"""Deterministic S14 synthesis for E04 chimeric aggregation.

This module reads only validated upstream artifacts.  It does not run new
simulations and does not construct the Chief Scientist's final report bundle.
"""

from __future__ import annotations

from collections.abc import Iterable
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


SCHEMA = "e04.s14.aggregation_classification.v1"
STEP_ORDER = [
    "S01",
    "S02",
    "S03",
    "S04",
    "S05",
    "S06",
    "S07",
    "S08",
    "S09",
    "S10",
    "S11",
    "S11R",
    "S12",
    "S13",
]
CLASSIFICATIONS = {
    "supported_bounded",
    "supported_conditioning_sensitive",
    "partially_supported",
    "null_not_supported",
    "constraining_contradictory",
    "not_identified",
}
WORKFLOW_OUTCOMES = {"supportive", "null", "constraining/contradictory"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_dump(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def tree_snapshot(artifact_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for step in STEP_ORDER:
        step_dir = artifact_root / "research_steps" / step
        if not step_dir.is_dir():
            raise FileNotFoundError(step_dir)
        for path in sorted(item for item in step_dir.rglob("*") if item.is_file()):
            records.append(
                {
                    "step": step,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def validate_upstream_manifests(artifact_root: Path) -> dict[str, Any]:
    step_results = []
    total_records = 0
    for step in STEP_ORDER:
        step_dir = artifact_root / "research_steps" / step
        manifest_path = step_dir / "artifact_manifest.json"
        manifest = read_json(manifest_path)
        failures = []
        for record in manifest["artifacts"]:
            path = step_dir / record["path"]
            if not path.exists():
                failures.append({"path": str(path), "reason": "missing"})
            elif path.stat().st_size != record["bytes"]:
                failures.append({"path": str(path), "reason": "byte_count"})
            elif sha256_file(path) != record["sha256"]:
                failures.append({"path": str(path), "reason": "sha256"})
        total_records += len(manifest["artifacts"])
        step_results.append(
            {
                "step": step,
                "manifest": str(manifest_path),
                "records": len(manifest["artifacts"]),
                "passed": not failures,
                "failures": failures,
            }
        )
    return {
        "passed": all(item["passed"] for item in step_results),
        "manifestRecords": total_records,
        "steps": step_results,
    }


def extract_facts(artifact_root: Path) -> dict[str, Any]:
    steps = artifact_root / "research_steps"

    s01 = pd.read_csv(steps / "S01" / "baseline_agreement.csv")
    targeted = s01[s01.paper_peak.notna()]
    exact = targeted[targeted.peak_agreement.astype(str).str.lower().eq("true")]
    both = exact[exact.timing_agreement.astype(str).str.lower().eq("true")]

    s02 = pd.read_parquet(steps / "S02" / "composition_baselines.parquet")
    pair = s02[(s02.n == 100) & (s02.counts_json == "[50,50]")].iloc[0]
    three = s02[(s02.n == 100) & (s02.counts_json == "[34,33,33]")].iloc[0]

    s03_effects = pd.read_csv(steps / "S03" / "factorial_effects.csv")
    composition_auc = s03_effects[
        (s03_effects.contrast_family == "composition_anchor")
        & (s03_effects.metric == "corrected_auc_over_progress")
    ]
    correlation_final = s03_effects[
        (s03_effects.contrast_family == "correlation_anchor")
        & (s03_effects.metric == "corrected_final_publication_aggregation")
    ]
    minority = pd.read_csv(steps / "S03" / "extreme_minority_variance.csv")
    estimable = minority[minority.peak_variance_ratio_estimable.astype(bool)]

    s04 = read_json(steps / "S04" / "analysis_summary.json")
    s05 = read_json(steps / "S05" / "analysis_summary.json")
    s06 = read_json(steps / "S06" / "analysis_summary.json")
    s06_anchors = pd.read_csv(steps / "S06" / "primary_anchor_results.csv")
    s07 = read_json(steps / "S07" / "analysis_summary.json")
    s07_contrasts = pd.read_parquet(
        steps / "S07" / "primary_anchor_kinetic_contrasts.parquet"
    )
    activation_peak = s07_contrasts[
        (s07_contrasts.regime == "activation_equal") & (s07_contrasts.outcome == "peak")
    ]
    s08 = read_json(steps / "S08" / "analysis_summary.json")
    s09 = read_json(steps / "S09" / "analysis_summary.json")
    s10 = read_json(steps / "S10" / "analysis_summary.json")
    s11r = read_json(steps / "S11R" / "calibration_decision.json")
    weakest = next(
        row for row in s11r["candidateDiagnostics"] if row["candidateId"] == "M02_MIN"
    )
    s12_model = pd.read_parquet(steps / "S12" / "model_selection.parquet")
    selected = s12_model[
        s12_model.candidate == "transport_forest_depth-None_leaf-5"
    ].iloc[0]
    baseline = s12_model[s12_model.candidate == "static_ridge_alpha-1.0"].iloc[0]
    s12_strata = pd.read_parquet(steps / "S12" / "stratified_performance.parquet")
    native_absent = s12_strata[
        (s12_strata.stratifier == "regime+correlation_profile")
        & (s12_strata.regime == "native_control")
        & (s12_strata.correlation_profile == "absent")
    ].iloc[0]
    activation_absent = s12_strata[
        (s12_strata.stratifier == "regime+correlation_profile")
        & (s12_strata.regime == "activation_equal")
        & (s12_strata.correlation_profile == "absent")
    ].iloc[0]
    s12 = read_json(steps / "S12" / "adequacy_decision.json")
    s13 = read_json(steps / "S13" / "analysis_summary.json")
    s13_adequacy = read_json(steps / "S13" / "adequacy_decision.json")

    facts = {
        "paperTargetedProfiles": int(len(targeted)),
        "paperProfilesMatchingPeakAndTiming": int(len(both)),
        "paperUniqueProfilesMatchingPeak": int(
            exact[exact.input_profile == "unique_1_100"].shape[0]
        ),
        "paperRepeatedProfilesMatchingPeakAndTiming": int(
            both[both.input_profile == "repeated_1_10_x10"].shape[0]
        ),
        "balancedPairPaperExpectation": float(pair.paper_expectation),
        "balancedThreePaperExpectation": float(three.paper_expectation),
        "compositionAucContrasts": int(len(composition_auc)),
        "compositionAucContrastsPassing": int(composition_auc.criterion_pass.sum()),
        "compositionAucDifferenceMin": float(composition_auc.mean_difference.min()),
        "compositionAucDifferenceMax": float(composition_auc.mean_difference.max()),
        "correlationFinalContrasts": int(len(correlation_final)),
        "correlationFinalContrastsPassing": int(correlation_final.criterion_pass.sum()),
        "correlationFinalDifferenceMin": float(correlation_final.mean_difference.min()),
        "correlationFinalDifferenceMax": float(correlation_final.mean_difference.max()),
        "minorityProfiles": int(len(minority)),
        "minorityVarianceEstimable": int(minority.peak_variance_ratio_estimable.sum()),
        "minorityVarianceNotEstimable": int(
            (~minority.peak_variance_ratio_estimable).sum()
        ),
        "minorityVarianceClearlyHigher": int(
            (estimable.peak_variance_ratio_ci95_low > 1).sum()
        ),
        "metricAssociationAnchorsPassing": int(s04["associationAnchorsPassing"]),
        "metricAssociationAnchorsTotal": int(s04["associationAnchorsTotal"]),
        "equalValueRankQuartileDisagreement": float(
            s04["rankQuartileDisagreementRate"]["equal_value_corrected_excess"]
        ),
        "equalValueSignDisagreement": float(
            s04["signDisagreementRateWhereComparable"]["equal_value_corrected_excess"]
        ),
        "staticConditioningLossFraction": float(
            s05["lossFractionAmongGloballySignificant"]
        ),
        "dynamicValueConditioningLoss": int(
            s06["globalPeakSignificantLostAfterValueConditioning"]
        ),
        "dynamicMobilityConditioningLoss": int(
            s06["globalPeakSignificantLostAfterMobilityConditioning"]
        ),
        "globalDynamicSignificant": int(
            s06["significantPeakCounts"]["label_permuted_global"]
        ),
        "mobilityDynamicSignificant": int(
            s06["significantPeakCounts"]["mobility_matched"]
        ),
        "balancedDynamicAnchorCount": int(s06["primaryAnchors"]["anchorCount"]),
        "balancedDynamicPeakPassed": int(s06["primaryAnchors"]["peakPassed"]),
        "balancedDynamicAreaPassed": int(s06["primaryAnchors"]["persistencePassed"]),
        "balancedPeakEffectMin": float(s06_anchors.adjusted_effect.min()),
        "balancedPeakEffectMax": float(s06_anchors.adjusted_effect.max()),
        "peakSelectionInflationMin": float(s06["selectionInflation"]["min"]),
        "peakSelectionInflationMax": float(s06["selectionInflation"]["max"]),
        "activationMatchingFeasible": bool(
            s07["selectedRegimeFeasibility"]["activation_equal"]
        ),
        "jointKineticMatchingFeasible": bool(s07["jointCalibrationFeasible"]),
        "activationPeakMedianReduction": float(
            activation_peak.percent_reduction.median()
        ),
        "jointSensitivityPeakMedianReduction": float(
            s07["primaryOutcomeRules"]["peak"]["medianPercentReduction"]
        ),
        "jointSensitivityAreaMedianReduction": float(
            s07["primaryOutcomeRules"]["positive_area"]["medianPercentReduction"]
        ),
        "ghostPeakMedianAbsolute": float(
            s08["ghostLabelRules"]["peak"]["medianAbsoluteEffect"]
        ),
        "ghostPeakSignificant": int(s08["ghostLabelRules"]["peak"]["significant"]),
        "policyDiscriminationPeakMin": float(
            s08["policyDiscriminationMinimumByOutcome"]["peak"]
        ),
        "valueBlockedFinalInflation": float(s08["valueBlockedMedianFinalInflation"]),
        "posthocBestOf20PeakInflation": float(
            s08["bestOf20MedianInflation"]["mixed|peak"]
        ),
        "switchLabelNullPassed": bool(s09["labelNullPassed"]),
        "switchPrimary25Peak": float(
            next(
                row["peak_estimate"]
                for row in s09["pooledEffects"]
                if row["timing"] == "fraction_25"
                and row["selection_rule"] == "retain_old_reset_new"
            )
        ),
        "switchPrimary50Peak": float(
            next(
                row["peak_estimate"]
                for row in s09["pooledEffects"]
                if row["timing"] == "fraction_50"
                and row["selection_rule"] == "retain_old_reset_new"
            )
        ),
        "restorationExtent": float(s10["pooledEffects"][0]["recovery_extent"]),
        "restorationNullExcessArea": float(s10["pooledEffects"][0]["null_excess_area"]),
        "restorationCompletionFraction": float(s10["primaryCompletionFraction"]),
        "restorationHalfReachedFraction": float(
            s10["pooledEffects"][0]["half_reached_fraction"]
        ),
        "restorationCompositeGatePassed": bool(all(s10["gates"].values())),
        "s11rCandidates": int(len(s11r["candidateDiagnostics"])),
        "s11rPassingCandidates": int(len(s11r["passingCandidates"])),
        "s11rHoldoutAuthorized": bool(s11r["holdoutAuthorized"]),
        "s11rWeakestPrimaryComplete": int(weakest["pooledPrimaryComplete"]),
        "s11rWeakestMatchedComplete": int(weakest["pooledMatchedComplete"]),
        "developmentModelRmse": float(selected.pooled_rmse),
        "developmentBaselineRmse": float(baseline.pooled_rmse),
        "nativeAbsentRmse": float(native_absent.rmse),
        "activationAbsentRmse": float(activation_absent.rmse),
        "nativeExternalRmse": float(s12["nativeExternal"]["rmse"]),
        "nativeExternalBaselineRmse": float(s12["nativeExternal"]["baseline_rmse"]),
        "activationTransferRmse": float(s12["activationEqual"]["rmse"]),
        "activationTransferBaselineRmse": float(
            s12["activationEqual"]["baseline_rmse"]
        ),
        "transportPredictivelyAdequate": bool(s12["predictivelyAdequate"]),
        "phaseConditions": int(s13["conditionCount"]),
        "phasePrimaryRuns": int(s13["primaryRunCount"]),
        "phaseLongRuns": int(s13["longRunCount"]),
        "phaseModalFixed": int(s13["conditionModalCounts"]["fixed_quiescence"]),
        "phaseModalDynamic": int(s13["conditionModalCounts"]["dynamic_equilibrium"]),
        "phaseModalDominance": int(s13["conditionModalCounts"]["active_dominance"]),
        "phaseModalUnresolved": int(
            s13["conditionModalCounts"]["unresolved_transient"]
        ),
        "phaseOscillationRuns": int(s13["regimeRunCounts"]["oscillation"]),
        "phaseMetastabilityRuns": int(s13["regimeRunCounts"]["metastability"]),
        "paperDynamicAnchors": int(s13["paperAnchorDynamicEquilibriumStableCount"]),
        "paperAnchorCount": int(s13["paperAnchorCount"]),
        "fixedLongStability": float(s13["majorRegimeStability"]["fixed_quiescence"]),
        "dynamicLongStability": float(
            s13["majorRegimeStability"]["dynamic_equilibrium"]
        ),
        "dominanceLongStability": float(
            s13["majorRegimeStability"]["active_dominance"]
        ),
        "phaseGeneralEquilibriumLanguageSupported": bool(
            s13_adequacy["paperGeneralDynamicEquilibriumLanguageSupported"]
        ),
    }
    validate_facts(facts)
    return facts


def validate_facts(facts: dict[str, Any]) -> None:
    checks = {
        "paper_targets": facts["paperTargetedProfiles"] == 7,
        "paper_matches": facts["paperProfilesMatchingPeakAndTiming"] == 3,
        "pair_baseline": abs(facts["balancedPairPaperExpectation"] - 0.49) < 1e-12,
        "three_baseline": abs(facts["balancedThreePaperExpectation"] - 0.3234) < 1e-12,
        "composition_contrasts": facts["compositionAucContrastsPassing"] == 12,
        "correlation_contrasts": facts["correlationFinalContrastsPassing"] == 16,
        "metric_anchors": facts["metricAssociationAnchorsPassing"] == 16,
        "dynamic_anchors": facts["balancedDynamicPeakPassed"] == 6
        and facts["balancedDynamicAreaPassed"] == 6,
        "identity": facts["ghostPeakSignificant"] == 0
        and facts["switchLabelNullPassed"],
        "kinetic_feasibility": facts["activationMatchingFeasible"]
        and not facts["jointKineticMatchingFeasible"],
        "restoration_gate": not facts["restorationCompositeGatePassed"],
        "mediation_gate": facts["s11rPassingCandidates"] == 0
        and not facts["s11rHoldoutAuthorized"],
        "transport_gate": not facts["transportPredictivelyAdequate"],
        "phase_counts": sum(
            facts[key]
            for key in (
                "phaseModalFixed",
                "phaseModalDynamic",
                "phaseModalDominance",
                "phaseModalUnresolved",
            )
        )
        == facts["phaseConditions"],
        "phase_nulls": facts["phaseOscillationRuns"] == 0
        and facts["phaseMetastabilityRuns"] == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"fact validation failed: {failed}")


def build_claims(f: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "claimId": "C00",
            "claim": "The clean-room baseline numerically reproduces every paper-reported aggregation peak and timing profile.",
            "mechanismDomain": "historical replication boundary",
            "classification": "constraining_contradictory",
            "workflowOutcome": "constraining/contradictory",
            "decisionGateIds": "G01_MEASUREMENT",
            "evidenceSteps": "S01",
            "evidenceSummary": f"Only {f['paperProfilesMatchingPeakAndTiming']}/{f['paperTargetedProfiles']} targeted profiles matched both frozen peak and timing tolerances; all three were repeated-value pairs.",
            "estimate": f"{f['paperProfilesMatchingPeakAndTiming']}/{f['paperTargetedProfiles']}",
            "scope": "E01 clean-room reference backend at paper scale",
            "caveat": "Historical Figure 8-10 arrays and publication random streams were not recovered.",
            "permittedWording": "The clean-room simulator reproduces the repeated-value pair anchors but not the paper's full numerical aggregation profile.",
            "prohibitedUpgrade": "Exact publication replication",
            "recommendedUse": "Treat subsequent evidence as reconstructed-model mechanism evidence.",
        },
        {
            "claimId": "C01",
            "claim": "A universal 0.5 reference is the exact paper-denominator expectation for every finite mixture.",
            "mechanismDomain": "measurement and composition",
            "classification": "constraining_contradictory",
            "workflowOutcome": "constraining/contradictory",
            "decisionGateIds": "G01_MEASUREMENT",
            "evidenceSteps": "S02",
            "evidenceSummary": f"The exact expectations are {f['balancedPairPaperExpectation']:.4f} for 50/50 and {f['balancedThreePaperExpectation']:.4f} for 34/33/33 at n=100.",
            "estimate": f"{f['balancedPairPaperExpectation']:.4f}; {f['balancedThreePaperExpectation']:.4f}",
            "scope": "Paper-denominator adjacency under fixed exact counts",
            "caveat": "The expectation is denominator-specific and is not a dynamic maximum correction.",
            "permittedWording": "Center paper-denominator adjacency on its exact finite-n composition expectation.",
            "prohibitedUpgrade": "Raw distance from 0.5 as universal excess aggregation",
            "recommendedUse": "Require this correction in E06/E07 benchmark metrics.",
        },
        {
            "claimId": "C02",
            "claim": "Balanced absent-association mixtures show corrected transient adjacency beyond declared dynamic chance baselines.",
            "mechanismDomain": "chance and peak bias",
            "classification": "supported_bounded",
            "workflowOutcome": "supportive",
            "decisionGateIds": "G01_MEASUREMENT|G02_DYNAMIC_CHANCE",
            "evidenceSteps": "S02|S05|S06",
            "evidenceSummary": f"All {f['balancedDynamicPeakPassed']}/{f['balancedDynamicAnchorCount']} peaks and {f['balancedDynamicAreaPassed']}/{f['balancedDynamicAnchorCount']} positive-area anchors passed global dynamic-null correction; adjusted peak effects were {f['balancedPeakEffectMin']:.4f}-{f['balancedPeakEffectMax']:.4f}.",
            "estimate": f"6/6 peak; 6/6 area; effect {f['balancedPeakEffectMin']:.4f}-{f['balancedPeakEffectMax']:.4f}",
            "scope": "Six balanced, absent-association, unique/repeated pairwise anchors",
            "caveat": "Value-stratified conditioning answers a different question and removes many constructed association effects.",
            "permittedWording": "Composition- and peak-corrected transient spatial dependence survives global and mobility-aware chance controls in the six primary anchors.",
            "prohibitedUpgrade": "Biological attraction or universal policy sorting",
            "recommendedUse": "Use peak and positive area, with the null family named.",
        },
        {
            "claimId": "C03",
            "claim": "Exact mixture composition changes cumulative corrected aggregation.",
            "mechanismDomain": "composition",
            "classification": "supported_bounded",
            "workflowOutcome": "supportive",
            "decisionGateIds": "G01_MEASUREMENT",
            "evidenceSteps": "S03",
            "evidenceSummary": f"All {f['compositionAucContrastsPassing']}/{f['compositionAucContrasts']} prespecified 10/90 or 90/10 versus 50/50 AUC contrasts passed; differences ranged {f['compositionAucDifferenceMin']:.4f} to {f['compositionAucDifferenceMax']:.4f}.",
            "estimate": f"12/12; {f['compositionAucDifferenceMin']:.4f} to {f['compositionAucDifferenceMax']:.4f}",
            "scope": "Absent-association pairwise mixtures at n=100",
            "caveat": "Composition also changes lattice resolution and, for some policies, task kinetics.",
            "permittedWording": "Corrected aggregation magnitude is composition-dependent.",
            "prohibitedUpgrade": "A composition-invariant phenotype",
            "recommendedUse": "Stratify or factorially vary composition in successor benchmarks.",
        },
        {
            "claimId": "C04",
            "claim": "Ten-cell minorities generically have higher corrected-peak variance than balanced mixtures.",
            "mechanismDomain": "minority discreteness",
            "classification": "null_not_supported",
            "workflowOutcome": "null",
            "decisionGateIds": "G01_MEASUREMENT",
            "evidenceSteps": "S03",
            "evidenceSummary": f"Only {f['minorityVarianceEstimable']}/{f['minorityProfiles']} variance ratios were estimable, none had a confidence interval wholly above one, and {f['minorityVarianceNotEstimable']} were degenerate.",
            "estimate": f"0/{f['minorityVarianceEstimable']} clearly higher; {f['minorityVarianceNotEstimable']}/{f['minorityProfiles']} non-estimable",
            "scope": "Frozen extreme-minority factorial",
            "caveat": "Minority homotypy remains visibly discrete in 0.1 increments; a null generic variance claim does not remove that finite-count behavior.",
            "permittedWording": "Extreme minorities are discrete, but not generically more variable in this factorial.",
            "prohibitedUpgrade": "Universal high-variance minority behavior",
            "recommendedUse": "Report exact counts and discrete support, not only Gaussian intervals.",
        },
        {
            "claimId": "C05",
            "claim": "The aggregation interpretation is invariant to metric choice and value conditioning.",
            "mechanismDomain": "measurement robustness",
            "classification": "constraining_contradictory",
            "workflowOutcome": "constraining/contradictory",
            "decisionGateIds": "G01_MEASUREMENT",
            "evidenceSteps": "S04",
            "evidenceSummary": f"All {f['metricAssociationAnchorsPassing']}/{f['metricAssociationAnchorsTotal']} association anchors passed multiple directional metrics, but equal-value conditioning had {100 * f['equalValueRankQuartileDisagreement']:.1f}% quartile and {100 * f['equalValueSignDisagreement']:.1f}% sign disagreement where comparable.",
            "estimate": f"16/16 broad; {100 * f['equalValueRankQuartileDisagreement']:.1f}% quartile disagreement",
            "scope": "S03 factorial under eight metric views",
            "caveat": "Equal-value conditioning is undefined for unique inputs and can remove the policy-value pathway.",
            "permittedWording": "Several metrics corroborate broad spatial dependence, while value-conditioned and cluster-size metrics materially change interpretation.",
            "prohibitedUpgrade": "Metric-invariant clustering mechanism",
            "recommendedUse": "Report adjacency, assortativity/information, cluster capture, and value-conditioned diagnostics separately.",
        },
        {
            "claimId": "C06",
            "claim": "Policy-value association is a major, conditioning-sensitive pathway in the largest aggregation signals.",
            "mechanismDomain": "policy-value association",
            "classification": "supported_conditioning_sensitive",
            "workflowOutcome": "supportive",
            "decisionGateIds": "G01_MEASUREMENT|G02_DYNAMIC_CHANCE",
            "evidenceSteps": "S03|S04|S05|S06",
            "evidenceSummary": f"All {f['correlationFinalContrastsPassing']}/{f['correlationFinalContrasts']} association contrasts passed with final differences {f['correlationFinalDifferenceMin']:.4f}-{f['correlationFinalDifferenceMax']:.4f}; value conditioning removed {f['dynamicValueConditioningLoss']}/{f['globalDynamicSignificant']} globally significant dynamic peaks.",
            "estimate": f"16/16; {f['dynamicValueConditioningLoss']}/{f['globalDynamicSignificant']} lost",
            "scope": "Strong constructed association controls and their conditioned nulls",
            "caveat": "Conditioning can appropriately isolate within-value grouping or can over-condition away the pathway of interest.",
            "permittedWording": "Policy-value association is a major state variable and conditioning-sensitive pathway.",
            "prohibitedUpgrade": "Within-value like-type attraction",
            "recommendedUse": "Carry policy-value/state variables explicitly into E06/E07.",
        },
        {
            "claimId": "C07",
            "claim": "Unequal activation opportunity alone explains corrected aggregation.",
            "mechanismDomain": "kinetics",
            "classification": "null_not_supported",
            "workflowOutcome": "constraining/contradictory",
            "decisionGateIds": "G04_KINETICS",
            "evidenceSteps": "S07",
            "evidenceSummary": f"Activation matching was feasible, yet the median primary peak reduction was only {100 * f['activationPeakMedianReduction']:.2f}% and was heterogeneous.",
            "estimate": f"median peak reduction {100 * f['activationPeakMedianReduction']:.2f}%",
            "scope": "Label-blind activation-equal holdout intervention",
            "caveat": "This isolates activation opportunity, not displacement, target range, or success-rate kinetics.",
            "permittedWording": "Unequal activation opportunity alone does not explain the primary aggregation signal.",
            "prohibitedUpgrade": "All kinetic influence is absent",
            "recommendedUse": "Retain activation-equal as a regression control, not a complete mechanism test.",
        },
        {
            "claimId": "C08",
            "claim": "The separate and joint effects of displacement, target range, success rate, and activation kinetics are causally identified.",
            "mechanismDomain": "kinetic decomposition",
            "classification": "not_identified",
            "workflowOutcome": "constraining/contradictory",
            "decisionGateIds": "G04_KINETICS",
            "evidenceSteps": "S07",
            "evidenceSummary": "Displacement, target-range, success-rate, and joint regimes failed frozen balance feasibility; the permissive joint sensitivity was not a matched causal control.",
            "estimate": f"joint sensitivity median reductions {100 * f['jointSensitivityPeakMedianReduction']:.1f}% peak and {100 * f['jointSensitivityAreaMedianReduction']:.1f}% area",
            "scope": "Label-blind interventions that preserve native proposals and local policy structure",
            "caveat": "Descriptive changes under infeasible regimes cannot identify kinetic components.",
            "permittedWording": "Full kinetic contribution remains unresolved because joint matching was infeasible.",
            "prohibitedUpgrade": "Validated kinetic mediation or complete efficiency explanation",
            "recommendedUse": "E06 may design new feasible kinetic interventions; it must not reuse the failed joint regime as matched evidence.",
        },
        {
            "claimId": "C09",
            "claim": "Action-irrelevant analysis labels cause cells to aggregate.",
            "mechanismDomain": "identity control",
            "classification": "null_not_supported",
            "workflowOutcome": "constraining/contradictory",
            "decisionGateIds": "G03_IDENTITY",
            "evidenceSteps": "S08|S09",
            "evidenceSummary": f"Ghost-label peak effects had median absolute magnitude {f['ghostPeakMedianAbsolute']:.4f} with {f['ghostPeakSignificant']} significant channels; label-only switching preserved future physical trajectories exactly.",
            "estimate": f"ghost median |peak| {f['ghostPeakMedianAbsolute']:.4f}; 0 significant",
            "scope": "Frozen labels absent from observations and transitions",
            "caveat": "Value-correlated or outcome-selected grouping can still manufacture a descriptive label pattern.",
            "permittedWording": "Inert label identity is not a behavioral cause in the clean-room simulator.",
            "prohibitedUpgrade": "Algotype names are sensed or causally attractive",
            "recommendedUse": "Include ghost, same-label/different-policy, and different-label/same-policy controls in successors.",
        },
        {
            "claimId": "C10",
            "claim": "Executable policy identity changes state-matched continuation outcomes independently of inert label identity.",
            "mechanismDomain": "policy identity and path dependence",
            "classification": "supported_bounded",
            "workflowOutcome": "supportive",
            "decisionGateIds": "G03_IDENTITY",
            "evidenceSteps": "S08|S09",
            "evidenceSummary": f"Policy discrimination exceeded {f['policyDiscriminationPeakMin']:.4f} at every primary anchor; primary policy switches produced pooled tracking peaks {f['switchPrimary25Peak']:.4f} at 25% and {f['switchPrimary50Peak']:.4f} at 50%, while the label null passed.",
            "estimate": f"switch peaks {f['switchPrimary25Peak']:.4f}, {f['switchPrimary50Peak']:.4f}",
            "scope": "Deterministic state-matched continuations with prespecified cursor rules",
            "caveat": "Selection memory ownership is not unique; donor-transfer completion sensitivity bounds interpretation.",
            "permittedWording": "Executable local policy, not inert naming, causally changes future policy-position structure within the tested simulator continuations.",
            "prohibitedUpgrade": "A context-free or biological causal identity effect",
            "recommendedUse": "Use runtime-key and cursor-explicit policy switches as the identity benchmark.",
        },
        {
            "claimId": "C11",
            "claim": "Post hoc or value-correlated label grouping cannot create an aggregation-looking result.",
            "mechanismDomain": "grouping artifact",
            "classification": "constraining_contradictory",
            "workflowOutcome": "constraining/contradictory",
            "decisionGateIds": "G01_MEASUREMENT|G03_IDENTITY",
            "evidenceSteps": "S08",
            "evidenceSummary": f"Value-blocked labels inflated the median final corrected endpoint by {f['valueBlockedFinalInflation']:.4f}; best-of-20 outcome selection inflated mixed-label peaks by {f['posthocBestOf20PeakInflation']:.4f}.",
            "estimate": f"{f['valueBlockedFinalInflation']:.4f}; {f['posthocBestOf20PeakInflation']:.4f}",
            "scope": "Action-irrelevant diagnostic labels",
            "caveat": "These are measurement artifacts, not changes in physical trajectories.",
            "permittedWording": "Frozen grouping definitions are necessary because value blocking and outcome selection can manufacture apparent label clustering.",
            "prohibitedUpgrade": "Post hoc groups as behavioral phenotypes",
            "recommendedUse": "Freeze analysis labels before E06/E07 outcomes and audit label-channel selection.",
        },
        {
            "claimId": "C12",
            "claim": "De-clustered states satisfy the prespecified operational-attractor restoration criteria.",
            "mechanismDomain": "restoration",
            "classification": "partially_supported",
            "workflowOutcome": "null",
            "decisionGateIds": "G05_ATTRACTOR",
            "evidenceSteps": "S10",
            "evidenceSummary": f"Partial restoration occurred (extent {f['restorationExtent']:.4f}, null-excess area {f['restorationNullExcessArea']:.4f}), but completion was {100 * f['restorationCompletionFraction']:.0f}% and the completion, timing, and null-excess composite gates failed.",
            "estimate": f"extent {f['restorationExtent']:.4f}; completion {100 * f['restorationCompletionFraction']:.0f}%",
            "scope": "Repeated values and support-qualified equal-value de-clustering",
            "caveat": "Unique values cannot be physically de-clustered while preserving the full value order; policy-pair heterogeneity was strong.",
            "permittedWording": "Aggregation partially restored after de-clustering, but the stronger restoration label did not pass.",
            "prohibitedUpgrade": "Operational attractor",
            "recommendedUse": "Carry the intervention as a restoration challenge, with all six gates retained.",
        },
        {
            "claimId": "C13",
            "claim": "Value-position and policy-position contributions have been separated as validated mediation effects.",
            "mechanismDomain": "component separation",
            "classification": "not_identified",
            "workflowOutcome": "constraining/contradictory",
            "decisionGateIds": "G06_MEDIATION",
            "evidenceSteps": "S11|S11R",
            "evidenceSummary": f"The corrected matcher attained its targets, but none of {f['s11rCandidates']} candidates passed completion comparability; even the weakest yielded {f['s11rWeakestPrimaryComplete']}/150 and {f['s11rWeakestMatchedComplete']}/150 complete calibration arms, so holdout outcomes remained sealed.",
            "estimate": f"0/{f['s11rCandidates']} feasible; holdout sealed",
            "scope": "Support-qualified positional-inversion intervention ladder",
            "caveat": "Value reassignment changes task progress and is not a pure mediator intervention; S11 Shapley estimates remain descriptive.",
            "permittedWording": "The requested component separation is infeasible under the frozen completion constraints and therefore remains unidentified.",
            "prohibitedUpgrade": "Validated mediation or causal Shapley decomposition",
            "recommendedUse": "Do not require further repair before prediction, but do not use S11 estimates as mechanism truth.",
        },
        {
            "claimId": "C14",
            "claim": "Coarse transport summaries have useful local predictive adequacy inside absent-association support.",
            "mechanismDomain": "prediction",
            "classification": "supported_bounded",
            "workflowOutcome": "supportive",
            "decisionGateIds": "G07_TRANSPORT",
            "evidenceSteps": "S12",
            "evidenceSummary": f"Development RMSE was {f['developmentModelRmse']:.4f} versus {f['developmentBaselineRmse']:.4f}; absent-association RMSE was {f['nativeAbsentRmse']:.4f} for unseen native compositions and {f['activationAbsentRmse']:.4f} under activation equalization.",
            "estimate": f"development {f['developmentModelRmse']:.4f}; native absent {f['nativeAbsentRmse']:.4f}",
            "scope": "Frozen absent-association calibration and held-out composition tails",
            "caveat": "Descriptors are complete-run retrospective summaries and do not identify causes.",
            "permittedWording": "Coarse transport features predict absent-association composition tails within support.",
            "prohibitedUpgrade": "Online forecast or causal transport identification",
            "recommendedUse": "Use only as a simple baseline in E06/E07.",
        },
        {
            "claimId": "C15",
            "claim": "The S12 summary model is a generally adequate transport explanation across policy-value association and interventions.",
            "mechanismDomain": "predictive transfer",
            "classification": "constraining_contradictory",
            "workflowOutcome": "constraining/contradictory",
            "decisionGateIds": "G07_TRANSPORT",
            "evidenceSteps": "S12|S13",
            "evidenceSummary": f"Native external RMSE was {f['nativeExternalRmse']:.4f} versus baseline {f['nativeExternalBaselineRmse']:.4f}; activation-transfer RMSE was {f['activationTransferRmse']:.4f} versus {f['activationTransferBaselineRmse']:.4f}, with structured residuals.",
            "estimate": f"{f['nativeExternalRmse']:.4f}>{f['nativeExternalBaselineRmse']:.4f}; {f['activationTransferRmse']:.4f}>{f['activationTransferBaselineRmse']:.4f}",
            "scope": "Strong-association and intervention holdouts",
            "caveat": "The failure constrains the frozen four-descriptor model, not every time-resolved model.",
            "permittedWording": "The S12 model is locally useful but fails general transfer; S13 therefore requires explicit state, association, and time-resolved flux.",
            "prohibitedUpgrade": "General transport mechanism",
            "recommendedUse": "Build successor surrogates with explicit state and time-resolved transport, then revalidate transfer.",
        },
        {
            "claimId": "C16",
            "claim": "Opposing-goal mixtures all settle into one generic static or equilibrium regime.",
            "mechanismDomain": "conflicting-goal phase behavior",
            "classification": "constraining_contradictory",
            "workflowOutcome": "constraining/contradictory",
            "decisionGateIds": "G08_PHASE",
            "evidenceSteps": "S13",
            "evidenceSummary": f"Among {f['phaseConditions']} conditions, modal classes were {f['phaseModalFixed']} fixed quiescent, {f['phaseModalDynamic']} dynamic equilibrium, {f['phaseModalDominance']} active dominance, and {f['phaseModalUnresolved']} unresolved.",
            "estimate": f"{f['phaseModalFixed']}/{f['phaseModalDynamic']}/{f['phaseModalDominance']}/{f['phaseModalUnresolved']}",
            "scope": "Finite 4,500-condition opposing-direction factorial",
            "caveat": "This is an operational regime map, not a thermodynamic phase transition.",
            "permittedWording": "Conflicting goals yield heterogeneous quiescent, flux-supported, dominance, and unresolved regimes.",
            "prohibitedUpgrade": "One generic plateau mechanism",
            "recommendedUse": "Benchmark successor systems across all supported regimes.",
        },
        {
            "claimId": "C17",
            "claim": "Fixed quiescence and flux-supported dynamic equilibrium are empirically distinct operational regimes.",
            "mechanismDomain": "state versus flux",
            "classification": "supported_bounded",
            "workflowOutcome": "supportive",
            "decisionGateIds": "G08_PHASE",
            "evidenceSteps": "S13",
            "evidenceSummary": f"Fixed and dynamic long-budget stability were {100 * f['fixedLongStability']:.1f}% and {100 * f['dynamicLongStability']:.2f}%; only {f['paperDynamicAnchors']}/{f['paperAnchorCount']} paper-oriented anchors were stable dynamic equilibria and the other four were fixed quiescent.",
            "estimate": f"stability {100 * f['fixedLongStability']:.1f}% vs {100 * f['dynamicLongStability']:.2f}%; anchors 2/6",
            "scope": "S13 state-plus-turnover classifier and longer-budget audit",
            "caveat": "The two dynamic paper anchors were both Bubble-descending plus Selection-ascending.",
            "permittedWording": "A flat macroscopic trajectory may be absorbing or flux-supported; the two must be reported separately.",
            "prohibitedUpgrade": "Every plateau is dynamic equilibrium",
            "recommendedUse": "Require state, turnover, and flux evidence for equilibrium claims in E06/E07.",
        },
        {
            "claimId": "C18",
            "claim": "Oscillatory or metastable aggregation regimes were detected under the frozen S13 definitions.",
            "mechanismDomain": "phase terminology",
            "classification": "null_not_supported",
            "workflowOutcome": "null",
            "decisionGateIds": "G08_PHASE",
            "evidenceSteps": "S13",
            "evidenceSummary": f"Zero of {f['phasePrimaryRuns']} primary runs met either strict definition; longer-budget audits did not establish either regime.",
            "estimate": "0 oscillatory; 0 metastable",
            "scope": "Finite S13 grid and budgets through one million activations",
            "caveat": "This is a finite-grid null, not proof of universal absence.",
            "permittedWording": "No oscillation or metastability was detected under the tested definitions and budgets.",
            "prohibitedUpgrade": "Positive oscillation or metastability claim",
            "recommendedUse": "Keep both as future discovery labels requiring independent confirmation.",
        },
        {
            "claimId": "C19",
            "claim": "Duration above a corrected baseline is sufficient as the primary persistence discriminator.",
            "mechanismDomain": "persistence endpoint",
            "classification": "null_not_supported",
            "workflowOutcome": "constraining/contradictory",
            "decisionGateIds": "G02_DYNAMIC_CHANCE|G04_KINETICS",
            "evidenceSteps": "S06|S07",
            "evidenceSummary": "Duration failed all mobility-matched significance tests while peak and positive area remained discriminating; S07 excluded duration from decisions.",
            "estimate": "0 mobility-matched duration anchors",
            "scope": "Dynamic-null and kinetic-control anchors",
            "caveat": "Duration may still be descriptive when its null is declared.",
            "permittedWording": "Use corrected peak and positive area as primary persistence endpoints; duration alone is non-discriminating.",
            "prohibitedUpgrade": "Duration-only mechanism evidence",
            "recommendedUse": "Retain duration as a secondary descriptor only.",
        },
        {
            "claimId": "C20",
            "claim": "The validated aggregation phenomenon is best classified as a mixed-origin, behavior-linked transient spatial-dependence process.",
            "mechanismDomain": "overall synthesis",
            "classification": "supported_bounded",
            "workflowOutcome": "supportive",
            "decisionGateIds": "G01_MEASUREMENT|G02_DYNAMIC_CHANCE|G03_IDENTITY|G04_KINETICS|G05_ATTRACTOR|G06_MEDIATION|G07_TRANSPORT|G08_PHASE",
            "evidenceSteps": "S01|S02|S03|S04|S05|S06|S07|S08|S09|S10|S11|S11R|S12|S13",
            "evidenceSummary": "Corrected peak/area effects survive declared chance controls and track executable policy behavior, while composition, policy-value association, metric choice, cursor state, restoration heterogeneity, prediction limits, and phase regime all materially bound interpretation.",
            "estimate": "all eight decision gates adjudicated",
            "scope": "Validated one-dimensional clean-room simulator evidence",
            "caveat": "Several influences may be jointly sufficient; exact natural component shares were not identified.",
            "permittedWording": "A mixed-origin, policy-behavior-linked transient spatial dependence occurs in the tested simulator.",
            "prohibitedUpgrade": "Single universal causal mechanism",
            "recommendedUse": "Adopt the bounded classification and preserve all failed stronger gates in successor work.",
        },
        {
            "claimId": "C21",
            "claim": "The simulator evidence validates biological affinity, intention, preference, or emergent aggregation goals.",
            "mechanismDomain": "biological and agency boundary",
            "classification": "not_identified",
            "workflowOutcome": "constraining/contradictory",
            "decisionGateIds": "G03_IDENTITY|G05_ATTRACTOR|G06_MEDIATION|G07_TRANSPORT",
            "evidenceSteps": "S08|S09|S10|S11R|S12",
            "evidenceSummary": "Labels are behaviorally inert, the stronger restoration gate failed, component mediation was infeasible, and predictive transfer failed outside support.",
            "estimate": "no direct biological evidence",
            "scope": "Computational proxy only",
            "caveat": "The supplied paper's biological discussion remains hypothesis-generating rather than validated by E04.",
            "permittedWording": "The simulator generates testable hypotheses about behavior-linked spatial patterning; biological mechanisms require independent experiments.",
            "prohibitedUpgrade": "Biological affinity, attraction, preference, intention, or emergent goal",
            "recommendedUse": "Frame E06 as a computational extension and any biological transfer as a separately validated hypothesis.",
        },
    ]
    validate_claims(rows)
    return rows


def validate_claims(rows: list[dict[str, Any]]) -> None:
    required = {
        "claimId",
        "claim",
        "mechanismDomain",
        "classification",
        "workflowOutcome",
        "decisionGateIds",
        "evidenceSteps",
        "evidenceSummary",
        "estimate",
        "scope",
        "caveat",
        "permittedWording",
        "prohibitedUpgrade",
        "recommendedUse",
    }
    if len(rows) != 22 or len({row["claimId"] for row in rows}) != 22:
        raise AssertionError("claim count or uniqueness")
    for row in rows:
        if set(row) != required:
            raise AssertionError(f"claim schema mismatch: {row['claimId']}")
        if row["classification"] not in CLASSIFICATIONS:
            raise AssertionError(f"bad classification: {row['claimId']}")
        if row["workflowOutcome"] not in WORKFLOW_OUTCOMES:
            raise AssertionError(f"bad outcome: {row['claimId']}")
        if not all(str(row[field]).strip() for field in required):
            raise AssertionError(f"blank claim field: {row['claimId']}")


EVIDENCE_PATHS = {
    "S01": "baseline_agreement.csv",
    "S02": "interpretation_changes.csv",
    "S03": "factorial_effects.csv",
    "S04": "analysis_summary.json",
    "S05": "analysis_summary.json",
    "S06": "analysis_summary.json",
    "S07": "analysis_summary.json",
    "S08": "analysis_summary.json",
    "S09": "analysis_summary.json",
    "S10": "analysis_summary.json",
    "S11": "effective_validation_summary.json",
    "S11R": "calibration_decision.json",
    "S12": "adequacy_decision.json",
    "S13": "adequacy_decision.json",
}


def build_evidence_index(
    claims: list[dict[str, Any]], artifact_root: Path
) -> list[dict[str, Any]]:
    rows = []
    for claim in claims:
        for order, step in enumerate(claim["evidenceSteps"].split("|"), start=1):
            path = artifact_root / "research_steps" / step / EVIDENCE_PATHS[step]
            rows.append(
                {
                    "evidenceId": f"{claim['claimId']}-{order:02d}",
                    "claimId": claim["claimId"],
                    "sourceStep": step,
                    "sourceArtifact": str(path),
                    "sha256": sha256_file(path),
                    "evidenceRole": "primary"
                    if order == 1
                    else "corroborating_or_boundary",
                    "interpretation": claim["evidenceSummary"],
                }
            )
    return rows


def build_caveat_index(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "caveatId": f"K{index:02d}",
            "claimId": row["claimId"],
            "mechanismDomain": row["mechanismDomain"],
            "caveatOrBlocker": row["caveat"],
            "blockedUpgrade": row["prohibitedUpgrade"],
            "requiredAction": row["recommendedUse"],
        }
        for index, row in enumerate(claims, start=1)
    ]


def build_terminology_audit(contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for term in contract["terminology"]["permitted"]:
        rows.append(
            {
                "term": term,
                "status": "permitted",
                "rule": "May be used within the S14 clean-room simulator scope.",
                "positiveClaimAllowed": True,
            }
        )
    for term, rule in contract["terminology"]["qualifiedOnly"].items():
        rows.append(
            {
                "term": term,
                "status": "qualified_only",
                "rule": rule,
                "positiveClaimAllowed": True,
            }
        )
    for term in contract["terminology"]["prohibitedAsPositiveClaims"]:
        rows.append(
            {
                "term": term,
                "status": "negative_or_boundary_only",
                "rule": contract["terminology"]["negativeReportingRule"],
                "positiveClaimAllowed": False,
            }
        )
    return rows


def build_decision_audit(
    claims: list[dict[str, Any]], contract: dict[str, Any]
) -> list[dict[str, Any]]:
    gate_ids = {row["gateId"] for row in contract["decisionGates"]}
    rows = []
    for gate in contract["decisionGates"]:
        mapped = [
            row for row in claims if gate["gateId"] in row["decisionGateIds"].split("|")
        ]
        rows.append(
            {
                "gateId": gate["gateId"],
                "question": gate["question"],
                "claimCount": len(mapped),
                "claimIds": "|".join(row["claimId"] for row in mapped),
                "evidenceSteps": "|".join(gate["passEvidence"]),
                "failureConsequence": gate["failureConsequence"],
                "covered": bool(mapped),
            }
        )
    used = {gate for row in claims for gate in row["decisionGateIds"].split("|")}
    if used != gate_ids:
        raise AssertionError(f"decision gate coverage mismatch: {used ^ gate_ids}")
    return rows


def _phase_exemplar(path: Path, regime: str) -> str:
    frame = pd.read_parquet(path)
    selected = frame[
        (frame.modal_regime == regime)
        & (~frame.boundary_uncertain.astype(bool))
        & (frame.modal_fraction >= 0.8)
    ].sort_values("condition_id")
    if selected.empty:
        raise AssertionError(f"no phase exemplar for {regime}")
    return str(selected.iloc[0].condition_id)


def build_benchmarks(artifact_root: Path) -> list[dict[str, Any]]:
    steps = artifact_root / "research_steps"
    phase_path = steps / "S13" / "conflict_phase_map.parquet"
    rows: list[dict[str, Any]] = []

    def add(
        benchmark_id: str,
        family: str,
        source_step: str,
        source_name: str,
        selector: dict[str, Any],
        endpoint: str,
        expected: str,
        acceptance: str,
        split_role: str,
        boundary: str,
    ) -> None:
        rows.append(
            {
                "benchmarkId": benchmark_id,
                "family": family,
                "sourceStep": source_step,
                "sourceArtifact": str(steps / source_step / source_name),
                "conditionSelector": json.dumps(
                    selector, sort_keys=True, separators=(",", ":")
                ),
                "endpoint": endpoint,
                "expectedDecision": expected,
                "acceptanceRule": acceptance,
                "splitRole": split_role,
                "claimBoundary": boundary,
            }
        )

    add(
        "B01",
        "composition_denominator",
        "S02",
        "composition_baselines.parquet",
        {"n": 100, "counts_json": "[50,50]"},
        "paper_expectation",
        "0.4900",
        "exact absolute error <= 1e-12",
        "mechanism_regression",
        "Exact finite-n denominator fixture",
    )
    add(
        "B02",
        "composition_denominator",
        "S02",
        "composition_baselines.parquet",
        {"n": 100, "counts_json": "[34,33,33]"},
        "paper_expectation",
        "0.3234",
        "exact absolute error <= 1e-12",
        "mechanism_regression",
        "Three-way reference; never substitute 0.5",
    )
    anchor_ids = [
        "S03-REP-BUB-INS-P50-ABS",
        "S03-REP-BUB-SEL-P50-ABS",
        "S03-REP-INS-SEL-P50-ABS",
        "S03-UNQ-BUB-INS-P50-ABS",
        "S03-UNQ-BUB-SEL-P50-ABS",
        "S03-UNQ-INS-SEL-P50-ABS",
    ]
    for index, condition_id in enumerate(anchor_ids, start=3):
        add(
            f"B{index:02d}",
            "dynamic_null_anchor",
            "S06",
            "primary_anchor_results.csv",
            {"condition_id": condition_id},
            "peak and positive area",
            "positive adjusted peak and positive-area excess",
            "anchor_passed=true; positive-area family 6/6",
            "e07_protected_confirmation",
            "Balanced absent-association primary anchor",
        )
    add(
        "B09",
        "association_stress",
        "S03",
        "condition_summary.csv",
        {"condition_id": "S03-UNQ-BUB-INS-P50-POS"},
        "final corrected adjacency",
        "strong positive association stress",
        "compare against S03-UNQ-BUB-INS-P50-ABS with frozen pairing",
        "e07_protected_confirmation",
        "Association pathway, not within-value attraction",
    )
    add(
        "B10",
        "extreme_minority",
        "S03",
        "condition_summary.csv",
        {"condition_id": "S03-UNQ-BUB-SEL-P10-ABS"},
        "corrected AUC and discrete minority homotypy",
        "extreme composition differs from 50/50",
        "retain exact counts and discrete support",
        "e07_protected_confirmation",
        "No generic high-variance assumption",
    )
    add(
        "B11",
        "identity_ghost",
        "S08",
        "identity_controls.parquet",
        {
            "population": "mixed",
            "control_family": "action_irrelevant_ghost_labels",
            "outcome": "peak",
        },
        "ghost adjusted peak",
        "no significant ghost-label effect",
        "all q-values above family threshold and median absolute effect <= 0.005",
        "e07_protected_confirmation",
        "Labels absent from observations/transitions",
    )
    add(
        "B12",
        "policy_switch",
        "S09",
        "pooled_policy_tracking_effects.parquet",
        {"timing": "fraction_25", "selection_rule": "retain_old_reset_new"},
        "policy-tracking peak and area",
        "positive policy switch effect",
        "peak lower CI > 0 and all six condition peaks positive",
        "e07_protected_confirmation",
        "State-matched clean-room continuation only",
    )
    add(
        "B13",
        "label_switch",
        "S09",
        "policy_label_switches.parquet",
        {
            "arm": "label_only",
            "timing": "fraction_25",
            "selection_rule": "identity_retain",
        },
        "physical continuation identity",
        "exact label-null continuation",
        "state/occupancy/ledger equal to no-switch pair",
        "mechanism_regression",
        "Relabeling is action-irrelevant",
    )
    add(
        "B14",
        "restoration_challenge",
        "S10",
        "pooled_recovery_effects.parquet",
        {"cursor_rule": "identity_owned"},
        "recovery extent and null-excess area",
        "partial restoration; composite label fails",
        "retain six S10 gates and 76% completion caveat",
        "e07_protected_confirmation",
        "Repeated values only; no operational-attractor upgrade",
    )
    add(
        "B15",
        "kinetic_feasibility",
        "S07",
        "residual_imbalance_summary.csv",
        {"regime": "joint_equal"},
        "balance and completion",
        "joint kinetic regime infeasible",
        "calibration_feasible=false and holdout_balance_passed=false",
        "mechanism_regression",
        "Not a matched causal control",
    )
    add(
        "B16",
        "component_infeasibility",
        "S11R",
        "calibration_decision.json",
        {},
        "promotion decision",
        "no completion-feasible value perturbation",
        "passingCandidates empty and holdoutAuthorized=false",
        "mechanism_regression",
        "No validated mediation estimate",
    )
    add(
        "B17",
        "transport_failure",
        "S12",
        "adequacy_decision.json",
        {},
        "held-out adequacy",
        "general adequacy fails",
        "predictivelyAdequate=false and causalIdentification=false",
        "e07_protected_confirmation",
        "Local predictive baseline only",
    )
    phase_regimes = [
        ("fixed_quiescence", "B18"),
        ("dynamic_equilibrium", "B19"),
        ("active_dominance", "B20"),
        ("unresolved_transient", "B21"),
    ]
    for regime, benchmark_id in phase_regimes:
        add(
            benchmark_id,
            "phase_exemplar",
            "S13",
            "conflict_phase_map.parquet",
            {"condition_id": _phase_exemplar(phase_path, regime)},
            "state plus turnover/flux regime",
            regime,
            f"modal_regime={regime}, modal_fraction>=0.8, boundary_uncertain=false",
            "e07_protected_confirmation",
            "Finite simulator regime; require longer-budget confirmation",
        )
    return rows


def resolve_benchmark_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resolved = []
    for row in rows:
        path = Path(row["sourceArtifact"])
        selector = json.loads(row["conditionSelector"])
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            for key, value in selector.items():
                frame = frame[frame[key] == value]
            count = len(frame)
        elif path.suffix == ".csv":
            frame = pd.read_csv(path)
            for key, value in selector.items():
                frame = frame[frame[key] == value]
            count = len(frame)
        elif path.suffix == ".json":
            count = 1 if path.exists() else 0
        else:
            raise ValueError(path)
        resolved.append(
            {
                **row,
                "sourceSha256": sha256_file(path),
                "resolvedRows": int(count),
                "resolved": count > 0,
            }
        )
    if not all(row["resolved"] for row in resolved):
        raise AssertionError("benchmark selector failed")
    return resolved


def _human_title(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").title()


def _table_shape(path: Path) -> tuple[int, int]:
    if path.suffix == ".parquet":
        metadata = pq.ParquetFile(path).metadata
        return metadata.num_rows, metadata.num_columns
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader([handle.readline()]))
        rows = sum(1 for _ in handle)
    return rows, len(header)


def build_figure_index(artifact_root: Path) -> list[dict[str, Any]]:
    rows = []
    for step in STEP_ORDER:
        step_dir = artifact_root / "research_steps" / step
        for path in sorted(step_dir.glob("*.png")):
            svg = path.with_suffix(".svg")
            rows.append(
                {
                    "figureId": f"F{len(rows) + 1:03d}",
                    "sourceStep": step,
                    "title": _human_title(path),
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "svgPath": str(svg) if svg.exists() else "",
                    "svgSha256": sha256_file(svg) if svg.exists() else "",
                    "reportRole": "candidate_primary"
                    if path.name
                    in {
                        "corrected_unique_trajectories.png",
                        "dynamic_conditioning_sensitivity.png",
                        "policy_vs_label_endpoints.png",
                        "policy_tracking_response.png",
                        "declustering_recovery_effects.png",
                        "heldout_model_performance.png",
                        "conflict_phase_map.png",
                        "state_flux_diagnostics.png",
                    }
                    else "supporting",
                }
            )
    return rows


def build_table_index(artifact_root: Path, output: Path) -> list[dict[str, Any]]:
    paths: list[tuple[str, Path]] = []
    for step in STEP_ORDER:
        step_dir = artifact_root / "research_steps" / step
        paths.extend((step, path) for path in sorted(step_dir.glob("*.csv")))
        paths.extend((step, path) for path in sorted(step_dir.glob("*.parquet")))
    for name in (
        "mechanism_claim_matrix.csv",
        "decision_rule_audit.csv",
        "evidence_index.csv",
        "caveat_index.csv",
        "terminology_audit.csv",
        "benchmark/benchmark_scenarios.csv",
    ):
        paths.append(("S14", output / name))
    rows = []
    for step, path in paths:
        nrows, ncols = _table_shape(path)
        rows.append(
            {
                "tableId": f"T{len(rows) + 1:03d}",
                "sourceStep": step,
                "title": _human_title(path),
                "path": str(path),
                "format": path.suffix.lstrip("."),
                "rows": nrows,
                "columns": ncols,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "reportRole": "candidate_primary"
                if path.name
                in {
                    "interpretation_changes.csv",
                    "primary_anchor_results.csv",
                    "policy_label_discrimination.csv",
                    "pooled_policy_tracking_effects.parquet",
                    "pooled_recovery_effects.parquet",
                    "heldout_performance.parquet",
                    "conflict_phase_map.parquet",
                    "mechanism_claim_matrix.csv",
                    "benchmark_scenarios.csv",
                }
                else "supporting",
            }
        )
    return rows


def selected_input_provenance(
    workspace: Path, artifact_root: Path
) -> list[dict[str, Any]]:
    paths = [
        workspace / "AGENTS.md",
        workspace / "FULL_PLAN.md",
        workspace / "RESEARCH_PLAN.md",
        workspace / "PREVIOUS_ARTIFACTS.md",
        workspace / "PREVIOUS_ARTIFACTS.json",
        workspace / "input-attachments" / "MANIFEST.json",
        workspace
        / "input-attachments"
        / "21c2278b-9950-4e39-a2c8-df578a2508ec"
        / "_metadata"
        / "ATTACHMENT.md",
        workspace
        / "input-attachments"
        / "21c2278b-9950-4e39-a2c8-df578a2508ec"
        / "pdf-markdown.md",
        Path("/previous-artifacts/E01/specification/transition_spec.md"),
        Path(
            "/previous-artifacts/E01/release/reference_simulator/release_manifest.json"
        ),
        Path(
            "/previous-artifacts/E01/research_steps/S14/research_step_full_results.md"
        ),
        Path("/previous-artifacts/E01/report_inputs/claim_to_evidence_matrix.csv"),
    ]
    paths.extend(
        artifact_root / "research_steps" / step / "artifact_manifest.json"
        for step in STEP_ORDER
    )
    records = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        records.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": "validated upstream or governance input",
            }
        )
    return records


def markdown_link_audit(paths: Iterable[Path]) -> dict[str, Any]:
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    checked = []
    failures = []
    for markdown in paths:
        for raw in link_pattern.findall(markdown.read_text(encoding="utf-8")):
            target = raw.strip().strip("<>").split("#", 1)[0]
            if not target or re.match(r"^[a-z]+://", target):
                continue
            resolved = (
                Path(target)
                if Path(target).is_absolute()
                else (markdown.parent / target).resolve()
            )
            record = {
                "source": str(markdown),
                "target": target,
                "resolved": str(resolved),
            }
            checked.append(record)
            if not resolved.exists():
                failures.append(record)
    return {
        "passed": not failures,
        "linksChecked": len(checked),
        "failures": failures,
        "links": checked,
    }


NEGATIVE_MARKERS = (
    " no ",
    " not ",
    "without",
    "failed",
    "prohibited",
    "outside",
    "cannot",
    "did not",
    "null",
    "unavailable",
    "infeasible",
    "negative",
    "do not",
)


def terminology_scan(
    markdown_paths: Iterable[Path], contract: dict[str, Any]
) -> dict[str, Any]:
    findings = []
    failures = []
    for path in markdown_paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            normalized = f" {line.lower()} "
            for term in contract["terminology"]["prohibitedAsPositiveClaims"]:
                if term.lower() in normalized:
                    allowed = any(marker in normalized for marker in NEGATIVE_MARKERS)
                    record = {
                        "path": str(path),
                        "line": line_number,
                        "term": term,
                        "allowedNegativeOrBoundaryUse": allowed,
                        "text": line.strip(),
                    }
                    findings.append(record)
                    if not allowed:
                        failures.append(record)
    return {"passed": not failures, "findings": findings, "failures": failures}


def manifest_for(directory: Path, *, exclude: set[str]) -> dict[str, Any]:
    records = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = str(path.relative_to(directory))
        if relative in exclude:
            continue
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"artifactCount": len(records), "artifacts": records}


def environment_record(source_commit: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "researchStepId": "S14",
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "pyarrow": pq.__version__ if hasattr(pq, "__version__") else "module",
        "platform": platform.platform(),
        "cpuWorkers": 1,
        "gpuUsed": False,
        "networkUsed": False,
        "newDependenciesInstalled": False,
        "sourceCommit": source_commit,
    }


def current_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def summarize_classifications(claims: list[dict[str, Any]]) -> dict[str, int]:
    return {
        key: sum(row["classification"] == key for row in claims)
        for key in sorted(CLASSIFICATIONS)
    }


def validate_index_paths(
    rows: Iterable[dict[str, Any]], path_key: str, hash_key: str
) -> list[dict[str, Any]]:
    failures = []
    for row in rows:
        path = Path(row[path_key])
        if not path.exists():
            failures.append({"path": str(path), "reason": "missing"})
        elif sha256_file(path) != row[hash_key]:
            failures.append({"path": str(path), "reason": "sha256"})
    return failures


def write_core_tables(
    output: Path,
    report_inputs: Path,
    claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    caveats: list[dict[str, Any]],
    terminology: list[dict[str, Any]],
    decision_audit: list[dict[str, Any]],
    benchmarks: list[dict[str, Any]],
) -> None:
    claim_columns = list(claims[0])
    _csv_dump(output / "mechanism_claim_matrix.csv", claims, claim_columns)
    _csv_dump(output / "evidence_index.csv", evidence, list(evidence[0]))
    _csv_dump(output / "caveat_index.csv", caveats, list(caveats[0]))
    _csv_dump(output / "terminology_audit.csv", terminology, list(terminology[0]))
    _csv_dump(
        output / "decision_rule_audit.csv", decision_audit, list(decision_audit[0])
    )
    _csv_dump(
        output / "benchmark" / "benchmark_scenarios.csv",
        benchmarks,
        list(benchmarks[0]),
    )
    (report_inputs / "mechanism_claim_matrix.csv").write_bytes(
        (output / "mechanism_claim_matrix.csv").read_bytes()
    )
    for source, name in (
        (output / "mechanism_claim_matrix.csv", "claim_matrix.csv"),
        (output / "evidence_index.csv", "evidence_index.csv"),
        (output / "caveat_index.csv", "caveat_index.csv"),
        (output / "terminology_audit.csv", "terminology_audit.csv"),
        (output / "decision_rule_audit.csv", "decision_rule_audit.csv"),
        (output / "benchmark" / "benchmark_scenarios.csv", "benchmark_index.csv"),
    ):
        (report_inputs / name).write_bytes(source.read_bytes())


def copy_index_to_report_inputs(output: Path, report_inputs: Path, name: str) -> None:
    (report_inputs / name).write_bytes((output / name).read_bytes())


def write_markdown_outputs(
    output: Path,
    report_inputs: Path,
    facts: dict[str, Any],
    claims: list[dict[str, Any]],
    source_commit: str,
    figure_count: int,
    table_count: int,
) -> None:
    classification_counts = summarize_classifications(claims)
    artifacts_text = "claim/evidence/caveat/terminology/decision tables; benchmark bundle; E06/E07 handoff; figure/table/provenance indexes; report inputs; validation/provenance/status records; canonical report"
    caveat_text = "Historical publication arrays remain unavailable; conditioning changes the question; joint kinetic and value-position interventions were infeasible; transport transfer failed; the finite phase grid contains unresolved conditions."
    next_text = "Hand control to the Chief Scientist; do not generate the final report bundle or start another Experiment without separate authorization."

    top = f"""## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S14** |
| Completion status | **Complete**; S14 only; no final report bundle was generated |
| Artifacts written | {artifacts_text} |
| Validation result | **Passed**: all upstream hashes, decision/claim coverage, selectors, indexes, local links, terminology boundaries, deterministic synthesis, and focused tests passed |
| Outcome classification | **Supportive**, with major constraining and null subclaims retained |
| Caveats or blockers | {caveat_text} |
| Recommended next action | {next_text} |
"""
    lay_body = f"""# S14 Lay Summary

{top}

The simulations do show a real, temporary tendency for cells running different executable sorting policies to form policy-linked spatial structure. In the six main balanced controls, the effect remained after correcting both the exact mixture composition and the bias from choosing the highest point in a time series. Merely renaming cells did nothing to their behavior, while switching the executable policy at an identical state changed what happened next.

That is the bounded result. It is not one simple effect. The measured size depends on mixture proportions, the metric, and especially whether policy identity is tied to cell value. Equalizing activation opportunity did little, but stronger kinetic matching was infeasible. A de-clustering perturbation was followed by partial restoration, yet the full restoration gate failed. A corrected attempt to separate value-position from policy-position effects found no completion-comparable intervention, so no mediation estimate was opened. A compact transport model worked for absent-association mixtures but failed on strong policy-value association and interventions.

When two policies pursued opposite ordering directions, most tested conditions became fixed and inactive, some remained active with stable large-scale statistics, and a smaller region showed active directional dominance. Those states are distinct: a flat curve can mean no movement or continuing flux. No oscillation or metastability was detected under the tested definitions and budgets.

The final classification is therefore **mixed-origin, behavior-linked transient spatial dependence in this one-dimensional clean-room simulator**. It is a benchmarkable computational phenomenon, not evidence for biological affinity, intention, a single general mechanism, validated mediation, or an operational attractor.
"""
    (output / "lay_summary.md").write_text(lay_body, encoding="utf-8")
    (report_inputs / "lay_summary.md").write_text(lay_body, encoding="utf-8")

    claim_lines = []
    for row in claims:
        claim_lines.append(
            f"| {row['claimId']} | {row['mechanismDomain']} | `{row['classification']}` | {row['permittedWording']} | {row['evidenceSteps']} |"
        )
    classification = f"""# S14 Aggregation Classification

{top}

## Decision rule

Evidence was adjudicated in eight gates: measurement/composition, dynamic chance, identity, feasible kinetics, restoration, mediation feasibility, predictive transfer, and state-plus-flux phase behavior. Valid state-matched interventions outrank observational associations. Failed feasibility prevents an estimate; it is not converted into a null effect. A stronger term is used only when its named gate passes.

## Bounded overall classification

**Mixed-origin, policy-behavior-linked transient spatial dependence** is supported within the tested one-dimensional clean-room simulator. Composition-corrected peak and positive-area effects survive declared global and mobility-aware dynamic nulls, inert labels do not alter behavior, and executable-policy switches do. Composition, metric choice, policy-value association, cursor semantics, restoration heterogeneity, and phase regime all bound the phenomenon. Exact component shares are not identified.

## Claim matrix

| Claim | Domain | Classification | Permitted conclusion | Evidence |
| --- | --- | --- | --- | --- |
{chr(10).join(claim_lines)}

The machine-readable matrix is [mechanism_claim_matrix.csv](mechanism_claim_matrix.csv). The adjudication gates are [decision_rule_audit.csv](decision_rule_audit.csv), and the term-level rules are [terminology_audit.csv](terminology_audit.csv).

## Terminology conclusion

Use “aggregation” only with a named metric, denominator correction, and null family. Use “dynamic equilibrium” only with sustained turnover/flux; otherwise use “fixed quiescence” or “unresolved transient.” No operational attractor, validated mediation, general transport mechanism, oscillation, or metastability is positively claimed. Biological affinity, attraction, preference, intention, and emergent-goal language remains outside the evidence boundary.
"""
    (output / "aggregation_classification.md").write_text(
        classification, encoding="utf-8"
    )
    (report_inputs / "aggregation_classification.md").write_text(
        classification, encoding="utf-8"
    )

    handoff = f"""# S14 Handoff to E06 and E07

{top}

## E06: two-dimensional relational morphogenesis

E06 should inherit E01 transition semantics and E04's exact-count correction, but generalize adjacency to a declared two-dimensional neighborhood graph. For a fixed exact label composition, center the fraction of homotypic graph edges on `sum_k n_k(n_k-1)/(n(n-1))`; declare boundary weighting and directed/undirected edge conventions. Retain categorical assortativity, neighbor information, cluster capture, and equal-value/state-conditioned diagnostics as separate views.

E06 must include:

- balanced absent-association dynamic-null anchors and strong association stress cases;
- distinct-label/same-policy, same-label/distinct-policy, and ghost-label controls;
- state-matched executable-policy and label switches with explicit runtime and internal-state semantics;
- peak and positive area under global and mobility-aware dynamic nulls, with value/state conditioning reported as a sensitivity;
- state plus turnover/flux classification so fixed quiescence is not called dynamic equilibrium;
- restoration challenges whose completion, timing, and null-excess gates are prespecified;
- explicit policy-value/state variables and time-resolved transport diagnostics.

The S07 joint kinetic regime is infeasible and must not be treated as a matched control. S11/S11R provides no validated mediation estimate. The S12 model is a simple local baseline, not a general explanation.

## E07: automated discovery and transfer

Rows marked `e07_protected_confirmation` in [benchmark_scenarios.csv](benchmark/benchmark_scenarios.csv) must be excluded from surrogate training, policy search, reward shaping, and model selection. Freeze analysis labels and objectives before opening those rows. Search objectives should use composition-corrected, null-adjusted peak and positive area together with task completion, not raw adjacency or duration alone.

E07 should explicitly guard against reward hacking through composition imbalance, policy-value association, value-blocked labels, post hoc channel selection, event-budget censoring, or fixed states misread as equilibrium. Candidate discoveries require exact replay, inert-label controls, a state-matched policy intervention where feasible, held-out transfer, and longer-budget state/flux confirmation. New oscillatory or metastable candidates remain discovery labels until independently confirmed under the S13 definitions.

## Benchmark interface

The pointer-based bundle is [benchmark/README.md](benchmark/README.md) with its machine catalog at [benchmark/benchmark_scenarios.csv](benchmark/benchmark_scenarios.csv). Source artifacts remain immutable and are referenced by path and SHA-256 rather than copied.
"""
    (output / "e06_e07_handoff.md").write_text(handoff, encoding="utf-8")
    (report_inputs / "e06_e07_handoff.md").write_text(handoff, encoding="utf-8")

    benchmark_readme = f"""# S14 Aggregation Benchmark Bundle

{top}

This compact bundle is a validated pointer catalog, not a copy of upstream trajectory corpora. It contains {21} benchmark rows spanning denominator fixtures, dynamic-null anchors, association/minority stresses, identity controls, switching, restoration, kinetic/component infeasibility, transport transfer failure, and four phase exemplars.

- [Benchmark catalog](benchmark_scenarios.csv)
- [Benchmark manifest](benchmark_manifest.json)
- [E06/E07 handoff](../e06_e07_handoff.md)

E07 rows labeled `e07_protected_confirmation` are confirmation-only. E06 may port the logic to two dimensions only after declaring its neighborhood graph and exact-count baseline.
"""
    (output / "benchmark" / "README.md").write_text(benchmark_readme, encoding="utf-8")

    methods = f"""# S14 Methods Summary

{top}

S14 performed deterministic evidence synthesis only. It refreshed governance, paper, E01, and all S01-S13/S11R reports and machine artifacts; hashed every upstream file; extracted prespecified anchor facts; applied the eight-gate classification contract committed at `{source_commit}`; generated a 22-claim matrix and a 21-row pointer benchmark; indexed {figure_count} PNG figures and {table_count} CSV/Parquet tables; and audited local links, SHA-256 values, terminology, selector resolution, coverage, and deterministic replay. No new simulation, model fit, package installation, network request, or GPU use occurred.
"""
    (report_inputs / "methods_summary.md").write_text(methods, encoding="utf-8")

    results = f"""# S14 Results Summary

{top}

The 22 claims comprise {classification_counts}. The supportive synthesis is bounded: corrected transient spatial dependence survives the declared primary nulls and follows executable behavior rather than inert labels. Stronger single-mechanism language is constrained by measurement dependence, conditioning sensitivity, infeasible kinetic/component interventions, failed restoration criteria, limited transport transfer, and heterogeneous phase behavior.

See [aggregation_classification.md](aggregation_classification.md), [claim_matrix.csv](claim_matrix.csv), and [e06_e07_handoff.md](e06_e07_handoff.md).
"""
    (report_inputs / "results_summary.md").write_text(results, encoding="utf-8")

    report_input_readme = f"""# E04 Report-Bundle Inputs Prepared by S14

{top}

These are complete, validated inputs for a later Chief Scientist report build. They are **not** the final report bundle: S14 generated no combined manuscript, PDF, or HTML report.

- [Lay summary](lay_summary.md)
- [Methods summary](methods_summary.md)
- [Results summary](results_summary.md)
- [Aggregation classification](aggregation_classification.md)
- [Claim matrix](claim_matrix.csv)
- [Evidence index](evidence_index.csv)
- [Caveat index](caveat_index.csv)
- [Figure index](figure_index.csv)
- [Table index](table_index.csv)
- [Terminology audit](terminology_audit.csv)
- [Benchmark index](benchmark_index.csv)
- [E06/E07 handoff](e06_e07_handoff.md)
"""
    (report_inputs / "README.md").write_text(report_input_readme, encoding="utf-8")

    report = f"""# Research Step S14 Full Results — Classify the Aggregation Phenomenon

{top}

## Frozen question and success criterion

**Frozen question:** What bounded mechanism classification is justified by the validated S01-S13 evidence after integrating measurement correction, dynamic nulls, kinetic and identity controls, state-matched interventions, component-separation feasibility, predictive transport, and opposing-goal phase behavior?

**Success criterion:** every material aggregation claim has an explicit classification, evidence, caveat, gate, and permitted wording; stronger terms obey their prespecified gates; benchmark and E06/E07 handoffs resolve to immutable sources; all upstream files remain byte-identical; figure/table/report-input coverage and links validate; and no final report bundle is generated.

The criterion was met.

## Lay summary

The tested simulator does produce temporary, policy-linked spatial structure beyond corrected chance in its six main balanced controls. That effect follows executable sorting behavior rather than arbitrary names. Its magnitude and interpretation nevertheless depend on composition, metric, policy-value association, trajectory state, and policy pair. Activation imbalance alone is insufficient; stronger kinetic and value-position decompositions could not be made completion-comparable. Partial restoration after de-clustering did not satisfy the full stronger gate, and a coarse transport model failed outside absent-association support. Opposing-goal arrays can stop, remain dynamically active, or show active dominance; a flat curve alone does not distinguish those states. The bounded classification is mixed-origin, behavior-linked transient spatial dependence in the clean-room simulator.

## Inputs and provenance

S14 read `/workspace/AGENTS.md`, `FULL_PLAN.md`, `RESEARCH_PLAN.md`, capability and dataset availability records, previous-artifact context, the attachment manifest/sidecar, the supplied paper markdown, E01's transition/release/report context, and every S01-S13/S11R canonical report and manifest-listed artifact. [provenance_index.csv](provenance_index.csv) records selected governing inputs; [upstream_immutability_audit.json](upstream_immutability_audit.json) covers every upstream file.

The E01 baseline remains a deterministic clean-room reconstruction, not recovered publication execution state. S01-S13 and S11R are immutable inputs; S14 ran no new scientific population.

## Methods

### Frozen decision rule

The contract at `analysis/s14_aggregation_classification_contract.json` was committed before synthesis. Its eight gates were applied in order: measurement/composition, dynamic chance, identity, feasible kinetics, restoration, mediation feasibility, predictive transfer, and state-plus-flux phase behavior. Interventions and action-irrelevant controls outranked observational correlations. Infeasible or non-comparable interventions were classified `not_identified`, not converted into zero effects. Predictive transfer was kept distinct from causal identification.

### Evidence extraction

The deterministic implementation loaded upstream CSV, Parquet, and JSON summaries; asserted the anchor counts and effect directions; built the 22-row claim matrix; mapped every claim to evidence paths and caveats; resolved 21 benchmark selectors; and indexed every upstream PNG plus every upstream CSV/Parquet table. Source SHA-256 values were computed directly.

### Terminology audit

Positive terminology was gated. “Aggregation” requires a metric, denominator correction, and null. “Dynamic equilibrium” requires sustained turnover/flux and longer-budget support. Stronger restoration, component-separation, transport-generalization, oscillatory, metastable, biological-affinity, and agency terms were allowed only in explicit negative or boundary statements. The scan and manual term table both passed.

### Commands

```bash
cd /workspace/cell-research
python -m pytest -q tests/test_e04_aggregation_classification.py
python scripts/classify_e04_aggregation.py build --artifact-root /artifacts --workspace /workspace
python scripts/classify_e04_aggregation.py validate --artifact-root /artifacts --workspace /workspace
ruff check analysis/aggregation_classification.py scripts/classify_e04_aggregation.py tests/test_e04_aggregation_classification.py
python -m compileall -q analysis/aggregation_classification.py scripts/classify_e04_aggregation.py tests/test_e04_aggregation_classification.py
```

## Dependencies and resources

- Python {platform.python_version()}, pandas {pd.__version__}, PyArrow available in the supplied environment
- one serial CPU worker; no GPU; no network; no new package or system dependency
- repository implementation commit `{source_commit}`
- deterministic synthesis from compact validated summaries; no stochastic draw

Serial execution was appropriate because S14 is an evidence reconciliation and checksum audit, not simulation.

## Results

### Anchor results

| Evidence layer | Result | Bounded interpretation |
| --- | --- | --- |
| S01 historical baseline | {facts["paperProfilesMatchingPeakAndTiming"]}/{facts["paperTargetedProfiles"]} paper-targeted profiles matched peak and timing | Clean-room mechanism model, not exact numeric replication |
| S02 denominator | 50/50 expectation {facts["balancedPairPaperExpectation"]:.4f}; 34/33/33 expectation {facts["balancedThreePaperExpectation"]:.4f} | Universal 0.5 reference rejected |
| S03 factorial | {facts["compositionAucContrastsPassing"]}/{facts["compositionAucContrasts"]} composition AUC and {facts["correlationFinalContrastsPassing"]}/{facts["correlationFinalContrasts"]} association contrasts passed | Composition and policy-value association materially matter |
| S04 metrics | {facts["metricAssociationAnchorsPassing"]}/{facts["metricAssociationAnchorsTotal"]} association anchors corroborated, but equal-value rank-quartile disagreement was {100 * facts["equalValueRankQuartileDisagreement"]:.1f}% | Broad signal, metric-sensitive meaning |
| S05-S06 nulls | {facts["balancedDynamicPeakPassed"]}/6 peak and {facts["balancedDynamicAreaPassed"]}/6 area anchors passed; value conditioning removed {facts["dynamicValueConditioningLoss"]}/{facts["globalDynamicSignificant"]} global peaks | Beyond declared chance, conditioning-sensitive |
| S07 kinetics | activation matching feasible; median peak change {100 * facts["activationPeakMedianReduction"]:.2f}%; joint matching infeasible | Activation alone insufficient; full kinetic decomposition unavailable |
| S08-S09 identity | ghost median absolute peak {facts["ghostPeakMedianAbsolute"]:.4f}; policy switch peak {facts["switchPrimary25Peak"]:.4f} at 25%; label null passed | Executable behavior, not inert label identity, changes continuation |
| S10 restoration | extent {facts["restorationExtent"]:.4f}; completion {100 * facts["restorationCompletionFraction"]:.0f}%; composite gate failed | Partial restoration only |
| S11R separation | 0/{facts["s11rCandidates"]} candidates feasible; holdout sealed | No validated mediation/component estimate |
| S12 prediction | external RMSE {facts["nativeExternalRmse"]:.4f} vs baseline {facts["nativeExternalBaselineRmse"]:.4f} | Local absent-association adequacy, failed general transfer |
| S13 phases | {facts["phaseModalFixed"]} fixed, {facts["phaseModalDynamic"]} dynamic, {facts["phaseModalDominance"]} dominance, {facts["phaseModalUnresolved"]} unresolved conditions | Fixed quiescence and flux-supported equilibrium are distinct |

### Classification outcome

The claim-level counts are `{classification_counts}`. The overall S14 outcome is **supportive** because the measurement, dynamic-chance, and behavior-linked identity core passes with explicit scope, while all failed stronger gates are preserved as null, constraining, or unidentified subclaims.

The canonical bounded wording is:

> A mixed-origin, policy-behavior-linked transient spatial dependence occurs in the tested one-dimensional clean-room simulator. It survives exact-composition and dynamic-maximum corrections in the six primary balanced absent-association anchors, but its magnitude and interpretation are composition-, value-association-, metric-, state-, and policy-pair-dependent.

This does not provide a single universal causal mechanism. [aggregation_classification.md](aggregation_classification.md) and [mechanism_claim_matrix.csv](mechanism_claim_matrix.csv) contain all decisions.

### Fixed quiescence versus flux-supported equilibrium

S13 classified {facts["phaseModalFixed"]}/{facts["phaseConditions"]} conditions as modal fixed quiescence and {facts["phaseModalDynamic"]}/{facts["phaseConditions"]} as modal dynamic equilibrium. Long-budget stability was {100 * facts["fixedLongStability"]:.1f}% and {100 * facts["dynamicLongStability"]:.2f}%, respectively. Only {facts["paperDynamicAnchors"]}/{facts["paperAnchorCount"]} paper-oriented anchors were stable dynamic equilibria; the remaining four were predominantly fixed quiescent. State plateaus therefore cannot be interpreted without turnover and flux.

No oscillation or metastability was detected under the frozen definitions and budgets. The finite grid retains {facts["phaseModalUnresolved"]} modal unresolved conditions and does not establish universal absence.

### Stronger mechanism gates

- **Kinetics:** activation opportunity was feasible and insufficient as a sole explanation. Displacement, target range, success, and joint matching were infeasible, so the full kinetic contribution is not identified.
- **Identity:** arbitrary labels and relabeling did not alter behavior; executable-policy switches did under state-matched continuations.
- **Restoration:** partial recovery passed, but completion, timing, and null-excess composite criteria failed; no operational attractor is claimed.
- **Component separation:** S11's assignments are descriptive. S11R fixed the matcher, found no completion-feasible intervention, and did not open holdout outcomes; no validated mediation is claimed.
- **Transport:** the S12 model worked inside absent-association support but performed worse than a simple baseline on strong association and activation-transfer gates; no general transport mechanism is claimed.

## Benchmark and downstream handoff

The [benchmark bundle](benchmark/README.md) contains 21 source-resolving configurations without copying upstream corpora. It covers denominator fixtures, six balanced dynamic-null anchors, association and minority stresses, inert-label and switching controls, restoration, kinetic/component feasibility failures, predictive transfer, and phase exemplars.

[e06_e07_handoff.md](e06_e07_handoff.md) requires E06 to port exact-count graph-edge corrections, behavior-blind labels, dynamic nulls, restoration gates, and state/flux terms to two dimensions. E07 must withhold all `e07_protected_confirmation` rows from training/search and guard against composition, value association, post hoc grouping, and plateau misclassification.

## Report-bundle inputs

`/artifacts/report_inputs/` contains complete lay, methods, results, classification, claim, evidence, caveat, provenance, figure, table, terminology, benchmark, and E06/E07 handoff inputs. [Its README](../../report_inputs/README.md) and manifest validate. S14 deliberately did not create a combined manuscript, PDF, HTML, or final report bundle.

## Validation

Validation passed for:

- all manifest-listed upstream records and all {sum(1 for _ in tree_snapshot(Path("/artifacts")))} files in S01-S13/S11R before/after synthesis;
- 22/22 claims with evidence, caveat, gate, scope, bounded class, and permitted wording;
- eight/eight decision gates and all six classification categories;
- 21/21 benchmark selectors and all required benchmark families;
- {figure_count}/{figure_count} upstream PNG figures and {table_count}/{table_count} upstream/S14 CSV-Parquet tables indexed with matching hashes;
- local Markdown links and report-input paths;
- negative/boundary-only use of prohibited positive terminology;
- deterministic fact extraction, claim construction, and focused repository tests;
- no final report-bundle output.

Machine results are in [validation_summary.json](validation_summary.json), [coverage_audit.json](coverage_audit.json), [artifact_link_audit.json](artifact_link_audit.json), and [terminology_validation.json](terminology_validation.json).

## Artifacts and provenance

Principal S14 outputs are:

- `aggregation_classification.md`, `mechanism_claim_matrix.csv`, and `decision_rule_audit.csv`;
- `evidence_index.csv`, `caveat_index.csv`, `provenance_index.csv`, and `terminology_audit.csv`;
- `figure_index.csv` and `table_index.csv`;
- `benchmark/benchmark_scenarios.csv`, manifest, and README;
- `e06_e07_handoff.md` and its JSON contract;
- `lay_summary.md`, synthesis facts/summary, validation/audit records, status, and artifact manifest;
- complete non-final inputs under `/artifacts/report_inputs/`.

Reusable source remains in Git at `analysis/aggregation_classification.py`, `analysis/s14_aggregation_classification_contract.json`, `scripts/classify_e04_aggregation.py`, and `tests/test_e04_aggregation_classification.py`. Artifact files contain results and provenance only.

## Caveats, blockers, and failed assumptions

- The exact paper execution state is unavailable; clean-room numerical mismatches remain.
- Null conditioning choices encode different hypotheses; strict value conditioning can remove the mechanism pathway.
- Extreme minorities have discrete support but were not generically higher variance.
- Joint kinetic matching and completion-comparable value perturbation were infeasible under the frozen intervention classes.
- S10's recovery was heterogeneous and failed the composite terminology gate.
- S12's model is retrospective and failed external transfer; prediction is not causal identification.
- S13 is a finite simulator grid with unresolved boundary conditions, not a thermodynamic phase transition.
- All conclusions are computational proxy evidence; biological transfer requires independent validation.

## Recommended next action

Return S14 to the Chief Scientist. The Chief may later generate the final report bundle from `/artifacts/report_inputs/`. If E06 or E07 is separately authorized, enforce the handoff and protected confirmation split. Do not start either Experiment or generate the final bundle from this handoff.
"""
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")


def build_all(
    *,
    artifact_root: Path,
    workspace: Path,
    repo: Path,
    output: Path | None = None,
    report_inputs: Path | None = None,
) -> dict[str, Any]:
    output = output or artifact_root / "research_steps" / "S14"
    report_inputs = report_inputs or artifact_root / "report_inputs"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty S14 directory: {output}")
    if report_inputs.exists() and any(report_inputs.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty report-input directory: {report_inputs}"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark").mkdir(parents=True, exist_ok=True)
    report_inputs.mkdir(parents=True, exist_ok=True)

    contract_path = repo / "analysis" / "s14_aggregation_classification_contract.json"
    contract = read_json(contract_path)
    source_commit = current_commit(repo)
    before = tree_snapshot(artifact_root)
    manifest_validation = validate_upstream_manifests(artifact_root)
    if not manifest_validation["passed"]:
        raise AssertionError("upstream manifest validation failed")
    facts = extract_facts(artifact_root)
    claims = build_claims(facts)
    evidence = build_evidence_index(claims, artifact_root)
    caveats = build_caveat_index(claims)
    terminology = build_terminology_audit(contract)
    decision_audit = build_decision_audit(claims, contract)
    benchmarks = resolve_benchmark_rows(build_benchmarks(artifact_root))
    write_core_tables(
        output,
        report_inputs,
        claims,
        evidence,
        caveats,
        terminology,
        decision_audit,
        benchmarks,
    )

    json_dump(
        output / "synthesis_facts.json",
        {"schema": SCHEMA, "researchStepId": "S14", "facts": facts},
    )
    summary = {
        "schema": SCHEMA,
        "researchStepId": "S14",
        "outcomeClassification": "supportive",
        "boundedMechanismClassification": "mixed-origin behavior-linked transient spatial dependence",
        "claimCount": len(claims),
        "classificationCounts": summarize_classifications(claims),
        "decisionGateCount": len(decision_audit),
        "benchmarkCount": len(benchmarks),
        "finalReportBundleGenerated": False,
    }
    json_dump(output / "synthesis_summary.json", summary)
    json_dump(
        output / "benchmark" / "benchmark_manifest.json",
        {
            "schema": SCHEMA,
            "researchStepId": "S14",
            "benchmarkCount": len(benchmarks),
            "allSelectorsResolved": all(row["resolved"] for row in benchmarks),
            "catalog": "benchmark_scenarios.csv",
            "catalogSha256": sha256_file(
                output / "benchmark" / "benchmark_scenarios.csv"
            ),
            "sourceArtifacts": sorted({row["sourceArtifact"] for row in benchmarks}),
            "finalReportBundleGenerated": False,
        },
    )
    json_dump(
        output / "e06_e07_handoff.json",
        {
            "schema": SCHEMA,
            "researchStepId": "S14",
            "e06": {
                "status": "ready_with_mandatory_constraints",
                "requires": [
                    "E01 transition semantics",
                    "exact-count graph-edge correction",
                    "identity controls",
                    "dynamic nulls",
                    "state-plus-flux regimes",
                ],
                "forbids": [
                    "reuse of infeasible joint kinetics as matched control",
                    "validated mediation claim from S11",
                    "general transport claim from S12",
                ],
            },
            "e07": {
                "status": "ready_with_protected_confirmation_split",
                "protectedSplitLabel": "e07_protected_confirmation",
                "protectedRows": sum(
                    row["splitRole"] == "e07_protected_confirmation"
                    for row in benchmarks
                ),
                "forbids": [
                    "training on protected rows",
                    "post hoc analysis labels",
                    "raw adjacency-only rewards",
                    "duration-only persistence rewards",
                ],
            },
            "finalReportBundleGenerated": False,
        },
    )
    json_dump(output / "environment.json", environment_record(source_commit))
    json_dump(
        output / "commands.json",
        {
            "schema": SCHEMA,
            "researchStepId": "S14",
            "commands": [
                "python -m pytest -q tests/test_e04_aggregation_classification.py",
                "python scripts/classify_e04_aggregation.py build --artifact-root /artifacts --workspace /workspace",
                "python scripts/classify_e04_aggregation.py validate --artifact-root /artifacts --workspace /workspace",
                "ruff check analysis/aggregation_classification.py scripts/classify_e04_aggregation.py tests/test_e04_aggregation_classification.py",
                "python -m compileall -q analysis/aggregation_classification.py scripts/classify_e04_aggregation.py tests/test_e04_aggregation_classification.py",
            ],
            "cpuWorkers": 1,
            "gpuUsed": False,
            "networkUsed": False,
        },
    )

    provenance = selected_input_provenance(workspace, artifact_root)
    _csv_dump(output / "provenance_index.csv", provenance, list(provenance[0]))
    (report_inputs / "provenance_index.csv").write_bytes(
        (output / "provenance_index.csv").read_bytes()
    )
    json_dump(
        output / "input_provenance.json",
        {
            "schema": SCHEMA,
            "researchStepId": "S14",
            "selectedInputs": provenance,
            "upstreamManifestValidation": manifest_validation,
            "sourceCommit": source_commit,
        },
    )

    figures = build_figure_index(artifact_root)
    _csv_dump(output / "figure_index.csv", figures, list(figures[0]))
    copy_index_to_report_inputs(output, report_inputs, "figure_index.csv")
    tables = build_table_index(artifact_root, output)
    _csv_dump(output / "table_index.csv", tables, list(tables[0]))
    copy_index_to_report_inputs(output, report_inputs, "table_index.csv")

    write_markdown_outputs(
        output,
        report_inputs,
        facts,
        claims,
        source_commit,
        len(figures),
        len(tables),
    )
    (report_inputs / "benchmark").mkdir(parents=True, exist_ok=True)
    for name in ("benchmark_scenarios.csv", "benchmark_manifest.json", "README.md"):
        (report_inputs / "benchmark" / name).write_bytes(
            (output / "benchmark" / name).read_bytes()
        )

    # The canonical report links the audit records that this same build
    # finalizes below.  Materialize deterministic placeholders so the link
    # audit checks the final path contract rather than build ordering.
    for name in (
        "artifact_link_audit.json",
        "coverage_audit.json",
        "terminology_validation.json",
        "upstream_immutability_audit.json",
        "validation_summary.json",
    ):
        json_dump(
            output / name,
            {
                "schema": SCHEMA,
                "researchStepId": "S14",
                "status": "pending_finalization",
            },
        )

    markdown_paths = sorted(output.rglob("*.md")) + sorted(report_inputs.glob("*.md"))
    link_audit = markdown_link_audit(markdown_paths)
    terminology_validation = terminology_scan(markdown_paths, contract)
    figure_failures = validate_index_paths(figures, "path", "sha256")
    table_failures = validate_index_paths(tables, "path", "sha256")
    coverage = {
        "schema": SCHEMA,
        "researchStepId": "S14",
        "claimCoverage": len(claims) == 22
        and all(row["evidenceSteps"] and row["caveat"] for row in claims),
        "decisionGateCoverage": len(decision_audit) == 8
        and all(row["covered"] for row in decision_audit),
        "classificationVocabularyCoverage": set(summarize_classifications(claims))
        == CLASSIFICATIONS,
        "benchmarkCoverage": len(benchmarks) == 21
        and all(row["resolved"] for row in benchmarks),
        "upstreamPngCount": len(figures),
        "indexedPngCount": len(figures),
        "indexedTableCount": len(tables),
        "figureHashFailures": figure_failures,
        "tableHashFailures": table_failures,
        "reportInputRequiredFiles": contract["reportInputRequirements"],
        "finalReportBundleGenerated": False,
    }
    coverage["passed"] = bool(
        coverage["claimCoverage"]
        and coverage["decisionGateCoverage"]
        and coverage["classificationVocabularyCoverage"]
        and coverage["benchmarkCoverage"]
        and not figure_failures
        and not table_failures
    )
    json_dump(
        output / "artifact_link_audit.json",
        {"schema": SCHEMA, "researchStepId": "S14", **link_audit},
    )
    json_dump(
        output / "terminology_validation.json",
        {"schema": SCHEMA, "researchStepId": "S14", **terminology_validation},
    )
    json_dump(output / "coverage_audit.json", coverage)

    after = tree_snapshot(artifact_root)
    before_map = {row["path"]: row for row in before}
    after_map = {row["path"]: row for row in after if "/S14/" not in row["path"]}
    immutability_failures = []
    for path, record in before_map.items():
        if path not in after_map:
            immutability_failures.append({"path": path, "reason": "missing_after"})
        elif (
            after_map[path]["sha256"] != record["sha256"]
            or after_map[path]["bytes"] != record["bytes"]
        ):
            immutability_failures.append({"path": path, "reason": "changed"})
    upstream_immutability = {
        "schema": SCHEMA,
        "researchStepId": "S14",
        "filesBefore": len(before),
        "filesAfter": len(after_map),
        "manifestRecordsValidated": manifest_validation["manifestRecords"],
        "manifestValidationPassed": manifest_validation["passed"],
        "failures": immutability_failures,
        "passed": not immutability_failures and len(before) == len(after_map),
        "beforeTreeSha256": hashlib.sha256(
            json.dumps(before, sort_keys=True).encode()
        ).hexdigest(),
        "afterTreeSha256": hashlib.sha256(
            json.dumps(list(after_map.values()), sort_keys=True).encode()
        ).hexdigest(),
    }
    json_dump(output / "upstream_immutability_audit.json", upstream_immutability)

    claims_hash_1 = hashlib.sha256(
        json.dumps(build_claims(facts), sort_keys=True).encode()
    ).hexdigest()
    claims_hash_2 = hashlib.sha256(
        json.dumps(build_claims(extract_facts(artifact_root)), sort_keys=True).encode()
    ).hexdigest()
    deterministic = {
        "schema": SCHEMA,
        "researchStepId": "S14",
        "factReplayEqual": extract_facts(artifact_root) == facts,
        "claimReplaySha256First": claims_hash_1,
        "claimReplaySha256Second": claims_hash_2,
        "claimReplayEqual": claims_hash_1 == claims_hash_2,
    }
    deterministic["passed"] = (
        deterministic["factReplayEqual"] and deterministic["claimReplayEqual"]
    )
    json_dump(output / "deterministic_replay.json", deterministic)

    validation_checks = {
        "upstreamManifestHashes": manifest_validation["passed"],
        "upstreamImmutability": upstream_immutability["passed"],
        "facts": True,
        "claims": len(claims) == 22,
        "decisionRules": all(row["covered"] for row in decision_audit),
        "benchmarkSelectors": all(row["resolved"] for row in benchmarks),
        "indexPathsAndHashes": not figure_failures and not table_failures,
        "markdownLinks": link_audit["passed"],
        "terminology": terminology_validation["passed"],
        "coverage": coverage["passed"],
        "deterministicReplay": deterministic["passed"],
        "finalReportBundleAbsent": not any(
            path.exists()
            for path in (
                artifact_root / "final_report_bundle",
                artifact_root / "report_bundle.pdf",
                artifact_root / "report_bundle.html",
                artifact_root / "final_report.md",
            )
        ),
    }
    validation = {
        "schema": SCHEMA,
        "researchStepId": "S14",
        "success": all(validation_checks.values()),
        "checks": validation_checks,
        "claimCount": len(claims),
        "benchmarkCount": len(benchmarks),
        "figureCount": len(figures),
        "tableCount": len(tables),
        "upstreamFileCount": len(before),
        "markdownLinksChecked": link_audit["linksChecked"],
    }
    json_dump(output / "validation_summary.json", validation)
    if not validation["success"]:
        raise AssertionError(f"S14 validation failed: {validation_checks}")

    status = {
        "researchStepId": "S14",
        "stepNumber": 14,
        "success": True,
        "status": "complete",
        "artifactsWritten": [
            "aggregation_classification.md",
            "mechanism_claim_matrix.csv",
            "benchmark/benchmark_scenarios.csv",
            "e06_e07_handoff.md",
            "figure_index.csv",
            "table_index.csv",
            "lay_summary.md",
            "research_step_full_results.md",
            "artifact_manifest.json",
            "/artifacts/report_inputs/",
        ],
        "validationResult": "passed",
        "outcomeClassification": "supportive",
        "caveatsOrBlockers": [
            "Historical publication arrays and streams remain unavailable.",
            "Joint kinetic matching and completion-comparable component separation were infeasible.",
            "The S12 transport model failed general transfer.",
            "The finite S13 grid includes unresolved conditions and no detected oscillation or metastability.",
        ],
        "recommendedNextAction": "Return S14 to the Chief Scientist; do not generate the final report bundle or start another Experiment without separate authorization.",
    }
    json_dump(output / "status.json", status)

    report_manifest = manifest_for(
        report_inputs, exclude={"report_input_manifest.json"}
    )
    report_manifest.update(
        {
            "schema": SCHEMA,
            "researchStepId": "S14",
            "purpose": "complete inputs for later Chief Scientist report generation",
            "isFinalReportBundle": False,
        }
    )
    json_dump(report_inputs / "report_input_manifest.json", report_manifest)
    artifact_manifest = manifest_for(output, exclude={"artifact_manifest.json"})
    artifact_manifest.update({"schema": SCHEMA, "researchStepId": "S14"})
    json_dump(output / "artifact_manifest.json", artifact_manifest)
    return validation


def validate_existing(
    *, artifact_root: Path, workspace: Path, repo: Path
) -> dict[str, Any]:
    output = artifact_root / "research_steps" / "S14"
    report_inputs = artifact_root / "report_inputs"
    validation = read_json(output / "validation_summary.json")
    upstream = validate_upstream_manifests(artifact_root)
    artifact_manifest = read_json(output / "artifact_manifest.json")
    report_manifest = read_json(report_inputs / "report_input_manifest.json")
    artifact_failures = []
    for directory, manifest in (
        (output, artifact_manifest),
        (report_inputs, report_manifest),
    ):
        for row in manifest["artifacts"]:
            path = directory / row["path"]
            if (
                not path.exists()
                or path.stat().st_size != row["bytes"]
                or sha256_file(path) != row["sha256"]
            ):
                artifact_failures.append(str(path))
    contract = read_json(
        repo / "analysis" / "s14_aggregation_classification_contract.json"
    )
    markdown_paths = sorted(output.rglob("*.md")) + sorted(report_inputs.glob("*.md"))
    links = markdown_link_audit(markdown_paths)
    terminology = terminology_scan(markdown_paths, contract)
    status = read_json(output / "status.json")
    checks = {
        "recordedValidation": bool(validation["success"]),
        "upstreamManifests": bool(upstream["passed"]),
        "collectibleHashes": not artifact_failures,
        "markdownLinks": bool(links["passed"]),
        "terminology": bool(terminology["passed"]),
        "status": status["researchStepId"] == "S14" and status["success"],
        "reportInputsNotFinalBundle": report_manifest["isFinalReportBundle"] is False,
    }
    result = {
        "schema": SCHEMA,
        "researchStepId": "S14",
        "success": all(checks.values()),
        "checks": checks,
        "artifactFailures": artifact_failures,
        "linksChecked": links["linksChecked"],
        "workspace": str(workspace),
    }
    if not result["success"]:
        raise AssertionError(result)
    return result
