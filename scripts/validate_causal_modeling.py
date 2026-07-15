#!/usr/bin/env python3
"""Validate S12 contracts, causal estimates, models, and diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from causal_simulator.causal_modeling import (  # noqa: E402
    ADDITIVE_S01_COSTS,
    COUPLING,
    ENDPOINTS,
    ESTIMANDS,
    canonical_json_bytes,
    sha256_file,
)


DEFAULT_OUTPUT = Path("/artifacts/research_steps/S12")
S11 = Path("/artifacts/research_steps/S11")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate(args: argparse.Namespace) -> None:
    output = args.output
    freeze = read_json(output / "model_freeze_manifest.json")
    spec = read_json(output / "modeling_prespecification.json")
    stability = read_json(output / "bootstrap_stability.json")
    fit_summary = read_json(output / "model_fit_summary.json")
    tradeoff = read_json(output / "e04_resource_tradeoff.json")
    inherited_holdout = read_json(S11 / "holdout_integrity_validation.json")
    inherited_ledger = read_json(S11 / "ledger_identity_validation.json")
    inherited_replay = read_json(S11 / "deterministic_replay_validation.json")

    arm = pd.read_parquet(output / "model_arm_data.parquet")
    paired = pd.read_parquet(output / "paired_analysis_data.parquet")
    marginal = pd.read_parquet(output / "marginal_effects.parquet")
    heterogeneity = pd.read_parquet(output / "heterogeneity_effects.parquet")
    two_way = pd.read_parquet(output / "two_way_heterogeneity.parquet")
    omnibus = pd.read_parquet(output / "heterogeneity_omnibus.parquet")
    rankings = pd.read_parquet(output / "heterogeneity_rankings.parquet")
    unsupported = pd.read_parquet(output / "unsupported_heterogeneity_cells.parquet")
    rng = pd.read_parquet(output / "rng_unpaired_sensitivity.parquet")
    components = pd.read_parquet(output / "e04_cost_components.parquet")
    cost_sensitivity = pd.read_parquet(output / "cost_transform_sensitivity.parquet")
    leave_scale = pd.read_parquet(output / "leave_one_scale_out.parquet")
    diagnostics = pd.read_parquet(output / "model_diagnostics.parquet")
    calibration = pd.read_parquet(output / "calibration_diagnostics.parquet")
    survival_diagnostics = pd.read_parquet(output / "survival_diagnostics.parquet")
    survival = pd.read_parquet(output / "survival_summaries.parquet")
    standardized = pd.read_parquet(output / "model_standardized_effects.parquet")
    piecewise_path = output / "piecewise_completion_rates.parquet"
    piecewise = pd.read_parquet(piecewise_path) if piecewise_path.exists() else pd.DataFrame()
    s11_primary = pd.read_parquet(S11 / "primary_effects.parquet")

    additive_sum = sum(arm[f"cost_{name}"] for name in ADDITIVE_S01_COSTS)
    ledger_checks = {
        "oneProposalPerActivation": bool((arm.cost_activations == arm.cost_proposals).all()),
        "proposalPartition": bool((
            arm.cost_proposals
            == arm.cost_noOps + arm.cost_rejections + arm.cost_memoryUpdates
            + arm.cost_acceptedSwaps + arm.cost_conflictLosses
        ).all()),
        "displacementIdentity": bool((arm.cost_displacedCells == 2 * arm.cost_acceptedSwaps).all()),
        "readAliases": bool((
            (arm.cost_valueReads == arm.cost_observationRecordReads)
            & (arm.cost_statusReads == arm.cost_observationRecordReads)
        ).all()),
        "candidateIdentity": bool((
            (arm.cost_targetCalculations == arm.cost_policyCandidateConstructions)
            & (
                arm.cost_targetCalculations
                == arm.cost_proposals - arm.cost_deferredRetryEnvelopeReuses
            )
            & (arm.cost_deferredRetryEnvelopeReuses == arm.cost_retryAttempts)
        ).all()),
        "failureIdentity": bool((
            arm.cost_failedProposals == arm.cost_rejections + arm.cost_conflictLosses
        ).all()),
        "coordinatorIdentity": bool((
            (arm.cost_coordinatorMessages == 2 * arm.cost_coordinatorEligibleDecisions)
            & (
                arm.cost_coordinatorCandidateEvaluations
                == arm.cost_coordinatorEligibleDecisions
            )
        ).all()),
        "schedulerCandidateInspectionsZero": bool((arm.cost_schedulerCandidateInspections == 0).all()),
        "s01ProjectionIdentity": bool((arm.projection_s01UnitWeightFullCost == additive_sum).all()),
        "zeroCostControl": bool((arm.projection_zeroCostControl == 0).all()),
        "opportunityIdentity": bool((
            (arm.activationCount == arm.cost_activations)
            & (arm.completionOpportunity == arm.cost_activations)
        ).all()),
    }

    primary_comparison = s11_primary[["estimandId", "endpoint", "estimate"]].merge(
        marginal[["estimandId", "endpoint", "estimate"]],
        on=["estimandId", "endpoint"], how="left", validate="one_to_one",
        suffixes=("S11", "S12"),
    )
    expected_omnibus = (
        omnibus.groupby("endpoint", observed=True).size().to_dict()
    )
    primary_diagnostics = diagnostics[
        diagnostics.modelFamily != "fractional_logit_sensitivity"
    ]
    incomplete_calibration = calibration[~calibration.completeOutOfFoldPrediction]
    calibration_disposition = bool(
        len(incomplete_calibration) == 0
        or incomplete_calibration.foldFailuresJson.str.contains(
            "nonfinite_predictions", regex=False
        ).all()
    )
    ph_trigger = survival_diagnostics.treatmentPhHolmP < 0.01
    piecewise_counts = (
        piecewise.groupby("estimandId", observed=True).size().to_dict()
        if len(piecewise) else {}
    )
    ph_rule_pass = True
    for row in survival_diagnostics.itertuples(index=False):
        if row.phAlternativeTriggered:
            ph_rule_pass &= piecewise_counts.get(row.estimandId, 0) == 3
            ph_rule_pass &= not row.singleHazardRatioInterpretable
        else:
            ph_rule_pass &= piecewise_counts.get(row.estimandId, 0) == 0
            ph_rule_pass &= row.singleHazardRatioInterpretable == row.inferenceValid

    checks = {
        "modelFreezePassedBeforeModeling": bool(
            freeze["success"] and freeze["frozenBeforeOutcomeModeling"]
            and freeze["outcomeModelFitsBeforeFreeze"] == 0
        ),
        "artifactPrespecificationHashUnchanged": (
            sha256_file(output / "modeling_prespecification.json")
            == freeze["modelingPrespecificationSha256"]
        ),
        "repositoryPrespecificationExactCopy": (
            (output / "modeling_prespecification.json").read_bytes()
            == (REPOSITORY / "design/s12/modeling_prespecification.json").read_bytes()
        ),
        "exactS11AnalysisPopulation": bool(
            len(arm) == 16000 and len(paired) == 8000
            and arm.runDesignId.nunique() == 12000
            and set(arm.confirmatoryLook) == {1, 2}
            and set(arm.estimandId) == set(ESTIMANDS)
            and (arm.groupby(["estimandId", "contrastRole"]).size() == 2000).all()
            and (paired.groupby("estimandId").size() == 2000).all()
        ),
        "protectedSupportAndHoldoutIntegrity": bool(
            arm.protected.all() and set(arm.split) == {"confirmatory_holdout"}
            and inherited_holdout["success"]
        ),
        "s11PrimaryEstimandsPreserved": bool(
            len(primary_comparison) == 4
            and primary_comparison.estimateS12.notna().all()
            and np.allclose(
                primary_comparison.estimateS11,
                primary_comparison.estimateS12,
                atol=1e-12,
            )
        ),
        "couplingClassificationsPreserved": bool(
            all(
                set(arm.loc[arm.estimandId == estimand, "couplingClassification"])
                == {classification}
                for estimand, classification in COUPLING.items()
            )
            and set(rng.estimandId) == {"E02-S01-E03", "E02-S01-E06"}
            and len(rng) == 10
        ),
        "completeLedgerIdentity": bool(
            inherited_ledger["success"] and all(ledger_checks.values())
            and set(arm.costSchemaVersion) == {"E02.complete-cost-ledger.v1"}
        ),
        "marginalFamilyComplete": bool(
            len(marginal) == 20 and set(marginal.endpoint) == set(ENDPOINTS)
            and set(marginal.estimandId) == set(ESTIMANDS)
            and (marginal.pairCount == 2000).all()
            and np.isfinite(marginal[
                ["estimate", "bootstrapStandardError", "bootstrapLow95", "bootstrapHigh95", "holmAdjustedP"]
            ]).all().all()
        ),
        "multiplicityFamiliesComplete": bool(
            expected_omnibus == {endpoint: 28 for endpoint in ENDPOINTS}
            and np.isfinite(omnibus[["unadjustedP", "bhAdjustedP"]]).all().all()
            and len(components) == 12
            and np.isfinite(components[["unadjustedBootstrapP", "bhAdjustedP"]]).all().all()
        ),
        "supportedHeterogeneityOnly": bool(
            len(heterogeneity) == 460 and heterogeneity.supported.all()
            and heterogeneity.pairCount.min() >= 100
            and len(two_way) == 820 and two_way.supported.all()
            and two_way.pairCount.min() >= 50
            and len(rankings) == 140 and len(unsupported) == 0
        ),
        "noExtrapolationOrOutcomeSelection": bool(
            spec["analysisPopulations"]["heterogeneity"]["unsupportedRule"].startswith("Do not estimate")
            and fit_summary["noFormulaSelection"]
            and freeze["checks"]["noOutcomeSelection"]
        ),
        "bootstrapStable": bool(
            stability["success"] and stability["rowsTotal"] == 20
            and stability["pointStabilityPassed"] == 20
            and stability["widthStabilityPassed"] == 20
        ),
        "continuousAndBinaryDesignsValid": bool(
            len(primary_diagnostics) == 12
            and primary_diagnostics.fullRank.all()
            and primary_diagnostics.conditionPass.all()
            and primary_diagnostics.chosenInferenceValid.all()
        ),
        "frozenModelAlternativesOnly": bool(
            set(
                diagnostics.loc[
                    diagnostics.alternativeTriggered, ["estimandId", "outcome", "chosenModel"]
                ].itertuples(index=False, name=None)
            ) == {
                ("E02-S01-E04", "normalizedResidualError", "gee_gaussian_exchangeable"),
                ("E02-S01-E04", "successByBudget", "gee_binomial_independence"),
            }
        ),
        "residualDiagnosticsComplete": bool(
            len(diagnostics[diagnostics.modelFamily == "continuous_hierarchical"]) == 8
            and np.isfinite(
                diagnostics.loc[
                    diagnostics.modelFamily == "continuous_hierarchical",
                    ["residualFittedSpearman", "breuschPaganLmP", "maximumAbsoluteScaleMeanResidual", "residualQ01", "residualQ99"],
                ]
            ).all().all()
        ),
        "calibrationAuditHasExplicitDisposition": bool(
            len(calibration) == 12 and calibration_disposition
            and calibration.predictionsAvailable.between(0, 4000).all()
        ),
        "survivalAssumptionsAndFrozenAlternatives": bool(
            len(survival_diagnostics) == 4
            and survival_diagnostics.designRank.eq(survival_diagnostics.designColumns).all()
            and (survival_diagnostics.phAlternativeTriggered == ph_trigger).all()
            and ph_rule_pass and len(survival) == 8
            and survival.completionCumulativeIncidenceAtBudget.between(0, 1).all()
            and survival.restrictedMeanCompletionFreeBudgetFraction.between(0, 1).all()
        ),
        "sensitivitySuiteComplete": bool(
            len(standardized) == 16 and len(cost_sensitivity) == 16
            and len(leave_scale) == 100 and len(rng) == 10
        ),
        "e04ResourceTradeoffComplete": bool(
            tradeoff["estimandId"] == "E02-S01-E04"
            and tradeoff["pairCount"] == 2000
            and tradeoff["successRiskDifference"] > 0
            and tradeoff["rawS01UnitWeightFullCostDifference"] > 0
            and tradeoff["incrementalRawS01CostPerSuccessPercentagePoint"] > 0
        ),
        "inheritedExactReplayStillValid": bool(inherited_replay["success"]),
        "noS13ArtifactStarted": not Path("/artifacts/research_steps/S13").exists(),
    }
    diagnostic_findings = {
        "chosenPrimaryModelsValid": int(primary_diagnostics.chosenInferenceValid.sum()),
        "chosenPrimaryModelRows": len(primary_diagnostics),
        "continuousFallbacksTriggered": fit_summary["continuousAlternativesTriggered"],
        "binaryFallbacksTriggered": fit_summary["binaryAlternativesTriggered"],
        "fractionalSensitivityModelsValid": fit_summary["sensitivityModelInferenceValid"],
        "fractionalSensitivityModelRows": fit_summary["sensitivityModelRows"],
        "completeCalibrationRows": int(calibration.completeOutOfFoldPrediction.sum()),
        "calibrationRows": len(calibration),
        "incompleteCalibrationRows": incomplete_calibration[
            ["estimandId", "outcome", "predictionsAvailable", "foldFailuresJson"]
        ].to_dict(orient="records"),
        "coxInferenceValid": int(survival_diagnostics.inferenceValid.sum()),
        "coxRows": len(survival_diagnostics),
        "phAlternativesTriggered": int(survival_diagnostics.phAlternativeTriggered.sum()),
        "phAlternativeEstimands": survival_diagnostics.loc[
            survival_diagnostics.phAlternativeTriggered, "estimandId"
        ].tolist(),
        "heteroskedasticContinuousRows": int((
            diagnostics.loc[
                diagnostics.modelFamily == "continuous_hierarchical", "breuschPaganLmP"
            ] < 0.05
        ).sum()),
        "leaveOneScaleOutSignChanges": int((~leave_scale.signPreserved).sum()),
        "unsupportedHeterogeneityCells": len(unsupported),
    }
    summary = {
        "schemaVersion": "e02.s12.validation_summary.v1",
        "researchStepId": "S12",
        "stepNumber": 12,
        "success": bool(all(checks.values())),
        "status": "complete_with_declared_model_limitations" if all(checks.values()) else "validation_failed",
        "checks": checks,
        "ledgerChecks": ledger_checks,
        "diagnosticFindings": diagnostic_findings,
        "validationResult": (
            "PASS_WITH_DECLARED_MODEL_LIMITATIONS"
            if all(checks.values()) else "FAIL"
        ),
    }
    (output / "validation_summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    print(json.dumps({
        "success": summary["success"],
        "checks": len(checks),
        "diagnosticFindings": diagnostic_findings,
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
