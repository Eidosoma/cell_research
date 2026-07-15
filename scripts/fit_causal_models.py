#!/usr/bin/env python3
"""Fit the frozen S12 continuous, binary, and completion model families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd
import patsy
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.formula.api import gee, mixedlm
from statsmodels.genmod.cov_struct import Exchangeable, Independence
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.stats.diagnostic import het_breuschpagan

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from causal_simulator.causal_modeling import (
    ESTIMANDS,
    MODIFIERS,
    canonical_json_bytes,
    hash_fold,
    holm_adjust,
    interval_specific_completion_rates,
)


DEFAULT_OUTPUT = Path("/artifacts/research_steps/S12")
MODIFIER_FORMULA = (
    "C(n)+C(valueProfile)+C(orderStructure)+C(policyProfile)+"
    "C(direction)+C(placementClass)+C(faultCount)"
)
FULL_RIGHT_HAND_SIDE = (
    f"treatment + {MODIFIER_FORMULA} + treatment:({MODIFIER_FORMULA})"
)
PH_MAIN = (
    "C(valueProfile)+C(orderStructure)+C(policyProfile)+C(direction)+"
    "C(placementClass)+C(faultCount)"
)
PH_RIGHT_HAND_SIDE = (
    f"treatment + {PH_MAIN} + treatment:C(n) + treatment:({PH_MAIN})"
)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), path,
        compression="zstd", compression_level=9, use_dictionary=True,
        write_statistics=True, version="2.6",
    )


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def finite_result(result: Any) -> tuple[bool, bool]:
    params = np.asarray(result.params, dtype=float)
    try:
        covariance = np.asarray(result.cov_params(), dtype=float)
    except Exception:
        covariance = np.asarray([[np.nan]])
    return bool(np.isfinite(params).all()), bool(np.isfinite(covariance).all())


def coefficient_rows(
    result: Any,
    *,
    estimand: str,
    outcome: str,
    model_type: str,
    chosen: bool,
    inference_valid: bool,
) -> list[dict[str, Any]]:
    names = list(getattr(result.model, "exog_names", []))
    params = np.asarray(result.params, dtype=float)
    bse = np.asarray(getattr(result, "bse", np.full(len(params), np.nan)), dtype=float)
    pvalues = np.asarray(getattr(result, "pvalues", np.full(len(params), np.nan)), dtype=float)
    rows = []
    for index, estimate in enumerate(params[:len(names)]):
        rows.append({
            "schemaVersion": "e02.s12.model_coefficient.v1",
            "estimandId": estimand, "outcome": outcome,
            "modelType": model_type, "term": names[index],
            "estimate": float(estimate),
            "standardError": float(bse[index]) if index < len(bse) and np.isfinite(bse[index]) else np.nan,
            "pValue": float(pvalues[index]) if index < len(pvalues) and np.isfinite(pvalues[index]) else np.nan,
            "chosenByFrozenDiagnosticRule": chosen,
            "coefficientInferenceValid": inference_valid,
            "claimType": "model_based_associational_heterogeneity" if ":" in names[index] else "model_parameter",
        })
    return rows


def standardized_prediction_effect(result: Any, data: pd.DataFrame) -> float:
    baseline = data[data.treatment == 0].copy()
    active = baseline.copy()
    active["treatment"] = 1
    reference = baseline.copy()
    reference["treatment"] = 0
    return float(np.mean(result.predict(active) - result.predict(reference)))


def design_diagnostics(formula: str, data: pd.DataFrame) -> dict[str, Any]:
    _, design = patsy.dmatrices(formula, data=data, return_type="dataframe")
    return {
        "rows": int(design.shape[0]), "columns": int(design.shape[1]),
        "rank": int(np.linalg.matrix_rank(design.to_numpy())),
        "conditionNumber": float(np.linalg.cond(design.to_numpy())),
        "fullRank": bool(np.linalg.matrix_rank(design.to_numpy()) == design.shape[1]),
        "conditionPass": bool(np.linalg.cond(design.to_numpy()) < 10000),
    }


def fit_gee_model(formula: str, data: pd.DataFrame, family: Any, covariance: str) -> tuple[Any, list[str]]:
    structure = Exchangeable() if covariance == "exchangeable" else Independence()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = gee(
            formula, "pairingBlockId", data,
            cov_struct=structure, family=family,
        ).fit(maxiter=500)
    return result, [str(item.message) for item in caught]


def continuous_models(arm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, str], tuple[str, Any]]]:
    coefficients = []
    diagnostics = []
    standardized = []
    chosen_models: dict[tuple[str, str], tuple[str, Any]] = {}
    for estimand in ESTIMANDS:
        data = arm[arm.estimandId == estimand].copy()
        for outcome in ("normalizedResidualError", "logS01UnitWeightFullCost"):
            formula = f"{outcome} ~ {FULL_RIGHT_HAND_SIDE}"
            design = design_diagnostics(formula, data)
            mixed_result = None
            mixed_messages: list[str] = []
            mixed_exception = None
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    mixed_result = mixedlm(
                        formula, data, groups=data.pairingBlockId, re_formula="1"
                    ).fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
                mixed_messages = [str(item.message) for item in caught]
            except Exception as error:
                mixed_exception = f"{type(error).__name__}: {error}"
            mixed_params_finite = mixed_cov_finite = False
            mixed_converged = False
            random_variance = np.nan
            if mixed_result is not None:
                mixed_params_finite, mixed_cov_finite = finite_result(mixed_result)
                mixed_converged = bool(mixed_result.converged)
                random_variance = float(mixed_result.cov_re.iloc[0, 0])
            mixed_valid = bool(
                mixed_result is not None and mixed_converged and mixed_params_finite
                and mixed_cov_finite and random_variance >= 1e-8
            )

            gee_result, gee_messages = fit_gee_model(formula, data, Gaussian(), "exchangeable")
            gee_params_finite, gee_cov_finite = finite_result(gee_result)
            gee_valid = bool(gee_result.converged and gee_params_finite and gee_cov_finite)
            chosen_type = "mixedlm_random_pair_intercept" if mixed_valid else "gee_gaussian_exchangeable"
            chosen = mixed_result if mixed_valid else gee_result
            chosen_valid = mixed_valid if mixed_valid else gee_valid
            chosen_models[(estimand, outcome)] = (chosen_type, chosen)
            if mixed_result is not None:
                coefficients.extend(coefficient_rows(
                    mixed_result, estimand=estimand, outcome=outcome,
                    model_type="mixedlm_random_pair_intercept", chosen=mixed_valid,
                    inference_valid=mixed_valid,
                ))
            coefficients.extend(coefficient_rows(
                gee_result, estimand=estimand, outcome=outcome,
                model_type="gee_gaussian_exchangeable", chosen=not mixed_valid,
                inference_valid=gee_valid,
            ))

            fitted = np.asarray(chosen.predict(data), dtype=float)
            residual = data[outcome].to_numpy(dtype=float) - fitted
            spearman = stats.spearmanr(fitted, residual)
            _, exog = patsy.dmatrices(formula, data=data, return_type="dataframe")
            bp = het_breuschpagan(residual, exog.to_numpy())
            scale_means = pd.DataFrame({"n": data.n.to_numpy(), "residual": residual}).groupby("n").residual.mean()
            diagnostics.append({
                "schemaVersion": "e02.s12.model_diagnostic.v1",
                "estimandId": estimand, "outcome": outcome,
                "modelFamily": "continuous_hierarchical",
                "frozenPrimaryModel": "mixedlm_random_pair_intercept",
                "chosenModel": chosen_type,
                "alternativeTriggered": not mixed_valid,
                "alternativeReason": (
                    mixed_exception or (
                        f"converged={mixed_converged};paramsFinite={mixed_params_finite};"
                        f"covarianceFinite={mixed_cov_finite};randomVariance={random_variance}"
                    )
                ) if not mixed_valid else None,
                "mixedWarningsJson": json.dumps(mixed_messages, sort_keys=True),
                "geeWarningsJson": json.dumps(gee_messages, sort_keys=True),
                "chosenInferenceValid": chosen_valid,
                "designRows": design["rows"], "designColumns": design["columns"],
                "designRank": design["rank"], "conditionNumber": design["conditionNumber"],
                "fullRank": design["fullRank"], "conditionPass": design["conditionPass"],
                "randomInterceptVariance": random_variance,
                "residualFittedSpearman": float(spearman.statistic),
                "residualFittedSpearmanP": float(spearman.pvalue),
                "breuschPaganLmP": float(bp[1]), "breuschPaganFP": float(bp[3]),
                "maximumAbsoluteScaleMeanResidual": float(scale_means.abs().max()),
                "residualMedian": float(np.median(residual)),
                "residualQ01": float(np.quantile(residual, 0.01)),
                "residualQ99": float(np.quantile(residual, 0.99)),
            })
            standardized.append({
                "estimandId": estimand, "outcome": outcome,
                "modelType": chosen_type,
                "standardizedActiveMinusReference": standardized_prediction_effect(chosen, data),
                "inferenceValid": chosen_valid,
                "claimType": "model_standardized_total_effect_sensitivity",
            })

            if outcome == "normalizedResidualError":
                fractional, frac_messages = fit_gee_model(
                    formula, data, Binomial(), "exchangeable"
                )
                frac_params, frac_cov = finite_result(fractional)
                coefficients.extend(coefficient_rows(
                    fractional, estimand=estimand, outcome=outcome,
                    model_type="fractional_logit_gee_sensitivity", chosen=False,
                    inference_valid=bool(fractional.converged and frac_params and frac_cov),
                ))
                diagnostics.append({
                    "schemaVersion": "e02.s12.model_diagnostic.v1",
                    "estimandId": estimand, "outcome": outcome,
                    "modelFamily": "fractional_logit_sensitivity",
                    "frozenPrimaryModel": "sensitivity_only", "chosenModel": "fractional_logit_gee_sensitivity",
                    "alternativeTriggered": False, "alternativeReason": None,
                    "mixedWarningsJson": "[]", "geeWarningsJson": json.dumps(frac_messages),
                    "chosenInferenceValid": bool(fractional.converged and frac_params and frac_cov),
                    "designRows": design["rows"], "designColumns": design["columns"],
                    "designRank": design["rank"], "conditionNumber": design["conditionNumber"],
                    "fullRank": design["fullRank"], "conditionPass": design["conditionPass"],
                    "randomInterceptVariance": np.nan, "residualFittedSpearman": np.nan,
                    "residualFittedSpearmanP": np.nan, "breuschPaganLmP": np.nan,
                    "breuschPaganFP": np.nan, "maximumAbsoluteScaleMeanResidual": np.nan,
                    "residualMedian": np.nan, "residualQ01": np.nan, "residualQ99": np.nan,
                })
                standardized.append({
                    "estimandId": estimand, "outcome": outcome,
                    "modelType": "fractional_logit_gee_sensitivity",
                    "standardizedActiveMinusReference": standardized_prediction_effect(fractional, data),
                    "inferenceValid": bool(fractional.converged and frac_params and frac_cov),
                    "claimType": "model_standardized_sensitivity",
                })
    return pd.DataFrame(coefficients), pd.DataFrame(diagnostics), pd.DataFrame(standardized), chosen_models


def success_models(arm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, tuple[str, Any]]]:
    coefficients = []
    diagnostics = []
    standardized = []
    chosen_models: dict[str, tuple[str, Any]] = {}
    for estimand in ESTIMANDS:
        data = arm[arm.estimandId == estimand].copy()
        formula = f"successByBudget ~ {FULL_RIGHT_HAND_SIDE}"
        design = design_diagnostics(formula, data)
        exchangeable, exchange_messages = fit_gee_model(formula, data, Binomial(), "exchangeable")
        ex_params, ex_cov = finite_result(exchangeable)
        exchange_valid = bool(exchangeable.converged and ex_params and ex_cov)
        independence = None
        independence_messages: list[str] = []
        if not exchange_valid:
            independence, independence_messages = fit_gee_model(
                formula, data, Binomial(), "independence"
            )
        chosen = exchangeable if exchange_valid else independence
        chosen_type = "gee_binomial_exchangeable" if exchange_valid else "gee_binomial_independence"
        if chosen is None:
            raise AssertionError("frozen binary alternative did not return a model")
        chosen_params, chosen_cov = finite_result(chosen)
        chosen_valid = bool(chosen.converged and chosen_params and chosen_cov)
        chosen_models[estimand] = (chosen_type, chosen)
        coefficients.extend(coefficient_rows(
            exchangeable, estimand=estimand, outcome="successByBudget",
            model_type="gee_binomial_exchangeable", chosen=exchange_valid,
            inference_valid=exchange_valid,
        ))
        if independence is not None:
            coefficients.extend(coefficient_rows(
                independence, estimand=estimand, outcome="successByBudget",
                model_type="gee_binomial_independence", chosen=True,
                inference_valid=chosen_valid,
            ))
        diagnostics.append({
            "schemaVersion": "e02.s12.model_diagnostic.v1",
            "estimandId": estimand, "outcome": "successByBudget",
            "modelFamily": "binary_logistic", "frozenPrimaryModel": "gee_binomial_exchangeable",
            "chosenModel": chosen_type, "alternativeTriggered": not exchange_valid,
            "alternativeReason": (
                f"converged={exchangeable.converged};paramsFinite={ex_params};covarianceFinite={ex_cov}"
                if not exchange_valid else None
            ),
            "exchangeableWarningsJson": json.dumps(exchange_messages),
            "independenceWarningsJson": json.dumps(independence_messages),
            "chosenInferenceValid": chosen_valid,
            "designRows": design["rows"], "designColumns": design["columns"],
            "designRank": design["rank"], "conditionNumber": design["conditionNumber"],
            "fullRank": design["fullRank"], "conditionPass": design["conditionPass"],
        })
        standardized.append({
            "estimandId": estimand, "outcome": "successByBudget",
            "modelType": chosen_type,
            "standardizedActiveMinusReference": standardized_prediction_effect(chosen, data),
            "inferenceValid": chosen_valid,
            "claimType": "model_standardized_total_effect_sensitivity",
        })
    return pd.DataFrame(coefficients), pd.DataFrame(diagnostics), pd.DataFrame(standardized), chosen_models


def calibration_metrics(observed: np.ndarray, predicted: np.ndarray, family: str) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if family == "continuous":
        design = sm.add_constant(predicted)
        calibration = sm.OLS(observed, design).fit()
        return {
            "rmse": float(np.sqrt(np.mean((observed - predicted) ** 2))),
            "mae": float(np.mean(np.abs(observed - predicted))),
            "calibrationIntercept": float(calibration.params[0]),
            "calibrationSlope": float(calibration.params[1]),
        }
    clipped = np.clip(predicted, 1e-8, 1 - 1e-8)
    logit = np.log(clipped / (1 - clipped))
    try:
        calibration = sm.GLM(observed, sm.add_constant(logit), family=sm.families.Binomial()).fit()
        intercept, slope = calibration.params
    except Exception:
        intercept = slope = np.nan
    bins = pd.qcut(pd.Series(clipped), q=10, duplicates="drop")
    table = pd.DataFrame({"observed": observed, "predicted": clipped, "bin": bins}).groupby(
        "bin", observed=True
    ).agg(observed=("observed", "mean"), predicted=("predicted", "mean"), size=("observed", "size"))
    ece = float(np.sum(np.abs(table.observed - table.predicted) * table.size) / len(observed))
    return {
        "brier": float(np.mean((observed - clipped) ** 2)),
        "logLoss": float(-np.mean(observed * np.log(clipped) + (1 - observed) * np.log(1 - clipped))),
        "calibrationIntercept": float(intercept),
        "calibrationSlope": float(slope),
        "expectedCalibrationError": ece,
    }


def cross_validated_calibration(
    arm: pd.DataFrame,
    freeze_sha: str,
    continuous_chosen: dict[tuple[str, str], tuple[str, Any]],
    success_chosen: dict[str, tuple[str, Any]],
) -> pd.DataFrame:
    rows = []
    for estimand in ESTIMANDS:
        data = arm[arm.estimandId == estimand].copy()
        data["fold"] = data.pairingBlockId.map(lambda value: hash_fold(freeze_sha, value))
        for outcome in ("normalizedResidualError", "logS01UnitWeightFullCost", "successByBudget"):
            formula = f"{outcome} ~ {FULL_RIGHT_HAND_SIDE}"
            predictions = np.full(len(data), np.nan)
            fold_failures = []
            chosen_type = (
                success_chosen[estimand][0] if outcome == "successByBudget"
                else continuous_chosen[(estimand, outcome)][0]
            )
            for fold in range(5):
                train = data[data.fold != fold]
                test = data[data.fold == fold]
                try:
                    if outcome == "successByBudget":
                        covariance = "exchangeable" if chosen_type.endswith("exchangeable") else "independence"
                        result, _ = fit_gee_model(formula, train, Binomial(), covariance)
                    elif chosen_type.startswith("mixedlm"):
                        result = mixedlm(
                            formula, train, groups=train.pairingBlockId, re_formula="1"
                        ).fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
                    else:
                        result, _ = fit_gee_model(formula, train, Gaussian(), "exchangeable")
                    fold_predictions = np.asarray(result.predict(test), dtype=float)
                    if not np.isfinite(fold_predictions).all():
                        fold_failures.append(
                            f"fold={fold}:nonfinite_predictions="
                            f"{int((~np.isfinite(fold_predictions)).sum())}"
                        )
                    predictions[data.fold.to_numpy() == fold] = fold_predictions
                except Exception as error:
                    fold_failures.append(f"fold={fold}:{type(error).__name__}:{error}")
            finite = np.isfinite(predictions)
            metrics = (
                calibration_metrics(
                    data.loc[finite, outcome].to_numpy(), predictions[finite],
                    "binary" if outcome == "successByBudget" else "continuous",
                ) if finite.any() else {}
            )
            rows.append({
                "schemaVersion": "e02.s12.calibration_diagnostic.v1",
                "estimandId": estimand, "outcome": outcome,
                "modelType": chosen_type, "folds": 5,
                "rows": len(data), "predictionsAvailable": int(finite.sum()),
                "foldFailuresJson": json.dumps(fold_failures),
                "completeOutOfFoldPrediction": bool(finite.all()),
                **metrics,
            })
    return pd.DataFrame(rows)


def survival_models(arm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficients = []
    diagnostics = []
    fit_records: list[tuple[str, pd.DataFrame, Any, list[str], bool, bool]] = []
    for estimand in ESTIMANDS:
        data = arm[arm.estimandId == estimand].copy()
        formula = f"completionOpportunity ~ {PH_RIGHT_HAND_SIDE}"
        _, design = patsy.dmatrices(formula, data=data, return_type="dataframe")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = PHReg.from_formula(
                formula, data=data, status=data.completionObserved,
                strata=data.n, ties="efron",
            )
            result = model.fit(groups=data.pairingBlockId)
        messages = [str(item.message) for item in caught]
        params_finite, covariance_finite = finite_result(result)
        convergence_warning = any("converg" in message.lower() for message in messages)
        inference_valid = bool(params_finite and covariance_finite and not convergence_warning)
        coefficients.extend(coefficient_rows(
            result, estimand=estimand, outcome="completionOpportunity",
            model_type="cause_specific_cox", chosen=True,
            inference_valid=inference_valid,
        ))
        treatment_index = model.exog_names.index("treatment")
        schoenfeld = np.asarray(result.schoenfeld_residuals[:, treatment_index], dtype=float)
        mask = np.isfinite(schoenfeld) & (data.completionObserved.to_numpy() == 1)
        if mask.sum() >= 3:
            test = stats.spearmanr(schoenfeld[mask], np.log1p(data.completionOpportunity.to_numpy()[mask]))
            ph_rho, ph_p = float(test.statistic), float(test.pvalue)
        else:
            ph_rho, ph_p = np.nan, 1.0
        fit_records.append((estimand, data, result, messages, inference_valid, convergence_warning))
        diagnostics.append({
            "schemaVersion": "e02.s12.survival_diagnostic.v1",
            "estimandId": estimand, "modelType": "cause_specific_cox",
            "eventRows": int(data.completionObserved.sum()),
            "competingFailureRows": int(data.competingFailure.sum()),
            "rightCensoredBudgetRows": int((data.stopReason == "event_budget").sum()),
            "designRows": len(data), "designColumns": design.shape[1],
            "designRank": int(np.linalg.matrix_rank(design.to_numpy())),
            "conditionNumber": float(np.linalg.cond(design.to_numpy())),
            "paramsFinite": params_finite, "covarianceFinite": covariance_finite,
            "convergenceWarning": convergence_warning,
            "inferenceValid": inference_valid,
            "treatmentSchoenfeldSpearman": ph_rho,
            "treatmentPhUnadjustedP": ph_p,
            "warningsJson": json.dumps(messages),
        })
    diagnostic_frame = pd.DataFrame(diagnostics)
    diagnostic_frame["treatmentPhHolmP"] = holm_adjust(diagnostic_frame.treatmentPhUnadjustedP)
    diagnostic_frame["phAlternativeTriggered"] = diagnostic_frame.treatmentPhHolmP < 0.01
    diagnostic_frame["singleHazardRatioInterpretable"] = (
        diagnostic_frame.inferenceValid & ~diagnostic_frame.phAlternativeTriggered
    )
    piecewise_rows = []
    for record in diagnostic_frame.itertuples(index=False):
        if not record.phAlternativeTriggered:
            continue
        data = arm[arm.estimandId == record.estimandId]
        active = interval_specific_completion_rates(data[data.treatment == 1]).rename(
            columns={"completionRate": "activeRate", "completionEvents": "activeEvents", "normalizedExposure": "activeExposure"}
        )
        reference = interval_specific_completion_rates(data[data.treatment == 0]).rename(
            columns={"completionRate": "referenceRate", "completionEvents": "referenceEvents", "normalizedExposure": "referenceExposure"}
        )
        merged = active.merge(reference, on=["intervalLower", "intervalUpper"])
        for row in merged.itertuples(index=False):
            piecewise_rows.append({
                "schemaVersion": "e02.s12.piecewise_completion_rate.v1",
                "estimandId": record.estimandId,
                "intervalLower": row.intervalLower, "intervalUpper": row.intervalUpper,
                "activeEvents": row.activeEvents, "activeExposure": row.activeExposure,
                "activeRate": row.activeRate, "referenceEvents": row.referenceEvents,
                "referenceExposure": row.referenceExposure, "referenceRate": row.referenceRate,
                "rateRatio": row.activeRate / row.referenceRate if row.referenceRate > 0 else np.nan,
                "claimType": "descriptive_frozen_ph_alternative",
            })
    return pd.DataFrame(coefficients), diagnostic_frame, pd.DataFrame(piecewise_rows)


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((args.output / "model_freeze_manifest.json").read_text())
    freeze_sha = freeze["modelFreezeSha256"]
    arm = pd.read_parquet(args.output / "model_arm_data.parquet")
    continuous_coef, continuous_diag, continuous_std, continuous_chosen = continuous_models(arm)
    success_coef, success_diag, success_std, success_chosen = success_models(arm)
    survival_coef, survival_diag, piecewise = survival_models(arm)
    calibration = cross_validated_calibration(
        arm, freeze_sha, continuous_chosen, success_chosen
    )
    coefficients = pd.concat(
        [continuous_coef, success_coef, survival_coef], ignore_index=True
    )
    diagnostics = pd.concat(
        [continuous_diag, success_diag], ignore_index=True, sort=False
    )
    chosen_mask = diagnostics.modelFamily != "fractional_logit_sensitivity"
    sensitivity_mask = diagnostics.modelFamily == "fractional_logit_sensitivity"
    standardized = pd.concat([continuous_std, success_std], ignore_index=True)
    write_parquet(args.output / "model_coefficients.parquet", coefficients)
    write_parquet(args.output / "model_diagnostics.parquet", diagnostics)
    write_parquet(args.output / "model_standardized_effects.parquet", standardized)
    write_parquet(args.output / "calibration_diagnostics.parquet", calibration)
    write_parquet(args.output / "survival_diagnostics.parquet", survival_diag)
    if len(piecewise):
        write_parquet(args.output / "piecewise_completion_rates.parquet", piecewise)
    write_json(args.output / "model_fit_summary.json", {
        "schemaVersion": "e02.s12.model_fit_summary.v1",
        "researchStepId": "S12", "modelFreezeSha256": freeze_sha,
        "coefficientRows": len(coefficients), "diagnosticRows": len(diagnostics),
        "calibrationRows": len(calibration), "survivalDiagnosticRows": len(survival_diag),
        "continuousAlternativesTriggered": int(
            continuous_diag.loc[continuous_diag.modelFamily == "continuous_hierarchical", "alternativeTriggered"].sum()
        ),
        "binaryAlternativesTriggered": int(success_diag.alternativeTriggered.sum()),
        "chosenModelInferenceValid": int(
            diagnostics.loc[chosen_mask, "chosenInferenceValid"].fillna(False).sum()
        ),
        "chosenModelRows": int(chosen_mask.sum()),
        "sensitivityModelInferenceValid": int(
            diagnostics.loc[sensitivity_mask, "chosenInferenceValid"].fillna(False).sum()
        ),
        "sensitivityModelRows": int(sensitivity_mask.sum()),
        "survivalInferenceValid": int(survival_diag.inferenceValid.sum()),
        "phAlternativesTriggered": int(survival_diag.phAlternativeTriggered.sum()),
        "completeCalibrationRows": int(calibration.completeOutOfFoldPrediction.sum()),
        "piecewiseRows": len(piecewise),
        "noFormulaSelection": True,
    })
    print(json.dumps({
        "coefficients": len(coefficients),
        "continuousAlternatives": int(
            continuous_diag.loc[continuous_diag.modelFamily == "continuous_hierarchical", "alternativeTriggered"].sum()
        ),
        "binaryAlternatives": int(success_diag.alternativeTriggered.sum()),
        "survivalValid": int(survival_diag.inferenceValid.sum()),
        "calibrationComplete": int(calibration.completeOutOfFoldPrediction.sum()),
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
