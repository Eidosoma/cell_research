#!/usr/bin/env python3
"""Fit the frozen S12 repeated-scenario context models and diagnostics."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import patsy
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan

from reference_simulator.model import canonical_json_bytes
from src.detours.context_sensitivity import METRIC_NAMES


OUTPUT = Path("/artifacts/research_steps/S12")
MODELS = OUTPUT / "context_models"
FORMULA_RHS = (
    "C(n)+C(policy_profile)+C(scheduler_profile)+C(direction)+"
    "C(initial_disorder_profile)+initial_inversion_fraction+C(fault_context)+"
    "C(intervention_type)+C(policy_profile):C(scheduler_profile)+"
    "C(policy_profile):initial_inversion_fraction"
)
BOOTSTRAPS = 2_000
FOLDS = 5


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(path: Path, frame: pd.DataFrame, schema: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {b"schemaVersion": schema.encode(), b"researchStepId": b"S12"}
    )
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )


def add_model_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["fault_context"] = result.fault_count.astype(str) + ":" + result.fault_location_class.astype(str)
    return result


def block_fold(block_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"E03/S12/fold/v1\x00{block_id}".encode()).digest()[:4], "big") % FOLDS


def calibration_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    eps = 1e-8
    p = np.clip(probability.astype(float), eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y)) == 2:
        calibration = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2_000).fit(logit, y)
        intercept = float(calibration.intercept_[0])
        slope = float(calibration.coef_[0, 0])
    else:
        intercept = math.nan
        slope = math.nan
    bins = pd.qcut(p, q=10, duplicates="drop")
    calibration_frame = pd.DataFrame({"y": y, "p": p, "bin": bins})
    grouped = calibration_frame.groupby("bin", observed=True).agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
    ece = float(((grouped.n / grouped.n.sum()) * (grouped.observed - grouped.predicted).abs()).sum())
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece_10": ece,
    }


def design_diagnostics(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    matrix = patsy.dmatrix(FORMULA_RHS, frame, return_type="dataframe")
    values = matrix.to_numpy()
    rank = int(np.linalg.matrix_rank(values))
    condition = float(np.linalg.cond(values))
    record = {
        "researchStepId": "S12",
        "rows": len(matrix),
        "columns": matrix.shape[1],
        "rank": rank,
        "conditionNumber": condition,
        "fullRank": rank == matrix.shape[1],
        "conditionBelow100": condition < 100,
        "columnsInOrder": matrix.columns.tolist(),
    }
    if not record["fullRank"] or not record["conditionBelow100"]:
        raise RuntimeError(f"S12 identifiability gate: {record}")
    return matrix, record


def fit_gee_binary(frame: pd.DataFrame, outcome: str, model_id: str) -> tuple[pd.DataFrame, dict, np.ndarray]:
    formula = f"{outcome} ~ {FORMULA_RHS}"
    model = smf.gee(
        formula,
        groups="scenario_block_id",
        data=frame,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    result = model.fit(maxiter=150)
    finite = bool(np.isfinite(result.params).all() and np.isfinite(result.cov_params()).all().all())
    converged = bool(getattr(result, "converged", False))
    probabilities = np.asarray(result.predict(frame), dtype=float)
    extreme = bool(((probabilities < 1e-8) | (probabilities > 1 - 1e-8)).any())
    fallback = None
    if not finite or not converged or extreme:
        matrix = patsy.dmatrix(FORMULA_RHS, frame, return_type="dataframe")
        ridge = LogisticRegression(C=1.0, solver="lbfgs", max_iter=4_000)
        ridge.fit(matrix.to_numpy(), frame[outcome].astype(int).to_numpy())
        probabilities = ridge.predict_proba(matrix.to_numpy())[:, 1]
        fallback = "ridge_logistic_prediction_due_nonfinite_nonconvergence_or_extreme_probability"
        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"S12 binary fallback nonfinite for {model_id}")
    coefficients = pd.DataFrame(
        {
            "model_id": model_id,
            "outcome": outcome,
            "term": result.params.index,
            "estimate": result.params.to_numpy(float),
            "standard_error": result.bse.to_numpy(float),
            "p_value": result.pvalues.to_numpy(float),
            "fit_family": "binomial_gee_exchangeable",
        }
    )
    diagnostic = {
        "modelId": model_id,
        "outcome": outcome,
        "rows": len(frame),
        "events": int(frame[outcome].sum()),
        "nonevents": int((~frame[outcome].astype(bool)).sum()),
        "clusters": int(frame.scenario_block_id.nunique()),
        "converged": converged,
        "finite": finite,
        "extremeProbability": extreme,
        "workingCorrelation": float(np.atleast_1d(result.cov_struct.dep_params)[0]),
        "predictionFallback": fallback,
        "inSampleCalibration": calibration_metrics(frame[outcome].astype(int).to_numpy(), probabilities),
    }
    return coefficients, diagnostic, probabilities


def fit_depth(frame: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, dict, np.ndarray]:
    positive = frame[frame.any_detour_episode].copy()
    positive["log1p_depth"] = np.log1p(positive.max_episode_depth.astype(float))
    if len(positive) < 25:
        raise RuntimeError(f"S12 depth identifiability gate for {metric}: {len(positive)}")
    formula = f"log1p_depth ~ {FORMULA_RHS}"
    fallback = None
    result = None
    fit_family = "gaussian_mixed_random_block_intercept"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = smf.mixedlm(
                formula,
                positive,
                groups=positive["scenario_block_id"],
                re_formula="1",
            ).fit(reml=False, method="lbfgs", maxiter=300, disp=False)
            covariance = np.asarray(result.cov_params())
            random_variance = float(np.asarray(result.cov_re).ravel()[0])
            invalid = (
                not bool(getattr(result, "converged", False))
                or not np.isfinite(result.params).all()
                or not np.isfinite(covariance).all()
                or random_variance < 1e-8
            )
        except Exception as exc:  # declared deterministic fallback
            invalid = True
            fallback = f"mixed_exception:{type(exc).__name__}"
        warning_text = [str(item.message) for item in caught]
    if invalid:
        fit_family = "gaussian_gee_exchangeable_fallback"
        fallback = fallback or "singular_nonconverged_or_nonfinite_mixed_fit"
        result = smf.gee(
            formula,
            groups="scenario_block_id",
            data=positive,
            family=sm.families.Gaussian(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit(maxiter=150)
        random_variance = math.nan
    predictions = np.asarray(result.predict(positive), dtype=float)
    residual = positive.log1p_depth.to_numpy() - predictions
    design = patsy.dmatrix(FORMULA_RHS, positive, return_type="dataframe").to_numpy()
    bp = het_breuschpagan(residual, design)
    coefficients = pd.DataFrame(
        {
            "model_id": f"depth_{metric}",
            "outcome": "log1p_depth",
            "term": result.params.index,
            "estimate": result.params.to_numpy(float),
            "standard_error": result.bse.to_numpy(float),
            "p_value": result.pvalues.to_numpy(float),
            "fit_family": fit_family,
        }
    )
    diagnostic = {
        "modelId": f"depth_{metric}",
        "metric": metric,
        "rows": len(positive),
        "clusters": int(positive.scenario_block_id.nunique()),
        "fitFamily": fit_family,
        "fallback": fallback,
        "randomInterceptVariance": random_variance,
        "finite": bool(np.isfinite(result.params).all() and np.isfinite(predictions).all()),
        "warnings": warning_text,
        "residualFittedSpearman": float(stats.spearmanr(predictions, residual).statistic),
        "residualMean": float(residual.mean()),
        "residualSd": float(residual.std(ddof=1)),
        "residualQuantiles": {str(q): float(np.quantile(residual, q)) for q in (0, 0.01, 0.25, 0.5, 0.75, 0.99, 1)},
        "breuschPaganLm": float(bp[0]),
        "breuschPaganP": float(bp[1]),
    }
    if not diagnostic["finite"]:
        raise RuntimeError(f"S12 depth fallback nonfinite for {metric}")
    return coefficients, diagnostic, predictions


def fit_recovery(frame: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, dict]:
    episodes = frame[frame.running_min_episode_count.gt(0)].copy()
    episodes["recovery_events"] = episodes.recovered_episode_count.astype(int)
    episodes["time_at_risk"] = episodes.total_episode_time_at_risk_opportunities.astype(float)
    episodes = episodes[episodes.time_at_risk.gt(0)].copy()
    if episodes.recovery_events.sum() < 25 or (episodes.running_min_episode_count - episodes.recovery_events).sum() < 25:
        raise RuntimeError(f"S12 recovery identifiability gate for {metric}")
    formula = f"recovery_events ~ {FORMULA_RHS}"
    result = smf.gee(
        formula,
        groups="scenario_block_id",
        data=episodes,
        family=sm.families.Poisson(),
        cov_struct=sm.cov_struct.Exchangeable(),
        offset=np.log(episodes.time_at_risk.to_numpy()),
    ).fit(maxiter=150)
    finite = bool(np.isfinite(result.params).all() and np.isfinite(result.cov_params()).all().all())
    if not bool(getattr(result, "converged", False)) or not finite:
        raise RuntimeError(f"S12 recovery model failed for {metric}")
    coefficients = pd.DataFrame(
        {
            "model_id": f"recovery_{metric}",
            "outcome": "recovery_events_per_opportunity",
            "term": result.params.index,
            "estimate": result.params.to_numpy(float),
            "standard_error": result.bse.to_numpy(float),
            "p_value": result.pvalues.to_numpy(float),
            "fit_family": "poisson_gee_piecewise_exponential",
        }
    )
    diagnostic = {
        "modelId": f"recovery_{metric}",
        "metric": metric,
        "rows": len(episodes),
        "clusters": int(episodes.scenario_block_id.nunique()),
        "recoveredEpisodes": int(episodes.recovery_events.sum()),
        "rightCensoredEpisodes": int((episodes.running_min_episode_count - episodes.recovery_events).sum()),
        "timeAtRiskOpportunities": float(episodes.time_at_risk.sum()),
        "converged": bool(getattr(result, "converged", False)),
        "finite": finite,
        "workingCorrelation": float(np.atleast_1d(result.cov_struct.dep_params)[0]),
        "modelInterpretation": "constant piecewise-exponential recovery hazard; open episodes contribute exposure without a recovery event",
    }
    return coefficients, diagnostic


def held_out_binary(frame: pd.DataFrame, outcome: str, model_id: str) -> tuple[dict, pd.DataFrame]:
    matrix = patsy.dmatrix(FORMULA_RHS, frame, return_type="dataframe")
    x = matrix.to_numpy(float)
    y = frame[outcome].astype(int).to_numpy()
    folds = frame.scenario_block_id.map(block_fold).to_numpy()
    probabilities = np.full(len(frame), np.nan)
    fold_rows = []
    for fold in range(FOLDS):
        train = folds != fold
        test = folds == fold
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=4_000)
        model.fit(x[train], y[train])
        probabilities[test] = model.predict_proba(x[test])[:, 1]
        fold_rows.append(
            {
                "model_id": model_id,
                "fold": fold,
                "train_rows": int(train.sum()),
                "test_rows": int(test.sum()),
                **calibration_metrics(y[test], probabilities[test]),
            }
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"held-out binary predictions missing for {model_id}")
    return calibration_metrics(y, probabilities), pd.DataFrame(fold_rows)


def held_out_depth(frame: pd.DataFrame, metric: str) -> tuple[dict, pd.DataFrame]:
    positive = frame[frame.any_detour_episode].copy()
    matrix = patsy.dmatrix(FORMULA_RHS, positive, return_type="dataframe").to_numpy(float)
    y = np.log1p(positive.max_episode_depth.to_numpy(float))
    folds = positive.scenario_block_id.map(block_fold).to_numpy()
    predictions = np.full(len(positive), np.nan)
    rows = []
    for fold in range(FOLDS):
        train = folds != fold
        test = folds == fold
        model = Ridge(alpha=1.0).fit(matrix[train], y[train])
        predictions[test] = model.predict(matrix[test])
        rows.append(
            {
                "model_id": f"depth_{metric}",
                "fold": fold,
                "train_rows": int(train.sum()),
                "test_rows": int(test.sum()),
                "rmse": float(mean_squared_error(y[test], predictions[test]) ** 0.5),
                "mae": float(mean_absolute_error(y[test], predictions[test])),
            }
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"held-out depth predictions missing for {metric}")
    return {
        "rmse": float(mean_squared_error(y, predictions) ** 0.5),
        "mae": float(mean_absolute_error(y, predictions)),
        "calibrationIntercept": float(np.polyfit(predictions, y, 1)[1]),
        "calibrationSlope": float(np.polyfit(predictions, y, 1)[0]),
    }, pd.DataFrame(rows)


def cluster_bootstrap_cells(frame: pd.DataFrame) -> pd.DataFrame:
    factors = (
        "n", "policy_profile", "scheduler_profile", "direction",
        "initial_disorder_profile", "fault_count", "fault_location_class", "intervention_type",
        "s10_strict_filter_exposed", "s11_rate_feasibility_stratum", "recurrent_active_status",
    )
    endpoints = {
        "detour_occurrence": "any_detour_episode",
        "completion": "completed",
        "max_depth": "max_episode_depth",
        "recovery_fraction": "recovered_episode_count",
    }
    rows = []
    for metric, metric_frame in frame.groupby("metric", observed=True):
        for factor in factors:
            for level, subset in metric_frame.groupby(factor, observed=True, dropna=False):
                block = subset.groupby("scenario_block_id", observed=True).agg(
                    detour_occurrence=("any_detour_episode", "mean"),
                    completion=("completed", "mean"),
                    max_depth=("max_episode_depth", "mean"),
                    recovered=("recovered_episode_count", "sum"),
                    episodes=("running_min_episode_count", "sum"),
                )
                if block.empty:
                    continue
                values = {
                    "detour_occurrence": block.detour_occurrence.to_numpy(float),
                    "completion": block.completion.to_numpy(float),
                    "max_depth": block.max_depth.to_numpy(float),
                }
                rng = np.random.default_rng(
                    int.from_bytes(hashlib.sha256(f"E03/S12/bootstrap/{metric}/{factor}/{level}".encode()).digest()[:8], "big")
                )
                indices = rng.integers(0, len(block), size=(BOOTSTRAPS, len(block)))
                for endpoint, array in values.items():
                    draws = array[indices].mean(axis=1)
                    rows.append(
                        {
                            "metric": metric,
                            "factor": factor,
                            "level": str(level),
                            "endpoint": endpoint,
                            "runs": len(subset),
                            "blocks": len(block),
                            "estimate": float(array.mean()),
                            "ci_low": float(np.quantile(draws, 0.025)),
                            "ci_high": float(np.quantile(draws, 0.975)),
                            "bootstrap_replicates": BOOTSTRAPS,
                        }
                    )
                recovered = block.recovered.to_numpy(float)
                episodes = block.episodes.to_numpy(float)
                estimate = float(recovered.sum() / episodes.sum()) if episodes.sum() else math.nan
                if episodes.sum():
                    sampled_recovered = recovered[indices].sum(axis=1)
                    sampled_episodes = episodes[indices].sum(axis=1)
                    draws = np.divide(sampled_recovered, sampled_episodes, out=np.full(BOOTSTRAPS, np.nan), where=sampled_episodes > 0)
                    ci_low = float(np.nanquantile(draws, 0.025))
                    ci_high = float(np.nanquantile(draws, 0.975))
                else:
                    ci_low = math.nan
                    ci_high = math.nan
                rows.append(
                    {
                        "metric": metric,
                        "factor": factor,
                        "level": str(level),
                        "endpoint": "recovery_fraction",
                        "runs": len(subset),
                        "blocks": len(block),
                        "estimate": estimate,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "bootstrap_replicates": BOOTSTRAPS,
                    }
                )
    return pd.DataFrame(rows)


def segmentation_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, subset in frame.groupby(["metric", "n", "policy_profile"], observed=True):
        denomin = len(subset)
        for definition, mask in (
            ("running_minimum_episode", subset.any_detour_episode),
            ("positive_delta_run", subset.positive_run_count.gt(0)),
            ("above_initial_level", subset.initial_level_excursion.gt(0)),
            ("s10_individual_proposal_exposure", subset.s10_strict_filter_exposed),
        ):
            rows.append(
                {
                    "metric": keys[0],
                    "n": int(keys[1]),
                    "policy_profile": keys[2],
                    "segmentation_definition": definition,
                    "runs": denomin,
                    "positive_runs": int(mask.sum()),
                    "rate": float(mask.mean()),
                }
            )
    return pd.DataFrame(rows)


def threshold_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, subset in frame.groupby("metric", observed=True):
        for threshold in (0, 1, 2, 4):
            column = f"worsening_proposals_gt_{threshold}"
            rows.append(
                {
                    "metric": metric,
                    "threshold_raw_units": threshold,
                    "runs": len(subset),
                    "exposed_runs": int(subset[column].gt(0).sum()),
                    "exposure_rate": float(subset[column].gt(0).mean()),
                    "worsening_proposals": int(subset[column].sum()),
                }
            )
    return pd.DataFrame(rows)


def rare_event_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cell_fields = ["metric", "n", "policy_profile", "scheduler_profile", "intervention_type"]
    for keys, subset in frame.groupby(cell_fields, observed=True):
        events = int(subset.any_detour_episode.sum())
        rows.append(
            {
                **dict(zip(cell_fields, keys, strict=True)),
                "runs": len(subset),
                "events": events,
                "nonevents": len(subset) - events,
                "complete_separation_cell": events == 0 or events == len(subset),
            }
        )
    return pd.DataFrame(rows)


def evidence_layer_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    bridge = pq.read_table(OUTPUT / "small_n_bridge_strata.parquet").to_pandas()
    large = (
        frame.groupby(["metric", "intervention_type"], observed=True)
        .agg(
            runs=("run_id", "size"),
            completion_rate=("completed", "mean"),
            detour_rate=("any_detour_episode", "mean"),
            mean_depth=("max_episode_depth", "mean"),
        )
        .reset_index()
    )
    large["evidence_layer"] = "empirical_large_n_native"
    small = (
        bridge[bridge.null_family.eq("open_loop_opportunity")]
        .groupby(["metric", "intervention_type"], observed=True)
        .apply(
            lambda group: pd.Series(
                {
                    "runs": int(group.runs.sum()),
                    "completion_rate": float(np.average(group.completion_rate, weights=group.runs)),
                    "detour_rate": float(np.average(group.detour_rate, weights=group.runs)),
                    "mean_depth": float(np.average(group.mean_excursion, weights=group.runs)),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    small["evidence_layer"] = "exact_small_n_open_loop_bridge"
    combined = pd.concat([large, small], ignore_index=True)
    combined["exact_reachability_available"] = combined.evidence_layer.str.startswith("exact")
    combined["direct_sign_conflict_test_permitted"] = False
    combined["comparison_boundary"] = "no shared n/exact-reachability support; report side-by-side, never pool or call a sign reversal material conflict"
    return combined


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    frame = add_model_fields(pq.read_table(OUTPUT / "larger_n_metric_outcomes.parquet").to_pandas())
    runs = add_model_fields(pq.read_table(OUTPUT / "larger_n_trajectories.parquet").to_pandas())
    _, design_record = design_diagnostics(frame.drop_duplicates("run_id"))
    write_json(MODELS / "design_diagnostics.json", design_record)

    event_counts = frame.groupby("metric", observed=True).any_detour_episode.agg(["sum", "count"])
    event_counts["nonevents"] = event_counts["count"] - event_counts["sum"]
    if (event_counts[["sum", "nonevents"]] < 25).any().any():
        write_json(
            MODELS / "identifiability_failure.json",
            {"researchStepId": "S12", "reason": "fewer than 25 events or nonevents", "counts": event_counts.reset_index().to_dict("records")},
        )
        raise RuntimeError("S12 occurrence identifiability gate failed")

    coefficients = []
    diagnostics = []
    held_out_rows = []
    held_out_summary = []
    for metric in METRIC_NAMES:
        subset = frame[frame.metric.eq(metric)].copy()
        coef, diag, _ = fit_gee_binary(subset, "any_detour_episode", f"occurrence_{metric}")
        coefficients.append(coef)
        diagnostics.append(diag)
        depth_coef, depth_diag, _ = fit_depth(subset, metric)
        coefficients.append(depth_coef)
        diagnostics.append(depth_diag)
        recovery_coef, recovery_diag = fit_recovery(subset, metric)
        coefficients.append(recovery_coef)
        diagnostics.append(recovery_diag)
        summary, folds = held_out_binary(subset, "any_detour_episode", f"occurrence_{metric}")
        held_out_summary.append({"model_id": f"occurrence_{metric}", **summary})
        held_out_rows.append(folds)
        depth_summary, depth_folds = held_out_depth(subset, metric)
        held_out_summary.append({"model_id": f"depth_{metric}", **depth_summary})
        held_out_rows.append(depth_folds)

    completion_coef, completion_diag, _ = fit_gee_binary(runs, "completed", "completion_all_metrics")
    coefficients.append(completion_coef)
    diagnostics.append(completion_diag)
    completion_summary, completion_folds = held_out_binary(runs, "completed", "completion_all_metrics")
    held_out_summary.append({"model_id": "completion_all_metrics", **completion_summary})
    held_out_rows.append(completion_folds)

    coefficient_frame = pd.concat(coefficients, ignore_index=True)
    held_out_frame = pd.concat(held_out_rows, ignore_index=True, sort=False)
    bootstrap = cluster_bootstrap_cells(frame)
    segmentation = segmentation_sensitivity(frame)
    thresholds = threshold_sensitivity(frame)
    rare = rare_event_diagnostics(frame)
    evidence = evidence_layer_comparison(frame)

    write_parquet(MODELS / "model_coefficients.parquet", coefficient_frame, "e03.s12.model_coefficients.v1")
    write_json(MODELS / "model_diagnostics.json", {"researchStepId": "S12", "models": diagnostics})
    write_parquet(MODELS / "held_out_prediction_by_fold.parquet", held_out_frame, "e03.s12.held_out_by_fold.v1")
    write_json(MODELS / "held_out_prediction_summary.json", {"researchStepId": "S12", "models": held_out_summary})
    write_parquet(OUTPUT / "marginal_effects.parquet", bootstrap, "e03.s12.marginal_effects.v1")
    write_parquet(OUTPUT / "episode_segmentation_sensitivity.parquet", segmentation, "e03.s12.episode_segmentation.v1")
    write_parquet(OUTPUT / "threshold_sensitivity.parquet", thresholds, "e03.s12.threshold_sensitivity.v1")
    write_parquet(OUTPUT / "rare_event_separation_diagnostics.parquet", rare, "e03.s12.rare_event_diagnostics.v1")
    write_parquet(OUTPUT / "evidence_layer_comparison.parquet", evidence, "e03.s12.evidence_layer_comparison.v1")

    result = {
        "researchStepId": "S12",
        "success": True,
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "models": len(diagnostics),
        "coefficientRows": len(coefficient_frame),
        "marginalEffectRows": len(bootstrap),
        "heldOutFoldRows": len(held_out_frame),
        "episodeOccurrence": [
            {"metric": index, "events": int(row["sum"]), "nonevents": int(row.nonevents)}
            for index, row in event_counts.iterrows()
        ],
        "designFullRank": design_record["fullRank"],
        "conditionNumber": design_record["conditionNumber"],
        "materialS11Conflict": False,
        "materialS11ConflictReason": "Exact small-n and empirical larger-n layers have no shared exact-reachability support and were not pooled; side-by-side comparisons remain descriptive.",
    }
    write_json(OUTPUT / "model_build_summary.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
