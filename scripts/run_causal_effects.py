#!/usr/bin/env python3
"""Estimate frozen S12 standardized and supported heterogeneous contrasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats
import statsmodels.formula.api as smf

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from causal_simulator.causal_modeling import (
    ADDITIVE_S01_COSTS,
    COUPLING,
    ENDPOINTS,
    ESTIMANDS,
    MODIFIERS,
    SCALES,
    TWO_WAY_MODIFIERS,
    bh_adjust,
    build_bootstrap_weights,
    build_paired_data,
    canonical_json_bytes,
    holm_adjust,
    load_arm_data,
    sign_tail_p,
    standardized_aalen_johansen,
)


DEFAULT_OUTPUT = Path("/artifacts/research_steps/S12")
DEFAULT_RESULTS = Path("/artifacts/research_steps/S11/confirmatory_results.parquet")
DEFAULT_CONTRAST = Path("/artifacts/research_steps/S11/confirmatory_contrast_map.parquet")
DEFAULT_PAIRING = Path("/artifacts/research_steps/S11/confirmatory_pairing_blocks.parquet")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), path,
        compression="zstd", compression_level=9, use_dictionary=True,
        write_statistics=True, version="2.6",
    )


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def ordered_blocks(paired: pd.DataFrame) -> pd.DataFrame:
    return paired[["pairingBlockId", "n"]].drop_duplicates().sort_values(
        ["n", "pairingBlockId"]
    ).reset_index(drop=True)


def align_pairs(part: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    aligned = blocks.merge(part, on=["pairingBlockId", "n"], validate="one_to_one")
    if len(aligned) != 2000:
        raise AssertionError("paired alignment lost S11 support")
    return aligned


def bootstrap_effects(
    arm: pd.DataFrame,
    paired: pd.DataFrame,
    freeze_sha: str,
    replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], np.ndarray]:
    blocks = ordered_blocks(paired)
    weights = build_bootstrap_weights(blocks, freeze_sha, replicates)
    weights_float = weights.astype(np.float64)
    marginal_rows = []
    heterogeneity_rows = []
    two_way_rows = []
    stability_rows = []
    bootstrap_cache: dict[tuple[str, str], np.ndarray] = {}

    for estimand in ESTIMANDS:
        part = align_pairs(paired[paired.estimandId == estimand], blocks)
        for endpoint in ENDPOINTS:
            values = part[endpoint].to_numpy(dtype=float)
            draws = weights_float @ values / 2000.0
            bootstrap_cache[(estimand, endpoint)] = draws
            estimate = float(values.mean())
            arm_part = arm[arm.estimandId == estimand]
            active_mean = float(
                arm_part.loc[arm_part.treatment == 1, endpoint].mean()
            )
            reference_mean = float(
                arm_part.loc[arm_part.treatment == 0, endpoint].mean()
            )
            marginal_rows.append({
                "schemaVersion": "e02.s12.marginal_effect.v1",
                "estimandId": estimand,
                "endpoint": endpoint,
                "effectOrientation": "active_minus_reference",
                "estimate": estimate,
                "bootstrapStandardError": float(draws.std(ddof=1)),
                "bootstrapLow95": float(np.quantile(draws, 0.025)),
                "bootstrapHigh95": float(np.quantile(draws, 0.975)),
                "unadjustedBootstrapP": sign_tail_p(draws),
                "activeMean": active_mean,
                "referenceMean": reference_mean,
                "riskRatio": (
                    active_mean / reference_mean
                    if endpoint == "successByBudget" and reference_mean > 0
                    else np.nan
                ),
                "pairCount": len(values),
                "couplingClassification": COUPLING[estimand],
                "claimType": "randomized_standardized_total_effect",
            })
            first = draws[:1000]
            full_se = draws.std(ddof=1)
            first_half = (np.quantile(first, 0.975) - np.quantile(first, 0.025)) / 2
            full_half = (np.quantile(draws, 0.975) - np.quantile(draws, 0.025)) / 2
            point_shift_se = abs(first.mean() - draws.mean()) / full_se if full_se > 0 else 0.0
            half_ratio = first_half / full_half if full_half > 0 else 1.0
            stability_rows.append({
                "estimandId": estimand, "endpoint": endpoint,
                "first1000Mean": float(first.mean()), "full2000Mean": float(draws.mean()),
                "pointShiftInFullSE": float(point_shift_se),
                "first1000HalfWidth95": float(first_half),
                "full2000HalfWidth95": float(full_half),
                "halfWidthRatio": float(half_ratio),
                "pointStabilityPass": bool(point_shift_se <= 0.10),
                "widthStabilityPass": bool(0.85 <= half_ratio <= 1.15),
            })

        for modifier in MODIFIERS:
            for level, group in part.groupby(modifier, observed=True, sort=True):
                if len(group) < 100:
                    continue
                mask = (part[modifier].to_numpy() == level).astype(float)
                denominator = weights_float @ mask
                for endpoint in ENDPOINTS:
                    values = part[endpoint].to_numpy(dtype=float)
                    draws = (weights_float @ (values * mask)) / denominator
                    heterogeneity_rows.append({
                        "schemaVersion": "e02.s12.heterogeneity_effect.v1",
                        "estimandId": estimand, "endpoint": endpoint,
                        "modifier": modifier, "level": str(level),
                        "pairCount": len(group), "estimate": float(group[endpoint].mean()),
                        "bootstrapLow95": float(np.quantile(draws, 0.025)),
                        "bootstrapHigh95": float(np.quantile(draws, 0.975)),
                        "supported": True,
                        "claimType": "conditional_causal_contrast_with_associational_modifier_comparison",
                    })

        for left, right in TWO_WAY_MODIFIERS:
            for levels, group in part.groupby([left, right], observed=True, sort=True):
                if len(group) < 50:
                    continue
                for endpoint in ENDPOINTS:
                    two_way_rows.append({
                        "schemaVersion": "e02.s12.two_way_heterogeneity.v1",
                        "estimandId": estimand, "endpoint": endpoint,
                        "modifierPair": f"{left}:{right}",
                        "leftLevel": str(levels[0]), "rightLevel": str(levels[1]),
                        "pairCount": len(group), "estimate": float(group[endpoint].mean()),
                        "supported": True, "inference": "descriptive_no_separate_test",
                    })

    marginal = pd.DataFrame(marginal_rows)
    marginal["holmAdjustedP"] = holm_adjust(marginal.unadjustedBootstrapP)
    marginal["holmReject05"] = marginal.holmAdjustedP < 0.05
    stability = pd.DataFrame(stability_rows)
    stability_summary = {
        "schemaVersion": "e02.s12.bootstrap_stability.v1",
        "researchStepId": "S12", "replicates": replicates,
        "comparisonReplicates": 1000,
        "rows": stability.to_dict(orient="records"),
        "pointStabilityPassed": int(stability.pointStabilityPass.sum()),
        "widthStabilityPassed": int(stability.widthStabilityPass.sum()),
        "rowsTotal": len(stability),
        "success": bool(stability.pointStabilityPass.all() and stability.widthStabilityPass.all()),
    }
    return (
        marginal,
        pd.DataFrame(heterogeneity_rows),
        pd.DataFrame(two_way_rows),
        stability_summary,
        weights,
    )


def heterogeneity_support_outputs(
    paired: pd.DataFrame,
    heterogeneity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the frozen supported-cell and effect-range ranking rules."""

    rankings = (
        heterogeneity.groupby(
            ["estimandId", "endpoint", "modifier"], observed=True, sort=True
        )
        .agg(
            supportedLevels=("level", "nunique"),
            minimumCellPairs=("pairCount", "min"),
            minimumEffect=("estimate", "min"),
            maximumEffect=("estimate", "max"),
        )
        .reset_index()
    )
    rankings["effectRange"] = rankings.maximumEffect - rankings.minimumEffect
    rankings["associationalRangeRank"] = (
        rankings.groupby(["estimandId", "endpoint"], observed=True).effectRange
        .rank(method="first", ascending=False)
        .astype(int)
    )
    rankings["schemaVersion"] = "e02.s12.heterogeneity_ranking.v1"
    rankings["claimType"] = "associational_supported_effect_range_ranking"
    rankings = rankings.sort_values(
        ["estimandId", "endpoint", "associationalRangeRank"]
    ).reset_index(drop=True)

    unsupported: list[dict[str, Any]] = []
    global_levels = {
        modifier: sorted(paired[modifier].astype(str).unique())
        for modifier in MODIFIERS
    }
    for estimand in ESTIMANDS:
        part = paired[paired.estimandId == estimand]
        for modifier in MODIFIERS:
            counts = part[modifier].astype(str).value_counts()
            for level in global_levels[modifier]:
                count = int(counts.get(level, 0))
                if count < 100:
                    unsupported.append({
                        "schemaVersion": "e02.s12.unsupported_cell.v1",
                        "estimandId": estimand,
                        "cellType": "one_way",
                        "modifier": modifier,
                        "level": level,
                        "pairCount": count,
                        "minimumPairs": 100,
                        "action": "not_estimated_no_pooling_or_extrapolation",
                    })
        for left, right in TWO_WAY_MODIFIERS:
            all_cells = (
                paired[[left, right]].astype(str).drop_duplicates()
                .sort_values([left, right]).itertuples(index=False, name=None)
            )
            counts = part.groupby([left, right], observed=True).size()
            for left_level, right_level in all_cells:
                matching = part[
                    (part[left].astype(str) == left_level)
                    & (part[right].astype(str) == right_level)
                ]
                count = len(matching)
                if count < 50:
                    unsupported.append({
                        "schemaVersion": "e02.s12.unsupported_cell.v1",
                        "estimandId": estimand,
                        "cellType": "two_way",
                        "modifier": f"{left}:{right}",
                        "level": f"{left_level}:{right_level}",
                        "pairCount": count,
                        "minimumPairs": 50,
                        "action": "not_estimated_no_pooling_or_extrapolation",
                    })
    columns = [
        "schemaVersion", "estimandId", "cellType", "modifier", "level",
        "pairCount", "minimumPairs", "action",
    ]
    return rankings, pd.DataFrame(unsupported, columns=columns)


def heterogeneity_omnibus(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for estimand in ESTIMANDS:
        part = paired[paired.estimandId == estimand]
        for endpoint in ENDPOINTS:
            for modifier in MODIFIERS:
                model = smf.ols(f"{endpoint} ~ C({modifier})", data=part).fit(cov_type="HC3")
                terms = [index for index, name in enumerate(model.params.index) if name != "Intercept"]
                restriction = np.zeros((len(terms), len(model.params)))
                for row_index, term_index in enumerate(terms):
                    restriction[row_index, term_index] = 1.0
                test = model.wald_test(restriction, scalar=True)
                rows.append({
                    "schemaVersion": "e02.s12.heterogeneity_omnibus.v1",
                    "estimandId": estimand, "endpoint": endpoint,
                    "modifier": modifier, "degreesOfFreedom": len(terms),
                    "waldStatistic": float(test.statistic),
                    "unadjustedP": float(test.pvalue),
                    "claimType": "associational_effect_modification_omnibus",
                })
    frame = pd.DataFrame(rows)
    frame["bhAdjustedP"] = np.nan
    for endpoint, indices in frame.groupby("endpoint").groups.items():
        frame.loc[indices, "bhAdjustedP"] = bh_adjust(frame.loc[indices, "unadjustedP"])
    frame["bhReject05"] = frame.bhAdjustedP < 0.05
    return frame


def rng_unpaired_sensitivity(
    arm: pd.DataFrame,
    paired: pd.DataFrame,
    freeze_sha: str,
    replicates: int,
) -> pd.DataFrame:
    blocks = ordered_blocks(paired)
    active_weights = build_bootstrap_weights(blocks, freeze_sha, replicates, stream="independent_active")
    reference_weights = build_bootstrap_weights(blocks, freeze_sha, replicates, stream="independent_reference")
    active_weights_float = active_weights.astype(np.float64)
    reference_weights_float = reference_weights.astype(np.float64)
    rows = []
    for estimand in ("E02-S01-E03", "E02-S01-E06"):
        part = arm[arm.estimandId == estimand]
        for treatment, label, weights in (
            (1, "active", active_weights), (0, "reference", reference_weights)
        ):
            sub = blocks.merge(
                part[part.treatment == treatment],
                on=["pairingBlockId", "n"], validate="one_to_one",
            )
            if treatment == 1:
                active = sub
            else:
                reference = sub
        for endpoint in ENDPOINTS:
            active_values = active[endpoint].to_numpy(dtype=float)
            reference_values = reference[endpoint].to_numpy(dtype=float)
            draws = (
                active_weights_float @ active_values / 2000.0
                - reference_weights_float @ reference_values / 2000.0
            )
            paired_values = align_pairs(paired[paired.estimandId == estimand], blocks)[endpoint]
            rows.append({
                "schemaVersion": "e02.s12.rng_unpaired_sensitivity.v1",
                "estimandId": estimand, "endpoint": endpoint,
                "pairedPointEstimateRetained": float(paired_values.mean()),
                "armIndependentBootstrapLow95": float(np.quantile(draws, 0.025)),
                "armIndependentBootstrapHigh95": float(np.quantile(draws, 0.975)),
                "armIndependentBootstrapSE": float(draws.std(ddof=1)),
                "unadjustedP": sign_tail_p(draws),
                "interpretation": "scenario paired point estimate; no pathwise CRN variance claim",
            })
    return pd.DataFrame(rows)


def survival_outputs(arm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    curves = []
    for estimand in ESTIMANDS:
        for treatment, role in ((0, "reference"), (1, "active")):
            part = arm[(arm.estimandId == estimand) & (arm.treatment == treatment)]
            metrics, curve = standardized_aalen_johansen(part)
            summaries.append({
                "schemaVersion": "e02.s12.survival_summary.v1",
                "estimandId": estimand, "contrastRole": role,
                "pairCount": len(part), **metrics,
                "completionEvents": int(part.completionObserved.sum()),
                "competingFailures": int(part.competingFailure.sum()),
                "rightCensoredBudgets": int((part.stopReason == "event_budget").sum()),
            })
            curve["estimandId"] = estimand
            curve["contrastRole"] = role
            curves.append(curve)
    return pd.DataFrame(summaries), pd.concat(curves, ignore_index=True)


def leave_one_scale_out(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for estimand in ESTIMANDS:
        part = paired[paired.estimandId == estimand]
        for endpoint in ENDPOINTS:
            full = float(part[endpoint].mean())
            for omitted in SCALES:
                estimate = float(part.loc[part.n != omitted, endpoint].mean())
                rows.append({
                    "estimandId": estimand, "endpoint": endpoint,
                    "omittedScale": omitted, "fullEstimate": full,
                    "leaveOneScaleOutEstimate": estimate,
                    "absoluteShift": abs(estimate - full),
                    "signPreserved": bool(np.sign(estimate) == np.sign(full) or full == 0),
                })
    return pd.DataFrame(rows)


def cost_sensitivities(arm: pd.DataFrame, weights: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = []
    sensitivity_rows = []
    part = arm[arm.estimandId == "E02-S01-E04"]
    active = part[part.treatment == 1].set_index("pairingBlockId").sort_index()
    reference = part[part.treatment == 0].set_index("pairingBlockId").loc[active.index]
    blocks = pd.DataFrame({"pairingBlockId": active.index, "n": active.n.to_numpy()}).sort_values(
        ["n", "pairingBlockId"]
    )
    active = active.loc[blocks.pairingBlockId]
    reference = reference.loc[blocks.pairingBlockId]
    weights_float = weights.astype(np.float64)
    for component in ADDITIVE_S01_COSTS:
        values = (
            active[f"cost_{component}"].to_numpy(dtype=float)
            - reference[f"cost_{component}"].to_numpy(dtype=float)
        )
        draws = weights_float @ values / 2000.0
        rows.append({
            "schemaVersion": "e02.s12.e04_cost_component.v1",
            "component": component, "unit": "native_count",
            "meanDifference": float(values.mean()),
            "bootstrapLow95": float(np.quantile(draws, 0.025)),
            "bootstrapHigh95": float(np.quantile(draws, 0.975)),
            "unadjustedBootstrapP": sign_tail_p(draws),
        })
    components = pd.DataFrame(rows)
    components["bhAdjustedP"] = bh_adjust(components.unadjustedBootstrapP)
    components["bhReject05"] = components.bhAdjustedP < 0.05

    for estimand in ESTIMANDS:
        sub = arm[arm.estimandId == estimand]
        a = sub[sub.treatment == 1].set_index("pairingBlockId").sort_index()
        b = sub[sub.treatment == 0].set_index("pairingBlockId").loc[a.index]
        for endpoint in (
            "logS01UnitWeightFullCost", "log1pS01UnitWeightFullCost",
            "logControllerExpandedCost", "logMechanismExpandedCost",
        ):
            values = a[endpoint].to_numpy() - b[endpoint].to_numpy()
            sensitivity_rows.append({
                "estimandId": estimand, "endpoint": endpoint,
                "estimate": float(values.mean()),
                "low95NormalApprox": float(values.mean() - 1.96 * values.std(ddof=1) / np.sqrt(len(values))),
                "high95NormalApprox": float(values.mean() + 1.96 * values.std(ddof=1) / np.sqrt(len(values))),
            })

    success_diff = active.successByBudget.to_numpy() - reference.successByBudget.to_numpy()
    residual_diff = active.normalizedResidualError.to_numpy() - reference.normalizedResidualError.to_numpy()
    raw_cost_diff = active.projection_s01UnitWeightFullCost.to_numpy(dtype=float) - reference.projection_s01UnitWeightFullCost.to_numpy(dtype=float)
    n2_cost_diff = active.s01CostPerN2.to_numpy() - reference.s01CostPerN2.to_numpy()
    log_cost_diff = active.logS01UnitWeightFullCost.to_numpy() - reference.logS01UnitWeightFullCost.to_numpy()
    log_opp_diff = active.logCompletionOpportunity.to_numpy() - reference.logCompletionOpportunity.to_numpy()
    risk_difference = float(success_diff.mean())
    tradeoff = {
        "schemaVersion": "e02.s12.e04_resource_tradeoff.v1",
        "researchStepId": "S12", "estimandId": "E02-S01-E04",
        "pairCount": len(active), "successRiskDifference": risk_difference,
        "normalizedResidualDifference": float(residual_diff.mean()),
        "logCompletionOpportunityDifference": float(log_opp_diff.mean()),
        "logS01UnitWeightFullCostDifference": float(log_cost_diff.mean()),
        "rawS01UnitWeightFullCostDifference": float(raw_cost_diff.mean()),
        "s01UnitWeightCostPerN2Difference": float(n2_cost_diff.mean()),
        "pairsWithStrictSuccessBenefit": int((success_diff > 0).sum()),
        "pairsWithCostIncrease": int((raw_cost_diff > 0).sum()),
        "pairsWithSuccessBenefitAndCostIncrease": int(((success_diff > 0) & (raw_cost_diff > 0)).sum()),
        "pairsTaskNoWorseAndCostIncrease": int(((success_diff >= 0) & (residual_diff <= 0) & (raw_cost_diff > 0)).sum()),
        "incrementalRawS01CostPerSuccessPercentagePoint": (
            float(raw_cost_diff.mean() / (100 * risk_difference)) if risk_difference > 0 else None
        ),
        "claimBoundary": "descriptive joint tradeoff of randomized total effects; not a mediator-adjusted direct effect or physical price",
    }
    return components, pd.DataFrame(sensitivity_rows), tradeoff


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((args.output / "model_freeze_manifest.json").read_text())
    freeze_sha = freeze["modelFreezeSha256"]
    arm = load_arm_data(args.results, args.contrast_map, args.pairing)
    paired = build_paired_data(arm)
    write_parquet(args.output / "model_arm_data.parquet", arm)
    write_parquet(args.output / "paired_analysis_data.parquet", paired)

    marginal, heterogeneity, two_way, stability, weights = bootstrap_effects(
        arm, paired, freeze_sha, args.bootstrap_replicates
    )
    rankings, unsupported = heterogeneity_support_outputs(
        paired, heterogeneity
    )
    omnibus = heterogeneity_omnibus(paired)
    rng = rng_unpaired_sensitivity(
        arm, paired, freeze_sha, args.bootstrap_replicates
    )
    survival_summary, survival_curves = survival_outputs(arm)
    loo = leave_one_scale_out(paired)
    components, cost_transforms, tradeoff = cost_sensitivities(arm, weights)

    write_parquet(args.output / "marginal_effects.parquet", marginal)
    marginal.to_csv(args.output / "marginal_effects.csv", index=False)
    write_parquet(args.output / "heterogeneity_effects.parquet", heterogeneity)
    write_parquet(args.output / "two_way_heterogeneity.parquet", two_way)
    write_parquet(args.output / "heterogeneity_omnibus.parquet", omnibus)
    write_parquet(args.output / "heterogeneity_rankings.parquet", rankings)
    rankings.to_csv(args.output / "heterogeneity_rankings.csv", index=False)
    write_parquet(args.output / "unsupported_heterogeneity_cells.parquet", unsupported)
    unsupported.to_csv(args.output / "unsupported_heterogeneity_cells.csv", index=False)
    write_parquet(args.output / "rng_unpaired_sensitivity.parquet", rng)
    write_parquet(args.output / "survival_summaries.parquet", survival_summary)
    write_parquet(args.output / "survival_curves.parquet", survival_curves)
    write_parquet(args.output / "leave_one_scale_out.parquet", loo)
    write_parquet(args.output / "e04_cost_components.parquet", components)
    write_parquet(args.output / "cost_transform_sensitivity.parquet", cost_transforms)
    write_json(args.output / "bootstrap_stability.json", stability)
    write_json(args.output / "e04_resource_tradeoff.json", tradeoff)
    write_json(args.output / "effect_execution_summary.json", {
        "schemaVersion": "e02.s12.effect_execution_summary.v1",
        "researchStepId": "S12", "modelFreezeSha256": freeze_sha,
        "armRows": len(arm), "pairedRows": len(paired),
        "pairsPerEstimand": 2000, "bootstrapReplicates": args.bootstrap_replicates,
        "marginalRows": len(marginal), "heterogeneityRows": len(heterogeneity),
        "twoWayRows": len(two_way), "omnibusRows": len(omnibus),
        "heterogeneityRankingRows": len(rankings),
        "unsupportedCellRows": len(unsupported),
        "protectedRowsOnly": bool(arm.protected.all()),
        "costSchemaVersions": sorted(arm.costSchemaVersion.unique().tolist()),
        "success": True,
    })
    print(json.dumps({
        "armRows": len(arm), "pairedRows": len(paired),
        "marginalRows": len(marginal), "heterogeneityRows": len(heterogeneity),
        "bootstrapStability": stability["success"],
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--contrast-map", type=Path, default=DEFAULT_CONTRAST)
    parser.add_argument("--pairing", type=Path, default=DEFAULT_PAIRING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
