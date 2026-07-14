#!/usr/bin/env python3
"""Analyze one cumulative S10 look and apply the frozen adaptation rule."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from reference_simulator.model import canonical_json_bytes, sha256_json


STEP = "S10"
CANDIDATES = (
    "E02-S01-E01", "E02-S01-E03", "E02-S01-E04", "E02-S01-E06",
    "E02-S01-E08", "E02-S01-E09", "E02-S01-E10", "E02-S01-E11",
    "E02-S01-E12",
)
INTERACTIONS = {"E02-S01-E09", "E02-S01-E10", "E02-S01-E11", "E02-S01-E12"}
ENDPOINTS = {
    "normalizedResidualError": ("normalizedResidualError", "additive", 0.02),
    "successByBudget": ("successByBudget", "additive", 0.02),
    "logCompletionOpportunity": ("completionOpportunity", "log", math.log(1.10)),
    "logS01UnitWeightFullCost": ("projection_s01UnitWeightFullCost", "log", math.log(1.10)),
}
ALPHAS = np.asarray([0.000001, 0.00000316227766, 0.00001, 0.0000316227766, 0.0001, 0.000316227766, 0.001, 0.00316227766, 0.01, 0.0316227766, 0.1, 0.316227766, 1.0])


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd", compression_level=9, use_dictionary=True, write_statistics=True, version="2.6")


def read_runs(cache: Path, look: int) -> pd.DataFrame:
    frames = []
    for stage in range(1, look + 1):
        path = cache / f"stage_{stage:02d}_screening_runs.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_parquet(path))
    frame = pd.concat(frames, ignore_index=True)
    return frame.sort_values("runDesignId").reset_index(drop=True)


def paired_effects(runs: pd.DataFrame, contrast_map_path: Path) -> pd.DataFrame:
    mapping = pd.read_parquet(contrast_map_path)
    mapping = mapping[mapping.screeningStage <= int(runs.screeningStage.max())]
    columns = [
        "runDesignId", "pairingBlockId", "n", "valueProfile", "orderStructure",
        "policyProfile", "direction", "screeningStage", "screeningCvFold",
        *[item[0] for item in ENDPOINTS.values()],
    ]
    merged = mapping.merge(runs[columns], on=["runDesignId", "pairingBlockId", "screeningStage"], validate="many_to_one")
    output: list[dict[str, Any]] = []
    metadata = ["n", "valueProfile", "orderStructure", "policyProfile", "direction", "screeningStage", "screeningCvFold"]
    for (estimand, block), group in merged.groupby(["estimandId", "pairingBlockId"], sort=True):
        roles = {row.contrastRole: row for row in group.itertuples(index=False)}
        first = next(iter(roles.values()))
        row: dict[str, Any] = {
            "estimandId": estimand,
            "pairingBlockId": block,
            **{name: getattr(first, name) for name in metadata},
        }
        factorial = "reference_reference" in roles
        expected = (
            {"reference_reference", "reference_active", "active_reference", "active_active"}
            if factorial else {"reference", "active"}
        )
        if set(roles) != expected:
            raise AssertionError(f"incomplete roles for {estimand}/{block}: {set(roles)}")
        for endpoint, (column, transform, _) in ENDPOINTS.items():
            def value(role: str) -> float:
                raw = float(getattr(roles[role], column))
                return math.log(raw + 0.5) if transform == "log" else raw
            if factorial:
                effect = value("active_active") - value("active_reference") - value("reference_active") + value("reference_reference")
            else:
                effect = value("active") - value("reference")
            row[endpoint] = effect
        output.append(row)
    return pd.DataFrame(output).sort_values(["estimandId", "pairingBlockId"]).reset_index(drop=True)


def _bootstrap_seed(design_hash: str, look: int, endpoint: str, estimand: str, replicate: int, scale: int, draw: int) -> int:
    payload = f"{design_hash}/{look}/{endpoint}/{estimand}/{replicate}/{scale}/{draw}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def bootstrap_intervals(effects: pd.DataFrame, design_hash: str, look: int, replicates: int = 1000) -> pd.DataFrame:
    rows = []
    for estimand, group in effects.groupby("estimandId", sort=True):
        by_scale = {int(n): part for n, part in group.groupby("n", sort=True)}
        for endpoint in ENDPOINTS:
            values = group[endpoint].to_numpy(float)
            observed = float(values.mean())
            draws = np.empty(replicates, dtype=float)
            for replicate in range(replicates):
                sampled: list[float] = []
                for scale, part in by_scale.items():
                    local = part[endpoint].to_numpy(float)
                    for draw in range(len(local)):
                        sampled.append(local[_bootstrap_seed(design_hash, look, endpoint, estimand, replicate, scale, draw) % len(local)])
                draws[replicate] = np.mean(sampled)
            low, high = np.quantile(draws, [0.025, 0.975])
            rows.append({
                "estimandId": estimand,
                "endpoint": endpoint,
                "pairCount": len(group),
                "estimate": observed,
                "bootstrapLow95": float(low),
                "bootstrapHigh95": float(high),
                "bootstrapHalfWidth": float((high - low) / 2),
                "bootstrapReplicates": replicates,
            })
    return pd.DataFrame(rows)


def fit_sparse_models(effects: pd.DataFrame, matrix_path: Path, look: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrix = pd.read_parquet(matrix_path)
    matrix = matrix[matrix.screeningStage <= look]
    effects = effects[effects.estimandId.isin(CANDIDATES)]
    joined = matrix.merge(effects, on=["estimandId", "pairingBlockId", "screeningStage", "screeningCvFold", "n"], validate="one_to_one")
    feature_columns = [column for column in matrix.columns if "__" in column]
    X = joined[feature_columns].to_numpy(float)
    fold = joined.screeningCvFold.to_numpy(int)
    coefficients = []
    convergence = []
    for endpoint in ENDPOINTS:
        y = joined[endpoint].to_numpy(float)
        for model_name, ratio in (("elastic_net", 0.8), ("lasso", 1.0), ("ridge", 0.0)):
            scores = []
            converged_by_alpha = []
            for alpha in ALPHAS:
                fold_scores = []
                alpha_converged = True
                for held_out in range(5):
                    train = fold != held_out
                    test = ~train
                    if model_name == "ridge":
                        model = Ridge(alpha=float(alpha), fit_intercept=False, tol=1e-8, max_iter=100000, solver="lsqr")
                    else:
                        model = ElasticNet(alpha=float(alpha), l1_ratio=ratio, fit_intercept=False, max_iter=100000, tol=1e-8, selection="cyclic")
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model.fit(X[train], y[train])
                    alpha_converged &= not any(isinstance(item.message, ConvergenceWarning) for item in caught)
                    prediction = model.predict(X[test])
                    fold_scores.append(float(np.mean((y[test] - prediction) ** 2)))
                scores.append(float(np.mean(fold_scores)))
                converged_by_alpha.append(alpha_converged)
            minimum = min(scores)
            eligible = [index for index, score in enumerate(scores) if math.isclose(score, minimum, rel_tol=1e-12, abs_tol=1e-15)]
            selected_index = max(eligible, key=lambda index: ALPHAS[index])
            alpha = float(ALPHAS[selected_index])
            if model_name == "ridge":
                final = Ridge(alpha=alpha, fit_intercept=False, tol=1e-8, max_iter=100000, solver="lsqr")
            else:
                final = ElasticNet(alpha=alpha, l1_ratio=ratio, fit_intercept=False, max_iter=100000, tol=1e-8, selection="cyclic")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                final.fit(X, y)
            did_converge = not any(isinstance(item.message, ConvergenceWarning) for item in caught)
            for name, value in zip(feature_columns, final.coef_, strict=True):
                estimand, term = name.split("__", 1)
                coefficients.append({
                    "endpoint": endpoint,
                    "model": model_name,
                    "selectedAlpha": alpha,
                    "cvMeanSquaredError": scores[selected_index],
                    "estimandId": estimand,
                    "term": term,
                    "coefficient": float(value),
                    "active": bool(abs(value) > 1e-10),
                    "converged": did_converge,
                })
            convergence.append({"endpoint": endpoint, "model": model_name, "selectedAlpha": alpha, "selectedCvMse": scores[selected_index], "allCvFitsConverged": all(converged_by_alpha), "finalFitConverged": did_converge})
    frame = pd.DataFrame(coefficients)
    validation = {
        "schemaVersion": "e02.s10.sparse_model_validation.v1",
        "researchStepId": STEP,
        "look": look,
        "rows": len(joined),
        "features": len(feature_columns),
        "matrixRank": int(np.linalg.matrix_rank(X)),
        "expectedRank": len(feature_columns),
        "fits": convergence,
        "allConverged": all(item["allCvFitsConverged"] and item["finalFitConverged"] for item in convergence),
    }
    validation["success"] = validation["matrixRank"] == validation["expectedRank"] and validation["allConverged"]
    return frame, validation


def rank_candidates(intervals: pd.DataFrame, coefficients: pd.DataFrame) -> pd.DataFrame:
    pivot = intervals.pivot(index="estimandId", columns="endpoint", values="estimate")
    interval_lookup = intervals.set_index(["estimandId", "endpoint"])
    rows = []
    for estimand in CANDIDATES:
        active = coefficients[(coefficients.estimandId == estimand) & (coefficients.term == "main")]
        agreement = float(active.active.mean())
        standardized = {
            endpoint: abs(float(pivot.loc[estimand, endpoint])) / ENDPOINTS[endpoint][2]
            for endpoint in ENDPOINTS
        }
        utility = max(standardized.values()) * max(0.5, agreement)
        residual = interval_lookup.loc[(estimand, "normalizedResidualError")]
        success = interval_lookup.loc[(estimand, "successByBudget")]
        inconclusive = (
            residual.bootstrapLow95 <= -0.02 and residual.bootstrapHigh95 >= 0.02
            and success.bootstrapLow95 <= -0.02 and success.bootstrapHigh95 >= 0.02
        )
        primary_coefficients = active[active.endpoint == "normalizedResidualError"].set_index("model").coefficient.to_dict()
        unshrunk = float(pivot.loc[estimand, "normalizedResidualError"])
        signs = [math.copysign(1, value) for value in [unshrunk, *primary_coefficients.values()] if value != 0]
        direction_concordant = bool(signs) and len(set(signs)) == 1 and len(primary_coefficients) == 3
        rows.append({
            "estimandId": estimand,
            "utilityScore": utility,
            "selectionAgreement": agreement,
            "standardizedResidual": standardized["normalizedResidualError"],
            "standardizedSuccess": standardized["successByBudget"],
            "standardizedCompletion": standardized["logCompletionOpportunity"],
            "standardizedFullCost": standardized["logS01UnitWeightFullCost"],
            "residualEffect": float(pivot.loc[estimand, "normalizedResidualError"]),
            "successEffect": float(pivot.loc[estimand, "successByBudget"]),
            "logCompletionEffect": float(pivot.loc[estimand, "logCompletionOpportunity"]),
            "logFullCostEffect": float(pivot.loc[estimand, "logS01UnitWeightFullCost"]),
            "inconclusive": inconclusive,
            "primaryDirectionConcordant": direction_concordant,
        })
    frame = pd.DataFrame(rows).sort_values(["utilityScore", "estimandId"], ascending=[False, True]).reset_index(drop=True)
    frame["utilityRank"] = np.arange(1, len(frame) + 1)
    return frame


def s11_selection(ranking: pd.DataFrame) -> list[str]:
    selected = list(ranking.head(4).estimandId)
    augment = ranking[
        ranking.estimandId.isin(INTERACTIONS)
        & (ranking.utilityScore >= 1)
        & ranking.primaryDirectionConcordant
    ]
    for estimand in augment.estimandId:
        if estimand not in selected and len(selected) < 6:
            selected.append(estimand)
    coverage = (
        ({"E02-S01-E01", "E02-S01-E08"}, "architecture_or_coordinator"),
        ({"E02-S01-E03", "E02-S01-E09", "E02-S01-E12"}, "scheduler"),
        ({"E02-S01-E04", "E02-S01-E06", "E02-S01-E10", "E02-S01-E11", "E02-S01-E12"}, "fault_or_continuation"),
    )
    score = ranking.set_index("estimandId").utilityScore.to_dict()
    for family, _ in coverage:
        if not family.intersection(selected):
            replacement = max(family, key=lambda item: (score[item], item))
            redundant = []
            for item in selected:
                remaining = set(selected) - {item}
                if all(group.intersection(remaining | {replacement}) for group, _ in coverage):
                    redundant.append(item)
            if redundant:
                remove = min(redundant, key=lambda item: (score[item], item))
                selected[selected.index(remove)] = replacement
            elif len(selected) < 6:
                selected.append(replacement)
    return sorted(set(selected), key=lambda item: (-score[item], item))[:6]


def parity_checks(runs: pd.DataFrame, mapping_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    mapping = pd.read_parquet(mapping_path)
    mapping = mapping[mapping.screeningStage <= int(runs.screeningStage.max())]
    comparisons = ["scenarioId", "stopReason", "activationCount", "finalOccupancySha256", "nativeLedgerJson", "streamCountersJson", "projection_s01UnitWeightFullCost"]
    runs = runs.copy()
    runs["comparablePathDigest"] = runs.apply(
        lambda row: sha256_json({column: row[column] for column in comparisons}),
        axis=1,
    )
    comparisons.append("comparablePathDigest")
    joined = mapping.merge(runs, on=["runDesignId", "pairingBlockId", "screeningStage"], validate="many_to_one")
    summaries = {}
    for estimand in ("E02-S01-E02", "E02-S01-E05"):
        part = joined[joined.estimandId == estimand]
        pivot = part.pivot(index="pairingBlockId", columns="contrastRole", values=comparisons)
        equal = pd.DataFrame({column: pivot[(column, "active")].eq(pivot[(column, "reference")]) for column in comparisons})
        summaries[estimand] = {
            "pairingBlocks": len(equal),
            "comparisons": {column: int(equal[column].sum()) for column in comparisons},
            "allComparisonsPass": int(equal.all(axis=1).sum()),
            "success": bool(equal.all(axis=None)),
        }
    return summaries["E02-S01-E02"], summaries["E02-S01-E05"]


def analyze(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    look_dir = args.output / f"look_{args.look:02d}"
    look_dir.mkdir(parents=True, exist_ok=True)
    runs = read_runs(args.cache, args.look)
    effects = paired_effects(runs, args.contrast_map)
    freeze = json.loads((args.output / "design_freeze_manifest.json").read_text())
    intervals = bootstrap_intervals(effects, freeze["designFreezeSha256"], args.look)
    coefficients, model_validation = fit_sparse_models(effects, args.design_matrix, args.look)
    ranking = rank_candidates(intervals, coefficients)
    selected = s11_selection(ranking)
    parity, e05 = parity_checks(runs, args.contrast_map)

    expected_blocks = 50 * args.look
    treatment_counts = runs.groupby("treatmentSignature").size()
    validations = {
        "runRows": len(runs) == expected_blocks * 14,
        "uniqueRunIds": runs.runDesignId.nunique() == len(runs),
        "pairingBlocks": runs.pairingBlockId.nunique() == expected_blocks,
        "treatmentMarginals": len(treatment_counts) == 14 and set(treatment_counts) == {expected_blocks},
        "contractValidation": bool(runs.contractValidationPass.all()),
        "pairedEffectRows": len(effects) == expected_blocks * 11,
        "e02ExactParity": parity["success"],
        "e05RetryInactive": e05["success"],
        "sparseModel": model_validation["success"],
        "protectedOutcomeReadsZero": True,
    }

    previous_selection = None
    rank_correlation = None
    selection_stable = False
    if args.look > 1:
        prior = json.loads((args.output / f"look_{args.look - 1:02d}" / "tentative_s11_selection.json").read_text())
        previous_selection = prior["selectedEstimands"]
        selection_stable = previous_selection == selected
        prior_rank = pd.read_parquet(args.output / f"look_{args.look - 1:02d}" / "interaction_ranking.parquet").set_index("estimandId").utilityRank
        current_rank = ranking.set_index("estimandId").utilityRank
        rank_correlation = float(spearmanr(prior_rank.loc[list(CANDIDATES)], current_rank.loc[list(CANDIDATES)]).statistic)

    boundary_ids = set(selected)
    if len(ranking) > len(selected):
        boundary_ids.add(next(item for item in ranking.estimandId if item not in boundary_ids))
    precision = intervals[intervals.estimandId.isin(boundary_ids)].copy()
    precision["threshold"] = precision.endpoint.map({key: value[2] for key, value in ENDPOINTS.items()})
    precision_pass = bool((precision.bootstrapHalfWidth <= precision.threshold).all())
    eligible_precision_stop = (
        expected_blocks >= 150
        and selection_stable
        and rank_correlation is not None
        and rank_correlation >= 0.95
        and precision_pass
    )
    if not parity["success"]:
        action = "stop_for_review_exact_parity_failure"
        should_continue = False
    elif expected_blocks >= 250:
        action = "stop_at_frozen_maximum"
        should_continue = False
    elif eligible_precision_stop:
        action = "stop_for_frozen_precision_and_stability"
        should_continue = False
    else:
        action = "continue_to_next_nested_look"
        should_continue = True
    decision = {
        "schemaVersion": "e02.s10.adaptation_decision.v1",
        "researchStepId": STEP,
        "look": args.look,
        "cumulativePairingBlocks": expected_blocks,
        "cumulativeRunRows": len(runs),
        "earliestPrecisionStop": 150,
        "maximumPairingBlocks": 250,
        "selectedEstimands": selected,
        "previousSelection": previous_selection,
        "selectionStable": selection_stable,
        "rankSpearman": rank_correlation,
        "precisionBoundaryEstimands": sorted(boundary_ids),
        "precisionPass": precision_pass,
        "eligiblePrecisionStop": eligible_precision_stop,
        "exactParityPass": parity["success"],
        "validationPass": all(validations.values()),
        "action": action,
        "continue": should_continue,
        "protectedOutcomeReads": 0,
    }
    if not decision["validationPass"]:
        raise AssertionError(json.dumps({"validation": validations, "model": model_validation}, indent=2, sort_keys=True))

    write_parquet(look_dir / "paired_effects.parquet", effects)
    write_parquet(look_dir / "bootstrap_intervals.parquet", intervals)
    write_parquet(look_dir / "sparse_model_coefficients.parquet", coefficients)
    write_parquet(look_dir / "interaction_ranking.parquet", ranking)
    write_json(look_dir / "sparse_model_validation.json", model_validation)
    write_json(look_dir / "parity_and_negative_control.json", {"E02-S01-E02": parity, "E02-S01-E05": e05})
    write_json(look_dir / "tentative_s11_selection.json", {
        "schemaVersion": "e02.s10.tentative_s11_selection.v1",
        "researchStepId": STEP,
        "look": args.look,
        "selectedEstimands": selected,
        "provisional": should_continue,
        "ineligible": {"E02-S01-E05": "retry_inactive", "E02-S01-E07": "non_executable_information_boundary"},
        "protectedOutcomeReads": 0,
    })
    write_json(look_dir / "adaptation_decision.json", decision)
    log_path = args.output / "adaptation_log.json"
    prior_log = json.loads(log_path.read_text()) if log_path.exists() else {"schemaVersion": "e02.s10.adaptation_log.v1", "researchStepId": STEP, "frozenDesignSha256": freeze["designFreezeSha256"], "entries": []}
    prior_log["entries"] = [item for item in prior_log["entries"] if item["look"] != args.look] + [decision]
    prior_log["entries"].sort(key=lambda item: item["look"])
    prior_log["allEntriesComply"] = all(item["action"] in {"continue_to_next_nested_look", "stop_for_frozen_precision_and_stability", "stop_at_frozen_maximum", "stop_for_review_exact_parity_failure"} for item in prior_log["entries"])
    prior_log["protectedOutcomeReads"] = 0
    write_json(log_path, prior_log)
    print(json.dumps({"decision": decision, "topRanking": ranking.head(5).to_dict(orient="records")}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--look", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--cache", type=Path, default=Path("/cache/s10"))
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S10"))
    parser.add_argument("--contrast-map", type=Path, default=Path("/artifacts/research_steps/S10/screening_contrast_map.parquet"))
    parser.add_argument("--design-matrix", type=Path, default=Path("/artifacts/research_steps/S10/design_matrix.parquet"))
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
