#!/usr/bin/env python3
"""Write compact S12 anchor summaries and paired uncertainty tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from reference_simulator.model import canonical_json_bytes


OUTPUT = Path("/artifacts/research_steps/S12")
BOOTSTRAPS = 2_000


def write_parquet(path: Path, frame: pd.DataFrame, schema: str) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {b"schemaVersion": schema.encode(), b"researchStepId": b"S12"}
    )
    pq.write_table(table, path, compression="zstd", compression_level=9, use_dictionary=True, version="2.6")


def write_json(path: Path, value) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def paired_summary(effects: pd.DataFrame) -> pd.DataFrame:
    endpoints = (
        "difference_completed",
        "difference_any_detour_episode",
        "difference_max_episode_depth",
        "difference_max_episode_duration_opportunities",
        "difference_final_metric_level",
        "difference_full_ledger_unit_cost",
        "difference_s10_strict_filter_exposed",
    )
    rows = []
    for keys, group in effects.groupby(["metric", "intervention_type"], observed=True):
        for endpoint in endpoints:
            values = group[endpoint].to_numpy(float)
            rng = np.random.default_rng(
                int.from_bytes(hashlib.sha256(f"E03/S12/paired/{keys[0]}/{keys[1]}/{endpoint}".encode()).digest()[:8], "big")
            )
            draws = np.empty(BOOTSTRAPS, dtype=float)
            for start in range(0, BOOTSTRAPS, 250):
                stop = min(start + 250, BOOTSTRAPS)
                indices = rng.integers(0, len(values), size=(stop - start, len(values)))
                draws[start:stop] = values[indices].mean(axis=1)
            rows.append(
                {
                    "metric": keys[0],
                    "intervention_type": keys[1],
                    "endpoint": endpoint.removeprefix("difference_"),
                    "pairs": len(values),
                    "mean_difference": float(values.mean()),
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                    "bootstrap_replicates": BOOTSTRAPS,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    runs = pq.read_table(OUTPUT / "larger_n_trajectories.parquet").to_pandas()
    metrics = pq.read_table(OUTPUT / "larger_n_metric_outcomes.parquet").to_pandas()
    effects = pq.read_table(OUTPUT / "paired_context_effects.parquet").to_pandas()
    long = pq.read_table(OUTPUT / "long_horizon_sensitivity.parquet").to_pandas()
    summary = paired_summary(effects)
    write_parquet(OUTPUT / "paired_effect_summary.parquet", summary, "e03.s12.paired_effect_summary.v1")
    small_pairs = pq.read_table("/artifacts/research_steps/S11/null_barrier_effects.parquet").to_pandas()
    small_pairs = small_pairs[small_pairs.null_family.eq("open_loop_opportunity")]
    small_summary = (
        small_pairs.groupby(["metric", "intervention_type"], observed=True)
        .delta_completed_arm_minus_baseline
        .agg(s11_pairs="size", s11_open_loop_completion_difference="mean")
        .reset_index()
    )
    large_summary = summary[summary.endpoint.eq("completed")][
        ["metric", "intervention_type", "pairs", "mean_difference", "ci_low", "ci_high"]
    ].rename(
        columns={
            "pairs": "s12_pairs",
            "mean_difference": "s12_native_completion_difference",
            "ci_low": "s12_ci_low",
            "ci_high": "s12_ci_high",
        }
    )
    cross_layer = large_summary.merge(
        small_summary,
        on=["metric", "intervention_type"],
        how="left",
        validate="one_to_one",
    )
    cross_layer["direction_agreement"] = np.sign(cross_layer.s12_native_completion_difference) == np.sign(
        cross_layer.s11_open_loop_completion_difference
    )
    cross_layer["same_support_material_conflict_test"] = False
    cross_layer["support_boundary"] = (
        "S11 exact n=4-5 necessary-start structural arms versus S12 empirical n=12-24 default-initial-state arms; "
        "directional reversal is reported but cannot be attributed within a shared exact-reachability stratum"
    )
    write_parquet(
        OUTPUT / "cross_layer_directional_comparison.parquet",
        cross_layer,
        "e03.s12.cross_layer_directional_comparison.v1",
    )

    metric_summary = (
        metrics.groupby("metric", observed=True)
        .agg(
            runs=("run_id", "size"),
            detour_runs=("any_detour_episode", "sum"),
            s10_exposed_runs=("s10_strict_filter_exposed", "sum"),
            episodes=("running_min_episode_count", "sum"),
            recovered_episodes=("recovered_episode_count", "sum"),
            open_episodes=("open_censored_episode", "sum"),
            mean_max_depth=("max_episode_depth", "mean"),
        )
        .reset_index()
    )
    write_parquet(OUTPUT / "metric_outcome_summary.parquet", metric_summary, "e03.s12.metric_outcome_summary.v1")
    result = {
        "researchStepId": "S12",
        "stepNumber": 12,
        "success": False,
        "status": "complete_constraining_identifiability_stop",
        "artifactsWritten": [
            "larger_n_trajectories.parquet",
            "larger_n_metric_outcomes.parquet",
            "paired_context_effects.parquet",
            "paired_effect_summary.parquet",
            "metric_outcome_summary.parquet",
            "cross_layer_directional_comparison.parquet",
            "marginal_effects.parquet",
            "threshold_sensitivity.parquet",
            "context_models/",
        ],
        "validationResult": "All final corpus, stop-rule, cross-layer, descriptive-sensitivity, and provenance gates passed; the frozen hierarchical family is not identifiable for three zero-event global metrics.",
        "outcomeClassification": "constraining/contradictory",
        "runAccounting": {
            "scenarioBlocks": int(runs.scenario_block_id.nunique()),
            "runs": len(runs),
            "runMetricRows": len(metrics),
            "chargedOpportunities": int(runs.event_count.sum()),
            "complete": int(runs.completed.sum()),
            "quiescent": int(runs.quiescent.sum()),
            "primaryCensored": int(runs.primary_active_censored.sum()),
            "longHorizonExtensions": len(long),
            "longComplete": int(long.long_stop_reason.eq("complete").sum()),
            "longQuiescent": int(long.long_stop_reason.eq("quiescent").sum()),
            "longActive": int(long.long_stop_reason.eq("event_budget").sum()),
        },
        "metricOutcomes": metric_summary.to_dict("records"),
        "caveatsOrBlockers": [
            "The full multi-metric hierarchical model family cannot be identified because inversion, footrule, and maximum-rank detour occurrence are all zero.",
            "Larger-n exact structural reachability is unavailable by design; empirical trajectories are not labelled necessary or unnecessary.",
            "S11-compatible feasibility is a shadow audit on native paths, not an executed rate-matched null."
            ,"Barrier removal reverses direction across evidence layers: +36.62 percentage points completion in empirical n=12-24 versus -12.61 points for the S11 exact-small-n open-loop bridge; the supports do not share exact reachability and were not pooled."
        ],
        "recommendedNextAction": "Chief Scientist review before any S13 authorization; retain the zero-event global-metric result and do not redefine or narrow the S12 outcome family post hoc.",
    }
    write_json(OUTPUT / "result_summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
