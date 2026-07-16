#!/usr/bin/env python3
"""Summarize frozen S13 memory/information ablations without pooling evidence tiers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import binomtest

from reference_simulator.model import canonical_json_bytes


OUTPUT = Path("/artifacts/research_steps/S13")
S11 = Path("/artifacts/research_steps/S11")
S12 = Path("/artifacts/research_steps/S12")
BOOTSTRAPS = 2_000
METRICS = ("adjacent_descents", "inversion_count", "spearman_footrule", "maximum_rank_error")


def write_parquet(name: str, frame: pd.DataFrame, schema: str) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {b"schemaVersion": schema.encode(), b"researchStepId": b"S13"}
    )
    pq.write_table(
        table, OUTPUT / name, compression="zstd", compression_level=9,
        use_dictionary=True, version="2.6",
    )


def write_json(name: str, value: object) -> None:
    (OUTPUT / name).write_bytes(canonical_json_bytes(value) + b"\n")


def seed_for(*parts: object) -> int:
    text = "/".join(map(str, parts))
    return int.from_bytes(hashlib.sha256(f"E03/S13/{text}".encode()).digest()[:8], "big")


def clustered_interval(values: np.ndarray, clusters: np.ndarray, seed: int) -> tuple[float, float]:
    frame = pd.DataFrame({"value": values.astype(float), "cluster": clusters})
    grouped = frame.groupby("cluster", sort=True).value.agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAPS, dtype=float)
    size = len(grouped)
    for start in range(0, BOOTSTRAPS, 100):
        stop = min(start + 100, BOOTSTRAPS)
        index = rng.integers(0, size, size=(stop - start, size))
        draws[start:stop] = sums[index].sum(axis=1) / counts[index].sum(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def attach_clusters(effects: pd.DataFrame) -> pd.DataFrame:
    s12_design = pq.read_table(
        S12 / "larger_n_design.parquet", columns=["run_id", "scenario_block_id"]
    ).to_pandas().rename(columns={"run_id": "source_s12_run_id"})
    source = pq.read_table(
        OUTPUT / "capability_design.parquet",
        columns=["physical_pair_id", "source_s12_run_id"],
    ).to_pandas().drop_duplicates("physical_pair_id")
    effects = effects.merge(source, on="physical_pair_id", validate="many_to_one")
    effects = effects.merge(s12_design, on="source_s12_run_id", how="left", validate="many_to_one")
    small = effects.evidence_tier.eq("exact_small_n_anchor_empirical_path")
    effects["bootstrap_cluster_id"] = effects.scenario_block_id.astype("string")
    effects.loc[small, "bootstrap_cluster_id"] = (
        "small:" + effects.loc[small, "source_family_ordinal"].astype(str)
        + ":" + effects.loc[small, "source_state_ordinal"].astype(str)
    )
    if effects.bootstrap_cluster_id.isna().any():
        raise AssertionError("missing frozen bootstrap cluster")
    return effects


def effect_summary(effects: pd.DataFrame) -> pd.DataFrame:
    endpoint_columns = {
        "completion_risk_difference": "difference_completed",
        "detour_occurrence_risk_difference": "difference_any_detour_episode",
        "episode_count_difference": "difference_running_min_episode_count",
        "maximum_depth_difference": "difference_max_episode_depth",
        "final_metric_level_difference": "difference_final_metric_level",
        "successful_pair_full_cost_difference": "successful_pair_difference_full_ledger_unit_cost",
    }
    rows: list[dict] = []
    keys = ["evidence_tier", "contrast_id", "first_arm", "second_arm", "metric"]
    for group_keys, group in effects.groupby(keys, observed=True, sort=True):
        base = dict(zip(keys, group_keys, strict=True))
        for endpoint, column in endpoint_columns.items():
            eligible = group[column].notna().to_numpy()
            values = group.loc[eligible, column].to_numpy(float)
            clusters = group.loc[eligible, "bootstrap_cluster_id"].to_numpy()
            if not len(values):
                low = high = mean = np.nan
            else:
                mean = float(values.mean())
                low, high = clustered_interval(values, clusters, seed_for(*group_keys, endpoint))
            rows.append({
                **base,
                "endpoint": endpoint,
                "pairs_total": len(group),
                "pairs_estimand": len(values),
                "clusters": len(np.unique(clusters)) if len(values) else 0,
                "mean_difference_first_minus_second": mean,
                "ci_low": low,
                "ci_high": high,
                "bootstrap_replicates": BOOTSTRAPS,
                "estimand_population": (
                    "paired_runs_both_successful_only" if endpoint.startswith("successful_pair")
                    else "all_paired_runs_including_failure_quiescence_and_censoring"
                ),
                "shorter_failed_run_counted_as_efficient": False,
            })
    return pd.DataFrame(rows)


def stratified_effect_summary(effects: pd.DataFrame) -> pd.DataFrame:
    """Retain every prospective factor without promoting it to the primary test."""

    endpoint_columns = {
        "completion_risk_difference": "difference_completed",
        "detour_occurrence_risk_difference": "difference_any_detour_episode",
        "maximum_depth_difference": "difference_max_episode_depth",
        "final_metric_level_difference": "difference_final_metric_level",
    }
    dimensions = (
        "n", "policy_profile", "scheduler_profile", "intervention_type",
        "s07_exact_start_classification", "source_necessary_signature",
    )
    rows: list[dict] = []
    for dimension in dimensions:
        eligible = effects.copy()
        if dimension in ("s07_exact_start_classification", "source_necessary_signature"):
            eligible = eligible[eligible.evidence_tier.eq("exact_small_n_anchor_empirical_path")]
        eligible = eligible[eligible[dimension].notna()]
        keys = ["evidence_tier", "contrast_id", "first_arm", "second_arm", "metric", dimension]
        for group_keys, group in eligible.groupby(keys, observed=True, sort=True):
            base = dict(zip(keys[:-1], group_keys[:-1], strict=True))
            for endpoint, column in endpoint_columns.items():
                values = group[column].to_numpy(float)
                clusters = group.bootstrap_cluster_id.to_numpy()
                low, high = clustered_interval(
                    values, clusters, seed_for("stratified", dimension, *group_keys, endpoint)
                )
                rows.append({
                    **base, "stratum_dimension": dimension,
                    "stratum_value": str(group_keys[-1]), "endpoint": endpoint,
                    "pairs": len(group), "clusters": len(np.unique(clusters)),
                    "mean_difference_first_minus_second": float(values.mean()),
                    "ci_low": low, "ci_high": high, "bootstrap_replicates": BOOTSTRAPS,
                    "primary_selection_role": "prospectively_retained_descriptive_stratum_not_used_for_selection",
                })
    return pd.DataFrame(rows)


def zero_event_diagnostics(metrics: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in metrics.groupby(["evidence_tier", "capability_arm", "metric"], observed=True):
        events = int(group.any_detour_episode.sum())
        rows.append({
            "record_type": "arm_event_count", "evidence_tier": keys[0],
            "capability_arm": keys[1], "contrast_id": None, "metric": keys[2],
            "runs_or_pairs": len(group), "first_only_events": None,
            "second_only_events": None, "both_event": None,
            "event_count": events,
            "zero_event_status": "zero_events_valid_result" if events == 0 else "events_observed",
            "paired_risk_difference": None, "exact_discordance_pvalue": None,
            "odds_ratio_fitted": False,
        })
    for keys, group in effects.groupby(["evidence_tier", "contrast_id", "metric"], observed=True):
        first = group.first_any_detour_episode.to_numpy(bool)
        second = group.second_any_detour_episode.to_numpy(bool)
        first_only = int((first & ~second).sum())
        second_only = int((~first & second).sum())
        discordant = first_only + second_only
        pvalue = float(binomtest(first_only, discordant, 0.5).pvalue) if discordant else 1.0
        if not first.any() and not second.any():
            status = "joint_zero_rd_exactly_zero"
        elif not first.any() or not second.any():
            status = "one_arm_zero_exact_discordance_reported"
        else:
            status = "events_in_both_arms"
        rows.append({
            "record_type": "paired_zero_handling", "evidence_tier": keys[0],
            "capability_arm": None, "contrast_id": keys[1], "metric": keys[2],
            "runs_or_pairs": len(group), "first_only_events": first_only,
            "second_only_events": second_only, "both_event": int((first & second).sum()),
            "event_count": int(first.sum() + second.sum()), "zero_event_status": status,
            "paired_risk_difference": float(first.mean() - second.mean()),
            "exact_discordance_pvalue": pvalue, "odds_ratio_fitted": False,
        })
    return pd.DataFrame(rows)


def capability_summary(runs: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    run_summary = runs.groupby(["evidence_tier", "capability_arm"], observed=True).agg(
        runs=("run_id", "size"), complete=("completed", "sum"), quiescent=("quiescent", "sum"),
        primary_active=("primary_active_censored", "sum"),
        recurrent_active=("recurrent_active_status", lambda x: int((x == "recurrent_active_4x").sum())),
        mean_cost=("full_ledger_unit_cost", "mean"),
        action_override_runs=("behavioral_action_overrides", lambda x: int((x > 0).sum())),
        maximum_observed_radius=("maximum_radius_read", "max"),
        capability_memory_writes=("capability_memory_writes", "sum"),
    ).reset_index()
    run_summary["completion_rate"] = run_summary.complete / run_summary.runs
    metric_summary = metrics.groupby(["evidence_tier", "capability_arm", "metric"], observed=True).agg(
        metric_runs=("run_id", "size"), detour_runs=("any_detour_episode", "sum"),
        episodes=("running_min_episode_count", "sum"), mean_max_depth=("max_episode_depth", "mean"),
        mean_final_metric=("final_metric_level", "mean"),
    ).reset_index()
    metric_summary["detour_rate"] = metric_summary.detour_runs / metric_summary.metric_runs
    return metric_summary.merge(run_summary, on=["evidence_tier", "capability_arm"], validate="many_to_one")


def barrier_effects(runs: pd.DataFrame) -> pd.DataFrame:
    s12_design = pq.read_table(
        S12 / "larger_n_design.parquet", columns=["run_id", "scenario_block_id"]
    ).to_pandas().rename(columns={"run_id": "source_s12_run_id"})
    large = runs[runs.evidence_tier.eq("empirical_large_n")].merge(
        s12_design, on="source_s12_run_id", validate="many_to_one"
    )
    pivot = large.pivot(index=["scenario_block_id", "capability_arm"], columns="intervention_type", values="completed")
    rows = []
    for arm in sorted(large.capability_arm.unique()):
        selected = pivot.xs(arm, level="capability_arm")
        for intervention in ("remove", "move", "add"):
            values = selected[intervention].astype(float).to_numpy() - selected.baseline.astype(float).to_numpy()
            low, high = clustered_interval(values, selected.index.to_numpy(), seed_for("barrier", arm, intervention))
            rows.append({
                "evidence_tier": "empirical_large_n", "capability_arm": arm,
                "intervention_type": intervention, "scenario_blocks": len(values),
                "completion_risk_difference_vs_baseline": float(values.mean()),
                "ci_low": low, "ci_high": high, "bootstrap_replicates": BOOTSTRAPS,
                "exact_reachability": "out_of_exact_domain_large_n",
            })
    return pd.DataFrame(rows)


def cross_layer_boundary(barrier: pd.DataFrame) -> pd.DataFrame:
    prior = pq.read_table(S12 / "cross_layer_directional_comparison.parquet").to_pandas()
    prior = prior[prior.metric.eq("adjacent_descents")]
    rows: list[dict] = []
    for row in prior.itertuples(index=False):
        rows.extend([
            {"source_step": "S11", "evidence_layer": "exact_small_n_open_loop_bridge",
             "capability_arm": "open_loop_opportunity", "intervention_type": row.intervention_type,
             "completion_difference": row.s11_open_loop_completion_difference,
             "pooled_estimate": False},
            {"source_step": "S12", "evidence_layer": "empirical_large_n_native",
             "capability_arm": "native", "intervention_type": row.intervention_type,
             "completion_difference": row.s12_native_completion_difference,
             "pooled_estimate": False},
        ])
    for row in barrier.itertuples(index=False):
        rows.append({"source_step": "S13", "evidence_layer": "empirical_large_n_capability_ablation",
                     "capability_arm": row.capability_arm, "intervention_type": row.intervention_type,
                     "completion_difference": row.completion_risk_difference_vs_baseline,
                     "pooled_estimate": False})
    frame = pd.DataFrame(rows)
    frame["support_boundary"] = (
        "Exact n=4-5 necessary-start and empirical n=12-24 default-initial-state supports are separate; "
        "directional comparisons are descriptive and barrier-removal effects are never pooled."
    )
    return frame


def complexity_effects(complexity: pd.DataFrame, summary: pd.DataFrame, capability: pd.DataFrame) -> pd.DataFrame:
    completion = summary[
        summary.endpoint.eq("completion_risk_difference")
        & summary.contrast_id.isin(["single_failure_bit", "single_counter_2bit", "single_recent_direction", "single_radius2", "full_vs_native"])
    ].copy()
    completion = completion[completion.metric.eq("adjacent_descents")]
    completion = completion.rename(columns={"first_arm": "arm"})
    result = completion.merge(complexity, on="arm", validate="many_to_one")
    result["absolute_completion_effect"] = result.mean_difference_first_minus_second.abs()
    result["bits_plus_radius"] = result.declared_stored_bits_per_actor + result.added_sensing_radius
    rates = capability[capability.metric.eq("adjacent_descents")][
        ["evidence_tier", "capability_arm", "completion_rate", "detour_rate"]
    ].rename(columns={"capability_arm": "arm"})
    return result.merge(rates, on=["evidence_tier", "arm"], validate="one_to_one")


def main() -> None:
    runs = pq.read_table(OUTPUT / "capability_runs.parquet").to_pandas()
    metrics = pq.read_table(OUTPUT / "memory_information_results.parquet").to_pandas()
    effects = attach_clusters(pq.read_table(OUTPUT / "paired_ablation_effects.parquet").to_pandas())
    complexity = pq.read_table(OUTPUT / "complexity_table.parquet").to_pandas()

    summary = effect_summary(effects)
    stratified = stratified_effect_summary(effects)
    zero = zero_event_diagnostics(metrics, effects)
    capability = capability_summary(runs, metrics)
    barrier = barrier_effects(runs)
    cross_layer = cross_layer_boundary(barrier)
    pareto = complexity_effects(complexity, summary, capability)

    write_parquet("ablation_effect_summary.parquet", summary, "e03.s13.ablation_effect_summary.v1")
    write_parquet("stratified_ablation_effects.parquet", stratified, "e03.s13.stratified_ablation_effects.v1")
    write_parquet("metric_zero_event_diagnostics.parquet", zero, "e03.s13.metric_zero_event_diagnostics.v1")
    write_parquet("capability_summary.parquet", capability, "e03.s13.capability_summary.v1")
    write_parquet("barrier_effects_by_capability.parquet", barrier, "e03.s13.barrier_effects_by_capability.v1")
    write_parquet("cross_layer_boundary_audit.parquet", cross_layer, "e03.s13.cross_layer_boundary_audit.v1")
    write_parquet("complexity_effect_pareto.parquet", pareto, "e03.s13.complexity_effect_pareto.v1")
    terminal = (
        runs.groupby(["evidence_tier", "capability_arm", "stop_reason", "recurrent_active_status"], observed=True)
        .agg(runs=("run_id", "size"), mean_event_count=("event_count", "mean"),
             mean_full_cost=("full_ledger_unit_cost", "mean"))
        .reset_index()
    )
    write_parquet("terminal_recurrence_summary.parquet", terminal, "e03.s13.terminal_recurrence_summary.v1")

    primary = summary[
        summary.evidence_tier.eq("empirical_large_n")
        & summary.endpoint.eq("completion_risk_difference")
        & summary.metric.eq("adjacent_descents")
    ]
    full = primary[primary.contrast_id.eq("full_vs_native")].iloc[0]
    leave_one_out = primary[primary.contrast_id.str.startswith("ablate_")].copy()
    full_gate = (
        abs(full.mean_difference_first_minus_second) >= 0.05
        and (full.ci_low > 0 or full.ci_high < 0)
    )
    loo_gate_rows = leave_one_out[
        leave_one_out.mean_difference_first_minus_second.abs().ge(0.02)
        & ((leave_one_out.ci_low > 0) | (leave_one_out.ci_high < 0))
    ]
    supportive = bool(full_gate and len(loo_gate_rows) > 0)
    mechanism = pd.DataFrame({
        "primary_full_vs_native_effect": [float(full.mean_difference_first_minus_second)],
        "primary_full_ci_low": [float(full.ci_low)], "primary_full_ci_high": [float(full.ci_high)],
        "full_gate_met": [bool(full_gate)], "leave_one_out_gate_count": [len(loo_gate_rows)],
        "leave_one_out_gate_contrasts": [",".join(loo_gate_rows.contrast_id.tolist())],
        "supportive_criterion_met": [supportive],
        "criterion": ["|full-native completion RD| >= 0.05 with clustered interval excluding zero and >=1 leave-one-out |RD| >= 0.02 with interval excluding zero"],
    })
    write_parquet("mechanism_candidate_summary.parquet", mechanism, "e03.s13.mechanism_candidate_summary.v1")

    global_large = zero[
        zero.record_type.eq("arm_event_count") & zero.evidence_tier.eq("empirical_large_n")
        & zero.metric.isin(["inversion_count", "spearman_footrule", "maximum_rank_error"])
    ]
    result = {
        "researchStepId": "S13", "stepNumber": 13, "success": True,
        "status": "complete", "outcomeClassification": "supportive" if supportive else "null",
        "supportiveCriterionMet": supportive,
        "runAccounting": {"runs": len(runs), "runMetricRows": len(metrics),
                          "pairedEffectRows": len(effects), "evidenceTiers": 2,
                          "capabilityArms": int(runs.capability_arm.nunique())},
        "primaryFullVsNativeCompletionRiskDifference": float(full.mean_difference_first_minus_second),
        "primaryFullVsNativeInterval": [float(full.ci_low), float(full.ci_high)],
        "leaveOneOutGateContrasts": loo_gate_rows.contrast_id.tolist(),
        "largeGlobalMetricZeroArmCells": int(global_large.event_count.eq(0).sum()),
        "largeGlobalMetricArmCells": len(global_large),
        "zeroEventHandling": "Retained as valid joint-zero or one-arm-zero results; paired risk differences and exact discordance are reported; no odds ratios or post-hoc metric collapse.",
        "evidenceBoundary": "Exact small-n anchor paths and empirical larger-n scenarios were estimated separately; opposing barrier-removal effects were not pooled.",
    }
    write_json("result_summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
