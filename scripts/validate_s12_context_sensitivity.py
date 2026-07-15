#!/usr/bin/env python3
"""Validate S12 corpus, models, sensitivity analyses, and stopping boundary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from reference_simulator.model import canonical_json_bytes


OUTPUT = Path("/artifacts/research_steps/S12")
EXPECTED_RUNS = 96_768
EXPECTED_BLOCKS = 24_192
EXPECTED_METRIC_ROWS = EXPECTED_RUNS * 4
EXPECTED_TRACE_RUNS = 968


def write_json(path: Path, value) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def check(condition: bool, label: str, checks: list[dict]) -> None:
    checks.append({"check": label, "success": bool(condition)})


def trace_episode_summary(levels: list[int], event_ends: list[int], batch_width: int) -> dict[str, int | bool]:
    initial = int(levels[0])
    running_min = initial
    anchor = initial
    prior = initial
    opened = False
    positive = False
    start = -1
    current_depth = 0
    episodes = recovered = max_depth = max_duration = open_duration = positive_runs = 0
    sum_depth = sum_duration = 0
    max_initial = 0
    for level, event_end in zip(levels[1:], event_ends[1:], strict=True):
        level = int(level)
        delta = level - prior
        if delta > 0:
            if not positive:
                positive_runs += 1
                positive = True
        else:
            positive = False
        max_initial = max(max_initial, level - initial)
        if not opened:
            if level > running_min:
                opened = True
                anchor = running_min
                start = int(event_end) - batch_width
                current_depth = level - anchor
                max_depth = max(max_depth, current_depth)
                episodes += 1
            elif level < running_min:
                running_min = level
        else:
            current_depth = max(current_depth, level - anchor)
            max_depth = max(max_depth, current_depth)
            if level <= anchor:
                recovered += 1
                duration = int(event_end) - start
                max_duration = max(max_duration, duration)
                sum_duration += duration
                sum_depth += current_depth
                current_depth = 0
                opened = False
                running_min = min(running_min, level)
        prior = level
    if opened:
        open_duration = int(event_ends[-1]) - start
        max_duration = max(max_duration, open_duration)
    return {
        "running_min_episode_count": episodes,
        "recovered_episode_count": recovered,
        "open_censored_episode": opened,
        "max_episode_depth": max_depth,
        "sum_recovered_episode_depth": sum_depth,
        "max_episode_duration_opportunities": max_duration,
        "sum_recovered_episode_duration_opportunities": sum_duration,
        "open_episode_duration_opportunities": open_duration,
        "positive_run_count": positive_runs,
        "initial_level_excursion": max_initial,
    }


def retained_trace_episode_matches(traces: pd.DataFrame, metrics: pd.DataFrame, runs: pd.DataFrame) -> bool:
    metric_index = metrics.set_index(["run_id", "metric"])
    widths = runs.set_index("run_id").batch_width.to_dict()
    for run_id, trace in traces.groupby("run_id", sort=False):
        ordered = trace.sort_values("checkpoint_index")
        events = ordered.event_count.astype(int).tolist()
        for metric in ("adjacent_descents", "inversion_count", "spearman_footrule", "maximum_rank_error"):
            summary = trace_episode_summary(
                ordered[f"distance_{metric}"].astype(int).tolist(),
                events,
                int(widths[run_id]),
            )
            expected = metric_index.loc[(run_id, metric)]
            for field, value in summary.items():
                observed = expected[field]
                if isinstance(value, bool):
                    if bool(observed) != value:
                        return False
                elif int(observed) != int(value):
                    return False
    return True


def main() -> int:
    runs = pq.read_table(OUTPUT / "larger_n_trajectories.parquet").to_pandas()
    metrics = pq.read_table(OUTPUT / "larger_n_metric_outcomes.parquet").to_pandas()
    design = pq.read_table(OUTPUT / "larger_n_design.parquet").to_pandas()
    effects = pq.read_table(OUTPUT / "paired_context_effects.parquet").to_pandas()
    long = pq.read_table(OUTPUT / "long_horizon_sensitivity.parquet").to_pandas()
    traces = pq.read_table(OUTPUT / "retained_metric_traces.parquet").to_pandas()
    parity = pq.read_table(OUTPUT / "reference_engine_parity.parquet").to_pandas()
    bridge = pq.read_table(OUTPUT / "small_n_bridge_strata.parquet").to_pandas()
    cross_layer = pq.read_table(OUTPUT / "cross_layer_directional_comparison.parquet").to_pandas()
    marginal = pq.read_table(OUTPUT / "marginal_effects.parquet").to_pandas()
    thresholds = pq.read_table(OUTPUT / "threshold_sensitivity.parquet").to_pandas()
    segmentation = pq.read_table(OUTPUT / "episode_segmentation_sensitivity.parquet").to_pandas()
    rare = pq.read_table(OUTPUT / "rare_event_separation_diagnostics.parquet").to_pandas()
    identifiability = pq.read_table(OUTPUT / "context_models/identifiability_by_metric.parquet").to_pandas()
    model_status = pq.read_table(OUTPUT / "context_models/model_status.parquet").to_pandas()
    modeling_status = json.loads((OUTPUT / "context_models/modeling_status.json").read_text())
    heldout_status = json.loads((OUTPUT / "context_models/held_out_prediction_status.json").read_text())
    residual_status = json.loads((OUTPUT / "context_models/calibration_residual_status.json").read_text())
    design_diag = json.loads((OUTPUT / "context_models/design_diagnostics.json").read_text())
    immutability = json.loads((OUTPUT / "input_immutability.json").read_text())
    build = json.loads((OUTPUT / "build_summary.json").read_text())
    model_build = json.loads((OUTPUT / "model_build_summary.json").read_text())

    checks: list[dict] = []
    check(len(design) == len(runs) == EXPECTED_RUNS, "96,768 design/run rows", checks)
    check(design.scenario_block_id.nunique() == EXPECTED_BLOCKS, "24,192 unique matched blocks", checks)
    check(design.groupby("scenario_block_id").size().eq(4).all(), "four intervention arms per block", checks)
    check(set(design.intervention_type) == {"baseline", "remove", "move", "add"}, "all intervention types", checks)
    check(set(design.n) == {12, 18, 24}, "all larger-n sizes", checks)
    check(design.policy_profile.nunique() == 7 and design.scheduler_profile.nunique() == 3, "policy and scheduler coverage", checks)
    check(design.direction.nunique() == 2 and design.initial_disorder_profile.nunique() == 3, "direction and disorder coverage", checks)
    check(len(metrics) == EXPECTED_METRIC_ROWS and metrics.groupby("run_id").size().eq(4).all(), "four metric rows per run", checks)
    check(metrics.metric.nunique() == 4, "all independent metrics", checks)
    check(len(effects) == EXPECTED_BLOCKS * 3 * 4, "complete paired intervention effects", checks)
    check(runs.run_id.is_unique and design.run_id.is_unique, "unique run identities", checks)

    proposal_partition = (
        runs.ledger_proposals
        == runs.ledger_noOps
        + runs.ledger_rejections
        + runs.ledger_memoryUpdates
        + runs.ledger_acceptedSwaps
        + runs.ledger_conflictLosses
    )
    check(runs.ledger_activations.eq(runs.ledger_proposals).all(), "one proposal per activation", checks)
    check(proposal_partition.all(), "complete proposal ledger partition", checks)
    check(runs.ledger_displacedCells.eq(2 * runs.ledger_acceptedSwaps).all(), "two displacements per swap", checks)
    check((runs.event_count == runs.ledger_activations).all(), "event/activation agreement", checks)
    check((runs[["completed", "quiescent", "primary_active_censored"]].sum(axis=1) == 1).all(), "disjoint terminal/censoring partition", checks)

    requested = runs[["shadow_requested_swap", "shadow_requested_memory", "shadow_requested_unchanged"]].sum(axis=1)
    deficits = runs[["shadow_infeasible_swap", "shadow_infeasible_memory", "shadow_infeasible_unchanged"]].sum(axis=1)
    check(requested.eq(runs.event_count).all(), "S11 shadow requests partition every event", checks)
    check((deficits.gt(0) == runs.s11_shadow_rate_feasibility_deficit).all(), "shadow feasibility stratum exact", checks)
    check(metrics.exact_structural_reachability.eq("out_of_exact_domain_large_n").all(), "large-n exact reachability boundary", checks)
    check(set(bridge.evidence_layer) == {"exact_small_n_structural_bridge"}, "small-n exact bridge separated", checks)
    check(set(bridge.n).issubset({4, 5}), "small-n bridge size support", checks)
    check(
        (~cross_layer.loc[cross_layer.intervention_type.eq("remove"), "direction_agreement"]).all()
        and cross_layer.loc[cross_layer.intervention_type.isin(["add", "move"]), "direction_agreement"].all()
        and not cross_layer.same_support_material_conflict_test.any(),
        "cross-layer removal reversal retained outside shared exact support",
        checks,
    )

    required_long = runs.replicate_index.eq(0) & runs.primary_active_censored
    check(len(long) == int(required_long.sum()), "every replicate-zero active run extended", checks)
    check(long.prefix_match.all(), "all long-horizon prefixes exact", checks)
    check(set(long.long_stop_reason).issubset({"complete", "quiescent", "event_budget"}), "long-horizon terminal classes", checks)
    recurrent_expected = set(long.loc[long.long_stop_reason.eq("event_budget"), "run_id"])
    recurrent_observed = set(runs.loc[runs.recurrent_active_status.eq("recurrent_active_4x"), "run_id"])
    check(recurrent_expected == recurrent_observed, "recurrent-active status exact", checks)

    check(len(parity) == EXPECTED_TRACE_RUNS and parity.all_match.all(), "968 authoritative engine parity runs", checks)
    check(traces.run_id.nunique() == EXPECTED_TRACE_RUNS, "at least one percent trace retention", checks)
    check(traces.groupby("run_id").checkpoint_index.min().eq(0).all(), "retained traces include initial checkpoint", checks)
    check(retained_trace_episode_matches(traces, metrics, runs), "independent retained-trace episode reconstruction", checks)
    check(build["nativeChargedOpportunities"] == build["shadowAuditChargedOpportunities"], "full deterministic audit opportunity count", checks)

    check(design_diag["fullRank"] and design_diag["conditionBelow100"], "full-rank conditioned model design", checks)
    check(
        modeling_status["status"] == "predeclared_identifiability_gate_triggered"
        and modeling_status["success"] is False,
        "predeclared model identifiability stop recorded",
        checks,
    )
    check(
        model_status.status.str.startswith("not_").all()
        and not model_status.coefficients_written.any(),
        "no silently narrowed hierarchical model fit",
        checks,
    )
    check(
        heldout_status["status"] == "not_run_due_predeclared_identifiability_gate"
        and residual_status["status"] == "not_estimable_due_predeclared_identifiability_gate",
        "calibration, residual, and held-out non-estimability explicit",
        checks,
    )
    check(model_build["materialS11Conflict"] is False, "no material same-support S11 conflict", checks)

    occurrence_counts = metrics.groupby("metric").any_detour_episode.agg(["sum", "count"])
    zero_metrics = set(occurrence_counts.index[occurrence_counts["sum"].eq(0)])
    check(
        zero_metrics == {"inversion_count", "spearman_footrule", "maximum_rank_error"}
        and int(occurrence_counts.loc["adjacent_descents", "sum"]) == 63_177,
        "rare-event gate reproducibly triggered by three zero-event metrics",
        checks,
    )
    check(
        identifiability.identifiable_binary_occurrence.sum() == 1
        and set(identifiability.loc[~identifiability.identifiable_binary_occurrence, "metric"]) == zero_metrics,
        "metric identifiability table matches corpus",
        checks,
    )
    check(len(rare) > 0 and rare.complete_separation_cell.notna().all(), "separation cells retained and flagged", checks)
    check(not (OUTPUT / "context_models/held_out_prediction_by_fold.parquet").exists(), "held-out predictions correctly absent after gate", checks)

    check(segmentation.segmentation_definition.nunique() == 4, "four episode segmentation definitions", checks)
    check(set(thresholds.threshold_raw_units) == {0, 1, 2, 4}, "all threshold sensitivities", checks)
    threshold_monotone = all(
        group.sort_values("threshold_raw_units").exposure_rate.is_monotonic_decreasing
        for _, group in thresholds.groupby("metric")
    )
    check(threshold_monotone, "threshold exposure is monotone", checks)
    check(len(marginal) > 0 and marginal.bootstrap_replicates.eq(2000).all(), "2,000-draw clustered marginal effects", checks)
    check(
        all((OUTPUT / name).exists() for name in (
            "context_marginal_effects.png", "threshold_plots.png",
            "episode_segmentation_sensitivity.png", "held_out_calibration.png",
        )),
        "required context, threshold, segmentation, and gate-status figures",
        checks,
    )
    check(immutability["success"] and all(item["unchanged"] for item in immutability["inputs"]), "all execution inputs immutable", checks)
    check(not Path("/artifacts/research_steps/S13").exists(), "S13 not started", checks)

    failed = [item["check"] for item in checks if not item["success"]]
    result = {
        "schemaVersion": "e03.s12.validation.v1",
        "researchStepId": "S12",
        "stepNumber": 12,
        "success": not failed,
        "status": "passed_with_predeclared_identifiability_stop" if not failed else "failed",
        "checksPassed": len(checks) - len(failed),
        "checksTotal": len(checks),
        "checks": checks,
        "failures": failed,
        "validationResult": f"{'PASS' if not failed else 'FAIL'}: {len(checks) - len(failed)}/{len(checks)} corpus, stop-rule, descriptive-sensitivity, and provenance gates; hierarchical calibration/residual/held-out checks are correctly non-estimable after the frozen gate",
        "caveatsOrBlockers": [
            "The predeclared hierarchical family is not identifiable because three independent global metrics have zero detour events; no hierarchical coefficients, residuals, calibration, or held-out predictions were produced."
        ] if not failed else failed,
        "recommendedNextAction": "Write the canonical constraining S12 report and hand control back without starting S13." if not failed else "Stop S12 and report the validation failures without simplifying the design.",
    }
    write_json(OUTPUT / "validation_results.json", result)
    write_json(
        OUTPUT / "deterministic_replay_validation.json",
        {
            "researchStepId": "S12",
            "success": bool(not failed and parity.all_match.all()),
            "fullCorpusNativeShadowAuditMatches": EXPECTED_RUNS,
            "authoritativeReferenceParityMatches": int(parity.all_match.sum()),
            "longHorizonPrefixMatches": int(long.prefix_match.sum()),
        },
    )
    print(json.dumps(result, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
