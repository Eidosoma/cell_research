#!/usr/bin/env python3
"""Validate the complete S13 corpus, traces, estimands, and evidence boundaries."""

from __future__ import annotations

import ast
import inspect
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from reference_simulator.model import canonical_json_bytes
from src.detours import memory_information as mi


OUTPUT = Path("/artifacts/research_steps/S13")
EXPECTED_RUNS = 257_400
EXPECTED_METRICS = 1_029_600
EXPECTED_EFFECTS = 926_640
ARMS = set(mi.ARM_NAMES)
METRICS = set(mi.METRIC_NAMES)


def check(condition: bool, label: str, checks: list[dict]) -> None:
    checks.append({"check": label, "success": bool(condition)})


def trace_episode(levels: np.ndarray, events: np.ndarray, width: int) -> dict[str, int | bool]:
    initial = int(levels[0])
    running_min = initial
    anchor = initial
    prior = initial
    opened = positive = False
    start = -1
    current_depth = 0
    episodes = recovered = max_depth = max_duration = open_duration = positive_runs = 0
    sum_depth = sum_duration = max_initial = 0
    for level, event_end in zip(levels[1:], events[1:], strict=True):
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
                start = int(event_end) - width
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
                opened = False
                current_depth = 0
                running_min = min(running_min, level)
        prior = level
    if opened:
        open_duration = int(events[-1]) - start
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


def trace_reconstruction(traces: pd.DataFrame, metrics: pd.DataFrame, runs: pd.DataFrame) -> bool:
    outcome = metrics.set_index(["run_id", "metric"])
    widths = runs.set_index("run_id").batch_width.to_dict()
    for run_id, frame in traces.groupby("run_id", sort=False):
        frame = frame.sort_values("metric_checkpoint_index")
        events = frame.event_count.to_numpy(int)
        for metric in mi.METRIC_NAMES:
            observed = trace_episode(frame[f"distance_{metric}"].to_numpy(int), events, int(widths[run_id]))
            expected = outcome.loc[(run_id, metric)]
            for field, value in observed.items():
                if isinstance(value, bool):
                    if bool(expected[field]) != value:
                        return False
                elif int(expected[field]) != int(value):
                    return False
    return True


def main() -> None:
    design = pq.read_table(OUTPUT / "capability_design.parquet").to_pandas()
    runs = pq.read_table(OUTPUT / "capability_runs.parquet").to_pandas()
    metrics = pq.read_table(OUTPUT / "memory_information_results.parquet").to_pandas()
    effects = pq.read_table(OUTPUT / "paired_ablation_effects.parquet").to_pandas()
    summary = pq.read_table(OUTPUT / "ablation_effect_summary.parquet").to_pandas()
    stratified = pq.read_table(OUTPUT / "stratified_ablation_effects.parquet").to_pandas()
    complexity = pq.read_table(OUTPUT / "complexity_table.parquet").to_pandas()
    coverage = pq.read_table(OUTPUT / "exact_start_coverage.parquet").to_pandas()
    long = pq.read_table(OUTPUT / "long_horizon_sensitivity.parquet").to_pandas()
    parity = pq.read_table(OUTPUT / "native_s12_parity.parquet").to_pandas()
    trace_index = pq.read_table(OUTPUT / "retained_metric_trace_index.parquet").to_pandas()
    traces = pq.read_table(OUTPUT / "retained_metric_traces.parquet").to_pandas()
    zero = pq.read_table(OUTPUT / "metric_zero_event_diagnostics.parquet").to_pandas()
    cross = pq.read_table(OUTPUT / "cross_layer_boundary_audit.parquet").to_pandas()
    terminal_summary = pq.read_table(OUTPUT / "terminal_recurrence_summary.parquet").to_pandas()
    replay = json.loads((OUTPUT / "deterministic_replay_validation.json").read_text())
    immutability = json.loads((OUTPUT / "input_immutability.json").read_text())
    trace_build = json.loads((OUTPUT / "retained_trace_build_summary.json").read_text())
    result = json.loads((OUTPUT / "result_summary.json").read_text())

    checks: list[dict] = []
    check(len(design) == len(runs) == EXPECTED_RUNS, "257,400 design and run rows", checks)
    check(len(metrics) == EXPECTED_METRICS and metrics.groupby("run_id").size().eq(4).all(), "1,029,600 complete run-metric rows", checks)
    check(len(effects) == EXPECTED_EFFECTS, "926,640 declared paired-effect rows", checks)
    check(set(runs.capability_arm) == ARMS and runs.capability_arm.nunique() == 10, "all ten frozen arms", checks)
    check(set(metrics.metric) == METRICS, "all four independent metrics retained", checks)
    check(int(runs.evidence_tier.eq("empirical_large_n").sum()) == 241_920 and int(runs.evidence_tier.eq("exact_small_n_anchor_empirical_path").sum()) == 15_480, "large and small evidence-tier accounting", checks)
    check(runs.run_id.is_unique and design.run_id.is_unique and set(runs.run_id) == set(design.run_id), "unique aligned run identities", checks)
    check(runs.groupby("physical_pair_id").size().eq(10).all() and runs.physical_pair_id.nunique() == 25_740, "ten-arm physical pairing", checks)
    check(design.groupby("physical_pair_id").stream_root_hex.nunique().eq(1).all(), "paired raw opportunity roots identical", checks)
    check(design.groupby("physical_pair_id").pre_physical_state_sha256.nunique().eq(1).all(), "paired pre-intervention physical states identical", checks)
    check(design.groupby("physical_pair_id").capability_arm.nunique().eq(10).all(), "every pair contains every arm", checks)

    unique_coverage = coverage[["family_ordinal", "state_ordinal", "capability_applicability"]].drop_duplicates()
    check(len(unique_coverage) == 407 and int(unique_coverage.capability_applicability.eq("eligible_local_cell_policy").sum()) == 387, "all 407 exact starts and 387 eligible starts retained", checks)
    check(int(unique_coverage.capability_applicability.eq("not_applicable_traditional_global_controller").sum()) == 20, "20 traditional-controller starts explicitly ineligible", checks)
    check(len(coverage) == 407 * 4 and set(coverage.metric) == METRICS, "four-metric exact-start coverage", checks)
    small = runs[runs.evidence_tier.eq("exact_small_n_anchor_empirical_path")]
    check(small.groupby(["source_family_ordinal", "source_state_ordinal"]).replicate_index.nunique().eq(4).all(), "four replicates per eligible exact start", checks)
    check(metrics.loc[metrics.evidence_tier.eq("empirical_large_n"), "s07_exact_start_classification"].eq("out_of_exact_domain_large_n").all(), "large-n exact reachability boundary preserved", checks)

    ledger_partition = runs.ledger_proposals.eq(
        runs.ledger_noOps + runs.ledger_rejections + runs.ledger_memoryUpdates
        + runs.ledger_acceptedSwaps + runs.ledger_conflictLosses
    )
    check(runs.ledger_activations.eq(runs.ledger_proposals).all() and runs.event_count.eq(runs.ledger_activations).all(), "event, activation, and proposal identity", checks)
    check(ledger_partition.all(), "complete proposal outcome partition", checks)
    check(runs.ledger_displacedCells.eq(2 * runs.ledger_acceptedSwaps).all(), "two displaced cells per accepted swap", checks)
    cost_columns = [column for column in runs if column.startswith("ledger_")]
    check(runs.full_ledger_unit_cost.eq(runs[cost_columns].sum(axis=1) + runs.capability_memory_writes).all(), "full cost and capability memory accounting", checks)
    check(runs[["completed", "quiescent", "primary_active_censored"]].sum(axis=1).eq(1).all(), "terminal/censoring partition", checks)
    check(runs.loc[runs.primary_active_censored, "event_count"].eq(runs.loc[runs.primary_active_censored, "event_budget"]).all(), "event-budget censoring exact", checks)
    complete_ids = set(runs.loc[runs.completed, "run_id"])
    check(metrics.loc[metrics.run_id.isin(complete_ids), "final_metric_level"].eq(0).all(), "complete terminal states agree with every metric goal", checks)

    flags = {name: tuple(map(bool, mi.ARM_FLAGS[index])) for index, name in enumerate(mi.ARM_NAMES)}
    complexity_ok = True
    for row in complexity.itertuples(index=False):
        failure, counter, recent, radius = flags[row.arm]
        product = (2 if failure else 1) * (4 if counter else 1) * (3 if recent else 1)
        minimum = math.ceil(math.log2(product)) if product > 1 else 0
        stored = int(failure) + 2 * int(counter) + 2 * int(recent)
        complexity_ok &= (
            row.mutable_state_count_per_actor == product
            and row.minimum_bits_per_actor == minimum
            and row.declared_stored_bits_per_actor == stored
            and row.added_sensing_radius == (2 if radius else 0)
        )
    check(complexity_ok and len(complexity) == 10, "complexity product, minimum bits, stored bits, and sensing radius exact", checks)
    radius_arms = {name for name, flag in flags.items() if flag[3]}
    check(runs.loc[runs.capability_arm.isin(radius_arms), "maximum_radius_read"].le(2).all() and runs.loc[~runs.capability_arm.isin(radius_arms), "maximum_radius_read"].eq(0).all(), "observed sensing never exceeds radius two", checks)
    memoryless = {name for name, flag in flags.items() if not any(flag[:3])}
    check(runs.loc[runs.capability_arm.isin(memoryless), "capability_memory_writes"].eq(0).all(), "memoryless arms have no hidden memory writes", checks)
    initial_full = traces[traces.metric_checkpoint_index.eq(0)].merge(
        runs[["run_id", "physical_pair_id"]], on="run_id", validate="one_to_one"
    )
    check(initial_full.groupby("physical_pair_id").physical_state_fingerprint_u64.nunique().eq(1).all(), "added memory resets to a common paired physical start", checks)

    source_tree = ast.parse(inspect.getsource(mi.capability_proposal.py_func))
    called = {node.func.id for node in ast.walk(source_tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    forbidden_calls = {"metric_profile", "swap_metric_delta", "_terminal_code", "_is_complete", "_state_fingerprint", "counter_draw"}
    check(not (called & forbidden_calls), "capability proposal gateway has no metric, terminal, future-stream, or analysis call", checks)
    parameters = set(inspect.signature(mi.capability_proposal.py_func).parameters)
    forbidden_parameters = {"metric", "reachability", "terminal", "outcome", "future", "rank"}
    check(not (parameters & forbidden_parameters), "capability proposal signature contains no hidden global outcome input", checks)
    check(runs.loc[runs.capability_arm.eq("native"), "behavioral_action_overrides"].eq(0).all(), "native arm has zero capability overrides", checks)
    overridden = runs.behavioral_action_overrides.gt(0)
    check(runs.loc[overridden, "first_action_override_event"].ge(0).all() and runs.loc[overridden, "first_override_pre_physical_fingerprint_u64"].notna().all(), "first override pre-state is audited", checks)

    check(len(parity) == 24_192 and parity.all_match.all(), "24,192 native rows exactly reproduce frozen S12", checks)
    check(replay["success"] and replay["runsReplayed"] == EXPECTED_RUNS and all(replay["matches"].values()), "full deterministic replay matches every declared field", checks)
    required_long = runs.replicate_index.eq(0) & runs.primary_active_censored
    check(len(long) == int(required_long.sum()) == 25_773 and long.prefix_match.all(), "all 25,773 required long-horizon prefixes exact", checks)
    recurrent_expected = set(long.loc[long.long_stop_reason.eq("event_budget"), "run_id"])
    recurrent_observed = set(runs.loc[runs.recurrent_active_status.eq("recurrent_active_4x"), "run_id"])
    check(recurrent_expected == recurrent_observed, "recurrent-active labels agree with long runs", checks)

    check(trace_build["instrumentedKernelEquivalentRuns"] == 3_194 and trace_build["largeTraceRuns"] == 2_420 and trace_build["smallTraceRuns"] == 774, "outcome-blind retained-trace quotas and source equivalence", checks)
    check(traces.run_id.nunique() == 3_194 and len(trace_index) == 6_388, "full traces and compact indexes cover all retained runs", checks)
    check(traces.groupby("run_id").metric_checkpoint_index.min().eq(0).all() and traces.groupby("run_id").event_count.apply(lambda x: x.is_monotonic_increasing).all(), "trace initial rows and monotone event ordering", checks)
    check(trace_reconstruction(traces, metrics, runs), "independent full-trace metric and episode reconstruction", checks)
    terminal_trace = traces.sort_values(["run_id", "metric_checkpoint_index"]).groupby("run_id", as_index=False).tail(1)
    terminal_trace = terminal_trace.merge(runs[["run_id", "event_count", "physical_state_fingerprint_u64", "full_state_fingerprint_u64"]], on="run_id", suffixes=("_trace", "_run"), validate="one_to_one")
    check(terminal_trace.event_count_trace.eq(terminal_trace.event_count_run).all() and terminal_trace.physical_state_fingerprint_u64_trace.eq(terminal_trace.physical_state_fingerprint_u64_run).all() and terminal_trace.full_state_fingerprint_u64_trace.eq(terminal_trace.full_state_fingerprint_u64_run).all(), "retained terminal events and physical/full hashes exact", checks)

    arm_zero = zero[zero.record_type.eq("arm_event_count")]
    check(set(arm_zero.metric) == METRICS and len(arm_zero) == 2 * 10 * 4, "zero-event status exists for every tier-arm-metric cell", checks)
    native_global = arm_zero[
        arm_zero.capability_arm.eq("native")
        & arm_zero.evidence_tier.eq("empirical_large_n")
        & arm_zero.metric.ne("adjacent_descents")
    ]
    check(
        len(native_global) == 3 and native_global.event_count.eq(0).all(),
        "S12 native global-metric zero result retained on its empirical larger-n support",
        checks,
    )
    check((~zero.odds_ratio_fitted).all(), "no odds ratio or continuity correction fitted", checks)
    joint = zero[zero.zero_event_status.eq("joint_zero_rd_exactly_zero")]
    check(joint.paired_risk_difference.fillna(0).eq(0).all() and joint.first_only_events.fillna(0).eq(0).all() and joint.second_only_events.fillna(0).eq(0).all(), "joint-zero convention exact", checks)
    check(summary.bootstrap_replicates.eq(2_000).all() and summary.estimand_population.notna().all(), "2,000-draw clustered paired intervals and explicit estimands", checks)
    check(
        set(stratified.stratum_dimension)
        == {"n", "policy_profile", "scheduler_profile", "intervention_type", "s07_exact_start_classification", "source_necessary_signature"}
        and stratified.bootstrap_replicates.eq(2_000).all(),
        "all prospective n, policy, scheduler, intervention, exact-class, and necessity strata retained",
        checks,
    )
    check(
        int(terminal_summary.runs.sum()) == len(runs)
        and set(terminal_summary.evidence_tier) == set(runs.evidence_tier),
        "terminal and recurrence strata account for every run",
        checks,
    )
    successful = summary.endpoint.eq("successful_pair_full_cost_difference")
    check(summary.loc[successful, "estimand_population"].eq("paired_runs_both_successful_only").all() and not summary.shorter_failed_run_counted_as_efficient.any(), "cost efficiency is survivor-only and never rewards shorter failure", checks)
    check(set(cross.source_step) == {"S11", "S12", "S13"} and not cross.pooled_estimate.any() and cross.support_boundary.nunique() == 1, "opposing barrier effects remain separate and unpooled", checks)
    check(result["supportiveCriterionMet"] and result["outcomeClassification"] == "supportive" and result["leaveOneOutGateContrasts"] == ["ablate_recent"], "frozen supportive criterion evaluated exactly", checks)
    repository_xml = ET.parse(OUTPUT / "repository_tests.junit.xml").getroot().find("testsuite")
    mounted_xml = ET.parse(OUTPUT / "mounted_reference_tests.junit.xml").getroot().find("testsuite")
    check(
        repository_xml is not None
        and int(repository_xml.attrib["tests"]) == 199
        and int(repository_xml.attrib["failures"]) == 0
        and int(repository_xml.attrib["errors"]) == 0,
        "199 repository regressions including eight state/reset/sensing fixtures pass",
        checks,
    )
    check(
        mounted_xml is not None
        and int(mounted_xml.attrib["tests"]) == 10
        and int(mounted_xml.attrib["failures"]) == 0
        and int(mounted_xml.attrib["errors"]) == 0,
        "10 mounted E01 reference regressions pass",
        checks,
    )
    check(immutability["success"] and all(item["unchanged"] for item in immutability["inputs"]), "execution inputs byte-immutable", checks)
    check(not Path("/artifacts/research_steps/S14").exists(), "S14 not started", checks)

    failures = [item["check"] for item in checks if not item["success"]]
    status = {
        "schemaVersion": "e03.s13.validation.v1", "researchStepId": "S13",
        "stepNumber": 13, "success": not failures,
        "status": "passed" if not failures else "failed",
        "checksPassed": len(checks) - len(failures), "checksTotal": len(checks),
        "checks": checks, "failures": failures,
        "validationResult": f"{'PASS' if not failures else 'FAIL'}: {len(checks) - len(failures)}/{len(checks)} corpus, semantic, trace, pairing, replay, zero-event, evidence-boundary, and provenance gates",
        "caveatsOrBlockers": [] if not failures else failures,
        "recommendedNextAction": "Write the canonical S13 report and return control without starting S14." if not failures else "Stop S13 and report validation failures without narrowing the frozen design.",
    }
    (OUTPUT / "validation_results.json").write_bytes(canonical_json_bytes(status) + b"\n")
    gateway = {
        "researchStepId": "S13", "success": not (called & forbidden_calls) and not (parameters & forbidden_parameters),
        "proposalParameters": sorted(parameters), "calledFunctions": sorted(called),
        "forbiddenCallsObserved": sorted(called & forbidden_calls),
        "forbiddenParametersObserved": sorted(parameters & forbidden_parameters),
        "interpretation": "The full occupancy representation is used to locate the actor and read only its declared native/radius-two neighborhood; it is not an analysis outcome or global-rank input.",
    }
    (OUTPUT / "information_gateway_audit.json").write_bytes(canonical_json_bytes(gateway) + b"\n")
    print(json.dumps(status, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
