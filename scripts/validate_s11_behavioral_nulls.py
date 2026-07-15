#!/usr/bin/env python3
"""Independent coverage, calibration, replay, and handoff checks for E03 S11."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.stats import chisquare

from src.detours.barrier_interventions import METRICS
from src.detours.behavioral_nulls import NULL_NAMES, bounded, counter_draw, stream_root
from scripts.build_s11_behavioral_nulls import (
    COST_COLUMNS,
    DELTA_COLUMNS,
    EDGE_COST_COLUMNS,
    OUTPUT,
    CACHE,
    S05,
    bootstrap_summary,
    graph_arrays,
    input_paths,
    load_controls,
    result_rows_for_family,
    sha256_file,
    write_json,
    write_parquet,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_graphs(family_ordinals: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edge_dataset = ds.dataset(
        sorted((S05 / "graphs").glob("edge_shard_*.parquet")), format="parquet"
    )
    node_dataset = ds.dataset(
        sorted((S05 / "graphs").glob("node_shard_*.parquet")), format="parquet"
    )
    edges = edge_dataset.to_table(
        columns=[
            "family_ordinal",
            "source_state_ordinal",
            "successor_state_ordinal",
            "opportunity_ordinal",
            "probability_numerator",
            "probability_denominator",
            "proposal_kind_code",
            "decision_code",
            "changed",
            *EDGE_COST_COLUMNS,
            *DELTA_COLUMNS,
        ],
        filter=ds.field("family_ordinal").isin(family_ordinals),
    ).to_pandas()
    nodes = node_dataset.to_table(
        columns=["family_ordinal", "state_ordinal", "terminal_code"],
        filter=ds.field("family_ordinal").isin(family_ordinals),
    ).to_pandas()
    manifest = pd.read_parquet(S05 / "graph_family_manifest.parquet").set_index(
        "family_ordinal"
    )
    return edges, nodes, manifest


def deterministic_replay(
    controls: pd.DataFrame,
    nulls: pd.DataFrame,
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    fields = [
        "stop_reason",
        "completed",
        "event_count",
        "final_state_ordinal",
        "trajectory_fingerprint_u64",
        "raw_stream_fingerprint_u64",
        "checkpoint_2048_state_ordinal",
        "checkpoint_2048_fingerprint_u64",
        "selected_swap_count",
        "selected_memory_count",
        "selected_unchanged_count",
        "requested_swap_count",
        "requested_memory_count",
        "requested_unchanged_count",
        "infeasible_swap_request_count",
        "infeasible_memory_request_count",
        "infeasible_unchanged_request_count",
        "s06_full_ledger_cost",
        "e01_proposal_inclusive_full_ledger_cost",
        *[f"cost_{name}" for name in COST_COLUMNS],
        *[
            f"{prefix}_{metric}"
            for metric in METRICS
            for prefix in ("start", "peak", "excursion", "final", "worsening_events")
        ],
    ]
    expected = nulls.set_index("null_run_id")
    rows: list[dict[str, Any]] = []
    family_ordinals = sorted(int(value) for value in controls.arm_family_ordinal.unique())
    for family_index, family_ordinal in enumerate(family_ordinals):
        arrays = graph_arrays(
            family_ordinal,
            edges[edges.family_ordinal == family_ordinal],
            nodes[nodes.family_ordinal == family_ordinal],
            int(manifest.loc[family_ordinal, "edge_count"]),
        )
        actual_rows = result_rows_for_family(
            controls[controls.arm_family_ordinal == family_ordinal], arrays
        )
        for actual in actual_rows:
            expected_row = expected.loc[actual["null_run_id"]]
            mismatches = []
            for field in fields:
                left = actual[field]
                right = expected_row[field]
                if pd.isna(left) and pd.isna(right):
                    continue
                if isinstance(left, float) or isinstance(right, float):
                    matched = bool(np.isclose(left, right, rtol=0.0, atol=0.0))
                else:
                    matched = left == right
                if not matched:
                    mismatches.append(field)
            rows.append(
                {
                    "schema_version": "e03.s11.replay_validation.v1",
                    "research_step_id": "S11",
                    "null_run_id": actual["null_run_id"],
                    "s09_anchor_run_id": actual["s09_anchor_run_id"],
                    "arm_family_ordinal": family_ordinal,
                    "null_family": actual["null_family"],
                    "fields_checked": len(fields),
                    "mismatch_count": len(mismatches),
                    "mismatched_fields": "+".join(mismatches),
                    "replay_matched": len(mismatches) == 0,
                }
            )
        if (family_index + 1) % 75 == 0:
            print(
                json.dumps(
                    {
                        "replayFamiliesCompleted": family_index + 1,
                        "replayFamiliesTotal": len(family_ordinals),
                        "replayRows": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return pd.DataFrame(rows)


def calibration_diagnostics(graph_audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    draws_per_test = 200_000
    root = stream_root("calibration", "random_legal")
    for bound_value in sorted(int(value) for value in graph_audit.opportunity_count_per_state.unique()):
        counts = np.zeros(bound_value, dtype=np.int64)
        for event in range(draws_per_test):
            index = int(
                bounded(np.uint64(counter_draw(root, event, 1)), bound_value)
            )
            counts[index] += 1
        statistic, p_value = chisquare(counts)
        if bound_value == 1:
            p_value = 1.0
        expected = draws_per_test / bound_value
        maximum_z = float(np.max(np.abs(counts - expected) / np.sqrt(expected)))
        rows.append(
            {
                "schema_version": "e03.s11.null_calibration_diagnostic.v1",
                "research_step_id": "S11",
                "diagnostic": "random_legal_uniform_edge_mapping",
                "opportunity_count": bound_value,
                "draw_count": draws_per_test,
                "chi_square": float(statistic),
                "p_value": float(p_value),
                "maximum_standardized_deviation": maximum_z,
                "tolerance": "singleton exact or p>1e-6 and max_z<6",
                "passed": bool(p_value > 1e-6 and maximum_z < 6.0),
            }
        )
    # Directly calibrate every distinct integer opportunity-weight shape present
    # in S11.  In S05 these are traditional [1] or cell-view 2n-unit profiles.
    for n in (4, 5):
        for bubble_count in range(n + 1):
            weights = np.asarray(
                [value for _ in range(bubble_count) for value in (1, 1)]
                + [2] * (n - bubble_count),
                dtype=np.int16,
            )
            if not len(weights):
                continue
            counts = np.zeros(len(weights), dtype=np.int64)
            open_root = stream_root(
                f"calibration-open-{n}-{bubble_count}", "open_loop_opportunity"
            )
            total = int(weights.sum())
            cumulative = np.cumsum(weights)
            for event in range(draws_per_test):
                wanted = int(
                    bounded(np.uint64(counter_draw(open_root, event, 1)), total)
                )
                counts[int(np.searchsorted(cumulative, wanted, side="right"))] += 1
            expected = draws_per_test * weights.astype(float) / total
            statistic, p_value = chisquare(counts, expected)
            maximum_z = float(np.max(np.abs(counts - expected) / np.sqrt(expected)))
            rows.append(
                {
                    "schema_version": "e03.s11.null_calibration_diagnostic.v1",
                    "research_step_id": "S11",
                    "diagnostic": "open_loop_declared_opportunity_measure",
                    "opportunity_count": len(weights),
                    "draw_count": draws_per_test,
                    "chi_square": float(statistic),
                    "p_value": float(p_value),
                    "maximum_standardized_deviation": maximum_z,
                    "tolerance": "p>1e-6 and max_z<6",
                    "passed": bool(p_value > 1e-6 and maximum_z < 6.0),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    nulls = pd.read_parquet(OUTPUT / "null_results.parquet")
    metric = pd.read_parquet(OUTPUT / "null_metric_results.parquet")
    matching = pd.read_parquet(OUTPUT / "action_rate_matching.parquet")
    comparisons = pd.read_parquet(OUTPUT / "observed_null_comparisons.parquet")
    barrier = pd.read_parquet(OUTPUT / "null_barrier_effects.parquet")
    distributions = pd.read_parquet(OUTPUT / "null_distributions.parquet")
    bootstrap = pd.read_parquet(OUTPUT / "null_calibration_bootstrap.parquet")
    graph_audit = pd.read_parquet(OUTPUT / "graph_coverage_diagnostics.parquet")
    sensitivity = pd.read_parquet(OUTPUT / "event_budget_sensitivity.parquet")
    controls = load_controls()

    gates: list[dict[str, Any]] = []

    def gate(name: str, passed: bool, detail: str) -> None:
        gates.append({"name": name, "passed": bool(passed), "detail": detail})
        require(bool(passed), f"validation gate failed: {name}: {detail}")

    gate(
        "complete_corpus_coverage",
        len(nulls) == 176_568
        and nulls.null_run_id.nunique() == 176_568
        and set(nulls.null_family) == set(NULL_NAMES)
        and len(metric) == 706_272
        and len(barrier) == 619_024,
        f"nulls={len(nulls)}, metric={len(metric)}, barrier={len(barrier)}",
    )
    gate(
        "complete_declared_strata",
        not metric[
            [
                "s09_exact_start_status",
                "s09_observed_status",
                "s10_strict_filter_exposed",
                "s10_strict_filtered_exact_status",
                "intervention_type",
                "opportunity_rate_bin",
            ]
        ].isna().any().any(),
        "all run-metric rows joined to S09/S10 exposure, status, intervention, and rate strata",
    )
    rate_partition = (
        matching.requested_swap_count
        + matching.requested_memory_count
        + matching.requested_unchanged_count
        == matching.event_count
    )
    selected_partition = (
        matching.selected_swap_count
        + matching.selected_memory_count
        + matching.selected_unchanged_count
        == matching.event_count
    )
    denominator = matching.target_rate_denominator.replace(0, np.nan)
    requested_changed = matching.requested_swap_count + matching.requested_memory_count
    target_changed = matching.event_count * (
        matching.target_swap_count + matching.target_memory_count
    ) / denominator
    swap_target = matching.event_count * matching.target_swap_count / denominator
    memory_target = matching.event_count * matching.target_memory_count / denominator
    quota_valid = (
        (requested_changed - target_changed).abs().fillna(0).lt(1.0 + 1e-12).all()
        and (matching.requested_swap_count - swap_target).abs().fillna(0).lt(2.0 + 1e-12).all()
        and (matching.requested_memory_count - memory_target).abs().fillna(0).lt(2.0 + 1e-12).all()
    )
    no_deficit = (
        matching.infeasible_swap_request_count
        + matching.infeasible_memory_request_count
        + matching.infeasible_unchanged_request_count
        == 0
    )
    no_deficit_exact = (
        matching.loc[no_deficit, "requested_swap_count"].to_numpy()
        == matching.loc[no_deficit, "selected_swap_count"].to_numpy()
    ).all() and (
        matching.loc[no_deficit, "requested_memory_count"].to_numpy()
        == matching.loc[no_deficit, "selected_memory_count"].to_numpy()
    ).all() and (
        matching.loc[no_deficit, "requested_unchanged_count"].to_numpy()
        == matching.loc[no_deficit, "selected_unchanged_count"].to_numpy()
    ).all()
    gate(
        "action_rate_matching",
        rate_partition.all()
        and selected_partition.all()
        and quota_valid
        and no_deficit_exact,
        f"rows={len(matching)}, no-deficit exact rows={int(no_deficit.sum())}, deficit rows={int((~no_deficit).sum())}",
    )
    gate(
        "feasible_action_handling",
        bool(nulls.edge_compatibility_valid.all())
        and bool(nulls.terminal_agreement_valid.all())
        and bool(
            (
                nulls.selected_swap_count
                + nulls.selected_memory_count
                + nulls.selected_unchanged_count
                == nulls.event_count
            ).all()
        ),
        f"authoritative selected events={int(nulls.event_count.sum())}",
    )
    root_counts = nulls.groupby(["pair_block_id", "null_family"]).stream_root_hex.nunique()
    family_root_counts = nulls.groupby("pair_block_id").stream_root_hex.nunique()
    gate(
        "independent_and_common_raw_streams",
        bool(root_counts.eq(1).all()) and bool(family_root_counts.eq(len(NULL_NAMES)).all()),
        f"paired blocks={nulls.pair_block_id.nunique()}, null-specific roots per block={len(NULL_NAMES)}",
    )
    calibration = calibration_diagnostics(graph_audit)
    write_parquet(
        OUTPUT / "null_calibration_diagnostics.parquet",
        calibration,
        "e03.s11.null_calibration_diagnostic.v1",
    )
    gate(
        "null_generator_calibration",
        bool(calibration.passed.all()),
        f"passed={int(calibration.passed.sum())}/{len(calibration)} deterministic frequency checks",
    )
    disjoint = (
        (~(metric.exact_impossibility & metric.completion_censored)).all()
        and (~(metric.quiescent_start & metric.reached_quiescence)).all()
        and metric.loc[metric.quiescent_start, "stopped_quiescent"].all()
        and metric.loc[metric.quiescent_start, "event_count"].eq(0).all()
        and metric.loc[~metric.completed, "activation_optimality_gap"].isna().all()
        and comparisons.loc[
            ~comparisons.successful_efficiency_comparable,
            "delta_activations_when_both_successful",
        ].isna().all()
        and barrier.loc[
            ~barrier.successful_efficiency_comparable,
            "delta_activations_when_both_successful",
        ].isna().all()
    )
    gate(
        "outcome_and_efficiency_separation",
        bool(disjoint),
        "impossibility, quiescence, reachable censoring, completion, residual error, and successful-only efficiency are disjoint",
    )
    executed = sensitivity.sensitivity_action.eq("extended_exact_reachable")
    gate(
        "event_budget_sensitivity",
        len(sensitivity) == 7_827
        and int(executed.sum()) == 2_731
        and int((~executed).sum()) == 5_096
        and bool(sensitivity.loc[executed, "prefix_valid"].all()),
        f"extended={int(executed.sum())}, valid prefixes={int(sensitivity.loc[executed, 'prefix_valid'].sum())}",
    )
    recomputed_bootstrap = bootstrap_summary(comparisons)
    bootstrap_columns = [
        "null_family",
        "metric",
        "estimand",
        "estimate",
        "ci_lower_95",
        "ci_upper_95",
        "bootstrap_seed",
    ]
    bootstrap_match = recomputed_bootstrap[bootstrap_columns].equals(
        bootstrap[bootstrap_columns]
    )
    gate(
        "bootstrap_reproducibility",
        bootstrap_match,
        f"rows={len(bootstrap)}, draws per row=10000",
    )
    family_ordinals = sorted(int(value) for value in controls.arm_family_ordinal.unique())
    edges, nodes, manifest = load_graphs(family_ordinals)
    replay = deterministic_replay(controls, nulls, edges, nodes, manifest)
    write_parquet(
        OUTPUT / "replay_validation.parquet",
        replay,
        "e03.s11.replay_validation.v1",
    )
    gate(
        "deterministic_full_replay",
        len(replay) == len(nulls) and bool(replay.replay_matched.all()),
        f"matched={int(replay.replay_matched.sum())}/{len(replay)}",
    )
    before = json.loads((CACHE / "input_hashes_before.json").read_text())["inputs"]
    after = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in input_paths()
    ]
    immutability = {
        "schemaVersion": "e03.s11.input_immutability.v1",
        "researchStepId": "S11",
        "success": before == after,
        "inputCount": len(after),
        "inputs": after,
    }
    write_json(OUTPUT / "input_immutability.json", immutability)
    gate(
        "input_immutability",
        immutability["success"],
        f"unchanged inputs={len(after)}",
    )
    gate(
        "scope_boundary",
        not Path("/artifacts/research_steps/S12").exists(),
        "S12 artifact directory absent",
    )
    validation = {
        "schemaVersion": "e03.s11.validation_results.v1",
        "researchStepId": "S11",
        "success": all(item["passed"] for item in gates),
        "gateCount": len(gates),
        "gatesPassed": sum(item["passed"] for item in gates),
        "gates": gates,
        "deterministicReplays": len(replay),
        "replayFieldsPerRun": int(replay.fields_checked.iloc[0]),
        "nullCalibrationChecks": len(calibration),
        "bootstrapRowsRecomputed": len(recomputed_bootstrap),
    }
    write_json(OUTPUT / "validation_results.json", validation)
    print(json.dumps(validation, sort_keys=True))


if __name__ == "__main__":
    main()
