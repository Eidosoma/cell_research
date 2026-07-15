#!/usr/bin/env python3
"""Extend reachable replicate-zero S11 nulls to 32,768 opportunities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from src.detours.barrier_interventions import METRICS
from src.detours.behavioral_nulls import NULL_NAMES, simulate_family, stream_root
from scripts.build_s11_behavioral_nulls import (
    COST_COLUMNS,
    DELTA_COLUMNS,
    EDGE_COST_COLUMNS,
    OUTPUT,
    S04,
    S05,
    STOP_LABELS,
    graph_arrays,
    load_controls,
    write_json,
    write_parquet,
)


EXTENDED_BUDGET = 32_768


def main() -> None:
    nulls = pd.read_parquet(OUTPUT / "null_results.parquet")
    metric = pd.read_parquet(
        OUTPUT / "null_metric_results.parquet",
        filters=[("metric", "=", "adjacent_descents")],
        columns=["null_run_id", "s09_exact_start_status"],
    )
    primary = nulls[
        (nulls.replicate_index == 0) & nulls.stop_reason.eq("event_budget")
    ].merge(metric, on="null_run_id", how="left", validate="one_to_one")
    if len(primary) != 7_827 or primary.s09_exact_start_status.isna().any():
        raise RuntimeError(f"hard stop: expected 7,827 primary budget rows, found {len(primary)}")
    extend = primary[primary.s09_exact_start_status == "reachable"].copy()
    if len(extend) != 2_731:
        raise RuntimeError(f"hard stop: expected 2,731 reachable extensions, found {len(extend)}")

    controls = load_controls().set_index("run_id", drop=False)
    anchors = controls.loc[extend.s09_anchor_run_id.unique()].copy()
    family_ordinals = sorted(int(value) for value in anchors.arm_family_ordinal.unique())
    edge_dataset = ds.dataset(
        sorted((S05 / "graphs").glob("edge_shard_*.parquet")), format="parquet"
    )
    node_dataset = ds.dataset(
        sorted((S05 / "graphs").glob("node_shard_*.parquet")), format="parquet"
    )
    edge_columns = [
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
    ]
    edges = edge_dataset.to_table(
        columns=edge_columns,
        filter=ds.field("family_ordinal").isin(family_ordinals),
    ).to_pandas()
    nodes = node_dataset.to_table(
        columns=["family_ordinal", "state_ordinal", "terminal_code"],
        filter=ds.field("family_ordinal").isin(family_ordinals),
    ).to_pandas()
    manifest = pd.read_parquet(S05 / "graph_family_manifest.parquet").set_index(
        "family_ordinal"
    )
    primary_lookup = primary.set_index("null_run_id")
    extend_lookup = {
        (str(row.s09_anchor_run_id), str(row.null_family)): str(row.null_run_id)
        for row in extend.itertuples(index=False)
    }
    long_rows: dict[str, dict[str, Any]] = {}
    for family_index, family_ordinal in enumerate(family_ordinals):
        family_anchors = anchors[anchors.arm_family_ordinal == family_ordinal]
        arrays = graph_arrays(
            family_ordinal,
            edges[edges.family_ordinal == family_ordinal],
            nodes[nodes.family_ordinal == family_ordinal],
            int(manifest.loc[family_ordinal, "edge_count"]),
        )
        roots = np.asarray(
            [
                [stream_root(str(row.pair_block_id), name) for name in NULL_NAMES]
                for row in family_anchors.itertuples(index=False)
            ],
            dtype=np.uint64,
        )
        start_levels = family_anchors[
            [f"start_{name}" for name in METRICS]
        ].to_numpy(dtype=np.int16)
        output = simulate_family(
            family_anchors.arm_state_ordinal.to_numpy(dtype=np.int64),
            roots,
            family_anchors.event_count.to_numpy(dtype=np.int32),
            family_anchors.cost_acceptedSwaps.to_numpy(dtype=np.int32),
            family_anchors.cost_memoryUpdates.to_numpy(dtype=np.int32),
            start_levels,
            int(arrays["slots"]),
            arrays["successors"],
            arrays["terminal"],
            arrays["opportunity_weights"],
            arrays["decision"],
            arrays["changed"],
            arrays["kind"],
            arrays["costs"],
            arrays["deltas"],
            EXTENDED_BUDGET,
        )
        (
            final_states,
            stop_codes,
            event_counts,
            ledgers,
            peaks,
            finals,
            worsening,
            requested,
            deficits,
            selected,
            fingerprints,
            raw_fingerprints,
            checkpoint_states,
            checkpoint_fingerprints,
            checkpoint_ledgers,
        ) = output
        for run_index, anchor in enumerate(family_anchors.itertuples(index=False)):
            for null_code, null_name in enumerate(NULL_NAMES):
                key = (str(anchor.run_id), null_name)
                if key not in extend_lookup:
                    continue
                null_run_id = extend_lookup[key]
                primary_row = primary_lookup.loc[null_run_id]
                events = int(event_counts[run_index, null_code])
                stop_reason = STOP_LABELS[int(stop_codes[run_index, null_code])]
                row: dict[str, Any] = {
                    "schema_version": "e03.s11.event_budget_sensitivity.v1",
                    "research_step_id": "S11",
                    "null_run_id": null_run_id,
                    "s09_anchor_run_id": anchor.run_id,
                    "source_family_ordinal": int(anchor.source_family_ordinal),
                    "source_state_ordinal": int(anchor.source_state_ordinal),
                    "arm_family_ordinal": family_ordinal,
                    "arm_state_ordinal": int(anchor.arm_state_ordinal),
                    "null_family": null_name,
                    "intervention_type": anchor.intervention_type,
                    "arm_variant": anchor.arm_variant,
                    "s09_exact_start_status": "reachable",
                    "sensitivity_action": "extended_exact_reachable",
                    "primary_event_budget": 2_048,
                    "primary_stop_reason": "event_budget",
                    "extended_event_budget": EXTENDED_BUDGET,
                    "extended_stop_reason": stop_reason,
                    "extended_completed": stop_reason == "complete",
                    "extended_reached_quiescence": stop_reason == "quiescent",
                    "extended_still_active": stop_reason == "event_budget",
                    "extended_event_count": events,
                    "extended_final_state_ordinal": int(final_states[run_index, null_code]),
                    "extended_trajectory_fingerprint_u64": f"{int(fingerprints[run_index, null_code]):016x}",
                    "extended_raw_stream_fingerprint_u64": f"{int(raw_fingerprints[run_index, null_code]):016x}",
                    "prefix_state_match": int(checkpoint_states[run_index, null_code])
                    == int(primary_row.final_state_ordinal),
                    "prefix_fingerprint_match": f"{int(checkpoint_fingerprints[run_index, null_code]):016x}"
                    == str(primary_row.trajectory_fingerprint_u64),
                }
                ledger_match = True
                for cost_index, cost_name in enumerate(COST_COLUMNS):
                    extended_value = int(ledgers[run_index, null_code, cost_index])
                    checkpoint_value = int(
                        checkpoint_ledgers[run_index, null_code, cost_index]
                    )
                    row[f"extended_cost_{cost_name}"] = extended_value
                    if checkpoint_value != int(primary_row[f"cost_{cost_name}"]):
                        ledger_match = False
                row["prefix_ledger_match"] = ledger_match
                row["prefix_valid"] = (
                    row["prefix_state_match"]
                    and row["prefix_fingerprint_match"]
                    and ledger_match
                )
                row["extended_cost_activations"] = events
                row["extended_cost_proposals"] = events
                row["extended_s06_full_ledger_cost"] = events + sum(
                    row[f"extended_cost_{name}"] for name in COST_COLUMNS
                )
                for metric_code, metric_name in enumerate(METRICS):
                    start = int(start_levels[run_index, metric_code])
                    peak = int(peaks[run_index, null_code, metric_code])
                    row[f"start_{metric_name}"] = start
                    row[f"extended_peak_{metric_name}"] = peak
                    row[f"extended_excursion_{metric_name}"] = peak - start
                    row[f"extended_final_{metric_name}"] = int(
                        finals[run_index, null_code, metric_code]
                    )
                    row[f"extended_worsening_events_{metric_name}"] = int(
                        worsening[run_index, null_code, metric_code]
                    )
                long_rows[null_run_id] = row
        if (family_index + 1) % 50 == 0:
            print(
                json.dumps(
                    {
                        "familiesCompleted": family_index + 1,
                        "familiesTotal": len(family_ordinals),
                        "extensionsCompleted": len(long_rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if len(long_rows) != len(extend):
        raise RuntimeError(
            f"hard stop: expected {len(extend)} extensions, built {len(long_rows)}"
        )
    rows = list(long_rows.values())
    for row in primary[primary.s09_exact_start_status != "reachable"].itertuples(
        index=False
    ):
        rows.append(
            {
                "schema_version": "e03.s11.event_budget_sensitivity.v1",
                "research_step_id": "S11",
                "null_run_id": row.null_run_id,
                "s09_anchor_run_id": row.s09_anchor_run_id,
                "source_family_ordinal": int(row.source_family_ordinal),
                "source_state_ordinal": int(row.source_state_ordinal),
                "arm_family_ordinal": int(row.arm_family_ordinal),
                "arm_state_ordinal": int(row.arm_state_ordinal),
                "null_family": row.null_family,
                "intervention_type": row.intervention_type,
                "arm_variant": row.arm_variant,
                "s09_exact_start_status": row.s09_exact_start_status,
                "sensitivity_action": "not_extended_exact_impossible",
                "primary_event_budget": 2_048,
                "primary_stop_reason": "event_budget",
                "extended_event_budget": EXTENDED_BUDGET,
                "extended_stop_reason": None,
                "extended_completed": False,
                "extended_reached_quiescence": False,
                "extended_still_active": False,
                "prefix_state_match": None,
                "prefix_fingerprint_match": None,
                "prefix_ledger_match": None,
                "prefix_valid": None,
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["null_family", "source_family_ordinal", "source_state_ordinal", "arm_variant"]
    )
    write_parquet(
        OUTPUT / "event_budget_sensitivity.parquet",
        result,
        "e03.s11.event_budget_sensitivity.v1",
    )
    executed = result.sensitivity_action.eq("extended_exact_reachable")
    summary = {
        "schemaVersion": "e03.s11.event_budget_sensitivity_summary.v1",
        "researchStepId": "S11",
        "primaryReplicateZeroBudgetRuns": len(result),
        "exactReachableExtended": int(executed.sum()),
        "exactImpossibleNotExtended": int((~executed).sum()),
        "extensionsCompleted": int(result.loc[executed, "extended_completed"].sum()),
        "extensionsReachedQuiescence": int(
            result.loc[executed, "extended_reached_quiescence"].sum()
        ),
        "extensionsStillActive": int(
            result.loc[executed, "extended_still_active"].sum()
        ),
        "prefixesValid": int(result.loc[executed, "prefix_valid"].sum()),
    }
    write_json(OUTPUT / "event_budget_sensitivity_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
