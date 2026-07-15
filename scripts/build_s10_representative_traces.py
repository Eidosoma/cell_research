#!/usr/bin/env python3
"""Select, replay, and plot representative S10 paired traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.detours.action_suppression import MetricWorseningFilter
from src.detours.barrier_interventions import METRICS, execute_intervention_arm
from src.detours.state_space import FamilySpec


OUTPUT = Path("/artifacts/research_steps/S10")
S04 = Path("/artifacts/research_steps/S04")


def select_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    primary = pairs[pairs.filter_threshold == 0].copy()
    selected: list[pd.Series] = []
    for metric in METRICS:
        subset = primary[primary.filter_metric == metric]
        candidates = [
            subset[subset.pair_outcome_classification == "filter_created_exact_impossibility"].sort_values(
                ["suppressed_count", "pair_id"], ascending=[False, True]
            ),
            subset[subset.pair_outcome_classification == "filter_failed_despite_exact_reachability"].sort_values(
                ["suppressed_count", "pair_id"], ascending=[False, True]
            ),
            subset[
                subset.efficiency_comparable
                & (subset.efficiency_delta_activations > 0)
            ].sort_values(["efficiency_delta_activations", "pair_id"], ascending=[False, True]),
            subset[subset.suppressed_count == 0].sort_values("pair_id"),
        ]
        labels = [
            "created_exact_impossibility",
            "reachable_completion_harm",
            "both_complete_cost_harm",
            "no_filter_exposure_control",
        ]
        for label, candidate in zip(labels, candidates):
            if candidate.empty:
                continue
            row = candidate.iloc[0].copy()
            row["representative_category"] = label
            selected.append(row)
    frame = pd.DataFrame(selected)
    if frame.empty:
        raise RuntimeError("hard stop: no representative S10 pairs selected")
    return frame.drop_duplicates("pair_id").reset_index(drop=True)


def main() -> None:
    pairs = pd.read_parquet(OUTPUT / "paired_effects.parquet")
    results = pd.read_parquet(OUTPUT / "action_suppression_results.parquet").set_index(
        "run_id", drop=False
    )
    audits = pd.read_parquet(OUTPUT / "filter_audit.parquet").set_index("run_id")
    inventory = pd.read_parquet(S04 / "state_family_inventory.parquet").set_index(
        "family_ordinal"
    )
    selected = select_pairs(pairs)
    trace_rows: list[dict[str, Any]] = []
    for panel_index, pair in selected.iterrows():
        filtered_expected = results.loc[pair.filtered_run_id]
        control_expected = results.loc[pair.control_run_id]
        family = FamilySpec.from_canonical_dict(
            json.loads(
                inventory.loc[
                    int(filtered_expected.arm_family_ordinal), "canonical_family_json"
                ]
            )
        )
        target = filtered_expected.target_cell_id
        target_index = None if pd.isna(target) else int(str(target)[1:])
        common = dict(
            source_family_ordinal=int(filtered_expected.source_family_ordinal),
            arm_family_ordinal=int(filtered_expected.arm_family_ordinal),
            intervention_type=str(filtered_expected.intervention_type),
            arm_variant=str(filtered_expected.arm_variant),
            focal_index=int(str(filtered_expected.focal_barrier_id)[1:]),
            target_index=target_index,
            replicate_index=int(filtered_expected.replicate_index),
            coupling_key=str(filtered_expected.coupling_key),
            coupling_seed=int(filtered_expected.coupling_seed),
            seed_search_attempts=int(filtered_expected.seed_search_attempts),
            max_activations=2_048,
            trace_proposal_details=True,
        )
        control_row, control_trace = execute_intervention_arm(
            family, int(filtered_expected.source_state_ordinal), **common
        )
        filter_ = MetricWorseningFilter(
            str(filtered_expected.filter_metric),
            int(filtered_expected.filter_threshold),
        )
        filtered_row, filtered_trace = execute_intervention_arm(
            family,
            int(filtered_expected.source_state_ordinal),
            proposal_validation_filter=filter_,
            **common,
        )
        compare_fields = [
            "scenario_id",
            "stop_reason",
            "completed",
            "event_count",
            "final_state_ordinal",
            "final_dynamic_state_sha256",
            "full_ledger_unit_cost",
        ]
        compare_fields.extend(
            column for column in filtered_expected.index if column.startswith("cost_")
        )
        compare_fields.extend(
            column
            for column in filtered_expected.index
            if column.startswith(("start_", "peak_", "excursion_", "final_", "worsening_events_"))
        )
        expected_audit = audits.loc[pair.filtered_run_id]
        if (
            control_row["event_digest_sha256"] != control_expected.event_digest_sha256
            or not all(
                filtered_row[field] == filtered_expected[field]
                for field in compare_fields
            )
            or filter_.audit_row()["suppressed_count"]
            != expected_audit.suppressed_count
        ):
            raise RuntimeError(f"hard stop: representative replay mismatch {pair.pair_id}")
        for condition, run_id, trace, run_row in (
            ("unfiltered_control", pair.control_run_id, control_trace, control_row),
            ("metric_filtered", pair.filtered_run_id, filtered_trace, filtered_row),
        ):
            initial = {
                name: int(run_row[f"start_{name}"]) for name in METRICS
            }
            trace_rows.append(
                {
                    "panel_index": panel_index,
                    "pair_id": pair.pair_id,
                    "representative_category": pair.representative_category,
                    "condition": condition,
                    "run_id": run_id,
                    "filter_metric": pair.filter_metric,
                    "event_index": -1,
                    "decision": "initial",
                    "terminal": None,
                    **{f"level_{name}": initial[name] for name in METRICS},
                }
            )
            for event in trace:
                trace_rows.append(
                    {
                        "panel_index": panel_index,
                        "pair_id": pair.pair_id,
                        "representative_category": pair.representative_category,
                        "condition": condition,
                        "run_id": run_id,
                        "filter_metric": pair.filter_metric,
                        "event_index": int(event["event_index"]),
                        "decision": event["decision"],
                        "terminal": event["terminal"],
                        **{
                            f"level_{name}": int(event["after_levels"][name])
                            for name in METRICS
                        },
                    }
                )
    traces = pd.DataFrame(trace_rows)
    table = pa.Table.from_pandas(traces, preserve_index=False).replace_schema_metadata(
        {
            b"schemaVersion": b"e03.s10.representative_traces.v1",
            b"researchStepId": b"S10",
        }
    )
    pq.write_table(
        table,
        OUTPUT / "representative_traces.parquet",
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        version="2.6",
    )

    panels = selected.reset_index(drop=True)
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3.6 * rows), squeeze=False)
    for panel_index, pair in panels.iterrows():
        axis = axes.flat[panel_index]
        subset = traces[traces.pair_id == pair.pair_id]
        metric = pair.filter_metric
        for condition, style in (
            ("unfiltered_control", "-"),
            ("metric_filtered", "--"),
        ):
            curve = subset[subset.condition == condition]
            axis.plot(
                curve.event_index + 1,
                curve[f"level_{metric}"],
                style,
                linewidth=1.3,
                label=condition.replace("_", " "),
            )
        axis.set_title(
            f"{metric}\n{pair.representative_category}", fontsize=9
        )
        axis.set_xlabel("charged opportunity")
        axis.set_ylabel("exact distance")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    for axis in axes.flat[len(panels) :]:
        axis.axis("off")
    figure.suptitle(
        "S10 metric-filtered versus identical unfiltered opportunity streams",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(OUTPUT / "suppression_effects.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"panels": len(panels), "traceRows": len(traces)}, sort_keys=True))


if __name__ == "__main__":
    main()
