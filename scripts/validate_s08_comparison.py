#!/usr/bin/env python3
"""Independent physical-artifact validation for E03 S08."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("/artifacts/research_steps/S08")
S02 = Path("/artifacts/research_steps/S02")
S05 = Path("/artifacts/research_steps/S05")
S07 = Path("/artifacts/research_steps/S07")
METRIC_CODES = {
    "adjacent_descents": 0,
    "inversion_count": 1,
    "spearman_footrule": 2,
    "maximum_rank_error": 3,
}
CLASS_LABELS = {
    0: "complete_start",
    1: "reachable_no_detour",
    2: "necessary_detour",
    3: "unreachable_active",
    4: "quiescent",
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def independent_behavior(solver: str, success: bool, excursion: int) -> str:
    if solver == "complete_start":
        return "complete_start" if success else "contradiction"
    if solver == "quiescent":
        return "quiescent_terminal" if not success else "contradiction"
    if success:
        if solver == "unreachable_active":
            return "contradiction"
        if solver == "necessary_detour":
            return "successful_necessary_detour"
        return "successful_no_detour" if excursion == 0 else "successful_unnecessary_detour"
    if solver == "necessary_detour":
        return "failed_where_detour_required"
    if solver == "reachable_no_detour":
        return "failed_despite_no_required_detour"
    if solver == "unreachable_active":
        return "failed_structurally_unreachable"
    return "contradiction"


def main() -> None:
    required = [
        "trajectory_coverage.parquet",
        "observed_structural_events.parquet",
        "accepted_action_projections.parquet",
        "behavior_necessity_comparison.parquet",
        "efficiency_gaps.parquet",
        "witness_panels.parquet",
        "witness_panels.png",
        "classification_summary.parquet",
        "trajectory_start_summary.parquet",
        "research_step_full_results.md",
        "validation_results.json",
        "artifact_manifest.json",
        "provenance_manifest.json",
        "environment.json",
        "input_immutability.json",
        "result_summary.json",
    ]
    gates: dict[str, bool] = {}
    gates["required_outputs_present_nonempty"] = all(
        (ROOT / name).is_file() and (ROOT / name).stat().st_size > 0 for name in required
    )
    report = (ROOT / "research_step_full_results.md").read_text()
    gates["markdown_handoff_contract"] = all(
        phrase in report
        for phrase in (
            "S08 — Compare observed behavior with necessity",
            "Completion status",
            "Artifacts written",
            "Validation result",
            "Outcome classification",
            "Caveats or blockers",
            "Recommended next action",
            "### Lay summary",
            "## Inputs",
            "## Methods",
            "## Results",
            "## Commands",
            "## Validation",
            "## Provenance",
        )
    )
    coverage = pq.read_table(ROOT / "trajectory_coverage.parquet").to_pandas()
    events = pq.read_table(ROOT / "observed_structural_events.parquet").to_pandas()
    accepted = pq.read_table(ROOT / "accepted_action_projections.parquet").to_pandas()
    comparison = pq.read_table(ROOT / "behavior_necessity_comparison.parquet").to_pandas()
    efficiency = pq.read_table(ROOT / "efficiency_gaps.parquet").to_pandas()
    panels = pq.read_table(ROOT / "witness_panels.parquet").to_pandas()

    gates["coverage_accounting"] = (
        len(coverage) == 91
        and int(coverage.primary_denominator.sum()) == 90
        and int(coverage.supplemental.sum()) == 1
        and int(coverage.exact_mapping_eligible.sum()) == 5
        and int(coverage.missing_ordered_state.sum()) == 6
    )
    gates["coverage_boundaries_explicit"] = (
        int(coverage.large_n_boundary.sum()) == 83
        and int(coverage.mixed_goal_boundary.sum()) == 12
        and int(coverage.selection_memory_boundary.sum()) == 2
        and bool(coverage.scheduler_projection_boundary.all())
    )
    gates["event_and_accepted_accounting"] = len(events) == 59 and len(accepted) == 17
    chain_ok = True
    for _, trace in events.sort_values(["logical_trace_id", "event_index"]).groupby("logical_trace_id"):
        chain_ok &= list(trace.event_index) == list(range(len(trace)))
        chain_ok &= all(
            left == right
            for left, right in zip(trace.native_post_hash.iloc[:-1], trace.native_pre_hash.iloc[1:])
        )
        chain_ok &= all(
            left == right
            for left, right in zip(trace.successor_state_ordinal.iloc[:-1], trace.source_state_ordinal.iloc[1:])
        )
    gates["native_and_structural_chains"] = chain_ok

    accepted_keys = set(
        zip(accepted.logical_trace_id, accepted.event_index, accepted.opportunity_ordinal)
    )
    event_accepted_keys = set(
        zip(
            events.loc[events.cost_accepted_swaps == 1, "logical_trace_id"],
            events.loc[events.cost_accepted_swaps == 1, "event_index"],
            events.loc[events.cost_accepted_swaps == 1, "opportunity_ordinal"],
        )
    )
    gates["accepted_subset_exact"] = accepted_keys == event_accepted_keys

    graph_ok = True
    graph_cache: dict[int, pd.DataFrame] = {}
    node_cache: dict[int, pd.DataFrame] = {}
    for family in sorted(events.family_ordinal.unique()):
        family = int(family)
        shard = family % 8
        graph_cache[family] = pq.read_table(
            S05 / f"graphs/edge_shard_{shard:02d}.parquet",
            filters=[("family_ordinal", "=", family)],
        ).to_pandas().reset_index(drop=True)
        node_cache[family] = pq.read_table(
            S05 / f"graphs/node_shard_{shard:02d}.parquet",
            filters=[("family_ordinal", "=", family)],
        ).to_pandas().sort_values("state_ordinal").reset_index(drop=True)
    for row in events.itertuples(index=False):
        graph = graph_cache[int(row.family_ordinal)]
        match = graph[
            (graph.source_state_ordinal == row.source_state_ordinal)
            & (graph.opportunity_ordinal == row.opportunity_ordinal)
        ]
        if len(match) != 1:
            graph_ok = False
            break
        edge = match.iloc[0]
        for column in (
            "successor_state_ordinal", "scheduler_actor_index", "scheduler_side_code",
            "observation_reads", "value_comparisons", "cost_no_ops", "cost_rejections",
            "cost_memory_updates", "cost_accepted_swaps", "cost_displaced_cells",
            "delta_adjacent_descents", "delta_inversion_count", "delta_spearman_footrule",
            "delta_maximum_rank_error",
        ):
            graph_ok &= int(edge[column]) == int(getattr(row, column))
    gates["authoritative_s05_edge_joins"] = graph_ok

    families = sorted(int(value) for value in comparison.family_ordinal.unique())
    paths = pq.read_table(
        S07 / "path_solutions.parquet", filters=[("family_ordinal", "in", families)]
    ).to_pandas()
    joined = comparison.merge(
        paths,
        left_on=["family_ordinal", "state_ordinal"],
        right_on=["family_ordinal", "state_ordinal"],
        how="left",
    )
    joined = joined[joined.metric_code == joined.metric.map(METRIC_CODES)]
    gates["exact_s07_join_cardinality"] = len(joined) == len(comparison) == 256
    s07_ok = True
    for row in joined.itertuples(index=False):
        s07_ok &= CLASS_LABELS[int(row.classification_code)] == row.solver_classification
        s07_ok &= int(row.metric_level) == int(row.start_distance)
        if int(row.minimum_peak_y) < 0:
            s07_ok &= pd.isna(row.minimum_peak_x) and pd.isna(row.minimum_excursion_x)
        else:
            s07_ok &= int(row.minimum_peak_y) == int(row.minimum_peak_x)
            s07_ok &= int(row.minimum_excursion_y) == int(row.minimum_excursion_x)
    gates["s07_values_exact"] = s07_ok

    behavior_ok = all(
        independent_behavior(row.solver_classification, bool(row.observed_success), int(row.observed_excursion))
        == row.behavior_classification
        for row in comparison.itertuples(index=False)
    )
    gates["independent_classification_consistency"] = behavior_ok
    counts = Counter(comparison.behavior_classification)
    gates["classification_partition"] = counts == Counter(
        {
            "failed_structurally_unreachable": 192,
            "successful_no_detour": 44,
            "quiescent_terminal": 16,
            "complete_start": 4,
        }
    )

    s02 = pq.read_table(
        S02 / "replayed_distances.parquet",
        filters=[("logical_trace_id", "in", sorted(accepted.logical_trace_id.unique()))],
    ).to_pandas()
    accepted_checkpoints = {
        (row.logical_trace_id, int(row.activation_count), int(row.accepted_swap_index))
        for row in s02.itertuples(index=False)
        if row.checkpoint_kind == "accepted_action" and row.direction == "ascending"
    }
    gates["accepted_actions_join_s02"] = all(
        (row.logical_trace_id, int(row.event_index) + 1, int(row.accepted_swap_index))
        in accepted_checkpoints
        for row in accepted.itertuples(index=False)
    )

    comparable = efficiency[efficiency.comparable]
    gates["efficiency_optimum_order"] = (
        len(efficiency) == 672
        and set(comparable.lexicographic_relation) <= {"equal", "above_optimum"}
        and bool((comparable.activation_gap >= 0).all())
        and not bool(efficiency[~efficiency.comparable].lexicographic_relation.notna().any())
    )

    panel_ok = True
    for (_, kind), path in panels.sort_values("step_index").groupby(["panel_id", "path_kind"]):
        path = path.sort_values("step_index")
        panel_ok &= list(path.step_index) == list(range(len(path)))
        family = int(
            comparison.loc[comparison.logical_trace_id == path.logical_trace_id.iloc[0], "family_ordinal"].iloc[0]
        )
        graph = graph_cache[family]
        for current, nxt in zip(path.itertuples(index=False), path.iloc[1:].itertuples(index=False)):
            edge = graph.iloc[int(current.edge_local_index)]
            panel_ok &= int(edge.source_state_ordinal) == int(current.state_ordinal)
            panel_ok &= int(edge.successor_state_ordinal) == int(nxt.state_ordinal)
        if kind == "s07_primary_optimum":
            nodes = node_cache[family]
            panel_ok &= int(nodes.iloc[int(path.state_ordinal.iloc[-1])].terminal_code) == 1
            initial = comparison[
                (comparison.logical_trace_id == path.logical_trace_id.iloc[0])
                & (comparison.metric == path.metric.iloc[0])
                & (comparison.checkpoint_activation == 0)
            ].iloc[0]
            panel_ok &= int(path.metric_level.max()) == int(initial.minimum_peak)
    gates["witness_panel_edge_replay"] = panel_ok
    immutable = json.loads((ROOT / "input_immutability.json").read_text())
    gates["input_immutability"] = bool(immutable["success"]) and immutable["preRunSha256"] == immutable["postRunSha256"]
    gates["s09_not_started"] = not (Path("/artifacts/research_steps/S09").exists())

    success = all(gates.values())
    result: dict[str, Any] = {
        "schemaVersion": "e03.s08.independent_validation.v1",
        "researchStepId": "S08",
        "success": success,
        "gates": gates,
        "gateCount": len(gates),
        "passedGateCount": sum(gates.values()),
        "counts": {
            "coverageRows": len(coverage),
            "eventRows": len(events),
            "acceptedRows": len(accepted),
            "comparisonRows": len(comparison),
            "efficiencyRows": len(efficiency),
            "panelRows": len(panels),
        },
    }
    write_json(ROOT / "independent_validation.json", result)
    (ROOT / "validation.log").write_text(
        "\n".join(f"{'PASS' if passed else 'FAIL'} {name}" for name, passed in gates.items()) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
