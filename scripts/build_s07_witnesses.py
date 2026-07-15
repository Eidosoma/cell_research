#!/usr/bin/env python3
"""Build a deterministic stratified S07 primary/secondary witness panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.detours.necessary_detour import family_metric_levels
from src.detours.path_solutions import (
    PROFILE_ACCEPTED_SWAPS,
    PROFILE_ACTIVATIONS,
    PROFILE_FULL_LEDGER,
    PROFILE_OBSERVATION_READS,
    PROFILE_VALUE_COMPARISONS,
    reconstruct_successor_witness,
    solve_all_starts_lexicographic,
)
from src.detours.state_space import FamilySpec


S04 = Path("/artifacts/research_steps/S04")
S05 = Path("/artifacts/research_steps/S05")
OUTPUT = Path("/artifacts/research_steps/S07")
METRICS = (
    "adjacent_descents",
    "inversion_count",
    "spearman_footrule",
    "maximum_rank_error",
)
METRIC_CODES = {name: index for index, name in enumerate(METRICS)}
COST_COLUMNS = (
    "observation_reads",
    "value_comparisons",
    "cost_no_ops",
    "cost_rejections",
    "cost_memory_updates",
    "cost_accepted_swaps",
    "cost_displaced_cells",
)
PROFILE_CODES = {
    "activations": PROFILE_ACTIVATIONS,
    "observation_reads": PROFILE_OBSERVATION_READS,
    "value_comparisons": PROFILE_VALUE_COMPARISONS,
    "accepted_swaps": PROFILE_ACCEPTED_SWAPS,
    "displaced_cells": PROFILE_ACCEPTED_SWAPS,
    "full_ledger": PROFILE_FULL_LEDGER,
}

WITNESS_SCHEMA = pa.schema(
    [
        pa.field("panel_case_id", pa.int32(), nullable=False),
        pa.field("selection_reasons", pa.string(), nullable=False),
        pa.field("family_ordinal", pa.int32(), nullable=False),
        pa.field("state_ordinal", pa.uint32(), nullable=False),
        pa.field("metric", pa.string(), nullable=False),
        pa.field("metric_code", pa.uint8(), nullable=False),
        pa.field("classification_code", pa.uint8(), nullable=False),
        pa.field("solution_scope", pa.string(), nullable=False),
        pa.field("cost_profile", pa.string(), nullable=False),
        pa.field("start_metric_level", pa.int16(), nullable=False),
        pa.field("primary_minimum_peak", pa.int16(), nullable=False),
        pa.field("primary_minimum_excursion", pa.int16(), nullable=False),
        pa.field("secondary_threshold", pa.int16(), nullable=False),
        pa.field("observed_path_peak", pa.int16(), nullable=False),
        pa.field("goal_state_ordinal", pa.uint32(), nullable=False),
        pa.field("path_edge_count", pa.int32(), nullable=False),
        pa.field("optimal_label_json", pa.string(), nullable=False),
        pa.field("ledger_activations", pa.int32(), nullable=False),
        pa.field("ledger_observation_reads", pa.int32(), nullable=False),
        pa.field("ledger_value_comparisons", pa.int32(), nullable=False),
        pa.field("ledger_no_ops", pa.int32(), nullable=False),
        pa.field("ledger_rejections", pa.int32(), nullable=False),
        pa.field("ledger_memory_updates", pa.int32(), nullable=False),
        pa.field("ledger_accepted_swaps", pa.int32(), nullable=False),
        pa.field("ledger_displaced_cells", pa.int32(), nullable=False),
        pa.field("node_ordinals", pa.list_(pa.uint32()), nullable=False),
        pa.field("edge_local_indices", pa.list_(pa.int32()), nullable=False),
        pa.field("opportunity_ordinals", pa.list_(pa.uint8()), nullable=False),
        pa.field("path_logical_digest", pa.string(), nullable=False),
    ],
    metadata={
        b"schemaVersion": b"e03.s07.witness_paths.v1",
        b"researchStepId": b"S07",
        b"edgeIdentity": b"family_ordinal plus source node from node_ordinals plus opportunity_ordinal",
        b"schedulerQuantification": b"existential finite S05 structural opportunity path",
    },
)


def family_from_row(row: dict[str, Any]) -> FamilySpec:
    return FamilySpec.from_canonical_dict(json.loads(row["canonical_family_json"]))


def choose_cases(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    frame["class_kind"] = np.where(
        frame.necessary_detour_count > 0, "necessary", "no_necessary"
    )
    frame["policy_regime"] = np.where(
        frame.policy_profile.str.startswith("pure_"), frame.policy_profile, "mixed"
    )
    eligible = frame[
        (frame.reachable_active_count > 0)
        & (
            (frame.class_kind == "necessary")
            | (
                (frame.class_kind == "no_necessary")
                & (frame.reachable_no_detour_count > 0)
            )
        )
    ]
    selected: list[pd.DataFrame] = []

    def pick(groups: list[str], reason: str) -> None:
        rows = (
            eligible.sort_values(
                ["necessary_detour_count", "reachable_active_count", "family_ordinal"],
                ascending=[False, False, True],
            )
            .groupby(groups, dropna=False, sort=True)
            .head(1)
            .copy()
        )
        rows["selection_reason"] = reason
        selected.append(rows)

    pick(["metric", "n", "class_kind"], "n_class")
    pick(
        ["metric", "architecture", "direction", "class_kind"],
        "architecture_direction_class",
    )
    pick(["metric", "fault_mode", "class_kind"], "fault_class")
    pick(["metric", "policy_regime", "class_kind"], "policy_class")
    boundary = eligible[eligible.family_ordinal.isin([5557, 7976])].copy()
    boundary["selection_reason"] = "largest_graph_boundary"
    selected.append(boundary)
    deepest = (
        eligible[eligible.class_kind == "necessary"]
        .sort_values(
            ["metric", "maximum_normalized_excursion", "necessary_detour_count", "family_ordinal"],
            ascending=[True, False, False, True],
        )
        .groupby("metric", sort=True)
        .head(3)
        .copy()
    )
    deepest["selection_reason"] = "deepest_excursion"
    selected.append(deepest)
    combined = pd.concat(selected, ignore_index=True)
    combined["state_ordinal"] = np.where(
        combined.class_kind == "necessary",
        combined.deepest_necessary_detour_state,
        combined.first_reachable_no_detour_state,
    ).astype(int)
    keys = ["family_ordinal", "metric", "state_ordinal"]
    reasons = (
        combined.groupby(keys, sort=True).selection_reason.apply(
            lambda values: ";".join(sorted(set(values)))
        )
    ).rename("selection_reasons")
    cases = combined.drop_duplicates(keys).set_index(keys).join(reasons).reset_index()
    cases = cases.sort_values(keys).reset_index(drop=True)
    cases["panel_case_id"] = np.arange(len(cases), dtype=np.int32)
    return cases


def read_family(path: Path, ordinal: int, columns: list[str] | None = None) -> pa.Table:
    return pq.read_table(
        path,
        columns=columns,
        filters=[("family_ordinal", "=", ordinal)],
    ).combine_chunks()


def profile_label(profile: str, ledger: tuple[int, ...]) -> tuple[int, ...]:
    activations, reads, comparisons, no_ops, rejections, memory, swaps, displaced = ledger
    if profile == "activations":
        return (activations,)
    if profile == "observation_reads":
        return (reads, activations)
    if profile == "value_comparisons":
        return (comparisons, activations)
    if profile == "accepted_swaps":
        return (swaps, activations)
    if profile == "displaced_cells":
        return (displaced, activations)
    if profile == "full_ledger":
        return ledger
    raise ValueError(profile)


def path_ledger(edge_indices: tuple[int, ...], costs: dict[str, np.ndarray]) -> tuple[int, ...]:
    if edge_indices:
        selected = np.asarray(edge_indices, dtype=np.int64)
        totals = tuple(int(costs[name][selected].sum()) for name in COST_COLUMNS)
    else:
        totals = (0,) * len(COST_COLUMNS)
    return (len(edge_indices), *totals[:2], *totals[2:])


def witness_digest(
    family_ordinal: int,
    nodes: tuple[int, ...],
    edges: tuple[int, ...],
    opportunities: tuple[int, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray([family_ordinal], dtype=np.int32).tobytes())
    digest.update(np.asarray(nodes, dtype=np.uint32).tobytes())
    digest.update(np.asarray(edges, dtype=np.int32).tobytes())
    digest.update(np.asarray(opportunities, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def add_row(
    rows: list[dict[str, Any]],
    case: pd.Series,
    scope: str,
    profile: str,
    nodes: tuple[int, ...],
    edges: tuple[int, ...],
    opportunities: np.ndarray,
    levels: np.ndarray,
    costs: dict[str, np.ndarray],
    optimal_label: tuple[int, ...],
    threshold: int,
) -> None:
    selected_opportunities = tuple(int(opportunities[edge]) for edge in edges)
    ledger = path_ledger(edges, costs)
    if profile != "none" and profile_label(profile, ledger) != optimal_label:
        raise AssertionError(
            f"case {case.panel_case_id} {scope}/{profile} ledger {ledger} != {optimal_label}"
        )
    observed_peak = int(np.max(levels[np.asarray(nodes, dtype=np.int64)]))
    if scope in {"primary_minimax", "secondary_minimax"} and observed_peak != int(
        case.minimum_peak
    ):
        raise AssertionError("minimax witness peak mismatch")
    rows.append(
        {
            "panel_case_id": int(case.panel_case_id),
            "selection_reasons": case.selection_reasons,
            "family_ordinal": int(case.family_ordinal),
            "state_ordinal": int(case.state_ordinal),
            "metric": case.metric,
            "metric_code": METRIC_CODES[case.metric],
            "classification_code": int(case.classification_code),
            "solution_scope": scope,
            "cost_profile": profile,
            "start_metric_level": int(case.metric_level),
            "primary_minimum_peak": int(case.minimum_peak),
            "primary_minimum_excursion": int(case.minimum_excursion),
            "secondary_threshold": threshold,
            "observed_path_peak": observed_peak,
            "goal_state_ordinal": int(nodes[-1]),
            "path_edge_count": len(edges),
            "optimal_label_json": json.dumps(list(optimal_label), separators=(",", ":")),
            "ledger_activations": ledger[0],
            "ledger_observation_reads": ledger[1],
            "ledger_value_comparisons": ledger[2],
            "ledger_no_ops": ledger[3],
            "ledger_rejections": ledger[4],
            "ledger_memory_updates": ledger[5],
            "ledger_accepted_swaps": ledger[6],
            "ledger_displaced_cells": ledger[7],
            "node_ordinals": list(nodes),
            "edge_local_indices": list(edges),
            "opportunity_ordinals": list(selected_opportunities),
            "path_logical_digest": witness_digest(
                int(case.family_ordinal), nodes, edges, selected_opportunities
            ),
        }
    )


def main() -> None:
    started = time.perf_counter()
    summary = pd.read_parquet(OUTPUT / "necessity_prevalence_by_family.parquet")
    cases = choose_cases(summary)
    family_metadata = (
        pd.read_parquet(S04 / "state_family_inventory.parquet")
        .set_index("family_ordinal")
        .to_dict("index")
    )
    graph_metadata = (
        pd.read_parquet(S05 / "graph_family_manifest.parquet")
        .set_index("family_ordinal")
        .to_dict("index")
    )
    witness_rows: list[dict[str, Any]] = []
    enriched_cases: list[dict[str, Any]] = []
    for position, (ordinal, family_cases) in enumerate(cases.groupby("family_ordinal", sort=True)):
        ordinal = int(ordinal)
        graph = graph_metadata[ordinal]
        family = family_from_row(family_metadata[ordinal])
        edge_table = read_family(
            S05 / graph["edge_shard"],
            ordinal,
            [
                "family_ordinal",
                "source_state_ordinal",
                "successor_state_ordinal",
                "opportunity_ordinal",
                *COST_COLUMNS,
            ],
        )
        node_table = read_family(
            S05 / graph["node_shard"], ordinal, ["family_ordinal", "terminal_code"]
        )
        sources = edge_table["source_state_ordinal"].to_numpy().astype(np.int64, copy=False)
        targets = edge_table["successor_state_ordinal"].to_numpy().astype(np.int64, copy=False)
        opportunities = edge_table["opportunity_ordinal"].to_numpy()
        costs = {name: edge_table[name].to_numpy() for name in COST_COLUMNS}
        terminal = node_table["terminal_code"].to_numpy()
        cost_output = read_family(
            OUTPUT / "minimum_cost_solutions.parquet", ordinal
        ).to_pandas().sort_values("state_ordinal")
        path_output = read_family(
            OUTPUT / "path_solutions.parquet", ordinal
        ).to_pandas()
        for _, case in family_cases.iterrows():
            start = int(case.state_ordinal)
            metric_code = METRIC_CODES[case.metric]
            path_metric = path_output[path_output.metric_code == metric_code].sort_values(
                "state_ordinal"
            )
            if len(path_metric) != len(terminal):
                raise AssertionError("path solution family/metric coverage mismatch")
            state_row = path_metric.iloc[start]
            case = case.copy()
            for name in (
                "metric_level",
                "minimum_peak",
                "minimum_excursion",
                "classification_code",
            ):
                case[name] = int(state_row[name])
            levels = family_metric_levels(family, case.metric)
            primary_successor = path_metric.primary_successor_edge_local_index.to_numpy()
            primary_nodes, primary_edges = reconstruct_successor_witness(
                start, sources, targets, terminal, primary_successor
            )
            add_row(
                witness_rows,
                case,
                "primary_minimax",
                "none",
                primary_nodes,
                primary_edges,
                opportunities,
                levels,
                costs,
                (int(case.minimum_peak),),
                int(case.minimum_peak),
            )

            threshold = int(case.minimum_peak)
            allowed = (levels[sources] <= threshold) & (levels[targets] <= threshold)
            solved_profiles: dict[str, Any] = {}
            for profile, code in PROFILE_CODES.items():
                if profile == "displaced_cells":
                    solved_profiles[profile] = solved_profiles["accepted_swaps"]
                    continue
                solved_profiles[profile] = solve_all_starts_lexicographic(
                    len(terminal),
                    sources,
                    targets,
                    terminal,
                    costs,
                    code,
                    edge_allowed=allowed,
                )
            for profile in PROFILE_CODES:
                solved = solved_profiles[profile]
                secondary_nodes, secondary_edges = reconstruct_successor_witness(
                    start, sources, targets, terminal, solved.successor_edges
                )
                label = tuple(int(value) for value in solved.labels[:, start])
                if profile == "displaced_cells":
                    label = (2 * label[0], label[1])
                add_row(
                    witness_rows,
                    case,
                    "secondary_minimax",
                    profile,
                    secondary_nodes,
                    secondary_edges,
                    opportunities,
                    levels,
                    costs,
                    label,
                    threshold,
                )

            cost_row = cost_output.iloc[start]
            global_profiles = {
                "activations": (
                    (int(cost_row.shortest_activations),),
                    "shortest_successor_edge_local_index",
                ),
                "observation_reads": (
                    (
                        int(cost_row.minimum_observation_reads),
                        int(cost_row.minimum_observation_reads_activations),
                    ),
                    "minimum_observation_reads_successor_edge_local_index",
                ),
                "value_comparisons": (
                    (
                        int(cost_row.minimum_value_comparisons),
                        int(cost_row.minimum_value_comparisons_activations),
                    ),
                    "minimum_value_comparisons_successor_edge_local_index",
                ),
                "accepted_swaps": (
                    (
                        int(cost_row.minimum_accepted_swaps),
                        int(cost_row.minimum_accepted_swaps_activations),
                    ),
                    "minimum_accepted_swaps_successor_edge_local_index",
                ),
                "displaced_cells": (
                    (
                        int(cost_row.minimum_displaced_cells),
                        int(cost_row.minimum_displaced_cells_activations),
                    ),
                    "minimum_displaced_cells_successor_edge_local_index",
                ),
                "full_ledger": (
                    tuple(
                        int(cost_row[f"minimum_full_ledger_{name}"])
                        for name in (
                            "activations",
                            "observation_reads",
                            "value_comparisons",
                            "no_ops",
                            "rejections",
                            "memory_updates",
                            "accepted_swaps",
                            "displaced_cells",
                        )
                    ),
                    "minimum_full_ledger_successor_edge_local_index",
                ),
            }
            for profile, (label, successor_column) in global_profiles.items():
                nodes, edge_indices = reconstruct_successor_witness(
                    start,
                    sources,
                    targets,
                    terminal,
                    cost_output[successor_column].to_numpy(),
                )
                add_row(
                    witness_rows,
                    case,
                    "unconstrained_minimum_cost",
                    profile,
                    nodes,
                    edge_indices,
                    opportunities,
                    levels,
                    costs,
                    label,
                    -1,
                )
            enriched_cases.append(
                {
                    **case.to_dict(),
                    "node_count": len(terminal),
                    "edge_count": len(sources),
                    "witness_row_count": 13,
                }
            )
        print(
            json.dumps(
                {
                    "familyPosition": position,
                    "familyOrdinal": ordinal,
                    "cases": len(family_cases),
                    "witnessRows": len(witness_rows),
                }
            ),
            flush=True,
        )

    table = pa.Table.from_pylist(witness_rows, schema=WITNESS_SCHEMA)
    pq.write_table(
        table,
        OUTPUT / "witness_paths.parquet",
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
    )
    panel_table = pa.Table.from_pandas(pd.DataFrame(enriched_cases), preserve_index=False)
    pq.write_table(
        panel_table,
        OUTPUT / "witness_panel_manifest.parquet",
        compression="zstd",
        compression_level=9,
    )
    result = {
        "schemaVersion": "e03.s07.witness_build.v1",
        "researchStepId": "S07",
        "success": True,
        "panelCases": len(cases),
        "families": int(cases.family_ordinal.nunique()),
        "metrics": sorted(cases.metric.unique()),
        "necessaryCases": int((cases.class_kind == "necessary").sum()),
        "reachableNoDetourCases": int((cases.class_kind == "no_necessary").sum()),
        "witnessRows": len(witness_rows),
        "profilesPerCase": 13,
        "runtimeSeconds": time.perf_counter() - started,
    }
    (OUTPUT / "witness_build.json").write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
