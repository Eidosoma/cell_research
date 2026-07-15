#!/usr/bin/env python3
"""Build the exhaustive E03 S11 structural behavioral-null corpus."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from reference_simulator.model import canonical_json_bytes
from src.detours.barrier_interventions import METRICS
from src.detours.behavioral_nulls import (
    NULL_NAMES,
    behavior_classification,
    exact_start_status,
    opportunity_integer_weights,
    opportunity_rate_bin,
    simulate_family,
    stream_root,
)
from src.detours.necessary_detour import CLASS_LABELS
from src.detours.state_space import FamilySpec


OUTPUT = Path("/artifacts/research_steps/S11")
CACHE = Path("/cache/e03_s11")
S03 = Path("/artifacts/research_steps/S03")
S04 = Path("/artifacts/research_steps/S04")
S05 = Path("/artifacts/research_steps/S05")
S07 = Path("/artifacts/research_steps/S07")
S08 = Path("/artifacts/research_steps/S08")
S09 = Path("/artifacts/research_steps/S09")
S10 = Path("/artifacts/research_steps/S10")
REPOSITORY = Path(__file__).resolve().parents[1]
PRIMARY_BUDGET = 2_048
BOOTSTRAPS = 10_000
METRIC_CODES = {name: index for index, name in enumerate(METRICS)}
STOP_LABELS = {1: "complete", 2: "quiescent", 3: "event_budget"}
COST_COLUMNS = (
    "observationReads",
    "valueComparisons",
    "noOps",
    "rejections",
    "memoryUpdates",
    "acceptedSwaps",
    "displacedCells",
)
EDGE_COST_COLUMNS = (
    "observation_reads",
    "value_comparisons",
    "cost_no_ops",
    "cost_rejections",
    "cost_memory_updates",
    "cost_accepted_swaps",
    "cost_displaced_cells",
)
DELTA_COLUMNS = (
    "delta_adjacent_descents",
    "delta_inversion_count",
    "delta_spearman_footrule",
    "delta_maximum_rank_error",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_id(domain: str, *parts: object) -> str:
    payload = "\x00".join([domain, *(str(part) for part in parts)]).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(
    path: Path,
    value: pd.DataFrame | Sequence[Mapping[str, Any]],
    schema_version: str,
) -> None:
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    if frame.empty:
        raise ValueError(f"refusing to write empty S11 table: {path.name}")
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {b"schemaVersion": schema_version.encode(), b"researchStepId": b"S11"}
    )
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )


def input_paths() -> list[Path]:
    paths = [
        Path("/workspace/AGENTS.md"),
        Path("/workspace/FULL_PLAN.md"),
        Path("/workspace/RESEARCH_PLAN.md"),
        Path("/workspace/PREVIOUS_ARTIFACTS.md"),
        Path("/workspace/PREVIOUS_ARTIFACTS.json"),
        Path("/workspace/input-attachments/MANIFEST.json"),
        S03 / "research_step_full_results.md",
        S03 / "metric_disagreement_atlas.parquet",
        S04 / "research_step_full_results.md",
        S04 / "state_family_inventory.parquet",
        S05 / "research_step_full_results.md",
        S05 / "graph_corpus_manifest.json",
        S05 / "graph_family_manifest.parquet",
        S07 / "research_step_full_results.md",
        S07 / "path_solutions.parquet",
        S07 / "minimum_cost_solutions.parquet",
        S08 / "research_step_full_results.md",
        S09 / "research_step_full_results.md",
        S09 / "barrier_interventions.parquet",
        S09 / "structural_optimum_changes.parquet",
        S09 / "event_budget_sensitivity.parquet",
        S10 / "research_step_full_results.md",
        S10 / "paired_effects.parquet",
        REPOSITORY / "analysis/s11_behavioral_null_contract.json",
        REPOSITORY / "reference_simulator/engine.py",
        REPOSITORY / "reference_simulator/model.py",
        REPOSITORY / "reference_simulator/policies.py",
        REPOSITORY / "reference_simulator/transition_primitives.py",
        REPOSITORY / "src/detours/behavioral_nulls.py",
    ]
    paths.extend(sorted(Path("/workspace/input-attachments").glob("*/_metadata/ATTACHMENT.md")))
    return paths


def load_controls() -> pd.DataFrame:
    controls = pd.read_parquet(S09 / "barrier_interventions.parquet")
    controls = controls[
        controls.intervention_type.isin(["baseline", "remove", "move", "add"])
    ].copy()
    controls = controls.sort_values(
        [
            "arm_family_ordinal",
            "source_family_ordinal",
            "source_state_ordinal",
            "arm_variant",
            "replicate_index",
        ],
        kind="stable",
    ).reset_index(drop=True)
    if len(controls) != 25_224:
        raise RuntimeError(f"hard stop: expected 25,224 S09 anchors, found {len(controls)}")
    contexts = controls[controls.replicate_index == 0][
        ["arm_family_ordinal", "arm_state_ordinal", "arm_variant"]
    ].drop_duplicates()
    if len(contexts) != 3_328:
        raise RuntimeError(f"hard stop: expected 3,328 contexts, found {len(contexts)}")
    if controls.arm_family_ordinal.nunique() != 394:
        raise RuntimeError("hard stop: expected 394 arm families")
    if controls[["source_family_ordinal", "source_state_ordinal"]].drop_duplicates().shape[0] != 407:
        raise RuntimeError("hard stop: expected 407 source starts")

    sensitivity = pd.read_parquet(S09 / "event_budget_sensitivity.parquet")
    budget = controls[controls.stop_reason == "event_budget"]
    joined = budget[["run_id"]].merge(
        sensitivity[["primary_run_id", "extended_stop_reason", "extended_event_count"]],
        left_on="run_id",
        right_on="primary_run_id",
        validate="one_to_one",
    )
    if len(joined) != 3_194 or not (
        joined.extended_stop_reason.eq("event_budget").all()
        and joined.extended_event_count.eq(32_768).all()
    ):
        raise RuntimeError("hard stop: S09 recurrent-active labels are incomplete")
    controls["s09_observed_status"] = np.select(
        [controls.completed, controls.stop_reason.eq("quiescent")],
        ["completed", "quiescent"],
        default="recurrent_active_32768",
    )
    controls["s09_recurrent_active"] = controls.s09_observed_status.eq(
        "recurrent_active_32768"
    )
    controls["anchor_swap_rate"] = np.divide(
        controls.cost_acceptedSwaps,
        controls.event_count,
        out=np.zeros(len(controls), dtype=float),
        where=controls.event_count.to_numpy() > 0,
    )
    controls["anchor_memory_rate"] = np.divide(
        controls.cost_memoryUpdates,
        controls.event_count,
        out=np.zeros(len(controls), dtype=float),
        where=controls.event_count.to_numpy() > 0,
    )
    controls["opportunity_rate_bin"] = [
        opportunity_rate_bin(int(events), int(swaps))
        for events, swaps in zip(controls.event_count, controls.cost_acceptedSwaps)
    ]
    return controls


def load_family_specs() -> dict[int, FamilySpec]:
    inventory = pd.read_parquet(S04 / "state_family_inventory.parquet")
    return {
        int(row.family_ordinal): FamilySpec.from_canonical_dict(
            json.loads(row.canonical_family_json)
        )
        for row in inventory.itertuples(index=False)
    }


def graph_arrays(
    family_ordinal: int,
    family_edges: pd.DataFrame,
    family_nodes: pd.DataFrame,
    expected_edge_count: int,
) -> dict[str, Any]:
    family_edges = family_edges.sort_values(
        ["source_state_ordinal", "opportunity_ordinal"], kind="stable"
    )
    family_nodes = family_nodes.sort_values("state_ordinal", kind="stable")
    node_count = len(family_nodes)
    if node_count == 0 or len(family_edges) != expected_edge_count:
        raise RuntimeError(f"graph slice size mismatch for family {family_ordinal}")
    count_series = family_edges.groupby("source_state_ordinal", sort=True).size()
    active_states = family_nodes.loc[
        family_nodes.terminal_code.eq(0), "state_ordinal"
    ].to_numpy(dtype=np.int64)
    if (
        len(count_series) != len(active_states)
        or not np.array_equal(count_series.index.to_numpy(dtype=np.int64), active_states)
        or not np.all(count_series.to_numpy() == count_series.iloc[0])
    ):
        raise RuntimeError(f"non-dense opportunity graph for family {family_ordinal}")
    slots = int(count_series.iloc[0])
    expected_sources = np.repeat(active_states, slots)
    expected_opportunities = np.tile(np.arange(slots, dtype=np.int16), len(active_states))
    if not (
        np.array_equal(
            family_edges.source_state_ordinal.to_numpy(dtype=np.int64),
            expected_sources,
        )
        and np.array_equal(
            family_edges.opportunity_ordinal.to_numpy(dtype=np.int16),
            expected_opportunities,
        )
        and np.array_equal(
            family_nodes.state_ordinal.to_numpy(dtype=np.int64),
            np.arange(node_count, dtype=np.int64),
        )
    ):
        raise RuntimeError(f"edge or node order mismatch for family {family_ordinal}")
    numerators = family_edges.probability_numerator.to_numpy(dtype=np.uint8).reshape(
        len(active_states), slots
    )
    denominators = family_edges.probability_denominator.to_numpy(dtype=np.uint8).reshape(
        len(active_states), slots
    )
    if not (
        np.all(numerators == numerators[0])
        and np.all(denominators == denominators[0])
    ):
        raise RuntimeError(f"state-dependent opportunity labels for family {family_ordinal}")
    # S05 intentionally emits no edges for terminal nodes.  The simulator never
    # selects from them; dense terminal slots below are storage-only self-loops.
    dense_count = node_count * slots
    dense_successors = np.repeat(np.arange(node_count, dtype=np.int64), slots)
    dense_decision = np.zeros(dense_count, dtype=np.int8)
    dense_changed = np.zeros(dense_count, dtype=np.int8)
    dense_kind = np.zeros(dense_count, dtype=np.int8)
    dense_costs = np.zeros((len(EDGE_COST_COLUMNS), dense_count), dtype=np.int16)
    dense_deltas = np.zeros((len(DELTA_COLUMNS), dense_count), dtype=np.int16)
    dense_indices = (
        family_edges.source_state_ordinal.to_numpy(dtype=np.int64) * slots
        + family_edges.opportunity_ordinal.to_numpy(dtype=np.int64)
    )
    dense_successors[dense_indices] = family_edges.successor_state_ordinal.to_numpy(
        dtype=np.int64
    )
    dense_decision[dense_indices] = family_edges.decision_code.to_numpy(dtype=np.int8)
    dense_changed[dense_indices] = family_edges.changed.to_numpy(dtype=np.int8)
    dense_kind[dense_indices] = family_edges.proposal_kind_code.to_numpy(dtype=np.int8)
    for index, column in enumerate(EDGE_COST_COLUMNS):
        dense_costs[index, dense_indices] = family_edges[column].to_numpy(dtype=np.int16)
    for index, column in enumerate(DELTA_COLUMNS):
        dense_deltas[index, dense_indices] = family_edges[column].to_numpy(dtype=np.int16)
    return {
        "slots": slots,
        "successors": dense_successors,
        "terminal": family_nodes.terminal_code.to_numpy(dtype=np.uint8),
        "opportunity_weights": opportunity_integer_weights(
            numerators[0], denominators[0]
        ),
        "decision": dense_decision,
        "changed": dense_changed,
        "kind": dense_kind,
        "costs": dense_costs,
        "deltas": dense_deltas,
    }


def result_rows_for_family(
    anchors: pd.DataFrame,
    arrays: Mapping[str, Any],
) -> list[dict[str, Any]]:
    start_levels = anchors[
        [f"start_{metric}" for metric in METRICS]
    ].to_numpy(dtype=np.int16)
    roots = np.asarray(
        [
            [stream_root(str(row.pair_block_id), name) for name in NULL_NAMES]
            for row in anchors.itertuples(index=False)
        ],
        dtype=np.uint64,
    )
    outputs = simulate_family(
        anchors.arm_state_ordinal.to_numpy(dtype=np.int64),
        roots,
        anchors.event_count.to_numpy(dtype=np.int32),
        anchors.cost_acceptedSwaps.to_numpy(dtype=np.int32),
        anchors.cost_memoryUpdates.to_numpy(dtype=np.int32),
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
        PRIMARY_BUDGET,
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
    ) = outputs
    metadata_columns = (
        "pair_block_id",
        "source_family_ordinal",
        "source_state_ordinal",
        "arm_family_ordinal",
        "arm_state_ordinal",
        "n",
        "architecture",
        "direction",
        "policy_profile",
        "intervention_type",
        "arm_variant",
        "focal_barrier_id",
        "target_cell_id",
        "replicate_index",
        "s09_observed_status",
        "s09_recurrent_active",
        "anchor_swap_rate",
        "anchor_memory_rate",
        "opportunity_rate_bin",
    )
    rows: list[dict[str, Any]] = []
    for run_index, anchor in enumerate(anchors.itertuples(index=False)):
        anchor_dict = anchor._asdict()
        for null_code, null_name in enumerate(NULL_NAMES):
            events = int(event_counts[run_index, null_code])
            stop_reason = STOP_LABELS[int(stop_codes[run_index, null_code])]
            row: dict[str, Any] = {
                "schema_version": "e03.s11.null_result.v1",
                "research_step_id": "S11",
                "null_run_id": "s11n:"
                + digest_id("E03/S11/run/v1", anchor.run_id, null_name),
                "s09_anchor_run_id": anchor.run_id,
                **{column: anchor_dict[column] for column in metadata_columns},
                "null_family": null_name,
                "greedy_metric": (
                    METRICS[null_code - 3] if null_code >= 3 else None
                ),
                "stream_root_hex": f"{int(roots[run_index, null_code]):016x}",
                "stream_profile": "sha256_root_splitmix64_counter_v1",
                "event_budget": PRIMARY_BUDGET,
                "initial_terminal": STOP_LABELS.get(
                    int(arrays["terminal"][int(anchor.arm_state_ordinal)]), "active"
                ),
                "stop_reason": stop_reason,
                "completed": stop_reason == "complete",
                "event_budget_censored_candidate": stop_reason == "event_budget",
                "event_count": events,
                "final_state_ordinal": int(final_states[run_index, null_code]),
                "trajectory_fingerprint_u64": f"{int(fingerprints[run_index, null_code]):016x}",
                "raw_stream_fingerprint_u64": f"{int(raw_fingerprints[run_index, null_code]):016x}",
                "checkpoint_2048_state_ordinal": int(
                    checkpoint_states[run_index, null_code]
                ),
                "checkpoint_2048_fingerprint_u64": (
                    f"{int(checkpoint_fingerprints[run_index, null_code]):016x}"
                ),
                "edge_compatibility_valid": True,
                "terminal_agreement_valid": True,
                "cost_activations": events,
                "cost_proposals": events,
                "cost_conflictLosses": 0,
                "selected_swap_count": int(selected[run_index, null_code, 0]),
                "selected_memory_count": int(selected[run_index, null_code, 1]),
                "selected_unchanged_count": int(selected[run_index, null_code, 2]),
                "requested_swap_count": int(requested[run_index, null_code, 0]),
                "requested_memory_count": int(requested[run_index, null_code, 1]),
                "requested_unchanged_count": int(requested[run_index, null_code, 2]),
                "infeasible_swap_request_count": int(deficits[run_index, null_code, 0]),
                "infeasible_memory_request_count": int(deficits[run_index, null_code, 1]),
                "infeasible_unchanged_request_count": int(deficits[run_index, null_code, 2]),
                "target_swap_count": int(anchor.cost_acceptedSwaps),
                "target_memory_count": int(anchor.cost_memoryUpdates),
                "target_rate_denominator": int(anchor.event_count),
            }
            for cost_index, cost_name in enumerate(COST_COLUMNS):
                row[f"cost_{cost_name}"] = int(ledgers[run_index, null_code, cost_index])
            row["s06_full_ledger_cost"] = events + sum(
                row[f"cost_{name}"] for name in COST_COLUMNS
            )
            row["e01_proposal_inclusive_full_ledger_cost"] = (
                row["s06_full_ledger_cost"] + events
            )
            row["rate_matched_target_swap_rate"] = float(anchor.anchor_swap_rate)
            row["rate_matched_target_memory_rate"] = float(anchor.anchor_memory_rate)
            row["achieved_swap_rate"] = selected[run_index, null_code, 0] / events if events else 0.0
            row["achieved_memory_rate"] = selected[run_index, null_code, 1] / events if events else 0.0
            for metric_code, metric in enumerate(METRICS):
                start = int(start_levels[run_index, metric_code])
                peak = int(peaks[run_index, null_code, metric_code])
                final = int(finals[run_index, null_code, metric_code])
                row[f"start_{metric}"] = start
                row[f"peak_{metric}"] = peak
                row[f"excursion_{metric}"] = peak - start
                row[f"final_{metric}"] = final
                row[f"worsening_events_{metric}"] = int(
                    worsening[run_index, null_code, metric_code]
                )
            rows.append(row)
    return rows


def build_metric_results(nulls: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    optimum = pd.read_parquet(S09 / "structural_optimum_changes.parquet")
    optimum = optimum.rename(
        columns={
            "arm_classification": "s09_unfiltered_classification",
            "arm_metric_level": "exact_start_metric_level",
            "arm_minimum_peak": "exact_minimum_peak",
            "arm_minimum_excursion": "exact_minimum_excursion",
        }
    )
    optimum_columns = [
        "arm_family_ordinal",
        "arm_state_ordinal",
        "arm_variant",
        "metric",
        "s09_unfiltered_classification",
        "exact_start_metric_level",
        "exact_minimum_peak",
        "exact_minimum_excursion",
    ]
    optimum = optimum[optimum_columns].drop_duplicates()
    if len(optimum) != 13_312:
        raise RuntimeError("hard stop: S09 exact optimum context table is incomplete")

    s10 = pd.read_parquet(S10 / "paired_effects.parquet")
    s10 = s10[s10.filter_threshold == 0][
        [
            "s09_control_run_id",
            "filter_metric",
            "suppressed_count",
            "filtered_classification",
            "filtered_exact_status",
            "intervention_created_impossibility",
            "preexisting_exact_impossibility",
        ]
    ].rename(
        columns={
            "s09_control_run_id": "s09_anchor_run_id",
            "filter_metric": "metric",
            "suppressed_count": "s10_strict_suppressed_count",
            "filtered_classification": "s10_strict_filtered_classification",
            "filtered_exact_status": "s10_strict_filtered_exact_status",
            "intervention_created_impossibility": "s10_intervention_created_impossibility",
            "preexisting_exact_impossibility": "s10_preexisting_exact_impossibility",
        }
    )
    if len(s10) != 100_896:
        raise RuntimeError("hard stop: S10 strict-exposure table is incomplete")
    s10["s10_strict_filter_exposed"] = s10.s10_strict_suppressed_count.gt(0)

    family_ordinals = sorted(int(value) for value in nulls.arm_family_ordinal.unique())
    cost = ds.dataset(S07 / "minimum_cost_solutions.parquet", format="parquet").to_table(
        filter=ds.field("family_ordinal").isin(family_ordinals)
    ).to_pandas()
    contexts = nulls[["arm_family_ordinal", "arm_state_ordinal"]].drop_duplicates()
    cost = contexts.merge(
        cost,
        left_on=["arm_family_ordinal", "arm_state_ordinal"],
        right_on=["family_ordinal", "state_ordinal"],
        how="left",
        validate="one_to_one",
    )
    if cost.shortest_activations.isna().any():
        raise RuntimeError("hard stop: an S11 context lacks an S07 cost solution")
    full_fields = [
        "minimum_full_ledger_activations",
        "minimum_full_ledger_observation_reads",
        "minimum_full_ledger_value_comparisons",
        "minimum_full_ledger_no_ops",
        "minimum_full_ledger_rejections",
        "minimum_full_ledger_memory_updates",
        "minimum_full_ledger_accepted_swaps",
        "minimum_full_ledger_displaced_cells",
    ]
    cost["minimum_s06_full_ledger_cost"] = cost[full_fields].sum(axis=1)
    cost = cost[
        [
            "arm_family_ordinal",
            "arm_state_ordinal",
            "shortest_activations",
            "minimum_accepted_swaps",
            "minimum_s06_full_ledger_cost",
        ]
    ]

    common = [
        "null_run_id",
        "s09_anchor_run_id",
        "pair_block_id",
        "source_family_ordinal",
        "source_state_ordinal",
        "arm_family_ordinal",
        "arm_state_ordinal",
        "n",
        "architecture",
        "direction",
        "policy_profile",
        "intervention_type",
        "arm_variant",
        "focal_barrier_id",
        "target_cell_id",
        "replicate_index",
        "null_family",
        "greedy_metric",
        "s09_observed_status",
        "s09_recurrent_active",
        "opportunity_rate_bin",
        "completed",
        "stop_reason",
        "event_count",
        "cost_acceptedSwaps",
        "s06_full_ledger_cost",
        "e01_proposal_inclusive_full_ledger_cost",
        "trajectory_fingerprint_u64",
    ]
    frames: list[pd.DataFrame] = []
    for metric in METRICS:
        frame = nulls[
            common
            + [
                f"start_{metric}",
                f"peak_{metric}",
                f"excursion_{metric}",
                f"final_{metric}",
                f"worsening_events_{metric}",
            ]
        ].copy()
        frame["metric"] = metric
        frame = frame.rename(
            columns={
                f"start_{metric}": "start_metric_level",
                f"peak_{metric}": "peak_metric_level",
                f"excursion_{metric}": "observed_excursion",
                f"final_{metric}": "final_metric_level",
                f"worsening_events_{metric}": "worsening_event_count",
            }
        )
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result = result.merge(
        optimum,
        on=["arm_family_ordinal", "arm_state_ordinal", "arm_variant", "metric"],
        how="left",
        validate="many_to_one",
    ).merge(
        s10,
        on=["s09_anchor_run_id", "metric"],
        how="left",
        validate="many_to_one",
    ).merge(
        cost,
        on=["arm_family_ordinal", "arm_state_ordinal"],
        how="left",
        validate="many_to_one",
    )
    if result[
        [
            "s09_unfiltered_classification",
            "s10_strict_suppressed_count",
            "shortest_activations",
        ]
    ].isna().any().any():
        raise RuntimeError("hard stop: S07/S09/S10 metric join is incomplete")
    result["s09_exact_start_status"] = result.s09_unfiltered_classification.map(
        exact_start_status
    )
    result["exact_impossibility"] = result.s09_exact_start_status.eq(
        "unreachable_active"
    )
    result["quiescent_start"] = result.s09_exact_start_status.eq("quiescent")
    result["stopped_quiescent"] = result.stop_reason.eq("quiescent")
    result["reached_quiescence"] = (
        result.stopped_quiescent & ~result.quiescent_start
    )
    result["completion_censored"] = (
        result.stop_reason.eq("event_budget")
        & result.s09_exact_start_status.eq("reachable")
    )
    result["exact_impossibility_not_censoring"] = (
        result.exact_impossibility & ~result.completion_censored
    )
    result["behavior_classification"] = [
        behavior_classification(
            completed=bool(row.completed),
            stop_reason=str(row.stop_reason),
            exact_classification=str(row.s09_unfiltered_classification),
            excursion=int(row.observed_excursion),
            minimum_excursion=int(row.exact_minimum_excursion),
        )
        for row in result.itertuples(index=False)
    ]
    result["activation_optimality_gap"] = np.where(
        result.completed,
        result.event_count - result.shortest_activations,
        np.nan,
    )
    result["accepted_swap_optimality_gap"] = np.where(
        result.completed,
        result.cost_acceptedSwaps - result.minimum_accepted_swaps,
        np.nan,
    )
    result["s06_full_ledger_optimality_gap"] = np.where(
        result.completed,
        result.s06_full_ledger_cost - result.minimum_s06_full_ledger_cost,
        np.nan,
    )
    result.insert(0, "schema_version", "e03.s11.null_metric_result.v1")
    result.insert(1, "research_step_id", "S11")
    return result


def build_observed_comparisons(
    metric_results: pd.DataFrame,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    observed_common = [
        "run_id",
        "completed",
        "stop_reason",
        "event_count",
        "cost_acceptedSwaps",
        "full_ledger_unit_cost",
    ]
    frames: list[pd.DataFrame] = []
    for metric in METRICS:
        frame = controls[
            observed_common
            + [f"excursion_{metric}", f"final_{metric}"]
        ].copy()
        frame["metric"] = metric
        frame = frame.rename(
            columns={
                "run_id": "s09_anchor_run_id",
                "completed": "observed_completed",
                "stop_reason": "observed_stop_reason",
                "event_count": "observed_event_count",
                "cost_acceptedSwaps": "observed_accepted_swaps",
                "full_ledger_unit_cost": "observed_e01_full_ledger_cost",
                f"excursion_{metric}": "observed_s09_excursion",
                f"final_{metric}": "observed_s09_final_metric_level",
            }
        )
        frames.append(frame)
    observed = pd.concat(frames, ignore_index=True)
    columns = [
        "null_run_id",
        "s09_anchor_run_id",
        "source_family_ordinal",
        "source_state_ordinal",
        "arm_family_ordinal",
        "arm_state_ordinal",
        "replicate_index",
        "null_family",
        "metric",
        "intervention_type",
        "s09_observed_status",
        "s09_exact_start_status",
        "s10_strict_filter_exposed",
        "opportunity_rate_bin",
        "completed",
        "stop_reason",
        "event_count",
        "cost_acceptedSwaps",
        "e01_proposal_inclusive_full_ledger_cost",
        "observed_excursion",
        "final_metric_level",
        "completion_censored",
        "exact_impossibility",
        "quiescent_start",
        "stopped_quiescent",
        "reached_quiescence",
    ]
    result = metric_results[columns].merge(
        observed,
        on=["s09_anchor_run_id", "metric"],
        how="left",
        validate="many_to_one",
    )
    result["delta_completed_null_minus_observed"] = (
        result.completed.astype(np.int8) - result.observed_completed.astype(np.int8)
    )
    result["delta_excursion_null_minus_observed"] = (
        result.observed_excursion - result.observed_s09_excursion
    )
    result["delta_final_error_null_minus_observed"] = (
        result.final_metric_level - result.observed_s09_final_metric_level
    )
    result["successful_efficiency_comparable"] = (
        result.completed & result.observed_completed
    )
    result["delta_activations_when_both_successful"] = np.where(
        result.successful_efficiency_comparable,
        result.event_count - result.observed_event_count,
        np.nan,
    )
    result["delta_swaps_when_both_successful"] = np.where(
        result.successful_efficiency_comparable,
        result.cost_acceptedSwaps - result.observed_accepted_swaps,
        np.nan,
    )
    result["delta_e01_full_cost_when_both_successful"] = np.where(
        result.successful_efficiency_comparable,
        result.e01_proposal_inclusive_full_ledger_cost
        - result.observed_e01_full_ledger_cost,
        np.nan,
    )
    result.insert(0, "schema_version", "e03.s11.observed_null_comparison.v1")
    result.insert(1, "research_step_id", "S11")
    return result


def build_barrier_effects(metric_results: pd.DataFrame) -> pd.DataFrame:
    baseline = metric_results[metric_results.intervention_type == "baseline"][
        [
            "pair_block_id",
            "null_family",
            "metric",
            "null_run_id",
            "completed",
            "stop_reason",
            "event_count",
            "cost_acceptedSwaps",
            "s06_full_ledger_cost",
            "observed_excursion",
            "final_metric_level",
            "trajectory_fingerprint_u64",
        ]
    ].rename(
        columns={
            "null_run_id": "baseline_null_run_id",
            "completed": "baseline_completed",
            "stop_reason": "baseline_stop_reason",
            "event_count": "baseline_event_count",
            "cost_acceptedSwaps": "baseline_accepted_swaps",
            "s06_full_ledger_cost": "baseline_s06_full_ledger_cost",
            "observed_excursion": "baseline_excursion",
            "final_metric_level": "baseline_final_metric_level",
            "trajectory_fingerprint_u64": "baseline_trajectory_fingerprint_u64",
        }
    )
    active = metric_results[metric_results.intervention_type != "baseline"].copy()
    result = active.merge(
        baseline,
        on=["pair_block_id", "null_family", "metric"],
        how="left",
        validate="many_to_one",
    )
    if result.baseline_null_run_id.isna().any():
        raise RuntimeError("hard stop: a null barrier arm lacks its matched baseline")
    result["delta_completed_arm_minus_baseline"] = (
        result.completed.astype(np.int8) - result.baseline_completed.astype(np.int8)
    )
    result["delta_excursion_arm_minus_baseline"] = (
        result.observed_excursion - result.baseline_excursion
    )
    result["delta_final_error_arm_minus_baseline"] = (
        result.final_metric_level - result.baseline_final_metric_level
    )
    result["successful_efficiency_comparable"] = (
        result.completed & result.baseline_completed
    )
    result["delta_activations_when_both_successful"] = np.where(
        result.successful_efficiency_comparable,
        result.event_count - result.baseline_event_count,
        np.nan,
    )
    result["delta_swaps_when_both_successful"] = np.where(
        result.successful_efficiency_comparable,
        result.cost_acceptedSwaps - result.baseline_accepted_swaps,
        np.nan,
    )
    result["delta_s06_full_cost_when_both_successful"] = np.where(
        result.successful_efficiency_comparable,
        result.s06_full_ledger_cost - result.baseline_s06_full_ledger_cost,
        np.nan,
    )
    keep = [
        "null_run_id",
        "baseline_null_run_id",
        "pair_block_id",
        "source_family_ordinal",
        "source_state_ordinal",
        "arm_family_ordinal",
        "arm_state_ordinal",
        "replicate_index",
        "null_family",
        "metric",
        "intervention_type",
        "arm_variant",
        "s09_observed_status",
        "s09_exact_start_status",
        "s10_strict_filter_exposed",
        "opportunity_rate_bin",
        "completed",
        "baseline_completed",
        "completion_censored",
        "exact_impossibility",
        "quiescent_start",
        "stopped_quiescent",
        "reached_quiescence",
        "delta_completed_arm_minus_baseline",
        "delta_excursion_arm_minus_baseline",
        "delta_final_error_arm_minus_baseline",
        "successful_efficiency_comparable",
        "delta_activations_when_both_successful",
        "delta_swaps_when_both_successful",
        "delta_s06_full_cost_when_both_successful",
    ]
    result = result[keep]
    result.insert(0, "schema_version", "e03.s11.null_barrier_effect.v1")
    result.insert(1, "research_step_id", "S11")
    return result


def build_distributions(metric_results: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "null_family",
        "metric",
        "intervention_type",
        "s09_exact_start_status",
        "s09_observed_status",
        "s10_strict_filter_exposed",
        "s10_strict_filtered_exact_status",
        "s10_intervention_created_impossibility",
        "opportunity_rate_bin",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in metric_results.groupby(group_columns, dropna=False, sort=True):
        successful = group[group.completed]
        row = dict(zip(group_columns, key))
        row.update(
            {
                "run_metric_count": len(group),
                "completion_count": int(group.completed.sum()),
                "completion_rate": float(group.completed.mean()),
                "exact_impossibility_count": int(group.exact_impossibility.sum()),
                "quiescent_start_count": int(group.quiescent_start.sum()),
                "reached_quiescence_count": int(group.reached_quiescence.sum()),
                "reachable_censored_count": int(group.completion_censored.sum()),
                "mean_final_metric_level": float(group.final_metric_level.mean()),
                "median_final_metric_level": float(group.final_metric_level.median()),
                "mean_excursion": float(group.observed_excursion.mean()),
                "p90_excursion": float(group.observed_excursion.quantile(0.9)),
                "successful_count": len(successful),
                "successful_mean_activation_gap": (
                    float(successful.activation_optimality_gap.mean())
                    if len(successful)
                    else np.nan
                ),
                "successful_mean_s06_full_cost_gap": (
                    float(successful.s06_full_ledger_optimality_gap.mean())
                    if len(successful)
                    else np.nan
                ),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    result.insert(0, "schema_version", "e03.s11.null_distribution.v1")
    result.insert(1, "research_step_id", "S11")
    return result


def bootstrap_summary(comparisons: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cluster_columns = ["source_family_ordinal", "source_state_ordinal"]
    estimands = (
        "delta_completed_null_minus_observed",
        "delta_excursion_null_minus_observed",
        "delta_final_error_null_minus_observed",
    )
    for (null_name, metric), group in comparisons.groupby(
        ["null_family", "metric"], sort=True
    ):
        grouped = group.groupby(cluster_columns, sort=True)
        clusters = grouped.size().to_numpy(dtype=np.int64)
        values = {
            estimand: grouped[estimand].sum().to_numpy(dtype=float)
            for estimand in estimands
        }
        seed = int.from_bytes(
            hashlib.sha256(
                f"E03/S11/bootstrap/v1:{null_name}:{metric}".encode()
            ).digest()[:8],
            "big",
        )
        rng = np.random.Generator(np.random.PCG64DXSM(seed))
        draws = np.empty((len(estimands), BOOTSTRAPS), dtype=float)
        for start in range(0, BOOTSTRAPS, 250):
            stop = min(start + 250, BOOTSTRAPS)
            indices = rng.integers(0, len(clusters), size=(stop - start, len(clusters)))
            denominators = clusters[indices].sum(axis=1)
            for index, estimand in enumerate(estimands):
                draws[index, start:stop] = values[estimand][indices].sum(axis=1) / denominators
        for index, estimand in enumerate(estimands):
            rows.append(
                {
                    "schema_version": "e03.s11.null_calibration_bootstrap.v1",
                    "research_step_id": "S11",
                    "null_family": null_name,
                    "metric": metric,
                    "estimand": estimand,
                    "row_count": len(group),
                    "cluster_count": len(clusters),
                    "estimate": float(group[estimand].mean()),
                    "bootstrap_draws": BOOTSTRAPS,
                    "bootstrap_seed": seed,
                    "ci_lower_95": float(np.quantile(draws[index], 0.025)),
                    "ci_upper_95": float(np.quantile(draws[index], 0.975)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    if Path("/artifacts/research_steps/S12").exists():
        raise RuntimeError("hard stop: S12 artifact directory already exists")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    started = time.time()
    inputs = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in input_paths()
    ]
    write_json(CACHE / "input_hashes_before.json", {"inputs": inputs})
    controls = load_controls()
    families = load_family_specs()
    family_ordinals = sorted(int(value) for value in controls.arm_family_ordinal.unique())
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
    rows: list[dict[str, Any]] = []
    graph_audit: list[dict[str, Any]] = []
    for ordinal_index, family_ordinal in enumerate(family_ordinals):
        family_edges = edges[edges.family_ordinal == family_ordinal]
        family_nodes = nodes[nodes.family_ordinal == family_ordinal]
        arrays = graph_arrays(
            family_ordinal,
            family_edges,
            family_nodes,
            int(manifest.loc[family_ordinal, "edge_count"]),
        )
        anchors = controls[controls.arm_family_ordinal == family_ordinal]
        rows.extend(result_rows_for_family(anchors, arrays))
        graph_audit.append(
            {
                "family_ordinal": family_ordinal,
                "node_count": len(family_nodes),
                "edge_count": len(family_edges),
                "opportunity_count_per_state": arrays["slots"],
                "anchor_count": len(anchors),
                "null_run_count": len(anchors) * len(NULL_NAMES),
                "dense_edge_order_valid": True,
                "opportunity_labels_state_independent": True,
                "opportunity_weight_sum": int(arrays["opportunity_weights"].sum()),
            }
        )
        if (ordinal_index + 1) % 50 == 0:
            print(
                json.dumps(
                    {
                        "familiesCompleted": ordinal_index + 1,
                        "familiesTotal": len(family_ordinals),
                        "nullRows": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    nulls = pd.DataFrame(rows)
    if len(nulls) != 25_224 * len(NULL_NAMES):
        raise RuntimeError(f"hard stop: null run coverage mismatch {len(nulls)}")
    write_parquet(OUTPUT / "null_results.parquet", nulls, "e03.s11.null_result.v1")
    write_parquet(
        OUTPUT / "graph_coverage_diagnostics.parquet",
        graph_audit,
        "e03.s11.graph_coverage_diagnostic.v1",
    )

    metric_results = build_metric_results(nulls, controls)
    if len(metric_results) != len(nulls) * len(METRICS):
        raise RuntimeError("hard stop: null metric coverage mismatch")
    write_parquet(
        OUTPUT / "null_metric_results.parquet",
        metric_results,
        "e03.s11.null_metric_result.v1",
    )
    comparisons = build_observed_comparisons(metric_results, controls)
    write_parquet(
        OUTPUT / "observed_null_comparisons.parquet",
        comparisons,
        "e03.s11.observed_null_comparison.v1",
    )
    barrier = build_barrier_effects(metric_results)
    write_parquet(
        OUTPUT / "null_barrier_effects.parquet",
        barrier,
        "e03.s11.null_barrier_effect.v1",
    )
    distributions = build_distributions(metric_results)
    write_parquet(
        OUTPUT / "null_distributions.parquet",
        distributions,
        "e03.s11.null_distribution.v1",
    )
    bootstrap = bootstrap_summary(comparisons)
    write_parquet(
        OUTPUT / "null_calibration_bootstrap.parquet",
        bootstrap,
        "e03.s11.null_calibration_bootstrap.v1",
    )

    matching_columns = [
        "null_run_id",
        "s09_anchor_run_id",
        "pair_block_id",
        "source_family_ordinal",
        "source_state_ordinal",
        "arm_family_ordinal",
        "arm_state_ordinal",
        "replicate_index",
        "null_family",
        "intervention_type",
        "s09_observed_status",
        "s09_recurrent_active",
        "opportunity_rate_bin",
        "target_rate_denominator",
        "target_swap_count",
        "target_memory_count",
        "rate_matched_target_swap_rate",
        "rate_matched_target_memory_rate",
        "requested_swap_count",
        "requested_memory_count",
        "requested_unchanged_count",
        "infeasible_swap_request_count",
        "infeasible_memory_request_count",
        "infeasible_unchanged_request_count",
        "selected_swap_count",
        "selected_memory_count",
        "selected_unchanged_count",
        "event_count",
        "achieved_swap_rate",
        "achieved_memory_rate",
        "stream_root_hex",
    ]
    write_parquet(
        OUTPUT / "matching_diagnostics.parquet",
        nulls[matching_columns],
        "e03.s11.matching_diagnostic.v1",
    )
    write_parquet(
        OUTPUT / "action_rate_matching.parquet",
        nulls[nulls.null_family == "rate_matched_random"][matching_columns],
        "e03.s11.action_rate_matching.v1",
    )
    feasible = nulls[
        [
            "null_run_id",
            "null_family",
            "arm_family_ordinal",
            "arm_state_ordinal",
            "event_count",
            "selected_swap_count",
            "selected_memory_count",
            "selected_unchanged_count",
            "infeasible_swap_request_count",
            "infeasible_memory_request_count",
            "infeasible_unchanged_request_count",
            "edge_compatibility_valid",
            "terminal_agreement_valid",
        ]
    ].copy()
    feasible["event_partition_valid"] = (
        feasible.selected_swap_count
        + feasible.selected_memory_count
        + feasible.selected_unchanged_count
        == feasible.event_count
    )
    write_parquet(
        OUTPUT / "feasible_action_diagnostics.parquet",
        feasible,
        "e03.s11.feasible_action_diagnostic.v1",
    )

    environment = {
        "schemaVersion": "e03.s11.environment.v1",
        "researchStepId": "S11",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "workers": 1,
        "threadEnvironment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "intentionalSerialExecution": "Numba family kernels use deterministic serial order; graph slices are small and shared-process execution avoids repeated reads.",
    }
    write_json(OUTPUT / "environment.json", environment)
    build_summary = {
        "schemaVersion": "e03.s11.build_summary.v1",
        "researchStepId": "S11",
        "observedAnchors": len(controls),
        "structuralContexts": 3_328,
        "sourceStarts": 407,
        "armFamilies": len(family_ordinals),
        "graphEdgesLoaded": len(edges),
        "graphNodesLoaded": len(nodes),
        "nullFamilies": list(NULL_NAMES),
        "nullRuns": len(nulls),
        "nullMetricRows": len(metric_results),
        "barrierMetricPairs": len(barrier),
        "bootstrapRows": len(bootstrap),
        "runtimeSeconds": time.time() - started,
    }
    write_json(OUTPUT / "build_summary.json", build_summary)
    print(json.dumps(build_summary, sort_keys=True))


if __name__ == "__main__":
    main()
