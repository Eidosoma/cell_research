#!/usr/bin/env python3
"""Build the E03 S08 observed-behavior versus exact-necessity comparison."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from reference_simulator.engine import evaluate_terminal, execute_batch, initial_state
from reference_simulator.model import RunState, Scenario, canonical_json_bytes, state_hash
from scripts.replay_e01_distances import (
    S06 as E01_S06,
    _materialize,
    build_tasks,
)
from src.detours.necessary_detour import (
    CLASS_LABELS,
    COST_PROFILES,
    PrimarySolution,
    UNREACHABLE,
    cost_matrix_from_s05,
    family_metric_levels,
    solve_lexicographic_witness,
)
from src.detours.observed_comparison import (
    ScenarioProjection,
    classify_observed_suffix,
    event_scheduler_label,
    lexicographic_relation,
    project_scenario,
)
from src.detours.replay import stable_trace_id, values_hash
from src.detours.state_space import FamilySpec, StructuralState, family_lookup_key
from src.detours.transition_graph import (
    DECISION_CODES,
    PROPOSAL_KIND_CODES,
    REASON_CODES,
)


OUTPUT = Path("/artifacts/research_steps/S08")
S02 = Path("/artifacts/research_steps/S02")
S04 = Path("/artifacts/research_steps/S04")
S05 = Path("/artifacts/research_steps/S05")
S06 = Path("/artifacts/research_steps/S06")
S07 = Path("/artifacts/research_steps/S07")
E01 = Path("/previous-artifacts/E01")
REPOSITORY = Path(__file__).resolve().parents[1]
METRICS = (
    "adjacent_descents",
    "inversion_count",
    "spearman_footrule",
    "maximum_rank_error",
)
METRIC_CODES = {name: index for index, name in enumerate(METRICS)}
CLASS_CONTRADICTION_PREFIX = "semantic_contradiction"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], schema_version: str) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty required table {path.name}")
    table = pa.Table.from_pylist(list(rows)).replace_schema_metadata(
        {
            b"schemaVersion": schema_version.encode(),
            b"researchStepId": b"S08",
        }
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


def clone_state(state: RunState) -> RunState:
    return state.clone()


def state_ordinal(state: StructuralState) -> int:
    return state.selection_cursor_code * math.factorial(state.family.n) + state.occupancy_rank


def native_values(scenario: Scenario, state: RunState) -> list[int | float]:
    return [scenario.cell_map[cell_id].value for cell_id in state.occupancy]


def native_terminal_code(terminal: str | None) -> int:
    return {None: 0, "complete": 1, "quiescent": 2}[terminal]


def input_paths() -> list[Path]:
    paths = [
        Path("/workspace/AGENTS.md"),
        Path("/workspace/FULL_PLAN.md"),
        Path("/workspace/RESEARCH_PLAN.md"),
        Path("/workspace/PREVIOUS_ARTIFACTS.md"),
        Path("/workspace/PREVIOUS_ARTIFACTS.json"),
        Path("/workspace/input-attachments/MANIFEST.json"),
        S02 / "replayed_distances.parquet",
        S02 / "trace_coverage_manifest.json",
        S02 / "replay_diagnostics.json",
        S04 / "state_family_inventory.parquet",
        S04 / "canonical_state_encoder.md",
        S05 / "graph_family_manifest.parquet",
        S05 / "graph_schema.md",
        S06 / "necessary_detour_spec.md",
        S07 / "path_solutions.parquet",
        S07 / "minimum_cost_solutions.parquet",
        E01 / "research_steps/S05/smoke_sample_result.json",
        E01 / "research_steps/S06/examples/reference_trace.parquet",
        E01 / "research_steps/S06/examples/reference_trace.manifest.json",
        REPOSITORY / "analysis/s08_comparison_contract.json",
        REPOSITORY / "scripts/build_s08_comparison.py",
        REPOSITORY / "src/detours/observed_comparison.py",
        REPOSITORY / "src/detours/state_space.py",
        REPOSITORY / "src/detours/transition_graph.py",
        REPOSITORY / "src/detours/necessary_detour.py",
        REPOSITORY / "reference_simulator/engine.py",
        REPOSITORY / "reference_simulator/model.py",
        REPOSITORY / "reference_simulator/policies.py",
        REPOSITORY / "reference_simulator/scheduler.py",
        REPOSITORY / "reference_simulator/transition_primitives.py",
    ]
    paths.extend(sorted(Path("/workspace/input-attachments").glob("*/_metadata/ATTACHMENT.md")))
    return paths


def load_family_context() -> tuple[pd.DataFrame, dict[tuple[object, ...], dict[str, Any]]]:
    families = pq.read_table(S04 / "state_family_inventory.parquet").to_pandas()
    lookup: dict[tuple[object, ...], dict[str, Any]] = {}
    for row in families.to_dict("records"):
        family = FamilySpec.from_canonical_dict(json.loads(row["canonical_family_json"]))
        lookup[family_lookup_key(family)] = row
    return families, lookup


def scenario_metadata(scenario: Scenario) -> dict[str, Any]:
    directions = {cell.direction.value for cell in scenario.cells}
    values = [cell.value for cell in scenario.cells]
    return {
        "n": len(scenario.cells),
        "architecture": scenario.architecture.value,
        "direction_scope": "homogeneous" if len(directions) == 1 else "mixed_goal",
        "direction": next(iter(directions)) if len(directions) == 1 else "mixed",
        "unique_values": len(values) == len(set(values)),
        "batch_width": scenario.batch_width,
        "has_selection": any(cell.policy.value == "Selection" for cell in scenario.cells),
    }


def coverage_rows(
    family_lookup: Mapping[tuple[object, ...], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], Scenario, ScenarioProjection, dict[str, Any]]]]:
    tasks, s13_tasks, gaps = build_tasks()
    rows: list[dict[str, Any]] = []
    eligible: list[tuple[dict[str, Any], Scenario, ScenarioProjection, dict[str, Any]]] = []

    def add_task(task: dict[str, Any], engine: str) -> None:
        scenario, _ = _materialize(task)
        metadata = scenario_metadata(scenario)
        projection: ScenarioProjection | None = None
        family_row: Mapping[str, Any] | None = None
        projection_error: str | None = None
        try:
            projection = project_scenario(scenario)
            family_row = family_lookup.get(family_lookup_key(projection.family))
        except ValueError as error:
            projection_error = str(error)
        mixed = metadata["direction_scope"] == "mixed_goal"
        large = metadata["n"] > 9
        selection_boundary = (
            not large
            and not mixed
            and metadata["has_selection"]
            and family_row is None
        )
        exact = (
            engine == "standard"
            and family_row is not None
            and scenario.batch_width == 1
            and projection is not None
        )
        if exact:
            reason = "exact_structural_mapping"
        elif mixed:
            reason = "mixed_goal_outside_s04"
        elif large:
            reason = "large_n_outside_s04_s07"
        elif selection_boundary:
            reason = "selection_memory_outside_retained_s04_size"
        elif engine != "standard":
            reason = "nonstandard_replay_path_outside_exact_graph_mapping"
        else:
            reason = "family_not_retained"
        row = {
            "logical_trace_id": task["logical_trace_id"],
            "scenario_id": scenario.scenario_id,
            "source_step_id": task["source_step_id"],
            "source_artifact": task["source_artifact"],
            "primary_denominator": True,
            "supplemental": False,
            "s02_alignment_status": "aligned",
            **metadata,
            "missing_ordered_state": False,
            "large_n_boundary": large,
            "mixed_goal_boundary": mixed,
            "selection_memory_boundary": selection_boundary,
            "scheduler_projection_boundary": True,
            "family_retained": family_row is not None,
            "family_ordinal": int(family_row["family_ordinal"]) if family_row is not None else None,
            "family_id": family_row["family_id"] if family_row is not None else None,
            "exact_mapping_eligible": exact,
            "coverage_reason": reason,
            "projection_detail": projection_error,
        }
        rows.append(row)
        if exact:
            eligible.append((task, scenario, projection, dict(family_row)))  # type: ignore[arg-type]

    for task in tasks:
        add_task(task, "standard")
    for task in s13_tasks:
        add_task(task, "s13")

    smoke = json.loads((E01 / "research_steps/S05/smoke_sample_result.json").read_text())
    reference_scenario = Scenario.from_dict(smoke["scenario"])
    reference_projection = project_scenario(reference_scenario)
    reference_family = family_lookup.get(family_lookup_key(reference_projection.family))
    if reference_family is None:
        raise AssertionError("retained n=4 reference family is missing from S04")
    reference_task = {
        "engine": "reference_retained",
        "source_step_id": "S06",
        "source_artifact": "research_steps/S06/examples/reference_trace.jsonl.zst",
        "logical_trace_id": stable_trace_id("S06/reference_trace", reference_scenario.scenario_id),
        "scenario_id": reference_scenario.scenario_id,
        "expected": {
            "final_state_hash": smoke["finalStateHash"],
            "activation_count": smoke["summary"]["activationCount"],
            "accepted_swaps": smoke["summary"]["ledger"]["acceptedSwaps"],
            "stop_reason": smoke["summary"]["stopReason"],
        },
        "retained_events": smoke["events"],
    }
    rows.append(
        {
            "logical_trace_id": reference_task["logical_trace_id"],
            "scenario_id": reference_scenario.scenario_id,
            "source_step_id": "S06",
            "source_artifact": reference_task["source_artifact"],
            "primary_denominator": True,
            "supplemental": False,
            "s02_alignment_status": "aligned",
            **scenario_metadata(reference_scenario),
            "missing_ordered_state": False,
            "large_n_boundary": False,
            "mixed_goal_boundary": False,
            "selection_memory_boundary": False,
            "scheduler_projection_boundary": True,
            "family_retained": True,
            "family_ordinal": int(reference_family["family_ordinal"]),
            "family_id": reference_family["family_id"],
            "exact_mapping_eligible": True,
            "coverage_reason": "exact_structural_mapping",
            "projection_detail": "native identities canonically relabelled by unique value rank",
        }
    )
    eligible.append((reference_task, reference_scenario, reference_projection, dict(reference_family)))

    historical = json.loads((E01_S06 / "examples/historical_trace.manifest.json").read_text())
    rows.append(
        {
            "logical_trace_id": stable_trace_id("S06/historical_trace", historical["scenarioId"]),
            "scenario_id": historical["scenarioId"],
            "source_step_id": "S06",
            "source_artifact": "research_steps/S06/examples/historical_trace.jsonl.zst",
            "primary_denominator": False,
            "supplemental": True,
            "s02_alignment_status": "aligned_observable_only",
            "n": 2,
            "architecture": "historical_unknown",
            "direction_scope": "homogeneous",
            "direction": "ascending",
            "unique_values": True,
            "batch_width": None,
            "has_selection": None,
            "missing_ordered_state": False,
            "large_n_boundary": False,
            "mixed_goal_boundary": False,
            "selection_memory_boundary": False,
            "scheduler_projection_boundary": True,
            "family_retained": False,
            "family_ordinal": None,
            "family_id": None,
            "exact_mapping_eligible": False,
            "coverage_reason": "below_s04_minimum_n_and_native_state_unavailable",
            "projection_detail": "supplemental observable snapshots only",
        }
    )
    for gap in gaps:
        rows.append(
            {
                "logical_trace_id": gap["logicalTraceId"],
                "scenario_id": gap["scenarioId"],
                "source_step_id": gap["sourceStepId"],
                "source_artifact": gap["sourceArtifact"],
                "primary_denominator": True,
                "supplemental": False,
                "s02_alignment_status": "coverage_gap",
                "n": 100,
                "architecture": "unknown_from_metric_only_trace",
                "direction_scope": "unknown",
                "direction": "unknown",
                "unique_values": True,
                "batch_width": None,
                "has_selection": None,
                "missing_ordered_state": True,
                "large_n_boundary": True,
                "mixed_goal_boundary": False,
                "selection_memory_boundary": False,
                "scheduler_projection_boundary": True,
                "family_retained": False,
                "family_ordinal": None,
                "family_id": None,
                "exact_mapping_eligible": False,
                "coverage_reason": "missing_ordered_state_snapshots",
                "projection_detail": gap["detail"],
            }
        )
    rows.sort(key=lambda row: (row["source_step_id"], row["scenario_id"], row["supplemental"]))
    eligible.sort(key=lambda item: (item[0]["source_step_id"], item[1].scenario_id))
    primary = [row for row in rows if row["primary_denominator"]]
    if len(primary) != 90 or len(rows) != 91:
        raise AssertionError(f"S02 denominator drift: primary={len(primary)}, all={len(rows)}")
    return rows, eligible


def replay_eligible(
    eligible: Sequence[tuple[dict[str, Any], Scenario, ScenarioProjection, dict[str, Any]]]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for task, scenario, projection, family_row in eligible:
        state = initial_state(scenario)
        state.terminal = evaluate_terminal(scenario, state)
        states = [clone_state(state)]
        events: list[dict[str, Any]] = []
        retained = task.get("retained_events")
        while state.terminal is None:
            produced, _ = execute_batch(scenario, state, retain_events=True, emit_event_records=True)
            if len(produced) != 1:
                raise AssertionError("S08 exact mapping requires serial one-event batches")
            event = produced[0]
            if retained is not None:
                expected = retained[len(events)]
                if event != expected:
                    raise AssertionError("retained reference event differs from deterministic replay")
            events.append(event)
            states.append(clone_state(state))
        expected = task["expected"]
        checks = {
            "final_state_hash": state_hash(scenario.scenario_id, state) == expected["final_state_hash"],
            "activation_count": state.activation_count == int(expected["activation_count"]),
            "accepted_swaps": state.ledger["acceptedSwaps"] == int(expected["accepted_swaps"]),
            "stop_reason": state.terminal == expected["stop_reason"],
            "retained_event_count": retained is None or len(events) == len(retained),
        }
        for key, passed in checks.items():
            if not passed:
                raise AssertionError(f"{task['logical_trace_id']} replay check failed: {key}")
        results.append(
            {
                "task": task,
                "scenario": scenario,
                "projection": projection,
                "family_row": family_row,
                "states": states,
                "events": events,
                "checks": checks,
            }
        )
    return results


def load_graph(family_ordinal: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    shard = family_ordinal % 8
    edges = pq.read_table(
        S05 / f"graphs/edge_shard_{shard:02d}.parquet",
        filters=[("family_ordinal", "=", family_ordinal)],
    ).to_pandas()
    nodes = pq.read_table(
        S05 / f"graphs/node_shard_{shard:02d}.parquet",
        filters=[("family_ordinal", "=", family_ordinal)],
    ).to_pandas()
    edges.reset_index(drop=True, inplace=True)
    nodes.sort_values("state_ordinal", inplace=True)
    nodes.reset_index(drop=True, inplace=True)
    return edges, nodes


def load_s07(families: Sequence[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = pq.read_table(
        S07 / "path_solutions.parquet",
        filters=[("family_ordinal", "in", list(families))],
    ).to_pandas()
    costs = pq.read_table(
        S07 / "minimum_cost_solutions.parquet",
        filters=[("family_ordinal", "in", list(families))],
    ).to_pandas()
    return paths, costs


def metric_lookup(paths: pd.DataFrame) -> dict[tuple[int, int, int], dict[str, Any]]:
    return {
        (int(row.family_ordinal), int(row.state_ordinal), int(row.metric_code)): row._asdict()
        for row in paths.itertuples(index=False)
    }


def graph_edge_rows(
    replays: Sequence[dict[str, Any]],
    graphs: Mapping[int, tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    event_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    for replay in replays:
        task, scenario, projection = replay["task"], replay["scenario"], replay["projection"]
        family_ordinal = int(replay["family_row"]["family_ordinal"])
        edges, _ = graphs[family_ordinal]
        edge_lookup = {
            (int(row.source_state_ordinal), int(row.opportunity_ordinal)): row
            for row in edges.itertuples(index=False)
        }
        for event_index, event in enumerate(replay["events"]):
            before_native = replay["states"][event_index]
            after_native = replay["states"][event_index + 1]
            before = projection.state(before_native)
            after = projection.state(after_native)
            source = state_ordinal(before)
            successor = state_ordinal(after)
            opportunity, actor, side = event_scheduler_label(projection, event)
            edge = edge_lookup.get((source, opportunity))
            if edge is None:
                raise AssertionError("recorded opportunity has no S05 authoritative edge")
            proposal = event["proposal"]
            actor_index = (
                projection.native_to_canonical[proposal["actorId"]]
                if proposal["actorId"] in projection.native_to_canonical
                else -1
            )
            expected = {
                "successor_state_ordinal": successor,
                "scheduler_actor_index": actor,
                "scheduler_side_code": side,
                "proposal_kind_code": PROPOSAL_KIND_CODES[proposal["kind"]],
                "proposal_reason_code": REASON_CODES[proposal["reason"]],
                "decision_code": DECISION_CODES[event["decision"]],
                "proposal_actor_index": actor_index,
                "actor_position": int(proposal["actorPos"]),
                "target_position": -1 if proposal["targetPos"] is None else int(proposal["targetPos"]),
                "new_cursor": -128 if proposal["newCursor"] is None else int(proposal["newCursor"]),
                "observation_reads": int(event["ledgerDelta"]["observationReads"]),
                "value_comparisons": int(event["ledgerDelta"]["valueComparisons"]),
                "cost_no_ops": int(event["ledgerDelta"]["noOps"]),
                "cost_rejections": int(event["ledgerDelta"]["rejections"]),
                "cost_memory_updates": int(event["ledgerDelta"]["memoryUpdates"]),
                "cost_accepted_swaps": int(event["ledgerDelta"]["acceptedSwaps"]),
                "cost_displaced_cells": int(event["ledgerDelta"]["displacedCells"]),
            }
            direct_metric_deltas = {
                "delta_adjacent_descents": (
                    int(family_metric_levels(projection.family, "adjacent_descents")[successor])
                    - int(family_metric_levels(projection.family, "adjacent_descents")[source])
                ),
                "delta_inversion_count": (
                    int(family_metric_levels(projection.family, "inversion_count")[successor])
                    - int(family_metric_levels(projection.family, "inversion_count")[source])
                ),
                "delta_spearman_footrule": (
                    int(family_metric_levels(projection.family, "spearman_footrule")[successor])
                    - int(family_metric_levels(projection.family, "spearman_footrule")[source])
                ),
                "delta_maximum_rank_error": (
                    int(family_metric_levels(projection.family, "maximum_rank_error")[successor])
                    - int(family_metric_levels(projection.family, "maximum_rank_error")[source])
                ),
            }
            expected.update(direct_metric_deltas)
            mismatches = {
                key: (int(getattr(edge, key)), value)
                for key, value in expected.items()
                if int(getattr(edge, key)) != value
            }
            if mismatches:
                raise AssertionError(f"S05 edge semantic mismatch: {mismatches}")
            row = {
                "logical_trace_id": task["logical_trace_id"],
                "scenario_id": scenario.scenario_id,
                "source_step_id": task["source_step_id"],
                "family_ordinal": family_ordinal,
                "family_id": replay["family_row"]["family_id"],
                "event_index": event_index,
                "source_state_ordinal": source,
                "successor_state_ordinal": successor,
                "opportunity_ordinal": opportunity,
                "scheduler_actor_index": actor,
                "scheduler_side_code": side,
                "proposal_kind": proposal["kind"],
                "proposal_reason": proposal["reason"],
                "decision": event["decision"],
                "accepted_swap_index": int(after_native.ledger["acceptedSwaps"]),
                "native_pre_hash": event["preStateHash"],
                "native_post_hash": event["postStateHash"],
                "terminal_after": after_native.terminal,
                "structural_changed": source != successor,
                "history_to_structure_many_to_one": True,
                **{
                    key: value
                    for key, value in expected.items()
                    if key.startswith("cost_")
                    or key.startswith("delta_")
                    or key in ("observation_reads", "value_comparisons")
                },
            }
            event_rows.append(row)
            if int(event["ledgerDelta"]["acceptedSwaps"]) == 1:
                accepted_rows.append(dict(row))
    return event_rows, accepted_rows


def validate_s02_checkpoints(
    replays: Sequence[dict[str, Any]],
    accepted_rows: Sequence[dict[str, Any]],
) -> dict[str, int]:
    trace_ids = [replay["task"]["logical_trace_id"] for replay in replays]
    table = pq.read_table(
        S02 / "replayed_distances.parquet",
        filters=[("logical_trace_id", "in", trace_ids)],
    ).to_pandas()
    checkpoint_rows = table[
        (table.goal_direction_scope == "native_homogeneous_goal")
        & (table.direction == "ascending")
    ]
    checks = Counter()
    replay_by_trace = {row["task"]["logical_trace_id"]: row for row in replays}
    metric_columns = {
        "adjacent_descents": "adjacent_descents",
        "inversion_count": "inversion_count",
        "spearman_footrule": "spearman_footrule",
        "maximum_rank_error": "maximum_rank_error",
    }
    for checkpoint in checkpoint_rows.itertuples(index=False):
        replay = replay_by_trace[checkpoint.logical_trace_id]
        activation = int(checkpoint.activation_count)
        native = replay["states"][activation]
        scenario = replay["scenario"]
        projection = replay["projection"]
        expected_hash = state_hash(scenario.scenario_id, native)
        if checkpoint.native_state_hash != expected_hash:
            raise AssertionError("S02 native checkpoint hash mismatch")
        if checkpoint.ordered_values_sha256 != values_hash(native_values(scenario, native)):
            raise AssertionError("S02 ordered-value checkpoint mismatch")
        structural = projection.state(native)
        ordinal = state_ordinal(structural)
        for metric, column in metric_columns.items():
            actual = int(family_metric_levels(projection.family, metric)[ordinal])
            if int(getattr(checkpoint, column)) != actual:
                raise AssertionError(f"S02 {metric} checkpoint mismatch")
            checks[f"metric_{metric}"] += 1
        checks["native_hash"] += 1
        checks["ordered_values"] += 1
    checkpoint_keys = {
        (row.logical_trace_id, int(row.activation_count), int(row.accepted_swap_index))
        for row in checkpoint_rows.itertuples(index=False)
        if row.checkpoint_kind == "accepted_action"
    }
    for row in accepted_rows:
        key = (row["logical_trace_id"], row["event_index"] + 1, row["accepted_swap_index"])
        if key not in checkpoint_keys:
            raise AssertionError("accepted S05 projection has no exact S02 accepted-action checkpoint")
        checks["accepted_action_projection"] += 1
    return dict(checks)


def cumulative_suffix_costs(events: Sequence[Mapping[str, Any]]) -> list[dict[str, int]]:
    names = (
        "activations",
        "observation_reads",
        "value_comparisons",
        "cost_no_ops",
        "cost_rejections",
        "cost_memory_updates",
        "cost_accepted_swaps",
        "cost_displaced_cells",
    )
    result = [{name: 0 for name in names} for _ in range(len(events) + 1)]
    for index in range(len(events) - 1, -1, -1):
        result[index] = dict(result[index + 1])
        result[index]["activations"] += 1
        for name in names[1:]:
            result[index][name] += int(events[index][name])
    return result


def s07_primary_solution(path_family_metric: pd.DataFrame) -> PrimarySolution:
    rows = path_family_metric.sort_values("state_ordinal")
    if list(rows.state_ordinal) != list(range(len(rows))):
        raise AssertionError("S07 family/metric rows are not a complete state sequence")
    peak = rows.minimum_peak.to_numpy(dtype=np.int64)
    excursion = rows.minimum_excursion.to_numpy(dtype=np.int64)
    peak[peak < 0] = UNREACHABLE
    excursion[excursion < 0] = UNREACHABLE
    return PrimarySolution(
        bottleneck_levels=peak,
        excursion_levels=excursion,
        successor_edges=rows.primary_successor_edge_local_index.to_numpy(dtype=np.int64),
        classifications=rows.classification_code.to_numpy(dtype=np.uint8),
    )


def global_cost_vectors(row: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    return {
        "activations": (int(row["shortest_activations"]),),
        "observation_reads": (
            int(row["minimum_observation_reads"]),
            int(row["minimum_observation_reads_activations"]),
        ),
        "value_comparisons": (
            int(row["minimum_value_comparisons"]),
            int(row["minimum_value_comparisons_activations"]),
        ),
        "accepted_swaps": (
            int(row["minimum_accepted_swaps"]),
            int(row["minimum_accepted_swaps_activations"]),
        ),
        "displaced_cells": (
            int(row["minimum_displaced_cells"]),
            int(row["minimum_displaced_cells_activations"]),
        ),
        "full_ledger": tuple(
            int(row[f"minimum_full_ledger_{name}"])
            for name in (
                "activations", "observation_reads", "value_comparisons", "no_ops",
                "rejections", "memory_updates", "accepted_swaps", "displaced_cells",
            )
        ),
    }


def observed_cost_vector(cost: Mapping[str, int], profile: str) -> tuple[int, ...]:
    return tuple(int(cost[name]) for name in COST_PROFILES[profile])


def activation_gap(
    profile: str,
    observed_vector: Sequence[int],
    optimum_vector: Sequence[int],
) -> int:
    """Return the activation coordinate gap for one frozen cost profile."""

    coordinate = 0 if profile in ("activations", "full_ledger") else 1
    return int(observed_vector[coordinate]) - int(optimum_vector[coordinate])


def build_comparisons(
    replays: Sequence[dict[str, Any]],
    event_rows: Sequence[dict[str, Any]],
    graphs: Mapping[int, tuple[pd.DataFrame, pd.DataFrame]],
    paths: pd.DataFrame,
    costs: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    path_lookup = metric_lookup(paths)
    cost_lookup = {
        (int(row.family_ordinal), int(row.state_ordinal)): row._asdict()
        for row in costs.itertuples(index=False)
    }
    event_by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        event_by_trace[row["logical_trace_id"]].append(row)
    comparisons: list[dict[str, Any]] = []
    efficiency: list[dict[str, Any]] = []
    panels: list[dict[str, Any]] = []
    validation = Counter()

    family_context: dict[tuple[int, str], tuple[np.ndarray, PrimarySolution]] = {}
    for replay in replays:
        family_ordinal = int(replay["family_row"]["family_ordinal"])
        family: FamilySpec = replay["projection"].family
        for metric in METRICS:
            selected = paths[
                (paths.family_ordinal == family_ordinal)
                & (paths.metric_code == METRIC_CODES[metric])
            ]
            family_context[(family_ordinal, metric)] = (
                family_metric_levels(family, metric),
                s07_primary_solution(selected),
            )

    # Validate the entire encountered successor forest, once per unique start.
    witnessed: set[tuple[int, int, str]] = set()
    witness_paths: dict[tuple[int, int, str], tuple[list[int], list[int]]] = {}
    for replay in replays:
        trace_id = replay["task"]["logical_trace_id"]
        family_ordinal = int(replay["family_row"]["family_ordinal"])
        edges, nodes = graphs[family_ordinal]
        terminal = nodes.terminal_code.to_numpy(dtype=np.uint8)
        for native in replay["states"]:
            start = state_ordinal(replay["projection"].state(native))
            for metric in METRICS:
                key = (family_ordinal, start, metric)
                if key in witnessed:
                    continue
                witnessed.add(key)
                row = path_lookup[(family_ordinal, start, METRIC_CODES[metric])]
                if int(row["minimum_peak"]) < 0:
                    continue
                levels, solution = family_context[(family_ordinal, metric)]
                current = start
                path_nodes = [current]
                path_edges: list[int] = []
                seen: set[int] = set()
                while terminal[current] != 1:
                    if current in seen:
                        raise AssertionError("S07 primary witness contains a cycle")
                    seen.add(current)
                    edge_index = int(solution.successor_edges[current])
                    edge = edges.iloc[edge_index]
                    if int(edge.source_state_ordinal) != current:
                        raise AssertionError("S07 successor edge source mismatch")
                    path_edges.append(edge_index)
                    current = int(edge.successor_state_ordinal)
                    path_nodes.append(current)
                if max(int(levels[node]) for node in path_nodes) != int(row["minimum_peak"]):
                    raise AssertionError("S07 witness peak mismatch")
                witness_paths[key] = (path_nodes, path_edges)
                validation["s07_witness_replay"] += 1

    for replay in replays:
        task = replay["task"]
        trace_id = task["logical_trace_id"]
        scenario = replay["scenario"]
        projection = replay["projection"]
        family = projection.family
        family_ordinal = int(replay["family_row"]["family_ordinal"])
        edges, nodes = graphs[family_ordinal]
        trace_events = sorted(event_by_trace[trace_id], key=lambda row: row["event_index"])
        suffix_costs = cumulative_suffix_costs(trace_events)
        structural_states = [state_ordinal(projection.state(state)) for state in replay["states"]]
        successful = replay["states"][-1].terminal == "complete"
        for checkpoint, (native, start) in enumerate(zip(replay["states"], structural_states)):
            actual_terminal = native_terminal_code(native.terminal)
            if int(nodes.iloc[start].terminal_code) != actual_terminal:
                raise AssertionError("native/S05 terminal-state disagreement")
            cost_row = cost_lookup[(family_ordinal, start)]
            obs_cost = suffix_costs[checkpoint]
            global_optima = global_cost_vectors(cost_row)
            for profile in COST_PROFILES:
                observed_vector = observed_cost_vector(obs_cost, profile)
                optimum_vector = global_optima[profile]
                reachable = optimum_vector[0] >= 0
                comparable = successful and reachable
                efficiency.append(
                    {
                        "logical_trace_id": trace_id,
                        "scenario_id": scenario.scenario_id,
                        "family_ordinal": family_ordinal,
                        "checkpoint_activation": checkpoint,
                        "state_ordinal": start,
                        "comparison_scope": "unconstrained_global_s07",
                        "metric": None,
                        "cost_profile": profile,
                        "observed_success": successful,
                        "observed_cost_vector_json": json.dumps(observed_vector),
                        "optimum_cost_vector_json": json.dumps(optimum_vector) if reachable else None,
                        "lexicographic_relation": (
                            lexicographic_relation(observed_vector, optimum_vector) if comparable else None
                        ),
                        "activation_gap": (
                            activation_gap(profile, observed_vector, optimum_vector)
                            if comparable else None
                        ),
                        "comparable": comparable,
                        "comparison_reason": (
                            "successful_suffix" if comparable else
                            ("observed_suffix_failed" if not successful else "structurally_unreachable")
                        ),
                    }
                )
            for metric in METRICS:
                metric_row = path_lookup[(family_ordinal, start, METRIC_CODES[metric])]
                levels, solution = family_context[(family_ordinal, metric)]
                observed_levels = [int(levels[node]) for node in structural_states[checkpoint:]]
                observed_peak = max(observed_levels)
                observed_excursion = observed_peak - observed_levels[0]
                solver_class = int(metric_row["classification_code"])
                behavior = classify_observed_suffix(
                    solver_classification=solver_class,
                    observed_success=successful,
                    observed_excursion=observed_excursion,
                )
                if behavior.startswith(CLASS_CONTRADICTION_PREFIX):
                    raise AssertionError(f"observed/solver semantic contradiction: {behavior}")
                minimum_peak = int(metric_row["minimum_peak"])
                minimum_excursion = int(metric_row["minimum_excursion"])
                reachable = minimum_peak >= 0
                denominator = {
                    "adjacent_descents": family.n - 1,
                    "inversion_count": family.n * (family.n - 1) // 2,
                    "spearman_footrule": family.n * family.n // 2,
                    "maximum_rank_error": family.n - 1,
                }[metric]
                comparison = {
                    "logical_trace_id": trace_id,
                    "scenario_id": scenario.scenario_id,
                    "source_step_id": task["source_step_id"],
                    "family_ordinal": family_ordinal,
                    "family_id": replay["family_row"]["family_id"],
                    "n": family.n,
                    "architecture": family.architecture.value,
                    "policy_profile": replay["family_row"]["policy_profile"],
                    "fault_mode": replay["family_row"]["fault_mode"],
                    "checkpoint_activation": checkpoint,
                    "checkpoint_kind": (
                        "initial" if checkpoint == 0 else
                        ("terminal" if checkpoint == len(structural_states) - 1 else "post_activation")
                    ),
                    "state_ordinal": start,
                    "native_state_hash": state_hash(scenario.scenario_id, native),
                    "metric": metric,
                    "metric_denominator": denominator,
                    "start_distance": observed_levels[0],
                    "observed_peak": observed_peak,
                    "observed_excursion": observed_excursion,
                    "minimum_peak": minimum_peak if reachable else None,
                    "minimum_excursion": minimum_excursion if reachable else None,
                    "excess_peak_depth": observed_peak - minimum_peak if successful and reachable else None,
                    "normalized_observed_excursion": observed_excursion / denominator,
                    "normalized_minimum_excursion": minimum_excursion / denominator if reachable else None,
                    "normalized_excess_peak_depth": (
                        (observed_peak - minimum_peak) / denominator if successful and reachable else None
                    ),
                    "solver_classification": CLASS_LABELS[solver_class],
                    "observed_success": successful,
                    "observed_stop_reason": replay["states"][-1].terminal,
                    "behavior_classification": behavior,
                    "observed_reached_minimum_peak": observed_peak >= minimum_peak if reachable else None,
                    "required_depth_shortfall_on_failure": (
                        max(0, minimum_peak - observed_peak)
                        if (not successful and solver_class == 2)
                        else None
                    ),
                    "scheduler_quantification": "existential_finite_S05_structural_opportunity_path",
                    "byte_exact_scheduler_claim": False,
                }
                comparisons.append(comparison)
                validation["exact_s07_join"] += 1

                # Start-conditioned secondary optima are meaningful only for an observed success.
                if successful and reachable:
                    edge_columns = {name: edges[name].to_numpy() for name in set(sum((list(v) for v in COST_PROFILES.values()), [])) if name != "activations"}
                    for profile in COST_PROFILES:
                        cost_matrix = cost_matrix_from_s05(edge_columns, profile)
                        witness = solve_lexicographic_witness(
                            start,
                            family.state_count,
                            edges.source_state_ordinal.to_numpy(),
                            edges.successor_state_ordinal.to_numpy(),
                            levels,
                            nodes.terminal_code.to_numpy(),
                            cost_matrix,
                            primary_solution=solution,
                        )
                        observed_vector = observed_cost_vector(obs_cost, profile)
                        within_threshold = observed_peak == minimum_peak
                        efficiency.append(
                            {
                                "logical_trace_id": trace_id,
                                "scenario_id": scenario.scenario_id,
                                "family_ordinal": family_ordinal,
                                "checkpoint_activation": checkpoint,
                                "state_ordinal": start,
                                "comparison_scope": "primary_preserving_s06_secondary",
                                "metric": metric,
                                "cost_profile": profile,
                                "observed_success": True,
                                "observed_cost_vector_json": json.dumps(observed_vector),
                                "optimum_cost_vector_json": json.dumps(witness.cost),
                                "lexicographic_relation": (
                                    lexicographic_relation(observed_vector, witness.cost)
                                    if within_threshold else None
                                ),
                                "activation_gap": (
                                    activation_gap(profile, observed_vector, witness.cost)
                                    if within_threshold else None
                                ),
                                "comparable": within_threshold,
                                "comparison_reason": (
                                    "observed_suffix_respects_primary_threshold"
                                    if within_threshold else "observed_suffix_exceeds_primary_threshold"
                                ),
                            }
                        )
                        validation["secondary_optimum_replay"] += 1

        # Panel: initial checkpoint for every metric, retaining observed and exact primary paths.
        start = structural_states[0]
        for metric in METRICS:
            levels, _ = family_context[(family_ordinal, metric)]
            panel_id = f"{trace_id}:{metric}:initial"
            for step, node in enumerate(structural_states):
                panels.append(
                    {
                        "panel_id": panel_id,
                        "logical_trace_id": trace_id,
                        "metric": metric,
                        "path_kind": "observed_suffix",
                        "step_index": step,
                        "state_ordinal": node,
                        "metric_level": int(levels[node]),
                        "edge_local_index": None if step == len(structural_states) - 1 else int(
                            next(
                                index
                                for index, row in edges.iterrows()
                                if int(row.source_state_ordinal) == node
                                and int(row.successor_state_ordinal) == structural_states[step + 1]
                                and int(row.opportunity_ordinal) == trace_events[step]["opportunity_ordinal"]
                            )
                        ),
                    }
                )
            witness_key = (family_ordinal, start, metric)
            if witness_key in witness_paths:
                witness_nodes, witness_edges = witness_paths[witness_key]
                for step, node in enumerate(witness_nodes):
                    panels.append(
                        {
                            "panel_id": panel_id,
                            "logical_trace_id": trace_id,
                            "metric": metric,
                            "path_kind": "s07_primary_optimum",
                            "step_index": step,
                            "state_ordinal": node,
                            "metric_level": int(levels[node]),
                            "edge_local_index": None if step == len(witness_nodes) - 1 else witness_edges[step],
                        }
                    )
    return comparisons, efficiency, panels, dict(validation)


def plot_panels(panel_rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    frame = pd.DataFrame(panel_rows)
    panel_ids = list(dict.fromkeys(frame.panel_id))
    # Five exact trajectories x four metrics.  A 5x4 grid keeps every initial witness visible.
    fig, axes = plt.subplots(5, 4, figsize=(16, 14), constrained_layout=True)
    for ax, panel_id in zip(axes.flat, panel_ids):
        panel = frame[frame.panel_id == panel_id]
        for kind, style in (("observed_suffix", "o-"), ("s07_primary_optimum", "x--")):
            selected = panel[panel.path_kind == kind]
            if not selected.empty:
                ax.plot(selected.step_index, selected.metric_level, style, label=kind, linewidth=1.2, markersize=3)
        trace = panel.logical_trace_id.iloc[0]
        metric = panel.metric.iloc[0]
        ax.set_title(f"{trace[5:13]} · {metric.replace('_', ' ')}", fontsize=8)
        ax.set_xlabel("opportunity step")
        ax.set_ylabel("exact distance")
        ax.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2)
    fig.suptitle("S08 observed suffixes versus existential S07 primary witnesses", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    start_time = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inputs = input_paths()
    pre_hashes = {str(path): sha256_file(path) for path in inputs}
    _, family_lookup = load_family_context()
    coverage, eligible = coverage_rows(family_lookup)
    replays = replay_eligible(eligible)
    family_ordinals = sorted({int(row["family_row"]["family_ordinal"]) for row in replays})
    graphs = {ordinal: load_graph(ordinal) for ordinal in family_ordinals}
    paths, costs = load_s07(family_ordinals)
    event_rows, accepted_rows = graph_edge_rows(replays, graphs)
    s02_checks = validate_s02_checkpoints(replays, accepted_rows)
    comparisons, efficiency, panels, comparison_checks = build_comparisons(
        replays, event_rows, graphs, paths, costs
    )
    contradictions = [
        row for row in comparisons
        if str(row["behavior_classification"]).startswith(CLASS_CONTRADICTION_PREFIX)
    ]
    if contradictions:
        raise AssertionError("semantic contradictions remain in final comparison")

    write_parquet(OUTPUT / "trajectory_coverage.parquet", coverage, "e03.s08.trajectory_coverage.v1")
    write_parquet(OUTPUT / "observed_structural_events.parquet", event_rows, "e03.s08.observed_structural_event.v1")
    write_parquet(OUTPUT / "accepted_action_projections.parquet", accepted_rows, "e03.s08.accepted_action_projection.v1")
    write_parquet(OUTPUT / "behavior_necessity_comparison.parquet", comparisons, "e03.s08.behavior_necessity_comparison.v1")
    write_parquet(OUTPUT / "efficiency_gaps.parquet", efficiency, "e03.s08.efficiency_gap.v1")
    write_parquet(OUTPUT / "witness_panels.parquet", panels, "e03.s08.witness_panel.v1")
    plot_panels(panels, OUTPUT / "witness_panels.png")

    comparison_frame = pd.DataFrame(comparisons)
    summary = (
        comparison_frame.groupby(["metric", "behavior_classification", "observed_success"], dropna=False)
        .size().rename("checkpoint_count").reset_index()
    )
    write_parquet(
        OUTPUT / "classification_summary.parquet",
        summary.to_dict("records"),
        "e03.s08.classification_summary.v1",
    )
    trace_summary = (
        comparison_frame[comparison_frame.checkpoint_activation == 0]
        .groupby(["logical_trace_id", "scenario_id", "source_step_id", "metric", "behavior_classification", "solver_classification", "observed_stop_reason"], dropna=False)
        .agg(
            start_distance=("start_distance", "first"),
            observed_peak=("observed_peak", "first"),
            observed_excursion=("observed_excursion", "first"),
            minimum_peak=("minimum_peak", "first"),
            minimum_excursion=("minimum_excursion", "first"),
            excess_peak_depth=("excess_peak_depth", "first"),
        ).reset_index()
    )
    write_parquet(
        OUTPUT / "trajectory_start_summary.parquet",
        trace_summary.to_dict("records"),
        "e03.s08.trajectory_start_summary.v1",
    )
    post_hashes = {str(path): sha256_file(path) for path in inputs}
    immutable = pre_hashes == post_hashes
    if not immutable:
        raise AssertionError("one or more frozen S08 inputs changed")

    coverage_counts = Counter(row["coverage_reason"] for row in coverage)
    behavior_counts = Counter(row["behavior_classification"] for row in comparisons)
    validation = {
        "schemaVersion": "e03.s08.validation_results.v1",
        "researchStepId": "S08",
        "success": True,
        "checks": {
            "primary_trace_accounting": sum(row["primary_denominator"] for row in coverage) == 90,
            "supplemental_trace_accounting": sum(row["supplemental"] for row in coverage) == 1,
            "exact_eligible_trace_count": len(replays) == 5,
            "all_recorded_opportunities_mapped": len(event_rows) == sum(len(row["events"]) for row in replays),
            "accepted_action_count": len(accepted_rows) == 17,
            "comparison_rows_complete": len(comparisons) == sum(len(row["states"]) * 4 for row in replays),
            "s07_joins_complete": comparison_checks["exact_s07_join"] == len(comparisons),
            "classification_partition_complete": sum(behavior_counts.values()) == len(comparisons),
            "semantic_contradictions_absent": not contradictions,
            "input_immutability": immutable,
        },
        "counts": {
            "coverageRows": len(coverage),
            "exactTrajectories": len(replays),
            "mappedEvents": len(event_rows),
            "acceptedActions": len(accepted_rows),
            "comparisonRows": len(comparisons),
            "efficiencyRows": len(efficiency),
            "witnessPanelRows": len(panels),
            **{f"s02_{key}": value for key, value in s02_checks.items()},
            **comparison_checks,
        },
        "coverageReasonCounts": dict(sorted(coverage_counts.items())),
        "behaviorCounts": dict(sorted(behavior_counts.items())),
    }
    if not all(validation["checks"].values()):
        raise AssertionError(f"validation gate failed: {validation['checks']}")
    write_json(OUTPUT / "validation_results.json", validation)
    write_json(
        OUTPUT / "input_immutability.json",
        {
            "schemaVersion": "e03.s08.input_immutability.v1",
            "researchStepId": "S08",
            "success": immutable,
            "preRunSha256": pre_hashes,
            "postRunSha256": post_hashes,
        },
    )
    result_summary = {
        "schemaVersion": "e03.s08.result_summary.v1",
        "researchStepId": "S08",
        "success": True,
        "primaryTraceCount": 90,
        "exactEligibleTraceCount": len(replays),
        "exactEligibleFraction": len(replays) / 90,
        "mappedEventCount": len(event_rows),
        "comparisonRowCount": len(comparisons),
        "coverageReasonCounts": dict(sorted(coverage_counts.items())),
        "behaviorCounts": dict(sorted(behavior_counts.items())),
        "semanticContradictionCount": 0,
        "elapsedSeconds": time.perf_counter() - start_time,
    }
    write_json(OUTPUT / "result_summary.json", result_summary)
    write_json(
        OUTPUT / "environment.json",
        {
            "schemaVersion": "e03.s08.environment.v1",
            "python": sys.version,
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "matplotlib": plt.matplotlib.__version__,
            "workerCount": 1,
            "threadEnvironment": {
                key: os.environ.get(key)
                for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            },
        },
    )
    write_json(
        OUTPUT / "provenance_manifest.json",
        {
            "schemaVersion": "e03.s08.provenance_manifest.v1",
            "researchStepId": "S08",
            "repositoryCommitAtRunStart": os.popen("git rev-parse HEAD").read().strip(),
            "inputs": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(pre_hashes.items())
            ],
            "command": "OPENSSL_FORCE_FIPS_MODE=0 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python scripts/build_s08_comparison.py",
            "schedulerClaim": "existential structural opportunities only; native replay remains byte-exact history",
        },
    )
    artifacts = []
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    write_json(
        OUTPUT / "artifact_manifest.json",
        {
            "schemaVersion": "e03.s08.artifact_manifest.v1",
            "researchStepId": "S08",
            "artifacts": artifacts,
        },
    )
    print(json.dumps(result_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
