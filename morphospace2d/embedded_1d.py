"""Embed the E01 one-dimensional cell-view row in an E05 2D substrate.

S06 is a baseline-preservation check.  The runner below keeps the E01
cell-view scheduler and cell classes intact, but places every cell on a chosen
row of an S01 square-grid substrate.  Policies therefore see the same row
semantics as E01 while row-position validation verifies that the 2D substrate
does not introduce off-row moves.
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from e02_deterministic_simulator.metrics import (
    aggregation,
    monotonicity_error,
    sortedness_percent,
    sortedness_raw,
    state_hash,
)
from modules.multithread.BubbleSortCell import BubbleSortCell
from modules.multithread.CellGroup import CellGroup, GroupStatus
from modules.multithread.InsertionSortCell import InsertionSortCell
from modules.multithread.SelectionSortCell import SelectionSortCell
from modules.multithread.StatusProbe import StatusProbe

from .identities import (
    IDENTITY_SCHEMA_VERSION,
    default_identity_schema,
    identities_from_scalar_values,
    identity_to_cell_state,
)
from .substrates import SUBSTRATE_SCHEMA_VERSION, Substrate


EMBEDDED_1D_SCHEMA_VERSION = "e05_s06_embedded_1d.v1"
EMBEDDED_TRACE_SCHEMA_VERSION = "e05_s06_embedded_trace.v1"
CLASSIC_ALGORITHMS = ("bubble", "insertion", "selection")
CELL_CLASSES = {
    "bubble": BubbleSortCell,
    "insertion": InsertionSortCell,
    "selection": SelectionSortCell,
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _compact_json(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def values_from_cells(cells: Sequence[Any]) -> list[int]:
    """Return row values in x-position order."""

    return [int(cell.value) for cell in cells]


def positions_from_cells(cells: Sequence[Any]) -> list[tuple[int, int]]:
    """Return current 2D positions in cell-array order."""

    return [(int(cell.current_position[0]), int(cell.current_position[1])) for cell in cells]


def algotypes_from_cells(cells: Sequence[Any], algotype_by_cell_id: Mapping[int, str]) -> list[str]:
    """Return position-ordered Algotype labels."""

    return [str(algotype_by_cell_id[int(cell.threadID)]) for cell in cells]


def is_sorted_values(values: Sequence[int]) -> bool:
    return all(int(left) <= int(right) for left, right in zip(values, values[1:]))


def row_position_errors(cells: Sequence[Any], substrate: Substrate, row_y: int) -> list[str]:
    """Validate that all live cells occupy unique nodes on the embedded row."""

    errors: list[str] = []
    positions = positions_from_cells(cells)
    node_set = set(substrate.nodes)
    if len(positions) != len(set(positions)):
        errors.append("duplicate 2D cell positions")
    for index, position in enumerate(positions):
        if position not in node_set:
            errors.append(f"cell array slot {index} position {position} is not a substrate node")
        if position[1] != int(row_y):
            errors.append(f"cell array slot {index} moved off row_y={row_y}: {position}")
        if position[0] != index:
            errors.append(f"cell array slot {index} has x-position {position[0]}")
    return errors


def dedup_consecutive(values: Sequence[float], tolerance: float = 1e-12) -> list[float]:
    deduped: list[float] = []
    for raw_value in values:
        value = float(raw_value)
        if not deduped or not math.isclose(value, deduped[-1], rel_tol=0.0, abs_tol=tolerance):
            deduped.append(value)
    return deduped


def signed_segments(values: Sequence[float], tolerance: float = 1e-12) -> list[float]:
    """Collapse a Sortedness trajectory into signed monotone segments."""

    segments: list[float] = []
    deduped = dedup_consecutive(values, tolerance=tolerance)
    for previous, current in zip(deduped, deduped[1:]):
        delta = float(current - previous)
        if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=tolerance):
            continue
        if segments and segments[-1] * delta > 0:
            segments[-1] += delta
        else:
            segments.append(delta)
    return segments


def delayed_gratification_from_sortedness(values: Sequence[float]) -> dict[str, Any]:
    """Compute E01's unit-tested Delayed Gratification proxy."""

    segments = signed_segments(values)
    event_scores: list[float] = []
    drops: list[float] = []
    recoveries: list[float] = []
    event_segment_indices: list[int] = []

    index = 0
    while index < len(segments) and segments[index] >= 0:
        index += 1

    while index < len(segments) - 1:
        drop_segment = segments[index]
        recovery_segment = segments[index + 1]
        if drop_segment < 0 and recovery_segment > 0:
            drop = -float(drop_segment)
            recovery = float(recovery_segment)
            event_scores.append((recovery - drop) / drop if drop else 0.0)
            drops.append(drop)
            recoveries.append(recovery)
            event_segment_indices.append(index)
            index += 2
        else:
            index += 1

    return {
        "delayedGratification": float(np.mean(event_scores)) if event_scores else 0.0,
        "dgEventCount": int(len(event_scores)),
        "dgPositiveEventCount": int(sum(1 for score in event_scores if score > 0)),
        "dgNegativeEventCount": int(sum(1 for score in event_scores if score < 0)),
        "dgZeroEventCount": int(sum(1 for score in event_scores if math.isclose(score, 0.0, abs_tol=1e-12))),
        "dgTotalDrop": float(sum(drops)),
        "dgTotalRecovery": float(sum(recoveries)),
        "dgMeanDrop": float(np.mean(drops)) if drops else 0.0,
        "dgMeanRecovery": float(np.mean(recoveries)) if recoveries else 0.0,
        "dgMaxEventScore": float(max(event_scores)) if event_scores else 0.0,
        "dgMinEventScore": float(min(event_scores)) if event_scores else 0.0,
        "dgSignedEventScoresJson": json.dumps([round(score, 12) for score in event_scores], separators=(",", ":")),
        "dgSignedSegmentsJson": json.dumps([round(segment, 12) for segment in segments], separators=(",", ":")),
        "dgEventSegmentIndicesJson": json.dumps(event_segment_indices, separators=(",", ":")),
    }


@dataclass(frozen=True)
class EmbeddedRowRunResult:
    """Result record for one E01-style run embedded in a 2D row."""

    schema_version: str
    condition_id: str
    algorithm: str
    substrate: Substrate
    row_y: int
    initial_values: list[int]
    final_values: list[int]
    initial_algotypes: list[str]
    final_algotypes: list[str]
    completed: bool
    stop_reason: str
    swap_count: int
    comparison_count: int
    archived_compare_and_swap_count: int
    event_count: int
    scheduler_rounds: int
    final_sortedness_percent: float
    final_monotonicity_error: int
    final_aggregation: float
    delayed_gratification: dict[str, Any]
    trace_rows: list[dict[str, Any]]
    wall_time_seconds: float
    row_validation_errors: tuple[str, ...] = ()
    identity_schema_version: str = IDENTITY_SCHEMA_VERSION
    substrate_schema_version: str = SUBSTRATE_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def row_restricted(self) -> bool:
        return not self.row_validation_errors and all(bool(row.get("row_restricted", False)) for row in self.trace_rows)

    def summary_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "condition_id": self.condition_id,
            "implementation": "embedded_2d_row_e01_cell_view",
            "algorithm": self.algorithm,
            "substrate_type": self.substrate.substrate_type,
            "substrate_schema_version": self.substrate_schema_version,
            "identity_schema_version": self.identity_schema_version,
            "grid_width": int(self.substrate.metadata.get("width", len(self.initial_values))),
            "grid_height": int(self.substrate.metadata.get("height", 1)),
            "row_y": int(self.row_y),
            "n": len(self.initial_values),
            "completed": bool(self.completed),
            "stop_reason": self.stop_reason,
            "row_restricted": bool(self.row_restricted),
            "row_validation_error_count": len(self.row_validation_errors),
            "row_validation_errors_json": _compact_json(list(self.row_validation_errors)),
            "swap_count": int(self.swap_count),
            "comparison_count": int(self.comparison_count),
            "archived_compare_and_swap_count": int(self.archived_compare_and_swap_count),
            "event_count": int(self.event_count),
            "scheduler_rounds": int(self.scheduler_rounds),
            "initial_state_hash": state_hash(self.initial_values),
            "final_state_hash": state_hash(self.final_values),
            "final_sortedness_percent": float(self.final_sortedness_percent),
            "final_monotonicity_error": int(self.final_monotonicity_error),
            "final_aggregation": float(self.final_aggregation),
            "delayed_gratification": float(self.delayed_gratification["delayedGratification"]),
            "dg_event_count": int(self.delayed_gratification["dgEventCount"]),
            "dg_total_drop": float(self.delayed_gratification["dgTotalDrop"]),
            "dg_total_recovery": float(self.delayed_gratification["dgTotalRecovery"]),
            "values_conserved": Counter(self.initial_values) == Counter(self.final_values),
            "algotypes_conserved": Counter(self.initial_algotypes) == Counter(self.final_algotypes),
            "initial_values_json": _compact_json(self.initial_values),
            "final_values_json": _compact_json(self.final_values),
            "initial_algotypes_json": _compact_json(self.initial_algotypes),
            "final_algotypes_json": _compact_json(self.final_algotypes),
            "wall_time_seconds": float(self.wall_time_seconds),
        }


def _trace_row(
    *,
    condition_id: str,
    algorithm: str,
    substrate: Substrate,
    row_y: int,
    cells: Sequence[Any],
    algotype_by_cell_id: Mapping[int, str],
    event_index: int,
    event_kind: str,
    swap_count: int,
    comparison_count: int,
    archived_compare_and_swap_count: int,
    scheduler_round: int,
    actor_cell_id: int | None = None,
    actor_algotype: str | None = None,
    state_values: Sequence[int] | None = None,
) -> dict[str, Any]:
    values = [int(value) for value in (state_values if state_values is not None else values_from_cells(cells))]
    raw = sortedness_raw(values)
    position_errors = row_position_errors(cells, substrate, row_y)
    position_algotypes = algotypes_from_cells(cells, algotype_by_cell_id)
    positions = positions_from_cells(cells)
    return {
        "schema_version": EMBEDDED_TRACE_SCHEMA_VERSION,
        "condition_id": condition_id,
        "implementation": "embedded_2d_row_e01_cell_view",
        "algorithm": algorithm,
        "event_index": int(event_index),
        "event_kind": event_kind,
        "swap_count": int(swap_count),
        "comparison_count": int(comparison_count),
        "archived_compare_and_swap_count": int(archived_compare_and_swap_count),
        "scheduler_round": int(scheduler_round),
        "sortedness_raw_count": int(raw),
        "sortedness_percent": sortedness_percent(values),
        "monotonicity_error": monotonicity_error(values),
        "state_hash": state_hash(values),
        "algotype_sequence_json": _compact_json(position_algotypes),
        "positions_json": _compact_json(positions),
        "actor_cell_id": actor_cell_id,
        "actor_algotype": actor_algotype,
        "row_y": int(row_y),
        "row_restricted": len(position_errors) == 0,
        "row_validation_error_count": len(position_errors),
        "row_validation_errors_json": _compact_json(position_errors),
        "substrate_type": substrate.substrate_type,
        "grid_width": int(substrate.metadata.get("width", len(values))),
        "grid_height": int(substrate.metadata.get("height", 1)),
    }


def _make_cells(
    *,
    algorithm: str,
    initial_values: Sequence[int],
    row_y: int,
    status_probe: StatusProbe,
    swapping_count: list[int],
    export_steps: list[list[int]],
    lock: threading.Lock,
) -> list[Any]:
    cell_class = CELL_CLASSES[algorithm]
    cells: list[Any] = [None] * len(initial_values)
    for index, value in enumerate(initial_values):
        cells[index] = cell_class(
            index,
            int(value),
            lock,
            (index, row_y),
            cells,
            (0, row_y),
            (len(initial_values) - 1, row_y),
            status_probe,
            disable_visualization=True,
            swapping_count=swapping_count,
            export_steps=export_steps,
            label=0,
            reverse_direction=False,
        )
    return cells


def run_embedded_row(
    algorithm: str,
    initial_values: Sequence[int],
    *,
    scheduler_seed: int,
    tie_breaker_seed: int,
    row_y: int = 1,
    height: int = 3,
    max_swaps: int = 500_000,
    max_rounds: int = 100_000,
    condition_id: str = "manual",
) -> EmbeddedRowRunResult:
    """Run one E01 cell-view policy on an embedded 2D row.

    The grid may contain off-row nodes, but occupancy starts and must remain on
    `row_y`.  Selection retains E01's same-row ideal-position exchange, which
    can be non-adjacent; this is intentional baseline preservation.
    """

    algorithm = str(algorithm)
    if algorithm not in CELL_CLASSES:
        raise ValueError(f"unknown classic algorithm: {algorithm!r}")
    values = [int(value) for value in initial_values]
    if not values:
        raise ValueError("initial_values must not be empty")
    if height < 1:
        raise ValueError("height must be positive")
    if row_y < 0 or row_y >= height:
        raise ValueError("row_y must be inside the grid height")

    substrate = Substrate.square_grid(width=len(values), height=height)
    identity_schema = default_identity_schema()
    identities = identities_from_scalar_values(values)
    cell_states = [identity_to_cell_state(identity, identity_schema, include_hidden=False) for identity in identities]
    metadata = {
        "cellStateCount": len(cell_states),
        "selectionTargetScope": "same_row_e01_ideal_position",
        "rowRestrictionMode": "x_axis_only_e01_cell_view",
        "offRowNodeCount": len(substrate.nodes) - len(values),
    }

    start = time.perf_counter()
    lock = threading.Lock()
    status_probe = StatusProbe()
    swapping_count = [0]
    export_steps: list[list[int]] = []
    random.seed(int(tie_breaker_seed))
    cells = _make_cells(
        algorithm=algorithm,
        initial_values=values,
        row_y=row_y,
        status_probe=status_probe,
        swapping_count=swapping_count,
        export_steps=export_steps,
        lock=lock,
    )
    group = CellGroup(
        cells,
        cells,
        0,
        (0, row_y),
        (len(values) - 1, row_y),
        GroupStatus.ACTIVE,
        lock,
        count_down=1,
        phase_period=1,
    )
    for cell in cells:
        cell.group = group

    cell_objects = list(cells)
    algotype_by_cell_id = {int(cell.threadID): algorithm for cell in cells}
    trace_rows = [
        _trace_row(
            condition_id=condition_id,
            algorithm=algorithm,
            substrate=substrate,
            row_y=row_y,
            cells=cells,
            algotype_by_cell_id=algotype_by_cell_id,
            event_index=0,
            event_kind="initial",
            swap_count=0,
            comparison_count=0,
            archived_compare_and_swap_count=0,
            scheduler_round=0,
        )
    ]

    scheduler_rng = np.random.default_rng(int(scheduler_seed))
    scheduler_rounds = 0
    no_progress_rounds = 0
    stop_reason = "sorted"

    while not is_sorted_values(values_from_cells(cells)):
        if status_probe.swap_count >= int(max_swaps):
            stop_reason = "max_step_cap"
            break
        if scheduler_rounds >= int(max_rounds):
            stop_reason = "max_round_cap"
            break
        swaps_before_round = status_probe.swap_count
        order = scheduler_rng.permutation(len(cell_objects))
        for cell_index in order:
            actor = cell_objects[int(cell_index)]
            actor_cell_id = int(actor.threadID)
            swaps_before_actor = status_probe.swap_count
            before_len = len(status_probe.sorting_steps)
            actor.move()
            if len(status_probe.sorting_steps) > before_len:
                for event_number, snapshot in zip(
                    range(swaps_before_actor + 1, status_probe.swap_count + 1),
                    status_probe.sorting_steps[before_len:],
                    strict=False,
                ):
                    trace_rows.append(
                        _trace_row(
                            condition_id=condition_id,
                            algorithm=algorithm,
                            substrate=substrate,
                            row_y=row_y,
                            cells=cells,
                            algotype_by_cell_id=algotype_by_cell_id,
                            event_index=event_number,
                            event_kind="swap",
                            swap_count=event_number,
                            comparison_count=status_probe.compare_and_swap_count + event_number,
                            archived_compare_and_swap_count=status_probe.compare_and_swap_count,
                            scheduler_round=scheduler_rounds,
                            actor_cell_id=actor_cell_id,
                            actor_algotype=algorithm,
                            state_values=[int(value) for value in snapshot],
                        )
                    )
                    if status_probe.swap_count >= int(max_swaps):
                        stop_reason = "max_step_cap"
                        break
            if is_sorted_values(values_from_cells(cells)) or stop_reason == "max_step_cap":
                break
        scheduler_rounds += 1
        if status_probe.swap_count == swaps_before_round:
            no_progress_rounds += 1
        else:
            no_progress_rounds = 0
        if no_progress_rounds >= 100 and not is_sorted_values(values_from_cells(cells)):
            stop_reason = "no_progress_round_cap"
            break

    final_values = values_from_cells(cells)
    completed = is_sorted_values(final_values)
    if completed:
        stop_reason = "sorted"
    final_algotypes = algotypes_from_cells(cells, algotype_by_cell_id)
    dg_metrics = delayed_gratification_from_sortedness([row["sortedness_percent"] for row in trace_rows])
    row_errors: list[str] = []
    for row in trace_rows:
        if not row["row_restricted"]:
            row_errors.extend(json.loads(str(row["row_validation_errors_json"])))
    row_errors.extend(row_position_errors(cells, substrate, row_y))

    return EmbeddedRowRunResult(
        schema_version=EMBEDDED_1D_SCHEMA_VERSION,
        condition_id=condition_id,
        algorithm=algorithm,
        substrate=substrate,
        row_y=int(row_y),
        initial_values=list(values),
        final_values=final_values,
        initial_algotypes=[algorithm] * len(values),
        final_algotypes=final_algotypes,
        completed=completed,
        stop_reason=stop_reason,
        swap_count=int(status_probe.swap_count),
        comparison_count=int(status_probe.compare_and_swap_count + status_probe.swap_count),
        archived_compare_and_swap_count=int(status_probe.compare_and_swap_count),
        event_count=len(trace_rows),
        scheduler_rounds=int(scheduler_rounds),
        final_sortedness_percent=sortedness_percent(final_values),
        final_monotonicity_error=monotonicity_error(final_values),
        final_aggregation=aggregation(final_algotypes),
        delayed_gratification=dg_metrics,
        trace_rows=trace_rows,
        wall_time_seconds=time.perf_counter() - start,
        row_validation_errors=tuple(sorted(set(row_errors))),
        metadata=metadata,
    )
