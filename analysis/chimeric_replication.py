"""S13 chimeric and conflicting-direction replication.

The module executes only unprotected S08 paper-scale rows.  Frozen public C is
an evidence source, not an executable mixed-policy oracle; all trajectories
created here are explicitly clean-room R trajectories.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

from analysis.no_fault_sorting import condition_from_dict
from reference_simulator.engine import (
    evaluate_terminal,
    execute_serial_summary_activation,
    initial_state,
    run as ordinary_run,
)
from reference_simulator.model import LEDGER_FIELDS, Scenario, canonical_json_bytes, state_hash
from reference_simulator.scheduler import scheduled_actor, scheduled_side
from scenario_bank.core import materialize_scenario


REPOSITORY = Path(__file__).resolve().parents[1]
S08_DIR = Path("/artifacts/research_steps/S08")
S09_RUNS = Path("/artifacts/research_steps/S09/no_fault_runs.parquet")
S01_CLAIMS = Path("/artifacts/research_steps/S01/claim_registry.parquet")
PREREGISTRATION = REPOSITORY / "analysis" / "s13_chimera_preregistration.json"
OUTPUT_SCHEMA = "e01.s13.chimeric_run.v1"
R_PROFILE = "R-clean-room-reference-E01-v1"
GRID = tuple(index / 100 for index in range(101))
EXECUTED_FAMILIES = {
    "same_direction_unique",
    "same_direction_repeated",
    "opposite_unique",
    "opposite_repeated",
    "identical_policy_label_control",
}
EXPECTED_EXECUTED = {
    "same_direction_unique": 800,
    "same_direction_repeated": 600,
    "opposite_unique": 600,
    "opposite_repeated": 600,
    "identical_policy_label_control": 300,
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT
    ).strip()


def _selected_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(str(row["conditionId"]), []).append(str(row["scenarioId"]))
    return {min(values) for values in grouped.values()}


def load_tasks() -> list[dict[str, Any]]:
    """Load exactly the S13 paper-scale population without opening holdouts."""
    table = pq.read_table(
        S08_DIR / "paired_scenario_bank.parquet",
        filters=[("split", "=", "paper_scale")],
    )
    rows = [row for row in table.to_pylist() if row["conditionFamily"] in EXECUTED_FAMILIES]
    counts = pd.Series([row["conditionFamily"] for row in rows]).value_counts().to_dict()
    if counts != EXPECTED_EXECUTED or len(rows) != 2900:
        raise ValueError(f"S13 population mismatch: {counts}")
    if any(bool(row["protected"]) for row in rows):
        raise PermissionError("S13 paper population unexpectedly contains protected rows")
    if any(row["split"] != "paper_scale" for row in rows):
        raise PermissionError("S13 task escaped paper_scale")
    if any(row["architecture"] != "cell_view" for row in rows):
        raise ValueError("S13 chimeras must use cell-view architecture")
    if any(row["faultMode"] != "none" for row in rows):
        raise ValueError("S13 population unexpectedly contains faults")

    base_table = pq.read_table(
        S08_DIR / "base_draw_bank.parquet",
        filters=[("split", "=", "paper_scale")],
    )
    bases = {
        row["baseDrawId"]: row
        for row in base_table.to_pylist()
        if row["inputProfile"] in {"unique_1_100", "repeated_1_10_x10"}
    }
    if len(bases) != 200:
        raise ValueError(f"expected 200 paper-scale base draws, found {len(bases)}")

    catalog = pq.read_table(S08_DIR / "condition_catalog.parquet").to_pylist()
    required = {str(row["conditionId"]) for row in rows}
    conditions = {
        str(row["conditionId"]): json.loads(row["conditionJson"])
        for row in catalog
        if row["conditionId"] in required
    }
    if set(conditions) != required:
        raise ValueError("condition catalog does not cover S13 rows")

    selected = _selected_ids(rows)
    tasks: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (item["conditionId"], item["replicateOrdinal"])):
        tasks.append(
            {
                "scenario_row": row,
                "condition": conditions[row["conditionId"]],
                "base": bases[row["baseDrawId"]],
                "retain_raw_trace": row["scenarioId"] in selected,
                "trace_selection_reason": (
                    "preregistered_lexicographically_smallest_scenario_id"
                    if row["scenarioId"] in selected
                    else None
                ),
            }
        )
    if sum(bool(task["retain_raw_trace"]) for task in tasks) != 29:
        raise ValueError("expected one selected trace for each of 29 executed conditions")
    return tasks


@dataclass
class MetricTracker:
    scenario: Scenario
    control: bool
    paper_strict_edges: int = 0
    reference_nonstrict_edges: int = 0
    same_label_edges: int = 0
    equal_value_edges: int = 0
    equal_value_same_label_edges: int = 0

    def __post_init__(self) -> None:
        for right in range(1, len(self.scenario.cells)):
            contribution = self._edge(self.scenario.initial_occupancy, right)
            self._add(contribution, 1)

    def _label(self, cell_id: str) -> str:
        cell = self.scenario.cell_map[cell_id]
        if self.control:
            if cell.analysis_label is None:
                raise ValueError("identical-policy control lacks its ghost analysis label")
            return cell.analysis_label
        return cell.policy.value

    def _edge(
        self,
        occupancy: Sequence[str],
        right: int,
        remap: Callable[[int], str] | None = None,
    ) -> tuple[int, int, int, int, int]:
        getter = remap or occupancy.__getitem__
        left_id, right_id = getter(right - 1), getter(right)
        left, right_cell = self.scenario.cell_map[left_id], self.scenario.cell_map[right_id]
        equal = int(left.value == right_cell.value)
        same = int(self._label(left_id) == self._label(right_id))
        return (
            int(right_cell.value > left.value),
            int(right_cell.value >= left.value),
            same,
            equal,
            equal * same,
        )

    def _add(self, values: tuple[int, int, int, int, int], sign: int) -> None:
        self.paper_strict_edges += sign * values[0]
        self.reference_nonstrict_edges += sign * values[1]
        self.same_label_edges += sign * values[2]
        self.equal_value_edges += sign * values[3]
        self.equal_value_same_label_edges += sign * values[4]

    def metrics(self) -> dict[str, float | None]:
        n = len(self.scenario.cells)
        return {
            "paper_sortedness_percent": 100.0 * (1 + self.paper_strict_edges) / n,
            "reference_sortedness_percent": 100.0 * (1 + self.reference_nonstrict_edges) / n,
            "publication_aggregation": self.same_label_edges / n,
            "reference_aggregation": self.same_label_edges / (n - 1),
            "duplicate_same_algotype_edge_rate": (
                self.equal_value_same_label_edges / self.equal_value_edges
                if self.equal_value_edges
                else None
            ),
            "equal_value_edge_count": float(self.equal_value_edges),
        }

    def after_swap(self, state: Any, first: int, second: int) -> None:
        occupancy = state.occupancy
        affected = {
            right
            for position in (first, second)
            for right in (position, position + 1)
            if 1 <= right < len(occupancy)
        }

        def before(index: int) -> str:
            if index == first:
                return occupancy[second]
            if index == second:
                return occupancy[first]
            return occupancy[index]

        for right in affected:
            self._add(self._edge(occupancy, right, before), -1)
        for right in affected:
            self._add(self._edge(occupancy, right), 1)


class AdmissibilityTracker:
    """O(log n) exact no-fault terminal projection for S13 serial runs.

    The ordinary S03 predicate scans the line after every state change.  Some
    opposing scenarios change on nearly every one of one million activations.
    This tracker preserves the same witness definition while updating only
    edges and identities touched by a committed swap or cursor update.
    """

    def __init__(self, scenario: Scenario, state: Any) -> None:
        if any(cell.fault.value != "normal" for cell in scenario.cells):
            raise ValueError("S13 admissibility tracker is lawful only without faults")
        self.scenario = scenario
        self.positions = {cell_id: index for index, cell_id in enumerate(state.occupancy)}
        n = len(state.occupancy)
        self.asc_invalid = [False] * (n - 1)
        self.desc_invalid = [False] * (n - 1)
        self.asc_heap: list[int] = []
        self.desc_heap: list[int] = []
        self.asc_invalid_count = 0
        self.desc_invalid_count = 0
        self.bubble_witness_count = 0
        for edge in range(n - 1):
            asc, desc, bubble = self._edge(state.occupancy, edge)
            self.asc_invalid[edge] = asc
            self.desc_invalid[edge] = desc
            self.asc_invalid_count += int(asc)
            self.desc_invalid_count += int(desc)
            self.bubble_witness_count += bubble
            if asc:
                heapq.heappush(self.asc_heap, edge)
            if desc:
                heapq.heappush(self.desc_heap, edge)
        self.selection_active_count = self._selection_active_count(state)

    def _edge(
        self,
        occupancy: Sequence[str],
        edge: int,
        remap: Callable[[int], str] | None = None,
    ) -> tuple[bool, bool, int]:
        getter = remap or occupancy.__getitem__
        left = self.scenario.cell_map[getter(edge)]
        right = self.scenario.cell_map[getter(edge + 1)]
        asc = left.value > right.value
        desc = left.value < right.value
        bubble = 0
        if asc:
            bubble += int(left.policy.value == "Bubble" and left.direction.value == "ascending")
            bubble += int(right.policy.value == "Bubble" and right.direction.value == "ascending")
        if desc:
            bubble += int(left.policy.value == "Bubble" and left.direction.value == "descending")
            bubble += int(right.policy.value == "Bubble" and right.direction.value == "descending")
        return asc, desc, bubble

    def _selection_active(self, actor_id: str, state: Any) -> bool:
        actor = self.scenario.cell_map[actor_id]
        if actor.policy.value != "Selection":
            return False
        cursor = state.selection_cursors[actor_id]
        return 0 <= cursor < len(state.occupancy) and cursor != self.positions[actor_id]

    def _selection_active_count(self, state: Any) -> int:
        return sum(self._selection_active(actor_id, state) for actor_id in state.selection_cursors)

    def after_swap(self, state: Any, first: int, second: int) -> None:
        occupancy = state.occupancy
        affected = {
            edge
            for position in (first, second)
            for edge in (position - 1, position)
            if 0 <= edge < len(occupancy) - 1
        }
        moved = {occupancy[first], occupancy[second]}
        old_selection = sum(
            self._selection_active(actor_id, state)
            for actor_id in moved
            if actor_id in state.selection_cursors
        )

        def before(index: int) -> str:
            if index == first:
                return occupancy[second]
            if index == second:
                return occupancy[first]
            return occupancy[index]

        for edge in affected:
            old_asc, old_desc, old_bubble = self._edge(occupancy, edge, before)
            if old_asc != self.asc_invalid[edge] or old_desc != self.desc_invalid[edge]:
                raise AssertionError("admissibility edge cache drifted before swap")
            self.bubble_witness_count -= old_bubble
        self.positions[occupancy[first]] = first
        self.positions[occupancy[second]] = second
        for edge in affected:
            asc, desc, bubble = self._edge(occupancy, edge)
            if asc != self.asc_invalid[edge]:
                self.asc_invalid_count += 1 if asc else -1
                self.asc_invalid[edge] = asc
                if asc:
                    heapq.heappush(self.asc_heap, edge)
            if desc != self.desc_invalid[edge]:
                self.desc_invalid_count += 1 if desc else -1
                self.desc_invalid[edge] = desc
                if desc:
                    heapq.heappush(self.desc_heap, edge)
            self.bubble_witness_count += bubble
        new_selection = sum(
            self._selection_active(actor_id, state)
            for actor_id in moved
            if actor_id in state.selection_cursors
        )
        self.selection_active_count += new_selection - old_selection

    def after_memory_update(self, state: Any) -> None:
        # Memory-only updates are at most O(n^2) per run.  A compact rescan is
        # simpler and independently checkable while dense swap cycles stay O(1).
        self.selection_active_count = self._selection_active_count(state)

    def _first_invalid(self, ascending: bool) -> int | None:
        heap = self.asc_heap if ascending else self.desc_heap
        flags = self.asc_invalid if ascending else self.desc_invalid
        while heap and not flags[heap[0]]:
            heapq.heappop(heap)
        return heap[0] if heap else None

    def has_change(self, state: Any) -> bool:
        if self.bubble_witness_count > 0 or self.selection_active_count > 0:
            return True
        for ascending in (True, False):
            edge = self._first_invalid(ascending)
            if edge is None:
                continue
            actor = self.scenario.cell_map[state.occupancy[edge + 1]]
            if actor.policy.value == "Insertion" and (actor.direction.value == "ascending") == ascending:
                return True
        return False

    def completion(self) -> bool:
        directions = {cell.direction.value for cell in self.scenario.cells}
        if directions == {"ascending"}:
            return self.asc_invalid_count == 0
        if directions == {"descending"}:
            return self.desc_invalid_count == 0
        return False

    def terminal_after_change(self, state: Any) -> str | None:
        if self.completion():
            return "complete"
        if not self.has_change(state):
            return "quiescent"
        if state.activation_count >= self.scenario.max_activations:
            return "event_budget"
        return None


def _execute_s13_activation(
    scenario: Scenario,
    state: Any,
    admissibility: AdmissibilityTracker,
    *,
    on_swap: Callable[[Any, int, int], None] | None = None,
) -> str:
    """Exact no-fault, serial S03 activation without repeated prefix scans.

    Returns ``swap``, ``memory``, or ``noop``.  Observation/read ledgers match
    ``cell_view_proposal``; the first-invalid-edge cache only changes runtime.
    """
    if state.activation_count >= scenario.max_activations:
        state.terminal = "event_budget"
        return "noop"
    event_index = state.activation_count
    actor_id, _, consumed = scheduled_actor(scenario, event_index, include_draws=False)
    state.stream_counters["actor_activation"] = state.stream_counters.get("actor_activation", 0) + consumed
    actor = scenario.cell_map[actor_id]
    position = admissibility.positions[actor_id]
    reads = 0
    comparisons = 0
    outcome = "noop"
    target_position: int | None = None
    new_cursor: int | None = None

    if actor.policy.value == "Bubble":
        side, _ = scheduled_side(scenario, event_index)
        state.stream_counters["bubble_side"] = state.stream_counters.get("bubble_side", 0) + 1
        target_position = position + (-1 if side == "left" else 1)
        if not 0 <= target_position < len(state.occupancy):
            reads = 1
            target_position = None
        else:
            reads, comparisons = 2, 1
            target = scenario.cell_map[state.occupancy[target_position]]
            if actor.direction.value == "ascending":
                inversion = actor.value < target.value if side == "left" else actor.value > target.value
            else:
                inversion = actor.value > target.value if side == "left" else actor.value < target.value
            if inversion:
                outcome = "swap"
    elif actor.policy.value == "Insertion":
        if position == 0:
            reads = 1
        else:
            ascending = actor.direction.value == "ascending"
            first_invalid_edge = admissibility._first_invalid(ascending)
            if first_invalid_edge is not None and first_invalid_edge < position - 1:
                # _prefix_is_ordered reads through the right endpoint of the
                # first invalid edge, then cell_view_proposal adds one read.
                reads = first_invalid_edge + 3
                comparisons = first_invalid_edge + 1
            else:
                reads = position + 2
                comparisons = position
                target_position = position - 1
                target = scenario.cell_map[state.occupancy[target_position]]
                inversion = actor.value < target.value if ascending else actor.value > target.value
                if inversion:
                    outcome = "swap"
    else:
        cursor = state.selection_cursors[actor_id]
        if not 0 <= cursor < len(state.occupancy) or cursor == position:
            reads = 1
        else:
            reads, comparisons = 2, 1
            target_position = cursor
            target = scenario.cell_map[state.occupancy[cursor]]
            delta = 1 if actor.direction.value == "ascending" else -1
            if target.value <= actor.value:
                outcome = "memory"
                new_cursor = cursor + delta
            else:
                outcome = "swap"

    ledger = state.ledger
    ledger["activations"] += 1
    ledger["observationReads"] += reads
    ledger["valueComparisons"] += comparisons
    ledger["proposals"] += 1
    if outcome == "noop":
        ledger["noOps"] += 1
    elif outcome == "memory":
        ledger["memoryUpdates"] += 1
        state.selection_cursors[actor_id] = new_cursor
    else:
        assert target_position is not None
        state.occupancy[position], state.occupancy[target_position] = (
            state.occupancy[target_position], state.occupancy[position]
        )
        if on_swap is not None:
            on_swap(state, position, target_position)
        ledger["acceptedSwaps"] += 1
        ledger["displacedCells"] += 2
    state.activation_count += 1
    if state.activation_count >= scenario.max_activations:
        state.terminal = "event_budget"
    return outcome


def _composition_null(counts: Mapping[str, int], n: int) -> tuple[float, float]:
    reference = sum(count * (count - 1) for count in counts.values()) / (n * (n - 1))
    return reference * (n - 1) / n, reference


def _floor_grid(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not points:
        raise ValueError("trajectory is empty")
    final = len(points) - 1
    return [dict(points[math.floor(progress * final)]) for progress in GRID]


def run_reference_summary(scenario: Scenario, *, family: str, retain_raw_trace: bool) -> dict[str, Any]:
    scenario.validate()
    state = initial_state(scenario)
    state.terminal = evaluate_terminal(scenario, state)
    control = family == "identical_policy_label_control"
    tracker = MetricTracker(scenario, control)
    admissibility = AdmissibilityTracker(scenario, state)
    initial_occupancy = list(state.occupancy)
    initial_values = [scenario.cell_map[cell_id].value for cell_id in initial_occupancy]
    initial_hash = state_hash(scenario.scenario_id, state)
    points: list[dict[str, Any]] = [{"swap_index": 0, **tracker.metrics()}]

    swap_committed = False

    def on_swap(current: Any, first: int, second: int) -> None:
        nonlocal swap_committed
        swap_committed = True
        tracker.after_swap(current, first, second)
        admissibility.after_swap(current, first, second)
        # The summary engine invokes the callback immediately after occupancy
        # commit and immediately before adding this swap to the ledger.
        points.append({"swap_index": int(current.ledger["acceptedSwaps"]) + 1, **tracker.metrics()})

    start = time.perf_counter()
    while state.terminal is None:
        swap_committed = False
        outcome = _execute_s13_activation(
            scenario, state, admissibility, on_swap=on_swap
        )
        if outcome != "noop":
            if not swap_committed:
                admissibility.after_memory_update(state)
            state.terminal = admissibility.terminal_after_change(state)
    elapsed = time.perf_counter() - start
    official_terminal = evaluate_terminal(scenario, state)
    if state.terminal != official_terminal:
        raise AssertionError(
            f"incremental terminal mismatch: {state.terminal} != {official_terminal}"
        )
    final_values = [scenario.cell_map[cell_id].value for cell_id in state.occupancy]
    if points[-1]["swap_index"] != state.ledger["acceptedSwaps"]:
        raise AssertionError("accepted-swap trajectory is incomplete")
    final_metrics = tracker.metrics()
    full_recomputed = MetricTracker(
        Scenario.create(
            scenario.cells,
            initial_occupancy=tuple(state.occupancy),
            seed=scenario.seed,
            max_activations=scenario.max_activations,
            architecture=scenario.architecture,
            batch_width=scenario.batch_width,
            generation_key="S13/final-metric-recomputation",
        ),
        control,
    ).metrics()
    for key in final_metrics:
        left, right = final_metrics[key], full_recomputed[key]
        if left is None or right is None:
            if left is not right:
                raise AssertionError(f"final metric null mismatch for {key}")
        elif not math.isclose(float(left), float(right), abs_tol=1e-12):
            raise AssertionError(f"incremental metric mismatch for {key}: {left} != {right}")

    if control:
        labels = [tracker._label(cell.cell_id) for cell in scenario.cells]
    else:
        labels = [cell.policy.value for cell in scenario.cells]
    label_counts = {label: labels.count(label) for label in sorted(set(labels))}
    policy_counts = {
        policy: sum(cell.policy.value == policy for cell in scenario.cells)
        for policy in sorted({cell.policy.value for cell in scenario.cells})
    }
    direction_counts = {
        direction: sum(cell.direction.value == direction for cell in scenario.cells)
        for direction in sorted({cell.direction.value for cell in scenario.cells})
    }
    publication_null, reference_null = _composition_null(label_counts, len(labels))
    homogeneous = len(direction_counts) == 1
    expected_order = next(iter(direction_counts)) if homogeneous else None
    if expected_order == "ascending":
        final_ordered: bool | None = all(a <= b for a, b in zip(final_values, final_values[1:]))
    elif expected_order == "descending":
        final_ordered = all(a >= b for a, b in zip(final_values, final_values[1:]))
    else:
        final_ordered = None

    ledger = dict(state.ledger)
    target_calculations = ledger["proposals"]
    rejected_actions = ledger["rejections"] + ledger["conflictLosses"]
    full_cost = (
        ledger["activations"] + ledger["observationReads"] + ledger["valueComparisons"]
        + target_calculations + ledger["proposals"] + ledger["noOps"]
        + ledger["rejections"] + ledger["memoryUpdates"] + ledger["acceptedSwaps"]
        + ledger["displacedCells"] + ledger["conflictLosses"]
    )
    grid = _floor_grid(points)
    return {
        "schema_version": OUTPUT_SCHEMA,
        "research_step_id": "S13",
        "backend_profile": R_PROFILE,
        "evidence_layer": "clean_room_reference",
        "publication_snapshot_claimed": False,
        "scenario_id": scenario.scenario_id,
        "architecture": scenario.architecture.value,
        "scheduler": scenario.scheduler,
        "sequence_basis": "accepted_swap_derived_from_activation",
        "family": family,
        "stop_reason": state.terminal,
        "completed": state.terminal == "complete",
        "censored": state.terminal in {"event_budget", "invariant_error"},
        "activation_count": state.activation_count,
        "successful_swap_count": ledger["acceptedSwaps"],
        "initial_state_hash": initial_hash,
        "final_state_hash": state_hash(scenario.scenario_id, state),
        "initial_values_sha256": canonical_hash(initial_values),
        "final_values_sha256": canonical_hash(final_values),
        "value_multiset_conserved": sorted(initial_values) == sorted(final_values),
        "final_consensus_ordered": final_ordered,
        "elapsed_seconds": elapsed,
        "policy_counts_json": json.dumps(policy_counts, sort_keys=True, separators=(",", ":")),
        "clustering_label_counts_json": json.dumps(label_counts, sort_keys=True, separators=(",", ":")),
        "direction_counts_json": json.dumps(direction_counts, sort_keys=True, separators=(",", ":")),
        "publication_aggregation_null": publication_null,
        "reference_aggregation_null": reference_null,
        "initial_paper_sortedness_percent": points[0]["paper_sortedness_percent"],
        "final_paper_sortedness_percent": final_metrics["paper_sortedness_percent"],
        "initial_reference_sortedness_percent": points[0]["reference_sortedness_percent"],
        "final_reference_sortedness_percent": final_metrics["reference_sortedness_percent"],
        "initial_publication_aggregation": points[0]["publication_aggregation"],
        "final_publication_aggregation": final_metrics["publication_aggregation"],
        "initial_reference_aggregation": points[0]["reference_aggregation"],
        "final_reference_aggregation": final_metrics["reference_aggregation"],
        "final_duplicate_same_algotype_edge_rate": final_metrics["duplicate_same_algotype_edge_rate"],
        "final_equal_value_edge_count": final_metrics["equal_value_edge_count"],
        "swap_only_cost": ledger["acceptedSwaps"],
        "publication_swap_plus_comparison_cost": ledger["acceptedSwaps"] + ledger["valueComparisons"],
        "target_calculations": target_calculations,
        "rejected_actions": rejected_actions,
        "unit_weight_full_ledger": full_cost,
        **{f"ledger_{key}": ledger[key] for key in LEDGER_FIELDS},
        "trajectory_sha256": canonical_hash(points),
        "curve_grid": grid,
        "raw_trace": points if retain_raw_trace else None,
    }


def _identity(task: Mapping[str, Any]) -> dict[str, Any]:
    row = task["scenario_row"]
    return {
        "condition_id": row["conditionId"],
        "condition_family": row["conditionFamily"],
        "profile_role": row["profileRole"],
        "assignment_profile": row["assignmentProfile"],
        "input_profile": row["inputProfile"],
        "policy_set_json": json.dumps(row["policySet"], separators=(",", ":")),
        "direction_profile": row["directionProfile"],
        "base_draw_id": row["baseDrawId"],
        "pairing_block_id": row["pairingBlockId"],
        "split": row["split"],
        "protected": bool(row["protected"]),
        "replicate_ordinal": int(row["replicateOrdinal"]),
        "runtime_seed": row["runtimeSeed"],
        "generation_key": row["generationKey"],
        "event_budget": int(row["maxActivations"]),
        "scenario_json_sha256": row["scenarioJsonSha256"],
        "backend_eligibility": row["backendEligibility"],
        "historical_random_stream_status": row["historicalRandomStreamStatus"],
        "trace_selection_reason": task["trace_selection_reason"],
    }


def _worker(task: Mapping[str, Any]) -> dict[str, Any]:
    condition = condition_from_dict(task["condition"])
    scenario, metadata = materialize_scenario(condition, task["base"])
    row = task["scenario_row"]
    if scenario.scenario_id != row["scenarioId"]:
        raise ValueError("S08 scenario identity failed rematerialization")
    if metadata["scenarioJsonSha256"] != row["scenarioJsonSha256"]:
        raise ValueError("S08 scenario JSON hash failed rematerialization")
    result = run_reference_summary(
        scenario,
        family=str(row["conditionFamily"]),
        retain_raw_trace=bool(task["retain_raw_trace"]),
    )
    result.update(_identity(task))
    result["run_id"] = "s13r:" + hashlib.sha256(
        f"{R_PROFILE}|{scenario.scenario_id}".encode()
    ).hexdigest()
    return result


def _read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["scenario_id"]] = row
    return rows


def run_population(
    tasks: Sequence[Mapping[str, Any]], checkpoint: Path, *, workers: int = 8
) -> list[dict[str, Any]]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1,8]")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_checkpoint(checkpoint)
    expected = {str(task["scenario_row"]["scenarioId"]) for task in tasks}
    if not set(completed) <= expected:
        raise ValueError("checkpoint contains rows outside the S13 population")
    pending = [task for task in tasks if task["scenario_row"]["scenarioId"] not in completed]
    with checkpoint.open("a", encoding="utf-8") as handle:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker, task): task for task in pending}
            for future in as_completed(futures):
                result = future.result()
                handle.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                completed[result["scenario_id"]] = result
                count = len(completed)
                if count % 50 == 0 or count == len(tasks):
                    print(f"S13 reference progress {count}/{len(tasks)}", file=sys.stderr, flush=True)
    if set(completed) != expected:
        raise ValueError("S13 run accounting mismatch")
    return [completed[key] for key in sorted(completed)]


def exact_replay_selected(tasks: Sequence[Mapping[str, Any]], *, workers: int = 8) -> list[dict[str, Any]]:
    selected = [task for task in tasks if task["retain_raw_trace"]]
    if len(selected) != 29:
        raise ValueError("selected replay population mismatch")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        reruns = list(executor.map(_worker, selected))
    return reruns


def ordinary_engine_comparisons(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compare six representative summary results to the ordinary engine."""
    wanted = (
        "C-UNQ-CHIM-BUB-INS-EXACT-ASC",
        "C-UNQ-CHIM-BUB-INS-SEL-EXACT-ASC",
        "C-REP-CHIM-BUB-SEL-EXACT-ASC",
        "C-UNQ-CHIM-BUB-INS-EXACT-OPP",
        "C-UNQ-CHIM-BUB-SEL-EXACT-OPP",
        "C-UNQ-CHIM-INS-SEL-EXACT-OPP",
    )
    samples = []
    for condition_id in wanted:
        candidates = [task for task in tasks if task["scenario_row"]["conditionId"] == condition_id]
        samples.append(min(candidates, key=lambda task: task["scenario_row"]["scenarioId"]))
    results: list[dict[str, Any]] = []
    for task in samples:
        condition = condition_from_dict(task["condition"])
        scenario, _ = materialize_scenario(condition, task["base"])
        summary = _worker(task)
        ordinary = ordinary_run(scenario, trace_mode="digest")
        checks = {
            "stopReason": summary["stop_reason"] == ordinary.summary["stopReason"],
            "activationCount": summary["activation_count"] == ordinary.summary["activationCount"],
            "acceptedSwaps": summary["successful_swap_count"] == ordinary.summary["ledger"]["acceptedSwaps"],
            "ledger": all(summary[f"ledger_{key}"] == ordinary.summary["ledger"][key] for key in LEDGER_FIELDS),
            "finalStateHash": summary["final_state_hash"] == ordinary.final_state_hash,
        }
        results.append(
            {
                "conditionId": condition_id,
                "scenarioId": scenario.scenario_id,
                "checks": checks,
                "passed": all(checks.values()),
                "ordinaryEventDigest": ordinary.event_digest,
            }
        )
    return results


def _bootstrap_ci(
    values: Sequence[float], *, address: str, confidence: float = 0.95, draws: int = 10_000
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return math.nan, math.nan
    if np.all(array == array[0]):
        return float(array[0]), float(array[0])
    seed = int.from_bytes(
        hashlib.sha256(f"E01/S13/bootstrap/{address}".encode()).digest()[:16], "big"
    )
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    means = np.empty(draws, dtype=np.float64)
    chunk = 1000
    for start in range(0, draws, chunk):
        stop = min(start + chunk, draws)
        indices = generator.integers(0, len(array), size=(stop - start, len(array)))
        means[start:stop] = array[indices].mean(axis=1)
    alpha = 1 - confidence
    return tuple(float(value) for value in np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def _wilson(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = stats.norm.ppf(0.5 + confidence / 2)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - radius, center + radius


def _pure_s09_rows() -> pd.DataFrame:
    source = pd.read_parquet(S09_RUNS)
    source = source[
        source.backend_profile.eq(R_PROFILE)
        & source.split.eq("paper_scale")
        & source.architecture.eq("cell_view")
    ].copy()
    if len(source) != 300 or set(source.policy) != {"Bubble", "Insertion", "Selection"}:
        raise ValueError("S09 pure reference population is not the expected 300 rows")
    rows: list[dict[str, Any]] = []
    for item in source.to_dict(orient="records"):
        ledger = {key: int(item[f"ledger_{key}"]) for key in LEDGER_FIELDS}
        target = ledger["proposals"]
        rejected = ledger["rejections"] + ledger["conflictLosses"]
        full = (
            ledger["activations"] + ledger["observationReads"] + ledger["valueComparisons"]
            + target + ledger["proposals"] + ledger["noOps"] + ledger["rejections"]
            + ledger["memoryUpdates"] + ledger["acceptedSwaps"]
            + ledger["displacedCells"] + ledger["conflictLosses"]
        )
        rows.append(
            {
                "schema_version": OUTPUT_SCHEMA,
                "research_step_id": "S13",
                "backend_profile": R_PROFILE,
                "evidence_layer": "clean_room_reference_reused_validated_S09",
                "publication_snapshot_claimed": False,
                "scenario_id": item["scenario_id"],
                "condition_id": item["condition_id"],
                "condition_family": "unique_pure",
                "family": "unique_pure",
                "profile_role": "reference_primary",
                "assignment_profile": "pure",
                "input_profile": "unique_1_100",
                "policy_set_json": json.dumps([item["policy"]], separators=(",", ":")),
                "policy_counts_json": json.dumps({item["policy"]: 100}, separators=(",", ":")),
                "clustering_label_counts_json": json.dumps({item["policy"]: 100}, separators=(",", ":")),
                "direction_profile": "consensus_ascending",
                "direction_counts_json": json.dumps({"ascending": 100}, separators=(",", ":")),
                "base_draw_id": item["base_draw_id"],
                "pairing_block_id": item["pairing_block_id"],
                "split": item["split"],
                "protected": bool(item["protected"]),
                "replicate_ordinal": int(item["replicate_ordinal"]),
                "runtime_seed": item["runtime_seed"],
                "generation_key": item["generation_key"],
                "event_budget": int(item["event_budget"]),
                "scenario_json_sha256": item["scenario_json_sha256"],
                "backend_eligibility": "C_adapter_endpoint_supported_no_fault",
                "historical_random_stream_status": item["historical_random_stream_status"],
                "architecture": "cell_view",
                "scheduler": item["scheduler"],
                "sequence_basis": item["sequence_basis"],
                "stop_reason": item["stop_reason"],
                "completed": bool(item["completed"]),
                "censored": item["stop_reason"] in {"event_budget", "invariant_error"},
                "activation_count": int(item["activation_count"]),
                "successful_swap_count": int(item["successful_swap_count"]),
                "initial_state_hash": None,
                "final_state_hash": item["final_state_hash"],
                "initial_values_sha256": item["initial_values_sha256"],
                "final_values_sha256": item["final_values_sha256"],
                "value_multiset_conserved": bool(item["value_multiset_conserved"]),
                "final_consensus_ordered": bool(item["final_nonstrictly_sorted"]),
                "elapsed_seconds": float(item["elapsed_seconds"]),
                "publication_aggregation_null": 0.99,
                "reference_aggregation_null": 1.0,
                "initial_paper_sortedness_percent": float(item["initial_sortedness_percent"]),
                "final_paper_sortedness_percent": float(item["final_sortedness_percent"]),
                "initial_reference_sortedness_percent": float(item["initial_sortedness_percent"]),
                "final_reference_sortedness_percent": float(item["final_sortedness_percent"]),
                "initial_publication_aggregation": 0.99,
                "final_publication_aggregation": 0.99,
                "initial_reference_aggregation": 1.0,
                "final_reference_aggregation": 1.0,
                "final_duplicate_same_algotype_edge_rate": math.nan,
                "final_equal_value_edge_count": 0.0,
                "swap_only_cost": ledger["acceptedSwaps"],
                "publication_swap_plus_comparison_cost": ledger["acceptedSwaps"] + ledger["valueComparisons"],
                "target_calculations": target,
                "rejected_actions": rejected,
                "unit_weight_full_ledger": full,
                **{f"ledger_{key}": ledger[key] for key in LEDGER_FIELDS},
                "trajectory_sha256": item["trace_sha256"],
                "trace_selection_reason": item["trace_selection_reason"],
                "run_id": "s13reuse:" + str(item["run_id"]),
                "source_s09_run_id": item["run_id"],
                "source_s09_sha256": sha256_file(S09_RUNS),
            }
        )
    return pd.DataFrame(rows)


def _results_frame(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = [
        {key: value for key, value in result.items() if key not in {"curve_grid", "raw_trace"}}
        for result in results
    ]
    frame = pd.concat([pd.DataFrame(rows), _pure_s09_rows()], ignore_index=True, sort=False)
    if len(frame) != 3200 or frame.run_id.duplicated().any():
        raise ValueError("combined S13 results must contain 3200 unique run IDs")
    return frame.sort_values(["condition_id", "replicate_ordinal"]).reset_index(drop=True)


def _trajectory_frame(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        curve = result["curve_grid"]
        for grid_index, point in enumerate(curve):
            warp_index = min(100, math.floor(1.25 * grid_index))
            warped = curve[warp_index]
            rows.append(
                {
                    "scenario_id": result["scenario_id"],
                    "condition_id": result["condition_id"],
                    "condition_family": result["condition_family"],
                    "assignment_profile": result["assignment_profile"],
                    "profile_role": result["profile_role"],
                    "input_profile": result["input_profile"],
                    "policy_set_json": result["policy_set_json"],
                    "direction_profile": result["direction_profile"],
                    "pairing_block_id": result["pairing_block_id"],
                    "replicate_ordinal": result["replicate_ordinal"],
                    "grid_index": grid_index,
                    "normalized_swap_progress": GRID[grid_index],
                    "source_swap_index": int(point["swap_index"]),
                    "paper_sortedness_percent": point["paper_sortedness_percent"],
                    "reference_sortedness_percent": point["reference_sortedness_percent"],
                    "publication_aggregation": point["publication_aggregation"],
                    "reference_aggregation": point["reference_aggregation"],
                    "duplicate_same_algotype_edge_rate": point["duplicate_same_algotype_edge_rate"],
                    "equal_value_edge_count": point["equal_value_edge_count"],
                    "frozen_disorder_warp_progress": min(1.0, 1.25 * GRID[grid_index]),
                    "warped_paper_sortedness_percent": warped["paper_sortedness_percent"],
                    "warped_publication_aggregation": warped["publication_aggregation"],
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != 2900 * 101:
        raise ValueError("trajectory-grid run accounting mismatch")
    return result


def _selected_trace_frame(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result["raw_trace"] is None:
            continue
        final_swaps = int(result["successful_swap_count"])
        for point in result["raw_trace"]:
            rows.append(
                {
                    "scenario_id": result["scenario_id"],
                    "condition_id": result["condition_id"],
                    "condition_family": result["condition_family"],
                    "assignment_profile": result["assignment_profile"],
                    "input_profile": result["input_profile"],
                    "stop_reason": result["stop_reason"],
                    "swap_index": int(point["swap_index"]),
                    "normalized_swap_progress": (
                        point["swap_index"] / final_swaps if final_swaps else 0.0
                    ),
                    **{key: value for key, value in point.items() if key != "swap_index"},
                }
            )
    result = pd.DataFrame(rows)
    if result.scenario_id.nunique() != 29:
        raise ValueError("selected trace coverage mismatch")
    return result


def _condition_label(condition_id: str) -> str:
    if "BUB-INS-SEL" in condition_id:
        return "Bubble+Insertion+Selection"
    if "BUB-INS" in condition_id:
        return "Bubble+Insertion"
    if "BUB-SEL" in condition_id:
        return "Bubble+Selection"
    if "INS-SEL" in condition_id:
        return "Insertion+Selection"
    if "PURE-CV-BUB" in condition_id:
        return "Bubble"
    if "PURE-CV-INS" in condition_id:
        return "Insertion"
    if "PURE-CV-SEL" in condition_id:
        return "Selection"
    return condition_id


def _condition_summary(frame: pd.DataFrame, trajectories: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition_id, group in frame.groupby("condition_id", sort=True):
        success = int(group.completed.sum())
        lower, upper = _wilson(success, len(group))
        row: dict[str, Any] = {
            "condition_id": condition_id,
            "condition_label": _condition_label(condition_id),
            "condition_family": group.condition_family.iloc[0],
            "profile_role": group.profile_role.iloc[0],
            "assignment_profile": group.assignment_profile.iloc[0],
            "input_profile": group.input_profile.iloc[0],
            "runs": len(group),
            "completed_runs": success,
            "completion_fraction": success / len(group),
            "completion_wilson95_low": lower,
            "completion_wilson95_high": upper,
            "censored_runs": int(group.censored.sum()),
            "stop_reason_counts_json": json.dumps(group.stop_reason.value_counts().sort_index().to_dict(), separators=(",", ":")),
        }
        metrics = (
            "successful_swap_count", "activation_count", "swap_only_cost",
            "publication_swap_plus_comparison_cost", "unit_weight_full_ledger",
            "initial_paper_sortedness_percent", "final_paper_sortedness_percent",
            "initial_reference_sortedness_percent", "final_reference_sortedness_percent",
            "initial_publication_aggregation", "final_publication_aggregation",
            "initial_reference_aggregation", "final_reference_aggregation",
            "final_duplicate_same_algotype_edge_rate", "elapsed_seconds",
        )
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"mean_{metric}"] = float(values.mean()) if len(values) else math.nan
            row[f"sd_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            if len(values):
                low, high = _bootstrap_ci(values, address=f"condition/{condition_id}/{metric}")
            else:
                low, high = math.nan, math.nan
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        curve = trajectories[trajectories.condition_id.eq(condition_id)]
        if len(curve):
            mean_curve = curve.groupby("grid_index", sort=True).publication_aggregation.mean()
            peak = float(mean_curve.max())
            peak_index = int(mean_curve[mean_curve.eq(peak)].index.min())
            row["mean_curve_publication_aggregation_peak"] = peak
            row["mean_curve_peak_progress_percent"] = peak_index
        else:
            row["mean_curve_publication_aggregation_peak"] = math.nan
            row["mean_curve_peak_progress_percent"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _peak_uncertainty(trajectories: pd.DataFrame) -> pd.DataFrame:
    source = trajectories[
        trajectories.condition_family.isin(
            ["same_direction_unique", "same_direction_repeated", "identical_policy_label_control"]
        )
    ]
    rows = []
    for condition_id, group in source.groupby("condition_id", sort=True):
        matrix = group.pivot(
            index="scenario_id", columns="grid_index", values="publication_aggregation"
        ).sort_index().to_numpy(dtype=np.float64)
        if matrix.shape != (100, 101):
            raise ValueError(f"peak uncertainty matrix mismatch for {condition_id}: {matrix.shape}")
        seed = int.from_bytes(
            hashlib.sha256(f"E01/S13/peak-bootstrap/{condition_id}".encode()).digest()[:16], "big"
        )
        generator = np.random.Generator(np.random.PCG64DXSM(seed))
        counts = generator.multinomial(100, [0.01] * 100, size=10_000)
        curves = counts @ matrix / 100.0
        peaks = curves.max(axis=1)
        locations = curves.argmax(axis=1).astype(np.float64)
        observed = matrix.mean(axis=0)
        rows.append(
            {
                "condition_id": condition_id,
                "condition_label": _condition_label(condition_id),
                "condition_family": group.condition_family.iloc[0],
                "assignment_profile": group.assignment_profile.iloc[0],
                "input_profile": group.input_profile.iloc[0],
                "runs": 100,
                "bootstrap_draws": 10000,
                "observed_mean_curve_peak": float(observed.max()),
                "observed_earliest_peak_progress_percent": int(observed.argmax()),
                "peak_ci95_low": float(np.quantile(peaks, 0.025)),
                "peak_ci95_high": float(np.quantile(peaks, 0.975)),
                "peak_progress_ci95_low": float(np.quantile(locations, 0.025)),
                "peak_progress_ci95_high": float(np.quantile(locations, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def _assignment_sensitivity(frame: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    exact_conditions = sorted(
        value
        for value in frame.condition_id.unique()
        if "-EXACT-" in value
        and frame.loc[frame.condition_id.eq(value), "condition_family"].iloc[0]
        in {"same_direction_unique", "same_direction_repeated", "opposite_unique", "opposite_repeated"}
    )
    by_summary = summary.set_index("condition_id")
    for exact_id in exact_conditions:
        random_id = exact_id.replace("-EXACT-", "-RANDOM-")
        exact = frame[frame.condition_id.eq(exact_id)]
        random = frame[frame.condition_id.eq(random_id)]
        merged = exact.merge(random, on="pairing_block_id", suffixes=("_exact", "_random"), validate="one_to_one")
        row: dict[str, Any] = {
            "exact_condition_id": exact_id,
            "random_condition_id": random_id,
            "condition_label": _condition_label(exact_id),
            "condition_family": exact.condition_family.iloc[0],
            "input_profile": exact.input_profile.iloc[0],
            "pairs": len(merged),
            "exact_composition": "50/50" if len(json.loads(exact.policy_counts_json.iloc[0])) == 2 else "rotating_34/33/33",
            "random_realized_composition_minmax_json": json.dumps(
                {
                    key: [
                        min(json.loads(value).get(key, 0) for value in random.policy_counts_json),
                        max(json.loads(value).get(key, 0) for value in random.policy_counts_json),
                    ]
                    for key in sorted(set().union(*(json.loads(value) for value in random.policy_counts_json)))
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        for metric in (
            "successful_swap_count", "final_paper_sortedness_percent",
            "final_publication_aggregation", "activation_count",
        ):
            delta = merged[f"{metric}_random"] - merged[f"{metric}_exact"]
            low, high = _bootstrap_ci(delta, address=f"assignment/{exact_id}/{metric}")
            row[f"paired_random_minus_exact_mean_{metric}"] = float(delta.mean())
            row[f"paired_random_minus_exact_ci95_low_{metric}"] = low
            row[f"paired_random_minus_exact_ci95_high_{metric}"] = high
        row["exact_censored_runs"] = int(exact.censored.sum())
        row["random_censored_runs"] = int(random.censored.sum())
        row["exact_mean_curve_peak"] = float(by_summary.loc[exact_id].mean_curve_publication_aggregation_peak)
        row["random_mean_curve_peak"] = float(by_summary.loc[random_id].mean_curve_publication_aggregation_peak)
        row["random_minus_exact_mean_curve_peak"] = row["random_mean_curve_peak"] - row["exact_mean_curve_peak"]
        pairs.append(row)
    return pd.DataFrame(pairs)


def _aggregation_control_contrasts(trajectories: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    source = trajectories[
        trajectories.input_profile.eq("unique_1_100")
        & trajectories.assignment_profile.eq("balanced_exact")
        & trajectories.condition_family.eq("same_direction_unique")
    ]
    controls = trajectories[trajectories.condition_family.eq("identical_policy_label_control")]
    by_summary = summary.set_index("condition_id")
    rows = []
    for condition_id, group in source.groupby("condition_id", sort=True):
        label = _condition_label(condition_id)
        peak_index = int(by_summary.loc[condition_id].mean_curve_peak_progress_percent)
        chimera = group[group.grid_index.eq(peak_index)][
            ["pairing_block_id", "publication_aggregation"]
        ].rename(columns={"publication_aggregation": "chimera"})
        if label == "Bubble+Insertion+Selection":
            control = (
                controls[controls.grid_index.eq(peak_index)]
                .groupby("pairing_block_id", as_index=False)
                .publication_aggregation.mean()
                .rename(columns={"publication_aggregation": "control"})
            )
            control_id = "mean_of_three_ghost_controls"
        else:
            code = {
                "Bubble+Insertion": "BUB-INS",
                "Bubble+Selection": "BUB-SEL",
                "Insertion+Selection": "INS-SEL",
            }[label]
            control_id = f"C-UNQ-CONTROL-GHOST-{code}"
            control = controls[
                controls.condition_id.eq(control_id) & controls.grid_index.eq(peak_index)
            ][["pairing_block_id", "publication_aggregation"]].rename(
                columns={"publication_aggregation": "control"}
            )
        merged = chimera.merge(control, on="pairing_block_id", validate="one_to_one")
        delta = merged.chimera - merged.control
        low, high = _bootstrap_ci(
            delta,
            address=f"aggregation-control/{condition_id}/{peak_index}",
            confidence=0.9875,
        )
        rows.append(
            {
                "condition_id": condition_id,
                "condition_label": label,
                "control_id": control_id,
                "peak_progress_percent": peak_index,
                "pairs": len(merged),
                "mean_chimera_aggregation": float(merged.chimera.mean()),
                "mean_control_aggregation": float(merged.control.mean()),
                "paired_chimera_minus_control_mean": float(delta.mean()),
                "bonferroni_98_75_ci_low": low,
                "bonferroni_98_75_ci_high": high,
                "clean_room_elevation_supported": low > 0,
            }
        )
    return pd.DataFrame(rows)


def _cost_interpolation(frame: pd.DataFrame) -> pd.DataFrame:
    pure = frame[frame.condition_family.eq("unique_pure")].copy()
    pure["policy"] = pure.condition_id.map(
        lambda value: "Bubble" if "BUB" in value else ("Insertion" if "INS" in value else "Selection")
    )
    pure_wide = pure.pivot(index="pairing_block_id", columns="policy", values="swap_only_cost")
    exact = frame[
        frame.condition_family.eq("same_direction_unique")
        & frame.assignment_profile.eq("balanced_exact")
        & ~frame.condition_id.str.contains("BUB-INS-SEL")
    ]
    rows = []
    for condition_id, group in exact.groupby("condition_id", sort=True):
        policies = json.loads(group.policy_set_json.iloc[0])
        merged = group[["pairing_block_id", "swap_only_cost"]].merge(
            pure_wide[list(policies)], left_on="pairing_block_id", right_index=True, validate="one_to_one"
        )
        midpoint = merged[policies].mean(axis=1)
        difference = merged.swap_only_cost - midpoint
        low, high = _bootstrap_ci(difference, address=f"cost-midpoint/{condition_id}")
        mixed_mean = float(merged.swap_only_cost.mean())
        pure_means = [float(merged[policy].mean()) for policy in policies]
        midpoint_mean = float(midpoint.mean())
        margin = 0.10 * midpoint_mean
        rows.append(
            {
                "condition_id": condition_id,
                "condition_label": _condition_label(condition_id),
                "pairs": len(merged),
                "policies_json": json.dumps(policies, separators=(",", ":")),
                "pure_first_mean_swaps": pure_means[0],
                "pure_second_mean_swaps": pure_means[1],
                "pure_midpoint_mean_swaps": midpoint_mean,
                "mixed_mean_swaps": mixed_mean,
                "paired_mixed_minus_midpoint_mean": float(difference.mean()),
                "paired_difference_ci95_low": low,
                "paired_difference_ci95_high": high,
                "equivalence_margin": margin,
                "between_pure_means": min(pure_means) <= mixed_mean <= max(pure_means),
                "midpoint_equivalent": low >= -margin and high <= margin,
            }
        )
    return pd.DataFrame(rows)


def _opposing_summary(frame: pd.DataFrame, trajectories: pd.DataFrame) -> pd.DataFrame:
    source = frame[frame.condition_family.isin(["opposite_unique", "opposite_repeated"])]
    rows: list[dict[str, Any]] = []
    confidence_adjusted = 1 - 0.05 / 3
    for condition_id, group in source.groupby("condition_id", sort=True):
        curve = trajectories[trajectories.condition_id.eq(condition_id)]
        mean_curve = curve.groupby("grid_index", sort=True).agg(
            paper_sortedness=("paper_sortedness_percent", "mean"),
            reference_sortedness=("reference_sortedness_percent", "mean"),
            publication_aggregation=("publication_aggregation", "mean"),
            warped_paper_sortedness=("warped_paper_sortedness_percent", "mean"),
        )
        rho = float(stats.spearmanr(np.arange(101), mean_curve.paper_sortedness).statistic)
        reference_rho = float(stats.spearmanr(np.arange(101), mean_curve.reference_sortedness).statistic)
        delta = group.final_publication_aggregation - group.initial_publication_aggregation
        low, high = _bootstrap_ci(
            delta,
            address=f"opposing-agg-adjusted/{condition_id}",
            confidence=confidence_adjusted,
        )
        condition_label = _condition_label(condition_id)
        first_quarter_min = float(mean_curve.paper_sortedness.iloc[:26].min())
        first_quarter_min_i = int(mean_curve.paper_sortedness.iloc[:26].idxmin())
        rebound_slice = mean_curve.paper_sortedness.loc[first_quarter_min_i:75]
        rebound = float(rebound_slice.max())
        final = float(mean_curve.paper_sortedness.iloc[-1])
        if condition_label == "Bubble+Selection":
            shape = (
                float(mean_curve.paper_sortedness.iloc[0]) - first_quarter_min >= 2
                and rebound - first_quarter_min >= 2
                and rebound - final >= 2
                and final < 44
            )
        elif condition_label == "Bubble+Insertion":
            shape = rho >= 0.9 and final > 50
        else:
            shape = rho <= -0.9 and final < 50
        reference_first_quarter_min = float(mean_curve.reference_sortedness.iloc[:26].min())
        reference_first_quarter_min_i = int(mean_curve.reference_sortedness.iloc[:26].idxmin())
        reference_rebound = float(
            mean_curve.reference_sortedness.loc[reference_first_quarter_min_i:75].max()
        )
        reference_final = float(mean_curve.reference_sortedness.iloc[-1])
        if condition_label == "Bubble+Selection":
            reference_shape = (
                float(mean_curve.reference_sortedness.iloc[0]) - reference_first_quarter_min >= 2
                and reference_rebound - reference_first_quarter_min >= 2
                and reference_rebound - reference_final >= 2
                and reference_final < 44
            )
        elif condition_label == "Bubble+Insertion":
            reference_shape = reference_rho >= 0.9 and reference_final > 50
        else:
            reference_shape = reference_rho <= -0.9 and reference_final < 50
        rows.append(
            {
                "condition_id": condition_id,
                "condition_label": condition_label,
                "input_profile": group.input_profile.iloc[0],
                "assignment_profile": group.assignment_profile.iloc[0],
                "runs": len(group),
                "mean_initial_paper_sortedness_percent": float(group.initial_paper_sortedness_percent.mean()),
                "mean_final_paper_sortedness_percent": float(group.final_paper_sortedness_percent.mean()),
                "mean_initial_reference_sortedness_percent": float(group.initial_reference_sortedness_percent.mean()),
                "mean_final_reference_sortedness_percent": float(group.final_reference_sortedness_percent.mean()),
                "mean_initial_publication_aggregation": float(group.initial_publication_aggregation.mean()),
                "mean_final_publication_aggregation": float(group.final_publication_aggregation.mean()),
                "mean_publication_aggregation_change": float(delta.mean()),
                "aggregation_change_adjusted_ci_low": low,
                "aggregation_change_adjusted_ci_high": high,
                "aggregation_rise_supported": low > 0,
                "mean_curve_spearman_rho": rho,
                "trajectory_shape_supported": shape,
                "reference_mean_curve_spearman_rho": reference_rho,
                "reference_trajectory_shape_supported": reference_shape,
                "duplicate_sortedness_shape_metric_agreement": shape == reference_shape,
                "completed_runs": int(group.completed.sum()),
                "quiescent_runs": int(group.stop_reason.eq("quiescent").sum()),
                "event_budget_runs": int(group.stop_reason.eq("event_budget").sum()),
                "censored_runs": int(group.censored.sum()),
                "stable_terminal_runs": int(group.stop_reason.eq("quiescent").sum()),
                "stop_reason_counts_json": json.dumps(group.stop_reason.value_counts().sort_index().to_dict(), separators=(",", ":")),
            }
        )
    return pd.DataFrame(rows)


def _target_classification(actual: float, target: float, kind: str) -> str:
    precision = 2 if len(str(target).split(".")[-1]) >= 2 else 1
    if round(actual, precision) == round(target, precision):
        return "reproduced_to_display_precision"
    if kind == "aggregation":
        margin = max(0.02, 0.05 * abs(target))
    elif kind == "progress":
        margin = 5.0
    elif kind == "swaps":
        margin = max(50.0, 0.05 * abs(target))
    else:
        raise ValueError(kind)
    return "approximately_reproduced" if abs(actual - target) <= margin else "not_reproduced"


def _claim_classifications(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    costs: pd.DataFrame,
    opposing: pd.DataFrame,
) -> pd.DataFrame:
    claims = pd.read_parquet(S01_CLAIMS)
    claims = claims[claims.figure.astype(str).isin(["8", "9", "10"])]
    if len(claims) != 35:
        raise ValueError("S01 Figure 8-10 claim registry is not the expected 35 rows")
    by_id = summary.set_index("condition_id")
    cost_by_id = costs.set_index("condition_id")
    opposing_by_id = opposing.set_index("condition_id")
    mapping = {
        "Bubble+Insertion": "BUB-INS",
        "Bubble+Selection": "BUB-SEL",
        "Insertion+Selection": "INS-SEL",
        "Selection+Insertion": "INS-SEL",
    }

    def cid(prefix: str, label: str, assignment: str = "EXACT", direction: str = "ASC") -> str:
        return f"C-{prefix}-CHIM-{mapping[label]}-{assignment}-{direction}"

    rows: list[dict[str, Any]] = []
    for claim in claims.to_dict(orient="records"):
        claim_id = claim["claim_id"]
        actual: Any = None
        target: Any = claim.get("reported_mean")
        classification = "not_recoverable"
        rationale = "No identified historical endpoint is available."
        evidence = "clean_room_reference"

        if "COMPLETION" in claim_id:
            label = (
                "Bubble+Insertion+Selection" if "BUBBLE-INSERTION-SELECTION" in claim_id
                else "Bubble+Insertion" if "BUBBLE-INSERTION" in claim_id
                else "Bubble+Selection" if "BUBBLE-SELECTION" in claim_id
                else "Insertion+Selection"
            )
            code = "BUB-INS-SEL" if label == "Bubble+Insertion+Selection" else mapping[label]
            row = by_id.loc[f"C-UNQ-CHIM-{code}-EXACT-ASC"]
            actual = row.completion_fraction * 100
            classification = "supported_clean_room" if row.completed_runs == 100 else "contradicted_clean_room"
            rationale = f"{int(row.completed_runs)}/100 exact-composition R runs completed."
        elif claim_id.startswith("F08-A-CHIMERA"):
            label = (
                "Bubble+Insertion+Selection" if "THREE-WAY" in claim_id
                else "Bubble+Insertion" if "BUBBLE-INSERTION" in claim_id
                else "Bubble+Selection" if "BUBBLE-SELECTION" in claim_id
                else "Insertion+Selection"
            )
            code = "BUB-INS-SEL" if label == "Bubble+Insertion+Selection" else mapping[label]
            row = by_id.loc[f"C-UNQ-CHIM-{code}-EXACT-ASC"]
            actual = float(row.mean_curve_publication_aggregation_peak)
            peak_class = _target_classification(actual, float(target), "aggregation")
            paper_progress = float(str(claim["reported_value_text"]).split("at ")[1].split("%")[0])
            progress_class = _target_classification(float(row.mean_curve_peak_progress_percent), paper_progress, "progress")
            classification = peak_class if peak_class == progress_class else f"peak:{peak_class};progress:{progress_class}"
            rationale = f"Mean-curve peak={actual:.4f} at {row.mean_curve_peak_progress_percent:.0f}% using the paper n denominator."
        elif claim_id.startswith("F08-A-CONTROL"):
            code = (
                "BUB-INS" if "BUBBLE-INSERTION" in claim_id
                else "BUB-SEL" if "BUBBLE-SELECTION" in claim_id
                else "INS-SEL"
            )
            row = by_id.loc[f"C-UNQ-CONTROL-GHOST-{code}"]
            actual = float(row.mean_curve_publication_aggregation_peak)
            classification = _target_classification(actual, float(target), "aggregation")
            rationale = f"Ghost-label Bubble control peak={actual:.4f}; the publication control recipe is contradictory and absent from C."
        elif claim_id == "F08-A-UNIQUE-START-END-BASELINE":
            group = summary[
                summary.condition_id.str.startswith("C-UNQ-CHIM-")
                & summary.condition_id.str.endswith("EXACT-ASC")
            ]
            actual = {
                row.condition_label: [row.mean_initial_publication_aggregation, row.mean_final_publication_aggregation]
                for row in group.itertuples()
            }
            universal = all(
                abs(value - 0.5) <= 0.02 for pair in actual.values() for value in pair
            )
            classification = "supported_clean_room" if universal else "contradicted_clean_room"
            rationale = "The three-type composition-conditioned null is near 0.323, so a universal 0.5 baseline is mathematically incompatible with equal three-way composition."
        elif claim_id == "F08-B-LINEAR-EFFICIENCY":
            actual = bool(costs.between_pure_means.all() and costs.midpoint_equivalent.all())
            classification = "supported_clean_room" if actual else "contradicted_clean_room"
            rationale = f"Between-range held for {int(costs.between_pure_means.sum())}/3 and midpoint equivalence for {int(costs.midpoint_equivalent.sum())}/3 exact pairs."
        elif claim_id.startswith("F08-B-MIX"):
            label = (
                "Bubble+Insertion" if "BUBBLE-INSERTION" in claim_id
                else "Bubble+Selection" if "BUBBLE-SELECTION" in claim_id
                else "Insertion+Selection"
            )
            row = by_id.loc[cid("UNQ", label)]
            actual = float(row.mean_successful_swap_count)
            classification = _target_classification(actual, float(target), "swaps")
            rationale = f"Exact-composition R mean accepted swaps={actual:.2f}."
        elif claim_id.startswith("F08-B-PURE"):
            policy = "Bubble" if "BUBBLE" in claim_id else ("Insertion" if "INSERTION" in claim_id else "Selection")
            code = {"Bubble": "BUB", "Insertion": "INS", "Selection": "SEL"}[policy]
            row = by_id.loc[f"C-UNQ-PURE-CV-{code}-ASC"]
            actual = float(row.mean_successful_swap_count)
            classification = _target_classification(actual, float(target), "swaps")
            rationale = f"Validated S09 R mean accepted swaps={actual:.2f}; historical C remains separate."
        elif claim_id == "F08-C-AGGREGATION-DEFINITION":
            actual = "same-label adjacent edges / n"
            classification = "implemented_with_boundary_decision"
            rationale = "S13 preserves the printed/frozen-code n denominator and also reports the n-1 reference metric."
        elif claim_id.startswith("F08-D-") and claim_id != "F08-E-DUPLICATE-EXAMPLES":
            label = (
                "Bubble+Insertion" if "BUBBLE-INSERTION" in claim_id
                else "Bubble+Selection" if "BUBBLE-SELECTION" in claim_id
                else "Insertion+Selection"
            )
            row = by_id.loc[cid("REP", label)]
            if claim_id.endswith("FINAL"):
                actual = float(row.mean_final_publication_aggregation)
                classification = _target_classification(actual, float(target), "aggregation")
                rationale = f"Exact-composition repeated-value final mean={actual:.4f}."
            else:
                actual = float(row.mean_curve_publication_aggregation_peak)
                peak_class = _target_classification(actual, float(target), "aggregation")
                paper_progress = float(str(claim["reported_value_text"]).split("at ")[1].split("%")[0])
                progress_class = _target_classification(float(row.mean_curve_peak_progress_percent), paper_progress, "progress")
                classification = peak_class if peak_class == progress_class else f"peak:{peak_class};progress:{progress_class}"
                rationale = f"Mean-curve peak={actual:.4f} at {row.mean_curve_peak_progress_percent:.0f}%."
        elif claim_id == "F08-E-DUPLICATE-EXAMPLES":
            values = frame[
                frame.condition_family.eq("same_direction_repeated")
                & frame.assignment_profile.eq("balanced_exact")
            ].final_duplicate_same_algotype_edge_rate.dropna()
            actual = {"min": float(values.min()), "max": float(values.max())}
            classification = "not_recoverable"
            rationale = "The two displayed historical identities are unavailable; clean-room examples span the reported qualitative range but are not substitutes."
        elif claim_id.startswith("F09-"):
            if claim_id.endswith("FINAL-SORTEDNESS"):
                label = "Bubble+Selection" if "F09-A" in claim_id else ("Bubble+Insertion" if "F09-B" in claim_id else "Insertion+Selection")
                row = opposing_by_id.loc[cid("UNQ", label, direction="OPP")]
                actual = float(row.mean_final_paper_sortedness_percent)
                number_class = _target_classification(actual, float(target), "progress")
                classification = number_class if row.trajectory_shape_supported else f"endpoint:{number_class};shape:not_reproduced"
                rationale = f"Exact R final strict Sortedness={actual:.2f}; stop counts {row.stop_reason_counts_json}."
            elif claim_id == "F09-ABC-AGGREGATION-RISE":
                group = opposing[
                    opposing.input_profile.eq("unique_1_100")
                    & opposing.assignment_profile.eq("balanced_exact")
                ]
                actual = group[["condition_label", "mean_publication_aggregation_change"]].to_dict(orient="records")
                classification = "supported_clean_room" if group.aggregation_rise_supported.all() else "contradicted_clean_room"
                rationale = f"Adjusted lower interval exceeded zero for {int(group.aggregation_rise_supported.sum())}/3 exact panels."
            elif claim_id == "F09-ABC-STARTING-SORTEDNESS":
                group = opposing[
                    opposing.input_profile.eq("unique_1_100")
                    & opposing.assignment_profile.eq("balanced_exact")
                ]
                actual = group.mean_initial_paper_sortedness_percent.to_list()
                classification = "supported_clean_room" if all(abs(value - 50) <= 5 for value in actual) else "contradicted_clean_room"
                rationale = "Applied the preregistered +/-5 percentage-point 'near 50' interpretation."
            else:
                group = opposing[
                    opposing.input_profile.eq("unique_1_100")
                    & opposing.assignment_profile.eq("balanced_exact")
                ].set_index("condition_label")
                actual = {
                    label: float(group.loc[label].mean_final_paper_sortedness_percent)
                    for label in ("Bubble+Selection", "Bubble+Insertion", "Insertion+Selection")
                }
                order = actual["Bubble+Selection"] < 50 and actual["Bubble+Insertion"] > 50 and actual["Insertion+Selection"] < 50
                classification = "supported_clean_room" if order else "contradicted_clean_room"
                rationale = "Winner order is a joint sign test under the frozen opposing-direction map."
        elif claim_id.startswith("F10-"):
            label = "Bubble+Selection" if "F10-A" in claim_id else ("Bubble+Insertion" if "F10-B" in claim_id else "Insertion+Selection")
            row = opposing_by_id.loc[cid("REP", label, direction="OPP")]
            actual = {
                "finalStrictSortedness": float(row.mean_final_paper_sortedness_percent),
                "aggregationChange": float(row.mean_publication_aggregation_change),
                "shapeSupported": bool(row.trajectory_shape_supported),
                "referenceShapeSupported": bool(row.reference_trajectory_shape_supported),
                "shapeMetricAgreement": bool(row.duplicate_sortedness_shape_metric_agreement),
                "quiescentRuns": int(row.quiescent_runs),
                "eventBudgetRuns": int(row.event_budget_runs),
            }
            qualitative = (
                row.mean_final_paper_sortedness_percent < 100
                and row.aggregation_rise_supported
                and row.trajectory_shape_supported
                and row.event_budget_runs == 0
            )
            if not row.duplicate_sortedness_shape_metric_agreement:
                classification = "metric_dependent_clean_room_qualitative"
            else:
                classification = "supported_clean_room_qualitative" if qualitative else "constrained_clean_room_qualitative"
            rationale = "No numeric historical endpoint is recoverable; strict-paper and non-strict duplicate Sortedness are retained separately, and event-budget censoring prevents a claim that nothing will change."

        rows.append(
            {
                "claim_id": claim_id,
                "figure": str(claim["figure"]),
                "panel": claim["panel"],
                "claim_group": claim["claim_group"],
                "claim_text": claim["claim_text"],
                "paper_target": target,
                "actual_json": json.dumps(actual, sort_keys=True, separators=(",", ":")) if isinstance(actual, (dict, list)) else actual,
                "classification": classification,
                "evidence_layer": evidence,
                "rationale": rationale,
                "historical_endpoint_available": False,
            }
        )
    return pd.DataFrame(rows).sort_values("claim_id")


def _plot_figure8(
    summary: pd.DataFrame, trajectories: pd.DataFrame, output: Path
) -> None:
    colors = {
        "Bubble+Insertion": "#2d6a9f",
        "Bubble+Selection": "#c74343",
        "Insertion+Selection": "#2b8c6b",
        "Bubble+Insertion+Selection": "#7651a3",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    unique = trajectories[
        trajectories.condition_family.eq("same_direction_unique")
        & trajectories.assignment_profile.eq("balanced_exact")
    ]
    for condition_id, group in unique.groupby("condition_id", sort=True):
        label = _condition_label(condition_id)
        curve = group.groupby("normalized_swap_progress").publication_aggregation.mean()
        axes[0, 0].plot(curve.index * 100, curve.values, label=label, color=colors[label])
    axes[0, 0].set(title="A. Unique values: transient aggregation", ylabel="Publication aggregation (/n)")
    axes[0, 0].legend(fontsize=8)

    bar_order = ("Bubble", "Insertion", "Selection", "Bubble+Insertion", "Bubble+Selection", "Insertion+Selection")
    bar_values = []
    for label in bar_order:
        row = summary[
            summary.condition_label.eq(label)
            & summary.input_profile.eq("unique_1_100")
            & summary.assignment_profile.isin(["pure", "balanced_exact"])
            & ~summary.condition_id.str.contains("OPP")
            & ~summary.condition_id.str.contains("CONTROL")
        ]
        if len(row) != 1:
            raise ValueError(f"Figure 8 cost bar {label} has {len(row)} source rows")
        bar_values.append(float(row.iloc[0].mean_successful_swap_count))
    axes[0, 1].bar(range(len(bar_order)), bar_values, color=["#6d9dc5", "#8bb174", "#d6a65b", "#777777", "#777777", "#777777"])
    axes[0, 1].set_xticks(range(len(bar_order)), [label.replace("+", "+\n") for label in bar_order], fontsize=8)
    axes[0, 1].set(title="B. Accepted-swap cost", ylabel="Mean accepted swaps")

    controls = trajectories[trajectories.condition_family.eq("identical_policy_label_control")]
    for condition_id, group in controls.groupby("condition_id", sort=True):
        label = _condition_label(condition_id)
        curve = group.groupby("normalized_swap_progress").publication_aggregation.mean()
        axes[1, 0].plot(curve.index * 100, curve.values, label=label, color=colors[label])
    axes[1, 0].axhline(0.49, color="black", linestyle=":", linewidth=1, label="exact 50/50 null")
    axes[1, 0].set(title="C. Same-policy ghost-label controls", xlabel="Accepted-swap progress (%)", ylabel="Publication aggregation (/n)")
    axes[1, 0].legend(fontsize=8)

    repeated = trajectories[
        trajectories.condition_family.eq("same_direction_repeated")
        & trajectories.assignment_profile.eq("balanced_exact")
    ]
    for condition_id, group in repeated.groupby("condition_id", sort=True):
        label = _condition_label(condition_id)
        curve = group.groupby("normalized_swap_progress").publication_aggregation.mean()
        axes[1, 1].plot(curve.index * 100, curve.values, label=label, color=colors[label])
    axes[1, 1].set(title="D. Repeated values (1–10, ten copies)", xlabel="Accepted-swap progress (%)", ylabel="Publication aggregation (/n)")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle("Figure 8 clean-room reconstruction — exact composition", fontsize=14)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"figure8_reconstruction.{suffix}", dpi=180)
    plt.close(fig)


def _plot_opposing(trajectories: pd.DataFrame, output: Path, *, repeated: bool) -> None:
    input_profile = "repeated_1_10_x10" if repeated else "unique_1_100"
    figure = 10 if repeated else 9
    source = trajectories[
        trajectories.input_profile.eq(input_profile)
        & trajectories.condition_family.eq("opposite_repeated" if repeated else "opposite_unique")
        & trajectories.assignment_profile.eq("balanced_exact")
    ]
    order = ("Bubble+Selection", "Bubble+Insertion", "Insertion+Selection")
    titles = (
        "Bubble ↓ + Selection ↑",
        "Bubble ↑ + Insertion ↓",
        "Insertion ↑ + Selection ↓",
    )
    paper_targets = (42.5, 73.73, 38.31)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, constrained_layout=True)
    for column, (label, title) in enumerate(zip(order, titles)):
        condition_id = next(
            value for value in source.condition_id.unique() if _condition_label(value) == label
        )
        group = source[source.condition_id.eq(condition_id)]
        curve = group.groupby("normalized_swap_progress").agg(
            sortedness=("paper_sortedness_percent", "mean"),
            reference_sortedness=("reference_sortedness_percent", "mean"),
            aggregation=("publication_aggregation", "mean"),
            warped_sortedness=("warped_paper_sortedness_percent", "mean"),
            warped_aggregation=("warped_publication_aggregation", "mean"),
        )
        x = curve.index.to_numpy() * 100
        axes[0, column].plot(x, curve.sortedness, label="actual accepted-swap progress", color="#2d6a9f")
        if repeated:
            axes[0, column].plot(
                x,
                curve.reference_sortedness,
                label="non-strict duplicate sensitivity",
                color="#7a5195",
                linestyle=":",
            )
        axes[0, column].plot(x, curve.warped_sortedness, label="frozen source 1.25 warp", color="#2d6a9f", linestyle="--", alpha=0.7)
        if not repeated:
            axes[0, column].axhline(paper_targets[column], color="black", linestyle=":", label="paper endpoint")
        axes[0, column].set_title(title)
        axes[1, column].plot(x, curve.aggregation, label="actual accepted-swap progress", color="#c74343")
        axes[1, column].plot(x, curve.warped_aggregation, label="frozen source 1.25 warp", color="#c74343", linestyle="--", alpha=0.7)
        axes[1, column].set_xlabel("Display progress (%)")
        for row in (0, 1):
            axes[row, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel("Strict paper Sortedness (%)")
    axes[1, 0].set_ylabel("Publication aggregation (/n)")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(f"Figure {figure} clean-room reconstruction — exact composition")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"figure{figure}_reconstruction.{suffix}", dpi=180)
    plt.close(fig)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, compression="zstd")


def build_artifacts(
    results: Sequence[Mapping[str, Any]],
    replays: Sequence[Mapping[str, Any]],
    ordinary: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    frame = _results_frame(results)
    trajectories = _trajectory_frame(results)
    traces = _selected_trace_frame(results)
    summary = _condition_summary(frame, trajectories)
    peak_uncertainty = _peak_uncertainty(trajectories)
    assignment_sensitivity = _assignment_sensitivity(frame, summary)
    aggregation_controls = _aggregation_control_contrasts(trajectories, summary)
    costs = _cost_interpolation(frame)
    opposing = _opposing_summary(frame, trajectories)
    claims = _claim_classifications(frame, summary, costs, opposing)

    _write_parquet(frame, output / "chimeric_results.parquet")
    _write_parquet(trajectories, output / "chimeric_trajectories.parquet")
    _write_parquet(traces, output / "selected_swap_traces.parquet")
    _write_parquet(
        frame[[
            "run_id", "condition_id", "scenario_id", "pairing_block_id", "assignment_profile",
            "stop_reason", "swap_only_cost", "publication_swap_plus_comparison_cost",
            "target_calculations", "rejected_actions", "unit_weight_full_ledger",
            *[f"ledger_{key}" for key in LEDGER_FIELDS], "elapsed_seconds",
        ]],
        output / "cost_ledger.parquet",
    )
    summary.to_csv(output / "condition_summary.csv", index=False)
    peak_uncertainty.to_csv(output / "aggregation_peak_uncertainty.csv", index=False)
    assignment_sensitivity.to_csv(output / "assignment_sensitivity.csv", index=False)
    aggregation_controls.to_csv(output / "aggregation_control_contrasts.csv", index=False)
    costs.to_csv(output / "cost_interpolation.csv", index=False)
    opposing.to_csv(output / "opposing_direction_summary.csv", index=False)
    claims[claims.figure.eq("8")].to_csv(output / "figure8_claim_classifications.csv", index=False)
    claims[claims.figure.isin(["9", "10"])].to_csv(output / "figure9_10_claim_classifications.csv", index=False)
    summary[summary.input_profile.eq("repeated_1_10_x10")][[
        "condition_id", "condition_label", "profile_role", "assignment_profile", "runs",
        "completion_fraction", "mean_final_duplicate_same_algotype_edge_rate",
        "mean_final_publication_aggregation", "mean_curve_publication_aggregation_peak",
        "mean_curve_peak_progress_percent", "stop_reason_counts_json",
    ]].to_csv(output / "duplicate_persistence_summary.csv", index=False)

    composition = frame[[
        "run_id", "scenario_id", "condition_id", "condition_family", "input_profile",
        "assignment_profile", "profile_role", "replicate_ordinal", "base_draw_id",
        "pairing_block_id", "policy_counts_json", "clustering_label_counts_json",
        "direction_counts_json", "publication_aggregation_null", "reference_aggregation_null",
    ]].copy()
    composition.to_csv(output / "composition_manifest.csv", index=False)

    original_by_scenario = {row["scenario_id"]: row for row in results}
    replay_rows = []
    replay_keys = (
        "stop_reason", "activation_count", "successful_swap_count", "final_state_hash",
        "trajectory_sha256", "initial_publication_aggregation", "final_publication_aggregation",
        *[f"ledger_{key}" for key in LEDGER_FIELDS],
    )
    for replay in replays:
        original = original_by_scenario[replay["scenario_id"]]
        checks = {key: replay[key] == original[key] for key in replay_keys}
        replay_rows.append(
            {
                "conditionId": replay["condition_id"],
                "scenarioId": replay["scenario_id"],
                "checks": checks,
                "passed": all(checks.values()),
                "trajectorySha256": replay["trajectory_sha256"],
            }
        )
    write_json(
        output / "exact_replay_samples.json",
        {"schema": "e01.s13.exact_replay.v1", "samples": replay_rows, "allPassed": all(row["passed"] for row in replay_rows)},
    )
    write_json(
        output / "ordinary_engine_comparisons.json",
        {"schema": "e01.s13.ordinary_engine_comparison.v1", "samples": list(ordinary), "allPassed": all(row["passed"] for row in ordinary)},
    )

    unavailable_rows = []
    for condition_id in sorted(frame[~frame.condition_family.eq("unique_pure")].condition_id.unique()):
        unavailable_rows.append(
            {
                "condition_id": condition_id,
                "frozen_public_commit": "1fd2bd5921c1f6b423a71f691d5189106a8a1020",
                "historical_endpoint_available": False,
                "historical_actor_algotype_available": False,
                "historical_direction_available": False,
                "historical_activation_ledger_available": False,
                "historical_random_stream_available": False,
                "reason": "raw_Figure_8_10_arrays_absent_and_mixed_driver_outside_S04_validated_adapter",
                "reference_substituted_for_historical": False,
            }
        )
    pd.DataFrame(unavailable_rows).to_csv(output / "historical_endpoint_availability.csv", index=False)

    _plot_figure8(summary, trajectories, output)
    _plot_opposing(trajectories, output, repeated=False)
    _plot_opposing(trajectories, output, repeated=True)

    accounting = {
        "schema": "e01.s13.run_accounting.v1",
        "researchStepId": "S13",
        "executedReferenceRuns": len(results),
        "reusedValidatedS09PureReferenceRuns": 300,
        "totalResultRows": len(frame),
        "trajectoryGridRows": len(trajectories),
        "selectedScenarios": traces.scenario_id.nunique(),
        "selectedTraceRows": len(traces),
        "byFamily": frame.condition_family.value_counts().sort_index().to_dict(),
        "byAssignment": frame.assignment_profile.value_counts().sort_index().to_dict(),
        "byStopReason": frame.stop_reason.value_counts().sort_index().to_dict(),
        "protectedRowsOpened": int(frame.protected.sum()),
        "forbiddenSplitRows": int((frame.split != "paper_scale").sum()),
    }
    write_json(output / "run_accounting.json", accounting)
    return {
        "runs": len(frame),
        "claims": len(claims),
        "accounting": accounting,
    }


def validate_artifacts(output: Path) -> dict[str, Any]:
    frame = pd.read_parquet(output / "chimeric_results.parquet")
    trajectories = pd.read_parquet(output / "chimeric_trajectories.parquet")
    traces = pd.read_parquet(output / "selected_swap_traces.parquet")
    summary = pd.read_csv(output / "condition_summary.csv")
    composition = pd.read_csv(output / "composition_manifest.csv")
    costs = pd.read_csv(output / "cost_interpolation.csv")
    peak_uncertainty = pd.read_csv(output / "aggregation_peak_uncertainty.csv")
    assignment_sensitivity = pd.read_csv(output / "assignment_sensitivity.csv")
    aggregation_controls = pd.read_csv(output / "aggregation_control_contrasts.csv")
    opposing = pd.read_csv(output / "opposing_direction_summary.csv")
    claims = pd.concat(
        [
            pd.read_csv(output / "figure8_claim_classifications.csv"),
            pd.read_csv(output / "figure9_10_claim_classifications.csv"),
        ],
        ignore_index=True,
    )
    replay = json.loads((output / "exact_replay_samples.json").read_text())
    ordinary = json.loads((output / "ordinary_engine_comparisons.json").read_text())
    historical = pd.read_csv(output / "historical_endpoint_availability.csv")
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    check("result_row_count", len(frame) == 3200, len(frame))
    check("run_ids_unique", frame.run_id.nunique() == 3200, frame.run_id.nunique())
    check("executed_run_count", int(frame.evidence_layer.eq("clean_room_reference").sum()) == 2900, int(frame.evidence_layer.eq("clean_room_reference").sum()))
    check("s09_reuse_count", int(frame.evidence_layer.str.contains("reused").sum()) == 300, int(frame.evidence_layer.str.contains("reused").sum()))
    check("paper_scale_only", set(frame.split) == {"paper_scale"}, sorted(frame.split.unique()))
    check("protected_rows_zero", not frame.protected.any(), int(frame.protected.sum()))
    check("trajectory_grid_accounting", len(trajectories) == 292900 and trajectories.scenario_id.nunique() == 2900, [len(trajectories), trajectories.scenario_id.nunique()])
    check("trajectory_grid_complete", trajectories.groupby("scenario_id").size().eq(101).all(), trajectories.groupby("scenario_id").size().value_counts().to_dict())
    check("selected_trace_coverage", traces.scenario_id.nunique() == 29, traces.scenario_id.nunique())
    check("claim_registry_coverage", len(claims) == 35 and claims.claim_id.nunique() == 35, [len(claims), claims.claim_id.nunique()])

    parsed_policy = composition.policy_counts_json.map(json.loads)
    parsed_labels = composition.clustering_label_counts_json.map(json.loads)
    parsed_directions = composition.direction_counts_json.map(json.loads)
    check("all_compositions_sum_to_n", all(sum(item.values()) == 100 for item in parsed_policy), None)
    check("all_clustering_labels_sum_to_n", all(sum(item.values()) == 100 for item in parsed_labels), None)
    check("all_directions_sum_to_n", all(sum(item.values()) == 100 for item in parsed_directions), None)

    exact_pair = composition[
        composition.assignment_profile.eq("balanced_exact")
        & composition.policy_counts_json.map(lambda value: len(json.loads(value)) == 2)
    ]
    check(
        "exact_pairwise_50_50",
        all(sorted(json.loads(value).values()) == [50, 50] for value in exact_pair.policy_counts_json),
        len(exact_pair),
    )
    exact_three = composition[
        composition.assignment_profile.eq("balanced_exact")
        & composition.policy_counts_json.map(lambda value: len(json.loads(value)) == 3)
    ]
    check(
        "exact_three_way_34_33_33",
        len(exact_three) == 100 and all(sorted(json.loads(value).values()) == [33, 33, 34] for value in exact_three.policy_counts_json),
        len(exact_three),
    )
    controls = composition[composition.condition_family.eq("identical_policy_label_control")]
    check(
        "control_policy_and_label_composition",
        len(controls) == 300
        and all(json.loads(value) == {"Bubble": 100} for value in controls.policy_counts_json)
        and all(sorted(json.loads(value).values()) == [50, 50] for value in controls.clustering_label_counts_json),
        len(controls),
    )
    random_rows = composition[composition.assignment_profile.eq("independent_random")]
    random_nonexact = sum(sorted(item.values()) not in ([50, 50], [33, 33, 34]) for item in random_rows.policy_counts_json.map(json.loads))
    check("random_composition_retained_not_forced", random_nonexact > 0, int(random_nonexact))

    opposite = composition[composition.condition_family.isin(["opposite_unique", "opposite_repeated"])]
    check("opposing_direction_encoding", all(set(item) == {"ascending", "descending"} for item in opposite.direction_counts_json.map(json.loads)), len(opposite))
    same = composition[~composition.condition_family.isin(["opposite_unique", "opposite_repeated"])]
    check("same_direction_encoding", all(item == {"ascending": 100} for item in same.direction_counts_json.map(json.loads)), len(same))

    pair_counts = frame.groupby(["condition_id", "pairing_block_id"]).size()
    check("one_run_per_condition_pairing_block", pair_counts.eq(1).all(), pair_counts.value_counts().to_dict())
    condition_counts = frame.groupby("condition_id").size()
    check("one_hundred_runs_per_condition", condition_counts.eq(100).all(), condition_counts.value_counts().to_dict())
    profile_block_counts = frame.groupby(["input_profile", "pairing_block_id"]).condition_id.nunique()
    expected_profiles = {"unique_1_100": 20, "repeated_1_10_x10": 12}
    observed_profile_counts = {
        profile: sorted(group.unique().tolist())
        for profile, group in profile_block_counts.groupby(level=0)
    }
    check("pairing_block_coverage", observed_profile_counts == {key: [value] for key, value in expected_profiles.items()}, observed_profile_counts)

    grid_start = trajectories[trajectories.grid_index.eq(0)].set_index("scenario_id")
    grid_end = trajectories[trajectories.grid_index.eq(100)].set_index("scenario_id")
    executed = frame[frame.evidence_layer.eq("clean_room_reference")].set_index("scenario_id")
    check("trajectory_start_aggregation", np.allclose(grid_start.loc[executed.index].publication_aggregation, executed.initial_publication_aggregation), None)
    check("trajectory_end_aggregation", np.allclose(grid_end.loc[executed.index].publication_aggregation, executed.final_publication_aggregation), None)
    check("trajectory_end_sortedness", np.allclose(grid_end.loc[executed.index].paper_sortedness_percent, executed.final_paper_sortedness_percent), None)

    check("value_multiset_conservation", frame.value_multiset_conserved.all(), int((~frame.value_multiset_conserved).sum()))
    check("no_invariant_error", not frame.stop_reason.eq("invariant_error").any(), frame.stop_reason.value_counts().to_dict())
    same_direction_results = frame[~frame.condition_family.isin(["opposite_unique", "opposite_repeated", "unique_pure"])]
    check("same_direction_all_complete", same_direction_results.completed.all(), int((~same_direction_results.completed).sum()))
    check("same_direction_final_order", same_direction_results.final_consensus_ordered.fillna(False).all(), int((~same_direction_results.final_consensus_ordered.fillna(False)).sum()))

    check("ledger_activations_equal_proposals", (frame.ledger_activations == frame.ledger_proposals).all(), None)
    check("ledger_displacement_identity", (frame.ledger_displacedCells == 2 * frame.ledger_acceptedSwaps).all(), None)
    check("swap_cost_identity", (frame.swap_only_cost == frame.ledger_acceptedSwaps).all(), None)
    check("publication_cost_identity", (frame.publication_swap_plus_comparison_cost == frame.ledger_acceptedSwaps + frame.ledger_valueComparisons).all(), None)
    check("target_calculation_identity", (frame.target_calculations == frame.ledger_proposals).all(), None)
    check("rejected_action_identity", (frame.rejected_actions == frame.ledger_rejections + frame.ledger_conflictLosses).all(), None)
    check("wall_time_separate_and_positive", (frame.elapsed_seconds > 0).all(), float(frame.elapsed_seconds.min()))

    check("exact_replay_all_passed", replay["allPassed"] and len(replay["samples"]) == 29, [replay["allPassed"], len(replay["samples"])])
    check("ordinary_engine_all_passed", ordinary["allPassed"] and len(ordinary["samples"]) == 6, [ordinary["allPassed"], len(ordinary["samples"])])
    check("historical_endpoints_explicitly_unavailable", len(historical) == 29 and not historical.historical_endpoint_available.any() and not historical.reference_substituted_for_historical.any(), len(historical))
    check("cost_interpolation_pairing", len(costs) == 3 and costs.pairs.eq(100).all(), costs.pairs.to_list())
    check("peak_uncertainty_accounting", len(peak_uncertainty) == 17 and peak_uncertainty.runs.eq(100).all() and peak_uncertainty.bootstrap_draws.eq(10000).all(), len(peak_uncertainty))
    check("assignment_sensitivity_pairing", len(assignment_sensitivity) == 13 and assignment_sensitivity.pairs.eq(100).all(), len(assignment_sensitivity))
    check("aggregation_control_pairing", len(aggregation_controls) == 4 and aggregation_controls.pairs.eq(100).all(), len(aggregation_controls))
    check("opposing_profile_accounting", len(opposing) == 12 and opposing.runs.eq(100).all(), len(opposing))

    summary_means = frame.groupby("condition_id").successful_swap_count.mean().sort_index()
    written_means = summary.set_index("condition_id").mean_successful_swap_count.sort_index()
    check("reported_means_recomputed", np.allclose(summary_means, written_means), None)
    figure_paths = [
        output / f"figure{figure}_reconstruction.{suffix}"
        for figure in (8, 9, 10)
        for suffix in ("png", "svg")
    ]
    check("figure_artifacts_present", all(path.is_file() and path.stat().st_size > 0 for path in figure_paths), [path.name for path in figure_paths])

    failed = [name for name, value in checks.items() if not value["passed"]]
    result = {
        "schema": "e01.s13.validation_summary.v1",
        "researchStepId": "S13",
        "success": not failed,
        "zeroUnexplainedFailures": not failed,
        "checkCount": len(checks),
        "failedChecks": failed,
        "checks": checks,
    }
    write_json(output / "validation_summary.json", result)
    if failed:
        raise AssertionError(f"S13 validation failed: {failed}")
    return result


def write_provenance(output: Path) -> dict[str, Any]:
    inputs = {
        "researchPlan": Path("/workspace/RESEARCH_PLAN.md"),
        "fullPlan": Path("/workspace/FULL_PLAN.md"),
        "agents": Path("/workspace/AGENTS.md"),
        "attachmentManifest": Path("/workspace/input-attachments/MANIFEST.json"),
        "attachmentSidecar": Path("/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md"),
        "paperMarkdown": Path("/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/pdf-markdown.md"),
        "claimRegistry": S01_CLAIMS,
        "scenarioBank": S08_DIR / "paired_scenario_bank.parquet",
        "conditionCatalog": S08_DIR / "condition_catalog.parquet",
        "baseDrawBank": S08_DIR / "base_draw_bank.parquet",
        "s09PureRuns": S09_RUNS,
        "preregistration": PREREGISTRATION,
    }
    missing = [name for name, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"provenance inputs missing: {missing}")
    value = {
        "schema": "e01.s13.provenance.v1",
        "researchStepId": "S13",
        "createdUtc": datetime.now(timezone.utc).isoformat(),
        "repositoryHeadAtPackaging": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "frozenPublicCommit": "1fd2bd5921c1f6b423a71f691d5189106a8a1020",
        "publicationSnapshotClaimed": False,
        "historicalMixedDriverExecuted": False,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in inputs.items()
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workers": 8,
            "threadEnvironment": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            },
            "packages": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
                "scipy": stats.__version__ if hasattr(stats, "__version__") else None,
                "matplotlib": matplotlib.__version__,
            },
        },
    }
    write_json(output / "provenance.json", value)
    write_json(output / "environment.json", value["environment"])
    return value


def write_artifact_manifest(output: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    value = {
        "schema": "e01.s13.artifact_manifest.v1",
        "researchStepId": "S13",
        "artifactCount": len(files),
        "artifacts": files,
    }
    write_json(output / "artifact_manifest.json", value)
    return value
