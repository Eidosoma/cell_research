"""Embedded 1D sorting validation helpers for E05 S06.

S06 uses the S01 substrate graph, S02 scalar identities, S03 target
morphologies, S04 swap-cost semantics, and S05 morphospace metrics to check
that an adjacent-swap 1D row behaves identically when embedded in a 2D grid.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.e05.actions import default_action_set
from src.e05.cell_identity import CellIdentity, attach_identity, scalar_identity, scalar_value_schema
from src.e05.morphospace_metrics import aggregate_morphospace_error, evaluate_morphology_metrics
from src.e05.substrates import SubstrateCell, SubstrateState, array_substrate, square_grid_substrate
from src.e05.targets import TargetMetricContract, TargetMorphology, sorted_row_target


ADJACENT_ALGORITHMS = ("bubble", "insertion")
EMBEDDED_ROW_LABEL = "embedded_row"
BARRIER_LABEL = "embedded_barrier"


@dataclass(frozen=True)
class RowSwapOutcome:
    """Compact result for one row-restricted adjacent swap attempt."""

    source_site_id: int
    target_site_id: int
    allowed: bool
    reason: str
    state_changed: bool
    energy_cost_charged: float

    def compact_dict(self) -> dict[str, Any]:
        return {
            "source_site_id": int(self.source_site_id),
            "target_site_id": int(self.target_site_id),
            "allowed": bool(self.allowed),
            "reason": self.reason,
            "state_changed": bool(self.state_changed),
            "energy_cost_charged": float(self.energy_cost_charged),
        }


@dataclass(frozen=True)
class EmbeddedSortResult:
    """One deterministic adjacent-swap sorting trajectory."""

    algorithm: str
    substrate_kind: str
    initial_values: tuple[int | float, ...]
    final_values: tuple[int | float, ...]
    row_site_ids: tuple[int, ...]
    final_state: SubstrateState
    row_target: TargetMorphology
    embedded_target: TargetMorphology | None
    swap_count: int
    comparison_count: int
    energy_total: float
    stop_reason: str
    records: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    row_constraint_violations: int = 0

    @property
    def compare_plus_swap_count(self) -> int:
        return int(self.comparison_count + self.swap_count)

    @property
    def final_sortedness_percent(self) -> float:
        return sortedness_percent(self.final_values)

    @property
    def final_monotonicity_error_count(self) -> int:
        return monotonicity_error_count(self.final_values)


def stable_json_sha256(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sortedness_percent(values: Sequence[int | float], direction: str = "increasing") -> float:
    """E01-compatible adjacent-pair Sortedness percentage."""

    values = tuple(values)
    if not values:
        return 100.0
    if direction in {"increasing", "nondecreasing", "all_increasing"}:
        ordered_pairs = sum(1 for idx in range(1, len(values)) if values[idx - 1] <= values[idx])
    elif direction in {"decreasing", "nonincreasing", "all_decreasing"}:
        ordered_pairs = sum(1 for idx in range(1, len(values)) if values[idx - 1] >= values[idx])
    else:
        raise ValueError(f"unsupported sort direction: {direction}")
    return float(100.0 * (1 + ordered_pairs) / len(values))


def monotonicity_error_count(values: Sequence[int | float], direction: str = "increasing") -> int:
    values = tuple(values)
    if direction in {"increasing", "nondecreasing", "all_increasing"}:
        return int(sum(1 for idx in range(1, len(values)) if values[idx] < values[idx - 1]))
    if direction in {"decreasing", "nonincreasing", "all_decreasing"}:
        return int(sum(1 for idx in range(1, len(values)) if values[idx] > values[idx - 1]))
    raise ValueError(f"unsupported sort direction: {direction}")


def row_site_ids(width: int, row_y: int, *, grid_height: int = 3) -> tuple[int, ...]:
    width = int(width)
    row_y = int(row_y)
    grid_height = int(grid_height)
    if width <= 0:
        raise ValueError("width must be positive")
    if grid_height <= 0:
        raise ValueError("grid_height must be positive")
    if row_y < 0 or row_y >= grid_height:
        raise ValueError("row_y must be inside the grid")
    return tuple(row_y * width + x for x in range(width))


def row_values(state: SubstrateState, row_ids: Sequence[int]) -> tuple[int | float, ...]:
    values: list[int | float] = []
    for site_id in row_ids:
        cell = state.cell_at(int(site_id))
        if cell is None or cell.value is None:
            raise ValueError(f"row site {site_id} is empty or has no scalar value")
        values.append(cell.value)
    return tuple(values)


def row_neighbor_swap_allowed(state: SubstrateState, row_ids: Sequence[int], source_site_id: int, target_site_id: int) -> tuple[bool, str]:
    """Return whether a requested swap is inside the embedded row and adjacent."""

    row_tuple = tuple(int(site_id) for site_id in row_ids)
    row_index = {site_id: idx for idx, site_id in enumerate(row_tuple)}
    source_site_id = int(source_site_id)
    target_site_id = int(target_site_id)
    if source_site_id not in row_index:
        return False, "source_not_in_embedded_row"
    if target_site_id not in row_index:
        return False, "target_not_in_embedded_row"
    if abs(row_index[source_site_id] - row_index[target_site_id]) != 1:
        return False, "target_not_adjacent_in_embedded_row"
    if not state.is_adjacent(source_site_id, target_site_id):
        return False, "target_not_adjacent_in_substrate"
    return True, "allowed"


def execute_row_swap(state: SubstrateState, row_ids: Sequence[int], source_site_id: int, target_site_id: int) -> RowSwapOutcome:
    """Execute one fast row-restricted swap using public S01 state methods."""

    source_site_id = int(source_site_id)
    target_site_id = int(target_site_id)
    row_allowed, row_reason = row_neighbor_swap_allowed(state, row_ids, source_site_id, target_site_id)
    if not row_allowed:
        return RowSwapOutcome(source_site_id, target_site_id, False, row_reason, False, 0.0)
    allowed, reason = state.can_swap(source_site_id, target_site_id)
    if not allowed:
        return RowSwapOutcome(source_site_id, target_site_id, False, reason, False, 0.0)
    source_cell = state.remove_cell(source_site_id)
    target_cell = state.remove_cell(target_site_id)
    state.place_cell(source_site_id, target_cell)
    state.place_cell(target_site_id, source_cell)
    cost = default_action_set()["swap"].energy_cost
    return RowSwapOutcome(source_site_id, target_site_id, True, "allowed", True, cost)


def build_1d_state(values: Sequence[int | float]) -> SubstrateState:
    state = array_substrate(len(values))
    state.fill_sites(tuple(_row_cell(idx, value) for idx, value in enumerate(values)))
    return state


def build_embedded_state(
    values: Sequence[int | float],
    *,
    grid_height: int = 3,
    row_y: int = 1,
    target: TargetMorphology | None = None,
) -> SubstrateState:
    """Return a 2D grid with active row cells and stuck off-row barriers."""

    values = tuple(values)
    target = target or embedded_sorted_row_target(values, grid_height=grid_height, row_y=row_y)
    state = target.substrate.copy_empty()
    row_ids = row_site_ids(len(values), row_y, grid_height=grid_height)
    row_by_site = dict(zip(row_ids, values, strict=True))
    for site_id in state.site_ids:
        if site_id in row_by_site:
            x = row_ids.index(site_id)
            state.place_cell(site_id, _row_cell(x, row_by_site[site_id]))
        else:
            identity = target.identities_by_site[site_id]
            value = float(identity.components["value"])
            state.place_cell(
                site_id,
                attach_identity(
                    SubstrateCell(
                        cell_id=f"barrier_site_{site_id}",
                        value=value,
                        label=BARRIER_LABEL,
                        status="stuck",
                    ),
                    identity,
                ),
            )
    return state


def project_row_state(state: SubstrateState, row_ids: Sequence[int]) -> SubstrateState:
    projected = array_substrate(len(tuple(row_ids)))
    for idx, source_site_id in enumerate(row_ids):
        cell = state.cell_at(int(source_site_id))
        if cell is None:
            raise ValueError(f"row site {source_site_id} is empty")
        projected.place_cell(idx, cell)
    return projected


def embedded_sorted_row_target(
    values: Sequence[int | float],
    *,
    grid_height: int = 3,
    row_y: int = 1,
    target_id: str = "embedded_sorted_row",
) -> TargetMorphology:
    """Return a full 2D target with a sorted active row and fixed barriers."""

    values = tuple(values)
    if not values:
        raise ValueError("values must not be empty")
    width = len(values)
    substrate = square_grid_substrate(width, int(grid_height))
    row_ids = set(row_site_ids(width, int(row_y), grid_height=int(grid_height)))
    sorted_values = tuple(sorted(values))
    barrier_values = {
        site_id: _barrier_value(site_id, width, int(grid_height))
        for site_id in substrate.site_ids
        if site_id not in row_ids
    }
    all_values = (*sorted_values, *barrier_values.values())
    schema = scalar_value_schema(float(min(all_values)), float(max(all_values)))
    identities: dict[int, CellIdentity] = {}
    row_sites = row_site_ids(width, int(row_y), grid_height=int(grid_height))
    for x, site_id in enumerate(row_sites):
        value = sorted_values[x]
        identities[site_id] = scalar_identity(value, identity_id=f"{target_id}_row_{x}_value_{value}")
    for site_id, value in barrier_values.items():
        identities[site_id] = scalar_identity(value, identity_id=f"{target_id}_barrier_site_{site_id}")
    return TargetMorphology(
        target_id=target_id,
        title="Embedded sorted row",
        substrate=substrate,
        schema=schema,
        identities_by_site=identities,
        target_kind="embedded_sorted_row",
        metric_contract=TargetMetricContract(
            primary_metric="mean_scalar_identity_distance",
            zero_error_definition="The middle row contains the sorted values and off-row barrier sites retain fixed identities.",
            compatible_components=("value",),
            notes="S06 continuity target for placing a 1D adjacent-swap row inside a 2D grid.",
        ),
        metadata={
            "input_values": list(values),
            "sorted_values": list(sorted_values),
            "grid_height": int(grid_height),
            "row_y": int(row_y),
            "row_site_ids": list(row_sites),
            "off_row_semantics": "stuck scalar barrier identities",
        },
    )


def run_adjacent_sort(
    values: Sequence[int | float],
    *,
    algorithm: str,
    substrate_kind: str,
    grid_height: int = 3,
    row_y: int = 1,
) -> EmbeddedSortResult:
    """Run a deterministic adjacent-swap Bubble or Insertion trajectory."""

    algorithm = str(algorithm)
    substrate_kind = str(substrate_kind)
    if algorithm not in ADJACENT_ALGORITHMS:
        raise ValueError(f"unsupported S06 adjacent algorithm: {algorithm}")
    values = tuple(values)
    row_target = sorted_row_target(values, target_id=f"s06_sorted_row_{algorithm}")
    embedded_target = None
    if substrate_kind == "array_1d":
        state = build_1d_state(values)
        active_row_ids = tuple(range(len(values)))
    elif substrate_kind == "embedded_square_grid_2d":
        embedded_target = embedded_sorted_row_target(
            values,
            grid_height=grid_height,
            row_y=row_y,
            target_id=f"s06_embedded_sorted_row_{algorithm}",
        )
        state = build_embedded_state(values, grid_height=grid_height, row_y=row_y, target=embedded_target)
        active_row_ids = row_site_ids(len(values), row_y, grid_height=grid_height)
    else:
        raise ValueError(f"unsupported substrate_kind: {substrate_kind}")

    records: list[dict[str, Any]] = []
    comparisons = 0
    swaps = 0
    energy_total = 0.0
    row_constraint_violations = 0
    current_values = list(values)

    def append_record(*, is_initial: bool = False) -> None:
        records.append(
            {
                "algorithm": algorithm,
                "substrate_kind": substrate_kind,
                "swap_step": int(swaps),
                "comparison_count_at_step": int(comparisons),
                "sortedness_percent": sortedness_percent(current_values),
                "monotonicity_error_count": monotonicity_error_count(current_values),
                "is_initial": bool(is_initial),
                "is_final": False,
            }
        )

    append_record(is_initial=True)
    if algorithm == "bubble":
        while True:
            inversion_idx = None
            for idx in range(len(current_values) - 1):
                comparisons += 1
                if current_values[idx] > current_values[idx + 1]:
                    inversion_idx = idx
                    break
            if inversion_idx is None:
                break
            idx = inversion_idx
            while idx < len(active_row_ids) - 1:
                comparisons += 1
                if current_values[idx] <= current_values[idx + 1]:
                    break
                outcome = execute_row_swap(state, active_row_ids, active_row_ids[idx], active_row_ids[idx + 1])
                if not outcome.allowed:
                    row_constraint_violations += 1
                    raise RuntimeError(f"row swap rejected unexpectedly: {outcome.reason}")
                current_values[idx], current_values[idx + 1] = current_values[idx + 1], current_values[idx]
                swaps += 1
                energy_total += outcome.energy_cost_charged
                append_record()
                idx += 1
    else:
        for unsorted_idx in range(1, len(active_row_ids)):
            idx = unsorted_idx
            while idx > 0:
                comparisons += 1
                if current_values[idx - 1] <= current_values[idx]:
                    break
                outcome = execute_row_swap(state, active_row_ids, active_row_ids[idx - 1], active_row_ids[idx])
                if not outcome.allowed:
                    row_constraint_violations += 1
                    raise RuntimeError(f"row swap rejected unexpectedly: {outcome.reason}")
                current_values[idx - 1], current_values[idx] = current_values[idx], current_values[idx - 1]
                swaps += 1
                energy_total += outcome.energy_cost_charged
                append_record()
                idx -= 1
    records[-1]["is_final"] = True
    return EmbeddedSortResult(
        algorithm=algorithm,
        substrate_kind=substrate_kind,
        initial_values=values,
        final_values=tuple(current_values),
        row_site_ids=active_row_ids,
        final_state=state,
        row_target=row_target,
        embedded_target=embedded_target,
        swap_count=swaps,
        comparison_count=comparisons,
        energy_total=energy_total,
        stop_reason="sorted" if monotonicity_error_count(current_values) == 0 else "not_sorted",
        records=tuple(records),
        row_constraint_violations=row_constraint_violations,
    )


def final_metric_summary(result: EmbeddedSortResult) -> dict[str, float]:
    projected = project_row_state(result.final_state, result.row_site_ids)
    row_metrics = evaluate_morphology_metrics(result.row_target, projected, state_label="final_row_projection")
    payload = {
        "final_row_morphospace_error": aggregate_morphospace_error(row_metrics),
        "final_row_target_identity_error": result.row_target.target_error(projected),
    }
    if result.embedded_target is not None:
        embedded_metrics = evaluate_morphology_metrics(result.embedded_target, result.final_state, state_label="final_embedded_state")
        payload["final_embedded_morphospace_error"] = aggregate_morphospace_error(embedded_metrics)
        payload["final_embedded_target_identity_error"] = result.embedded_target.target_error(result.final_state)
    else:
        payload["final_embedded_morphospace_error"] = float("nan")
        payload["final_embedded_target_identity_error"] = float("nan")
    return payload


def run_summary_row(
    result: EmbeddedSortResult,
    *,
    repeat_index: int,
    initial_array_seed: int,
    condition_id: str,
    matched_group_id: str,
    value_bank_id: str,
) -> dict[str, Any]:
    metrics = final_metric_summary(result)
    return {
        "research_step_id": "S06",
        "experiment_id": "E05",
        "source_experiment_id": "E01",
        "condition_id": condition_id,
        "algorithm": result.algorithm,
        "substrate_kind": result.substrate_kind,
        "repeat_index": int(repeat_index),
        "initial_array_seed": int(initial_array_seed),
        "matched_group_id": matched_group_id,
        "value_bank_id": value_bank_id,
        "array_length": len(result.initial_values),
        "initial_array_sha256": stable_json_sha256(list(result.initial_values)),
        "final_array_sha256": stable_json_sha256(list(result.final_values)),
        "swap_only_steps": int(result.swap_count),
        "comparison_steps_observed": int(result.comparison_count),
        "compare_plus_swap_steps": int(result.compare_plus_swap_count),
        "energy_total": float(result.energy_total),
        "final_sortedness_percent": float(result.final_sortedness_percent),
        "final_monotonicity_error_count": int(result.final_monotonicity_error_count),
        "stop_reason": result.stop_reason,
        "row_constraint_violations": int(result.row_constraint_violations),
        **metrics,
    }


def trace_rows(
    result: EmbeddedSortResult,
    *,
    repeat_index: int,
    initial_array_seed: int,
    condition_id: str,
    matched_group_id: str,
    value_bank_id: str,
) -> list[dict[str, Any]]:
    initial_hash = stable_json_sha256(list(result.initial_values))
    final_hash = stable_json_sha256(list(result.final_values))
    rows = []
    for record in result.records:
        rows.append(
            {
                "research_step_id": "S06",
                "experiment_id": "E05",
                "source_experiment_id": "E01",
                "condition_id": condition_id,
                "algorithm": result.algorithm,
                "substrate_kind": result.substrate_kind,
                "matched_group_id": matched_group_id,
                "value_bank_id": value_bank_id,
                "repeat_index": int(repeat_index),
                "initial_array_seed": int(initial_array_seed),
                "swap_step": int(record["swap_step"]),
                "comparison_count_at_step": int(record["comparison_count_at_step"]),
                "sortedness_percent": float(record["sortedness_percent"]),
                "monotonicity_error_count": int(record["monotonicity_error_count"]),
                "is_initial": bool(record["is_initial"]),
                "is_final": bool(record["is_final"]),
                "stop_reason": result.stop_reason,
                "initial_array_sha256": initial_hash,
                "final_array_sha256": final_hash,
            }
        )
    return rows


def _row_cell(idx: int, value: int | float) -> SubstrateCell:
    identity = scalar_identity(value, identity_id=f"row_cell_{idx}_value_{value}")
    return attach_identity(
        SubstrateCell(
            cell_id=f"row_cell_{idx}",
            value=value,
            label=EMBEDDED_ROW_LABEL,
            status="active",
        ),
        identity,
    )


def _barrier_value(site_id: int, width: int, grid_height: int) -> float:
    # Negative, unique, and outside the E01 1..100 row value range.
    _ = width, grid_height
    return float(-(int(site_id) + 1))
