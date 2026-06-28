"""Morphospace metric library for E05 S05.

The metrics are report-facing computational proxies over S01 substrates, S02
identity vectors, S03 targets, and S04 action traces. They are intentionally
deterministic and normalized where practical so later benchmark sweeps can
compare perfect targets, scrambled states, partial repairs, and non-conservative
birth/death/detach states with the same contract.
"""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .identities import CellIdentity, evaluate_neighbor_preferences
from .substrates import CellState, Position, Substrate
from .targets import TargetMorphology, evaluate_target_energy


METRIC_SCHEMA_VERSION = "e05_s05_morphospace_metrics.v1"
TRAJECTORY_METRIC_VERSION = "e05_s05_trajectory_metrics.v1"

METRIC_IDS = (
    "target_energy",
    "target_neighborhood_error",
    "graph_edit_approx",
    "boundary_error",
    "topology_error",
    "shape_moment_error",
    "hausdorff_distance",
    "earth_mover_distance",
    "trajectory_curvature",
)


def _as_position(value: int | Sequence[int]) -> Position:
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(item) for item in value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_ready(item) for item in value)
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def _components(cell: CellIdentity | CellState | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(cell, CellIdentity):
        return dict(cell.components)
    if isinstance(cell, CellState):
        components = dict(cell.identity)
        components.setdefault("scalar_value", cell.value)
        return components
    if "components" in cell and isinstance(cell["components"], Mapping):
        return dict(cell["components"])
    components = dict(cell)
    if "value" in components:
        components.setdefault("scalar_value", components["value"])
    return components


def _cell_id(cell: CellIdentity | CellState | Mapping[str, Any], fallback: int) -> int:
    if isinstance(cell, CellIdentity | CellState):
        return int(cell.cell_id)
    for key in ("cell_id", "cellId"):
        if key in cell:
            return int(cell[key])
    return int(fallback)


def _to_identity(cell: CellIdentity | CellState | Mapping[str, Any], fallback_cell_id: int) -> CellIdentity:
    if isinstance(cell, CellIdentity):
        return cell
    components = _components(cell)
    components.setdefault("ap_coordinate", 0.0)
    components.setdefault("organ_type", "axis")
    components.setdefault("polarity", [1.0, 0.0])
    components.setdefault("adhesion_type", "adhesion_a")
    components.setdefault("target_neighbor_preferences", [])
    components.setdefault("internal_state", {})
    return CellIdentity(cell_id=_cell_id(cell, fallback_cell_id), components=components)


def normalize_observed_state(
    observed_by_position: Mapping[Position, CellIdentity | CellState | Mapping[str, Any]],
) -> dict[Position, CellIdentity]:
    """Convert CellState or mapping payloads into S02 CellIdentity records."""

    normalized: dict[Position, CellIdentity] = {}
    for index, (position, cell) in enumerate(observed_by_position.items()):
        normalized[_as_position(position)] = _to_identity(cell, fallback_cell_id=index)
    return normalized


def _position_array(positions: Sequence[Position]) -> np.ndarray:
    if not positions:
        return np.zeros((0, 2), dtype=float)
    dims = max(len(position) for position in positions)
    rows = []
    for position in positions:
        row = [float(item) for item in position]
        if len(row) < dims:
            row.extend([0.0] * (dims - len(row)))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _bbox_diagonal(positions: Sequence[Position]) -> float:
    array = _position_array(positions)
    if array.size == 0:
        return 1.0
    span = array.max(axis=0) - array.min(axis=0)
    return max(1.0, float(np.linalg.norm(span)))


def _euclidean(left: Position, right: Position) -> float:
    dims = max(len(left), len(right))
    left_row = list(left) + [0] * (dims - len(left))
    right_row = list(right) + [0] * (dims - len(right))
    return float(np.linalg.norm(np.asarray(left_row, dtype=float) - np.asarray(right_row, dtype=float)))


def _metric_record(metric_id: str, value: float, normalized_value: float, detail: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schemaVersion": METRIC_SCHEMA_VERSION,
        "metricId": metric_id,
        "value": float(value),
        "normalizedValue": float(normalized_value),
        "direction": "lower_is_better",
        "zeroMeaning": "exact match for this metric component",
        "detail": _json_ready(dict(detail)),
    }


def metric_catalog() -> list[dict[str, Any]]:
    """Return metric definitions, direction, normalization, and tie behavior."""

    return [
        {
            "schema_version": METRIC_SCHEMA_VERSION,
            "metric_id": "target_energy",
            "direction": "lower_is_better",
            "normalization": "total target-energy divided by target node count",
            "tie_case": "0 means all weighted identity and target-neighborhood constraints are satisfied",
            "description": "Wraps S03 global target-energy evaluator as a report-facing metric.",
        },
        {
            "schema_version": METRIC_SCHEMA_VERSION,
            "metric_id": "target_neighborhood_error",
            "direction": "lower_is_better",
            "normalization": "preference penalties transformed to penalty/(penalty + actor count)",
            "tie_case": "0 means all local target-neighborhood preference rules are satisfied",
            "description": "Measures S02 target-neighborhood rule violations in the observed state.",
        },
        {
            "schema_version": METRIC_SCHEMA_VERSION,
            "metric_id": "graph_edit_approx",
            "direction": "lower_is_better",
            "normalization": "node, edge, and organ-label edit counts divided by target graph size terms",
            "tie_case": "0 means occupied graph and organ-type labels match the target exactly",
            "description": "A deterministic graph-edit approximation; exact graph edit distance is deferred for large graphs.",
        },
        {
            "schema_version": METRIC_SCHEMA_VERSION,
            "metric_id": "boundary_error",
            "direction": "lower_is_better",
            "normalization": "average of organ boundary-label Jaccard error and exposed-shape boundary Jaccard error",
            "tie_case": "0 means boundary labels and exposed occupied boundary positions match",
            "description": "Tracks both target boundary identities and geometric boundary exposure.",
        },
        {
            "schema_version": METRIC_SCHEMA_VERSION,
            "metric_id": "topology_error",
            "direction": "lower_is_better",
            "normalization": "connected-component and hole-count differences divided by target node count",
            "tie_case": "0 means component and hole counts match the target",
            "description": "Counts occupied connected components and integer-lattice holes for 2D states.",
        },
        {
            "schema_version": METRIC_SCHEMA_VERSION,
            "metric_id": "shape_moment_error",
            "direction": "lower_is_better",
            "normalization": "area, centroid, and covariance errors normalized by target size and bounding box",
            "tie_case": "0 means occupied shape moments match",
            "description": "A compact shape proxy based on occupancy count, centroid, and covariance eigenvalues.",
        },
        {
            "schema_version": METRIC_SCHEMA_VERSION,
            "metric_id": "hausdorff_distance",
            "direction": "lower_is_better",
            "normalization": "bidirectional Hausdorff distance divided by target bounding-box diagonal",
            "tie_case": "0 means target and observed occupied position sets are identical",
            "description": "Measures worst-case geometric separation between occupied target and observed nodes.",
        },
        {
            "schema_version": METRIC_SCHEMA_VERSION,
            "metric_id": "earth_mover_distance",
            "direction": "lower_is_better",
            "normalization": "identity-aware assignment cost divided by target bounding-box diagonal",
            "tie_case": "0 means identities occupy their target positions under the scalar/organ matching cost",
            "description": "Approximates how far identity mass must move to recover the target assignment.",
        },
        {
            "schema_version": TRAJECTORY_METRIC_VERSION,
            "metric_id": "trajectory_curvature",
            "direction": "lower_is_better",
            "normalization": "path length divided by direct endpoint distance, minus one",
            "tie_case": "0 means a straight two-point or collinear path; 0 also for stationary paths",
            "description": "Quantifies route inefficiency through a sequence of state-level feature points.",
        },
    ]


def _induced_edges(substrate: Substrate, occupied_positions: set[Position]) -> set[tuple[Position, Position]]:
    return {
        tuple(sorted((left, right)))  # type: ignore[arg-type]
        for left, right in substrate.edges()
        if left in occupied_positions and right in occupied_positions
    }


def graph_edit_approximation(
    target: TargetMorphology,
    observed_by_position: Mapping[Position, CellIdentity | CellState | Mapping[str, Any]],
) -> dict[str, Any]:
    observed = normalize_observed_state(observed_by_position)
    target_nodes = set(target.substrate.nodes)
    observed_nodes = set(observed)
    node_edits = len(target_nodes ^ observed_nodes)
    target_edges = _induced_edges(target.substrate, target_nodes)
    observed_edges = _induced_edges(target.substrate, observed_nodes)
    edge_edits = len(target_edges ^ observed_edges)
    label_edits = 0
    for position in target_nodes & observed_nodes:
        expected_label = target.identities_by_position[position].components.get("organ_type")
        observed_label = observed[position].components.get("organ_type")
        label_edits += int(expected_label != observed_label)
    value = float(node_edits + edge_edits + label_edits)
    denominator = max(1.0, float(len(target_nodes) + len(target_edges) + len(target_nodes)))
    return _metric_record(
        "graph_edit_approx",
        value,
        value / denominator,
        {
            "nodeEdits": node_edits,
            "edgeEdits": edge_edits,
            "organLabelEdits": label_edits,
            "denominator": denominator,
        },
    )


def target_neighborhood_error(
    target: TargetMorphology,
    observed_by_position: Mapping[Position, CellIdentity | CellState | Mapping[str, Any]],
) -> dict[str, Any]:
    observed = normalize_observed_state(observed_by_position)
    total_penalty = 0.0
    missing_actor_count = 0
    rule_count = 0
    for position, expected_identity in target.identities_by_position.items():
        if position not in observed:
            missing_actor_count += 1
            total_penalty += 1.0
            continue
        neighbors = [observed[neighbor] for neighbor in target.substrate.neighbors(position) if neighbor in observed]
        result = evaluate_neighbor_preferences(expected_identity, neighbors)
        total_penalty += float(result["penalty"])
        rule_count += int(result["ruleCount"])
    denominator = max(1.0, total_penalty + len(target.identities_by_position))
    normalized = total_penalty / denominator
    return _metric_record(
        "target_neighborhood_error",
        total_penalty,
        normalized,
        {
            "preferencePenalty": total_penalty,
            "missingActorCount": missing_actor_count,
            "ruleCount": rule_count,
        },
    )


def _organ_positions(
    state: Mapping[Position, CellIdentity],
    organ_type: str,
) -> set[Position]:
    return {
        position
        for position, identity in state.items()
        if identity.components.get("organ_type") == organ_type
    }


def _shape_boundary_positions(substrate: Substrate, occupied_positions: set[Position]) -> set[Position]:
    if not occupied_positions:
        return set()
    max_degree = max((len(substrate.neighbors(position)) for position in substrate.nodes), default=0)
    boundary: set[Position] = set()
    for position in occupied_positions:
        if position not in set(substrate.nodes):
            boundary.add(position)
            continue
        neighbors = set(substrate.neighbors(position))
        if len(neighbors) < max_degree or bool(neighbors - occupied_positions):
            boundary.add(position)
    return boundary


def _jaccard_error(left: set[Position], right: set[Position]) -> float:
    union = left | right
    if not union:
        return 0.0
    return 1.0 - (len(left & right) / len(union))


def boundary_error(
    target: TargetMorphology,
    observed_by_position: Mapping[Position, CellIdentity | CellState | Mapping[str, Any]],
) -> dict[str, Any]:
    observed = normalize_observed_state(observed_by_position)
    target_state = target.target_state()
    target_boundary_labels = _organ_positions(target_state, "boundary")
    observed_boundary_labels = _organ_positions(observed, "boundary")
    target_shape_boundary = _shape_boundary_positions(target.substrate, set(target.substrate.nodes))
    observed_shape_boundary = _shape_boundary_positions(target.substrate, set(observed))
    label_error = _jaccard_error(target_boundary_labels, observed_boundary_labels)
    shape_error = _jaccard_error(target_shape_boundary, observed_shape_boundary)
    value = (label_error + shape_error) / 2.0
    return _metric_record(
        "boundary_error",
        value,
        value,
        {
            "boundaryLabelJaccardError": label_error,
            "shapeBoundaryJaccardError": shape_error,
            "targetBoundaryLabelCount": len(target_boundary_labels),
            "observedBoundaryLabelCount": len(observed_boundary_labels),
            "targetShapeBoundaryCount": len(target_shape_boundary),
            "observedShapeBoundaryCount": len(observed_shape_boundary),
        },
    )


def _component_count(substrate: Substrate, occupied_positions: set[Position]) -> int:
    remaining = {position for position in occupied_positions if position in set(substrate.nodes)}
    extra = occupied_positions - remaining
    count = len(extra)
    while remaining:
        start = remaining.pop()
        count += 1
        queue: deque[Position] = deque([start])
        while queue:
            position = queue.popleft()
            for neighbor in substrate.neighbors(position):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
    return count


def _hole_count_2d(occupied_positions: set[Position]) -> int:
    two_d = {position for position in occupied_positions if len(position) >= 2}
    if not two_d:
        return 0
    xs = [position[0] for position in two_d]
    ys = [position[1] for position in two_d]
    min_x, max_x = min(xs) - 1, max(xs) + 1
    min_y, max_y = min(ys) - 1, max(ys) + 1
    all_cells = {(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)}
    empty = all_cells - {(position[0], position[1]) for position in two_d}
    holes = 0
    while empty:
        start = empty.pop()
        touches_exterior = start[0] in {min_x, max_x} or start[1] in {min_y, max_y}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            touches_exterior = touches_exterior or x in {min_x, max_x} or y in {min_y, max_y}
            for candidate in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if candidate in empty:
                    empty.remove(candidate)
                    queue.append(candidate)
        if not touches_exterior:
            holes += 1
    return holes


def topology_error(
    target: TargetMorphology,
    observed_by_position: Mapping[Position, CellIdentity | CellState | Mapping[str, Any]],
) -> dict[str, Any]:
    observed_positions = set(normalize_observed_state(observed_by_position))
    target_positions = set(target.substrate.nodes)
    target_components = _component_count(target.substrate, target_positions)
    observed_components = _component_count(target.substrate, observed_positions)
    target_holes = _hole_count_2d(target_positions)
    observed_holes = _hole_count_2d(observed_positions)
    value = float(abs(target_components - observed_components) + abs(target_holes - observed_holes))
    normalized = value / max(1.0, float(len(target_positions)))
    return _metric_record(
        "topology_error",
        value,
        normalized,
        {
            "targetComponentCount": target_components,
            "observedComponentCount": observed_components,
            "targetHoleCount": target_holes,
            "observedHoleCount": observed_holes,
        },
    )


def _shape_summary(positions: set[Position]) -> dict[str, Any]:
    ordered = sorted(positions)
    array = _position_array(ordered)
    if array.size == 0:
        return {
            "count": 0,
            "centroid": [],
            "covarianceEigenvalues": [],
        }
    centroid = array.mean(axis=0)
    if len(array) <= 1:
        covariance = np.zeros((array.shape[1], array.shape[1]), dtype=float)
    else:
        covariance = np.cov(array, rowvar=False, bias=True)
        if np.ndim(covariance) == 0:
            covariance = np.asarray([[float(covariance)]], dtype=float)
    eigenvalues = np.linalg.eigvalsh(np.asarray(covariance, dtype=float))
    return {
        "count": int(len(array)),
        "centroid": [float(item) for item in centroid.tolist()],
        "covarianceEigenvalues": [float(item) for item in eigenvalues.tolist()],
    }


def shape_moment_error(
    target: TargetMorphology,
    observed_by_position: Mapping[Position, CellIdentity | CellState | Mapping[str, Any]],
) -> dict[str, Any]:
    observed_positions = set(normalize_observed_state(observed_by_position))
    target_positions = set(target.substrate.nodes)
    target_summary = _shape_summary(target_positions)
    observed_summary = _shape_summary(observed_positions)
    bbox = _bbox_diagonal(tuple(target_positions | observed_positions))
    area_error = abs(target_summary["count"] - observed_summary["count"]) / max(1.0, float(target_summary["count"]))
    if target_summary["centroid"] and observed_summary["centroid"]:
        centroid_error = float(
            np.linalg.norm(np.asarray(target_summary["centroid"], dtype=float) - np.asarray(observed_summary["centroid"], dtype=float))
        ) / bbox
    else:
        centroid_error = 0.0 if target_summary["centroid"] == observed_summary["centroid"] else 1.0
    target_eigs = np.asarray(target_summary["covarianceEigenvalues"], dtype=float)
    observed_eigs = np.asarray(observed_summary["covarianceEigenvalues"], dtype=float)
    if target_eigs.size != observed_eigs.size:
        cov_error = 1.0
    else:
        cov_denom = max(1.0, float(np.linalg.norm(target_eigs)))
        cov_error = float(np.linalg.norm(target_eigs - observed_eigs)) / cov_denom
    value = area_error + centroid_error + cov_error
    normalized = value / 3.0
    return _metric_record(
        "shape_moment_error",
        value,
        normalized,
        {
            "areaError": area_error,
            "centroidError": centroid_error,
            "covarianceError": cov_error,
            "targetShapeSummary": target_summary,
            "observedShapeSummary": observed_summary,
        },
    )


def hausdorff_distance(
    target: TargetMorphology,
    observed_by_position: Mapping[Position, CellIdentity | CellState | Mapping[str, Any]],
) -> dict[str, Any]:
    observed_positions = set(normalize_observed_state(observed_by_position))
    target_positions = set(target.substrate.nodes)
    if not target_positions and not observed_positions:
        value = 0.0
    elif not target_positions or not observed_positions:
        value = _bbox_diagonal(tuple(target_positions | observed_positions))
    else:
        target_list = sorted(target_positions)
        observed_list = sorted(observed_positions)
        target_to_observed = max(min(_euclidean(left, right) for right in observed_list) for left in target_list)
        observed_to_target = max(min(_euclidean(left, right) for right in target_list) for left in observed_list)
        value = max(target_to_observed, observed_to_target)
    bbox = _bbox_diagonal(tuple(target_positions | observed_positions))
    return _metric_record(
        "hausdorff_distance",
        value,
        value / bbox,
        {
            "targetOccupiedCount": len(target_positions),
            "observedOccupiedCount": len(observed_positions),
            "bboxDiagonal": bbox,
        },
    )


def earth_mover_distance(
    target: TargetMorphology,
    observed_by_position: Mapping[Position, CellIdentity | CellState | Mapping[str, Any]],
) -> dict[str, Any]:
    observed = normalize_observed_state(observed_by_position)
    target_items = sorted(target.identities_by_position.items())
    observed_items = sorted(observed.items())
    bbox = _bbox_diagonal(tuple(set(target.substrate.nodes) | set(observed)))
    if not target_items and not observed_items:
        return _metric_record("earth_mover_distance", 0.0, 0.0, {"matchedCount": 0, "unmatchedPenalty": 0.0})
    if not target_items or not observed_items:
        unmatched = max(len(target_items), len(observed_items))
        value = float(unmatched) * bbox
        return _metric_record("earth_mover_distance", value, value / max(1.0, unmatched * bbox), {"matchedCount": 0, "unmatchedPenalty": value})

    cost = np.zeros((len(observed_items), len(target_items)), dtype=float)
    for obs_i, (observed_position, observed_identity) in enumerate(observed_items):
        observed_components = observed_identity.components
        for tgt_i, (target_position, target_identity) in enumerate(target_items):
            target_components = target_identity.components
            spatial = _euclidean(observed_position, target_position)
            scalar_penalty = 0.0
            try:
                scalar_penalty = abs(float(observed_components.get("scalar_value")) - float(target_components.get("scalar_value"))) / max(1.0, len(target_items))
            except (TypeError, ValueError):
                scalar_penalty = 0.0 if observed_components.get("scalar_value") == target_components.get("scalar_value") else 1.0
            organ_penalty = 0.0 if observed_components.get("organ_type") == target_components.get("organ_type") else 1.0
            cost[obs_i, tgt_i] = spatial + scalar_penalty + organ_penalty
    row_ind, col_ind = linear_sum_assignment(cost)
    assignment_cost = float(cost[row_ind, col_ind].sum())
    unmatched_count = abs(len(observed_items) - len(target_items))
    unmatched_penalty = float(unmatched_count) * bbox
    value = assignment_cost + unmatched_penalty
    denominator = max(1.0, float(max(len(observed_items), len(target_items))) * bbox)
    return _metric_record(
        "earth_mover_distance",
        value,
        value / denominator,
        {
            "matchedCount": len(row_ind),
            "unmatchedCount": unmatched_count,
            "assignmentCost": assignment_cost,
            "unmatchedPenalty": unmatched_penalty,
            "bboxDiagonal": bbox,
        },
    )


def target_energy_metric(
    target: TargetMorphology,
    observed_by_position: Mapping[Position, CellIdentity | CellState | Mapping[str, Any]],
) -> dict[str, Any]:
    observed = normalize_observed_state(observed_by_position)
    energy = evaluate_target_energy(target, observed)
    total = float(energy["totalEnergy"])
    normalized = total / max(1.0, float(len(target.substrate.nodes)))
    return _metric_record(
        "target_energy",
        total,
        normalized,
        energy,
    )


def evaluate_morphospace_metrics(
    target: TargetMorphology,
    observed_by_position: Mapping[Position, CellIdentity | CellState | Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the S05 report-facing metric suite for one target/state pair."""

    metrics = [
        target_energy_metric(target, observed_by_position),
        target_neighborhood_error(target, observed_by_position),
        graph_edit_approximation(target, observed_by_position),
        boundary_error(target, observed_by_position),
        topology_error(target, observed_by_position),
        shape_moment_error(target, observed_by_position),
        hausdorff_distance(target, observed_by_position),
        earth_mover_distance(target, observed_by_position),
    ]
    normalized_values = [float(metric["normalizedValue"]) for metric in metrics]
    composite_error = float(np.mean(normalized_values)) if normalized_values else 0.0
    return {
        "schemaVersion": METRIC_SCHEMA_VERSION,
        "targetId": target.target_id,
        "motif": target.motif,
        "metricCount": len(metrics),
        "compositeError": composite_error,
        "metrics": metrics,
    }


def metric_rows(
    target: TargetMorphology,
    case_id: str,
    observed_by_position: Mapping[Position, CellIdentity | CellState | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = evaluate_morphospace_metrics(target, observed_by_position)
    rows = []
    for metric in result["metrics"]:
        rows.append(
            {
                "schema_version": METRIC_SCHEMA_VERSION,
                "target_id": target.target_id,
                "motif": target.motif,
                "case_id": case_id,
                "metric_id": metric["metricId"],
                "value": metric["value"],
                "normalized_value": metric["normalizedValue"],
                "direction": metric["direction"],
                "detail_json": canonical_json(metric["detail"]),
                "composite_error": result["compositeError"],
            }
        )
    return rows


def trajectory_curvature(points: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Compute path curvature from a sequence of metric-space feature points."""

    arrays = [np.asarray(point, dtype=float) for point in points]
    if len(arrays) < 2:
        value = 0.0
        path_length = 0.0
        direct_distance = 0.0
    else:
        path_length = float(sum(np.linalg.norm(arrays[index + 1] - arrays[index]) for index in range(len(arrays) - 1)))
        direct_distance = float(np.linalg.norm(arrays[-1] - arrays[0]))
        if direct_distance == 0.0:
            value = 0.0 if path_length == 0.0 else path_length
        else:
            value = max(0.0, path_length / direct_distance - 1.0)
    return {
        "schemaVersion": TRAJECTORY_METRIC_VERSION,
        "metricId": "trajectory_curvature",
        "value": float(value),
        "normalizedValue": float(value),
        "direction": "lower_is_better",
        "zeroMeaning": "straight or stationary trajectory",
        "detail": {
            "pointCount": len(arrays),
            "pathLength": path_length,
            "directDistance": direct_distance,
        },
    }


def centroid_feature(observed_by_position: Mapping[Position, CellIdentity | CellState | Mapping[str, Any]]) -> tuple[float, ...]:
    """Return an occupancy centroid feature useful for trajectory curvature."""

    positions = sorted(normalize_observed_state(observed_by_position))
    summary = _shape_summary(set(positions))
    return tuple(float(value) for value in summary["centroid"])


def remove_positions(
    state: Mapping[Position, CellIdentity | CellState | Mapping[str, Any]],
    positions: Sequence[Position],
) -> dict[Position, CellIdentity | CellState | Mapping[str, Any]]:
    remove = {_as_position(position) for position in positions}
    return {_as_position(position): cell for position, cell in state.items() if _as_position(position) not in remove}


def rotate_state_180(target: TargetMorphology) -> dict[Position, CellIdentity]:
    """Rotate identity assignment 180 degrees within the target bounding box."""

    positions = sorted(target.identities_by_position)
    if not positions:
        return {}
    dims = max(len(position) for position in positions)
    mins = [min((position[index] if index < len(position) else 0) for position in positions) for index in range(dims)]
    maxs = [max((position[index] if index < len(position) else 0) for position in positions) for index in range(dims)]
    rotated: dict[Position, CellIdentity] = {}
    for position, identity in target.identities_by_position.items():
        padded = list(position) + [0] * (dims - len(position))
        rotated_position = tuple(int(mins[index] + maxs[index] - padded[index]) for index in range(dims))
        if len(position) < dims:
            rotated_position = rotated_position[: len(position)]
        rotated[rotated_position] = identity
    return rotated


def add_extra_cell(
    state: Mapping[Position, CellIdentity | CellState | Mapping[str, Any]],
    position: Position,
    identity: CellIdentity,
) -> dict[Position, CellIdentity | CellState | Mapping[str, Any]]:
    updated = {_as_position(existing_position): cell for existing_position, cell in state.items()}
    updated[_as_position(position)] = identity
    return updated
