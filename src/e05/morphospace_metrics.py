"""Morphospace metrics for E05 S05 target-state evaluation.

The metrics operate on fixed S01 substrate graphs and S03 target morphologies.
Several shape/topology distances are intentionally labeled approximations:
they are compact, deterministic proxies suitable for benchmark sanity checks,
not exact continuous tissue geometry or exact graph-edit distances.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from src.e05.cell_identity import CellIdentity, identity_from_substrate_cell
from src.e05.substrates import SubstrateState
from src.e05.targets import TargetMorphology


EMPTY_LABEL = "__empty__"


@dataclass(frozen=True)
class MorphologyMetricSpec:
    """Metadata and claim boundary for one morphospace metric."""

    metric_id: str
    metric_family: str
    lower_bound: float
    zero_definition: str
    exactness: str
    approximation_label: str
    local_or_global: str
    caveat: str = ""

    def __post_init__(self) -> None:
        if not self.metric_id:
            raise ValueError("metric_id must not be empty")
        if float(self.lower_bound) < 0.0:
            raise ValueError("lower_bound must be non-negative")
        if self.exactness not in {"exact_for_declared_state", "approximation"}:
            raise ValueError(f"unsupported exactness label: {self.exactness}")
        if self.exactness == "approximation" and not self.approximation_label:
            raise ValueError("approximations must carry an approximation_label")

    def compact_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "metric_family": self.metric_family,
            "lower_bound": self.lower_bound,
            "zero_definition": self.zero_definition,
            "exactness": self.exactness,
            "approximation_label": self.approximation_label,
            "local_or_global": self.local_or_global,
            "caveat": self.caveat,
        }


@dataclass(frozen=True)
class MorphologyMetricResult:
    """One evaluated metric value with its specification."""

    spec: MorphologyMetricSpec
    value: float
    target_id: str
    state_label: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def compact_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "state_label": self.state_label,
            "metric_id": self.spec.metric_id,
            "metric_family": self.spec.metric_family,
            "value": float(self.value),
            "lower_bound": self.spec.lower_bound,
            "exactness": self.spec.exactness,
            "approximation_label": self.spec.approximation_label,
            "details": dict(self.details),
        }


MetricFunction = Callable[[TargetMorphology, SubstrateState], tuple[float, Mapping[str, Any]]]


def default_metric_specs() -> tuple[MorphologyMetricSpec, ...]:
    """Return the S05 morphospace metric catalog."""

    return (
        MorphologyMetricSpec(
            "target_identity_error",
            "spatial_identity",
            0.0,
            "Every occupied site carries the identity assigned by the target morphology.",
            "exact_for_declared_state",
            "not_approximate",
            "global_fixed_site",
        ),
        MorphologyMetricSpec(
            "target_neighborhood_error",
            "target_neighborhood",
            0.0,
            "Every site has the same multiset of neighboring target labels as the target state.",
            "exact_for_declared_state",
            "not_approximate",
            "local_neighborhood_aggregated",
        ),
        MorphologyMetricSpec(
            "boundary_error",
            "boundary",
            0.0,
            "The set of sites labeled boundary matches the target boundary set.",
            "exact_for_declared_state",
            "not_approximate",
            "global_fixed_site",
        ),
        MorphologyMetricSpec(
            "topology_component_error",
            "topology",
            0.0,
            "Connected-component counts per label match the target state on the fixed substrate graph.",
            "approximation",
            "connected_component_count_proxy",
            "global_graph",
            "Counts connected components by label; it does not compute full topological equivalence.",
        ),
        MorphologyMetricSpec(
            "shape_moment_error",
            "shape_moments",
            0.0,
            "Per-label centroid and spread moments match the target state.",
            "approximation",
            "centroid_second_moment_proxy",
            "global_coordinate",
            "Low-order moments can miss shape differences with matching centroids and spreads.",
        ),
        MorphologyMetricSpec(
            "hausdorff_distance",
            "hausdorff",
            0.0,
            "Per-label occupied site coordinate sets match the target state.",
            "approximation",
            "discrete_site_coordinate_hausdorff",
            "global_coordinate",
            "Computes Hausdorff distance on discrete site coordinates, not continuous tissue boundaries.",
        ),
        MorphologyMetricSpec(
            "earth_mover_distance",
            "earth_mover",
            0.0,
            "Per-label coordinate distributions match the target state.",
            "approximation",
            "sliced_coordinate_wasserstein_proxy",
            "global_coordinate",
            "Averages independent coordinate-axis Wasserstein distances, not exact multidimensional EMD.",
        ),
        MorphologyMetricSpec(
            "graph_edit_distance_proxy",
            "graph_edit",
            0.0,
            "Node labels and same-label edge relations match the target state on the fixed graph.",
            "approximation",
            "fixed_graph_label_edge_proxy",
            "global_graph",
            "The substrate graph is fixed; this is a labeled-node/edge disagreement proxy, not exact graph edit distance.",
        ),
    )


def metric_spec_rows(specs: Sequence[MorphologyMetricSpec] | None = None) -> list[dict[str, Any]]:
    rows = []
    for spec in specs or default_metric_specs():
        row = spec.compact_dict()
        row["spec_sha256"] = stable_metric_sha256(row)
        rows.append(row)
    return rows


def evaluate_morphology_metrics(
    target: TargetMorphology,
    state: SubstrateState,
    *,
    state_label: str = "state",
    specs: Sequence[MorphologyMetricSpec] | None = None,
) -> tuple[MorphologyMetricResult, ...]:
    """Evaluate all requested S05 metrics for one target/state pair."""

    _validate_state_matches_target(target, state)
    spec_items = tuple(specs or default_metric_specs())
    functions = _metric_functions()
    results = []
    for spec in spec_items:
        if spec.metric_id not in functions:
            raise KeyError(f"no evaluator for metric_id: {spec.metric_id}")
        value, details = functions[spec.metric_id](target, state)
        results.append(
            MorphologyMetricResult(
                spec=spec,
                value=max(0.0, float(value)),
                target_id=target.target_id,
                state_label=state_label,
                details=details,
            )
        )
    return tuple(results)


def aggregate_morphospace_error(results: Sequence[MorphologyMetricResult]) -> float:
    if not results:
        return 0.0
    return float(sum(result.value for result in results) / len(results))


def metric_result_rows(results: Sequence[MorphologyMetricResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        payload = result.compact_dict()
        payload["details_json"] = json.dumps(payload.pop("details"), sort_keys=True, separators=(",", ":"), default=str)
        rows.append(payload)
    return rows


def stable_metric_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def target_identity_error(target: TargetMorphology, state: SubstrateState) -> tuple[float, Mapping[str, Any]]:
    value = target.target_error(state)
    return value, {"identity_distance": value}


def target_neighborhood_error(target: TargetMorphology, state: SubstrateState) -> tuple[float, Mapping[str, Any]]:
    target_labels = target_site_labels(target)
    observed_labels = state_site_labels(target, state)
    per_site = []
    for site_id in target.substrate.site_ids:
        target_counts = collections.Counter(target_labels[neighbor] for neighbor in target.substrate.neighbors(site_id))
        observed_counts = collections.Counter(observed_labels[neighbor] for neighbor in target.substrate.neighbors(site_id))
        keys = set(target_counts) | set(observed_counts)
        mismatch = sum(abs(target_counts[key] - observed_counts[key]) for key in keys)
        denom = max(1, sum(target_counts.values()) + sum(observed_counts.values()))
        per_site.append(mismatch / denom)
    value = float(sum(per_site) / len(per_site)) if per_site else 0.0
    return value, {"site_count": len(per_site)}


def boundary_error(target: TargetMorphology, state: SubstrateState) -> tuple[float, Mapping[str, Any]]:
    target_boundary = _sites_with_label(target_site_labels(target), "boundary")
    observed_boundary = _sites_with_label(state_site_labels(target, state), "boundary")
    union = target_boundary | observed_boundary
    if not union:
        return 0.0, {"target_boundary_count": 0, "observed_boundary_count": 0}
    value = len(target_boundary ^ observed_boundary) / len(union)
    return value, {"target_boundary_count": len(target_boundary), "observed_boundary_count": len(observed_boundary)}


def topology_component_error(target: TargetMorphology, state: SubstrateState) -> tuple[float, Mapping[str, Any]]:
    target_labels = target_site_labels(target)
    observed_labels = state_site_labels(target, state)
    labels = sorted((set(target_labels.values()) | set(observed_labels.values())) - {EMPTY_LABEL})
    if not labels:
        return 0.0, {"labels": []}
    errors = []
    component_counts = {}
    for label in labels:
        target_count = _connected_component_count(target.substrate, target_labels, label)
        observed_count = _connected_component_count(target.substrate, observed_labels, label)
        component_counts[label] = {"target": target_count, "observed": observed_count}
        errors.append(abs(target_count - observed_count) / max(1, target_count, observed_count))
    return float(sum(errors) / len(errors)), {"component_counts": component_counts}


def shape_moment_error(target: TargetMorphology, state: SubstrateState) -> tuple[float, Mapping[str, Any]]:
    target_labels = target_site_labels(target)
    observed_labels = state_site_labels(target, state)
    labels = sorted((set(target_labels.values()) | set(observed_labels.values())) - {EMPTY_LABEL})
    diagonal = _coordinate_diagonal(target.substrate)
    if not labels:
        return 0.0, {"labels": []}
    errors = []
    for label in labels:
        target_coords = _coordinates_for_label(target.substrate, target_labels, label)
        observed_coords = _coordinates_for_label(target.substrate, observed_labels, label)
        if not target_coords and not observed_coords:
            errors.append(0.0)
            continue
        if not target_coords or not observed_coords:
            errors.append(1.0)
            continue
        target_centroid, target_spread = _centroid_and_spread(target_coords)
        observed_centroid, observed_spread = _centroid_and_spread(observed_coords)
        centroid_delta = _euclidean(target_centroid, observed_centroid) / diagonal
        spread_delta = abs(target_spread - observed_spread) / diagonal
        errors.append(min(1.0, 0.5 * centroid_delta + 0.5 * spread_delta))
    return float(sum(errors) / len(errors)), {"label_count": len(labels), "normalizer": diagonal}


def hausdorff_distance(target: TargetMorphology, state: SubstrateState) -> tuple[float, Mapping[str, Any]]:
    target_labels = target_site_labels(target)
    observed_labels = state_site_labels(target, state)
    labels = sorted((set(target_labels.values()) | set(observed_labels.values())) - {EMPTY_LABEL})
    diagonal = _coordinate_diagonal(target.substrate)
    if not labels:
        return 0.0, {"labels": []}
    errors = []
    for label in labels:
        target_coords = _coordinates_for_label(target.substrate, target_labels, label)
        observed_coords = _coordinates_for_label(target.substrate, observed_labels, label)
        errors.append(_symmetric_hausdorff(target_coords, observed_coords) / diagonal)
    return float(sum(errors) / len(errors)), {"label_count": len(labels), "normalizer": diagonal}


def earth_mover_distance(target: TargetMorphology, state: SubstrateState) -> tuple[float, Mapping[str, Any]]:
    target_labels = target_site_labels(target)
    observed_labels = state_site_labels(target, state)
    labels = sorted((set(target_labels.values()) | set(observed_labels.values())) - {EMPTY_LABEL})
    diagonal = _coordinate_diagonal(target.substrate)
    if not labels:
        return 0.0, {"labels": []}
    errors = []
    for label in labels:
        target_coords = _coordinates_for_label(target.substrate, target_labels, label)
        observed_coords = _coordinates_for_label(target.substrate, observed_labels, label)
        if not target_coords and not observed_coords:
            errors.append(0.0)
        elif not target_coords or not observed_coords:
            errors.append(1.0)
        else:
            axis_distances = []
            dims = max(len(target_coords[0]), len(observed_coords[0]))
            for axis in range(dims):
                left = [coord[axis] if axis < len(coord) else 0.0 for coord in target_coords]
                right = [coord[axis] if axis < len(coord) else 0.0 for coord in observed_coords]
                axis_distances.append(_wasserstein_1d(left, right))
            errors.append(min(1.0, math.sqrt(sum(value * value for value in axis_distances)) / diagonal))
    return float(sum(errors) / len(errors)), {"label_count": len(labels), "normalizer": diagonal}


def graph_edit_distance_proxy(target: TargetMorphology, state: SubstrateState) -> tuple[float, Mapping[str, Any]]:
    target_labels = target_site_labels(target)
    observed_labels = state_site_labels(target, state)
    sites = target.substrate.site_ids
    node_mismatch = sum(1 for site_id in sites if target_labels[site_id] != observed_labels[site_id]) / len(sites)
    edges = _undirected_edges(target.substrate)
    if not edges:
        edge_mismatch = 0.0
    else:
        mismatches = 0
        for left, right in edges:
            target_same = target_labels[left] == target_labels[right]
            observed_same = observed_labels[left] == observed_labels[right]
            if target_same != observed_same:
                mismatches += 1
        edge_mismatch = mismatches / len(edges)
    value = 0.5 * node_mismatch + 0.5 * edge_mismatch
    return value, {"node_mismatch": node_mismatch, "edge_mismatch": edge_mismatch, "edge_count": len(edges)}


def target_site_labels(target: TargetMorphology) -> dict[int, str]:
    return {
        site_id: label_from_identity(identity)
        for site_id, identity in target.identities_by_site.items()
    }


def state_site_labels(target: TargetMorphology, state: SubstrateState) -> dict[int, str]:
    _validate_state_matches_target(target, state)
    labels: dict[int, str] = {}
    for site_id in target.substrate.site_ids:
        cell = state.cell_at(site_id)
        if cell is None:
            labels[site_id] = EMPTY_LABEL
            continue
        try:
            labels[site_id] = label_from_identity(identity_from_substrate_cell(cell))
        except ValueError:
            labels[site_id] = str(cell.label)
    return labels


def label_from_identity(identity: CellIdentity) -> str:
    components = identity.components
    if "organ_type" in components:
        return str(components["organ_type"])
    if "value" in components:
        value = float(components["value"])
        return f"value:{value:.12g}"
    for key in sorted(components):
        value = components[key]
        if isinstance(value, (str, int, float)):
            return f"{key}:{value}"
    return identity.identity_id


def reordered_target_state(target: TargetMorphology, site_order: Sequence[int]) -> SubstrateState:
    """Return a state with target cells placed according to ``site_order``.

    ``site_order`` maps target site order to source site IDs.  For example, if
    the first element is 3, site 0 receives the cell identity assigned to target
    site 3.  This is useful for deterministic monotonic sanity fixtures.
    """

    order = tuple(int(item) for item in site_order)
    if set(order) != set(target.substrate.site_ids) or len(order) != len(target.substrate.site_ids):
        raise ValueError("site_order must be a permutation of target site IDs")
    source_state = target.constructed_substrate()
    state = target.substrate.copy_empty()
    cells = []
    for source_site_id in order:
        cell = source_state.cell_at(source_site_id)
        if cell is None:
            raise ValueError("constructed target unexpectedly has an empty source site")
        cells.append(cell)
    state.fill_sites(tuple(cells))
    return state


def swapped_target_state(target: TargetMorphology, left_site_id: int, right_site_id: int) -> SubstrateState:
    order = list(target.substrate.site_ids)
    left_index = order.index(int(left_site_id))
    right_index = order.index(int(right_site_id))
    order[left_index], order[right_index] = order[right_index], order[left_index]
    return reordered_target_state(target, order)


def _metric_functions() -> dict[str, MetricFunction]:
    return {
        "target_identity_error": target_identity_error,
        "target_neighborhood_error": target_neighborhood_error,
        "boundary_error": boundary_error,
        "topology_component_error": topology_component_error,
        "shape_moment_error": shape_moment_error,
        "hausdorff_distance": hausdorff_distance,
        "earth_mover_distance": earth_mover_distance,
        "graph_edit_distance_proxy": graph_edit_distance_proxy,
    }


def _validate_state_matches_target(target: TargetMorphology, state: SubstrateState) -> None:
    if tuple(state.site_ids) != tuple(target.substrate.site_ids):
        raise ValueError("state site IDs do not match target substrate")


def _sites_with_label(labels: Mapping[int, str], label: str) -> set[int]:
    return {site_id for site_id, observed in labels.items() if observed == label}


def _connected_component_count(substrate: SubstrateState, labels: Mapping[int, str], label: str) -> int:
    remaining = _sites_with_label(labels, label)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            site_id = stack.pop()
            for neighbor in substrate.neighbors(site_id):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


def _coordinates_for_label(substrate: SubstrateState, labels: Mapping[int, str], label: str) -> list[tuple[float, ...]]:
    return [tuple(float(item) for item in substrate.coordinate(site_id)) for site_id, observed in labels.items() if observed == label]


def _coordinate_diagonal(substrate: SubstrateState) -> float:
    coords = [tuple(float(item) for item in substrate.coordinate(site_id)) for site_id in substrate.site_ids]
    if len(coords) <= 1:
        return 1.0
    dims = max(len(coord) for coord in coords)
    mins = []
    maxs = []
    for axis in range(dims):
        values = [coord[axis] if axis < len(coord) else 0.0 for coord in coords]
        mins.append(min(values))
        maxs.append(max(values))
    diagonal = math.sqrt(sum((right - left) ** 2 for left, right in zip(mins, maxs, strict=True)))
    return diagonal if diagonal > 0.0 else 1.0


def _centroid_and_spread(coords: Sequence[tuple[float, ...]]) -> tuple[tuple[float, ...], float]:
    dims = max(len(coord) for coord in coords)
    centroid = tuple(
        sum(coord[axis] if axis < len(coord) else 0.0 for coord in coords) / len(coords)
        for axis in range(dims)
    )
    spread = math.sqrt(
        sum(_euclidean(coord, centroid) ** 2 for coord in coords) / len(coords)
    )
    return centroid, spread


def _symmetric_hausdorff(left: Sequence[tuple[float, ...]], right: Sequence[tuple[float, ...]]) -> float:
    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0
    return max(_directed_hausdorff(left, right), _directed_hausdorff(right, left))


def _directed_hausdorff(left: Sequence[tuple[float, ...]], right: Sequence[tuple[float, ...]]) -> float:
    return max(min(_euclidean(a, b) for b in right) for a in left)


def _euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    dims = max(len(left), len(right))
    return math.sqrt(
        sum(
            ((left[axis] if axis < len(left) else 0.0) - (right[axis] if axis < len(right) else 0.0)) ** 2
            for axis in range(dims)
        )
    )


def _wasserstein_1d(left: Sequence[float], right: Sequence[float]) -> float:
    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0
    left_sorted = sorted(float(item) for item in left)
    right_sorted = sorted(float(item) for item in right)
    values = sorted(set(left_sorted + right_sorted))
    if len(values) <= 1:
        return 0.0
    distance = 0.0
    left_index = 0
    right_index = 0
    for low, high in zip(values[:-1], values[1:], strict=True):
        while left_index < len(left_sorted) and left_sorted[left_index] <= low:
            left_index += 1
        while right_index < len(right_sorted) and right_sorted[right_index] <= low:
            right_index += 1
        left_cdf = left_index / len(left_sorted)
        right_cdf = right_index / len(right_sorted)
        distance += abs(left_cdf - right_cdf) * (high - low)
    return distance


def _undirected_edges(substrate: SubstrateState) -> tuple[tuple[int, int], ...]:
    edges = []
    for source in substrate.site_ids:
        for target in substrate.neighbors(source):
            if source < target:
                edges.append((source, target))
    return tuple(edges)
