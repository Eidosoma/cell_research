"""Scrambled target recovery benchmarks for E05 S07.

The S07 simulator is deliberately narrower than the S04 action world: it
benchmarks adjacent local swaps on fully occupied S03 targets after a seeded
identity scramble.  Policies are allowed to use actor/neighbor-visible identity
components, actor-internal neighbor-preference rules, and local substrate
coordinates.  They are not given the whole target identity assignment.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .identities import CellIdentity, evaluate_neighbor_preferences
from .metrics import evaluate_morphospace_metrics, trajectory_curvature
from .substrates import Position, Substrate
from .targets import TargetMorphology, canonical_json


RECOVERY_SCHEMA_VERSION = "e05_s07_scrambled_recovery.v1"
POLICY_AUDIT_VERSION = "e05_s07_policy_leakage_audit.v1"
SCRAMBLE_SCHEMA_VERSION = "e05_s07_scramble_severity.v1"

PROHIBITED_POLICY_PAYLOAD_KEYS = {
    "all_target_positions",
    "all_target_identities",
    "expected_identity_by_position",
    "global_state",
    "global_target_state",
    "hidden",
    "identity_by_position",
    "internal_state",
    "oracle",
    "target_identities",
    "target_map",
    "target_state",
    "whole_target",
}


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


def _sha16(value: Any) -> str:
    payload = json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _as_position(value: int | Sequence[int]) -> Position:
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(item) for item in value)


def _position_array(position: Position, dims: int) -> np.ndarray:
    row = list(position) + [0] * (dims - len(position))
    return np.asarray(row[:dims], dtype=float)


def _euclidean(left: Position, right: Position) -> float:
    dims = max(len(left), len(right))
    return float(np.linalg.norm(_position_array(left, dims) - _position_array(right, dims)))


def _linear_index(substrate: Substrate, position: Position) -> int:
    position = _as_position(position)
    if substrate.substrate_type == "row_1d" and "length" in substrate.metadata:
        return int(position[0])
    if substrate.substrate_type == "square_grid_2d" and {"width", "height"}.issubset(substrate.metadata):
        width = int(substrate.metadata["width"])
        return int(position[1]) * width + int(position[0])
    return {node: index for index, node in enumerate(sorted(substrate.nodes))}[position]


def _linear_index_to_position(substrate: Substrate, index: int) -> Position:
    if substrate.substrate_type == "row_1d" and "length" in substrate.metadata:
        return (max(0, min(int(substrate.metadata["length"]) - 1, int(index))),)
    if substrate.substrate_type == "square_grid_2d" and {"width", "height"}.issubset(substrate.metadata):
        width = int(substrate.metadata["width"])
        height = int(substrate.metadata["height"])
        clipped = max(0, min(width * height - 1, int(index)))
        return (clipped % width, clipped // width)
    nodes = sorted(substrate.nodes)
    return nodes[max(0, min(len(nodes) - 1, int(index)))]


def _position_axis_fraction(substrate: Substrate, position: Position) -> float:
    position = _as_position(position)
    if len(position) == 1:
        length = int(substrate.metadata.get("length", len(substrate.nodes)))
        return float(position[0]) / max(1.0, float(length - 1))
    width = int(substrate.metadata.get("width", max((node[0] for node in substrate.nodes), default=0) + 1))
    return float(position[0]) / max(1.0, float(width - 1))


def _rank_target_index(identity: CellIdentity, substrate: Substrate) -> float:
    scalar = identity.components.get("scalar_value")
    try:
        return float(scalar) - 1.0
    except (TypeError, ValueError):
        ap = float(identity.components.get("ap_coordinate", 0.0))
        return ap * max(1, len(substrate.nodes) - 1)


def _rank_position_error(identity: CellIdentity, position: Position, substrate: Substrate) -> float:
    denominator = max(1.0, float(len(substrate.nodes) - 1))
    return abs(float(_linear_index(substrate, position)) - _rank_target_index(identity, substrate)) / denominator


def _pair_order_error(
    left_identity: CellIdentity,
    left_position: Position,
    right_identity: CellIdentity,
    right_position: Position,
    substrate: Substrate,
) -> float:
    denominator = max(1.0, float(len(substrate.nodes) - 1))
    left_index = _linear_index(substrate, left_position)
    right_index = _linear_index(substrate, right_position)
    left_rank = _rank_target_index(left_identity, substrate)
    right_rank = _rank_target_index(right_identity, substrate)
    if left_index == right_index:
        return 0.0
    if left_index < right_index:
        return max(0.0, left_rank - right_rank) / denominator
    return max(0.0, right_rank - left_rank) / denominator


def _axis_position_error(identity: CellIdentity, position: Position, substrate: Substrate) -> float:
    try:
        ap_coordinate = float(identity.components.get("ap_coordinate", 0.0))
    except (TypeError, ValueError):
        ap_coordinate = 0.0
    return abs(ap_coordinate - _position_axis_fraction(substrate, position))


def _local_preference_score(
    identity: CellIdentity,
    position: Position,
    observed_by_position: Mapping[Position, CellIdentity],
    substrate: Substrate,
) -> float:
    neighbors = [
        observed_by_position[neighbor]
        for neighbor in substrate.neighbors(position)
        if neighbor in observed_by_position
    ]
    result = evaluate_neighbor_preferences(identity, neighbors)
    score = float(result["score"])
    same_adhesion = sum(
        1
        for neighbor in neighbors
        if neighbor.components.get("adhesion_type") == identity.components.get("adhesion_type")
    )
    same_organ = sum(
        1
        for neighbor in neighbors
        if neighbor.components.get("organ_type") == identity.components.get("organ_type")
    )
    degree = max(1.0, float(len(neighbors)))
    return score + 0.25 * float(same_adhesion) / degree + 0.15 * float(same_organ) / degree


@dataclass(frozen=True)
class RecoveryPolicySpec:
    """Local swap policy declaration used by the S07 benchmark."""

    policy_id: str
    family: str
    description: str
    rank_weight: float
    axis_weight: float = 0.0
    affinity_weight: float = 0.0
    exploration_epsilon: float = 0.0
    memory_weight: float = 0.0
    signal_weight: float = 0.0
    source_artifact: str | None = None
    source_policy_id: str | None = None
    source_policy_label: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "schemaVersion": RECOVERY_SCHEMA_VERSION,
            "policyId": self.policy_id,
            "family": self.family,
            "description": self.description,
            "allowedInputs": [
                "actor_visible_identity",
                "adjacent_neighbor_visible_identity",
                "adjacent_neighbor_position",
                "public_substrate_local_geometry",
                "actor_internal_neighbor_preferences",
                "bounded_actor_memory_for_e04_variants",
                "local_signal_at_actor_and_adjacent_neighbor_for_e04_variants",
            ],
            "rankWeight": float(self.rank_weight),
            "axisWeight": float(self.axis_weight),
            "affinityWeight": float(self.affinity_weight),
            "explorationEpsilon": float(self.exploration_epsilon),
            "memoryWeight": float(self.memory_weight),
            "signalWeight": float(self.signal_weight),
            "sourceArtifact": self.source_artifact,
            "sourcePolicyId": self.source_policy_id,
            "sourcePolicyLabel": self.source_policy_label,
            "parameters": dict(self.parameters),
            "leakageBoundary": (
                "No whole target identity map, target-state table, global score, "
                "or nonlocal occupancy view is supplied to the policy."
            ),
        }


def standard_recovery_policy_specs(
    *,
    e03_source: Mapping[str, Any] | None = None,
    e04_source: Mapping[str, Any] | None = None,
) -> tuple[RecoveryPolicySpec, ...]:
    """Return the fixed S07 policy panel."""

    e03_policy_id = None if e03_source is None else str(e03_source.get("policyId", "unknown"))
    e03_label = None if e03_source is None else str(e03_source.get("description", "E03 frontier candidate"))
    e04_policy_id = None if e04_source is None else str(e04_source.get("policyId", "unknown"))
    e04_label = None if e04_source is None else str(e04_source.get("policyLabel", "E04 handoff candidate"))
    return (
        RecoveryPolicySpec(
            policy_id="classic_scalar_rank_swap",
            family="classic_derived",
            description="Adjacent swap if visible scalar rank is closer to the neighbor position after swapping.",
            rank_weight=1.0,
            axis_weight=0.15,
            affinity_weight=0.0,
        ),
        RecoveryPolicySpec(
            policy_id="classic_neighbor_affinity_swap",
            family="classic_derived",
            description="Adjacent swap if actor and neighbor local preference/adhesion scores improve.",
            rank_weight=0.0,
            axis_weight=0.0,
            affinity_weight=1.0,
        ),
        RecoveryPolicySpec(
            policy_id="e03_frontier_rank_affinity",
            family="e03_frontier_feasible",
            description="S07 2D analogue of an E03 local frontier policy: rank-gradient swap with local affinity tie-breaking.",
            rank_weight=0.8,
            axis_weight=0.1,
            affinity_weight=0.25,
            exploration_epsilon=0.01,
            source_artifact=None if e03_source is None else str(e03_source.get("sourceArtifact")),
            source_policy_id=e03_policy_id,
            source_policy_label=e03_label,
            parameters={"catalogBackedAnalogue": e03_source is not None},
        ),
        RecoveryPolicySpec(
            policy_id="e04_memory_signal_rank_affinity",
            family="e04_memory_signal_feasible",
            description="S07 2D analogue of an E04 no-oracle memory/signal policy with rank and local affinity scoring.",
            rank_weight=0.75,
            axis_weight=0.1,
            affinity_weight=0.35,
            exploration_epsilon=0.02,
            memory_weight=0.08,
            signal_weight=0.05,
            source_artifact=None if e04_source is None else str(e04_source.get("sourceArtifact")),
            source_policy_id=e04_policy_id,
            source_policy_label=e04_label,
            parameters={"catalogBackedAnalogue": e04_source is not None, "signalDecay": 0.1, "signalDiffusion": 0.2},
        ),
        RecoveryPolicySpec(
            policy_id="random_local_swap_null",
            family="random_null",
            description="Uniform random adjacent occupied swap with no target or identity scoring.",
            rank_weight=0.0,
            axis_weight=0.0,
            affinity_weight=0.0,
            exploration_epsilon=1.0,
        ),
    )


def _audit_mapping_keys(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in PROHIBITED_POLICY_PAYLOAD_KEYS:
                errors.append(f"{path}.{key_text} exposes prohibited key")
            errors.extend(_audit_mapping_keys(item, f"{path}.{key_text}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_audit_mapping_keys(item, f"{path}[{index}]"))
    return errors


def audit_recovery_policy_payload(policy: RecoveryPolicySpec | Mapping[str, Any]) -> dict[str, Any]:
    """Audit a policy declaration for hidden whole-target leakage."""

    record = policy.to_record() if isinstance(policy, RecoveryPolicySpec) else dict(policy)
    errors = _audit_mapping_keys(record)
    serialized = json.dumps(_json_ready(record), sort_keys=True)
    suspicious_tokens = [
        token
        for token in ("global_target_state", "target_map", "whole_target", "all_target_identities")
        if token in serialized
    ]
    if suspicious_tokens:
        errors.append(f"serialized policy mentions prohibited token(s): {sorted(set(suspicious_tokens))}")
    return {
        "schemaVersion": POLICY_AUDIT_VERSION,
        "policyId": str(record.get("policyId", record.get("policy_id", "unknown"))),
        "success": len(errors) == 0,
        "errorCount": len(errors),
        "errors": errors,
        "payloadHash": _sha16(record),
    }


def _deranged_permutation(size: int, rng: np.random.Generator) -> np.ndarray:
    if size <= 1:
        return np.arange(size, dtype=int)
    permutation = rng.permutation(size)
    fixed = np.flatnonzero(permutation == np.arange(size))
    if len(fixed) == 1:
        left = int(fixed[0])
        right = 0 if left != 0 else 1
        permutation[left], permutation[right] = permutation[right], permutation[left]
    elif len(fixed) > 1:
        permutation[fixed] = permutation[np.roll(fixed, 1)]
    return permutation


def scramble_target_state(
    target: TargetMorphology,
    *,
    seed: int,
) -> tuple[dict[Position, CellIdentity], dict[str, Any]]:
    """Return a seeded severe identity scramble plus severity metadata."""

    rng = np.random.default_rng(int(seed))
    positions = sorted(target.identities_by_position)
    identities = [target.identities_by_position[position] for position in positions]
    permutation = _deranged_permutation(len(positions), rng)
    observed = {position: identities[int(permutation[index])] for index, position in enumerate(positions)}
    severity = scrambling_severity(target, observed, seed=seed, permutation=permutation)
    return observed, severity


def scrambling_severity(
    target: TargetMorphology,
    observed_by_position: Mapping[Position, CellIdentity],
    *,
    seed: int | None = None,
    permutation: Sequence[int] | None = None,
) -> dict[str, Any]:
    target_position_by_cell = {
        identity.cell_id: position
        for position, identity in target.identities_by_position.items()
    }
    observed_position_by_cell = {
        identity.cell_id: position
        for position, identity in observed_by_position.items()
    }
    displacements = []
    linear_displacements = []
    moved = 0
    for cell_id, target_position in target_position_by_cell.items():
        observed_position = observed_position_by_cell[cell_id]
        displacement = _euclidean(target_position, observed_position)
        linear_displacement = abs(_linear_index(target.substrate, target_position) - _linear_index(target.substrate, observed_position))
        displacements.append(displacement)
        linear_displacements.append(float(linear_displacement))
        moved += int(displacement > 0)
    dims = max((len(position) for position in target.substrate.nodes), default=1)
    arrays = [_position_array(position, dims) for position in target.substrate.nodes]
    if arrays:
        stack = np.vstack(arrays)
        span = stack.max(axis=0) - stack.min(axis=0)
        bbox_diagonal = max(1.0, float(np.linalg.norm(span)))
    else:
        bbox_diagonal = 1.0
    metric_result = evaluate_morphospace_metrics(target, observed_by_position)
    metric_by_id = {metric["metricId"]: metric for metric in metric_result["metrics"]}
    mean_displacement = float(np.mean(displacements)) if displacements else 0.0
    moved_fraction = moved / max(1.0, float(len(displacements)))
    normalized = mean_displacement / bbox_diagonal
    if moved_fraction >= 0.9 and normalized >= 0.25:
        severity_class = "severe"
    elif moved_fraction >= 0.5:
        severity_class = "moderate"
    else:
        severity_class = "mild"
    return {
        "schemaVersion": SCRAMBLE_SCHEMA_VERSION,
        "targetId": target.target_id,
        "motif": target.motif,
        "seed": seed,
        "nodeCount": len(target.substrate.nodes),
        "movedCellCount": moved,
        "movedFraction": moved_fraction,
        "meanEuclideanDisplacement": mean_displacement,
        "maxEuclideanDisplacement": float(max(displacements) if displacements else 0.0),
        "meanLinearDisplacement": float(np.mean(linear_displacements)) if linear_displacements else 0.0,
        "maxLinearDisplacement": float(max(linear_displacements) if linear_displacements else 0.0),
        "bboxDiagonal": bbox_diagonal,
        "normalizedMeanEuclideanDisplacement": normalized,
        "initialCompositeError": float(metric_result["compositeError"]),
        "initialTargetEnergy": float(metric_by_id["target_energy"]["value"]),
        "initialEarthMoverDistance": float(metric_by_id["earth_mover_distance"]["value"]),
        "severityClass": severity_class,
        "permutationHash": _sha16(list(permutation) if permutation is not None else sorted((position, cell.cell_id) for position, cell in observed_by_position.items())),
    }


def _state_hash(state: Mapping[Position, CellIdentity]) -> str:
    payload = [
        {"position": list(position), "cellId": identity.cell_id}
        for position, identity in sorted(state.items())
    ]
    return _sha16(payload)


def _metric_lookup(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(metric["metricId"]): metric for metric in result["metrics"]}


def _local_score(
    identity: CellIdentity,
    position: Position,
    state: Mapping[Position, CellIdentity],
    substrate: Substrate,
    policy: RecoveryPolicySpec,
    memory_by_cell: Mapping[int, int],
    signal_by_position: Mapping[Position, float],
) -> float:
    score = 0.0
    if policy.rank_weight:
        score -= float(policy.rank_weight) * _rank_position_error(identity, position, substrate)
    if policy.axis_weight:
        score -= float(policy.axis_weight) * _axis_position_error(identity, position, substrate)
    if policy.affinity_weight:
        score += float(policy.affinity_weight) * _local_preference_score(identity, position, state, substrate)
    if policy.memory_weight:
        score -= float(policy.memory_weight) * min(5, int(memory_by_cell.get(identity.cell_id, 0))) / 5.0
    if policy.signal_weight:
        score -= float(policy.signal_weight) * float(signal_by_position.get(position, 0.0))
    return score


def _candidate_delta(
    state: Mapping[Position, CellIdentity],
    left: Position,
    right: Position,
    substrate: Substrate,
    policy: RecoveryPolicySpec,
    memory_by_cell: Mapping[int, int],
    signal_by_position: Mapping[Position, float],
) -> float:
    left_identity = state[left]
    right_identity = state[right]
    before = (
        _local_score(left_identity, left, state, substrate, policy, memory_by_cell, signal_by_position)
        + _local_score(right_identity, right, state, substrate, policy, memory_by_cell, signal_by_position)
    )
    if policy.rank_weight:
        before -= float(policy.rank_weight) * _pair_order_error(
            left_identity,
            left,
            right_identity,
            right,
            substrate,
        )
    swapped = dict(state)
    swapped[left], swapped[right] = swapped[right], swapped[left]
    after = (
        _local_score(right_identity, left, swapped, substrate, policy, memory_by_cell, signal_by_position)
        + _local_score(left_identity, right, swapped, substrate, policy, memory_by_cell, signal_by_position)
    )
    if policy.rank_weight:
        after -= float(policy.rank_weight) * _pair_order_error(
            right_identity,
            left,
            left_identity,
            right,
            substrate,
        )
    return float(after - before)


def _propose_swap(
    state: Mapping[Position, CellIdentity],
    actor_position: Position,
    substrate: Substrate,
    policy: RecoveryPolicySpec,
    rng: np.random.Generator,
    memory_by_cell: Mapping[int, int],
    signal_by_position: Mapping[Position, float],
) -> tuple[Position | None, float, str]:
    neighbors = [neighbor for neighbor in sorted(substrate.neighbors(actor_position)) if neighbor in state]
    if not neighbors:
        return None, 0.0, "no_occupied_neighbors"
    if policy.policy_id == "random_local_swap_null":
        index = int(rng.integers(0, len(neighbors)))
        return neighbors[index], 0.0, "random_null"

    scored = [
        (
            _candidate_delta(
                state,
                actor_position,
                neighbor,
                substrate,
                policy,
                memory_by_cell,
                signal_by_position,
            ),
            neighbor,
        )
        for neighbor in neighbors
    ]
    best_delta, best_neighbor = max(scored, key=lambda item: (item[0], -_linear_index(substrate, item[1])))
    if best_delta > 1e-12:
        return best_neighbor, float(best_delta), "improving_local_delta"
    if float(policy.exploration_epsilon) > 0 and float(rng.random()) < float(policy.exploration_epsilon):
        index = int(rng.integers(0, len(neighbors)))
        return neighbors[index], float(best_delta), "epsilon_exploration"
    return None, float(best_delta), "no_local_improvement"


def _diffuse_signals(signal_by_position: dict[Position, float], substrate: Substrate, *, decay: float, diffusion: float) -> None:
    if not signal_by_position:
        return
    updated: dict[Position, float] = {}
    for position in substrate.nodes:
        current = float(signal_by_position.get(position, 0.0))
        neighbors = substrate.neighbors(position)
        neighbor_mean = 0.0
        if neighbors:
            neighbor_mean = float(np.mean([signal_by_position.get(neighbor, 0.0) for neighbor in neighbors]))
        updated[position] = max(0.0, current * (1.0 - decay - diffusion) + neighbor_mean * diffusion)
    signal_by_position.update(updated)


def _feature_point(result: Mapping[str, Any]) -> tuple[float, float, float]:
    metrics = _metric_lookup(result)
    return (
        float(result["compositeError"]),
        float(metrics["target_energy"]["normalizedValue"]),
        float(metrics["earth_mover_distance"]["normalizedValue"]),
    )


def _snapshot_row(
    *,
    run_id: str,
    target: TargetMorphology,
    policy: RecoveryPolicySpec,
    seed: int,
    step: int,
    state: Mapping[Position, CellIdentity],
    accepted_swap_count: int,
    wait_count: int,
    reason: str,
) -> dict[str, Any]:
    result = evaluate_morphospace_metrics(target, state)
    metrics = _metric_lookup(result)
    return {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "research_step_id": "S07",
        "run_id": run_id,
        "target_id": target.target_id,
        "motif": target.motif,
        "policy_id": policy.policy_id,
        "policy_family": policy.family,
        "seed": int(seed),
        "step": int(step),
        "state_hash": _state_hash(state),
        "accepted_swap_count": int(accepted_swap_count),
        "wait_count": int(wait_count),
        "snapshot_reason": reason,
        "cell_position_assignment_json": canonical_json(
            [
                {"position": list(position), "cellId": identity.cell_id}
                for position, identity in sorted(state.items())
            ]
        ),
        "composite_error": float(result["compositeError"]),
        "target_energy": float(metrics["target_energy"]["value"]),
        "target_energy_normalized": float(metrics["target_energy"]["normalizedValue"]),
        "target_neighborhood_error": float(metrics["target_neighborhood_error"]["value"]),
        "earth_mover_distance": float(metrics["earth_mover_distance"]["value"]),
        "earth_mover_normalized": float(metrics["earth_mover_distance"]["normalizedValue"]),
        "graph_edit_normalized": float(metrics["graph_edit_approx"]["normalizedValue"]),
        "boundary_error": float(metrics["boundary_error"]["normalizedValue"]),
        "topology_error": float(metrics["topology_error"]["normalizedValue"]),
        "shape_moment_error": float(metrics["shape_moment_error"]["normalizedValue"]),
        "hausdorff_normalized": float(metrics["hausdorff_distance"]["normalizedValue"]),
    }


def run_recovery_benchmark(
    target: TargetMorphology,
    policy: RecoveryPolicySpec,
    *,
    seed: int,
    max_steps: int = 6000,
    snapshot_interval: int = 250,
    recovery_threshold: float = 1e-12,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Run one seeded scrambled-recovery benchmark."""

    rng = np.random.default_rng(int(seed))
    state, severity = scramble_target_state(target, seed=int(seed))
    initial_state_hash = _state_hash(state)
    initial_metrics = evaluate_morphospace_metrics(target, state)
    positions = sorted(state)
    run_id = f"S07::{target.target_id}::{policy.policy_id}::seed{int(seed)}"
    accepted_swap_count = 0
    wait_count = 0
    rejected_count = 0
    exploration_swap_count = 0
    memory_update_count = 0
    signal_update_count = 0
    first_recovery_step: int | None = None
    best_composite = float(initial_metrics["compositeError"])
    best_step = 0
    memory_by_cell: dict[int, int] = {}
    signal_by_position: dict[Position, float] = {position: 0.0 for position in positions}
    trace_rows = [
        _snapshot_row(
            run_id=run_id,
            target=target,
            policy=policy,
            seed=seed,
            step=0,
            state=state,
            accepted_swap_count=accepted_swap_count,
            wait_count=wait_count,
            reason="initial_scrambled",
        )
    ]
    feature_points = [_feature_point(initial_metrics)]

    for step in range(1, int(max_steps) + 1):
        actor_position = positions[int(rng.integers(0, len(positions)))]
        actor_identity = state[actor_position]
        neighbor, delta, reason = _propose_swap(
            state,
            actor_position,
            target.substrate,
            policy,
            rng,
            memory_by_cell,
            signal_by_position,
        )
        if neighbor is None:
            wait_count += 1
            rejected_count += 1
            if policy.memory_weight:
                memory_by_cell[actor_identity.cell_id] = min(100, int(memory_by_cell.get(actor_identity.cell_id, 0)) + 1)
                memory_update_count += 1
            if policy.signal_weight:
                signal_by_position[actor_position] = min(10.0, signal_by_position.get(actor_position, 0.0) + 1.0)
                signal_update_count += 1
        else:
            state = dict(state)
            state[actor_position], state[neighbor] = state[neighbor], state[actor_position]
            accepted_swap_count += 1
            exploration_swap_count += int(reason == "epsilon_exploration" or reason == "random_null")
            if policy.memory_weight:
                memory_by_cell[state[actor_position].cell_id] = max(0, int(memory_by_cell.get(state[actor_position].cell_id, 0)) - 1)
                memory_by_cell[state[neighbor].cell_id] = max(0, int(memory_by_cell.get(state[neighbor].cell_id, 0)) - 1)
                memory_update_count += 2

        if policy.signal_weight and (step % 5 == 0):
            _diffuse_signals(signal_by_position, target.substrate, decay=0.1, diffusion=0.2)
            signal_update_count += len(signal_by_position)

        if step % int(snapshot_interval) == 0 or step == int(max_steps):
            snapshot = _snapshot_row(
                run_id=run_id,
                target=target,
                policy=policy,
                seed=seed,
                step=step,
                state=state,
                accepted_swap_count=accepted_swap_count,
                wait_count=wait_count,
                reason="interval" if step < int(max_steps) else "final",
            )
            trace_rows.append(snapshot)
            feature_points.append(
                (
                    float(snapshot["composite_error"]),
                    float(snapshot["target_energy_normalized"]),
                    float(snapshot["earth_mover_normalized"]),
                )
            )
            if float(snapshot["composite_error"]) < best_composite:
                best_composite = float(snapshot["composite_error"])
                best_step = step
            if first_recovery_step is None and float(snapshot["composite_error"]) <= float(recovery_threshold):
                first_recovery_step = step
                if policy.policy_id != "random_local_swap_null":
                    break

    final_metrics = evaluate_morphospace_metrics(target, state)
    final_lookup = _metric_lookup(final_metrics)
    initial_lookup = _metric_lookup(initial_metrics)
    final_composite = float(final_metrics["compositeError"])
    if first_recovery_step is None and final_composite <= float(recovery_threshold):
        first_recovery_step = step
    curvature = trajectory_curvature(feature_points)
    exact_recovery = final_composite <= float(recovery_threshold)
    improved = final_composite < float(initial_metrics["compositeError"])
    cell_ids_preserved = sorted(identity.cell_id for identity in state.values()) == sorted(
        identity.cell_id for identity in target.identities_by_position.values()
    )
    occupancy_preserved = set(state) == set(target.substrate.nodes)
    row = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "research_step_id": "S07",
        "run_id": run_id,
        "target_id": target.target_id,
        "motif": target.motif,
        "substrate_type": target.substrate.substrate_type,
        "node_count": len(target.substrate.nodes),
        "policy_id": policy.policy_id,
        "policy_family": policy.family,
        "source_policy_id": policy.source_policy_id,
        "source_artifact": policy.source_artifact,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "snapshot_interval": int(snapshot_interval),
        "initial_state_hash": initial_state_hash,
        "final_state_hash": _state_hash(state),
        "initial_composite_error": float(initial_metrics["compositeError"]),
        "final_composite_error": final_composite,
        "delta_composite_error": final_composite - float(initial_metrics["compositeError"]),
        "relative_error_reduction": (
            (float(initial_metrics["compositeError"]) - final_composite) / max(1e-12, float(initial_metrics["compositeError"]))
        ),
        "initial_target_energy": float(initial_lookup["target_energy"]["value"]),
        "final_target_energy": float(final_lookup["target_energy"]["value"]),
        "initial_earth_mover_distance": float(initial_lookup["earth_mover_distance"]["value"]),
        "final_earth_mover_distance": float(final_lookup["earth_mover_distance"]["value"]),
        "initial_target_neighborhood_error": float(initial_lookup["target_neighborhood_error"]["value"]),
        "final_target_neighborhood_error": float(final_lookup["target_neighborhood_error"]["value"]),
        "exact_recovery": bool(exact_recovery),
        "improved": bool(improved),
        "recovery_time_steps": first_recovery_step,
        "best_composite_error": float(best_composite),
        "best_step": int(best_step),
        "accepted_swap_count": int(accepted_swap_count),
        "wait_count": int(wait_count),
        "rejected_count": int(rejected_count),
        "exploration_swap_count": int(exploration_swap_count),
        "memory_update_count": int(memory_update_count),
        "signal_update_count": int(signal_update_count),
        "trajectory_curvature": float(curvature["value"]),
        "trajectory_point_count": int(curvature["detail"]["pointCount"]),
        "occupancy_preserved": bool(occupancy_preserved),
        "cell_ids_preserved": bool(cell_ids_preserved),
        "collision_count": 0,
        "birth_count": 0,
        "death_count": 0,
        "detach_count": 0,
        "conservation_mode": "conservative_adjacent_swaps_only",
        "uses_whole_target_leakage": False,
        "severity_class": severity["severityClass"],
        "scramble_moved_fraction": float(severity["movedFraction"]),
        "scramble_normalized_mean_displacement": float(severity["normalizedMeanEuclideanDisplacement"]),
    }
    return row, trace_rows, severity
