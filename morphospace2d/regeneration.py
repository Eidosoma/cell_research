"""Damage-like perturbation and repair benchmarks for E05 S08.

S08 extends the S07 scrambled-recovery panel to states with missing cells,
frozen patches, rotated grafts, duplicate identities, and inserted foreign
patches.  Repair policies receive only actor-local state, adjacent neighbor
state, public substrate geometry, actor memory/signal summaries, and whether
the actor is off-substrate.  They are not given whole target maps or full
perturbation masks.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .identities import CellIdentity
from .metrics import evaluate_morphospace_metrics, trajectory_curvature
from .recovery import (
    PROHIBITED_POLICY_PAYLOAD_KEYS,
    _candidate_delta,
    _diffuse_signals,
    _json_ready,
    _linear_index,
    _local_score,
    _metric_lookup,
    _sha16,
    _state_hash,
)
from .substrates import Position, Substrate
from .targets import TargetMorphology, canonical_json


REGENERATION_SCHEMA_VERSION = "e05_s08_regeneration.v1"
PERTURBATION_SCHEMA_VERSION = "e05_s08_perturbation.v1"
REGENERATION_POLICY_AUDIT_VERSION = "e05_s08_policy_leakage_audit.v1"

PROHIBITED_REGENERATION_POLICY_KEYS = PROHIBITED_POLICY_PAYLOAD_KEYS | {
    "all_damage_positions",
    "all_frozen_positions",
    "all_graft_transforms",
    "damage_mask",
    "full_perturbation_mask",
    "graft_transform_map",
    "perturbation_map",
    "removed_target_cells",
}


@dataclass(frozen=True)
class PerturbationSpec:
    """Versioned damage-like perturbation recipe."""

    perturbation_id: str
    family: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "schemaVersion": PERTURBATION_SCHEMA_VERSION,
            "perturbationId": self.perturbation_id,
            "family": self.family,
            "description": self.description,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class PerturbedState:
    """Concrete perturbed target state plus mask/provenance metadata."""

    target_id: str
    perturbation: PerturbationSpec
    seed: int
    observed_by_position: Mapping[Position, CellIdentity]
    frozen_positions: frozenset[Position] = frozenset()
    mask_positions: tuple[Position, ...] = ()
    source_positions: tuple[Position, ...] = ()
    target_positions: tuple[Position, ...] = ()
    extra_positions: tuple[Position, ...] = ()
    removed_cell_ids: tuple[int, ...] = ()
    replaced_cell_ids: tuple[int, ...] = ()
    inserted_cell_ids: tuple[int, ...] = ()
    duplicated_cell_ids: tuple[int, ...] = ()
    graft_transform: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self, target: TargetMorphology) -> dict[str, Any]:
        target_nodes = set(target.substrate.nodes)
        observed_nodes = set(self.observed_by_position)
        return {
            "schema_version": PERTURBATION_SCHEMA_VERSION,
            "target_id": self.target_id,
            "motif": target.motif,
            "perturbation_id": self.perturbation.perturbation_id,
            "perturbation_family": self.perturbation.family,
            "seed": int(self.seed),
            "description": self.perturbation.description,
            "parameters_json": canonical_json(self.perturbation.parameters),
            "mask_positions_json": canonical_json(self.mask_positions),
            "source_positions_json": canonical_json(self.source_positions),
            "target_positions_json": canonical_json(self.target_positions),
            "extra_positions_json": canonical_json(self.extra_positions),
            "frozen_positions_json": canonical_json(tuple(sorted(self.frozen_positions))),
            "removed_cell_ids_json": canonical_json(self.removed_cell_ids),
            "replaced_cell_ids_json": canonical_json(self.replaced_cell_ids),
            "inserted_cell_ids_json": canonical_json(self.inserted_cell_ids),
            "duplicated_cell_ids_json": canonical_json(self.duplicated_cell_ids),
            "graft_transform_json": canonical_json(self.graft_transform),
            "target_node_count": len(target.substrate.nodes),
            "observed_position_count": len(self.observed_by_position),
            "missing_position_count": len(target_nodes - observed_nodes),
            "extra_position_count": len(observed_nodes - target_nodes),
            "cell_count_delta": len(self.observed_by_position) - len(target.substrate.nodes),
            "requires_nonconservative_repair": self.perturbation.family in {"contiguous_chunk_removal", "insert_foreign_patch"},
            "state_hash": _state_hash(self.observed_by_position),
        }


@dataclass(frozen=True)
class RegenerationPolicySpec:
    """Local repair policy declaration for S08."""

    policy_id: str
    family: str
    description: str
    rank_weight: float
    axis_weight: float = 0.0
    affinity_weight: float = 0.0
    exploration_epsilon: float = 0.0
    memory_weight: float = 0.0
    signal_weight: float = 0.0
    allow_divide: bool = False
    allow_die: bool = False
    allow_rotate: bool = False
    source_artifact: str | None = None
    source_policy_id: str | None = None
    source_policy_label: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        allowed_actions = ["wait", "swap", "crawl"]
        if self.allow_divide:
            allowed_actions.append("divide")
        if self.allow_die:
            allowed_actions.append("die")
        if self.allow_rotate:
            allowed_actions.append("rotate")
        return {
            "schemaVersion": REGENERATION_SCHEMA_VERSION,
            "policyId": self.policy_id,
            "family": self.family,
            "description": self.description,
            "allowedActions": allowed_actions,
            "allowedInputs": [
                "actor_visible_identity",
                "actor_position",
                "actor_off_substrate_flag",
                "actor_frozen_flag",
                "adjacent_neighbor_visible_identity",
                "adjacent_neighbor_position",
                "adjacent_empty_or_occupied_flag",
                "adjacent_frozen_flag",
                "public_substrate_local_geometry",
                "actor_internal_neighbor_preferences",
                "bounded_actor_memory_for_memory_signal_variants",
                "local_signal_at_actor_and_adjacent_neighbor_for_memory_signal_variants",
            ],
            "rankWeight": float(self.rank_weight),
            "axisWeight": float(self.axis_weight),
            "affinityWeight": float(self.affinity_weight),
            "explorationEpsilon": float(self.exploration_epsilon),
            "memoryWeight": float(self.memory_weight),
            "signalWeight": float(self.signal_weight),
            "allowDivide": bool(self.allow_divide),
            "allowDie": bool(self.allow_die),
            "allowRotate": bool(self.allow_rotate),
            "sourceArtifact": self.source_artifact,
            "sourcePolicyId": self.source_policy_id,
            "sourcePolicyLabel": self.source_policy_label,
            "parameters": dict(self.parameters),
            "leakageBoundary": (
                "No full perturbation mask, whole target identity map, target-state table, "
                "or global score is supplied to the repair policy."
            ),
        }


def _audit_mapping_keys(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in PROHIBITED_REGENERATION_POLICY_KEYS:
                errors.append(f"{path}.{key_text} exposes prohibited key")
            errors.extend(_audit_mapping_keys(item, f"{path}.{key_text}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_audit_mapping_keys(item, f"{path}[{index}]"))
    return errors


def audit_regeneration_policy_payload(policy: RegenerationPolicySpec | Mapping[str, Any]) -> dict[str, Any]:
    """Audit an S08 policy declaration for target or perturbation leakage."""

    record = policy.to_record() if isinstance(policy, RegenerationPolicySpec) else dict(policy)
    errors = _audit_mapping_keys(record)
    serialized = json.dumps(_json_ready(record), sort_keys=True)
    suspicious = [
        token
        for token in (
            "global_target_state",
            "target_map",
            "whole_target",
            "all_target_identities",
            "full_perturbation_mask",
            "damage_mask",
            "graft_transform_map",
        )
        if token in serialized
    ]
    if suspicious:
        errors.append(f"serialized policy mentions prohibited token(s): {sorted(set(suspicious))}")
    return {
        "schemaVersion": REGENERATION_POLICY_AUDIT_VERSION,
        "policyId": str(record.get("policyId", record.get("policy_id", "unknown"))),
        "success": len(errors) == 0,
        "errorCount": len(errors),
        "errors": errors,
        "payloadHash": _sha16(record),
    }


def standard_perturbation_specs() -> tuple[PerturbationSpec, ...]:
    return (
        PerturbationSpec(
            "remove_center_chunk",
            "contiguous_chunk_removal",
            "Remove a central contiguous chunk, creating missing substrate positions that require growth-like filling for full occupancy.",
            {"width": 2, "height": 2, "rowLength": 2},
        ),
        PerturbationSpec(
            "freeze_center_patch",
            "freeze_patch",
            "Freeze a central patch so local repair policies cannot move those cells or swap through them.",
            {"width": 2, "height": 2, "rowLength": 2},
        ),
        PerturbationSpec(
            "rotate_center_graft",
            "rotate_graft",
            "Rotate a central graft 180 degrees and rotate graft-cell polarity vectors by the same amount.",
            {"width": 3, "height": 3, "rowLength": 3, "quarterTurns": 2},
        ),
        PerturbationSpec(
            "duplicate_left_region_into_right",
            "duplicate_region",
            "Duplicate a left-side source patch into a right-side target patch using new cell IDs, replacing the original target-patch cells.",
            {"width": 2, "height": 2, "rowLength": 2},
        ),
        PerturbationSpec(
            "insert_foreign_patch_outside",
            "insert_foreign_patch",
            "Insert a small foreign patch just outside the target substrate to test extra-cell metrics and local off-substrate pruning.",
            {"width": 2, "height": 2, "rowLength": 2},
        ),
    )


def standard_regeneration_policy_specs(
    *,
    e03_source: Mapping[str, Any] | None = None,
    e04_source: Mapping[str, Any] | None = None,
) -> tuple[RegenerationPolicySpec, ...]:
    e03_policy_id = None if e03_source is None else str(e03_source.get("policyId", "unknown"))
    e03_label = None if e03_source is None else str(e03_source.get("description", "E03 frontier candidate"))
    e04_policy_id = None if e04_source is None else str(e04_source.get("policyId", "unknown"))
    e04_label = None if e04_source is None else str(e04_source.get("policyLabel", "E04 handoff candidate"))
    return (
        RegenerationPolicySpec(
            policy_id="classic_rank_swap_crawl_repair",
            family="classic_derived",
            description="Visible scalar-rank local swap/crawl repair without non-conservative growth or pruning.",
            rank_weight=1.0,
            axis_weight=0.15,
            affinity_weight=0.0,
        ),
        RegenerationPolicySpec(
            policy_id="classic_growth_prune_rank_repair",
            family="classic_derived_growth_prune",
            description="Scalar-rank local repair that can divide into adjacent empty substrate nodes and die when off-substrate.",
            rank_weight=0.9,
            axis_weight=0.12,
            affinity_weight=0.1,
            allow_divide=True,
            allow_die=True,
            allow_rotate=True,
            parameters={"growthBias": 0.2, "offSubstrateDeathRule": "actor_local"},
        ),
        RegenerationPolicySpec(
            policy_id="e03_frontier_repair",
            family="e03_frontier_feasible",
            description="S08 analogue of an E03 frontier local policy with rank, crawl, and affinity repair.",
            rank_weight=0.8,
            axis_weight=0.1,
            affinity_weight=0.25,
            exploration_epsilon=0.01,
            allow_divide=True,
            source_artifact=None if e03_source is None else str(e03_source.get("sourceArtifact")),
            source_policy_id=e03_policy_id,
            source_policy_label=e03_label,
            parameters={"catalogBackedAnalogue": e03_source is not None, "growthBias": 0.1},
        ),
        RegenerationPolicySpec(
            policy_id="e04_memory_signal_regeneration",
            family="e04_memory_signal_feasible",
            description="S08 analogue of an E04 no-oracle memory/signal repair policy with local growth and off-substrate pruning.",
            rank_weight=0.75,
            axis_weight=0.1,
            affinity_weight=0.35,
            exploration_epsilon=0.02,
            memory_weight=0.08,
            signal_weight=0.05,
            allow_divide=True,
            allow_die=True,
            allow_rotate=True,
            source_artifact=None if e04_source is None else str(e04_source.get("sourceArtifact")),
            source_policy_id=e04_policy_id,
            source_policy_label=e04_label,
            parameters={"catalogBackedAnalogue": e04_source is not None, "signalDecay": 0.1, "signalDiffusion": 0.2},
        ),
        RegenerationPolicySpec(
            policy_id="random_local_repair_null",
            family="random_null",
            description="Uniform random legal local swap/crawl move; no division, death, target, or identity scoring.",
            rank_weight=0.0,
            axis_weight=0.0,
            affinity_weight=0.0,
            exploration_epsilon=1.0,
        ),
    )


def _dims(target: TargetMorphology) -> tuple[int, int | None]:
    metadata = target.substrate.metadata
    if target.substrate.substrate_type == "row_1d":
        return int(metadata["length"]), None
    return int(metadata.get("width", 1)), int(metadata.get("height", 1))


def _center_patch(target: TargetMorphology, *, width: int, height: int, row_length: int) -> tuple[Position, ...]:
    x_count, y_count = _dims(target)
    if y_count is None:
        length = min(row_length, x_count)
        start = max(0, (x_count - length) // 2)
        return tuple((x,) for x in range(start, start + length))
    patch_width = min(width, x_count)
    patch_height = min(height, y_count)
    x0 = max(0, (x_count - patch_width) // 2)
    y0 = max(0, (y_count - patch_height) // 2)
    return tuple((x, y) for y in range(y0, y0 + patch_height) for x in range(x0, x0 + patch_width))


def _side_patches(target: TargetMorphology, *, width: int, height: int, row_length: int) -> tuple[tuple[Position, ...], tuple[Position, ...]]:
    x_count, y_count = _dims(target)
    if y_count is None:
        length = min(row_length, max(1, x_count // 3))
        source = tuple((x,) for x in range(0, length))
        target_patch = tuple((x_count - length + x,) for x in range(length))
        return source, target_patch
    patch_width = min(width, max(1, x_count // 3))
    patch_height = min(height, y_count)
    y0 = max(0, (y_count - patch_height) // 2)
    source = tuple((x, y) for y in range(y0, y0 + patch_height) for x in range(0, patch_width))
    target_patch = tuple((x_count - patch_width + x, y) for y in range(y0, y0 + patch_height) for x in range(patch_width))
    return source, target_patch


def _extra_patch(target: TargetMorphology, *, width: int, height: int, row_length: int) -> tuple[Position, ...]:
    x_count, y_count = _dims(target)
    if y_count is None:
        return tuple((x_count + x,) for x in range(min(row_length, 2)))
    patch_width = min(width, 2)
    patch_height = min(height, max(1, y_count))
    y0 = max(0, (y_count - patch_height) // 2)
    return tuple((x_count + x, y) for y in range(y0, y0 + patch_height) for x in range(patch_width))


def _rotated_positions(patch: Sequence[Position]) -> dict[Position, Position]:
    if not patch:
        return {}
    dims = max(len(position) for position in patch)
    mins = [min((position[index] if index < len(position) else 0) for position in patch) for index in range(dims)]
    maxs = [max((position[index] if index < len(position) else 0) for position in patch) for index in range(dims)]
    mapping: dict[Position, Position] = {}
    for position in patch:
        padded = list(position) + [0] * (dims - len(position))
        rotated = tuple(int(mins[index] + maxs[index] - padded[index]) for index in range(dims))
        mapping[position] = rotated[: len(position)]
    return mapping


def _rotate_polarity(components: Mapping[str, Any], quarter_turns: int) -> dict[str, Any]:
    updated = dict(components)
    polarity = updated.get("polarity")
    if not isinstance(polarity, Sequence) or len(polarity) < 2:
        return updated
    vector = [float(polarity[0]), float(polarity[1])]
    for _ in range(int(quarter_turns) % 4):
        vector = [-vector[1], vector[0]]
    updated["polarity"] = [round(vector[0], 12), round(vector[1], 12)]
    return updated


def _clone_identity(identity: CellIdentity, *, cell_id: int, component_updates: Mapping[str, Any] | None = None) -> CellIdentity:
    components = dict(identity.components)
    components.update(dict(component_updates or {}))
    return CellIdentity(cell_id=int(cell_id), components=components)


def _foreign_identity(cell_id: int, index: int) -> CellIdentity:
    return CellIdentity(
        cell_id=int(cell_id),
        components={
            "scalar_value": 10_000 + int(index),
            "ap_coordinate": 1.25,
            "organ_type": "appendage",
            "polarity": [-1.0, 0.0],
            "adhesion_type": "adhesion_b",
            "target_neighbor_preferences": [],
            "internal_state": {"s08_foreign_patch": True},
        },
    )


def apply_perturbation(
    target: TargetMorphology,
    spec: PerturbationSpec,
    *,
    seed: int = 0,
) -> PerturbedState:
    """Apply one deterministic S08 perturbation to a target state."""

    rng = np.random.default_rng(int(seed))
    state = dict(target.target_state())
    next_cell_id = max((identity.cell_id for identity in state.values()), default=-1) + 1
    params = dict(spec.parameters)
    width = int(params.get("width", 2))
    height = int(params.get("height", 2))
    row_length = int(params.get("rowLength", 2))
    frozen_positions: frozenset[Position] = frozenset()
    mask_positions: tuple[Position, ...] = ()
    source_positions: tuple[Position, ...] = ()
    target_positions: tuple[Position, ...] = ()
    extra_positions: tuple[Position, ...] = ()
    removed_cell_ids: tuple[int, ...] = ()
    replaced_cell_ids: tuple[int, ...] = ()
    inserted_cell_ids: tuple[int, ...] = ()
    duplicated_cell_ids: tuple[int, ...] = ()
    graft_transform: dict[str, Any] = {"transform": "none"}

    if spec.family == "contiguous_chunk_removal":
        mask_positions = _center_patch(target, width=width, height=height, row_length=row_length)
        removed = []
        for position in mask_positions:
            if position in state:
                removed.append(state[position].cell_id)
                del state[position]
        removed_cell_ids = tuple(sorted(removed))
        graft_transform = {"transform": "remove", "maskPositions": mask_positions}
    elif spec.family == "freeze_patch":
        mask_positions = _center_patch(target, width=width, height=height, row_length=row_length)
        frozen_positions = frozenset(mask_positions)
        graft_transform = {"transform": "freeze", "frozenPositions": mask_positions}
    elif spec.family == "rotate_graft":
        mask_positions = _center_patch(target, width=width, height=height, row_length=row_length)
        rotation = _rotated_positions(mask_positions)
        quarter_turns = int(params.get("quarterTurns", 2))
        rotated_state = dict(state)
        for target_position, source_position in rotation.items():
            source_identity = state[source_position]
            rotated_state[target_position] = CellIdentity(
                cell_id=source_identity.cell_id,
                components=_rotate_polarity(source_identity.components, quarter_turns),
            )
        state = rotated_state
        graft_transform = {
            "transform": "rotate_180",
            "quarterTurns": quarter_turns,
            "positionMap": {str(position): list(source) for position, source in sorted(rotation.items())},
        }
    elif spec.family == "duplicate_region":
        source_positions, target_positions = _side_patches(target, width=width, height=height, row_length=row_length)
        replaced = []
        duplicated = []
        for index, (source_position, target_position) in enumerate(zip(source_positions, target_positions, strict=False)):
            if target_position in state:
                replaced.append(state[target_position].cell_id)
            clone = _clone_identity(state[source_position], cell_id=next_cell_id + index)
            state[target_position] = clone
            duplicated.append(clone.cell_id)
        replaced_cell_ids = tuple(sorted(replaced))
        duplicated_cell_ids = tuple(sorted(duplicated))
        graft_transform = {
            "transform": "duplicate_replace",
            "sourcePositions": source_positions,
            "targetPositions": target_positions,
        }
    elif spec.family == "insert_foreign_patch":
        extra_positions = _extra_patch(target, width=width, height=height, row_length=row_length)
        inserted = []
        shuffled = list(extra_positions)
        rng.shuffle(shuffled)
        for index, position in enumerate(shuffled):
            foreign = _foreign_identity(next_cell_id + index, index)
            state[position] = foreign
            inserted.append(foreign.cell_id)
        inserted_cell_ids = tuple(sorted(inserted))
        graft_transform = {"transform": "insert_extra_foreign_patch", "extraPositions": extra_positions}
    else:
        raise ValueError(f"unsupported perturbation family: {spec.family}")

    return PerturbedState(
        target_id=target.target_id,
        perturbation=spec,
        seed=int(seed),
        observed_by_position={position: state[position] for position in sorted(state)},
        frozen_positions=frozen_positions,
        mask_positions=tuple(sorted(mask_positions)),
        source_positions=tuple(sorted(source_positions)),
        target_positions=tuple(sorted(target_positions)),
        extra_positions=tuple(sorted(extra_positions)),
        removed_cell_ids=removed_cell_ids,
        replaced_cell_ids=replaced_cell_ids,
        inserted_cell_ids=inserted_cell_ids,
        duplicated_cell_ids=duplicated_cell_ids,
        graft_transform=graft_transform,
    )


def _clone_child(identity: CellIdentity, child_id: int) -> CellIdentity:
    return _clone_identity(identity, cell_id=int(child_id))


def _polarity_similarity_score(identity: CellIdentity, position: Position, state: Mapping[Position, CellIdentity], substrate: Substrate) -> float:
    polarity = identity.components.get("polarity")
    if not isinstance(polarity, Sequence) or len(polarity) < 2:
        return 0.0
    vector = tuple(round(float(item), 6) for item in polarity[:2])
    neighbors = [state[neighbor] for neighbor in substrate.neighbors(position) if neighbor in state]
    if not neighbors:
        return 0.0
    same = 0
    for neighbor in neighbors:
        neighbor_polarity = neighbor.components.get("polarity")
        if isinstance(neighbor_polarity, Sequence) and len(neighbor_polarity) >= 2:
            same += tuple(round(float(item), 6) for item in neighbor_polarity[:2]) == vector
    return same / max(1.0, float(len(neighbors)))


def _candidate_crawl_delta(
    state: Mapping[Position, CellIdentity],
    source: Position,
    target_position: Position,
    substrate: Substrate,
    policy: RegenerationPolicySpec,
    memory_by_cell: Mapping[int, int],
    signal_by_position: Mapping[Position, float],
) -> float:
    actor = state[source]
    before = _local_score(actor, source, state, substrate, policy, memory_by_cell, signal_by_position)
    after_state = dict(state)
    del after_state[source]
    after_state[target_position] = actor
    after = _local_score(actor, target_position, after_state, substrate, policy, memory_by_cell, signal_by_position)
    return float(after - before)


def _candidate_divide_delta(
    state: Mapping[Position, CellIdentity],
    source: Position,
    target_position: Position,
    substrate: Substrate,
    policy: RegenerationPolicySpec,
    next_cell_id: int,
    memory_by_cell: Mapping[int, int],
    signal_by_position: Mapping[Position, float],
) -> float:
    actor = state[source]
    child = _clone_child(actor, next_cell_id)
    before = _local_score(actor, source, state, substrate, policy, memory_by_cell, signal_by_position)
    after_state = dict(state)
    after_state[target_position] = child
    after = _local_score(actor, source, after_state, substrate, policy, memory_by_cell, signal_by_position)
    after += _local_score(child, target_position, after_state, substrate, policy, memory_by_cell, signal_by_position)
    growth_bias = float(policy.parameters.get("growthBias", 0.0))
    return float(after - before + growth_bias)


def _candidate_rotate_delta(
    state: Mapping[Position, CellIdentity],
    source: Position,
    substrate: Substrate,
    policy: RegenerationPolicySpec,
) -> tuple[int, float]:
    actor = state[source]
    before = _polarity_similarity_score(actor, source, state, substrate)
    best_turn = 0
    best_delta = 0.0
    for turns in (1, 2, 3):
        rotated = CellIdentity(actor.cell_id, _rotate_polarity(actor.components, turns))
        after_state = dict(state)
        after_state[source] = rotated
        after = _polarity_similarity_score(rotated, source, after_state, substrate)
        delta = float(after - before) + (0.01 if policy.family.endswith("growth_prune") else 0.0)
        if delta > best_delta:
            best_delta = delta
            best_turn = turns
    return best_turn, best_delta


def _legal_candidates(
    state: Mapping[Position, CellIdentity],
    actor_position: Position,
    substrate: Substrate,
    frozen_positions: frozenset[Position],
    policy: RegenerationPolicySpec,
    rng: np.random.Generator,
    next_cell_id: int,
    memory_by_cell: Mapping[int, int],
    signal_by_position: Mapping[Position, float],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if actor_position not in set(substrate.nodes):
        if policy.allow_die:
            candidates.append({"action": "die", "targetPosition": None, "delta": 1.0, "reason": "off_substrate_local_prune"})
        return candidates
    if actor_position in frozen_positions:
        return candidates
    for neighbor in sorted(substrate.neighbors(actor_position)):
        if neighbor in frozen_positions:
            continue
        if neighbor in state:
            delta = _candidate_delta(state, actor_position, neighbor, substrate, policy, memory_by_cell, signal_by_position)
            candidates.append({"action": "swap", "targetPosition": neighbor, "delta": delta, "reason": "local_swap_delta"})
        else:
            crawl_delta = _candidate_crawl_delta(state, actor_position, neighbor, substrate, policy, memory_by_cell, signal_by_position)
            candidates.append({"action": "crawl", "targetPosition": neighbor, "delta": crawl_delta, "reason": "local_crawl_delta"})
            if policy.allow_divide:
                divide_delta = _candidate_divide_delta(
                    state,
                    actor_position,
                    neighbor,
                    substrate,
                    policy,
                    next_cell_id,
                    memory_by_cell,
                    signal_by_position,
                )
                candidates.append({"action": "divide", "targetPosition": neighbor, "delta": divide_delta, "reason": "local_divide_delta"})
    if policy.allow_rotate:
        turns, rotate_delta = _candidate_rotate_delta(state, actor_position, substrate, policy)
        if turns:
            candidates.append({"action": "rotate", "targetPosition": None, "delta": rotate_delta, "quarterTurns": turns, "reason": "local_polarity_rotation"})
    if policy.policy_id == "random_local_repair_null":
        candidates = [candidate for candidate in candidates if candidate["action"] in {"swap", "crawl"}]
        if candidates:
            index = int(rng.integers(0, len(candidates)))
            chosen = dict(candidates[index])
            chosen["delta"] = 0.0
            return [chosen]
    return candidates


def _snapshot_row(
    *,
    run_id: str,
    target: TargetMorphology,
    perturbation: PerturbationSpec,
    policy: RegenerationPolicySpec,
    seed: int,
    step: int,
    state: Mapping[Position, CellIdentity],
    frozen_positions: frozenset[Position],
    action_counts: Mapping[str, int],
    reason: str,
) -> dict[str, Any]:
    result = evaluate_morphospace_metrics(target, state)
    metrics = _metric_lookup(result)
    target_nodes = set(target.substrate.nodes)
    observed_nodes = set(state)
    return {
        "schema_version": REGENERATION_SCHEMA_VERSION,
        "research_step_id": "S08",
        "run_id": run_id,
        "target_id": target.target_id,
        "motif": target.motif,
        "perturbation_id": perturbation.perturbation_id,
        "perturbation_family": perturbation.family,
        "policy_id": policy.policy_id,
        "policy_family": policy.family,
        "seed": int(seed),
        "step": int(step),
        "state_hash": _state_hash(state),
        "snapshot_reason": reason,
        "cell_position_assignment_json": canonical_json(
            [{"position": list(position), "cellId": identity.cell_id} for position, identity in sorted(state.items())]
        ),
        "frozen_positions_json": canonical_json(tuple(sorted(frozen_positions))),
        "cell_count": len(state),
        "occupied_substrate_count": len(observed_nodes & target_nodes),
        "missing_position_count": len(target_nodes - observed_nodes),
        "extra_position_count": len(observed_nodes - target_nodes),
        "swap_count": int(action_counts.get("swap", 0)),
        "crawl_count": int(action_counts.get("crawl", 0)),
        "divide_count": int(action_counts.get("divide", 0)),
        "die_count": int(action_counts.get("die", 0)),
        "rotate_count": int(action_counts.get("rotate", 0)),
        "wait_count": int(action_counts.get("wait", 0)),
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


def _feature_point(result: Mapping[str, Any]) -> tuple[float, float, float]:
    metrics = _metric_lookup(result)
    return (
        float(result["compositeError"]),
        float(metrics["target_energy"]["normalizedValue"]),
        float(metrics["earth_mover_distance"]["normalizedValue"]),
    )


def run_regeneration_benchmark(
    target: TargetMorphology,
    perturbation: PerturbationSpec,
    policy: RegenerationPolicySpec,
    *,
    seed: int,
    max_steps: int = 1500,
    snapshot_interval: int = 150,
    recovery_threshold: float = 1e-12,
) -> tuple[dict[str, Any], list[dict[str, Any]], PerturbedState]:
    """Run one local repair benchmark from a deterministic perturbed state."""

    rng = np.random.default_rng(int(seed))
    perturbed = apply_perturbation(target, perturbation, seed=int(seed))
    state: dict[Position, CellIdentity] = dict(perturbed.observed_by_position)
    initial_state_hash = _state_hash(state)
    initial_metrics = evaluate_morphospace_metrics(target, state)
    run_id = f"S08::{target.target_id}::{perturbation.perturbation_id}::{policy.policy_id}::seed{int(seed)}"
    next_cell_id = max((identity.cell_id for identity in state.values()), default=-1) + 1
    action_counts: dict[str, int] = {"swap": 0, "crawl": 0, "divide": 0, "die": 0, "rotate": 0, "wait": 0}
    illegal_action_count = 0
    frozen_violation_count = 0
    memory_by_cell: dict[int, int] = {}
    signal_by_position: dict[Position, float] = {position: 0.0 for position in target.substrate.nodes}
    frozen_initial = {position: state[position].cell_id for position in perturbed.frozen_positions if position in state}
    first_recovery_step: int | None = None
    best_composite = float(initial_metrics["compositeError"])
    best_step = 0
    feature_points = [_feature_point(initial_metrics)]
    trace_rows = [
        _snapshot_row(
            run_id=run_id,
            target=target,
            perturbation=perturbation,
            policy=policy,
            seed=seed,
            step=0,
            state=state,
            frozen_positions=perturbed.frozen_positions,
            action_counts=action_counts,
            reason="initial_perturbed",
        )
    ]

    for step in range(1, int(max_steps) + 1):
        actor_positions = sorted(state)
        if not actor_positions:
            action_counts["wait"] += 1
            continue
        actor_position = actor_positions[int(rng.integers(0, len(actor_positions)))]
        actor_identity = state[actor_position]
        candidates = _legal_candidates(
            state,
            actor_position,
            target.substrate,
            perturbed.frozen_positions,
            policy,
            rng,
            next_cell_id,
            memory_by_cell,
            signal_by_position,
        )
        if not candidates:
            action_counts["wait"] += 1
            if policy.memory_weight:
                memory_by_cell[actor_identity.cell_id] = min(100, int(memory_by_cell.get(actor_identity.cell_id, 0)) + 1)
            if policy.signal_weight and actor_position in signal_by_position:
                signal_by_position[actor_position] = min(10.0, signal_by_position.get(actor_position, 0.0) + 1.0)
        else:
            if policy.policy_id == "random_local_repair_null":
                chosen = candidates[0]
            else:
                chosen = max(candidates, key=lambda item: (float(item["delta"]), item["action"]))
                if float(chosen["delta"]) <= 1e-12 and float(policy.exploration_epsilon) > 0:
                    if float(rng.random()) < float(policy.exploration_epsilon):
                        chosen = candidates[int(rng.integers(0, len(candidates)))]
                    else:
                        chosen = {"action": "wait", "targetPosition": None, "delta": 0.0, "reason": "no_local_improvement"}
                elif float(chosen["delta"]) <= 1e-12 and chosen["action"] != "die":
                    chosen = {"action": "wait", "targetPosition": None, "delta": 0.0, "reason": "no_local_improvement"}
            action = str(chosen["action"])
            target_position = chosen.get("targetPosition")
            if action == "swap" and target_position in state:
                state[actor_position], state[target_position] = state[target_position], state[actor_position]  # type: ignore[index]
                action_counts["swap"] += 1
            elif action == "crawl" and target_position is not None and target_position not in state:
                state[target_position] = state.pop(actor_position)  # type: ignore[index]
                action_counts["crawl"] += 1
            elif action == "divide" and target_position is not None and target_position not in state:
                state[target_position] = _clone_child(actor_identity, next_cell_id)  # type: ignore[index]
                next_cell_id += 1
                action_counts["divide"] += 1
            elif action == "die" and actor_position not in set(target.substrate.nodes):
                del state[actor_position]
                action_counts["die"] += 1
            elif action == "rotate":
                turns = int(chosen.get("quarterTurns", 1))
                state[actor_position] = CellIdentity(actor_identity.cell_id, _rotate_polarity(actor_identity.components, turns))
                action_counts["rotate"] += 1
            else:
                action_counts["wait"] += 1
                illegal_action_count += int(action != "wait")
            if policy.memory_weight and action != "wait":
                memory_by_cell[actor_identity.cell_id] = max(0, int(memory_by_cell.get(actor_identity.cell_id, 0)) - 1)

        if policy.signal_weight and (step % 5 == 0):
            _diffuse_signals(signal_by_position, target.substrate, decay=0.1, diffusion=0.2)

        for frozen_position, frozen_cell_id in frozen_initial.items():
            if state.get(frozen_position) is None or state[frozen_position].cell_id != frozen_cell_id:
                frozen_violation_count += 1

        if step % int(snapshot_interval) == 0 or step == int(max_steps):
            snapshot = _snapshot_row(
                run_id=run_id,
                target=target,
                perturbation=perturbation,
                policy=policy,
                seed=seed,
                step=step,
                state=state,
                frozen_positions=perturbed.frozen_positions,
                action_counts=action_counts,
                reason="interval" if step < int(max_steps) else "final",
            )
            trace_rows.append(snapshot)
            feature_points.append((snapshot["composite_error"], snapshot["target_energy_normalized"], snapshot["earth_mover_normalized"]))
            if float(snapshot["composite_error"]) < best_composite:
                best_composite = float(snapshot["composite_error"])
                best_step = step
            if first_recovery_step is None and float(snapshot["composite_error"]) <= float(recovery_threshold):
                first_recovery_step = step
                if policy.policy_id != "random_local_repair_null":
                    break

    final_metrics = evaluate_morphospace_metrics(target, state)
    initial_lookup = _metric_lookup(initial_metrics)
    final_lookup = _metric_lookup(final_metrics)
    curvature = trajectory_curvature(feature_points)
    target_nodes = set(target.substrate.nodes)
    final_nodes = set(state)
    final_composite = float(final_metrics["compositeError"])
    initial_composite = float(initial_metrics["compositeError"])
    if initial_composite <= 1e-12:
        relative_error_reduction = 0.0 if final_composite <= float(recovery_threshold) else -final_composite
    else:
        relative_error_reduction = (initial_composite - final_composite) / initial_composite
    exact_recovery = final_composite <= float(recovery_threshold)
    improved = final_composite < initial_composite
    nonconservative_action_count = int(action_counts["divide"] + action_counts["die"])
    row = {
        "schema_version": REGENERATION_SCHEMA_VERSION,
        "research_step_id": "S08",
        "run_id": run_id,
        "target_id": target.target_id,
        "motif": target.motif,
        "substrate_type": target.substrate.substrate_type,
        "node_count": len(target.substrate.nodes),
        "perturbation_id": perturbation.perturbation_id,
        "perturbation_family": perturbation.family,
        "policy_id": policy.policy_id,
        "policy_family": policy.family,
        "source_policy_id": policy.source_policy_id,
        "source_artifact": policy.source_artifact,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "snapshot_interval": int(snapshot_interval),
        "initial_state_hash": initial_state_hash,
        "final_state_hash": _state_hash(state),
        "initial_composite_error": initial_composite,
        "final_composite_error": final_composite,
        "delta_composite_error": final_composite - initial_composite,
        "relative_error_reduction": relative_error_reduction,
        "initial_target_energy": float(initial_lookup["target_energy"]["value"]),
        "final_target_energy": float(final_lookup["target_energy"]["value"]),
        "initial_earth_mover_distance": float(initial_lookup["earth_mover_distance"]["value"]),
        "final_earth_mover_distance": float(final_lookup["earth_mover_distance"]["value"]),
        "initial_target_neighborhood_error": float(initial_lookup["target_neighborhood_error"]["value"]),
        "final_target_neighborhood_error": float(final_lookup["target_neighborhood_error"]["value"]),
        "initial_cell_count": len(perturbed.observed_by_position),
        "final_cell_count": len(state),
        "target_cell_count": len(target.substrate.nodes),
        "initial_cell_count_delta": len(perturbed.observed_by_position) - len(target.substrate.nodes),
        "final_cell_count_delta": len(state) - len(target.substrate.nodes),
        "initial_missing_position_count": len(target_nodes - set(perturbed.observed_by_position)),
        "final_missing_position_count": len(target_nodes - final_nodes),
        "initial_extra_position_count": len(set(perturbed.observed_by_position) - target_nodes),
        "final_extra_position_count": len(final_nodes - target_nodes),
        "exact_recovery": bool(exact_recovery),
        "improved": bool(improved),
        "recovery_time_steps": first_recovery_step,
        "best_composite_error": float(best_composite),
        "best_step": int(best_step),
        "swap_count": int(action_counts["swap"]),
        "crawl_count": int(action_counts["crawl"]),
        "divide_count": int(action_counts["divide"]),
        "die_count": int(action_counts["die"]),
        "rotate_count": int(action_counts["rotate"]),
        "wait_count": int(action_counts["wait"]),
        "nonconservative_action_count": nonconservative_action_count,
        "illegal_action_count": int(illegal_action_count),
        "collision_count": 0,
        "frozen_violation_count": int(frozen_violation_count),
        "trajectory_curvature": float(curvature["value"]),
        "trajectory_point_count": int(curvature["detail"]["pointCount"]),
        "uses_whole_target_leakage": False,
        "metric_handling_finite": bool(
            math.isfinite(final_composite)
            and all(math.isfinite(float(metric["value"])) and math.isfinite(float(metric["normalizedValue"])) for metric in final_metrics["metrics"])
        ),
        "requires_nonconservative_repair": perturbation.family in {"contiguous_chunk_removal", "insert_foreign_patch"},
    }
    return row, trace_rows, perturbed


def perturbation_catalog_rows(targets: Sequence[TargetMorphology], specs: Sequence[PerturbationSpec], seeds: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        for spec in specs:
            for seed in seeds:
                rows.append(apply_perturbation(target, spec, seed=int(seed)).to_record(target))
    return rows
