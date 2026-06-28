"""Symmetry-breaking benchmark helpers for E05 S10."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .recovery import PROHIBITED_POLICY_PAYLOAD_KEYS, _json_ready, _sha16
from .substrates import Position, Substrate
from .targets import canonical_json


SYMMETRY_SCHEMA_VERSION = "e05_s10_symmetry_breaking.v1"
SYMMETRY_INITIAL_STATE_VERSION = "e05_s10_symmetric_initial_state.v1"
SYMMETRY_POLICY_AUDIT_VERSION = "e05_s10_policy_axis_leakage_audit.v1"

PROHIBITED_LOCAL_SYMMETRY_KEYS = PROHIBITED_POLICY_PAYLOAD_KEYS | {
    "axis_oracle",
    "expected_axis",
    "global_axis",
    "global_gradient",
    "global_gradient_vector",
    "gradient_axis",
    "organizer_position",
    "target_axis",
    "target_axis_map",
    "whole_axis_field",
}


@dataclass(frozen=True)
class SymmetryTaskSpec:
    """Symmetry-breaking task over a square grid."""

    task_id: str
    motif: str
    target_kind: str
    grid_size: int
    description: str
    target_axis: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "schemaVersion": SYMMETRY_SCHEMA_VERSION,
            "taskId": self.task_id,
            "motif": self.motif,
            "targetKind": self.target_kind,
            "gridSize": int(self.grid_size),
            "targetAxis": self.target_axis,
            "description": self.description,
            "parameters": dict(self.parameters),
            "symmetricInitialClass": "uniform_neutral_square",
        }


@dataclass(frozen=True)
class SymmetryPolicySpec:
    """Policy declaration for S10 mutable-polarity symmetry tests."""

    policy_id: str
    family: str
    description: str
    information_scope: str
    is_local_only: bool
    nucleation_probability: float = 0.0
    alignment_strength: float = 0.0
    noise_strength: float = 0.0
    uses_local_signal: bool = False
    uses_local_polarity: bool = False
    uses_organizer: bool = False
    uses_global_gradient: bool = False
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        allowed_inputs = [
            "actor_current_label",
            "actor_current_polarity",
            "adjacent_neighbor_labels",
            "adjacent_neighbor_polarities",
            "public_local_degree",
            "local_random_seed_stream",
        ]
        if self.uses_local_signal:
            allowed_inputs.extend(["actor_local_signal", "adjacent_neighbor_local_signals"])
        if self.uses_organizer:
            allowed_inputs.extend(["explicit_organizer_position", "organizer_emitted_axis_or_radial_cue"])
        if self.uses_global_gradient:
            allowed_inputs.extend(["explicit_global_gradient_axis"])
        return {
            "schemaVersion": SYMMETRY_SCHEMA_VERSION,
            "policyId": self.policy_id,
            "family": self.family,
            "description": self.description,
            "informationScope": self.information_scope,
            "isLocalOnly": bool(self.is_local_only),
            "isGlobalInformationBaseline": bool(self.uses_organizer or self.uses_global_gradient),
            "usesLocalSignal": bool(self.uses_local_signal),
            "usesLocalPolarity": bool(self.uses_local_polarity),
            "usesOrganizer": bool(self.uses_organizer),
            "usesGlobalGradient": bool(self.uses_global_gradient),
            "allowedInputs": allowed_inputs,
            "nucleationProbability": float(self.nucleation_probability),
            "alignmentStrength": float(self.alignment_strength),
            "noiseStrength": float(self.noise_strength),
            "parameters": dict(self.parameters),
            "leakageBoundary": (
                "Local-only policies receive only actor and neighbor polarity/label/signal state plus local degree. "
                "Organizer and global-gradient policies are explicit nonlocal baselines and are flagged as such."
            ),
        }


def standard_symmetry_task_specs() -> tuple[SymmetryTaskSpec, ...]:
    return (
        SymmetryTaskSpec(
            task_id="axis_gradient_9x9",
            motif="axis_gradient",
            target_kind="axis",
            grid_size=9,
            target_axis="x",
            description="Choose a stable x-axis orientation from a fully symmetric neutral square.",
        ),
        SymmetryTaskSpec(
            task_id="ring_from_neutral_9x9",
            motif="ring",
            target_kind="ring",
            grid_size=9,
            description="Form a radial core/ring/outer pattern from a fully symmetric neutral square.",
        ),
        SymmetryTaskSpec(
            task_id="asymmetric_appendage_9x9",
            motif="asymmetric_appendage",
            target_kind="asymmetric_appendage",
            grid_size=9,
            target_axis="x",
            description="Form a right-sided appendage-like computational motif from a symmetric neutral square.",
        ),
    )


def standard_symmetry_policy_specs() -> tuple[SymmetryPolicySpec, ...]:
    return (
        SymmetryPolicySpec(
            policy_id="local_no_gradient_alignment",
            family="local_only_no_gradient",
            description="Local neighbor-alignment rule with rare random polarity nucleation and no axis cue.",
            information_scope="local_only",
            is_local_only=True,
            nucleation_probability=0.004,
            alignment_strength=0.65,
            noise_strength=0.0,
            uses_local_polarity=True,
        ),
        SymmetryPolicySpec(
            policy_id="local_polarity_noise_alignment",
            family="local_only_polarity",
            description="Local polarity-alignment rule with stronger stochastic nucleation and local noise.",
            information_scope="local_only",
            is_local_only=True,
            nucleation_probability=0.018,
            alignment_strength=0.78,
            noise_strength=0.04,
            uses_local_polarity=True,
        ),
        SymmetryPolicySpec(
            policy_id="local_signal_alignment",
            family="local_only_signal",
            description="Local polarity alignment with an actor-local diffusive signal field and no global axis.",
            information_scope="local_only",
            is_local_only=True,
            nucleation_probability=0.010,
            alignment_strength=0.82,
            noise_strength=0.025,
            uses_local_signal=True,
            uses_local_polarity=True,
            parameters={"signalDecay": 0.06, "signalDiffusion": 0.25},
        ),
        SymmetryPolicySpec(
            policy_id="organizer_center_baseline",
            family="organizer_baseline",
            description="Explicit center organizer baseline that supplies radial or x-axis cues.",
            information_scope="explicit_organizer_baseline",
            is_local_only=False,
            alignment_strength=1.0,
            uses_organizer=True,
            parameters={"organizer": "center"},
        ),
        SymmetryPolicySpec(
            policy_id="global_gradient_baseline",
            family="global_gradient_baseline",
            description="Explicit top-down x-axis gradient baseline.",
            information_scope="explicit_global_gradient_baseline",
            is_local_only=False,
            alignment_strength=1.0,
            uses_global_gradient=True,
            parameters={"globalGradientAxis": "x"},
        ),
    )


def _square_substrate(size: int) -> Substrate:
    return Substrate.square_grid(int(size), int(size))


def _neutral_state(substrate: Substrate) -> dict[Position, dict[str, Any]]:
    return {
        position: {"vector": (0.0, 0.0), "label": "neutral", "signal": 0.0}
        for position in substrate.nodes
    }


def _state_hash(state: Mapping[Position, Mapping[str, Any]]) -> str:
    payload = [
        {
            "position": list(position),
            "vector": [round(float(value), 6) for value in cell["vector"]],
            "label": str(cell["label"]),
        }
        for position, cell in sorted(state.items())
    ]
    return _sha16(payload)


def make_symmetric_initial_state(task: SymmetryTaskSpec, *, seed: int) -> dict[str, Any]:
    substrate = _square_substrate(task.grid_size)
    state = _neutral_state(substrate)
    audit = verify_initial_symmetry(task, state, seed=seed)
    return {
        "schemaVersion": SYMMETRY_INITIAL_STATE_VERSION,
        "researchStepId": "S10",
        "taskId": task.task_id,
        "motif": task.motif,
        "seed": int(seed),
        "gridSize": int(task.grid_size),
        "nodeCount": len(substrate.nodes),
        "state": state,
        "stateHash": _state_hash(state),
        "audit": audit,
    }


def _transform_position(position: Position, size: int, transform: str) -> Position:
    x, y = int(position[0]), int(position[1])
    n = int(size) - 1
    if transform == "identity":
        return (x, y)
    if transform == "rot90":
        return (n - y, x)
    if transform == "rot180":
        return (n - x, n - y)
    if transform == "rot270":
        return (y, n - x)
    if transform == "reflect_x":
        return (n - x, y)
    if transform == "reflect_y":
        return (x, n - y)
    if transform == "reflect_diag":
        return (y, x)
    if transform == "reflect_antidiag":
        return (n - y, n - x)
    raise ValueError(f"unknown D4 transform: {transform}")


def verify_initial_symmetry(
    task: SymmetryTaskSpec,
    state: Mapping[Position, Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    transforms = ("identity", "rot90", "rot180", "rot270", "reflect_x", "reflect_y", "reflect_diag", "reflect_antidiag")
    mismatches: list[dict[str, Any]] = []
    for transform in transforms:
        for position, cell in state.items():
            transformed = _transform_position(position, task.grid_size, transform)
            other = state.get(transformed)
            if other is None:
                mismatches.append({"transform": transform, "position": position, "reason": "missing_transformed_position"})
                continue
            if str(other["label"]) != str(cell["label"]):
                mismatches.append({"transform": transform, "position": position, "reason": "label_mismatch"})
                continue
            if tuple(float(v) for v in other["vector"]) != tuple(float(v) for v in cell["vector"]):
                mismatches.append({"transform": transform, "position": position, "reason": "vector_mismatch"})
    labels = Counter(str(cell["label"]) for cell in state.values())
    vectors = Counter(tuple(float(v) for v in cell["vector"]) for cell in state.values())
    return {
        "schemaVersion": SYMMETRY_INITIAL_STATE_VERSION,
        "researchStepId": "S10",
        "taskId": task.task_id,
        "seed": int(seed),
        "gridSize": int(task.grid_size),
        "transformCount": len(transforms),
        "nodeCount": len(state),
        "success": len(mismatches) == 0,
        "mismatchCount": len(mismatches),
        "mismatches": mismatches[:10],
        "labelCounts": dict(sorted(labels.items())),
        "vectorCounts": {str(key): value for key, value in sorted(vectors.items())},
        "stateHash": _state_hash(state),
    }


def _audit_mapping_keys(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in PROHIBITED_LOCAL_SYMMETRY_KEYS:
                errors.append(f"{path}.{key_text} exposes prohibited key")
            errors.extend(_audit_mapping_keys(item, f"{path}.{key_text}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_audit_mapping_keys(item, f"{path}[{index}]"))
    return errors


def audit_symmetry_policy_payload(policy: SymmetryPolicySpec | Mapping[str, Any]) -> dict[str, Any]:
    record = policy.to_record() if isinstance(policy, SymmetryPolicySpec) else dict(policy)
    is_local_only = bool(record.get("isLocalOnly", record.get("is_local_only", False)))
    is_global_baseline = bool(record.get("isGlobalInformationBaseline", False))
    errors: list[str] = []
    if is_local_only:
        errors.extend(_audit_mapping_keys(record))
        serialized = json.dumps(_json_ready(record), sort_keys=True)
        suspicious = [
            token
            for token in ("global_axis", "target_axis", "target_map", "organizer_position", "global_gradient")
            if token in serialized
        ]
        if suspicious:
            errors.append(f"serialized local-only policy mentions prohibited token(s): {sorted(set(suspicious))}")
    else:
        if not is_global_baseline:
            errors.append("non-local policy is not explicitly flagged as a global-information baseline")
    return {
        "schemaVersion": SYMMETRY_POLICY_AUDIT_VERSION,
        "policyId": str(record.get("policyId", record.get("policy_id", "unknown"))),
        "isLocalOnly": is_local_only,
        "isGlobalInformationBaseline": is_global_baseline,
        "usesOrganizer": bool(record.get("usesOrganizer", False)),
        "usesGlobalGradient": bool(record.get("usesGlobalGradient", False)),
        "success": len(errors) == 0,
        "errorCount": len(errors),
        "errors": errors,
        "payloadHash": _sha16(record),
    }


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros(2, dtype=float)
    return vector / norm


def _random_cardinal(rng: np.random.Generator) -> np.ndarray:
    choices = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=float)
    return choices[int(rng.integers(0, len(choices)))]


def _center(task: SymmetryTaskSpec) -> tuple[float, float]:
    center = (float(task.grid_size - 1) / 2.0, float(task.grid_size - 1) / 2.0)
    return center


def _target_label(task: SymmetryTaskSpec, position: Position) -> str:
    x, y = int(position[0]), int(position[1])
    center_x, center_y = _center(task)
    if task.target_kind == "ring":
        dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        inner = max(1.0, center_x) * 0.55
        outer = max(1.0, center_x) * 0.95
        if dist < inner:
            return "core"
        if dist <= outer:
            return "ring"
        return "outer"
    if task.target_kind == "asymmetric_appendage":
        center = int(center_x)
        band_radius = max(1, round(task.grid_size / 5))
        if (x, y) == (center, center):
            return "organizer"
        if x >= center + 1 and center - band_radius <= y <= center + band_radius:
            return "appendage"
        if max(1, center - max(1, round(task.grid_size / 3))) <= x <= center and center - band_radius <= y <= center + band_radius:
            return "core"
        return "boundary"
    return "axis"


def _target_vector(task: SymmetryTaskSpec, position: Position) -> np.ndarray:
    x, y = int(position[0]), int(position[1])
    if task.target_kind == "ring":
        center_x, center_y = _center(task)
        return _normalize(np.asarray([float(x) - center_x, float(y) - center_y], dtype=float))
    if task.target_axis == "y":
        return np.asarray([0.0, 1.0], dtype=float)
    return np.asarray([1.0, 0.0], dtype=float)


def _diffuse_signal(state: dict[Position, dict[str, Any]], substrate: Substrate, *, decay: float, diffusion: float) -> None:
    updated: dict[Position, float] = {}
    for position in substrate.nodes:
        current = float(state[position].get("signal", 0.0))
        neighbors = substrate.neighbors(position)
        neighbor_mean = 0.0
        if neighbors:
            neighbor_mean = float(np.mean([float(state[neighbor].get("signal", 0.0)) for neighbor in neighbors]))
        updated[position] = max(0.0, current * (1.0 - decay - diffusion) + neighbor_mean * diffusion)
    for position, value in updated.items():
        state[position]["signal"] = value


def _apply_policy_step(
    task: SymmetryTaskSpec,
    policy: SymmetryPolicySpec,
    state: dict[Position, dict[str, Any]],
    substrate: Substrate,
    rng: np.random.Generator,
) -> None:
    position = sorted(substrate.nodes)[int(rng.integers(0, len(substrate.nodes)))]
    cell = state[position]
    if policy.uses_global_gradient:
        cell["vector"] = tuple(_target_vector(task, position if task.target_kind == "ring" else (task.grid_size - 1, 0)))
        cell["label"] = _target_label(task, position)
        return
    if policy.uses_organizer:
        cell["vector"] = tuple(_target_vector(task, position))
        cell["label"] = _target_label(task, position)
        cell["signal"] = max(float(cell.get("signal", 0.0)), 1.0)
        return

    neighbors = [neighbor for neighbor in substrate.neighbors(position)]
    neighbor_vectors = [
        np.asarray(state[neighbor]["vector"], dtype=float)
        for neighbor in neighbors
        if np.linalg.norm(np.asarray(state[neighbor]["vector"], dtype=float)) > 1e-9
    ]
    current = np.asarray(cell["vector"], dtype=float)
    if neighbor_vectors:
        local_mean = _normalize(np.mean(neighbor_vectors, axis=0))
        noise = np.zeros(2, dtype=float)
        if policy.noise_strength:
            noise = rng.normal(0.0, float(policy.noise_strength), size=2)
        updated = _normalize((1.0 - float(policy.alignment_strength)) * current + float(policy.alignment_strength) * local_mean + noise)
        cell["vector"] = tuple(float(value) for value in updated)
        cell["label"] = "axis"
        if policy.uses_local_signal:
            cell["signal"] = min(10.0, float(cell.get("signal", 0.0)) + float(np.linalg.norm(updated)))
        return
    nucleation_probability = float(policy.nucleation_probability)
    if policy.uses_local_signal:
        nucleation_probability += min(0.04, float(cell.get("signal", 0.0)) * 0.01)
    if float(rng.random()) < nucleation_probability:
        vector = _random_cardinal(rng)
        if policy.noise_strength:
            vector = _normalize(vector + rng.normal(0.0, float(policy.noise_strength), size=2))
        cell["vector"] = tuple(float(value) for value in vector)
        cell["label"] = "axis"
        if policy.uses_local_signal:
            cell["signal"] = min(10.0, float(cell.get("signal", 0.0)) + 1.0)


def _axis_metrics(task: SymmetryTaskSpec, state: Mapping[Position, Mapping[str, Any]]) -> dict[str, Any]:
    vectors = []
    for cell in state.values():
        vector = np.asarray(cell["vector"], dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            vectors.append(vector / norm)
    if not vectors:
        return {
            "orientedFraction": 0.0,
            "axisStrength": 0.0,
            "axisAngleRadians": None,
            "selectedAxis": "none",
            "targetAxisAlignment": 0.0 if task.target_axis else None,
        }
    rows = np.vstack(vectors)
    qx = float(np.mean(rows[:, 0] ** 2 - rows[:, 1] ** 2))
    qy = float(np.mean(2.0 * rows[:, 0] * rows[:, 1]))
    strength = float(math.sqrt(qx * qx + qy * qy))
    angle = 0.5 * math.atan2(qy, qx)
    selected = "x" if abs(math.cos(angle)) >= abs(math.sin(angle)) else "y"
    target_alignment = None
    if task.target_axis == "x":
        target_alignment = abs(math.cos(angle))
    elif task.target_axis == "y":
        target_alignment = abs(math.sin(angle))
    return {
        "orientedFraction": len(vectors) / max(1.0, float(len(state))),
        "axisStrength": strength,
        "axisAngleRadians": angle,
        "selectedAxis": selected,
        "targetAxisAlignment": None if target_alignment is None else float(target_alignment),
    }


def _label_metrics(task: SymmetryTaskSpec, state: Mapping[Position, Mapping[str, Any]]) -> dict[str, Any]:
    labels = [str(cell["label"]) for cell in state.values()]
    label_counts = Counter(labels)
    diversity = len(label_counts)
    if task.target_kind not in {"ring", "asymmetric_appendage"}:
        return {"labelMatchFraction": None, "labelDiversity": diversity, "labelCounts": dict(sorted(label_counts.items()))}
    matches = 0
    for position, cell in state.items():
        matches += int(str(cell["label"]) == _target_label(task, position))
    return {
        "labelMatchFraction": matches / max(1.0, float(len(state))),
        "labelDiversity": diversity,
        "labelCounts": dict(sorted(label_counts.items())),
    }


def symmetry_metrics(task: SymmetryTaskSpec, state: Mapping[Position, Mapping[str, Any]]) -> dict[str, Any]:
    axis = _axis_metrics(task, state)
    labels = _label_metrics(task, state)
    if task.target_kind == "axis":
        pattern_score = float(axis["axisStrength"]) * float(axis["orientedFraction"]) * float(axis["targetAxisAlignment"] or 0.0)
        success = pattern_score >= 0.45 and axis["selectedAxis"] == task.target_axis
    else:
        label_match = float(labels["labelMatchFraction"] or 0.0)
        pattern_score = label_match
        success = label_match >= 0.75
    symmetry_broken = bool(float(axis["axisStrength"]) >= 0.35 and float(axis["orientedFraction"]) >= 0.35)
    if task.target_kind in {"ring", "asymmetric_appendage"}:
        symmetry_broken = symmetry_broken or int(labels["labelDiversity"]) >= 3
    return {
        "axis": axis,
        "labels": labels,
        "patternScore": float(pattern_score),
        "symmetryBroken": symmetry_broken,
        "success": bool(success),
    }


def _state_assignment_json(state: Mapping[Position, Mapping[str, Any]]) -> str:
    payload = [
        {
            "position": list(position),
            "vx": round(float(cell["vector"][0]), 5),
            "vy": round(float(cell["vector"][1]), 5),
            "label": str(cell["label"]),
        }
        for position, cell in sorted(state.items())
    ]
    return canonical_json(payload)


def _snapshot_row(
    *,
    run_id: str,
    task: SymmetryTaskSpec,
    policy: SymmetryPolicySpec,
    seed: int,
    step: int,
    state: Mapping[Position, Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    metrics = symmetry_metrics(task, state)
    axis = metrics["axis"]
    labels = metrics["labels"]
    return {
        "schema_version": SYMMETRY_SCHEMA_VERSION,
        "research_step_id": "S10",
        "run_id": run_id,
        "task_id": task.task_id,
        "motif": task.motif,
        "target_kind": task.target_kind,
        "policy_id": policy.policy_id,
        "policy_family": policy.family,
        "seed": int(seed),
        "step": int(step),
        "snapshot_reason": reason,
        "state_hash": _state_hash(state),
        "state_assignment_json": _state_assignment_json(state),
        "oriented_fraction": float(axis["orientedFraction"]),
        "axis_strength": float(axis["axisStrength"]),
        "axis_angle_radians": axis["axisAngleRadians"],
        "selected_axis": axis["selectedAxis"],
        "target_axis_alignment": axis["targetAxisAlignment"],
        "label_match_fraction": labels["labelMatchFraction"],
        "label_diversity": int(labels["labelDiversity"]),
        "label_counts_json": canonical_json(labels["labelCounts"]),
        "pattern_score": float(metrics["patternScore"]),
        "symmetry_broken": bool(metrics["symmetryBroken"]),
        "success": bool(metrics["success"]),
    }


def run_symmetry_breaking_benchmark(
    task: SymmetryTaskSpec,
    policy: SymmetryPolicySpec,
    *,
    seed: int,
    max_steps: int = 600,
    snapshot_interval: int = 100,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    substrate = _square_substrate(task.grid_size)
    initial = make_symmetric_initial_state(task, seed=int(seed))
    state: dict[Position, dict[str, Any]] = {
        position: dict(cell)
        for position, cell in initial["state"].items()
    }
    initial_hash = _state_hash(state)
    rng = np.random.default_rng(int(seed))
    run_id = f"S10::{task.task_id}::{policy.policy_id}::seed{int(seed)}"
    trace_rows = [
        _snapshot_row(
            run_id=run_id,
            task=task,
            policy=policy,
            seed=int(seed),
            step=0,
            state=state,
            reason="initial_symmetric",
        )
    ]

    for step in range(1, int(max_steps) + 1):
        _apply_policy_step(task, policy, state, substrate, rng)
        if policy.uses_local_signal and step % 5 == 0:
            _diffuse_signal(
                state,
                substrate,
                decay=float(policy.parameters.get("signalDecay", 0.06)),
                diffusion=float(policy.parameters.get("signalDiffusion", 0.25)),
            )
        if step % int(snapshot_interval) == 0 or step == int(max_steps):
            trace_rows.append(
                _snapshot_row(
                    run_id=run_id,
                    task=task,
                    policy=policy,
                    seed=int(seed),
                    step=step,
                    state=state,
                    reason="interval" if step < int(max_steps) else "final",
                )
            )

    final = symmetry_metrics(task, state)
    axis = final["axis"]
    labels = final["labels"]
    row = {
        "schema_version": SYMMETRY_SCHEMA_VERSION,
        "research_step_id": "S10",
        "run_id": run_id,
        "task_id": task.task_id,
        "motif": task.motif,
        "target_kind": task.target_kind,
        "target_axis": task.target_axis,
        "grid_size": int(task.grid_size),
        "node_count": len(substrate.nodes),
        "policy_id": policy.policy_id,
        "policy_family": policy.family,
        "information_scope": policy.information_scope,
        "is_local_only_policy": bool(policy.is_local_only),
        "is_global_information_baseline": bool(policy.uses_organizer or policy.uses_global_gradient),
        "uses_organizer": bool(policy.uses_organizer),
        "uses_global_gradient": bool(policy.uses_global_gradient),
        "uses_hidden_axis_leakage": False,
        "uses_target_map_leakage": False,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "snapshot_interval": int(snapshot_interval),
        "initial_state_hash": initial_hash,
        "final_state_hash": _state_hash(state),
        "initial_symmetry_success": bool(initial["audit"]["success"]),
        "oriented_fraction": float(axis["orientedFraction"]),
        "axis_strength": float(axis["axisStrength"]),
        "axis_angle_radians": axis["axisAngleRadians"],
        "selected_axis": axis["selectedAxis"],
        "target_axis_alignment": axis["targetAxisAlignment"],
        "label_match_fraction": labels["labelMatchFraction"],
        "label_diversity": int(labels["labelDiversity"]),
        "label_counts_json": canonical_json(labels["labelCounts"]),
        "pattern_score": float(final["patternScore"]),
        "symmetry_broken": bool(final["symmetryBroken"]),
        "success": bool(final["success"]),
        "occupancy_preserved": True,
        "cell_count": len(state),
    }
    return row, trace_rows, initial["audit"]


def symmetry_task_catalog_rows(tasks: Sequence[SymmetryTaskSpec]) -> list[dict[str, Any]]:
    return [
        {
            "research_step_id": "S10",
            "schema_version": SYMMETRY_SCHEMA_VERSION,
            "task_id": task.task_id,
            "motif": task.motif,
            "target_kind": task.target_kind,
            "target_axis": task.target_axis,
            "grid_size": int(task.grid_size),
            "description": task.description,
            "task_record_json": canonical_json(task.to_record()),
        }
        for task in tasks
    ]


def symmetry_policy_catalog_rows(policies: Sequence[SymmetryPolicySpec]) -> list[dict[str, Any]]:
    rows = []
    for policy in policies:
        audit = audit_symmetry_policy_payload(policy)
        rows.append(
            {
                "research_step_id": "S10",
                "schema_version": SYMMETRY_SCHEMA_VERSION,
                "policy_id": policy.policy_id,
                "policy_family": policy.family,
                "information_scope": policy.information_scope,
                "is_local_only_policy": bool(policy.is_local_only),
                "is_global_information_baseline": bool(policy.uses_organizer or policy.uses_global_gradient),
                "uses_organizer": bool(policy.uses_organizer),
                "uses_global_gradient": bool(policy.uses_global_gradient),
                "description": policy.description,
                "policy_record_json": canonical_json(policy.to_record()),
                "audit_success": bool(audit["success"]),
                "audit_errors_json": canonical_json(audit["errors"]),
                "audit_payload_hash": audit["payloadHash"],
            }
        )
    return rows


def axis_consistency_rows(run_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in run_rows:
        groups.setdefault((str(row["task_id"]), str(row["policy_id"])), []).append(row)
    rows: list[dict[str, Any]] = []
    for (task_id, policy_id), group in sorted(groups.items()):
        axes = [str(row["selected_axis"]) for row in group]
        counts = Counter(axes)
        dominant_axis, dominant_count = counts.most_common(1)[0]
        rows.append(
            {
                "research_step_id": "S10",
                "task_id": task_id,
                "motif": str(group[0]["motif"]),
                "policy_id": policy_id,
                "policy_family": str(group[0]["policy_family"]),
                "is_local_only_policy": bool(group[0]["is_local_only_policy"]),
                "is_global_information_baseline": bool(group[0]["is_global_information_baseline"]),
                "run_count": len(group),
                "dominant_selected_axis": dominant_axis,
                "dominant_axis_fraction": dominant_count / max(1.0, float(len(group))),
                "x_axis_fraction": counts.get("x", 0) / max(1.0, float(len(group))),
                "y_axis_fraction": counts.get("y", 0) / max(1.0, float(len(group))),
                "none_axis_fraction": counts.get("none", 0) / max(1.0, float(len(group))),
                "success_rate": sum(bool(row["success"]) for row in group) / max(1.0, float(len(group))),
                "symmetry_broken_rate": sum(bool(row["symmetry_broken"]) for row in group) / max(1.0, float(len(group))),
                "axis_counts_json": canonical_json(dict(sorted(counts.items()))),
            }
        )
    return rows
