"""Batched GPU grid simulations for E05 S11.

The S11 engine is a deliberately narrow torch implementation for fixed,
fully occupied square-grid label dynamics.  It validates GPU batching against
the same deterministic CPU tensor kernel before using the GPU for larger
scrambled-pattern sweeps.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .recovery import PROHIBITED_POLICY_PAYLOAD_KEYS, _json_ready, _sha16
from .targets import canonical_json


GPU_BATCH_SCHEMA_VERSION = "e05_s11_gpu_batch_sweeps.v1"
GPU_POLICY_AUDIT_VERSION = "e05_s11_gpu_policy_leakage_audit.v1"
GPU_VALIDATION_VERSION = "e05_s11_cpu_gpu_validation.v1"

PROHIBITED_GPU_LOCAL_KEYS = PROHIBITED_POLICY_PAYLOAD_KEYS | {
    "axis_oracle",
    "expected_labels",
    "global_gradient",
    "global_grid",
    "global_target",
    "organizer_position",
    "target_label_map",
    "target_labels",
    "target_map",
    "whole_target_labels",
}


@dataclass(frozen=True)
class GPUTargetSpec:
    """Square-grid label target used by the S11 batch engine."""

    target_id: str
    motif: str
    width: int
    height: int
    class_names: tuple[str, ...]
    target_labels: tuple[tuple[int, ...], ...]
    description: str

    @property
    def class_count(self) -> int:
        return len(self.class_names)

    @property
    def node_count(self) -> int:
        return int(self.width) * int(self.height)

    def labels_array(self) -> np.ndarray:
        return np.asarray(self.target_labels, dtype=np.int64)

    def to_record(self) -> dict[str, Any]:
        return {
            "schemaVersion": GPU_BATCH_SCHEMA_VERSION,
            "targetId": self.target_id,
            "motif": self.motif,
            "width": int(self.width),
            "height": int(self.height),
            "nodeCount": int(self.node_count),
            "classNames": list(self.class_names),
            "classCount": int(self.class_count),
            "description": self.description,
            "targetLabels": [list(row) for row in self.target_labels],
        }


@dataclass(frozen=True)
class GPUPolicySpec:
    """Policy declaration for S11 batched label dynamics."""

    policy_id: str
    family: str
    description: str
    information_scope: str
    update_rule: str
    is_local_only: bool
    self_weight: float = 0.0
    uses_target_map: bool = False
    uses_global_gradient: bool = False
    uses_organizer: bool = False
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_global_information_baseline(self) -> bool:
        return bool(self.uses_target_map or self.uses_global_gradient or self.uses_organizer)

    def to_record(self) -> dict[str, Any]:
        allowed_inputs = [
            "actor_current_label",
            "von_neumann_neighbor_labels",
            "public_batch_index",
            "local_random_seed_stream_for_initial_scramble_only",
        ]
        if self.is_global_information_baseline:
            allowed_inputs.append("explicit_target_label_map_or_global_cue")
        return {
            "schemaVersion": GPU_BATCH_SCHEMA_VERSION,
            "policyId": self.policy_id,
            "family": self.family,
            "description": self.description,
            "informationScope": self.information_scope,
            "updateRule": self.update_rule,
            "isLocalOnly": bool(self.is_local_only),
            "isGlobalInformationBaseline": self.is_global_information_baseline,
            "usesTargetMap": bool(self.uses_target_map),
            "usesGlobalGradient": bool(self.uses_global_gradient),
            "usesOrganizer": bool(self.uses_organizer),
            "selfWeight": float(self.self_weight),
            "parameters": dict(self.parameters),
            "allowedInputs": allowed_inputs,
            "leakageBoundary": (
                "Local-only GPU policies receive only actor and von Neumann neighbor labels. "
                "Target-map and global-gradient access is allowed only for explicitly flagged baselines."
            ),
        }


def torch_device_summary(device: str | torch.device | None = None) -> dict[str, Any]:
    requested = "cuda" if device is None else str(device)
    cuda_available = bool(torch.cuda.is_available())
    selected = torch.device(requested if requested != "cuda" or cuda_available else "cpu")
    payload: dict[str, Any] = {
        "schemaVersion": GPU_BATCH_SCHEMA_VERSION,
        "torchVersion": torch.__version__,
        "cudaAvailable": cuda_available,
        "requestedDevice": requested,
        "selectedDevice": str(selected),
        "cudaDeviceCount": int(torch.cuda.device_count()) if cuda_available else 0,
    }
    if cuda_available:
        index = selected.index if selected.type == "cuda" and selected.index is not None else 0
        props = torch.cuda.get_device_properties(index)
        payload.update(
            {
                "cudaDeviceName": torch.cuda.get_device_name(index),
                "cudaCapability": list(torch.cuda.get_device_capability(index)),
                "cudaTotalMemoryBytes": int(props.total_memory),
            }
        )
    return payload


def _labels_hash(labels: np.ndarray | torch.Tensor) -> str:
    if isinstance(labels, torch.Tensor):
        array = labels.detach().to("cpu").contiguous().numpy()
    else:
        array = np.ascontiguousarray(labels)
    digest = hashlib.sha256()
    digest.update(array.astype(np.int16, copy=False).tobytes())
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    return digest.hexdigest()[:16]


def _target_from_labels(
    *,
    target_id: str,
    motif: str,
    labels: np.ndarray,
    class_names: Sequence[str],
    description: str,
) -> GPUTargetSpec:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 2:
        raise ValueError("target labels must be a 2D array")
    if int(labels.min()) < 0 or int(labels.max()) >= len(class_names):
        raise ValueError("target labels reference classes outside class_names")
    return GPUTargetSpec(
        target_id=target_id,
        motif=motif,
        width=int(labels.shape[1]),
        height=int(labels.shape[0]),
        class_names=tuple(str(item) for item in class_names),
        target_labels=tuple(tuple(int(value) for value in row) for row in labels.tolist()),
        description=description,
    )


def build_gpu_target_spec(motif: str, *, width: int, height: int | None = None) -> GPUTargetSpec:
    """Build a compact label target for the S11 batch engine."""

    motif = str(motif)
    height = int(width if height is None else height)
    width = int(width)
    yy, xx = np.mgrid[0:height, 0:width]
    if motif == "gradient":
        labels = np.minimum(2, np.floor(3 * xx / max(1, width)).astype(np.int64))
        return _target_from_labels(
            target_id=f"gpu_gradient_{width}x{height}",
            motif=motif,
            labels=labels,
            class_names=("left", "middle", "right"),
            description="Three-band left-right computational gradient target for batched GPU sweeps.",
        )
    if motif == "stripes":
        labels = (xx % 2).astype(np.int64)
        return _target_from_labels(
            target_id=f"gpu_stripes_{width}x{height}",
            motif=motif,
            labels=labels,
            class_names=("stripe_a", "stripe_b"),
            description="Alternating vertical stripe label target for batched GPU sweeps.",
        )
    if motif == "ring":
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        radius = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        max_radius = max(1.0, min(center_x, center_y))
        labels = np.full((height, width), 2, dtype=np.int64)
        labels[radius <= 0.50 * max_radius] = 0
        labels[(radius > 0.50 * max_radius) & (radius <= 0.90 * max_radius)] = 1
        return _target_from_labels(
            target_id=f"gpu_ring_{width}x{height}",
            motif=motif,
            labels=labels,
            class_names=("core", "ring", "outer"),
            description="Core, ring, and outer-region target for batched GPU sweeps.",
        )
    if motif == "boundary":
        labels = np.zeros((height, width), dtype=np.int64)
        labels[1:-1, 1:-1] = 1
        return _target_from_labels(
            target_id=f"gpu_boundary_{width}x{height}",
            motif=motif,
            labels=labels,
            class_names=("boundary", "core"),
            description="Perimeter boundary around core target for batched GPU sweeps.",
        )
    if motif == "appendage":
        labels = np.zeros((height, width), dtype=np.int64)
        center_x = width // 2
        center_y = height // 2
        band = max(1, round(height / 6))
        labels[max(0, center_y - band) : min(height, center_y + band + 1), max(0, center_x - 2) : center_x + 1] = 1
        labels[max(0, center_y - band) : min(height, center_y + band + 1), center_x + 1 :] = 2
        labels[center_y, center_x] = 3
        return _target_from_labels(
            target_id=f"gpu_appendage_{width}x{height}",
            motif=motif,
            labels=labels,
            class_names=("boundary", "core", "appendage", "organizer"),
            description="Abstract organizer-core-appendage label target for batched GPU sweeps.",
        )
    raise ValueError(f"unknown GPU target motif: {motif}")


def standard_gpu_validation_targets() -> tuple[GPUTargetSpec, ...]:
    return (
        build_gpu_target_spec("gradient", width=7, height=5),
        build_gpu_target_spec("ring", width=9, height=9),
        build_gpu_target_spec("appendage", width=9, height=7),
    )


def standard_gpu_sweep_targets() -> tuple[GPUTargetSpec, ...]:
    return (
        build_gpu_target_spec("gradient", width=24, height=18),
        build_gpu_target_spec("stripes", width=24, height=18),
        build_gpu_target_spec("ring", width=25, height=25),
        build_gpu_target_spec("boundary", width=24, height=18),
        build_gpu_target_spec("appendage", width=25, height=17),
    )


def standard_gpu_policy_specs() -> tuple[GPUPolicySpec, ...]:
    return (
        GPUPolicySpec(
            policy_id="local_neighbor_majority_gpu",
            family="local_only_label_diffusion",
            description="Batched local-only von Neumann neighbor-majority label update.",
            information_scope="local_only",
            update_rule="neighbor_majority",
            is_local_only=True,
            self_weight=0.0,
        ),
        GPUPolicySpec(
            policy_id="local_inertia_smoothing_gpu",
            family="local_only_label_diffusion",
            description="Batched local-only neighbor-majority update with actor self-label inertia.",
            information_scope="local_only",
            update_rule="neighbor_majority",
            is_local_only=True,
            self_weight=2.0,
            parameters={"selfWeightInterpretation": "center-cell vote weight in local majority kernel"},
        ),
        GPUPolicySpec(
            policy_id="explicit_target_relaxation_gpu",
            family="global_target_map_baseline",
            description="Explicit target-map relaxation baseline that writes mismatched labels toward the target in phases.",
            information_scope="explicit_target_map_baseline",
            update_rule="target_relaxation",
            is_local_only=False,
            uses_target_map=True,
            parameters={"updatePeriod": 4},
        ),
    )


def _audit_mapping_keys(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in PROHIBITED_GPU_LOCAL_KEYS:
                errors.append(f"{path}.{key_text} exposes prohibited key")
            errors.extend(_audit_mapping_keys(item, f"{path}.{key_text}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_audit_mapping_keys(item, f"{path}[{index}]"))
    return errors


def audit_gpu_policy_payload(policy: GPUPolicySpec | Mapping[str, Any]) -> dict[str, Any]:
    record = policy.to_record() if isinstance(policy, GPUPolicySpec) else dict(policy)
    is_local_only = bool(record.get("isLocalOnly", record.get("is_local_only", False)))
    is_global_baseline = bool(record.get("isGlobalInformationBaseline", False))
    errors: list[str] = []
    if is_local_only:
        errors.extend(_audit_mapping_keys(record))
        serialized = json.dumps(_json_ready(record), sort_keys=True)
        suspicious = [
            token
            for token in ("target_map", "target_labels", "global_gradient", "organizer_position", "whole_target")
            if token in serialized
        ]
        if suspicious:
            errors.append(f"serialized local-only policy mentions prohibited token(s): {sorted(set(suspicious))}")
    else:
        if not is_global_baseline:
            errors.append("non-local GPU policy is not explicitly flagged as a global-information baseline")
    return {
        "schemaVersion": GPU_POLICY_AUDIT_VERSION,
        "researchStepId": "S11",
        "policyId": str(record.get("policyId", record.get("policy_id", "unknown"))),
        "isLocalOnly": is_local_only,
        "isGlobalInformationBaseline": is_global_baseline,
        "usesTargetMap": bool(record.get("usesTargetMap", False)),
        "usesGlobalGradient": bool(record.get("usesGlobalGradient", False)),
        "usesOrganizer": bool(record.get("usesOrganizer", False)),
        "success": len(errors) == 0,
        "errorCount": len(errors),
        "errors": errors,
        "payloadHash": _sha16(record),
    }


def gpu_target_catalog_rows(targets: Sequence[GPUTargetSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        labels = target.labels_array()
        values, counts = np.unique(labels, return_counts=True)
        class_counts = {target.class_names[int(value)]: int(count) for value, count in zip(values, counts)}
        rows.append(
            {
                "research_step_id": "S11",
                "schema_version": GPU_BATCH_SCHEMA_VERSION,
                "target_id": target.target_id,
                "motif": target.motif,
                "width": int(target.width),
                "height": int(target.height),
                "node_count": int(target.node_count),
                "class_count": int(target.class_count),
                "class_names_json": canonical_json(list(target.class_names)),
                "class_counts_json": canonical_json(class_counts),
                "target_hash": _labels_hash(labels),
                "target_record_json": canonical_json(target.to_record()),
            }
        )
    return rows


def gpu_policy_catalog_rows(policies: Sequence[GPUPolicySpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        audit = audit_gpu_policy_payload(policy)
        rows.append(
            {
                "research_step_id": "S11",
                "schema_version": GPU_BATCH_SCHEMA_VERSION,
                "policy_id": policy.policy_id,
                "policy_family": policy.family,
                "information_scope": policy.information_scope,
                "update_rule": policy.update_rule,
                "is_local_only_policy": bool(policy.is_local_only),
                "is_global_information_baseline": policy.is_global_information_baseline,
                "uses_target_map": bool(policy.uses_target_map),
                "uses_global_gradient": bool(policy.uses_global_gradient),
                "uses_organizer": bool(policy.uses_organizer),
                "description": policy.description,
                "policy_record_json": canonical_json(policy.to_record()),
                "audit_success": bool(audit["success"]),
                "audit_errors_json": canonical_json(audit["errors"]),
                "audit_payload_hash": audit["payloadHash"],
            }
        )
    return rows


def make_scrambled_label_batch(
    target: GPUTargetSpec,
    seeds: Sequence[int],
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Create one deterministic deranged label grid per seed."""

    target_flat = target.labels_array().reshape(-1)
    rows: list[np.ndarray] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        permuted = target_flat[rng.permutation(target_flat.size)]
        if np.array_equal(permuted, target_flat) and target_flat.size > 1:
            permuted = np.roll(permuted, 1)
        rows.append(permuted.reshape(target.height, target.width).astype(np.int64, copy=False))
    return torch.as_tensor(np.stack(rows, axis=0), dtype=torch.long, device=torch.device(device))


def _neighbor_majority_step(state: torch.Tensor, *, class_count: int, self_weight: float) -> torch.Tensor:
    one_hot = F.one_hot(state, num_classes=int(class_count)).permute(0, 3, 1, 2).to(torch.float32)
    kernel = torch.zeros((int(class_count), 1, 3, 3), dtype=torch.float32, device=state.device)
    kernel[:, 0, 0, 1] = 1.0
    kernel[:, 0, 1, 0] = 1.0
    kernel[:, 0, 1, 2] = 1.0
    kernel[:, 0, 2, 1] = 1.0
    kernel[:, 0, 1, 1] = float(self_weight)
    counts = F.conv2d(one_hot, kernel, padding=1, groups=int(class_count))
    return torch.argmax(counts, dim=1).to(torch.long)


def _target_relaxation_step(
    state: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    step: int,
    update_period: int,
) -> torch.Tensor:
    yy = torch.arange(state.shape[1], dtype=torch.long, device=state.device).view(1, -1, 1)
    xx = torch.arange(state.shape[2], dtype=torch.long, device=state.device).view(1, 1, -1)
    phase = (xx + 3 * yy + int(step)) % max(1, int(update_period))
    mask = phase.eq(0) & state.ne(target_labels)
    return torch.where(mask, target_labels, state)


def _metric_tensors(state: torch.Tensor, target_labels: torch.Tensor) -> dict[str, torch.Tensor]:
    match = state.eq(target_labels)
    label_match = match.to(torch.float32).mean(dim=(1, 2))
    horizontal_disagreement = state[:, :, 1:].ne(state[:, :, :-1]).to(torch.float32).mean(dim=(1, 2))
    vertical_disagreement = state[:, 1:, :].ne(state[:, :-1, :]).to(torch.float32).mean(dim=(1, 2))
    edge_disagreement = 0.5 * (horizontal_disagreement + vertical_disagreement)
    exact = match.flatten(1).all(dim=1)
    return {
        "label_match_fraction": label_match,
        "hamming_error": 1.0 - label_match,
        "edge_disagreement": edge_disagreement,
        "exact_match": exact,
    }


def simulate_batched_labels(
    target: GPUTargetSpec,
    policy: GPUPolicySpec,
    initial_labels: torch.Tensor,
    *,
    steps: int,
    device: str | torch.device,
    trace_interval: int = 10,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Run a deterministic batched tensor simulation and return trace rows."""

    selected_device = torch.device(device)
    state = initial_labels.to(selected_device, dtype=torch.long).clone()
    target_labels = torch.as_tensor(target.labels_array(), dtype=torch.long, device=selected_device).unsqueeze(0)
    if state.ndim != 3 or state.shape[1:] != target_labels.shape[1:]:
        raise ValueError("initial label batch shape does not match target labels")
    trace_rows: list[dict[str, Any]] = []

    def append_trace(step: int, reason: str) -> None:
        metrics = _metric_tensors(state, target_labels)
        for batch_index in range(state.shape[0]):
            trace_rows.append(
                {
                    "research_step_id": "S11",
                    "schema_version": GPU_BATCH_SCHEMA_VERSION,
                    "target_id": target.target_id,
                    "motif": target.motif,
                    "policy_id": policy.policy_id,
                    "policy_family": policy.family,
                    "batch_index": int(batch_index),
                    "step": int(step),
                    "snapshot_reason": reason,
                    "device": str(selected_device),
                    "label_match_fraction": float(metrics["label_match_fraction"][batch_index].detach().cpu().item()),
                    "hamming_error": float(metrics["hamming_error"][batch_index].detach().cpu().item()),
                    "edge_disagreement": float(metrics["edge_disagreement"][batch_index].detach().cpu().item()),
                    "exact_match": bool(metrics["exact_match"][batch_index].detach().cpu().item()),
                    "state_hash": _labels_hash(state[batch_index]),
                }
            )

    append_trace(0, "initial_scrambled")
    for step in range(1, int(steps) + 1):
        if policy.update_rule == "neighbor_majority":
            state = _neighbor_majority_step(state, class_count=target.class_count, self_weight=policy.self_weight)
        elif policy.update_rule == "target_relaxation":
            state = _target_relaxation_step(
                state,
                target_labels,
                step=step,
                update_period=int(policy.parameters.get("updatePeriod", 4)),
            )
        else:
            raise ValueError(f"unknown GPU policy update rule: {policy.update_rule}")
        if step % int(trace_interval) == 0 or step == int(steps):
            append_trace(step, "interval" if step < int(steps) else "final")
    return state, trace_rows


def run_batched_gpu_sweep(
    target: GPUTargetSpec,
    policy: GPUPolicySpec,
    seeds: Sequence[int],
    *,
    steps: int,
    device: str | torch.device,
    trace_interval: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[np.ndarray]]:
    """Run one target-policy batch and return per-seed rows, trace rows, final labels."""

    device = torch.device(device)
    initial = make_scrambled_label_batch(target, seeds, device=device)
    initial_hashes = [_labels_hash(initial[index]) for index in range(initial.shape[0])]
    final, trace_rows = simulate_batched_labels(
        target,
        policy,
        initial,
        steps=int(steps),
        device=device,
        trace_interval=int(trace_interval),
    )
    target_labels = torch.as_tensor(target.labels_array(), dtype=torch.long, device=device).unsqueeze(0)
    initial_metrics = _metric_tensors(initial, target_labels)
    final_metrics = _metric_tensors(final, target_labels)
    rows: list[dict[str, Any]] = []
    final_arrays: list[np.ndarray] = []
    for index, seed in enumerate(seeds):
        initial_error = float(initial_metrics["hamming_error"][index].detach().cpu().item())
        final_error = float(final_metrics["hamming_error"][index].detach().cpu().item())
        reduction = 0.0 if initial_error <= 1e-12 else (initial_error - final_error) / initial_error
        final_array = final[index].detach().to("cpu").numpy().astype(np.int64, copy=True)
        final_arrays.append(final_array)
        rows.append(
            {
                "research_step_id": "S11",
                "schema_version": GPU_BATCH_SCHEMA_VERSION,
                "run_id": f"S11::{target.target_id}::{policy.policy_id}::seed{int(seed)}",
                "target_id": target.target_id,
                "motif": target.motif,
                "width": int(target.width),
                "height": int(target.height),
                "node_count": int(target.node_count),
                "class_count": int(target.class_count),
                "policy_id": policy.policy_id,
                "policy_family": policy.family,
                "update_rule": policy.update_rule,
                "information_scope": policy.information_scope,
                "is_local_only_policy": bool(policy.is_local_only),
                "is_global_information_baseline": policy.is_global_information_baseline,
                "uses_target_map": bool(policy.uses_target_map),
                "uses_global_gradient": bool(policy.uses_global_gradient),
                "uses_organizer": bool(policy.uses_organizer),
                "uses_hidden_target_map_leakage": False,
                "seed": int(seed),
                "steps": int(steps),
                "trace_interval": int(trace_interval),
                "device": str(device),
                "initial_state_hash": initial_hashes[index],
                "final_state_hash": _labels_hash(final[index]),
                "target_hash": _labels_hash(target.labels_array()),
                "initial_label_match_fraction": float(initial_metrics["label_match_fraction"][index].detach().cpu().item()),
                "final_label_match_fraction": float(final_metrics["label_match_fraction"][index].detach().cpu().item()),
                "initial_hamming_error": initial_error,
                "final_hamming_error": final_error,
                "relative_error_reduction": float(reduction),
                "initial_edge_disagreement": float(initial_metrics["edge_disagreement"][index].detach().cpu().item()),
                "final_edge_disagreement": float(final_metrics["edge_disagreement"][index].detach().cpu().item()),
                "exact_match": bool(final_metrics["exact_match"][index].detach().cpu().item()),
                "improved": bool(final_error < initial_error - 1e-12),
                "occupancy_preserved": True,
                "cell_count": int(target.node_count),
            }
        )
    return rows, trace_rows, final_arrays


def simulate_label_frame_sequence(
    target: GPUTargetSpec,
    policy: GPUPolicySpec,
    *,
    seed: int,
    steps: int,
    device: str | torch.device,
    frame_interval: int,
) -> list[np.ndarray]:
    """Return label-grid frames for one seeded run."""

    selected_device = torch.device(device)
    state = make_scrambled_label_batch(target, [int(seed)], device=selected_device)
    target_labels = torch.as_tensor(target.labels_array(), dtype=torch.long, device=selected_device).unsqueeze(0)
    frames = [state[0].detach().to("cpu").numpy().astype(np.int64, copy=True)]
    for step in range(1, int(steps) + 1):
        if policy.update_rule == "neighbor_majority":
            state = _neighbor_majority_step(state, class_count=target.class_count, self_weight=policy.self_weight)
        elif policy.update_rule == "target_relaxation":
            state = _target_relaxation_step(
                state,
                target_labels,
                step=step,
                update_period=int(policy.parameters.get("updatePeriod", 4)),
            )
        else:
            raise ValueError(f"unknown GPU policy update rule: {policy.update_rule}")
        if step % int(frame_interval) == 0 or step == int(steps):
            frames.append(state[0].detach().to("cpu").numpy().astype(np.int64, copy=True))
    return frames


def cpu_gpu_validation_rows(
    targets: Sequence[GPUTargetSpec],
    policies: Sequence[GPUPolicySpec],
    seeds: Sequence[int],
    *,
    steps: int,
    trace_interval: int,
    gpu_device: str = "cuda",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not torch.cuda.is_available():
        return [
            {
                "research_step_id": "S11",
                "schema_version": GPU_VALIDATION_VERSION,
                "check_id": "cuda_available_for_cpu_gpu_validation",
                "target_id": None,
                "policy_id": None,
                "success": False,
                "max_abs_metric_delta": None,
                "cpu_hashes_json": "[]",
                "gpu_hashes_json": "[]",
                "detail": "torch.cuda.is_available() is false",
            }
        ]
    for target in targets:
        for policy in policies:
            initial = make_scrambled_label_batch(target, seeds, device="cpu")
            cpu_final, _ = simulate_batched_labels(
                target,
                policy,
                initial,
                steps=int(steps),
                device="cpu",
                trace_interval=int(trace_interval),
            )
            gpu_final, _ = simulate_batched_labels(
                target,
                policy,
                initial,
                steps=int(steps),
                device=gpu_device,
                trace_interval=int(trace_interval),
            )
            gpu_cpu = gpu_final.detach().to("cpu")
            cpu_target = torch.as_tensor(target.labels_array(), dtype=torch.long, device="cpu").unsqueeze(0)
            cpu_metrics = _metric_tensors(cpu_final, cpu_target)
            gpu_metrics = _metric_tensors(gpu_cpu, cpu_target)
            deltas = [
                torch.max(torch.abs(cpu_metrics[key].to(torch.float64) - gpu_metrics[key].to(torch.float64))).item()
                for key in ("label_match_fraction", "hamming_error", "edge_disagreement")
            ]
            cpu_hashes = [_labels_hash(cpu_final[index]) for index in range(cpu_final.shape[0])]
            gpu_hashes = [_labels_hash(gpu_cpu[index]) for index in range(gpu_cpu.shape[0])]
            rows.append(
                {
                    "research_step_id": "S11",
                    "schema_version": GPU_VALIDATION_VERSION,
                    "check_id": "cpu_gpu_tensor_parity",
                    "target_id": target.target_id,
                    "motif": target.motif,
                    "policy_id": policy.policy_id,
                    "steps": int(steps),
                    "seed_count": len(seeds),
                    "success": bool(torch.equal(cpu_final, gpu_cpu) and max(deltas) <= 1e-12),
                    "max_abs_metric_delta": float(max(deltas)),
                    "cpu_hashes_json": canonical_json(cpu_hashes),
                    "gpu_hashes_json": canonical_json(gpu_hashes),
                    "detail": "final label tensors and selected metrics match exactly between CPU and GPU kernels",
                }
            )
    return rows


def seed_reproducibility_rows(
    target: GPUTargetSpec,
    policy: GPUPolicySpec,
    seeds: Sequence[int],
    *,
    steps: int,
    trace_interval: int,
    device: str,
) -> list[dict[str, Any]]:
    first_rows, _, _ = run_batched_gpu_sweep(
        target,
        policy,
        seeds,
        steps=int(steps),
        device=device,
        trace_interval=int(trace_interval),
    )
    second_rows, _, _ = run_batched_gpu_sweep(
        target,
        policy,
        seeds,
        steps=int(steps),
        device=device,
        trace_interval=int(trace_interval),
    )
    rows: list[dict[str, Any]] = []
    for left, right in zip(first_rows, second_rows):
        rows.append(
            {
                "research_step_id": "S11",
                "schema_version": GPU_VALIDATION_VERSION,
                "check_id": "seed_reproducibility",
                "target_id": target.target_id,
                "policy_id": policy.policy_id,
                "seed": int(left["seed"]),
                "device": device,
                "success": bool(
                    left["initial_state_hash"] == right["initial_state_hash"]
                    and left["final_state_hash"] == right["final_state_hash"]
                    and abs(float(left["final_hamming_error"]) - float(right["final_hamming_error"])) <= 1e-12
                ),
                "initial_hash_first": left["initial_state_hash"],
                "initial_hash_second": right["initial_state_hash"],
                "final_hash_first": left["final_state_hash"],
                "final_hash_second": right["final_state_hash"],
                "detail": "same seed list and GPU batch kernel reproduce identical initial and final state hashes",
            }
        )
    return rows


def gpu_sweep_summary_rows(run_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in run_rows:
        groups.setdefault((str(row["target_id"]), str(row["policy_id"])), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (target_id, policy_id), rows in sorted(groups.items()):
        summaries.append(
            {
                "research_step_id": "S11",
                "schema_version": GPU_BATCH_SCHEMA_VERSION,
                "target_id": target_id,
                "motif": str(rows[0]["motif"]),
                "policy_id": policy_id,
                "policy_family": str(rows[0]["policy_family"]),
                "is_local_only_policy": bool(rows[0]["is_local_only_policy"]),
                "is_global_information_baseline": bool(rows[0]["is_global_information_baseline"]),
                "run_count": len(rows),
                "exact_match_rate": float(np.mean([bool(row["exact_match"]) for row in rows])),
                "improved_rate": float(np.mean([bool(row["improved"]) for row in rows])),
                "initial_hamming_error_mean": float(np.mean([float(row["initial_hamming_error"]) for row in rows])),
                "final_hamming_error_mean": float(np.mean([float(row["final_hamming_error"]) for row in rows])),
                "relative_error_reduction_mean": float(np.mean([float(row["relative_error_reduction"]) for row in rows])),
                "final_label_match_fraction_mean": float(np.mean([float(row["final_label_match_fraction"]) for row in rows])),
                "final_edge_disagreement_mean": float(np.mean([float(row["final_edge_disagreement"]) for row in rows])),
            }
        )
    return summaries


def labels_to_rgb(labels: np.ndarray, *, class_count: int) -> np.ndarray:
    palette = np.asarray(
        [
            [46, 113, 190],
            [232, 173, 53],
            [78, 160, 100],
            [183, 71, 87],
            [118, 88, 166],
            [80, 80, 80],
        ],
        dtype=np.uint8,
    )
    labels = np.asarray(labels, dtype=np.int64)
    return palette[labels % min(len(palette), max(1, int(class_count)))]
