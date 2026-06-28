"""Scale-transfer benchmark helpers for E05 S09."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import evaluate_morphospace_metrics
from .recovery import (
    PROHIBITED_POLICY_PAYLOAD_KEYS,
    RecoveryPolicySpec,
    _json_ready,
    _sha16,
    audit_recovery_policy_payload,
    run_recovery_benchmark,
    scramble_target_state,
)
from .targets import (
    TargetMorphology,
    build_boundary_target,
    build_gradient_target,
    build_organ_like_target,
    build_ring_target,
    build_stripes_target,
    canonical_json,
)


SCALING_SCHEMA_VERSION = "e05_s09_scaling_transfer.v1"
TARGET_SCALING_RULES_VERSION = "e05_s09_target_scaling_rules.v1"
SCALING_POLICY_AUDIT_VERSION = "e05_s09_scaling_policy_leakage_audit.v1"

PROHIBITED_SCALING_POLICY_KEYS = PROHIBITED_POLICY_PAYLOAD_KEYS | {
    "all_grid_sizes",
    "global_grid_size",
    "global_height",
    "global_node_count",
    "global_size",
    "global_width",
    "heldout_size",
    "heldout_sizes",
    "scale_oracle",
    "target_size_map",
    "whole_scaled_target",
}

SIZE_CLASS_ORDER = {
    "training_small": 0,
    "heldout_medium": 1,
    "heldout_large": 2,
}


@dataclass(frozen=True)
class ScaledTargetSpec:
    """Target builder plus scale-transfer metadata."""

    motif: str
    size_class: str
    width: int
    height: int
    scale_rule_id: str
    scale_rule: str
    is_heldout_size: bool

    @property
    def size_key(self) -> str:
        if self.motif == "ring":
            return f"{self.width}x{self.width}"
        return f"{self.width}x{self.height}"

    def to_record(self, target: TargetMorphology | None = None) -> dict[str, Any]:
        return {
            "schemaVersion": TARGET_SCALING_RULES_VERSION,
            "motif": self.motif,
            "sizeClass": self.size_class,
            "sizeKey": self.size_key,
            "width": int(self.width),
            "height": int(self.height),
            "nodeCount": None if target is None else len(target.substrate.nodes),
            "targetId": None if target is None else target.target_id,
            "scaleRuleId": self.scale_rule_id,
            "scaleRule": self.scale_rule,
            "isHeldoutSize": bool(self.is_heldout_size),
        }


def standard_scaling_target_specs() -> tuple[ScaledTargetSpec, ...]:
    """Return small training anchors plus medium/large held-out target sizes."""

    rules = {
        "gradient": (
            "gradient_x_normalized",
            "Width and height increase; scalar rank follows row-major order and AP coordinate remains x/(width-1).",
        ),
        "stripes": (
            "vertical_stripe_period_two",
            "Width and height increase; alternating two-column local stripe grammar is preserved, adding more stripes.",
        ),
        "ring": (
            "radial_threshold_fraction",
            "Odd square sizes increase; core/ring/outer roles use fixed fractions of the substrate radius.",
        ),
        "boundary": (
            "perimeter_core_fraction",
            "Width and height increase; outer perimeter remains boundary and the interior scales as core.",
        ),
        "organ_like": (
            "centered_core_appendage_band",
            "Width and height increase; organizer stays centered, core occupies a center-left band, and appendage extends rightward.",
        ),
    }
    sizes = {
        "gradient": (("training_small", 5, 4), ("heldout_medium", 8, 6), ("heldout_large", 10, 8)),
        "stripes": (("training_small", 6, 4), ("heldout_medium", 10, 6), ("heldout_large", 12, 8)),
        "ring": (("training_small", 7, 7), ("heldout_medium", 9, 9), ("heldout_large", 11, 11)),
        "boundary": (("training_small", 6, 5), ("heldout_medium", 9, 7), ("heldout_large", 12, 9)),
        "organ_like": (("training_small", 7, 5), ("heldout_medium", 10, 7), ("heldout_large", 12, 10)),
    }
    specs: list[ScaledTargetSpec] = []
    for motif, motif_sizes in sizes.items():
        rule_id, rule = rules[motif]
        for size_class, width, height in motif_sizes:
            specs.append(
                ScaledTargetSpec(
                    motif=motif,
                    size_class=size_class,
                    width=int(width),
                    height=int(height),
                    scale_rule_id=rule_id,
                    scale_rule=rule,
                    is_heldout_size=size_class.startswith("heldout_"),
                )
            )
    return tuple(specs)


def build_scaled_target(spec: ScaledTargetSpec) -> TargetMorphology:
    if spec.motif == "gradient":
        return build_gradient_target(width=spec.width, height=spec.height)
    if spec.motif == "stripes":
        return build_stripes_target(width=spec.width, height=spec.height)
    if spec.motif == "ring":
        return build_ring_target(size=spec.width)
    if spec.motif == "boundary":
        return build_boundary_target(width=spec.width, height=spec.height)
    if spec.motif == "organ_like":
        return build_organ_like_target(width=spec.width, height=spec.height)
    raise ValueError(f"unknown scaling motif: {spec.motif}")


def build_scaling_target_panel(
    specs: Sequence[ScaledTargetSpec] | None = None,
) -> tuple[tuple[ScaledTargetSpec, TargetMorphology], ...]:
    selected = standard_scaling_target_specs() if specs is None else tuple(specs)
    return tuple((spec, build_scaled_target(spec)) for spec in selected)


def target_scaling_rule_rows(panel: Sequence[tuple[ScaledTargetSpec, TargetMorphology]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec, target in panel:
        row = spec.to_record(target)
        row.update(
            {
                "research_step_id": "S09",
                "target_record_hash": _sha16(target.to_record()),
                "target_record_json": canonical_json(target.to_record()),
            }
        )
        rows.append(row)
    return rows


def _audit_mapping_keys(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in PROHIBITED_SCALING_POLICY_KEYS:
                errors.append(f"{path}.{key_text} exposes prohibited key")
            errors.extend(_audit_mapping_keys(item, f"{path}.{key_text}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_audit_mapping_keys(item, f"{path}[{index}]"))
    return errors


def audit_scaling_policy_payload(policy: RecoveryPolicySpec | Mapping[str, Any]) -> dict[str, Any]:
    """Audit S09 policy declarations for hidden size or target-map leakage."""

    record = policy.to_record() if isinstance(policy, RecoveryPolicySpec) else dict(policy)
    base_audit = audit_recovery_policy_payload(policy)
    errors = list(base_audit.get("errors", []))
    errors.extend(_audit_mapping_keys(record))
    serialized = json.dumps(_json_ready(record), sort_keys=True)
    suspicious = [
        token
        for token in (
            "global_size",
            "global_grid_size",
            "target_map",
            "whole_scaled_target",
            "heldout_sizes",
            "scale_oracle",
        )
        if token in serialized
    ]
    if suspicious:
        errors.append(f"serialized policy mentions prohibited scaling token(s): {sorted(set(suspicious))}")
    return {
        "schemaVersion": SCALING_POLICY_AUDIT_VERSION,
        "policyId": str(record.get("policyId", record.get("policy_id", "unknown"))),
        "success": len(errors) == 0,
        "errorCount": len(errors),
        "errors": errors,
        "payloadHash": _sha16(record),
    }


def metric_normalization_rows(
    panel: Sequence[tuple[ScaledTargetSpec, TargetMorphology]],
    *,
    seed: int,
    max_ratio: float = 3.0,
) -> list[dict[str, Any]]:
    """Check exact-zero and matched-scramble normalized metric comparability."""

    exact_rows: list[dict[str, Any]] = []
    scramble_rows: list[dict[str, Any]] = []
    for spec, target in panel:
        exact = evaluate_morphospace_metrics(target, target.target_state())
        exact_success = abs(float(exact["compositeError"])) <= 1e-12 and all(
            abs(float(metric["normalizedValue"])) <= 1e-12 for metric in exact["metrics"]
        )
        exact_rows.append(
            {
                "research_step_id": "S09",
                "check_scope": "exact_target_zero",
                "motif": spec.motif,
                "size_class": spec.size_class,
                "target_id": target.target_id,
                "metric_id": "composite",
                "value": float(exact["compositeError"]),
                "normalized_value": float(exact["compositeError"]),
                "success": bool(exact_success),
                "detail_json": canonical_json({"metricCount": len(exact["metrics"])}),
            }
        )
        scrambled, severity = scramble_target_state(target, seed=int(seed))
        result = evaluate_morphospace_metrics(target, scrambled)
        for metric in result["metrics"]:
            normalized = float(metric["normalizedValue"])
            scramble_rows.append(
                {
                    "research_step_id": "S09",
                    "check_scope": "matched_scramble_metric",
                    "motif": spec.motif,
                    "size_class": spec.size_class,
                    "target_id": target.target_id,
                    "metric_id": str(metric["metricId"]),
                    "value": float(metric["value"]),
                    "normalized_value": normalized,
                    "success": bool(math.isfinite(normalized) and normalized >= 0.0),
                    "detail_json": canonical_json(
                        {
                            "nodeCount": len(target.substrate.nodes),
                            "severityClass": severity["severityClass"],
                            "movedFraction": severity["movedFraction"],
                            "compositeError": result["compositeError"],
                        }
                    ),
                }
            )

    ratio_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in scramble_rows:
        grouped.setdefault((str(row["motif"]), str(row["metric_id"])), []).append(row)
    for (motif, metric_id), rows in sorted(grouped.items()):
        values = [float(row["normalized_value"]) for row in rows if float(row["normalized_value"]) > 1e-12]
        if not values:
            ratio = 1.0
        else:
            ratio = max(values) / max(1e-12, min(values))
        success = math.isfinite(ratio) and ratio <= float(max_ratio)
        ratio_rows.append(
            {
                "research_step_id": "S09",
                "check_scope": "matched_scramble_size_ratio",
                "motif": motif,
                "size_class": "all_size_classes",
                "target_id": "all_scaled_targets",
                "metric_id": metric_id,
                "value": ratio,
                "normalized_value": ratio,
                "success": bool(success),
                "detail_json": canonical_json({"maxAllowedRatio": float(max_ratio), "positiveValueCount": len(values)}),
            }
        )
    return exact_rows + scramble_rows + ratio_rows


def run_scaling_recovery_benchmark(
    spec: ScaledTargetSpec,
    target: TargetMorphology,
    policy: RecoveryPolicySpec,
    *,
    seed: int,
    max_steps: int,
    snapshot_interval: int,
    recovery_threshold: float = 1e-12,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Run one S09 scaled scrambled-recovery transfer benchmark."""

    row, trace_rows, severity = run_recovery_benchmark(
        target,
        policy,
        seed=int(seed),
        max_steps=int(max_steps),
        snapshot_interval=int(snapshot_interval),
        recovery_threshold=float(recovery_threshold),
    )
    run_id = f"S09::{target.target_id}::{spec.size_class}::{policy.policy_id}::seed{int(seed)}"
    initial_composite = float(row["initial_composite_error"])
    final_composite = float(row["final_composite_error"])
    node_count = max(1, len(target.substrate.nodes))
    row.update(
        {
            "schema_version": SCALING_SCHEMA_VERSION,
            "research_step_id": "S09",
            "run_id": run_id,
            "size_class": spec.size_class,
            "size_key": spec.size_key,
            "scale_rule_id": spec.scale_rule_id,
            "is_heldout_size": bool(spec.is_heldout_size),
            "training_size_class": "training_small",
            "perturbation_class": "deranged_identity_scramble",
            "max_steps_per_cell": float(max_steps) / float(node_count),
            "accepted_swaps_per_cell": float(row["accepted_swap_count"]) / float(node_count),
            "initial_composite_error_per_cell": initial_composite / float(node_count),
            "final_composite_error_per_cell": final_composite / float(node_count),
            "uses_hidden_global_size": False,
            "uses_whole_target_leakage": False,
        }
    )
    for trace_row in trace_rows:
        trace_row.update(
            {
                "schema_version": SCALING_SCHEMA_VERSION,
                "research_step_id": "S09",
                "run_id": run_id,
                "size_class": spec.size_class,
                "size_key": spec.size_key,
                "scale_rule_id": spec.scale_rule_id,
                "is_heldout_size": bool(spec.is_heldout_size),
            }
        )
    severity = dict(severity)
    severity.update(
        {
            "schemaVersion": SCALING_SCHEMA_VERSION,
            "researchStepId": "S09",
            "targetId": target.target_id,
            "sizeClass": spec.size_class,
            "sizeKey": spec.size_key,
            "scaleRuleId": spec.scale_rule_id,
            "isHeldoutSize": bool(spec.is_heldout_size),
        }
    )
    return row, trace_rows, severity


def transfer_gap_rows(run_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize held-out transfer gaps relative to training-small behavior."""

    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in run_rows:
        by_key.setdefault((str(row["motif"]), str(row["policy_id"])), []).append(row)
    rows: list[dict[str, Any]] = []
    for (motif, policy_id), group in sorted(by_key.items()):
        train = [float(row["relative_error_reduction"]) for row in group if row["size_class"] == "training_small"]
        train_mean = float(np.mean(train)) if train else float("nan")
        for size_class in ("heldout_medium", "heldout_large"):
            heldout = [float(row["relative_error_reduction"]) for row in group if row["size_class"] == size_class]
            heldout_mean = float(np.mean(heldout)) if heldout else float("nan")
            rows.append(
                {
                    "research_step_id": "S09",
                    "motif": motif,
                    "policy_id": policy_id,
                    "training_size_class": "training_small",
                    "heldout_size_class": size_class,
                    "training_relative_error_reduction_mean": train_mean,
                    "heldout_relative_error_reduction_mean": heldout_mean,
                    "transfer_gap": train_mean - heldout_mean,
                    "heldout_improved": bool(heldout and heldout_mean > 0.0),
                }
            )
    return rows
