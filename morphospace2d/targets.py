"""Target morphology definitions and validation for E05 S03."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import numpy as np

from .identities import (
    CellIdentity,
    NeighborPreferenceRule,
    actor_components,
    default_identity_schema,
    evaluate_neighbor_preferences,
    validate_identity_catalog,
)
from .substrates import Position, Substrate


TARGET_SCHEMA_VERSION = "e05_s03_target_morphology.v1"
TARGET_ENERGY_VERSION = "e05_s03_target_energy.v1"
LOCAL_TARGET_PAYLOAD_VERSION = "e05_s03_local_target_payload.v1"

COMPONENT_WEIGHTS = {
    "scalar_value": 0.25,
    "ap_coordinate": 0.5,
    "organ_type": 1.0,
    "polarity": 0.25,
    "adhesion_type": 0.5,
}
ORGAN_COLORS = {
    "axis": "#6f8fcf",
    "boundary": "#1f9e89",
    "core": "#f0c541",
    "appendage": "#d95f5f",
    "organizer": "#7b3294",
}
PROHIBITED_LOCAL_PAYLOAD_KEYS = {
    "all_target_positions",
    "all_target_identities",
    "motif",
    "target_id",
    "target_map",
    "global_target_state",
    "hidden",
    "internal_state",
}


def _as_position(value: Sequence[int] | int) -> Position:
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


def _substrate_config(substrate: Substrate) -> dict[str, Any]:
    return {
        "schemaVersion": substrate.schema_version,
        "substrateType": substrate.substrate_type,
        "boundary": substrate.boundary,
        "nodes": [list(node) for node in substrate.nodes],
        "edges": [[list(left), list(right)] for left, right in substrate.edges()],
        "metadata": dict(substrate.metadata),
    }


def substrate_from_config(record: Mapping[str, Any]) -> Substrate:
    nodes = [tuple(int(item) for item in node) for node in record["nodes"]]
    edges = [
        (tuple(int(item) for item in edge[0]), tuple(int(item) for item in edge[1]))
        for edge in record["edges"]
    ]
    adjacency: dict[Position, list[Position]] = {node: [] for node in nodes}
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    return Substrate(
        str(record["substrateType"]),
        tuple(nodes),
        {node: tuple(neighbors) for node, neighbors in adjacency.items()},
        boundary=str(record.get("boundary", "open")),
        metadata=dict(record.get("metadata", {})),
    )


@dataclass(frozen=True)
class TargetMorphology:
    """Target identity assignment plus local constraints on a substrate."""

    target_id: str
    title: str
    motif: str
    substrate: Substrate
    identities_by_position: Mapping[Position, CellIdentity]
    description: str
    tolerance: float = 0.0
    component_weights: Mapping[str, float] = field(default_factory=lambda: dict(COMPONENT_WEIGHTS))
    schema_version: str = TARGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        normalized = {_as_position(position): identity for position, identity in self.identities_by_position.items()}
        object.__setattr__(self, "identities_by_position", normalized)

    def target_state(self) -> dict[Position, CellIdentity]:
        return dict(self.identities_by_position)

    def local_policy_payload(self, position: Position) -> dict[str, Any]:
        """Return actor-local target hints without exposing a whole target map."""

        position = _as_position(position)
        identity = self.identities_by_position[position]
        schema = default_identity_schema()
        payload = {
            "schemaVersion": LOCAL_TARGET_PAYLOAD_VERSION,
            "localDegree": len(self.substrate.neighbors(position)),
            "actorTargetPreferences": identity.components.get("target_neighbor_preferences", []),
            "actorVisibleIdentity": actor_components(identity, schema),
        }
        return _json_ready(payload)

    def validate(self) -> list[str]:
        errors: list[str] = []
        errors.extend(self.substrate.validate())
        substrate_nodes = set(self.substrate.nodes)
        target_nodes = set(self.identities_by_position)
        if target_nodes != substrate_nodes:
            missing = sorted(substrate_nodes - target_nodes)
            extra = sorted(target_nodes - substrate_nodes)
            if missing:
                errors.append(f"missing target identities for substrate nodes: {missing}")
            if extra:
                errors.append(f"target identities outside substrate: {extra}")
        errors.extend(validate_identity_catalog(tuple(self.identities_by_position.values()), default_identity_schema()))
        for position in sorted(target_nodes):
            payload = self.local_policy_payload(position)
            errors.extend(audit_local_target_payload(payload))
        return errors

    def to_record(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "targetId": self.target_id,
            "title": self.title,
            "motif": self.motif,
            "description": self.description,
            "tolerance": self.tolerance,
            "componentWeights": dict(self.component_weights),
            "substrate": _substrate_config(self.substrate),
            "targetCells": [
                {
                    "position": list(position),
                    "identity": identity.to_record(schema=default_identity_schema(), include_hidden=True),
                }
                for position, identity in sorted(self.identities_by_position.items())
            ],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "TargetMorphology":
        substrate = substrate_from_config(record["substrate"])
        identities = {
            tuple(int(item) for item in cell["position"]): CellIdentity.from_record(cell["identity"])
            for cell in record["targetCells"]
        }
        return cls(
            target_id=str(record["targetId"]),
            title=str(record["title"]),
            motif=str(record["motif"]),
            substrate=substrate,
            identities_by_position=identities,
            description=str(record.get("description", "")),
            tolerance=float(record.get("tolerance", 0.0)),
            component_weights=dict(record.get("componentWeights", COMPONENT_WEIGHTS)),
            schema_version=str(record.get("schemaVersion", TARGET_SCHEMA_VERSION)),
        )


def audit_local_target_payload(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    keys = set(payload)
    leaked = sorted(PROHIBITED_LOCAL_PAYLOAD_KEYS & keys)
    if leaked:
        errors.append(f"payload exposes prohibited keys: {leaked}")
    actor_identity = payload.get("actorVisibleIdentity", {})
    if isinstance(actor_identity, Mapping):
        leaked_identity = sorted(PROHIBITED_LOCAL_PAYLOAD_KEYS & set(actor_identity))
        if leaked_identity:
            errors.append(f"actor identity exposes hidden keys: {leaked_identity}")
    else:
        errors.append("actorVisibleIdentity is not a mapping")
    if "actorTargetPreferences" not in payload:
        errors.append("payload is missing actorTargetPreferences")
    return errors


def _numeric_distance(left: Any, right: Any) -> float:
    try:
        return abs(float(left) - float(right))
    except (TypeError, ValueError):
        return 0.0 if left == right else 1.0


def _component_penalty(
    expected: CellIdentity,
    observed: CellIdentity,
    weights: Mapping[str, float],
    numeric_normalizers: Mapping[str, float] | None = None,
) -> float:
    numeric_normalizers = {} if numeric_normalizers is None else dict(numeric_normalizers)
    penalty = 0.0
    for component, weight in weights.items():
        expected_value = expected.components.get(component)
        observed_value = observed.components.get(component)
        if component in {"scalar_value", "ap_coordinate"}:
            normalizer = max(1.0, float(numeric_normalizers.get(component, 1.0)))
            penalty += float(weight) * _numeric_distance(expected_value, observed_value) / normalizer
        elif component == "polarity":
            expected_vec = np.asarray(expected_value, dtype=float)
            observed_vec = np.asarray(observed_value, dtype=float)
            if expected_vec.shape != observed_vec.shape:
                penalty += float(weight)
            else:
                penalty += float(weight) * float(np.linalg.norm(expected_vec - observed_vec))
        else:
            penalty += 0.0 if expected_value == observed_value else float(weight)
    return penalty


def evaluate_target_energy(
    target: TargetMorphology,
    observed_by_position: Mapping[Position, CellIdentity],
) -> dict[str, Any]:
    """Evaluate global target constraints. This is not a local policy payload."""

    observed = {_as_position(position): identity for position, identity in observed_by_position.items()}
    scalar_values = [
        float(identity.components["scalar_value"])
        for identity in target.identities_by_position.values()
        if "scalar_value" in identity.components
    ]
    scalar_range = max(scalar_values) - min(scalar_values) if scalar_values else 1.0
    numeric_normalizers = {"scalar_value": max(1.0, scalar_range), "ap_coordinate": 1.0}
    component_penalty = 0.0
    missing_penalty = 0.0
    extra_penalty = 0.0
    preference_penalty = 0.0
    for position, expected_identity in target.identities_by_position.items():
        observed_identity = observed.get(position)
        if observed_identity is None:
            missing_penalty += 1.0
            continue
        component_penalty += _component_penalty(
            expected_identity,
            observed_identity,
            target.component_weights,
            numeric_normalizers=numeric_normalizers,
        )
        neighbor_identities = [
            observed[neighbor_position]
            for neighbor_position in target.substrate.neighbors(position)
            if neighbor_position in observed
        ]
        preference_penalty += float(evaluate_neighbor_preferences(expected_identity, neighbor_identities)["penalty"])
    extra_positions = sorted(set(observed) - set(target.identities_by_position))
    extra_penalty = float(len(extra_positions))
    total = component_penalty + missing_penalty + extra_penalty + preference_penalty
    return {
        "schemaVersion": TARGET_ENERGY_VERSION,
        "targetId": target.target_id,
        "motif": target.motif,
        "componentPenalty": component_penalty,
        "missingPenalty": missing_penalty,
        "extraPenalty": extra_penalty,
        "preferencePenalty": preference_penalty,
        "totalEnergy": total,
        "withinTolerance": total <= float(target.tolerance),
    }


def _identity_for_position(
    cell_id: int,
    scalar_value: float,
    ap_coordinate: float,
    organ_type: str,
    polarity: Sequence[float],
    adhesion_type: str,
) -> CellIdentity:
    return CellIdentity(
        cell_id=cell_id,
        components={
            "scalar_value": scalar_value,
            "ap_coordinate": ap_coordinate,
            "organ_type": organ_type,
            "polarity": [float(item) for item in polarity],
            "adhesion_type": adhesion_type,
            "target_neighbor_preferences": [],
            "internal_state": {},
        },
    )


def _attach_satisfied_neighbor_preferences(
    substrate: Substrate,
    identities_by_position: Mapping[Position, CellIdentity],
) -> dict[Position, CellIdentity]:
    updated: dict[Position, CellIdentity] = {}
    for position, identity in identities_by_position.items():
        neighbor_identities = [
            identities_by_position[neighbor]
            for neighbor in substrate.neighbors(position)
            if neighbor in identities_by_position
        ]
        organ_counts = Counter(neighbor.components["organ_type"] for neighbor in neighbor_identities)
        adhesion_counts = Counter(neighbor.components["adhesion_type"] for neighbor in neighbor_identities)
        rules: list[dict[str, Any]] = []
        for organ_type, count in sorted(organ_counts.items()):
            if count > 0:
                rules.append(NeighborPreferenceRule("organ_type", organ_type, min_count=count, max_count=count, weight=0.5).to_record())
        own_adhesion_count = adhesion_counts.get(identity.components["adhesion_type"], 0)
        if own_adhesion_count > 0:
            rules.append(
                NeighborPreferenceRule(
                    "adhesion_type",
                    identity.components["adhesion_type"],
                    min_count=own_adhesion_count,
                    max_count=own_adhesion_count,
                    weight=0.25,
                ).to_record()
            )
        components = dict(identity.components)
        components["target_neighbor_preferences"] = rules
        updated[position] = CellIdentity(cell_id=identity.cell_id, components=components)
    return updated


def _build_target(
    target_id: str,
    title: str,
    motif: str,
    substrate: Substrate,
    role_by_position: Mapping[Position, tuple[float, float, str, Sequence[float], str]],
    description: str,
) -> TargetMorphology:
    identities: dict[Position, CellIdentity] = {}
    for cell_id, position in enumerate(sorted(substrate.nodes)):
        scalar_value, ap_coordinate, organ_type, polarity, adhesion_type = role_by_position[position]
        identities[position] = _identity_for_position(
            cell_id,
            scalar_value=scalar_value,
            ap_coordinate=ap_coordinate,
            organ_type=organ_type,
            polarity=polarity,
            adhesion_type=adhesion_type,
        )
    identities = _attach_satisfied_neighbor_preferences(substrate, identities)
    target = TargetMorphology(
        target_id=target_id,
        title=title,
        motif=motif,
        substrate=substrate,
        identities_by_position=identities,
        description=description,
    )
    errors = target.validate()
    if errors:
        raise ValueError(f"invalid target {target_id}: {'; '.join(errors)}")
    return target


def build_gradient_target(width: int = 5, height: int = 4) -> TargetMorphology:
    substrate = Substrate.square_grid(width, height)
    roles = {}
    for x, y in substrate.nodes:
        ap = x / max(1, width - 1)
        roles[(x, y)] = (1 + x + y * width, ap, "axis", [1.0, 0.0], "adhesion_a" if x < width / 2 else "adhesion_b")
    return _build_target(
        f"gradient_x_{width}x{height}",
        "Left-right gradient",
        "gradient",
        substrate,
        roles,
        "Monotone computational anterior-posterior-like gradient on a square grid.",
    )


def build_stripes_target(width: int = 6, height: int = 4) -> TargetMorphology:
    substrate = Substrate.square_grid(width, height)
    roles = {}
    for x, y in substrate.nodes:
        is_boundary_stripe = x % 2 == 0
        roles[(x, y)] = (
            1 + x + y * width,
            x / max(1, width - 1),
            "boundary" if is_boundary_stripe else "core",
            [0.0, 1.0],
            "adhesion_boundary" if is_boundary_stripe else "adhesion_a",
        )
    return _build_target(
        f"vertical_stripes_{width}x{height}",
        "Vertical stripes",
        "stripes",
        substrate,
        roles,
        "Alternating vertical computational role stripes.",
    )


def build_ring_target(size: int = 7) -> TargetMorphology:
    substrate = Substrate.square_grid(size, size)
    center = (size - 1) / 2.0
    roles = {}
    for x, y in substrate.nodes:
        dist = math.sqrt((x - center) ** 2 + (y - center) ** 2)
        outer_threshold = max(1.0, center) * 0.9333333333333333
        inner_threshold = max(1.0, center) * 0.6
        if inner_threshold <= dist <= outer_threshold:
            organ_type = "boundary"
            adhesion = "adhesion_boundary"
        elif dist < inner_threshold:
            organ_type = "core"
            adhesion = "adhesion_a"
        else:
            organ_type = "axis"
            adhesion = "adhesion_b"
        roles[(x, y)] = (1 + x + y * size, x / max(1, size - 1), organ_type, [0.0, 1.0], adhesion)
    return _build_target(f"ring_{size}x{size}", "Ring motif", "ring", substrate, roles, "Concentric computational ring around a core.")


def build_sorted_row_target(length: int = 8) -> TargetMorphology:
    substrate = Substrate.row(length)
    roles = {
        (x,): (x + 1, x / max(1, length - 1), "axis", [1.0, 0.0], "adhesion_a" if x % 2 == 0 else "adhesion_b")
        for x in range(length)
    }
    return _build_target(f"sorted_row_{length}", "Sorted row", "sorted_row", substrate, roles, "One-dimensional sorted scalar baseline target.")


def build_boundary_target(width: int = 6, height: int = 5) -> TargetMorphology:
    substrate = Substrate.square_grid(width, height)
    roles = {}
    for x, y in substrate.nodes:
        is_boundary = x == 0 or y == 0 or x == width - 1 or y == height - 1
        roles[(x, y)] = (
            1 + x + y * width,
            x / max(1, width - 1),
            "boundary" if is_boundary else "core",
            [1.0, 0.0] if is_boundary else [0.0, 1.0],
            "adhesion_boundary" if is_boundary else "adhesion_a",
        )
    return _build_target(
        f"perimeter_boundary_{width}x{height}",
        "Boundary and core",
        "boundary",
        substrate,
        roles,
        "Perimeter boundary around computational core.",
    )


def build_organ_like_target(width: int = 7, height: int = 5) -> TargetMorphology:
    substrate = Substrate.square_grid(width, height)
    center = (width // 2, height // 2)
    band_radius = max(1, round(height / 5))
    y_min = max(1, center[1] - band_radius)
    y_max = min(height - 2, center[1] + band_radius) if height > 2 else center[1]
    core_left = max(1, center[0] - max(1, round(width / 3)))
    roles = {}
    for x, y in substrate.nodes:
        if (x, y) == center:
            organ_type = "organizer"
            adhesion = "adhesion_boundary"
            polarity = [1.0, 0.0]
        elif x >= center[0] + 1 and y_min <= y <= y_max:
            organ_type = "appendage"
            adhesion = "adhesion_b"
            polarity = [1.0, 0.0]
        elif core_left <= x <= center[0] and y_min <= y <= y_max:
            organ_type = "core"
            adhesion = "adhesion_a"
            polarity = [0.0, 1.0]
        else:
            organ_type = "boundary"
            adhesion = "adhesion_boundary"
            polarity = [-1.0, 0.0]
        roles[(x, y)] = (1 + x + y * width, x / max(1, width - 1), organ_type, polarity, adhesion)
    return _build_target(
        f"abstract_organ_like_{width}x{height}",
        "Abstract organ-like motif",
        "organ_like",
        substrate,
        roles,
        "Computational organizer, core, appendage, and boundary motif; not an anatomical claim.",
    )


def build_standard_target_library() -> tuple[TargetMorphology, ...]:
    return (
        build_gradient_target(),
        build_stripes_target(),
        build_ring_target(),
        build_sorted_row_target(),
        build_boundary_target(),
        build_organ_like_target(),
    )


def target_catalog_rows(targets: Sequence[TargetMorphology]) -> list[dict[str, Any]]:
    rows = []
    for target in targets:
        organ_counts = Counter(identity.components["organ_type"] for identity in target.identities_by_position.values())
        rows.append(
            {
                "schema_version": target.schema_version,
                "target_id": target.target_id,
                "title": target.title,
                "motif": target.motif,
                "substrate_type": target.substrate.substrate_type,
                "node_count": len(target.substrate.nodes),
                "edge_count": len(target.substrate.edges()),
                "organ_type_counts_json": canonical_json(dict(sorted(organ_counts.items()))),
                "description": target.description,
                "target_record_json": canonical_json(target.to_record()),
            }
        )
    return rows


def write_target_library_json(path: Path, targets: Sequence[TargetMorphology]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": TARGET_SCHEMA_VERSION,
        "targetCount": len(targets),
        "targets": [target.to_record() for target in targets],
    }
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_target_library_json(path: Path) -> tuple[TargetMorphology, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(TargetMorphology.from_record(record) for record in payload["targets"])


def scrambled_state(target: TargetMorphology) -> dict[Position, CellIdentity]:
    positions = sorted(target.identities_by_position)
    identities = [target.identities_by_position[position] for position in positions]
    rotated = identities[1:] + identities[:1]
    return dict(zip(positions, rotated))


def render_target_panel(targets: Sequence[TargetMorphology], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = 3
    rows = math.ceil(len(targets) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.3, rows * 3.0), squeeze=False)
    color_values = list(ORGAN_COLORS.values())
    for axis, target in zip(axes.ravel(), targets):
        axis.set_title(target.motif.replace("_", " "))
        axis.set_aspect("equal")
        if target.substrate.substrate_type == "row_1d":
            xs = [position[0] for position in target.substrate.nodes]
            ys = [0 for _ in xs]
        else:
            xs = [position[0] for position in target.substrate.nodes]
            ys = [position[1] if len(position) > 1 else 0 for position in target.substrate.nodes]
        if target.motif in {"gradient", "sorted_row"}:
            values = [float(target.identities_by_position[position].components["ap_coordinate"]) for position in target.substrate.nodes]
            vmin = min(values)
            vmax = max(values)
            denom = max(1e-12, vmax - vmin)
            cmap = plt.get_cmap("viridis")
            colors = [mcolors.to_hex(cmap((value - vmin) / denom)) for value in values]
        else:
            colors = [
                ORGAN_COLORS.get(target.identities_by_position[position].components["organ_type"], "#cccccc")
                for position in target.substrate.nodes
            ]
        axis.scatter(xs, ys, c=colors, s=220, edgecolors="#222222", linewidths=0.6)
        for left, right in target.substrate.edges():
            lx, ly = left[0], left[1] if len(left) > 1 else 0
            rx, ry = right[0], right[1] if len(right) > 1 else 0
            axis.plot([lx, rx], [ly, ry], color="#c7c7c7", linewidth=0.6, zorder=0)
        axis.set_xticks([])
        axis.set_yticks([])
        if target.substrate.substrate_type == "row_1d":
            axis.set_ylim(-0.8, 0.8)
        axis.invert_yaxis()
    for axis in axes.ravel()[len(targets) :]:
        axis.axis("off")
    labels = [plt.Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color, markersize=8) for label, color in ORGAN_COLORS.items()]
    fig.legend(handles=labels, loc="lower center", ncol=min(len(labels), 5), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)
