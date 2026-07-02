"""Target morphology specifications for E05 S03.

Targets are explicit site-to-identity maps over S01 substrates.  They are toy
computational morphology goals, not biological anatomy; S05 will add richer
metric families.  S03 only ensures targets can be represented, rendered, and
validated against constructed states with zero target error.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.e05.cell_identity import (
    CellIdentity,
    IdentitySchema,
    attach_identity,
    default_morphogenesis_identity_schema,
    scalar_identity,
    scalar_value_schema,
)
from src.e05.substrates import SubstrateCell, SubstrateState, array_substrate, square_grid_substrate


ORGAN_COLORS = {
    "neural": "#4C78A8",
    "epidermis": "#F58518",
    "mesenchyme": "#54A24B",
    "boundary": "#B279A2",
}


@dataclass(frozen=True)
class TargetMetricContract:
    """Metric-compatibility metadata for a target morphology."""

    primary_metric: str
    zero_error_definition: str
    compatible_components: tuple[str, ...]
    notes: str = ""

    def compact_dict(self) -> dict[str, Any]:
        return {
            "primary_metric": self.primary_metric,
            "zero_error_definition": self.zero_error_definition,
            "compatible_components": list(self.compatible_components),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class TargetMorphology:
    """A target pattern over a fixed S01 substrate."""

    target_id: str
    title: str
    substrate: SubstrateState
    schema: IdentitySchema
    identities_by_site: Mapping[int, CellIdentity]
    target_kind: str
    metric_contract: TargetMetricContract
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("target_id must not be empty")
        site_ids = set(self.substrate.site_ids)
        target_ids = {int(site_id) for site_id in self.identities_by_site}
        if target_ids != site_ids:
            missing = sorted(site_ids - target_ids)
            extra = sorted(target_ids - site_ids)
            raise ValueError(f"target identities must cover every site exactly; missing={missing}, extra={extra}")
        validated = {
            int(site_id): self.schema.validate_identity(identity)
            for site_id, identity in self.identities_by_site.items()
        }
        object.__setattr__(self, "identities_by_site", dict(sorted(validated.items())))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def target_hash(self) -> str:
        return stable_target_sha256(self.compact_dict(include_hash=False))

    def compact_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "target_id": self.target_id,
            "title": self.title,
            "target_kind": self.target_kind,
            "substrate": self.substrate.graph_spec(),
            "schema": self.schema.compact_dict(),
            "identities_by_site": {
                str(site_id): identity.compact_dict()
                for site_id, identity in sorted(self.identities_by_site.items())
            },
            "metric_contract": self.metric_contract.compact_dict(),
            "metadata": dict(self.metadata),
        }
        if include_hash:
            payload["target_hash"] = self.target_hash
        return payload

    def constructed_substrate(self) -> SubstrateState:
        """Return a state exactly matching the target identities."""

        state = self.substrate.copy_empty()
        cells = []
        for site_id in state.site_ids:
            identity = self.identities_by_site[site_id]
            value = None
            if self.schema.order_component is not None:
                try:
                    value = self.schema.order_key(identity)
                except ValueError:
                    value = None
            cells.append(
                attach_identity(
                    SubstrateCell(
                        cell_id=f"{self.target_id}_site_{site_id}",
                        value=value,
                        label=str(identity.components.get("organ_type", self.target_kind)),
                    ),
                    identity,
                )
            )
        state.fill_sites(tuple(cells))
        return state

    def target_error(self, state: SubstrateState) -> float:
        """Average identity distance to target, with empty/wrong-site penalty."""

        if tuple(state.site_ids) != tuple(self.substrate.site_ids):
            raise ValueError("state site IDs do not match target substrate")
        penalties: list[float] = []
        for site_id in self.substrate.site_ids:
            cell = state.cell_at(site_id)
            if cell is None or "e05_identity" not in cell.metadata:
                penalties.append(1.0)
                continue
            observed = CellIdentity.from_dict(cell.metadata["e05_identity"])
            penalties.append(self.schema.distance(observed, self.identities_by_site[site_id]))
        return float(sum(penalties) / len(penalties)) if penalties else 0.0

    def organ_grid(self) -> list[list[str]]:
        width = int(self.substrate.dimensions.get("width", len(self.substrate.site_ids)))
        height = int(self.substrate.dimensions.get("height", 1))
        rows: list[list[str]] = []
        for y in range(height):
            row = []
            for x in range(width):
                site_id = y * width + x
                identity = self.identities_by_site[site_id]
                row.append(str(identity.components.get("organ_type", identity.components.get("value", ""))))
            rows.append(row)
        return rows


def stable_target_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sorted_row_target(values: Sequence[int | float], *, target_id: str = "sorted_row") -> TargetMorphology:
    values = tuple(values)
    if not values:
        raise ValueError("values must not be empty")
    sorted_values = tuple(sorted(values))
    lower = min(sorted_values)
    upper = max(sorted_values)
    schema = scalar_value_schema(lower, upper) if lower < upper else scalar_value_schema()
    substrate = array_substrate(len(sorted_values))
    identities = {
        site_id: scalar_identity(value, identity_id=f"{target_id}_value_{value}_site_{site_id}")
        for site_id, value in enumerate(sorted_values)
    }
    return TargetMorphology(
        target_id=target_id,
        title="Sorted row",
        substrate=substrate,
        schema=schema,
        identities_by_site=identities,
        target_kind="sorted_row",
        metric_contract=TargetMetricContract(
            primary_metric="mean_scalar_identity_distance",
            zero_error_definition="Each site contains the scalar identity assigned by the sorted target row.",
            compatible_components=("value",),
            notes="Continuity target for the original 1D sorting abstraction.",
        ),
        metadata={"input_values": list(values), "sorted_values": list(sorted_values)},
    )


def gradient_target(width: int, height: int, *, target_id: str = "ap_gradient") -> TargetMorphology:
    substrate = square_grid_substrate(width, height)
    schema = default_morphogenesis_identity_schema()
    identities: dict[int, CellIdentity] = {}
    denom = max(1, int(width) - 1)
    for site_id in substrate.site_ids:
        x, y = substrate.coordinate(site_id)
        ap = float(x / denom)
        identities[site_id] = morph_identity(
            f"{target_id}_{site_id}",
            ap,
            "mesenchyme",
            (1.0, 0.0),
            "medium",
            metadata={"x": x, "y": y},
        )
    return _target(
        target_id,
        "Anterior-posterior gradient",
        "gradient",
        substrate,
        schema,
        identities,
        "mean_identity_distance",
        ("ap_coordinate",),
        "A constructed state has each site's ap_coordinate equal to the target gradient value.",
    )


def stripes_target(width: int, height: int, *, stripe_axis: str = "x", target_id: str = "organ_stripes") -> TargetMorphology:
    if stripe_axis not in {"x", "y"}:
        raise ValueError("stripe_axis must be x or y")
    substrate = square_grid_substrate(width, height)
    schema = default_morphogenesis_identity_schema()
    organs = ("neural", "epidermis")
    identities = {}
    for site_id in substrate.site_ids:
        x, y = substrate.coordinate(site_id)
        stripe_index = x if stripe_axis == "x" else y
        organ = organs[stripe_index % len(organs)]
        identities[site_id] = morph_identity(f"{target_id}_{site_id}", _ap_from_x(x, width), organ, (0.0, 1.0), "medium")
    return _target(
        target_id,
        "Alternating organ stripes",
        "stripes",
        substrate,
        schema,
        identities,
        "categorical_organ_match",
        ("organ_type",),
        "A constructed state has the expected alternating organ_type labels at every site.",
        metadata={"stripe_axis": stripe_axis},
    )


def boundary_target(width: int, height: int, *, target_id: str = "boundary_pattern") -> TargetMorphology:
    substrate = square_grid_substrate(width, height)
    schema = default_morphogenesis_identity_schema()
    identities = {}
    for site_id in substrate.site_ids:
        x, y = substrate.coordinate(site_id)
        is_boundary = x == 0 or y == 0 or x == width - 1 or y == height - 1
        organ = "boundary" if is_boundary else "mesenchyme"
        adhesion = "high" if is_boundary else "medium"
        identities[site_id] = morph_identity(f"{target_id}_{site_id}", _ap_from_x(x, width), organ, (1.0, 0.0), adhesion)
    return _target(
        target_id,
        "Boundary and interior",
        "boundary",
        substrate,
        schema,
        identities,
        "boundary_label_match",
        ("organ_type", "adhesion_type"),
        "A constructed state has boundary labels on the perimeter and mesenchyme labels inside.",
    )


def ring_target(width: int, height: int, *, target_id: str = "ring_pattern") -> TargetMorphology:
    substrate = square_grid_substrate(width, height)
    schema = default_morphogenesis_identity_schema()
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    radius = max(1.0, min(width, height) / 3.0)
    identities = {}
    for site_id in substrate.site_ids:
        x, y = substrate.coordinate(site_id)
        distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        if abs(distance - radius) <= 0.75:
            organ = "boundary"
            adhesion = "high"
        elif distance < radius:
            organ = "neural"
            adhesion = "medium"
        else:
            organ = "epidermis"
            adhesion = "low"
        polarity = _unit_vector(x - center_x, y - center_y)
        identities[site_id] = morph_identity(f"{target_id}_{site_id}", _ap_from_x(x, width), organ, polarity, adhesion)
    return _target(
        target_id,
        "Ring with inside/outside",
        "ring",
        substrate,
        schema,
        identities,
        "radial_region_match",
        ("organ_type", "polarity"),
        "A constructed state has center, ring, and outside region identities at the intended sites.",
        metadata={"radius": radius, "center": [center_x, center_y]},
    )


def symmetry_target(width: int, height: int, *, target_id: str = "bilateral_symmetry") -> TargetMorphology:
    substrate = square_grid_substrate(width, height)
    schema = default_morphogenesis_identity_schema()
    identities = {}
    mid = (width - 1) / 2.0
    for site_id in substrate.site_ids:
        x, y = substrate.coordinate(site_id)
        distance_from_mid = abs(x - mid)
        if distance_from_mid <= 0.5:
            organ = "neural"
        elif y in {0, height - 1}:
            organ = "boundary"
        else:
            organ = "mesenchyme"
        polarity = (-1.0, 0.0) if x < mid else (1.0, 0.0)
        if x == mid:
            polarity = (0.0, 1.0)
        identities[site_id] = morph_identity(f"{target_id}_{site_id}", _ap_from_x(x, width), organ, polarity, "medium")
    return _target(
        target_id,
        "Bilateral symmetry",
        "symmetry",
        substrate,
        schema,
        identities,
        "mirror_label_match",
        ("organ_type", "polarity"),
        "A constructed state has left-right mirrored organ labels with opposing polarity.",
    )


def organ_like_target(width: int, height: int, *, target_id: str = "toy_organ_like") -> TargetMorphology:
    substrate = square_grid_substrate(width, height)
    schema = default_morphogenesis_identity_schema()
    identities = {}
    body_x_min = max(1, width // 3)
    body_x_max = min(width - 2, (2 * width) // 3)
    limb_y = height // 2
    for site_id in substrate.site_ids:
        x, y = substrate.coordinate(site_id)
        if x < width // 4 and y == limb_y:
            organ = "neural"
            adhesion = "high"
            polarity = (-1.0, 0.0)
        elif body_x_min <= x <= body_x_max and 1 <= y <= height - 2:
            organ = "mesenchyme"
            adhesion = "medium"
            polarity = (1.0, 0.0)
        elif x == body_x_max + 1 and 1 <= y <= height - 2:
            organ = "boundary"
            adhesion = "high"
            polarity = (1.0, 0.0)
        else:
            organ = "epidermis"
            adhesion = "low"
            polarity = (0.0, 1.0)
        identities[site_id] = morph_identity(f"{target_id}_{site_id}", _ap_from_x(x, width), organ, polarity, adhesion)
    return _target(
        target_id,
        "Toy organ-like body and appendage",
        "organ_like",
        substrate,
        schema,
        identities,
        "region_label_match",
        ("organ_type", "adhesion_type", "polarity"),
        "A constructed state has a simple body, boundary, and appendage-like region.",
        metadata={"toy_model": True},
    )


def default_target_gallery() -> tuple[TargetMorphology, ...]:
    return (
        sorted_row_target((5, 1, 4, 2, 3)),
        gradient_target(6, 4),
        stripes_target(6, 4),
        boundary_target(6, 4),
        ring_target(7, 7),
        symmetry_target(7, 5),
        organ_like_target(8, 5),
    )


def constructed_target_error(target: TargetMorphology) -> float:
    return target.target_error(target.constructed_substrate())


def target_summary_rows(targets: Sequence[TargetMorphology]) -> list[dict[str, Any]]:
    rows = []
    for target in targets:
        rows.append(
            {
                "target_id": target.target_id,
                "title": target.title,
                "target_kind": target.target_kind,
                "substrate_kind": target.substrate.kind,
                "site_count": len(target.substrate.site_ids),
                "schema_id": target.schema.schema_id,
                "target_hash": target.target_hash,
                "constructed_target_error": constructed_target_error(target),
                "primary_metric": target.metric_contract.primary_metric,
                "compatible_components_json": json.dumps(list(target.metric_contract.compatible_components), separators=(",", ":")),
                "render_mode": target_render_mode(target),
            }
        )
    return rows


def target_render_mode(target: TargetMorphology) -> str:
    """Return the target component used for gallery rendering."""

    if target.schema.schema_id == "e05.scalar_value_identity.v1":
        return "value"
    if target.target_kind == "gradient" and "ap_coordinate" in target.metric_contract.compatible_components:
        return "ap_coordinate"
    return "organ_type"


def render_target_gallery(targets: Sequence[TargetMorphology], output_path: Path) -> None:
    """Render target identities to a compact PNG gallery."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = 3
    rows = math.ceil(len(targets) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.4), constrained_layout=True)
    flat_axes = np.array(axes).reshape(-1)
    color_keys = ["neural", "epidermis", "mesenchyme", "boundary"]
    cmap = ListedColormap([ORGAN_COLORS[key] for key in color_keys])
    organ_to_index = {organ: idx for idx, organ in enumerate(color_keys)}
    for ax, target in zip(flat_axes, targets, strict=False):
        width = int(target.substrate.dimensions.get("width", len(target.substrate.site_ids)))
        height = int(target.substrate.dimensions.get("height", 1))
        render_mode = target_render_mode(target)
        if render_mode in {"value", "ap_coordinate"}:
            values = [target.identities_by_site[site_id].components[render_mode] for site_id in target.substrate.site_ids]
            matrix = np.array(values, dtype=float).reshape(height, width)
            kwargs = {"vmin": 0.0, "vmax": 1.0} if render_mode == "ap_coordinate" else {}
            image = ax.imshow(matrix, cmap="viridis", interpolation="nearest", aspect="auto", **kwargs)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        else:
            matrix = np.zeros((height, width), dtype=int)
            for site_id in target.substrate.site_ids:
                x, y = target.substrate.coordinate(site_id)
                organ = str(target.identities_by_site[site_id].components["organ_type"])
                matrix[y, x] = organ_to_index[organ]
            ax.imshow(matrix, cmap=cmap, vmin=0, vmax=len(color_keys) - 1, interpolation="nearest", aspect="equal")
        ax.set_title(target.title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(target.target_kind, fontsize=8)
    for ax in flat_axes[len(targets) :]:
        ax.axis("off")
    fig.suptitle("E05 S03 target morphology gallery", fontsize=12)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _target(
    target_id: str,
    title: str,
    target_kind: str,
    substrate: SubstrateState,
    schema: IdentitySchema,
    identities: Mapping[int, CellIdentity],
    primary_metric: str,
    compatible_components: tuple[str, ...],
    zero_error_definition: str,
    metadata: Mapping[str, Any] | None = None,
) -> TargetMorphology:
    return TargetMorphology(
        target_id=target_id,
        title=title,
        substrate=substrate,
        schema=schema,
        identities_by_site=identities,
        target_kind=target_kind,
        metric_contract=TargetMetricContract(
            primary_metric=primary_metric,
            zero_error_definition=zero_error_definition,
            compatible_components=compatible_components,
            notes="S03 metric contract; S05 will implement richer morphology metrics.",
        ),
        metadata=dict(metadata or {}),
    )


def morph_identity(
    identity_id: str,
    ap_coordinate: float,
    organ_type: str,
    polarity: tuple[float, float],
    adhesion_type: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> CellIdentity:
    return CellIdentity(
        identity_id=identity_id,
        components={
            "ap_coordinate": float(ap_coordinate),
            "organ_type": organ_type,
            "polarity": tuple(float(item) for item in polarity),
            "adhesion_type": adhesion_type,
        },
        metadata=dict(metadata or {}),
    )


def _ap_from_x(x: int, width: int) -> float:
    return float(x / max(1, width - 1))


def _unit_vector(dx: float, dy: float) -> tuple[float, float]:
    norm = math.sqrt(dx * dx + dy * dy)
    if norm == 0.0:
        return (0.0, 1.0)
    return (float(dx / norm), float(dy / norm))
