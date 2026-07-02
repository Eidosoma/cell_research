"""Identity-vector layer for E05 higher-dimensional morphogenesis tasks.

S02 keeps scalar ``Value`` as a recoverable special case while adding typed
identity components such as anterior-posterior coordinate, organ type, polarity,
adhesion type, and local target-neighborhood preferences.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.e05.substrates import SubstrateCell, SubstrateState, array_substrate


IDENTITY_METADATA_KEY = "e05_identity"
COMPONENT_KINDS = frozenset({"continuous", "categorical", "polarity"})
PREFERENCE_RELATIONS = frozenset({"equals", "within", "less_than", "greater_than"})
SORT_DIRECTIONS = frozenset({"increasing", "decreasing"})


@dataclass(frozen=True)
class IdentityComponentSpec:
    """Schema entry for one identity-vector component."""

    name: str
    kind: str
    weight: float = 1.0
    ordered: bool = False
    categories: tuple[str, ...] = ()
    lower_bound: float | None = None
    upper_bound: float | None = None
    vector_length: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("component name must not be empty")
        kind = str(self.kind)
        if kind not in COMPONENT_KINDS:
            raise ValueError(f"unsupported component kind: {kind}")
        if float(self.weight) < 0.0:
            raise ValueError("component weight must be non-negative")
        if self.lower_bound is not None and self.upper_bound is not None and self.lower_bound >= self.upper_bound:
            raise ValueError("lower_bound must be less than upper_bound")
        if kind == "categorical":
            object.__setattr__(self, "categories", tuple(str(item) for item in self.categories))
        elif self.categories:
            raise ValueError("categories are only supported for categorical components")
        if kind == "polarity" and self.vector_length is not None and self.vector_length <= 0:
            raise ValueError("vector_length must be positive")
        if kind != "polarity" and self.vector_length is not None:
            raise ValueError("vector_length is only supported for polarity components")

    def validate_value(self, value: Any) -> Any:
        if self.kind == "continuous":
            numeric = float(value)
            if self.lower_bound is not None and numeric < self.lower_bound:
                raise ValueError(f"{self.name} below lower bound")
            if self.upper_bound is not None and numeric > self.upper_bound:
                raise ValueError(f"{self.name} above upper bound")
            return numeric
        if self.kind == "categorical":
            label = str(value)
            if self.categories and label not in self.categories:
                raise ValueError(f"{self.name} category {label!r} not in schema")
            return label
        vector = tuple(float(item) for item in value)
        if self.vector_length is not None and len(vector) != self.vector_length:
            raise ValueError(f"{self.name} polarity length must be {self.vector_length}")
        if not vector:
            raise ValueError(f"{self.name} polarity vector must not be empty")
        return vector

    def compact_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "weight": float(self.weight),
            "ordered": bool(self.ordered),
            "categories": list(self.categories),
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "vector_length": self.vector_length,
        }


@dataclass(frozen=True)
class TargetNeighborPreference:
    """One local target-neighborhood preference carried by a cell identity."""

    direction: str | None
    component: str
    expected_value: Any
    relation: str = "equals"
    tolerance: float = 0.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.component:
            raise ValueError("preference component must not be empty")
        if self.relation not in PREFERENCE_RELATIONS:
            raise ValueError(f"unsupported preference relation: {self.relation}")
        if float(self.tolerance) < 0.0:
            raise ValueError("preference tolerance must be non-negative")
        if float(self.weight) < 0.0:
            raise ValueError("preference weight must be non-negative")
        if self.direction is not None:
            object.__setattr__(self, "direction", str(self.direction))

    def compact_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "component": self.component,
            "expected_value": _jsonable(self.expected_value),
            "relation": self.relation,
            "tolerance": float(self.tolerance),
            "weight": float(self.weight),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TargetNeighborPreference":
        return cls(
            direction=None if payload.get("direction") is None else str(payload["direction"]),
            component=str(payload["component"]),
            expected_value=payload["expected_value"],
            relation=str(payload.get("relation", "equals")),
            tolerance=float(payload.get("tolerance", 0.0)),
            weight=float(payload.get("weight", 1.0)),
        )


@dataclass(frozen=True)
class CellIdentity:
    """Typed identity vector plus optional local neighborhood preferences."""

    identity_id: str
    components: Mapping[str, Any]
    target_preferences: tuple[TargetNeighborPreference, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.identity_id:
            raise ValueError("identity_id must not be empty")
        object.__setattr__(self, "components", dict(self.components))
        object.__setattr__(self, "target_preferences", tuple(self.target_preferences))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def compact_dict(self) -> dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "components": {name: _jsonable(value) for name, value in sorted(self.components.items())},
            "target_preferences": [preference.compact_dict() for preference in self.target_preferences],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CellIdentity":
        return cls(
            identity_id=str(payload["identity_id"]),
            components=dict(payload.get("components", {})),
            target_preferences=tuple(TargetNeighborPreference.from_dict(item) for item in payload.get("target_preferences", ())),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class IdentitySchema:
    """Schema, comparison, distance, and compatibility contract."""

    schema_id: str
    components: tuple[IdentityComponentSpec, ...]
    order_component: str | None = None
    compatibility_components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.schema_id:
            raise ValueError("schema_id must not be empty")
        components = tuple(self.components)
        if not components:
            raise ValueError("schema must include at least one component")
        names = [component.name for component in components]
        if len(names) != len(set(names)):
            raise ValueError("component names must be unique")
        object.__setattr__(self, "components", components)
        if self.order_component is not None:
            component = self.component(self.order_component)
            if not component.ordered:
                raise ValueError("order_component must reference an ordered component")
        if self.compatibility_components:
            for name in self.compatibility_components:
                self.component(name)

    def component(self, name: str) -> IdentityComponentSpec:
        for component in self.components:
            if component.name == name:
                return component
        raise KeyError(f"unknown identity component: {name}")

    def component_names(self) -> tuple[str, ...]:
        return tuple(component.name for component in self.components)

    def validate_identity(self, identity: CellIdentity) -> CellIdentity:
        missing = set(self.component_names()) - set(identity.components)
        if missing:
            raise ValueError(f"identity {identity.identity_id!r} missing components: {sorted(missing)}")
        validated = {
            component.name: component.validate_value(identity.components[component.name])
            for component in self.components
        }
        return CellIdentity(
            identity_id=identity.identity_id,
            components=validated,
            target_preferences=identity.target_preferences,
            metadata=identity.metadata,
        )

    def component_distance(self, component_name: str, left: CellIdentity, right: CellIdentity) -> float:
        component = self.component(component_name)
        left_value = component.validate_value(left.components[component_name])
        right_value = component.validate_value(right.components[component_name])
        if component.kind == "continuous":
            return _continuous_distance(left_value, right_value, component.lower_bound, component.upper_bound)
        if component.kind == "categorical":
            return 0.0 if left_value == right_value else 1.0
        return _polarity_distance(left_value, right_value)

    def distance(self, left: CellIdentity, right: CellIdentity, components: Sequence[str] | None = None) -> float:
        left = self.validate_identity(left)
        right = self.validate_identity(right)
        names = tuple(components or self.component_names())
        weighted = []
        for name in names:
            component = self.component(name)
            if component.weight == 0.0:
                continue
            weighted.append((component.weight, self.component_distance(name, left, right)))
        if not weighted:
            return 0.0
        total_weight = sum(weight for weight, _distance in weighted)
        return sum(weight * distance for weight, distance in weighted) / total_weight

    def compatibility_score(self, left: CellIdentity, right: CellIdentity) -> float:
        components = self.compatibility_components or self.component_names()
        return max(0.0, min(1.0, 1.0 - self.distance(left, right, components)))

    def order_key(self, identity: CellIdentity) -> float:
        if self.order_component is None:
            raise ValueError("schema has no order_component")
        validated = self.validate_identity(identity)
        component = self.component(self.order_component)
        if component.kind != "continuous":
            raise ValueError("only continuous ordered components are currently supported")
        return float(validated.components[self.order_component])

    def compare_order(self, left: CellIdentity, right: CellIdentity) -> int:
        left_key = self.order_key(left)
        right_key = self.order_key(right)
        if left_key < right_key:
            return -1
        if left_key > right_key:
            return 1
        return 0

    def compact_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "components": [component.compact_dict() for component in self.components],
            "order_component": self.order_component,
            "compatibility_components": list(self.compatibility_components),
        }

    @property
    def sha256(self) -> str:
        return stable_identity_sha256(self.compact_dict())


@dataclass(frozen=True)
class NeighborIdentityRelation:
    """Actor-local identity relation to one neighboring site."""

    site_id: int
    direction: str
    occupied: bool
    neighbor_cell_id: str | None
    neighbor_identity: CellIdentity | None
    identity_distance: float | None
    compatibility_score: float | None
    order_relation: int | None
    target_preference_score: float | None

    def compact_dict(self) -> dict[str, Any]:
        return {
            "site_id": int(self.site_id),
            "direction": self.direction,
            "occupied": bool(self.occupied),
            "neighbor_cell_id": self.neighbor_cell_id,
            "neighbor_identity": None if self.neighbor_identity is None else self.neighbor_identity.compact_dict(),
            "identity_distance": self.identity_distance,
            "compatibility_score": self.compatibility_score,
            "order_relation": self.order_relation,
            "target_preference_score": self.target_preference_score,
        }


@dataclass(frozen=True)
class LocalIdentityObservation:
    """Substrate-local observation augmented with identity relations."""

    substrate_kind: str
    actor_site_id: int
    actor_cell_id: str
    actor_identity: CellIdentity
    neighbor_relations: tuple[NeighborIdentityRelation, ...]

    def compact_dict(self) -> dict[str, Any]:
        return {
            "substrate_kind": self.substrate_kind,
            "actor_site_id": int(self.actor_site_id),
            "actor_cell_id": self.actor_cell_id,
            "actor_identity": self.actor_identity.compact_dict(),
            "neighbor_relations": [relation.compact_dict() for relation in self.neighbor_relations],
        }


def stable_identity_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scalar_value_schema(min_value: float | None = None, max_value: float | None = None) -> IdentitySchema:
    """Return a one-component schema that recovers scalar Value ordering."""

    return IdentitySchema(
        schema_id="e05.scalar_value_identity.v1",
        components=(
            IdentityComponentSpec(
                name="value",
                kind="continuous",
                ordered=True,
                lower_bound=min_value,
                upper_bound=max_value,
            ),
        ),
        order_component="value",
        compatibility_components=("value",),
    )


def default_morphogenesis_identity_schema() -> IdentitySchema:
    """Return the S02 toy multi-component identity schema."""

    return IdentitySchema(
        schema_id="e05.default_morphogenesis_identity.v1",
        components=(
            IdentityComponentSpec("ap_coordinate", "continuous", weight=1.0, ordered=True, lower_bound=0.0, upper_bound=1.0),
            IdentityComponentSpec(
                "organ_type",
                "categorical",
                weight=1.0,
                categories=("neural", "epidermis", "mesenchyme", "boundary"),
            ),
            IdentityComponentSpec("polarity", "polarity", weight=0.5, vector_length=2),
            IdentityComponentSpec("adhesion_type", "categorical", weight=0.5, categories=("low", "medium", "high")),
        ),
        order_component="ap_coordinate",
        compatibility_components=("organ_type", "adhesion_type"),
    )


def scalar_identity(value: int | float, identity_id: str | None = None) -> CellIdentity:
    return CellIdentity(identity_id=identity_id or f"value:{value}", components={"value": float(value)})


def attach_identity(cell: SubstrateCell, identity: CellIdentity, *, value_from_order_key: float | None = None) -> SubstrateCell:
    """Return a copy of ``cell`` with identity metadata attached."""

    metadata = dict(cell.metadata)
    metadata[IDENTITY_METADATA_KEY] = identity.compact_dict()
    value = cell.value if value_from_order_key is None else value_from_order_key
    return SubstrateCell(
        cell_id=cell.cell_id,
        value=value,
        label=cell.label,
        status=cell.status,
        metadata=metadata,
    )


def identity_from_substrate_cell(cell: SubstrateCell, *, fallback_scalar: bool = True) -> CellIdentity:
    payload = cell.metadata.get(IDENTITY_METADATA_KEY)
    if payload is not None:
        return CellIdentity.from_dict(payload)
    if fallback_scalar and cell.value is not None:
        return scalar_identity(cell.value, identity_id=str(cell.cell_id))
    raise ValueError(f"cell {cell.cell_id!r} has no identity metadata")


def make_scalar_identity_cells(values: Sequence[int | float], *, label: str = "scalar") -> tuple[SubstrateCell, ...]:
    cells = []
    for idx, value in enumerate(values):
        identity = scalar_identity(value, identity_id=f"scalar_{idx}")
        cells.append(
            attach_identity(
                SubstrateCell(cell_id=f"cell_{idx}", value=value, label=label),
                identity,
            )
        )
    return tuple(cells)


def should_swap_for_identity_order(
    actor_identity: CellIdentity,
    target_identity: CellIdentity,
    schema: IdentitySchema,
    *,
    direction: str = "increasing",
) -> bool:
    if direction not in SORT_DIRECTIONS:
        raise ValueError(f"unsupported sort direction: {direction}")
    relation = schema.compare_order(actor_identity, target_identity)
    return relation > 0 if direction == "increasing" else relation < 0


def identity_ordered_values(
    values: Sequence[int | float],
    *,
    direction: str = "increasing",
) -> dict[str, Any]:
    """Sort scalar identities by local adjacent swaps on an S01 1D substrate."""

    values = tuple(values)
    if not values:
        raise ValueError("values must not be empty")
    lower = min(values)
    upper = max(values)
    schema = scalar_value_schema(lower, upper) if lower < upper else scalar_value_schema()
    substrate = array_substrate(len(values))
    substrate.fill_sites(make_scalar_identity_cells(values))
    swap_count = 0
    pass_count = 0
    while True:
        changed = False
        for site_id in range(len(values) - 1):
            actor = substrate.cell_at(site_id)
            target = substrate.cell_at(site_id + 1)
            if actor is None or target is None:
                continue
            if should_swap_for_identity_order(
                identity_from_substrate_cell(actor),
                identity_from_substrate_cell(target),
                schema,
                direction=direction,
            ):
                result = substrate.apply_action("swap", site_id, site_id + 1)
                if not result.allowed:
                    raise RuntimeError(f"identity-order swap was unexpectedly rejected: {result.reason}")
                swap_count += 1
                changed = True
        pass_count += 1
        if not changed:
            break
        if pass_count > len(values) * len(values):
            raise RuntimeError("identity-order bubble sort exceeded safety guard")
    return {
        "final_values": substrate.values_in_site_order(),
        "swap_count": swap_count,
        "pass_count": pass_count,
        "schema_sha256": schema.sha256,
    }


def local_identity_observation(substrate: SubstrateState, site_id: int, schema: IdentitySchema) -> LocalIdentityObservation:
    observation = substrate.local_observation(site_id)
    actor_identity = schema.validate_identity(identity_from_substrate_cell(observation.actor_cell))
    relations: list[NeighborIdentityRelation] = []
    for neighbor in observation.neighbors:
        neighbor_cell = substrate.cell_at(neighbor.site_id)
        if neighbor_cell is None:
            relations.append(
                NeighborIdentityRelation(
                    site_id=neighbor.site_id,
                    direction=neighbor.direction,
                    occupied=False,
                    neighbor_cell_id=None,
                    neighbor_identity=None,
                    identity_distance=None,
                    compatibility_score=None,
                    order_relation=None,
                    target_preference_score=None,
                )
            )
            continue
        neighbor_identity = schema.validate_identity(identity_from_substrate_cell(neighbor_cell))
        relations.append(
            NeighborIdentityRelation(
                site_id=neighbor.site_id,
                direction=neighbor.direction,
                occupied=True,
                neighbor_cell_id=neighbor_cell.cell_id,
                neighbor_identity=neighbor_identity,
                identity_distance=schema.distance(actor_identity, neighbor_identity),
                compatibility_score=schema.compatibility_score(actor_identity, neighbor_identity),
                order_relation=None if schema.order_component is None else schema.compare_order(actor_identity, neighbor_identity),
                target_preference_score=target_preference_score(actor_identity, neighbor_identity, neighbor.direction, schema),
            )
        )
    return LocalIdentityObservation(
        substrate_kind=observation.substrate_kind,
        actor_site_id=observation.actor_site_id,
        actor_cell_id=observation.actor_cell.cell_id,
        actor_identity=actor_identity,
        neighbor_relations=tuple(relations),
    )


def target_preference_score(
    actor_identity: CellIdentity,
    neighbor_identity: CellIdentity,
    direction: str,
    schema: IdentitySchema,
) -> float | None:
    matching = [
        preference
        for preference in actor_identity.target_preferences
        if preference.direction is None or preference.direction == direction
    ]
    if not matching:
        return None
    weighted_scores = []
    for preference in matching:
        component = schema.component(preference.component)
        observed_value = neighbor_identity.components[preference.component]
        score = _preference_component_score(component, observed_value, preference)
        weighted_scores.append((float(preference.weight), score))
    total_weight = sum(weight for weight, _score in weighted_scores)
    if total_weight == 0.0:
        return 0.0
    return sum(weight * score for weight, score in weighted_scores) / total_weight


def _preference_component_score(
    component: IdentityComponentSpec,
    observed_value: Any,
    preference: TargetNeighborPreference,
) -> float:
    observed = component.validate_value(observed_value)
    expected = component.validate_value(preference.expected_value)
    if component.kind == "continuous":
        delta = abs(float(observed) - float(expected))
        if preference.relation == "equals":
            if preference.tolerance == 0.0:
                return 1.0 if delta == 0.0 else 0.0
            return max(0.0, 1.0 - delta / preference.tolerance)
        if preference.relation == "within":
            return 1.0 if delta <= preference.tolerance else 0.0
        if preference.relation == "less_than":
            return 1.0 if float(observed) < float(expected) else 0.0
        if preference.relation == "greater_than":
            return 1.0 if float(observed) > float(expected) else 0.0
    if component.kind == "categorical":
        if preference.relation != "equals":
            raise ValueError("categorical preferences only support equals")
        return 1.0 if observed == expected else 0.0
    distance = _polarity_distance(observed, expected)
    if preference.relation == "equals":
        return max(0.0, 1.0 - distance)
    if preference.relation == "within":
        return 1.0 if distance <= preference.tolerance else 0.0
    raise ValueError("polarity preferences only support equals or within")


def _continuous_distance(left: float, right: float, lower_bound: float | None, upper_bound: float | None) -> float:
    diff = abs(float(left) - float(right))
    if lower_bound is not None and upper_bound is not None:
        scale = float(upper_bound) - float(lower_bound)
        return min(1.0, diff / scale)
    return diff / (1.0 + diff)


def _polarity_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("polarity vectors must have equal length")
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    cosine = sum(float(a) * float(b) for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    cosine = max(-1.0, min(1.0, cosine))
    return 0.5 * (1.0 - cosine)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    return value
