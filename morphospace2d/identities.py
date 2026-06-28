"""Identity-vector schema utilities for E05 S02.

S02 keeps identity records independent from target morphology definitions. The
schema names which components are fixed, mutable, visible to neighbors, or
hidden, and provides a reversible scalar mapping for the one-dimensional
baseline.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .substrates import CellState


IDENTITY_SCHEMA_VERSION = "e05_s02_identity_schema.v1"
NEIGHBOR_PREFERENCE_VERSION = "e05_s02_neighbor_preference.v1"


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


def stable_identity_id(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class IdentityFieldSpec:
    """Schema entry for one identity component."""

    name: str
    kind: str
    fixed: bool
    mutable: bool
    observable_to_neighbors: bool
    hidden: bool = False
    required: bool = True
    allowed_values: tuple[str, ...] = ()
    description: str = ""

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.name:
            errors.append("field name is empty")
        if self.fixed and self.mutable:
            errors.append(f"{self.name} cannot be both fixed and mutable")
        if self.hidden and self.observable_to_neighbors:
            errors.append(f"{self.name} cannot be hidden and neighbor-observable")
        if self.kind not in {"numeric", "categorical", "vector", "rules", "mapping"}:
            errors.append(f"{self.name} has unsupported kind {self.kind!r}")
        return errors

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "fixed": self.fixed,
            "mutable": self.mutable,
            "observableToNeighbors": self.observable_to_neighbors,
            "hidden": self.hidden,
            "required": self.required,
            "allowedValues": list(self.allowed_values),
            "description": self.description,
        }


@dataclass(frozen=True)
class IdentitySchema:
    """Collection of field specs and scalar mapping metadata."""

    fields: tuple[IdentityFieldSpec, ...]
    schema_version: str = IDENTITY_SCHEMA_VERSION
    scalar_component: str = "scalar_value"
    notes: str = "Computational identity schema; biological names are analogical labels."

    def field_map(self) -> dict[str, IdentityFieldSpec]:
        return {field.name: field for field in self.fields}

    def validate(self) -> list[str]:
        errors: list[str] = []
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            errors.append("duplicate identity field names")
        for field in self.fields:
            errors.extend(field.validate())
        if self.scalar_component not in names:
            errors.append(f"scalar component {self.scalar_component!r} is not in schema")
        scalar_field = self.field_map().get(self.scalar_component)
        if scalar_field is not None and scalar_field.kind != "numeric":
            errors.append("scalar component must be numeric")
        return errors

    def assert_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("; ".join(errors))

    def to_record(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "scalarComponent": self.scalar_component,
            "notes": self.notes,
            "fields": [field.to_record() for field in self.fields],
        }


@dataclass(frozen=True)
class NeighborPreferenceRule:
    """Local target-neighborhood preference for an identity."""

    component: str
    expected_value: Any
    min_count: int = 0
    max_count: int | None = None
    weight: float = 1.0

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "NeighborPreferenceRule":
        return cls(
            component=str(record["component"]),
            expected_value=record.get("expectedValue", record.get("expected_value")),
            min_count=int(record.get("minCount", record.get("min_count", 0))),
            max_count=None if record.get("maxCount", record.get("max_count")) is None else int(record.get("maxCount", record.get("max_count"))),
            weight=float(record.get("weight", 1.0)),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schemaVersion": NEIGHBOR_PREFERENCE_VERSION,
            "component": self.component,
            "expectedValue": self.expected_value,
            "minCount": self.min_count,
            "maxCount": self.max_count,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class CellIdentity:
    """Identity vector associated with one simulated cell."""

    cell_id: int
    components: Mapping[str, Any]
    schema_version: str = IDENTITY_SCHEMA_VERSION
    identity_id: str | None = None

    def __post_init__(self) -> None:
        normalized = {str(key): _json_ready(value) for key, value in self.components.items()}
        object.__setattr__(self, "components", normalized)
        if self.identity_id is None:
            digest_source = {"cellId": int(self.cell_id), "components": normalized}
            object.__setattr__(self, "identity_id", f"id_{stable_identity_id(digest_source)}")

    @property
    def scalar_value(self) -> Any:
        return self.components["scalar_value"]

    def preference_rules(self) -> tuple[NeighborPreferenceRule, ...]:
        return tuple(
            NeighborPreferenceRule.from_record(record)
            for record in self.components.get("target_neighbor_preferences", [])
        )

    def to_record(self, *, schema: IdentitySchema | None = None, include_hidden: bool = True) -> dict[str, Any]:
        components = dict(self.components)
        if schema is not None and not include_hidden:
            components = observable_components(self, schema)
        return {
            "schemaVersion": self.schema_version,
            "identityId": self.identity_id,
            "cellId": int(self.cell_id),
            "components": _json_ready(components),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "CellIdentity":
        return cls(
            cell_id=int(record["cellId"]),
            components=dict(record["components"]),
            schema_version=str(record.get("schemaVersion", IDENTITY_SCHEMA_VERSION)),
            identity_id=None if record.get("identityId") is None else str(record["identityId"]),
        )


def default_identity_schema() -> IdentitySchema:
    schema = IdentitySchema(
        fields=(
            IdentityFieldSpec(
                "scalar_value",
                "numeric",
                fixed=True,
                mutable=False,
                observable_to_neighbors=True,
                description="Original one-dimensional sorting Value; reversible baseline scalar.",
            ),
            IdentityFieldSpec(
                "ap_coordinate",
                "numeric",
                fixed=True,
                mutable=False,
                observable_to_neighbors=True,
                description="Normalized anterior-posterior-like coordinate used as a computational axis label.",
            ),
            IdentityFieldSpec(
                "organ_type",
                "categorical",
                fixed=True,
                mutable=False,
                observable_to_neighbors=True,
                allowed_values=("axis", "boundary", "core", "appendage", "organizer"),
                description="Computational region or role label; not an anatomical claim.",
            ),
            IdentityFieldSpec(
                "polarity",
                "vector",
                fixed=False,
                mutable=True,
                observable_to_neighbors=True,
                description="Local orientation vector available for later action and symmetry tasks.",
            ),
            IdentityFieldSpec(
                "adhesion_type",
                "categorical",
                fixed=False,
                mutable=True,
                observable_to_neighbors=True,
                allowed_values=("adhesion_a", "adhesion_b", "adhesion_boundary"),
                description="Computational adhesion class for local compatibility scoring.",
            ),
            IdentityFieldSpec(
                "target_neighbor_preferences",
                "rules",
                fixed=True,
                mutable=False,
                observable_to_neighbors=False,
                description="Actor-internal target-neighborhood rules used by local policies or metrics.",
            ),
            IdentityFieldSpec(
                "internal_state",
                "mapping",
                fixed=False,
                mutable=True,
                observable_to_neighbors=False,
                hidden=True,
                required=False,
                description="Hidden mutable cell-internal state; excluded from neighbor observations.",
            ),
        )
    )
    schema.assert_valid()
    return schema


def scalar_identity(
    cell_id: int,
    value: int | float,
    *,
    n: int | None = None,
    organ_type: str = "axis",
) -> CellIdentity:
    denom = max(1, int(n) - 1) if n is not None else 1
    ap_coordinate = 0.0 if n in {None, 1} else (float(value) - 1.0) / float(denom)
    return CellIdentity(
        cell_id=int(cell_id),
        components={
            "scalar_value": value,
            "ap_coordinate": ap_coordinate,
            "organ_type": organ_type,
            "polarity": [1.0, 0.0],
            "adhesion_type": "adhesion_a" if int(value) % 2 else "adhesion_b",
            "target_neighbor_preferences": [],
            "internal_state": {},
        },
    )


def identities_from_scalar_values(values: Sequence[int | float]) -> tuple[CellIdentity, ...]:
    return tuple(scalar_identity(index, value, n=len(values)) for index, value in enumerate(values))


def scalar_values_from_identities(identities: Sequence[CellIdentity]) -> list[Any]:
    return [identity.scalar_value for identity in identities]


def observable_components(identity: CellIdentity, schema: IdentitySchema) -> dict[str, Any]:
    field_map = schema.field_map()
    return {
        key: value
        for key, value in identity.components.items()
        if key in field_map and field_map[key].observable_to_neighbors and not field_map[key].hidden
    }


def actor_components(identity: CellIdentity, schema: IdentitySchema) -> dict[str, Any]:
    """Return components available to the actor itself, excluding hidden internals."""

    field_map = schema.field_map()
    return {
        key: value
        for key, value in identity.components.items()
        if key in field_map and not field_map[key].hidden
    }


def identity_to_cell_state(identity: CellIdentity, schema: IdentitySchema, *, include_hidden: bool = False) -> CellState:
    components = dict(identity.components) if include_hidden else actor_components(identity, schema)
    return CellState(cell_id=identity.cell_id, value=identity.scalar_value, identity=components)


def validate_identity(identity: CellIdentity, schema: IdentitySchema) -> list[str]:
    errors: list[str] = []
    field_map = schema.field_map()
    for field in schema.fields:
        if field.required and field.name not in identity.components:
            errors.append(f"cell {identity.cell_id} missing required field {field.name}")
    for name, value in identity.components.items():
        field = field_map.get(name)
        if field is None:
            errors.append(f"cell {identity.cell_id} has unknown field {name}")
            continue
        if field.kind == "numeric" and not isinstance(value, (int, float)):
            errors.append(f"cell {identity.cell_id} field {name} is not numeric")
        if field.kind == "categorical" and field.allowed_values and value not in field.allowed_values:
            errors.append(f"cell {identity.cell_id} field {name} has invalid value {value!r}")
        if field.kind == "vector":
            if not isinstance(value, list) or not all(isinstance(item, (int, float)) for item in value):
                errors.append(f"cell {identity.cell_id} field {name} is not a numeric vector")
        if field.kind == "rules":
            if not isinstance(value, list):
                errors.append(f"cell {identity.cell_id} field {name} is not a rule list")
            else:
                for record in value:
                    try:
                        NeighborPreferenceRule.from_record(record)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"cell {identity.cell_id} invalid preference rule: {exc}")
        if field.kind == "mapping" and not isinstance(value, Mapping):
            errors.append(f"cell {identity.cell_id} field {name} is not a mapping")
    return errors


def validate_identity_catalog(identities: Sequence[CellIdentity], schema: IdentitySchema) -> list[str]:
    errors = schema.validate()
    ids = [identity.cell_id for identity in identities]
    if len(ids) != len(set(ids)):
        errors.append("duplicate cell_id values")
    for identity in identities:
        errors.extend(validate_identity(identity, schema))
    scalar_values = scalar_values_from_identities(identities)
    if any(value is None for value in scalar_values):
        errors.append("scalar mapping produced None")
    return errors


def _component_value(identity: CellIdentity, component: str) -> Any:
    return identity.components.get(component)


def evaluate_neighbor_preferences(
    actor: CellIdentity,
    neighbors: Sequence[CellIdentity],
) -> dict[str, Any]:
    """Evaluate actor target-neighborhood rules against local neighbors."""

    rule_records: list[dict[str, Any]] = []
    total_penalty = 0.0
    satisfied = 0
    for rule in actor.preference_rules():
        count = sum(1 for neighbor in neighbors if _component_value(neighbor, rule.component) == rule.expected_value)
        under = max(0, rule.min_count - count)
        over = 0 if rule.max_count is None else max(0, count - rule.max_count)
        penalty = (under + over) * float(rule.weight)
        total_penalty += penalty
        is_satisfied = penalty == 0.0
        satisfied += int(is_satisfied)
        record = rule.to_record()
        record.update({"observedCount": count, "penalty": penalty, "satisfied": is_satisfied})
        rule_records.append(record)
    score = 1.0 / (1.0 + total_penalty)
    return {
        "schemaVersion": NEIGHBOR_PREFERENCE_VERSION,
        "actorCellId": actor.cell_id,
        "neighborCellIds": [neighbor.cell_id for neighbor in neighbors],
        "ruleCount": len(rule_records),
        "satisfiedRuleCount": satisfied,
        "penalty": total_penalty,
        "score": score,
        "rules": rule_records,
    }


def catalog_rows(identities: Sequence[CellIdentity], schema: IdentitySchema) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identity in identities:
        rows.append(
            {
                "schema_version": identity.schema_version,
                "identity_id": identity.identity_id,
                "cell_id": identity.cell_id,
                "scalar_value": identity.scalar_value,
                "ap_coordinate": identity.components.get("ap_coordinate"),
                "organ_type": identity.components.get("organ_type"),
                "polarity_json": canonical_json(identity.components.get("polarity")),
                "adhesion_type": identity.components.get("adhesion_type"),
                "target_neighbor_preferences_json": canonical_json(identity.components.get("target_neighbor_preferences", [])),
                "observable_components_json": canonical_json(observable_components(identity, schema)),
                "actor_components_json": canonical_json(actor_components(identity, schema)),
                "full_record_json": canonical_json(identity.to_record(schema=schema, include_hidden=True)),
            }
        )
    return rows


def build_example_identity_catalog() -> tuple[CellIdentity, ...]:
    """Build a small deterministic 2D/graph-ready identity catalog."""

    rules_boundary = [
        NeighborPreferenceRule("organ_type", "core", min_count=1, max_count=3, weight=1.0).to_record(),
        NeighborPreferenceRule("adhesion_type", "adhesion_boundary", min_count=1, weight=0.5).to_record(),
    ]
    rules_core = [
        NeighborPreferenceRule("organ_type", "core", min_count=2, weight=0.75).to_record(),
        NeighborPreferenceRule("organ_type", "boundary", min_count=1, max_count=4, weight=0.5).to_record(),
    ]
    rules_appendage = [
        NeighborPreferenceRule("organ_type", "appendage", min_count=1, weight=0.75).to_record(),
        NeighborPreferenceRule("organ_type", "core", min_count=1, weight=0.5).to_record(),
    ]
    rows = [
        (0, 1, 0.0, "boundary", [1.0, 0.0], "adhesion_boundary", rules_boundary),
        (1, 2, 0.25, "core", [1.0, 0.0], "adhesion_a", rules_core),
        (2, 3, 0.50, "core", [0.0, 1.0], "adhesion_a", rules_core),
        (3, 4, 0.75, "appendage", [0.0, 1.0], "adhesion_b", rules_appendage),
        (4, 5, 1.0, "boundary", [-1.0, 0.0], "adhesion_boundary", rules_boundary),
    ]
    return tuple(
        CellIdentity(
            cell_id=cell_id,
            components={
                "scalar_value": scalar_value,
                "ap_coordinate": ap_coordinate,
                "organ_type": organ_type,
                "polarity": polarity,
                "adhesion_type": adhesion_type,
                "target_neighbor_preferences": preferences,
                "internal_state": {"debug_seed": 20260628 + cell_id},
            },
        )
        for cell_id, scalar_value, ap_coordinate, organ_type, polarity, adhesion_type, preferences in rows
    )
