"""Action-set contract for E05 S04 morphogenesis tasks.

S04 keeps actions local to one actor site and, when applicable, one adjacent
target site.  The executor records energy costs and population accounting for
neutral movement, metadata-only actions, and explicit birth/death actions.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.e05.cell_identity import CellIdentity, attach_identity, identity_from_substrate_cell
from src.e05.substrates import SubstrateCell, SubstrateState


ACTION_TYPES = frozenset(
    {
        "wait",
        "swap",
        "crawl",
        "rotate_polarity",
        "divide",
        "die",
        "adhere",
        "detach",
        "exchange_signal",
    }
)
REQUESTED_S04_ACTIONS = (
    "swap",
    "crawl",
    "rotate_polarity",
    "divide",
    "die",
    "adhere",
    "detach",
    "exchange_signal",
)
SIGNAL_CHANNEL_RANGES = {
    "blocked": (0.0, 1.0),
    "sorted": (0.0, 1.0),
    "frustrated": (0.0, 1.0),
    "target_seeking": (-1.0, 1.0),
    "morphogen": (0.0, 1.0),
}
ADHESION_BONDS_KEY = "e05_adhesion_bonds"
SIGNAL_OUTBOX_KEY = "e05_signal_outbox"
SIGNAL_INBOX_KEY = "e05_signal_inbox"
DIVISION_PARENT_KEY = "e05_division_parent_cell_id"
ACTIVE_STATUSES = frozenset({"active", "cellstatus.active"})


@dataclass(frozen=True)
class ActionSpec:
    """One configured local action with cost and accounting semantics."""

    action_type: str
    energy_cost: float
    requires_target: bool
    target_occupancy: str
    population_delta: int
    local_semantics: str
    conservation_rule: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        action_type = str(self.action_type)
        if action_type not in ACTION_TYPES:
            raise ValueError(f"unsupported action_type: {action_type}")
        if float(self.energy_cost) < 0.0:
            raise ValueError("energy_cost must be non-negative")
        if self.target_occupancy not in {"none", "empty", "occupied", "occupied_or_empty", "occupied_movable"}:
            raise ValueError(f"unsupported target_occupancy: {self.target_occupancy}")
        if int(self.population_delta) not in {-1, 0, 1}:
            raise ValueError("population_delta must be -1, 0, or 1")
        object.__setattr__(self, "action_type", action_type)
        object.__setattr__(self, "energy_cost", float(self.energy_cost))
        object.__setattr__(self, "population_delta", int(self.population_delta))
        object.__setattr__(self, "parameters", dict(self.parameters))

    def compact_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "energy_cost": self.energy_cost,
            "requires_target": bool(self.requires_target),
            "target_occupancy": self.target_occupancy,
            "population_delta": self.population_delta,
            "local_semantics": self.local_semantics,
            "conservation_rule": self.conservation_rule,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class MorphogenesisActionRequest:
    """One local action proposal from a policy or validation fixture."""

    action_type: str
    source_site_id: int
    target_site_id: int | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_type", str(self.action_type))
        object.__setattr__(self, "source_site_id", int(self.source_site_id))
        object.__setattr__(self, "target_site_id", None if self.target_site_id is None else int(self.target_site_id))
        object.__setattr__(self, "parameters", dict(self.parameters))

    def compact_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "source_site_id": self.source_site_id,
            "target_site_id": self.target_site_id,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class MorphogenesisActionResult:
    """Auditable action outcome with cost and population accounting."""

    action_type: str
    source_site_id: int
    target_site_id: int | None
    allowed: bool
    reason: str
    state_changed: bool
    nominal_energy_cost: float
    energy_cost_charged: float
    population_before: int
    population_after: int
    birth_count: int
    death_count: int
    conservation_delta: int
    signature_before: tuple[tuple[Any, ...], ...]
    signature_after: tuple[tuple[Any, ...], ...]
    details: Mapping[str, Any] = field(default_factory=dict)

    def compact_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "source_site_id": self.source_site_id,
            "target_site_id": self.target_site_id,
            "allowed": bool(self.allowed),
            "reason": self.reason,
            "state_changed": bool(self.state_changed),
            "nominal_energy_cost": self.nominal_energy_cost,
            "energy_cost_charged": self.energy_cost_charged,
            "population_before": self.population_before,
            "population_after": self.population_after,
            "birth_count": self.birth_count,
            "death_count": self.death_count,
            "conservation_delta": self.conservation_delta,
            "signature_before": [list(row) for row in self.signature_before],
            "signature_after": [list(row) for row in self.signature_after],
            "details": dict(self.details),
        }


class ActionExecutor:
    """Execute configured S04 actions on an S01 substrate state."""

    def __init__(self, action_set: Mapping[str, ActionSpec] | None = None) -> None:
        self.action_set = dict(action_set or default_action_set())

    def execute(self, state: SubstrateState, request: MorphogenesisActionRequest) -> MorphogenesisActionResult:
        spec = self.action_set.get(request.action_type)
        signature_before = action_state_signature(state)
        population_before = occupied_site_count(state)
        if spec is None:
            return self._finish(state, request, None, False, "unsupported_action", signature_before, population_before)
        if request.source_site_id not in state.site_ids:
            return self._finish(state, request, spec, False, "source_site_missing", signature_before, population_before)
        source_cell = state.cell_at(request.source_site_id)
        if source_cell is None:
            return self._finish(state, request, spec, False, "actor_missing", signature_before, population_before)
        if request.action_type != "wait" and not _is_active(source_cell):
            return self._finish(state, request, spec, False, "actor_not_active_or_missing", signature_before, population_before)
        if spec.target_occupancy == "none" and request.target_site_id is not None:
            return self._finish(state, request, spec, False, "target_not_allowed", signature_before, population_before)
        if spec.requires_target:
            target_reason = self._validate_target(state, request, spec)
            if target_reason != "allowed":
                return self._finish(state, request, spec, False, target_reason, signature_before, population_before)

        if request.action_type == "wait":
            return self._finish(state, request, spec, True, "allowed", signature_before, population_before)
        if request.action_type == "swap":
            low_level = state.apply_action("swap", request.source_site_id, request.target_site_id)
            return self._finish(state, request, spec, low_level.allowed, low_level.reason, signature_before, population_before)
        if request.action_type == "crawl":
            low_level = state.apply_action("crawl", request.source_site_id, request.target_site_id)
            return self._finish(state, request, spec, low_level.allowed, low_level.reason, signature_before, population_before)
        if request.action_type == "rotate_polarity":
            return self._rotate_polarity(state, request, spec, signature_before, population_before)
        if request.action_type == "divide":
            return self._divide(state, request, spec, signature_before, population_before)
        if request.action_type == "die":
            state.remove_cell(request.source_site_id)
            return self._finish(state, request, spec, True, "allowed", signature_before, population_before, death_count=1)
        if request.action_type == "adhere":
            return self._adhere(state, request, spec, signature_before, population_before)
        if request.action_type == "detach":
            return self._detach(state, request, spec, signature_before, population_before)
        if request.action_type == "exchange_signal":
            return self._exchange_signal(state, request, spec, signature_before, population_before)
        return self._finish(state, request, spec, False, "unsupported_action", signature_before, population_before)

    def legal_action_results(self, state: SubstrateState, source_site_id: int) -> tuple[MorphogenesisActionResult, ...]:
        """Dry-run configured actions against local neighbor sites."""

        source_site_id = int(source_site_id)
        actions: list[MorphogenesisActionResult] = [
            self.dry_run(state, MorphogenesisActionRequest("wait", source_site_id)),
            self.dry_run(state, MorphogenesisActionRequest("die", source_site_id)),
            self.dry_run(state, MorphogenesisActionRequest("rotate_polarity", source_site_id, parameters={"polarity": (1.0, 0.0)})),
        ]
        for target_site_id in state.neighbors(source_site_id):
            actions.extend(
                (
                    self.dry_run(state, MorphogenesisActionRequest("swap", source_site_id, target_site_id)),
                    self.dry_run(state, MorphogenesisActionRequest("crawl", source_site_id, target_site_id)),
                    self.dry_run(state, MorphogenesisActionRequest("divide", source_site_id, target_site_id)),
                    self.dry_run(state, MorphogenesisActionRequest("adhere", source_site_id, target_site_id)),
                    self.dry_run(state, MorphogenesisActionRequest("detach", source_site_id, target_site_id)),
                    self.dry_run(
                        state,
                        MorphogenesisActionRequest(
                            "exchange_signal",
                            source_site_id,
                            target_site_id,
                            {"channel": "blocked", "value": 1.0},
                        ),
                    ),
                )
            )
        return tuple(actions)

    def dry_run(self, state: SubstrateState, request: MorphogenesisActionRequest) -> MorphogenesisActionResult:
        clone = clone_substrate_state(state)
        return self.execute(clone, request)

    def _validate_target(self, state: SubstrateState, request: MorphogenesisActionRequest, spec: ActionSpec) -> str:
        if request.target_site_id is None:
            return "target_required"
        if request.target_site_id not in state.site_ids:
            return "target_site_missing"
        if request.source_site_id == request.target_site_id:
            return "source_equals_target"
        if not state.is_adjacent(request.source_site_id, request.target_site_id):
            return "target_not_adjacent"
        target_cell = state.cell_at(request.target_site_id)
        if spec.target_occupancy == "empty" and target_cell is not None:
            return "target_not_empty"
        if spec.target_occupancy in {"occupied", "occupied_movable"} and target_cell is None:
            return "target_empty"
        return "allowed"

    def _rotate_polarity(
        self,
        state: SubstrateState,
        request: MorphogenesisActionRequest,
        spec: ActionSpec,
        signature_before: tuple[tuple[Any, ...], ...],
        population_before: int,
    ) -> MorphogenesisActionResult:
        vector = request.parameters.get("polarity", request.parameters.get("vector"))
        if vector is None:
            return self._finish(state, request, spec, False, "polarity_vector_required", signature_before, population_before)
        source_cell = state.cell_at(request.source_site_id)
        assert source_cell is not None
        try:
            identity = identity_from_substrate_cell(source_cell, fallback_scalar=False)
        except ValueError:
            return self._finish(state, request, spec, False, "identity_with_polarity_required", signature_before, population_before)
        if "polarity" not in identity.components:
            return self._finish(state, request, spec, False, "identity_with_polarity_required", signature_before, population_before)
        try:
            polarity = _normalize_vector(vector)
        except (TypeError, ValueError):
            return self._finish(state, request, spec, False, "invalid_polarity_vector", signature_before, population_before)
        updated = CellIdentity(
            identity_id=identity.identity_id,
            components={**identity.components, "polarity": polarity},
            target_preferences=identity.target_preferences,
            metadata={**identity.metadata, "last_action": "rotate_polarity"},
        )
        _replace_cell(state, request.source_site_id, attach_identity(source_cell, updated))
        return self._finish(state, request, spec, True, "allowed", signature_before, population_before, details={"polarity": polarity})

    def _divide(
        self,
        state: SubstrateState,
        request: MorphogenesisActionRequest,
        spec: ActionSpec,
        signature_before: tuple[tuple[Any, ...], ...],
        population_before: int,
    ) -> MorphogenesisActionResult:
        source_cell = state.cell_at(request.source_site_id)
        assert source_cell is not None
        daughter_id = str(request.parameters.get("daughter_cell_id", f"{source_cell.cell_id}_daughter_{request.target_site_id}"))
        if _cell_id_exists(state, daughter_id):
            return self._finish(state, request, spec, False, "daughter_cell_id_not_unique", signature_before, population_before)
        daughter_metadata = dict(source_cell.metadata)
        daughter_metadata[DIVISION_PARENT_KEY] = str(source_cell.cell_id)
        daughter = SubstrateCell(
            cell_id=daughter_id,
            value=source_cell.value,
            label=source_cell.label,
            status="active",
            metadata=daughter_metadata,
        )
        try:
            identity = identity_from_substrate_cell(source_cell, fallback_scalar=False)
        except ValueError:
            identity = None
        if identity is not None:
            daughter_identity = CellIdentity(
                identity_id=f"{identity.identity_id}:daughter:{request.target_site_id}",
                components=dict(identity.components),
                target_preferences=identity.target_preferences,
                metadata={**identity.metadata, DIVISION_PARENT_KEY: str(source_cell.cell_id)},
            )
            daughter = attach_identity(daughter, daughter_identity)
        state.place_cell(request.target_site_id, daughter)
        return self._finish(
            state,
            request,
            spec,
            True,
            "allowed",
            signature_before,
            population_before,
            birth_count=1,
            details={"daughter_cell_id": daughter_id},
        )

    def _adhere(
        self,
        state: SubstrateState,
        request: MorphogenesisActionRequest,
        spec: ActionSpec,
        signature_before: tuple[tuple[Any, ...], ...],
        population_before: int,
    ) -> MorphogenesisActionResult:
        source_cell, target_cell = _source_target_cells(state, request)
        if _has_bond(source_cell, target_cell):
            return self._finish(state, request, spec, False, "adhesion_bond_exists", signature_before, population_before)
        _replace_two_cells(
            state,
            request.source_site_id,
            _with_bond(source_cell, target_cell.cell_id, add=True),
            request.target_site_id,
            _with_bond(target_cell, source_cell.cell_id, add=True),
        )
        return self._finish(state, request, spec, True, "allowed", signature_before, population_before)

    def _detach(
        self,
        state: SubstrateState,
        request: MorphogenesisActionRequest,
        spec: ActionSpec,
        signature_before: tuple[tuple[Any, ...], ...],
        population_before: int,
    ) -> MorphogenesisActionResult:
        source_cell, target_cell = _source_target_cells(state, request)
        if not _has_bond(source_cell, target_cell):
            return self._finish(state, request, spec, False, "adhesion_bond_missing", signature_before, population_before)
        _replace_two_cells(
            state,
            request.source_site_id,
            _with_bond(source_cell, target_cell.cell_id, add=False),
            request.target_site_id,
            _with_bond(target_cell, source_cell.cell_id, add=False),
        )
        return self._finish(state, request, spec, True, "allowed", signature_before, population_before)

    def _exchange_signal(
        self,
        state: SubstrateState,
        request: MorphogenesisActionRequest,
        spec: ActionSpec,
        signature_before: tuple[tuple[Any, ...], ...],
        population_before: int,
    ) -> MorphogenesisActionResult:
        channel = request.parameters.get("channel")
        if channel is None:
            return self._finish(state, request, spec, False, "signal_channel_required", signature_before, population_before)
        channel = str(channel)
        if channel not in SIGNAL_CHANNEL_RANGES:
            return self._finish(state, request, spec, False, "unsupported_signal_channel", signature_before, population_before)
        try:
            value = float(request.parameters["value"])
        except (KeyError, TypeError, ValueError):
            return self._finish(state, request, spec, False, "signal_value_required", signature_before, population_before)
        lower, upper = SIGNAL_CHANNEL_RANGES[channel]
        if value < lower or value > upper:
            return self._finish(state, request, spec, False, "signal_value_out_of_range", signature_before, population_before)
        source_cell, target_cell = _source_target_cells(state, request)
        record = {
            "channel": channel,
            "value": value,
            "source_site_id": request.source_site_id,
            "target_site_id": request.target_site_id,
            "source_cell_id": source_cell.cell_id,
            "target_cell_id": target_cell.cell_id,
            "access_scope": "adjacent_neighbor",
        }
        _replace_two_cells(
            state,
            request.source_site_id,
            _append_metadata_record(source_cell, SIGNAL_OUTBOX_KEY, record),
            request.target_site_id,
            _append_metadata_record(target_cell, SIGNAL_INBOX_KEY, record),
        )
        return self._finish(state, request, spec, True, "allowed", signature_before, population_before, details=record)

    def _finish(
        self,
        state: SubstrateState,
        request: MorphogenesisActionRequest,
        spec: ActionSpec | None,
        allowed: bool,
        reason: str,
        signature_before: tuple[tuple[Any, ...], ...],
        population_before: int,
        *,
        birth_count: int = 0,
        death_count: int = 0,
        details: Mapping[str, Any] | None = None,
    ) -> MorphogenesisActionResult:
        signature_after = action_state_signature(state)
        population_after = occupied_site_count(state)
        nominal = 0.0 if spec is None else spec.energy_cost
        return MorphogenesisActionResult(
            action_type=request.action_type,
            source_site_id=request.source_site_id,
            target_site_id=request.target_site_id,
            allowed=bool(allowed),
            reason=str(reason),
            state_changed=signature_before != signature_after,
            nominal_energy_cost=nominal,
            energy_cost_charged=nominal if allowed else 0.0,
            population_before=population_before,
            population_after=population_after,
            birth_count=int(birth_count) if allowed else 0,
            death_count=int(death_count) if allowed else 0,
            conservation_delta=population_after - population_before,
            signature_before=signature_before,
            signature_after=signature_after,
            details=dict(details or {}),
        )


def default_action_set() -> dict[str, ActionSpec]:
    """Return the default S04 local action set and toy energy costs."""

    return {
        "wait": ActionSpec("wait", 0.0, False, "none", 0, "actor remains unchanged at its current site", "population conserved"),
        "swap": ActionSpec("swap", 1.0, True, "occupied_movable", 0, "actor swaps with one adjacent occupied movable target", "population conserved"),
        "crawl": ActionSpec("crawl", 1.2, True, "empty", 0, "actor moves into one adjacent empty target site", "population conserved"),
        "rotate_polarity": ActionSpec("rotate_polarity", 0.25, False, "none", 0, "actor updates only its own polarity vector", "population conserved"),
        "divide": ActionSpec("divide", 2.5, True, "empty", 1, "actor creates one daughter in an adjacent empty site", "birth_count increments by one"),
        "die": ActionSpec("die", 0.5, False, "none", -1, "actor is removed from its current site", "death_count increments by one"),
        "adhere": ActionSpec("adhere", 0.4, True, "occupied", 0, "actor creates a reciprocal adhesion bond with an adjacent occupied target", "population conserved"),
        "detach": ActionSpec("detach", 0.3, True, "occupied", 0, "actor removes a reciprocal adhesion bond with an adjacent occupied target", "population conserved"),
        "exchange_signal": ActionSpec(
            "exchange_signal",
            0.2,
            True,
            "occupied",
            0,
            "actor writes one bounded E04-style signal record to an adjacent occupied target",
            "population conserved",
            parameters={"channels": sorted(SIGNAL_CHANNEL_RANGES)},
        ),
    }


def action_spec_rows(action_set: Mapping[str, ActionSpec] | None = None) -> list[dict[str, Any]]:
    specs = action_set or default_action_set()
    rows = []
    for action_type in sorted(specs):
        spec = specs[action_type]
        rows.append(
            {
                "action_type": spec.action_type,
                "energy_cost": spec.energy_cost,
                "requires_target": spec.requires_target,
                "target_occupancy": spec.target_occupancy,
                "population_delta": spec.population_delta,
                "local_semantics": spec.local_semantics,
                "conservation_rule": spec.conservation_rule,
                "spec_sha256": stable_action_sha256(spec.compact_dict()),
            }
        )
    return rows


def stable_action_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def occupied_site_count(state: SubstrateState) -> int:
    return sum(1 for site_id in state.site_ids if state.cell_at(site_id) is not None)


def action_state_signature(state: SubstrateState) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            int(site_id),
            None if state.cell_at(site_id) is None else stable_action_sha256(state.cell_at(site_id).compact_dict()),
        )
        for site_id in state.site_ids
    )


def clone_substrate_state(state: SubstrateState) -> SubstrateState:
    clone = state.copy_empty()
    for site_id in state.site_ids:
        cell = state.cell_at(site_id)
        if cell is not None:
            clone.place_cell(site_id, cell)
    return clone


def action_outcome_rows(results: Sequence[MorphogenesisActionResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        rows.append(
            {
                "action_type": result.action_type,
                "source_site_id": result.source_site_id,
                "target_site_id": result.target_site_id,
                "allowed": result.allowed,
                "reason": result.reason,
                "state_changed": result.state_changed,
                "nominal_energy_cost": result.nominal_energy_cost,
                "energy_cost_charged": result.energy_cost_charged,
                "population_before": result.population_before,
                "population_after": result.population_after,
                "birth_count": result.birth_count,
                "death_count": result.death_count,
                "conservation_delta": result.conservation_delta,
            }
        )
    return rows


def _status_key(status: str) -> str:
    return str(status).strip().lower()


def _is_active(cell: SubstrateCell) -> bool:
    return _status_key(cell.status) in ACTIVE_STATUSES


def _normalize_vector(vector: Any) -> tuple[float, ...]:
    values = tuple(float(item) for item in vector)
    if not values:
        raise ValueError("polarity vector must not be empty")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise ValueError("polarity vector must not be zero")
    return tuple(value / norm for value in values)


def _replace_cell(state: SubstrateState, site_id: int, cell: SubstrateCell) -> None:
    state.remove_cell(site_id)
    state.place_cell(site_id, cell)


def _replace_two_cells(
    state: SubstrateState,
    left_site_id: int,
    left_cell: SubstrateCell,
    right_site_id: int,
    right_cell: SubstrateCell,
) -> None:
    state.remove_cell(left_site_id)
    state.remove_cell(right_site_id)
    state.place_cell(left_site_id, left_cell)
    state.place_cell(right_site_id, right_cell)


def _source_target_cells(state: SubstrateState, request: MorphogenesisActionRequest) -> tuple[SubstrateCell, SubstrateCell]:
    source_cell = state.cell_at(request.source_site_id)
    target_cell = state.cell_at(request.target_site_id)
    assert source_cell is not None and target_cell is not None
    return source_cell, target_cell


def _with_metadata(cell: SubstrateCell, metadata: Mapping[str, Any]) -> SubstrateCell:
    return SubstrateCell(
        cell_id=cell.cell_id,
        value=cell.value,
        label=cell.label,
        status=cell.status,
        metadata=dict(metadata),
    )


def _with_bond(cell: SubstrateCell, other_cell_id: str, *, add: bool) -> SubstrateCell:
    metadata = dict(cell.metadata)
    bonds = set(str(item) for item in metadata.get(ADHESION_BONDS_KEY, ()))
    if add:
        bonds.add(str(other_cell_id))
    else:
        bonds.discard(str(other_cell_id))
    metadata[ADHESION_BONDS_KEY] = sorted(bonds)
    return _with_metadata(cell, metadata)


def _has_bond(left: SubstrateCell, right: SubstrateCell) -> bool:
    left_bonds = set(str(item) for item in left.metadata.get(ADHESION_BONDS_KEY, ()))
    right_bonds = set(str(item) for item in right.metadata.get(ADHESION_BONDS_KEY, ()))
    return str(right.cell_id) in left_bonds and str(left.cell_id) in right_bonds


def _append_metadata_record(cell: SubstrateCell, key: str, record: Mapping[str, Any]) -> SubstrateCell:
    metadata = dict(cell.metadata)
    records = list(metadata.get(key, ()))
    records.append(dict(record))
    metadata[key] = records
    return _with_metadata(cell, metadata)


def _cell_id_exists(state: SubstrateState, cell_id: str) -> bool:
    try:
        state.site_of_cell(cell_id)
    except KeyError:
        return False
    return True
