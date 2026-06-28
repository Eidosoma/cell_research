"""Generalized local action semantics for E05 S04.

S01 keeps movement deliberately conservative. This module adds an S04 action
world that can represent births, deaths, detached live cells, adhesion bonds,
local signal exchange, and explicit energy/conservation accounting.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .substrates import CellState, Position, Substrate


ACTION_SCHEMA_VERSION = "e05_s04_action_semantics.v1"
ACTION_TRACE_SCHEMA_VERSION = "e05_s04_action_trace.v1"
LOCAL_ACTION_OBSERVATION_VERSION = "e05_s04_local_action_observation.v1"

SUPPORTED_ACTIONS = (
    "wait",
    "swap",
    "crawl",
    "rotate",
    "divide",
    "die",
    "adhere",
    "detach",
    "exchange_signal",
)
ACTION_ENERGY_COSTS = {
    "wait": 0.0,
    "swap": 1.0,
    "crawl": 1.0,
    "rotate": 0.25,
    "divide": 2.5,
    "die": 0.5,
    "adhere": 0.4,
    "detach": 0.6,
    "exchange_signal": 0.2,
}
CONSERVATION_MODES = {
    "wait": "conservative",
    "swap": "conservative",
    "crawl": "conservative",
    "rotate": "conservative_identity_mutation",
    "divide": "non_conservative_birth",
    "die": "non_conservative_death",
    "adhere": "conservative_topology_mutation",
    "detach": "non_conservative_detach",
    "exchange_signal": "conservative_signal_mutation",
}
LOCAL_SIGNAL_KEY = "local_signals"
LOCAL_SIGNAL_EMITTED_KEY = "local_signals_emitted"
SIGNAL_CHANNEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
PROHIBITED_LOCAL_ACTION_KEYS = {
    "all_cells",
    "all_positions",
    "all_target_identities",
    "all_target_positions",
    "cells",
    "global_target_state",
    "hidden",
    "internal_state",
    "occupancy",
    "target_map",
    "trace_rows",
}


def _as_position(value: int | Sequence[int]) -> Position:
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
    if isinstance(value, set):
        return sorted(_json_ready(item) for item in value)
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


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:16]


def _bond(left: int, right: int) -> tuple[int, int]:
    if int(left) == int(right):
        raise ValueError("adhesion bond cannot connect a cell to itself")
    return tuple(sorted((int(left), int(right))))  # type: ignore[return-value]


def _visible_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _json_ready(value)
        for key, value in identity.items()
        if str(key) not in {"internal_state", "hidden"} and not str(key).startswith("_hidden")
    }


@dataclass(frozen=True)
class ActionProposal:
    """An actor-local action proposal before S04 world constraints are applied."""

    action: str
    actor_cell_id: int
    target_position: Position | None = None
    reason: str = "manual"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        action = str(self.action)
        target = None if self.target_position is None else _as_position(self.target_position)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "actor_cell_id", int(self.actor_cell_id))
        object.__setattr__(self, "target_position", target)
        object.__setattr__(self, "parameters", dict(self.parameters))

    @classmethod
    def wait(cls, actor_cell_id: int, reason: str = "wait") -> "ActionProposal":
        return cls("wait", int(actor_cell_id), None, reason)

    @classmethod
    def swap(cls, actor_cell_id: int, target_position: Position, reason: str = "swap") -> "ActionProposal":
        return cls("swap", int(actor_cell_id), _as_position(target_position), reason)

    @classmethod
    def crawl(cls, actor_cell_id: int, target_position: Position, reason: str = "crawl") -> "ActionProposal":
        return cls("crawl", int(actor_cell_id), _as_position(target_position), reason)

    @classmethod
    def rotate(cls, actor_cell_id: int, quarter_turns: int = 1, reason: str = "rotate") -> "ActionProposal":
        return cls("rotate", int(actor_cell_id), None, reason, {"quarter_turns": int(quarter_turns)})

    @classmethod
    def divide(
        cls,
        actor_cell_id: int,
        target_position: Position,
        *,
        child_value: Any | None = None,
        child_identity_updates: Mapping[str, Any] | None = None,
        reason: str = "divide",
    ) -> "ActionProposal":
        params: dict[str, Any] = {"child_identity_updates": dict(child_identity_updates or {})}
        if child_value is not None:
            params["child_value"] = child_value
        return cls("divide", int(actor_cell_id), _as_position(target_position), reason, params)

    @classmethod
    def die(cls, actor_cell_id: int, reason: str = "die") -> "ActionProposal":
        return cls("die", int(actor_cell_id), None, reason)

    @classmethod
    def adhere(cls, actor_cell_id: int, target_position: Position, reason: str = "adhere") -> "ActionProposal":
        return cls("adhere", int(actor_cell_id), _as_position(target_position), reason)

    @classmethod
    def detach(cls, actor_cell_id: int, reason: str = "detach") -> "ActionProposal":
        return cls("detach", int(actor_cell_id), None, reason)

    @classmethod
    def exchange_signal(
        cls,
        actor_cell_id: int,
        target_position: Position,
        *,
        channel: str = "morphogen",
        amount: float = 1.0,
        reason: str = "exchange_signal",
    ) -> "ActionProposal":
        return cls(
            "exchange_signal",
            int(actor_cell_id),
            _as_position(target_position),
            reason,
            {"channel": str(channel), "amount": float(amount)},
        )


@dataclass(frozen=True)
class ActionNeighborView:
    position: Position
    relative_offset: Position | None
    occupied: bool
    cell: CellState | None = None
    adhered_to_actor: bool = False


@dataclass(frozen=True)
class ActionObservation:
    """Local neighborhood exposed to a policy before it proposes an action."""

    schema_version: str
    substrate_type: str
    actor: CellState
    actor_position: Position
    neighbors: tuple[ActionNeighborView, ...]
    boundary: str
    actor_signal_state: Mapping[str, Any] = field(default_factory=dict)

    @property
    def occupied_neighbors(self) -> tuple[ActionNeighborView, ...]:
        return tuple(neighbor for neighbor in self.neighbors if neighbor.occupied)

    @property
    def empty_neighbors(self) -> tuple[ActionNeighborView, ...]:
        return tuple(neighbor for neighbor in self.neighbors if not neighbor.occupied)

    def to_policy_payload(self) -> dict[str, Any]:
        return local_action_policy_payload(self)


@dataclass(frozen=True)
class ActionOutcome:
    step_index: int
    action: str
    actor_cell_id: int
    source_position_before: Position | None
    target_position: Position | None
    actor_position_after: Position | None
    target_cell_id_before: int | None
    accepted: bool
    legal: bool
    reason: str
    collision: bool
    energy_cost: float
    conservation_mode: str
    cell_count_before: int
    cell_count_after: int
    occupied_count_before: int
    occupied_count_after: int
    detached_count_before: int
    detached_count_after: int
    born_cell_id: int | None = None
    died_cell_id: int | None = None
    detached_cell_id: int | None = None
    adhesion_bond_delta: int = 0
    signal_delta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def cell_delta(self) -> int:
        return int(self.cell_count_after - self.cell_count_before)

    @property
    def occupied_delta(self) -> int:
        return int(self.occupied_count_after - self.occupied_count_before)

    @property
    def detached_delta(self) -> int:
        return int(self.detached_count_after - self.detached_count_before)

    def to_record(self) -> dict[str, Any]:
        return {
            "stepIndex": int(self.step_index),
            "action": self.action,
            "actorCellId": int(self.actor_cell_id),
            "sourcePositionBefore": None if self.source_position_before is None else list(self.source_position_before),
            "targetPosition": None if self.target_position is None else list(self.target_position),
            "actorPositionAfter": None if self.actor_position_after is None else list(self.actor_position_after),
            "targetCellIdBefore": self.target_cell_id_before,
            "accepted": bool(self.accepted),
            "legal": bool(self.legal),
            "reason": self.reason,
            "collision": bool(self.collision),
            "energyCost": float(self.energy_cost),
            "conservationMode": self.conservation_mode,
            "cellDelta": self.cell_delta,
            "occupiedDelta": self.occupied_delta,
            "detachedDelta": self.detached_delta,
            "bornCellId": self.born_cell_id,
            "diedCellId": self.died_cell_id,
            "detachedCellId": self.detached_cell_id,
            "adhesionBondDelta": int(self.adhesion_bond_delta),
            "signalDelta": dict(self.signal_delta),
        }


@dataclass
class ActionWorld:
    """Mutable S04 world with conservative and non-conservative local actions."""

    substrate: Substrate
    cells: Mapping[int, CellState]
    occupancy: Mapping[Position, int]
    detached_cell_ids: set[int] = field(default_factory=set)
    adhesion_bonds: set[tuple[int, int]] = field(default_factory=set)
    condition_id: str = "manual"
    trace_rows: list[dict[str, Any]] = field(default_factory=list)
    step_index: int = 0
    energy_spent: float = 0.0
    next_cell_id: int | None = None

    def __post_init__(self) -> None:
        self.cells = {int(cell_id): cell for cell_id, cell in self.cells.items()}
        self.occupancy = {_as_position(position): int(cell_id) for position, cell_id in self.occupancy.items()}
        self.detached_cell_ids = {int(cell_id) for cell_id in self.detached_cell_ids}
        self.adhesion_bonds = {_bond(left, right) for left, right in self.adhesion_bonds}
        if self.next_cell_id is None:
            self.next_cell_id = 0 if not self.cells else max(self.cells) + 1
        self.assert_valid_state()
        if not self.trace_rows:
            self._append_trace_row(
                event_kind="initial",
                proposal=ActionProposal.wait(-1, "initial"),
                outcome=None,
                counts_before=self._counts(),
                counts_after=self._counts(),
            )

    @classmethod
    def from_position_cells(
        cls,
        substrate: Substrate,
        position_cells: Mapping[Position, CellState],
        *,
        condition_id: str = "manual",
    ) -> "ActionWorld":
        cells: dict[int, CellState] = {}
        occupancy: dict[Position, int] = {}
        for position, cell in position_cells.items():
            position = _as_position(position)
            cells[int(cell.cell_id)] = cell
            occupancy[position] = int(cell.cell_id)
        return cls(substrate=substrate, cells=cells, occupancy=occupancy, condition_id=condition_id)

    def clone(self) -> "ActionWorld":
        return ActionWorld(
            substrate=self.substrate,
            cells=dict(self.cells),
            occupancy=dict(self.occupancy),
            detached_cell_ids=set(self.detached_cell_ids),
            adhesion_bonds=set(self.adhesion_bonds),
            condition_id=self.condition_id,
            trace_rows=list(self.trace_rows),
            step_index=self.step_index,
            energy_spent=self.energy_spent,
            next_cell_id=self.next_cell_id,
        )

    def occupied_count(self) -> int:
        return len(self.occupancy)

    def detached_count(self) -> int:
        return len(self.detached_cell_ids)

    def cell_position(self, cell_id: int) -> Position | None:
        for position, occupied_cell_id in self.occupancy.items():
            if occupied_cell_id == int(cell_id):
                return position
        return None

    def position_values(self) -> dict[Position, Any]:
        return {position: self.cells[cell_id].value for position, cell_id in self.occupancy.items()}

    def cell_id_counter(self) -> Counter[int]:
        return Counter(self.occupancy.values())

    def value_counter(self) -> Counter[Any]:
        return Counter(cell.value for cell in self.cells.values())

    def occupancy_hash(self) -> str:
        rows = [
            {"position": position, "cell_id": cell_id, "value": self.cells[cell_id].value}
            for position, cell_id in sorted(self.occupancy.items())
        ]
        return _stable_hash(rows)

    def cell_identity_hash(self) -> str:
        rows = [self.cells[cell_id].to_record() for cell_id in sorted(self.cells)]
        payload = {
            "cells": rows,
            "detachedCellIds": sorted(self.detached_cell_ids),
            "adhesionBonds": sorted(self.adhesion_bonds),
        }
        return _stable_hash(payload)

    def _counts(self) -> dict[str, int]:
        return {
            "cell": len(self.cells),
            "occupied": len(self.occupancy),
            "detached": len(self.detached_cell_ids),
        }

    def validate_state(self) -> list[str]:
        errors: list[str] = []
        node_set = set(self.substrate.nodes)
        if any(position not in node_set for position in self.occupancy):
            errors.append("occupancy references non-substrate position")
        occupied_ids = list(self.occupancy.values())
        if any(cell_id not in self.cells for cell_id in occupied_ids):
            errors.append("occupancy references unknown cell")
        counts = Counter(occupied_ids)
        duplicated = sorted(cell_id for cell_id, count in counts.items() if count != 1)
        if duplicated:
            errors.append(f"cell ids occupy multiple positions or invalid counts: {duplicated}")
        unknown_detached = sorted(cell_id for cell_id in self.detached_cell_ids if cell_id not in self.cells)
        if unknown_detached:
            errors.append(f"detached cell ids are unknown: {unknown_detached}")
        overlap = sorted(set(occupied_ids) & self.detached_cell_ids)
        if overlap:
            errors.append(f"cells cannot be occupied and detached simultaneously: {overlap}")
        untracked = sorted(set(self.cells) - set(occupied_ids) - self.detached_cell_ids)
        if untracked:
            errors.append(f"live cells are neither occupied nor detached: {untracked}")
        if len(self.occupancy) > len(self.substrate.nodes):
            errors.append("more occupied cells than substrate nodes")
        for left, right in sorted(self.adhesion_bonds):
            if left not in self.cells or right not in self.cells:
                errors.append(f"adhesion bond references unknown cell: {(left, right)}")
            elif left in self.detached_cell_ids or right in self.detached_cell_ids:
                errors.append(f"adhesion bond references detached cell: {(left, right)}")
            else:
                left_position = self.cell_position(left)
                right_position = self.cell_position(right)
                if left_position is None or right_position is None:
                    errors.append(f"adhesion bond references unoccupied cell: {(left, right)}")
                elif not self.substrate.edge_exists(left_position, right_position):
                    errors.append(f"adhesion bond cells are not adjacent: {(left, right)}")
        return errors

    def assert_valid_state(self) -> None:
        errors = self.validate_state()
        if errors:
            raise ValueError("; ".join(errors))

    def local_observation(self, cell_id: int) -> ActionObservation:
        position = self.cell_position(cell_id)
        if position is None:
            raise KeyError(f"cell {cell_id} is not currently occupying a substrate node")
        actor = self.cells[int(cell_id)]
        neighbors: list[ActionNeighborView] = []
        for neighbor_position in self.substrate.neighbors(position):
            neighbor_cell_id = self.occupancy.get(neighbor_position)
            adhered = False if neighbor_cell_id is None else _bond(cell_id, neighbor_cell_id) in self.adhesion_bonds
            neighbors.append(
                ActionNeighborView(
                    position=neighbor_position,
                    relative_offset=self.substrate.relative_offset(position, neighbor_position),
                    occupied=neighbor_cell_id is not None,
                    cell=None if neighbor_cell_id is None else self.cells[neighbor_cell_id],
                    adhered_to_actor=adhered,
                )
            )
        return ActionObservation(
            schema_version=LOCAL_ACTION_OBSERVATION_VERSION,
            substrate_type=self.substrate.substrate_type,
            actor=actor,
            actor_position=position,
            neighbors=tuple(neighbors),
            boundary=self.substrate.boundary,
            actor_signal_state=dict(actor.identity.get(LOCAL_SIGNAL_KEY, {})),
        )

    def _target_context(self, proposal: ActionProposal) -> tuple[Position | None, Position | None, int | None]:
        source = self.cell_position(proposal.actor_cell_id)
        target = None if proposal.target_position is None else _as_position(proposal.target_position)
        target_cell_id = None if target is None else self.occupancy.get(target)
        return source, target, target_cell_id

    def is_legal_action(self, proposal: ActionProposal) -> tuple[bool, str, bool]:
        action = proposal.action
        if action not in SUPPORTED_ACTIONS:
            return False, f"unknown_action:{action}", False
        if action == "wait":
            return True, proposal.reason, False
        actor_exists = proposal.actor_cell_id in self.cells
        if not actor_exists:
            return False, "actor_unknown_or_dead", False
        source, target, target_cell_id = self._target_context(proposal)
        if action == "die":
            return True, proposal.reason, False
        if source is None:
            return False, "actor_detached_or_unoccupied", False
        if action == "rotate":
            polarity = self.cells[proposal.actor_cell_id].identity.get("polarity")
            if not _valid_2d_vector(polarity):
                return False, "rotate_requires_2d_polarity", False
            try:
                int(proposal.parameters.get("quarter_turns", 1))
            except (TypeError, ValueError):
                return False, "rotate_requires_integer_quarter_turns", False
            return True, proposal.reason, False
        if action == "detach":
            return True, proposal.reason, False
        if target is None:
            return False, "missing_target", False
        if target not in set(self.substrate.nodes):
            return False, "target_not_in_substrate", False
        if target == source:
            return False, "target_is_source", False
        if not self.substrate.edge_exists(source, target):
            return False, "target_not_adjacent", False
        target_occupied = target_cell_id is not None
        if action == "swap":
            return (True, proposal.reason, False) if target_occupied else (False, "swap_target_empty", False)
        if action == "crawl":
            return (True, proposal.reason, False) if not target_occupied else (False, "crawl_target_occupied", True)
        if action == "divide":
            return (True, proposal.reason, False) if not target_occupied else (False, "divide_target_occupied", True)
        if action == "adhere":
            if not target_occupied or target_cell_id is None:
                return False, "adhere_target_empty", False
            if _bond(proposal.actor_cell_id, target_cell_id) in self.adhesion_bonds:
                return False, "adhesion_bond_exists", False
            return True, proposal.reason, False
        if action == "exchange_signal":
            if not target_occupied:
                return False, "signal_target_empty", False
            channel = str(proposal.parameters.get("channel", ""))
            amount = float(proposal.parameters.get("amount", 0.0))
            if not SIGNAL_CHANNEL_RE.match(channel):
                return False, "invalid_signal_channel", False
            if amount <= 0.0:
                return False, "signal_amount_must_be_positive", False
            return True, proposal.reason, False
        return False, f"unknown_action:{action}", False

    def apply_action(self, proposal: ActionProposal) -> ActionOutcome:
        source, target, target_cell_id = self._target_context(proposal)
        counts_before = self._counts()
        legal, reason, collision = self.is_legal_action(proposal)
        accepted = False
        born_cell_id: int | None = None
        died_cell_id: int | None = None
        detached_cell_id: int | None = None
        adhesion_bond_delta = 0
        signal_delta: dict[str, Any] = {}
        if legal:
            if proposal.action == "wait":
                accepted = True
            elif proposal.action == "swap" and source is not None and target is not None and target_cell_id is not None:
                self.occupancy[source] = target_cell_id
                self.occupancy[target] = proposal.actor_cell_id
                accepted = True
            elif proposal.action == "crawl" and source is not None and target is not None:
                del self.occupancy[source]
                self.occupancy[target] = proposal.actor_cell_id
                accepted = True
            elif proposal.action == "rotate":
                self._rotate_actor_polarity(proposal.actor_cell_id, int(proposal.parameters.get("quarter_turns", 1)))
                accepted = True
            elif proposal.action == "divide" and source is not None and target is not None:
                born_cell_id = self._divide_actor(proposal.actor_cell_id, target, proposal.parameters)
                accepted = True
            elif proposal.action == "die":
                died_cell_id = self._die_cell(proposal.actor_cell_id)
                accepted = True
            elif proposal.action == "adhere" and target_cell_id is not None:
                self.adhesion_bonds.add(_bond(proposal.actor_cell_id, target_cell_id))
                adhesion_bond_delta = 1
                accepted = True
            elif proposal.action == "detach" and source is not None:
                detached_cell_id = proposal.actor_cell_id
                del self.occupancy[source]
                self.detached_cell_ids.add(proposal.actor_cell_id)
                adhesion_bond_delta = -self._remove_bonds_for(proposal.actor_cell_id)
                accepted = True
            elif proposal.action == "exchange_signal" and target_cell_id is not None:
                signal_delta = self._exchange_signal(proposal.actor_cell_id, target_cell_id, proposal.parameters)
                accepted = True
        energy_cost = float(ACTION_ENERGY_COSTS.get(proposal.action, 0.0)) if accepted else 0.0
        self.energy_spent += energy_cost
        self.step_index += 1
        counts_after = self._counts()
        outcome = ActionOutcome(
            step_index=self.step_index,
            action=proposal.action,
            actor_cell_id=proposal.actor_cell_id,
            source_position_before=source,
            target_position=target,
            actor_position_after=self.cell_position(proposal.actor_cell_id),
            target_cell_id_before=target_cell_id,
            accepted=accepted,
            legal=legal,
            reason=reason,
            collision=collision,
            energy_cost=energy_cost,
            conservation_mode=CONSERVATION_MODES.get(proposal.action, "unknown"),
            cell_count_before=counts_before["cell"],
            cell_count_after=counts_after["cell"],
            occupied_count_before=counts_before["occupied"],
            occupied_count_after=counts_after["occupied"],
            detached_count_before=counts_before["detached"],
            detached_count_after=counts_after["detached"],
            born_cell_id=born_cell_id,
            died_cell_id=died_cell_id,
            detached_cell_id=detached_cell_id,
            adhesion_bond_delta=adhesion_bond_delta,
            signal_delta=signal_delta,
        )
        self.assert_valid_state()
        self._append_trace_row(
            event_kind="action",
            proposal=proposal,
            outcome=outcome,
            counts_before=counts_before,
            counts_after=counts_after,
        )
        return outcome

    def _replace_cell_identity(self, cell_id: int, identity: Mapping[str, Any], value: Any | None = None) -> None:
        old = self.cells[int(cell_id)]
        new_value = old.value if value is None else value
        self.cells[int(cell_id)] = CellState(cell_id=old.cell_id, value=new_value, identity=dict(identity))

    def _rotate_actor_polarity(self, cell_id: int, quarter_turns: int) -> None:
        identity = dict(self.cells[int(cell_id)].identity)
        vector = [float(identity["polarity"][0]), float(identity["polarity"][1])]
        turns = int(quarter_turns) % 4
        for _ in range(turns):
            vector = [-vector[1], vector[0]]
        identity["polarity"] = [round(vector[0], 12), round(vector[1], 12)]
        self._replace_cell_identity(cell_id, identity)

    def _divide_actor(self, actor_cell_id: int, target: Position, parameters: Mapping[str, Any]) -> int:
        if self.next_cell_id is None:
            self.next_cell_id = 0 if not self.cells else max(self.cells) + 1
        child_id = int(self.next_cell_id)
        self.next_cell_id += 1
        parent = self.cells[int(actor_cell_id)]
        child_identity = dict(parent.identity)
        child_identity.update(dict(parameters.get("child_identity_updates", {})))
        child_value = parameters.get("child_value", parent.value)
        if "scalar_value" in child_identity:
            child_identity["scalar_value"] = child_value
        self.cells[child_id] = CellState(cell_id=child_id, value=child_value, identity=child_identity)
        self.occupancy[_as_position(target)] = child_id
        return child_id

    def _die_cell(self, cell_id: int) -> int:
        cell_id = int(cell_id)
        position = self.cell_position(cell_id)
        if position is not None:
            del self.occupancy[position]
        self.detached_cell_ids.discard(cell_id)
        self._remove_bonds_for(cell_id)
        del self.cells[cell_id]
        return cell_id

    def _remove_bonds_for(self, cell_id: int) -> int:
        cell_id = int(cell_id)
        before = len(self.adhesion_bonds)
        self.adhesion_bonds = {bond for bond in self.adhesion_bonds if cell_id not in bond}
        return before - len(self.adhesion_bonds)

    def _exchange_signal(self, actor_cell_id: int, target_cell_id: int, parameters: Mapping[str, Any]) -> dict[str, Any]:
        channel = str(parameters.get("channel", "morphogen"))
        amount = float(parameters.get("amount", 1.0))
        target = self.cells[int(target_cell_id)]
        target_identity = dict(target.identity)
        signal_state = dict(target_identity.get(LOCAL_SIGNAL_KEY, {}))
        channel_state = dict(signal_state.get(channel, {}))
        channel_state["received"] = float(channel_state.get("received", 0.0)) + amount
        channel_state["count"] = int(channel_state.get("count", 0)) + 1
        channel_state["last_from"] = int(actor_cell_id)
        signal_state[channel] = channel_state
        target_identity[LOCAL_SIGNAL_KEY] = signal_state
        self._replace_cell_identity(target_cell_id, target_identity)

        actor = self.cells[int(actor_cell_id)]
        actor_identity = dict(actor.identity)
        emitted = dict(actor_identity.get(LOCAL_SIGNAL_EMITTED_KEY, {}))
        emitted[channel] = float(emitted.get(channel, 0.0)) + amount
        actor_identity[LOCAL_SIGNAL_EMITTED_KEY] = emitted
        self._replace_cell_identity(actor_cell_id, actor_identity)

        return {
            "channel": channel,
            "amount": amount,
            "sourceCellId": int(actor_cell_id),
            "targetCellId": int(target_cell_id),
        }

    def _append_trace_row(
        self,
        *,
        event_kind: str,
        proposal: ActionProposal,
        outcome: ActionOutcome | None,
        counts_before: Mapping[str, int],
        counts_after: Mapping[str, int],
    ) -> None:
        actor_position = self.cell_position(proposal.actor_cell_id)
        row = {
            "schema_version": ACTION_TRACE_SCHEMA_VERSION,
            "condition_id": self.condition_id,
            "step_index": int(self.step_index),
            "event_kind": event_kind,
            "substrate_type": self.substrate.substrate_type,
            "node_count": len(self.substrate.nodes),
            "edge_count": len(self.substrate.edges()),
            "occupied_count": int(counts_after["occupied"]),
            "cell_count": int(counts_after["cell"]),
            "detached_count": int(counts_after["detached"]),
            "action": proposal.action,
            "actor_cell_id": None if proposal.actor_cell_id < 0 else int(proposal.actor_cell_id),
            "actor_position": None if actor_position is None else list(actor_position),
            "source_position_before": None if outcome is None or outcome.source_position_before is None else list(outcome.source_position_before),
            "target_position": None if proposal.target_position is None else list(proposal.target_position),
            "actor_position_after": None if outcome is None or outcome.actor_position_after is None else list(outcome.actor_position_after),
            "target_cell_id_before": None if outcome is None else outcome.target_cell_id_before,
            "accepted": None if outcome is None else bool(outcome.accepted),
            "legal": None if outcome is None else bool(outcome.legal),
            "reason": proposal.reason if outcome is None else outcome.reason,
            "collision": False if outcome is None else bool(outcome.collision),
            "energy_cost": 0.0 if outcome is None else float(outcome.energy_cost),
            "energy_spent": float(self.energy_spent),
            "conservation_mode": "initial" if outcome is None else outcome.conservation_mode,
            "cell_count_before": int(counts_before["cell"]),
            "cell_count_after": int(counts_after["cell"]),
            "cell_delta": 0 if outcome is None else outcome.cell_delta,
            "occupied_count_before": int(counts_before["occupied"]),
            "occupied_count_after": int(counts_after["occupied"]),
            "occupied_delta": 0 if outcome is None else outcome.occupied_delta,
            "detached_count_before": int(counts_before["detached"]),
            "detached_count_after": int(counts_after["detached"]),
            "detached_delta": 0 if outcome is None else outcome.detached_delta,
            "born_cell_id": None if outcome is None else outcome.born_cell_id,
            "died_cell_id": None if outcome is None else outcome.died_cell_id,
            "detached_cell_id": None if outcome is None else outcome.detached_cell_id,
            "adhesion_bond_delta": 0 if outcome is None else outcome.adhesion_bond_delta,
            "adhesion_bonds_json": canonical_json(sorted(self.adhesion_bonds)),
            "signal_delta_json": canonical_json({} if outcome is None else outcome.signal_delta),
            "proposal_parameters_json": canonical_json(proposal.parameters),
            "occupancy_hash": self.occupancy_hash(),
            "cell_identity_hash": self.cell_identity_hash(),
        }
        self.trace_rows.append(row)


def _valid_2d_vector(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    )


def local_action_policy_payload(observation: ActionObservation) -> dict[str, Any]:
    """Return actor-local S04 action context without global maps or traces."""

    payload = {
        "schemaVersion": LOCAL_ACTION_OBSERVATION_VERSION,
        "substrateType": observation.substrate_type,
        "boundary": observation.boundary,
        "actor": {
            "cellId": int(observation.actor.cell_id),
            "value": observation.actor.value,
            "identity": _visible_identity(observation.actor.identity),
            "localSignals": _json_ready(observation.actor_signal_state),
        },
        "actorPosition": list(observation.actor_position),
        "localDegree": len(observation.neighbors),
        "neighbors": [
            {
                "position": list(neighbor.position),
                "relativeOffset": None if neighbor.relative_offset is None else list(neighbor.relative_offset),
                "occupied": bool(neighbor.occupied),
                "adheredToActor": bool(neighbor.adhered_to_actor),
                "cell": None
                if neighbor.cell is None
                else {
                    "cellId": int(neighbor.cell.cell_id),
                    "value": neighbor.cell.value,
                    "identity": _visible_identity(neighbor.cell.identity),
                },
            }
            for neighbor in observation.neighbors
        ],
    }
    return _json_ready(payload)


def _recursive_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_recursive_keys(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            keys.update(_recursive_keys(item))
    return keys


def audit_local_action_payload(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    keys = _recursive_keys(payload)
    leaked = sorted(PROHIBITED_LOCAL_ACTION_KEYS & keys)
    if leaked:
        errors.append(f"payload exposes prohibited keys: {leaked}")
    if payload.get("schemaVersion") != LOCAL_ACTION_OBSERVATION_VERSION:
        errors.append("unexpected local action observation schema version")
    if "neighbors" not in payload:
        errors.append("payload is missing neighbors")
    return errors


ACTION_TRACE_REQUIRED_COLUMNS = {
    "schema_version",
    "condition_id",
    "step_index",
    "event_kind",
    "substrate_type",
    "node_count",
    "edge_count",
    "occupied_count",
    "cell_count",
    "detached_count",
    "action",
    "actor_cell_id",
    "actor_position",
    "source_position_before",
    "target_position",
    "actor_position_after",
    "target_cell_id_before",
    "accepted",
    "legal",
    "reason",
    "collision",
    "energy_cost",
    "energy_spent",
    "conservation_mode",
    "cell_count_before",
    "cell_count_after",
    "cell_delta",
    "occupied_count_before",
    "occupied_count_after",
    "occupied_delta",
    "detached_count_before",
    "detached_count_after",
    "detached_delta",
    "born_cell_id",
    "died_cell_id",
    "detached_cell_id",
    "adhesion_bond_delta",
    "adhesion_bonds_json",
    "signal_delta_json",
    "proposal_parameters_json",
    "occupancy_hash",
    "cell_identity_hash",
}


def validate_action_trace_schema(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not rows:
        return ["trace has no rows"]
    for index, row in enumerate(rows):
        missing = sorted(ACTION_TRACE_REQUIRED_COLUMNS - set(row))
        if missing:
            errors.append(f"row {index} missing columns: {missing}")
        if row.get("schema_version") != ACTION_TRACE_SCHEMA_VERSION:
            errors.append(f"row {index} has unexpected schema_version {row.get('schema_version')!r}")
        if not isinstance(row.get("step_index"), int):
            errors.append(f"row {index} step_index is not int")
        if row.get("event_kind") == "action" and row.get("accepted"):
            cell_delta = int(row.get("cell_count_after", 0)) - int(row.get("cell_count_before", 0))
            occupied_delta = int(row.get("occupied_count_after", 0)) - int(row.get("occupied_count_before", 0))
            detached_delta = int(row.get("detached_count_after", 0)) - int(row.get("detached_count_before", 0))
            if cell_delta != row.get("cell_delta"):
                errors.append(f"row {index} has inconsistent cell_delta")
            if occupied_delta != row.get("occupied_delta"):
                errors.append(f"row {index} has inconsistent occupied_delta")
            if detached_delta != row.get("detached_delta"):
                errors.append(f"row {index} has inconsistent detached_delta")
    return errors


def action_semantics_spec() -> dict[str, Any]:
    rows = []
    target_rules = {
        "wait": "no target",
        "swap": "adjacent occupied target",
        "crawl": "adjacent empty target",
        "rotate": "occupied actor with 2D polarity; integer quarter turns",
        "divide": "adjacent empty target; creates one child cell",
        "die": "live occupied or detached actor; removes cell",
        "adhere": "adjacent occupied target; creates one undirected adhesion bond",
        "detach": "occupied actor; removes actor from substrate into detached pool",
        "exchange_signal": "adjacent occupied target; positive amount and valid local channel",
    }
    collision_rules = {
        "crawl": "occupied target is rejected and flagged collision=true",
        "divide": "occupied target is rejected and flagged collision=true",
    }
    for action in SUPPORTED_ACTIONS:
        rows.append(
            {
                "action": action,
                "energyCost": ACTION_ENERGY_COSTS[action],
                "conservationMode": CONSERVATION_MODES[action],
                "targetRule": target_rules[action],
                "collisionHandling": collision_rules.get(action, "illegal proposals are rejected without state mutation"),
            }
        )
    return {
        "schemaVersion": ACTION_SCHEMA_VERSION,
        "traceSchemaVersion": ACTION_TRACE_SCHEMA_VERSION,
        "localObservationVersion": LOCAL_ACTION_OBSERVATION_VERSION,
        "supportedActions": list(SUPPORTED_ACTIONS),
        "energyCosts": dict(ACTION_ENERGY_COSTS),
        "conservationModes": dict(CONSERVATION_MODES),
        "actions": rows,
        "traceContract": sorted(ACTION_TRACE_REQUIRED_COLUMNS),
        "localPolicyBoundary": {
            "allowed": "actor cell, actor position, local neighbors, local adhesion flags, and actor-local signal state",
            "prohibitedKeys": sorted(PROHIBITED_LOCAL_ACTION_KEYS),
            "wholeWorldMapsExposed": False,
        },
    }


def make_validation_cell(
    cell_id: int,
    value: int,
    *,
    organ_type: str = "axis",
    polarity: Sequence[float] = (1.0, 0.0),
    adhesion_type: str = "adhesion_a",
) -> CellState:
    return CellState(
        cell_id=int(cell_id),
        value=int(value),
        identity={
            "scalar_value": int(value),
            "ap_coordinate": float(value - 1) / 4.0,
            "organ_type": organ_type,
            "polarity": [float(polarity[0]), float(polarity[1])],
            "adhesion_type": adhesion_type,
            "target_neighbor_preferences": [],
        },
    )


def build_s04_validation_world() -> ActionWorld:
    substrate = Substrate.square_grid(3, 3)
    cells = {
        (0, 0): make_validation_cell(0, 1, organ_type="axis", polarity=(1.0, 0.0), adhesion_type="adhesion_a"),
        (1, 0): make_validation_cell(1, 2, organ_type="core", polarity=(0.0, 1.0), adhesion_type="adhesion_a"),
        (0, 1): make_validation_cell(2, 3, organ_type="boundary", polarity=(-1.0, 0.0), adhesion_type="adhesion_boundary"),
        (2, 0): make_validation_cell(3, 4, organ_type="appendage", polarity=(0.0, -1.0), adhesion_type="adhesion_b"),
    }
    return ActionWorld.from_position_cells(substrate, cells, condition_id="e05_s04_validation")


def standard_action_sequence() -> tuple[ActionProposal, ...]:
    return (
        ActionProposal.swap(0, (1, 0), "swap_adjacent_occupied"),
        ActionProposal.crawl(0, (0, 0), "crawl_collision_occupied_target"),
        ActionProposal.crawl(0, (1, 1), "crawl_adjacent_empty"),
        ActionProposal.rotate(0, quarter_turns=1, reason="rotate_quarter_turn"),
        ActionProposal.divide(0, (2, 1), child_value=5, reason="divide_into_empty_neighbor"),
        ActionProposal.adhere(0, (2, 1), "adhere_to_child"),
        ActionProposal.exchange_signal(0, (2, 1), channel="morphogen", amount=0.75, reason="local_signal_to_child"),
        ActionProposal.detach(4, "detach_child_from_substrate"),
        ActionProposal.die(4, "die_detached_child"),
    )


def run_standard_action_sequence() -> tuple[ActionWorld, tuple[ActionOutcome, ...], list[dict[str, Any]]]:
    world = build_s04_validation_world()
    outcomes: list[ActionOutcome] = []
    validation_rows: list[dict[str, Any]] = []
    for proposal in standard_action_sequence():
        before_hash = world.occupancy_hash()
        outcome = world.apply_action(proposal)
        outcomes.append(outcome)
        validation_rows.append(
            {
                "check_id": f"{outcome.step_index}_{proposal.reason}",
                "action": outcome.action,
                "accepted": outcome.accepted,
                "legal": outcome.legal,
                "collision": outcome.collision,
                "reason": outcome.reason,
                "state_valid_after": world.validate_state() == [],
                "occupancy_changed": before_hash != world.occupancy_hash(),
                "cell_delta": outcome.cell_delta,
                "occupied_delta": outcome.occupied_delta,
                "detached_delta": outcome.detached_delta,
                "energy_cost": outcome.energy_cost,
                "conservation_mode": outcome.conservation_mode,
            }
        )
    return world, tuple(outcomes), validation_rows
