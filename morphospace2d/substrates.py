"""Substrate and local-move simulator abstractions for E05 S01.

The module keeps S01 deliberately small: substrates are undirected graphs with
optional geometry metadata, and movement is limited to local swaps or crawls to
adjacent empty positions. One-dimensional rows, square grids, hex grids, and
irregular graphs are all represented through the same graph contract.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


Position = tuple[int, ...]
SUBSTRATE_SCHEMA_VERSION = "e05_s01_substrate.v1"
TRACE_SCHEMA_VERSION = "e05_s01_trace.v1"


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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _stable_hash(value: Any) -> str:
    payload = json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Substrate:
    """Undirected graph substrate with optional coordinate geometry."""

    substrate_type: str
    nodes: tuple[Position, ...]
    adjacency: Mapping[Position, tuple[Position, ...]]
    boundary: str = "open"
    geometry: Mapping[Position, Mapping[str, Any]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SUBSTRATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        nodes = tuple(_as_position(node) for node in self.nodes)
        node_set = set(nodes)
        normalized: dict[Position, tuple[Position, ...]] = {}
        for node in nodes:
            neighbors = tuple(sorted(set(_as_position(neighbor) for neighbor in self.adjacency.get(node, ()))))
            normalized[node] = neighbors
        for node, neighbors in tuple(normalized.items()):
            for neighbor in neighbors:
                if neighbor not in node_set:
                    raise ValueError(f"neighbor {neighbor} is not a substrate node")
                if neighbor == node:
                    raise ValueError(f"self-loop at {node}")
                back_neighbors = set(normalized.get(neighbor, ()))
                if node not in back_neighbors:
                    normalized[neighbor] = tuple(sorted(back_neighbors | {node}))
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "adjacency", normalized)
        self.assert_valid()

    @classmethod
    def row(cls, length: int, *, periodic: bool = False) -> "Substrate":
        if length < 1:
            raise ValueError("row length must be positive")
        nodes = tuple((index,) for index in range(length))
        adjacency: dict[Position, list[Position]] = {node: [] for node in nodes}
        for index in range(length - 1):
            left = (index,)
            right = (index + 1,)
            adjacency[left].append(right)
            adjacency[right].append(left)
        if periodic and length > 2:
            adjacency[(0,)].append((length - 1,))
            adjacency[(length - 1,)].append((0,))
        geometry = {node: {"coordinate": node, "axis": "x"} for node in nodes}
        return cls(
            "row_1d",
            nodes,
            {node: tuple(neighbors) for node, neighbors in adjacency.items()},
            boundary="periodic" if periodic else "open",
            geometry=geometry,
            metadata={"length": length, "periodic": periodic},
        )

    @classmethod
    def square_grid(cls, width: int, height: int, *, periodic: bool = False) -> "Substrate":
        if width < 1 or height < 1:
            raise ValueError("square grid dimensions must be positive")
        nodes = tuple((x, y) for y in range(height) for x in range(width))
        node_set = set(nodes)
        adjacency: dict[Position, list[Position]] = {node: [] for node in nodes}
        for x, y in nodes:
            candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            if periodic:
                candidates = [((cx % width), (cy % height)) for cx, cy in candidates]
            for candidate in candidates:
                if candidate in node_set and candidate != (x, y):
                    adjacency[(x, y)].append(candidate)
        geometry = {node: {"coordinate": node, "lattice": "square"} for node in nodes}
        return cls(
            "square_grid_2d",
            nodes,
            {node: tuple(neighbors) for node, neighbors in adjacency.items()},
            boundary="periodic" if periodic else "open",
            geometry=geometry,
            metadata={"width": width, "height": height, "periodic": periodic},
        )

    @classmethod
    def hex_grid(cls, width: int, height: int) -> "Substrate":
        """Create an open axial-coordinate hex grid."""

        if width < 1 or height < 1:
            raise ValueError("hex grid dimensions must be positive")
        nodes = tuple((q, r) for r in range(height) for q in range(width))
        node_set = set(nodes)
        directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))
        adjacency: dict[Position, list[Position]] = {node: [] for node in nodes}
        for q, r in nodes:
            for dq, dr in directions:
                candidate = (q + dq, r + dr)
                if candidate in node_set:
                    adjacency[(q, r)].append(candidate)
        geometry = {node: {"coordinate": node, "lattice": "hex_axial"} for node in nodes}
        return cls(
            "hex_grid_2d",
            nodes,
            {node: tuple(neighbors) for node, neighbors in adjacency.items()},
            geometry=geometry,
            metadata={"width": width, "height": height, "coordinate_system": "axial"},
        )

    @classmethod
    def irregular_graph(
        cls,
        edges: Iterable[tuple[int | Sequence[int], int | Sequence[int]]],
        *,
        nodes: Iterable[int | Sequence[int]] | None = None,
    ) -> "Substrate":
        edge_positions = [(_as_position(left), _as_position(right)) for left, right in edges]
        node_set = set(_as_position(node) for node in nodes) if nodes is not None else set()
        for left, right in edge_positions:
            node_set.add(left)
            node_set.add(right)
        if not node_set:
            raise ValueError("irregular graph must have at least one node")
        ordered_nodes = tuple(sorted(node_set))
        adjacency: dict[Position, list[Position]] = {node: [] for node in ordered_nodes}
        for left, right in edge_positions:
            if left == right:
                raise ValueError(f"self-loop at {left}")
            adjacency[left].append(right)
            adjacency[right].append(left)
        geometry = {node: {"coordinate": node, "lattice": "irregular"} for node in ordered_nodes}
        return cls(
            "irregular_graph",
            ordered_nodes,
            {node: tuple(neighbors) for node, neighbors in adjacency.items()},
            geometry=geometry,
            metadata={"edge_count": len(edge_positions)},
        )

    @classmethod
    def lattice3d(cls, width: int, height: int, depth: int) -> "Substrate":
        if width < 1 or height < 1 or depth < 1:
            raise ValueError("3D lattice dimensions must be positive")
        nodes = tuple((x, y, z) for z in range(depth) for y in range(height) for x in range(width))
        node_set = set(nodes)
        adjacency: dict[Position, list[Position]] = {node: [] for node in nodes}
        for x, y, z in nodes:
            candidates = [
                (x - 1, y, z),
                (x + 1, y, z),
                (x, y - 1, z),
                (x, y + 1, z),
                (x, y, z - 1),
                (x, y, z + 1),
            ]
            adjacency[(x, y, z)].extend(candidate for candidate in candidates if candidate in node_set)
        geometry = {node: {"coordinate": node, "lattice": "cubic"} for node in nodes}
        return cls(
            "lattice_3d",
            nodes,
            {node: tuple(neighbors) for node, neighbors in adjacency.items()},
            geometry=geometry,
            metadata={"width": width, "height": height, "depth": depth},
        )

    def neighbors(self, position: Position) -> tuple[Position, ...]:
        position = _as_position(position)
        if position not in self.adjacency:
            raise KeyError(f"unknown substrate position: {position}")
        return self.adjacency[position]

    def edge_exists(self, left: Position, right: Position) -> bool:
        left = _as_position(left)
        right = _as_position(right)
        return right in set(self.adjacency.get(left, ()))

    def edges(self) -> tuple[tuple[Position, Position], ...]:
        edge_set: set[tuple[Position, Position]] = set()
        for left, neighbors in self.adjacency.items():
            for right in neighbors:
                edge_set.add(tuple(sorted((left, right))))
        return tuple(sorted(edge_set))

    def relative_offset(self, source: Position, target: Position) -> Position | None:
        source = _as_position(source)
        target = _as_position(target)
        if len(source) != len(target):
            return None
        return tuple(target[index] - source[index] for index in range(len(source)))

    def validate(self) -> list[str]:
        errors: list[str] = []
        node_set = set(self.nodes)
        if len(node_set) != len(self.nodes):
            errors.append("duplicate nodes")
        if set(self.adjacency) != node_set:
            errors.append("adjacency keys do not match nodes")
        for node, neighbors in self.adjacency.items():
            for neighbor in neighbors:
                if neighbor not in node_set:
                    errors.append(f"neighbor {neighbor} is not a node")
                if neighbor == node:
                    errors.append(f"self-loop at {node}")
                if node not in self.adjacency.get(neighbor, ()):
                    errors.append(f"asymmetric edge {node}->{neighbor}")
        return errors

    def assert_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("; ".join(errors))

    def to_record(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "substrateType": self.substrate_type,
            "boundary": self.boundary,
            "nodeCount": len(self.nodes),
            "edgeCount": len(self.edges()),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CellState:
    """Identity-bearing cell record. S01 keeps identity payloads opaque."""

    cell_id: int
    value: Any
    identity: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {"cellId": self.cell_id, "value": self.value, "identity": dict(self.identity)}


@dataclass(frozen=True)
class NeighborView:
    position: Position
    relative_offset: Position | None
    occupied: bool
    cell: CellState | None = None


@dataclass(frozen=True)
class LocalObservation:
    """Local neighborhood exposed to a policy invocation."""

    substrate_type: str
    actor: CellState
    actor_position: Position
    neighbors: tuple[NeighborView, ...]
    boundary: str

    @property
    def occupied_neighbors(self) -> tuple[NeighborView, ...]:
        return tuple(neighbor for neighbor in self.neighbors if neighbor.occupied)

    @property
    def empty_neighbors(self) -> tuple[NeighborView, ...]:
        return tuple(neighbor for neighbor in self.neighbors if not neighbor.occupied)


@dataclass(frozen=True)
class MoveProposal:
    action: str
    actor_cell_id: int
    target_position: Position | None = None
    reason: str = "manual"

    @classmethod
    def wait(cls, actor_cell_id: int, reason: str = "wait") -> "MoveProposal":
        return cls("wait", int(actor_cell_id), None, reason)

    @classmethod
    def swap(cls, actor_cell_id: int, target_position: Position, reason: str = "swap") -> "MoveProposal":
        return cls("swap", int(actor_cell_id), _as_position(target_position), reason)

    @classmethod
    def crawl(cls, actor_cell_id: int, target_position: Position, reason: str = "crawl") -> "MoveProposal":
        return cls("crawl", int(actor_cell_id), _as_position(target_position), reason)


@dataclass(frozen=True)
class MoveStepOutcome:
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


class LocalMovePolicy(Protocol):
    policy_id: str

    def propose(self, observation: LocalObservation, rng: np.random.Generator) -> MoveProposal:
        """Return a local movement proposal from the supplied observation."""


class GreedyLowerValueSwapPolicy:
    """Swap with the lowest-valued occupied neighbor when it is lower than actor."""

    policy_id = "greedy_lower_value_swap"

    def propose(self, observation: LocalObservation, rng: np.random.Generator) -> MoveProposal:
        candidates = [
            neighbor
            for neighbor in observation.occupied_neighbors
            if neighbor.cell is not None and observation.actor.value > neighbor.cell.value
        ]
        if not candidates:
            return MoveProposal.wait(observation.actor.cell_id, "no_lower_neighbor")
        candidates.sort(key=lambda neighbor: (neighbor.cell.value if neighbor.cell is not None else 0, neighbor.position))
        return MoveProposal.swap(observation.actor.cell_id, candidates[0].position, "swap_with_lower_neighbor")


class ClockwiseCrawlPolicy:
    """Move into the first empty local neighbor in deterministic neighbor order."""

    policy_id = "first_empty_neighbor_crawl"

    def propose(self, observation: LocalObservation, rng: np.random.Generator) -> MoveProposal:
        if not observation.empty_neighbors:
            return MoveProposal.wait(observation.actor.cell_id, "no_empty_neighbor")
        target = sorted(observation.empty_neighbors, key=lambda neighbor: neighbor.position)[0].position
        return MoveProposal.crawl(observation.actor.cell_id, target, "crawl_to_empty_neighbor")


@dataclass
class MorphologyWorld:
    """Mutable cell occupancy state on a substrate."""

    substrate: Substrate
    cells: Mapping[int, CellState]
    occupancy: Mapping[Position, int]
    condition_id: str = "manual"
    trace_rows: list[dict[str, Any]] = field(default_factory=list)
    step_index: int = 0

    def __post_init__(self) -> None:
        cells = {int(cell_id): cell for cell_id, cell in self.cells.items()}
        occupancy = {_as_position(position): int(cell_id) for position, cell_id in self.occupancy.items()}
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "occupancy", occupancy)
        self.assert_valid_occupancy()
        if not self.trace_rows:
            self._append_trace_row(
                event_kind="initial",
                proposal=MoveProposal.wait(next(iter(cells), -1), "initial"),
                outcome=None,
            )

    @classmethod
    def from_position_values(
        cls,
        substrate: Substrate,
        position_values: Mapping[Position, Any],
        *,
        condition_id: str = "manual",
    ) -> "MorphologyWorld":
        cells: dict[int, CellState] = {}
        occupancy: dict[Position, int] = {}
        normalized_values = {_as_position(position): value for position, value in position_values.items()}
        for cell_id, position in enumerate(sorted(normalized_values)):
            cells[cell_id] = CellState(cell_id=cell_id, value=normalized_values[position])
            occupancy[position] = cell_id
        return cls(substrate=substrate, cells=cells, occupancy=occupancy, condition_id=condition_id)

    def clone(self) -> "MorphologyWorld":
        return MorphologyWorld(
            substrate=self.substrate,
            cells=dict(self.cells),
            occupancy=dict(self.occupancy),
            condition_id=self.condition_id,
            trace_rows=list(self.trace_rows),
            step_index=self.step_index,
        )

    def occupied_count(self) -> int:
        return len(self.occupancy)

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
        return _stable_hash(rows)

    def validate_occupancy(self) -> list[str]:
        errors: list[str] = []
        node_set = set(self.substrate.nodes)
        if any(position not in node_set for position in self.occupancy):
            errors.append("occupancy references non-substrate position")
        if any(cell_id not in self.cells for cell_id in self.occupancy.values()):
            errors.append("occupancy references unknown cell")
        counts = self.cell_id_counter()
        duplicated = sorted(cell_id for cell_id, count in counts.items() if count != 1)
        if duplicated:
            errors.append(f"cell ids occupy multiple positions or invalid counts: {duplicated}")
        missing = sorted(set(self.cells) - set(self.occupancy.values()))
        if missing:
            errors.append(f"cells missing from occupancy: {missing}")
        if len(self.cells) > len(self.substrate.nodes):
            errors.append("more cells than substrate nodes")
        if len(missing) + len(self.occupancy) != len(self.cells):
            errors.append("cell/occupancy cardinality mismatch")
        return errors

    def assert_valid_occupancy(self) -> None:
        errors = self.validate_occupancy()
        if errors:
            raise ValueError("; ".join(errors))

    def local_observation(self, cell_id: int) -> LocalObservation:
        position = self.cell_position(cell_id)
        if position is None:
            raise KeyError(f"cell {cell_id} is not currently occupying a substrate node")
        neighbors: list[NeighborView] = []
        for neighbor_position in self.substrate.neighbors(position):
            neighbor_cell_id = self.occupancy.get(neighbor_position)
            neighbors.append(
                NeighborView(
                    position=neighbor_position,
                    relative_offset=self.substrate.relative_offset(position, neighbor_position),
                    occupied=neighbor_cell_id is not None,
                    cell=None if neighbor_cell_id is None else self.cells[neighbor_cell_id],
                )
            )
        return LocalObservation(
            substrate_type=self.substrate.substrate_type,
            actor=self.cells[int(cell_id)],
            actor_position=position,
            neighbors=tuple(neighbors),
            boundary=self.substrate.boundary,
        )

    def is_legal_move(self, proposal: MoveProposal) -> tuple[bool, str]:
        if proposal.action == "wait":
            return True, proposal.reason
        source = self.cell_position(proposal.actor_cell_id)
        if source is None:
            return False, "actor_not_occupied"
        if proposal.target_position is None:
            return False, "missing_target"
        target = _as_position(proposal.target_position)
        if target not in set(self.substrate.nodes):
            return False, "target_not_in_substrate"
        if target == source:
            return False, "target_is_source"
        if not self.substrate.edge_exists(source, target):
            return False, "target_not_adjacent"
        target_occupied = target in self.occupancy
        if proposal.action == "swap":
            return (True, proposal.reason) if target_occupied else (False, "swap_target_empty")
        if proposal.action == "crawl":
            return (True, proposal.reason) if not target_occupied else (False, "crawl_target_occupied")
        return False, f"unknown_action:{proposal.action}"

    def apply_move(self, proposal: MoveProposal) -> MoveStepOutcome:
        source = self.cell_position(proposal.actor_cell_id)
        target = None if proposal.target_position is None else _as_position(proposal.target_position)
        target_cell_id = None if target is None else self.occupancy.get(target)
        legal, reason = self.is_legal_move(proposal)
        accepted = False
        if legal and proposal.action == "swap" and source is not None and target is not None and target_cell_id is not None:
            self.occupancy[source] = target_cell_id
            self.occupancy[target] = proposal.actor_cell_id
            accepted = True
        elif legal and proposal.action == "crawl" and source is not None and target is not None:
            del self.occupancy[source]
            self.occupancy[target] = proposal.actor_cell_id
            accepted = True
        elif legal and proposal.action == "wait":
            accepted = True
        self.step_index += 1
        outcome = MoveStepOutcome(
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
        )
        self.assert_valid_occupancy()
        self._append_trace_row(event_kind="move", proposal=proposal, outcome=outcome)
        return outcome

    def _append_trace_row(
        self,
        *,
        event_kind: str,
        proposal: MoveProposal,
        outcome: MoveStepOutcome | None,
    ) -> None:
        actor_position = self.cell_position(proposal.actor_cell_id)
        row = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "condition_id": self.condition_id,
            "step_index": int(self.step_index),
            "event_kind": event_kind,
            "substrate_type": self.substrate.substrate_type,
            "node_count": len(self.substrate.nodes),
            "edge_count": len(self.substrate.edges()),
            "occupied_count": self.occupied_count(),
            "cell_count": len(self.cells),
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
            "occupancy_hash": self.occupancy_hash(),
            "cell_identity_hash": self.cell_identity_hash(),
        }
        self.trace_rows.append(row)


def run_local_dynamics(
    world: MorphologyWorld,
    policy: LocalMovePolicy,
    *,
    steps: int,
    seed: int = 0,
    actor_order: Sequence[int] | None = None,
) -> list[MoveStepOutcome]:
    """Run a deterministic round-robin local policy simulation."""

    rng = np.random.default_rng(seed)
    outcomes: list[MoveStepOutcome] = []
    if actor_order is None:
        actor_order = tuple(sorted(world.cells))
    if not actor_order:
        return outcomes
    for step in range(int(steps)):
        actor_cell_id = int(actor_order[step % len(actor_order)])
        if world.cell_position(actor_cell_id) is None:
            continue
        observation = world.local_observation(actor_cell_id)
        outcomes.append(world.apply_move(policy.propose(observation, rng)))
    return outcomes


REQUIRED_TRACE_COLUMNS = {
    "schema_version",
    "condition_id",
    "step_index",
    "event_kind",
    "substrate_type",
    "node_count",
    "edge_count",
    "occupied_count",
    "cell_count",
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
    "occupancy_hash",
    "cell_identity_hash",
}


def validate_trace_schema(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not rows:
        return ["trace has no rows"]
    for index, row in enumerate(rows):
        missing = sorted(REQUIRED_TRACE_COLUMNS - set(row))
        if missing:
            errors.append(f"row {index} missing columns: {missing}")
        if row.get("schema_version") != TRACE_SCHEMA_VERSION:
            errors.append(f"row {index} has unexpected schema_version {row.get('schema_version')!r}")
        if not isinstance(row.get("step_index"), int):
            errors.append(f"row {index} step_index is not int")
    return errors
