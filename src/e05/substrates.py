"""Substrate abstractions for E05 higher-dimensional morphogenesis tasks.

The module keeps the E03 local-policy contract intact: policies see an actor,
its bounded local neighborhood, and a small action set.  Substrates differ only
in their site graph and boundary rules; occupancy, swap, crawl/move, and local
observation semantics are shared.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


ACTIVE_STATUSES = frozenset({"active", "cellstatus.active"})
MOVABLE_TARGET_STATUSES = frozenset({"active", "cellstatus.active", "freeze", "frozen", "passive_frozen", "cellstatus.freeze"})


Coordinate = tuple[int, ...]


@dataclass(frozen=True)
class SubstrateCell:
    """One cell identity occupying a substrate site."""

    cell_id: str
    value: int | float | None = None
    label: str = "cell"
    status: str = "active"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def compact_dict(self) -> dict[str, Any]:
        return {
            "cell_id": str(self.cell_id),
            "value": self.value,
            "label": str(self.label),
            "status": str(self.status),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Site:
    """One position in a substrate graph."""

    site_id: int
    coordinate: Coordinate
    boundary: bool = False

    def compact_dict(self) -> dict[str, Any]:
        return {
            "site_id": int(self.site_id),
            "coordinate": list(self.coordinate),
            "boundary": bool(self.boundary),
        }


@dataclass(frozen=True)
class NeighborObservation:
    """Actor-local view of one neighboring site."""

    site_id: int
    coordinate: Coordinate
    direction: str
    occupied: bool
    cell_id: str | None
    value: int | float | None
    label: str | None
    status: str | None
    movable_target: bool

    def compact_dict(self) -> dict[str, Any]:
        return {
            "site_id": int(self.site_id),
            "coordinate": list(self.coordinate),
            "direction": str(self.direction),
            "occupied": bool(self.occupied),
            "cell_id": self.cell_id,
            "value": self.value,
            "label": self.label,
            "status": self.status,
            "movable_target": bool(self.movable_target),
        }


@dataclass(frozen=True)
class LocalSubstrateObservation:
    """Policy-visible bounded observation on any substrate."""

    substrate_kind: str
    actor_site_id: int
    actor_coordinate: Coordinate
    actor_cell: SubstrateCell
    neighbors: tuple[NeighborObservation, ...]
    boundary: bool
    action_set: tuple[str, ...] = ("swap", "move", "wait")

    @property
    def neighbor_site_ids(self) -> tuple[int, ...]:
        return tuple(item.site_id for item in self.neighbors)

    def compact_dict(self) -> dict[str, Any]:
        return {
            "substrate_kind": str(self.substrate_kind),
            "actor_site_id": int(self.actor_site_id),
            "actor_coordinate": list(self.actor_coordinate),
            "actor_cell": self.actor_cell.compact_dict(),
            "neighbors": [item.compact_dict() for item in self.neighbors],
            "boundary": bool(self.boundary),
            "action_set": list(self.action_set),
        }


@dataclass(frozen=True)
class SubstrateActionResult:
    """Result of one constrained local substrate action."""

    action_type: str
    source_site_id: int
    target_site_id: int | None
    allowed: bool
    reason: str
    state_changed: bool
    signature_before: tuple[tuple[Any, ...], ...]
    signature_after: tuple[tuple[Any, ...], ...]

    def compact_dict(self) -> dict[str, Any]:
        return {
            "action_type": str(self.action_type),
            "source_site_id": int(self.source_site_id),
            "target_site_id": None if self.target_site_id is None else int(self.target_site_id),
            "allowed": bool(self.allowed),
            "reason": str(self.reason),
            "state_changed": bool(self.state_changed),
            "signature_before": [list(row) for row in self.signature_before],
            "signature_after": [list(row) for row in self.signature_after],
        }


def _status_key(status: str) -> str:
    return str(status).strip().lower()


def _is_active_actor(cell: SubstrateCell | None) -> bool:
    return cell is not None and _status_key(cell.status) in ACTIVE_STATUSES


def _is_movable_target(cell: SubstrateCell | None) -> bool:
    return cell is not None and _status_key(cell.status) in MOVABLE_TARGET_STATUSES


def stable_substrate_sha256(payload: Any) -> str:
    """Hash a substrate payload using stable JSON serialization."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SubstrateState:
    """Mutable occupancy state over a fixed site graph."""

    def __init__(
        self,
        *,
        kind: str,
        sites: Sequence[Site],
        adjacency: Mapping[int, Sequence[int]],
        directions: Mapping[tuple[int, int], str] | None = None,
        boundary_condition: str = "open",
        dimensions: Mapping[str, Any] | None = None,
    ) -> None:
        self.kind = str(kind)
        self.boundary_condition = str(boundary_condition)
        self.dimensions = dict(dimensions or {})
        self.sites: tuple[Site, ...] = tuple(sites)
        self._site_by_id = {site.site_id: site for site in self.sites}
        if len(self._site_by_id) != len(self.sites):
            raise ValueError("site IDs must be unique")
        self._adjacency = _normalize_adjacency(adjacency, self._site_by_id.keys())
        self._directions = dict(directions or {})
        self._cells_by_site: dict[int, SubstrateCell | None] = {site.site_id: None for site in self.sites}
        self._site_by_cell_id: dict[str, int] = {}

    @property
    def site_ids(self) -> tuple[int, ...]:
        return tuple(site.site_id for site in self.sites)

    def copy_empty(self) -> "SubstrateState":
        return SubstrateState(
            kind=self.kind,
            sites=self.sites,
            adjacency=self._adjacency,
            directions=self._directions,
            boundary_condition=self.boundary_condition,
            dimensions=self.dimensions,
        )

    def graph_spec(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "boundary_condition": self.boundary_condition,
            "dimensions": dict(self.dimensions),
            "site_count": len(self.sites),
            "edge_count": sum(len(neighbors) for neighbors in self._adjacency.values()) // 2,
            "sites": [site.compact_dict() for site in self.sites],
            "adjacency": {str(site_id): list(self._adjacency[site_id]) for site_id in self.site_ids},
            "directions": {
                f"{source}->{target}": direction
                for (source, target), direction in sorted(self._directions.items())
            },
        }

    def graph_hash(self) -> str:
        return stable_substrate_sha256(self.graph_spec())

    def validate_graph(self) -> None:
        _normalize_adjacency(self._adjacency, self._site_by_id.keys())

    def site(self, site_id: int) -> Site:
        self._require_site(site_id)
        return self._site_by_id[int(site_id)]

    def coordinate(self, site_id: int) -> Coordinate:
        return self.site(site_id).coordinate

    def neighbors(self, site_id: int) -> tuple[int, ...]:
        self._require_site(site_id)
        return self._adjacency[int(site_id)]

    def neighbor_count(self, site_id: int) -> int:
        return len(self.neighbors(site_id))

    def is_adjacent(self, source_site_id: int, target_site_id: int) -> bool:
        self._require_site(source_site_id)
        self._require_site(target_site_id)
        return int(target_site_id) in self._adjacency[int(source_site_id)]

    def direction_between(self, source_site_id: int, target_site_id: int) -> str:
        self._require_site(source_site_id)
        self._require_site(target_site_id)
        return self._directions.get((int(source_site_id), int(target_site_id)), f"neighbor_{target_site_id}")

    def cell_at(self, site_id: int) -> SubstrateCell | None:
        self._require_site(site_id)
        return self._cells_by_site[int(site_id)]

    def site_of_cell(self, cell_id: str) -> int:
        key = str(cell_id)
        if key not in self._site_by_cell_id:
            raise KeyError(f"cell_id is not placed: {cell_id!r}")
        return self._site_by_cell_id[key]

    def occupied(self, site_id: int) -> bool:
        return self.cell_at(site_id) is not None

    def place_cell(self, site_id: int, cell: SubstrateCell) -> None:
        site_id = int(site_id)
        self._require_site(site_id)
        if self._cells_by_site[site_id] is not None:
            raise ValueError(f"site {site_id} is already occupied")
        key = str(cell.cell_id)
        if key in self._site_by_cell_id:
            raise ValueError(f"cell_id is already placed: {cell.cell_id!r}")
        self._cells_by_site[site_id] = cell
        self._site_by_cell_id[key] = site_id

    def remove_cell(self, site_id: int) -> SubstrateCell:
        site_id = int(site_id)
        self._require_site(site_id)
        cell = self._cells_by_site[site_id]
        if cell is None:
            raise ValueError(f"site {site_id} is empty")
        self._cells_by_site[site_id] = None
        self._site_by_cell_id.pop(str(cell.cell_id), None)
        return cell

    def fill_sites(self, cells: Sequence[SubstrateCell], site_ids: Sequence[int] | None = None) -> None:
        target_site_ids = tuple(self.site_ids if site_ids is None else site_ids)
        if len(cells) != len(target_site_ids):
            raise ValueError("cells and site_ids must have the same length")
        for site_id, cell in zip(target_site_ids, cells, strict=True):
            self.place_cell(site_id, cell)

    def can_swap(self, source_site_id: int, target_site_id: int) -> tuple[bool, str]:
        source_site_id = int(source_site_id)
        target_site_id = int(target_site_id)
        if source_site_id == target_site_id:
            return False, "source_equals_target"
        if source_site_id not in self._site_by_id:
            return False, "source_site_missing"
        if target_site_id not in self._site_by_id:
            return False, "target_site_missing"
        if not self.is_adjacent(source_site_id, target_site_id):
            return False, "target_not_adjacent"
        source_cell = self.cell_at(source_site_id)
        target_cell = self.cell_at(target_site_id)
        if not _is_active_actor(source_cell):
            return False, "actor_not_active_or_missing"
        if not _is_movable_target(target_cell):
            return False, "target_not_movable_or_missing"
        return True, "allowed"

    def can_move(self, source_site_id: int, target_site_id: int) -> tuple[bool, str]:
        source_site_id = int(source_site_id)
        target_site_id = int(target_site_id)
        if source_site_id == target_site_id:
            return False, "source_equals_target"
        if source_site_id not in self._site_by_id:
            return False, "source_site_missing"
        if target_site_id not in self._site_by_id:
            return False, "target_site_missing"
        if not self.is_adjacent(source_site_id, target_site_id):
            return False, "target_not_adjacent"
        source_cell = self.cell_at(source_site_id)
        if not _is_active_actor(source_cell):
            return False, "actor_not_active_or_missing"
        if self.cell_at(target_site_id) is not None:
            return False, "target_not_empty"
        return True, "allowed"

    def apply_action(self, action_type: str, source_site_id: int, target_site_id: int | None = None) -> SubstrateActionResult:
        action_type = str(action_type)
        source_site_id = int(source_site_id)
        target_site_id = None if target_site_id is None else int(target_site_id)
        signature_before = self.signature()
        allowed = False
        reason = "unsupported_action"
        if source_site_id not in self._site_by_id:
            reason = "source_site_missing"
        elif self.cell_at(source_site_id) is None:
            reason = "actor_missing"
        elif action_type == "wait":
            allowed = True
            reason = "allowed"
        elif action_type == "swap":
            if target_site_id is None:
                reason = "target_required"
            else:
                allowed, reason = self.can_swap(source_site_id, target_site_id)
                if allowed:
                    self._swap_unchecked(source_site_id, target_site_id)
        elif action_type in {"move", "crawl"}:
            if target_site_id is None:
                reason = "target_required"
            else:
                allowed, reason = self.can_move(source_site_id, target_site_id)
                if allowed:
                    self._move_unchecked(source_site_id, target_site_id)
        signature_after = self.signature()
        return SubstrateActionResult(
            action_type=action_type,
            source_site_id=source_site_id,
            target_site_id=target_site_id,
            allowed=bool(allowed),
            reason=reason,
            state_changed=signature_before != signature_after,
            signature_before=signature_before,
            signature_after=signature_after,
        )

    def apply_indexed_policy_action(self, source_site_id: int, policy_action: Any) -> SubstrateActionResult:
        """Apply an E03-style indexed action on a 1D-compatible substrate."""

        action_type = str(getattr(policy_action, "action_type", "wait"))
        target_index = getattr(policy_action, "target_index", None)
        if action_type == "swap":
            return self.apply_action("swap", source_site_id, None if target_index is None else int(target_index))
        return self.apply_action("wait", source_site_id, None if target_index is None else int(target_index))

    def legal_actions(self, source_site_id: int) -> tuple[SubstrateActionResult, ...]:
        actions = [self.apply_action("wait", source_site_id)]
        for target_site_id in self.neighbors(source_site_id):
            before = self.signature()
            actions.append(self._dry_run_result("swap", source_site_id, target_site_id, *self.can_swap(source_site_id, target_site_id), before))
            actions.append(self._dry_run_result("move", source_site_id, target_site_id, *self.can_move(source_site_id, target_site_id), before))
        return tuple(actions)

    def local_observation(self, source_site_id: int) -> LocalSubstrateObservation:
        source_site_id = int(source_site_id)
        self._require_site(source_site_id)
        actor_cell = self.cell_at(source_site_id)
        if actor_cell is None:
            raise ValueError(f"actor site {source_site_id} is empty")
        neighbor_items: list[NeighborObservation] = []
        for target_site_id in self.neighbors(source_site_id):
            target_cell = self.cell_at(target_site_id)
            neighbor_items.append(
                NeighborObservation(
                    site_id=target_site_id,
                    coordinate=self.coordinate(target_site_id),
                    direction=self.direction_between(source_site_id, target_site_id),
                    occupied=target_cell is not None,
                    cell_id=None if target_cell is None else str(target_cell.cell_id),
                    value=None if target_cell is None else target_cell.value,
                    label=None if target_cell is None else str(target_cell.label),
                    status=None if target_cell is None else str(target_cell.status),
                    movable_target=_is_movable_target(target_cell),
                )
            )
        return LocalSubstrateObservation(
            substrate_kind=self.kind,
            actor_site_id=source_site_id,
            actor_coordinate=self.coordinate(source_site_id),
            actor_cell=actor_cell,
            neighbors=tuple(neighbor_items),
            boundary=self.site(source_site_id).boundary,
        )

    def signature(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                int(site_id),
                None if cell is None else str(cell.cell_id),
                None if cell is None else cell.value,
                None if cell is None else str(cell.label),
                None if cell is None else str(cell.status),
            )
            for site_id, cell in sorted(self._cells_by_site.items())
        )

    def values_in_site_order(self) -> tuple[int | float | None, ...]:
        return tuple(None if self.cell_at(site_id) is None else self.cell_at(site_id).value for site_id in self.site_ids)

    def labels_in_site_order(self) -> tuple[str | None, ...]:
        return tuple(None if self.cell_at(site_id) is None else str(self.cell_at(site_id).label) for site_id in self.site_ids)

    def _swap_unchecked(self, source_site_id: int, target_site_id: int) -> None:
        source_cell = self._cells_by_site[source_site_id]
        target_cell = self._cells_by_site[target_site_id]
        self._cells_by_site[source_site_id] = target_cell
        self._cells_by_site[target_site_id] = source_cell
        if source_cell is not None:
            self._site_by_cell_id[str(source_cell.cell_id)] = target_site_id
        if target_cell is not None:
            self._site_by_cell_id[str(target_cell.cell_id)] = source_site_id

    def _move_unchecked(self, source_site_id: int, target_site_id: int) -> None:
        source_cell = self._cells_by_site[source_site_id]
        self._cells_by_site[source_site_id] = None
        self._cells_by_site[target_site_id] = source_cell
        if source_cell is not None:
            self._site_by_cell_id[str(source_cell.cell_id)] = target_site_id

    def _dry_run_result(
        self,
        action_type: str,
        source_site_id: int,
        target_site_id: int,
        allowed: bool,
        reason: str,
        signature: tuple[tuple[Any, ...], ...],
    ) -> SubstrateActionResult:
        return SubstrateActionResult(
            action_type=action_type,
            source_site_id=int(source_site_id),
            target_site_id=int(target_site_id),
            allowed=bool(allowed),
            reason=str(reason),
            state_changed=False,
            signature_before=signature,
            signature_after=signature,
        )

    def _require_site(self, site_id: int) -> None:
        if int(site_id) not in self._site_by_id:
            raise KeyError(f"unknown site_id: {site_id}")


def make_cells(values: Sequence[int | float], *, label: str = "cell", status: str = "active", id_prefix: str = "cell") -> tuple[SubstrateCell, ...]:
    """Create a deterministic set of cells for fixtures and validation."""

    return tuple(
        SubstrateCell(cell_id=f"{id_prefix}_{idx}", value=value, label=label, status=status)
        for idx, value in enumerate(values)
    )


def array_substrate(length: int, *, periodic: bool = False) -> SubstrateState:
    if int(length) <= 0:
        raise ValueError("length must be positive")
    length = int(length)
    sites = [Site(site_id=idx, coordinate=(idx,), boundary=not periodic and idx in {0, length - 1}) for idx in range(length)]
    adjacency: dict[int, list[int]] = {idx: [] for idx in range(length)}
    directions: dict[tuple[int, int], str] = {}
    for idx in range(length):
        candidates = ((idx - 1, "left"), (idx + 1, "right"))
        for raw_target, direction in candidates:
            target = raw_target % length if periodic else raw_target
            if 0 <= target < length and target != idx and target not in adjacency[idx]:
                adjacency[idx].append(target)
                directions[(idx, target)] = direction
    return SubstrateState(
        kind="array_1d",
        sites=sites,
        adjacency=adjacency,
        directions=directions,
        boundary_condition="periodic" if periodic else "open",
        dimensions={"length": length},
    )


def square_grid_substrate(width: int, height: int, *, periodic: bool = False) -> SubstrateState:
    width, height = _positive_dimensions(width, height)
    sites = []
    adjacency: dict[int, list[int]] = {}
    directions: dict[tuple[int, int], str] = {}
    for y in range(height):
        for x in range(width):
            site_id = _square_id(x, y, width)
            boundary = not periodic and (x == 0 or y == 0 or x == width - 1 or y == height - 1)
            sites.append(Site(site_id=site_id, coordinate=(x, y), boundary=boundary))
            adjacency[site_id] = []
    for y in range(height):
        for x in range(width):
            source = _square_id(x, y, width)
            for dx, dy, direction in ((0, -1, "north"), (1, 0, "east"), (0, 1, "south"), (-1, 0, "west")):
                nx, ny = _neighbor_coordinate(x, y, dx, dy, width, height, periodic)
                if nx is None or ny is None:
                    continue
                target = _square_id(nx, ny, width)
                if target != source and target not in adjacency[source]:
                    adjacency[source].append(target)
                    directions[(source, target)] = direction
    return SubstrateState(
        kind="square_grid_2d",
        sites=sites,
        adjacency=adjacency,
        directions=directions,
        boundary_condition="periodic" if periodic else "open",
        dimensions={"width": width, "height": height},
    )


def hex_grid_substrate(width: int, height: int, *, periodic: bool = False) -> SubstrateState:
    width, height = _positive_dimensions(width, height)
    sites = []
    adjacency: dict[int, list[int]] = {}
    directions: dict[tuple[int, int], str] = {}
    for r in range(height):
        for q in range(width):
            site_id = _square_id(q, r, width)
            boundary = not periodic and (q == 0 or r == 0 or q == width - 1 or r == height - 1)
            sites.append(Site(site_id=site_id, coordinate=(q, r), boundary=boundary))
            adjacency[site_id] = []
    axial_neighbors = (
        (1, 0, "east"),
        (1, -1, "north_east"),
        (0, -1, "north_west"),
        (-1, 0, "west"),
        (-1, 1, "south_west"),
        (0, 1, "south_east"),
    )
    for r in range(height):
        for q in range(width):
            source = _square_id(q, r, width)
            for dq, dr, direction in axial_neighbors:
                nq, nr = _neighbor_coordinate(q, r, dq, dr, width, height, periodic)
                if nq is None or nr is None:
                    continue
                target = _square_id(nq, nr, width)
                if target != source and target not in adjacency[source]:
                    adjacency[source].append(target)
                    directions[(source, target)] = direction
    return SubstrateState(
        kind="hex_grid_2d",
        sites=sites,
        adjacency=adjacency,
        directions=directions,
        boundary_condition="periodic" if periodic else "open",
        dimensions={"width": width, "height": height, "coordinate_system": "axial_rectangular"},
    )


def lattice3d_substrate(width: int, height: int, depth: int, *, periodic: bool = False) -> SubstrateState:
    width, height, depth = _positive_dimensions(width, height, depth)
    sites = []
    adjacency: dict[int, list[int]] = {}
    directions: dict[tuple[int, int], str] = {}
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                site_id = _lattice3d_id(x, y, z, width, height)
                boundary = not periodic and (
                    x == 0 or y == 0 or z == 0 or x == width - 1 or y == height - 1 or z == depth - 1
                )
                sites.append(Site(site_id=site_id, coordinate=(x, y, z), boundary=boundary))
                adjacency[site_id] = []
    steps = (
        (0, -1, 0, "north"),
        (1, 0, 0, "east"),
        (0, 1, 0, "south"),
        (-1, 0, 0, "west"),
        (0, 0, 1, "up"),
        (0, 0, -1, "down"),
    )
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                source = _lattice3d_id(x, y, z, width, height)
                for dx, dy, dz, direction in steps:
                    nx = _wrap_or_bound(x + dx, width, periodic)
                    ny = _wrap_or_bound(y + dy, height, periodic)
                    nz = _wrap_or_bound(z + dz, depth, periodic)
                    if nx is None or ny is None or nz is None:
                        continue
                    target = _lattice3d_id(nx, ny, nz, width, height)
                    if target != source and target not in adjacency[source]:
                        adjacency[source].append(target)
                        directions[(source, target)] = direction
    return SubstrateState(
        kind="lattice_3d",
        sites=sites,
        adjacency=adjacency,
        directions=directions,
        boundary_condition="periodic" if periodic else "open",
        dimensions={"width": width, "height": height, "depth": depth},
    )


def graph_substrate(
    adjacency: Mapping[int, Iterable[int]],
    *,
    coordinates: Mapping[int, Sequence[int]] | None = None,
    kind: str = "irregular_graph",
) -> SubstrateState:
    normalized = _normalize_adjacency({int(k): tuple(int(v) for v in values) for k, values in adjacency.items()}, adjacency.keys())
    coordinate_map = {int(key): tuple(int(v) for v in value) for key, value in (coordinates or {}).items()}
    sites = [
        Site(
            site_id=site_id,
            coordinate=coordinate_map.get(site_id, (site_id,)),
            boundary=len(neighbors) <= 1,
        )
        for site_id, neighbors in sorted(normalized.items())
    ]
    directions = {
        (source, target): f"edge_to_{target}"
        for source, targets in normalized.items()
        for target in targets
    }
    return SubstrateState(
        kind=kind,
        sites=sites,
        adjacency=normalized,
        directions=directions,
        boundary_condition="graph",
        dimensions={"node_count": len(normalized)},
    )


def _normalize_adjacency(adjacency: Mapping[int, Sequence[int]], site_ids: Iterable[int]) -> dict[int, tuple[int, ...]]:
    site_set = {int(site_id) for site_id in site_ids}
    normalized: dict[int, tuple[int, ...]] = {}
    for site_id in sorted(site_set):
        if site_id not in adjacency:
            raise ValueError(f"missing adjacency entry for site {site_id}")
        seen: set[int] = set()
        neighbors: list[int] = []
        for target in adjacency[site_id]:
            target = int(target)
            if target == site_id:
                raise ValueError(f"self-loop is not allowed for site {site_id}")
            if target not in site_set:
                raise ValueError(f"adjacency for site {site_id} references unknown site {target}")
            if target not in seen:
                neighbors.append(target)
                seen.add(target)
        normalized[site_id] = tuple(neighbors)
    for source, targets in normalized.items():
        for target in targets:
            if source not in normalized[target]:
                raise ValueError(f"adjacency must be undirected: {source}->{target} lacks reciprocal edge")
    return normalized


def _positive_dimensions(*values: int) -> tuple[int, ...]:
    dims = tuple(int(value) for value in values)
    if any(value <= 0 for value in dims):
        raise ValueError("dimensions must be positive")
    return dims


def _square_id(x: int, y: int, width: int) -> int:
    return int(y) * int(width) + int(x)


def _lattice3d_id(x: int, y: int, z: int, width: int, height: int) -> int:
    return int(z) * int(width) * int(height) + int(y) * int(width) + int(x)


def _wrap_or_bound(value: int, limit: int, periodic: bool) -> int | None:
    if periodic:
        return int(value) % int(limit)
    if 0 <= int(value) < int(limit):
        return int(value)
    return None


def _neighbor_coordinate(
    x: int,
    y: int,
    dx: int,
    dy: int,
    width: int,
    height: int,
    periodic: bool,
) -> tuple[int | None, int | None]:
    return _wrap_or_bound(x + dx, width, periodic), _wrap_or_bound(y + dy, height, periodic)
