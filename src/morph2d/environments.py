"""Typed spatial environment specifications for E06 S03.

S03 compiles square, axial-hexagonal, and explicit irregular fixtures into one
canonical undirected graph representation.  It defines topology, occupancy,
site roles, boundary observations, serialization, and static evaluation
plumbing only.  Movement proposals and conflict resolution remain S04 work.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .grammar import RelationalGrammar, score_grid
from .targets import TargetDefinition, evaluate_success


CATALOG_VERSION = "e06.s03.environment-catalog.v1"
ENVIRONMENT_VERSION = "e06.s03.environment.v1"
SUPPORTED_GEOMETRIES = {"square", "hexagonal", "irregular"}
SUPPORTED_BOUNDARIES = {"bounded", "periodic"}
SUPPORTED_OCCUPANCY = {"fully_occupied", "vacancy_enabled"}
SITE_ROLES = {"active", "obstacle", "fixed_boundary"}
BOUNDARY_SIGNAL_MODES = {"none", "domain_edge_flags", "explicit_site_tags"}
SQUARE_DIRECTIONS = (
    ("north", -1, 0),
    ("east", 0, 1),
    ("south", 1, 0),
    ("west", 0, -1),
)
HEX_DIRECTIONS = (
    ("q_plus", 1, 0),
    ("q_minus", -1, 0),
    ("r_plus", 0, 1),
    ("r_minus", 0, -1),
    ("s_plus", 1, -1),
    ("s_minus", -1, 1),
)


class EnvironmentValidationError(ValueError):
    """Raised when an environment declaration is invalid or unsupported."""


@dataclass(frozen=True)
class Site:
    site_id: str
    coordinate: tuple[int, int]
    role: str
    fixed_token: str | None
    boundary_tags: tuple[str, ...]


@dataclass(frozen=True)
class Environment:
    environment_id: str
    title: str
    geometry: str
    boundary_mode: str
    occupancy_mode: str
    vacancy_label: str
    obstacle_token: str
    generator: Mapping[str, Any]
    sites: tuple[Site, ...]
    edges: tuple[tuple[str, str], ...]
    initial_state: Mapping[str, str]
    boundary_signal: Mapping[str, Any]
    target_binding: Mapping[str, str] | None
    require_connected: bool
    notes: str
    expected: Mapping[str, Any]

    @property
    def occupiable_sites(self) -> tuple[Site, ...]:
        return tuple(site for site in self.sites if site.role != "obstacle")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _site_id(geometry: str, first: int, second: int) -> str:
    if geometry == "square":
        return f"r{first}_c{second}"
    return f"q{first}_r{second}"


def _canonical_edge(first: str, second: str) -> tuple[str, str]:
    if first == second:
        raise EnvironmentValidationError("self edges are not allowed")
    return tuple(sorted((first, second)))


def _regular_coordinates(
    geometry: str, boundary_mode: str, generator: Mapping[str, Any]
) -> tuple[tuple[int, int], ...]:
    kind = generator.get("kind")
    if geometry == "square":
        if kind != "rectangle" or set(generator) != {"kind", "rows", "columns"}:
            raise EnvironmentValidationError(
                "square generator must be rectangle with rows and columns"
            )
        rows, columns = int(generator["rows"]), int(generator["columns"])
        if min(rows, columns) <= 0:
            raise EnvironmentValidationError("square dimensions must be positive")
        if boundary_mode == "periodic" and min(rows, columns) < 3:
            raise EnvironmentValidationError(
                "periodic square dimensions must each be at least three"
            )
        return tuple((row, col) for row in range(rows) for col in range(columns))
    if geometry == "hexagonal" and boundary_mode == "bounded":
        if kind != "axial_radius" or set(generator) != {"kind", "radius"}:
            raise EnvironmentValidationError(
                "bounded hexagonal generator must declare axial_radius"
            )
        radius = int(generator["radius"])
        if radius < 1:
            raise EnvironmentValidationError("hexagonal radius must be positive")
        return tuple(
            sorted(
                (q, r)
                for q in range(-radius, radius + 1)
                for r in range(-radius, radius + 1)
                if max(abs(q), abs(r), abs(-q - r)) <= radius
            )
        )
    if geometry == "hexagonal" and boundary_mode == "periodic":
        if kind != "axial_torus" or set(generator) != {
            "kind",
            "rows",
            "columns",
        }:
            raise EnvironmentValidationError(
                "periodic hexagonal generator must declare axial_torus"
            )
        rows, columns = int(generator["rows"]), int(generator["columns"])
        if min(rows, columns) < 3:
            raise EnvironmentValidationError(
                "periodic hexagonal dimensions must each be at least three"
            )
        return tuple((q, r) for q in range(columns) for r in range(rows))
    raise EnvironmentValidationError(
        f"unsupported generator for {geometry}/{boundary_mode}"
    )


def _regular_edges(
    geometry: str,
    boundary_mode: str,
    generator: Mapping[str, Any],
    coordinates: set[tuple[int, int]],
    obstacles: set[tuple[int, int]],
) -> tuple[tuple[str, str], ...]:
    directions = SQUARE_DIRECTIONS if geometry == "square" else HEX_DIRECTIONS
    edges: set[tuple[str, str]] = set()
    for first in sorted(coordinates - obstacles):
        for _, delta_first, delta_second in directions:
            neighbor = (first[0] + delta_first, first[1] + delta_second)
            if boundary_mode == "periodic":
                rows, columns = int(generator["rows"]), int(generator["columns"])
                if geometry == "square":
                    neighbor = (neighbor[0] % rows, neighbor[1] % columns)
                else:
                    neighbor = (neighbor[0] % columns, neighbor[1] % rows)
            if neighbor not in coordinates or neighbor in obstacles:
                continue
            edges.add(
                _canonical_edge(
                    _site_id(geometry, *first), _site_id(geometry, *neighbor)
                )
            )
    return tuple(sorted(edges))


def _regular_boundary_tags(
    geometry: str,
    boundary_mode: str,
    coordinate: tuple[int, int],
    coordinates: set[tuple[int, int]],
) -> tuple[str, ...]:
    if boundary_mode == "periodic":
        return ()
    directions = SQUARE_DIRECTIONS if geometry == "square" else HEX_DIRECTIONS
    tags = [
        name
        for name, delta_first, delta_second in directions
        if (coordinate[0] + delta_first, coordinate[1] + delta_second)
        not in coordinates
    ]
    return tuple(sorted(tags))


def _compile_irregular(
    generator: Mapping[str, Any], obstacle_ids: set[str]
) -> tuple[tuple[Site, ...], tuple[tuple[str, str], ...]]:
    if (
        set(generator) != {"kind", "sites", "edges"}
        or generator.get("kind") != "explicit_graph"
    ):
        raise EnvironmentValidationError(
            "irregular generator must be an explicit_graph with sites and edges"
        )
    sites: list[Site] = []
    for raw in generator["sites"]:
        if set(raw) != {"siteId", "coordinate", "boundaryTags"}:
            raise EnvironmentValidationError("invalid explicit irregular site")
        site_id = str(raw["siteId"])
        coordinate = tuple(int(item) for item in raw["coordinate"])
        if len(coordinate) != 2:
            raise EnvironmentValidationError("site coordinates must have length two")
        sites.append(
            Site(
                site_id=site_id,
                coordinate=(coordinate[0], coordinate[1]),
                role="obstacle" if site_id in obstacle_ids else "active",
                fixed_token=None,
                boundary_tags=tuple(sorted(str(item) for item in raw["boundaryTags"])),
            )
        )
    site_ids = {site.site_id for site in sites}
    edges: set[tuple[str, str]] = set()
    for raw_edge in generator["edges"]:
        if len(raw_edge) != 2:
            raise EnvironmentValidationError("explicit edges must contain two IDs")
        first, second = str(raw_edge[0]), str(raw_edge[1])
        if first not in site_ids or second not in site_ids:
            raise EnvironmentValidationError("explicit edge references unknown site")
        if first in obstacle_ids or second in obstacle_ids:
            continue
        edge = _canonical_edge(first, second)
        if edge in edges:
            raise EnvironmentValidationError("duplicate explicit edge")
        edges.add(edge)
    return tuple(sorted(sites, key=lambda item: item.site_id)), tuple(sorted(edges))


def _initial_state(
    raw: Mapping[str, Any],
    sites: Sequence[Site],
    geometry: str,
    generator: Mapping[str, Any],
    vacancy_label: str,
    obstacle_token: str,
) -> dict[str, str]:
    if set(raw) != {"defaultToken", "overrides", "rows"}:
        raise EnvironmentValidationError("invalid initialState keys")
    state = {site.site_id: str(raw["defaultToken"]) for site in sites}
    rows = raw["rows"]
    if rows is not None:
        if geometry != "square" or generator["kind"] != "rectangle":
            raise EnvironmentValidationError("row state is supported only for square")
        if len(rows) != int(generator["rows"]) or any(
            len(row) != int(generator["columns"]) for row in rows
        ):
            raise EnvironmentValidationError("initial rows do not match dimensions")
        for row_index, row in enumerate(rows):
            for column_index, token in enumerate(row):
                state[_site_id("square", row_index, column_index)] = token
    unknown_overrides = set(raw["overrides"]) - set(state)
    if unknown_overrides:
        raise EnvironmentValidationError(
            f"initialState overrides unknown sites: {sorted(unknown_overrides)}"
        )
    state.update({str(key): str(value) for key, value in raw["overrides"].items()})
    for site in sites:
        if site.role == "obstacle":
            state[site.site_id] = obstacle_token
        elif site.fixed_token is not None:
            state[site.site_id] = site.fixed_token
    if vacancy_label == obstacle_token:
        raise EnvironmentValidationError("vacancy and obstacle tokens must differ")
    return dict(sorted(state.items()))


def _fixed_site_ids(
    raw: Mapping[str, Any],
    geometry: str,
    boundary_mode: str,
    sites: Sequence[Site],
) -> tuple[set[str], str | None]:
    if set(raw) != {"mode", "sites", "token"}:
        raise EnvironmentValidationError("invalid fixedBoundary keys")
    mode = raw["mode"]
    if mode not in {"none", "perimeter", "sites"}:
        raise EnvironmentValidationError("invalid fixedBoundary mode")
    if boundary_mode == "periodic" and mode != "none":
        raise EnvironmentValidationError(
            "periodic plus fixed_boundary is unsupported in S03"
        )
    if mode == "none":
        if raw["sites"] or raw["token"] is not None:
            raise EnvironmentValidationError("empty fixedBoundary must not carry data")
        return set(), None
    token = str(raw["token"])
    if mode == "perimeter":
        if geometry == "irregular":
            raise EnvironmentValidationError(
                "irregular fixed boundaries require explicit site IDs"
            )
        return {site.site_id for site in sites if site.boundary_tags}, token
    fixed = {str(item) for item in raw["sites"]}
    if not fixed or fixed - {site.site_id for site in sites}:
        raise EnvironmentValidationError("fixedBoundary references unknown/empty sites")
    return fixed, token


def parse_environment_spec(raw: Mapping[str, Any]) -> Environment:
    """Compile one high-level fixture specification to a canonical graph."""

    required = {
        "environmentId",
        "title",
        "geometry",
        "boundaryMode",
        "occupancyMode",
        "vacancyLabel",
        "obstacleToken",
        "generator",
        "obstacles",
        "fixedBoundary",
        "boundarySignal",
        "initialState",
        "targetBinding",
        "requireConnected",
        "notes",
        "expected",
    }
    if set(raw) != required:
        raise EnvironmentValidationError(
            f"environment keys must be exactly {sorted(required)}"
        )
    geometry = str(raw["geometry"])
    boundary_mode = str(raw["boundaryMode"])
    occupancy_mode = str(raw["occupancyMode"])
    if geometry not in SUPPORTED_GEOMETRIES:
        raise EnvironmentValidationError("unsupported geometry")
    if boundary_mode not in SUPPORTED_BOUNDARIES:
        raise EnvironmentValidationError("unsupported boundary mode")
    if occupancy_mode not in SUPPORTED_OCCUPANCY:
        raise EnvironmentValidationError("unsupported occupancy mode")
    boundary_signal = dict(raw["boundarySignal"])
    if set(boundary_signal) != {
        "mode",
        "observable",
        "includeObstacleContact",
        "includeFixedRole",
    }:
        raise EnvironmentValidationError("invalid boundarySignal keys")
    if boundary_signal["mode"] not in BOUNDARY_SIGNAL_MODES:
        raise EnvironmentValidationError("invalid boundary signal mode")
    if boundary_mode == "periodic" and (
        boundary_signal["mode"] != "none" or boundary_signal["includeFixedRole"]
    ):
        raise EnvironmentValidationError(
            "periodic domains have no natural exterior boundary signal"
        )
    if geometry == "irregular" and boundary_signal["mode"] == "domain_edge_flags":
        raise EnvironmentValidationError(
            "irregular boundaries must use explicit_site_tags, not inferred degree"
        )
    if geometry != "irregular" and boundary_signal["mode"] == "explicit_site_tags":
        raise EnvironmentValidationError(
            "explicit_site_tags is reserved for irregular graphs"
        )
    if bool(boundary_signal["observable"]) != (
        boundary_signal["mode"] != "none"
        or bool(boundary_signal["includeObstacleContact"])
        or bool(boundary_signal["includeFixedRole"])
    ):
        raise EnvironmentValidationError(
            "boundary observable flag must match declared signal channels"
        )

    generator = dict(raw["generator"])
    obstacle_raw = raw["obstacles"]
    if geometry == "irregular":
        obstacle_ids = {str(item) for item in obstacle_raw}
        sites, edges = _compile_irregular(generator, obstacle_ids)
    else:
        coordinates = set(_regular_coordinates(geometry, boundary_mode, generator))
        obstacle_coordinates = {
            tuple(int(value) for value in coordinate) for coordinate in obstacle_raw
        }
        if any(len(item) != 2 for item in obstacle_coordinates) or not (
            obstacle_coordinates <= coordinates
        ):
            raise EnvironmentValidationError("obstacle coordinate is outside domain")
        sites = tuple(
            Site(
                site_id=_site_id(geometry, *coordinate),
                coordinate=coordinate,
                role="obstacle" if coordinate in obstacle_coordinates else "active",
                fixed_token=None,
                boundary_tags=_regular_boundary_tags(
                    geometry, boundary_mode, coordinate, coordinates
                ),
            )
            for coordinate in sorted(coordinates)
        )
        edges = _regular_edges(
            geometry,
            boundary_mode,
            generator,
            coordinates,
            obstacle_coordinates,
        )

    fixed_ids, fixed_token = _fixed_site_ids(
        raw["fixedBoundary"], geometry, boundary_mode, sites
    )
    if fixed_ids & {site.site_id for site in sites if site.role == "obstacle"}:
        raise EnvironmentValidationError(
            "obstacle sites cannot be fixed boundary sites"
        )
    sites = tuple(
        Site(
            site_id=site.site_id,
            coordinate=site.coordinate,
            role="fixed_boundary" if site.site_id in fixed_ids else site.role,
            fixed_token=fixed_token if site.site_id in fixed_ids else None,
            boundary_tags=site.boundary_tags,
        )
        for site in sites
    )
    initial_state = _initial_state(
        raw["initialState"],
        sites,
        geometry,
        generator,
        str(raw["vacancyLabel"]),
        str(raw["obstacleToken"]),
    )
    target_binding = raw["targetBinding"]
    if target_binding is not None:
        if set(target_binding) != {"targetId", "grammarId", "evaluationProfile"}:
            raise EnvironmentValidationError("invalid targetBinding keys")
        target_binding = {key: str(value) for key, value in target_binding.items()}
        if target_binding["evaluationProfile"] != "s01_s02_conjunctive_square_v1":
            raise EnvironmentValidationError("unsupported target evaluation profile")
    environment = Environment(
        environment_id=str(raw["environmentId"]),
        title=str(raw["title"]),
        geometry=geometry,
        boundary_mode=boundary_mode,
        occupancy_mode=occupancy_mode,
        vacancy_label=str(raw["vacancyLabel"]),
        obstacle_token=str(raw["obstacleToken"]),
        generator=generator,
        sites=tuple(sorted(sites, key=lambda item: item.site_id)),
        edges=tuple(sorted(edges)),
        initial_state=initial_state,
        boundary_signal=boundary_signal,
        target_binding=target_binding,
        require_connected=bool(raw["requireConnected"]),
        notes=str(raw["notes"]),
        expected=dict(raw["expected"]),
    )
    validate_environment(environment)
    return environment


def _site_to_dict(site: Site) -> dict[str, Any]:
    return {
        "siteId": site.site_id,
        "coordinate": list(site.coordinate),
        "role": site.role,
        "fixedToken": site.fixed_token,
        "boundaryTags": list(site.boundary_tags),
    }


def environment_to_dict(environment: Environment) -> dict[str, Any]:
    """Serialize the compiled semantic environment, not its generator shorthand."""

    return {
        "schemaVersion": ENVIRONMENT_VERSION,
        "environmentId": environment.environment_id,
        "title": environment.title,
        "geometry": environment.geometry,
        "boundaryMode": environment.boundary_mode,
        "occupancyMode": environment.occupancy_mode,
        "vacancyLabel": environment.vacancy_label,
        "obstacleToken": environment.obstacle_token,
        "generator": dict(environment.generator),
        "sites": [_site_to_dict(site) for site in environment.sites],
        "edges": [list(edge) for edge in environment.edges],
        "initialState": dict(environment.initial_state),
        "boundarySignal": dict(environment.boundary_signal),
        "targetBinding": (
            None
            if environment.target_binding is None
            else dict(environment.target_binding)
        ),
        "requireConnected": environment.require_connected,
        "notes": environment.notes,
        "expected": dict(environment.expected),
    }


def parse_serialized_environment(raw: Mapping[str, Any]) -> Environment:
    required = {
        "schemaVersion",
        "environmentId",
        "title",
        "geometry",
        "boundaryMode",
        "occupancyMode",
        "vacancyLabel",
        "obstacleToken",
        "generator",
        "sites",
        "edges",
        "initialState",
        "boundarySignal",
        "targetBinding",
        "requireConnected",
        "notes",
        "expected",
    }
    if set(raw) != required or raw["schemaVersion"] != ENVIRONMENT_VERSION:
        raise EnvironmentValidationError("serialized environment schema mismatch")
    sites: list[Site] = []
    for item in raw["sites"]:
        if set(item) != {
            "siteId",
            "coordinate",
            "role",
            "fixedToken",
            "boundaryTags",
        }:
            raise EnvironmentValidationError("invalid serialized site")
        coordinate = tuple(int(value) for value in item["coordinate"])
        if len(coordinate) != 2:
            raise EnvironmentValidationError("invalid serialized coordinate")
        sites.append(
            Site(
                site_id=str(item["siteId"]),
                coordinate=(coordinate[0], coordinate[1]),
                role=str(item["role"]),
                fixed_token=(
                    None if item["fixedToken"] is None else str(item["fixedToken"])
                ),
                boundary_tags=tuple(str(value) for value in item["boundaryTags"]),
            )
        )
    target_binding = raw["targetBinding"]
    environment = Environment(
        environment_id=str(raw["environmentId"]),
        title=str(raw["title"]),
        geometry=str(raw["geometry"]),
        boundary_mode=str(raw["boundaryMode"]),
        occupancy_mode=str(raw["occupancyMode"]),
        vacancy_label=str(raw["vacancyLabel"]),
        obstacle_token=str(raw["obstacleToken"]),
        generator=dict(raw["generator"]),
        sites=tuple(sites),
        edges=tuple(tuple(str(value) for value in edge) for edge in raw["edges"]),
        initial_state={
            str(key): str(value) for key, value in raw["initialState"].items()
        },
        boundary_signal=dict(raw["boundarySignal"]),
        target_binding=(
            None
            if target_binding is None
            else {str(key): str(value) for key, value in target_binding.items()}
        ),
        require_connected=bool(raw["requireConnected"]),
        notes=str(raw["notes"]),
        expected=dict(raw["expected"]),
    )
    validate_environment(environment)
    return environment


def canonical_environment_bytes(environment: Environment) -> bytes:
    return _canonical_json_bytes(environment_to_dict(environment))


def environment_sha256(environment: Environment) -> str:
    return hashlib.sha256(canonical_environment_bytes(environment)).hexdigest()


def neighbor_map(environment: Environment) -> dict[str, tuple[str, ...]]:
    neighbors = {site.site_id: [] for site in environment.occupiable_sites}
    for first, second in environment.edges:
        neighbors[first].append(second)
        neighbors[second].append(first)
    return {key: tuple(sorted(values)) for key, values in sorted(neighbors.items())}


def connected_components(environment: Environment) -> tuple[tuple[str, ...], ...]:
    neighbors = neighbor_map(environment)
    remaining = set(neighbors)
    components: list[tuple[str, ...]] = []
    while remaining:
        start = min(remaining)
        queue = deque([start])
        component = {start}
        remaining.remove(start)
        while queue:
            current = queue.popleft()
            for other in neighbors[current]:
                if other in remaining:
                    remaining.remove(other)
                    component.add(other)
                    queue.append(other)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def boundary_observation(environment: Environment, site_id: str) -> dict[str, Any]:
    sites = {site.site_id: site for site in environment.sites}
    if site_id not in sites or sites[site_id].role == "obstacle":
        raise EnvironmentValidationError(
            "boundary observation requires occupiable site"
        )
    site = sites[site_id]
    signal = environment.boundary_signal
    boundary_tags = list(site.boundary_tags) if signal["mode"] != "none" else []
    coordinate_to_site = {site.coordinate: site for site in environment.sites}
    obstacle_contacts: list[str] = []
    if signal["includeObstacleContact"] and environment.geometry != "irregular":
        directions = (
            SQUARE_DIRECTIONS if environment.geometry == "square" else HEX_DIRECTIONS
        )
        for _, delta_first, delta_second in directions:
            other = coordinate_to_site.get(
                (
                    site.coordinate[0] + delta_first,
                    site.coordinate[1] + delta_second,
                )
            )
            if other is not None and other.role == "obstacle":
                obstacle_contacts.append(other.site_id)
    neighbors = neighbor_map(environment)[site_id]
    fixed_neighbors = (
        sorted(other for other in neighbors if sites[other].role == "fixed_boundary")
        if signal["includeFixedRole"]
        else []
    )
    return {
        "observable": bool(signal["observable"]),
        "boundaryTags": sorted(boundary_tags),
        "obstacleContacts": sorted(obstacle_contacts),
        "isFixedBoundary": bool(
            signal["includeFixedRole"] and site.role == "fixed_boundary"
        ),
        "fixedBoundaryNeighbors": fixed_neighbors,
    }


def topology_summary(environment: Environment) -> dict[str, Any]:
    neighbors = neighbor_map(environment)
    roles = Counter(site.role for site in environment.sites)
    degree_histogram = Counter(len(values) for values in neighbors.values())
    observations = {
        site.site_id: boundary_observation(environment, site.site_id)
        for site in environment.occupiable_sites
    }
    signaled = sum(
        bool(
            record["boundaryTags"]
            or record["obstacleContacts"]
            or record["isFixedBoundary"]
            or record["fixedBoundaryNeighbors"]
        )
        for record in observations.values()
    )
    return {
        "environmentId": environment.environment_id,
        "geometry": environment.geometry,
        "boundaryMode": environment.boundary_mode,
        "occupancyMode": environment.occupancy_mode,
        "siteCount": len(environment.sites),
        "occupiableSiteCount": len(environment.occupiable_sites),
        "edgeCount": len(environment.edges),
        "componentCount": len(connected_components(environment)),
        "degreeHistogram": {
            str(key): value for key, value in sorted(degree_histogram.items())
        },
        "vacancyCount": sum(
            environment.initial_state[site.site_id] == environment.vacancy_label
            for site in environment.occupiable_sites
        ),
        "obstacleCount": roles["obstacle"],
        "fixedBoundaryCount": roles["fixed_boundary"],
        "naturalBoundarySiteCount": sum(
            bool(site.boundary_tags) for site in environment.occupiable_sites
        ),
        "signaledSiteCount": signaled,
    }


def validate_environment(environment: Environment) -> None:
    if environment.geometry not in SUPPORTED_GEOMETRIES:
        raise EnvironmentValidationError("unsupported geometry")
    if environment.boundary_mode not in SUPPORTED_BOUNDARIES:
        raise EnvironmentValidationError("unsupported boundary mode")
    if environment.occupancy_mode not in SUPPORTED_OCCUPANCY:
        raise EnvironmentValidationError("unsupported occupancy mode")
    site_ids = [site.site_id for site in environment.sites]
    if not site_ids or len(site_ids) != len(set(site_ids)):
        raise EnvironmentValidationError("site IDs must be nonempty and unique")
    coordinates = [site.coordinate for site in environment.sites]
    if len(coordinates) != len(set(coordinates)):
        raise EnvironmentValidationError("site coordinates must be unique")
    if any(site.role not in SITE_ROLES for site in environment.sites):
        raise EnvironmentValidationError("invalid site role")
    if tuple(sorted(environment.edges)) != environment.edges or len(
        environment.edges
    ) != len(set(environment.edges)):
        raise EnvironmentValidationError("edges must be sorted and unique")
    sites = {site.site_id: site for site in environment.sites}
    for first, second in environment.edges:
        if _canonical_edge(first, second) != (first, second):
            raise EnvironmentValidationError("edge endpoints must be canonical")
        if first not in sites or second not in sites:
            raise EnvironmentValidationError("edge references unknown site")
        if sites[first].role == "obstacle" or sites[second].role == "obstacle":
            raise EnvironmentValidationError("obstacles cannot have graph edges")
    if set(environment.initial_state) != set(site_ids):
        raise EnvironmentValidationError("initial state must cover every site exactly")
    for site in environment.sites:
        token = environment.initial_state[site.site_id]
        if site.role == "obstacle" and token != environment.obstacle_token:
            raise EnvironmentValidationError("obstacle token mismatch")
        if site.role != "obstacle" and token == environment.obstacle_token:
            raise EnvironmentValidationError("obstacle token on occupiable site")
        if site.role == "fixed_boundary" and token != site.fixed_token:
            raise EnvironmentValidationError("fixed boundary token mismatch")
    vacancies = sum(
        environment.initial_state[site.site_id] == environment.vacancy_label
        for site in environment.occupiable_sites
    )
    if environment.occupancy_mode == "fully_occupied" and vacancies:
        raise EnvironmentValidationError("fully occupied fixture contains vacancy")
    if environment.occupancy_mode == "vacancy_enabled" and vacancies == 0:
        raise EnvironmentValidationError("vacancy-enabled fixture needs a vacancy")
    if environment.require_connected and len(connected_components(environment)) != 1:
        raise EnvironmentValidationError("occupiable graph is disconnected")
    if environment.boundary_mode == "periodic":
        if any(site.boundary_tags for site in environment.sites):
            raise EnvironmentValidationError("periodic sites cannot have boundary tags")
        if environment.boundary_signal["mode"] != "none":
            raise EnvironmentValidationError("periodic exterior signal is unsupported")
        if any(site.role == "fixed_boundary" for site in environment.sites):
            raise EnvironmentValidationError("periodic fixed boundary is unsupported")
    if environment.geometry == "irregular" and environment.boundary_signal[
        "mode"
    ] not in {"none", "explicit_site_tags"}:
        raise EnvironmentValidationError("irregular boundary inference is unsupported")
    if environment.target_binding is not None:
        if not (
            environment.geometry == "square"
            and environment.boundary_mode == "bounded"
            and all(site.role == "active" for site in environment.sites)
            and environment.generator.get("kind") == "rectangle"
        ):
            raise EnvironmentValidationError(
                "S01/S02 target binding supports only unobstructed bounded square rectangles"
            )
    expected = dict(environment.expected)
    observed = topology_summary(environment)
    for key, value in expected.items():
        if key not in observed or observed[key] != value:
            raise EnvironmentValidationError(
                f"{environment.environment_id}: expected {key}={value!r}, "
                f"observed {observed.get(key)!r}"
            )


def replay_environment(environment: Environment) -> dict[str, Any]:
    """Replay deterministic topology/occupancy queries without S04 movement."""

    neighbors = neighbor_map(environment)
    records = [
        {
            "siteId": site.site_id,
            "coordinate": list(site.coordinate),
            "role": site.role,
            "token": environment.initial_state[site.site_id],
            "neighbors": list(neighbors.get(site.site_id, ())),
            "boundaryObservation": (
                None
                if site.role == "obstacle"
                else boundary_observation(environment, site.site_id)
            ),
        }
        for site in environment.sites
    ]
    payload = {
        "environmentId": environment.environment_id,
        "environmentSha256": environment_sha256(environment),
        "topology": topology_summary(environment),
        "components": [list(item) for item in connected_components(environment)],
        "siteRecords": records,
    }
    return {
        **payload,
        "replaySha256": hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
    }


def environment_grid(environment: Environment) -> tuple[tuple[str, ...], ...]:
    if (
        environment.geometry != "square"
        or environment.generator.get("kind") != "rectangle"
    ):
        raise EnvironmentValidationError("only rectangular square state has a grid")
    rows, columns = (
        int(environment.generator["rows"]),
        int(environment.generator["columns"]),
    )
    return tuple(
        tuple(
            environment.initial_state[_site_id("square", row, column)]
            for column in range(columns)
        )
        for row in range(rows)
    )


def evaluate_conjunctive(
    environment: Environment,
    target: TargetDefinition,
    grammar: RelationalGrammar,
) -> dict[str, Any]:
    """Record local grammar and independent global target gates separately."""

    if environment.target_binding is None:
        raise EnvironmentValidationError("environment has no target binding")
    if environment.target_binding["targetId"] != target.target_id:
        raise EnvironmentValidationError("target binding mismatch")
    if environment.target_binding["grammarId"] != grammar.grammar_id:
        raise EnvironmentValidationError("grammar binding mismatch")
    if (
        target.equivalence["boundaryCueRequired"]
        and not environment.boundary_signal["observable"]
    ):
        raise EnvironmentValidationError(
            "target requires an observable boundary signal"
        )
    grid = environment_grid(environment)
    local = score_grid(grid, grammar)
    global_audit = evaluate_success(grid, target)
    completion = bool(local["accepted"] and global_audit["success"])
    return {
        "environmentId": environment.environment_id,
        "targetId": target.target_id,
        "grammarId": grammar.grammar_id,
        "applicable": True,
        "localGrammar": {
            "accepted": local["accepted"],
            "softScore": local["softScore"],
            "relationalScore": local["relationalScore"],
            "hardViolationCount": local["hardViolationCount"],
        },
        "globalAudit": {
            "success": global_audit["success"],
            "countMatch": global_audit["countMatch"],
            "equivalenceOrGeometryMatch": global_audit["geometryMatch"],
            "componentMatch": global_audit["componentMatch"],
            "topologyMatch": global_audit["topologyMatch"],
            "mismatchCount": global_audit["mismatchCount"],
            "componentCounts": global_audit["componentCounts"],
            "vacancyHoles": global_audit["vacancyHoles"],
        },
        "completion": completion,
        "completionRule": "localGrammar.accepted AND independentS01GlobalAudit.success",
    }


def parse_environment_catalog(
    raw: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[Environment, ...]]:
    required = {
        "schemaVersion",
        "researchStepId",
        "libraryVersion",
        "coordinateConventions",
        "evaluationContract",
        "unsupportedCombinations",
        "environments",
    }
    if set(raw) != required:
        raise EnvironmentValidationError("environment catalog keys mismatch")
    if raw["schemaVersion"] != CATALOG_VERSION or raw["researchStepId"] != "S03":
        raise EnvironmentValidationError("environment catalog version mismatch")
    environments = tuple(parse_environment_spec(item) for item in raw["environments"])
    identifiers = [environment.environment_id for environment in environments]
    if len(identifiers) != len(set(identifiers)):
        raise EnvironmentValidationError("environment IDs must be unique")
    metadata = {key: raw[key] for key in required - {"environments"}}
    return metadata, environments


def load_environment_catalog(
    path: str | Path,
) -> tuple[Mapping[str, Any], tuple[Environment, ...]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise EnvironmentValidationError("environment catalog root must be a mapping")
    return parse_environment_catalog(raw)
