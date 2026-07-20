"""Deterministic legal-movement semantics for E06 S04.

The movement layer consumes the canonical undirected graph and site roles frozen
in S03.  It defines side-effect-free authenticated proposals, validation,
deterministic simultaneous conflict resolution, atomic commit, complete logical
cost accounting, canonical serialization, and exact replay.  It does not define
an Algotype or expose grammar/target evaluation to movement legality.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .environments import Environment, environment_sha256, neighbor_map


STATE_VERSION = "e06.s04.movement-state.v1"
PROPOSAL_VERSION = "e06.s04.movement-proposal.v1"
BATCH_VERSION = "e06.s04.movement-batch-result.v1"
ENABLED_KINDS = {
    "adjacent_swap",
    "vacancy_move",
    "short_exchange",
    "rotation",
}
DEFERRED_KINDS = {"long_range_exchange", "division", "removal"}
OCCUPANT_KINDS = {"cell", "vacancy", "fixed_boundary"}
LEDGER_FIELDS = (
    "submittedProposals",
    "validProposals",
    "invalidProposals",
    "conflictCandidates",
    "conflictPriorityEvaluations",
    "conflictLosses",
    "acceptedMovements",
    "stateHashChecks",
    "identityChecks",
    "siteRoleReads",
    "occupancyReads",
    "adjacencyChecks",
    "reservedSiteClaims",
    "boundarySignalReads",
    "displacedEntities",
    "displacedCells",
    "displacedVacancies",
    "cellGraphDisplacement",
    "vacancyGraphDisplacement",
    "totalGraphDisplacement",
    "adjacentSwaps",
    "vacancyMoves",
    "shortExchanges",
    "rotations",
)


class MovementValidationError(ValueError):
    """Raised for malformed movement state, proposal, or batch records."""


@dataclass(frozen=True)
class Occupant:
    occupant_id: str
    token: str
    kind: str


@dataclass(frozen=True)
class MovementState:
    environment_id: str
    environment_sha256: str
    transition_index: int
    occupancy: tuple[tuple[str, Occupant], ...]

    @property
    def occupant_map(self) -> dict[str, Occupant]:
        return dict(self.occupancy)


@dataclass(frozen=True)
class MovementProposal:
    proposal_id: str
    kind: str
    actor_id: str
    source_site: str
    target_site: str | None
    route: tuple[str, ...]
    rotation_direction: int
    observed_state_sha256: str
    expected_occupants: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ProposalValidation:
    proposal: MovementProposal
    valid: bool
    reason: str
    reserved_sites: tuple[str, ...]
    adjacency_checks: int
    max_individual_displacement: int
    logical_cost: Mapping[str, int]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_payload(domain: str, payload: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + _canonical_json_bytes(payload)
    ).hexdigest()


def empty_ledger() -> dict[str, int]:
    return {field: 0 for field in LEDGER_FIELDS}


def _add_ledger(target: dict[str, int], delta: Mapping[str, int]) -> None:
    if set(delta) != set(LEDGER_FIELDS):
        raise MovementValidationError("cost ledger fields mismatch")
    for field in LEDGER_FIELDS:
        value = int(delta[field])
        if value < 0:
            raise MovementValidationError("cost ledger values must be nonnegative")
        target[field] += value


def _state_body(state: MovementState) -> dict[str, Any]:
    return {
        "schemaVersion": STATE_VERSION,
        "environmentId": state.environment_id,
        "environmentSha256": state.environment_sha256,
        "transitionIndex": state.transition_index,
        "occupancy": [
            {
                "siteId": site_id,
                "occupantId": occupant.occupant_id,
                "token": occupant.token,
                "kind": occupant.kind,
            }
            for site_id, occupant in state.occupancy
        ],
    }


def movement_state_sha256(state: MovementState) -> str:
    return _sha256_payload("E06/S04/state/v1", _state_body(state))


def movement_state_to_dict(state: MovementState) -> dict[str, Any]:
    validate_movement_state_shape(state)
    return {**_state_body(state), "stateSha256": movement_state_sha256(state)}


def canonical_movement_state_bytes(state: MovementState) -> bytes:
    return _canonical_json_bytes(movement_state_to_dict(state))


def initial_movement_state(environment: Environment) -> MovementState:
    """Compile stable identities for every S03 occupiable site."""

    occupancy: list[tuple[str, Occupant]] = []
    for site in environment.occupiable_sites:
        token = environment.initial_state[site.site_id]
        if site.role == "fixed_boundary":
            kind = "fixed_boundary"
            prefix = "fixed"
        elif token == environment.vacancy_label:
            kind = "vacancy"
            prefix = "vacancy"
        else:
            kind = "cell"
            prefix = "cell"
        occupancy.append(
            (
                site.site_id,
                Occupant(
                    occupant_id=f"{prefix}:{environment.environment_id}:{site.site_id}",
                    token=token,
                    kind=kind,
                ),
            )
        )
    state = MovementState(
        environment_id=environment.environment_id,
        environment_sha256=environment_sha256(environment),
        transition_index=0,
        occupancy=tuple(sorted(occupancy)),
    )
    validate_state_against_environment(environment, state)
    return state


def parse_movement_state(raw: Mapping[str, Any]) -> MovementState:
    required = {
        "schemaVersion",
        "environmentId",
        "environmentSha256",
        "transitionIndex",
        "occupancy",
        "stateSha256",
    }
    if set(raw) != required or raw["schemaVersion"] != STATE_VERSION:
        raise MovementValidationError("movement state schema mismatch")
    occupancy = []
    for item in raw["occupancy"]:
        if set(item) != {"siteId", "occupantId", "token", "kind"}:
            raise MovementValidationError("movement occupancy record mismatch")
        occupancy.append(
            (
                str(item["siteId"]),
                Occupant(
                    occupant_id=str(item["occupantId"]),
                    token=str(item["token"]),
                    kind=str(item["kind"]),
                ),
            )
        )
    state = MovementState(
        environment_id=str(raw["environmentId"]),
        environment_sha256=str(raw["environmentSha256"]),
        transition_index=int(raw["transitionIndex"]),
        occupancy=tuple(occupancy),
    )
    validate_movement_state_shape(state)
    if raw["stateSha256"] != movement_state_sha256(state):
        raise MovementValidationError("movement state hash mismatch")
    return state


def validate_movement_state_shape(state: MovementState) -> None:
    if state.transition_index < 0:
        raise MovementValidationError("transition index must be nonnegative")
    site_ids = [site_id for site_id, _ in state.occupancy]
    occupants = [occupant for _, occupant in state.occupancy]
    if tuple(sorted(site_ids)) != tuple(site_ids) or len(site_ids) != len(
        set(site_ids)
    ):
        raise MovementValidationError("movement occupancy sites must be sorted unique")
    occupant_ids = [item.occupant_id for item in occupants]
    if len(occupant_ids) != len(set(occupant_ids)):
        raise MovementValidationError("occupant identities must be unique")
    if any(item.kind not in OCCUPANT_KINDS for item in occupants):
        raise MovementValidationError("invalid occupant kind")


def validate_state_against_environment(
    environment: Environment, state: MovementState
) -> dict[str, Any]:
    validate_movement_state_shape(state)
    if state.environment_id != environment.environment_id:
        raise MovementValidationError("state environment ID mismatch")
    if state.environment_sha256 != environment_sha256(environment):
        raise MovementValidationError("state environment hash mismatch")
    occupiable = {site.site_id: site for site in environment.occupiable_sites}
    occupancy = state.occupant_map
    if set(occupancy) != set(occupiable):
        raise MovementValidationError("state must cover every occupiable site exactly")
    fixed_mismatches = []
    for site_id, site in occupiable.items():
        occupant = occupancy[site_id]
        if occupant.token == environment.obstacle_token:
            raise MovementValidationError(
                "obstacle token cannot occupy a movement site"
            )
        if site.role == "fixed_boundary" and (
            occupant.kind != "fixed_boundary"
            or occupant.token != site.fixed_token
            or not occupant.occupant_id.startswith("fixed:")
        ):
            fixed_mismatches.append(site_id)
        if site.role != "fixed_boundary" and occupant.kind == "fixed_boundary":
            raise MovementValidationError("fixed occupant left its fixed site")
        if (occupant.token == environment.vacancy_label) != (
            occupant.kind == "vacancy"
        ):
            raise MovementValidationError("vacancy token and occupant kind disagree")
    if fixed_mismatches:
        raise MovementValidationError(
            f"fixed boundary occupant mismatch: {sorted(fixed_mismatches)}"
        )
    counts = Counter(item.token for item in occupancy.values())
    kinds = Counter(item.kind for item in occupancy.values())
    return {
        "environmentId": environment.environment_id,
        "stateSha256": movement_state_sha256(state),
        "siteCoverage": True,
        "uniqueOccupants": True,
        "obstaclesExcluded": True,
        "fixedBoundaryPreserved": True,
        "tokenCounts": dict(sorted(counts.items())),
        "occupantKindCounts": dict(sorted(kinds.items())),
        "occupantIds": sorted(item.occupant_id for item in occupancy.values()),
        "success": True,
    }


def state_tokens(state: MovementState) -> dict[str, str]:
    return {site_id: occupant.token for site_id, occupant in state.occupancy}


def _proposal_body(proposal: MovementProposal) -> dict[str, Any]:
    return {
        "schemaVersion": PROPOSAL_VERSION,
        "kind": proposal.kind,
        "actorId": proposal.actor_id,
        "sourceSite": proposal.source_site,
        "targetSite": proposal.target_site,
        "route": list(proposal.route),
        "rotationDirection": proposal.rotation_direction,
        "observedStateSha256": proposal.observed_state_sha256,
        "expectedOccupants": [
            {"siteId": site_id, "occupantId": occupant_id}
            for site_id, occupant_id in proposal.expected_occupants
        ],
    }


def movement_proposal_id(proposal: MovementProposal) -> str:
    return _sha256_payload("E06/S04/proposal/v1", _proposal_body(proposal))


def proposal_to_dict(proposal: MovementProposal) -> dict[str, Any]:
    return {**_proposal_body(proposal), "proposalId": proposal.proposal_id}


def canonical_proposal_bytes(proposal: MovementProposal) -> bytes:
    return _canonical_json_bytes(proposal_to_dict(proposal))


def make_proposal(
    state: MovementState,
    kind: str,
    route: Sequence[str],
    *,
    rotation_direction: int = 0,
    observed_state_sha256: str | None = None,
    actor_id: str | None = None,
    expected_occupants: Mapping[str, str] | None = None,
) -> MovementProposal:
    """Create an engine-authenticated proposal envelope from an intent route."""

    route_tuple = tuple(str(item) for item in route)
    occupancy = state.occupant_map
    source = route_tuple[0] if route_tuple else ""
    if kind == "rotation" and route_tuple:
        if rotation_direction == 1:
            target = route_tuple[1] if len(route_tuple) > 1 else None
        elif rotation_direction == -1:
            target = route_tuple[-1] if len(route_tuple) > 1 else None
        else:
            target = None
    else:
        target = route_tuple[-1] if len(route_tuple) > 1 else None
    if actor_id is None:
        actor_id = occupancy[source].occupant_id if source in occupancy else "unknown"
    if expected_occupants is None:
        expected_occupants = {
            site_id: occupancy[site_id].occupant_id
            for site_id in route_tuple
            if site_id in occupancy
        }
    proposal_without_id = MovementProposal(
        proposal_id="",
        kind=str(kind),
        actor_id=str(actor_id),
        source_site=source,
        target_site=target,
        route=route_tuple,
        rotation_direction=int(rotation_direction),
        observed_state_sha256=(
            movement_state_sha256(state)
            if observed_state_sha256 is None
            else str(observed_state_sha256)
        ),
        expected_occupants=tuple(
            sorted((str(key), str(value)) for key, value in expected_occupants.items())
        ),
    )
    return MovementProposal(
        **{
            **proposal_without_id.__dict__,
            "proposal_id": movement_proposal_id(proposal_without_id),
        }
    )


def parse_proposal(raw: Mapping[str, Any]) -> MovementProposal:
    required = {
        "schemaVersion",
        "proposalId",
        "kind",
        "actorId",
        "sourceSite",
        "targetSite",
        "route",
        "rotationDirection",
        "observedStateSha256",
        "expectedOccupants",
    }
    if set(raw) != required or raw["schemaVersion"] != PROPOSAL_VERSION:
        raise MovementValidationError("movement proposal schema mismatch")
    proposal = MovementProposal(
        proposal_id=str(raw["proposalId"]),
        kind=str(raw["kind"]),
        actor_id=str(raw["actorId"]),
        source_site=str(raw["sourceSite"]),
        target_site=(None if raw["targetSite"] is None else str(raw["targetSite"])),
        route=tuple(str(item) for item in raw["route"]),
        rotation_direction=int(raw["rotationDirection"]),
        observed_state_sha256=str(raw["observedStateSha256"]),
        expected_occupants=tuple(
            (str(item["siteId"]), str(item["occupantId"]))
            for item in raw["expectedOccupants"]
        ),
    )
    if proposal.proposal_id != movement_proposal_id(proposal):
        raise MovementValidationError("movement proposal ID mismatch")
    if tuple(sorted(proposal.expected_occupants)) != proposal.expected_occupants:
        raise MovementValidationError("expected occupants must be canonical")
    return proposal


def _shortest_distance(
    environment: Environment,
    source: str,
    target: str,
    maximum: int,
    *,
    _neighbors: Mapping[str, Sequence[str]] | None = None,
) -> int | None:
    neighbors = neighbor_map(environment) if _neighbors is None else _neighbors
    if source not in neighbors or target not in neighbors:
        return None
    queue = deque([(source, 0)])
    seen = {source}
    while queue:
        current, distance = queue.popleft()
        if current == target:
            return distance
        if distance >= maximum:
            continue
        for other in neighbors[current]:
            if other not in seen:
                seen.add(other)
                queue.append((other, distance + 1))
    return None


def _invalid_validation(
    proposal: MovementProposal,
    reason: str,
    ledger: dict[str, int],
    reserved_sites: Sequence[str] = (),
    adjacency_checks: int = 0,
) -> ProposalValidation:
    ledger["invalidProposals"] = 1
    return ProposalValidation(
        proposal=proposal,
        valid=False,
        reason=reason,
        reserved_sites=tuple(reserved_sites),
        adjacency_checks=adjacency_checks,
        max_individual_displacement=0,
        logical_cost=ledger,
    )


def validate_proposal(
    environment: Environment,
    state: MovementState,
    proposal: MovementProposal,
    *,
    _state_sha256: str | None = None,
    _neighbors: Mapping[str, Sequence[str]] | None = None,
) -> ProposalValidation:
    """Validate one proposal against the immutable pre-batch snapshot."""

    ledger = empty_ledger()
    ledger["submittedProposals"] = 1
    if proposal.kind in DEFERRED_KINDS:
        return _invalid_validation(proposal, f"deferred_kind:{proposal.kind}", ledger)
    if proposal.kind not in ENABLED_KINDS:
        return _invalid_validation(proposal, "unknown_kind", ledger)
    ledger["stateHashChecks"] = 1
    state_sha256 = (
        movement_state_sha256(state) if _state_sha256 is None else _state_sha256
    )
    if proposal.observed_state_sha256 != state_sha256:
        return _invalid_validation(proposal, "stale_observation", ledger)
    route = proposal.route
    if not route or proposal.source_site != route[0]:
        return _invalid_validation(proposal, "source_route_mismatch", ledger)
    if proposal.kind == "rotation":
        expected_target = (
            route[1]
            if proposal.rotation_direction == 1 and len(route) > 1
            else route[-1]
            if proposal.rotation_direction == -1 and len(route) > 1
            else None
        )
    else:
        expected_target = route[-1] if len(route) > 1 else None
    if proposal.target_site != expected_target:
        return _invalid_validation(proposal, "target_route_mismatch", ledger)
    if len(route) != len(set(route)):
        return _invalid_validation(proposal, "route_repeats_site", ledger)

    sites = {site.site_id: site for site in environment.sites}
    occupancy = state.occupant_map
    for site_id in route:
        if site_id not in sites:
            return _invalid_validation(proposal, "unknown_site", ledger, route)
        ledger["siteRoleReads"] += 1
        if sites[site_id].role == "obstacle":
            return _invalid_validation(proposal, "obstacle_site", ledger, route)
        if sites[site_id].role == "fixed_boundary":
            return _invalid_validation(proposal, "fixed_boundary_site", ledger, route)

    expected = dict(proposal.expected_occupants)
    if set(expected) != set(route):
        return _invalid_validation(
            proposal, "expected_occupants_must_cover_route", ledger, route
        )
    for site_id in route:
        ledger["occupancyReads"] += 1
        ledger["identityChecks"] += 1
        if occupancy[site_id].occupant_id != expected[site_id]:
            return _invalid_validation(
                proposal, "stale_resource_identity", ledger, route
            )
    if occupancy[route[0]].occupant_id != proposal.actor_id:
        return _invalid_validation(proposal, "actor_identity_mismatch", ledger, route)

    edge_set = set(environment.edges)

    def is_edge(first: str, second: str) -> bool:
        ledger["adjacencyChecks"] += 1
        return tuple(sorted((first, second))) in edge_set

    max_displacement = 0
    if proposal.kind in {"adjacent_swap", "vacancy_move"}:
        if len(route) != 2:
            return _invalid_validation(proposal, "edge_move_requires_two_sites", ledger)
        if not is_edge(route[0], route[1]):
            return _invalid_validation(
                proposal, "non_adjacent_edge", ledger, route, ledger["adjacencyChecks"]
            )
        source, target = occupancy[route[0]], occupancy[route[1]]
        if proposal.kind == "adjacent_swap":
            if source.kind == "vacancy" or target.kind == "vacancy":
                return _invalid_validation(
                    proposal, "adjacent_swap_requires_two_cells", ledger, route, 1
                )
        elif source.kind == "vacancy" or target.kind != "vacancy":
            return _invalid_validation(
                proposal, "vacancy_move_requires_cell_to_vacancy", ledger, route, 1
            )
        max_displacement = 1
    elif proposal.kind == "short_exchange":
        if len(route) != 3:
            return _invalid_validation(
                proposal, "short_exchange_requires_three_site_path", ledger
            )
        if not is_edge(route[0], route[1]) or not is_edge(route[1], route[2]):
            return _invalid_validation(
                proposal,
                "short_exchange_path_not_contiguous",
                ledger,
                route,
                ledger["adjacencyChecks"],
            )
        if (
            _shortest_distance(
                environment,
                route[0],
                route[2],
                2,
                _neighbors=_neighbors,
            )
            != 2
        ):
            return _invalid_validation(
                proposal, "short_exchange_endpoints_not_distance_two", ledger, route, 2
            )
        if (
            occupancy[route[0]].kind == "vacancy"
            or occupancy[route[2]].kind == "vacancy"
        ):
            return _invalid_validation(
                proposal, "short_exchange_requires_cell_endpoints", ledger, route, 2
            )
        max_displacement = 2
    else:
        if not 3 <= len(route) <= 6:
            return _invalid_validation(
                proposal, "rotation_cycle_length_must_be_3_to_6", ledger
            )
        if proposal.rotation_direction not in {-1, 1}:
            return _invalid_validation(
                proposal, "rotation_direction_must_be_plus_or_minus_one", ledger
            )
        pairs = list(zip(route, route[1:])) + [(route[-1], route[0])]
        if not all(is_edge(first, second) for first, second in pairs):
            return _invalid_validation(
                proposal,
                "rotation_route_not_closed_cycle",
                ledger,
                route,
                ledger["adjacencyChecks"],
            )
        if any(occupancy[site_id].kind == "vacancy" for site_id in route):
            return _invalid_validation(
                proposal,
                "rotation_requires_fully_occupied_cycle",
                ledger,
                route,
                len(route),
            )
        max_displacement = 1

    ledger["validProposals"] = 1
    ledger["conflictCandidates"] = 1
    ledger["reservedSiteClaims"] = len(route)
    return ProposalValidation(
        proposal=proposal,
        valid=True,
        reason="valid",
        reserved_sites=tuple(route),
        adjacency_checks=ledger["adjacencyChecks"],
        max_individual_displacement=max_displacement,
        logical_cost=ledger,
    )


def conflict_priority_uint64(batch_id: str, proposal_id: str) -> int:
    digest = hashlib.sha256(
        b"E06/S04/conflict/v1\x00"
        + batch_id.encode("ascii")
        + b"\x00"
        + proposal_id.encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def conflict_order_key(
    batch_id: str,
    proposal: MovementProposal,
    *,
    priority_override: int | None = None,
) -> tuple[int, str]:
    priority = (
        conflict_priority_uint64(batch_id, proposal.proposal_id)
        if priority_override is None
        else int(priority_override)
    )
    return priority, proposal.proposal_id


def movement_batch_id(
    state: MovementState, proposals: Sequence[MovementProposal], batch_nonce: str
) -> str:
    payload = {
        "stateSha256": movement_state_sha256(state),
        "proposalIds": sorted(item.proposal_id for item in proposals),
        "batchNonce": str(batch_nonce),
    }
    return _sha256_payload("E06/S04/batch/v1", payload)


def _movement_commit_delta(
    proposal: MovementProposal, occupancy: Mapping[str, Occupant]
) -> dict[str, int]:
    delta = empty_ledger()
    route = proposal.route
    delta["acceptedMovements"] = 1
    if proposal.kind == "adjacent_swap":
        delta["adjacentSwaps"] = 1
        distances = {route[0]: 1, route[1]: 1}
    elif proposal.kind == "vacancy_move":
        delta["vacancyMoves"] = 1
        distances = {route[0]: 1, route[1]: 1}
    elif proposal.kind == "short_exchange":
        delta["shortExchanges"] = 1
        distances = {route[0]: 2, route[2]: 2}
    else:
        delta["rotations"] = 1
        distances = {site_id: 1 for site_id in route}
    for site_id, distance in distances.items():
        occupant = occupancy[site_id]
        delta["displacedEntities"] += 1
        delta["totalGraphDisplacement"] += distance
        if occupant.kind == "vacancy":
            delta["displacedVacancies"] += 1
            delta["vacancyGraphDisplacement"] += distance
        else:
            delta["displacedCells"] += 1
            delta["cellGraphDisplacement"] += distance
    return delta


def _apply_disjoint(
    state: MovementState, accepted: Sequence[MovementProposal]
) -> MovementState:
    before = state.occupant_map
    after = dict(before)
    for proposal in accepted:
        route = proposal.route
        if proposal.kind in {"adjacent_swap", "vacancy_move", "short_exchange"}:
            first, second = route[0], route[-1]
            after[first], after[second] = before[second], before[first]
        else:
            direction = proposal.rotation_direction
            for index, site_id in enumerate(route):
                target = route[(index + direction) % len(route)]
                after[target] = before[site_id]
    return MovementState(
        environment_id=state.environment_id,
        environment_sha256=state.environment_sha256,
        transition_index=state.transition_index + 1,
        occupancy=tuple(sorted(after.items())),
    )


def _invariant_comparison(
    environment: Environment, before: MovementState, after: MovementState
) -> dict[str, Any]:
    before_audit = validate_state_against_environment(environment, before)
    after_audit = validate_state_against_environment(environment, after)
    checks = {
        "siteCoveragePreserved": before_audit["siteCoverage"]
        and after_audit["siteCoverage"],
        "identityConserved": before_audit["occupantIds"] == after_audit["occupantIds"],
        "compositionConserved": before_audit["tokenCounts"]
        == after_audit["tokenCounts"],
        "occupantKindsConserved": before_audit["occupantKindCounts"]
        == after_audit["occupantKindCounts"],
        "obstaclesExcluded": before_audit["obstaclesExcluded"]
        and after_audit["obstaclesExcluded"],
        "fixedBoundaryPreserved": before_audit["fixedBoundaryPreserved"]
        and after_audit["fixedBoundaryPreserved"],
    }
    return {
        "checks": checks,
        "before": before_audit,
        "after": after_audit,
        "success": all(checks.values()),
    }


def _decision_record(
    validation: ProposalValidation,
    outcome: str,
    priority: int | None,
    conflicts_with: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "proposalId": validation.proposal.proposal_id,
        "kind": validation.proposal.kind,
        "validation": "valid" if validation.valid else "invalid",
        "reason": validation.reason if not validation.valid else outcome,
        "outcome": outcome,
        "conflictPriorityUint64": priority,
        "reservedSites": list(validation.reserved_sites),
        "conflictsWithAcceptedProposalIds": sorted(conflicts_with),
        "maxIndividualGraphDisplacement": validation.max_individual_displacement,
        "logicalValidationCost": dict(validation.logical_cost),
    }


def resolve_batch(
    environment: Environment,
    state: MovementState,
    proposals: Sequence[MovementProposal],
    *,
    batch_nonce: str,
) -> dict[str, Any]:
    """Resolve one simultaneous proposal batch independently of input order."""

    validate_state_against_environment(environment, state)
    proposal_ids = [item.proposal_id for item in proposals]
    if len(proposal_ids) != len(set(proposal_ids)):
        raise MovementValidationError("batch proposal IDs must be unique")
    batch_id = movement_batch_id(state, proposals, batch_nonce)
    validations = [
        validate_proposal(environment, state, proposal)
        for proposal in sorted(proposals, key=lambda item: item.proposal_id)
    ]
    ledger = empty_ledger()
    for validation in validations:
        _add_ledger(ledger, validation.logical_cost)

    valid = [item for item in validations if item.valid]
    priorities = {
        item.proposal.proposal_id: conflict_priority_uint64(
            batch_id, item.proposal.proposal_id
        )
        for item in valid
    }
    ledger["conflictPriorityEvaluations"] = len(valid)
    ordered = sorted(
        valid,
        key=lambda item: (
            priorities[item.proposal.proposal_id],
            item.proposal.proposal_id,
        ),
    )
    accepted_validations: list[ProposalValidation] = []
    claimed_by: dict[str, str] = {}
    decisions: list[dict[str, Any]] = []
    for validation in ordered:
        conflicts = sorted(
            {
                claimed_by[site_id]
                for site_id in validation.reserved_sites
                if site_id in claimed_by
            }
        )
        if conflicts:
            ledger["conflictLosses"] += 1
            decisions.append(
                _decision_record(
                    validation,
                    "conflict_lost",
                    priorities[validation.proposal.proposal_id],
                    conflicts,
                )
            )
            continue
        accepted_validations.append(validation)
        for site_id in validation.reserved_sites:
            claimed_by[site_id] = validation.proposal.proposal_id
        decisions.append(
            _decision_record(
                validation,
                "accepted",
                priorities[validation.proposal.proposal_id],
            )
        )
    for validation in validations:
        if not validation.valid:
            decisions.append(_decision_record(validation, "rejected", None))

    accepted = [item.proposal for item in accepted_validations]
    before_occupancy = state.occupant_map
    for proposal in accepted:
        _add_ledger(ledger, _movement_commit_delta(proposal, before_occupancy))
    post_state = _apply_disjoint(state, accepted)
    invariants = _invariant_comparison(environment, state, post_state)
    if not invariants["success"]:
        raise MovementValidationError("post-commit invariant failure")
    if ledger["boundarySignalReads"] != 0:
        raise MovementValidationError(
            "movement legality exceeded boundary-signal budget"
        )

    body = {
        "schemaVersion": BATCH_VERSION,
        "batchId": batch_id,
        "batchNonce": str(batch_nonce),
        "environmentId": environment.environment_id,
        "preState": movement_state_to_dict(state),
        "proposals": [
            proposal_to_dict(item)
            for item in sorted(proposals, key=lambda item: item.proposal_id)
        ],
        "resolutionOrder": [item.proposal.proposal_id for item in ordered],
        "decisions": sorted(decisions, key=lambda item: item["proposalId"]),
        "acceptedProposalIds": sorted(item.proposal_id for item in accepted),
        "postState": movement_state_to_dict(post_state),
        "costLedger": ledger,
        "invariants": invariants,
        "boundarySignalBudget": {
            "allowedReadsForMovementLegality": 0,
            "observedReads": ledger["boundarySignalReads"],
            "preserved": ledger["boundarySignalReads"] == 0,
            "note": "Site-role enforcement is engine metadata, not a policy boundary signal.",
        },
    }
    return {
        **body,
        "transitionSha256": _sha256_payload("E06/S04/transition/v1", body),
    }


def canonical_batch_result_bytes(result: Mapping[str, Any]) -> bytes:
    return _canonical_json_bytes(dict(result))


def replay_batch_result(
    environment: Environment, result: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "schemaVersion",
        "batchId",
        "batchNonce",
        "environmentId",
        "preState",
        "proposals",
        "resolutionOrder",
        "decisions",
        "acceptedProposalIds",
        "postState",
        "costLedger",
        "invariants",
        "boundarySignalBudget",
        "transitionSha256",
    }
    if set(result) != required or result["schemaVersion"] != BATCH_VERSION:
        raise MovementValidationError("batch result schema mismatch")
    body = {key: result[key] for key in required - {"transitionSha256"}}
    if result["transitionSha256"] != _sha256_payload("E06/S04/transition/v1", body):
        raise MovementValidationError("batch transition hash mismatch")
    state = parse_movement_state(result["preState"])
    proposals = [parse_proposal(item) for item in result["proposals"]]
    replayed = resolve_batch(
        environment, state, proposals, batch_nonce=str(result["batchNonce"])
    )
    if canonical_batch_result_bytes(replayed) != canonical_batch_result_bytes(result):
        raise MovementValidationError("batch replay mismatch")
    return replayed


def inverse_proposal(
    post_state: MovementState, proposal: MovementProposal
) -> MovementProposal:
    if proposal.kind not in ENABLED_KINDS:
        raise MovementValidationError("cannot invert a disabled movement kind")
    if proposal.kind == "rotation":
        return make_proposal(
            post_state,
            proposal.kind,
            proposal.route,
            rotation_direction=-proposal.rotation_direction,
        )
    return make_proposal(post_state, proposal.kind, tuple(reversed(proposal.route)))
