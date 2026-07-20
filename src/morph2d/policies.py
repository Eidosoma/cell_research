"""Priced local spatial Algotypes for E06 S05.

Policies receive only an immutable, explicitly budgeted projection of a
pre-activation state.  Opaque candidate handles separate policy intent from
S04's authenticated proposal envelope: site IDs, routes, occupant identities,
state hashes, conflict priorities, analysis labels, and completion evaluators
remain engine-only.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .environments import Environment, boundary_observation, neighbor_map
from .grammar import ConstraintKind, Priority, RelationalGrammar
from .movements import (
    ENABLED_KINDS,
    MovementProposal,
    MovementState,
    make_proposal,
    movement_state_sha256,
    resolve_batch,
    validate_proposal,
)


POLICY_CATALOG_VERSION = "e06.s05.policy-catalog.v1"
OBSERVATION_VERSION = "e06.s05.policy-observation.v1"
DECISION_VERSION = "e06.s05.policy-decision.v1"
EVENT_VERSION = "e06.s05.policy-event.v1"
RELATION_PROFILE_VERSION = "e06.s05.local-relation-profile.v1"
SUPPORTED_STRATEGIES = {
    "greedy_neighbor_satisfaction",
    "boundary_seeking",
    "gradient_following",
    "exploration",
    "memory_based_recovery",
    "conflict_avoidance",
}
FEATURES_BY_STRATEGY = {
    "greedy_neighbor_satisfaction": {"actor_token", "local_relation_delta"},
    "boundary_seeking": {"actor_token", "natural_boundary_delta"},
    "gradient_following": {"gradient_delta"},
    "exploration": {"counter_exploration_index"},
    "memory_based_recovery": {
        "actor_token",
        "local_relation_delta",
        "own_memory",
    },
    "conflict_avoidance": {
        "actor_token",
        "local_relation_delta",
        "lagged_conflict_count",
        "movement_cost",
    },
}
OBSERVATION_LEDGER_FIELDS = (
    "actorTokenReads",
    "localTokenReads",
    "candidateDescriptorReads",
    "boundarySignalReads",
    "gradientSignalReads",
    "laggedConflictReads",
    "memoryReads",
    "counterRandomDraws",
    "affordanceTopologyReads",
    "affordanceSiteRoleReads",
    "utilityEvaluations",
    "comparisonOperations",
    "communicatedBitsUpperBound",
    "persistentMemoryBits",
)
FORBIDDEN_OBSERVATION_KEYS = {
    "actorid",
    "analysislabel",
    "batchid",
    "boundarytags",
    "conflictpriority",
    "conflictpriorityuint64",
    "expectedoccupants",
    "fixedboundaryneighbors",
    "fixedtoken",
    "globalaudit",
    "independents01globalaudit",
    "isfixedboundary",
    "localgrammar",
    "obstaclecontacts",
    "occupantid",
    "observedstatesha256",
    "proposalid",
    "route",
    "s01globalcompletionaudit",
    "siteid",
    "statesha256",
    "targetsite",
    "transitionsha256",
}


class PolicyValidationError(ValueError):
    """Raised when a policy, observation, memory, or decision is invalid."""


@dataclass(frozen=True)
class PolicyDefinition:
    policy_id: str
    strategy: str
    allowed_movement_kinds: tuple[str, ...]
    max_candidates: int
    observation_features: tuple[str, ...]
    information_budget_max_bits: int
    relation_profile_required: bool
    target_specificity: str
    parameters: Mapping[str, Any]
    complexity: Mapping[str, int]
    intended_behavior: str
    non_guarantees: tuple[str, ...]


@dataclass(frozen=True)
class RelationProfile:
    grammar_id: str
    target_id: str
    actor_neighbor_weights: tuple[tuple[str, str, int], ...]
    projected_constraint_ids: tuple[str, ...]
    excluded_constraint_ids: tuple[str, ...]

    @property
    def weight_map(self) -> dict[tuple[str, str], int]:
        return {
            (actor, neighbor): weight
            for actor, neighbor, weight in self.actor_neighbor_weights
        }


@dataclass(frozen=True)
class PolicyMemory:
    best_local_utility: int
    frustration: int


@dataclass(frozen=True)
class CandidateAffordance:
    candidate_key: str
    proposal: MovementProposal
    target_site: str
    movement_cost: int
    reservation_size: int


@dataclass(frozen=True)
class PolicyObservation:
    policy_id: str
    strategy: str
    payload: Mapping[str, Any]
    budget: Mapping[str, int]
    source_state_sha256: str
    observation_sha256: str


@dataclass(frozen=True)
class ObservationBuild:
    observation: PolicyObservation
    candidate_map: Mapping[str, CandidateAffordance]


@dataclass(frozen=True)
class PolicyDecision:
    policy_id: str
    action: str
    selected_candidate_key: str | None
    reason: str
    decision_score: int | None


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_payload(domain: str, payload: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + _canonical_json_bytes(payload)
    ).hexdigest()


def empty_observation_ledger() -> dict[str, int]:
    return {field: 0 for field in OBSERVATION_LEDGER_FIELDS}


def _bits_for_count(maximum_inclusive: int) -> int:
    return max(1, math.ceil(math.log2(maximum_inclusive + 1)))


def compile_relation_profile(grammar: RelationalGrammar) -> RelationProfile:
    """Project only S02 edge/neighbor constraints into actor-local contacts.

    The projection deliberately excludes boundary, motif, symmetry, whole-grid
    interval totals, acceptance, and every S01 audit.  Positive minimum edge
    requirements become positive local contact weights.  Hard zero-maximum
    edges become amplified local penalties.  Neighbor-count rules contribute
    only to their declared subject token.
    """

    weights: dict[tuple[str, str], int] = {}
    projected: list[str] = []
    excluded: list[str] = []
    for constraint in grammar.constraints:
        parameters = constraint.parameters
        base_weight = max(1, int(round(constraint.weight)))
        if constraint.kind == ConstraintKind.EDGE_COUNT:
            tokens = tuple(str(item) for item in parameters["tokens"])
            minimum = float(parameters["min"])
            maximum = float(parameters["max"])
            if minimum > 0:
                contribution = base_weight
            elif maximum == 0 and constraint.priority == Priority.HARD:
                contribution = -4 * base_weight
            else:
                excluded.append(constraint.constraint_id)
                continue
            first, second = tokens
            weights[(first, second)] = weights.get((first, second), 0) + contribution
            weights[(second, first)] = weights.get((second, first), 0) + contribution
            projected.append(constraint.constraint_id)
        elif constraint.kind == ConstraintKind.NEIGHBOR_COUNT:
            if float(parameters["min"]) <= 0:
                excluded.append(constraint.constraint_id)
                continue
            actor = str(parameters["subject"])
            for neighbor in parameters["neighborTokens"]:
                key = (actor, str(neighbor))
                weights[key] = weights.get(key, 0) + base_weight
            projected.append(constraint.constraint_id)
        else:
            excluded.append(constraint.constraint_id)
    return RelationProfile(
        grammar_id=grammar.grammar_id,
        target_id=grammar.target_id,
        actor_neighbor_weights=tuple(
            (actor, neighbor, weight)
            for (actor, neighbor), weight in sorted(weights.items())
        ),
        projected_constraint_ids=tuple(sorted(projected)),
        excluded_constraint_ids=tuple(sorted(excluded)),
    )


def relation_profile_to_dict(profile: RelationProfile) -> dict[str, Any]:
    return {
        "schemaVersion": RELATION_PROFILE_VERSION,
        "grammarId": profile.grammar_id,
        "targetId": profile.target_id,
        "actorNeighborWeights": [
            {"actorToken": actor, "neighborToken": neighbor, "weight": weight}
            for actor, neighbor, weight in profile.actor_neighbor_weights
        ],
        "projectedConstraintIds": list(profile.projected_constraint_ids),
        "excludedConstraintIds": list(profile.excluded_constraint_ids),
        "claimBoundary": "Actor-local contact utility only; not the S02 whole-grid score or S01 completion.",
    }


def local_contact_utility(
    environment: Environment,
    state: MovementState,
    actor_id: str,
    profile: RelationProfile,
    *,
    _neighbors: Mapping[str, Sequence[str]] | None = None,
    _occupancy: Mapping[str, Any] | None = None,
    _locations: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    occupancy = state.occupant_map if _occupancy is None else _occupancy
    locations = (
        {occupant.occupant_id: site_id for site_id, occupant in occupancy.items()}
        if _locations is None
        else _locations
    )
    if actor_id not in locations:
        raise PolicyValidationError("actor identity is absent from movement state")
    site_id = locations[actor_id]
    actor = occupancy[site_id]
    if actor.kind != "cell":
        raise PolicyValidationError("only cell occupants can execute an Algotype")
    weights = profile.weight_map
    neighbors = (neighbor_map(environment) if _neighbors is None else _neighbors)[
        site_id
    ]
    utility = sum(
        weights.get((actor.token, occupancy[other].token), 0) for other in neighbors
    )
    return utility, len(neighbors)


def _counter_u64(
    decision_key: str,
    policy_id: str,
    actor_id: str,
    activation_index: int,
    draw_index: int,
) -> int:
    if activation_index < 0 or draw_index < 0:
        raise PolicyValidationError("counter indices must be nonnegative")
    payload = (
        b"E06/S05/policy-draw/v1\x00"
        + str(decision_key).encode("utf-8")
        + b"\x00"
        + policy_id.encode("ascii")
        + b"\x00"
        + actor_id.encode("utf-8")
        + b"\x00"
        + activation_index.to_bytes(8, "big")
        + draw_index.to_bytes(4, "big")
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _counter_bounded(
    decision_key: str,
    policy_id: str,
    actor_id: str,
    activation_index: int,
    bound: int,
) -> int:
    if bound <= 0:
        raise PolicyValidationError("bounded draw requires a positive bound")
    limit = ((1 << 64) // bound) * bound
    draw_index = 0
    while True:
        value = _counter_u64(
            decision_key, policy_id, actor_id, activation_index, draw_index
        )
        if value < limit:
            return value % bound
        draw_index += 1


def _movement_cost(proposal: MovementProposal) -> int:
    """S04 single-proposal ``totalGraphDisplacement`` for a legal intent."""

    if proposal.kind in {"adjacent_swap", "vacancy_move"}:
        return 2
    if proposal.kind == "short_exchange":
        return 4
    return len(proposal.route)


def _actor_target(proposal: MovementProposal) -> str:
    if proposal.target_site is None:
        raise PolicyValidationError("candidate proposal has no target")
    return proposal.target_site


def _simple_cycles_from_actor(
    environment: Environment,
    source: str,
    maximum_length: int = 6,
    *,
    _neighbors: Mapping[str, Sequence[str]] | None = None,
) -> tuple[tuple[str, ...], ...]:
    neighbors = neighbor_map(environment) if _neighbors is None else _neighbors
    cycles: set[tuple[str, ...]] = set()

    def visit(path: tuple[str, ...]) -> None:
        current = path[-1]
        if len(path) >= 3 and source in neighbors[current]:
            reverse = (source, *reversed(path[1:]))
            cycles.add(min(path, reverse))
        if len(path) >= maximum_length:
            return
        for other in neighbors[current]:
            if other == source or other in path:
                continue
            visit((*path, other))

    visit((source,))
    return tuple(sorted(cycles))


def enumerate_candidate_affordances(
    environment: Environment,
    state: MovementState,
    actor_id: str,
    definition: PolicyDefinition,
    *,
    _state_sha256: str | None = None,
    _neighbors: Mapping[str, Sequence[str]] | None = None,
) -> tuple[tuple[CandidateAffordance, ...], dict[str, int]]:
    """Build legal S04 candidates while keeping their routes engine-private."""

    occupancy = state.occupant_map
    locations = {
        occupant.occupant_id: site_id for site_id, occupant in occupancy.items()
    }
    if actor_id not in locations:
        raise PolicyValidationError("unknown actor")
    source = locations[actor_id]
    if occupancy[source].kind != "cell":
        raise PolicyValidationError("Algotype actor must be a cell")
    allowed = set(definition.allowed_movement_kinds)
    neighbors = neighbor_map(environment) if _neighbors is None else _neighbors
    state_sha256 = (
        movement_state_sha256(state) if _state_sha256 is None else _state_sha256
    )
    raw: list[MovementProposal] = []
    topology_reads = 0
    role_reads = 0

    if allowed & {"adjacent_swap", "vacancy_move"}:
        for target in neighbors[source]:
            topology_reads += 1
            role_reads += 1
            kind = (
                "vacancy_move"
                if occupancy[target].kind == "vacancy"
                else "adjacent_swap"
            )
            if kind not in allowed:
                continue
            proposal = make_proposal(
                state,
                kind,
                (source, target),
                observed_state_sha256=state_sha256,
            )
            if validate_proposal(
                environment,
                state,
                proposal,
                _state_sha256=state_sha256,
                _neighbors=neighbors,
            ).valid:
                raw.append(proposal)

    if "short_exchange" in allowed:
        for middle in neighbors[source]:
            topology_reads += 1
            role_reads += 1
            for target in neighbors[middle]:
                topology_reads += 1
                role_reads += 1
                if target == source:
                    continue
                proposal = make_proposal(
                    state,
                    "short_exchange",
                    (source, middle, target),
                    observed_state_sha256=state_sha256,
                )
                if validate_proposal(
                    environment,
                    state,
                    proposal,
                    _state_sha256=state_sha256,
                    _neighbors=neighbors,
                ).valid:
                    raw.append(proposal)

    if "rotation" in allowed:
        for cycle in _simple_cycles_from_actor(
            environment, source, _neighbors=neighbors
        ):
            topology_reads += len(cycle)
            role_reads += len(cycle)
            for direction in (-1, 1):
                proposal = make_proposal(
                    state,
                    "rotation",
                    cycle,
                    rotation_direction=direction,
                    observed_state_sha256=state_sha256,
                )
                if validate_proposal(
                    environment,
                    state,
                    proposal,
                    _state_sha256=state_sha256,
                    _neighbors=neighbors,
                ).valid:
                    raw.append(proposal)

    unique = {proposal.proposal_id: proposal for proposal in raw}

    # Preserve at least one affordance from every available, declared movement
    # kind before applying the candidate cap.  The final activation-local
    # ordering is a state-keyed permutation: c00 therefore does not
    # systematically reveal a movement kind or geometric direction.
    canonical = sorted(
        unique.values(),
        key=lambda item: (item.kind, item.route, item.rotation_direction),
    )
    ranked = sorted(
        canonical,
        key=lambda item: _sha256_payload(
            "E06/S05/opaque-candidate-rank/v1",
            {
                "policyId": definition.policy_id,
                "stateSha256": state_sha256,
                "proposalId": item.proposal_id,
            },
        ),
    )
    required = []
    for kind in definition.allowed_movement_kinds:
        first = next((item for item in ranked if item.kind == kind), None)
        if first is not None:
            required.append(first)
    selected_ids = {item.proposal_id for item in required}
    selected = (
        required
        + [item for item in ranked if item.proposal_id not in selected_ids][
            : definition.max_candidates - len(required)
        ]
    )
    ordered = sorted(
        selected,
        key=lambda item: _sha256_payload(
            "E06/S05/opaque-candidate-order/v1",
            {
                "policyId": definition.policy_id,
                "stateSha256": state_sha256,
                "proposalId": item.proposal_id,
            },
        ),
    )
    candidates = tuple(
        CandidateAffordance(
            candidate_key=f"c{index:02d}",
            proposal=proposal,
            target_site=_actor_target(proposal),
            movement_cost=_movement_cost(proposal),
            reservation_size=len(proposal.route),
        )
        for index, proposal in enumerate(ordered)
    )
    return candidates, {
        "affordanceTopologyReads": topology_reads,
        "affordanceSiteRoleReads": role_reads,
    }


def _preview_state(
    environment: Environment, state: MovementState, proposal: MovementProposal
) -> MovementState:
    # Candidate enumeration has already passed the proposal through the
    # canonical S04 validator.  A singleton valid batch is conflict-free, so
    # constructing its occupant permutation directly is semantically identical
    # to ``resolve_batch`` while avoiding repeated hashes, ledgers, and full
    # invariant serialization for every actor-local utility preview.
    before = state.occupant_map
    after = dict(before)
    route = proposal.route
    if proposal.kind in {"adjacent_swap", "vacancy_move", "short_exchange"}:
        first, second = route[0], route[-1]
        after[first], after[second] = before[second], before[first]
    elif proposal.kind == "rotation":
        for index, site_id in enumerate(route):
            target = route[(index + proposal.rotation_direction) % len(route)]
            after[target] = before[site_id]
    else:  # pragma: no cover - candidates are limited by the frozen S04 set.
        raise PolicyValidationError("candidate preview received unknown movement kind")
    return MovementState(
        environment_id=state.environment_id,
        environment_sha256=state.environment_sha256,
        transition_index=state.transition_index + 1,
        occupancy=tuple(sorted(after.items())),
    )


def _natural_boundary_level(environment: Environment, site_id: str) -> int:
    observation = boundary_observation(environment, site_id)
    return len(observation["boundaryTags"])


def _observation_body(
    policy_id: str,
    strategy: str,
    payload: Mapping[str, Any],
    budget: Mapping[str, int],
    source_state_sha256: str,
) -> dict[str, Any]:
    return {
        "schemaVersion": OBSERVATION_VERSION,
        "policyId": policy_id,
        "strategy": strategy,
        "payload": payload,
        "budget": budget,
        "sourceStateReference": {
            "availability": "engine_only_not_disclosed_to_policy",
            "hashCommitted": True,
        },
        "permissionAudit": {
            "forbiddenFieldsAbsent": True,
            "authenticationFieldsDisclosed": False,
            "globalEvaluationFieldsDisclosed": False,
        },
        "sourceStateSha256EngineAudit": source_state_sha256,
    }


def _payload_key_audit(value: Any, path: str = "payload") -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = "".join(
                character for character in str(key).lower() if character.isalnum()
            )
            if normalized in FORBIDDEN_OBSERVATION_KEYS:
                violations.append(f"{path}.{key}")
            violations.extend(_payload_key_audit(item, f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            violations.extend(_payload_key_audit(item, f"{path}[{index}]"))
    return violations


def validate_policy_payload(payload: Mapping[str, Any]) -> None:
    violations = _payload_key_audit(payload)
    if violations:
        raise PolicyValidationError(
            f"forbidden policy observation fields: {', '.join(violations)}"
        )


def observation_to_dict(observation: PolicyObservation) -> dict[str, Any]:
    body = _observation_body(
        observation.policy_id,
        observation.strategy,
        observation.payload,
        observation.budget,
        observation.source_state_sha256,
    )
    return {**body, "observationSha256": observation.observation_sha256}


def canonical_observation_bytes(observation: PolicyObservation) -> bytes:
    return _canonical_json_bytes(observation_to_dict(observation))


def build_policy_observation(
    environment: Environment,
    state: MovementState,
    actor_id: str,
    definition: PolicyDefinition,
    *,
    relation_profile: RelationProfile | None = None,
    boundary_direction: str | None = None,
    boundary_tokens: Sequence[str] | None = None,
    gradient_levels: Mapping[str, int] | None = None,
    gradient_direction: str | None = None,
    lagged_conflicts: Mapping[str, int] | None = None,
    memory: PolicyMemory | None = None,
    decision_key: str = "default",
    activation_index: int = 0,
    _state_sha256: str | None = None,
    _neighbors: Mapping[str, Sequence[str]] | None = None,
) -> ObservationBuild:
    """Construct an immutable observation envelope and engine-only handle map."""

    source_hash = (
        movement_state_sha256(state) if _state_sha256 is None else _state_sha256
    )
    neighbors = neighbor_map(environment) if _neighbors is None else _neighbors
    candidates, affordance_cost = enumerate_candidate_affordances(
        environment,
        state,
        actor_id,
        definition,
        _state_sha256=source_hash,
        _neighbors=neighbors,
    )
    occupancy = state.occupant_map
    locations = {
        occupant.occupant_id: site_id for site_id, occupant in occupancy.items()
    }
    source = locations[actor_id]
    actor_token = occupancy[source].token
    payload: dict[str, Any] = {"candidates": []}
    ledger = empty_observation_ledger()
    ledger.update(affordance_cost)
    ledger["candidateDescriptorReads"] = len(candidates)
    candidate_key_bits = _bits_for_count(definition.max_candidates - 1)
    ledger["communicatedBitsUpperBound"] = (
        _bits_for_count(definition.max_candidates)
        + len(candidates) * candidate_key_bits
    )

    if "actor_token" in definition.observation_features:
        payload["actorToken"] = actor_token
        ledger["actorTokenReads"] = 1
        token_count = len({occupant.token for occupant in state.occupant_map.values()})
        ledger["communicatedBitsUpperBound"] += _bits_for_count(token_count - 1)

    current_utility: int | None = None
    utility_deltas: dict[str, int] = {}
    if "local_relation_delta" in definition.observation_features:
        if relation_profile is None:
            raise PolicyValidationError("policy requires an S02 local relation profile")
        current_utility, reads = local_contact_utility(
            environment,
            state,
            actor_id,
            relation_profile,
            _neighbors=neighbors,
            _occupancy=occupancy,
            _locations=locations,
        )
        ledger["localTokenReads"] += reads
        ledger["utilityEvaluations"] += 1
        payload["currentLocalRelationUtility"] = current_utility
        ledger["communicatedBitsUpperBound"] += 8
        for candidate in candidates:
            preview = _preview_state(environment, state, candidate.proposal)
            preview_occupancy = preview.occupant_map
            after, after_reads = local_contact_utility(
                environment,
                preview,
                actor_id,
                relation_profile,
                _neighbors=neighbors,
                _occupancy=preview_occupancy,
                _locations={actor_id: candidate.target_site},
            )
            utility_deltas[candidate.candidate_key] = after - current_utility
            ledger["localTokenReads"] += after_reads
            ledger["utilityEvaluations"] += 1

    current_boundary: int | None = None
    boundary_deltas: dict[str, int] = {}
    if "natural_boundary_delta" in definition.observation_features:
        if boundary_direction not in {"seek", "avoid"}:
            raise PolicyValidationError(
                "boundary policy requires seek or avoid direction"
            )
        if not boundary_tokens:
            raise PolicyValidationError(
                "boundary policy requires explicit actor tokens"
            )
        current_boundary = _natural_boundary_level(environment, source)
        ledger["boundarySignalReads"] += 1
        payload["boundaryDirectiveApplies"] = actor_token in set(boundary_tokens)
        payload["currentNaturalBoundaryLevel"] = current_boundary
        payload["boundaryDirection"] = boundary_direction
        ledger["communicatedBitsUpperBound"] += 5
        for candidate in candidates:
            target_level = _natural_boundary_level(environment, candidate.target_site)
            ledger["boundarySignalReads"] += 1
            raw_delta = target_level - current_boundary
            boundary_deltas[candidate.candidate_key] = (
                raw_delta if boundary_direction == "seek" else -raw_delta
            )

    current_gradient: int | None = None
    gradient_deltas: dict[str, int] = {}
    if "gradient_delta" in definition.observation_features:
        if gradient_levels is None or gradient_direction not in {"up", "down"}:
            raise PolicyValidationError(
                "gradient policy requires an explicit field and direction"
            )
        required_sites = {source, *(item.target_site for item in candidates)}
        if set(gradient_levels) < required_sites:
            raise PolicyValidationError(
                "gradient field does not cover local candidates"
            )
        if any(not 0 <= int(gradient_levels[site]) <= 255 for site in required_sites):
            raise PolicyValidationError("gradient levels must be uint8")
        current_gradient = int(gradient_levels[source])
        ledger["gradientSignalReads"] += 1
        payload["currentGradientLevel"] = current_gradient
        payload["gradientDirection"] = gradient_direction
        ledger["communicatedBitsUpperBound"] += 9
        for candidate in candidates:
            target_level = int(gradient_levels[candidate.target_site])
            ledger["gradientSignalReads"] += 1
            raw_delta = target_level - current_gradient
            gradient_deltas[candidate.candidate_key] = (
                raw_delta if gradient_direction == "up" else -raw_delta
            )

    conflict_counts: dict[str, int] = {}
    if "lagged_conflict_count" in definition.observation_features:
        if lagged_conflicts is None:
            raise PolicyValidationError(
                "conflict-avoidance policy requires a lagged conflict field"
            )
        required_sites = {item.target_site for item in candidates}
        if set(lagged_conflicts) < required_sites:
            raise PolicyValidationError(
                "lagged conflict field does not cover local candidates"
            )
        for candidate in candidates:
            count = int(lagged_conflicts[candidate.target_site])
            if not 0 <= count <= 3:
                raise PolicyValidationError(
                    "lagged conflict count must be saturated 0..3"
                )
            conflict_counts[candidate.candidate_key] = count
            ledger["laggedConflictReads"] += 1

    if "own_memory" in definition.observation_features:
        if memory is None:
            raise PolicyValidationError("memory policy requires identity-owned memory")
        validate_policy_memory(memory)
        payload["ownMemory"] = {
            "bestLocalRelationUtility": memory.best_local_utility,
            "frustration": memory.frustration,
        }
        ledger["memoryReads"] = 2
        ledger["persistentMemoryBits"] = 10
        ledger["communicatedBitsUpperBound"] += 10

    for candidate in candidates:
        record: dict[str, Any] = {"candidateKey": candidate.candidate_key}
        if utility_deltas:
            record["localRelationDelta"] = utility_deltas[candidate.candidate_key]
            ledger["communicatedBitsUpperBound"] += 8
        if boundary_deltas:
            record["naturalBoundaryDelta"] = boundary_deltas[candidate.candidate_key]
            ledger["communicatedBitsUpperBound"] += 4
        if gradient_deltas:
            record["gradientDelta"] = gradient_deltas[candidate.candidate_key]
            ledger["communicatedBitsUpperBound"] += 9
        if conflict_counts:
            record["laggedConflictCount"] = conflict_counts[candidate.candidate_key]
            ledger["communicatedBitsUpperBound"] += 2
        if "movement_cost" in definition.observation_features:
            record["movementCost"] = candidate.movement_cost
            ledger["communicatedBitsUpperBound"] += 4
        payload["candidates"].append(record)

    if "counter_exploration_index" in definition.observation_features:
        if candidates:
            payload["explorationIndex"] = _counter_bounded(
                decision_key,
                definition.policy_id,
                actor_id,
                activation_index,
                len(candidates),
            )
            ledger["counterRandomDraws"] = 1
            ledger["communicatedBitsUpperBound"] += candidate_key_bits
        else:
            payload["explorationIndex"] = None
            ledger["communicatedBitsUpperBound"] += 1

    ledger["comparisonOperations"] = len(candidates)
    if ledger["communicatedBitsUpperBound"] > definition.information_budget_max_bits:
        raise PolicyValidationError("policy observation exceeds declared bit budget")
    validate_policy_payload(payload)
    if set(ledger) != set(OBSERVATION_LEDGER_FIELDS):
        raise PolicyValidationError("observation ledger schema mismatch")
    body = _observation_body(
        definition.policy_id,
        definition.strategy,
        payload,
        ledger,
        source_hash,
    )
    observation = PolicyObservation(
        policy_id=definition.policy_id,
        strategy=definition.strategy,
        payload=payload,
        budget=ledger,
        source_state_sha256=source_hash,
        observation_sha256=_sha256_payload("E06/S05/observation/v1", body),
    )
    return ObservationBuild(
        observation=observation,
        candidate_map={item.candidate_key: item for item in candidates},
    )


def validate_policy_memory(memory: PolicyMemory) -> None:
    if not -128 <= memory.best_local_utility <= 127:
        raise PolicyValidationError("best local utility must fit signed int8")
    if not 0 <= memory.frustration <= 3:
        raise PolicyValidationError("frustration must fit two bits")


def _decision(
    definition: PolicyDefinition,
    action: str,
    selected: str | None,
    reason: str,
    score: int | None,
) -> PolicyDecision:
    return PolicyDecision(
        policy_id=definition.policy_id,
        action=action,
        selected_candidate_key=selected,
        reason=reason,
        decision_score=score,
    )


def _maximum_candidate(
    records: Sequence[Mapping[str, Any]], field: str
) -> tuple[Mapping[str, Any] | None, int | None]:
    if not records:
        return None, None
    best = max(
        records, key=lambda item: (int(item[field]), -int(item["candidateKey"][1:]))
    )
    return best, int(best[field])


def decide_policy(
    definition: PolicyDefinition, payload: Mapping[str, Any]
) -> PolicyDecision:
    """Pure deterministic decision over the disclosed payload only."""

    validate_policy_payload(payload)
    candidates = list(payload["candidates"])
    strategy = definition.strategy
    if not candidates:
        return _decision(definition, "noop", None, "no_candidate", None)
    if strategy == "greedy_neighbor_satisfaction":
        best, score = _maximum_candidate(candidates, "localRelationDelta")
        if score is None or score <= 0:
            return _decision(definition, "noop", None, "no_local_improvement", score)
        return _decision(
            definition,
            "proposal",
            str(best["candidateKey"]),
            "best_local_improvement",
            score,
        )
    if strategy == "boundary_seeking":
        if not payload["boundaryDirectiveApplies"]:
            return _decision(
                definition, "noop", None, "boundary_directive_not_for_actor_token", None
            )
        best, score = _maximum_candidate(candidates, "naturalBoundaryDelta")
        if score is None or score <= 0:
            return _decision(definition, "noop", None, "no_boundary_improvement", score)
        return _decision(
            definition,
            "proposal",
            str(best["candidateKey"]),
            "best_boundary_step",
            score,
        )
    if strategy == "gradient_following":
        best, score = _maximum_candidate(candidates, "gradientDelta")
        if score is None or score <= 0:
            return _decision(definition, "noop", None, "no_gradient_improvement", score)
        return _decision(
            definition,
            "proposal",
            str(best["candidateKey"]),
            "best_gradient_step",
            score,
        )
    if strategy == "exploration":
        index = payload["explorationIndex"]
        if index is None:
            return _decision(definition, "noop", None, "no_candidate", None)
        selected = candidates[int(index)]
        return _decision(
            definition,
            "proposal",
            str(selected["candidateKey"]),
            "counter_addressed_exploration",
            int(index),
        )
    if strategy == "memory_based_recovery":
        memory = payload["ownMemory"]
        current = int(payload["currentLocalRelationUtility"])
        if current >= int(memory["bestLocalRelationUtility"]):
            return _decision(
                definition, "noop", None, "remembered_reference_met", current
            )
        best, score = _maximum_candidate(candidates, "localRelationDelta")
        if score is None or score <= 0:
            return _decision(definition, "noop", None, "no_recovery_step", score)
        return _decision(
            definition,
            "proposal",
            str(best["candidateKey"]),
            "restore_remembered_relation",
            score,
        )
    if strategy == "conflict_avoidance":
        penalty = int(definition.parameters["laggedConflictPenalty"])
        cost_weight = int(definition.parameters["movementCostWeight"])
        scored = [
            (
                int(item["localRelationDelta"])
                - penalty * int(item["laggedConflictCount"])
                - cost_weight * int(item["movementCost"]),
                str(item["candidateKey"]),
                item,
            )
            for item in candidates
        ]
        score, _, best = max(scored, key=lambda item: (item[0], -int(item[1][1:])))
        if score <= 0:
            return _decision(
                definition, "noop", None, "no_net_conflict_avoiding_gain", score
            )
        return _decision(
            definition,
            "proposal",
            str(best["candidateKey"]),
            "best_lagged_conflict_adjusted_step",
            score,
        )
    raise PolicyValidationError(f"unsupported strategy: {strategy}")


def decision_to_dict(decision: PolicyDecision) -> dict[str, Any]:
    return {
        "schemaVersion": DECISION_VERSION,
        "policyId": decision.policy_id,
        "action": decision.action,
        "selectedCandidateKey": decision.selected_candidate_key,
        "reason": decision.reason,
        "decisionScore": decision.decision_score,
    }


def canonical_decision_bytes(decision: PolicyDecision) -> bytes:
    return _canonical_json_bytes(decision_to_dict(decision))


def materialize_decision(
    build: ObservationBuild, decision: PolicyDecision
) -> MovementProposal | None:
    """Engine adapter from opaque policy choice to authenticated S04 proposal."""

    if decision.policy_id != build.observation.policy_id:
        raise PolicyValidationError("decision policy mismatch")
    if decision.action == "noop":
        if decision.selected_candidate_key is not None:
            raise PolicyValidationError("noop cannot select a candidate")
        return None
    if decision.action != "proposal" or decision.selected_candidate_key is None:
        raise PolicyValidationError("invalid policy decision action")
    if decision.selected_candidate_key not in build.candidate_map:
        raise PolicyValidationError("decision selected an unknown candidate")
    return build.candidate_map[decision.selected_candidate_key].proposal


def update_policy_memory(
    memory: PolicyMemory,
    observation: PolicyObservation,
    decision: PolicyDecision,
    outcome: str,
) -> PolicyMemory:
    validate_policy_memory(memory)
    current = int(
        observation.payload.get(
            "currentLocalRelationUtility", memory.best_local_utility
        )
    )
    realized = current
    if outcome == "accepted" and decision.decision_score is not None:
        if decision.reason in {
            "restore_remembered_relation",
            "best_local_improvement",
            "best_lagged_conflict_adjusted_step",
        }:
            selected = next(
                (
                    item
                    for item in observation.payload["candidates"]
                    if item["candidateKey"] == decision.selected_candidate_key
                ),
                None,
            )
            if selected is not None and "localRelationDelta" in selected:
                realized += int(selected["localRelationDelta"])
    best = max(memory.best_local_utility, realized)
    best = max(-128, min(127, best))
    frustration = 0 if outcome == "accepted" else min(3, memory.frustration + 1)
    updated = PolicyMemory(best_local_utility=best, frustration=frustration)
    validate_policy_memory(updated)
    return updated


def execute_policy_activation(
    environment: Environment,
    state: MovementState,
    actor_id: str,
    definition: PolicyDefinition,
    *,
    relation_profile: RelationProfile | None = None,
    boundary_direction: str | None = None,
    boundary_tokens: Sequence[str] | None = None,
    gradient_levels: Mapping[str, int] | None = None,
    gradient_direction: str | None = None,
    lagged_conflicts: Mapping[str, int] | None = None,
    memory: PolicyMemory | None = None,
    decision_key: str = "default",
    activation_index: int = 0,
    batch_nonce: str = "policy-activation",
) -> dict[str, Any]:
    build = build_policy_observation(
        environment,
        state,
        actor_id,
        definition,
        relation_profile=relation_profile,
        boundary_direction=boundary_direction,
        boundary_tokens=boundary_tokens,
        gradient_levels=gradient_levels,
        gradient_direction=gradient_direction,
        lagged_conflicts=lagged_conflicts,
        memory=memory,
        decision_key=decision_key,
        activation_index=activation_index,
    )
    decision = decide_policy(definition, build.observation.payload)
    proposal = materialize_decision(build, decision)
    batch = resolve_batch(
        environment,
        state,
        () if proposal is None else (proposal,),
        batch_nonce=batch_nonce,
    )
    if proposal is None:
        outcome = "noop"
    elif proposal.proposal_id in batch["acceptedProposalIds"]:
        outcome = "accepted"
    else:
        outcome = "rejected"
    memory_after = None
    if memory is not None:
        memory_after = update_policy_memory(
            memory, build.observation, decision, outcome
        )
    feedback = {
        "profile": "s02_actor_local_contact_projection_v1",
        "currentLocalRelationUtility": build.observation.payload.get(
            "currentLocalRelationUtility"
        ),
        "selectedLocalRelationDelta": next(
            (
                item.get("localRelationDelta")
                for item in build.observation.payload["candidates"]
                if item["candidateKey"] == decision.selected_candidate_key
            ),
            None,
        ),
        "wholeGridGrammarScoreDisclosed": False,
        "s01GlobalEvaluationDisclosed": False,
    }
    body = {
        "schemaVersion": EVENT_VERSION,
        "policyId": definition.policy_id,
        "strategy": definition.strategy,
        "observation": observation_to_dict(build.observation),
        "decision": decision_to_dict(decision),
        "materializationAudit": {
            "proposalCreated": proposal is not None,
            "authenticationAddedAfterPolicyDecision": proposal is not None,
            "policySawProposalEnvelope": False,
            "proposalKindEngineAudit": None if proposal is None else proposal.kind,
        },
        "movementBatchResult": batch,
        "outcome": outcome,
        "memoryBefore": memory_to_dict(memory),
        "memoryAfter": memory_to_dict(memory_after),
        "s02LocalFeedback": feedback,
        "s01GlobalEvaluation": {
            "availability": "offline_only_not_computed_in_policy_activation",
            "disclosedToPolicy": False,
        },
    }
    return {**body, "eventSha256": _sha256_payload("E06/S05/event/v1", body)}


def canonical_policy_event_bytes(event: Mapping[str, Any]) -> bytes:
    return _canonical_json_bytes(dict(event))


def memory_to_dict(memory: PolicyMemory | None) -> dict[str, int] | None:
    if memory is None:
        return None
    return {
        "bestLocalRelationUtility": memory.best_local_utility,
        "frustration": memory.frustration,
    }


def policy_complexity(definition: PolicyDefinition) -> dict[str, Any]:
    complexity = {key: int(value) for key, value in definition.complexity.items()}
    required = {
        "decisionRuleCount",
        "branchCount",
        "tunableScalarCount",
        "persistentMemoryBits",
    }
    if set(complexity) != required or any(value < 0 for value in complexity.values()):
        raise PolicyValidationError("invalid policy complexity declaration")
    score = (
        complexity["decisionRuleCount"]
        + complexity["branchCount"]
        + complexity["tunableScalarCount"]
        + math.ceil(complexity["persistentMemoryBits"] / 4)
    )
    return {
        **complexity,
        "complexityScore": score,
        "observationFeatureCount": len(definition.observation_features),
        "allowedMovementKindCount": len(definition.allowed_movement_kinds),
        "targetSpecificity": definition.target_specificity,
        "informationBudgetMaxBits": definition.information_budget_max_bits,
        "metricDefinition": "rules + branches + tunable scalars + ceil(memory bits / 4)",
    }


def policy_to_dict(definition: PolicyDefinition) -> dict[str, Any]:
    return {
        "policyId": definition.policy_id,
        "strategy": definition.strategy,
        "allowedMovementKinds": list(definition.allowed_movement_kinds),
        "maxCandidates": definition.max_candidates,
        "observationFeatures": list(definition.observation_features),
        "informationBudgetMaxBits": definition.information_budget_max_bits,
        "relationProfileRequired": definition.relation_profile_required,
        "targetSpecificity": definition.target_specificity,
        "parameters": dict(definition.parameters),
        "complexity": dict(definition.complexity),
        "intendedBehavior": definition.intended_behavior,
        "nonGuarantees": list(definition.non_guarantees),
    }


def _parse_policy(raw: Mapping[str, Any]) -> PolicyDefinition:
    required = {
        "policyId",
        "strategy",
        "allowedMovementKinds",
        "maxCandidates",
        "observationFeatures",
        "informationBudgetMaxBits",
        "relationProfileRequired",
        "targetSpecificity",
        "parameters",
        "complexity",
        "intendedBehavior",
        "nonGuarantees",
    }
    if set(raw) != required:
        raise PolicyValidationError("policy declaration keys mismatch")
    strategy = str(raw["strategy"])
    if strategy not in SUPPORTED_STRATEGIES:
        raise PolicyValidationError("unsupported policy strategy")
    allowed = tuple(str(item) for item in raw["allowedMovementKinds"])
    if not allowed or len(allowed) != len(set(allowed)) or set(allowed) - ENABLED_KINDS:
        raise PolicyValidationError("invalid allowed movement kinds")
    maximum = int(raw["maxCandidates"])
    if not 1 <= maximum <= 16:
        raise PolicyValidationError("maxCandidates must be 1..16")
    features = tuple(str(item) for item in raw["observationFeatures"])
    if len(features) != len(set(features)):
        raise PolicyValidationError("observation features must be unique")
    if set(features) != FEATURES_BY_STRATEGY[strategy]:
        raise PolicyValidationError("strategy observation feature set mismatch")
    definition = PolicyDefinition(
        policy_id=str(raw["policyId"]),
        strategy=strategy,
        allowed_movement_kinds=allowed,
        max_candidates=maximum,
        observation_features=features,
        information_budget_max_bits=int(raw["informationBudgetMaxBits"]),
        relation_profile_required=bool(raw["relationProfileRequired"]),
        target_specificity=str(raw["targetSpecificity"]),
        parameters=dict(raw["parameters"]),
        complexity={str(key): int(value) for key, value in raw["complexity"].items()},
        intended_behavior=str(raw["intendedBehavior"]),
        non_guarantees=tuple(str(item) for item in raw["nonGuarantees"]),
    )
    policy_complexity(definition)
    if definition.information_budget_max_bits <= 0:
        raise PolicyValidationError("information budget must be positive")
    relation_strategies = {
        "greedy_neighbor_satisfaction",
        "memory_based_recovery",
        "conflict_avoidance",
    }
    if definition.relation_profile_required != (strategy in relation_strategies):
        raise PolicyValidationError("relation-profile requirement mismatch")
    if strategy == "conflict_avoidance":
        if set(definition.parameters) != {
            "laggedConflictPenalty",
            "movementCostWeight",
        } or any(int(value) < 0 for value in definition.parameters.values()):
            raise PolicyValidationError("invalid conflict-avoidance parameters")
    elif definition.parameters:
        raise PolicyValidationError("unexpected strategy parameters")
    return definition


def parse_policy_catalog(
    raw: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[PolicyDefinition, ...]]:
    required = {
        "schemaVersion",
        "researchStepId",
        "libraryVersion",
        "observationContract",
        "informationCostModel",
        "complexityModel",
        "policies",
        "toyFixtures",
    }
    if set(raw) != required:
        raise PolicyValidationError("policy catalog keys mismatch")
    if raw["schemaVersion"] != POLICY_CATALOG_VERSION or raw["researchStepId"] != "S05":
        raise PolicyValidationError("policy catalog version mismatch")
    policies = tuple(_parse_policy(item) for item in raw["policies"])
    identifiers = [item.policy_id for item in policies]
    strategies = [item.strategy for item in policies]
    if len(identifiers) != len(set(identifiers)):
        raise PolicyValidationError("policy IDs must be unique")
    if set(strategies) != SUPPORTED_STRATEGIES or len(strategies) != len(
        SUPPORTED_STRATEGIES
    ):
        raise PolicyValidationError("catalog must define each planned strategy once")
    metadata = {key: raw[key] for key in required - {"policies"}}
    return metadata, policies


def load_policy_catalog(
    path: str | Path,
) -> tuple[Mapping[str, Any], tuple[PolicyDefinition, ...]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise PolicyValidationError("policy catalog root must be a mapping")
    return parse_policy_catalog(raw)


def decision_source_forbidden_accesses() -> list[str]:
    """Static audit of the pure policy decision function's source text."""

    source = inspect.getsource(decide_policy).lower()
    forbidden = [
        "analysis_label",
        "state_sha",
        "proposal_id",
        "actor_id",
        "target_site",
        "globalaudit",
        "global_completion",
        "boundarytags",
        "fixed_boundary",
        "conflict_priority",
    ]
    return sorted(item for item in forbidden if item in source)
