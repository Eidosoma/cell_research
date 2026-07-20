"""Batched masked-policy and deterministic transition tensors for E06 S07.

SHA authentication, exact opaque candidate ordering, source projection, and
proposal validation remain in the frozen CPU control plane.  This module moves
only the numeric policy decision and already-validated disjoint-claim/commit
data plane to PyTorch.  Consequently it can be compared exactly at the scoped
transition boundary without granting a policy access to engine-private routes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .environments import Environment
from .movements import (
    MovementProposal,
    MovementState,
    conflict_priority_uint64,
    movement_batch_id,
    resolve_batch,
    validate_proposal,
)
from .policies import ObservationBuild, PolicyDefinition, decide_policy


MAX_CANDIDATES = 16
MAX_ROUTE_LENGTH = 6
STRATEGY_CODES = {
    "greedy_neighbor_satisfaction": 0,
    "boundary_seeking": 1,
    "gradient_following": 2,
    "exploration": 3,
    "memory_based_recovery": 4,
    "conflict_avoidance": 5,
}
KIND_CODES = {
    "adjacent_swap": 0,
    "vacancy_move": 1,
    "short_exchange": 2,
    "rotation": 3,
}


class GPUValidationError(ValueError):
    """Raised for malformed masked observations or proposal tensors."""


@dataclass(frozen=True)
class MaskedObservationTensors:
    candidate_mask: torch.Tensor
    strategy_code: torch.Tensor
    local_relation_delta: torch.Tensor
    natural_boundary_delta: torch.Tensor
    gradient_delta: torch.Tensor
    lagged_conflict_count: torch.Tensor
    movement_cost: torch.Tensor
    boundary_directive_applies: torch.Tensor
    exploration_index: torch.Tensor
    current_local_utility: torch.Tensor
    memory_best_utility: torch.Tensor
    lagged_conflict_penalty: torch.Tensor
    movement_cost_weight: torch.Tensor


@dataclass(frozen=True)
class MaskedDecisionTensors:
    selected_candidate_index: torch.Tensor
    decision_score: torch.Tensor


@dataclass(frozen=True)
class CompiledProposalBatch:
    state: torch.Tensor
    route: torch.Tensor
    route_length: torch.Tensor
    kind_code: torch.Tensor
    rotation_direction: torch.Tensor
    priority_rank: torch.Tensor
    proposal_mask: torch.Tensor
    proposal_displacement: torch.Tensor
    site_ids: tuple[str, ...]
    occupant_ids: tuple[tuple[str, ...], ...]
    proposal_ids: tuple[tuple[str | None, ...], ...]


@dataclass(frozen=True)
class GPUTransitionResult:
    post_state: torch.Tensor
    accepted_mask: torch.Tensor
    accepted_movements: torch.Tensor
    total_graph_displacement: torch.Tensor


def _shape(builds: Sequence[Sequence[ObservationBuild | None]]) -> tuple[int, int]:
    if not builds or not builds[0]:
        raise GPUValidationError("masked observation batch cannot be empty")
    width = len(builds[0])
    if any(len(row) != width for row in builds):
        raise GPUValidationError("masked observation batch must be rectangular")
    return len(builds), width


def encode_masked_observations(
    builds: Sequence[Sequence[ObservationBuild | None]],
    definitions: Sequence[Sequence[PolicyDefinition | None]],
    *,
    device: torch.device | str,
) -> MaskedObservationTensors:
    """Encode only S05-disclosed scalars; routes and identities are absent."""

    batch, actors = _shape(builds)
    if len(definitions) != batch or any(len(row) != actors for row in definitions):
        raise GPUValidationError("definition and observation shapes differ")
    shape = (batch, actors, MAX_CANDIDATES)
    candidate_mask = torch.zeros(shape, dtype=torch.bool)
    strategy_code = torch.full((batch, actors), -1, dtype=torch.int8)
    local_delta = torch.zeros(shape, dtype=torch.int16)
    boundary_delta = torch.zeros(shape, dtype=torch.int16)
    gradient_delta = torch.zeros(shape, dtype=torch.int16)
    lagged_conflict = torch.zeros(shape, dtype=torch.int16)
    movement_cost = torch.zeros(shape, dtype=torch.int16)
    boundary_applies = torch.zeros((batch, actors), dtype=torch.bool)
    exploration_index = torch.full((batch, actors), -1, dtype=torch.int16)
    current_utility = torch.zeros((batch, actors), dtype=torch.int16)
    memory_best = torch.zeros((batch, actors), dtype=torch.int16)
    conflict_penalty = torch.zeros((batch, actors), dtype=torch.int16)
    cost_weight = torch.zeros((batch, actors), dtype=torch.int16)

    for batch_index, (build_row, definition_row) in enumerate(
        zip(builds, definitions, strict=True)
    ):
        for actor_index, (build, definition) in enumerate(
            zip(build_row, definition_row, strict=True)
        ):
            if build is None or definition is None:
                if build is not None or definition is not None:
                    raise GPUValidationError("partial masked actor slot")
                continue
            if build.observation.policy_id != definition.policy_id:
                raise GPUValidationError("masked observation policy mismatch")
            payload = build.observation.payload
            records = list(payload["candidates"])
            if len(records) > MAX_CANDIDATES:
                raise GPUValidationError("candidate count exceeds S05 cap")
            strategy_code[batch_index, actor_index] = STRATEGY_CODES[
                definition.strategy
            ]
            boundary_applies[batch_index, actor_index] = bool(
                payload.get("boundaryDirectiveApplies", False)
            )
            exploration_index[batch_index, actor_index] = int(
                -1
                if payload.get("explorationIndex") is None
                else payload["explorationIndex"]
            )
            current_utility[batch_index, actor_index] = int(
                payload.get("currentLocalRelationUtility", 0)
            )
            memory_best[batch_index, actor_index] = int(
                payload.get("ownMemory", {}).get("bestLocalRelationUtility", 0)
            )
            conflict_penalty[batch_index, actor_index] = int(
                definition.parameters.get("laggedConflictPenalty", 0)
            )
            cost_weight[batch_index, actor_index] = int(
                definition.parameters.get("movementCostWeight", 0)
            )
            for candidate_index, record in enumerate(records):
                if record["candidateKey"] != f"c{candidate_index:02d}":
                    raise GPUValidationError("opaque candidate handle is not canonical")
                candidate_mask[batch_index, actor_index, candidate_index] = True
                local_delta[batch_index, actor_index, candidate_index] = int(
                    record.get("localRelationDelta", 0)
                )
                boundary_delta[batch_index, actor_index, candidate_index] = int(
                    record.get("naturalBoundaryDelta", 0)
                )
                gradient_delta[batch_index, actor_index, candidate_index] = int(
                    record.get("gradientDelta", 0)
                )
                lagged_conflict[batch_index, actor_index, candidate_index] = int(
                    record.get("laggedConflictCount", 0)
                )
                movement_cost[batch_index, actor_index, candidate_index] = int(
                    record.get("movementCost", 0)
                )
    return MaskedObservationTensors(
        candidate_mask=candidate_mask.to(device),
        strategy_code=strategy_code.to(device),
        local_relation_delta=local_delta.to(device),
        natural_boundary_delta=boundary_delta.to(device),
        gradient_delta=gradient_delta.to(device),
        lagged_conflict_count=lagged_conflict.to(device),
        movement_cost=movement_cost.to(device),
        boundary_directive_applies=boundary_applies.to(device),
        exploration_index=exploration_index.to(device),
        current_local_utility=current_utility.to(device),
        memory_best_utility=memory_best.to(device),
        lagged_conflict_penalty=conflict_penalty.to(device),
        movement_cost_weight=cost_weight.to(device),
    )


def masked_observation_tensor_names() -> tuple[str, ...]:
    return tuple(MaskedObservationTensors.__dataclass_fields__)


def decide_masked_observations(
    observations: MaskedObservationTensors,
) -> MaskedDecisionTensors:
    """Vectorized exact S05 pure decision over the disclosed fields."""

    mask = observations.candidate_mask
    if mask.ndim != 3 or mask.shape[-1] != MAX_CANDIDATES:
        raise GPUValidationError("masked candidate tensor shape mismatch")
    sentinel = torch.iinfo(torch.int32).min
    local = observations.local_relation_delta.to(torch.int32)
    boundary = observations.natural_boundary_delta.to(torch.int32)
    gradient = observations.gradient_delta.to(torch.int32)
    conflict_score = (
        local
        - observations.lagged_conflict_penalty.to(torch.int32).unsqueeze(-1)
        * observations.lagged_conflict_count.to(torch.int32)
        - observations.movement_cost_weight.to(torch.int32).unsqueeze(-1)
        * observations.movement_cost.to(torch.int32)
    )
    strategy = observations.strategy_code.to(torch.int64)
    scores = torch.where(
        strategy.unsqueeze(-1) == STRATEGY_CODES["boundary_seeking"],
        boundary,
        torch.where(
            strategy.unsqueeze(-1) == STRATEGY_CODES["gradient_following"],
            gradient,
            torch.where(
                strategy.unsqueeze(-1) == STRATEGY_CODES["conflict_avoidance"],
                conflict_score,
                local,
            ),
        ),
    )
    scores = torch.where(mask, scores, sentinel)
    best_score, best_index = scores.max(dim=-1)
    has_candidate = mask.any(dim=-1)
    selection = best_index.to(torch.int64)
    exploration = strategy == STRATEGY_CODES["exploration"]
    selection = torch.where(
        exploration,
        observations.exploration_index.to(torch.int64),
        selection,
    )
    selection = torch.where(has_candidate, selection, -1)
    positive_required = torch.isin(
        strategy,
        torch.tensor(
            [
                STRATEGY_CODES["greedy_neighbor_satisfaction"],
                STRATEGY_CODES["boundary_seeking"],
                STRATEGY_CODES["gradient_following"],
                STRATEGY_CODES["memory_based_recovery"],
                STRATEGY_CODES["conflict_avoidance"],
            ],
            device=strategy.device,
        ),
    )
    selection = torch.where(positive_required & (best_score <= 0), -1, selection)
    selection = torch.where(
        (strategy == STRATEGY_CODES["boundary_seeking"])
        & ~observations.boundary_directive_applies,
        -1,
        selection,
    )
    selection = torch.where(
        (strategy == STRATEGY_CODES["memory_based_recovery"])
        & (observations.current_local_utility >= observations.memory_best_utility),
        -1,
        selection,
    )
    selection = torch.where(strategy < 0, -1, selection)
    decision_score = torch.where(selection >= 0, best_score, best_score)
    decision_score = torch.where(has_candidate, decision_score, 0)
    decision_score = torch.where(
        exploration & (selection >= 0), selection.to(torch.int32), decision_score
    )
    return MaskedDecisionTensors(selection, decision_score)


def validate_masked_decision_parity(
    observations: MaskedObservationTensors,
    builds: Sequence[Sequence[ObservationBuild | None]],
    definitions: Sequence[Sequence[PolicyDefinition | None]],
) -> list[dict[str, Any]]:
    decisions = decide_masked_observations(observations)
    selected = decisions.selected_candidate_index.detach().cpu()
    rows: list[dict[str, Any]] = []
    for batch_index, (build_row, definition_row) in enumerate(
        zip(builds, definitions, strict=True)
    ):
        for actor_index, (build, definition) in enumerate(
            zip(build_row, definition_row, strict=True)
        ):
            if build is None or definition is None:
                continue
            cpu = decide_policy(definition, build.observation.payload)
            gpu_key = (
                None
                if int(selected[batch_index, actor_index]) < 0
                else f"c{int(selected[batch_index, actor_index]):02d}"
            )
            rows.append(
                {
                    "batchIndex": batch_index,
                    "actorIndex": actor_index,
                    "policyId": definition.policy_id,
                    "cpuCandidateKey": cpu.selected_candidate_key,
                    "gpuCandidateKey": gpu_key,
                    "match": cpu.selected_candidate_key == gpu_key,
                }
            )
    return rows


def _proposal_displacement(proposal: MovementProposal) -> int:
    if proposal.kind in {"adjacent_swap", "vacancy_move"}:
        return 2
    if proposal.kind == "short_exchange":
        return 4
    return len(proposal.route)


def compile_validated_proposal_batch(
    environment: Environment,
    states: Sequence[MovementState],
    proposals: Sequence[Sequence[MovementProposal]],
    batch_nonces: Sequence[str],
    *,
    device: torch.device | str,
    maximum_proposals: int | None = None,
) -> tuple[CompiledProposalBatch, tuple[dict[str, Any], ...]]:
    """Compile exact S04-valid proposals and SHA priorities to tensors."""

    if not states or len(states) != len(proposals) or len(states) != len(batch_nonces):
        raise GPUValidationError("proposal compiler batch shape mismatch")
    sites = tuple(site.site_id for site in environment.occupiable_sites)
    site_index = {site_id: index for index, site_id in enumerate(sites)}
    width = maximum_proposals or max(1, max(len(row) for row in proposals))
    if width > MAX_CANDIDATES:
        raise GPUValidationError("proposal tensor width exceeds S05 cap")
    batch = len(states)
    state_tensor = torch.empty((batch, len(sites)), dtype=torch.int32)
    route = torch.full((batch, width, MAX_ROUTE_LENGTH), -1, dtype=torch.int64)
    route_length = torch.zeros((batch, width), dtype=torch.int64)
    kind = torch.full((batch, width), -1, dtype=torch.int8)
    rotation = torch.zeros((batch, width), dtype=torch.int8)
    priority_rank = torch.full((batch, width), width, dtype=torch.int64)
    mask = torch.zeros((batch, width), dtype=torch.bool)
    displacement = torch.zeros((batch, width), dtype=torch.int16)
    occupant_ids: list[tuple[str, ...]] = []
    proposal_ids: list[tuple[str | None, ...]] = []
    expected_results: list[dict[str, Any]] = []

    for batch_index, (state, proposal_row, nonce) in enumerate(
        zip(states, proposals, batch_nonces, strict=True)
    ):
        if state.environment_id != environment.environment_id:
            raise GPUValidationError("compiled state environment mismatch")
        if len(proposal_row) > width:
            raise GPUValidationError("compiled proposal row exceeds width")
        ordered_occupants = tuple(
            sorted(occupant.occupant_id for _, occupant in state.occupancy)
        )
        occupant_index = {
            occupant_id: index for index, occupant_id in enumerate(ordered_occupants)
        }
        state_map = state.occupant_map
        state_tensor[batch_index] = torch.tensor(
            [occupant_index[state_map[site_id].occupant_id] for site_id in sites],
            dtype=torch.int32,
        )
        occupant_ids.append(ordered_occupants)
        expected = resolve_batch(
            environment, state, tuple(proposal_row), batch_nonce=str(nonce)
        )
        expected_results.append(expected)
        batch_id = movement_batch_id(state, proposal_row, str(nonce))
        priorities = {
            proposal.proposal_id: conflict_priority_uint64(
                batch_id, proposal.proposal_id
            )
            for proposal in proposal_row
        }
        ranked_ids = [
            item.proposal_id
            for item in sorted(
                proposal_row,
                key=lambda item: (priorities[item.proposal_id], item.proposal_id),
            )
        ]
        rank_by_id = {proposal_id: rank for rank, proposal_id in enumerate(ranked_ids)}
        ids: list[str | None] = [None] * width
        for slot, proposal in enumerate(proposal_row):
            validation = validate_proposal(environment, state, proposal)
            if not validation.valid:
                raise GPUValidationError(
                    f"GPU compiler received invalid S04 proposal: {validation.reason}"
                )
            ids[slot] = proposal.proposal_id
            mask[batch_index, slot] = True
            route_length[batch_index, slot] = len(proposal.route)
            kind[batch_index, slot] = KIND_CODES[proposal.kind]
            rotation[batch_index, slot] = proposal.rotation_direction
            priority_rank[batch_index, slot] = rank_by_id[proposal.proposal_id]
            displacement[batch_index, slot] = _proposal_displacement(proposal)
            for route_index, site_id in enumerate(proposal.route):
                route[batch_index, slot, route_index] = site_index[site_id]
        proposal_ids.append(tuple(ids))
    return (
        CompiledProposalBatch(
            state=state_tensor.to(device),
            route=route.to(device),
            route_length=route_length.to(device),
            kind_code=kind.to(device),
            rotation_direction=rotation.to(device),
            priority_rank=priority_rank.to(device),
            proposal_mask=mask.to(device),
            proposal_displacement=displacement.to(device),
            site_ids=sites,
            occupant_ids=tuple(occupant_ids),
            proposal_ids=tuple(proposal_ids),
        ),
        tuple(expected_results),
    )


def resolve_compiled_proposals(
    compiled: CompiledProposalBatch,
) -> GPUTransitionResult:
    """Deterministic greedy conflict claiming and atomic integer commit."""

    state = compiled.state
    route = compiled.route
    batch, proposal_count, route_width = route.shape
    if route_width != MAX_ROUTE_LENGTH or state.ndim != 2:
        raise GPUValidationError("compiled transition tensor shape mismatch")
    site_count = state.shape[1]
    sentinel_site = site_count
    route_positions = torch.arange(MAX_ROUTE_LENGTH, device=state.device)
    masked_rank = torch.where(
        compiled.proposal_mask, compiled.priority_rank, proposal_count
    )
    proposal_order = torch.argsort(masked_rank, dim=1, stable=True)
    ordered_route = route.gather(
        1,
        proposal_order.unsqueeze(-1).expand(-1, -1, MAX_ROUTE_LENGTH),
    )
    ordered_length = compiled.route_length.gather(1, proposal_order)
    ordered_exists = compiled.proposal_mask.gather(1, proposal_order)
    claimed = torch.zeros(
        (batch, site_count + 1), dtype=torch.bool, device=state.device
    )
    accepted_order = torch.zeros(
        (batch, proposal_count), dtype=torch.bool, device=state.device
    )
    for rank in range(proposal_count):
        selected_route = ordered_route[:, rank]
        selected_length = ordered_length[:, rank]
        exists = ordered_exists[:, rank]
        route_mask = (
            route_positions.unsqueeze(0) < selected_length.unsqueeze(1)
        ) & exists.unsqueeze(1)
        safe_route = torch.where(route_mask, selected_route, sentinel_site)
        already_claimed = claimed.gather(1, safe_route)
        conflict = (already_claimed & route_mask).any(dim=1)
        wins = exists & ~conflict
        accepted_order[:, rank] = wins
        updates = already_claimed | (route_mask & wins.unsqueeze(1))
        claimed.scatter_(1, safe_route, updates)
    accepted = torch.zeros_like(accepted_order)
    accepted.scatter_(1, proposal_order, accepted_order)

    valid_route = route_positions.view(1, 1, -1) < compiled.route_length.unsqueeze(-1)
    rotation = compiled.kind_code == KIND_CODES["rotation"]
    last_index = (compiled.route_length - 1).clamp_min(0)
    first_site = route[:, :, 0]
    last_site = route.gather(2, last_index.unsqueeze(-1)).squeeze(-1)
    endpoint_target = torch.where(
        route_positions.view(1, 1, -1) == 0,
        last_site.unsqueeze(-1),
        first_site.unsqueeze(-1),
    )
    safe_length = compiled.route_length.clamp_min(1)
    rotation_target_index = (
        route_positions.view(1, 1, -1)
        + compiled.rotation_direction.to(torch.int64).unsqueeze(-1)
    ) % safe_length.unsqueeze(-1)
    rotation_target = route.gather(2, rotation_target_index)
    target_site = torch.where(rotation.unsqueeze(-1), rotation_target, endpoint_target)
    endpoint_mask = (route_positions.view(1, 1, -1) == 0) | (
        route_positions.view(1, 1, -1) == last_index.unsqueeze(-1)
    )
    commit_mask = (
        accepted.unsqueeze(-1) & valid_route & (rotation.unsqueeze(-1) | endpoint_mask)
    )
    source_site = torch.where(commit_mask, route, sentinel_site)
    target_site = torch.where(commit_mask, target_site, sentinel_site)
    mapping = torch.arange(
        site_count + 1, dtype=torch.int64, device=state.device
    ).repeat(batch, 1)
    mapping.scatter_(1, target_site.flatten(1), source_site.flatten(1))
    after = state.gather(1, mapping[:, :site_count])
    accepted_movements = accepted.sum(dim=1, dtype=torch.int64)
    displacement = (
        accepted.to(torch.int64) * compiled.proposal_displacement.to(torch.int64)
    ).sum(dim=1)
    return GPUTransitionResult(after, accepted, accepted_movements, displacement)


def decode_post_state_occupant_ids(
    compiled: CompiledProposalBatch, result: GPUTransitionResult
) -> tuple[tuple[str, ...], ...]:
    values = result.post_state.detach().cpu().tolist()
    return tuple(
        tuple(compiled.occupant_ids[row][index] for index in state_row)
        for row, state_row in enumerate(values)
    )


def expected_post_state_occupant_ids(
    compiled: CompiledProposalBatch,
    expected_results: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], ...]:
    output = []
    for expected in expected_results:
        occupancy = {
            item["siteId"]: item["occupantId"]
            for item in expected["postState"]["occupancy"]
        }
        output.append(tuple(occupancy[site_id] for site_id in compiled.site_ids))
    return tuple(output)


def synthetic_compiled_workload(
    batch_size: int,
    *,
    sites: int = 81,
    proposals: int = 8,
    device: torch.device | str,
) -> CompiledProposalBatch:
    """Create a deterministic mixed-conflict integer workload for benchmarks."""

    if batch_size <= 0 or sites < 24 or proposals != 8:
        raise GPUValidationError("synthetic benchmark contract mismatch")
    state = torch.arange(sites, dtype=torch.int32).repeat(batch_size, 1)
    routes = [
        (0, 1),
        (1, 2),
        (3, 4, 5),
        (6, 7, 8, 9),
        (10, 11),
        (12, 13, 14),
        (15, 16, 17, 18, 19, 20),
        (21, 22),
    ]
    kinds = [0, 0, 2, 3, 1, 2, 3, 0]
    directions = [0, 0, 0, 1, 0, 0, -1, 0]
    displacement = [2, 2, 4, 4, 2, 4, 6, 2]
    route_tensor = torch.full(
        (batch_size, proposals, MAX_ROUTE_LENGTH), -1, dtype=torch.int64
    )
    length_tensor = torch.zeros((batch_size, proposals), dtype=torch.int64)
    for index, item in enumerate(routes):
        route_tensor[:, index, : len(item)] = torch.tensor(item)
        length_tensor[:, index] = len(item)
    # Metadata are immutable and identical across this synthetic workload.
    # Reuse the rows so the target-batch device-memory probe does not create
    # millions of irrelevant host-side strings.
    proposal_id_row = tuple(f"synthetic-{index}" for index in range(proposals))
    proposal_ids = (proposal_id_row,) * batch_size
    occupant_id_row = tuple(f"o{index}" for index in range(sites))
    return CompiledProposalBatch(
        state=state.to(device),
        route=route_tensor.to(device),
        route_length=length_tensor.to(device),
        kind_code=torch.tensor(kinds, dtype=torch.int8)
        .repeat(batch_size, 1)
        .to(device),
        rotation_direction=torch.tensor(directions, dtype=torch.int8)
        .repeat(batch_size, 1)
        .to(device),
        priority_rank=torch.arange(proposals, dtype=torch.int64)
        .repeat(batch_size, 1)
        .to(device),
        proposal_mask=torch.ones(
            (batch_size, proposals), dtype=torch.bool, device=device
        ),
        proposal_displacement=torch.tensor(displacement, dtype=torch.int16)
        .repeat(batch_size, 1)
        .to(device),
        site_ids=tuple(f"s{index}" for index in range(sites)),
        occupant_ids=(occupant_id_row,) * batch_size,
        proposal_ids=proposal_ids,
    )


def tensor_storage_bytes(compiled: CompiledProposalBatch) -> int:
    tensors = (
        compiled.state,
        compiled.route,
        compiled.route_length,
        compiled.kind_code,
        compiled.rotation_direction,
        compiled.priority_rank,
        compiled.proposal_mask,
        compiled.proposal_displacement,
    )
    return sum(item.nelement() * item.element_size() for item in tensors)


def tensor_permutation_invariants(
    before: torch.Tensor, after: torch.Tensor
) -> torch.Tensor:
    if before.shape != after.shape or before.ndim != 2:
        raise GPUValidationError("state invariant tensor shape mismatch")
    return torch.eq(
        torch.sort(before.to(torch.int64), dim=1).values,
        torch.sort(after.to(torch.int64), dim=1).values,
    ).all(dim=1)
