"""S08 differential validation for the frozen S07 CPU/GPU engine boundary.

The CPU remains authoritative for S04 authentication/validation, S05 source
projection, and S06 channel/controller construction.  This module shadows the
promised GPU fields and records exact comparisons.  It never substitutes a GPU
answer into the CPU oracle, and it preserves complete fixtures for every
mismatch rather than weakening a comparison.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Mapping, Sequence

import torch

from .engine import EngineContext, EpisodeDefinition, run_cpu_episode
from .environments import Environment, neighbor_map
from .gpu_engine import (
    compile_validated_proposal_batch,
    decode_post_state_occupant_ids,
    decide_masked_observations,
    encode_masked_observations,
    expected_post_state_occupant_ids,
    resolve_compiled_proposals,
    tensor_permutation_invariants,
)
from .movements import (
    MovementProposal,
    MovementState,
    initial_movement_state,
    make_proposal,
    movement_state_to_dict,
    proposal_to_dict,
    validate_proposal,
    validate_state_against_environment,
)
from .policies import ObservationBuild, PolicyDecision, PolicyDefinition


@dataclass(frozen=True)
class TransitionCase:
    case_id: str
    phase: str
    state: MovementState
    proposals: tuple[MovementProposal, ...]
    batch_nonce: str
    metadata: Mapping[str, Any]
    expected_batch: Mapping[str, Any] | None = None


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _digest_rank(domain: str, address: str, value: str) -> bytes:
    return hashlib.sha256(
        domain.encode("ascii")
        + b"\x00"
        + address.encode("utf-8")
        + b"\x00"
        + value.encode("utf-8")
    ).digest()


def exhaustive_identity_states(environment: Environment) -> tuple[MovementState, ...]:
    """Enumerate every identity permutation on a small environment."""

    initial = initial_movement_state(environment)
    sites = tuple(site_id for site_id, _ in initial.occupancy)
    occupants = tuple(occupant for _, occupant in initial.occupancy)
    fixed_sites = {
        site.site_id
        for site in environment.occupiable_sites
        if site.role == "fixed_boundary"
    }
    if fixed_sites:
        raise ValueError(
            "S08 exhaustive permutation fixtures cannot contain fixed sites"
        )
    states = []
    for index, permutation in enumerate(itertools.permutations(occupants)):
        state = MovementState(
            environment_id=initial.environment_id,
            environment_sha256=initial.environment_sha256,
            transition_index=index,
            occupancy=tuple(zip(sites, permutation, strict=True)),
        )
        validate_state_against_environment(environment, state)
        states.append(state)
    return tuple(states)


def counter_permuted_state(
    environment: Environment, address: str, *, transition_index: int
) -> MovementState:
    """Create a state using only domain-separated content ranks."""

    initial = initial_movement_state(environment)
    site_roles = {site.site_id: site.role for site in environment.occupiable_sites}
    fixed = {
        site_id: occupant
        for site_id, occupant in initial.occupancy
        if site_roles[site_id] == "fixed_boundary"
    }
    movable_sites = tuple(
        site_id
        for site_id, _ in initial.occupancy
        if site_roles[site_id] != "fixed_boundary"
    )
    movable_occupants = tuple(
        occupant
        for site_id, occupant in initial.occupancy
        if site_roles[site_id] != "fixed_boundary"
    )
    ranked = tuple(
        sorted(
            movable_occupants,
            key=lambda item: _digest_rank(
                "E06/S08/state-permutation/v1", address, item.occupant_id
            ),
        )
    )
    occupancy = dict(fixed)
    occupancy.update(zip(movable_sites, ranked, strict=True))
    state = MovementState(
        environment_id=initial.environment_id,
        environment_sha256=initial.environment_sha256,
        transition_index=transition_index,
        occupancy=tuple(sorted(occupancy.items())),
    )
    validate_state_against_environment(environment, state)
    return state


@lru_cache(maxsize=None)
def _cycle_routes(
    nodes: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, ...], ...]:
    neighbors = {node: [] for node in nodes}
    for first, second in edges:
        neighbors[first].append(second)
        neighbors[second].append(first)
    routes: set[tuple[str, ...]] = set()
    for start in nodes:
        stack: list[tuple[str, ...]] = [(start,)]
        while stack:
            path = stack.pop()
            current = path[-1]
            for other in neighbors[current]:
                if other == start and 3 <= len(path) <= 6:
                    routes.add(path)
                    continue
                if other in path or len(path) >= 6:
                    continue
                stack.append((*path, other))
    return tuple(sorted(routes))


def enumerate_legal_proposals(
    environment: Environment,
    state: MovementState,
    *,
    cycle_candidate_cap: int | None = None,
    cycle_rank_address: str = "",
) -> tuple[MovementProposal, ...]:
    """Enumerate every S04-valid local intent, optionally capping cycles."""

    occupancy = state.occupant_map
    neighbors = neighbor_map(environment)
    proposals: dict[str, MovementProposal] = {}

    def retain(proposal: MovementProposal) -> None:
        validation = validate_proposal(environment, state, proposal)
        if validation.valid:
            proposals[proposal.proposal_id] = proposal

    for first, second in environment.edges:
        for source, target in ((first, second), (second, first)):
            source_kind = occupancy[source].kind
            target_kind = occupancy[target].kind
            if source_kind == "cell" and target_kind == "cell":
                retain(make_proposal(state, "adjacent_swap", (source, target)))
            elif source_kind == "cell" and target_kind == "vacancy":
                retain(make_proposal(state, "vacancy_move", (source, target)))

    for source in sorted(neighbors):
        if occupancy[source].kind != "cell":
            continue
        for middle in neighbors[source]:
            for target in neighbors[middle]:
                if target == source or occupancy[target].kind != "cell":
                    continue
                retain(make_proposal(state, "short_exchange", (source, middle, target)))

    rotation_specs = tuple(
        (route, direction)
        for route in _cycle_routes(tuple(sorted(neighbors)), environment.edges)
        for direction in (-1, 1)
    )
    if cycle_candidate_cap is not None and len(rotation_specs) > cycle_candidate_cap:
        rotation_specs = tuple(
            sorted(
                rotation_specs,
                key=lambda item: _digest_rank(
                    "E06/S08/cycle-retention/v1",
                    cycle_rank_address,
                    f"{'|'.join(item[0])}:{item[1]}",
                ),
            )[:cycle_candidate_cap]
        )
    rotations: dict[str, MovementProposal] = {}
    for route, direction in rotation_specs:
        proposal = make_proposal(state, "rotation", route, rotation_direction=direction)
        if validate_proposal(environment, state, proposal).valid:
            rotations[proposal.proposal_id] = proposal
    rotation_values = tuple(rotations.values())
    proposals.update({item.proposal_id: item for item in rotation_values})
    return tuple(sorted(proposals.values(), key=lambda item: item.proposal_id))


def select_stress_batch(
    proposals: Sequence[MovementProposal],
    size: int,
    *,
    address: str,
) -> tuple[MovementProposal, ...]:
    """Select a deterministic batch biased toward shared-route conflicts."""

    if not proposals or size <= 0:
        return ()
    ranked = sorted(
        proposals,
        key=lambda item: _digest_rank(
            "E06/S08/proposal-selection/v1", address, item.proposal_id
        ),
    )
    selected = [ranked[0]]
    reserved = set(ranked[0].route)
    remaining = ranked[1:]
    while remaining and len(selected) < min(size, len(ranked)):
        remaining.sort(
            key=lambda item: (
                -len(reserved.intersection(item.route)),
                _digest_rank(
                    "E06/S08/proposal-selection-step/v1",
                    f"{address}:{len(selected)}",
                    item.proposal_id,
                ),
            )
        )
        chosen = remaining.pop(0)
        selected.append(chosen)
        reserved.update(chosen.route)
    return tuple(selected)


def _expected_tensor_row(
    occupant_ids: Sequence[str], cpu_projection: Sequence[str]
) -> torch.Tensor:
    index = {occupant_id: slot for slot, occupant_id in enumerate(occupant_ids)}
    return torch.tensor([index[item] for item in cpu_projection], dtype=torch.int32)


def _root_cause(checks: Mapping[str, bool]) -> str:
    if not checks["cpuReexecutionExact"]:
        return "cpu_oracle_reexecution_divergence"
    if not checks["acceptedProposalIdsExact"]:
        return "gpu_priority_or_conflict_claim_divergence"
    if not checks["postStateBitsExact"]:
        return "gpu_atomic_commit_divergence"
    if not checks["acceptedAndDisplacementExact"]:
        return "gpu_movement_cost_divergence"
    if not checks["gpuReplayExact"]:
        return "gpu_nondeterministic_replay"
    if not checks["permutationInvariant"]:
        return "gpu_conservation_invariant_failure"
    return "unclassified_release_critical_divergence"


def compare_transition_cases(
    environment: Environment,
    cases: Sequence[TransitionCase],
    *,
    device: torch.device | str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare a rectangular batch of valid S04 cases at zero tolerance."""

    if not cases:
        return [], []
    maximum = max(1, max(len(item.proposals) for item in cases))
    compiled, cpu_results = compile_validated_proposal_batch(
        environment,
        [item.state for item in cases],
        [item.proposals for item in cases],
        [item.batch_nonce for item in cases],
        device=device,
        maximum_proposals=maximum,
    )
    gpu = resolve_compiled_proposals(compiled)
    replay = resolve_compiled_proposals(compiled)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))
    gpu_projection = decode_post_state_occupant_ids(compiled, gpu)
    cpu_projection = expected_post_state_occupant_ids(compiled, cpu_results)
    invariant_rows = (
        tensor_permutation_invariants(compiled.state, gpu.post_state).detach().cpu()
    )
    accepted_mask = gpu.accepted_mask.detach().cpu()
    replay_state = replay.post_state.detach().cpu()
    replay_accepted = replay.accepted_mask.detach().cpu()
    gpu_state = gpu.post_state.detach().cpu()
    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for index, (case, cpu) in enumerate(zip(cases, cpu_results, strict=True)):
        gpu_ids = sorted(
            proposal_id
            for proposal_id, accepted in zip(
                compiled.proposal_ids[index], accepted_mask[index].tolist(), strict=True
            )
            if proposal_id is not None and accepted
        )
        cpu_ids = sorted(cpu["acceptedProposalIds"])
        expected_tensor = _expected_tensor_row(
            compiled.occupant_ids[index], cpu_projection[index]
        )
        cpu_reexecution = (
            case.expected_batch is None
            or case.expected_batch["transitionSha256"] == cpu["transitionSha256"]
        )
        checks = {
            "cpuReexecutionExact": cpu_reexecution,
            "postStateBitsExact": torch.equal(gpu_state[index], expected_tensor),
            "postOccupantProjectionExact": gpu_projection[index]
            == cpu_projection[index],
            "acceptedProposalIdsExact": gpu_ids == cpu_ids,
            "acceptedAndDisplacementExact": (
                int(gpu.accepted_movements[index].detach().cpu())
                == cpu["costLedger"]["acceptedMovements"]
                and int(gpu.total_graph_displacement[index].detach().cpu())
                == cpu["costLedger"]["totalGraphDisplacement"]
            ),
            "gpuReplayExact": torch.equal(gpu_state[index], replay_state[index])
            and torch.equal(accepted_mask[index], replay_accepted[index]),
            "permutationInvariant": bool(invariant_rows[index]),
            "int32State": gpu_state.dtype == torch.int32,
        }
        success = all(checks.values())
        row = {
            "caseId": case.case_id,
            "phase": case.phase,
            "environmentId": environment.environment_id,
            "geometry": environment.geometry,
            "boundaryMode": environment.boundary_mode,
            "occupancyMode": environment.occupancy_mode,
            "proposalCount": len(case.proposals),
            "movementKinds": "|".join(sorted({item.kind for item in case.proposals})),
            "acceptedCount": cpu["costLedger"]["acceptedMovements"],
            "conflictLosses": cpu["costLedger"]["conflictLosses"],
            "totalGraphDisplacement": cpu["costLedger"]["totalGraphDisplacement"],
            **checks,
            "success": success,
            **dict(case.metadata),
        }
        rows.append(row)
        if not success:
            mismatches.append(
                {
                    "schemaVersion": "e06.s08.mismatch-fixture.v1",
                    "researchStepId": "S08",
                    "caseId": case.case_id,
                    "phase": case.phase,
                    "environmentId": environment.environment_id,
                    "severity": "release_critical",
                    "triageStatus": "unresolved",
                    "rootCauseClassification": _root_cause(checks),
                    "checks": checks,
                    "metadata": dict(case.metadata),
                    "input": {
                        "state": movement_state_to_dict(case.state),
                        "proposals": [
                            proposal_to_dict(item) for item in case.proposals
                        ],
                        "batchNonce": case.batch_nonce,
                    },
                    "cpu": cpu,
                    "gpu": {
                        "postStateOccupantIds": list(gpu_projection[index]),
                        "postStateIntegerIndices": gpu_state[index].tolist(),
                        "acceptedProposalIds": gpu_ids,
                        "acceptedMovements": int(
                            gpu.accepted_movements[index].detach().cpu()
                        ),
                        "totalGraphDisplacement": int(
                            gpu.total_graph_displacement[index].detach().cpu()
                        ),
                    },
                }
            )
    return rows, mismatches


@dataclass
class DifferentialRecorder:
    """Collect exact GPU shadows for a complete canonical CPU episode."""

    device: torch.device | str
    phase: str
    scenario_id: str
    policy_rows: list[dict[str, Any]] = field(default_factory=list)
    transition_rows: list[dict[str, Any]] = field(default_factory=list)
    mismatches: list[dict[str, Any]] = field(default_factory=list)

    def policy_batch_audit(
        self,
        transition_index: int,
        policy: PolicyDefinition,
        builds: Sequence[ObservationBuild],
        payloads: Sequence[Mapping[str, Any]],
        decisions: Sequence[PolicyDecision],
    ) -> None:
        audit_builds = []
        for build, payload in zip(builds, payloads, strict=True):
            audit_builds.append(
                replace(
                    build,
                    observation=replace(build.observation, payload=dict(payload)),
                )
            )
        encoded = encode_masked_observations(
            [audit_builds], [[policy] * len(audit_builds)], device=self.device
        )
        gpu = decide_masked_observations(encoded)
        selected = gpu.selected_candidate_index.detach().cpu()[0].tolist()
        expected = [item.selected_candidate_key for item in decisions]
        observed = [None if index < 0 else f"c{index:02d}" for index in selected]
        for actor_slot, (cpu_key, gpu_key) in enumerate(
            zip(expected, observed, strict=True)
        ):
            success = cpu_key == gpu_key
            case_id = (
                f"{self.phase}:{self.scenario_id}:policy:{transition_index}:"
                f"{actor_slot}"
            )
            self.policy_rows.append(
                {
                    "caseId": case_id,
                    "phase": self.phase,
                    "scenarioId": self.scenario_id,
                    "transitionIndex": transition_index,
                    "actorSlot": actor_slot,
                    "policyId": policy.policy_id,
                    "cpuCandidateKey": cpu_key,
                    "gpuCandidateKey": gpu_key,
                    "success": success,
                }
            )
            if not success:
                self.mismatches.append(
                    {
                        "schemaVersion": "e06.s08.mismatch-fixture.v1",
                        "researchStepId": "S08",
                        "caseId": case_id,
                        "phase": self.phase,
                        "severity": "release_critical",
                        "triageStatus": "unresolved",
                        "rootCauseClassification": "gpu_masked_policy_decision_divergence",
                        "input": {
                            "policyId": policy.policy_id,
                            "payload": payloads[actor_slot],
                        },
                        "cpu": {"selectedCandidateKey": cpu_key},
                        "gpu": {"selectedCandidateKey": gpu_key},
                    }
                )

    def transition_audit(
        self,
        transition_index: int,
        transition_kind: str,
        environment: Environment,
        state: MovementState,
        proposals: Sequence[MovementProposal],
        batch_nonce: str,
        cpu_batch: Mapping[str, Any],
    ) -> None:
        case_id = (
            f"{self.phase}:{self.scenario_id}:transition:{transition_index}:"
            f"{transition_kind}"
        )
        rows, mismatches = compare_transition_cases(
            environment,
            [
                TransitionCase(
                    case_id=case_id,
                    phase=self.phase,
                    state=state,
                    proposals=tuple(proposals),
                    batch_nonce=batch_nonce,
                    metadata={
                        "scenarioId": self.scenario_id,
                        "transitionIndex": transition_index,
                        "transitionKind": transition_kind,
                    },
                    expected_batch=cpu_batch,
                )
            ],
            device=self.device,
        )
        self.transition_rows.extend(rows)
        self.mismatches.extend(mismatches)


def run_differential_episode(
    context: EngineContext,
    definition: EpisodeDefinition,
    *,
    device: torch.device | str,
    phase: str,
    include_selected_traces: bool = False,
) -> tuple[dict[str, Any], DifferentialRecorder]:
    recorder = DifferentialRecorder(
        device=device, phase=phase, scenario_id=definition.scenario_id
    )
    result = run_cpu_episode(
        context,
        definition,
        include_selected_traces=include_selected_traces,
        policy_batch_audit=recorder.policy_batch_audit,
        transition_audit=recorder.transition_audit,
    )
    return result, recorder
