"""Frozen S10 perturbations, feasibility, repair metrics, and CPU execution.

All simulated lesions are external, deterministic transformations of an exact
S01/S02 target state. They preserve the S04 identity/token/kind/site contract.
Temporary barriers and immobilization suppress authenticated proposals through
the S10 engine gate without exposing their metadata to S05 policy payloads.
Count-changing deletion and excess conditions are feasibility audits only.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .baseline import (
    GRAMMAR_POLICIES,
    TargetMetricTracker,
    load_baseline_assets,
    state_grid,
)
from .engine import EpisodeDefinition, canonical_episode_result_bytes, run_cpu_episode
from .environments import Environment
from .grammar import RelationalGrammar, score_grid
from .movements import (
    MovementProposal,
    MovementState,
    Occupant,
    initial_movement_state,
    movement_state_sha256,
    validate_state_against_environment,
)
from .targets import TargetDefinition, evaluate_success


PERTURBATION_CATALOG_VERSION = "e06.s10.perturbation-catalog.v1"
RUN_RESULT_VERSION = "e06.s10.perturbation-run.v1"
MASTER_SEED_HEX = "0xe0610000000000000000000000000001"
SIMULATED_LESIONS = {
    "compact_hole",
    "temporary_barrier",
    "immobile_cells",
    "compact_wound",
    "formed_structure_displacement",
}
FEASIBILITY_ONLY_LESIONS = {"region_deletion", "cell_type_excess"}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def sha256_value(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_bytes(value)
    ).hexdigest()


def _rank(domain: str, address: str, value: str) -> bytes:
    return hashlib.sha256(
        domain.encode("ascii")
        + b"\x00"
        + address.encode("utf-8")
        + b"\x00"
        + value.encode("utf-8")
    ).digest()


def scenario_identity(
    split: str,
    target_id: str,
    lesion_id: str,
    severity: str,
    replicate: int,
    arm: str,
) -> dict[str, Any]:
    address = {
        "split": split,
        "targetId": target_id,
        "lesionId": lesion_id,
        "severity": severity,
        "replicate": int(replicate),
    }
    seeded = {"masterSeedHex": MASTER_SEED_HEX, "address": address}
    digest = sha256_value("E06/S10/pairing-block/v1", seeded)
    scenario_id = f"s10-{split[:4]}-{digest[:24]}"
    run_id = "run1:" + sha256_value("E06/S10/run/v1", {**seeded, "arm": arm})
    seed_digest = sha256_value("E06/S10/seed/v1", seeded)
    return {
        "scenarioId": scenario_id,
        "pairingBlockId": "pb1:" + digest,
        "runId": run_id,
        "seedHex": "0x" + seed_digest[:32],
        "seedDecimal": str(int(seed_digest[:32], 16)),
    }


def simulated_target_lesion_pairs(catalog: Mapping[str, Any]) -> list[dict[str, str]]:
    targets = list(catalog["targetPolicyMatrix"])
    output: list[dict[str, str]] = []
    for lesion_id, lesion in catalog["lesions"].items():
        if lesion["execution"] != "simulated":
            continue
        eligible = lesion["eligibleTargets"]
        for target in targets:
            if eligible == "all" or target["targetId"] in eligible:
                output.append(
                    {
                        "targetId": str(target["targetId"]),
                        "grammarId": str(target["grammarId"]),
                        "policyId": str(target["policyId"]),
                        "lesionId": str(lesion_id),
                    }
                )
    return sorted(output, key=lambda item: tuple(item.values()))


def _state_with_assignment(
    environment: Environment,
    initial: MovementState,
    assignment: Mapping[str, Occupant],
) -> MovementState:
    state = MovementState(
        environment_id=initial.environment_id,
        environment_sha256=initial.environment_sha256,
        transition_index=0,
        occupancy=tuple(sorted(assignment.items())),
    )
    validate_state_against_environment(environment, state)
    return state


def _coordinate_map(environment: Environment) -> dict[str, tuple[int, int]]:
    return {
        site.site_id: tuple(site.coordinate) for site in environment.occupiable_sites
    }


def _site_map(environment: Environment) -> dict[tuple[int, int], str]:
    return {
        tuple(site.coordinate): site.site_id for site in environment.occupiable_sites
    }


def _neighbor_map(environment: Environment) -> dict[str, set[str]]:
    neighbors = {site.site_id: set() for site in environment.occupiable_sites}
    for first, second in environment.edges:
        neighbors[first].add(second)
        neighbors[second].add(first)
    return neighbors


def _connected_after_removals(
    environment: Environment,
    removed_sites: set[str],
    removed_edges: set[tuple[str, str]] | None = None,
) -> bool:
    removed_edges = removed_edges or set()
    sites = {
        site.site_id
        for site in environment.occupiable_sites
        if site.site_id not in removed_sites
    }
    if not sites:
        return False
    neighbors = _neighbor_map(environment)
    first = min(sites)
    seen = {first}
    queue = deque([first])
    while queue:
        site = queue.popleft()
        for other in neighbors[site]:
            edge = tuple(sorted((site, other)))
            if other in sites and edge not in removed_edges and other not in seen:
                seen.add(other)
                queue.append(other)
    return seen == sites


def _shortest_distance(environment: Environment, source: str, target: str) -> int:
    if source == target:
        return 0
    neighbors = _neighbor_map(environment)
    queue = deque([(source, 0)])
    seen = {source}
    while queue:
        site, distance = queue.popleft()
        for other in neighbors[site]:
            if other == target:
                return distance + 1
            if other not in seen:
                seen.add(other)
                queue.append((other, distance + 1))
    raise ValueError("disconnected perturbation displacement")


def _external_ledger(
    environment: Environment,
    before: MovementState,
    after: MovementState,
) -> dict[str, int]:
    before_site = {item.occupant_id: site for site, item in before.occupancy}
    after_site = {item.occupant_id: site for site, item in after.occupancy}
    by_id = {item.occupant_id: item for _, item in before.occupancy}
    moved = [
        identity
        for identity in before_site
        if before_site[identity] != after_site[identity]
    ]
    distances = {
        identity: _shortest_distance(
            environment, before_site[identity], after_site[identity]
        )
        for identity in moved
    }
    return {
        "externalDisplacedEntities": len(moved),
        "externalDisplacedCells": sum(by_id[item].kind == "cell" for item in moved),
        "externalDisplacedVacancies": sum(
            by_id[item].kind == "vacancy" for item in moved
        ),
        "externalGraphDisplacement": sum(distances.values()),
    }


def _compact_swaps(
    environment: Environment,
    state: MovementState,
    address: str,
    count: int,
) -> tuple[MovementState, list[tuple[str, str]]]:
    coordinates = _coordinate_map(environment)
    rows = int(environment.generator["rows"])
    columns = int(environment.generator["columns"])
    center = ((rows - 1) / 2, (columns - 1) / 2)
    candidates = [
        edge
        for edge in environment.edges
        if state.occupant_map[edge[0]].token != state.occupant_map[edge[1]].token
    ]
    anchors = sorted(
        candidates,
        key=lambda edge: _rank("E06/S10/wound-anchor/v1", address, "|".join(edge)),
    )
    if not anchors:
        raise ValueError("no cross-token edge for compact wound")
    anchor = anchors[0]
    anchor_mid = tuple(
        (coordinates[anchor[0]][axis] + coordinates[anchor[1]][axis]) / 2
        for axis in range(2)
    )
    ranked = sorted(
        candidates,
        key=lambda edge: (
            sum(
                abs(
                    (coordinates[edge[0]][axis] + coordinates[edge[1]][axis]) / 2
                    - anchor_mid[axis]
                )
                for axis in range(2)
            ),
            sum(
                abs(
                    (coordinates[edge[0]][axis] + coordinates[edge[1]][axis]) / 2
                    - center[axis]
                )
                for axis in range(2)
            ),
            _rank("E06/S10/wound-edge/v1", address, "|".join(edge)),
        ),
    )
    selected: list[tuple[str, str]] = []
    reserved: set[str] = set()
    for edge in ranked:
        if reserved.intersection(edge):
            continue
        selected.append(edge)
        reserved.update(edge)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError("insufficient disjoint compact wound edges")
    assignment = state.occupant_map
    for first, second in selected:
        assignment[first], assignment[second] = assignment[second], assignment[first]
    return _state_with_assignment(environment, state, assignment), selected


def _hole_swaps(
    environment: Environment,
    state: MovementState,
    address: str,
    count: int,
) -> tuple[MovementState, list[tuple[str, str]]]:
    coordinates = _coordinate_map(environment)
    rows = int(environment.generator["rows"])
    columns = int(environment.generator["columns"])
    center = ((rows - 1) / 2, (columns - 1) / 2)
    cells = [site for site, occupant in state.occupancy if occupant.kind == "cell"]
    vacancies = [
        site for site, occupant in state.occupancy if occupant.kind == "vacancy"
    ]
    cells.sort(
        key=lambda site: (
            sum(abs(coordinates[site][axis] - center[axis]) for axis in range(2)),
            _rank("E06/S10/hole-cell/v1", address, site),
        )
    )
    vacancies.sort(
        key=lambda site: (
            -sum(abs(coordinates[site][axis] - center[axis]) for axis in range(2)),
            _rank("E06/S10/hole-vacancy/v1", address, site),
        )
    )
    actual = min(count, len(cells), len(vacancies))
    if actual < 1:
        raise ValueError("compact hole requires existing vacancy identities")
    pairs = list(zip(cells[:actual], vacancies[:actual], strict=True))
    assignment = state.occupant_map
    for cell_site, vacancy_site in pairs:
        assignment[cell_site], assignment[vacancy_site] = (
            assignment[vacancy_site],
            assignment[cell_site],
        )
    return _state_with_assignment(environment, state, assignment), pairs


def _displaced_state(
    environment: Environment,
    state: MovementState,
    target: TargetDefinition,
    address: str,
    shift: int,
) -> tuple[MovementState, dict[str, Any]]:
    rows = int(environment.generator["rows"])
    columns = int(environment.generator["columns"])
    sites = _site_map(environment)
    transforms = []
    for axis, delta in ((0, shift), (1, shift), (0, -shift), (1, -shift)):
        transforms.append((axis, delta, False))
    transforms.extend(((0, shift, True), (1, shift, True)))
    transforms.sort(
        key=lambda item: _rank(
            "E06/S10/displacement-choice/v1", address, json.dumps(item)
        )
    )
    for axis, delta, shear in transforms:
        assignment: dict[str, Occupant] = {}
        for row in range(rows):
            for column in range(columns):
                coordinate = [row, column]
                applied = delta
                if shear and ((column if axis == 0 else row) % 2):
                    applied = -delta
                coordinate[axis] = coordinate[axis] - applied
                coordinate[axis] %= rows if axis == 0 else columns
                assignment[sites[(row, column)]] = state.occupant_map[
                    sites[tuple(coordinate)]
                ]
        candidate = _state_with_assignment(environment, state, assignment)
        if not evaluate_success(state_grid(environment, candidate), target)["success"]:
            return candidate, {"axis": axis, "shift": delta, "shear": shear}
    raise ValueError("failed to construct S01-incomplete structure displacement")


def _barrier_edges(
    environment: Environment, address: str, fraction: float
) -> set[tuple[str, str]]:
    rows = int(environment.generator["rows"])
    columns = int(environment.generator["columns"])
    sites = _site_map(environment)
    cut = columns // 2 - 1
    candidates = [
        tuple(sorted((sites[(row, cut)], sites[(row, cut + 1)]))) for row in range(rows)
    ]
    count = min(rows - 1, max(1, round(fraction * rows)))
    ranked = sorted(
        candidates,
        key=lambda edge: (
            abs(_coordinate_map(environment)[edge[0]][0] - (rows - 1) / 2),
            _rank("E06/S10/barrier-edge/v1", address, "|".join(edge)),
        ),
    )
    selected = set(ranked[:count])
    if not _connected_after_removals(environment, set(), selected):
        raise ValueError("barrier must retain a connected movement graph")
    return selected


def _immobile_identities(
    environment: Environment,
    exact_state: MovementState,
    excluded_sites: set[str],
    address: str,
    count: int,
) -> tuple[set[str], set[str]]:
    coordinates = _coordinate_map(environment)
    rows = int(environment.generator["rows"])
    columns = int(environment.generator["columns"])
    candidates = [
        site
        for site, occupant in exact_state.occupancy
        if occupant.kind == "cell" and site not in excluded_sites
    ]
    candidates.sort(
        key=lambda site: (
            min(
                coordinates[site][0],
                rows - 1 - coordinates[site][0],
                coordinates[site][1],
                columns - 1 - coordinates[site][1],
            ),
            _rank("E06/S10/immobile/v1", address, site),
        )
    )
    selected_sites: set[str] = set()
    for site in candidates:
        proposal = selected_sites | {site}
        if _connected_after_removals(environment, proposal):
            selected_sites = proposal
        if len(selected_sites) == count:
            break
    if len(selected_sites) < count:
        raise ValueError("insufficient non-disconnecting immobile identities")
    return (
        {exact_state.occupant_map[site].occupant_id for site in selected_sites},
        selected_sites,
    )


@dataclass(frozen=True)
class LesionApplication:
    lesion_id: str
    severity: str
    state: MovementState
    mask: Mapping[str, Any]
    external_ledger: Mapping[str, int]
    barrier_edges: frozenset[tuple[str, str]]
    immobile_identity_ids: frozenset[str]
    gate_duration_transitions: int
    target_feasible: bool
    feasibility_reason: str


def apply_lesion(
    environment: Environment,
    target: TargetDefinition,
    exact_state: MovementState,
    lesion_id: str,
    severity: str,
    address: str,
    catalog: Mapping[str, Any],
) -> LesionApplication:
    if lesion_id not in SIMULATED_LESIONS:
        raise ValueError("S10 run requested for a feasibility-only lesion")
    parameters = catalog["severities"][severity]
    site_count = len(exact_state.occupancy)
    swap_count = max(
        int(parameters["minimumCrossTokenSwaps"]),
        math.ceil(float(parameters["affectedSiteFraction"]) * site_count / 2),
    )
    barrier_edges: set[tuple[str, str]] = set()
    immobile_ids: set[str] = set()
    mask: dict[str, Any]
    if lesion_id == "compact_hole":
        state, pairs = _hole_swaps(
            environment,
            exact_state,
            address,
            int(parameters["holeSwapCount"]),
        )
        mask = {"sitePairs": [list(item) for item in pairs]}
    elif lesion_id in {"temporary_barrier", "immobile_cells", "compact_wound"}:
        state, pairs = _compact_swaps(environment, exact_state, address, swap_count)
        mask = {"woundSwapEdges": [list(item) for item in pairs]}
        touched = {site for edge in pairs for site in edge}
        if lesion_id == "temporary_barrier":
            barrier_edges = _barrier_edges(
                environment,
                address,
                float(parameters["barrierBlockedEdgeFraction"]),
            )
            mask["barrierEdges"] = [list(item) for item in sorted(barrier_edges)]
        elif lesion_id == "immobile_cells":
            immobile_ids, immobile_sites = _immobile_identities(
                environment,
                exact_state,
                touched,
                address,
                int(parameters["immobileIdentityCount"]),
            )
            mask["immobileSites"] = sorted(immobile_sites)
            mask["immobileIdentityIds"] = sorted(immobile_ids)
    else:
        state, displacement = _displaced_state(
            environment,
            exact_state,
            target,
            address,
            int(parameters["displacementShift"]),
        )
        mask = {
            "displacement": displacement,
            "affectedSites": sorted(state.occupant_map),
        }
    global_after = evaluate_success(state_grid(environment, state), target)
    if global_after["success"]:
        raise ValueError("simulated lesion did not leave the S01 target set")
    before_ids = {item.occupant_id for _, item in exact_state.occupancy}
    after_ids = {item.occupant_id for _, item in state.occupancy}
    before_tokens = Counter(item.token for _, item in exact_state.occupancy)
    after_tokens = Counter(item.token for _, item in state.occupancy)
    before_kinds = Counter(item.kind for _, item in exact_state.occupancy)
    after_kinds = Counter(item.kind for _, item in state.occupancy)
    target_feasible = (
        before_ids == after_ids
        and before_tokens == after_tokens
        and before_kinds == after_kinds
        and _connected_after_removals(
            environment,
            {
                site
                for site, occupant in exact_state.occupancy
                if occupant.occupant_id in immobile_ids
            },
            barrier_edges,
        )
    )
    reason = (
        "exact composition and identity conserved; mobile movement graph connected; "
        "immobile identities, if any, remain on correct target tokens"
        if target_feasible
        else "count-preserving lesion failed the frozen connected-reachability gate"
    )
    mask_record = {
        "schemaVersion": "e06.s10.lesion-mask.v1",
        "lesionId": lesion_id,
        "severity": severity,
        "damageTransition": int(catalog["timing"]["damageTransition"]),
        **mask,
    }
    mask_record["lesionMaskSha256"] = sha256_value(
        "E06/S10/lesion-mask/v1", mask_record
    )
    return LesionApplication(
        lesion_id=lesion_id,
        severity=severity,
        state=state,
        mask=mask_record,
        external_ledger=_external_ledger(environment, exact_state, state),
        barrier_edges=frozenset(barrier_edges),
        immobile_identity_ids=frozenset(immobile_ids),
        gate_duration_transitions=int(parameters["gateDurationTransitions"]),
        target_feasible=target_feasible,
        feasibility_reason=reason,
    )


class PerturbationProposalGate:
    """Deterministically suppress proposals under a frozen S10 affordance lesion."""

    def __init__(self, application: LesionApplication) -> None:
        self.application = application
        self.attempted_proposals = 0
        self.suppressed_proposals = 0
        self.suppressed_route_site_claims = 0
        self.active_transition_count = 0

    def __call__(
        self,
        transition_index: int,
        _environment: Environment,
        _state: MovementState,
        proposals: Sequence[MovementProposal],
    ) -> Sequence[MovementProposal]:
        self.attempted_proposals += len(proposals)
        if transition_index >= self.application.gate_duration_transitions:
            return tuple(proposals)
        self.active_transition_count += 1
        retained = []
        for proposal in proposals:
            route_edges = {
                tuple(sorted(edge)) for edge in zip(proposal.route, proposal.route[1:])
            }
            if proposal.kind == "rotation" and len(proposal.route) > 2:
                route_edges.add(tuple(sorted((proposal.route[-1], proposal.route[0]))))
            expected_ids = {identity for _, identity in proposal.expected_occupants}
            suppressed = bool(
                route_edges.intersection(self.application.barrier_edges)
                or expected_ids.intersection(self.application.immobile_identity_ids)
            )
            if suppressed:
                self.suppressed_proposals += 1
                self.suppressed_route_site_claims += len(proposal.route)
            else:
                retained.append(proposal)
        return tuple(retained)

    def ledger(self) -> dict[str, int]:
        return {
            "gateAttemptedProposals": self.attempted_proposals,
            "suppressedProposals": self.suppressed_proposals,
            "suppressedRouteSiteClaims": self.suppressed_route_site_claims,
            "activeGateTransitions": self.active_transition_count,
        }


def count_changing_feasibility_records(
    catalog: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for target in catalog["targetPolicyMatrix"]:
        for lesion_id in sorted(FEASIBILITY_ONLY_LESIONS):
            for severity in sorted(catalog["severities"]):
                reason = (
                    "S04 removal is deferred; deletion changes identity, kind, and exact S01 counts"
                    if lesion_id == "region_deletion"
                    else "insertion/conversion/division is deferred; excess changes immutable token and exact S01 counts"
                )
                records.append(
                    {
                        "researchStepId": "S10",
                        "targetId": target["targetId"],
                        "lesionId": lesion_id,
                        "severity": severity,
                        "execution": "feasibility_only",
                        "targetFeasible": False,
                        "lesionArmLaunched": False,
                        "noDamageArmLaunched": False,
                        "vacancyAdjustedSemanticsAvailable": False,
                        "countAdjustedTargetAvailable": False,
                        "classification": "infeasible_no_adjusted_contract",
                        "reason": reason,
                    }
                )
    return records


def _episode_definition(
    scenario_id: str,
    environment: Environment,
    grammar: RelationalGrammar,
    policy_id: str,
    event_budget: int,
    catalog: Mapping[str, Any],
) -> EpisodeDefinition:
    parameters: dict[str, Any] = {}
    if policy_id == "memory_based_recovery_v1":
        parameters["memoryInitialBestLocalUtility"] = 12
    return EpisodeDefinition(
        scenario_id=scenario_id,
        environment_id=environment.environment_id,
        policy_id=policy_id,
        relation_grammar_id=(
            grammar.grammar_id if policy_id in GRAMMAR_POLICIES else None
        ),
        channel_mode="none",
        transitions=int(event_budget),
        actor_batch_size=int(catalog["simulation"]["actorBatchSize"]),
        parameters=parameters,
    )


def run_perturbation_once(
    specification: Mapping[str, Any],
) -> tuple[dict[str, Any], Any, Mapping[str, Any]]:
    context, targets, grammars, environments = load_baseline_assets()
    target = targets[str(specification["targetId"])]
    grammar = grammars[str(specification["grammarId"])]
    environment = environments[target.target_id]
    catalog = specification["catalog"]
    identity = scenario_identity(
        str(specification["split"]),
        target.target_id,
        str(specification["lesionId"]),
        str(specification["severity"]),
        int(specification["replicate"]),
        str(specification["arm"]),
    )
    exact_state = initial_movement_state(environment)
    exact_grid = state_grid(environment, exact_state)
    pre_global = evaluate_success(exact_grid, target)
    pre_local = score_grid(exact_grid, grammar)
    pre_conjunctive = bool(pre_global["success"] and pre_local["accepted"])
    if not pre_conjunctive:
        raise ValueError("S10 source must be independently verified as formed")
    application = apply_lesion(
        environment,
        target,
        exact_state,
        str(specification["lesionId"]),
        str(specification["severity"]),
        identity["pairingBlockId"],
        catalog,
    )
    arm = str(specification["arm"])
    initial_state = exact_state if arm == "no_damage" else application.state
    initial_grid = state_grid(environment, initial_state)
    immediate_global = evaluate_success(initial_grid, target)
    immediate_local = score_grid(initial_grid, grammar)
    immediate_conjunctive = bool(
        immediate_global["success"] and immediate_local["accepted"]
    )
    repair_eligible = bool(
        arm == "lesion"
        and pre_conjunctive
        and not immediate_conjunctive
        and application.target_feasible
    )
    tracker = TargetMetricTracker(environment, target, grammar)
    gate = PerturbationProposalGate(application)
    definition = _episode_definition(
        identity["scenarioId"],
        environment,
        grammar,
        str(specification["policyId"]),
        int(specification["eventBudget"]),
        catalog,
    )
    started = time.perf_counter()
    gate_active = arm == "lesion" and application.lesion_id in {
        "temporary_barrier",
        "immobile_cells",
    }
    result = run_cpu_episode(
        context,
        definition,
        include_selected_traces=bool(specification.get("retainTrace", False)),
        initial_state_override=initial_state,
        state_audit=tracker.observe,
        proposal_gate=(gate if gate_active else None),
    )
    elapsed = time.perf_counter() - started
    metrics = tracker.finalize()
    final_state = tracker.last_state
    assert final_state is not None
    gate_ledger = (
        gate.ledger()
        if gate_active
        else {
            "gateAttemptedProposals": 0,
            "suppressedProposals": 0,
            "suppressedRouteSiteClaims": 0,
            "activeGateTransitions": 0,
        }
    )
    external = (
        dict(application.external_ledger)
        if arm == "lesion"
        else {
            "externalDisplacedEntities": 0,
            "externalDisplacedCells": 0,
            "externalDisplacedVacancies": 0,
            "externalGraphDisplacement": 0,
        }
    )
    intervention = {
        "lesionMaskSha256": application.mask["lesionMaskSha256"],
        "preDamageStateSha256": movement_state_sha256(exact_state),
        "postDamageStateSha256": movement_state_sha256(initial_state),
        "targetFeasible": application.target_feasible,
        **external,
        **gate_ledger,
    }
    intervention["interventionSummarySha256"] = sha256_value(
        "E06/S10/intervention-summary/v1", intervention
    )
    initial_ids = {item.occupant_id for _, item in exact_state.occupancy}
    final_ids = {item.occupant_id for _, item in final_state.occupancy}
    initial_tokens = Counter(item.token for _, item in exact_state.occupancy)
    final_tokens = Counter(item.token for _, item in final_state.occupancy)
    initial_kinds = Counter(item.kind for _, item in exact_state.occupancy)
    final_kinds = Counter(item.kind for _, item in final_state.occupancy)
    movement = result["movementLedger"]
    observation = result["observationLedger"]
    channel = result["channelLedger"]
    episode_bytes = canonical_episode_result_bytes(result)
    initial_mismatch = int(metrics["initialS01MismatchCount"])
    closure = (
        (initial_mismatch - int(metrics["minimumS01MismatchCount"])) / initial_mismatch
        if initial_mismatch > 0
        else 0.0
    )
    row = {
        "schemaVersion": RUN_RESULT_VERSION,
        "phase": str(specification["phase"]),
        "split": str(specification["split"]),
        **identity,
        "targetId": target.target_id,
        "grammarId": grammar.grammar_id,
        "environmentId": environment.environment_id,
        "policyId": str(specification["policyId"]),
        "lesionId": str(specification["lesionId"]),
        "severity": str(specification["severity"]),
        "arm": arm,
        "replicate": int(specification["replicate"]),
        "backend": "canonical_s07_cpu_oracle_fallback",
        "eventBudgetTransitions": int(specification["eventBudget"]),
        "damageTransition": int(catalog["timing"]["damageTransition"]),
        "runStatus": "completed",
        "stopReason": "fixed_post_damage_event_budget",
        "failed": False,
        "errorType": None,
        "errorMessage": None,
        "preDamageConjunctive": pre_conjunctive,
        "immediatePostDamageConjunctive": immediate_conjunctive,
        "repairEligible": repair_eligible,
        "lesionRepairByBudget": bool(
            repair_eligible and metrics["conjunctiveCompletionByBudget"]
        ),
        "firstRepairTransition": (
            metrics["firstCompletionTransition"] if repair_eligible else None
        ),
        "censored": bool(
            repair_eligible and not metrics["conjunctiveCompletionByBudget"]
        ),
        "minimumMismatchClosureFraction": closure if repair_eligible else 0.0,
        "targetFeasible": application.target_feasible,
        "feasibilityReason": application.feasibility_reason,
        "lesionMaskSha256": application.mask["lesionMaskSha256"],
        "preDamageStateSha256": movement_state_sha256(exact_state),
        "postDamageStateSha256": movement_state_sha256(initial_state),
        "episodeSha256": result["episodeSha256"],
        "episodeCanonicalBytesSha256": hashlib.sha256(episode_bytes).hexdigest(),
        "initialGridRowsJson": json.dumps(
            ["".join(row_) for row_ in initial_grid], separators=(",", ":")
        ),
        **metrics,
        **external,
        **gate_ledger,
        "interventionSummarySha256": intervention["interventionSummarySha256"],
        "acceptedMovements": int(movement["acceptedMovements"]),
        "submittedProposals": int(movement["submittedProposals"]),
        "conflictLosses": int(movement["conflictLosses"]),
        "invalidProposals": int(movement["invalidProposals"]),
        "totalGraphDisplacement": int(movement["totalGraphDisplacement"]),
        "observationCommunicatedBitsUpperBound": int(
            observation["communicatedBitsUpperBound"]
        ),
        "observationUtilityEvaluations": int(observation["utilityEvaluations"]),
        "channelConfigurationBits": int(channel["configurationBits"]),
        "channelTotalInformationBits": int(channel["totalInformationBits"]),
        "totalInterventionCostUnits": int(
            external["externalGraphDisplacement"]
            + gate_ledger["suppressedProposals"]
            + gate_ledger["suppressedRouteSiteClaims"]
        ),
        "invariantSuccess": bool(
            initial_ids == final_ids
            and initial_tokens == final_tokens
            and initial_kinds == final_kinds
            and movement_state_sha256(final_state)
            == result["finalState"]["stateSha256"]
            and len(result["transitionSummaries"]) == int(specification["eventBudget"])
        ),
        "permissionAuditSuccess": all(
            value is False for value in result["permissionAudit"].values()
        ),
        "wallSeconds": elapsed,
        "traceSelected": bool(specification.get("retainTrace", False)),
    }
    row["metricSummarySha256"] = sha256_value(
        "E06/S10/metric-summary/v1",
        {
            key: row[key]
            for key in sorted(row)
            if key.startswith(
                ("initialS01", "minimumS01", "terminalS01", "terminalS02")
            )
            or key
            in {
                "lesionRepairByBudget",
                "firstRepairTransition",
                "terminalConjunctiveCompletion",
                "localGlobalDiscordance",
                "minimumMismatchClosureFraction",
            }
        },
    )
    trace = result if bool(specification.get("retainTrace", False)) else None
    mask_record = {
        "pairingBlockId": identity["pairingBlockId"],
        "targetId": target.target_id,
        "lesionId": application.lesion_id,
        "severity": application.severity,
        "replicate": int(specification["replicate"]),
        "preDamageStateSha256": movement_state_sha256(exact_state),
        "lesionStateSha256": movement_state_sha256(application.state),
        "targetFeasible": application.target_feasible,
        **application.mask,
        **application.external_ledger,
    }
    return row, trace, mask_record


def run_condition_task(task: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    traces = []
    masks: dict[str, Mapping[str, Any]] = {}
    trace_ids = set(task["traceRunIds"])
    for replicate in task["replicates"]:
        specification = {**task, "replicate": int(replicate)}
        specification.pop("replicates", None)
        identity = scenario_identity(
            str(specification["split"]),
            str(specification["targetId"]),
            str(specification["lesionId"]),
            str(specification["severity"]),
            int(replicate),
            str(specification["arm"]),
        )
        specification["retainTrace"] = identity["runId"] in trace_ids
        try:
            row, trace, mask = run_perturbation_once(specification)
            rows.append(row)
            masks[str(mask["pairingBlockId"])] = mask
            if trace is not None:
                traces.append(
                    {
                        "runId": row["runId"],
                        "reason": "preregistered_one_percent_condition_arm_sample",
                        "episode": trace,
                    }
                )
        except Exception as error:
            rows.append(
                {
                    "schemaVersion": RUN_RESULT_VERSION,
                    "phase": str(specification["phase"]),
                    "split": str(specification["split"]),
                    **identity,
                    "targetId": str(specification["targetId"]),
                    "grammarId": str(specification["grammarId"]),
                    "policyId": str(specification["policyId"]),
                    "lesionId": str(specification["lesionId"]),
                    "severity": str(specification["severity"]),
                    "arm": str(specification["arm"]),
                    "replicate": int(replicate),
                    "runStatus": "failed",
                    "failed": True,
                    "censored": False,
                    "errorType": type(error).__name__,
                    "errorMessage": str(error),
                }
            )
    return {
        "conditionId": task["conditionId"],
        "rows": rows,
        "traces": traces,
        "masks": list(masks.values()),
    }
