"""Frozen S11 two-dimensional chimera construction and CPU execution.

The executable chimera group is an engine-private, identity-owned assignment of
an existing S05 policy plus a priced actor-local relation profile.  Analysis
labels are never passed to :func:`morph2d.engine.run_cpu_episode`.  Physical
occupancy, tokens, identities, topology, movement legality, and the canonical
S02/S01 evaluation contract remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .baseline import (
    TargetMetricTracker,
    load_baseline_assets,
    make_initial_state,
    state_grid,
)
from .engine import EpisodeDefinition, canonical_episode_result_bytes, run_cpu_episode
from .environments import Environment, neighbor_map
from .grammar import RelationalGrammar, score_grid
from .movements import (
    MovementProposal,
    MovementState,
    initial_movement_state,
    movement_state_sha256,
)
from .policies import (
    RelationProfile,
    compile_relation_profile,
    local_contact_utility,
)
from .targets import TargetDefinition, evaluate_success


ROOT = Path(__file__).resolve().parents[2]
CHIMERA_CATALOG_VERSION = "e06.s11.chimera-catalog.v1"
CHIMERA_RUN_VERSION = "e06.s11.chimera-run.v1"
MASTER_SEED_HEX = "0xe0611000000000000000000000000001"
BASE_TARGET_ID = "layers_three_ordered_tissues"
BASE_GRAMMAR_ID = "layers_ordered_contacts_v1"


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


@lru_cache(maxsize=1)
def load_chimera_assets() -> tuple[
    Any,
    TargetDefinition,
    RelationalGrammar,
    Environment,
    Mapping[str, Any],
]:
    context, targets, grammars, environments = load_baseline_assets()
    catalog = yaml.safe_load(
        (ROOT / "configs/morphologies/chimera_catalog.yaml").read_text(encoding="utf-8")
    )
    baseline_catalog = yaml.safe_load(
        (ROOT / "configs/morphologies/baseline_catalog.yaml").read_text(
            encoding="utf-8"
        )
    )
    return (
        context,
        targets[BASE_TARGET_ID],
        grammars[BASE_GRAMMAR_ID],
        environments[BASE_TARGET_ID],
        {"chimera": catalog, "baseline": baseline_catalog},
    )


def transform_relation_profile(
    base: RelationProfile, relation_class: str
) -> RelationProfile:
    """Create the two frozen S11 actor-local profile transformations."""

    if relation_class == "aligned":
        return base
    if relation_class == "overlapping":
        translation = {"A": "A", "B": "C", "C": "B"}
        weights = tuple(
            sorted(
                (
                    translation.get(actor, actor),
                    translation.get(neighbor, neighbor),
                    weight,
                )
                for actor, neighbor, weight in base.actor_neighbor_weights
            )
        )
        return RelationProfile(
            grammar_id="s11_layers_B_C_token_permutation_v1",
            target_id=base.target_id,
            actor_neighbor_weights=weights,
            projected_constraint_ids=tuple(
                f"s11_token_permuted:{item}" for item in base.projected_constraint_ids
            ),
            excluded_constraint_ids=base.excluded_constraint_ids,
        )
    if relation_class == "contradictory":
        return RelationProfile(
            grammar_id="s11_layers_sign_inverted_actor_contacts_v1",
            target_id=base.target_id,
            actor_neighbor_weights=tuple(
                (actor, neighbor, -weight)
                for actor, neighbor, weight in base.actor_neighbor_weights
            ),
            projected_constraint_ids=tuple(
                f"s11_sign_inverted:{item}" for item in base.projected_constraint_ids
            ),
            excluded_constraint_ids=base.excluded_constraint_ids,
        )
    raise ValueError(f"unknown S11 relation class: {relation_class}")


def relation_profile_overlap_audit(base: RelationProfile) -> list[dict[str, Any]]:
    base_map = base.weight_map
    rows = []
    for relation_class in ("aligned", "overlapping", "contradictory"):
        profile = transform_relation_profile(base, relation_class)
        transformed = profile.weight_map
        common = sorted(set(base_map) & set(transformed))
        signs = [
            int(np.sign(base_map[key]) == np.sign(transformed[key])) for key in common
        ]
        exact = [int(base_map[key] == transformed[key]) for key in common]
        rows.append(
            {
                "relationClass": relation_class,
                "profileId": profile.grammar_id,
                "baseNonzeroPairs": len(base_map),
                "transformedNonzeroPairs": len(transformed),
                "commonNonzeroPairs": len(common),
                "sameSignCommonPairs": sum(signs),
                "oppositeSignCommonPairs": len(common) - sum(signs),
                "exactWeightCommonPairs": sum(exact),
                "allBasePairsSignOpposed": bool(
                    common
                    and len(common) == len(base_map) == len(transformed)
                    and sum(signs) == 0
                ),
            }
        )
    return rows


def scenario_identity(
    split: str,
    start_family: str,
    composition_id: str,
    replicate: int,
    mixture_id: str,
    arm: str,
) -> dict[str, str]:
    pairing_address = {
        "masterSeedHex": MASTER_SEED_HEX,
        "split": split,
        "startFamily": start_family,
        "compositionId": composition_id,
        "replicate": int(replicate),
    }
    pairing_digest = sha256_value("E06/S11/pairing-block/v1", pairing_address)
    run_digest = sha256_value(
        "E06/S11/run/v1",
        {**pairing_address, "mixtureId": mixture_id, "arm": arm},
    )
    seed_digest = sha256_value("E06/S11/seed/v1", pairing_address)
    return {
        "scenarioId": f"s11-{split[:4]}-{pairing_digest[:24]}",
        "pairingBlockId": "pb1:" + pairing_digest,
        "runId": "run1:" + run_digest,
        "seedHex": "0x" + seed_digest[:32],
    }


def _token_by_identity(state: MovementState) -> dict[str, str]:
    return {item.occupant_id: item.token for _, item in state.occupancy}


def assign_executable_groups(
    state: MovementState,
    composition_id: str,
    pairing_key: str,
    replicate: int,
) -> dict[str, int]:
    by_token: dict[str, list[str]] = defaultdict(list)
    for _, occupant in state.occupancy:
        by_token[occupant.token].append(occupant.occupant_id)
    if sorted(by_token) != ["A", "B", "C"] or any(
        len(values) != 27 for values in by_token.values()
    ):
        raise ValueError("S11 group assignment requires the 27/27/27 layers target")
    if composition_id == "minority_27_54":
        quotas = {token: 9 for token in by_token}
    elif composition_id == "near_balanced_41_40":
        low_token = ("A", "B", "C")[int(replicate) % 3]
        quotas = {token: 13 if token == low_token else 14 for token in by_token}
    else:
        raise ValueError(f"unknown S11 composition: {composition_id}")
    assignment: dict[str, int] = {}
    for token, identities in sorted(by_token.items()):
        ranked = sorted(
            identities,
            key=lambda item: _rank(
                "E06/S11/executable-group/v1", pairing_key, f"{token}:{item}"
            ),
        )
        group0 = set(ranked[: quotas[token]])
        assignment.update(
            {identity: int(identity not in group0) for identity in ranked}
        )
    return assignment


def assign_ghost_labels(
    state: MovementState, runtime_groups: Mapping[str, int], pairing_key: str
) -> dict[str, int]:
    """Assign exact-count, token-stratified labels nearly orthogonal to runtime group."""

    by_token_group: dict[tuple[str, int], list[str]] = defaultdict(list)
    for _, occupant in state.occupancy:
        by_token_group[
            (occupant.token, int(runtime_groups[occupant.occupant_id]))
        ].append(occupant.occupant_id)
    labels: dict[str, int] = {}
    for token in ("A", "B", "C"):
        group0 = by_token_group[(token, 0)]
        group1 = by_token_group[(token, 1)]
        target_zero = len(group0)
        from_group0 = int(round(target_zero * len(group0) / 27))
        from_group0 = max(0, min(len(group0), from_group0))
        from_group1 = target_zero - from_group0
        if not 0 <= from_group1 <= len(group1):
            raise ValueError("ghost-label contingency is infeasible")
        selected: set[str] = set()
        for runtime_group, identities, count in (
            (0, group0, from_group0),
            (1, group1, from_group1),
        ):
            ranked = sorted(
                identities,
                key=lambda item: _rank(
                    "E06/S11/ghost-label/v1",
                    pairing_key,
                    f"{token}:{runtime_group}:{item}",
                ),
            )
            selected.update(ranked[:count])
        for identity in group0 + group1:
            labels[identity] = int(identity not in selected)
    if Counter(labels.values()) != Counter(runtime_groups.values()):
        raise ValueError("ghost labels changed exact composition")
    return labels


def group_assignment_audit(
    state: MovementState,
    runtime_groups: Mapping[str, int],
    ghost_labels: Mapping[str, int],
) -> dict[str, Any]:
    tokens = _token_by_identity(state)
    runtime_counts = Counter(runtime_groups.values())
    ghost_counts = Counter(ghost_labels.values())
    by_token = Counter(
        (tokens[key], int(value)) for key, value in runtime_groups.items()
    )
    contingency = Counter(
        (int(runtime_groups[key]), int(ghost_labels[key])) for key in runtime_groups
    )
    n = len(runtime_groups)
    expected00 = runtime_counts[0] * ghost_counts[0] / n
    phi = (contingency[(0, 0)] - expected00) / math.sqrt(
        max(
            1e-12,
            runtime_counts[0]
            * runtime_counts[1]
            * ghost_counts[0]
            * ghost_counts[1]
            / (n * n),
        )
    )
    return {
        "runtimeCounts": {
            str(key): int(value) for key, value in sorted(runtime_counts.items())
        },
        "ghostCounts": {
            str(key): int(value) for key, value in sorted(ghost_counts.items())
        },
        "runtimeByToken": {
            f"{token}:{group}": int(value)
            for (token, group), value in sorted(by_token.items())
        },
        "runtimeGhostContingency": {
            f"{first}:{second}": int(value)
            for (first, second), value in sorted(contingency.items())
        },
        "runtimeGhostPhi": float(phi),
        "success": bool(
            set(runtime_groups) == set(ghost_labels) == set(tokens)
            and runtime_counts == ghost_counts
        ),
    }


def _site_labels(
    state: MovementState, labels_by_identity: Mapping[str, int]
) -> dict[str, int]:
    return {
        site_id: int(labels_by_identity[occupant.occupant_id])
        for site_id, occupant in state.occupancy
    }


def _same_edge_count(environment: Environment, labels: Mapping[str, int]) -> int:
    return sum(
        int(labels[first] == labels[second]) for first, second in environment.edges
    )


def graph_label_metrics(
    environment: Environment,
    state: MovementState,
    labels_by_identity: Mapping[str, int],
) -> dict[str, Any]:
    site_labels = _site_labels(state, labels_by_identity)
    counts = Counter(site_labels.values())
    n = sum(counts.values())
    edges = environment.edges
    same = _same_edge_count(environment, site_labels)
    raw = same / len(edges)
    expected = sum(value * (value - 1) for value in counts.values()) / (n * (n - 1))
    stub_counts = Counter()
    directed = Counter()
    for first, second in edges:
        left, right = site_labels[first], site_labels[second]
        stub_counts[left] += 1
        stub_counts[right] += 1
        directed[(left, right)] += 1
        directed[(right, left)] += 1
    total_stubs = 2 * len(edges)
    chance_stub = sum((value / total_stubs) ** 2 for value in stub_counts.values())
    assortativity = (
        None if chance_stub == 1 else (raw - chance_stub) / (1 - chance_stub)
    )
    mutual_information = 0.0
    entropy = 0.0
    for value in stub_counts.values():
        probability = value / total_stubs
        entropy -= probability * math.log2(probability)
    for (left, right), value in directed.items():
        joint = value / total_stubs
        left_p = stub_counts[left] / total_stubs
        right_p = stub_counts[right] / total_stubs
        mutual_information += joint * math.log2(joint / (left_p * right_p))
    nmi = None if entropy == 0 else mutual_information / entropy
    signed_nmi = (
        None
        if nmi is None or assortativity is None
        else math.copysign(nmi, assortativity)
    )
    neighbors = neighbor_map(environment)
    visited: set[str] = set()
    component_sizes: dict[int, list[int]] = defaultdict(list)
    for site_id in sorted(site_labels):
        if site_id in visited:
            continue
        label = site_labels[site_id]
        queue = deque([site_id])
        visited.add(site_id)
        size = 0
        while queue:
            current = queue.popleft()
            size += 1
            for other in neighbors[current]:
                if other not in visited and site_labels[other] == label:
                    visited.add(other)
                    queue.append(other)
        component_sizes[label].append(size)
    captures = {
        label: max(values) / counts[label] for label, values in component_sizes.items()
    }
    return {
        "population": n,
        "edgeCount": len(edges),
        "sameGroupEdgeCount": same,
        "crossGroupInterfaceEdges": len(edges) - same,
        "rawHomotypicEdgeFraction": raw,
        "exactCompositionExpectation": expected,
        "compositionCorrectedHomotypy": raw - expected,
        "categoricalAssortativity": assortativity,
        "neighborNmi": nmi,
        "signedNeighborNmi": signed_nmi,
        "domainCount": sum(len(values) for values in component_sizes.values()),
        "largestGroupComponentCapture": max(captures.values()),
        "meanGroupComponentCapture": float(np.mean(list(captures.values()))),
    }


def decluster_assignment(
    environment: Environment,
    state: MovementState,
    old_groups: Mapping[str, int],
    pairing_key: str,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Reduce same-group edges while preserving exact within-token counts."""

    tokens = _token_by_identity(state)
    positions = {occupant.occupant_id: site_id for site_id, occupant in state.occupancy}
    coordinates = {
        site.site_id: site.coordinate for site in environment.occupiable_sites
    }
    by_token_sites: dict[str, list[str]] = defaultdict(list)
    for identity, token in tokens.items():
        by_token_sites[token].append(positions[identity])
    quotas = Counter(
        (tokens[identity], int(group)) for identity, group in old_groups.items()
    )
    old_site_labels = _site_labels(state, old_groups)
    old_same = _same_edge_count(environment, old_site_labels)
    neighbors = neighbor_map(environment)

    candidates: list[tuple[int, int, str, dict[str, int], int]] = []
    for phase in (0, 1):
        site_labels: dict[str, int] = {}
        for token, sites in sorted(by_token_sites.items()):
            group0_count = quotas[(token, 0)]
            ranked = sorted(
                sites,
                key=lambda site_id: (
                    int(sum(coordinates[site_id]) % 2 != phase),
                    -len(neighbors[site_id]),
                    _rank(
                        "E06/S11/decluster-seed/v1",
                        pairing_key,
                        f"{phase}:{token}:{site_id}",
                    ),
                ),
            )
            chosen = set(ranked[:group0_count])
            site_labels.update(
                {site_id: int(site_id not in chosen) for site_id in sites}
            )
        compute_units = 0
        while True:
            current_same = _same_edge_count(environment, site_labels)
            best: tuple[int, bytes, str, str] | None = None
            for token, sites in sorted(by_token_sites.items()):
                zeros = [site for site in sites if site_labels[site] == 0]
                ones = [site for site in sites if site_labels[site] == 1]
                for first in zeros:
                    for second in ones:
                        compute_units += 1
                        changed = dict(site_labels)
                        changed[first], changed[second] = 1, 0
                        delta = _same_edge_count(environment, changed) - current_same
                        tie = _rank(
                            "E06/S11/decluster-swap/v1",
                            pairing_key,
                            f"{phase}:{token}:{first}:{second}",
                        )
                        proposal = (delta, tie, first, second)
                        if best is None or proposal < best:
                            best = proposal
            if best is None or best[0] >= 0:
                break
            site_labels[best[2]], site_labels[best[3]] = 1, 0
        new_groups = {
            occupant.occupant_id: site_labels[site_id]
            for site_id, occupant in state.occupancy
        }
        changes = sum(int(new_groups[key] != old_groups[key]) for key in old_groups)
        same = _same_edge_count(environment, site_labels)
        candidates.append(
            (
                same,
                changes,
                sha256_value("E06/S11/decluster-candidate/v1", new_groups),
                new_groups,
                compute_units,
            )
        )
    best_candidate = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    if best_candidate[0] > old_same:
        new_groups = dict(old_groups)
        new_same = old_same
        changes = 0
    else:
        new_groups = dict(best_candidate[3])
        new_same = int(best_candidate[0])
        changes = int(best_candidate[1])
    if Counter(new_groups.values()) != Counter(old_groups.values()):
        raise ValueError("declustering changed global group composition")
    if Counter((tokens[key], value) for key, value in new_groups.items()) != Counter(
        (tokens[key], value) for key, value in old_groups.items()
    ):
        raise ValueError("declustering changed group-by-token composition")
    return new_groups, {
        "algorithm": "deterministic_checkerboard_seed_plus_within_token_pair_swap_descent",
        "preSameGroupEdges": old_same,
        "postSameGroupEdges": new_same,
        "immediateEdgeDrop": old_same - new_same,
        "assignmentChanges": changes,
        "computeUnits": int(sum(item[4] for item in candidates)),
        "preGroupAssignmentSha256": sha256_value(
            "E06/S11/group-assignment/v1", old_groups
        ),
        "postGroupAssignmentSha256": sha256_value(
            "E06/S11/group-assignment/v1", new_groups
        ),
        "physicalStateSha256Before": movement_state_sha256(state),
        "physicalStateSha256After": movement_state_sha256(state),
        "statePreserved": True,
        "compositionPreserved": True,
        "sameGroupEdgesNonincreasing": new_same <= old_same,
    }


def _profile_and_policy_maps(
    actor_groups: Mapping[str, int], mixture: Mapping[str, Any], base: RelationProfile
) -> tuple[dict[str, str], dict[str, RelationProfile]]:
    group_profiles = {
        0: base,
        1: transform_relation_profile(base, str(mixture["relationClass"])),
    }
    group_policies = {
        0: str(mixture["group0Policy"]),
        1: str(mixture["group1Policy"]),
    }
    return (
        {
            identity: group_policies[int(group)]
            for identity, group in actor_groups.items()
        },
        {
            identity: group_profiles[int(group)]
            for identity, group in actor_groups.items()
        },
    )


class ChimeraTracker:
    """Read-only state, target, label-state, and flux tracker."""

    def __init__(
        self,
        environment: Environment,
        target: TargetDefinition,
        grammar: RelationalGrammar,
        initial_groups: Mapping[str, int],
        initial_profiles: Mapping[str, RelationProfile],
        ghost_labels: Mapping[str, int],
        *,
        checkpoint_every: int,
        transition_budget: int,
        switch_index: int | None = None,
        switched_groups: Mapping[str, int] | None = None,
        switched_profiles: Mapping[str, RelationProfile] | None = None,
        retain_projection: bool = False,
    ) -> None:
        self.environment = environment
        self.target_tracker = TargetMetricTracker(environment, target, grammar)
        self.grammar = grammar
        self.initial_groups = dict(initial_groups)
        self.initial_profiles = dict(initial_profiles)
        self.ghost_labels = dict(ghost_labels)
        self.checkpoint_every = int(checkpoint_every)
        self.transition_budget = int(transition_budget)
        self.switch_index = switch_index
        self.switched_groups = (
            None if switched_groups is None else dict(switched_groups)
        )
        self.switched_profiles = (
            None if switched_profiles is None else dict(switched_profiles)
        )
        self.retain_projection = retain_projection
        self.checkpoints: list[dict[str, Any]] = []
        self.checkpoint_states: dict[int, MovementState] = {}
        self.previous_state: MovementState | None = None
        self.window_moved_events = 0
        self.window_turnover = 0
        self.window_accepted = 0
        self.window_group_moved = Counter()

    def groups_at(self, transition_index: int) -> Mapping[str, int]:
        if (
            self.switch_index is not None
            and transition_index >= self.switch_index
            and self.switched_groups is not None
        ):
            return self.switched_groups
        return self.initial_groups

    def profiles_at(self, transition_index: int) -> Mapping[str, RelationProfile]:
        if (
            self.switch_index is not None
            and transition_index >= self.switch_index
            and self.switched_profiles is not None
        ):
            return self.switched_profiles
        return self.initial_profiles

    def observe(
        self, transition_index: int, state: MovementState, summary: Mapping[str, Any]
    ) -> None:
        self.target_tracker.observe(transition_index, state, summary)
        groups = self.groups_at(transition_index)
        if self.previous_state is not None:
            before = {
                occupant.occupant_id: site_id
                for site_id, occupant in self.previous_state.occupancy
            }
            after = {
                occupant.occupant_id: site_id for site_id, occupant in state.occupancy
            }
            moved = [
                identity for identity in before if before[identity] != after[identity]
            ]
            self.window_moved_events += len(moved)
            self.window_turnover += sum(
                int(
                    self.previous_state.occupant_map[site_id].occupant_id
                    != state.occupant_map[site_id].occupant_id
                )
                for site_id in self.previous_state.occupant_map
            )
            self.window_group_moved.update(groups[identity] for identity in moved)
            self.window_accepted += int(summary.get("acceptedCount", 0))
        self.previous_state = state
        checkpoint = transition_index == -1 or (
            (transition_index + 1) % self.checkpoint_every == 0
            or transition_index == self.transition_budget - 1
        )
        if not checkpoint:
            return
        primary = graph_label_metrics(self.environment, state, groups)
        ghost = graph_label_metrics(self.environment, state, self.ghost_labels)
        collapsed = graph_label_metrics(
            self.environment, state, {identity: 0 for identity in groups}
        )
        grid = state_grid(self.environment, state)
        local = score_grid(grid, self.grammar)
        profiles = self.profiles_at(transition_index)
        utilities = [
            local_contact_utility(
                self.environment, state, identity, profiles[identity]
            )[0]
            for identity in sorted(groups)
        ]
        record = {
            "transitionIndex": transition_index,
            "primary": primary,
            "ghost": ghost,
            "collapsed": collapsed,
            "s02GrammarAccepted": bool(local["accepted"]),
            "s02RelationalScore": float(local["relationalScore"]),
            "meanPrivateActorLocalUtility": float(np.mean(utilities)),
            "acceptedMovementsSincePriorCheckpoint": self.window_accepted,
            "identityMoveEventsSincePriorCheckpoint": self.window_moved_events,
            "occupancyTurnoverSitesSincePriorCheckpoint": self.window_turnover,
            "group0MoveEventsSincePriorCheckpoint": int(self.window_group_moved[0]),
            "group1MoveEventsSincePriorCheckpoint": int(self.window_group_moved[1]),
            "stateSha256": movement_state_sha256(state),
        }
        if self.retain_projection:
            record["occupantIdsBySite"] = [
                [site_id, occupant.occupant_id] for site_id, occupant in state.occupancy
            ]
        self.checkpoints.append(record)
        self.checkpoint_states[transition_index] = state
        self.window_moved_events = 0
        self.window_turnover = 0
        self.window_accepted = 0
        self.window_group_moved = Counter()

    def finalize(self) -> dict[str, Any]:
        target = self.target_tracker.finalize()
        values = [
            item["primary"]["compositionCorrectedHomotypy"] for item in self.checkpoints
        ]
        ghost_values = [
            item["ghost"]["compositionCorrectedHomotypy"] for item in self.checkpoints
        ]
        progress = np.linspace(0.0, 1.0, len(values))
        positive_area = float(np.trapezoid(np.maximum(values, 0.0), progress))
        ghost_area = float(np.trapezoid(np.maximum(ghost_values, 0.0), progress))
        tail = self.checkpoints[-4:]
        tail_accepted = sum(
            item["acceptedMovementsSincePriorCheckpoint"] for item in tail
        )
        tail_turnover = sum(
            item["occupancyTurnoverSitesSincePriorCheckpoint"] for item in tail
        )
        tail_range = max(
            item["primary"]["compositionCorrectedHomotypy"] for item in tail
        ) - min(item["primary"]["compositionCorrectedHomotypy"] for item in tail)
        if tail_accepted == 0 and tail_turnover == 0:
            flux_class = "fixed_tail_not_convergence_claim"
        elif tail_range <= 0.02 and tail_turnover > 0:
            flux_class = "active_stable_macrostate_description"
        else:
            flux_class = "active_changing_macrostate_description"
        return {
            **target,
            "initialCompositionCorrectedHomotypy": values[0],
            "terminalCompositionCorrectedHomotypy": values[-1],
            "peakCompositionCorrectedHomotypy": max(values),
            "positiveAreaCompositionCorrectedHomotypy": positive_area,
            "terminalCategoricalAssortativity": self.checkpoints[-1]["primary"][
                "categoricalAssortativity"
            ],
            "terminalSignedNeighborNmi": self.checkpoints[-1]["primary"][
                "signedNeighborNmi"
            ],
            "terminalLargestGroupComponentCapture": self.checkpoints[-1]["primary"][
                "largestGroupComponentCapture"
            ],
            "terminalDomainCount": self.checkpoints[-1]["primary"]["domainCount"],
            "ghostPeakCompositionCorrectedHomotypy": max(ghost_values),
            "ghostPositiveAreaCompositionCorrectedHomotypy": ghost_area,
            "collapsedMaxAbsoluteCorrectedHomotypy": max(
                abs(item["collapsed"]["compositionCorrectedHomotypy"])
                for item in self.checkpoints
            ),
            "tailAcceptedMovements": tail_accepted,
            "tailOccupancyTurnoverSites": tail_turnover,
            "tailCorrectedHomotypyRange": tail_range,
            "stateFluxClass": flux_class,
            "checkpointCount": len(self.checkpoints),
        }


class KineticAudit:
    def __init__(
        self,
        initial_groups: Mapping[str, int],
        *,
        switch_index: int | None = None,
        switched_groups: Mapping[str, int] | None = None,
    ) -> None:
        self.initial_groups = dict(initial_groups)
        self.switch_index = switch_index
        self.switched_groups = (
            None if switched_groups is None else dict(switched_groups)
        )
        self.submitted = Counter()
        self.accepted = Counter()
        self.conflict_lost = Counter()
        self.cross_group_targets = 0

    def groups_at(self, transition_index: int) -> Mapping[str, int]:
        if (
            self.switch_index is not None
            and transition_index >= self.switch_index
            and self.switched_groups is not None
        ):
            return self.switched_groups
        return self.initial_groups

    def observe(
        self,
        transition_index: int,
        _kind: str,
        _environment: Environment,
        pre_state: MovementState,
        proposals: Sequence[MovementProposal],
        _nonce: str,
        batch: Mapping[str, Any],
    ) -> None:
        groups = self.groups_at(transition_index)
        proposal_by_id = {item.proposal_id: item for item in proposals}
        accepted = set(batch["acceptedProposalIds"])
        locations = {
            occupant.occupant_id: site_id for site_id, occupant in pre_state.occupancy
        }
        for proposal in proposals:
            group = groups[proposal.actor_id]
            self.submitted[group] += 1
            if proposal.proposal_id in accepted:
                self.accepted[group] += 1
            if proposal.target_site is not None:
                target = pre_state.occupant_map[proposal.target_site]
                if target.kind == "cell" and groups[target.occupant_id] != group:
                    self.cross_group_targets += 1
            if proposal.actor_id not in locations:
                raise ValueError("proposal actor missing from kinetic audit state")
        for decision in batch["decisions"]:
            if decision["outcome"] == "conflict_lost":
                proposal = proposal_by_id[decision["proposalId"]]
                self.conflict_lost[groups[proposal.actor_id]] += 1

    def finalize(self) -> dict[str, int]:
        return {
            "group0SubmittedProposals": int(self.submitted[0]),
            "group1SubmittedProposals": int(self.submitted[1]),
            "group0AcceptedActorProposals": int(self.accepted[0]),
            "group1AcceptedActorProposals": int(self.accepted[1]),
            "group0ConflictLosses": int(self.conflict_lost[0]),
            "group1ConflictLosses": int(self.conflict_lost[1]),
            "crossGroupActorTargetProposals": int(self.cross_group_targets),
        }


def _activation_opportunities(
    scenario_id: str,
    actor_ids: Sequence[str],
    groups_before: Mapping[str, int],
    groups_after: Mapping[str, int] | None,
    switch_index: int | None,
    transitions: int,
    batch_size: int,
) -> Counter[int]:
    counts: Counter[int] = Counter()
    for transition_index in range(transitions):
        groups = (
            groups_after
            if groups_after is not None
            and switch_index is not None
            and transition_index >= switch_index
            else groups_before
        )
        assert groups is not None
        ranked = sorted(
            actor_ids,
            key=lambda actor_id: hashlib.sha256(
                b"E06/S07/actor-schedule/v1\x00"
                + scenario_id.encode("utf-8")
                + b"\x00"
                + transition_index.to_bytes(8, "big")
                + b"\x00"
                + actor_id.encode("utf-8")
            ).digest(),
        )
        counts.update(groups[identity] for identity in ranked[:batch_size])
    return counts


def _episode_definition(
    scenario_id: str, environment: Environment, transitions: int, batch_size: int
) -> EpisodeDefinition:
    return EpisodeDefinition(
        scenario_id=scenario_id,
        environment_id=environment.environment_id,
        policy_id="greedy_neighbor_satisfaction_v1",
        relation_grammar_id=BASE_GRAMMAR_ID,
        channel_mode="none",
        transitions=int(transitions),
        actor_batch_size=int(batch_size),
        parameters={},
    )


def _sham_trajectory(
    environment: Environment,
    checkpoints: Sequence[Mapping[str, Any]],
    checkpoint_states: Mapping[int, MovementState],
    pre_groups: Mapping[str, int],
    post_labels: Mapping[str, int],
    switch_index: int,
) -> list[dict[str, Any]]:
    rows = []
    for checkpoint in checkpoints:
        transition_index = int(checkpoint["transitionIndex"])
        labels = post_labels if transition_index >= switch_index else pre_groups
        rows.append(
            {
                "transitionIndex": transition_index,
                "metrics": graph_label_metrics(
                    environment, checkpoint_states[transition_index], labels
                ),
            }
        )
    return rows


def run_chimera_pair_once(
    specification: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    context, target, grammar, environment, catalogs = load_chimera_assets()
    catalog = specification["catalog"]
    mixture = next(
        item
        for item in catalog["mixtureArchetypes"]
        if item["mixtureId"] == specification["mixtureId"]
    )
    identity = scenario_identity(
        str(specification["split"]),
        str(specification["startFamily"]),
        str(specification["compositionId"]),
        int(specification["replicate"]),
        str(specification["mixtureId"]),
        "native",
    )
    exact_state = initial_movement_state(environment)
    if specification["startFamily"] == "exact_formed":
        initial_state = exact_state
    elif specification["startFamily"] == "partially_correct":
        initial_state = make_initial_state(
            environment,
            target,
            "partially_correct",
            identity["pairingBlockId"],
            catalogs["baseline"],
        )
    else:
        raise ValueError("unsupported S11 initial-state family")
    initial_groups = assign_executable_groups(
        initial_state,
        str(specification["compositionId"]),
        identity["pairingBlockId"],
        int(specification["replicate"]),
    )
    ghost_labels = assign_ghost_labels(
        initial_state, initial_groups, identity["pairingBlockId"]
    )
    assignment_audit = group_assignment_audit(
        initial_state, initial_groups, ghost_labels
    )
    base_profile = compile_relation_profile(grammar)
    initial_policies, initial_profiles = _profile_and_policy_maps(
        initial_groups, mixture, base_profile
    )
    transition_budget = int(specification["eventBudget"])
    switch_index = int(specification["interventionTransition"])
    definition = _episode_definition(
        identity["scenarioId"],
        environment,
        transition_budget,
        int(catalog["simulation"]["actorBatchSize"]),
    )
    retain_projection = bool(specification.get("retainNullTrajectory", False))
    requested_arms = tuple(
        map(
            str,
            specification.get("requestedArms", ("native", "executable_decluster")),
        )
    )
    if not requested_arms or not set(requested_arms) <= {
        "native",
        "executable_decluster",
    }:
        raise ValueError("invalid requested S11 arm set")
    native_tracker = ChimeraTracker(
        environment,
        target,
        grammar,
        initial_groups,
        initial_profiles,
        ghost_labels,
        checkpoint_every=int(catalog["stateMetrics"]["checkpointEveryTransitions"]),
        transition_budget=transition_budget,
        retain_projection=retain_projection,
    )
    native_kinetic = KineticAudit(initial_groups)
    started = time.perf_counter()
    native_result = run_cpu_episode(
        context,
        definition,
        include_selected_traces=bool(specification.get("retainTrace", False)),
        initial_state_override=initial_state,
        state_audit=native_tracker.observe,
        transition_audit=native_kinetic.observe,
        actor_policy_assignments=initial_policies,
        actor_relation_profiles=initial_profiles,
    )
    native_elapsed = time.perf_counter() - started
    checkpoint_state = native_tracker.checkpoint_states[switch_index - 1]
    switched_groups, intervention = decluster_assignment(
        environment,
        checkpoint_state,
        initial_groups,
        identity["pairingBlockId"] + f":{specification['mixtureId']}",
    )
    switched_policies, switched_profiles = _profile_and_policy_maps(
        switched_groups, mixture, base_profile
    )
    intervention["policySwitches"] = sum(
        int(initial_policies[key] != switched_policies[key]) for key in initial_policies
    )
    intervention["grammarSwitches"] = sum(
        int(initial_profiles[key].grammar_id != switched_profiles[key].grammar_id)
        for key in initial_profiles
    )
    intervention["controllerInputBits"] = int(
        catalog["interventions"]["costs"]["stateReadBitsPerIdentity"]
    ) * len(initial_groups)
    intervention["configurationBits"] = int(
        catalog["interventions"]["costs"]["fixedControllerConfigurationBits"]
    )
    intervention["physicalGraphDisplacement"] = 0
    intervention["s04MovementLedgerMutation"] = False
    decluster_tracker = None
    decluster_kinetic = None
    decluster_result = None
    decluster_elapsed = 0.0
    prefix_equal = True
    if "executable_decluster" in requested_arms:
        decluster_tracker = ChimeraTracker(
            environment,
            target,
            grammar,
            initial_groups,
            initial_profiles,
            ghost_labels,
            checkpoint_every=int(catalog["stateMetrics"]["checkpointEveryTransitions"]),
            transition_budget=transition_budget,
            switch_index=switch_index,
            switched_groups=switched_groups,
            switched_profiles=switched_profiles,
            retain_projection=retain_projection,
        )
        decluster_kinetic = KineticAudit(
            initial_groups,
            switch_index=switch_index,
            switched_groups=switched_groups,
        )
        started = time.perf_counter()
        decluster_result = run_cpu_episode(
            context,
            definition,
            include_selected_traces=bool(specification.get("retainTrace", False)),
            initial_state_override=initial_state,
            state_audit=decluster_tracker.observe,
            transition_audit=decluster_kinetic.observe,
            actor_policy_assignments=initial_policies,
            actor_relation_profiles=initial_profiles,
            actor_assignment_switches={
                switch_index: (switched_policies, switched_profiles)
            },
        )
        decluster_elapsed = time.perf_counter() - started
        prefix_equal = (
            native_result["transitionSummaries"][:switch_index]
            == decluster_result["transitionSummaries"][:switch_index]
        )
        if not prefix_equal:
            raise ValueError("paired chimera arms diverged before the intervention")
    sham_trajectory = _sham_trajectory(
        environment,
        native_tracker.checkpoints,
        native_tracker.checkpoint_states,
        initial_groups,
        switched_groups,
        switch_index,
    )
    old_immediate = graph_label_metrics(environment, checkpoint_state, initial_groups)
    new_immediate = graph_label_metrics(environment, checkpoint_state, switched_groups)
    intervention["immediateCorrectedHomotypyBefore"] = old_immediate[
        "compositionCorrectedHomotypy"
    ]
    intervention["immediateCorrectedHomotypyAfter"] = new_immediate[
        "compositionCorrectedHomotypy"
    ]
    intervention["labelShamUsesSamePartition"] = True
    intervention["labelShamRuntimeExecuted"] = False
    intervention["labelShamRuntimeBytesEqualByConstruction"] = True
    sham_post = [
        item["metrics"]["compositionCorrectedHomotypy"]
        for item in sham_trajectory
        if int(item["transitionIndex"]) >= switch_index
    ]
    sham_recovery = (
        max(sham_post) - float(new_immediate["compositionCorrectedHomotypy"])
        if sham_post
        else 0.0
    )

    actor_ids = tuple(sorted(initial_groups))
    native_opportunities = _activation_opportunities(
        identity["scenarioId"],
        actor_ids,
        initial_groups,
        None,
        None,
        transition_budget,
        definition.actor_batch_size,
    )
    decluster_opportunities = _activation_opportunities(
        identity["scenarioId"],
        actor_ids,
        initial_groups,
        switched_groups,
        switch_index,
        transition_budget,
        definition.actor_batch_size,
    )
    rows = []
    traces = []
    arm_records = {
        "native": (
            native_result,
            native_tracker,
            native_kinetic,
            native_elapsed,
            native_opportunities,
        ),
        "executable_decluster": (
            decluster_result,
            decluster_tracker,
            decluster_kinetic,
            decluster_elapsed,
            decluster_opportunities,
        ),
    }
    for arm in requested_arms:
        result, tracker, kinetic, elapsed, opportunities = arm_records[arm]
        if result is None or tracker is None or kinetic is None:
            raise ValueError("requested S11 arm was not executed")
        arm_identity = scenario_identity(
            str(specification["split"]),
            str(specification["startFamily"]),
            str(specification["compositionId"]),
            int(specification["replicate"]),
            str(specification["mixtureId"]),
            arm,
        )
        metrics = tracker.finalize()
        arm_post = [
            item["primary"]["compositionCorrectedHomotypy"]
            for item in tracker.checkpoints
            if int(item["transitionIndex"]) >= switch_index
        ]
        arm_recovery = (
            max(arm_post) - float(new_immediate["compositionCorrectedHomotypy"])
            if arm == "executable_decluster" and arm_post
            else 0.0
        )
        movement = result["movementLedger"]
        observation = result["observationLedger"]
        channel = result["channelLedger"]
        final_tokens = Counter(
            item[2] for item in result["finalState"]["occupantProjection"]
        )
        initial_tokens = Counter(item.token for _, item in initial_state.occupancy)
        initial_global = evaluate_success(
            state_grid(environment, initial_state), target
        )
        row = {
            "schemaVersion": CHIMERA_RUN_VERSION,
            "phase": str(specification["phase"]),
            "split": str(specification["split"]),
            **arm_identity,
            "targetId": target.target_id,
            "grammarId": grammar.grammar_id,
            "environmentId": environment.environment_id,
            "mixtureId": str(specification["mixtureId"]),
            "relationClass": str(mixture["relationClass"]),
            "group0Policy": str(mixture["group0Policy"]),
            "group1Policy": str(mixture["group1Policy"]),
            "compositionId": str(specification["compositionId"]),
            "startFamily": str(specification["startFamily"]),
            "replicate": int(specification["replicate"]),
            "interventionArm": arm,
            "interventionTransition": switch_index,
            "backend": catalog["backend"]["production"],
            "eventBudgetTransitions": transition_budget,
            "actorBatchSize": definition.actor_batch_size,
            "runStatus": "completed",
            "stopReason": "fixed_event_budget",
            "failed": False,
            "censored": bool(
                specification["startFamily"] == "partially_correct"
                and not metrics["conjunctiveCompletionByBudget"]
            ),
            "initialConjunctiveCompletion": bool(
                initial_global["success"]
                and score_grid(state_grid(environment, initial_state), grammar)[
                    "accepted"
                ]
            ),
            "initialStateSha256": movement_state_sha256(initial_state),
            "finalStateSha256": result["finalState"]["stateSha256"],
            "episodeSha256": result["episodeSha256"],
            "episodeCanonicalBytesSha256": hashlib.sha256(
                canonical_episode_result_bytes(result)
            ).hexdigest(),
            "initialGroupAssignmentSha256": sha256_value(
                "E06/S11/group-assignment/v1", initial_groups
            ),
            "terminalGroupAssignmentSha256": sha256_value(
                "E06/S11/group-assignment/v1",
                switched_groups if arm == "executable_decluster" else initial_groups,
            ),
            "ghostLabelAssignmentSha256": sha256_value(
                "E06/S11/ghost-assignment/v1", ghost_labels
            ),
            "group0Count": int(Counter(initial_groups.values())[0]),
            "group1Count": int(Counter(initial_groups.values())[1]),
            **metrics,
            **kinetic.finalize(),
            "group0ActivationOpportunities": int(opportunities[0]),
            "group1ActivationOpportunities": int(opportunities[1]),
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
            "chimeraConfigurationBits": 96,
            "interventionControllerInputBits": int(
                intervention["controllerInputBits"]
                if arm == "executable_decluster"
                else 0
            ),
            "interventionConfigurationBits": int(
                intervention["configurationBits"]
                if arm == "executable_decluster"
                else 0
            ),
            "interventionAssignmentChanges": int(
                intervention["assignmentChanges"]
                if arm == "executable_decluster"
                else 0
            ),
            "interventionPolicySwitches": int(
                intervention["policySwitches"] if arm == "executable_decluster" else 0
            ),
            "interventionGrammarSwitches": int(
                intervention["grammarSwitches"] if arm == "executable_decluster" else 0
            ),
            "interventionPhysicalDisplacement": 0,
            "postInterventionRecoveryExtent": arm_recovery,
            "matchedLabelShamRecoveryExtent": sham_recovery,
            "recoveryBeyondMatchedLabelSham": (
                arm_recovery - sham_recovery if arm == "executable_decluster" else 0.0
            ),
            "prefixTransitionIdentityPassed": prefix_equal,
            "invariantSuccess": bool(
                final_tokens == initial_tokens
                and len(result["finalState"]["occupantProjection"])
                == len(initial_state.occupancy)
            ),
            "permissionAuditSuccess": all(
                value is False for value in result["permissionAudit"].values()
            ),
            "wallSeconds": elapsed,
            "traceSelected": bool(specification.get("retainTrace", False)),
        }
        row["metricSummarySha256"] = sha256_value(
            "E06/S11/metric-summary/v1",
            {
                key: row[key]
                for key in sorted(row)
                if key.startswith(
                    (
                        "terminalS01",
                        "terminalS02",
                        "peakComposition",
                        "positiveAreaComposition",
                        "terminalComposition",
                    )
                )
                or key
                in {
                    "conjunctiveCompletionByBudget",
                    "terminalConjunctiveCompletion",
                    "localGlobalDiscordance",
                }
            },
        )
        rows.append(row)
        if specification.get("retainTrace", False):
            traces.append(
                {
                    "runId": row["runId"],
                    "arm": arm,
                    "episode": result,
                    "checkpointMetrics": tracker.checkpoints,
                }
            )
    trajectory_record: dict[str, Any] = {}
    if retain_projection:
        trajectory_record = {
            "pairingBlockId": identity["pairingBlockId"],
            "mixtureId": specification["mixtureId"],
            "compositionId": specification["compositionId"],
            "startFamily": specification["startFamily"],
            "replicate": int(specification["replicate"]),
            "initialGroups": initial_groups,
            "switchedGroups": switched_groups,
            "ghostLabels": ghost_labels,
            "tokenByIdentity": _token_by_identity(initial_state),
            "nativeCheckpoints": native_tracker.checkpoints,
            "declusterCheckpoints": (
                [] if decluster_tracker is None else decluster_tracker.checkpoints
            ),
            "labelShamTrajectory": sham_trajectory,
        }
    return (
        rows,
        traces,
        {
            "pairingBlockId": identity["pairingBlockId"],
            "mixtureId": specification["mixtureId"],
            "compositionId": specification["compositionId"],
            "startFamily": specification["startFamily"],
            "replicate": int(specification["replicate"]),
            "assignmentAudit": assignment_audit,
            "intervention": intervention,
            "trajectory": trajectory_record,
        },
    )


def run_condition_task(task: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    null_replicates = set(
        map(int, task["catalog"]["nulls"]["trajectorySubsetReplicates"])
    )
    trace_run_ids = set(task.get("traceRunIds", []))
    for replicate in task["replicates"]:
        specification = {**task, "replicate": int(replicate)}
        specification.pop("replicates", None)
        requested_arms = tuple(
            map(
                str,
                specification.get("requestedArms", ("native", "executable_decluster")),
            )
        )
        identities = [
            scenario_identity(
                str(specification["split"]),
                str(specification["startFamily"]),
                str(specification["compositionId"]),
                int(replicate),
                str(specification["mixtureId"]),
                arm,
            )["runId"]
            for arm in requested_arms
        ]
        specification["retainTrace"] = any(item in trace_run_ids for item in identities)
        specification["retainNullTrajectory"] = int(replicate) in null_replicates
        try:
            pair_rows, pair_traces, audit = run_chimera_pair_once(specification)
            rows.extend(pair_rows)
            traces.extend(pair_traces)
            audits.append(audit)
        except Exception as error:
            for arm in requested_arms:
                identity = scenario_identity(
                    str(specification["split"]),
                    str(specification["startFamily"]),
                    str(specification["compositionId"]),
                    int(replicate),
                    str(specification["mixtureId"]),
                    arm,
                )
                rows.append(
                    {
                        "schemaVersion": CHIMERA_RUN_VERSION,
                        "phase": specification["phase"],
                        "split": specification["split"],
                        **identity,
                        "mixtureId": specification["mixtureId"],
                        "compositionId": specification["compositionId"],
                        "startFamily": specification["startFamily"],
                        "replicate": int(replicate),
                        "interventionArm": arm,
                        "runStatus": "failed",
                        "stopReason": "execution_error",
                        "failed": True,
                        "censored": False,
                        "errorType": type(error).__name__,
                        "errorMessage": str(error),
                        "traceSelected": True,
                    }
                )
            traces.append(
                {
                    "reason": "execution_failure",
                    "replicate": int(replicate),
                    "errorType": type(error).__name__,
                    "errorMessage": str(error),
                }
            )
    return {"rows": rows, "traces": traces, "audits": audits}
