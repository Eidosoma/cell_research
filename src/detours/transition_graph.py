"""Exact finite legal-opportunity graphs for the E03 S04 state families.

The production kernels are deliberately independent of the object-oriented E01
proposal implementation.  The public oracle helper below calls E01 directly so
tests and artifact validation can compare two separate implementations.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Iterable

import numba as nb
import numpy as np

from reference_simulator.engine import evaluate_terminal
from reference_simulator.model import Architecture, Direction, FaultMode, Policy, ProposalKind
from reference_simulator.policies import cell_view_proposal, traditional_proposal
from reference_simulator.transition_primitives import commit_proposal, cost_delta, validate_proposal
from src.detours.distances import distance_profile
from src.detours.state_space import FamilySpec, StructuralState, cell_index


EDGE_SCHEMA = "e03.s05.legal_opportunity_edge.v1"
NODE_SCHEMA = "e03.s05.graph_node_annotation.v1"
GRAPH_PROJECTION = "fair_serial_legal_opportunity_set_v1"
EDGE_DIGEST_DOMAIN = b"E03/S05/family-edges/v1\x00"
NODE_DIGEST_DOMAIN = b"E03/S05/family-nodes/v1\x00"

POLICY_CODES = {Policy.BUBBLE: 0, Policy.INSERTION: 1, Policy.SELECTION: 2}
FAULT_CODES = {FaultMode.NORMAL: 0, FaultMode.PASSIVE: 1, FaultMode.STUCK: 2}
ARCHITECTURE_CODES = {Architecture.CELL_VIEW: 0, Architecture.TRADITIONAL: 1}
DIRECTION_CODES = {Direction.ASCENDING: 0, Direction.DESCENDING: 1}

PROPOSAL_KIND_LABELS = {0: "NoOp", 1: "Swap", 2: "MemoryUpdate"}
PROPOSAL_KIND_CODES = {value: key for key, value in PROPOSAL_KIND_LABELS.items()}
DECISION_LABELS = {
    0: "no_op",
    1: "accepted",
    2: "rejected_target_stuck",
    3: "rejected_actor_stuck",
}
DECISION_CODES = {value: key for key, value in DECISION_LABELS.items()}
REASON_LABELS = {
    0: "actor_fault",
    1: "boundary",
    2: "ordered_or_equal",
    3: "prefix_not_ordered",
    4: "cursor_exhausted",
    5: "at_cursor",
    6: "skip_stuck_target",
    7: "target_already_extreme",
    8: "strict_adjacent_inversion",
    9: "insertion_into_ordered_prefix",
    10: "selection_target",
    11: "traditional_global_bubble_first_inversion",
    12: "traditional_global_insertion_adjacent_action",
    13: "traditional_global_selection_extreme",
    14: "traditional_no_inversion",
    15: "traditional_singleton",
}
REASON_CODES = {value: key for key, value in REASON_LABELS.items()}
TERMINAL_LABELS = {0: "active", 1: "complete", 2: "quiescent"}

EDGE_ARRAY_COLUMNS = (
    "source_state_ordinal",
    "successor_state_ordinal",
    "opportunity_ordinal",
    "scheduler_actor_index",
    "scheduler_side_code",
    "probability_numerator",
    "probability_denominator",
    "proposal_kind_code",
    "proposal_reason_code",
    "decision_code",
    "proposal_actor_index",
    "actor_position",
    "target_position",
    "new_cursor",
    "changed",
    "observation_reads",
    "value_comparisons",
    "cost_no_ops",
    "cost_rejections",
    "cost_memory_updates",
    "cost_accepted_swaps",
    "cost_displaced_cells",
    "delta_adjacent_descents",
    "delta_paper_distance_numerator",
    "delta_inversion_count",
    "delta_spearman_footrule",
    "delta_maximum_rank_error",
    "delta_emd_numerator",
)


def metric_denominators(n: int) -> dict[str, int]:
    if n < 2:
        raise ValueError("S05 graph families require n >= 2")
    return {
        "normalized_adjacent_descents": n - 1,
        "paper_sortedness_distance": n,
        "normalized_kendall_distance": n * (n - 1) // 2,
        "normalized_spearman_footrule": n * n // 2,
        "normalized_maximum_rank_error": n - 1,
        "duplicate_aware_earth_movers_distance": n,
        "normalized_duplicate_aware_earth_movers_distance": n * n // 2,
    }


def opportunity_arrays(family: FamilySpec) -> dict[str, np.ndarray]:
    """Return the frozen, state-independent opportunity labels for a family."""

    actors: list[int] = []
    sides: list[int] = []
    numerators: list[int] = []
    denominators: list[int] = []
    if family.architecture == Architecture.TRADITIONAL:
        actors.append(-1)
        sides.append(-1)
        numerators.append(1)
        denominators.append(1)
    else:
        for actor, (policy, fault) in enumerate(zip(family.policies, family.faults)):
            if policy == Policy.BUBBLE and fault == FaultMode.NORMAL:
                for side in (0, 1):
                    actors.append(actor)
                    sides.append(side)
                    numerators.append(1)
                    denominators.append(2 * family.n)
            else:
                actors.append(actor)
                sides.append(-1)
                numerators.append(1)
                denominators.append(family.n)
    return {
        "actors": np.asarray(actors, dtype=np.int8),
        "sides": np.asarray(sides, dtype=np.int8),
        "probability_numerators": np.asarray(numerators, dtype=np.uint8),
        "probability_denominators": np.asarray(denominators, dtype=np.uint8),
    }


def numeric_family(family: FamilySpec) -> dict[str, Any]:
    owner_index = np.full(family.n, -1, dtype=np.int8)
    owner_multiplier = np.zeros(family.n, dtype=np.int64)
    multiplier = 1
    for owner_position, owner in enumerate(family.selection_owners):
        owner_index[owner] = owner_position
        owner_multiplier[owner] = multiplier
        multiplier *= family.n + 1
    facts = np.asarray([math.factorial(value) for value in range(family.n + 1)], dtype=np.int64)
    opportunities = opportunity_arrays(family)
    return {
        "n": family.n,
        "architecture": ARCHITECTURE_CODES[family.architecture],
        "direction": DIRECTION_CODES[family.direction],
        "policies": np.asarray([POLICY_CODES[item] for item in family.policies], dtype=np.int8),
        "faults": np.asarray([FAULT_CODES[item] for item in family.faults], dtype=np.int8),
        "owner_index": owner_index,
        "owner_multiplier": owner_multiplier,
        "facts": facts,
        **opportunities,
    }


@nb.njit(cache=True)
def _unrank_into(n: int, rank: int, facts: np.ndarray, occupancy: np.ndarray) -> None:
    available = np.empty(n, dtype=np.int8)
    for index in range(n):
        available[index] = index
    available_count = n
    remainder = rank
    for position in range(n):
        factor = facts[n - position - 1]
        choice = remainder // factor
        remainder %= factor
        occupancy[position] = available[choice]
        for shift in range(choice, available_count - 1):
            available[shift] = available[shift + 1]
        available_count -= 1


@nb.njit(cache=True)
def _rank_occupancy(occupancy: np.ndarray, n: int, facts: np.ndarray) -> int:
    available = np.empty(n, dtype=np.int8)
    for index in range(n):
        available[index] = index
    available_count = n
    rank = 0
    for position in range(n):
        value = occupancy[position]
        choice = 0
        while available[choice] != value:
            choice += 1
        rank += choice * facts[n - position - 1]
        for shift in range(choice, available_count - 1):
            available[shift] = available[shift + 1]
        available_count -= 1
    return rank


@nb.njit(cache=True)
def _positions(occupancy: np.ndarray, n: int) -> np.ndarray:
    result = np.empty(n, dtype=np.int8)
    for position in range(n):
        result[occupancy[position]] = position
    return result


@nb.njit(cache=True)
def _metrics(occupancy: np.ndarray, n: int, direction: int) -> tuple[int, int, int, int]:
    adjacent = 0
    inversions = 0
    footrule = 0
    maximum_rank = 0
    for position in range(n):
        value = occupancy[position]
        target = value if direction == 0 else n - 1 - value
        displacement = abs(position - target)
        footrule += displacement
        if displacement > maximum_rank:
            maximum_rank = displacement
        if position + 1 < n:
            if (direction == 0 and value > occupancy[position + 1]) or (
                direction == 1 and value < occupancy[position + 1]
            ):
                adjacent += 1
        for right in range(position + 1, n):
            if (direction == 0 and value > occupancy[right]) or (
                direction == 1 and value < occupancy[right]
            ):
                inversions += 1
    return adjacent, inversions, footrule, maximum_rank


@nb.njit(cache=True)
def _is_complete(occupancy: np.ndarray, n: int, direction: int) -> bool:
    for position in range(n - 1):
        if (direction == 0 and occupancy[position] > occupancy[position + 1]) or (
            direction == 1 and occupancy[position] < occupancy[position + 1]
        ):
            return False
    return True


@nb.njit(cache=True)
def _finish_swap(
    occupancy: np.ndarray,
    n: int,
    direction: int,
    facts: np.ndarray,
    source_rank: int,
    cursor_code: int,
    actor: int,
    actor_position: int,
    target_position: int,
    faults: np.ndarray,
    reason: int,
    reads: int,
    comparisons: int,
    source_adjacent: int,
    source_inversions: int,
    source_footrule: int,
    source_maximum_rank: int,
    compute_metrics: bool,
) -> tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]:
    target = occupancy[target_position]
    if faults[actor] == 2:
        return (
            cursor_code * facts[n] + source_rank, 1, reason, 3, actor,
            actor_position, target_position, -128, 0, reads, comparisons, 0, 0, 0, 0,
        )
    if faults[target] == 2:
        return (
            cursor_code * facts[n] + source_rank, 1, reason, 2, actor,
            actor_position, target_position, -128, 0, reads, comparisons, 0, 0, 0, 0,
        )
    occupancy[actor_position], occupancy[target_position] = (
        occupancy[target_position], occupancy[actor_position]
    )
    successor_rank = _rank_occupancy(occupancy, n, facts)
    delta_adjacent = 0
    delta_inversions = 0
    delta_footrule = 0
    delta_maximum_rank = 0
    if compute_metrics:
        adjacent, inversions, footrule, maximum_rank = _metrics(occupancy, n, direction)
        delta_adjacent = adjacent - source_adjacent
        delta_inversions = inversions - source_inversions
        delta_footrule = footrule - source_footrule
        delta_maximum_rank = maximum_rank - source_maximum_rank
    occupancy[actor_position], occupancy[target_position] = (
        occupancy[target_position], occupancy[actor_position]
    )
    return (
        cursor_code * facts[n] + successor_rank, 1, reason, 1, actor,
        actor_position, target_position, -128, 1, reads, comparisons,
        delta_adjacent, delta_inversions, delta_footrule, delta_maximum_rank,
    )


@nb.njit(cache=True)
def _transition_one(
    occupancy: np.ndarray,
    positions: np.ndarray,
    n: int,
    architecture: int,
    direction: int,
    policies: np.ndarray,
    faults: np.ndarray,
    owner_multiplier: np.ndarray,
    facts: np.ndarray,
    source_rank: int,
    cursor_code: int,
    scheduled_actor: int,
    scheduled_side: int,
    source_adjacent: int,
    source_inversions: int,
    source_footrule: int,
    source_maximum_rank: int,
    compute_metrics: bool,
) -> tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]:
    source_ordinal = cursor_code * facts[n] + source_rank
    if architecture == 1:
        policy = policies[0]
        reads = 0
        comparisons = 0
        if n < 2:
            return source_ordinal, 0, 15, 0, -1, -1, -1, -128, 0, 0, 0, 0, 0, 0, 0
        if policy == 0:
            for left in range(n - 1):
                a = occupancy[left]
                b = occupancy[left + 1]
                reads += 2
                comparisons += 1
                inversion = (direction == 0 and a > b) or (direction == 1 and a < b)
                if inversion:
                    return _finish_swap(
                        occupancy, n, direction, facts, source_rank, cursor_code,
                        a, left, left + 1, faults, 11, reads, comparisons,
                        source_adjacent, source_inversions, source_footrule,
                        source_maximum_rank, compute_metrics,
                    )
        elif policy == 1:
            for right in range(1, n):
                a = occupancy[right - 1]
                b = occupancy[right]
                reads += 2
                comparisons += 1
                inversion = (direction == 0 and a > b) or (direction == 1 and a < b)
                if inversion:
                    return _finish_swap(
                        occupancy, n, direction, facts, source_rank, cursor_code,
                        b, right, right - 1, faults, 12, reads, comparisons,
                        source_adjacent, source_inversions, source_footrule,
                        source_maximum_rank, compute_metrics,
                    )
        else:
            for boundary in range(n - 1):
                extreme = boundary
                for index in range(boundary + 1, n):
                    reads += 2
                    comparisons += 1
                    candidate = occupancy[index]
                    current = occupancy[extreme]
                    if (direction == 0 and candidate < current) or (
                        direction == 1 and candidate > current
                    ):
                        extreme = index
                if extreme != boundary:
                    actor = occupancy[extreme]
                    return _finish_swap(
                        occupancy, n, direction, facts, source_rank, cursor_code,
                        actor, extreme, boundary, faults, 13, reads, comparisons,
                        source_adjacent, source_inversions, source_footrule,
                        source_maximum_rank, compute_metrics,
                    )
        return source_ordinal, 0, 14, 0, -1, -1, -1, -128, 0, reads, comparisons, 0, 0, 0, 0

    actor = scheduled_actor
    actor_position = positions[actor]
    if faults[actor] != 0:
        return source_ordinal, 0, 0, 0, actor, actor_position, -1, -128, 0, 1, 0, 0, 0, 0, 0
    policy = policies[actor]
    if policy == 0:
        target_position = actor_position - 1 if scheduled_side == 0 else actor_position + 1
        if target_position < 0 or target_position >= n:
            return source_ordinal, 0, 1, 0, actor, actor_position, -1, -128, 0, 1, 0, 0, 0, 0, 0
        target = occupancy[target_position]
        if direction == 0:
            inversion = actor < target if scheduled_side == 0 else actor > target
        else:
            inversion = actor > target if scheduled_side == 0 else actor < target
        if not inversion:
            return source_ordinal, 0, 2, 0, actor, actor_position, -1, -128, 0, 2, 1, 0, 0, 0, 0
        return _finish_swap(
            occupancy, n, direction, facts, source_rank, cursor_code,
            actor, actor_position, target_position, faults, 8, 2, 1,
            source_adjacent, source_inversions, source_footrule,
            source_maximum_rank, compute_metrics,
        )

    if policy == 1:
        if actor_position == 0:
            return source_ordinal, 0, 1, 0, actor, actor_position, -1, -128, 0, 1, 0, 0, 0, 0, 0
        reads = 0
        comparisons = 0
        prior_value = -1
        prior_normal = False
        for position in range(actor_position):
            value = occupancy[position]
            reads += 1
            if faults[value] != 0:
                prior_normal = False
                prior_value = -1
                continue
            if prior_normal:
                comparisons += 1
                ordered = (direction == 0 and prior_value <= value) or (
                    direction == 1 and prior_value >= value
                )
                if not ordered:
                    return source_ordinal, 0, 3, 0, actor, actor_position, -1, -128, 0, reads + 1, comparisons, 0, 0, 0, 0
            prior_normal = True
            prior_value = value
        target_position = actor_position - 1
        target = occupancy[target_position]
        inversion = (direction == 0 and actor < target) or (direction == 1 and actor > target)
        if not inversion:
            return source_ordinal, 0, 2, 0, actor, actor_position, -1, -128, 0, reads + 2, comparisons + 1, 0, 0, 0, 0
        return _finish_swap(
            occupancy, n, direction, facts, source_rank, cursor_code,
            actor, actor_position, target_position, faults, 9,
            reads + 2, comparisons + 1, source_adjacent, source_inversions,
            source_footrule, source_maximum_rank, compute_metrics,
        )

    multiplier = owner_multiplier[actor]
    digit = (cursor_code // multiplier) % (n + 1)
    cursor = digit if direction == 0 else n - 1 - digit
    if cursor < 0 or cursor >= n:
        return source_ordinal, 0, 4, 0, actor, actor_position, -1, -128, 0, 1, 0, 0, 0, 0, 0
    if cursor == actor_position:
        return source_ordinal, 0, 5, 0, actor, actor_position, -1, -128, 0, 1, 0, 0, 0, 0, 0
    target = occupancy[cursor]
    new_cursor = cursor + (1 if direction == 0 else -1)
    if faults[target] == 2:
        return (
            (cursor_code + multiplier) * facts[n] + source_rank,
            2, 6, 1, actor, actor_position, -1, new_cursor, 1, 2, 0, 0, 0, 0, 0,
        )
    if target <= actor:
        return (
            (cursor_code + multiplier) * facts[n] + source_rank,
            2, 7, 1, actor, actor_position, -1, new_cursor, 1, 2, 1, 0, 0, 0, 0,
        )
    return _finish_swap(
        occupancy, n, direction, facts, source_rank, cursor_code,
        actor, actor_position, cursor, faults, 10, 2, 1,
        source_adjacent, source_inversions, source_footrule,
        source_maximum_rank, compute_metrics,
    )


@nb.njit(cache=True)
def _classify_states_kernel(
    state_count: int,
    n: int,
    architecture: int,
    direction: int,
    policies: np.ndarray,
    faults: np.ndarray,
    owner_multiplier: np.ndarray,
    facts: np.ndarray,
    opportunity_actors: np.ndarray,
    opportunity_sides: np.ndarray,
) -> np.ndarray:
    result = np.empty(state_count, dtype=np.uint8)
    occupancy = np.empty(n, dtype=np.int8)
    factorial = facts[n]
    for state_ordinal in range(state_count):
        rank = state_ordinal % factorial
        cursor_code = state_ordinal // factorial
        _unrank_into(n, rank, facts, occupancy)
        if _is_complete(occupancy, n, direction):
            result[state_ordinal] = 1
            continue
        positions = _positions(occupancy, n)
        changed = False
        for opportunity in range(len(opportunity_actors)):
            transition = _transition_one(
                occupancy, positions, n, architecture, direction, policies, faults,
                owner_multiplier, facts, rank, cursor_code,
                opportunity_actors[opportunity], opportunity_sides[opportunity],
                0, 0, 0, 0, False,
            )
            if transition[8] == 1:
                changed = True
                break
        result[state_ordinal] = 0 if changed else 2
    return result


@nb.njit(cache=True)
def _reachability_kernel(
    terminal_codes: np.ndarray,
    n: int,
    architecture: int,
    direction: int,
    policies: np.ndarray,
    faults: np.ndarray,
    owner_multiplier: np.ndarray,
    facts: np.ndarray,
    opportunity_actors: np.ndarray,
    opportunity_sides: np.ndarray,
) -> np.ndarray:
    state_count = len(terminal_codes)
    reachable = np.zeros(state_count, dtype=np.uint8)
    queue = np.empty(state_count, dtype=np.uint32)
    queue[0] = 0
    reachable[0] = 1
    head = 0
    tail = 1
    occupancy = np.empty(n, dtype=np.int8)
    factorial = facts[n]
    while head < tail:
        state_ordinal = int(queue[head])
        head += 1
        if terminal_codes[state_ordinal] != 0:
            continue
        rank = state_ordinal % factorial
        cursor_code = state_ordinal // factorial
        _unrank_into(n, rank, facts, occupancy)
        positions = _positions(occupancy, n)
        for opportunity in range(len(opportunity_actors)):
            transition = _transition_one(
                occupancy, positions, n, architecture, direction, policies, faults,
                owner_multiplier, facts, rank, cursor_code,
                opportunity_actors[opportunity], opportunity_sides[opportunity],
                0, 0, 0, 0, False,
            )
            successor = transition[0]
            if reachable[successor] == 0:
                reachable[successor] = 1
                queue[tail] = successor
                tail += 1
    return reachable


@nb.njit(cache=True)
def _edge_batch_kernel(
    start: int,
    stop: int,
    terminal_codes: np.ndarray,
    n: int,
    architecture: int,
    direction: int,
    policies: np.ndarray,
    faults: np.ndarray,
    owner_multiplier: np.ndarray,
    facts: np.ndarray,
    opportunity_actors: np.ndarray,
    opportunity_sides: np.ndarray,
    probability_numerators: np.ndarray,
    probability_denominators: np.ndarray,
) -> tuple:
    maximum = (stop - start) * len(opportunity_actors)
    source = np.empty(maximum, dtype=np.uint32)
    successor = np.empty(maximum, dtype=np.uint32)
    opportunity_ordinal = np.empty(maximum, dtype=np.uint8)
    scheduler_actor = np.empty(maximum, dtype=np.int8)
    scheduler_side = np.empty(maximum, dtype=np.int8)
    probability_numerator = np.empty(maximum, dtype=np.uint8)
    probability_denominator = np.empty(maximum, dtype=np.uint8)
    proposal_kind = np.empty(maximum, dtype=np.uint8)
    proposal_reason = np.empty(maximum, dtype=np.uint8)
    decision = np.empty(maximum, dtype=np.uint8)
    proposal_actor = np.empty(maximum, dtype=np.int8)
    actor_position = np.empty(maximum, dtype=np.int8)
    target_position = np.empty(maximum, dtype=np.int8)
    new_cursor = np.empty(maximum, dtype=np.int8)
    changed = np.empty(maximum, dtype=np.uint8)
    observation_reads = np.empty(maximum, dtype=np.uint16)
    value_comparisons = np.empty(maximum, dtype=np.uint16)
    cost_no_ops = np.empty(maximum, dtype=np.uint8)
    cost_rejections = np.empty(maximum, dtype=np.uint8)
    cost_memory_updates = np.empty(maximum, dtype=np.uint8)
    cost_accepted_swaps = np.empty(maximum, dtype=np.uint8)
    cost_displaced_cells = np.empty(maximum, dtype=np.uint8)
    delta_adjacent = np.empty(maximum, dtype=np.int8)
    delta_paper = np.empty(maximum, dtype=np.int8)
    delta_inversions = np.empty(maximum, dtype=np.int16)
    delta_footrule = np.empty(maximum, dtype=np.int16)
    delta_maximum_rank = np.empty(maximum, dtype=np.int8)
    delta_emd = np.empty(maximum, dtype=np.int16)
    occupancy = np.empty(n, dtype=np.int8)
    factorial = facts[n]
    output = 0
    for state_ordinal in range(start, stop):
        if terminal_codes[state_ordinal] != 0:
            continue
        rank = state_ordinal % factorial
        cursor_code = state_ordinal // factorial
        _unrank_into(n, rank, facts, occupancy)
        positions = _positions(occupancy, n)
        src_adjacent, src_inversions, src_footrule, src_maximum_rank = _metrics(
            occupancy, n, direction
        )
        for opportunity in range(len(opportunity_actors)):
            transition = _transition_one(
                occupancy, positions, n, architecture, direction, policies, faults,
                owner_multiplier, facts, rank, cursor_code,
                opportunity_actors[opportunity], opportunity_sides[opportunity],
                src_adjacent, src_inversions, src_footrule, src_maximum_rank, True,
            )
            source[output] = state_ordinal
            successor[output] = transition[0]
            opportunity_ordinal[output] = opportunity
            scheduler_actor[output] = opportunity_actors[opportunity]
            scheduler_side[output] = opportunity_sides[opportunity]
            probability_numerator[output] = probability_numerators[opportunity]
            probability_denominator[output] = probability_denominators[opportunity]
            proposal_kind[output] = transition[1]
            proposal_reason[output] = transition[2]
            decision[output] = transition[3]
            proposal_actor[output] = transition[4]
            actor_position[output] = transition[5]
            target_position[output] = transition[6]
            new_cursor[output] = transition[7]
            changed[output] = transition[8]
            observation_reads[output] = transition[9]
            value_comparisons[output] = transition[10]
            cost_no_ops[output] = 1 if transition[1] == 0 else 0
            cost_rejections[output] = 1 if transition[3] >= 2 else 0
            cost_memory_updates[output] = 1 if transition[1] == 2 and transition[3] == 1 else 0
            cost_accepted_swaps[output] = 1 if transition[1] == 1 and transition[3] == 1 else 0
            cost_displaced_cells[output] = 2 * cost_accepted_swaps[output]
            delta_adjacent[output] = transition[11]
            delta_paper[output] = transition[11]
            delta_inversions[output] = transition[12]
            delta_footrule[output] = transition[13]
            delta_maximum_rank[output] = transition[14]
            delta_emd[output] = transition[13]
            output += 1
    return (
        output, source, successor, opportunity_ordinal, scheduler_actor,
        scheduler_side, probability_numerator, probability_denominator,
        proposal_kind, proposal_reason, decision, proposal_actor,
        actor_position, target_position, new_cursor, changed,
        observation_reads, value_comparisons, cost_no_ops, cost_rejections,
        cost_memory_updates, cost_accepted_swaps, cost_displaced_cells,
        delta_adjacent, delta_paper, delta_inversions, delta_footrule,
        delta_maximum_rank, delta_emd,
    )


def classify_states(family: FamilySpec) -> np.ndarray:
    values = numeric_family(family)
    return _classify_states_kernel(
        family.state_count,
        values["n"], values["architecture"], values["direction"],
        values["policies"], values["faults"], values["owner_multiplier"],
        values["facts"], values["actors"], values["sides"],
    )


def reachable_states(family: FamilySpec, terminal_codes: np.ndarray) -> np.ndarray:
    values = numeric_family(family)
    return _reachability_kernel(
        terminal_codes,
        values["n"], values["architecture"], values["direction"],
        values["policies"], values["faults"], values["owner_multiplier"],
        values["facts"], values["actors"], values["sides"],
    )


def edge_batch(
    family: FamilySpec,
    terminal_codes: np.ndarray,
    start: int,
    stop: int,
) -> dict[str, np.ndarray]:
    if not 0 <= start <= stop <= family.state_count:
        raise ValueError("invalid state batch")
    values = numeric_family(family)
    result = _edge_batch_kernel(
        start, stop, terminal_codes,
        values["n"], values["architecture"], values["direction"],
        values["policies"], values["faults"], values["owner_multiplier"],
        values["facts"], values["actors"], values["sides"],
        values["probability_numerators"], values["probability_denominators"],
    )
    count = int(result[0])
    return {
        column: result[index + 1][:count]
        for index, column in enumerate(EDGE_ARRAY_COLUMNS)
    }


def warm_graph_kernels() -> None:
    """Compile kernels once in the parent before forking workers."""

    family = FamilySpec(
        n=4,
        architecture=Architecture.CELL_VIEW,
        direction=Direction.ASCENDING,
        policies=(Policy.BUBBLE,) * 4,
        faults=(FaultMode.NORMAL,) * 4,
    )
    terminal = classify_states(family)
    reachable_states(family, terminal)
    edge_batch(family, terminal, 0, family.state_count)


def _oracle_metrics(predecessor: StructuralState, successor: StructuralState) -> dict[str, int]:
    before = asdict(
        distance_profile(predecessor.occupancy, direction=predecessor.family.direction.value)
    )
    after = asdict(
        distance_profile(successor.occupancy, direction=successor.family.direction.value)
    )
    return {
        "delta_adjacent_descents": int(after["adjacent_descents"] - before["adjacent_descents"]),
        "delta_paper_distance_numerator": int(after["adjacent_descents"] - before["adjacent_descents"]),
        "delta_inversion_count": int(after["inversion_count"] - before["inversion_count"]),
        "delta_spearman_footrule": int(after["spearman_footrule"] - before["spearman_footrule"]),
        "delta_maximum_rank_error": int(after["maximum_rank_error"] - before["maximum_rank_error"]),
        "delta_emd_numerator": int(after["spearman_footrule"] - before["spearman_footrule"]),
    }


def e01_outgoing_edges(family: FamilySpec, state_ordinal: int) -> list[dict[str, int]]:
    """Independently enumerate one node using the authoritative E01 objects."""

    if not 0 <= state_ordinal < family.state_count:
        raise ValueError("state ordinal outside family")
    factorial = family.occupancy_state_count
    predecessor = StructuralState(
        family,
        occupancy_rank=state_ordinal % factorial,
        selection_cursor_code=state_ordinal // factorial,
    )
    scenario = family.scenario()
    run_state = predecessor.to_run_state()
    terminal = evaluate_terminal(scenario, run_state)
    if terminal in {"complete", "quiescent"}:
        return []
    opportunities = opportunity_arrays(family)
    rows: list[dict[str, int]] = []
    for ordinal, (actor, side, probability_numerator, probability_denominator) in enumerate(
        zip(
            opportunities["actors"],
            opportunities["sides"],
            opportunities["probability_numerators"],
            opportunities["probability_denominators"],
        )
    ):
        state = predecessor.to_run_state()
        if family.architecture == Architecture.TRADITIONAL:
            proposal = traditional_proposal(scenario, state)
        else:
            side_value = None if side < 0 else ("left" if side == 0 else "right")
            proposal = cell_view_proposal(scenario, state, f"c{int(actor)}", side=side_value)
        validation = validate_proposal(scenario, state, proposal)
        decision = "accepted" if validation.eligible_for_commit else validation.decision
        if validation.eligible_for_commit:
            snapshot = state.clone()
            commit_proposal(state, snapshot, proposal, decision)
        successor = StructuralState.from_run_state(family, state)
        ledger_delta = cost_delta(predecessor.to_run_state().ledger, proposal, decision)
        proposal_actor = -1 if proposal.actor_id == "__controller__" else cell_index(proposal.actor_id)
        row = {
            "source_state_ordinal": state_ordinal,
            "successor_state_ordinal": (
                successor.selection_cursor_code * factorial + successor.occupancy_rank
            ),
            "opportunity_ordinal": ordinal,
            "scheduler_actor_index": int(actor),
            "scheduler_side_code": int(side),
            "probability_numerator": int(probability_numerator),
            "probability_denominator": int(probability_denominator),
            "proposal_kind_code": PROPOSAL_KIND_CODES[proposal.kind.value],
            "proposal_reason_code": REASON_CODES[proposal.reason],
            "decision_code": DECISION_CODES[decision],
            "proposal_actor_index": proposal_actor,
            "actor_position": proposal.actor_pos,
            "target_position": -1 if proposal.target_pos is None else proposal.target_pos,
            "new_cursor": -128 if proposal.new_cursor is None else proposal.new_cursor,
            "changed": int(successor != predecessor),
            "observation_reads": proposal.observation_reads,
            "value_comparisons": proposal.value_comparisons,
            "cost_no_ops": ledger_delta["noOps"],
            "cost_rejections": ledger_delta["rejections"],
            "cost_memory_updates": ledger_delta["memoryUpdates"],
            "cost_accepted_swaps": ledger_delta["acceptedSwaps"],
            "cost_displaced_cells": ledger_delta["displacedCells"],
            **_oracle_metrics(predecessor, successor),
        }
        rows.append(row)
    return rows


def kernel_outgoing_edges(family: FamilySpec, state_ordinal: int) -> list[dict[str, int]]:
    terminal = classify_states(family)
    batch = edge_batch(family, terminal, state_ordinal, state_ordinal + 1)
    return [
        {column: int(batch[column][row]) for column in EDGE_ARRAY_COLUMNS}
        for row in range(len(batch["source_state_ordinal"]))
    ]


def compare_edge_records(
    observed: Iterable[dict[str, int]], expected: Iterable[dict[str, int]]
) -> tuple[bool, str]:
    left = list(observed)
    right = list(expected)
    if left == right:
        return True, "exact"
    if len(left) != len(right):
        return False, f"edge count {len(left)} != {len(right)}"
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            differing = sorted(key for key in a if a.get(key) != b.get(key))
            return False, f"edge {index} differs in {differing}"
    return False, "unclassified mismatch"
