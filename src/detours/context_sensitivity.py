"""Deterministic larger-n trajectory kernel for E03 S12.

The kernel is a compact projection of the frozen E01 cell-view transition
semantics.  It uses the public external-scheduler contract: actor, Bubble-side,
and synchronous conflict-priority tokens are counter addressed and state
blind.  A parity harness in the S12 builder compares selected runs against the
authoritative Python reference engine supplied with the identical tokens.

Only unique integer values and homogeneous directions are in scope.  Exact
small-state reachability remains outside this module and is carried separately
from S07/S11.
"""

from __future__ import annotations

import hashlib
from typing import Final

import numba as nb
import numpy as np


METRIC_NAMES: Final[tuple[str, ...]] = (
    "adjacent_descents",
    "inversion_count",
    "spearman_footrule",
    "maximum_rank_error",
)
POLICY_NAMES: Final[tuple[str, ...]] = ("Bubble", "Insertion", "Selection")
THRESHOLDS: Final[tuple[int, ...]] = (0, 1, 2, 4)

STOP_COMPLETE: Final[int] = 1
STOP_QUIESCENT: Final[int] = 2
STOP_EVENT_BUDGET: Final[int] = 3

KIND_NOOP: Final[int] = 0
KIND_SWAP: Final[int] = 1
KIND_MEMORY: Final[int] = 2

LEDGER_NAMES: Final[tuple[str, ...]] = (
    "activations",
    "observationReads",
    "valueComparisons",
    "noOps",
    "rejections",
    "memoryUpdates",
    "acceptedSwaps",
    "conflictLosses",
)


def stream_root(block_id: str) -> np.uint64:
    payload = f"E03/S12/context/v1\x00{block_id}".encode()
    return np.uint64(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


@nb.njit(cache=True)
def splitmix64(value: np.uint64) -> np.uint64:
    value = np.uint64(value + np.uint64(0x9E3779B97F4A7C15))
    value = np.uint64((value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9))
    value = np.uint64((value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB))
    return np.uint64(value ^ (value >> np.uint64(31)))


@nb.njit(cache=True)
def counter_draw(root: np.uint64, event: int, substream: int) -> np.uint64:
    address = np.uint64(event) * np.uint64(0xD2B74407B1CE6E93)
    address ^= np.uint64(substream) * np.uint64(0xCA5A826395121157)
    return splitmix64(np.uint64(root ^ address))


@nb.njit(cache=True)
def bounded(draw: np.uint64, bound: int) -> int:
    if bound <= 0:
        return np.int64(0)
    high = draw >> np.uint64(32)
    return np.int64((high * np.uint64(bound)) >> np.uint64(32))


@nb.njit(cache=True)
def systematic_request(
    event: int,
    denominator: int,
    swap_numerator: int,
    memory_numerator: int,
    changed_phase: int,
    swap_phase: int,
) -> int:
    """S11-compatible request code: 0 unchanged, 1 swap, 2 memory."""

    if denominator <= 0:
        return 0
    changed = swap_numerator + memory_numerator
    if changed <= 0:
        return 0
    before = (event * changed + changed_phase) // denominator
    after = ((event + 1) * changed + changed_phase) // denominator
    if after == before:
        return 0
    if swap_numerator <= 0:
        return 2
    if memory_numerator <= 0:
        return 1
    swap_before = (before * swap_numerator + swap_phase) // changed
    swap_after = ((before + 1) * swap_numerator + swap_phase) // changed
    return 1 if swap_after > swap_before else 2


@nb.njit(cache=True)
def _goal_rank(identity: int, n: int, direction: int) -> int:
    return identity if direction == 0 else n - 1 - identity


@nb.njit(cache=True)
def metric_profile(occupancy: np.ndarray, n: int, direction: int) -> np.ndarray:
    result = np.zeros(4, dtype=np.int32)
    maximum = 0
    for position in range(n):
        rank = _goal_rank(int(occupancy[position]), n, direction)
        error = abs(position - rank)
        result[2] += error
        if error > maximum:
            maximum = error
        if position + 1 < n:
            left = _goal_rank(int(occupancy[position]), n, direction)
            right = _goal_rank(int(occupancy[position + 1]), n, direction)
            if left > right:
                result[0] += 1
    inversions = 0
    for left in range(n - 1):
        left_rank = _goal_rank(int(occupancy[left]), n, direction)
        for right in range(left + 1, n):
            if left_rank > _goal_rank(int(occupancy[right]), n, direction):
                inversions += 1
    result[1] = inversions
    result[3] = maximum
    return result


@nb.njit(cache=True)
def swap_metric_delta(
    occupancy: np.ndarray,
    n: int,
    direction: int,
    left_pos: int,
    right_pos: int,
    current: np.ndarray,
) -> np.ndarray:
    """Metric delta for one proposed swap against an immutable snapshot."""

    result = np.zeros(4, dtype=np.int32)
    if left_pos == right_pos:
        return result
    a = min(left_pos, right_pos)
    b = max(left_pos, right_pos)
    x = int(occupancy[a])
    y = int(occupancy[b])
    rx = _goal_rank(x, n, direction)
    ry = _goal_rank(y, n, direction)

    before_adj = 0
    after_adj = 0
    for edge in range(max(0, a - 1), min(n - 1, b + 1)):
        old_left = int(occupancy[edge])
        old_right = int(occupancy[edge + 1])
        if _goal_rank(old_left, n, direction) > _goal_rank(old_right, n, direction):
            before_adj += 1
        new_left = y if edge == a else (x if edge == b else old_left)
        new_right = y if edge + 1 == a else (x if edge + 1 == b else old_right)
        if _goal_rank(new_left, n, direction) > _goal_rank(new_right, n, direction):
            after_adj += 1
    result[0] = after_adj - before_adj

    before_inv = 1 if rx > ry else 0
    after_inv = 1 if ry > rx else 0
    for pos in range(a + 1, b):
        rz = _goal_rank(int(occupancy[pos]), n, direction)
        before_inv += (1 if rx > rz else 0) + (1 if rz > ry else 0)
        after_inv += (1 if ry > rz else 0) + (1 if rz > rx else 0)
    result[1] = after_inv - before_inv

    before_foot = abs(a - rx) + abs(b - ry)
    after_foot = abs(b - rx) + abs(a - ry)
    result[2] = after_foot - before_foot

    maximum = 0
    for pos in range(n):
        identity = int(occupancy[pos])
        new_pos = b if pos == a else (a if pos == b else pos)
        error = abs(new_pos - _goal_rank(identity, n, direction))
        if error > maximum:
            maximum = error
    result[3] = maximum - int(current[3])
    return result


@nb.njit(cache=True)
def _is_complete(occupancy: np.ndarray, n: int, direction: int) -> bool:
    for pos in range(n - 1):
        if _goal_rank(int(occupancy[pos]), n, direction) > _goal_rank(
            int(occupancy[pos + 1]), n, direction
        ):
            return False
    return True


@nb.njit(cache=True)
def _positions(occupancy: np.ndarray, n: int) -> np.ndarray:
    positions = np.empty(n, dtype=np.int16)
    for pos in range(n):
        positions[int(occupancy[pos])] = pos
    return positions


@nb.njit(cache=True)
def _proposal(
    occupancy: np.ndarray,
    cursors: np.ndarray,
    policies: np.ndarray,
    faults: np.ndarray,
    n: int,
    direction: int,
    actor: int,
    side: int,
) -> tuple[int, int, int, int, int, int]:
    """Return kind, actor position, target, new cursor, reads, comparisons."""

    positions = _positions(occupancy, n)
    position = int(positions[actor])
    if faults[actor] != 0:
        return KIND_NOOP, position, -1, 0, 1, 0
    policy = int(policies[actor])
    if policy == 0:
        target = position - 1 if side == 0 else position + 1
        if target < 0 or target >= n:
            return KIND_NOOP, position, -1, 0, 1, 0
        other = int(occupancy[target])
        if direction == 0:
            inversion = actor < other if side == 0 else actor > other
        else:
            inversion = actor > other if side == 0 else actor < other
        if not inversion:
            return KIND_NOOP, position, -1, 0, 2, 1
        return KIND_SWAP, position, target, 0, 2, 1

    if policy == 1:
        if position == 0:
            return KIND_NOOP, position, -1, 0, 1, 0
        reads = 0
        comparisons = 0
        prior = -1
        prior_normal = False
        valid = True
        for pos in range(position):
            identity = int(occupancy[pos])
            reads += 1
            if faults[identity] != 0:
                prior = -1
                prior_normal = False
                continue
            if prior_normal:
                comparisons += 1
                ordered = prior <= identity if direction == 0 else prior >= identity
                if not ordered:
                    valid = False
                    break
            prior = identity
            prior_normal = True
        if not valid:
            return KIND_NOOP, position, -1, 0, reads + 1, comparisons
        target = position - 1
        other = int(occupancy[target])
        inversion = actor < other if direction == 0 else actor > other
        if not inversion:
            return KIND_NOOP, position, -1, 0, reads + 2, comparisons + 1
        return KIND_SWAP, position, target, 0, reads + 2, comparisons + 1

    cursor = int(cursors[actor])
    if cursor < 0 or cursor >= n:
        return KIND_NOOP, position, -1, 0, 1, 0
    if cursor == position:
        return KIND_NOOP, position, -1, 0, 1, 0
    target_identity = int(occupancy[cursor])
    delta = 1 if direction == 0 else -1
    if faults[target_identity] != 0:
        return KIND_MEMORY, position, -1, cursor + delta, 2, 0
    if target_identity <= actor:
        return KIND_MEMORY, position, -1, cursor + delta, 2, 1
    return KIND_SWAP, position, cursor, 0, 2, 1


@nb.njit(cache=True)
def _category_availability(
    occupancy: np.ndarray,
    cursors: np.ndarray,
    policies: np.ndarray,
    faults: np.ndarray,
    n: int,
    direction: int,
) -> tuple[bool, bool, bool]:
    swap = False
    memory = False
    unchanged = False
    positions = _positions(occupancy, n)
    prefix_valid = np.ones(n + 1, dtype=np.uint8)
    for stop in range(2, n + 1):
        left = int(occupancy[stop - 2])
        right = int(occupancy[stop - 1])
        pair_valid = (
            faults[left] != 0
            or faults[right] != 0
            or (left <= right if direction == 0 else left >= right)
        )
        prefix_valid[stop] = 1 if prefix_valid[stop - 1] and pair_valid else 0
    for actor in range(n):
        policy = int(policies[actor])
        if faults[actor] != 0:
            unchanged = True
            continue
        position = int(positions[actor])
        if policy == 0:
            for side in range(2):
                target = position - 1 if side == 0 else position + 1
                if target < 0 or target >= n:
                    unchanged = True
                    continue
                other = int(occupancy[target])
                if direction == 0:
                    inversion = actor < other if side == 0 else actor > other
                else:
                    inversion = actor > other if side == 0 else actor < other
                if inversion and faults[other] == 0:
                    swap = True
                else:
                    unchanged = True
        elif policy == 1:
            if position == 0 or prefix_valid[position] == 0:
                unchanged = True
                continue
            target_identity = int(occupancy[position - 1])
            inversion = actor < target_identity if direction == 0 else actor > target_identity
            if inversion and faults[target_identity] == 0:
                swap = True
            else:
                unchanged = True
        else:
            cursor = int(cursors[actor])
            if cursor < 0 or cursor >= n or cursor == position:
                unchanged = True
                continue
            target_identity = int(occupancy[cursor])
            if faults[target_identity] != 0 or target_identity <= actor:
                memory = True
            else:
                swap = True
    return swap, memory, unchanged


@nb.njit(cache=True)
def _is_quiescent(
    occupancy: np.ndarray,
    cursors: np.ndarray,
    policies: np.ndarray,
    faults: np.ndarray,
    n: int,
    direction: int,
) -> bool:
    swap, memory, _ = _category_availability(
        occupancy, cursors, policies, faults, n, direction
    )
    return not swap and not memory


@nb.njit(cache=True)
def _state_fingerprint(
    root: np.uint64,
    occupancy: np.ndarray,
    cursors: np.ndarray,
    n: int,
    events: int,
) -> np.uint64:
    value = splitmix64(root ^ np.uint64(events) ^ np.uint64(0x5354415445))
    for pos in range(n):
        value = splitmix64(
            value
            ^ np.uint64((pos + 1) * 1315423911)
            ^ np.uint64(int(occupancy[pos]) + 1)
        )
    for identity in range(n):
        cursor = int(cursors[identity]) + n + 2
        value = splitmix64(
            value
            ^ np.uint64((identity + 1) * 2654435761)
            ^ np.uint64(cursor)
        )
    return value


@nb.njit(cache=True)
def _terminal_code(
    occupancy: np.ndarray,
    cursors: np.ndarray,
    policies: np.ndarray,
    faults: np.ndarray,
    n: int,
    direction: int,
) -> int:
    if _is_complete(occupancy, n, direction):
        return STOP_COMPLETE
    if _is_quiescent(occupancy, cursors, policies, faults, n, direction):
        return STOP_QUIESCENT
    return 0


@nb.njit(cache=True)
def _simulate_one(
    n: int,
    policies: np.ndarray,
    faults: np.ndarray,
    initial_occupancy: np.ndarray,
    direction: int,
    batch_width: int,
    root: np.uint64,
    event_budget: int,
    target_swaps: int,
    target_memory: int,
    target_denominator: int,
    checkpoint_event: int,
) -> tuple:
    occupancy = initial_occupancy[:n].copy()
    cursors = np.full(n, -32768, dtype=np.int16)
    for identity in range(n):
        if policies[identity] == 2:
            cursors[identity] = 0 if direction == 0 else n - 1

    initial = metric_profile(occupancy, n, direction)
    levels = initial.copy()
    peaks = initial.copy()
    running_min = initial.copy()
    anchor = initial.copy()
    episode_start_event = np.full(4, -1, dtype=np.int32)
    episode_start_checkpoint = np.full(4, -1, dtype=np.int32)
    open_episode = np.zeros(4, dtype=np.uint8)
    episode_count = np.zeros(4, dtype=np.int32)
    recovered_count = np.zeros(4, dtype=np.int32)
    max_depth = np.zeros(4, dtype=np.int32)
    current_episode_depth = np.zeros(4, dtype=np.int32)
    sum_depth = np.zeros(4, dtype=np.int64)
    max_duration = np.zeros(4, dtype=np.int32)
    sum_duration = np.zeros(4, dtype=np.int64)
    open_duration = np.zeros(4, dtype=np.int32)
    max_initial_excursion = np.zeros(4, dtype=np.int32)
    worsening_actions = np.zeros(4, dtype=np.int32)
    positive_runs = np.zeros(4, dtype=np.int32)
    in_positive_run = np.zeros(4, dtype=np.uint8)
    threshold_counts = np.zeros((4, 4), dtype=np.int32)
    requested = np.zeros(3, dtype=np.int32)
    deficits = np.zeros(3, dtype=np.int32)
    ledger = np.zeros(8, dtype=np.int64)
    checkpoint_ledger = np.full(8, -1, dtype=np.int64)
    checkpoint_levels = np.full(4, -1, dtype=np.int32)
    checkpoint_fingerprint = np.uint64(0)
    checkpoint_stop = 0

    events = 0
    state_checkpoints = 0
    stop = _terminal_code(
        occupancy, cursors, policies, faults, n, direction
    )
    changed_phase = (
        bounded(counter_draw(root, 0, 41), target_denominator)
        if target_swaps >= 0 and target_denominator > 0
        else 0
    )
    swap_phase = (
        bounded(counter_draw(root, 0, 43), target_swaps + target_memory)
        if target_swaps + target_memory > 0
        else 0
    )

    while stop == 0 and events < event_budget:
        width = min(batch_width, event_budget - events)
        snapshot = occupancy.copy()
        snapshot_cursors = cursors.copy()
        kinds = np.zeros(4, dtype=np.int8)
        actors = np.zeros(4, dtype=np.int16)
        actor_positions = np.zeros(4, dtype=np.int16)
        targets = np.full(4, -1, dtype=np.int16)
        new_cursors = np.zeros(4, dtype=np.int16)
        reads = np.zeros(4, dtype=np.int16)
        comparisons = np.zeros(4, dtype=np.int16)
        priorities = np.zeros(4, dtype=np.uint64)
        eligible = np.zeros(4, dtype=np.uint8)
        accepted = np.zeros(4, dtype=np.uint8)
        conflict = np.zeros(4, dtype=np.uint8)

        for ordinal in range(width):
            event = events + ordinal
            if target_swaps >= 0:
                request = systematic_request(
                    event,
                    target_denominator,
                    target_swaps,
                    target_memory,
                    changed_phase,
                    swap_phase,
                )
                request_index = 2 if request == 0 else request - 1
                requested[request_index] += 1
                has_swap, has_memory, has_unchanged = _category_availability(
                    snapshot,
                    snapshot_cursors,
                    policies,
                    faults,
                    n,
                    direction,
                )
                feasible = (
                    has_swap if request == 1 else (has_memory if request == 2 else has_unchanged)
                )
                if not feasible:
                    deficits[request_index] += 1

            actor = np.int64(bounded(counter_draw(root, event, 1), n))
            side = np.int64(bounded(counter_draw(root, event, 2), 2))
            kind, actor_pos, target, new_cursor, read_count, comparison_count = _proposal(
                snapshot,
                snapshot_cursors,
                policies,
                faults,
                n,
                direction,
                actor,
                side,
            )
            kinds[ordinal] = kind
            actors[ordinal] = actor
            actor_positions[ordinal] = actor_pos
            targets[ordinal] = target
            new_cursors[ordinal] = new_cursor
            reads[ordinal] = read_count
            comparisons[ordinal] = comparison_count
            priorities[ordinal] = counter_draw(root, event, 3)
            if kind == KIND_MEMORY:
                eligible[ordinal] = 1
            elif kind == KIND_SWAP and faults[int(snapshot[target])] == 0:
                eligible[ordinal] = 1

        # Reference resolve_conflicts order: priority, actor identity, target,
        # ordinal.  Greedily reserve actor identities and swap positions.
        used = np.zeros(4, dtype=np.uint8)
        reserved_actor = np.zeros(n, dtype=np.uint8)
        reserved_position = np.zeros(n, dtype=np.uint8)
        for _ in range(width):
            chosen = -1
            for ordinal in range(width):
                if used[ordinal] != 0 or eligible[ordinal] == 0:
                    continue
                if chosen < 0:
                    chosen = ordinal
                    continue
                better = priorities[ordinal] < priorities[chosen]
                if priorities[ordinal] == priorities[chosen]:
                    if actors[ordinal] < actors[chosen]:
                        better = True
                    elif actors[ordinal] == actors[chosen]:
                        target_ord = int(targets[ordinal])
                        target_chosen = int(targets[chosen])
                        if target_ord < target_chosen or (
                            target_ord == target_chosen and ordinal < chosen
                        ):
                            better = True
                if better:
                    chosen = ordinal
            if chosen < 0:
                break
            used[chosen] = 1
            actor = int(actors[chosen])
            collision = reserved_actor[actor] != 0
            if kinds[chosen] == KIND_SWAP:
                a = int(actor_positions[chosen])
                b = int(targets[chosen])
                collision = collision or reserved_position[a] != 0 or reserved_position[b] != 0
            if collision:
                conflict[chosen] = 1
            else:
                accepted[chosen] = 1
                reserved_actor[actor] = 1
                if kinds[chosen] == KIND_SWAP:
                    reserved_position[int(actor_positions[chosen])] = 1
                    reserved_position[int(targets[chosen])] = 1

        any_change = False
        any_swap = False
        for ordinal in range(width):
            ledger[0] += 1
            ledger[1] += int(reads[ordinal])
            ledger[2] += int(comparisons[ordinal])
            kind = int(kinds[ordinal])
            if kind == KIND_NOOP:
                ledger[3] += 1
            elif eligible[ordinal] == 0:
                ledger[4] += 1
            elif conflict[ordinal] != 0:
                ledger[7] += 1
            elif accepted[ordinal] != 0 and kind == KIND_MEMORY:
                ledger[5] += 1
                cursors[int(actors[ordinal])] = int(new_cursors[ordinal])
                any_change = True
            elif accepted[ordinal] != 0 and kind == KIND_SWAP:
                ledger[6] += 1
                a = int(actor_positions[ordinal])
                b = int(targets[ordinal])
                deltas = swap_metric_delta(snapshot, n, direction, a, b, levels)
                for metric in range(4):
                    if deltas[metric] > 0:
                        worsening_actions[metric] += 1
                    for threshold_index in range(4):
                        if deltas[metric] > THRESHOLDS[threshold_index]:
                            threshold_counts[metric, threshold_index] += 1
                occupancy[a] = snapshot[b]
                occupancy[b] = snapshot[a]
                any_change = True
                any_swap = True

        events += width
        if any_swap:
            new_levels = metric_profile(occupancy, n, direction)
            state_checkpoints += 1
            for metric in range(4):
                delta = int(new_levels[metric] - levels[metric])
                if delta > 0:
                    if in_positive_run[metric] == 0:
                        positive_runs[metric] += 1
                        in_positive_run[metric] = 1
                else:
                    in_positive_run[metric] = 0

                if new_levels[metric] > peaks[metric]:
                    peaks[metric] = new_levels[metric]
                initial_excursion = int(new_levels[metric] - initial[metric])
                if initial_excursion > max_initial_excursion[metric]:
                    max_initial_excursion[metric] = initial_excursion

                if open_episode[metric] == 0:
                    if new_levels[metric] > running_min[metric]:
                        open_episode[metric] = 1
                        anchor[metric] = running_min[metric]
                        episode_start_event[metric] = events - width
                        episode_start_checkpoint[metric] = state_checkpoints
                        episode_count[metric] += 1
                        depth = int(new_levels[metric] - anchor[metric])
                        current_episode_depth[metric] = depth
                        if depth > max_depth[metric]:
                            max_depth[metric] = depth
                    elif new_levels[metric] < running_min[metric]:
                        running_min[metric] = new_levels[metric]
                else:
                    depth = int(new_levels[metric] - anchor[metric])
                    if depth > current_episode_depth[metric]:
                        current_episode_depth[metric] = depth
                    if depth > max_depth[metric]:
                        max_depth[metric] = depth
                    if new_levels[metric] <= anchor[metric]:
                        recovered_count[metric] += 1
                        duration = events - int(episode_start_event[metric])
                        if duration > max_duration[metric]:
                            max_duration[metric] = duration
                        sum_duration[metric] += duration
                        sum_depth[metric] += max(0, current_episode_depth[metric])
                        open_episode[metric] = 0
                        current_episode_depth[metric] = 0
                        running_min[metric] = min(running_min[metric], new_levels[metric])
                levels[metric] = new_levels[metric]
        if any_change:
            stop = _terminal_code(occupancy, cursors, policies, faults, n, direction)

        if checkpoint_event > 0 and events == checkpoint_event:
            checkpoint_stop = stop
            checkpoint_fingerprint = _state_fingerprint(
                root, occupancy, cursors, n, events
            )
            checkpoint_ledger[:] = ledger
            checkpoint_levels[:] = levels

    if stop == 0:
        stop = STOP_EVENT_BUDGET
    for metric in range(4):
        if open_episode[metric] != 0:
            duration = events - int(episode_start_event[metric])
            open_duration[metric] = duration
            if duration > max_duration[metric]:
                max_duration[metric] = duration
    fingerprint = _state_fingerprint(root, occupancy, cursors, n, events)
    return (
        stop,
        events,
        ledger,
        initial,
        levels,
        peaks,
        episode_count,
        recovered_count,
        open_episode,
        max_depth,
        sum_depth,
        max_duration,
        sum_duration,
        open_duration,
        max_initial_excursion,
        worsening_actions,
        positive_runs,
        threshold_counts,
        requested,
        deficits,
        fingerprint,
        occupancy,
        cursors,
        checkpoint_stop,
        checkpoint_fingerprint,
        checkpoint_ledger,
        checkpoint_levels,
    )


@nb.njit(cache=False)
def simulate_population(
    ns: np.ndarray,
    policies: np.ndarray,
    faults: np.ndarray,
    initial_occupancies: np.ndarray,
    directions: np.ndarray,
    batch_widths: np.ndarray,
    roots: np.ndarray,
    event_budgets: np.ndarray,
    target_swaps: np.ndarray,
    target_memory: np.ndarray,
    target_denominators: np.ndarray,
    checkpoint_events: np.ndarray,
) -> tuple:
    run_count = len(ns)
    max_n = policies.shape[1]
    stops = np.zeros(run_count, dtype=np.uint8)
    events = np.zeros(run_count, dtype=np.int32)
    ledgers = np.zeros((run_count, 8), dtype=np.int64)
    initial = np.zeros((run_count, 4), dtype=np.int32)
    final = np.zeros((run_count, 4), dtype=np.int32)
    peak = np.zeros((run_count, 4), dtype=np.int32)
    episode_count = np.zeros((run_count, 4), dtype=np.int32)
    recovered_count = np.zeros((run_count, 4), dtype=np.int32)
    open_episode = np.zeros((run_count, 4), dtype=np.uint8)
    max_depth = np.zeros((run_count, 4), dtype=np.int32)
    sum_depth = np.zeros((run_count, 4), dtype=np.int64)
    max_duration = np.zeros((run_count, 4), dtype=np.int32)
    sum_duration = np.zeros((run_count, 4), dtype=np.int64)
    open_duration = np.zeros((run_count, 4), dtype=np.int32)
    max_initial_excursion = np.zeros((run_count, 4), dtype=np.int32)
    worsening_actions = np.zeros((run_count, 4), dtype=np.int32)
    positive_runs = np.zeros((run_count, 4), dtype=np.int32)
    threshold_counts = np.zeros((run_count, 4, 4), dtype=np.int32)
    requested = np.zeros((run_count, 3), dtype=np.int32)
    deficits = np.zeros((run_count, 3), dtype=np.int32)
    fingerprints = np.zeros(run_count, dtype=np.uint64)
    final_occupancies = np.full((run_count, max_n), -1, dtype=np.int16)
    final_cursors = np.full((run_count, max_n), -32768, dtype=np.int16)
    checkpoint_stops = np.zeros(run_count, dtype=np.uint8)
    checkpoint_fingerprints = np.zeros(run_count, dtype=np.uint64)
    checkpoint_ledgers = np.full((run_count, 8), -1, dtype=np.int64)
    checkpoint_levels = np.full((run_count, 4), -1, dtype=np.int32)

    for run in range(run_count):
        result = _simulate_one(
            int(ns[run]),
            policies[run],
            faults[run],
            initial_occupancies[run],
            int(directions[run]),
            int(batch_widths[run]),
            roots[run],
            int(event_budgets[run]),
            int(target_swaps[run]),
            int(target_memory[run]),
            int(target_denominators[run]),
            int(checkpoint_events[run]),
        )
        stops[run] = result[0]
        events[run] = result[1]
        ledgers[run] = result[2]
        initial[run] = result[3]
        final[run] = result[4]
        peak[run] = result[5]
        episode_count[run] = result[6]
        recovered_count[run] = result[7]
        open_episode[run] = result[8]
        max_depth[run] = result[9]
        sum_depth[run] = result[10]
        max_duration[run] = result[11]
        sum_duration[run] = result[12]
        open_duration[run] = result[13]
        max_initial_excursion[run] = result[14]
        worsening_actions[run] = result[15]
        positive_runs[run] = result[16]
        threshold_counts[run] = result[17]
        requested[run] = result[18]
        deficits[run] = result[19]
        fingerprints[run] = result[20]
        n = int(ns[run])
        final_occupancies[run, :n] = result[21]
        final_cursors[run, :n] = result[22]
        checkpoint_stops[run] = result[23]
        checkpoint_fingerprints[run] = result[24]
        checkpoint_ledgers[run] = result[25]
        checkpoint_levels[run] = result[26]

    return (
        stops,
        events,
        ledgers,
        initial,
        final,
        peak,
        episode_count,
        recovered_count,
        open_episode,
        max_depth,
        sum_depth,
        max_duration,
        sum_duration,
        open_duration,
        max_initial_excursion,
        worsening_actions,
        positive_runs,
        threshold_counts,
        requested,
        deficits,
        fingerprints,
        final_occupancies,
        final_cursors,
        checkpoint_stops,
        checkpoint_fingerprints,
        checkpoint_ledgers,
        checkpoint_levels,
    )


def result_dict(result: tuple) -> dict[str, np.ndarray]:
    names = (
        "stop",
        "events",
        "ledger",
        "initial",
        "final",
        "peak",
        "episode_count",
        "recovered_count",
        "open_episode",
        "max_depth",
        "sum_depth",
        "max_duration",
        "sum_duration",
        "open_duration",
        "max_initial_excursion",
        "worsening_actions",
        "positive_runs",
        "threshold_counts",
        "requested",
        "deficits",
        "fingerprint",
        "final_occupancy",
        "final_cursors",
        "checkpoint_stop",
        "checkpoint_fingerprint",
        "checkpoint_ledger",
        "checkpoint_levels",
    )
    return dict(zip(names, result, strict=True))
