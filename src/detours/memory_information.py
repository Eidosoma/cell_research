"""Frozen local capability augmentations for E03 S13.

The module extends the deterministic S12 trajectory kernel with four explicitly
bounded additions: a one-bit failure flag, a saturating two-bit stagnation
counter, three-state recent-displacement memory, and a radius-two local sensing
overlay.  Mechanical validation, conflicts, commit, metrics, ledger fields, and
terminal precedence remain projections of the frozen E01 reference semantics.

The capability gateway never receives metric levels, exact reachability,
terminal labels, analysis labels, scenario outcomes, or future stream draws.
Only unique integer values and homogeneous directions are in scope.
"""

from __future__ import annotations

import hashlib
import math
from typing import Final

import numba as nb
import numpy as np

from src.detours.context_sensitivity import (
    KIND_MEMORY,
    KIND_NOOP,
    KIND_SWAP,
    LEDGER_NAMES,
    METRIC_NAMES,
    STOP_COMPLETE,
    STOP_EVENT_BUDGET,
    STOP_QUIESCENT,
    THRESHOLDS,
    _is_complete,
    _is_quiescent,
    _positions,
    _proposal,
    _state_fingerprint,
    bounded,
    counter_draw,
    metric_profile,
    splitmix64,
    swap_metric_delta,
)


ARM_NAMES: Final[tuple[str, ...]] = (
    "native",
    "failure_bit",
    "counter_2bit",
    "recent_direction",
    "radius2",
    "full",
    "full_minus_failure",
    "full_minus_counter",
    "full_minus_recent",
    "full_minus_radius2",
)

# failure bit, counter, recent direction, radius-two overlay
ARM_FLAGS: Final[tuple[tuple[int, int, int, int], ...]] = (
    (0, 0, 0, 0),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
    (1, 1, 1, 1),
    (0, 1, 1, 1),
    (1, 0, 1, 1),
    (1, 1, 0, 1),
    (1, 1, 1, 0),
)

CAPABILITY_AUDIT_NAMES: Final[tuple[str, ...]] = (
    "failure_side_overrides",
    "counter_side_overrides",
    "radius2_proposals",
    "recent_direction_vetoes",
    "failure_bit_sets",
    "counter_saturation_entries",
    "recent_direction_updates",
    "capability_memory_writes",
    "capability_bit_writes",
    "behavioral_action_overrides",
)

RECENT_NONE: Final[int] = 0
RECENT_LEFT: Final[int] = 1
RECENT_RIGHT: Final[int] = 2


def arm_flags(arm: str) -> tuple[int, int, int, int]:
    try:
        return ARM_FLAGS[ARM_NAMES.index(arm)]
    except ValueError as error:
        raise ValueError(f"unknown S13 capability arm {arm!r}") from error


def stream_root(key: str) -> np.uint64:
    payload = f"E03/S13/memory-information/v1\x00{key}".encode()
    return np.uint64(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


def complexity_rows() -> list[dict[str, object]]:
    """Return the prospective finite-state and information ledger by arm."""

    rows: list[dict[str, object]] = []
    for arm, flags in zip(ARM_NAMES, ARM_FLAGS, strict=True):
        failure, counter, recent, radius = (bool(value) for value in flags)
        states = (2 if failure else 1) * (4 if counter else 1) * (3 if recent else 1)
        minimum_bits = math.ceil(math.log2(states)) if states > 1 else 0
        stored_bits = (1 if failure else 0) + (2 if counter else 0) + (2 if recent else 0)
        rows.append(
            {
                "arm": arm,
                "failure_bit": failure,
                "counter_2bit": counter,
                "recent_direction": recent,
                "radius2": radius,
                "mutable_state_count_per_actor": states,
                "minimum_bits_per_actor": minimum_bits,
                "declared_stored_bits_per_actor": stored_bits,
                "added_sensing_radius": 2 if radius else 0,
                "maximum_added_neighbor_reads_per_activation": 4 if radius else 0,
                "maximum_added_local_comparisons_per_activation": 16 if radius else 0,
                "native_observation_scope": (
                    "Bubble one selected adjacent; Insertion strict prefix; Selection cursor occupant"
                ),
                "action_scope": (
                    "native actions plus adjacent radius-two overlay swap"
                    if radius
                    else "native action targets only; memory may change side or veto"
                ),
                "forbidden_information": (
                    "metrics, global ranks, exact reachability, terminal evaluator, analysis labels, "
                    "scenario outcomes, future draws, and positions outside the radius-two overlay"
                ),
            }
        )
    return rows


@nb.njit(cache=True)
def _local_inversions_after_swap(
    occupancy: np.ndarray,
    direction: int,
    start: int,
    stop: int,
    actor_pos: int,
    target_pos: int,
) -> tuple[int, int, int]:
    """Return before, after, comparisons inside one radius-two window."""

    before = 0
    after = 0
    comparisons = 0
    actor_identity = int(occupancy[actor_pos])
    target_identity = int(occupancy[target_pos])
    for edge in range(start, stop):
        left = int(occupancy[edge])
        right = int(occupancy[edge + 1])
        before += 1 if (left > right if direction == 0 else left < right) else 0
        new_left = (
            target_identity
            if edge == actor_pos
            else actor_identity if edge == target_pos else left
        )
        new_right = (
            target_identity
            if edge + 1 == actor_pos
            else actor_identity if edge + 1 == target_pos else right
        )
        after += 1 if (new_left > new_right if direction == 0 else new_left < new_right) else 0
        comparisons += 2
    return before, after, comparisons


@nb.njit(cache=True)
def capability_proposal(
    occupancy: np.ndarray,
    cursors: np.ndarray,
    policies: np.ndarray,
    faults: np.ndarray,
    failure_memory: np.ndarray,
    stagnation_counter: np.ndarray,
    recent_direction: np.ndarray,
    n: int,
    direction: int,
    actor: int,
    raw_side: int,
    failure_enabled: int,
    counter_enabled: int,
    recent_enabled: int,
    radius2_enabled: int,
) -> tuple[int, int, int, int, int, int, int, int, int, int, int]:
    """Construct one capability proposal without access to global outcomes.

    Returns kind, actor position, target, new cursor, reads, comparisons,
    failure-side override, counter-side override, radius proposal, recent veto,
    and action override relative to the native raw-side proposal.
    """

    native = _proposal(
        occupancy, cursors, policies, faults, n, direction, actor, raw_side
    )
    kind, actor_pos, target, new_cursor, reads, comparisons = native
    if faults[actor] != 0:
        return kind, actor_pos, target, new_cursor, reads, comparisons, 0, 0, 0, 0, 0

    side = raw_side
    failure_override = 0
    counter_override = 0
    if int(policies[actor]) == 0:
        if failure_enabled != 0 and failure_memory[actor] != 0:
            side = 1 - raw_side
            failure_override = 1
        elif counter_enabled != 0 and stagnation_counter[actor] >= 3:
            side = 1 - raw_side
            counter_override = 1

    if side != raw_side:
        kind, actor_pos, target, new_cursor, reads, comparisons = _proposal(
            occupancy, cursors, policies, faults, n, direction, actor, side
        )

    radius_selected = 0
    if radius2_enabled != 0:
        positions = _positions(occupancy, n)
        actor_pos = int(positions[actor])
        start = max(0, actor_pos - 2)
        stop = min(n - 1, actor_pos + 2)
        overlay_reads = stop - start + 1
        best_target = -1
        best_gain = 0
        overlay_comparisons = 0
        for candidate_side in range(2):
            candidate = actor_pos - 1 if candidate_side == 0 else actor_pos + 1
            if candidate < 0 or candidate >= n:
                continue
            target_identity = int(occupancy[candidate])
            if faults[target_identity] != 0:
                continue
            before, after, used = _local_inversions_after_swap(
                occupancy, direction, start, stop, actor_pos, candidate
            )
            overlay_comparisons += used
            gain = before - after
            if gain > best_gain:
                best_gain = gain
                best_target = candidate
            elif gain == best_gain and gain > 0:
                current_side = 0 if candidate < actor_pos else 1
                best_side = 0 if best_target < actor_pos else 1
                if current_side == side and best_side != side:
                    best_target = candidate
                elif current_side == best_side and candidate < best_target:
                    best_target = candidate
        reads += overlay_reads
        comparisons += overlay_comparisons
        if best_target >= 0:
            kind = KIND_SWAP
            target = best_target
            new_cursor = 0
            radius_selected = 1

    recent_veto = 0
    if recent_enabled != 0 and kind == KIND_SWAP and target >= 0:
        proposed = RECENT_LEFT if target < actor_pos else RECENT_RIGHT
        remembered = int(recent_direction[actor])
        if (
            (remembered == RECENT_LEFT and proposed == RECENT_RIGHT)
            or (remembered == RECENT_RIGHT and proposed == RECENT_LEFT)
        ):
            kind = KIND_NOOP
            target = -1
            new_cursor = 0
            recent_veto = 1

    action_override = 1 if (
        kind != int(native[0])
        or target != int(native[2])
        or new_cursor != int(native[3])
    ) else 0
    return (
        kind,
        actor_pos,
        target,
        new_cursor,
        reads,
        comparisons,
        failure_override,
        counter_override,
        radius_selected,
        recent_veto,
        action_override,
    )


@nb.njit(cache=True)
def _capability_quiescent(
    occupancy: np.ndarray,
    cursors: np.ndarray,
    policies: np.ndarray,
    faults: np.ndarray,
    failure_memory: np.ndarray,
    stagnation_counter: np.ndarray,
    recent_direction: np.ndarray,
    n: int,
    direction: int,
    failure_enabled: int,
    counter_enabled: int,
    recent_enabled: int,
    radius2_enabled: int,
) -> bool:
    if failure_enabled == 0 and counter_enabled == 0 and recent_enabled == 0 and radius2_enabled == 0:
        return _is_quiescent(occupancy, cursors, policies, faults, n, direction)
    for actor in range(n):
        side_count = 2 if int(policies[actor]) == 0 or radius2_enabled != 0 else 1
        for raw_side in range(side_count):
            proposal = capability_proposal(
                occupancy,
                cursors,
                policies,
                faults,
                failure_memory,
                stagnation_counter,
                recent_direction,
                n,
                direction,
                actor,
                raw_side,
                failure_enabled,
                counter_enabled,
                recent_enabled,
                radius2_enabled,
            )
            kind = int(proposal[0])
            target = int(proposal[2])
            if kind == KIND_MEMORY:
                return False
            if kind == KIND_SWAP and target >= 0 and faults[int(occupancy[target])] == 0:
                return False
            target_stuck = (
                kind == KIND_SWAP
                and target >= 0
                and faults[int(occupancy[target])] != 0
            )
            if failure_enabled != 0:
                next_failure = 1 if target_stuck else 0
                if next_failure != int(failure_memory[actor]):
                    return False
            if counter_enabled != 0 and stagnation_counter[actor] < 3:
                return False
    return True


@nb.njit(cache=True)
def _terminal_code(
    occupancy: np.ndarray,
    cursors: np.ndarray,
    policies: np.ndarray,
    faults: np.ndarray,
    failure_memory: np.ndarray,
    stagnation_counter: np.ndarray,
    recent_direction: np.ndarray,
    n: int,
    direction: int,
    failure_enabled: int,
    counter_enabled: int,
    recent_enabled: int,
    radius2_enabled: int,
) -> int:
    if _is_complete(occupancy, n, direction):
        return STOP_COMPLETE
    if _capability_quiescent(
        occupancy,
        cursors,
        policies,
        faults,
        failure_memory,
        stagnation_counter,
        recent_direction,
        n,
        direction,
        failure_enabled,
        counter_enabled,
        recent_enabled,
        radius2_enabled,
    ):
        return STOP_QUIESCENT
    return 0


@nb.njit(cache=True)
def _full_state_fingerprint(
    root: np.uint64,
    occupancy: np.ndarray,
    cursors: np.ndarray,
    failure_memory: np.ndarray,
    stagnation_counter: np.ndarray,
    recent_direction: np.ndarray,
    n: int,
    events: int,
    arm_code: int,
) -> np.uint64:
    value = _state_fingerprint(root, occupancy, cursors, n, events)
    value = splitmix64(value ^ np.uint64(arm_code + 1) ^ np.uint64(0x53413343))
    for identity in range(n):
        packed = (
            int(failure_memory[identity])
            + 4 * int(stagnation_counter[identity])
            + 32 * int(recent_direction[identity])
        )
        value = splitmix64(
            value
            ^ np.uint64((identity + 1) * 2246822519)
            ^ np.uint64(packed + 1)
        )
    return value


@nb.njit(cache=True)
def _simulate_one(
    n: int,
    policies: np.ndarray,
    faults: np.ndarray,
    initial_occupancy: np.ndarray,
    initial_cursors: np.ndarray,
    direction: int,
    batch_width: int,
    root: np.uint64,
    event_budget: int,
    arm_code: int,
    checkpoint_event: int,
) -> tuple:
    occupancy = initial_occupancy[:n].copy()
    cursors = initial_cursors[:n].copy()
    for identity in range(n):
        if policies[identity] == 2 and cursors[identity] == -32768:
            cursors[identity] = 0 if direction == 0 else n - 1
    failure_memory = np.zeros(n, dtype=np.uint8)
    stagnation_counter = np.zeros(n, dtype=np.uint8)
    recent_direction = np.zeros(n, dtype=np.uint8)
    flags = ARM_FLAGS[arm_code]
    failure_enabled = int(flags[0])
    counter_enabled = int(flags[1])
    recent_enabled = int(flags[2])
    radius2_enabled = int(flags[3])

    initial = metric_profile(occupancy, n, direction)
    levels = initial.copy()
    peaks = initial.copy()
    running_min = initial.copy()
    anchor = initial.copy()
    episode_start_event = np.full(4, -1, dtype=np.int32)
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
    ledger = np.zeros(8, dtype=np.int64)
    capability_audit = np.zeros(10, dtype=np.int64)
    maximum_radius_read = 0
    first_action_override_event = -1
    first_override_pre_physical_fingerprint = np.uint64(0)
    trajectory_digest = splitmix64(root ^ np.uint64(0x5331335452414345))
    checkpoint_ledger = np.full(8, -1, dtype=np.int64)
    checkpoint_capability_audit = np.full(10, -1, dtype=np.int64)
    checkpoint_levels = np.full(4, -1, dtype=np.int32)
    checkpoint_physical_fingerprint = np.uint64(0)
    checkpoint_full_fingerprint = np.uint64(0)
    checkpoint_stop = 0

    events = 0
    state_checkpoints = 0
    stop = _terminal_code(
        occupancy,
        cursors,
        policies,
        faults,
        failure_memory,
        stagnation_counter,
        recent_direction,
        n,
        direction,
        failure_enabled,
        counter_enabled,
        recent_enabled,
        radius2_enabled,
    )

    while stop == 0 and events < event_budget:
        width = min(batch_width, event_budget - events)
        snapshot = occupancy.copy()
        snapshot_cursors = cursors.copy()
        snapshot_failure = failure_memory.copy()
        snapshot_counter = stagnation_counter.copy()
        snapshot_recent = recent_direction.copy()
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
        proposal_flags = np.zeros((4, 5), dtype=np.uint8)

        for ordinal in range(width):
            event = events + ordinal
            actor = int(bounded(counter_draw(root, event, 1), n))
            raw_side = int(bounded(counter_draw(root, event, 2), 2))
            proposal = capability_proposal(
                snapshot,
                snapshot_cursors,
                policies,
                faults,
                snapshot_failure,
                snapshot_counter,
                snapshot_recent,
                n,
                direction,
                actor,
                raw_side,
                failure_enabled,
                counter_enabled,
                recent_enabled,
                radius2_enabled,
            )
            (
                kind,
                actor_pos,
                target,
                new_cursor,
                read_count,
                comparison_count,
                failure_side_override,
                counter_side_override,
                radius_selected,
                recent_veto,
                action_override,
            ) = proposal
            kinds[ordinal] = kind
            actors[ordinal] = actor
            actor_positions[ordinal] = actor_pos
            targets[ordinal] = target
            new_cursors[ordinal] = new_cursor
            reads[ordinal] = read_count
            comparisons[ordinal] = comparison_count
            priorities[ordinal] = counter_draw(root, event, 3)
            proposal_flags[ordinal, 0] = failure_side_override
            proposal_flags[ordinal, 1] = counter_side_override
            proposal_flags[ordinal, 2] = radius_selected
            proposal_flags[ordinal, 3] = recent_veto
            proposal_flags[ordinal, 4] = action_override
            if action_override != 0 and first_action_override_event < 0:
                first_action_override_event = event
                first_override_pre_physical_fingerprint = _state_fingerprint(
                    root, snapshot, snapshot_cursors, n, events
                )
            if radius2_enabled != 0 and faults[actor] == 0:
                actor_position = int(actor_pos)
                observed_radius = max(
                    actor_position - max(0, actor_position - 2),
                    min(n - 1, actor_position + 2) - actor_position,
                )
                if observed_radius > maximum_radius_read:
                    maximum_radius_read = observed_radius
            if kind == KIND_MEMORY:
                eligible[ordinal] = 1
            elif kind == KIND_SWAP and target >= 0 and faults[int(snapshot[target])] == 0:
                eligible[ordinal] = 1

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
                collision = (
                    collision
                    or reserved_position[a] != 0
                    or reserved_position[b] != 0
                )
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
            for flag in range(4):
                capability_audit[flag] += int(proposal_flags[ordinal, flag])
            capability_audit[9] += int(proposal_flags[ordinal, 4])
            kind = int(kinds[ordinal])
            actor = int(actors[ordinal])
            target = int(targets[ordinal])
            decision_accepted = accepted[ordinal] != 0 and conflict[ordinal] == 0
            if kind == KIND_NOOP:
                ledger[3] += 1
            elif eligible[ordinal] == 0:
                ledger[4] += 1
            elif conflict[ordinal] != 0:
                ledger[7] += 1
            elif decision_accepted and kind == KIND_MEMORY:
                ledger[5] += 1
                cursors[actor] = int(new_cursors[ordinal])
                any_change = True
            elif decision_accepted and kind == KIND_SWAP:
                ledger[6] += 1
                a = int(actor_positions[ordinal])
                b = target
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

            target_stuck = (
                kind == KIND_SWAP
                and target >= 0
                and faults[int(snapshot[target])] != 0
            )
            if failure_enabled != 0:
                new_failure = 1 if target_stuck else 0
                if new_failure != int(failure_memory[actor]):
                    failure_memory[actor] = new_failure
                    capability_audit[7] += 1
                    capability_audit[8] += 1
                    any_change = True
                if target_stuck:
                    capability_audit[4] += 1

            accepted_change = decision_accepted and kind in (KIND_SWAP, KIND_MEMORY)
            if counter_enabled != 0:
                old_counter = int(stagnation_counter[actor])
                new_counter = 0 if accepted_change else min(3, old_counter + 1)
                if new_counter != old_counter:
                    stagnation_counter[actor] = new_counter
                    capability_audit[7] += 1
                    capability_audit[8] += 2
                    any_change = True
                    if new_counter == 3:
                        capability_audit[5] += 1

            if recent_enabled != 0 and decision_accepted and kind == KIND_SWAP:
                a = int(actor_positions[ordinal])
                b = target
                target_identity = int(snapshot[b])
                actor_direction = RECENT_LEFT if b < a else RECENT_RIGHT
                target_direction = RECENT_RIGHT if b < a else RECENT_LEFT
                if recent_direction[actor] != actor_direction:
                    recent_direction[actor] = actor_direction
                    capability_audit[6] += 1
                    capability_audit[7] += 1
                    capability_audit[8] += 2
                    any_change = True
                if recent_direction[target_identity] != target_direction:
                    recent_direction[target_identity] = target_direction
                    capability_audit[6] += 1
                    capability_audit[7] += 1
                    capability_audit[8] += 2
                    any_change = True

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
            trajectory_digest = splitmix64(
                trajectory_digest ^ np.uint64(events) ^ np.uint64(state_checkpoints)
            )
            for metric in range(4):
                trajectory_digest = splitmix64(
                    trajectory_digest
                    ^ np.uint64((metric + 1) * 2654435761)
                    ^ np.uint64(int(new_levels[metric]) + 1)
                )
            for position in range(n):
                trajectory_digest = splitmix64(
                    trajectory_digest
                    ^ np.uint64((position + 1) * 2246822519)
                    ^ np.uint64(int(occupancy[position]) + 1)
                )

        if any_change:
            stop = _terminal_code(
                occupancy,
                cursors,
                policies,
                faults,
                failure_memory,
                stagnation_counter,
                recent_direction,
                n,
                direction,
                failure_enabled,
                counter_enabled,
                recent_enabled,
                radius2_enabled,
            )

        if checkpoint_event > 0 and events == checkpoint_event:
            checkpoint_stop = stop
            checkpoint_physical_fingerprint = _state_fingerprint(
                root, occupancy, cursors, n, events
            )
            checkpoint_full_fingerprint = _full_state_fingerprint(
                root,
                occupancy,
                cursors,
                failure_memory,
                stagnation_counter,
                recent_direction,
                n,
                events,
                arm_code,
            )
            checkpoint_ledger[:] = ledger
            checkpoint_capability_audit[:] = capability_audit
            checkpoint_levels[:] = levels

    if stop == 0:
        stop = STOP_EVENT_BUDGET
    for metric in range(4):
        if open_episode[metric] != 0:
            duration = events - int(episode_start_event[metric])
            open_duration[metric] = duration
            if duration > max_duration[metric]:
                max_duration[metric] = duration
    physical_fingerprint = _state_fingerprint(root, occupancy, cursors, n, events)
    full_fingerprint = _full_state_fingerprint(
        root,
        occupancy,
        cursors,
        failure_memory,
        stagnation_counter,
        recent_direction,
        n,
        events,
        arm_code,
    )
    return (
        stop,
        events,
        ledger,
        capability_audit,
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
        state_checkpoints,
        trajectory_digest,
        physical_fingerprint,
        full_fingerprint,
        occupancy,
        cursors,
        failure_memory,
        stagnation_counter,
        recent_direction,
        maximum_radius_read,
        first_action_override_event,
        first_override_pre_physical_fingerprint,
        checkpoint_stop,
        checkpoint_physical_fingerprint,
        checkpoint_full_fingerprint,
        checkpoint_ledger,
        checkpoint_capability_audit,
        checkpoint_levels,
    )


@nb.njit(cache=False)
def simulate_population(
    ns: np.ndarray,
    policies: np.ndarray,
    faults: np.ndarray,
    initial_occupancies: np.ndarray,
    initial_cursors: np.ndarray,
    directions: np.ndarray,
    batch_widths: np.ndarray,
    roots: np.ndarray,
    event_budgets: np.ndarray,
    arm_codes: np.ndarray,
    checkpoint_events: np.ndarray,
) -> tuple:
    run_count = len(ns)
    max_n = policies.shape[1]
    stops = np.zeros(run_count, dtype=np.uint8)
    events = np.zeros(run_count, dtype=np.int32)
    ledgers = np.zeros((run_count, 8), dtype=np.int64)
    capability_audits = np.zeros((run_count, 10), dtype=np.int64)
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
    state_checkpoints = np.zeros(run_count, dtype=np.int32)
    trajectory_digests = np.zeros(run_count, dtype=np.uint64)
    physical_fingerprints = np.zeros(run_count, dtype=np.uint64)
    full_fingerprints = np.zeros(run_count, dtype=np.uint64)
    final_occupancies = np.full((run_count, max_n), -1, dtype=np.int16)
    final_cursors = np.full((run_count, max_n), -32768, dtype=np.int16)
    final_failure = np.zeros((run_count, max_n), dtype=np.uint8)
    final_counter = np.zeros((run_count, max_n), dtype=np.uint8)
    final_recent = np.zeros((run_count, max_n), dtype=np.uint8)
    maximum_radius_reads = np.zeros(run_count, dtype=np.uint8)
    first_action_override_events = np.full(run_count, -1, dtype=np.int32)
    first_override_pre_physical_fingerprints = np.zeros(run_count, dtype=np.uint64)
    checkpoint_stops = np.zeros(run_count, dtype=np.uint8)
    checkpoint_physical_fingerprints = np.zeros(run_count, dtype=np.uint64)
    checkpoint_full_fingerprints = np.zeros(run_count, dtype=np.uint64)
    checkpoint_ledgers = np.full((run_count, 8), -1, dtype=np.int64)
    checkpoint_capability_audits = np.full((run_count, 10), -1, dtype=np.int64)
    checkpoint_levels = np.full((run_count, 4), -1, dtype=np.int32)

    for run in range(run_count):
        result = _simulate_one(
            int(ns[run]),
            policies[run],
            faults[run],
            initial_occupancies[run],
            initial_cursors[run],
            int(directions[run]),
            int(batch_widths[run]),
            roots[run],
            int(event_budgets[run]),
            int(arm_codes[run]),
            int(checkpoint_events[run]),
        )
        stops[run] = result[0]
        events[run] = result[1]
        ledgers[run] = result[2]
        capability_audits[run] = result[3]
        initial[run] = result[4]
        final[run] = result[5]
        peak[run] = result[6]
        episode_count[run] = result[7]
        recovered_count[run] = result[8]
        open_episode[run] = result[9]
        max_depth[run] = result[10]
        sum_depth[run] = result[11]
        max_duration[run] = result[12]
        sum_duration[run] = result[13]
        open_duration[run] = result[14]
        max_initial_excursion[run] = result[15]
        worsening_actions[run] = result[16]
        positive_runs[run] = result[17]
        threshold_counts[run] = result[18]
        state_checkpoints[run] = result[19]
        trajectory_digests[run] = result[20]
        physical_fingerprints[run] = result[21]
        full_fingerprints[run] = result[22]
        n = int(ns[run])
        final_occupancies[run, :n] = result[23]
        final_cursors[run, :n] = result[24]
        final_failure[run, :n] = result[25]
        final_counter[run, :n] = result[26]
        final_recent[run, :n] = result[27]
        maximum_radius_reads[run] = result[28]
        first_action_override_events[run] = result[29]
        first_override_pre_physical_fingerprints[run] = result[30]
        checkpoint_stops[run] = result[31]
        checkpoint_physical_fingerprints[run] = result[32]
        checkpoint_full_fingerprints[run] = result[33]
        checkpoint_ledgers[run] = result[34]
        checkpoint_capability_audits[run] = result[35]
        checkpoint_levels[run] = result[36]

    return (
        stops,
        events,
        ledgers,
        capability_audits,
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
        state_checkpoints,
        trajectory_digests,
        physical_fingerprints,
        full_fingerprints,
        final_occupancies,
        final_cursors,
        final_failure,
        final_counter,
        final_recent,
        maximum_radius_reads,
        first_action_override_events,
        first_override_pre_physical_fingerprints,
        checkpoint_stops,
        checkpoint_physical_fingerprints,
        checkpoint_full_fingerprints,
        checkpoint_ledgers,
        checkpoint_capability_audits,
        checkpoint_levels,
    )


def result_dict(result: tuple) -> dict[str, np.ndarray]:
    names = (
        "stop",
        "events",
        "ledger",
        "capability_audit",
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
        "state_checkpoints",
        "trajectory_digest",
        "physical_fingerprint",
        "full_fingerprint",
        "final_occupancy",
        "final_cursors",
        "final_failure",
        "final_counter",
        "final_recent",
        "maximum_radius_read",
        "first_action_override_event",
        "first_override_pre_physical_fingerprint",
        "checkpoint_stop",
        "checkpoint_physical_fingerprint",
        "checkpoint_full_fingerprint",
        "checkpoint_ledger",
        "checkpoint_capability_audit",
        "checkpoint_levels",
    )
    return dict(zip(names, result, strict=True))
