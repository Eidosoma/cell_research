"""Clean-room policy observations and side-effect-free proposal functions."""

from __future__ import annotations

from typing import Literal

from .model import (
    Architecture,
    Direction,
    FaultMode,
    Policy,
    Proposal,
    ProposalKind,
    RunState,
    Scenario,
)


def _noop(actor_id: str, actor_pos: int, reason: str, reads: int = 0, comparisons: int = 0) -> Proposal:
    return Proposal(
        ProposalKind.NO_OP,
        actor_id,
        actor_pos,
        reason=reason,
        observation_reads=reads,
        value_comparisons=comparisons,
    )


def _ordered(left: float, right: float, direction: Direction) -> bool:
    return left <= right if direction == Direction.ASCENDING else left >= right


def _prefix_is_ordered(
    scenario: Scenario,
    state: RunState,
    stop: int,
    direction: Direction,
) -> tuple[bool, int, int]:
    """Check normal-cell segments of strict prefix ``[0, stop)``."""
    cells = scenario.cell_map
    reads = 0
    comparisons = 0
    prior_value: float | None = None
    prior_normal = False
    for cell_id in state.occupancy[:stop]:
        cell = cells[cell_id]
        reads += 1
        if cell.fault != FaultMode.NORMAL:
            prior_normal = False
            prior_value = None
            continue
        if prior_normal:
            comparisons += 1
            if not _ordered(prior_value, cell.value, direction):  # type: ignore[arg-type]
                return False, reads, comparisons
        prior_normal = True
        prior_value = cell.value
    return True, reads, comparisons


def cell_view_proposal(
    scenario: Scenario,
    state: RunState,
    actor_id: str,
    *,
    side: Literal["left", "right"] | None = None,
) -> Proposal:
    """Evaluate exactly one S03 cell-view activation against a snapshot."""
    cells = scenario.cell_map
    positions = {cell_id: position for position, cell_id in enumerate(state.occupancy)}
    actor = cells[actor_id]
    position = positions[actor_id]
    if actor.fault != FaultMode.NORMAL:
        return _noop(actor_id, position, "actor_fault", reads=1)

    if actor.policy == Policy.BUBBLE:
        if side is None:
            raise ValueError("Bubble activation requires a side")
        target_position = position + (-1 if side == "left" else 1)
        if not 0 <= target_position < len(state.occupancy):
            return _noop(actor_id, position, "boundary", reads=1)
        target = cells[state.occupancy[target_position]]
        if actor.direction == Direction.ASCENDING:
            inversion = actor.value < target.value if side == "left" else actor.value > target.value
        else:
            inversion = actor.value > target.value if side == "left" else actor.value < target.value
        if not inversion:
            return _noop(actor_id, position, "ordered_or_equal", reads=2, comparisons=1)
        return Proposal(
            ProposalKind.SWAP,
            actor_id,
            position,
            target_pos=target_position,
            reason="strict_adjacent_inversion",
            observation_reads=2,
            value_comparisons=1,
        )

    if actor.policy == Policy.INSERTION:
        if position == 0:
            return _noop(actor_id, position, "boundary", reads=1)
        valid_prefix, reads, comparisons = _prefix_is_ordered(
            scenario, state, position, actor.direction
        )
        if not valid_prefix:
            return _noop(actor_id, position, "prefix_not_ordered", reads + 1, comparisons)
        target_position = position - 1
        target = cells[state.occupancy[target_position]]
        inversion = (
            actor.value < target.value
            if actor.direction == Direction.ASCENDING
            else actor.value > target.value
        )
        if not inversion:
            return _noop(actor_id, position, "ordered_or_equal", reads + 2, comparisons + 1)
        return Proposal(
            ProposalKind.SWAP,
            actor_id,
            position,
            target_pos=target_position,
            reason="insertion_into_ordered_prefix",
            observation_reads=reads + 2,
            value_comparisons=comparisons + 1,
        )

    if actor.policy == Policy.SELECTION:
        cursor = state.selection_cursors[actor_id]
        if not 0 <= cursor < len(state.occupancy):
            return _noop(actor_id, position, "cursor_exhausted", reads=1)
        if cursor == position:
            return _noop(actor_id, position, "at_cursor", reads=1)
        target = cells[state.occupancy[cursor]]
        delta = 1 if actor.direction == Direction.ASCENDING else -1
        if target.fault == FaultMode.STUCK:
            return Proposal(
                ProposalKind.MEMORY_UPDATE,
                actor_id,
                position,
                new_cursor=cursor + delta,
                reason="skip_stuck_target",
                observation_reads=2,
            )
        # Frozen S03 behavior: this same numeric comparison is intentional for
        # ascending and descending Selection. Direction changes only the cursor.
        if target.value <= actor.value:
            return Proposal(
                ProposalKind.MEMORY_UPDATE,
                actor_id,
                position,
                new_cursor=cursor + delta,
                reason="target_already_extreme",
                observation_reads=2,
                value_comparisons=1,
            )
        return Proposal(
            ProposalKind.SWAP,
            actor_id,
            position,
            target_pos=cursor,
            reason="selection_target",
            observation_reads=2,
            value_comparisons=1,
        )

    raise AssertionError(f"unsupported policy {actor.policy}")


def traditional_proposal(scenario: Scenario, state: RunState) -> Proposal:
    """One action from the separately named conventional controller R spec.

    ``traditional_global_primary_action_v1`` is a clean-room control, not an
    inference about the missing publication generator. Passive cells are movable;
    a swap touching a stuck cell is proposed and then rejected by validation.
    """
    policy = scenario.traditional_policy
    if policy is None:
        raise ValueError("traditional scenario has no controller policy")
    cells = scenario.cell_map
    order = state.occupancy
    n = len(order)
    reads = 0
    comparisons = 0

    if policy == Policy.BUBBLE:
        for left in range(n - 1):
            a, b = cells[order[left]], cells[order[left + 1]]
            reads += 2
            comparisons += 1
            direction = a.direction
            if not _ordered(a.value, b.value, direction):
                return Proposal(
                    ProposalKind.SWAP, a.cell_id, left, target_pos=left + 1,
                    reason="traditional_global_bubble_first_inversion",
                    observation_reads=reads, value_comparisons=comparisons,
                )
        return _noop("__controller__", -1, "traditional_no_inversion", reads, comparisons)

    if policy == Policy.INSERTION:
        if n < 2:
            return _noop("__controller__", -1, "traditional_singleton")
        direction = cells[order[0]].direction
        for right in range(1, n):
            a, b = cells[order[right - 1]], cells[order[right]]
            reads += 2
            comparisons += 1
            if not _ordered(a.value, b.value, direction):
                return Proposal(
                    ProposalKind.SWAP, b.cell_id, right, target_pos=right - 1,
                    reason="traditional_global_insertion_adjacent_action",
                    observation_reads=reads, value_comparisons=comparisons,
                )
        return _noop("__controller__", -1, "traditional_no_inversion", reads, comparisons)

    if policy == Policy.SELECTION:
        if n < 2:
            return _noop("__controller__", -1, "traditional_singleton")
        direction = cells[order[0]].direction
        for boundary in range(n - 1):
            extreme = boundary
            for index in range(boundary + 1, n):
                reads += 2
                comparisons += 1
                candidate = cells[order[index]].value
                current = cells[order[extreme]].value
                if (direction == Direction.ASCENDING and candidate < current) or (
                    direction == Direction.DESCENDING and candidate > current
                ):
                    extreme = index
            if extreme != boundary:
                actor_id = order[extreme]
                return Proposal(
                    ProposalKind.SWAP, actor_id, extreme, target_pos=boundary,
                    reason="traditional_global_selection_extreme",
                    observation_reads=reads, value_comparisons=comparisons,
                )
        return _noop("__controller__", -1, "traditional_no_inversion", reads, comparisons)

    raise AssertionError(f"unsupported traditional policy {policy}")


def has_admissible_cell_view_change(scenario: Scenario, state: RunState) -> bool:
    """Proof-equivalent optimized quiescence predicate for the S03 policies."""
    cells = scenario.cell_map
    occupancy = state.occupancy
    n = len(occupancy)

    # Bubble: enumerate both adjacent choices for each normal actor.
    for position, cell_id in enumerate(occupancy):
        actor = cells[cell_id]
        if actor.fault != FaultMode.NORMAL or actor.policy != Policy.BUBBLE:
            continue
        for side, target_position in (("left", position - 1), ("right", position + 1)):
            if not 0 <= target_position < n:
                continue
            target = cells[occupancy[target_position]]
            if target.fault == FaultMode.STUCK:
                continue
            if actor.direction == Direction.ASCENDING:
                inversion = actor.value < target.value if side == "left" else actor.value > target.value
            else:
                inversion = actor.value > target.value if side == "left" else actor.value < target.value
            if inversion:
                return True

    # Insertion: build prefix validity for both directions in O(n).
    for direction in (Direction.ASCENDING, Direction.DESCENDING):
        prefix_valid = [True] * (n + 1)
        for stop in range(2, n + 1):
            left = cells[occupancy[stop - 2]]
            right = cells[occupancy[stop - 1]]
            pair_valid = (
                left.fault != FaultMode.NORMAL
                or right.fault != FaultMode.NORMAL
                or _ordered(left.value, right.value, direction)
            )
            prefix_valid[stop] = prefix_valid[stop - 1] and pair_valid
        for position in range(1, n):
            actor = cells[occupancy[position]]
            if (
                actor.fault != FaultMode.NORMAL
                or actor.policy != Policy.INSERTION
                or actor.direction != direction
                or not prefix_valid[position]
            ):
                continue
            target = cells[occupancy[position - 1]]
            if target.fault == FaultMode.STUCK:
                continue
            inversion = actor.value < target.value if direction == Direction.ASCENDING else actor.value > target.value
            if inversion:
                return True

    # Selection: every in-bounds cursor not at actor position changes memory or swaps.
    positions = {cell_id: position for position, cell_id in enumerate(occupancy)}
    for actor_id, cursor in state.selection_cursors.items():
        actor = cells[actor_id]
        if (
            actor.fault == FaultMode.NORMAL
            and 0 <= cursor < n
            and cursor != positions[actor_id]
        ):
            return True
    return False


def has_admissible_change(scenario: Scenario, state: RunState) -> bool:
    if scenario.architecture == Architecture.CELL_VIEW:
        return has_admissible_cell_view_change(scenario, state)
    proposal = traditional_proposal(scenario, state)
    if proposal.kind != ProposalKind.SWAP or proposal.target_pos is None:
        return proposal.kind == ProposalKind.MEMORY_UPDATE
    cells = scenario.cell_map
    return (
        cells[state.occupancy[proposal.actor_pos]].fault != FaultMode.STUCK
        and cells[state.occupancy[proposal.target_pos]].fault != FaultMode.STUCK
    )
