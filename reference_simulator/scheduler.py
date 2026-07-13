"""Deterministic activation scheduling and batch conflict resolution."""

from __future__ import annotations

from typing import Iterable

from .model import Proposal, ProposalKind, Scenario
from .rng import bounded, u64


def scheduled_actor(
    scenario: Scenario, event_index: int, *, include_draws: bool = True
) -> tuple[str, tuple[tuple[str, int, int, int], ...], int]:
    index, consumed = bounded(
        scenario.seed,
        scenario.scenario_id,
        "actor_activation",
        event_index,
        len(scenario.cells),
    )
    actor_id = scenario.cells[index].cell_id
    draws = (
        tuple(
            (
                "actor_activation",
                event_index,
                draw_index,
                u64(scenario.seed, scenario.scenario_id, "actor_activation", event_index, draw_index),
            )
            for draw_index in range(consumed)
        )
        if include_draws
        else ()
    )
    return actor_id, draws, consumed


def scheduled_side(scenario: Scenario, event_index: int) -> tuple[str, tuple[str, int, int, int]]:
    value = u64(scenario.seed, scenario.scenario_id, "bubble_side", event_index, 0)
    return ("left" if value < (1 << 63) else "right"), ("bubble_side", event_index, 0, value)


def scheduled_priority(scenario: Scenario, event_index: int) -> tuple[int, tuple[str, int, int, int]]:
    value = u64(scenario.seed, scenario.scenario_id, "conflict_priority", event_index, 0)
    return value, ("conflict_priority", event_index, 0, value)


def _resources(proposal: Proposal) -> set[tuple[str, str | int]]:
    resources: set[tuple[str, str | int]] = {("actor", proposal.actor_id)}
    if proposal.kind == ProposalKind.SWAP:
        resources.add(("position", proposal.actor_pos))
        assert proposal.target_pos is not None
        resources.add(("position", proposal.target_pos))
    return resources


def resolve_conflicts(proposals: Iterable[Proposal]) -> tuple[set[int], set[int]]:
    """Greedy maximal disjoint set under S03's frozen priority tuple."""
    candidates = [
        proposal for proposal in proposals
        if proposal.kind in (ProposalKind.SWAP, ProposalKind.MEMORY_UPDATE)
    ]
    ordered = sorted(
        candidates,
        key=lambda p: (
            p.priority,
            p.actor_id,
            p.target_pos if p.target_pos is not None else -1,
            p.ordinal,
        ),
    )
    accepted: set[int] = set()
    lost: set[int] = set()
    reserved: set[tuple[str, str | int]] = set()
    for proposal in ordered:
        resources = _resources(proposal)
        if resources & reserved:
            lost.add(proposal.ordinal)
        else:
            accepted.add(proposal.ordinal)
            reserved.update(resources)
    return accepted, lost
