"""Deterministic activation scheduling and batch conflict resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from .model import Proposal, ProposalKind, Scenario
from .rng import bounded, u64


RandomDraw = tuple[str, int, int, int]


@dataclass(frozen=True, slots=True)
class ScheduledOpportunity:
    """One externally selected, fully charged actor opportunity.

    This is deliberately a data-only bridge.  A scheduler supplies an actor
    identity plus any counter-addressed selection draws it consumed; it cannot
    receive the transition state through this type.  An external common-random
    schedule may also freeze Bubble's left/right draw.  Leaving ``bubble_side``
    unset preserves the original scenario-keyed Bubble draw exactly.
    """

    actor_id: str
    random_draws: tuple[RandomDraw, ...] = ()
    stream_consumption: tuple[tuple[str, int], ...] = ()
    bubble_side: Literal["left", "right"] | None = None

    def __post_init__(self) -> None:
        if not self.actor_id:
            raise ValueError("scheduled actor identity must be nonempty")
        if any(not stream or count < 1 for stream, count in self.stream_consumption):
            raise ValueError("scheduler stream consumption must be positive")
        consumption = dict(self.stream_consumption)
        if len(consumption) != len(self.stream_consumption):
            raise ValueError("scheduler stream consumption may name each stream once")
        observed: dict[str, int] = {}
        for stream, event_index, draw_index, value in self.random_draws:
            if (
                not stream
                or event_index < 0
                or draw_index < 0
                or not 0 <= value < (1 << 64)
            ):
                raise ValueError("invalid counter-addressed scheduler draw")
            observed[stream] = observed.get(stream, 0) + 1
        if observed != consumption:
            raise ValueError("scheduler draws and stream consumption disagree")
        bubble_draws = observed.get("bubble_side", 0)
        if self.bubble_side is None and bubble_draws:
            raise ValueError("external bubble-side draws require bubble_side")
        if self.bubble_side is not None and bubble_draws != 1:
            raise ValueError("external bubble_side requires exactly one matching draw")


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
