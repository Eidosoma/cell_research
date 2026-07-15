"""Observed E01 trajectory projection and S08 behavior classifications.

This module is intentionally small.  It translates byte-exact E01 state into
the finite S04 structural quotient, but it does not pretend that the quotient
contains scheduler history.  Exact events are then matched to one labelled S05
opportunity by actor and (when E01 actually drew it) Bubble side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from reference_simulator.model import Architecture, Direction, FaultMode, Policy, RunState, Scenario
from src.detours.necessary_detour import (
    CLASS_COMPLETE_START,
    CLASS_NECESSARY_DETOUR,
    CLASS_QUIESCENT,
    CLASS_REACHABLE_NO_DETOUR,
    CLASS_UNREACHABLE_ACTIVE,
)
from src.detours.state_space import FamilySpec, StructuralState
from src.detours.transition_graph import opportunity_arrays


@dataclass(frozen=True, slots=True)
class ScenarioProjection:
    """One value-rank isomorphism from native identities to S04 identities."""

    family: FamilySpec
    native_to_canonical: Mapping[str, int]

    def state(self, native: RunState) -> StructuralState:
        occupancy = tuple(self.native_to_canonical[cell_id] for cell_id in native.occupancy)
        cursors = {
            self.native_to_canonical[cell_id]: int(cursor)
            for cell_id, cursor in native.selection_cursors.items()
        }
        return StructuralState.from_components(self.family, occupancy, cursors)


def project_scenario(scenario: Scenario) -> ScenarioProjection:
    """Canonicalize a homogeneous, unique-valued E01 scenario by value rank.

    S04 names value rank ``i`` as identity ``ci``.  Native E01 identity labels
    can differ (the retained n=4 trace is the important example), so policy,
    fault and cursor ownership are transported through this bijection.
    """

    directions = {cell.direction for cell in scenario.cells}
    if len(directions) != 1:
        raise ValueError("mixed-direction scenarios have no S04 family projection")
    values = [cell.value for cell in scenario.cells]
    if len(set(values)) != len(values):
        raise ValueError("tied-valued scenarios have no unique S04 identity projection")
    ranked = sorted(scenario.cells, key=lambda cell: cell.value)
    mapping = {cell.cell_id: index for index, cell in enumerate(ranked)}
    if scenario.architecture == Architecture.TRADITIONAL:
        if scenario.traditional_policy is None:
            raise ValueError("traditional scenario has no controller policy")
        policies = (scenario.traditional_policy,) * len(ranked)
    else:
        policies = tuple(cell.policy for cell in ranked)
    family = FamilySpec(
        n=len(ranked),
        architecture=scenario.architecture,
        direction=next(iter(directions)),
        policies=policies,
        faults=tuple(cell.fault for cell in ranked),
    )
    return ScenarioProjection(family, mapping)


def event_scheduler_label(
    projection: ScenarioProjection, event: Mapping[str, object]
) -> tuple[int, int, int]:
    """Return ``(opportunity, actor, side)`` for one retained serial event."""

    family = projection.family
    if family.architecture == Architecture.TRADITIONAL:
        actor = -1
        side = -1
    else:
        actor_id = str(event["actorId"])
        actor = projection.native_to_canonical[actor_id]
        draws = [
            row
            for row in event.get("randomAddressesAndDraws", [])  # type: ignore[union-attr]
            if row["stream"] == "bubble_side"
        ]
        expects_side = (
            family.policies[actor] == Policy.BUBBLE
            and family.faults[actor] == FaultMode.NORMAL
        )
        if expects_side:
            if len(draws) != 1:
                raise ValueError("normal Bubble event must retain exactly one side draw")
            side = 0 if int(draws[0]["value"]) < (1 << 63) else 1
        else:
            if draws:
                raise ValueError("non-Bubble or faulty actor unexpectedly consumed a side draw")
            side = -1
    opportunities = opportunity_arrays(family)
    matches = [
        index
        for index, (candidate_actor, candidate_side) in enumerate(
            zip(opportunities["actors"], opportunities["sides"])
        )
        if int(candidate_actor) == actor and int(candidate_side) == side
    ]
    if len(matches) != 1:
        raise ValueError("retained scheduler label is not one unique S05 opportunity")
    return matches[0], actor, side


def classify_observed_suffix(
    *,
    solver_classification: int,
    observed_success: bool,
    observed_excursion: int,
) -> str:
    """Partition an observed suffix without converting failure into detour depth."""

    if observed_excursion < 0:
        raise ValueError("observed excursion cannot be negative")
    if solver_classification == CLASS_COMPLETE_START:
        return "complete_start" if observed_success else "semantic_contradiction_complete_failed"
    if solver_classification == CLASS_QUIESCENT:
        return "quiescent_terminal" if not observed_success else "semantic_contradiction_quiescent_succeeded"
    if observed_success:
        if solver_classification == CLASS_UNREACHABLE_ACTIVE:
            return "semantic_contradiction_unreachable_succeeded"
        if solver_classification == CLASS_NECESSARY_DETOUR:
            return "successful_necessary_detour"
        if solver_classification == CLASS_REACHABLE_NO_DETOUR:
            return (
                "successful_no_detour"
                if observed_excursion == 0
                else "successful_unnecessary_detour"
            )
    else:
        if solver_classification == CLASS_NECESSARY_DETOUR:
            return "failed_where_detour_required"
        if solver_classification == CLASS_REACHABLE_NO_DETOUR:
            return "failed_despite_no_required_detour"
        if solver_classification == CLASS_UNREACHABLE_ACTIVE:
            return "failed_structurally_unreachable"
    raise ValueError("unrecognized observed/solver classification combination")


def lexicographic_relation(observed: Sequence[int], optimum: Sequence[int]) -> str:
    """Compare two equal-width integer cost vectors."""

    left = tuple(int(value) for value in observed)
    right = tuple(int(value) for value in optimum)
    if len(left) != len(right):
        raise ValueError("cost vectors have different widths")
    return "equal" if left == right else ("above_optimum" if left > right else "below_optimum")
