"""Policy-native read, compare, proposal, and one-proposal routing primitives.

The interface deliberately separates policy information from the trusted
transition kernel.  A policy endpoint receives an immutable typed observation;
the central-local relay receives only an :class:`ActionEnvelope`; and E01's
mechanical validator/committer remains the common transition authority.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal

from reference_simulator.model import (
    Direction,
    FaultMode,
    Policy,
    Proposal,
    ProposalKind,
    RunState,
    Scenario,
)


ACTION_INTERFACE_VERSION = "E02-common-action-v1"


class ControlTopology(str, Enum):
    DISTRIBUTED_LOCAL = "distributed_local"
    CENTRAL_LOCAL_PROPOSAL_K1 = "central_local_proposal_k1"


class InformationPermission(str, Enum):
    POLICY_NATIVE_LOCAL = "policy_native_local"
    FULL_GLOBAL_STATE = "full_global_state"


class ReadCapability(str, Enum):
    ACTOR_SELF = "actor_self"
    SELECTED_NEIGHBOR = "selected_neighbor"
    STRICT_PREFIX = "strict_prefix"
    CURSOR_TARGET = "cursor_target"
    GLOBAL_OCCUPANCY = "global_occupancy"
    ANALYSIS_LABEL = "analysis_label"
    NEIGHBOR_ALGOTYPE = "neighbor_algotype"
    LEDGER_TOTALS = "ledger_totals"
    FUTURE_RANDOM_DRAW = "future_random_draw"


class ForbiddenInformationError(PermissionError):
    """Raised before an undeclared policy read can return information."""


_POLICY_CAPABILITIES = {
    Policy.BUBBLE: frozenset({ReadCapability.ACTOR_SELF, ReadCapability.SELECTED_NEIGHBOR}),
    Policy.INSERTION: frozenset({ReadCapability.ACTOR_SELF, ReadCapability.STRICT_PREFIX}),
    Policy.SELECTION: frozenset({ReadCapability.ACTOR_SELF, ReadCapability.CURSOR_TARGET}),
}


@dataclass(frozen=True, slots=True)
class VisibleCell:
    position: int
    value: int | float
    fault: FaultMode


@dataclass(frozen=True, slots=True)
class ActorView:
    cell_id: str
    position: int
    value: int | float
    policy: Policy
    direction: Direction
    fault: FaultMode
    line_length: int
    cursor: int | None


@dataclass(frozen=True, slots=True)
class BubbleObservation:
    actor: ActorView
    selected_side: Literal["left", "right"] | None
    target: VisibleCell | None


@dataclass(frozen=True, slots=True)
class InsertionObservation:
    actor: ActorView
    prefix_read: tuple[VisibleCell, ...]
    prefix_is_ordered: bool | None
    left_target: VisibleCell | None


@dataclass(frozen=True, slots=True)
class SelectionObservation:
    actor: ActorView
    cursor_target: VisibleCell | None


PolicyObservation = BubbleObservation | InsertionObservation | SelectionObservation


@dataclass(slots=True)
class OperationMeter:
    observation_reads: int = 0
    value_comparisons: int = 0

    def read(self) -> None:
        self.observation_reads += 1

    def ordered(self, left: int | float, right: int | float, direction: Direction) -> bool:
        self.value_comparisons += 1
        return left <= right if direction == Direction.ASCENDING else left >= right

    def strict_less(self, left: int | float, right: int | float) -> bool:
        self.value_comparisons += 1
        return left < right

    def strict_greater(self, left: int | float, right: int | float) -> bool:
        self.value_comparisons += 1
        return left > right

    def less_equal(self, left: int | float, right: int | float) -> bool:
        self.value_comparisons += 1
        return left <= right


class PolicyNativeReadGateway:
    """Capability-checked projection of one actor's legal E01 information."""

    __slots__ = (
        "__scenario",
        "__state",
        "__observed_identities",
        "actor_id",
        "meter",
        "used_capabilities",
    )

    def __init__(self, scenario: Scenario, state: RunState, actor_id: str, meter: OperationMeter):
        if actor_id not in scenario.cell_map:
            raise KeyError(actor_id)
        self.__scenario = scenario
        self.__state = state
        self.__observed_identities: dict[int, str] = {}
        self.actor_id = actor_id
        self.meter = meter
        self.used_capabilities: list[ReadCapability] = []

    @property
    def allowed_capabilities(self) -> frozenset[ReadCapability]:
        return _POLICY_CAPABILITIES[self.__scenario.cell_map[self.actor_id].policy]

    def _authorize(self, capability: ReadCapability) -> None:
        if capability not in self.allowed_capabilities:
            raise ForbiddenInformationError(
                f"{self.__scenario.cell_map[self.actor_id].policy.value} cannot read "
                f"{capability.value} under {InformationPermission.POLICY_NATIVE_LOCAL.value}"
            )
        self.used_capabilities.append(capability)

    def read(
        self,
        capability: ReadCapability,
        *,
        side: Literal["left", "right"] | None = None,
        position: int | None = None,
    ) -> ActorView | VisibleCell | None:
        """Return one authorized record and charge the E01 logical-read convention."""
        self._authorize(capability)
        cells = self.__scenario.cell_map
        state = self.__state
        actor = cells[self.actor_id]
        actor_position = state.occupancy.index(self.actor_id)

        if capability == ReadCapability.ACTOR_SELF:
            self.meter.read()
            cursor = state.selection_cursors.get(self.actor_id)
            return ActorView(
                self.actor_id,
                actor_position,
                actor.value,
                actor.policy,
                actor.direction,
                actor.fault,
                len(state.occupancy),
                cursor,
            )

        if capability == ReadCapability.SELECTED_NEIGHBOR:
            if side not in {"left", "right"}:
                raise ValueError("selected-neighbor read requires left or right")
            target_position = actor_position + (-1 if side == "left" else 1)
            if not 0 <= target_position < len(state.occupancy):
                return None
            target = cells[state.occupancy[target_position]]
            self.meter.read()
            self.__observed_identities[target_position] = target.cell_id
            return VisibleCell(target_position, target.value, target.fault)

        if capability == ReadCapability.STRICT_PREFIX:
            if position is None or not 0 <= position < actor_position:
                raise ValueError("strict-prefix read position must be left of the actor")
            target = cells[state.occupancy[position]]
            self.meter.read()
            self.__observed_identities[position] = target.cell_id
            return VisibleCell(position, target.value, target.fault)

        if capability == ReadCapability.CURSOR_TARGET:
            cursor = state.selection_cursors[self.actor_id]
            if position is not None and position != cursor:
                raise ValueError("Selection can read only its current cursor position")
            if not 0 <= cursor < len(state.occupancy) or cursor == actor_position:
                return None
            target = cells[state.occupancy[cursor]]
            self.meter.read()
            self.__observed_identities[cursor] = target.cell_id
            return VisibleCell(cursor, target.value, target.fault)

        raise AssertionError(f"authorized capability has no reader: {capability.value}")

    def observed_identity_at(self, position: int) -> str | None:
        """Trusted envelope builder lookup; identities are absent from policy views."""
        return self.__observed_identities.get(position)


@dataclass(frozen=True, slots=True)
class ActionEnvelope:
    proposal: Proposal
    observed_target_id: str | None
    permission: InformationPermission
    used_capabilities: tuple[ReadCapability, ...]


def _noop(actor: ActorView, reason: str, meter: OperationMeter) -> Proposal:
    return Proposal(
        ProposalKind.NO_OP,
        actor.cell_id,
        actor.position,
        reason=reason,
        observation_reads=meter.observation_reads,
        value_comparisons=meter.value_comparisons,
    )


def build_policy_native_observation(
    scenario: Scenario,
    state: RunState,
    actor_id: str,
    *,
    side: Literal["left", "right"] | None = None,
) -> tuple[PolicyObservation, OperationMeter, PolicyNativeReadGateway]:
    meter = OperationMeter()
    gateway = PolicyNativeReadGateway(scenario, state, actor_id, meter)
    actor = gateway.read(ReadCapability.ACTOR_SELF)
    assert isinstance(actor, ActorView)

    if actor.policy == Policy.BUBBLE:
        target = None
        if actor.fault == FaultMode.NORMAL:
            if side not in {"left", "right"}:
                raise ValueError("Bubble activation requires a side")
            target = gateway.read(ReadCapability.SELECTED_NEIGHBOR, side=side)
            assert target is None or isinstance(target, VisibleCell)
        return BubbleObservation(actor, side, target), meter, gateway

    if actor.policy == Policy.INSERTION:
        prefix: list[VisibleCell] = []
        prefix_is_ordered: bool | None = None
        left_target = None
        if actor.fault == FaultMode.NORMAL and actor.position > 0:
            prefix_is_ordered = True
            prior_value: int | float | None = None
            prior_normal = False
            for position in range(actor.position):
                item = gateway.read(ReadCapability.STRICT_PREFIX, position=position)
                assert isinstance(item, VisibleCell)
                prefix.append(item)
                if item.fault != FaultMode.NORMAL:
                    prior_normal = False
                    prior_value = None
                    continue
                if prior_normal and not meter.ordered(prior_value, item.value, actor.direction):  # type: ignore[arg-type]
                    prefix_is_ordered = False
                    break
                prior_normal = True
                prior_value = item.value
            if prefix_is_ordered:
                left_target = gateway.read(
                    ReadCapability.STRICT_PREFIX, position=actor.position - 1
                )
                assert isinstance(left_target, VisibleCell)
        return (
            InsertionObservation(actor, tuple(prefix), prefix_is_ordered, left_target),
            meter,
            gateway,
        )

    if actor.policy == Policy.SELECTION:
        target = None
        if (
            actor.fault == FaultMode.NORMAL
            and actor.cursor is not None
            and 0 <= actor.cursor < actor.line_length
            and actor.cursor != actor.position
        ):
            target = gateway.read(ReadCapability.CURSOR_TARGET, position=actor.cursor)
            assert isinstance(target, VisibleCell)
        return SelectionObservation(actor, target), meter, gateway

    raise AssertionError(f"unsupported policy {actor.policy}")


def propose_from_observation(observation: PolicyObservation, meter: OperationMeter) -> Proposal:
    """Apply the frozen E01 rule using no object that can reveal raw state."""
    actor = observation.actor
    if actor.fault != FaultMode.NORMAL:
        return _noop(actor, "actor_fault", meter)

    if isinstance(observation, BubbleObservation):
        if observation.target is None:
            return _noop(actor, "boundary", meter)
        target = observation.target
        if actor.direction == Direction.ASCENDING:
            inversion = (
                meter.strict_less(actor.value, target.value)
                if observation.selected_side == "left"
                else meter.strict_greater(actor.value, target.value)
            )
        else:
            inversion = (
                meter.strict_greater(actor.value, target.value)
                if observation.selected_side == "left"
                else meter.strict_less(actor.value, target.value)
            )
        if not inversion:
            return _noop(actor, "ordered_or_equal", meter)
        return Proposal(
            ProposalKind.SWAP,
            actor.cell_id,
            actor.position,
            target_pos=target.position,
            reason="strict_adjacent_inversion",
            observation_reads=meter.observation_reads,
            value_comparisons=meter.value_comparisons,
        )

    if isinstance(observation, InsertionObservation):
        if actor.position == 0:
            return _noop(actor, "boundary", meter)
        if observation.prefix_is_ordered is False:
            return _noop(actor, "prefix_not_ordered", meter)
        target = observation.left_target
        assert target is not None
        inversion = (
            meter.strict_less(actor.value, target.value)
            if actor.direction == Direction.ASCENDING
            else meter.strict_greater(actor.value, target.value)
        )
        if not inversion:
            return _noop(actor, "ordered_or_equal", meter)
        return Proposal(
            ProposalKind.SWAP,
            actor.cell_id,
            actor.position,
            target_pos=target.position,
            reason="insertion_into_ordered_prefix",
            observation_reads=meter.observation_reads,
            value_comparisons=meter.value_comparisons,
        )

    if isinstance(observation, SelectionObservation):
        cursor = actor.cursor
        assert cursor is not None
        if not 0 <= cursor < actor.line_length:
            return _noop(actor, "cursor_exhausted", meter)
        if cursor == actor.position:
            return _noop(actor, "at_cursor", meter)
        target = observation.cursor_target
        assert target is not None
        delta = 1 if actor.direction == Direction.ASCENDING else -1
        if target.fault == FaultMode.STUCK:
            return Proposal(
                ProposalKind.MEMORY_UPDATE,
                actor.cell_id,
                actor.position,
                new_cursor=cursor + delta,
                reason="skip_stuck_target",
                observation_reads=meter.observation_reads,
                value_comparisons=meter.value_comparisons,
            )
        if meter.less_equal(target.value, actor.value):
            return Proposal(
                ProposalKind.MEMORY_UPDATE,
                actor.cell_id,
                actor.position,
                new_cursor=cursor + delta,
                reason="target_already_extreme",
                observation_reads=meter.observation_reads,
                value_comparisons=meter.value_comparisons,
            )
        return Proposal(
            ProposalKind.SWAP,
            actor.cell_id,
            actor.position,
            target_pos=cursor,
            reason="selection_target",
            observation_reads=meter.observation_reads,
            value_comparisons=meter.value_comparisons,
        )

    raise AssertionError(type(observation))


class CentralK1Relay:
    """A coordinator transport that can see exactly one proposal envelope."""

    __slots__ = ()

    def forward_one(self, envelopes: tuple[ActionEnvelope, ...]) -> ActionEnvelope:
        if len(envelopes) != 1:
            raise ValueError("central_local_proposal_k1 requires exactly one proposal")
        envelope = envelopes[0]
        if not isinstance(envelope, ActionEnvelope):
            raise TypeError("central relay accepts ActionEnvelope only")
        return envelope


class CommonActionInterface:
    """Common observation/proposal route used by both frozen S02 topologies."""

    __slots__ = ("central_relay",)

    def __init__(self) -> None:
        self.central_relay = CentralK1Relay()

    def envelope_for(
        self,
        topology: ControlTopology,
        scenario: Scenario,
        state: RunState,
        actor_id: str,
        *,
        side: Literal["left", "right"] | None = None,
    ) -> ActionEnvelope:
        observation, meter, gateway = build_policy_native_observation(
            scenario, state, actor_id, side=side
        )
        proposal = propose_from_observation(observation, meter)
        observed_target_id = (
            gateway.observed_identity_at(proposal.target_pos)
            if proposal.kind == ProposalKind.SWAP and proposal.target_pos is not None
            else None
        )
        proposal = replace(proposal, observed_target_id=observed_target_id)
        envelope = ActionEnvelope(
            proposal=proposal,
            observed_target_id=observed_target_id,
            permission=InformationPermission.POLICY_NATIVE_LOCAL,
            used_capabilities=tuple(gateway.used_capabilities),
        )
        if topology == ControlTopology.DISTRIBUTED_LOCAL:
            return envelope
        if topology == ControlTopology.CENTRAL_LOCAL_PROPOSAL_K1:
            return self.central_relay.forward_one((envelope,))
        raise AssertionError(topology)

    def proposal_for(
        self,
        topology: ControlTopology,
        scenario: Scenario,
        state: RunState,
        actor_id: str,
        *,
        side: Literal["left", "right"] | None = None,
    ) -> Proposal:
        return self.envelope_for(topology, scenario, state, actor_id, side=side).proposal
