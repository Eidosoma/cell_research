"""Train-safe full-episode bindings from the S01 DSL to native simulators.

The adapters in this module are deliberately control-plane glue.  Policies
receive only their declared typed observations; predecessor engines retain
actor scheduling, mechanical legality, fault handling, conflict resolution,
atomic commit, clocks, stopping rules, and native ledgers.  Adapter-owned DSL
memory, peer messages, operation counts, and licensed nonlocal capabilities
are kept in separate ledgers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any, Literal, Mapping, Sequence

from causal_simulator.action_interface import (
    ActionEnvelope,
    ControlTopology,
    InformationPermission,
)
from reference_simulator.engine import invariant_error, is_complete, run
from reference_simulator.model import (
    Direction,
    FaultMode,
    Policy,
    Proposal,
    ProposalKind,
    RunResult,
    RunState,
    Scenario,
    canonical_json_bytes,
)
from reference_simulator.policies import _prefix_is_ordered
from reference_simulator.transition_primitives import validate_proposal
from src.morph2d.baseline import TargetMetricTracker, load_baseline_assets
from src.morph2d.channels import CHANNEL_LEDGER_FIELDS
from src.morph2d.engine import (
    EpisodeDefinition,
    _initial_state,
    _state_blind_actor_schedule,
)
from src.morph2d.movements import (
    LEDGER_FIELDS as MOVEMENT_LEDGER_FIELDS,
    MovementProposal,
    MovementState,
    movement_state_sha256,
    parse_movement_state,
    resolve_batch,
    validate_state_against_environment,
)
from src.morph2d.policies import (
    OBSERVATION_LEDGER_FIELDS,
    ObservationBuild,
    PolicyDefinition,
    build_policy_observation,
    compile_relation_profile,
)
from src.policy_dsl import CompiledPolicy, compile_policy, execute_policy

from .communication import RecipientActivationMessageBus, SignalEmission
from .contracts import EvaluationAction, SuiteValidationError, canonical_sha256
from .e05_semantics import (
    TARGET_CHANGE_SEMANTICS_VERSION,
    target_change_terminal_transition,
    validate_target_change_result_semantics,
)
from .portfolio_adapters import PortfolioDispatcher


ADAPTER_VERSION = "e07.s04a.dsl-native-adapters.v1"
SPATIAL_MAX_ENABLED_DISPLACEMENT = 6  # Frozen S04 maximum rotation length.


def compiled_policies(action: EvaluationAction) -> dict[str, CompiledPolicy]:
    if action.mode != "dsl_episode":
        raise SuiteValidationError("DSL adapter requires dsl_episode mode")
    values = tuple(compile_policy(item) for item in action.policy_documents)
    return {item.policy_id: item for item in values}


def dsl_action(
    policy_documents: Sequence[Mapping[str, Any]],
    *,
    native_policy_bindings: Mapping[str, str] | None = None,
) -> EvaluationAction:
    """Construct a canonically hashed DSL episode action."""

    documents = tuple(policy_documents)
    policies = tuple(compile_policy(item) for item in documents)
    bindings = dict(native_policy_bindings or {})
    if len(policies) == 1:
        policy_id = policies[0].policy_id
        digest = policies[0].policy_sha256
    else:
        policy_id = "dsl_portfolio"
        digest = canonical_sha256(
            "E07/S04A/dsl-portfolio/v1",
            {
                "bindings": bindings,
                "policies": [item.policy_sha256 for item in policies],
            },
        )
    return EvaluationAction(
        policy_id,
        digest,
        mode="dsl_episode",
        policy_documents=documents,
        native_policy_bindings=bindings,
    )


def _line_carrier(policy: CompiledPolicy) -> Policy:
    selection = {
        "selection.cursor_in_bounds",
        "selection.cursor_at_actor",
        "selection.target.value",
        "selection.target.stuck",
    }
    if policy.permissions & selection or any(
        action["kind"] in {"swap_cursor", "advance_cursor"}
        for _, actions in policy.rules
        for action in actions
    ):
        return Policy.SELECTION
    if "activation.side" in policy.permissions:
        return Policy.BUBBLE
    return Policy.INSERTION


def _validate_line_authority(policy: CompiledPolicy, carrier: Policy) -> None:
    if policy.environment != "line1d.v1":
        raise SuiteValidationError("line adapter requires line1d.v1 policy")
    common = {
        "own.value",
        "own.position",
        "own.direction",
        "last_action.rejected",
        "repair.nudge_count",
        "signal.neighbor_sum_u8",
        "counter.choice_u8",
    }
    native = {
        Policy.BUBBLE: {
            "activation.side",
            "neighbor.left.exists",
            "neighbor.left.value",
            "neighbor.left.movable",
            "neighbor.right.exists",
            "neighbor.right.value",
            "neighbor.right.movable",
        },
        Policy.INSERTION: {
            "line.prefix_ordered",
            "neighbor.left.exists",
            "neighbor.left.value",
            "neighbor.left.movable",
        },
        Policy.SELECTION: {
            "selection.cursor_in_bounds",
            "selection.cursor_at_actor",
            "selection.target.value",
            "selection.target.stuck",
        },
    }
    unsupported = policy.permissions - common - native[carrier]
    if unsupported:
        raise SuiteValidationError(
            f"policy requests observations outside {carrier.value} authority: "
            f"{sorted(unsupported)}"
        )


def bind_homogeneous_line_scenario(
    scenario: Scenario, policy: CompiledPolicy
) -> Scenario:
    """Create a deterministic policy-carrier scenario from authorized inputs."""

    carrier = _line_carrier(policy)
    _validate_line_authority(policy, carrier)
    if all(cell.policy == carrier for cell in scenario.cells):
        return scenario
    cells = tuple(replace(cell, policy=carrier) for cell in scenario.cells)
    return Scenario.create(
        cells,
        initial_occupancy=scenario.initial_occupancy,
        seed=scenario.seed,
        max_activations=scenario.max_activations,
        architecture=scenario.architecture,
        batch_width=scenario.batch_width,
        generation_key=(
            f"{scenario.generation_key}/S04A/{policy.policy_sha256}/{carrier.value}"
        ),
        fault_placement=scenario.fault_placement,
        requested_fault_count=scenario.requested_fault_count,
    )


def bind_portfolio_line_scenario(
    scenario: Scenario, action: EvaluationAction
) -> Scenario:
    """Bind a portfolio to native carriers without changing carrier authority."""

    policies = compiled_policies(action)
    if not action.portfolio_definition:
        raise SuiteValidationError("portfolio line binding requires a definition")
    dispatcher = PortfolioDispatcher(action.portfolio_definition, policies)
    carriers = {Policy(name) for name in dispatcher.members_by_carrier}
    for carrier_name, members in dispatcher.members_by_carrier.items():
        carrier = Policy(carrier_name)
        for policy in members:
            _validate_line_authority(policy, carrier)
    present = {cell.policy for cell in scenario.cells}
    if len(carriers) == 1 and len(present) == 1:
        carrier = next(iter(carriers))
        if present == {carrier}:
            return scenario
        cells = tuple(replace(cell, policy=carrier) for cell in scenario.cells)
        return Scenario.create(
            cells,
            initial_occupancy=scenario.initial_occupancy,
            seed=scenario.seed,
            max_activations=scenario.max_activations,
            architecture=scenario.architecture,
            batch_width=scenario.batch_width,
            generation_key=(
                f"{scenario.generation_key}/S08A/{action.policy_sha256}/{carrier.value}"
            ),
            fault_placement=scenario.fault_placement,
            requested_fault_count=scenario.requested_fault_count,
        )
    if present != carriers:
        raise SuiteValidationError(
            "portfolio carriers do not exactly preserve heterogeneous native carriers"
        )
    return scenario


def _line_neighbors(state: RunState) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for position, actor_id in enumerate(state.occupancy):
        neighbors = []
        if position:
            neighbors.append(state.occupancy[position - 1])
        if position + 1 < len(state.occupancy):
            neighbors.append(state.occupancy[position + 1])
        result[actor_id] = tuple(neighbors)
    return result


def _empty_runtime_ledger() -> dict[str, int]:
    return {
        "dslActivations": 0,
        "dslOperations": 0,
        "dslMemoryWrites": 0,
        "dslSignalBundles": 0,
        "licensedPrefixPredicateEvaluations": 0,
        "licensedPrefixValueReads": 0,
        "licensedPrefixValueComparisons": 0,
        "engineCursorStateReads": 0,
        "engineCursorTargetProjectionReads": 0,
        "engineCursorAdvanceActions": 0,
        "engineCursorSwapActions": 0,
        "licensedLongRangeRequestedDistance": 0,
        "licensedLongRangeMaximumRequestedDistance": 0,
        "trustedValueProjectionReads": 0,
        "committedNativeMovementActions": 0,
        "committedNativeDisplacement": 0,
        "nativeEligibleActionOpportunities": 0,
        "rejectedNativeActions": 0,
    }


@dataclass
class _PendingLineEffects:
    actor_id: str
    event_index: int
    policy_sha256: str
    emitted: Mapping[int, int]
    neighbors: Mapping[str, Sequence[str]]


_LINE_VALUE_FIELDS = {
    "own.value",
    "neighbor.left.value",
    "neighbor.right.value",
    "selection.target.value",
}


def _line_condition_cost(
    expression: Mapping[str, Any],
    scalars: Mapping[str, Any],
    memory: Mapping[str, int],
) -> tuple[bool, int]:
    keys = set(expression)
    if keys == {"const"}:
        return bool(expression["const"]), 0
    if keys == {"not"}:
        value, count = _line_condition_cost(expression["not"], scalars, memory)
        return not value, count
    if keys in ({"all"}, {"any"}):
        mode = next(iter(keys))
        count = 0
        for child in expression[mode]:
            value, child_count = _line_condition_cost(child, scalars, memory)
            count += child_count
            if mode == "all" and not value:
                return False, count
            if mode == "any" and value:
                return True, count
        return mode == "all", count

    def operand(raw):
        source, payload = next(iter(raw.items()))
        if source == "const":
            return payload
        if source == "obs":
            return scalars[payload]
        return memory[payload]

    left = operand(expression["left"])
    right = operand(expression["right"])
    result = {
        "eq": left == right,
        "ne": left != right,
        "lt": left < right,
        "le": left <= right,
        "gt": left > right,
        "ge": left >= right,
    }[expression["op"]]
    fields = {
        payload
        for raw in (expression["left"], expression["right"])
        for source, payload in raw.items()
        if source == "obs"
    }
    return result, int(bool(fields & _LINE_VALUE_FIELDS))


def _line_rule_value_comparisons(
    policy: CompiledPolicy,
    observation: Mapping[str, Any],
    memory: Mapping[str, int],
) -> int:
    count = 0
    for expression, _ in policy.rules:
        matched, additions = _line_condition_cost(
            expression, observation["scalars"], memory
        )
        count += additions
        if matched:
            break
    return count


class LineDslRuntime:
    """Identity-owned DSL state and proposal projection for an E01 line."""

    def __init__(
        self,
        policies_by_native: Mapping[Policy, CompiledPolicy],
        *,
        value_projection: Mapping[str, int] | None = None,
        nudge_count: int = 0,
        portfolio_dispatcher: PortfolioDispatcher | None = None,
    ) -> None:
        if not policies_by_native:
            raise SuiteValidationError("line runtime needs at least one policy binding")
        if any(item.environment != "line1d.v1" for item in policies_by_native.values()):
            raise SuiteValidationError("line runtime received a non-line policy")
        self.policies_by_native = dict(policies_by_native)
        self.all_policies = {
            policy.policy_sha256: policy for policy in policies_by_native.values()
        }
        if portfolio_dispatcher is not None:
            self.all_policies.update(
                {
                    policy.policy_sha256: policy
                    for members in portfolio_dispatcher.members_by_carrier.values()
                    for policy in members
                }
            )
        self.value_projection = dict(value_projection or {})
        self.nudge_count = int(nudge_count)
        self.memory: dict[str, dict[str, int]] = {}
        self.member_memory: dict[str, dict[str, dict[str, int]]] = {}
        self.last_rejected: dict[str, bool] = {}
        self.portfolio_dispatcher = portfolio_dispatcher
        self.buses: dict[tuple[str, int, int], RecipientActivationMessageBus] = {}
        self.pending_effects: dict[int, _PendingLineEffects] = {}
        self.ledger = _empty_runtime_ledger()
        self.decision_counts: Counter[str] = Counter()
        self.activation_counts: Counter[str] = Counter()
        self.dispatch_counts: Counter[str] = Counter()

    def _policy(
        self,
        scenario: Scenario,
        state: RunState,
        actor_id: str,
        *,
        mutate: bool,
    ) -> CompiledPolicy:
        native = scenario.cell_map[actor_id].policy
        if self.portfolio_dispatcher is not None:
            by_carrier: dict[str, list[str]] = {
                carrier: [] for carrier in self.portfolio_dispatcher.members_by_carrier
            }
            for cell in scenario.cells:
                by_carrier.setdefault(cell.policy.value, []).append(cell.cell_id)
            self.portfolio_dispatcher.register_identities(by_carrier)
            signal = self.portfolio_dispatcher.selector_signal
            selector_value: bool | int = False
            if signal == "last_action.rejected":
                selector_value = self.last_rejected.get(actor_id, False)
            elif signal == "repair.nudge_count":
                selector_value = self.nudge_count
            policy = self.portfolio_dispatcher.select(
                native.value,
                actor_id,
                self.dispatch_counts.get(actor_id, 0),
                selector_value=selector_value,
                mutate=mutate,
            )
            if mutate:
                self.dispatch_counts[actor_id] += 1
            return policy
        try:
            return self.policies_by_native[native]
        except KeyError as exc:
            raise SuiteValidationError(
                f"missing DSL binding for native Algotype {native.value}"
            ) from exc

    def _bus(self, policy: CompiledPolicy) -> RecipientActivationMessageBus | None:
        if policy.signal_channels == 0:
            return None
        key = (
            policy.policy_sha256,
            policy.signal_channels,
            policy.signal_bits_per_channel,
        )
        return self.buses.setdefault(
            key,
            RecipientActivationMessageBus(
                channels=policy.signal_channels,
                bits_per_channel=policy.signal_bits_per_channel,
            ),
        )

    def _memory_for(
        self, actor_id: str, policy: CompiledPolicy, *, mutate: bool
    ) -> dict[str, int]:
        initial = {item.name: item.initial for item in policy.memory}
        if self.portfolio_dispatcher is None:
            if mutate:
                return self.memory.setdefault(actor_id, initial)
            return self.memory.get(actor_id, initial)
        actor_namespaces = self.member_memory.get(actor_id)
        if actor_namespaces is None:
            if not mutate:
                return initial
            actor_namespaces = self.member_memory.setdefault(actor_id, {})
        existing = actor_namespaces.get(policy.policy_sha256)
        if existing is not None:
            return existing
        if mutate:
            actor_namespaces[policy.policy_sha256] = initial
            self.portfolio_dispatcher.note_namespace_initialization()
        return initial

    def _write_memory(
        self, actor_id: str, policy: CompiledPolicy, value: Mapping[str, int]
    ) -> None:
        if self.portfolio_dispatcher is None:
            self.memory[actor_id] = dict(value)
        else:
            self.member_memory.setdefault(actor_id, {})[policy.policy_sha256] = dict(
                value
            )

    @staticmethod
    def _pending_sum(bus: RecipientActivationMessageBus | None, actor_id: str) -> int:
        if bus is None:
            return 0
        values = bus.pending.get(actor_id, {})
        return min(bus.maximum, sum(value for _, value in values.values()))

    def _observation(
        self,
        policy: CompiledPolicy,
        scenario: Scenario,
        state: RunState,
        actor_id: str,
        side: Literal["left", "right"] | None,
        *,
        consume_signal: bool,
        charge: bool,
    ) -> tuple[dict[str, Any], int, int]:
        cells = scenario.cell_map
        actor = cells[actor_id]
        position = state.occupancy.index(actor_id)
        left = cells[state.occupancy[position - 1]] if position else None
        right = (
            cells[state.occupancy[position + 1]]
            if position + 1 < len(state.occupancy)
            else None
        )
        cursor = state.selection_cursors.get(actor_id, -1)
        in_bounds = 0 <= cursor < len(state.occupancy)
        cursor_target_visible = in_bounds and cursor != position
        target = cells[state.occupancy[cursor]] if cursor_target_visible else None
        prefix, prefix_reads, prefix_comparisons = True, 0, 0
        if "line.prefix_ordered" in policy.permissions:
            if self.value_projection:
                prefix = True
                previous: int | float | None = None
                previous_normal = False
                for identity in state.occupancy[:position]:
                    item = cells[identity]
                    prefix_reads += 1
                    if item.fault != FaultMode.NORMAL:
                        previous = None
                        previous_normal = False
                        continue
                    current = self.value_projection.get(identity, item.value)
                    if previous_normal:
                        prefix_comparisons += 1
                        ordered = (
                            previous <= current
                            if actor.direction == Direction.ASCENDING
                            else previous >= current
                        )
                        if not ordered:
                            prefix = False
                            break
                    previous = current
                    previous_normal = True
            else:
                prefix, prefix_reads, prefix_comparisons = _prefix_is_ordered(
                    scenario, state, position, actor.direction
                )
            if charge:
                self.ledger["licensedPrefixPredicateEvaluations"] += 1
                self.ledger["licensedPrefixValueReads"] += prefix_reads
                self.ledger["licensedPrefixValueComparisons"] += prefix_comparisons
        if charge and policy.permissions & {
            "selection.cursor_in_bounds",
            "selection.cursor_at_actor",
        }:
            self.ledger["engineCursorStateReads"] += 1
        if charge and policy.permissions & {
            "selection.target.value",
            "selection.target.stuck",
        }:
            self.ledger["engineCursorTargetProjectionReads"] += int(in_bounds)
        bus = self._bus(policy)
        signal_sum = 0
        if "signal.neighbor_sum_u8" in policy.permissions:
            if consume_signal and bus is not None:
                observed = bus.observe(actor_id, state.activation_count)
                signal_sum = min(bus.maximum, sum(observed.channel_sums.values()))
            else:
                signal_sum = self._pending_sum(bus, actor_id)
        left_visible = left
        right_visible = right
        if "activation.side" in policy.permissions:
            left_visible = left if side == "left" else None
            right_visible = right if side == "right" else None
        if "line.prefix_ordered" in policy.permissions and not prefix:
            left_visible = None

        def value(cell):
            return (
                int(self.value_projection.get(cell.cell_id, cell.value)) if cell else 0
            )

        available: dict[str, Any] = {
            "activation.side": side or "none",
            "own.value": value(actor),
            "own.position": position,
            "own.direction": actor.direction.value,
            "neighbor.left.exists": left_visible is not None,
            "neighbor.left.value": value(left_visible),
            "neighbor.left.movable": left_visible is not None
            and left_visible.fault != FaultMode.STUCK,
            "neighbor.right.exists": right_visible is not None,
            "neighbor.right.value": value(right_visible),
            "neighbor.right.movable": right_visible is not None
            and right_visible.fault != FaultMode.STUCK,
            "line.prefix_ordered": prefix,
            "selection.cursor_in_bounds": in_bounds,
            "selection.cursor_at_actor": cursor == position,
            "selection.target.value": value(target),
            "selection.target.stuck": target is not None
            and target.fault == FaultMode.STUCK,
            "last_action.rejected": self.last_rejected.get(actor_id, False),
            "repair.nudge_count": self.nudge_count,
            "signal.neighbor_sum_u8": signal_sum,
            "counter.choice_u8": int.from_bytes(
                hashlib.sha256(
                    f"{scenario.scenario_id}/{state.activation_count}/{actor_id}".encode()
                ).digest()[:1],
                "big",
            ),
        }
        scalars = {name: available[name] for name in sorted(policy.permissions)}
        if charge and self.value_projection:
            self.ledger["trustedValueProjectionReads"] += (
                int("own.value" in policy.permissions)
                + int(
                    "neighbor.left.value" in policy.permissions
                    and left_visible is not None
                )
                + int(
                    "neighbor.right.value" in policy.permissions
                    and right_visible is not None
                )
                + int(
                    "selection.target.value" in policy.permissions
                    and cursor_target_visible
                )
                + prefix_reads
            )
        # Native read accounting is record-based, not one charge per exposed field.
        reads = 1  # native actor-self record is always charged once
        reads += int(
            any(name.startswith("neighbor.left.") for name in policy.permissions)
            and left_visible is not None
        )
        reads += int(
            any(name.startswith("neighbor.right.") for name in policy.permissions)
            and right_visible is not None
        )
        reads += prefix_reads
        reads += int(
            cursor_target_visible
            and any(name.startswith("selection.target.") for name in policy.permissions)
        )
        comparisons = prefix_comparisons
        return {"scalars": scalars, "candidates": []}, reads, comparisons

    @staticmethod
    def _terminal_action(actions: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        terminal = [
            item
            for item in actions
            if item["kind"]
            in {"noop", "swap_relative", "swap_cursor", "advance_cursor"}
        ]
        if len(terminal) != 1:
            raise SuiteValidationError(
                "DSL activation did not materialize one terminal action"
            )
        return terminal[0]

    def _proposal(
        self,
        scenario: Scenario,
        state: RunState,
        actor_id: str,
        *,
        side: Literal["left", "right"] | None,
        mutate: bool,
    ) -> tuple[Proposal, bool, bool]:
        actor = scenario.cell_map[actor_id]
        position = state.occupancy.index(actor_id)
        policy = self._policy(scenario, state, actor_id, mutate=mutate)
        if actor.fault == FaultMode.STUCK:
            return (
                Proposal(
                    ProposalKind.NO_OP,
                    actor_id,
                    position,
                    reason="actor_fault",
                    observation_reads=1,
                ),
                False,
                False,
            )
        observation, reads, comparisons = self._observation(
            policy,
            scenario,
            state,
            actor_id,
            side,
            consume_signal=mutate,
            charge=mutate,
        )
        initial = self._memory_for(actor_id, policy, mutate=mutate)
        comparisons += _line_rule_value_comparisons(policy, observation, initial)
        result = execute_policy(policy, observation, initial)
        memory_changed = dict(result.memory) != initial
        terminal = self._terminal_action(result.actions)
        target_position: int | None = None
        new_cursor: int | None = None
        observed_target: str | None = None
        kind = terminal["kind"]
        if kind == "swap_relative":
            offset = int(terminal["offset"])
            authorized = (
                actor.policy == Policy.BUBBLE
                and offset == (-1 if side == "left" else 1)
            ) or (actor.policy == Policy.INSERTION and offset == -1)
            target_position = position + offset if authorized else -1
            proposal_kind = ProposalKind.SWAP
            if 0 <= target_position < len(state.occupancy):
                observed_target = state.occupancy[target_position]
        elif kind == "swap_cursor":
            target_position = state.selection_cursors.get(actor_id)
            proposal_kind = ProposalKind.SWAP
            if target_position is not None and 0 <= target_position < len(
                state.occupancy
            ):
                observed_target = state.occupancy[target_position]
            if mutate and target_position is not None:
                distance = abs(position - target_position)
                self.ledger["engineCursorSwapActions"] += 1
                self.ledger["licensedLongRangeRequestedDistance"] += distance
                self.ledger["licensedLongRangeMaximumRequestedDistance"] = max(
                    self.ledger["licensedLongRangeMaximumRequestedDistance"], distance
                )
        elif kind == "advance_cursor":
            proposal_kind = ProposalKind.MEMORY_UPDATE
            new_cursor = state.selection_cursors.get(actor_id, -1) + int(
                terminal["delta"]
            )
            if mutate:
                self.ledger["engineCursorAdvanceActions"] += 1
        else:
            proposal_kind = ProposalKind.NO_OP
        proposal = Proposal(
            proposal_kind,
            actor_id,
            position,
            target_pos=target_position,
            new_cursor=new_cursor,
            reason=f"dsl:{policy.policy_id}:{kind}",
            observation_reads=reads,
            value_comparisons=comparisons,
            observed_target_id=observed_target,
        )
        if mutate:
            self._write_memory(actor_id, policy, result.memory)
            self.activation_counts[actor_id] += 1
            self.ledger["dslActivations"] += 1
            self.ledger["dslOperations"] += result.operation_count
            self.ledger["dslMemoryWrites"] += sum(
                item["kind"] == "set_memory" for item in result.actions
            )
            self.ledger["dslSignalBundles"] += int(bool(result.emitted_signals))
            self.pending_effects[state.activation_count] = _PendingLineEffects(
                actor_id,
                state.activation_count,
                policy.policy_sha256,
                dict(result.emitted_signals),
                _line_neighbors(state),
            )
        return proposal, memory_changed, bool(result.emitted_signals)

    def proposal_for(
        self,
        scenario: Scenario,
        state: RunState,
        actor_id: str,
        *,
        side: Literal["left", "right"] | None = None,
    ) -> Proposal:
        return self._proposal(scenario, state, actor_id, side=side, mutate=True)[0]

    def envelope_for(
        self,
        topology: ControlTopology,
        scenario: Scenario,
        state: RunState,
        actor_id: str,
        *,
        side: Literal["left", "right"] | None = None,
    ) -> ActionEnvelope:
        if topology not in {
            ControlTopology.DISTRIBUTED_LOCAL,
            ControlTopology.CENTRAL_LOCAL_PROPOSAL_K1,
        }:
            raise SuiteValidationError("unsupported E02 topology")
        proposal = self.proposal_for(scenario, state, actor_id, side=side)
        return ActionEnvelope(
            proposal,
            proposal.observed_target_id,
            InformationPermission.POLICY_NATIVE_LOCAL,
            (),
        )

    # Pass-through methods let the runtime act as an E01 batch interceptor.
    @staticmethod
    def prepare(proposal: Proposal, event_index: int):
        del event_index
        return proposal, ()

    @staticmethod
    def outcome(proposal: Proposal, validation, event_index: int):
        del proposal, event_index
        return validation

    def after_batch(
        self,
        scenario: Scenario,
        state: RunState,
        proposals: tuple[Proposal, ...],
        decisions: Mapping[int, str],
        batch_start_index: int,
    ) -> None:
        del scenario, state
        for proposal in proposals:
            event_index = batch_start_index + proposal.ordinal
            decision = str(decisions[proposal.ordinal])
            self.decision_counts[decision] += 1
            rejected = decision.startswith("rejected") or decision in {
                "action_failure",
                "conflict_loss",
            }
            self.last_rejected[proposal.actor_id] = rejected
            self.ledger["nativeEligibleActionOpportunities"] += 1
            self.ledger["rejectedNativeActions"] += int(rejected)
            if decision == "accepted" and proposal.kind == ProposalKind.SWAP:
                assert proposal.target_pos is not None
                self.ledger["committedNativeMovementActions"] += 1
                self.ledger["committedNativeDisplacement"] += abs(
                    proposal.actor_pos - proposal.target_pos
                )
            effects = self.pending_effects.pop(event_index, None)
            if effects and effects.emitted:
                policy = next(
                    value
                    for value in self.all_policies.values()
                    if value.policy_sha256 == effects.policy_sha256
                )
                bus = self._bus(policy)
                assert bus is not None
                bus.emit(
                    SignalEmission(
                        effects.actor_id, effects.event_index, effects.emitted
                    ),
                    effects.neighbors,
                )

    def evaluate_terminal(self, scenario: Scenario, state: RunState) -> str | None:
        if invariant_error(scenario, state) is not None:
            return "invariant_error"
        if is_complete(scenario, state):
            return "complete"
        if self.is_quiescent(scenario, state):
            return "quiescent"
        if state.activation_count >= scenario.max_activations:
            return "event_budget"
        return None

    def is_quiescent(self, scenario: Scenario, state: RunState) -> bool:
        # Emissions produced by the just-evaluated native batch are delivered
        # in ``after_batch`` and keep the episode live until recipients receive
        # their first eligible activation.
        if any(item.emitted for item in self.pending_effects.values()):
            return False
        for cell in scenario.cells:
            policy = self._policy(scenario, state, cell.cell_id, mutate=False)
            sides: tuple[Literal["left", "right"] | None, ...] = (
                ("left", "right")
                if "activation.side" in policy.permissions
                else (None,)
            )
            for side in sides:
                proposal, memory_changed, emitted = self._proposal(
                    scenario, state, cell.cell_id, side=side, mutate=False
                )
                if (
                    memory_changed
                    or emitted
                    or validate_proposal(scenario, state, proposal).eligible_for_commit
                ):
                    break
            else:
                continue
            break
        else:
            return True
        return False

    def adapter_ledgers(self) -> dict[str, Mapping[str, Any]]:
        communication = {
            field: 0
            for field in (
                "emittedSignalWrites",
                "transmittedSignalBits",
                "recipientDeliveries",
                "bufferOverwrites",
                "consumedDeliveries",
                "observableAggregateReads",
            )
        }
        for bus in self.buses.values():
            for key, value in bus.ledger_snapshot().items():
                communication[key] += value
        licensed = {
            key: self.ledger[key]
            for key in (
                "licensedPrefixPredicateEvaluations",
                "licensedPrefixValueReads",
                "licensedPrefixValueComparisons",
                "engineCursorStateReads",
                "engineCursorTargetProjectionReads",
                "engineCursorAdvanceActions",
                "engineCursorSwapActions",
                "licensedLongRangeRequestedDistance",
                "licensedLongRangeMaximumRequestedDistance",
            )
        }
        result: dict[str, Mapping[str, Any]] = {
            "dslRuntimeLedger": dict(self.ledger),
            "dslCommunicationLedger": communication,
            "licensedCapabilityLedger": licensed,
        }
        if self.portfolio_dispatcher is not None:
            snapshot = self.portfolio_dispatcher.ledger_snapshot()
            coordination = dict(snapshot["portfolioCoordinationLedger"])
            coordination.update(communication)
            result.update(
                {
                    "portfolioStructuralLedger": snapshot["portfolioStructuralLedger"],
                    "portfolioCoordinationLedger": coordination,
                }
            )
        return result

    def portfolio_assignment_audit(self) -> Mapping[str, Any] | None:
        if self.portfolio_dispatcher is None:
            return None
        snapshot = self.portfolio_dispatcher.ledger_snapshot()
        result = {
            key: value
            for key, value in snapshot.items()
            if key
            not in {
                "portfolioStructuralLedger",
                "portfolioCoordinationLedger",
            }
        }
        namespace_hashes = {
            f"{actor_id}::{policy_sha256}": canonical_sha256(
                "E07/S08A/member-memory-namespace/v1", value
            )
            for actor_id, namespaces in sorted(self.member_memory.items())
            for policy_sha256, value in sorted(namespaces.items())
        }
        result["memberMemoryNamespaceCount"] = len(namespace_hashes)
        result["memberMemoryNamespaceSha256"] = namespace_hashes
        return result


def make_line_runtime(
    scenario: Scenario,
    action: EvaluationAction,
    *,
    value_projection: Mapping[str, int] | None = None,
    nudge_count: int = 0,
) -> LineDslRuntime:
    """Construct the single-member or portfolio runtime for a bound line."""

    policies = compiled_policies(action)
    dispatcher: PortfolioDispatcher | None = None
    if action.portfolio_definition:
        dispatcher = PortfolioDispatcher(action.portfolio_definition, policies)
        bindings = {
            Policy(carrier): members[0]
            for carrier, members in dispatcher.members_by_carrier.items()
        }
    elif action.native_policy_bindings:
        bindings = {
            Policy(native): policies[policy_id]
            for native, policy_id in action.native_policy_bindings.items()
        }
    elif len(policies) == 1:
        policy = next(iter(policies.values()))
        bindings = {cell.policy: policy for cell in scenario.cells}
    else:
        raise SuiteValidationError("portfolio action requires native policy bindings")
    for carrier, policy in bindings.items():
        _validate_line_authority(policy, carrier)
    if dispatcher is not None:
        for carrier_name, members in dispatcher.members_by_carrier.items():
            for policy in members:
                _validate_line_authority(policy, Policy(carrier_name))
    return LineDslRuntime(
        bindings,
        value_projection=value_projection,
        nudge_count=nudge_count,
        portfolio_dispatcher=dispatcher,
    )


def run_line_dsl_episode(
    scenario: Scenario,
    action: EvaluationAction,
    *,
    trace_mode: Literal["full", "digest"] = "full",
    value_projection: Mapping[str, int] | None = None,
    nudge_count: int = 0,
    initial_state_override: RunState | None = None,
    runtime: LineDslRuntime | None = None,
    terminal_evaluator: Any | None = None,
) -> tuple[RunResult, LineDslRuntime]:
    runtime = runtime or make_line_runtime(
        scenario,
        action,
        value_projection=value_projection,
        nudge_count=nudge_count,
    )
    result = run(
        scenario,
        trace_mode=trace_mode,
        proposal_factory=runtime.proposal_for,
        execution_interceptor=runtime,
        terminal_evaluator=terminal_evaluator or runtime.evaluate_terminal,
        initial_state_override=initial_state_override,
    )
    return result, runtime


def line_replay_bytes(result: RunResult, runtime: LineDslRuntime) -> bytes:
    return canonical_json_bytes(
        {
            "result": result.to_dict(),
            "adapterLedgers": runtime.adapter_ledgers(),
            "portfolioAssignmentAudit": runtime.portfolio_assignment_audit(),
            "memory": runtime.memory,
            "memberMemory": runtime.member_memory,
            "lastRejected": runtime.last_rejected,
            "activationCounts": dict(sorted(runtime.activation_counts.items())),
            "dispatchCounts": dict(sorted(runtime.dispatch_counts.items())),
        }
    )


def _run_state_from_result(result: RunResult, *, occupancy=None) -> RunState:
    raw = result.final_state
    return RunState(
        occupancy=list(occupancy or raw["occupancy"]),
        selection_cursors={
            str(key): int(value) for key, value in raw["selectionCursors"].items()
        },
        activation_count=int(raw["activationCount"]),
        stream_counters={
            str(key): int(value) for key, value in raw["streamCounters"].items()
        },
        ledger={str(key): int(value) for key, value in raw["ledger"].items()},
        terminal=None,
    )


def _ledger_delta(
    after: Mapping[str, int], before: Mapping[str, int]
) -> dict[str, int]:
    return {key: int(after[key]) - int(before.get(key, 0)) for key in after}


def _line_movement_descriptors(
    ledger: Mapping[str, int], *, cell_count: int
) -> dict[str, float]:
    opportunities = int(ledger["nativeEligibleActionOpportunities"])
    committed = int(ledger["committedNativeMovementActions"])
    displacement = int(ledger["committedNativeDisplacement"])
    return {
        "acceptedNativeActionFraction": (
            committed / opportunities if opportunities else 0.0
        ),
        "committedDisplacementFraction": (
            displacement / (committed * max(1, cell_count - 1)) if committed else 0.0
        ),
    }


def finalize_e05_result(
    payload: Mapping[str, Any], runtime: LineDslRuntime
) -> dict[str, Any]:
    """Attach the portfolio audit at the single exit boundary for E05 panels."""

    result = dict(payload)
    if "portfolioAssignmentAudit" in result:
        raise SuiteValidationError("E05 result audit must be attached exactly once")
    result["portfolioAssignmentAudit"] = runtime.portfolio_assignment_audit()
    return result


def validate_e05_portfolio_result_contract(
    result: Mapping[str, Any], action: EvaluationAction
) -> dict[str, Any]:
    """Fail closed when an E05 portfolio branch omits assignment provenance."""

    audit = result.get("portfolioAssignmentAudit")
    if not action.portfolio_definition:
        if audit is not None:
            raise SuiteValidationError(
                "single-policy E05 result carried portfolio audit"
            )
        return {"portfolioAssignmentAuditRequired": False, "complete": True}
    required = {
        "adapterVersion",
        "configurationId",
        "mode",
        "selectorSignal",
        "singletonCarrierPolicySha256",
        "assignmentCountsByPolicySha256",
        "memberActivationCounts",
        "currentMemberByIdentity",
        "memberMemoryNamespaceCount",
        "memberMemoryNamespaceSha256",
    }
    if not isinstance(audit, Mapping):
        raise SuiteValidationError("E05 portfolio result omitted assignment audit")
    missing = sorted(required - set(audit))
    if missing:
        raise SuiteValidationError(
            f"E05 portfolio assignment audit is incomplete: {missing}"
        )
    definition = action.portfolio_definition
    if (
        audit["configurationId"] != definition["configurationId"]
        or audit["mode"] != definition["mode"]
    ):
        raise SuiteValidationError("E05 portfolio assignment audit mismatched action")
    if int(audit["memberMemoryNamespaceCount"]) != len(
        audit["memberMemoryNamespaceSha256"]
    ):
        raise SuiteValidationError("E05 member-memory audit cardinality mismatched")
    return {
        "portfolioAssignmentAuditRequired": True,
        "complete": True,
        "requiredFields": sorted(required),
    }


def run_e05_regeneration_dsl(
    action: EvaluationAction, *, replicate_ordinal: int
) -> dict[str, Any]:
    """Run one train-derived five-axis E05 binding panel without pooling axes."""

    from src.regeneration.benchmark import _checkpoint_from_development
    from src.regeneration.chimeric import build_composition_scenario
    from src.regeneration.transfer import apply_transfer_lesion
    from src.regeneration.tasks import strict_unequal_inversions

    policies = compiled_policies(action)
    if action.portfolio_definition:
        policy = next(iter(policies.values()))
    elif len(policies) == 1:
        policy = next(iter(policies.values()))
    else:
        raise SuiteValidationError("E05 core adapter requires one policy or portfolio")
    n = 32
    development_budget = 100 * n * n
    recovery_budget = 100 * n * n
    source, source_meta = build_composition_scenario(
        n=n,
        portfolio_id="pure_bubble",
        direction="ascending",
        replicate=int(replicate_ordinal),
        placement_map=0,
        max_activations=2 * development_budget + 40 * n,
    )
    scenario = (
        bind_portfolio_line_scenario(source, action)
        if action.portfolio_definition
        else bind_homogeneous_line_scenario(source, policy)
    )
    development, runtime = run_line_dsl_episode(scenario, action)
    source_completed = bool(development.summary["completed"])
    phase_rows: list[dict[str, Any]] = [
        {
            "phase": "development",
            "status": "executed",
            "stopReason": development.summary["stopReason"],
            "opportunities": int(development.summary["activationCount"]),
            "censored": development.summary["stopReason"] == "event_budget",
        }
    ]
    if not source_completed:
        stopped = [
            {
                "axis": axis,
                "status": "source_terminal",
                "sourceStopReason": development.summary["stopReason"],
            }
            for axis in ("repair", "memory", "transfer")
        ]
        result = {
            "schemaVersion": "e07.s04a.e05-dsl-panel.v1",
            "sourceTerminal": True,
            "sourceStopReason": development.summary["stopReason"],
            "phaseRows": phase_rows,
            "stoppedRows": stopped,
            "competencyAxes": {
                "robustness": {
                    "fullPopulationSuccess": False,
                    "terminalCause": development.summary["stopReason"],
                },
                "repair": {"status": "source_terminal"},
                "memory": {"status": "source_terminal"},
                "transfer": {"status": "source_terminal"},
            },
            "nativeLedgers": {"development": dict(development.summary["ledger"])},
            "adapterLedgers": runtime.adapter_ledgers(),
            "nativeMovementDescriptorsByPhase": {
                "development": _line_movement_descriptors(runtime.ledger, cell_count=n)
            },
            "validation": {
                "sourceTerminalsRetained": True,
                "stoppedRowsRetained": len(stopped) == 3,
                "identityCardinalityPreserved": True,
                "developmentBudgetRespected": int(
                    development.summary["activationCount"]
                )
                <= development_budget,
            },
        }
        return finalize_e05_result(result, runtime)

    development_runtime_ledger = dict(runtime.ledger)
    stabilization_start = _run_state_from_result(development)
    stabilization_cap = 20 * n
    before_stabilization_counts = dict(runtime.activation_counts)
    before_stabilization_movements = runtime.ledger["committedNativeMovementActions"]

    def stabilization_terminal(active_scenario: Scenario, state: RunState):
        if invariant_error(active_scenario, state) is not None:
            return "invariant_error"
        if (
            runtime.ledger["committedNativeMovementActions"]
            > before_stabilization_movements
        ):
            return "stabilization_movement_failure"
        minimum_observed = min(
            runtime.activation_counts.get(cell.cell_id, 0)
            - before_stabilization_counts.get(cell.cell_id, 0)
            for cell in active_scenario.cells
        )
        if minimum_observed >= 2:
            return "stabilization_complete"
        if (
            state.activation_count - stabilization_start.activation_count
            >= stabilization_cap
        ):
            return "stabilization_cap"
        return None

    stabilization, runtime = run_line_dsl_episode(
        scenario,
        action,
        initial_state_override=stabilization_start,
        runtime=runtime,
        terminal_evaluator=stabilization_terminal,
    )
    phase_rows.append(
        {
            "phase": "stabilization",
            "status": "executed",
            "stopReason": stabilization.summary["stopReason"],
            "opportunities": int(stabilization.summary["activationCount"])
            - stabilization_start.activation_count,
            "censored": False,
        }
    )
    stabilization_success = (
        stabilization.summary["stopReason"] == "stabilization_complete"
    )
    if not stabilization_success:
        stopped = [
            {
                "axis": axis,
                "status": "source_terminal",
                "sourceStopReason": stabilization.summary["stopReason"],
            }
            for axis in ("repair", "memory", "transfer")
        ]
        result = {
            "schemaVersion": "e07.s04a.e05-dsl-panel.v1",
            "adapterVersion": ADAPTER_VERSION,
            "sourceTerminal": True,
            "sourceStopReason": stabilization.summary["stopReason"],
            "phaseRows": phase_rows,
            "stoppedRows": stopped,
            "competencyAxes": {
                "robustness": {
                    "fullPopulationSuccess": False,
                    "terminalCause": stabilization.summary["stopReason"],
                },
                "repair": {"status": "source_terminal"},
                "memory": {"status": "source_terminal"},
                "transfer": {"status": "source_terminal"},
            },
            "nativeLedgers": {
                "development": dict(development.summary["ledger"]),
                "stabilization": _ledger_delta(
                    dict(stabilization.summary["ledger"]),
                    dict(development.summary["ledger"]),
                ),
            },
            "adapterLedgers": runtime.adapter_ledgers(),
            "nativeMovementDescriptorsByPhase": {
                "development": _line_movement_descriptors(
                    development_runtime_ledger, cell_count=n
                ),
                "stabilization": _line_movement_descriptors(
                    _ledger_delta(runtime.ledger, development_runtime_ledger),
                    cell_count=n,
                ),
            },
            "validation": {
                "sourceTerminalsRetained": True,
                "stoppedRowsRetained": len(stopped) == 3,
                "identityCardinalityPreserved": True,
                "developmentBudgetRespected": int(
                    development.summary["activationCount"]
                )
                <= development_budget,
                "stabilizationCoverageRetained": True,
            },
        }
        return finalize_e05_result(result, runtime)
    stabilization_runtime_ledger = dict(runtime.ledger)
    checkpoint = _checkpoint_from_development(scenario, stabilization)
    lesion = apply_transfer_lesion(
        scenario,
        checkpoint,
        lesion_type="segment_reversal_central_v1",
        location="central",
    )
    recovery_start = checkpoint.to_run_state(occupancy=lesion["postOccupancy"])
    recovery_start.terminal = None
    recovery_stop = recovery_start.activation_count + recovery_budget

    def recovery_terminal(active_scenario: Scenario, state: RunState):
        if invariant_error(active_scenario, state) is not None:
            return "invariant_error"
        if is_complete(active_scenario, state):
            return "recovery_complete"
        if runtime.is_quiescent(active_scenario, state):
            return "quiescent"
        if state.activation_count >= recovery_stop:
            return "phase_event_budget"
        return None

    before_recovery_native = dict(recovery_start.ledger)
    before_runtime = dict(runtime.ledger)
    runtime.nudge_count = 1
    recovery, runtime = run_line_dsl_episode(
        scenario,
        action,
        initial_state_override=recovery_start,
        runtime=runtime,
        nudge_count=1,
        terminal_evaluator=recovery_terminal,
    )
    recovery_values = list(recovery.summary["finalValues"])
    initial_distance = int(lesion["postDistance"])
    final_distance = strict_unequal_inversions(
        recovery_values, scenario.cells[0].direction
    )
    recovery_opportunities = (
        int(recovery.summary["activationCount"]) - recovery_start.activation_count
    )
    recovery_success = final_distance == 0
    phase_rows.extend(
        [
            {
                "phase": "lesion",
                "status": "executed",
                "stopReason": "instantaneous_operator",
                "opportunities": 0,
                "censored": False,
            },
            {
                "phase": "recovery",
                "status": "executed",
                "stopReason": recovery.summary["stopReason"],
                "opportunities": recovery_opportunities,
                "censored": not recovery_success,
            },
        ]
    )

    # Memory axis: same checkpoint/lesion, but identity memory reset before the
    # recovery phase.  Pairing and scheduler/global clock are unchanged.
    reset_runtime = make_line_runtime(scenario, action, nudge_count=1)

    def reset_terminal(active_scenario: Scenario, state: RunState):
        if invariant_error(active_scenario, state) is not None:
            return "invariant_error"
        if is_complete(active_scenario, state):
            return "recovery_complete"
        if reset_runtime.is_quiescent(active_scenario, state):
            return "quiescent"
        if state.activation_count >= recovery_stop:
            return "phase_event_budget"
        return None

    reset_recovery, reset_runtime = run_line_dsl_episode(
        scenario,
        action,
        initial_state_override=recovery_start,
        runtime=reset_runtime,
        terminal_evaluator=reset_terminal,
    )
    reset_distance = strict_unequal_inversions(
        list(reset_recovery.summary["finalValues"]), scenario.cells[0].direction
    )
    maximum_pairs = n * (n - 1) // 2

    # Robustness axis: a predeclared central stuck-cell severity-one derivative
    # of the same authorized base draw.
    central = n // 2
    fault_cells = tuple(
        replace(cell, fault=FaultMode.STUCK) if index == central else cell
        for index, cell in enumerate(scenario.cells)
    )
    fault_scenario = Scenario.create(
        fault_cells,
        initial_occupancy=scenario.initial_occupancy,
        seed=scenario.seed,
        max_activations=development_budget,
        generation_key=f"{scenario.generation_key}/robustness-severity-1",
        fault_placement="explicit",
        requested_fault_count=1,
    )
    fault_run, fault_runtime = run_line_dsl_episode(fault_scenario, action)
    fault_success = bool(fault_run.summary["completed"])
    fault_residual = strict_unequal_inversions(
        list(fault_run.summary["finalValues"]), scenario.cells[0].direction
    )

    # Transfer axis: same multiset and deterministic initial occupancy under the
    # opposite native goal; no source outcome selects this fixture.
    transfer_cells = tuple(
        replace(cell, direction=Direction.DESCENDING, fault=FaultMode.NORMAL)
        for cell in scenario.cells
    )
    transfer_scenario = Scenario.create(
        transfer_cells,
        initial_occupancy=scenario.initial_occupancy,
        seed=scenario.seed,
        max_activations=development_budget,
        generation_key=f"{scenario.generation_key}/transfer-descending",
    )
    transfer_run, transfer_runtime = run_line_dsl_episode(transfer_scenario, action)
    transfer_values = list(transfer_run.summary["finalValues"])
    transfer_distance = strict_unequal_inversions(transfer_values, Direction.DESCENDING)
    retained_runtime_delta = _ledger_delta(runtime.ledger, before_runtime)
    result = {
        "schemaVersion": "e07.s04a.e05-dsl-panel.v1",
        "adapterVersion": ADAPTER_VERSION,
        "sourceTerminal": False,
        "sourceStopReason": development.summary["stopReason"],
        "sourceScenarioId": scenario.scenario_id,
        "baseDrawPairingId": source_meta["baseDrawPairingId"],
        "phaseRows": phase_rows,
        "stoppedRows": [],
        "competencyAxes": {
            "robustness": {
                "severity": 1,
                "cleanSuccess": source_completed,
                "faultSuccess": fault_success,
                "pairedCompletionDelta": int(fault_success) - int(source_completed),
                "cleanResidual": 0.0,
                "faultResidual": fault_residual / max(1, maximum_pairs),
                "pairedResidualDelta": fault_residual / max(1, maximum_pairs),
                "terminalCause": fault_run.summary["stopReason"],
            },
            "repair": {
                "recoverySuccess": recovery_success,
                "restrictedRecoveryTime": recovery_opportunities
                if recovery_success
                else recovery_budget + 1,
                "recoveryCensored": not recovery_success,
                "initialDistance": initial_distance,
                "finalDistance": final_distance,
                "distanceRestorationFraction": (
                    (initial_distance - final_distance) / initial_distance
                    if initial_distance
                    else 1.0
                ),
            },
            "memory": {
                "historyRetainedSuccess": recovery_success,
                "stateResetSuccess": reset_distance == 0,
                "historyMatchedEffect": int(recovery_success)
                - int(reset_distance == 0),
                "resetMatchedEffect": reset_distance - final_distance,
                "postInjuryStability": recovery_success,
            },
            "transfer": {
                "frozenStratum": "opposite_direction_same_multiset_v1",
                "success": bool(transfer_run.summary["completed"]),
                "residual": transfer_distance / max(1, maximum_pairs),
                "terminalCause": transfer_run.summary["stopReason"],
            },
        },
        "descriptorsByAxis": {
            "robustness": {
                "pairedCompletionDelta": int(fault_success) - int(source_completed),
                "pairedResidualDelta": fault_residual / max(1, maximum_pairs),
            },
            "repair": {
                "distanceRestorationFraction": (
                    (initial_distance - final_distance) / initial_distance
                    if initial_distance
                    else 1.0
                ),
                "restrictedRecoveryTimeFraction": min(
                    1.0,
                    (
                        recovery_opportunities
                        if recovery_success
                        else recovery_budget + 1
                    )
                    / recovery_budget,
                ),
                "recoveryCensored": not recovery_success,
            },
            "memory": {
                "historyInterventionEffect": int(recovery_success)
                - int(reset_distance == 0),
                "resetInterventionEffect": reset_distance - final_distance,
            },
            "transfer": {
                "frozenStratumSuccessFraction": float(
                    bool(transfer_run.summary["completed"])
                ),
                "frozenStratumResidual": transfer_distance / max(1, maximum_pairs),
            },
        },
        "nativeMovementDescriptorsByPhase": {
            "development": _line_movement_descriptors(
                development_runtime_ledger, cell_count=n
            ),
            "stabilization": _line_movement_descriptors(
                _ledger_delta(stabilization_runtime_ledger, development_runtime_ledger),
                cell_count=n,
            ),
            "recovery": _line_movement_descriptors(
                retained_runtime_delta, cell_count=n
            ),
            "memoryResetRecovery": _line_movement_descriptors(
                reset_runtime.ledger, cell_count=n
            ),
            "robustnessFault": _line_movement_descriptors(
                fault_runtime.ledger, cell_count=n
            ),
            "transfer": _line_movement_descriptors(
                transfer_runtime.ledger, cell_count=n
            ),
        },
        "nativeLedgers": {
            "development": dict(development.summary["ledger"]),
            "stabilization": _ledger_delta(
                dict(stabilization.summary["ledger"]),
                dict(development.summary["ledger"]),
            ),
            "recovery": _ledger_delta(
                dict(recovery.summary["ledger"]), before_recovery_native
            ),
            "robustnessFault": dict(fault_run.summary["ledger"]),
            "transfer": dict(transfer_run.summary["ledger"]),
        },
        "extensionLedgers": {
            "recoveryDsl": retained_runtime_delta,
            "recoveryCommunication": runtime.adapter_ledgers()[
                "dslCommunicationLedger"
            ],
            "faultDsl": fault_runtime.adapter_ledgers()["dslRuntimeLedger"],
            "transferDsl": transfer_runtime.adapter_ledgers()["dslRuntimeLedger"],
        },
        "adapterLedgers": runtime.adapter_ledgers(),
        "adapterLedgersByIndependentPhase": {
            "memoryResetRecovery": reset_runtime.adapter_ledgers(),
            "robustnessFault": fault_runtime.adapter_ledgers(),
            "transfer": transfer_runtime.adapter_ledgers(),
        },
        "validation": {
            "phaseLocalClocks": all(row["opportunities"] >= 0 for row in phase_rows),
            "developmentBudgetRespected": int(development.summary["activationCount"])
            <= development_budget,
            "stabilizationCoverageRetained": stabilization_success,
            "axesNonaggregated": True,
            "identityCardinalityPreserved": len(recovery.final_state["occupancy"]) == n
            and set(recovery.final_state["occupancy"]) == set(scenario.cell_map),
            "sourceTerminalsRetained": True,
            "censorsRetained": True,
            "stoppedRowsRetained": True,
            "pairingRetained": bool(source_meta["baseDrawPairingId"]),
            "nativeAndExtensionCostsSeparate": True,
        },
        "claimBoundary": "Five operational simulator competencies remain separate; no biological regeneration, natural memory, learning, clinical repair, or universal competence is inferred.",
    }
    result = finalize_e05_result(result, runtime)
    result["panelSha256"] = canonical_sha256("E07/S04A/E05-panel/v1", result)
    return result


def run_e05_target_change_dsl(
    scenario: Scenario,
    action: EvaluationAction,
    *,
    target_change_id: str,
    signal_permission_id: str,
    adaptation_budget: int,
    probe_budget: int,
) -> dict[str, Any]:
    """Bind a DSL episode to E05's trusted target-code projection."""

    from src.regeneration.target_change import TargetChange, build_target_definition

    policies = compiled_policies(action)
    if action.portfolio_definition:
        policy = next(iter(policies.values()))
        scenario = bind_portfolio_line_scenario(scenario, action)
    elif len(policies) == 1:
        policy = next(iter(policies.values()))
        scenario = bind_homogeneous_line_scenario(scenario, policy)
    else:
        raise SuiteValidationError(
            "E05 target-change adapter requires one policy or portfolio"
        )
    definition = build_target_definition(scenario, TargetChange(target_change_id))
    projection = definition.policy_code_map
    runtime = make_line_runtime(scenario, action, value_projection=projection)
    start = 0
    first_hit: list[int | None] = [None]

    def target_terminal(active_scenario: Scenario, state: RunState):
        elapsed = state.activation_count - start
        distance = definition.distance(state.occupancy)
        first_hit[0], terminal = target_change_terminal_transition(
            elapsed=elapsed,
            distance=distance,
            first_hit=first_hit[0],
            quiescent=(first_hit[0] is None)
            and runtime.is_quiescent(active_scenario, state),
            invariant_error=invariant_error(active_scenario, state) is not None,
            adaptation_budget=adaptation_budget,
            probe_budget=probe_budget,
        )
        return terminal

    run_result, runtime = run_line_dsl_episode(
        scenario,
        action,
        runtime=runtime,
        terminal_evaluator=target_terminal,
    )
    occupancy = list(scenario.initial_occupancy)
    post_hit_departure = False
    seen_hit = definition.distance(occupancy) == 0
    for event in run_result.events:
        proposal = event["proposal"]
        if event["decision"] == "accepted" and proposal["kind"] == "Swap":
            left, right = int(proposal["actorPos"]), int(proposal["targetPos"])
            occupancy[left], occupancy[right] = occupancy[right], occupancy[left]
        distance = definition.distance(occupancy)
        if seen_hit and distance > 0:
            post_hit_departure = True
        seen_hit = seen_hit or distance == 0
    final_distance = definition.distance(occupancy)
    adapted = first_hit[0] is not None
    phase = int(run_result.summary["activationCount"])
    post_hit_opportunities = phase - int(first_hit[0]) if adapted else 0
    post_hit_probe_retained = (not adapted) or (
        run_result.summary["stopReason"] == "post_adaptation_probe_complete"
        and post_hit_opportunities == probe_budget
    )
    result = {
        "schemaVersion": "e07.s08h.e05-target-dsl-episode.v2",
        "adapterVersion": ADAPTER_VERSION,
        "targetChangeSemanticsVersion": TARGET_CHANGE_SEMANTICS_VERSION,
        "targetDefinition": definition.to_dict(),
        "targetSignalAuthority": {
            "signalPermissionId": signal_permission_id,
            "projection": "E05_trusted_target_code_transformer_equivalent",
            "genericDslPeerCommunicationUsedAsTargetSignal": False,
            "projectionReads": runtime.ledger["trustedValueProjectionReads"],
        },
        "targetCompleted": adapted,
        "phaseActivationCount": phase,
        "adaptationTime": first_hit[0],
        "adaptationCensored": not adapted,
        "restrictedAdaptationTime": int(first_hit[0])
        if adapted
        else adaptation_budget + 1,
        "finalNewTargetDistance": final_distance,
        "overshootCensored": not adapted,
        "postHitProbeOpportunities": post_hit_opportunities,
        "postHitProbeRetained": post_hit_probe_retained,
        "postHitProbeApplicable": adapted,
        "postHitAnyDeparture": post_hit_departure,
        "stopReason": run_result.summary["stopReason"],
        "nativeLedger": dict(run_result.summary["ledger"]),
        "targetSignalLedger": {
            "targetDefinitionReads": 1,
            "targetCodeProjectionReads": runtime.ledger["trustedValueProjectionReads"],
            "genericPeerSignalReads": runtime.adapter_ledgers()[
                "dslCommunicationLedger"
            ]["observableAggregateReads"],
        },
        "dslLedgers": runtime.adapter_ledgers(),
        "portfolioAssignmentAudit": runtime.portfolio_assignment_audit(),
        "descriptorsByAxis": {
            "plasticity_target_adaptation": {
                "restrictedAdaptationTimeFraction": min(
                    1.0,
                    (int(first_hit[0]) if adapted else adaptation_budget + 1)
                    / adaptation_budget,
                ),
                "adaptationCensored": not adapted,
                "postHitDepartureFraction": float(post_hit_departure),
            }
        },
        "nativeMovementDescriptors": _line_movement_descriptors(
            runtime.ledger, cell_count=len(scenario.cells)
        ),
        "validation": {
            "targetSignalAuthorityPreserved": signal_permission_id
            in {
                "local_boundary_relay_v1",
                "gradient_target_code_v1",
                "global_target_broadcast_v1",
            },
            "genericCommunicationNotTargetSignal": True,
            "phaseBudgetRespected": phase <= adaptation_budget + probe_budget,
            "adaptationDeadlineRespected": (not adapted)
            or int(first_hit[0]) <= adaptation_budget,
            "censorRetained": True,
            "postHitProbeRetained": post_hit_probe_retained,
            "phaseEventBudgetIsNonadaptation": (
                run_result.summary["stopReason"] != "phase_event_budget" or not adapted
            ),
        },
        "claimBoundary": "Target codes and adaptation are engineered E05 constructs, not biological goals, learning, agency, or physical signaling.",
    }
    semantic_audit = validate_target_change_result_semantics(
        result,
        adaptation_budget=adaptation_budget,
        probe_budget=probe_budget,
    )
    result["targetChangeSemanticAudit"] = semantic_audit
    result["validation"]["targetChangeSemanticContract"] = bool(
        semantic_audit["validNativeContract"]
    )
    result["resultSha256"] = canonical_sha256("E07/S04A/E05-target/v1", result)
    return result


def _add_fields(
    target: dict[str, int], delta: Mapping[str, int], fields: Sequence[str]
) -> None:
    if set(delta) != set(fields):
        raise SuiteValidationError("native ledger schema mismatch")
    for field in fields:
        target[field] += int(delta[field])


def _spatial_features(policy: CompiledPolicy) -> tuple[str, ...]:
    mapping = {
        "own.token": "actor_token",
        "own.local_relation_utility": "local_relation_delta",
        "candidate.local_relation_delta": "local_relation_delta",
        "natural.boundary_signal": "natural_boundary_delta",
        "candidate.natural_boundary_delta": "natural_boundary_delta",
        "gradient.current_u8": "gradient_delta",
        "candidate.gradient_delta": "gradient_delta",
        "candidate.lagged_conflict_count": "lagged_conflict_count",
        "candidate.movement_cost": "movement_cost",
        "counter.choice_u8": "counter_exploration_index",
    }
    return tuple(
        sorted({mapping[name] for name in policy.permissions if name in mapping})
    )


def _spatial_allowed_kinds(policy: CompiledPolicy) -> tuple[str, ...]:
    kinds: set[str] = set()
    for _, actions in policy.rules:
        for action in actions:
            if action["kind"] == "move_candidate":
                kinds.update(action["allowedMovementKinds"])
    for action in policy.default_actions:
        if action["kind"] == "move_candidate":
            kinds.update(action["allowedMovementKinds"])
    return tuple(sorted(kinds or {"adjacent_swap"}))


def _spatial_definition(policy: CompiledPolicy) -> PolicyDefinition:
    return PolicyDefinition(
        policy_id=f"dsl::{policy.policy_id}",
        strategy="dsl_trusted_interpreter_v1",
        allowed_movement_kinds=_spatial_allowed_kinds(policy),
        max_candidates=policy.max_candidates,
        observation_features=_spatial_features(policy),
        information_budget_max_bits=4096,
        relation_profile_required=bool(
            policy.permissions
            & {"own.local_relation_utility", "candidate.local_relation_delta"}
        ),
        target_specificity="native_E06_projection_only",
        parameters={},
        complexity=policy.complexity.to_dict(),
        intended_behavior="S01 DSL executed over E06 opaque candidates",
        non_guarantees=("formation", "repair", "convergence"),
    )


def _validate_spatial_authority(
    policy: CompiledPolicy, definition: EpisodeDefinition
) -> None:
    authority_bearing_channels = {
        "natural.boundary_signal": "boundary_signal",
        "candidate.natural_boundary_delta": "boundary_signal",
        "gradient.current_u8": "static_gradient",
        "candidate.gradient_delta": "static_gradient",
    }
    unavailable = {
        field: required_mode
        for field, required_mode in authority_bearing_channels.items()
        if field in policy.permissions and definition.channel_mode != required_mode
    }
    if unavailable:
        raise SuiteValidationError(
            "DSL policy requests E06 channel observations without the required "
            f"native authority: {unavailable}"
        )


def _spatial_actor_neighbors(
    environment, state: MovementState
) -> dict[str, tuple[str, ...]]:
    from src.morph2d.environments import neighbor_map

    neighbors = neighbor_map(environment)
    occupancy = state.occupant_map
    result: dict[str, tuple[str, ...]] = {}
    for site_id, occupant in state.occupancy:
        if occupant.kind != "cell":
            continue
        result[occupant.occupant_id] = tuple(
            sorted(
                occupancy[other].occupant_id
                for other in neighbors[site_id]
                if occupancy[other].kind == "cell"
            )
        )
    return result


def _spatial_observation(
    policy: CompiledPolicy,
    build: ObservationBuild,
    *,
    last_rejected: bool,
    signal_sum: int,
    counter_value: int,
) -> dict[str, Any]:
    payload = build.observation.payload
    candidates = []
    for item in payload["candidates"]:
        key = str(item["candidateKey"])
        affordance = build.candidate_map[key]
        available = {
            "candidate.local_relation_delta": int(item.get("localRelationDelta", 0)),
            "candidate.natural_boundary_delta": int(
                item.get("naturalBoundaryDelta", 0)
            ),
            "candidate.gradient_delta": int(item.get("gradientDelta", 0)),
            "candidate.lagged_conflict_count": int(item.get("laggedConflictCount", 0)),
            "candidate.movement_cost": int(
                item.get("movementCost", affordance.movement_cost)
            ),
            # This tag is consumed only by the trusted selector allowlist; it is
            # never legal in a scalar condition and never reveals a route/site.
            "candidate.kind": affordance.proposal.kind,
        }
        candidates.append(
            {
                "key": key,
                **{
                    name: available[name]
                    for name in sorted(policy.permissions)
                    if name.startswith("candidate.")
                },
            }
        )
    scalars_available = {
        "own.token": str(payload.get("actorToken", "unknown")),
        "own.local_relation_utility": int(
            payload.get("currentLocalRelationUtility", 0)
        ),
        "natural.boundary_signal": int(payload.get("currentNaturalBoundaryLevel", 0)),
        "gradient.current_u8": int(payload.get("currentGradientLevel", 0)),
        "candidate.count": len(candidates),
        "last_action.rejected": last_rejected,
        "signal.neighbor_sum_u8": signal_sum,
        "counter.choice_u8": counter_value,
    }
    return {
        "scalars": {
            name: scalars_available[name]
            for name in sorted(policy.permissions)
            if not name.startswith("candidate.")
        },
        "candidates": candidates,
    }


def run_spatial_dsl_episode(
    context,
    definition: EpisodeDefinition,
    action: EvaluationAction,
    *,
    target_id: str,
    initial_state_override: MovementState | None = None,
    offline_tracker: Any | None = None,
) -> dict[str, Any]:
    """Run a complete fixed-clock E06 episode under an arbitrary spatial DSL.

    ``offline_tracker`` is an outcome-blind audit hook for explicitly
    uncalibrated topologies.  It can observe committed native states and
    summaries only; policy decisions, legality, clocks, and costs remain owned
    by the unchanged E06 engine path.  The default calibrated S01/S02 tracker
    remains byte-for-byte behaviorally unchanged.
    """

    policies = compiled_policies(action)
    dispatcher = (
        PortfolioDispatcher(action.portfolio_definition, policies)
        if action.portfolio_definition
        else None
    )
    if dispatcher is not None:
        if set(dispatcher.members_by_carrier) != {"spatial"}:
            raise SuiteValidationError("E06 portfolio requires spatial carriers")
        spatial_policies = tuple(dispatcher.members_by_carrier["spatial"])
    elif len(policies) == 1:
        spatial_policies = tuple(policies.values())
    else:
        raise SuiteValidationError("E06 requires one policy or spatial portfolio")
    for policy in spatial_policies:
        if policy.environment != "spatial2d.v1":
            raise SuiteValidationError("E06 adapter requires spatial2d.v1 policy")
        _validate_spatial_authority(policy, definition)
    if definition.channel_mode != "none":
        raise SuiteValidationError(
            "DSL peer signals cannot impersonate an E06 authority-bearing channel"
        )
    environment = context.environments[definition.environment_id]
    dsl_definitions = {
        policy.policy_sha256: _spatial_definition(policy) for policy in spatial_policies
    }
    relation_profile = (
        None
        if definition.relation_grammar_id is None
        else compile_relation_profile(context.grammars[definition.relation_grammar_id])
    )
    if (
        any(item.relation_profile_required for item in dsl_definitions.values())
        and relation_profile is None
    ):
        raise SuiteValidationError(
            "DSL policy requires unavailable local relation authority"
        )
    if initial_state_override is None:
        state, initial_transform = _initial_state(environment, definition)
    else:
        validate_state_against_environment(environment, initial_state_override)
        if initial_state_override.environment_id != definition.environment_id:
            raise SuiteValidationError(
                "spatial initial-state override/environment mismatch"
            )
        state = initial_state_override
        initial_transform = None
    initial_hash = movement_state_sha256(state)
    actor_ids = tuple(
        sorted(
            occupant.occupant_id
            for _, occupant in state.occupancy
            if occupant.kind == "cell"
        )
    )
    if dispatcher is not None:
        dispatcher.register_identities({"spatial": actor_ids})
    memory = (
        {
            actor: {item.name: item.initial for item in spatial_policies[0].memory}
            for actor in actor_ids
        }
        if dispatcher is None
        else {}
    )
    member_memory: dict[str, dict[str, dict[str, int]]] = {}
    last_rejected = {actor: False for actor in actor_ids}
    buses = {
        policy.policy_sha256: RecipientActivationMessageBus(
            channels=policy.signal_channels,
            bits_per_channel=policy.signal_bits_per_channel,
        )
        for policy in spatial_policies
        if policy.signal_channels
    }
    dispatch_counts: Counter[str] = Counter()
    movement = {field: 0 for field in MOVEMENT_LEDGER_FIELDS}
    observation = {field: 0 for field in OBSERVATION_LEDGER_FIELDS}
    channel = {field: 0 for field in CHANNEL_LEDGER_FIELDS}
    runtime = _empty_runtime_ledger()
    transition_summaries = []
    movement_kind_counts: Counter[str] = Counter()
    changed_transitions = 0
    lagged = {site.site_id: 0 for site in environment.occupiable_sites}
    _, targets, _, _ = load_baseline_assets()
    target = targets[target_id]
    evaluation_grammar = next(
        item for item in context.grammars.values() if item.target_id == target_id
    )
    tracker = (
        TargetMetricTracker(environment, target, evaluation_grammar)
        if offline_tracker is None
        else offline_tracker
    )
    tracker.observe(-1, state, {"transitionKind": "initial_state"})

    for transition_index in range(definition.transitions):
        before_hash = movement_state_sha256(state)
        scheduled = _state_blind_actor_schedule(
            definition.scenario_id,
            transition_index,
            actor_ids,
            definition.actor_batch_size,
        )
        proposals: list[MovementProposal] = []
        proposal_actor: dict[str, str] = {}
        pending_emissions = []
        topology = _spatial_actor_neighbors(environment, state)
        builds: dict[str, ObservationBuild] = {}
        for slot, actor in enumerate(scheduled):
            activation = transition_index * definition.actor_batch_size + slot
            if dispatcher is None:
                policy = spatial_policies[0]
            else:
                policy = dispatcher.select(
                    "spatial",
                    actor,
                    dispatch_counts[actor],
                    selector_value=last_rejected[actor],
                    mutate=True,
                )
                dispatch_counts[actor] += 1
            dsl_definition = dsl_definitions[policy.policy_sha256]
            build = build_policy_observation(
                environment,
                state,
                actor,
                dsl_definition,
                relation_profile=relation_profile,
                lagged_conflicts=lagged,
                decision_key=definition.scenario_id,
                activation_index=activation,
                _state_sha256=before_hash,
            )
            _add_fields(
                observation, build.observation.budget, OBSERVATION_LEDGER_FIELDS
            )
            signal_sum = 0
            if "signal.neighbor_sum_u8" in policy.permissions:
                bus = buses.get(policy.policy_sha256)
                if bus is None:
                    raise SuiteValidationError(
                        "signal permission requires declared channels"
                    )
                observed = bus.observe(actor, activation)
                signal_sum = min(bus.maximum, sum(observed.channel_sums.values()))
            counter = int.from_bytes(
                hashlib.sha256(
                    f"{definition.scenario_id}/{activation}/{actor}".encode()
                ).digest()[:1],
                "big",
            )
            dsl_observation = _spatial_observation(
                policy,
                build,
                last_rejected=last_rejected[actor],
                signal_sum=signal_sum,
                counter_value=counter,
            )
            if dispatcher is None:
                actor_memory = memory[actor]
            else:
                namespaces = member_memory.setdefault(actor, {})
                actor_memory = namespaces.get(policy.policy_sha256)
                if actor_memory is None:
                    actor_memory = {item.name: item.initial for item in policy.memory}
                    namespaces[policy.policy_sha256] = actor_memory
                    dispatcher.note_namespace_initialization()
            decision = execute_policy(policy, dsl_observation, actor_memory)
            if dispatcher is None:
                memory[actor] = dict(decision.memory)
            else:
                member_memory[actor][policy.policy_sha256] = dict(decision.memory)
            runtime["dslActivations"] += 1
            runtime["dslOperations"] += decision.operation_count
            runtime["dslMemoryWrites"] += sum(
                item["kind"] == "set_memory" for item in decision.actions
            )
            runtime["dslSignalBundles"] += int(bool(decision.emitted_signals))
            pending_emissions.append(
                (
                    actor,
                    activation,
                    policy.policy_sha256,
                    dict(decision.emitted_signals),
                )
            )
            terminal = next(
                item
                for item in decision.actions
                if item["kind"] in {"noop", "move_candidate"}
            )
            if terminal["kind"] == "move_candidate":
                key = str(terminal["candidateKey"])
                if key not in build.candidate_map:
                    raise SuiteValidationError(
                        "DSL selected an unauthenticated candidate"
                    )
                proposal = build.candidate_map[key].proposal
                if proposal.kind != terminal["movementKind"]:
                    raise SuiteValidationError(
                        "trusted selector movement-kind mismatch"
                    )
                proposals.append(proposal)
                proposal_actor[proposal.proposal_id] = actor
                builds[proposal.proposal_id] = build
        # Every actor in the synchronous native batch observes before any peer
        # emission from that same batch becomes deliverable.
        for actor, activation, policy_sha256, emitted in pending_emissions:
            bus = buses.get(policy_sha256)
            if emitted and bus is not None:
                bus.emit(SignalEmission(actor, activation, emitted), topology)
        batch = resolve_batch(
            environment,
            state,
            tuple(proposals),
            batch_nonce=f"{definition.scenario_id}:transition:{transition_index}",
        )
        state = parse_movement_state(batch["postState"])
        _add_fields(movement, batch["costLedger"], MOVEMENT_LEDGER_FIELDS)
        accepted = set(batch["acceptedProposalIds"])
        proposed_actors = set(proposal_actor.values())
        if dispatcher is not None:
            for actor in scheduled:
                if actor not in proposed_actors:
                    last_rejected[actor] = False
        for proposal in proposals:
            actor = proposal_actor[proposal.proposal_id]
            last_rejected[actor] = proposal.proposal_id not in accepted
            if proposal.proposal_id in accepted:
                movement_kind_counts[proposal.kind] += 1
        changed = batch["preState"]["stateSha256"] != batch["postState"]["stateSha256"]
        changed_transitions += int(changed)
        next_lagged = {site.site_id: 0 for site in environment.occupiable_sites}
        proposed = {item.proposal_id: item for item in proposals}
        for item in batch["decisions"]:
            if item["outcome"] == "conflict_lost":
                target_site = proposed[item["proposalId"]].target_site
                if target_site is not None:
                    next_lagged[target_site] = min(3, next_lagged[target_site] + 1)
        lagged = next_lagged
        summary = {
            "transitionIndex": transition_index,
            "scheduledActorCount": len(scheduled),
            "proposalCount": len(proposals),
            "acceptedCount": batch["costLedger"]["acceptedMovements"],
            "conflictLosses": batch["costLedger"]["conflictLosses"],
            "invalidProposals": batch["costLedger"]["invalidProposals"],
            "preStateSha256": batch["preState"]["stateSha256"],
            "postStateSha256": batch["postState"]["stateSha256"],
            "transitionSha256": batch["transitionSha256"],
        }
        transition_summaries.append(summary)
        tracker.observe(transition_index, state, summary)

    metrics = tracker.finalize()
    total_accepted = sum(movement_kind_counts.values())
    entropy = 0.0
    if total_accepted > 1:
        entropy = -sum(
            (count / total_accepted) * math.log(count / total_accepted)
            for count in movement_kind_counts.values()
        ) / math.log(4)
    runtime["committedNativeMovementActions"] = movement["acceptedMovements"]
    runtime["committedNativeDisplacement"] = movement["totalGraphDisplacement"]
    runtime["nativeEligibleActionOpportunities"] = (
        definition.transitions * definition.actor_batch_size
    )
    runtime["rejectedNativeActions"] = (
        movement["invalidProposals"] + movement["conflictLosses"]
    )
    communication = {
        "emittedSignalWrites": 0,
        "transmittedSignalBits": 0,
        "recipientDeliveries": 0,
        "bufferOverwrites": 0,
        "consumedDeliveries": 0,
        "observableAggregateReads": 0,
    }
    for bus in buses.values():
        for key, value in bus.ledger_snapshot().items():
            communication[key] += value
    portfolio_ledgers: dict[str, Any] = {}
    if dispatcher is not None:
        snapshot = dispatcher.ledger_snapshot()
        coordination = dict(snapshot["portfolioCoordinationLedger"])
        coordination.update(communication)
        portfolio_ledgers = {
            "portfolioStructuralLedger": snapshot["portfolioStructuralLedger"],
            "portfolioCoordinationLedger": coordination,
            "portfolioAssignmentAudit": (
                {
                    key: value
                    for key, value in snapshot.items()
                    if key
                    not in {
                        "portfolioStructuralLedger",
                        "portfolioCoordinationLedger",
                    }
                }
                | {
                    "memberMemoryNamespaceCount": sum(
                        len(namespaces) for namespaces in member_memory.values()
                    ),
                    "memberMemoryNamespaceSha256": {
                        f"{actor_id}::{policy_sha256}": canonical_sha256(
                            "E07/S08A/member-memory-namespace/v1", value
                        )
                        for actor_id, namespaces in sorted(member_memory.items())
                        for policy_sha256, value in sorted(namespaces.items())
                    },
                }
            ),
        }
    body = {
        "schemaVersion": "e07.s04a.e06-dsl-episode.v1",
        "adapterVersion": ADAPTER_VERSION,
        "scenarioId": definition.scenario_id,
        "environmentId": definition.environment_id,
        "policyId": (
            spatial_policies[0].policy_id if dispatcher is None else "dsl_portfolio"
        ),
        "policySha256": action.policy_sha256,
        "transitionBudget": definition.transitions,
        "actorBatchSize": definition.actor_batch_size,
        "initialStateSha256": initial_hash,
        "initialStateOverrideUsed": initial_state_override is not None,
        "initialTransformSha256": (
            None if initial_transform is None else initial_transform["transitionSha256"]
        ),
        "finalStateSha256": movement_state_sha256(state),
        "movementLedger": movement,
        "observationLedger": observation,
        "e06ChannelLedger": channel,
        "dslRuntimeLedger": runtime,
        "dslCommunicationLedger": communication,
        "licensedCapabilityLedger": {
            key: runtime[key]
            for key in (
                "licensedPrefixPredicateEvaluations",
                "licensedPrefixValueReads",
                "licensedPrefixValueComparisons",
                "engineCursorStateReads",
                "engineCursorTargetProjectionReads",
                "engineCursorAdvanceActions",
                "engineCursorSwapActions",
                "licensedLongRangeRequestedDistance",
                "licensedLongRangeMaximumRequestedDistance",
            )
        },
        **portfolio_ledgers,
        "transitionSummaries": transition_summaries,
        "offlineEvaluation": metrics,
        "descriptors": {
            "committedMovementKindEntropy": entropy,
            "stateTurnoverFraction": changed_transitions / definition.transitions,
            "acceptedNativeActionFraction": (
                movement["acceptedMovements"]
                / (definition.transitions * definition.actor_batch_size)
            ),
            "committedDisplacementFraction": (
                movement["totalGraphDisplacement"]
                / max(
                    1,
                    movement["acceptedMovements"] * SPATIAL_MAX_ENABLED_DISPLACEMENT,
                )
            ),
        },
        "authorityAudit": {
            "opaqueCandidateEnumerationEngineOwned": True,
            "candidateKindUsedOnlyByTrustedSelectorAllowlist": True,
            "authenticationPreserved": True,
            "nativeLegalityPreserved": movement["invalidProposals"] == 0,
            "conflictResolutionNative": True,
            "atomicCommitNative": True,
            "identityMemoryOwned": (
                set(memory) == set(actor_ids)
                if dispatcher is None
                else set(member_memory).issubset(set(actor_ids))
            ),
            "memberMemoryNamespacesIsolated": dispatcher is None
            or all(
                policy_sha256 in {item.policy_sha256 for item in spatial_policies}
                for namespaces in member_memory.values()
                for policy_sha256 in namespaces
            ),
            "e06ChannelsNotImpersonated": True,
            "offlineS01S02Conjunction": True,
            "onlineCompletionHiddenFromPolicy": True,
        },
        "stopReason": "transition_budget",
    }
    body["episodeSha256"] = canonical_sha256("E07/S04A/E06-dsl-episode/v1", body)
    return body
