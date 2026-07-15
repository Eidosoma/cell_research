"""Atomic deterministic execution engine for the clean-room reference semantics."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Any, Callable, Literal, Mapping, Protocol

from .model import (
    Architecture,
    Direction,
    FaultMode,
    Policy,
    Proposal,
    ProposalKind,
    RunResult,
    RunState,
    Scenario,
    canonical_json_bytes,
    state_hash,
)
from .policies import cell_view_proposal, has_admissible_change, traditional_proposal
from .scheduler import (
    ScheduledOpportunity,
    resolve_conflicts,
    scheduled_actor,
    scheduled_priority,
    scheduled_side,
)
from .transition_primitives import (
    ValidationDecision,
    commit_proposal,
    cost_delta,
    validate_proposal,
)


TRACE_SCHEMA = "E01.reference-event-pre-S06.v1"
EMPTY_DIGEST = hashlib.sha256(b"E01/reference/trace/v1").hexdigest()
_PROGRESS_GUARANTEE_CACHE: dict[str, bool] = {}


class CellViewProposalFactory(Protocol):
    """Controller-neutral callback for one policy-native proposal."""

    def __call__(
        self,
        scenario: Scenario,
        state: RunState,
        actor_id: str,
        *,
        side: Literal["left", "right"] | None = None,
    ) -> Proposal: ...


class ScheduleFactory(Protocol):
    """Select the next charged batch without access to dynamic run state."""

    def __call__(
        self,
        event_index: int,
        remaining_opportunities: int,
    ) -> tuple[ScheduledOpportunity, ...]: ...


class BatchExecutionInterceptor(Protocol):
    """Optional post-construction fault hook; absent on the frozen E01 path."""

    def prepare(
        self,
        proposal: Proposal,
        event_index: int,
    ) -> tuple[Proposal, tuple[tuple[str, int], ...]]: ...

    def outcome(
        self,
        proposal: Proposal,
        validation: ValidationDecision,
        event_index: int,
    ) -> ValidationDecision: ...

    def after_batch(
        self,
        scenario: Scenario,
        state: RunState,
        proposals: tuple[Proposal, ...],
        decisions: Mapping[int, str],
        batch_start_index: int,
    ) -> None: ...


def initial_state(scenario: Scenario) -> RunState:
    return RunState(
        occupancy=list(scenario.initial_occupancy),
        selection_cursors=dict(scenario.initial_selection_cursors),
    )


def invariant_error(scenario: Scenario, state: RunState) -> str | None:
    expected = {cell.cell_id for cell in scenario.cells}
    if len(state.occupancy) != len(expected) or set(state.occupancy) != expected:
        return "occupancy_not_bijection"
    if state.activation_count < 0:
        return "negative_activation_count"
    for cell_id, cursor in state.selection_cursors.items():
        cell = scenario.cell_map.get(cell_id)
        if cell is None or cell.policy != Policy.SELECTION or not isinstance(cursor, int):
            return "invalid_selection_cursor"
    if any(value < 0 for value in state.stream_counters.values()):
        return "negative_stream_counter"
    if any(value < 0 for value in state.ledger.values()):
        return "negative_ledger"
    return None


def is_complete(scenario: Scenario, state: RunState) -> bool:
    cells = scenario.cell_map
    directions = {cell.direction for cell in scenario.cells}
    if len(directions) != 1:
        return False
    direction = next(iter(directions))
    values = [cells[cell_id].value for cell_id in state.occupancy]
    if direction == Direction.ASCENDING:
        return all(left <= right for left, right in zip(values, values[1:]))
    return all(left >= right for left, right in zip(values, values[1:]))


def _incomplete_state_has_progress_witness(scenario: Scenario) -> bool:
    """Return a cached proof obligation used only to skip quiescence scans.

    With no faults and one common direction, an incomplete pure Bubble or
    Insertion line contains an adjacent inversion.  Bubble can propose that
    inversion; the first inversion also has an ordered strict prefix and is
    therefore an admissible Insertion proposal.  The clean-room traditional
    controllers likewise propose a change from every incomplete no-fault
    state, including Selection.  Cell-view Selection is intentionally excluded
    because its identity-owned cursors can exhaust.
    """
    cached = _PROGRESS_GUARANTEE_CACHE.get(scenario.scenario_id)
    if cached is not None:
        return cached
    no_faults = all(cell.fault == FaultMode.NORMAL for cell in scenario.cells)
    one_direction = len({cell.direction for cell in scenario.cells}) == 1
    policies = {cell.policy for cell in scenario.cells}
    if scenario.architecture == Architecture.TRADITIONAL:
        result = no_faults and one_direction and len(policies) == 1
    else:
        result = (
            no_faults
            and one_direction
            and len(policies) == 1
            and next(iter(policies)) in {Policy.BUBBLE, Policy.INSERTION}
        )
    _PROGRESS_GUARANTEE_CACHE[scenario.scenario_id] = result
    return result


def evaluate_terminal(scenario: Scenario, state: RunState) -> str | None:
    """Apply S03 terminal precedence exactly."""
    if invariant_error(scenario, state) is not None:
        return "invariant_error"
    if is_complete(scenario, state):
        return "complete"
    if (
        not _incomplete_state_has_progress_witness(scenario)
        and not has_admissible_change(scenario, state)
    ):
        return "quiescent"
    if state.activation_count >= scenario.max_activations:
        return "event_budget"
    return None


def _update_counter(state: RunState, stream: str, count: int = 1) -> None:
    state.stream_counters[stream] = state.stream_counters.get(stream, 0) + count


def _proposal_for_slot(
    scenario: Scenario,
    snapshot: RunState,
    event_index: int,
    ordinal: int,
    *,
    capture_random_draws: bool = True,
    proposal_factory: CellViewProposalFactory | None = None,
    scheduled_opportunity: ScheduledOpportunity | None = None,
    actual_batch_width: int | None = None,
) -> Proposal:
    draws: list[tuple[str, int, int, int]] = []
    if scenario.architecture == Architecture.TRADITIONAL:
        proposal = traditional_proposal(scenario, snapshot)
        return replace(proposal, ordinal=ordinal)

    if scheduled_opportunity is None:
        actor_id, actor_draws, actor_consumed = scheduled_actor(
            scenario, event_index, include_draws=capture_random_draws
        )
        draws.extend(actor_draws)
        _update_counter(snapshot, "actor_activation", actor_consumed)
    else:
        actor_id = scheduled_opportunity.actor_id
        if actor_id not in scenario.cell_map:
            raise ValueError("scheduler selected an unknown actor identity")
        if capture_random_draws:
            draws.extend(scheduled_opportunity.random_draws)
        for stream, count in scheduled_opportunity.stream_consumption:
            _update_counter(snapshot, stream, count)
    actor = scenario.cell_map[actor_id]
    side = None
    if scheduled_opportunity is not None and scheduled_opportunity.bubble_side is not None:
        if actor.policy != Policy.BUBBLE:
            raise ValueError("external Bubble side supplied for a non-Bubble actor")
        side = scheduled_opportunity.bubble_side
    elif actor.policy == Policy.BUBBLE and actor.fault == FaultMode.NORMAL:
        side, side_draw = scheduled_side(scenario, event_index)
        draws.append(side_draw)
        _update_counter(snapshot, "bubble_side")
    factory = cell_view_proposal if proposal_factory is None else proposal_factory
    proposal_state = snapshot
    if proposal_factory is not None and snapshot.activation_count != event_index:
        # A synchronous batch shares occupancy/cursors but each charged slot
        # retains its own global opportunity clock for the frozen S03 weak
        # coordinator frequency rule.  The policy gateway cannot read this
        # field; only the separately typed coordinator signal encoder can.
        proposal_state = snapshot.clone()
        proposal_state.activation_count = event_index
    proposal = factory(scenario, proposal_state, actor_id, side=side)
    # Optional policy-gateway randomness (for example S05 sensing noise) is
    # already part of the returned proposal. Count and retain it without
    # changing byte output on the default empty-draw path.
    if proposal.random_draws:
        draws.extend(proposal.random_draws)
        counts: dict[str, int] = {}
        for stream, _, _, _ in proposal.random_draws:
            counts[stream] = counts.get(stream, 0) + 1
        for stream, count in counts.items():
            _update_counter(snapshot, stream, count)
    priority = 0
    effective_batch_width = (
        scenario.batch_width if actual_batch_width is None else actual_batch_width
    )
    if effective_batch_width > 1:
        priority, priority_draw = scheduled_priority(scenario, event_index)
        draws.append(priority_draw)
        _update_counter(snapshot, "conflict_priority")
    return replace(
        proposal,
        ordinal=ordinal,
        priority=priority,
        random_draws=tuple(draws),
    )


def _event_actor_fields(scenario: Scenario, proposal: Proposal) -> tuple[str, str]:
    cell = scenario.cell_map.get(proposal.actor_id)
    if cell is None:
        policy = scenario.traditional_policy.value if scenario.traditional_policy else "controller"
        return f"Traditional:{policy}", "controller"
    return cell.policy.value, cell.direction.value


def execute_batch(
    scenario: Scenario,
    state: RunState,
    *,
    retain_events: bool,
    emit_event_records: bool = True,
    proposal_factory: CellViewProposalFactory | None = None,
    schedule_factory: ScheduleFactory | None = None,
    execution_interceptor: BatchExecutionInterceptor | None = None,
) -> tuple[list[dict[str, Any]], list[bytes]]:
    """Generate from one snapshot, validate, resolve, and commit atomically.

    ``emit_event_records=False`` is a summary-only execution path.  It applies
    exactly the same proposals and state transition but avoids constructing or
    hashing per-activation trace payloads.  The default is deliberately the
    validated S05/S06 behavior, and callers requesting retained events may not
    disable event emission.
    """
    if retain_events and not emit_event_records:
        raise ValueError("retained events require emit_event_records=True")
    remaining = scenario.max_activations - state.activation_count
    scheduled: tuple[ScheduledOpportunity, ...] | None = None
    if schedule_factory is None:
        width = min(scenario.batch_width, max(remaining, 0))
    else:
        scheduled = tuple(schedule_factory(state.activation_count, max(remaining, 0)))
        width = len(scheduled)
        if width > max(remaining, 0):
            raise ValueError("scheduler batch exceeds remaining opportunity budget")
        if remaining > 0 and width == 0:
            raise ValueError("scheduler returned an empty batch before terminal state")
    if width == 0:
        return [], []
    snapshot = state.clone()
    # Draw counters are part of state; generation updates the working counter
    # copy, never transition-visible occupancy/cursors.
    proposals: list[Proposal] = []
    for ordinal in range(width):
        event_index = state.activation_count + ordinal
        proposal = _proposal_for_slot(
                scenario, snapshot, event_index, ordinal,
                capture_random_draws=emit_event_records,
                proposal_factory=proposal_factory,
                scheduled_opportunity=(scheduled[ordinal] if scheduled is not None else None),
                actual_batch_width=width,
            )
        if execution_interceptor is not None:
            proposal, consumption = execution_interceptor.prepare(proposal, event_index)
            for stream, count in consumption:
                _update_counter(snapshot, stream, count)
        proposals.append(proposal)

    decisions: dict[int, str] = {}
    valid_changes: list[Proposal] = []
    for proposal in proposals:
        validation = validate_proposal(scenario, state, proposal)
        if execution_interceptor is not None:
            validation = execution_interceptor.outcome(
                proposal,
                validation,
                state.activation_count + proposal.ordinal,
            )
        decisions[proposal.ordinal] = validation.decision
        if validation.eligible_for_commit:
            valid_changes.append(proposal)

    if width > 1:
        accepted, lost = resolve_conflicts(valid_changes)
    else:
        accepted = {proposal.ordinal for proposal in valid_changes}
        lost = set()
    for ordinal in lost:
        decisions[ordinal] = "conflict_loss"
    for ordinal in accepted:
        decisions[ordinal] = "accepted"

    pre_hash = state_hash(scenario.scenario_id, state) if emit_event_records else None
    ledger_deltas: list[dict[str, int]] = []
    for proposal in proposals:
        decision = decisions[proposal.ordinal]
        delta = cost_delta(state.ledger, proposal, decision)
        commit_proposal(state, snapshot, proposal, decision)
        for key, value in delta.items():
            state.ledger[key] += value
        ledger_deltas.append(delta)

    state.activation_count += width
    state.stream_counters = snapshot.stream_counters
    state.terminal = evaluate_terminal(scenario, state)
    if execution_interceptor is not None:
        execution_interceptor.after_batch(
            scenario,
            state,
            tuple(proposals),
            decisions,
            snapshot.activation_count,
        )
    post_hash = state_hash(scenario.scenario_id, state) if emit_event_records else None

    if not emit_event_records:
        return [], []

    events: list[dict[str, Any]] = []
    event_bytes: list[bytes] = []
    for proposal, delta in zip(proposals, ledger_deltas):
        event_index = snapshot.activation_count + proposal.ordinal
        actor_policy, actor_direction = _event_actor_fields(scenario, proposal)
        event = {
            "schemaVersion": TRACE_SCHEMA,
            "scenarioId": scenario.scenario_id,
            "eventIndex": event_index,
            "batchWidth": width,
            "batchOrdinal": proposal.ordinal,
            "actorId": proposal.actor_id,
            "actorAlgotype": actor_policy,
            "actorDirection": actor_direction,
            "preStateHash": pre_hash,
            "observation": {
                "reads": proposal.observation_reads,
                "valueComparisons": proposal.value_comparisons,
            },
            "randomAddressesAndDraws": proposal.to_dict()["randomDraws"],
            "proposal": proposal.to_dict(),
            "decision": decisions[proposal.ordinal],
            "ledgerDelta": delta,
            "postStateHash": post_hash,
            "stopReasonIfAny": state.terminal,
        }
        encoded = canonical_json_bytes(event)
        event_bytes.append(encoded)
        if retain_events:
            events.append(event)
    return events, event_bytes


def execute_serial_summary_activation(
    scenario: Scenario,
    state: RunState,
    *,
    on_accepted_swap: Callable[[RunState, int, int], None] | None = None,
    proposal_factory: CellViewProposalFactory | None = None,
) -> bool:
    """Execute one serial activation without trace or snapshot allocation.

    This is a performance-only projection of ``execute_batch`` for replicate
    summaries.  It is lawful only for batch width one, returns whether an
    accepted swap/memory update changed state, and evaluates only the event
    budget.  Its caller must apply ``evaluate_terminal`` after every returned
    state change.  The ordinary event-emitting path remains the validation and
    replay authority.
    """
    if scenario.batch_width != 1:
        raise ValueError("serial summary activation requires batch_width=1")
    if state.terminal is not None:
        return False
    if state.activation_count >= scenario.max_activations:
        state.terminal = "event_budget"
        return False

    proposal = _proposal_for_slot(
        scenario,
        state,
        state.activation_count,
        0,
        capture_random_draws=False,
        proposal_factory=proposal_factory,
    )
    snapshot = state.clone()
    validation = validate_proposal(scenario, state, proposal)
    decision = "accepted" if validation.eligible_for_commit else validation.decision
    delta = cost_delta(state.ledger, proposal, decision)
    changed = False
    if validation.eligible_for_commit:
        changed = commit_proposal(state, snapshot, proposal, "accepted")
        if changed and proposal.kind == ProposalKind.SWAP and on_accepted_swap is not None:
            assert proposal.target_pos is not None
            on_accepted_swap(state, proposal.actor_pos, proposal.target_pos)
    for key, value in delta.items():
        state.ledger[key] += value
    state.activation_count += 1
    if state.activation_count >= scenario.max_activations:
        state.terminal = "event_budget"
    return changed


def run(
    scenario: Scenario,
    *,
    trace_mode: Literal["full", "digest", "none"] = "digest",
    proposal_factory: CellViewProposalFactory | None = None,
    schedule_factory: ScheduleFactory | None = None,
    execution_interceptor: BatchExecutionInterceptor | None = None,
) -> RunResult:
    scenario.validate()
    if scenario.architecture == Architecture.TRADITIONAL and scenario.batch_width != 1:
        raise ValueError("traditional controller currently requires batch_width=1")
    state = initial_state(scenario)
    state.terminal = evaluate_terminal(scenario, state)
    initial_hash = state_hash(scenario.scenario_id, state)
    retained: list[Mapping[str, Any]] = []
    digest = bytes.fromhex(EMPTY_DIGEST)
    while state.terminal is None:
        events, encoded_events = execute_batch(
            scenario,
            state,
            retain_events=(trace_mode == "full"),
            proposal_factory=proposal_factory,
            schedule_factory=schedule_factory,
            execution_interceptor=execution_interceptor,
        )
        if not encoded_events and state.terminal is None:
            state.terminal = evaluate_terminal(scenario, state) or "event_budget"
        for encoded in encoded_events:
            digest = hashlib.sha256(digest + encoded).digest()
        retained.extend(events)
    final_hash = state_hash(scenario.scenario_id, state)
    cells = scenario.cell_map
    values = [cells[cell_id].value for cell_id in state.occupancy]
    summary = {
        "semanticsVersion": scenario.semantics_version,
        "architecture": scenario.architecture.value,
        "policy": (
            scenario.traditional_policy.value
            if scenario.architecture == Architecture.TRADITIONAL
            else "cell_view_mixed_or_identity_owned"
        ),
        "scheduler": scenario.scheduler,
        "faultPlacement": scenario.fault_placement,
        "requestedFaultCount": scenario.requested_fault_count,
        "realizedFaultCount": scenario.realized_fault_count,
        "stopReason": state.terminal,
        "completed": state.terminal == "complete",
        "activationCount": state.activation_count,
        "finalOccupancy": list(state.occupancy),
        "finalValues": values,
        "ledger": dict(state.ledger),
        "traceMode": trace_mode,
        "retainedEventCount": len(retained),
    }
    return RunResult(
        scenario=scenario,
        final_state=state.to_dict(),
        initial_state_hash=initial_hash,
        final_state_hash=final_hash,
        event_digest=digest.hex(),
        events=tuple(retained),
        summary=summary,
    )
