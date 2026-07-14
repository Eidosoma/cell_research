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
from .scheduler import resolve_conflicts, scheduled_actor, scheduled_priority, scheduled_side
from .transition_primitives import commit_proposal, cost_delta, validate_proposal


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
) -> Proposal:
    draws: list[tuple[str, int, int, int]] = []
    if scenario.architecture == Architecture.TRADITIONAL:
        proposal = traditional_proposal(scenario, snapshot)
        return replace(proposal, ordinal=ordinal)

    actor_id, actor_draws, actor_consumed = scheduled_actor(
        scenario, event_index, include_draws=capture_random_draws
    )
    draws.extend(actor_draws)
    _update_counter(snapshot, "actor_activation", actor_consumed)
    actor = scenario.cell_map[actor_id]
    side = None
    if actor.policy == Policy.BUBBLE and actor.fault == FaultMode.NORMAL:
        side, side_draw = scheduled_side(scenario, event_index)
        draws.append(side_draw)
        _update_counter(snapshot, "bubble_side")
    factory = cell_view_proposal if proposal_factory is None else proposal_factory
    proposal = factory(scenario, snapshot, actor_id, side=side)
    priority = 0
    if scenario.batch_width > 1:
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
    width = min(scenario.batch_width, max(remaining, 0))
    if width == 0:
        return [], []
    snapshot = state.clone()
    # Draw counters are part of state; generation updates the working counter
    # copy, never transition-visible occupancy/cursors.
    proposals: list[Proposal] = []
    for ordinal in range(width):
        event_index = state.activation_count + ordinal
        proposals.append(
            _proposal_for_slot(
                scenario, snapshot, event_index, ordinal,
                capture_random_draws=emit_event_records,
                proposal_factory=proposal_factory,
            )
        )

    decisions: dict[int, str] = {}
    valid_changes: list[Proposal] = []
    for proposal in proposals:
        validation = validate_proposal(scenario, state, proposal)
        decisions[proposal.ordinal] = validation.decision
        if validation.eligible_for_commit:
            valid_changes.append(proposal)

    if scenario.batch_width > 1:
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
