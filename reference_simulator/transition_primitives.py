"""Controller-neutral validation, commit, and cost-accounting primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .model import FaultMode, Proposal, ProposalKind, RunState, Scenario


@dataclass(frozen=True, slots=True)
class ValidationDecision:
    decision: str
    eligible_for_commit: bool


def validate_proposal(
    scenario: Scenario,
    state: RunState,
    proposal: Proposal,
) -> ValidationDecision:
    """Validate one proposal mechanically without making a policy choice."""
    if proposal.kind == ProposalKind.NO_OP:
        return ValidationDecision("no_op", False)

    if proposal.kind == ProposalKind.MEMORY_UPDATE:
        valid = proposal.actor_id in state.selection_cursors and proposal.new_cursor is not None
        return ValidationDecision(
            "valid" if valid else "rejected_invalid_memory_update",
            valid,
        )

    if proposal.target_pos is None or not 0 <= proposal.target_pos < len(state.occupancy):
        return ValidationDecision("rejected_invalid_target", False)
    if not 0 <= proposal.actor_pos < len(state.occupancy):
        return ValidationDecision("rejected_invalid_actor_position", False)
    if state.occupancy[proposal.actor_pos] != proposal.actor_id:
        return ValidationDecision("rejected_stale_actor", False)
    cells = scenario.cell_map
    actor = cells[proposal.actor_id]
    target = cells[state.occupancy[proposal.target_pos]]
    if (
        proposal.observed_target_id is not None
        and target.cell_id != proposal.observed_target_id
    ):
        return ValidationDecision("rejected_stale_target", False)
    if actor.fault == FaultMode.STUCK:
        return ValidationDecision("rejected_actor_stuck", False)
    if target.fault == FaultMode.STUCK:
        return ValidationDecision("rejected_target_stuck", False)
    return ValidationDecision("valid", True)


def cost_delta(
    ledger_fields: Iterable[str],
    proposal: Proposal,
    decision: str,
) -> dict[str, int]:
    """Charge exactly one opportunity and classify its terminal disposition."""
    delta = {key: 0 for key in ledger_fields}
    delta["activations"] = 1
    delta["observationReads"] = proposal.observation_reads
    delta["valueComparisons"] = proposal.value_comparisons
    delta["proposals"] = 1
    if proposal.kind == ProposalKind.NO_OP:
        delta["noOps"] = 1
    elif decision.startswith("rejected"):
        delta["rejections"] = 1
    elif decision == "conflict_loss":
        delta["conflictLosses"] = 1
    elif proposal.kind == ProposalKind.MEMORY_UPDATE and decision == "accepted":
        delta["memoryUpdates"] = 1
    elif proposal.kind == ProposalKind.SWAP and decision == "accepted":
        delta["acceptedSwaps"] = 1
        delta["displacedCells"] = 2
    return delta


def commit_proposal(
    state: RunState,
    snapshot: RunState,
    proposal: Proposal,
    decision: str,
) -> bool:
    """Commit one accepted change atomically; return whether state changed."""
    if decision != "accepted":
        return False
    if proposal.kind == ProposalKind.MEMORY_UPDATE:
        assert proposal.new_cursor is not None
        state.selection_cursors[proposal.actor_id] = proposal.new_cursor
        return True
    if proposal.kind == ProposalKind.SWAP:
        assert proposal.target_pos is not None
        state.occupancy[proposal.actor_pos], state.occupancy[proposal.target_pos] = (
            snapshot.occupancy[proposal.target_pos],
            snapshot.occupancy[proposal.actor_pos],
        )
        return True
    return False


def ledger_identity(ledger: Mapping[str, int]) -> dict[str, bool]:
    """Return the three exact accounting identities frozen by E01/S02."""
    return {
        "oneProposalPerActivation": ledger["activations"] == ledger["proposals"],
        "proposalPartition": ledger["proposals"]
        == ledger["noOps"]
        + ledger["rejections"]
        + ledger["memoryUpdates"]
        + ledger["acceptedSwaps"]
        + ledger["conflictLosses"],
        "twoDisplacementsPerSwap": ledger["displacedCells"]
        == 2 * ledger["acceptedSwaps"],
    }
