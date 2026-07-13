"""Evidence-bounded invariant audit for frozen historical swap records.

This module deliberately validates only what the S04 StatusProbe preserved.  It
does not infer actors, activations, observations, or native runtime state.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from .model import TraceBundle
from .schema import observable_state_hash
from .validation import validate_trace


class HistoricalInvariantViolation(AssertionError):
    """A historical observable-record invariant was violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HistoricalInvariantViolation(message)


def audit_historical_trace(
    bundle: TraceBundle, run: Mapping[str, Any]
) -> dict[str, Any]:
    """Audit a C trace against its source run without claiming activation parity."""
    raw = dict(run)
    validate_trace(bundle)
    _require(
        bundle.backend == "historical_frozen_public_commit",
        "historical audit received a non-historical backend",
    )
    _require(bundle.sequence_basis == "recorded_swap", "historical sequence is not recorded_swap")
    steps = list(raw["sortingSteps"])
    _require(len(bundle.events) == len(steps), "event count differs from successful-swap snapshots")
    previous = list(raw["inputValues"])
    _require(
        Counter(previous) == Counter(raw["finalValues"]),
        "input/final value multisets differ",
    )
    invisible = 0
    exact = 0
    for ordinal, (event, step) in enumerate(zip(bundle.events, steps)):
        post = list(step)
        _require(event["run"]["eventIndex"] == ordinal, f"event {ordinal} index mismatch")
        _require(event["run"]["sequenceBasis"] == "recorded_swap", f"event {ordinal} basis mismatch")
        _require(event["proposal"]["kind"] == "Swap", f"event {ordinal} is not a swap record")
        _require(event["decision"] == {"status": "accepted", "accepted": True}, f"event {ordinal} is not accepted")
        _require(Counter(previous) == Counter(post), f"event {ordinal} changes the value multiset")
        _require(
            event["state"]["observablePreHash"] == observable_state_hash(previous),
            f"event {ordinal} observable pre-hash mismatch",
        )
        _require(
            event["state"]["observablePostHash"] == observable_state_hash(post),
            f"event {ordinal} observable post-hash mismatch",
        )
        payload = event["backendPayload"]
        _require(payload["sourceSwapOrdinal"] == ordinal, f"event {ordinal} source ordinal mismatch")
        _require(payload["sourceSortingStep"] == post, f"event {ordinal} source snapshot mismatch")
        _require(event["costDelta"]["acceptedSwaps"] == 1, f"event {ordinal} swap cost mismatch")
        _require(event["costDelta"]["displacedCells"] == 2, f"event {ordinal} displacement cost mismatch")
        _require(event["displacement"]["count"] == 2, f"event {ordinal} displacement count mismatch")
        changed = [index for index, pair in enumerate(zip(previous, post)) if pair[0] != pair[1]]
        if not changed:
            invisible += 1
            _require(
                event["displacement"]["positions"] is None,
                f"event {ordinal} invents positions for an equal-value swap",
            )
        else:
            exact += 1
            _require(len(changed) == 2, f"event {ordinal} changes other than two positions")
            _require(
                previous[changed[0]] == post[changed[1]]
                and previous[changed[1]] == post[changed[0]],
                f"event {ordinal} is not an observable value exchange",
            )
            _require(
                event["displacement"]["positions"]["changed"] == changed,
                f"event {ordinal} changed-position evidence mismatch",
            )
        previous = post
    _require(previous == list(raw["finalValues"]), "last observable snapshot differs from final values")
    return {
        "success": True,
        "sequenceBasis": "recorded_swap",
        "recordedSwapCount": len(steps),
        "exactPositionSwapCount": exact,
        "observationallyInvisibleSwapCount": invisible,
        "activationParityClaimed": False,
    }
