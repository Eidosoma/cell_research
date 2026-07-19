"""Deterministic count-preserving falsification for E06 S02 grammars."""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Sequence

from .grammar import RelationalGrammar, score_grid
from .targets import Grid, TargetDefinition, evaluate_success, exact_equivalence_orbit


def _swap(grid: Grid, first: tuple[int, int], second: tuple[int, int]) -> Grid:
    mutable = [list(row) for row in grid]
    mutable[first[0]][first[1]], mutable[second[0]][second[1]] = (
        mutable[second[0]][second[1]],
        mutable[first[0]][first[1]],
    )
    return tuple(tuple(row) for row in mutable)


def _grid_rows(grid: Grid) -> list[str]:
    return ["".join(row) for row in grid]


def _grid_sha256(grid: Grid) -> str:
    return hashlib.sha256(
        json.dumps(_grid_rows(grid), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _hamming(first: Grid, second: Grid) -> int:
    return sum(
        first[row][col] != second[row][col]
        for row in range(len(first))
        for col in range(len(first[0]))
    )


def enumerate_single_swap_counterexamples(
    target: TargetDefinition, grammar: RelationalGrammar
) -> dict[str, Any]:
    """Exhaust every one-transposition, exact-count neighbor of the fixture."""

    coordinates = [
        (row, col) for row in range(target.shape[0]) for col in range(target.shape[1])
    ]
    proposal_count = 0
    grammar_accepted = 0
    target_accepted = 0
    false_positive_count = 0
    best_false_positive: dict[str, Any] | None = None
    orbit = exact_equivalence_orbit(target)
    for first_index, first in enumerate(coordinates):
        for second in coordinates[first_index + 1 :]:
            if target.grid[first[0]][first[1]] == target.grid[second[0]][second[1]]:
                continue
            proposal_count += 1
            candidate = _swap(target.grid, first, second)
            grammar_result = score_grid(candidate, grammar)
            target_result = evaluate_success(candidate, target, equivalence_orbit=orbit)
            grammar_accepted += int(grammar_result["accepted"])
            target_accepted += int(target_result["success"])
            if grammar_result["accepted"] and not target_result["success"]:
                false_positive_count += 1
                record = {
                    "swap": [list(first), list(second)],
                    "rows": _grid_rows(candidate),
                    "gridSha256": _grid_sha256(candidate),
                    "grammar": grammar_result,
                    "targetAudit": target_result,
                }
                rank = (
                    grammar_result["softScore"],
                    grammar_result["relationalScore"],
                    target_result["mismatchCount"],
                    record["gridSha256"],
                )
                if best_false_positive is None or rank > best_false_positive["_rank"]:
                    best_false_positive = {**record, "_rank": rank}
    if best_false_positive is not None:
        best_false_positive.pop("_rank")
    return {
        "targetId": target.target_id,
        "proposalCount": proposal_count,
        "grammarAcceptedCount": grammar_accepted,
        "s01AcceptedCount": target_accepted,
        "grammarAcceptedOutsideS01Count": false_positive_count,
        "bestFalsePositive": best_false_positive,
    }


def _objective(
    grammar_result: dict[str, Any], hamming_from_fixture: int, cell_count: int
) -> float:
    return (
        grammar_result["softScore"]
        + 0.15 * grammar_result["relationalScore"]
        + 0.30 * min(1.0, hamming_from_fixture / max(8, cell_count // 3))
        + 0.20 * int(grammar_result["accepted"])
        - 0.60 * grammar_result["hardViolationCount"]
    )


def search_distant_counterexample(
    target: TargetDefinition,
    grammar: RelationalGrammar,
    *,
    proposals: int = 100_000,
    restarts: int = 8,
    minimum_hamming: int = 8,
    seed: int | None = None,
    initial_states: Sequence[Grid] = (),
) -> dict[str, Any]:
    """Search for a distant grammar-accepted grid outside the S01 target set.

    The search uses exact-count swaps, hard-first grammar scoring, and a fixed
    simulated-annealing schedule.  S01 target membership is consulted only to
    audit improving accepted candidates, never as an optimization signal.
    """

    if seed is None:
        seed = int.from_bytes(
            hashlib.sha256(f"E06/S02/{target.target_id}".encode()).digest()[:8],
            "big",
        )
    rng = random.Random(seed)
    coordinates = [
        (row, col) for row in range(target.shape[0]) for col in range(target.shape[1])
    ]
    steps_per_restart = max(1, proposals // restarts)
    evaluated = 0
    accepted_transitions = 0
    audited_candidates = 0
    best: dict[str, Any] | None = None
    best_audit_trigger: tuple[float, float, int] | None = None
    orbit = exact_equivalence_orbit(target)

    for restart in range(restarts):
        current = (
            initial_states[restart] if restart < len(initial_states) else target.grid
        )
        # Deterministically diversify restart states without changing counts.
        for _ in range(max(0, restart - len(initial_states) + 1)):
            for _attempt in range(100):
                first, second = rng.sample(coordinates, 2)
                if current[first[0]][first[1]] != current[second[0]][second[1]]:
                    current = _swap(current, first, second)
                    break
        current_result = score_grid(current, grammar)
        current_hamming = _hamming(current, target.grid)
        current_objective = _objective(
            current_result, current_hamming, target.shape[0] * target.shape[1]
        )

        if current_result["accepted"] and current_hamming >= minimum_hamming:
            target_audit = evaluate_success(current, target, equivalence_orbit=orbit)
            audited_candidates += 1
            if not target_audit["success"]:
                audit_failures = int(not target_audit["componentMatch"]) + int(
                    not target_audit["topologyMatch"]
                )
                grid_sha256 = _grid_sha256(current)
                rank = (
                    current_result["softScore"],
                    current_result["relationalScore"],
                    audit_failures,
                    target_audit["mismatchCount"],
                    grid_sha256,
                )
                best_audit_trigger = (
                    round(current_result["softScore"], 12),
                    round(current_result["relationalScore"], 12),
                    current_hamming,
                )
                best = {
                    "_rank": rank,
                    "rows": _grid_rows(current),
                    "gridSha256": grid_sha256,
                    "hammingFromCanonicalFixture": current_hamming,
                    "grammar": current_result,
                    "targetAudit": target_audit,
                    "restart": restart,
                    "discoveryProposalIndex": evaluated,
                }

        for step in range(steps_per_restart):
            for _attempt in range(100):
                first, second = rng.sample(coordinates, 2)
                if current[first[0]][first[1]] != current[second[0]][second[1]]:
                    break
            proposal = _swap(current, first, second)
            proposal_result = score_grid(proposal, grammar)
            proposal_hamming = _hamming(proposal, target.grid)
            proposal_objective = _objective(
                proposal_result,
                proposal_hamming,
                target.shape[0] * target.shape[1],
            )
            progress = step / max(1, steps_per_restart - 1)
            temperature = 0.08 * (1.0 - progress) + 0.005
            delta = proposal_objective - current_objective
            if delta >= 0 or rng.random() < math.exp(delta / temperature):
                current = proposal
                current_result = proposal_result
                current_hamming = proposal_hamming
                current_objective = proposal_objective
                accepted_transitions += 1
            evaluated += 1

            if not current_result["accepted"] or current_hamming < minimum_hamming:
                continue
            trigger = (
                round(current_result["softScore"], 12),
                round(current_result["relationalScore"], 12),
                current_hamming,
            )
            if best_audit_trigger is not None and trigger <= best_audit_trigger:
                continue
            target_audit = evaluate_success(current, target, equivalence_orbit=orbit)
            audited_candidates += 1
            if target_audit["success"]:
                continue
            audit_failures = int(not target_audit["componentMatch"]) + int(
                not target_audit["topologyMatch"]
            )
            rank = (
                current_result["softScore"],
                current_result["relationalScore"],
                audit_failures,
                target_audit["mismatchCount"],
                _grid_sha256(current),
            )
            if best is None or rank > best["_rank"]:
                best_audit_trigger = trigger
                best = {
                    "_rank": rank,
                    "rows": _grid_rows(current),
                    "gridSha256": _grid_sha256(current),
                    "hammingFromCanonicalFixture": current_hamming,
                    "grammar": current_result,
                    "targetAudit": target_audit,
                    "restart": restart,
                    "discoveryProposalIndex": evaluated,
                }

    if best is not None:
        best.pop("_rank")
    return {
        "targetId": target.target_id,
        "seed": seed,
        "proposalBudget": proposals,
        "restartCount": restarts,
        "minimumHamming": minimum_hamming,
        "proposalsEvaluated": evaluated,
        "acceptedTransitions": accepted_transitions,
        "s01AuditedImprovingCandidates": audited_candidates,
        "counterexampleFound": best is not None,
        "counterexample": best,
    }
