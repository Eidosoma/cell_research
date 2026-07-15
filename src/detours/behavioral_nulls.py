"""Finite-graph behavioral nulls for E03 S11.

Every realized transition is one labelled edge in the frozen S05 structural
opportunity graph.  The functions here deliberately do not invoke or emulate a
byte-exact E01 scheduler.  They define counter-addressed policies over the S05
opportunity set and retain loops, multiedges, native no-ops, and rejections.
"""

from __future__ import annotations

import hashlib
from typing import Final

import numba as nb
import numpy as np


NULL_NAMES: Final[tuple[str, ...]] = (
    "random_legal",
    "rate_matched_random",
    "open_loop_opportunity",
    "greedy_adjacent_descents",
    "greedy_inversion_count",
    "greedy_spearman_footrule",
    "greedy_maximum_rank_error",
)
METRIC_NAMES: Final[tuple[str, ...]] = (
    "adjacent_descents",
    "inversion_count",
    "spearman_footrule",
    "maximum_rank_error",
)

STOP_ACTIVE_BUDGET: Final[int] = 3


def stream_root(pair_block_id: str, null_name: str) -> np.uint64:
    """Return the frozen SHA-256-derived S11 stream root."""

    if null_name not in NULL_NAMES:
        raise ValueError(f"unknown S11 null: {null_name}")
    payload = f"E03/S11/null/v1\x00{pair_block_id}\x00{null_name}".encode()
    return np.uint64(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


@nb.njit(cache=True)
def splitmix64(value: np.uint64) -> np.uint64:
    """One portable unsigned SplitMix64 permutation."""

    value = np.uint64(value + np.uint64(0x9E3779B97F4A7C15))
    value = np.uint64((value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9))
    value = np.uint64((value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB))
    return np.uint64(value ^ (value >> np.uint64(31)))


@nb.njit(cache=True)
def counter_draw(root: np.uint64, event: int, substream: int) -> np.uint64:
    address = np.uint64(event) * np.uint64(0xD2B74407B1CE6E93)
    address ^= np.uint64(substream) * np.uint64(0xCA5A826395121157)
    return splitmix64(np.uint64(root ^ address))


@nb.njit(cache=True)
def bounded(draw: np.uint64, bound: int) -> int:
    """Map the high 32 bits into ``[0, bound)`` by fixed multiply-high."""

    if bound <= 0:
        return 0
    high = draw >> np.uint64(32)
    return int((high * np.uint64(bound)) >> np.uint64(32))


@nb.njit(cache=True)
def systematic_request(
    event: int,
    denominator: int,
    swap_numerator: int,
    memory_numerator: int,
    changed_phase: int,
    swap_phase: int,
) -> int:
    """Return 0/1/2 for no requested change/swap/memory update.

    A randomized-phase systematic quota has exact changed/unchanged totals over
    every complete ``denominator``-event block and changed-count discrepancy
    below one over a prefix.  The nested swap/memory allocation can add one
    further count of subtype discrepancy before feasibility fallback.
    """

    if denominator <= 0:
        return 0
    changed = swap_numerator + memory_numerator
    if changed <= 0:
        return 0
    before = (event * changed + changed_phase) // denominator
    after = ((event + 1) * changed + changed_phase) // denominator
    if after == before:
        return 0
    if swap_numerator <= 0:
        return 2
    if memory_numerator <= 0:
        return 1
    swap_before = (before * swap_numerator + swap_phase) // changed
    swap_after = ((before + 1) * swap_numerator + swap_phase) // changed
    return 1 if swap_after > swap_before else 2


@nb.njit(cache=True)
def _select_category_edge(
    base: int,
    slots: int,
    category: int,
    decision: np.ndarray,
    changed: np.ndarray,
    kind: np.ndarray,
    draw: np.uint64,
) -> int:
    count = 0
    for slot in range(slots):
        edge = base + slot
        matches = (
            (category == 0 and not (decision[edge] == 1 and changed[edge] == 1))
            or (
                category > 0
                and decision[edge] == 1
                and changed[edge] == 1
                and kind[edge] == category
            )
        )
        if matches:
            count += 1
    if count == 0:
        return -1
    wanted = bounded(draw, count)
    seen = 0
    for slot in range(slots):
        edge = base + slot
        matches = (
            (category == 0 and not (decision[edge] == 1 and changed[edge] == 1))
            or (
                category > 0
                and decision[edge] == 1
                and changed[edge] == 1
                and kind[edge] == category
            )
        )
        if matches:
            if seen == wanted:
                return edge
            seen += 1
    return -1


@nb.njit(cache=True)
def _select_greedy_edge(
    base: int,
    slots: int,
    metric: int,
    deltas: np.ndarray,
    draw: np.uint64,
) -> int:
    best = np.int64(1 << 60)
    ties = 0
    for slot in range(slots):
        edge = base + slot
        value = deltas[metric, edge]
        if value < best:
            best = value
            ties = 1
        elif value == best:
            ties += 1
    wanted = bounded(draw, ties)
    seen = 0
    for slot in range(slots):
        edge = base + slot
        if deltas[metric, edge] == best:
            if seen == wanted:
                return edge
            seen += 1
    return base


@nb.njit(cache=True)
def _select_open_loop_edge(
    base: int,
    slots: int,
    opportunity_weights: np.ndarray,
    draw: np.uint64,
) -> int:
    total = 0
    for slot in range(slots):
        total += int(opportunity_weights[slot])
    wanted = bounded(draw, total)
    cumulative = 0
    for slot in range(slots):
        cumulative += int(opportunity_weights[slot])
        if wanted < cumulative:
            return base + slot
    return base + slots - 1


@nb.njit(cache=True)
def simulate_family(
    start_states: np.ndarray,
    roots: np.ndarray,
    target_denominators: np.ndarray,
    target_swaps: np.ndarray,
    target_memory: np.ndarray,
    start_levels: np.ndarray,
    slots: int,
    successors: np.ndarray,
    terminal: np.ndarray,
    opportunity_weights: np.ndarray,
    decision: np.ndarray,
    changed: np.ndarray,
    kind: np.ndarray,
    costs: np.ndarray,
    deltas: np.ndarray,
    maximum_events: int,
    checkpoint_event: int = 2048,
) -> tuple[np.ndarray, ...]:
    """Simulate all seven nulls for every supplied family start.

    Arrays are returned in ``(run, null[, metric/cost])`` order.  Cost columns
    are reads, comparisons, no-ops, rejections, memory updates, accepted swaps,
    and displaced cells; activations and proposals equal the event count.
    """

    run_count = len(start_states)
    null_count = len(NULL_NAMES)
    final_states = np.empty((run_count, null_count), dtype=np.int64)
    stop_codes = np.empty((run_count, null_count), dtype=np.uint8)
    event_counts = np.zeros((run_count, null_count), dtype=np.int32)
    ledgers = np.zeros((run_count, null_count, costs.shape[0]), dtype=np.int64)
    peaks = np.empty((run_count, null_count, 4), dtype=np.int16)
    finals = np.empty((run_count, null_count, 4), dtype=np.int16)
    worsening = np.zeros((run_count, null_count, 4), dtype=np.int32)
    requested = np.zeros((run_count, null_count, 3), dtype=np.int32)
    deficits = np.zeros((run_count, null_count, 3), dtype=np.int32)
    selected = np.zeros((run_count, null_count, 3), dtype=np.int32)
    fingerprints = np.zeros((run_count, null_count), dtype=np.uint64)
    raw_fingerprints = np.zeros((run_count, null_count), dtype=np.uint64)
    checkpoint_states = np.full((run_count, null_count), -1, dtype=np.int64)
    checkpoint_fingerprints = np.zeros((run_count, null_count), dtype=np.uint64)
    checkpoint_ledgers = np.zeros(
        (run_count, null_count, costs.shape[0]), dtype=np.int64
    )

    for run in range(run_count):
        for null_code in range(null_count):
            state = int(start_states[run])
            levels = start_levels[run].copy()
            peak = levels.copy()
            root = roots[run, null_code]
            fingerprint = splitmix64(root ^ np.uint64(0x5354415445))
            raw_fingerprint = splitmix64(root ^ np.uint64(0x524157))
            denominator = int(target_denominators[run])
            swap_target = int(target_swaps[run])
            memory_target = int(target_memory[run])
            changed_total = swap_target + memory_target
            changed_phase = bounded(counter_draw(root, 0, 41), denominator) if denominator > 0 else 0
            swap_phase = bounded(counter_draw(root, 0, 43), changed_total) if changed_total > 0 else 0
            stop = int(terminal[state])
            event = 0
            while stop == 0 and event < maximum_events:
                base = state * slots
                raw = counter_draw(root, event, 1)
                raw_fingerprint = splitmix64(raw_fingerprint ^ raw ^ np.uint64(event))
                if null_code == 0:
                    edge = base + bounded(raw, slots)
                elif null_code == 1:
                    request = systematic_request(
                        event,
                        denominator,
                        swap_target,
                        memory_target,
                        changed_phase,
                        swap_phase,
                    )
                    request_index = 2 if request == 0 else request - 1
                    requested[run, null_code, request_index] += 1
                    edge = _select_category_edge(
                        base,
                        slots,
                        request,
                        decision,
                        changed,
                        kind,
                        counter_draw(root, event, 7),
                    )
                    if edge < 0:
                        deficits[run, null_code, request_index] += 1
                    if edge < 0:
                        edge = base + bounded(counter_draw(root, event, 11), slots)
                elif null_code == 2:
                    edge = _select_open_loop_edge(
                        base, slots, opportunity_weights, raw
                    )
                else:
                    edge = _select_greedy_edge(
                        base,
                        slots,
                        null_code - 3,
                        deltas,
                        counter_draw(root, event, 17),
                    )

                edge = int(edge)

                if decision[edge] == 1 and changed[edge] == 1:
                    if kind[edge] == 1:
                        selected[run, null_code, 0] += 1
                    elif kind[edge] == 2:
                        selected[run, null_code, 1] += 1
                else:
                    selected[run, null_code, 2] += 1
                for cost in range(costs.shape[0]):
                    ledgers[run, null_code, cost] += costs[cost, edge]
                for metric in range(4):
                    delta = int(deltas[metric, edge])
                    levels[metric] += delta
                    if delta > 0:
                        worsening[run, null_code, metric] += 1
                    if levels[metric] > peak[metric]:
                        peak[metric] = levels[metric]
                state = int(successors[edge])
                fingerprint = splitmix64(
                    fingerprint
                    ^ np.uint64(edge)
                    ^ np.uint64(state)
                    ^ np.uint64(event) * np.uint64(0x9E3779B97F4A7C15)
                )
                event += 1
                if event == checkpoint_event:
                    checkpoint_states[run, null_code] = state
                    checkpoint_fingerprints[run, null_code] = fingerprint
                    for cost in range(costs.shape[0]):
                        checkpoint_ledgers[run, null_code, cost] = ledgers[
                            run, null_code, cost
                        ]
                stop = int(terminal[state])

            final_states[run, null_code] = state
            stop_codes[run, null_code] = stop if stop != 0 else STOP_ACTIVE_BUDGET
            event_counts[run, null_code] = event
            for metric in range(4):
                peaks[run, null_code, metric] = peak[metric]
                finals[run, null_code, metric] = levels[metric]
            fingerprints[run, null_code] = fingerprint
            raw_fingerprints[run, null_code] = raw_fingerprint

    return (
        final_states,
        stop_codes,
        event_counts,
        ledgers,
        peaks,
        finals,
        worsening,
        requested,
        deficits,
        selected,
        fingerprints,
        raw_fingerprints,
        checkpoint_states,
        checkpoint_fingerprints,
        checkpoint_ledgers,
    )


def opportunity_integer_weights(
    probability_numerators: np.ndarray,
    probability_denominators: np.ndarray,
) -> np.ndarray:
    """Return exact small integer weights for one S05 family opportunity set."""

    denominators = [int(value) for value in probability_denominators]
    common = int(np.lcm.reduce(np.asarray(denominators, dtype=np.int64)))
    weights = np.asarray(
        [
            int(numerator) * (common // int(denominator))
            for numerator, denominator in zip(
                probability_numerators, probability_denominators
            )
        ],
        dtype=np.int16,
    )
    if weights.min(initial=1) <= 0:
        raise ValueError("opportunity weights must be positive")
    if sum(
        int(numerator) / int(denominator)
        for numerator, denominator in zip(
            probability_numerators, probability_denominators
        )
    ) != 1.0:
        raise ValueError("S05 opportunity probabilities do not sum to one")
    return weights


def exact_start_status(classification: str) -> str:
    if classification in {
        "complete_start",
        "reachable_no_detour",
        "necessary_detour",
    }:
        return "reachable"
    return classification


def opportunity_rate_bin(event_count: int, accepted_swaps: int) -> str:
    if event_count == 0:
        return "terminal_zero_events"
    if accepted_swaps == 0:
        return "zero_movement"
    rate = accepted_swaps / event_count
    if rate <= 0.05:
        return "low_le_0.05"
    if rate <= 0.15:
        return "mid_le_0.15"
    if rate <= 0.30:
        return "high_le_0.30"
    return "very_high_gt_0.30"


def behavior_classification(
    *,
    completed: bool,
    stop_reason: str,
    exact_classification: str,
    excursion: int,
    minimum_excursion: int,
) -> str:
    """Apply the frozen mutually exclusive run-metric classification."""

    if exact_classification == "quiescent":
        return "quiescent_start"
    if exact_classification == "unreachable_active":
        return "exact_impossibility"
    if completed:
        if minimum_excursion > 0:
            return "successful_necessary_detour"
        if excursion > 0:
            return "successful_unnecessary_detour"
        return "successful_no_detour"
    if stop_reason == "quiescent":
        if minimum_excursion > 0:
            return "failed_quiescent_where_detour_required"
        return "failed_reached_quiescence"
    if stop_reason == "event_budget":
        if minimum_excursion > 0:
            return "censored_failed_where_detour_required"
        return "censored_reachable_no_detour"
    raise ValueError("unclassified S11 outcome")
