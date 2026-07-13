#!/usr/bin/env python3
"""Build and validate S03 transition-specification support artifacts.

This is deliberately not a simulator.  It emits a machine-readable semantic
contract, maps the S01 registry into that contract, and checks a small set of
hand-authored one-transition fixtures with an independent one-step oracle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


SCHEMA = "e01.s03.transition_contract.v1"
FROZEN_COMMIT = "1fd2bd5921c1f6b423a71f691d5189106a8a1020"

SECTIONS = {
    "EVIDENCE": "Evidence layers and authority",
    "STATE": "State, scenario, identity, and invariants",
    "OBS": "Observations and information boundaries",
    "PROPOSALS": "Proposal types and atomic commit",
    "POL-BUBBLE": "Bubble cell policy",
    "POL-INSERTION": "Insertion cell policy",
    "POL-SELECTION": "Selection cell policy and cursor memory",
    "FAULTS": "Normal, passive, and stuck fault semantics",
    "SCHEDULER": "Scheduler, randomness, and conflict semantics",
    "ORDER": "Direction, duplicates, and order predicates",
    "TERMINAL": "Completion, quiescence, and event-budget termination",
    "METRICS": "Paper, commit-aligned, and reference metric profiles",
    "COST": "Full operation ledger and historical projections",
    "SCENARIOS": "Scenario construction and condition profiles",
    "TRADITIONAL": "Traditional-controller boundary and missing history",
    "EVENTS": "Minimum event record required by the transition contract",
    "AMBIGUITIES": "Explicit ambiguity decisions and unresolved evidence",
    "CLAIMS": "Claim-registry condition mapping",
}

CONDITION_COLUMNS = [
    "array_size",
    "repetition_count",
    "value_distribution",
    "architecture",
    "algorithm",
    "algotype_composition",
    "sorting_direction",
    "fault_type",
    "fault_count",
    "placement_sampling",
    "stopping_rule",
    "metric_name",
    "metric_formula",
    "cost_definition",
]

AMBIGUITY_DECISIONS = {
    "AGGREGATION_BOUNDARY_UNDEFINED": {
        "referenceDecision": "Use adjacent edges i,i+1 and divide by n-1; retain commit-aligned divide-by-n as a named historical metric profile.",
        "residual": "The paper does not state its boundary convention.",
    },
    "AGGREGATION_NULL_CONFLICT": {
        "referenceDecision": "Use the exact composition-conditioned without-replacement same-neighbor expectation; never impose 0.5 on three-type mixtures.",
        "residual": "The paper's universal 0.5 statement remains contradictory.",
    },
    "ALGOTYPE_ASSIGNMENT_CONFLICT": {
        "referenceDecision": "Use exact declared counts followed by a seeded permutation; expose independent random assignment as a separate profile.",
        "residual": "Exact paper compositions are not recoverable for all panels.",
    },
    "COST_OPERATION_UNDEFINED": {
        "referenceDecision": "Record activations, reads, comparisons, proposals, rejections, memory updates, swaps, displacements, and conflict losses separately.",
        "residual": "Historical comparison instrumentation cannot be reconstructed from absent traces.",
    },
    "DG_DEFINITION_CONFLICT": {
        "referenceDecision": "Preserve the commit-aligned normalized net recovery formula as historical_dg_net; retain recovery/drop as a separately named sensitivity metric.",
        "residual": "S12 must preregister exact trajectory segmentation and compare both paper wordings.",
    },
    "DG_SIGN_CONFLICT": {
        "referenceDecision": "Do not choose a sign; retain both reported z signs as conflicting historical targets.",
        "residual": "Raw arrays are missing.",
    },
    "ERROR_BAR_TYPE_UNREPORTED": {
        "referenceDecision": "Report population SD only for a commit-aligned reconstruction and paired intervals for scientific comparisons; label both.",
        "residual": "Panel error-bar type is not attributable without raw plot data or author clarification.",
    },
    "F10_NUMERIC_NOT_RECOVERABLE": {
        "referenceDecision": "Keep the endpoint unavailable; do not digitize corrupted pixels or substitute regenerated values.",
        "residual": "Exact Figure 10 numerical targets are missing.",
    },
    "FAULT_PLACEMENT_UNREPORTED": {
        "referenceDecision": "Primary reference scenarios sample exactly f distinct identities without replacement; legacy_with_replacement is a separate profile.",
        "residual": "The paper does not identify the sampler and the commit can realize fewer than f distinct faults.",
    },
    "FIGURE_EXTRACTION_CORRUPT": {
        "referenceDecision": "Treat affected numeric values as unavailable unless supported by readable text or a separately provenance-labelled source.",
        "residual": "Supplied Figure 10 pixels are not adequate for exact extraction.",
    },
    "LINEARITY_NOT_TESTED": {
        "referenceDecision": "Treat linearity as a future estimand requiring an explicit fitted contrast and uncertainty, not as a transition assumption.",
        "residual": "No paper equivalence margin or fitted model exists.",
    },
    "METRIC_SCALE_AMBIGUOUS": {
        "referenceDecision": "Store all normalized metrics as fractions in [0,1] and multiply by 100 only in percent-labelled presentation fields.",
        "residual": "Paper formula/axis conventions differ.",
    },
    "NEGATIVE_CONTROL_CONTRADICTION": {
        "referenceDecision": "Define the control as distinct immutable labels attached to identical Bubble policies with exact composition; do not attribute paper control numbers to the frozen commit.",
        "residual": "No executable preserved control recipe or raw data exists.",
    },
    "NO_FAULT_STOP_UNREPORTED": {
        "referenceDecision": "Use exact non-strict completion for consensus-direction no-fault reference runs; preserve script-specific is_sorted as the commit profile.",
        "residual": "The claim-level paper rule is absent.",
    },
    "PANEL_VALUE_NOT_QUOTED": {
        "referenceDecision": "Retain the transcribed panel value as a lower-provenance target and never treat it as raw data.",
        "residual": "Original panel arrays are missing.",
    },
    "PASSIVE_INSERTION_CONTRADICTION": {
        "referenceDecision": "Keep each Figure 5 bar as its own target and reject the blanket direction when bar-level evidence reverses it.",
        "residual": "Replication is deferred; the contradiction is not resolved by semantics.",
    },
    "P_VALUE_CONFLICT": {
        "referenceDecision": "Retain both p-value strings as historical text; use recomputed, fully specified tests only on regenerated/reference data.",
        "residual": "No historical arrays exist for adjudication.",
    },
    "SAMPLE_SIZE_UNREPORTED": {
        "referenceDecision": "Require every executable scenario batch to declare N; never infer an unreported N from neighboring claims.",
        "residual": "The historical N remains unknown for affected claims.",
    },
    "SORTEDNESS_DUPLICATE_CONFLICT": {
        "referenceDecision": "Reference completion and Sortedness use non-strict order; the paper's strict printed score is retained as paper_sortedness_strict.",
        "residual": "Paper duplicate panels and printed formula remain inconsistent.",
    },
    "STOP_RULE_CONFLICT": {
        "referenceDecision": "Use ordered terminal precedence invariant_error, complete, quiescent, event_budget; expose two-unchanged-check and script rules only as historical profiles.",
        "residual": "The exact publication stop profile is not attributable.",
    },
    "TEST_SPECIFICATION_INCOMPLETE": {
        "referenceDecision": "Historical z values are targets only; future inference must declare estimand, pairing, variance, sidedness, and multiplicity family.",
        "residual": "Paper test construction is incomplete.",
    },
    "VARIABILITY_UNREPORTED": {
        "referenceDecision": "Do not impute variability; regenerated experiments must report declared uncertainty from complete run accounting.",
        "residual": "Historical variability is unavailable.",
    },
}


def contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "researchStepId": "S03",
        "frozenPublicCommit": FROZEN_COMMIT,
        "publicationSnapshotAssertion": False,
        "sections": [{"id": key, "title": value} for key, value in SECTIONS.items()],
        "evidenceLayers": {
            "paper": "Journal prose, equations, captions, and supplied figures; authoritative only for what was reported.",
            "frozen_public_commit": "Static behavior at the named commit; not asserted to be the publication snapshot.",
            "clean_room_reference": "Chosen deterministic semantics for future scientific simulation; departures are explicit decisions, not historical claims.",
        },
        "cellImmutableFields": ["cellId", "value", "algotype", "direction", "faultMode", "analysisLabel"],
        "stateFields": {
            "occupancy": "Length-n bijection from positions 0..n-1 to cellId.",
            "selectionCursor": "Per-Selection-cell integer cursor that travels with identity.",
            "activationCount": "Number of actor activations consumed, including no-ops and rejections.",
            "streamCounters": "Per-named-stream counter positions for counter-addressed draws.",
            "ledger": "Separate monotone operation counters.",
            "terminal": "Null or one of invariant_error, complete, quiescent, event_budget.",
        },
        "invariants": [
            "occupancy positions are unique and cover 0..n-1",
            "cell identities and immutable fields are conserved",
            "only atomic two-position swaps change occupancy",
            "only the owning Selection cell changes its cursor",
            "a stuck cell is never displaced",
            "ledger and activation counters never decrease",
        ],
        "directions": {
            "ascending": "Adjacent values are ordered when left <= right.",
            "descending": "Adjacent values are ordered when left >= right.",
            "duplicates": "Equality is ordered and never directly triggers a swap; identity stability is not guaranteed.",
        },
        "faultModes": {
            "normal": "May initiate and may be displaced.",
            "passive": "Cannot initiate; may be the target of and displaced by an accepted swap.",
            "stuck": "Cannot initiate and cannot be displaced; a swap targeting it is rejected.",
        },
        "proposalTypes": ["NoOp", "Swap", "MemoryUpdate"],
        "policies": {
            "Bubble": {
                "observation": "Own value/position/fault plus the selected adjacent position, its value/fault, and boundaries.",
                "choice": "Draw left or right with probability 1/2 even at a boundary.",
                "action": "Swap exactly when the selected adjacent pair is a strict inversion for the actor's direction.",
            },
            "Insertion": {
                "observation": "Own state, left neighbor, boundaries, and the full strict prefix; fault cells split the prefix into ordered segments.",
                "action": "If every prefix segment is direction-ordered, swap left exactly when actor/left-neighbor is a strict inversion.",
            },
            "Selection": {
                "observation": "Own state, cursor, boundaries, and the cell at cursor.",
                "cursor": "Initialize left for ascending and right for descending; advance toward the opposite boundary while target.value <= actor.value.",
                "action": "Swap with the cursor occupant when target.value > actor.value; a stuck target causes one cursor advance and no swap.",
            },
        },
        "scheduler": {
            "referenceDefault": "serial_counter_addressed",
            "actorPopulation": "all immutable cell identities, including faulty cells, to preserve paired streams",
            "activationDraw": "uniform cellId index from a counter-addressed stream",
            "bubbleChoiceDraw": "independent counter-addressed left/right draw",
            "historicalBoundary": "Frozen commit uses contending Python threads and one lock; OS timing is not represented as publication truth.",
        },
        "randomness": {
            "rootKey": "SHA256(ASCII('E01/RNG/v1') || 0x00 || uint128_be(masterSeed) || 0x00 || UTF8(scenarioId))",
            "block": "SHA256(rootKey || 0x00 || ASCII(streamName) || 0x00 || uint64_be(eventIndex) || uint32_be(drawIndex))",
            "uint64": "Interpret the first 8 block bytes as unsigned big-endian.",
            "unitInterval": "uint64 / 2^64, yielding [0,1).",
            "boundedInteger": "Reject uint64 >= floor(2^64/m)*m, increment drawIndex, then return uint64 mod m.",
            "streamNames": ["actor_activation", "bubble_side", "conflict_priority", "scenario_permutation"],
        },
        "conflicts": {
            "serialDefault": "At batch width one, conflicts are impossible.",
            "batchExtension": "Compute from one snapshot; proposals conflict when position resources overlap or the same actor has multiple state changes.",
            "resolution": "Lowest tuple (conflictPriorityUint64, actorId, targetPosition, proposalOrdinal) wins; accept only a maximal disjoint set in that order.",
        },
        "termination": {
            "precedence": ["invariant_error", "complete", "quiescent", "event_budget"],
            "complete": "All cells declare one common direction and occupancy is non-strictly ordered in it.",
            "quiescent": "No normal actor has any admissible Swap or MemoryUpdate over the full support of its policy choices.",
            "event_budget": "activationCount reaches maxActivations; no-op and rejected activations count.",
            "mixedDirection": "Never consensus-complete; it ends only by quiescence, budget, or invariant error.",
        },
        "metrics": {
            "referenceSortedness": "(1 + non-strict ordered adjacent pairs)/n for n>0",
            "paperSortednessStrict": "(1 + strict ordered adjacent pairs)/n for n>0",
            "monotonicityError": "Count strict adjacent inversions; equality is not an error.",
            "referenceAggregation": "same-Algotype adjacent edges/(n-1), n>1",
            "commitAggregation": "same-Algotype adjacent edges/n, terminal boundary contributes zero",
            "aggregationNull": "sum_k n_k(n_k-1)/(n(n-1)) under random permutation of fixed labels",
            "scale": "Store fractions; percent is presentation-only.",
        },
        "ledgerFields": [
            "activations",
            "observationReads",
            "valueComparisons",
            "proposals",
            "noOps",
            "rejections",
            "memoryUpdates",
            "acceptedSwaps",
            "displacedCells",
            "conflictLosses",
        ],
        "eventRequiredFields": [
            "schemaVersion",
            "scenarioId",
            "eventIndex",
            "actorId",
            "actorAlgotype",
            "actorDirection",
            "preStateHash",
            "observation",
            "proposal",
            "decision",
            "postStateHash",
            "ledgerDelta",
            "stopReason",
        ],
        "ambiguityDecisions": AMBIGUITY_DECISIONS,
    }


def cell(cell_id: str, value: int, policy: str, direction: str = "ascending", fault: str = "normal", cursor: int | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": cell_id,
        "value": value,
        "policy": policy,
        "direction": direction,
        "fault": fault,
    }
    if cursor is not None:
        item["cursor"] = cursor
    return item


def fixture_data() -> list[dict[str, Any]]:
    return [
        {
            "fixtureId": "T01",
            "title": "Bubble ascending right inversion",
            "pre": [cell("a", 2, "Bubble"), cell("b", 1, "Bubble")],
            "activation": {"actorId": "a", "neighborChoice": "right"},
            "expected": {"proposal": "Swap", "decision": "accepted", "postOrder": ["b", "a"]},
        },
        {
            "fixtureId": "T02",
            "title": "Bubble equality is not an inversion",
            "pre": [cell("a", 1, "Bubble"), cell("b", 1, "Bubble")],
            "activation": {"actorId": "a", "neighborChoice": "right"},
            "expected": {"proposal": "NoOp", "reason": "ordered_or_equal", "postOrder": ["a", "b"]},
        },
        {
            "fixtureId": "T03",
            "title": "Bubble descending left inversion",
            "pre": [cell("a", 1, "Bubble", "descending"), cell("b", 3, "Bubble", "descending")],
            "activation": {"actorId": "b", "neighborChoice": "left"},
            "expected": {"proposal": "Swap", "decision": "accepted", "postOrder": ["b", "a"]},
        },
        {
            "fixtureId": "T04",
            "title": "Bubble boundary draw consumes a no-op",
            "pre": [cell("a", 2, "Bubble"), cell("b", 1, "Bubble")],
            "activation": {"actorId": "a", "neighborChoice": "left"},
            "expected": {"proposal": "NoOp", "reason": "boundary", "postOrder": ["a", "b"]},
        },
        {
            "fixtureId": "T05",
            "title": "Insertion ordered-prefix swap",
            "pre": [cell("a", 1, "Insertion"), cell("b", 3, "Insertion"), cell("c", 2, "Insertion")],
            "activation": {"actorId": "c"},
            "expected": {"proposal": "Swap", "decision": "accepted", "postOrder": ["a", "c", "b"]},
        },
        {
            "fixtureId": "T06",
            "title": "Insertion unordered prefix blocks movement",
            "pre": [cell("a", 3, "Insertion"), cell("b", 1, "Insertion"), cell("c", 2, "Insertion")],
            "activation": {"actorId": "c"},
            "expected": {"proposal": "NoOp", "reason": "prefix_not_ordered", "postOrder": ["a", "b", "c"]},
        },
        {
            "fixtureId": "T07",
            "title": "Passive target can be displaced",
            "pre": [cell("p", 2, "Insertion", fault="passive"), cell("a", 1, "Insertion")],
            "activation": {"actorId": "a"},
            "expected": {"proposal": "Swap", "decision": "accepted", "postOrder": ["a", "p"]},
        },
        {
            "fixtureId": "T08",
            "title": "Stuck target rejects an otherwise legal swap",
            "pre": [cell("s", 2, "Insertion", fault="stuck"), cell("a", 1, "Insertion")],
            "activation": {"actorId": "a"},
            "expected": {"proposal": "Swap", "decision": "rejected_target_stuck", "postOrder": ["s", "a"]},
        },
        {
            "fixtureId": "T09",
            "title": "Selection equality advances cursor without swapping",
            "pre": [cell("t", 1, "Bubble"), cell("a", 1, "Selection", cursor=0)],
            "activation": {"actorId": "a"},
            "expected": {"proposal": "MemoryUpdate", "newCursor": 1, "postOrder": ["t", "a"]},
        },
        {
            "fixtureId": "T10",
            "title": "Selection swaps a smaller actor into its cursor",
            "pre": [cell("t", 3, "Bubble"), cell("a", 1, "Selection", cursor=0)],
            "activation": {"actorId": "a"},
            "expected": {"proposal": "Swap", "decision": "accepted", "postOrder": ["a", "t"]},
        },
        {
            "fixtureId": "T11",
            "title": "Selection skips a stuck target by cursor update",
            "pre": [cell("s", 3, "Bubble", fault="stuck"), cell("a", 1, "Selection", cursor=0)],
            "activation": {"actorId": "a"},
            "expected": {"proposal": "MemoryUpdate", "newCursor": 1, "postOrder": ["s", "a"]},
        },
        {
            "fixtureId": "T12",
            "title": "Selection descending starts at the right boundary",
            "pre": [cell("a", 1, "Selection", "descending", cursor=2), cell("m", 2, "Bubble", "descending"), cell("t", 3, "Bubble", "descending")],
            "activation": {"actorId": "a"},
            "expected": {"proposal": "Swap", "decision": "accepted", "postOrder": ["t", "m", "a"]},
        },
        {
            "fixtureId": "T13",
            "title": "Passive actor never initiates",
            "pre": [cell("a", 2, "Bubble", fault="passive"), cell("b", 1, "Bubble")],
            "activation": {"actorId": "a", "neighborChoice": "right"},
            "expected": {"proposal": "NoOp", "reason": "actor_fault", "postOrder": ["a", "b"]},
        },
        {
            "fixtureId": "T14",
            "title": "Batch conflict accepts the lowest priority disjoint proposal",
            "pre": [cell("a", 3, "Bubble"), cell("b", 2, "Bubble"), cell("c", 1, "Bubble")],
            "batch": [
                {"actorId": "a", "neighborChoice": "right", "priority": 9},
                {"actorId": "c", "neighborChoice": "left", "priority": 4},
            ],
            "expected": {"winnerActorId": "c", "loserActorIds": ["a"], "postOrder": ["a", "c", "b"]},
        },
        {
            "fixtureId": "T15",
            "title": "Duplicate consensus state is complete under non-strict order",
            "pre": [cell("a", 1, "Bubble"), cell("b", 1, "Insertion"), cell("c", 2, "Selection", cursor=0)],
            "terminal": {"activationCount": 0, "maxActivations": 10},
            "expected": {"stopReason": "complete", "referenceSortedness": 1.0, "paperSortednessStrict": 2 / 3},
        },
        {
            "fixtureId": "T16",
            "title": "Mixed directions are not consensus-complete and can hit budget",
            "pre": [cell("a", 1, "Bubble", "ascending"), cell("b", 2, "Bubble", "descending")],
            "terminal": {"activationCount": 5, "maxActivations": 5},
            "expected": {"stopReason": "event_budget"},
        },
        {
            "fixtureId": "T17",
            "title": "A stuck barrier can make a state quiescent",
            "pre": [cell("s", 2, "Insertion", fault="stuck"), cell("a", 1, "Insertion")],
            "terminal": {"activationCount": 1, "maxActivations": 10},
            "expected": {"stopReason": "quiescent"},
        },
    ]


def position(cells: list[dict[str, Any]], actor_id: str) -> int:
    for i, item in enumerate(cells):
        if item["id"] == actor_id:
            return i
    raise ValueError(f"unknown actor {actor_id}")


def ordered_pair(left: int, right: int, direction: str, strict: bool = False) -> bool:
    if direction == "ascending":
        return left < right if strict else left <= right
    return left > right if strict else left >= right


def counter_u64(master_seed: int, scenario_id: str, stream_name: str, event_index: int, draw_index: int) -> int:
    if not (0 <= master_seed < 2**128):
        raise ValueError("master_seed must be an unsigned 128-bit integer")
    if not (0 <= event_index < 2**64 and 0 <= draw_index < 2**32):
        raise ValueError("counter out of range")
    if not stream_name.isascii():
        raise ValueError("stream_name must be ASCII")
    root = hashlib.sha256(
        b"E01/RNG/v1\x00" + master_seed.to_bytes(16, "big") + b"\x00" + scenario_id.encode("utf-8")
    ).digest()
    block = hashlib.sha256(
        root
        + b"\x00"
        + stream_name.encode("ascii")
        + b"\x00"
        + event_index.to_bytes(8, "big")
        + draw_index.to_bytes(4, "big")
    ).digest()
    return int.from_bytes(block[:8], "big")


def prefix_ordered(cells: list[dict[str, Any]], end: int, direction: str) -> bool:
    segment: list[int] = []
    for item in cells[:end]:
        if item["fault"] != "normal":
            segment = []
            continue
        if segment and not ordered_pair(segment[-1], item["value"], direction):
            return False
        segment.append(item["value"])
    return True


def proposal(cells: list[dict[str, Any]], activation: dict[str, Any]) -> dict[str, Any]:
    actor_id = activation["actorId"]
    pos = position(cells, actor_id)
    actor = cells[pos]
    if actor["fault"] != "normal":
        return {"type": "NoOp", "reason": "actor_fault"}
    if actor["policy"] == "Bubble":
        choice = activation["neighborChoice"]
        target = pos + (1 if choice == "right" else -1)
        if target < 0 or target >= len(cells):
            return {"type": "NoOp", "reason": "boundary"}
        target_value = cells[target]["value"]
        inversion = (
            actor["value"] > target_value
            if (actor["direction"], choice) in {("ascending", "right"), ("descending", "left")}
            else actor["value"] < target_value
        )
        if not inversion:
            return {"type": "NoOp", "reason": "ordered_or_equal"}
        return {"type": "Swap", "actorId": actor_id, "targetPosition": target}
    if actor["policy"] == "Insertion":
        if pos == 0:
            return {"type": "NoOp", "reason": "boundary"}
        if not prefix_ordered(cells, pos, actor["direction"]):
            return {"type": "NoOp", "reason": "prefix_not_ordered"}
        left_value = cells[pos - 1]["value"]
        inversion = actor["value"] < left_value if actor["direction"] == "ascending" else actor["value"] > left_value
        if not inversion:
            return {"type": "NoOp", "reason": "ordered_or_equal"}
        return {"type": "Swap", "actorId": actor_id, "targetPosition": pos - 1}
    if actor["policy"] == "Selection":
        cursor = actor["cursor"]
        if cursor < 0 or cursor >= len(cells):
            return {"type": "NoOp", "reason": "cursor_exhausted"}
        if cursor == pos:
            return {"type": "NoOp", "reason": "at_cursor"}
        target = cells[cursor]
        delta = 1 if actor["direction"] == "ascending" else -1
        if target["fault"] == "stuck" or target["value"] <= actor["value"]:
            return {"type": "MemoryUpdate", "actorId": actor_id, "newCursor": cursor + delta}
        return {"type": "Swap", "actorId": actor_id, "targetPosition": cursor}
    raise ValueError(f"unknown policy {actor['policy']}")


def commit(cells: list[dict[str, Any]], prop: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    post = deepcopy(cells)
    if prop["type"] == "NoOp":
        return post, "no_op"
    if prop["type"] == "MemoryUpdate":
        post[position(post, prop["actorId"])]["cursor"] = prop["newCursor"]
        return post, "accepted"
    actor_pos = position(post, prop["actorId"])
    target_pos = prop["targetPosition"]
    if post[target_pos]["fault"] == "stuck":
        return post, "rejected_target_stuck"
    post[actor_pos], post[target_pos] = post[target_pos], post[actor_pos]
    return post, "accepted"


def state_change_exists(cells: list[dict[str, Any]]) -> bool:
    for actor in cells:
        if actor["fault"] != "normal":
            continue
        activations = [{"actorId": actor["id"]}]
        if actor["policy"] == "Bubble":
            activations = [
                {"actorId": actor["id"], "neighborChoice": "left"},
                {"actorId": actor["id"], "neighborChoice": "right"},
            ]
        for activation in activations:
            prop = proposal(cells, activation)
            if prop["type"] == "MemoryUpdate":
                return True
            if prop["type"] == "Swap" and cells[prop["targetPosition"]]["fault"] != "stuck":
                return True
    return False


def complete(cells: list[dict[str, Any]]) -> bool:
    directions = {item["direction"] for item in cells}
    if len(directions) != 1:
        return False
    direction = next(iter(directions))
    return all(ordered_pair(cells[i]["value"], cells[i + 1]["value"], direction) for i in range(len(cells) - 1))


def terminal(cells: list[dict[str, Any]], activation_count: int, max_activations: int) -> str | None:
    if complete(cells):
        return "complete"
    if not state_change_exists(cells):
        return "quiescent"
    if activation_count >= max_activations:
        return "event_budget"
    return None


def validate_fixtures(fixtures: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for fixture in fixtures:
        fid = fixture["fixtureId"]
        if fid in seen:
            errors.append(f"duplicate fixture ID: {fid}")
        seen.add(fid)
        cells = fixture["pre"]
        identities = [item["id"] for item in cells]
        if len(identities) != len(set(identities)):
            errors.append(f"{fid}: duplicate cell IDs")
            continue
        expected = fixture["expected"]
        if "activation" in fixture:
            prop = proposal(cells, fixture["activation"])
            if prop["type"] != expected["proposal"]:
                errors.append(f"{fid}: proposal {prop['type']} != {expected['proposal']}")
            if expected.get("reason") != prop.get("reason"):
                errors.append(f"{fid}: reason {prop.get('reason')} != {expected.get('reason')}")
            post, decision = commit(cells, prop)
            if expected.get("decision") and decision != expected["decision"]:
                errors.append(f"{fid}: decision {decision} != {expected['decision']}")
            if expected.get("newCursor") is not None:
                actual_cursor = post[position(post, fixture["activation"]["actorId"])].get("cursor")
                if actual_cursor != expected["newCursor"]:
                    errors.append(f"{fid}: cursor {actual_cursor} != {expected['newCursor']}")
            if [item["id"] for item in post] != expected["postOrder"]:
                errors.append(f"{fid}: post order mismatch")
            by_id_pre = {item["id"]: item for item in cells}
            by_id_post = {item["id"]: item for item in post}
            for cell_id in by_id_pre:
                for key in ["value", "policy", "direction", "fault"]:
                    if by_id_pre[cell_id][key] != by_id_post[cell_id][key]:
                        errors.append(f"{fid}: immutable {key} changed for {cell_id}")
        elif "batch" in fixture:
            ranked = sorted(fixture["batch"], key=lambda item: (item["priority"], item["actorId"]))
            proposals = [(item, proposal(cells, item)) for item in ranked]
            used: set[int] = set()
            winners: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for activation, prop in proposals:
                actor_pos = position(cells, activation["actorId"])
                resources = {actor_pos, prop["targetPosition"]} if prop["type"] == "Swap" else {actor_pos}
                if resources.isdisjoint(used):
                    winners.append((activation, prop))
                    used.update(resources)
            if [item[0]["actorId"] for item in winners] != [expected["winnerActorId"]]:
                errors.append(f"{fid}: conflict winner mismatch")
            post, decision = commit(cells, winners[0][1])
            if decision != "accepted" or [item["id"] for item in post] != expected["postOrder"]:
                errors.append(f"{fid}: conflict commit mismatch")
        else:
            reason = terminal(cells, fixture["terminal"]["activationCount"], fixture["terminal"]["maxActivations"])
            if reason != expected["stopReason"]:
                errors.append(f"{fid}: terminal {reason} != {expected['stopReason']}")
            if "referenceSortedness" in expected:
                direction = cells[0]["direction"]
                ref = (1 + sum(ordered_pair(cells[i]["value"], cells[i + 1]["value"], direction) for i in range(len(cells) - 1))) / len(cells)
                paper = (1 + sum(ordered_pair(cells[i]["value"], cells[i + 1]["value"], direction, strict=True) for i in range(len(cells) - 1))) / len(cells)
                if abs(ref - expected["referenceSortedness"]) > 1e-12 or abs(paper - expected["paperSortednessStrict"]) > 1e-12:
                    errors.append(f"{fid}: sortedness mismatch")
    return errors


def sections_for(row: dict[str, str]) -> list[str]:
    sections = {"EVIDENCE", "STATE", "SCENARIOS", "CLAIMS"}
    algorithm = row.get("algorithm", "")
    for policy in ["Bubble", "Insertion", "Selection"]:
        if policy.lower() in algorithm.lower():
            sections.add(f"POL-{policy.upper()}")
    architecture = row.get("architecture", "")
    if "traditional" in architecture:
        sections.add("TRADITIONAL")
    if "cell_view" in architecture or "cell-view" in row.get("claim_text", "").lower():
        sections.update({"OBS", "PROPOSALS", "SCHEDULER", "EVENTS"})
    if row.get("fault_type") not in {"", "none", "not applicable", "<NA>"} or row.get("fault_count") not in {"", "0", "<NA>"}:
        sections.add("FAULTS")
    if row.get("sorting_direction") not in {"", "<NA>"} or "sorted" in row.get("metric_name", "").lower():
        sections.add("ORDER")
    if row.get("stopping_rule") and "not applicable" not in row["stopping_rule"]:
        sections.add("TERMINAL")
    if row.get("metric_name") or row.get("metric_formula"):
        sections.add("METRICS")
    if row.get("cost_definition") and "not applicable" not in row["cost_definition"]:
        sections.add("COST")
    if row.get("ambiguity_codes"):
        sections.add("AMBIGUITIES")
    return sorted(sections)


def unresolved_for(row: dict[str, str]) -> str:
    issues: list[str] = []
    if "traditional" in row.get("architecture", ""):
        issues.append("traditional generators and fault behavior absent from public history")
    if row.get("evidence_status") in {"not_recoverable", "contradictory", "partially_specified"}:
        issues.append(f"paper evidence status={row['evidence_status']}")
    if row.get("ambiguity_codes"):
        issues.append("paper ambiguities=" + row["ambiguity_codes"])
    return "; ".join(issues) if issues else "none beyond general publication-snapshot uncertainty"


def map_claims(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    mapped: list[dict[str, str]] = []
    for row in rows:
        sections = sections_for(row)
        if row["evidence_status"] == "not_recoverable":
            status = "mapped_explicit_unavailable_outcome"
        elif row["evidence_status"] == "not_empirical":
            status = "mapped_context_only"
        elif unresolved_for(row).startswith("none"):
            status = "mapped"
        else:
            status = "mapped_with_historical_constraints"
        condition = {key: row.get(key, "") or "not reported" for key in CONDITION_COLUMNS}
        mapped.append(
            {
                "claimId": row["claim_id"],
                "figure": row["figure"],
                "panel": row["panel"],
                "paperEvidenceStatus": row["evidence_status"],
                "mappingStatus": status,
                "specSectionIds": "|".join(sections),
                "conditionJson": json.dumps(condition, sort_keys=True, separators=(",", ":")),
                "referenceProfile": "clean_room_reference_v1",
                "historicalProfile": "frozen_public_commit_static" if "cell_view" in row.get("architecture", "") else "missing_or_cross_architecture",
                "unresolvedHistoricalEvidence": unresolved_for(row),
                "ambiguityCodes": row.get("ambiguity_codes", ""),
            }
        )
    return mapped


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_coverage(rows: list[dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    section_by_column = {
        "array_size": "SCENARIOS",
        "repetition_count": "SCENARIOS",
        "value_distribution": "SCENARIOS",
        "architecture": "EVIDENCE",
        "algorithm": "SCENARIOS",
        "algotype_composition": "SCENARIOS",
        "sorting_direction": "ORDER",
        "fault_type": "FAULTS",
        "fault_count": "FAULTS",
        "placement_sampling": "SCENARIOS",
        "stopping_rule": "TERMINAL",
        "metric_name": "METRICS",
        "metric_formula": "METRICS",
        "cost_definition": "COST",
    }
    for column in CONDITION_COLUMNS:
        values = sorted({row.get(column, "") or "not reported" for row in rows})
        result[column] = {"specSectionId": section_by_column[column], "distinctValues": values, "valueCount": len(values)}
    return result


def validate_contract(value: dict[str, Any], claim_rows: list[dict[str, str]], mapped: list[dict[str, str]], fixtures: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[str]]:
    checks: list[dict[str, str]] = []
    errors: list[str] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"check": name, "status": "pass" if condition else "fail", "detail": detail})
        if not condition:
            errors.append(f"{name}: {detail}")

    claim_ids = [row["claim_id"] for row in claim_rows]
    check("claim_registry_rows", len(claim_rows) == 118, f"{len(claim_rows)} rows (expected 118)")
    check("claim_registry_unique_ids", len(claim_ids) == len(set(claim_ids)), f"{len(set(claim_ids))}/{len(claim_ids)} unique")
    check("one_mapping_per_claim", len(mapped) == len(claim_rows) and {row['claimId'] for row in mapped} == set(claim_ids), f"{len(mapped)} mappings")
    valid_sections = set(SECTIONS)
    invalid = sorted({section for row in mapped for section in row["specSectionIds"].split("|") if section not in valid_sections})
    check("mapping_section_references", not invalid, f"invalid sections={invalid}")
    empty_conditions = [row["claimId"] for row in mapped if any(value in {None, ""} for value in json.loads(row["conditionJson"]).values())]
    check("claim_condition_explicitness", not empty_conditions, f"rows with empty conditions={len(empty_conditions)}")
    registry_codes = {code for row in claim_rows for code in row.get("ambiguity_codes", "").split("|") if code}
    missing_codes = sorted(registry_codes - set(AMBIGUITY_DECISIONS))
    check("ambiguity_decision_coverage", not missing_codes, f"{len(registry_codes)} codes covered; missing={missing_codes}")
    required_contract_keys = {"stateFields", "invariants", "faultModes", "proposalTypes", "policies", "scheduler", "randomness", "conflicts", "termination", "metrics", "ledgerFields"}
    check("critical_term_coverage", required_contract_keys <= set(value), f"required keys present={sorted(required_contract_keys <= set(value) and required_contract_keys or [])}")
    fixture_errors = validate_fixtures(fixtures)
    check("toy_fixture_oracle", not fixture_errors, f"{len(fixtures)} fixtures checked; errors={fixture_errors}")
    return checks, errors


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_manifest(output_dir: Path, code_commit: str | None) -> None:
    files = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append({"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schema": "e01.s03.artifact_manifest.v1",
        "researchStepId": "S03",
        "frozenPublicCommit": FROZEN_COMMIT,
        "publicationSnapshotAssertion": False,
        "repositoryCodeCommit": code_commit,
        "artifacts": files,
    }
    (output_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--finalize-manifest", action="store_true")
    parser.add_argument("--code-commit")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.finalize_manifest:
        finalize_manifest(args.output_dir, args.code_commit)
        return 0
    with args.claim_registry.open(newline="", encoding="utf-8") as handle:
        claim_rows = list(csv.DictReader(handle))
    value = contract()
    fixtures = fixture_data()
    mapped = map_claims(claim_rows)
    checks, errors = validate_contract(value, claim_rows, mapped, fixtures)
    (args.output_dir / "transition_contract.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "toy_fixtures.json").write_text(json.dumps({"schema": "e01.s03.toy_fixtures.v1", "researchStepId": "S03", "fixtures": fixtures}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "ambiguity_decisions.json").write_text(json.dumps({"schema": "e01.s03.ambiguity_decisions.v1", "researchStepId": "S03", "decisions": AMBIGUITY_DECISIONS}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "claim_condition_coverage.json").write_text(json.dumps({"schema": "e01.s03.claim_condition_coverage.v1", "researchStepId": "S03", "columns": condition_coverage(claim_rows)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "claim_specification_map.csv", mapped)
    summary = {
        "schema": "e01.s03.validation.v1",
        "researchStepId": "S03",
        "stepNumber": 3,
        "success": not errors,
        "validationResult": "PASS" if not errors else "FAIL",
        "counts": {
            "claimRegistryRows": len(claim_rows),
            "claimMappings": len(mapped),
            "conditionColumns": len(CONDITION_COLUMNS),
            "ambiguityCodesCovered": len(AMBIGUITY_DECISIONS),
            "specSections": len(SECTIONS),
            "toyFixtures": len(fixtures),
        },
        "checks": checks,
        "errors": errors,
    }
    (args.output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
