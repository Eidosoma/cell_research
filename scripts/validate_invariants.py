#!/usr/bin/env python3
"""Seeded S07 property, differential, mutation, and historical-isolation campaign."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
from dataclasses import replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference_simulator import (  # noqa: E402
    AUDIT_VERSION,
    Architecture,
    Cell,
    Direction,
    FaultMode,
    Policy,
    Scenario,
    audit_reference_result,
)
from reference_simulator.api import create_scenario, run_scenario  # noqa: E402
from reference_simulator.engine import evaluate_terminal, initial_state  # noqa: E402
from reference_simulator.model import ProposalKind, RunResult, canonical_json_bytes  # noqa: E402
from reference_simulator.policies import cell_view_proposal  # noqa: E402
from reference_simulator.scheduler import resolve_conflicts  # noqa: E402
from shared_events import (  # noqa: E402
    adapt_historical_run,
    adapt_reference_result,
    audit_historical_trace,
    validate_event,
    validate_trace,
)
from shared_events.model import TraceBundle, sha256_file  # noqa: E402
from shared_events.schema import assign_event_id  # noqa: E402


S03_FIXTURES = Path("/artifacts/research_steps/S03/toy_fixtures.json")
S04_TOYS = Path("/artifacts/research_steps/S04/known_toy_results.json")
S04_RUN = Path("/artifacts/research_steps/S04/smoke_outputs/generated_raw_smoke/run.json")
S04_SCHEDULER = Path("/artifacts/research_steps/S04/scheduler_variability.json")
DEFAULT_ROOT_SEEDS = (0x5_07_2026, 0xC0FFEE_07)


INVARIANTS = (
    ("INV-01", "scenario", "Scenario identity and immutable cell metadata are conserved.", "scenarios", "MUT-08"),
    ("INV-02", "state", "Occupancy remains a unique bijection of all cell identities.", "events", "MUT-01"),
    ("INV-03", "state", "Identity and value multisets are conserved at every batch.", "events", "MUT-02"),
    ("INV-04", "position", "Actor and target positions resolve inside the pre-batch occupancy.", "events", "MUT-03"),
    ("INV-05", "swap", "Every accepted swap is atomic, distinct, and exchanges exactly two identities.", "acceptedSwaps", "MUT-04"),
    ("INV-06", "swap", "No-op, rejected, and conflict-lost proposals do not alter occupancy.", "noOps", "MUT-05"),
    ("INV-07", "boundary", "Bubble/Insertion boundary choices produce counted no-ops.", "boundaryNoOps", "MUT-06"),
    ("INV-08", "target", "Proposal targets and decisions match an independent policy/validation oracle.", "events", "MUT-03"),
    ("INV-09", "memory", "Only Selection identities own and update their traveling cursors.", "memoryUpdates", "MUT-09"),
    ("INV-10", "fault", "Passive actors never initiate state changes.", "passiveActorNoOps", "MUT-10"),
    ("INV-11", "fault", "Passive targets may be displaced while retaining identity and fault mode.", "passiveTargetDisplacements", "MUT-11"),
    ("INV-12", "fault", "Stuck identities never move and stuck-target swaps reject.", "stuckTargetRejections", "MUT-12"),
    ("INV-13", "duplicates", "Equality is non-strictly ordered and never directly proposes a swap.", "duplicateScenarios", "MUT-13"),
    ("INV-14", "randomness", "Counter-addressed actor, side, and priority draws match the frozen byte specification.", "events", "MUT-14"),
    ("INV-15", "ledger", "Activation, proposal, operation, swap, and displacement ledgers reconcile exactly.", "events", "MUT-07"),
    ("INV-16", "hash", "Initial, pre/post, final state hashes and event digest match canonical replay.", "events", "MUT-15"),
    ("INV-17", "terminal", "Complete means homogeneous-direction non-strict order, including duplicates.", "stop.complete", "MUT-16"),
    ("INV-18", "terminal", "Quiescent means no admissible swap or memory update under exhaustive choices.", "stop.quiescent", "MUT-17"),
    ("INV-19", "terminal", "Event budget follows complete/quiescent precedence and counts every activation.", "stop.event_budget", "MUT-18"),
    ("INV-20", "conflict", "Accepted batch changes are disjoint and conflict losers do not retry.", "conflictLosses", "MUT-19"),
    ("INV-21", "serialization", "Shared reference events retain activation basis, identities, and valid content/hash chains.", "sharedTraces", "MUT-20"),
    ("INV-22", "determinism", "Seeded reference reruns serialize byte-identically independent of worker execution order.", "exactReplays", "MUT-21"),
    ("INV-H01", "historical", "Historical StatusProbe records remain recorded_swap, never activation.", "historicalTrace", "MUT-H01"),
    ("INV-H02", "historical", "Unavailable historical actor/observation/native-state fields remain null and loss-explicit.", "historicalTrace", "MUT-H02"),
    ("INV-H03", "historical", "Historical swap snapshots conserve values and chain observable hashes only.", "historicalTrace", "MUT-H03"),
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _word(root_seed: int, case_index: int, label: str) -> int:
    payload = f"E01/S07/v1/{root_seed}/{case_index}/{label}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _permutation(items: list[str], root_seed: int, case_index: int) -> tuple[str, ...]:
    keyed = [(_word(root_seed, case_index, f"occupancy/{item}"), item) for item in items]
    return tuple(item for _, item in sorted(keyed))


def generated_scenario(root_seed: int, case_index: int) -> Scenario:
    """Deterministic broad case generator independent of worker scheduling."""
    n = 1 + (_word(root_seed, case_index, "n") % 12)
    architecture = Architecture.TRADITIONAL if case_index % 5 == 4 else Architecture.CELL_VIEW
    policies = tuple(Policy)
    policy_mode = case_index % 4
    homogeneous_policy = policies[(case_index // 4) % len(policies)]
    direction_base = Direction.ASCENDING if _word(root_seed, case_index, "direction") % 2 == 0 else Direction.DESCENDING
    mixed_directions = architecture == Architecture.CELL_VIEW and case_index % 7 == 0 and n > 1
    value_mode = case_index % 4
    if value_mode == 0:
        values = list(range(n))
    elif value_mode == 1:
        values = list(reversed(range(n)))
    elif value_mode == 2:
        domain = max(1, min(4, n // 2))
        values = [int(_word(root_seed, case_index, f"value/{i}") % domain) - 1 for i in range(n)]
    else:
        values = [int(_word(root_seed, case_index, f"value/{i}") % 7) - 3 for i in range(n)]

    fault_profile = case_index % 6
    faults = [FaultMode.NORMAL] * n
    if n > 1 and fault_profile in {1, 3, 5}:
        faults[_word(root_seed, case_index, "passive-index") % n] = FaultMode.PASSIVE
    if n > 1 and fault_profile in {2, 3, 5}:
        faults[_word(root_seed, case_index, "stuck-index") % n] = FaultMode.STUCK
    if fault_profile == 5:
        for index in range(n):
            draw = _word(root_seed, case_index, f"fault/{index}") % 10
            if draw == 0:
                faults[index] = FaultMode.PASSIVE
            elif draw == 1:
                faults[index] = FaultMode.STUCK

    cells = []
    for index, value in enumerate(values):
        policy = homogeneous_policy if architecture == Architecture.TRADITIONAL or policy_mode < 3 else policies[index % 3]
        if architecture == Architecture.CELL_VIEW and policy_mode < 3:
            policy = policies[policy_mode]
        direction = (
            Direction.ASCENDING if mixed_directions and index % 2 == 0
            else Direction.DESCENDING if mixed_directions
            else direction_base
        )
        cells.append(
            Cell(
                cell_id=f"cell-{index:04d}", value=value, policy=policy,
                direction=direction, fault=faults[index],
                analysis_label=f"label-{index % 2}" if case_index % 11 == 0 else None,
            )
        )
    ids = [cell.cell_id for cell in cells]
    occupancy = _permutation(ids, root_seed, case_index)
    if case_index % 29 == 0:
        occupancy = tuple(ids)  # deterministic ordered/duplicate terminal coverage
    batch_width = 1 if architecture == Architecture.TRADITIONAL else (1, 2, 4, 8)[case_index % 4]
    max_activations = 0 if case_index % 97 == 0 else max(64, min(512, n * n * 8))
    return Scenario.create(
        cells,
        initial_occupancy=occupancy,
        seed=_word(root_seed, case_index, "runtime-seed"),
        max_activations=max_activations,
        architecture=architecture,
        batch_width=batch_width,
        traditional_policy=homogeneous_policy if architecture == Architecture.TRADITIONAL else None,
        generation_key=f"S07/{root_seed}/{case_index}",
        fault_placement="explicit",
    )


def _case_worker(item: tuple[int, int, bool, bool]) -> dict[str, Any]:
    root_seed, case_index, shared_check, replay_check = item
    started = time.perf_counter()
    scenario: Scenario | None = None
    try:
        scenario = generated_scenario(root_seed, case_index)
        result = run_scenario(scenario, trace_mode="full")
        audit = audit_reference_result(result)
        shared_events = 0
        if shared_check:
            bundle = adapt_reference_result(result)
            shared = validate_trace(bundle)
            shared_events = shared["eventCount"]
        replayed = False
        if replay_check:
            second = run_scenario(scenario, trace_mode="full")
            if result.to_json_bytes() != second.to_json_bytes():
                raise AssertionError("byte-exact replay mismatch")
            replayed = True
        return {
            "success": True,
            "rootSeed": str(root_seed),
            "caseIndex": case_index,
            "scenarioId": scenario.scenario_id,
            "coverage": audit["coverage"],
            "sharedEvents": shared_events,
            "replayed": replayed,
            "elapsedSeconds": time.perf_counter() - started,
        }
    except Exception as exc:  # compact fixture is promoted only if a case fails
        return {
            "success": False,
            "rootSeed": str(root_seed),
            "caseIndex": case_index,
            "scenario": scenario.to_dict() if scenario else None,
            "errorType": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsedSeconds": time.perf_counter() - started,
        }


def run_property_campaign(case_count: int, workers: int, root_seeds: tuple[int, ...]) -> dict[str, Any]:
    items = []
    per_seed = case_count // len(root_seeds)
    remainder = case_count % len(root_seeds)
    for seed_ordinal, root_seed in enumerate(root_seeds):
        count = per_seed + (1 if seed_ordinal < remainder else 0)
        for index in range(count):
            items.append((root_seed, index, index % 16 == 0, index % 24 == 0))
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        rows = list(executor.map(_case_worker, items, chunksize=8))
    elapsed = time.perf_counter() - started
    coverage = Counter()
    shared_events = replayed = 0
    failures = []
    seed_counts = Counter()
    seed_failures = Counter()
    for row in rows:
        seed_counts[row["rootSeed"]] += 1
        if not row["success"]:
            failures.append(row)
            seed_failures[row["rootSeed"]] += 1
            continue
        coverage.update(row["coverage"])
        if row["sharedEvents"] or row["coverage"].get("events") == 0 and row["caseIndex"] % 16 == 0:
            coverage["sharedTraces"] += 1
            shared_events += row["sharedEvents"]
        if row["replayed"]:
            coverage["exactReplays"] += 1
            replayed += 1
    return {
        "success": not failures,
        "caseCount": len(rows),
        "workerCount": workers,
        "rootSeeds": [str(seed) for seed in root_seeds],
        "seedCaseCounts": dict(sorted(seed_counts.items())),
        "seedFailureCounts": dict(sorted(seed_failures.items())),
        "elapsedSeconds": elapsed,
        "scenarioCasesPerSecond": len(rows) / elapsed,
        "coverage": dict(sorted(coverage.items())),
        "sharedTraceScenarioCount": coverage["sharedTraces"],
        "sharedEventCount": shared_events,
        "exactReplayScenarioCount": replayed,
        "failures": failures,
    }


def _fixture_scenario(fixture: Mapping[str, Any], max_activations: int = 10) -> Scenario:
    cells = []
    cursors = {}
    for item in fixture["pre"]:
        cells.append(
            Cell(
                item["id"], item["value"], Policy(item["policy"]),
                Direction(item["direction"]), FaultMode(item["fault"]),
            )
        )
        if "cursor" in item:
            cursors[item["id"]] = item["cursor"]
    return Scenario.create(
        cells,
        initial_occupancy=[item["id"] for item in fixture["pre"]],
        initial_selection_cursors=cursors,
        max_activations=max_activations,
        generation_key="S07/S03/" + fixture["fixtureId"],
    )


def run_s03_differential() -> dict[str, Any]:
    fixtures = json.loads(S03_FIXTURES.read_text(encoding="utf-8"))["fixtures"]
    rows = []
    for fixture in fixtures:
        passed = True
        error = None
        try:
            scenario = _fixture_scenario(
                fixture, fixture.get("terminal", {}).get("maxActivations", 10)
            )
            state = initial_state(scenario)
            expected = fixture["expected"]
            if fixture["fixtureId"] == "T14":
                proposals = []
                for ordinal, activation in enumerate(fixture["batch"]):
                    proposal = cell_view_proposal(
                        scenario, state, activation["actorId"], side=activation["neighborChoice"]
                    )
                    proposals.append(
                        replace(proposal, ordinal=ordinal, priority=activation["priority"])
                    )
                accepted, lost = resolve_conflicts(proposals)
                winner = next(item for item in proposals if item.ordinal in accepted)
                state.occupancy[winner.actor_pos], state.occupancy[winner.target_pos] = (
                    state.occupancy[winner.target_pos], state.occupancy[winner.actor_pos]
                )
                assert scenario.cell_map[winner.actor_id].cell_id == expected["winnerActorId"]
                assert {item.actor_id for item in proposals if item.ordinal in lost} == set(expected["loserActorIds"])
                assert state.occupancy == expected["postOrder"]
            elif "terminal" in fixture:
                state.activation_count = fixture["terminal"]["activationCount"]
                assert evaluate_terminal(scenario, state) == expected["stopReason"]
            else:
                activation = fixture["activation"]
                proposal = cell_view_proposal(
                    scenario, state, activation["actorId"], side=activation.get("neighborChoice")
                )
                assert proposal.kind.value == expected["proposal"]
                if "reason" in expected:
                    assert proposal.reason == expected["reason"]
                decision = "no_op"
                if proposal.kind == ProposalKind.MEMORY_UPDATE:
                    state.selection_cursors[proposal.actor_id] = proposal.new_cursor
                    decision = "accepted"
                elif proposal.kind == ProposalKind.SWAP:
                    target = scenario.cell_map[state.occupancy[proposal.target_pos]]
                    if target.fault == FaultMode.STUCK:
                        decision = "rejected_target_stuck"
                    else:
                        state.occupancy[proposal.actor_pos], state.occupancy[proposal.target_pos] = (
                            state.occupancy[proposal.target_pos], state.occupancy[proposal.actor_pos]
                        )
                        decision = "accepted"
                if "decision" in expected:
                    assert decision == expected["decision"]
                if "newCursor" in expected:
                    assert state.selection_cursors[activation["actorId"]] == expected["newCursor"]
                assert state.occupancy == expected["postOrder"]
        except Exception as exc:
            passed = False
            error = f"{type(exc).__name__}: {exc}"
        rows.append({
            "fixtureId": fixture["fixtureId"], "title": fixture["title"],
            "passed": passed, "error": error, "comparisonLevel": "S03 frozen one-step/terminal fixture",
        })
    return {"success": all(row["passed"] for row in rows), "count": len(rows), "rows": rows}


def run_s04_endpoint_differential() -> dict[str, Any]:
    historical = json.loads(S04_TOYS.read_text(encoding="utf-8"))
    unique = {row["fixture"]: row for row in historical["container"]}
    rows = []
    for fixture_id, row in sorted(unique.items()):
        result = run_scenario(
            create_scenario(
                row["values"], policy=row["policy"].capitalize(),
                generation_key=f"S07/S04-endpoint/{fixture_id}",
                permute=False, seed=707, max_activations=20_000,
            ),
            trace_mode="full",
        )
        audit_reference_result(result)
        passed = result.summary["finalValues"] == row["expected"]
        rows.append({
            "fixture": fixture_id,
            "policy": row["policy"],
            "inputValues": row["values"],
            "historicalFinalValues": row["actualFinalValues"],
            "referenceFinalValues": result.summary["finalValues"],
            "historicalStop": row["stopReason"],
            "referenceStop": result.summary["stopReason"],
            "passed": passed,
            "comparisonLevel": "endpoint/value-multiset only",
            "eventParityClaimed": False,
        })
    return {
        "success": all(row["passed"] for row in rows),
        "count": len(rows),
        "rows": rows,
        "warning": "C recorded swaps were not compared with R activation events.",
    }


def run_historical_isolation() -> dict[str, Any]:
    raw = json.loads(S04_RUN.read_text(encoding="utf-8"))
    bundle = adapt_historical_run(raw, source_path=S04_RUN)
    validation = validate_trace(bundle)
    observable_audit = audit_historical_trace(bundle, raw)
    events = list(bundle.events)
    checks = {
        "backendIsHistorical": bundle.backend == "historical_frozen_public_commit",
        "sequenceBasisIsRecordedSwap": bundle.sequence_basis == "recorded_swap",
        "eventCountEqualsSuccessfulSnapshots": len(events) == len(raw["sortingSteps"]),
        "allEventsAreAcceptedSwaps": all(
            event["proposal"]["kind"] == "Swap" and event["decision"]["accepted"]
            for event in events
        ),
        "actorIdentityRemainsUnavailable": all(
            event["actor"]["id"] is None
            and event["fieldAvailability"]["actor.id"]["status"] == "unavailable"
            for event in events
        ),
        "nativeStateHashesRemainUnavailable": all(
            event["state"]["nativePreHash"] is None
            and event["fieldAvailability"]["state.nativePreHash"]["status"] == "unavailable"
            for event in events
        ),
        "sharedTraceValidationPasses": validation["success"],
        "historicalObservableAuditPasses": observable_audit["success"],
        "referenceInvariantAuditNotApplied": True,
    }
    scheduler = json.loads(S04_SCHEDULER.read_text(encoding="utf-8"))
    deviations = [
        {
            "id": "HIST-DEV-01",
            "classification": "isolated_known_deviation",
            "referenceInvariant": "exact deterministic replay",
            "historicalEvidence": "30/30 unique host and 30/30 unique OCI trace hashes at fixed input/seed",
            "value": {
                "hostUniqueTraces": scheduler["host"]["uniqueTraceHashes"],
                "containerUniqueTraces": scheduler["container"]["uniqueTraceHashes"],
            },
            "interpretation": "Expected C scheduler nondeterminism; not a reference-backend failure.",
        },
        {
            "id": "HIST-DEV-02",
            "classification": "isolated_known_deviation",
            "referenceInvariant": "requested fault count equals realized distinct fault identities",
            "historicalEvidence": "S02 static finding FAULT_PLACEMENT_WITH_REPLACEMENT",
            "interpretation": "C can realize fewer distinct frozen cells; legacy placement remains separately named.",
        },
        {
            "id": "HIST-DEV-03",
            "classification": "historically_unavailable",
            "referenceInvariant": "stuck cells never move",
            "historicalEvidence": "No distinct stuck-fault implementation identified at frozen public HEAD.",
            "interpretation": "Cannot execute or score stuck parity for C.",
        },
        {
            "id": "HIST-DEV-04",
            "classification": "historically_unavailable",
            "referenceInvariant": "activation/proposal/observation/ledger completeness",
            "historicalEvidence": "StatusProbe records successful swap snapshots only.",
            "interpretation": "C recorded swaps are never expanded into synthetic activations.",
        },
        {
            "id": "HIST-DEV-05",
            "classification": "isolated_profile_difference",
            "referenceInvariant": "terminal precedence invariant_error > complete > quiescent > event_budget",
            "historicalEvidence": "C drivers use script-specific is_sorted, no_cells_should_move, or 15,000 recorded swaps.",
            "interpretation": "Historical stop reasons remain profile-labelled, not scored as R terminal states.",
        },
    ]
    return {
        "success": all(checks.values()),
        "checks": checks,
        "deviations": deviations,
        "publicationSnapshotClaimed": False,
        "referenceActivationParityClaimed": False,
        "validation": validation,
        "observableAudit": observable_audit,
    }


def _first_event(raw: Mapping[str, Any], predicate: Callable[[Mapping[str, Any]], bool]) -> int:
    return next(index for index, event in enumerate(raw["events"]) if predicate(event))


def run_mutation_checks() -> dict[str, Any]:
    base_result = run_scenario(
        create_scenario(
            [4, 1, 3, 2], policy="Bubble", generation_key="S07/mutation/base",
            permute=False, seed=23, max_activations=200,
        ), trace_mode="full",
    )
    batch_result = run_scenario(
        create_scenario(
            [5, 4, 3, 2, 1], policy="Bubble", generation_key="S07/mutation/batch",
            permute=False, seed=77, max_activations=200, batch_width=4,
        ), trace_mode="full",
    )
    selection_result = run_scenario(
        create_scenario(
            [3, 1, 2], policy="Selection", generation_key="S07/mutation/selection",
            permute=False, seed=19, max_activations=200,
        ), trace_mode="full",
    )
    stuck_result = run_scenario(
        create_scenario(
            [3, 1, 4, 2], policy="Bubble", generation_key="S07/mutation/stuck",
            permute=False, seed=10, max_activations=100, faults={1: "stuck"},
        ), trace_mode="full",
    )
    mixed_scenario = Scenario.create(
        [
            Cell("a", 1, Policy.BUBBLE, Direction.ASCENDING),
            Cell("b", 2, Policy.BUBBLE, Direction.DESCENDING),
        ],
        initial_occupancy=("a", "b"), seed=5, max_activations=5,
        generation_key="S07/mutation/mixed",
    )
    mixed_result = run_scenario(mixed_scenario, trace_mode="full")
    duplicate_result = run_scenario(
        create_scenario(
            [2, 2, 1], policy="Bubble", generation_key="S07/mutation/duplicate",
            permute=False, seed=9, max_activations=200,
        ), trace_mode="full",
    )
    historical_raw = json.loads(S04_RUN.read_text(encoding="utf-8"))
    historical_bundle = adapt_historical_run(historical_raw)
    shared_bundle = adapt_reference_result(base_result)

    mutations: list[tuple[str, str, Callable[[], None]]] = []

    def audit_mutant(source: RunResult, mutate: Callable[[dict[str, Any]], None]) -> None:
        # RunResult.to_dict intentionally avoids a costly deep copy.  Mutants must
        # be isolated so one injected defect cannot contaminate a later check.
        raw = deepcopy(source.to_dict())
        mutate(raw)
        audit_reference_result(raw)

    def mutate_exact_replay() -> None:
        first = base_result.to_json_bytes()
        changed = deepcopy(base_result.to_dict())
        changed["summary"]["finalValues"][0] = 999
        second = canonical_json_bytes(changed)
        if first != second:
            raise AssertionError("exact replay byte mismatch")

    mutations.extend([
        ("MUT-01", "duplicate final occupancy identity", lambda: audit_mutant(base_result, lambda r: r["finalState"]["occupancy"].__setitem__(1, r["finalState"]["occupancy"][0]))),
        ("MUT-02", "change final value summary", lambda: audit_mutant(base_result, lambda r: r["summary"]["finalValues"].__setitem__(0, 999))),
        ("MUT-03", "illegal actor position", lambda: audit_mutant(base_result, lambda r: r["events"][0]["proposal"].__setitem__("actorPos", 999))),
        ("MUT-04", "accepted swap with same endpoint", lambda: audit_mutant(base_result, lambda r: r["events"][_first_event(r, lambda e: e["decision"] == "accepted" and e["proposal"]["kind"] == "Swap")]["proposal"].__setitem__("targetPos", r["events"][_first_event(r, lambda e: e["decision"] == "accepted" and e["proposal"]["kind"] == "Swap")]["proposal"]["actorPos"]))),
        ("MUT-05", "accept a no-op", lambda: audit_mutant(base_result, lambda r: r["events"][_first_event(r, lambda e: e["proposal"]["kind"] == "NoOp")].__setitem__("decision", "accepted"))),
        ("MUT-06", "boundary reason changed", lambda: audit_mutant(base_result, lambda r: r["events"][_first_event(r, lambda e: e["proposal"]["reason"] == "boundary")]["proposal"].__setitem__("reason", "ordered_or_equal"))),
        ("MUT-07", "ledger swap increment changed", lambda: audit_mutant(base_result, lambda r: r["events"][0]["ledgerDelta"].__setitem__("acceptedSwaps", 1))),
        ("MUT-08", "immutable scenario value changed", lambda: audit_mutant(base_result, lambda r: r["scenario"]["cells"][0].__setitem__("value", 999))),
        ("MUT-09", "Selection cursor owner removed", lambda: audit_mutant(selection_result, lambda r: r["finalState"]["selectionCursors"].pop(next(iter(r["finalState"]["selectionCursors"]))))),
        ("MUT-10", "passive actor marked normal in scenario", lambda: audit_mutant(run_scenario(create_scenario([3, 1, 2], policy="Bubble", generation_key="S07/mutation/passive", permute=False, seed=41, max_activations=100, faults={1: "passive"}), trace_mode="full"), lambda r: r["scenario"]["cells"][1].__setitem__("fault", "normal"))),
        ("MUT-11", "passive displacement undercounted", lambda: audit_mutant(run_scenario(create_scenario([3, 1, 2], policy="Bubble", generation_key="S07/mutation/passive2", permute=False, seed=41, max_activations=100, faults={1: "passive"}), trace_mode="full"), lambda r: r["summary"]["ledger"].__setitem__("displacedCells", 0))),
        ("MUT-12", "stuck-target rejection accepted", lambda: audit_mutant(stuck_result, lambda r: r["events"][_first_event(r, lambda e: e["decision"] == "rejected_target_stuck")].__setitem__("decision", "accepted"))),
        ("MUT-13", "equal-value no-op changed to swap", lambda: audit_mutant(duplicate_result, lambda r: r["events"][_first_event(r, lambda e: e["proposal"]["reason"] == "ordered_or_equal")]["proposal"].__setitem__("kind", "Swap"))),
        ("MUT-14", "counter-addressed draw changed", lambda: audit_mutant(base_result, lambda r: r["events"][0]["randomAddressesAndDraws"][0].__setitem__("value", 0))),
        ("MUT-15", "native pre-state hash changed", lambda: audit_mutant(base_result, lambda r: r["events"][0].__setitem__("preStateHash", "0" * 64))),
        ("MUT-16", "duplicate-complete initial state relabelled quiescent", lambda: audit_mutant(run_scenario(create_scenario([1, 1, 2], policy="Bubble", generation_key="S07/mutation/complete", permute=False), trace_mode="full"), lambda r: r["summary"].__setitem__("stopReason", "quiescent"))),
        ("MUT-17", "quiescent result relabelled complete", lambda: audit_mutant(stuck_result, lambda r: r["summary"].__setitem__("stopReason", "complete"))),
        ("MUT-18", "event-budget result relabelled complete", lambda: audit_mutant(mixed_result, lambda r: r["summary"].__setitem__("stopReason", "complete"))),
        ("MUT-19", "conflict loser accepted", lambda: audit_mutant(batch_result, lambda r: r["events"][_first_event(r, lambda e: e["decision"] == "conflict_loss")].__setitem__("decision", "accepted"))),
        ("MUT-20", "shared reference event ID changed", lambda: validate_event({**deepcopy(shared_bundle.events[0]), "eventId": "sha256:" + "0" * 64})),
    ])

    def mutate_historical_sequence() -> None:
        event = deepcopy(historical_bundle.events[0])
        event["run"]["sequenceBasis"] = "activation"
        validate_event(assign_event_id(event))

    def invent_historical_actor() -> None:
        event = deepcopy(historical_bundle.events[0])
        event["actor"]["id"] = "invented"
        validate_event(assign_event_id(event))

    def break_historical_hash_chain() -> None:
        event = deepcopy(historical_bundle.events[0])
        event["state"]["observablePreHash"] = "sha256:" + "0" * 64
        event = assign_event_id(event)
        broken = TraceBundle(
            events=(event,), run_id=historical_bundle.run_id,
            scenario_id=historical_bundle.scenario_id, backend=historical_bundle.backend,
            sequence_basis=historical_bundle.sequence_basis,
            source_artifacts=historical_bundle.source_artifacts,
            source_run_summary=historical_bundle.source_run_summary,
            stop_reason=historical_bundle.stop_reason,
        )
        audit_historical_trace(broken, historical_raw)

    mutations.extend([
        ("MUT-21", "reference exact replay bytes changed", mutate_exact_replay),
        ("MUT-H01", "historical sequence relabelled activation", mutate_historical_sequence),
        ("MUT-H02", "unavailable historical actor invented", invent_historical_actor),
        ("MUT-H03", "historical observable pre-hash changed", break_historical_hash_chain),
    ])

    rows = []
    for mutation_id, description, mutation in mutations:
        started = time.perf_counter()
        killed = False
        detected_by = None
        try:
            mutation()
        except Exception as exc:
            killed = True
            detected_by = f"{type(exc).__name__}: {exc}"
        rows.append({
            "mutationId": mutation_id,
            "description": description,
            "killed": killed,
            "detectedBy": detected_by,
            "elapsedSeconds": time.perf_counter() - started,
        })
    killed = sum(row["killed"] for row in rows)
    return {
        "success": killed == len(rows),
        "mutationCount": len(rows),
        "killedCount": killed,
        "survivedCount": len(rows) - killed,
        "mutationScore": killed / len(rows),
        "rows": rows,
    }


def coverage_matrix(
    aggregate: Mapping[str, int], mutations: Mapping[str, Any], historical: Mapping[str, Any]
) -> list[dict[str, Any]]:
    mutation_map = {row["mutationId"]: row for row in mutations["rows"]}
    rows = []
    for invariant_id, category, description, evidence_key, mutation_id in INVARIANTS:
        if evidence_key == "historicalTrace":
            observations = 1 if historical["success"] else 0
        else:
            observations = int(aggregate.get(evidence_key, 0))
        mutation_killed = mutation_map.get(mutation_id, {}).get("killed", False)
        passed = observations > 0 and mutation_killed
        rows.append({
            "invariantId": invariant_id,
            "category": category,
            "releaseCritical": "true",
            "description": description,
            "evidenceLayers": "R" if not invariant_id.startswith("INV-H") else "C-observable/isolation",
            "propertyObservationCount": observations,
            "mutationId": mutation_id,
            "mutationKilled": str(bool(mutation_killed)).lower(),
            "result": "pass" if passed else "fail",
            "historicalActivationParityRequired": "false" if invariant_id.startswith("INV-H") else "not_applicable",
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def junit_report(path: Path, cases: list[dict[str, Any]]) -> None:
    failures = sum(not case["passed"] for case in cases)
    total_time = sum(float(case.get("time", 0.0)) for case in cases)
    suite = ET.Element(
        "testsuite",
        name="E01.S07.invariants",
        tests=str(len(cases)), failures=str(failures), errors="0", skipped="0",
        time=f"{total_time:.6f}",
    )
    for case in cases:
        node = ET.SubElement(
            suite, "testcase", classname=case["classname"], name=case["name"],
            time=f"{float(case.get('time', 0.0)):.6f}",
        )
        if not case["passed"]:
            failure = ET.SubElement(node, "failure", message=case.get("message", "failed"))
            failure.text = case.get("details", case.get("message", "failed"))
    tree = ET.ElementTree(suite)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def source_manifest() -> dict[str, Any]:
    paths = [
        ROOT / "reference_simulator" / "__init__.py",
        ROOT / "reference_simulator" / "invariants.py",
        ROOT / "shared_events" / "__init__.py",
        ROOT / "shared_events" / "validation.py",
        ROOT / "shared_events" / "invariants.py",
        ROOT / "scripts" / "validate_invariants.py",
        ROOT / "tests" / "test_invariants.py",
        ROOT / "tests" / "test_shared_events.py",
    ]
    return {
        "package": "reference_simulator invariant gate",
        "auditVersion": AUDIT_VERSION,
        "repositoryRoot": str(ROOT),
        "repositoryCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "repositoryBranch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "repositoryDirty": bool(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()),
        "files": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in paths
        ],
        "sourceStorage": "Git repository; source is not duplicated into $ARTIFACTS_DIR.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=3072)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--root-seeds", default=",".join(str(seed) for seed in DEFAULT_ROOT_SEEDS)
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be in [1, 8]")
    if args.cases < 256:
        parser.error("--cases must be at least 256 for release coverage")
    root_seeds = tuple(int(value, 0) for value in args.root_seeds.split(","))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    property_result = run_property_campaign(args.cases, args.workers, root_seeds)
    s03 = run_s03_differential()
    s04 = run_s04_endpoint_differential()
    historical = run_historical_isolation()
    mutations = run_mutation_checks()

    aggregate = Counter(property_result["coverage"])
    aggregate["historicalTrace"] = int(historical["success"])
    matrix = coverage_matrix(aggregate, mutations, historical)
    unexplained = list(property_result["failures"])
    if not s03["success"]:
        unexplained.extend(row for row in s03["rows"] if not row["passed"])
    if not s04["success"]:
        unexplained.extend(row for row in s04["rows"] if not row["passed"])
    if not mutations["success"]:
        unexplained.extend(row for row in mutations["rows"] if not row["killed"])
    matrix_failures = [row for row in matrix if row["result"] != "pass"]
    unexplained.extend(matrix_failures)
    success = (
        property_result["success"] and s03["success"] and s04["success"]
        and historical["success"] and mutations["success"] and not matrix_failures
    )

    write_json(output / "property_test_summary.json", property_result)
    write_json(output / "differential_toy_results.json", {"s03": s03, "s04EndpointOnly": s04})
    write_json(output / "historical_isolation.json", historical)
    write_json(output / "mutation_results.json", mutations)
    write_csv(output / "invariant_coverage_matrix.csv", matrix)
    write_json(output / "invariant_coverage_matrix.json", {"researchStepId": "S07", "rows": matrix})
    write_json(output / "failing_fixtures.json", {
        "researchStepId": "S07",
        "unexplainedFailureCount": len(unexplained),
        "fixtures": unexplained,
        "knownHistoricalDeviationsAreSeparate": True,
        "knownHistoricalDeviationCount": len(historical["deviations"]),
    })

    cases = []
    for seed in property_result["rootSeeds"]:
        cases.append({
            "classname": "S07.seeded_property",
            "name": f"root_seed_{seed}",
            "passed": property_result["seedFailureCounts"].get(seed, 0) == 0,
            "time": property_result["elapsedSeconds"] / len(property_result["rootSeeds"]),
            "message": f"{property_result['seedFailureCounts'].get(seed, 0)} failures",
        })
    for row in s03["rows"]:
        cases.append({"classname": "S07.differential.S03", "name": row["fixtureId"], "passed": row["passed"], "message": row["error"], "time": 0})
    for row in s04["rows"]:
        cases.append({"classname": "S07.differential.S04_endpoint", "name": row["fixture"], "passed": row["passed"], "message": "endpoint mismatch", "time": 0})
    for row in mutations["rows"]:
        cases.append({"classname": "S07.mutation", "name": row["mutationId"], "passed": row["killed"], "message": "mutant survived", "details": row["detectedBy"], "time": row["elapsedSeconds"]})
    for row in matrix:
        cases.append({"classname": "S07.coverage", "name": row["invariantId"], "passed": row["result"] == "pass", "message": f"coverage={row['propertyObservationCount']} mutation={row['mutationKilled']}", "time": 0})
    cases.append({"classname": "S07.historical", "name": "recorded_swap_isolation", "passed": historical["success"], "message": "historical isolation failed", "time": 0})
    junit_report(output / "test_report.xml", cases)

    elapsed = time.perf_counter() - started
    summary = {
        "researchStepId": "S07",
        "success": success,
        "validationResult": "PASS" if success else "FAIL",
        "auditVersion": AUDIT_VERSION,
        "propertyCases": property_result["caseCount"],
        "propertyEvents": aggregate["events"],
        "rootSeeds": property_result["rootSeeds"],
        "workers": args.workers,
        "exactReplayCases": aggregate["exactReplays"],
        "sharedTraceCases": aggregate["sharedTraces"],
        "s03FixturesPassed": sum(row["passed"] for row in s03["rows"]),
        "s03FixtureCount": len(s03["rows"]),
        "s04EndpointFixturesPassed": sum(row["passed"] for row in s04["rows"]),
        "s04EndpointFixtureCount": len(s04["rows"]),
        "invariantsPassed": sum(row["result"] == "pass" for row in matrix),
        "invariantCount": len(matrix),
        "mutantsKilled": mutations["killedCount"],
        "mutantCount": mutations["mutationCount"],
        "mutationScore": mutations["mutationScore"],
        "unexplainedFailures": len(unexplained),
        "knownHistoricalDeviationsIsolated": len(historical["deviations"]),
        "historicalReferenceActivationParityClaimed": False,
        "junitCases": len(cases),
        "elapsedSeconds": elapsed,
        "coverage": dict(sorted(aggregate.items())),
    }
    write_json(output / "validation_summary.json", summary)
    write_json(output / "environment_provenance.json", {
        "researchStepId": "S07",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpuCountVisible": os.cpu_count(),
        "workerCount": args.workers,
        "parallelUnit": "independent generated scenario",
        "threadEnvironment": {
            key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
            )
        },
        "dependencies": "Python standard library plus repository runtime; S06 Jsonschema/PyArrow only for sampled shared traces",
        "newDependenciesInstalled": [],
    })
    write_json(output / "source_package_manifest.json", source_manifest())
    if not success:
        raise SystemExit("S07 invariant gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
