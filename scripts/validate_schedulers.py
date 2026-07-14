#!/usr/bin/env python3
"""Generate the compact S04 scheduler validation evidence package."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
from typing import Any, Iterable

from causal_simulator.architectures import (
    ArchitectureExecutionContract,
    run_architecture,
)
from causal_simulator.schedulers import (
    PERMUTATION_STREAM,
    SCHEDULER_PRESPECIFICATION_SHA256,
    FrozenSchedulerController,
    SchedulerExecutionContract,
    SchedulerFamily,
    exact_replay_scheduler,
    run_scheduled_architecture,
    scheduler_controller_fields,
)
from reference_simulator.api import create_scenario
from reference_simulator.model import Direction, Policy


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "s04_scheduler_fixtures.json"
PRESPEC_PATH = Path(
    "/artifacts/research_steps/S04/scheduler_package/scheduler_prespecification.json"
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def controller(
    family: SchedulerFamily,
    n: int,
    *,
    seed: int,
    scenario_id: str,
) -> FrozenSchedulerController:
    return FrozenSchedulerController(
        SchedulerExecutionContract(family),
        seed=seed,
        scenario_id=scenario_id,
        actor_ids=tuple(f"cell-{index:04d}" for index in range(n)),
    )


def sequence(
    family: SchedulerFamily,
    opportunities: int,
    n: int,
    *,
    seed: int,
    scenario_id: str,
) -> tuple[tuple[str, ...], FrozenSchedulerController]:
    item = controller(family, n, seed=seed, scenario_id=scenario_id)
    actors: list[str] = []
    while len(actors) < opportunities:
        slots = item(len(actors), opportunities - len(actors))
        actors.extend(slot.actor_id for slot in slots)
    return tuple(actors), item


def max_gap(actors: tuple[str, ...]) -> int:
    positions: dict[str, list[int]] = defaultdict(list)
    for index, actor_id in enumerate(actors):
        positions[actor_id].append(index)
    gaps = [
        right - left
        for values in positions.values()
        for left, right in zip(values, values[1:])
    ]
    return max(gaps, default=0)


def distribution_validation() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = 16
    seeds = 64
    opportunities_per_seed = 16_000
    expected = opportunities_per_seed / n
    standard_deviation = math.sqrt(opportunities_per_seed * (1 / n) * (1 - 1 / n))
    rows: list[dict[str, Any]] = []
    pooled = Counter()
    for seed in range(seeds):
        actors, _ = sequence(
            SchedulerFamily.UNIFORM_RANDOM_ACTIVATION,
            opportunities_per_seed,
            n,
            seed=seed,
            scenario_id=f"S04/uniform-distribution/seed-{seed:02d}",
        )
        counts = Counter(actors)
        pooled.update(counts)
        for actor_id in sorted(counts):
            deviation = counts[actor_id] - expected
            rows.append(
                {
                    "seed": seed,
                    "actorId": actor_id,
                    "opportunities": opportunities_per_seed,
                    "observed": counts[actor_id],
                    "expected": expected,
                    "deviation": deviation,
                    "standardizedDeviation": deviation / standard_deviation,
                }
            )
    pooled_expected = seeds * expected
    pooled_sd = math.sqrt(seeds * opportunities_per_seed * (1 / n) * (1 - 1 / n))
    max_cell_z = max(abs(float(row["standardizedDeviation"])) for row in rows)
    max_pooled_z = max(
        abs(count - pooled_expected) / pooled_sd for count in pooled.values()
    )
    result = {
        "passed": max_cell_z <= 6.0,
        "seeds": seeds,
        "identities": n,
        "opportunitiesPerSeed": opportunities_per_seed,
        "totalOpportunities": seeds * opportunities_per_seed,
        "prespecifiedMaximumAbsoluteStandardizedDeviation": 6.0,
        "observedMaximumAbsoluteSeedIdentityDeviation": max_cell_z,
        "observedMaximumAbsolutePooledIdentityDeviation": max_pooled_z,
        "pooledCounts": dict(sorted(pooled.items())),
    }
    return rows, result


def fairness_validation() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exact_families = (
        SchedulerFamily.DETERMINISTIC_SCAN,
        SchedulerFamily.RANDOM_PERMUTATION_SWEEP,
        SchedulerFamily.SYNCHRONOUS_BATCH_DETERMINISTIC_CONFLICT,
        SchedulerFamily.FAIR_ADVERSARIAL,
    )
    for family in exact_families:
        for n in (1, 2, 3, 8, 16, 31):
            opportunities = max(20_000, 400 * n)
            actors, audit = sequence(
                family,
                opportunities,
                n,
                seed=314159,
                scenario_id=f"S04/fairness/{family.value}/n-{n}",
            )
            identities = set(actors[:n])
            aligned_violations = 0
            window_violations = 0
            if family in {
                SchedulerFamily.DETERMINISTIC_SCAN,
                SchedulerFamily.RANDOM_PERMUTATION_SWEEP,
                SchedulerFamily.SYNCHRONOUS_BATCH_DETERMINISTIC_CONFLICT,
            }:
                aligned_violations = sum(
                    set(actors[start : start + n]) != identities
                    for start in range(0, opportunities - n + 1, n)
                )
            if family == SchedulerFamily.FAIR_ADVERSARIAL:
                window = 2 * n
                window_violations = sum(
                    set(actors[start : start + window]) != identities
                    for start in range(n, opportunities - window + 1)
                )
                bound = window
            elif family == SchedulerFamily.RANDOM_PERMUTATION_SWEEP:
                bound = max(0, 2 * n - 1)
            else:
                bound = n
            observed_gap = max_gap(actors)
            pass_gap = n == 1 or observed_gap <= bound
            passed = pass_gap and aligned_violations == 0 and window_violations == 0
            rows.append(
                {
                    "family": family.value,
                    "n": n,
                    "opportunities": opportunities,
                    "declaredMaximumGap": bound,
                    "observedMaximumGap": observed_gap,
                    "alignedBlockViolations": aligned_violations,
                    "slidingWindowViolations": window_violations,
                    "maximumIdentitiesScored": max(
                        item.identities_scored for item in audit.audits
                    ),
                    "policyProposalCandidatesInspected": sum(
                        item.proposal_candidates_inspected for item in audit.audits
                    ),
                    "passed": passed,
                }
            )
    uniform, _ = sequence(
        SchedulerFamily.UNIFORM_RANDOM_ACTIVATION,
        100_000,
        16,
        seed=271828,
        scenario_id="S04/fairness/uniform-observation",
    )
    rows.append(
        {
            "family": SchedulerFamily.UNIFORM_RANDOM_ACTIVATION.value,
            "n": 16,
            "opportunities": len(uniform),
            "declaredMaximumGap": "none",
            "observedMaximumGap": max_gap(uniform),
            "alignedBlockViolations": "not_applicable",
            "slidingWindowViolations": "not_applicable",
            "maximumIdentitiesScored": 1,
            "policyProposalCandidatesInspected": 0,
            "passed": True,
        }
    )
    exact_rows = [row for row in rows if row["family"] != "uniform_random_activation"]
    summary = {
        "passed": all(bool(row["passed"]) for row in rows),
        "exactFairnessCases": len(exact_rows),
        "exactFairnessCasesPassed": sum(bool(row["passed"]) for row in exact_rows),
        "uniformFiniteWindowGuarantee": False,
        "uniformObservedMaximumGapDiagnostic": rows[-1]["observedMaximumGap"],
        "totalPolicyProposalCandidatesInspectedBySchedulers": sum(
            int(row["policyProposalCandidatesInspected"]) for row in rows
        ),
    }
    return rows, summary


def proposal_resources(event: dict[str, Any]) -> set[tuple[str, str | int]]:
    proposal = event["proposal"]
    resources: set[tuple[str, str | int]] = {("actor", event["actorId"])}
    if proposal["kind"] == "Swap":
        resources.add(("position", int(proposal["actorPos"])))
        resources.add(("position", int(proposal["targetPos"])))
    return resources


def validate_conflict_batches(events: Iterable[dict[str, Any]]) -> dict[str, int | bool]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        groups[int(event["eventIndex"]) - int(event["batchOrdinal"])].append(event)
    batch_count = 0
    conflict_losses = 0
    priority_draws = 0
    failures = 0
    for batch in groups.values():
        batch_count += 1
        batch.sort(key=lambda item: int(item["batchOrdinal"]))
        width = int(batch[0]["batchWidth"])
        if len(batch) != width or [item["batchOrdinal"] for item in batch] != list(range(width)):
            failures += 1
        if len({item["preStateHash"] for item in batch}) != 1:
            failures += 1
        if len({item["postStateHash"] for item in batch}) != 1:
            failures += 1
        if len({item["actorId"] for item in batch}) != width:
            failures += 1
        if width > 1:
            for event in batch:
                draws = [
                    draw
                    for draw in event["randomAddressesAndDraws"]
                    if draw["stream"] == "conflict_priority"
                ]
                priority_draws += len(draws)
                if len(draws) != 1:
                    failures += 1
        changing = [
            item
            for item in batch
            if item["decision"] in {"accepted", "conflict_loss"}
        ]
        ordered = sorted(
            changing,
            key=lambda item: (
                item["proposal"]["priority"],
                item["actorId"],
                item["proposal"]["targetPos"]
                if item["proposal"]["targetPos"] is not None
                else -1,
                item["batchOrdinal"],
            ),
        )
        reserved: set[tuple[str, str | int]] = set()
        expected_accepted: set[int] = set()
        for event in ordered:
            resources = proposal_resources(event)
            if not resources & reserved:
                expected_accepted.add(int(event["batchOrdinal"]))
                reserved.update(resources)
        observed_accepted = {
            int(item["batchOrdinal"])
            for item in batch
            if item["decision"] == "accepted"
        }
        if expected_accepted != observed_accepted:
            failures += 1
        conflict_losses += sum(item["decision"] == "conflict_loss" for item in batch)
    return {
        "passed": failures == 0,
        "batchesChecked": batch_count,
        "conflictLossesObserved": conflict_losses,
        "conflictPriorityDrawsChecked": priority_draws,
        "invariantFailures": failures,
    }


def execution_grid_validation() -> tuple[
    list[dict[str, Any]], dict[str, Any], dict[str, Any]
]:
    input_profiles = {
        "reverse_unique": list(range(8, 0, -1)),
        "block_scrambled": [5, 6, 7, 8, 1, 2, 3, 4],
        "balanced_duplicates": [3, 1, 3, 2, 2, 1, 4, 4],
    }
    architectures = (
        ArchitectureExecutionContract.distributed_local(),
        ArchitectureExecutionContract.central_local_k1(),
        ArchitectureExecutionContract.distributed_weak(enabled=False),
        ArchitectureExecutionContract.distributed_weak(),
    )
    records: list[dict[str, Any]] = []
    total_replay = 0
    total_ledger = 0
    total_completed = 0
    total_legal = 0
    uniform_native_parity = 0
    uniform_native_pairs = 0
    topology_parity_pairs = 0
    topology_parity_passed = 0
    weak_active_native_differences = 0
    noncompletion_records: list[dict[str, Any]] = []

    for input_name, values in input_profiles.items():
        for policy in Policy:
            for direction in Direction:
                for seed in range(4):
                    scenario = create_scenario(
                        values,
                        policy=policy,
                        direction=direction,
                        seed=seed,
                        max_activations=4_096,
                        generation_key=(
                            f"S04/grid/{input_name}/{policy.value}/{direction.value}/{seed}"
                        ),
                        permute=False,
                    )
                    for family in SchedulerFamily:
                        scheduler_contract = SchedulerExecutionContract(family)
                        family_runs = []
                        for architecture_contract in architectures:
                            item = run_scheduled_architecture(
                                scenario,
                                architecture_contract,
                                scheduler_contract,
                                trace_mode="digest",
                            )
                            replay = exact_replay_scheduler(item)
                            replay_pass = replay.to_json_bytes() == item.to_json_bytes()
                            ledger_pass = all(item.opportunity_validation().values())
                            legal_pass = scheduler_contract.legal_primitives == (
                                "NoOp",
                                "Swap",
                                "MemoryUpdate",
                            )
                            complete = bool(item.result.summary["completed"])
                            total_replay += replay_pass
                            total_ledger += ledger_pass
                            total_legal += legal_pass
                            total_completed += complete
                            if family == SchedulerFamily.UNIFORM_RANDOM_ACTIVATION:
                                native = run_architecture(
                                    scenario,
                                    architecture_contract,
                                    trace_mode="digest",
                                )
                                uniform_native_pairs += 1
                                uniform_native_parity += (
                                    native.result.to_json_bytes()
                                    == item.result.to_json_bytes()
                                )
                            if not complete:
                                noncompletion_records.append(
                                    {
                                        "scenarioId": scenario.scenario_id,
                                        "inputProfile": input_name,
                                        "policy": policy.value,
                                        "direction": direction.value,
                                        "seed": seed,
                                        "architecture": architecture_contract.architecture.value,
                                        "coordinatorProfile": architecture_contract.coordinator_profile.value,
                                        "scheduler": family.value,
                                        "stopReason": item.result.summary["stopReason"],
                                        "activationCount": item.result.summary[
                                            "activationCount"
                                        ],
                                    }
                                )
                            records.append(
                                {
                                    "scenarioId": scenario.scenario_id,
                                    "inputProfile": input_name,
                                    "policy": policy.value,
                                    "direction": direction.value,
                                    "seed": seed,
                                    "architecture": architecture_contract.architecture.value,
                                    "coordinatorProfile": architecture_contract.coordinator_profile.value,
                                    "scheduler": family.value,
                                    "stopReason": item.result.summary["stopReason"],
                                    "completed": complete,
                                    "activationCount": item.result.summary["activationCount"],
                                    "conflictLosses": item.result.summary["ledger"][
                                        "conflictLosses"
                                    ],
                                    "coordinatorInterventions": item.architecture_run.architecture_ledger[
                                        "coordinatorInterventions"
                                    ],
                                    "replayPassed": replay_pass,
                                    "ledgerPassed": ledger_pass,
                                    "legalPrimitivesPassed": legal_pass,
                                    "eventDigest": item.result.event_digest,
                                }
                            )
                            family_runs.append(item)
                        for comparison in (1, 2):
                            topology_parity_pairs += 1
                            topology_parity_passed += (
                                family_runs[0].result.to_json_bytes()
                                == family_runs[comparison].result.to_json_bytes()
                            )
                        weak_active_native_differences += (
                            family_runs[0].result.to_json_bytes()
                            != family_runs[3].result.to_json_bytes()
                        )

    expected_runs = (
        len(input_profiles)
        * len(tuple(Policy))
        * len(tuple(Direction))
        * 4
        * len(tuple(SchedulerFamily))
        * len(architectures)
    )
    summary = {
        "passed": all(
            (
                len(records) == expected_runs,
                total_replay == expected_runs,
                total_ledger == expected_runs,
                total_legal == expected_runs,
                uniform_native_parity == uniform_native_pairs,
                topology_parity_passed == topology_parity_pairs,
            )
        ),
        "runs": len(records),
        "expectedRuns": expected_runs,
        "replaysPassed": total_replay,
        "ledgerOpportunityChecksPassed": total_ledger,
        "noFaultCompletions": total_completed,
        "noFaultNoncompletions": expected_runs - total_completed,
        "noFaultNoncompletionRecords": noncompletion_records,
        "noFaultCompletionIsOutcomeNotEngineInvariant": True,
        "legalPrimitiveChecksPassed": total_legal,
        "uniformNativeParityPairs": uniform_native_pairs,
        "uniformNativeParityPairsPassed": uniform_native_parity,
        "centralAndWeakDisabledParityPairs": topology_parity_pairs,
        "centralAndWeakDisabledParityPairsPassed": topology_parity_passed,
        "weakActiveNativeDifferences": weak_active_native_differences,
    }

    # Budget truncation is separate because a deliberately small budget is an
    # expected non-completion, not part of the no-fault completion grid.
    truncated_scenario = create_scenario(
        list(range(8, 0, -1)),
        policy=Policy.BUBBLE,
        seed=811,
        max_activations=11,
        generation_key="S04/budget-truncation",
        permute=False,
    )
    truncated = run_scheduled_architecture(
        truncated_scenario,
        ArchitectureExecutionContract.distributed_local(),
        SchedulerExecutionContract(
            SchedulerFamily.SYNCHRONOUS_BATCH_DETERMINISTIC_CONFLICT
        ),
        trace_mode="full",
    )
    batches = defaultdict(list)
    for event in truncated.result.events:
        batches[int(event["eventIndex"]) - int(event["batchOrdinal"])].append(event)
    widths = [len(items) for _, items in sorted(batches.items())]
    truncation = {
        "passed": widths == [8, 3]
        and truncated.result.summary["activationCount"] == 11
        and all(truncated.opportunity_validation().values()),
        "declaredBudget": 11,
        "observedBatchWidths": widths,
        "activationCount": truncated.result.summary["activationCount"],
        "stopReason": truncated.result.summary["stopReason"],
    }
    return records, summary, truncation


def traced_runtime_validation() -> tuple[dict[str, Any], dict[str, Any]]:
    """Inspect actual proposals and reconstruct synchronous conflicts."""

    legal_checks = 0
    opportunity_checks = 0
    traced_runs = 0
    traced_events = 0
    conflict_batches = 0
    conflict_losses = 0
    conflict_priority_draws = 0
    conflict_failures = 0
    architectures = (
        ArchitectureExecutionContract.distributed_local(),
        ArchitectureExecutionContract.central_local_k1(),
        ArchitectureExecutionContract.distributed_weak(enabled=False),
        ArchitectureExecutionContract.distributed_weak(),
    )
    for family in SchedulerFamily:
        for policy in Policy:
            for direction in Direction:
                scenario = create_scenario(
                    [6, 1, 5, 2, 4, 3],
                    policy=policy,
                    direction=direction,
                    seed=37,
                    max_activations=1_024,
                    generation_key=f"S04/traced/{family.value}/{policy.value}/{direction.value}",
                    permute=False,
                )
                for architecture in architectures:
                    item = run_scheduled_architecture(
                        scenario,
                        architecture,
                        SchedulerExecutionContract(family),
                        trace_mode="full",
                    )
                    traced_runs += 1
                    traced_events += len(item.result.events)
                    legal_checks += {
                        event["proposal"]["kind"] for event in item.result.events
                    } <= {"NoOp", "Swap", "MemoryUpdate"}
                    opportunity_checks += all(item.opportunity_validation().values())
                    if family == SchedulerFamily.SYNCHRONOUS_BATCH_DETERMINISTIC_CONFLICT:
                        checked = validate_conflict_batches(
                            dict(event) for event in item.result.events
                        )
                        conflict_batches += int(checked["batchesChecked"])
                        conflict_losses += int(checked["conflictLossesObserved"])
                        conflict_priority_draws += int(
                            checked["conflictPriorityDrawsChecked"]
                        )
                        conflict_failures += int(checked["invariantFailures"])

    # Frozen hand fixture guarantees that the conflict-loss branch is actually
    # exercised, even if a future traced grid happens to be disjoint.
    hand = create_scenario(
        [5, 4, 3, 2, 1],
        policy=Policy.BUBBLE,
        direction=Direction.ASCENDING,
        seed=0,
        max_activations=5,
        generation_key="S04/synchronous-conflict",
        permute=False,
    )
    hand_run = run_scheduled_architecture(
        hand,
        ArchitectureExecutionContract.distributed_local(),
        SchedulerExecutionContract(
            SchedulerFamily.SYNCHRONOUS_BATCH_DETERMINISTIC_CONFLICT
        ),
        trace_mode="full",
    )
    hand_checked = validate_conflict_batches(dict(event) for event in hand_run.result.events)
    conflict_batches += int(hand_checked["batchesChecked"])
    conflict_losses += int(hand_checked["conflictLossesObserved"])
    conflict_priority_draws += int(hand_checked["conflictPriorityDrawsChecked"])
    conflict_failures += int(hand_checked["invariantFailures"])
    hand_branch_pass = hand_run.result.summary["ledger"]["conflictLosses"] == 1

    primitive = {
        "passed": legal_checks == traced_runs and opportunity_checks == traced_runs,
        "tracedRuns": traced_runs,
        "tracedEvents": traced_events,
        "legalPrimitiveChecksPassed": legal_checks,
        "opportunityChecksPassed": opportunity_checks,
        "legalPrimitives": ["NoOp", "Swap", "MemoryUpdate"],
    }
    conflicts = {
        "passed": conflict_failures == 0 and conflict_losses > 0 and hand_branch_pass,
        "batchesChecked": conflict_batches,
        "conflictLossesObserved": conflict_losses,
        "conflictPriorityDrawsChecked": conflict_priority_draws,
        "invariantFailures": conflict_failures,
        "frozenHandFixtureConflictLossPassed": hand_branch_pass,
    }
    return primitive, conflicts


def fixture_validation() -> dict[str, Any]:
    fixture = json.loads(FIXTURE_PATH.read_text())
    cases: list[dict[str, Any]] = []
    for name, expected in fixture["expectedSequences"].items():
        family = SchedulerFamily(name)
        item = FrozenSchedulerController(
            SchedulerExecutionContract(family),
            seed=int(fixture["seed"]),
            scenario_id=str(fixture["scenarioId"]),
            actor_ids=tuple(fixture["actorIds"]),
        )
        observed: list[str] = []
        batches: list[list[str]] = []
        while len(observed) < int(fixture["opportunities"]):
            slots = item(len(observed), int(fixture["opportunities"]) - len(observed))
            batch = [slot.actor_id for slot in slots]
            batches.append(batch)
            observed.extend(batch)
        passed = observed == expected
        if family == SchedulerFamily.SYNCHRONOUS_BATCH_DETERMINISTIC_CONFLICT:
            passed = passed and batches == fixture["synchronousExpectedBatches"]
        cases.append(
            {
                "family": name,
                "passed": passed,
                "observedSequence": observed,
                "observedBatches": batches,
            }
        )
    result = {
        "schemaVersion": "E02.scheduler-fixture-results.v1",
        "sourceFixture": str(FIXTURE_PATH),
        "sourceFixtureSha256": hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest(),
        "passed": all(item["passed"] for item in cases),
        "cases": cases,
    }
    return result


def rng_validation() -> dict[str, Any]:
    rows = []
    for family in SchedulerFamily:
        actors, item = sequence(
            family,
            257,
            8,
            seed=12345,
            scenario_id=f"S04/rng/{family.value}",
        )
        blocks = Counter()
        for audit in item.audits:
            if audit.actor_selection_stream_blocks:
                stream = (
                    "actor_activation"
                    if family == SchedulerFamily.UNIFORM_RANDOM_ACTIVATION
                    else PERMUTATION_STREAM
                )
                blocks[stream] += audit.actor_selection_stream_blocks
        rows.append(
            {
                "family": family.value,
                "opportunities": len(actors),
                "actorSelectionStreams": dict(sorted(blocks.items())),
                "deterministicActorSelection": family
                in {
                    SchedulerFamily.DETERMINISTIC_SCAN,
                    SchedulerFamily.SYNCHRONOUS_BATCH_DETERMINISTIC_CONFLICT,
                    SchedulerFamily.FAIR_ADVERSARIAL,
                },
                "workerOrderUsed": False,
            }
        )
    passed = all(
        (row["family"] in {"uniform_random_activation", "random_permutation_sweep"})
        == bool(row["actorSelectionStreams"])
        for row in rows
    )
    return {
        "passed": passed,
        "generator": "sha256_counter_E01_v1",
        "rows": rows,
        "exactCouplingClaim": "only where named-stream consumption matches",
    }


def information_boundary_validation() -> dict[str, Any]:
    fields = set(scheduler_controller_fields())
    forbidden = {
        "scenario",
        "state",
        "occupancy",
        "cells",
        "values",
        "policies",
        "directions",
        "faults",
        "proposals",
        "ledger",
        "outcomes",
    }
    signature = tuple(inspect.signature(FrozenSchedulerController.__call__).parameters)
    result = {
        "passed": not fields & forbidden
        and signature == ("self", "event_index", "remaining_opportunities"),
        "controllerFields": sorted(fields),
        "forbiddenFields": sorted(forbidden),
        "forbiddenFieldsPresent": sorted(fields & forbidden),
        "callbackParameters": list(signature),
        "policyProposalCandidatesInspected": 0,
        "schedulerCanReceiveRunState": False,
    }
    return result


def capability_matrix() -> list[dict[str, Any]]:
    rows = []
    for family in SchedulerFamily:
        rows.append(
            {
                "family": family.value,
                "batchMode": "synchronous_population"
                if family == SchedulerFamily.SYNCHRONOUS_BATCH_DETERMINISTIC_CONFLICT
                else "serial",
                "pathwiseFairness": {
                    SchedulerFamily.UNIFORM_RANDOM_ACTIVATION: "none",
                    SchedulerFamily.DETERMINISTIC_SCAN: "aligned_n_block",
                    SchedulerFamily.RANDOM_PERMUTATION_SWEEP: "aligned_n_sweep",
                    SchedulerFamily.SYNCHRONOUS_BATCH_DETERMINISTIC_CONFLICT: "full_batch",
                    SchedulerFamily.FAIR_ADVERSARIAL: "sliding_2n_window_after_warmup",
                }[family],
                "actorSelectionStream": {
                    SchedulerFamily.UNIFORM_RANDOM_ACTIVATION: "actor_activation",
                    SchedulerFamily.RANDOM_PERMUTATION_SWEEP: PERMUTATION_STREAM,
                }.get(family, "none"),
                "dynamicGlobalReads": False,
                "policyProposalCandidatesPerOpportunity": 1,
                "matchedS03ArchitecturesSupported": 4,
                "legacyGlobalMatchedSupport": False,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    prespec_hash = hashlib.sha256(PRESPEC_PATH.read_bytes()).hexdigest()
    if prespec_hash != SCHEDULER_PRESPECIFICATION_SHA256:
        raise AssertionError("scheduler prespecification changed after implementation freeze")

    distribution_rows, distribution = distribution_validation()
    fairness_rows, fairness = fairness_validation()
    records, execution, truncation = execution_grid_validation()
    primitive_runtime, conflicts = traced_runtime_validation()
    fixtures = fixture_validation()
    rng = rng_validation()
    information = information_boundary_validation()
    capabilities = capability_matrix()

    write_csv(args.output / "activation_distribution.csv", distribution_rows)
    write_csv(args.output / "fairness_diagnostics.csv", fairness_rows)
    write_csv(args.output / "scheduler_capability_matrix.csv", capabilities)
    write_json(args.output / "scheduler_fixtures.json", fixtures)
    write_json(args.output / "run_records.json", records)
    write_json(args.output / "conflict_validation.json", conflicts)
    write_json(args.output / "primitive_runtime_validation.json", primitive_runtime)
    write_json(args.output / "budget_truncation_validation.json", truncation)
    write_json(args.output / "rng_consumption_validation.json", rng)
    write_json(args.output / "information_boundary_validation.json", information)
    write_json(
        args.output / "opportunity_ledger_validation.json",
        {
            "passed": execution["ledgerOpportunityChecksPassed"] == execution["runs"],
            "runs": execution["runs"],
            "checksPassed": execution["ledgerOpportunityChecksPassed"],
            "budgetTruncation": truncation,
        },
    )
    write_json(
        args.output.parent / "scheduler_fixtures.json",
        fixtures,
    )

    components = {
        "prespecificationHash": prespec_hash == SCHEDULER_PRESPECIFICATION_SHA256,
        "activationDistribution": bool(distribution["passed"]),
        "fairness": bool(fairness["passed"]),
        "executionGrid": bool(execution["passed"]),
        "primitiveRuntime": bool(primitive_runtime["passed"]),
        "conflicts": bool(conflicts["passed"]),
        "budgetTruncation": bool(truncation["passed"]),
        "fixtures": bool(fixtures["passed"]),
        "rngConsumption": bool(rng["passed"]),
        "informationBoundary": bool(information["passed"]),
    }
    summary = {
        "schemaVersion": "E02.scheduler-validation.v1",
        "researchStepId": "S04",
        "passed": all(components.values()),
        "components": components,
        "distribution": distribution,
        "fairness": fairness,
        "execution": execution,
        "primitiveRuntime": primitive_runtime,
        "conflicts": conflicts,
        "budgetTruncation": truncation,
        "fixtures": {
            "passed": fixtures["passed"],
            "cases": len(fixtures["cases"]),
        },
        "rng": rng,
        "informationBoundary": information,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "workers": 1,
            "threadEnvironmentOverrides": "none",
        },
    }
    write_json(args.output / "validation_summary.json", summary)
    if not summary["passed"]:
        raise SystemExit("S04 validation failed: " + json.dumps(components, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
