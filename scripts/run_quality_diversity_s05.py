#!/usr/bin/env python3
"""Execute the frozen, train-only E07 S05 Pareto MAP-Elites protocol."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import yaml

from src.policy_dsl import compile_policy
from src.quality_diversity.core import (
    TASK_IDS,
    aggregate_candidate,
    archive_insert,
    bootstrap_objective_intervals,
    compatible_with_task,
    counter_u64,
    dominates,
    evaluate_work_item,
    finite_behavior_deduplicate,
    load_seed_records,
    mutate_policy,
    policy_body_sha256,
    stable_cell_audit,
    validate_gate_and_inputs,
)


REPOSITORY = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPOSITORY / "configs/search/s05_qd_protocol.yaml"
OUTPUT = Path("/artifacts/research_steps/S05")
CACHE = Path("/cache/e07_s05")
EVALUATION_CACHE = CACHE / "evaluations"
PLAN_CACHE = CACHE / "generation_plans"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_hash(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_bytes(value)
    ).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_bytes(row).decode("ascii") + "\n")


def protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def protocol_hash() -> str:
    return hash_file(PROTOCOL_PATH)


def assert_protocol_lock(*, create: bool = False) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = CACHE / "protocol.sha256"
    actual = protocol_hash()
    if create and not lock.exists():
        lock.write_text(actual + "\n", encoding="ascii")
    if not lock.exists() or lock.read_text(encoding="ascii").strip() != actual:
        raise RuntimeError("S05 protocol hash changed after freeze")
    copied = OUTPUT / "search_protocol.yaml"
    if copied.exists() and hash_file(copied) != actual:
        raise RuntimeError(
            "artifact protocol copy differs from frozen repository protocol"
        )


def policy_record_from_seed(seed: Mapping[str, Any]) -> dict[str, Any]:
    document = deepcopy(seed["canonicalPolicy"])
    compiled = compile_policy(document)
    return {
        "schemaVersion": "e07.s05.policy-catalog-row.v1",
        "policyId": str(seed["policyId"]),
        "policySha256": compiled.policy_sha256,
        "policyBodySha256": policy_body_sha256(document),
        "document": document,
        "environment": str(seed["environment"]),
        "origin": "frozen_s03_seed",
        "family": str(seed["family"]),
        "generation": 0,
        "parents": list(seed.get("parents", [])),
        "mutation": None,
        "s03BoundedSemanticSha256": seed.get("boundedSemanticSha256"),
    }


def build_seed_catalog() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    records = [policy_record_from_seed(seed) for seed in load_seed_records()]
    by_hash = {item["policySha256"]: item for item in records}
    if len(by_hash) != len(records):
        raise RuntimeError("canonical duplicate present in frozen seed library")
    by_id = {item["policyId"]: item for item in records}
    return by_hash, by_id


def faithful_documents(
    by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        "Bubble": deepcopy(by_id["bubble_cell_view_v1"]["document"]),
        "Insertion": deepcopy(by_id["insertion_cell_view_v1"]["document"]),
    }


def evaluation_key(work: Mapping[str, Any]) -> str:
    return canonical_hash(
        "E07/S05/evaluation-cache-key/v1",
        {
            "protocolSha256": protocol_hash(),
            "taskId": work["taskId"],
            "policySha256": compile_policy(work["document"]).policy_sha256,
            "scenarioOrdinal": work["scenarioOrdinal"],
            "stage": work["stage"],
        },
    )


def evaluate_batch(
    works: Sequence[Mapping[str, Any]],
    *,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert_protocol_lock()
    validate_gate_and_inputs(PROTOCOL_PATH)
    EVALUATION_CACHE.mkdir(parents=True, exist_ok=True)
    hits: list[dict[str, Any]] = []
    pending: list[tuple[str, Mapping[str, Any]]] = []
    for work in works:
        key = evaluation_key(work)
        cache_path = EVALUATION_CACHE / f"{key}.json"
        if cache_path.exists():
            hits.append(json.loads(cache_path.read_text(encoding="utf-8")))
        else:
            pending.append((key, work))
    rows = list(hits)
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(evaluate_work_item, work): (key, work)
                for key, work in pending
            }
            for future in as_completed(futures):
                key, _ = futures[future]
                row = json.loads(canonical_bytes(future.result()).decode("ascii"))
                write_json(EVALUATION_CACHE / f"{key}.json", row)
                rows.append(row)
    elapsed = time.perf_counter() - started
    validate_gate_and_inputs(PROTOCOL_PATH)
    return (
        sorted(
            rows,
            key=lambda item: (
                item["taskId"],
                item["policySha256"],
                item["scenarioOrdinal"],
            ),
        ),
        {
            "requested": len(works),
            "cacheHits": len(hits),
            "nativeTopLevelEvaluations": len(pending),
            "workers": workers,
            "wallSeconds": elapsed,
        },
    )


def smoke() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    gate_start = validate_gate_and_inputs(PROTOCOL_PATH)
    assert_protocol_lock(create=True)
    shutil.copyfile(PROTOCOL_PATH, OUTPUT / "search_protocol.yaml")
    by_hash, by_id = build_seed_catalog()
    _ = by_hash
    faithful = faithful_documents(by_id)
    chosen = {
        task_id: (
            "spatial_greedy_adjacent_only_v1"
            if task_id.startswith("e07_s02_spatial2d_")
            else "bubble_cell_view_v1"
        )
        for task_id in TASK_IDS
    }
    works = [
        {
            "taskId": task_id,
            "scenarioOrdinal": 0,
            "document": by_id[chosen[task_id]]["document"],
            "faithfulDocuments": faithful,
            "stage": "primary",
        }
        for task_id in TASK_IDS
    ]
    rows, accounting = evaluate_batch(works, workers=8)
    if len(rows) != len(TASK_IDS) or any(
        row["failed"] or not row["replayPass"] or not all(row["validation"].values())
        for row in rows
    ):
        raise RuntimeError("S05 smoke did not produce eight valid native episodes")
    gate_end = validate_gate_and_inputs(PROTOCOL_PATH)
    gate_artifact = {
        "schemaVersion": "e07.s05.gate-revalidation-checkpoints.v1",
        "researchStepId": "S05",
        "success": True,
        "checkpoints": {"beforeSmoke": gate_start, "afterSmoke": gate_end},
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
    }
    write_json(OUTPUT / "gate_revalidation.json", gate_artifact)
    throughput = {
        "schemaVersion": "e07.s05.smoke-throughput.v1",
        "researchStepId": "S05",
        "success": True,
        "accounting": accounting,
        "rows": [
            {
                "taskId": row["taskId"],
                "policyId": row["policyId"],
                "elapsedSeconds": row["elapsedSeconds"],
                "failed": row["failed"],
                "replayPass": row["replayPass"],
                "nativeUnit": row["nativeUnit"],
                "stopReason": row["stopReason"],
            }
            for row in rows
        ],
        "threadsPerWorker": 1,
        "environmentVariables": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    write_json(OUTPUT / "smoke_throughput.json", throughput)
    write_json(CACHE / "smoke_rows.json", rows)


def _weighted_operator(operators: Sequence[Mapping[str, Any]], address: int) -> str:
    total = sum(float(item["weight"]) for item in operators)
    needle = (address % 10**12) / 10**12 * total
    cumulative = 0.0
    for item in operators:
        cumulative += float(item["weight"])
        if needle < cumulative:
            return str(item["id"])
    return str(operators[-1]["id"])


def _archive_parent(
    archive: Mapping[tuple[str, tuple[int, ...]], Sequence[Mapping[str, Any]]],
    address: int,
) -> Mapping[str, Any]:
    cells = sorted(archive)
    if not cells:
        raise RuntimeError("cannot select a parent from an empty archive")
    cell = cells[address % len(cells)]
    front = sorted(archive[cell], key=lambda item: item["policySha256"])
    return front[(address // max(1, len(cells))) % len(front)]


def _build_archive(
    aggregates: Sequence[Mapping[str, Any]],
    *,
    descriptor_filters: Mapping[str, set[str]] | None = None,
) -> tuple[dict[tuple[str, tuple[int, ...]], list[Mapping[str, Any]]], dict[str, int]]:
    archive: dict[tuple[str, tuple[int, ...]], list[Mapping[str, Any]]] = {}
    totals = {"newCells": 0, "frontAdditions": 0, "dominatedRemoved": 0, "rejected": 0}
    for item in sorted(aggregates, key=lambda row: row["policySha256"]):
        descriptor_filter = None
        if descriptor_filters is not None:
            descriptor_filter = descriptor_filters.get(str(item["policySha256"]), set())
        stats = archive_insert(archive, item, descriptor_filter=descriptor_filter)
        for key, value in stats.items():
            totals[key] += value
    return archive, totals


def _panel_rows(
    ledger: Mapping[tuple[str, str, int], Mapping[str, Any]],
    task_id: str,
    policy_hash: str,
    ordinals: Sequence[int],
) -> list[Mapping[str, Any]]:
    return [ledger[(task_id, policy_hash, ordinal)] for ordinal in ordinals]


def _candidate_aggregates(
    task_id: str,
    candidates: Mapping[str, Mapping[str, Any]],
    ledger: Mapping[tuple[str, str, int], Mapping[str, Any]],
    ordinals: Sequence[int],
) -> list[dict[str, Any]]:
    return [
        aggregate_candidate(
            candidate,
            _panel_rows(ledger, task_id, policy_hash, ordinals),
        )
        for policy_hash, candidate in sorted(candidates.items())
        if all((task_id, policy_hash, ordinal) in ledger for ordinal in ordinals)
    ]


def _works_for(
    task_id: str,
    policies: Sequence[Mapping[str, Any]],
    ordinals: Sequence[int],
    faithful: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "taskId": task_id,
            "scenarioOrdinal": ordinal,
            "document": policy["document"],
            "faithfulDocuments": faithful,
            "stage": "primary"
            if ordinal in primary_ordinals(task_id)
            else "reevaluation",
        }
        for policy in policies
        for ordinal in ordinals
    ]


def primary_ordinals(task_id: str) -> tuple[int, ...]:
    allocation = protocol()["allocation"]
    values = (
        allocation["e05RegenerationPrimaryOrdinals"]
        if task_id == "e07_s02_regeneration_1d"
        else allocation["primaryScenarioOrdinalsDefault"]
    )
    return tuple(int(value) for value in values)


def reevaluation_ordinals(task_id: str) -> tuple[int, ...]:
    allocation = protocol()["allocation"]
    values = (
        allocation["e05RegenerationReevaluationOrdinals"]
        if task_id == "e07_s02_regeneration_1d"
        else allocation["reevaluationScenarioOrdinalsDefault"]
    )
    return tuple(int(value) for value in values)


def _update_ledger(
    ledger: dict[tuple[str, str, int], dict[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    for row in rows:
        key = (
            str(row["taskId"]),
            str(row["policySha256"]),
            int(row["scenarioOrdinal"]),
        )
        previous = ledger.get(key)
        if (
            previous is not None
            and previous["stableEvaluationSha256"] != row["stableEvaluationSha256"]
        ):
            raise RuntimeError(f"evaluation cache disagreement for {key}")
        ledger[key] = dict(row)


def _archive_snapshot(
    archive: Mapping[tuple[str, tuple[int, ...]], Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        {
            "archiveId": archive_id,
            "cell": list(cell),
            "policySha256": item["policySha256"],
            "qualityMinimization": item["qualityMinimization"],
        }
        for (archive_id, cell), front in sorted(archive.items())
        for item in sorted(front, key=lambda row: row["policySha256"])
    ]


def freeze_generation_plan(
    task_id: str,
    generation: int,
    offspring: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Freeze first decision and replay that authoritative plan after restart."""

    PLAN_CACHE.mkdir(parents=True, exist_ok=True)
    path = PLAN_CACHE / f"{task_id}.generation-{generation:02d}.json"
    payload = {
        "schemaVersion": "e07.s05.generation-plan.v1",
        "protocolSha256": protocol_hash(),
        "taskId": task_id,
        "generation": generation,
        "offspring": [
            {
                "policyId": item["policyId"],
                "policySha256": item["policySha256"],
                "policyBodySha256": item["policyBodySha256"],
                "parents": item["parents"],
                "mutation": item["mutation"],
                "document": item["document"],
            }
            for item in offspring
        ],
    }
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if (
            prior.get("schemaVersion") != payload["schemaVersion"]
            or prior.get("protocolSha256") != payload["protocolSha256"]
            or prior.get("taskId") != task_id
            or prior.get("generation") != generation
        ):
            raise RuntimeError(f"invalid persisted generation plan: {path}")
        matches = canonical_bytes(prior) == canonical_bytes(payload)
        resolved = [
            {
                "schemaVersion": "e07.s05.policy-catalog-row.v1",
                **item,
                "environment": item["document"]["environment"],
                "origin": "s05_search",
                "family": "mutated_dsl",
                "generation": generation,
                "searchTaskId": task_id,
            }
            for item in prior["offspring"]
        ]
        return resolved, matches
    else:
        write_json(path, payload)
        return [dict(item) for item in offspring], True


def _seed_improvements(
    final_items: Sequence[Mapping[str, Any]],
    seed_items: Sequence[Mapping[str, Any]],
    stable_filters: Mapping[str, set[str]],
    seed_hashes: set[str],
) -> dict[str, Any]:
    seed_archive, _ = _build_archive(
        [item for item in seed_items if item["policySha256"] in seed_hashes],
        descriptor_filters=stable_filters,
    )
    improvements = []
    for candidate in final_items:
        if candidate["policySha256"] in seed_hashes:
            continue
        for descriptor in candidate["descriptors"]:
            if descriptor["archiveId"] not in stable_filters.get(
                candidate["policySha256"], set()
            ):
                continue
            key = (descriptor["archiveId"], tuple(descriptor["cell"]))
            seed_front = list(seed_archive.get(key, []))
            if not seed_front:
                improvements.append(
                    {
                        "type": "stable_new_cell_not_occupied_by_seed",
                        "archiveId": descriptor["archiveId"],
                        "cell": descriptor["cell"],
                        "policySha256": candidate["policySha256"],
                    }
                )
            elif any(dominates(candidate, seed) for seed in seed_front) and not any(
                dominates(seed, candidate) for seed in seed_front
            ):
                improvements.append(
                    {
                        "type": "same_cell_pareto_seed_improvement",
                        "archiveId": descriptor["archiveId"],
                        "cell": descriptor["cell"],
                        "policySha256": candidate["policySha256"],
                        "dominatedSeedPolicySha256": sorted(
                            seed["policySha256"]
                            for seed in seed_front
                            if dominates(candidate, seed)
                        ),
                    }
                )
    return {
        "seedStableCells": len(seed_archive),
        "stableImprovementCount": len(improvements),
        "improvements": improvements,
    }


def search() -> None:
    assert_protocol_lock()
    if not (OUTPUT / "smoke_throughput.json").exists():
        raise RuntimeError("smoke must pass before substantive search")
    frozen = protocol()
    gate_before = validate_gate_and_inputs(PROTOCOL_PATH)
    seed_catalog, by_id = build_seed_catalog()
    faithful = faithful_documents(by_id)
    config = frozen["allocation"]
    operators = frozen["mutationOperators"]["operators"]
    workers = int(config["maximumParallelWorkers"])
    ledger: dict[tuple[str, str, int], dict[str, Any]] = {}
    _update_ledger(
        ledger, json.loads((CACHE / "smoke_rows.json").read_text(encoding="utf-8"))
    )
    policy_catalog: dict[str, dict[str, Any]] = dict(seed_catalog)
    task_candidates: dict[str, dict[str, dict[str, Any]]] = {}
    task_seed_hashes: dict[str, set[str]] = {}
    convergence_rows: list[dict[str, Any]] = []
    duplicate_audit: dict[str, Any] = {
        "schemaVersion": "e07.s05.duplicate-audit.v1",
        "structuralRejectedBeforeEvaluation": 0,
        "canonicalRejectedBeforeEvaluation": 0,
        "invalidOrNoChangeMutations": 0,
        "finiteSemanticPrimary": [],
        "finiteSemanticFinal": [],
        "generationPlanReplayMismatches": [],
        "criterion": frozen["duplicates"]["boundedSemanticCriterion"],
        "finiteTrainingSupportOnly": True,
    }
    batch_accounting = []

    for task_id in TASK_IDS:
        seeds = {
            policy_hash: policy
            for policy_hash, policy in seed_catalog.items()
            if compatible_with_task(task_id, policy["document"])
            and not (
                task_id == "e07_s02_chimera_1d"
                and "selection" in policy["policyId"].lower()
            )
        }
        if not seeds:
            raise RuntimeError(f"no compatible S03 seed for {task_id}")
        task_candidates[task_id] = dict(seeds)
        task_seed_hashes[task_id] = set(seeds)
        rows, accounting = evaluate_batch(
            _works_for(
                task_id, list(seeds.values()), primary_ordinals(task_id), faithful
            ),
            workers=workers,
        )
        accounting.update({"taskId": task_id, "phase": "complete_seed_baselines"})
        batch_accounting.append(accounting)
        _update_ledger(ledger, rows)
        aggregates = _candidate_aggregates(
            task_id, task_candidates[task_id], ledger, primary_ordinals(task_id)
        )
        retained, audit = finite_behavior_deduplicate(aggregates)
        duplicate_audit["finiteSemanticPrimary"].extend(
            [{"taskId": task_id, **item} for item in audit]
        )
        archive, _ = _build_archive(retained)
        prior_cells = len(archive)
        stalled = 0
        generation = 1
        attempts = 0
        maximum_attempts = int(frozen["mutationOperators"]["maxAttemptsPerTask"])
        while generation <= int(config["maximumGenerations"]) and len(
            task_candidates[task_id]
        ) < int(config["candidateBudgetPerTask"]):
            remaining_generations = int(config["maximumGenerations"]) - generation + 1
            needed = int(config["candidateBudgetPerTask"]) - len(
                task_candidates[task_id]
            )
            batch_target = math.ceil(needed / remaining_generations)
            offspring: list[dict[str, Any]] = []
            while len(offspring) < batch_target and attempts < maximum_attempts:
                local_attempt = attempts
                attempts += 1
                parent_aggregate = _archive_parent(
                    archive,
                    counter_u64(task_id, generation, local_attempt, stream="parent"),
                )
                parent = task_candidates[task_id][parent_aggregate["policySha256"]]
                operator_id = _weighted_operator(
                    operators,
                    counter_u64(task_id, generation, local_attempt, stream="operator"),
                )
                mutated = mutate_policy(
                    parent["document"],
                    operator_id,
                    address=counter_u64(
                        task_id, generation, local_attempt, stream="mutation"
                    ),
                )
                if mutated is None:
                    duplicate_audit["invalidOrNoChangeMutations"] += 1
                    continue
                document, detail = mutated
                if not compatible_with_task(task_id, document):
                    duplicate_audit["invalidOrNoChangeMutations"] += 1
                    continue
                compiled = compile_policy(document)
                body_hash = policy_body_sha256(document)
                if any(
                    item["policyBodySha256"] == body_hash
                    for item in task_candidates[task_id].values()
                ):
                    duplicate_audit["structuralRejectedBeforeEvaluation"] += 1
                    continue
                if compiled.policy_sha256 in task_candidates[task_id]:
                    duplicate_audit["canonicalRejectedBeforeEvaluation"] += 1
                    continue
                offspring_record = {
                    "schemaVersion": "e07.s05.policy-catalog-row.v1",
                    "policyId": document["policyId"],
                    "policySha256": compiled.policy_sha256,
                    "policyBodySha256": body_hash,
                    "document": document,
                    "environment": document["environment"],
                    "origin": "s05_search",
                    "family": "mutated_dsl",
                    "generation": generation,
                    "parents": [parent["policySha256"]],
                    "mutation": detail,
                    "searchTaskId": task_id,
                }
                task_candidates[task_id][compiled.policy_sha256] = offspring_record
                policy_catalog.setdefault(compiled.policy_sha256, offspring_record)
                offspring.append(offspring_record)
            if not offspring:
                convergence_rows.append(
                    {
                        "taskId": task_id,
                        "generation": generation,
                        "evaluatedCandidateCount": len(task_candidates[task_id]),
                        "newCells": 0,
                        "newSeedFrontDominance": 0,
                        "occupancyGrowth": 0.0,
                        "stalled": True,
                        "stopReason": "mutation_representation_exhaustion",
                    }
                )
                break
            computed_offspring = list(offspring)
            offspring, plan_matches = freeze_generation_plan(
                task_id, generation, computed_offspring
            )
            if not plan_matches:
                duplicate_audit["generationPlanReplayMismatches"].append(
                    {
                        "taskId": task_id,
                        "generation": generation,
                        "computedPolicySha256": [
                            item["policySha256"] for item in computed_offspring
                        ],
                        "authoritativePolicySha256": [
                            item["policySha256"] for item in offspring
                        ],
                        "resolution": "persisted_pre_evaluation_plan_retained",
                        "cause": "pre_fix_in_memory_vs_canonical_json_tie_representation",
                    }
                )
                for item in computed_offspring:
                    task_candidates[task_id].pop(item["policySha256"], None)
                    catalog_item = policy_catalog.get(item["policySha256"])
                    if (
                        catalog_item is not None
                        and catalog_item.get("searchTaskId") == task_id
                        and catalog_item.get("generation") == generation
                    ):
                        policy_catalog.pop(item["policySha256"], None)
                for item in offspring:
                    task_candidates[task_id][item["policySha256"]] = item
                    policy_catalog.setdefault(item["policySha256"], item)
            rows, accounting = evaluate_batch(
                _works_for(task_id, offspring, primary_ordinals(task_id), faithful),
                workers=workers,
            )
            accounting.update({"taskId": task_id, "phase": f"generation_{generation}"})
            batch_accounting.append(accounting)
            _update_ledger(ledger, rows)
            aggregates = _candidate_aggregates(
                task_id, task_candidates[task_id], ledger, primary_ordinals(task_id)
            )
            retained, audit = finite_behavior_deduplicate(aggregates)
            duplicate_audit["finiteSemanticPrimary"].extend(
                [
                    {"taskId": task_id, "generation": generation, **item}
                    for item in audit
                ]
            )
            new_archive, _ = _build_archive(retained)
            new_cells = len(set(new_archive) - set(archive))
            occupancy_growth = (len(new_archive) - prior_cells) / max(1, prior_cells)
            seed_aggregates = [
                item
                for item in aggregates
                if item["policySha256"] in task_seed_hashes[task_id]
            ]
            seed_archive, _ = _build_archive(seed_aggregates)
            seed_dominance = 0
            for item in aggregates:
                if item["policySha256"] in task_seed_hashes[task_id]:
                    continue
                for descriptor in item["descriptors"]:
                    key = (descriptor["archiveId"], tuple(descriptor["cell"]))
                    if any(dominates(item, seed) for seed in seed_archive.get(key, [])):
                        seed_dominance += 1
                        break
            stall_now = (
                new_cells == 0
                and seed_dominance == 0
                and occupancy_growth < 0.01
                and len(task_candidates[task_id])
                >= int(config["minimumEvaluatedCandidatesPerTask"])
            )
            stalled = stalled + 1 if stall_now else 0
            stop_reason = None
            if stalled >= int(frozen["convergence"]["patienceGenerations"]):
                stop_reason = "prespecified_stall"
            elif len(task_candidates[task_id]) >= int(config["candidateBudgetPerTask"]):
                stop_reason = "candidate_budget"
            convergence_rows.append(
                {
                    "taskId": task_id,
                    "generation": generation,
                    "offspringEvaluated": len(offspring),
                    "evaluatedCandidateCount": len(task_candidates[task_id]),
                    "archiveCells": len(new_archive),
                    "archiveFrontEntries": sum(
                        len(front) for front in new_archive.values()
                    ),
                    "newCells": new_cells,
                    "newSeedFrontDominance": seed_dominance,
                    "occupancyGrowth": occupancy_growth,
                    "stalled": stall_now,
                    "consecutiveStalls": stalled,
                    "stopReason": stop_reason,
                }
            )
            archive = new_archive
            prior_cells = len(archive)
            generation += 1
            if stop_reason is not None:
                break

    gate_after_primary = validate_gate_and_inputs(PROTOCOL_PATH)
    provisional_archives: dict[str, Any] = {}
    reevaluate: dict[str, set[str]] = {}
    for task_id in TASK_IDS:
        aggregates = _candidate_aggregates(
            task_id, task_candidates[task_id], ledger, primary_ordinals(task_id)
        )
        retained, _ = finite_behavior_deduplicate(aggregates)
        archive, _ = _build_archive(retained)
        provisional_archives[task_id] = archive
        reevaluate[task_id] = {
            item["policySha256"] for front in archive.values() for item in front
        } | task_seed_hashes[task_id]
        policies = [
            task_candidates[task_id][policy_hash]
            for policy_hash in sorted(reevaluate[task_id])
        ]
        rows, accounting = evaluate_batch(
            _works_for(task_id, policies, reevaluation_ordinals(task_id), faithful),
            workers=workers,
        )
        accounting.update({"taskId": task_id, "phase": "archive_reevaluation"})
        batch_accounting.append(accounting)
        _update_ledger(ledger, rows)

    gate_after_reevaluation = validate_gate_and_inputs(PROTOCOL_PATH)
    stability_rows = []
    interval_rows = []
    final_archives = {}
    seed_comparisons = {}
    final_aggregates_by_task = {}
    final_duplicate_retained = {}
    bootstrap_reps = int(
        frozen["uncertaintyAndReevaluation"]["scenarioBlockBootstrapReplicates"]
    )
    interval_reps = int(
        frozen["uncertaintyAndReevaluation"]["objectiveIntervalBootstrapReplicates"]
    )
    stability_threshold = float(
        frozen["uncertaintyAndReevaluation"]["stableCellProbabilityThreshold"]
    )
    for task_id in TASK_IDS:
        all_ordinals = primary_ordinals(task_id) + reevaluation_ordinals(task_id)
        policies = {
            policy_hash: task_candidates[task_id][policy_hash]
            for policy_hash in reevaluate[task_id]
        }
        aggregates = _candidate_aggregates(task_id, policies, ledger, all_ordinals)
        retained, audit = finite_behavior_deduplicate(aggregates)
        final_duplicate_retained[task_id] = retained
        duplicate_audit["finiteSemanticFinal"].extend(
            [{"taskId": task_id, **item} for item in audit]
        )
        stable_filters: dict[str, set[str]] = {}
        for aggregate in retained:
            policy = policies[aggregate["policySha256"]]
            rows = _panel_rows(ledger, task_id, aggregate["policySha256"], all_ordinals)
            stability = stable_cell_audit(
                policy, rows, bootstrap_replicates=bootstrap_reps
            )
            stable_filters[aggregate["policySha256"]] = {
                item["archiveId"]
                for item in stability
                if item["sameCellProbability"] >= stability_threshold
            }
            stability_rows.extend(
                [
                    {
                        "taskId": task_id,
                        "policySha256": aggregate["policySha256"],
                        "stable": item["sameCellProbability"] >= stability_threshold,
                        **item,
                    }
                    for item in stability
                ]
            )
            interval_rows.append(
                {
                    "taskId": task_id,
                    "policySha256": aggregate["policySha256"],
                    "scenarioOrdinals": list(all_ordinals),
                    "intervals": bootstrap_objective_intervals(
                        policy, rows, bootstrap_replicates=interval_reps
                    ),
                    "bootstrapReplicates": interval_reps,
                }
            )
        archive, insertion_stats = _build_archive(
            retained, descriptor_filters=stable_filters
        )
        final_archives[task_id] = archive
        seed_comparisons[task_id] = {
            "taskId": task_id,
            **_seed_improvements(
                retained,
                retained,
                stable_filters,
                task_seed_hashes[task_id],
            ),
        }
        final_aggregates_by_task[task_id] = {
            "aggregates": retained,
            "stableFilters": stable_filters,
            "insertionStats": insertion_stats,
        }

    snapshots = {
        task_id: _archive_snapshot(archive)
        for task_id, archive in final_archives.items()
    }
    reverse_snapshots = {}
    for task_id in TASK_IDS:
        data = final_aggregates_by_task[task_id]
        reversed_archive: dict[
            tuple[str, tuple[int, ...]], list[Mapping[str, Any]]
        ] = {}
        for item in sorted(
            data["aggregates"], key=lambda row: row["policySha256"], reverse=True
        ):
            archive_insert(
                reversed_archive,
                item,
                descriptor_filter=data["stableFilters"].get(
                    item["policySha256"], set()
                ),
            )
        reverse_snapshots[task_id] = _archive_snapshot(reversed_archive)
    worker_order_pass = snapshots == reverse_snapshots
    if not worker_order_pass:
        raise RuntimeError("final archive depends on insertion/worker order")

    archive_entries = [
        {
            "schemaVersion": "e07.s05.archive-entry.v1",
            "taskId": task_id,
            "archiveId": archive_id,
            "cell": list(cell),
            "policySha256": item["policySha256"],
            "policyId": item["policyId"],
            "origin": item["origin"],
            "generation": item["generation"],
            "objectives": item["objectives"],
            "objectiveDirections": item["objectiveDirections"],
            "nativeCostVector": item["costVector"],
            "complexityVector": item["complexityVector"],
            "qualityMinimization": item["qualityMinimization"],
            "boundedSemanticSha256": item["boundedSemanticSha256"],
            "scenarioOrdinals": item["scenarioOrdinals"],
        }
        for task_id, archive in final_archives.items()
        for (archive_id, cell), front in sorted(archive.items())
        for item in sorted(front, key=lambda row: row["policySha256"])
    ]
    lineage_rows = [
        {
            "schemaVersion": "e07.s05.lineage-row.v1",
            "policySha256": item["policySha256"],
            "policyBodySha256": item["policyBodySha256"],
            "policyId": item["policyId"],
            "origin": item["origin"],
            "generation": item["generation"],
            "parents": item["parents"],
            "mutation": item["mutation"],
            "searchTaskId": item.get("searchTaskId"),
        }
        for item in sorted(policy_catalog.values(), key=lambda row: row["policySha256"])
    ]
    selected_ledger_rows = sorted(
        ledger.values(),
        key=lambda row: (row["taskId"], row["policySha256"], row["scenarioOrdinal"]),
    )
    selected_keys = {
        (row["taskId"], row["policySha256"], row["scenarioOrdinal"], row["stage"])
        for row in selected_ledger_rows
    }
    all_executed_rows = []
    for path in sorted(EVALUATION_CACHE.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = (
            row["taskId"],
            row["policySha256"],
            row["scenarioOrdinal"],
            row["stage"],
        )
        all_executed_rows.append(
            {**row, "selectedForCanonicalSearch": key in selected_keys}
        )
    all_executed_rows.sort(
        key=lambda row: (
            row["taskId"],
            row["policySha256"],
            row["scenarioOrdinal"],
            row["stage"],
        )
    )
    evaluation_hashes = [row["stableEvaluationSha256"] for row in all_executed_rows]
    total_native = len(all_executed_rows)
    selected_native = len(selected_ledger_rows)
    recovery_orphans = total_native - selected_native
    if total_native > int(config["maximumTopLevelEvaluations"]):
        raise RuntimeError("top-level evaluation budget exceeded")
    per_task_evaluations = Counter(row["taskId"] for row in all_executed_rows)
    per_task_candidates = {
        task: len(values) for task, values in task_candidates.items()
    }
    minimum_met = all(
        count >= int(config["minimumEvaluatedCandidatesPerTask"])
        for count in per_task_candidates.values()
    )
    all_replay = all(row["replayPass"] for row in all_executed_rows)
    failed_rows = [row for row in all_executed_rows if row["failed"]]
    failed_policy_pairs = {(row["taskId"], row["policySha256"]) for row in failed_rows}
    archived_policy_pairs = {
        (entry["taskId"], entry["policySha256"]) for entry in archive_entries
    }
    failure_handling_pass = not (failed_policy_pairs & archived_policy_pairs)
    archived_native_validation_pass = all(
        not row["failed"] and row["replayPass"] and all(row["validation"].values())
        for row in selected_ledger_rows
        if (row["taskId"], row["policySha256"]) in archived_policy_pairs
    )
    improvements = sum(
        item["stableImprovementCount"] for item in seed_comparisons.values()
    )
    outcome = "supportive" if improvements else "null"

    write_jsonl(OUTPUT / "evaluation_ledger.jsonl", all_executed_rows)
    write_jsonl(
        OUTPUT / "policy_catalog.jsonl",
        sorted(policy_catalog.values(), key=lambda row: row["policySha256"]),
    )
    write_jsonl(
        OUTPUT / "generation_plan_ledger.jsonl",
        [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(PLAN_CACHE.glob("*.json"))
        ],
    )
    write_jsonl(OUTPUT / "lineage_graph.jsonl", lineage_rows)
    write_jsonl(OUTPUT / "archive/archive_entries.jsonl", archive_entries)
    write_jsonl(OUTPUT / "archive/policy_objective_intervals.jsonl", interval_rows)
    write_jsonl(OUTPUT / "archive/descriptor_stability.jsonl", stability_rows)
    archive_manifest = {
        "schemaVersion": "e07.s05.archive-manifest.v1",
        "researchStepId": "S05",
        "taskCount": len(TASK_IDS),
        "archiveCellCount": sum(len(archive) for archive in final_archives.values()),
        "archiveEntryCount": len(archive_entries),
        "perTask": {
            task_id: {
                "archiveCellCount": len(final_archives[task_id]),
                "archiveEntryCount": sum(
                    len(front) for front in final_archives[task_id].values()
                ),
                "snapshotSha256": canonical_hash(
                    "E07/S05/archive-snapshot/v1", snapshots[task_id]
                ),
            }
            for task_id in TASK_IDS
        },
        "taskStratified": True,
        "crossTaskDominance": False,
        "universalNormalizedScore": False,
        "claimBoundary": frozen["claimBoundary"],
    }
    write_json(OUTPUT / "archive/archive_manifest.json", archive_manifest)
    write_json(
        OUTPUT / "convergence_diagnostics.json",
        {
            "schemaVersion": "e07.s05.convergence.v1",
            "researchStepId": "S05",
            "minimumCandidateCoverageMet": minimum_met,
            "rows": convergence_rows,
            "perTaskFinalCandidateCount": per_task_candidates,
        },
    )
    with (OUTPUT / "convergence_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = sorted({key for row in convergence_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(convergence_rows)
    write_json(
        OUTPUT / "seed_comparison.json",
        {
            "schemaVersion": "e07.s05.seed-comparison.v1",
            "researchStepId": "S05",
            "outcomeClassification": outcome,
            "stableImprovementCount": improvements,
            "perTask": seed_comparisons,
            "comparisonBoundary": "same task, archive, cell, scenario panel, and native contract only",
        },
    )
    write_json(OUTPUT / "duplicate_audit.json", duplicate_audit)
    write_json(
        OUTPUT / "scenario_resampling_uncertainty.json",
        {
            "schemaVersion": "e07.s05.resampling-uncertainty.v1",
            "researchStepId": "S05",
            "trainingOnly": True,
            "validationOutcomeEvaluations": 0,
            "confirmationOutcomeEvaluations": 0,
            "primaryOrdinalsByTask": {
                task: list(primary_ordinals(task)) for task in TASK_IDS
            },
            "reevaluationOrdinalsByTask": {
                task: list(reevaluation_ordinals(task)) for task in TASK_IDS
            },
            "descriptorBootstrapReplicates": bootstrap_reps,
            "objectiveBootstrapReplicates": interval_reps,
            "stableCellThreshold": stability_threshold,
            "stableDescriptorRows": sum(item["stable"] for item in stability_rows),
            "unstableDescriptorRows": sum(
                not item["stable"] for item in stability_rows
            ),
            "spatialResampling": frozen["scenarioResampling"]["spatialResample"],
            "regenerationProtectedOrdinalsExcluded": True,
        },
    )
    write_json(
        OUTPUT / "replay_worker_order_validation.json",
        {
            "schemaVersion": "e07.s05.replay-worker-order.v1",
            "researchStepId": "S05",
            "allNativeEpisodesReplayPass": all(
                row["replayPass"] for row in all_executed_rows
            ),
            "allArchiveEntriesNativeValidationPass": archived_native_validation_pass,
            "failedRowsRetainedAndExcludedFromArchive": failure_handling_pass,
            "failedEvaluationRowCount": len(failed_rows),
            "workerOrderIndependencePass": worker_order_pass,
            "forwardSnapshotSha256": canonical_hash(
                "E07/S05/all-archives/v1", snapshots
            ),
            "reverseSnapshotSha256": canonical_hash(
                "E07/S05/all-archives/v1", reverse_snapshots
            ),
            "evaluationLedgerSha256": canonical_hash(
                "E07/S05/evaluation-ledger/v1", evaluation_hashes
            ),
        },
    )
    budget = {
        "schemaVersion": "e07.s05.budget-accounting.v1",
        "researchStepId": "S05",
        "maximumTopLevelEvaluations": int(config["maximumTopLevelEvaluations"]),
        "actualUniqueTopLevelEvaluationsIncludingSmokeAndRecovery": total_native,
        "selectedCanonicalSearchEvaluations": selected_native,
        "recoveryOrphanEvaluationsRetainedInLedger": recovery_orphans,
        "remainingTopLevelEvaluationBudget": int(config["maximumTopLevelEvaluations"])
        - total_native,
        "perTaskTopLevelEvaluations": dict(sorted(per_task_evaluations.items())),
        "perTaskUniqueCandidates": per_task_candidates,
        "batches": batch_accounting,
        "nativeInternalReplayAdditional": True,
        "licensedCapabilityCostsKeptSeparate": True,
        "wallClockStopUsed": False,
        "runtimeDrivenWeakeningUsed": False,
        "maximumWorkers": workers,
        "threadsPerWorker": int(config["threadsPerWorker"]),
    }
    write_json(OUTPUT / "budget_accounting.json", budget)
    gate_final = validate_gate_and_inputs(PROTOCOL_PATH)
    gate_artifact = json.loads(
        (OUTPUT / "gate_revalidation.json").read_text(encoding="utf-8")
    )
    gate_artifact["checkpoints"].update(
        {
            "beforeSubstantiveSearch": gate_before,
            "afterPrimarySearch": gate_after_primary,
            "afterArchiveReevaluation": gate_after_reevaluation,
            "final": gate_final,
        }
    )
    write_json(OUTPUT / "gate_revalidation.json", gate_artifact)
    input_provenance = {
        "schemaVersion": "e07.s05.input-provenance.v1",
        "researchStepId": "S05",
        "protocolSha256": protocol_hash(),
        "gateCheckpoints": list(gate_artifact["checkpoints"]),
        "frozenInputs": frozen["frozenInputs"],
        "eligibilityInputs": frozen["eligibility"],
        "implementationSources": {
            str(path.relative_to(REPOSITORY)): hash_file(path)
            for path in (
                REPOSITORY / "src/quality_diversity/core.py",
                REPOSITORY / "src/environment_suite/runners.py",
                Path(__file__),
            )
        },
        "protectedOutcomeReads": 0,
    }
    write_json(OUTPUT / "input_provenance.json", input_provenance)
    environment = {
        "schemaVersion": "e07.s05.environment.v1",
        "python": sys.version,
        "platform": platform.platform(),
        "cpuCount": os.cpu_count(),
        "workers": workers,
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "repositoryBaseCommit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip(),
        "command": " ".join(sys.argv),
    }
    write_json(OUTPUT / "environment.json", environment)
    validation = {
        "schemaVersion": "e07.s05.validation-summary.v1",
        "researchStepId": "S05",
        "success": (
            all_replay
            and archived_native_validation_pass
            and failure_handling_pass
            and worker_order_pass
            and minimum_met
            and total_native <= int(config["maximumTopLevelEvaluations"])
        ),
        "checks": {
            "gatePassedAtEveryCheckpoint": True,
            "authorizedTrainingOnly": True,
            "validationOutcomeEvaluationsZero": True,
            "confirmationOutcomeEvaluationsZero": True,
            "allArchiveEntriesNativeValid": archived_native_validation_pass,
            "failedRowsRetainedAndArchiveExcluded": failure_handling_pass,
            "failedEvaluationRowCount": len(failed_rows),
            "allNativeReplayPass": all(row["replayPass"] for row in all_executed_rows),
            "workerOrderIndependence": worker_order_pass,
            "candidateMinimumEveryTask": minimum_met,
            "evaluationBudgetRespected": total_native
            <= int(config["maximumTopLevelEvaluations"]),
            "licensedCostFieldsRetained": any(
                "licensedPrefixPredicateEvaluations" in key
                or "engineCursorStateReads" in key
                or "licensedLongRangeRequestedDistance" in key
                for entry in archive_entries
                for key in entry["nativeCostVector"]
            ),
            "finiteSemanticDuplicatesAudited": True,
            "lineageAndPolicyHashesComplete": all(
                item["policySha256"] and item["policyBodySha256"]
                for item in policy_catalog.values()
            ),
        },
        "outcomeClassification": outcome,
        "stableImprovementCount": improvements,
        "caveatsOrBlockers": [
            (
                f"{len(failed_rows)} E05 rows reached source-terminal failure with "
                "developmentBudgetRespected=false; they are retained in the ledger "
                "and excluded from all archives."
            )
        ],
    }
    if not validation["success"]:
        raise RuntimeError("S05 final validation failed")
    write_json(OUTPUT / "validation_summary.json", validation)
    write_json(
        OUTPUT / "result_summary.json",
        {
            "schemaVersion": "e07.s05.result-summary.v1",
            "researchStepId": "S05",
            "success": True,
            "outcomeClassification": outcome,
            "stableImprovementCount": improvements,
            "archiveCellCount": len(archive_entries)
            and sum(len(item) for item in final_archives.values()),
            "archiveEntryCount": len(archive_entries),
            "uniquePolicies": len(policy_catalog),
            "uniqueEvaluatedCandidatesByTask": per_task_candidates,
            "topLevelEvaluations": total_native,
            "selectedCanonicalSearchEvaluations": selected_native,
            "recoveryOrphanEvaluations": recovery_orphans,
            "minimumCandidateCoverageMet": minimum_met,
            "archiveEligibleNativeValidationPass": archived_native_validation_pass,
            "failedEvaluationRowCount": len(failed_rows),
            "failureHandlingPass": failure_handling_pass,
            "workerOrderIndependencePass": worker_order_pass,
            "validationOutcomeEvaluations": 0,
            "confirmationOutcomeEvaluations": 0,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("smoke", "search"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "smoke":
        smoke()
    else:
        search()


if __name__ == "__main__":
    main()
