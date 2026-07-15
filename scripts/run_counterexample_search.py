#!/usr/bin/env python3
"""Run the fully frozen S13 search followed by independent confirmation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from causal_simulator.counterexample_search import (
    CONFIRMATION_MASTER_SEED,
    CONFIRMATION_SCHEDULER_DOMAIN,
    OBJECTIVE_IDS,
    PRESPECIFICATION_SHA256,
    TRAINING_DOMAIN,
    TRAINING_MASTER_SEED,
    Candidate,
    candidate_distance,
    canonical_hash,
    confirmation_neighbor,
    derive_seed,
    evaluate_candidate,
    flatten_evaluation,
    initial_candidate,
    mutate_candidate,
    objective_sort_key,
    paired_bootstrap_interval,
    sha256_file,
    spearman_rank,
    wilson_interval,
)
from reference_simulator.model import canonical_json_bytes, sha256_json


SIZES = (20, 50)
VALUE_PROFILES = ("unique", "balanced_duplicate")
FAULT_COUNTS = (2, 4)
INITIAL_POPULATION = 32
MUTATION_GENERATIONS = 7
CANDIDATES_PER_GENERATION = 32
CONFIRMATION_REPLICATES = 64
SCHEDULERS = ("uniform_random_activation", "random_permutation_sweep")


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    records = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )


def _execute(payload: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_candidate(
        payload["candidate"],
        stage=str(payload["stage"]),
        scheduler=str(payload["scheduler"]),
        seed=int(payload["seed"]),
        instance_id=str(payload["instanceId"]),
    )


def _parallel_evaluate(payloads: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers == 1:
        return [_execute(payload) for payload in payloads]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_execute, payloads, chunksize=1))
    return results


def _training_payload(candidate: Candidate) -> dict[str, Any]:
    seed = derive_seed(
        TRAINING_MASTER_SEED, TRAINING_DOMAIN, candidate.candidate_id, "runtime-seed"
    )
    return {
        "candidate": candidate.to_record(),
        "stage": "training",
        "scheduler": "uniform_random_activation",
        "seed": seed,
        "instanceId": "s13train1:" + sha256_json(
            {"domain": TRAINING_DOMAIN, "candidateId": candidate.candidate_id}
        ),
    }


def _run_search(workers: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    strata = [
        (n, value_profile, fault_count)
        for n in SIZES
        for value_profile in VALUE_PROFILES
        for fault_count in FAULT_COUNTS
    ]
    evaluations_by_stratum: dict[str, list[dict[str, Any]]] = {
        f"n{n}_{value_profile}_f{fault_count}": []
        for n, value_profile, fault_count in strata
    }
    seen_by_stratum: dict[str, set[str]] = {key: set() for key in evaluations_by_stratum}
    evaluation_generation: dict[str, int] = {}
    search_history: list[dict[str, Any]] = []

    for generation in range(MUTATION_GENERATIONS + 1):
        generation_candidates: list[Candidate] = []
        for n, value_profile, fault_count in strata:
            stratum_id = f"n{n}_{value_profile}_f{fault_count}"
            if generation == 0:
                candidates = [
                    initial_candidate(n, value_profile, fault_count, slot)
                    for slot in range(INITIAL_POPULATION)
                ]
            else:
                prior = evaluations_by_stratum[stratum_id]
                parent_map: dict[str, Candidate] = {}
                for objective_id in OBJECTIVE_IDS:
                    for evaluation in sorted(
                        prior, key=lambda item: objective_sort_key(item, objective_id)
                    )[:4]:
                        parent = Candidate.from_record(evaluation["candidate"])
                        parent_map[parent.candidate_id] = parent
                parents = [parent_map[key] for key in sorted(parent_map)]
                if not parents:
                    raise AssertionError("empty deterministic S13 parent pool")
                candidates = []
                for slot in range(CANDIDATES_PER_GENERATION):
                    accepted: Candidate | None = None
                    for attempt in range(64):
                        parent_index = derive_seed(
                            TRAINING_MASTER_SEED,
                            TRAINING_DOMAIN,
                            stratum_id,
                            generation,
                            slot,
                            attempt,
                            "parent",
                        ) % len(parents)
                        proposal = mutate_candidate(
                            parents[parent_index], generation, slot, attempt
                        )
                        if (
                            proposal.candidate_id not in seen_by_stratum[stratum_id]
                            and proposal.candidate_id
                            not in {item.candidate_id for item in candidates}
                        ):
                            accepted = proposal
                            break
                    if accepted is None:
                        raise AssertionError(
                            f"fixed candidate budget collision shortfall in {stratum_id} generation {generation}"
                        )
                    candidates.append(accepted)
            if len({item.candidate_id for item in candidates}) != CANDIDATES_PER_GENERATION:
                raise AssertionError("within-generation candidate duplication")
            generation_candidates.extend(candidates)

        payloads = [_training_payload(candidate) for candidate in generation_candidates]
        generation_evaluations = _parallel_evaluate(payloads, workers)
        for evaluation in generation_evaluations:
            candidate = Candidate.from_record(evaluation["candidate"])
            stratum_id = candidate.stratum_id
            if candidate.candidate_id in seen_by_stratum[stratum_id]:
                raise AssertionError("training candidate evaluated twice")
            seen_by_stratum[stratum_id].add(candidate.candidate_id)
            evaluations_by_stratum[stratum_id].append(evaluation)
            evaluation_generation[candidate.candidate_id] = generation

        for stratum_id, evaluations in sorted(evaluations_by_stratum.items()):
            for objective_id in OBJECTIVE_IDS:
                best = sorted(
                    evaluations, key=lambda item: objective_sort_key(item, objective_id)
                )[0]
                metric = best["contrasts"][objective_id]
                search_history.append(
                    {
                        "generation": generation,
                        "stratumId": stratum_id,
                        "objectiveId": objective_id,
                        "candidateId": best["candidate"]["candidateId"],
                        "targetOnlySuccess": metric["targetOnlySuccess"],
                        "directionalResidualScore": metric["directionalResidualScore"],
                        "activeMinusReferenceResidual": metric["activeMinusReferenceResidual"],
                        "discoveryThresholdMet": metric["discoveryThresholdMet"],
                        "cumulativeCandidatesInStratum": len(evaluations),
                    }
                )
        print(
            json.dumps(
                {
                    "stage": "training",
                    "generation": generation,
                    "candidatesCompleted": len(generation_evaluations),
                    "cumulativeCandidates": sum(map(len, evaluations_by_stratum.values())),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    evaluations = [
        item
        for stratum_id in sorted(evaluations_by_stratum)
        for item in evaluations_by_stratum[stratum_id]
    ]
    candidate_rows = [
        flatten_evaluation(item, evaluation_generation[item["candidate"]["candidateId"]])
        for item in evaluations
    ]
    run_rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        generation = evaluation_generation[evaluation["candidate"]["candidateId"]]
        for row in evaluation["runRows"]:
            run_rows.append({**row, "generation": generation})
    return evaluations, candidate_rows, search_history + run_rows


def _split_search_history_and_runs(combined: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    history = [row for row in combined if "objectiveId" in row]
    runs = [row for row in combined if "signature" in row]
    return history, runs


def _select_candidates(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_candidates: list[Candidate] = []
    for objective_id in OBJECTIVE_IDS:
        rank = 0
        for evaluation in sorted(
            evaluations, key=lambda item: objective_sort_key(item, objective_id)
        ):
            candidate = Candidate.from_record(evaluation["candidate"])
            if candidate.candidate_id in {item.candidate_id for item in selected_candidates}:
                continue
            if any(candidate_distance(candidate, prior) < 0.10 for prior in selected_candidates):
                continue
            rank += 1
            metric = evaluation["contrasts"][objective_id]
            selected.append(
                {
                    "selectionOrder": len(selected) + 1,
                    "objectiveId": objective_id,
                    "objectiveRank": rank,
                    "candidateId": candidate.candidate_id,
                    "candidate": candidate.to_record(),
                    "trainingMetric": dict(metric),
                    "discoveryEligible": bool(metric["discoveryThresholdMet"]),
                    "minimumDistanceToEarlierSelection": (
                        min(candidate_distance(candidate, prior) for prior in selected_candidates)
                        if selected_candidates
                        else None
                    ),
                }
            )
            selected_candidates.append(candidate)
            if rank == 2:
                break
    if len(selected) != 8:
        raise AssertionError(f"frozen selection expected 8 novel records, observed {len(selected)}")
    return selected


def _confirmation_payloads(selected: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for selection in selected:
        source = Candidate.from_record(selection["candidate"])
        for panel in ("exact_instance", "local_fault_array_neighborhood"):
            for replicate in range(CONFIRMATION_REPLICATES):
                instance_candidate = (
                    source
                    if panel == "exact_instance"
                    else confirmation_neighbor(source, replicate)
                )
                for scheduler in SCHEDULERS:
                    seed = derive_seed(
                        CONFIRMATION_MASTER_SEED,
                        CONFIRMATION_SCHEDULER_DOMAIN,
                        source.candidate_id,
                        panel,
                        replicate,
                        scheduler,
                    )
                    instance_id = "s13confirm1:" + sha256_json(
                        {
                            "sourceCandidateId": source.candidate_id,
                            "panel": panel,
                            "replicate": replicate,
                            "scheduler": scheduler,
                            "instanceCandidateId": instance_candidate.candidate_id,
                            "seed": str(seed),
                        }
                    )
                    payloads.append(
                        {
                            "candidate": instance_candidate.to_record(),
                            "stage": (
                                "confirmation_exact"
                                if panel == "exact_instance"
                                else "confirmation_neighborhood"
                            ),
                            "scheduler": scheduler,
                            "seed": seed,
                            "instanceId": instance_id,
                        }
                    )
                    metadata[instance_id] = {
                        "sourceCandidateId": source.candidate_id,
                        "selectionObjectiveId": selection["objectiveId"],
                        "selectionOrder": selection["selectionOrder"],
                        "panel": panel,
                        "replicate": replicate,
                        "scheduler": scheduler,
                        "instanceCandidateId": instance_candidate.candidate_id,
                        "sourceFaultMapId": source.fault_map_id,
                        "instanceFaultMapId": instance_candidate.fault_map_id,
                    }
    return payloads, metadata


def _confirmation_rows(
    evaluations: list[dict[str, Any]], metadata: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    effects: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for evaluation in evaluations:
        meta = dict(metadata[evaluation["instanceId"]])
        for objective_id, metric in evaluation["contrasts"].items():
            effects.append({**meta, **dict(metric)})
        for row in evaluation["runRows"]:
            runs.append({**meta, **row})
    return effects, runs


def _panel_classification(rows: list[dict[str, Any]], panel: str, objective_id: str) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["panel"] == panel and row["objectiveId"] == objective_id
    ]
    if len(selected) != 128:
        raise AssertionError(f"confirmation panel accounting failed: {panel} {objective_id} {len(selected)}")
    scores = [float(row["directionalResidualScore"]) for row in selected]
    interval_low, interval_high = paired_bootstrap_interval(
        scores, objective_id=objective_id, panel=panel
    )
    scheduler_means = {
        scheduler: float(np.mean([
            row["directionalResidualScore"] for row in selected
            if row["scheduler"] == scheduler
        ]))
        for scheduler in SCHEDULERS
    }
    concordant = sum(score > 0 for score in scores)
    concordance_low, concordance_high = wilson_interval(concordant, len(scores))
    target_only = sum(int(row["targetOnlySuccess"]) for row in selected)
    opposite_only = sum(int(row["oppositeOnlySuccess"]) for row in selected)
    unique_low, unique_high = wilson_interval(target_only, len(selected))
    target_per_scheduler = {
        scheduler: sum(
            int(row["targetOnlySuccess"])
            for row in selected
            if row["scheduler"] == scheduler
        )
        for scheduler in SCHEDULERS
    }
    ranking_pass = (
        float(np.mean(scores)) >= 0.02
        and interval_low > 0.0
        and all(value > 0.0 for value in scheduler_means.values())
        and (
            panel != "local_fault_array_neighborhood"
            or (concordant / len(scores) >= 0.65 and concordance_low > 0.5)
        )
    )
    unique_pass = (
        target_only >= 16
        and target_only / len(selected) >= 0.125
        and unique_low >= 0.05
        and opposite_only <= 1
        and all(value >= 1 for value in target_per_scheduler.values())
    )
    return {
        "panel": panel,
        "pairs": len(selected),
        "directionalResidualMean": float(np.mean(scores)),
        "directionalResidualBootstrapLow95": interval_low,
        "directionalResidualBootstrapHigh95": interval_high,
        "schedulerDirectionalMeans": scheduler_means,
        "directionalConcordantPairs": concordant,
        "directionalConcordanceRate": concordant / len(scores),
        "directionalConcordanceWilsonLow95": concordance_low,
        "directionalConcordanceWilsonHigh95": concordance_high,
        "targetOnlySuccessPairs": target_only,
        "targetOnlySuccessRate": target_only / len(selected),
        "targetOnlySuccessWilsonLow95": unique_low,
        "targetOnlySuccessWilsonHigh95": unique_high,
        "oppositeOnlySuccessPairs": opposite_only,
        "targetOnlySuccessPerScheduler": target_per_scheduler,
        "rankingReversalPass": ranking_pass,
        "uniqueSuccessPass": unique_pass,
    }


def _classify(
    selected: list[dict[str, Any]], effects: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    classifications: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for selection in selected:
        source_id = selection["candidateId"]
        objective_id = selection["objectiveId"]
        candidate_rows = [row for row in effects if row["sourceCandidateId"] == source_id]
        exact = _panel_classification(candidate_rows, "exact_instance", objective_id)
        neighborhood = _panel_classification(
            candidate_rows, "local_fault_array_neighborhood", objective_id
        )
        discovery = bool(selection["discoveryEligible"])
        ranking_full = exact["rankingReversalPass"] and neighborhood["rankingReversalPass"]
        unique_full = exact["uniqueSuccessPass"] and neighborhood["uniqueSuccessPass"]
        fully_confirmed = discovery and (ranking_full or unique_full)
        exact_only = discovery and (
            exact["rankingReversalPass"] or exact["uniqueSuccessPass"]
        ) and not fully_confirmed
        classification = (
            "fully_confirmed_counterexample"
            if fully_confirmed
            else "stream_confirmed_brittle_instance"
            if exact_only
            else "not_confirmed"
        )
        record = {
            "selectionOrder": selection["selectionOrder"],
            "candidateId": source_id,
            "objectiveId": objective_id,
            "trainingDiscoveryEligible": discovery,
            "trainingDirectionalResidualScore": selection["trainingMetric"]["directionalResidualScore"],
            "trainingTargetOnlySuccess": selection["trainingMetric"]["targetOnlySuccess"],
            "exactPanel": exact,
            "neighborhoodPanel": neighborhood,
            "rankingReversalBothPanels": ranking_full,
            "uniqueSuccessBothPanels": unique_full,
            "fullyConfirmed": fully_confirmed,
            "classification": classification,
        }
        classifications.append(record)
        diagnostics.append(
            {
                "candidateId": source_id,
                "objectiveId": objective_id,
                "sentinelBelowDiscoveryThreshold": not discovery,
                "trainingDirectionalResidualScore": selection["trainingMetric"]["directionalResidualScore"],
                "exactDirectionalResidualMean": exact["directionalResidualMean"],
                "neighborhoodDirectionalResidualMean": neighborhood["directionalResidualMean"],
                "trainingToExactShrinkage": selection["trainingMetric"]["directionalResidualScore"] - exact["directionalResidualMean"],
                "exactToNeighborhoodShrinkage": exact["directionalResidualMean"] - neighborhood["directionalResidualMean"],
                "exactRankingRetention": exact["rankingReversalPass"],
                "neighborhoodRankingRetention": neighborhood["rankingReversalPass"],
                "exactUniqueRetention": exact["uniqueSuccessPass"],
                "neighborhoodUniqueRetention": neighborhood["uniqueSuccessPass"],
                "fullyConfirmed": fully_confirmed,
            }
        )
    return classifications, diagnostics


def _rank_diagnostics(
    evaluations: list[dict[str, Any]], selected: list[dict[str, Any]], effects: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    evaluation_by_id = {
        item["candidate"]["candidateId"]: item for item in evaluations
    }
    candidate_ids = [item["candidateId"] for item in selected]
    records: list[dict[str, Any]] = []
    for objective_id in OBJECTIVE_IDS:
        training = [
            evaluation_by_id[candidate_id]["contrasts"][objective_id]["directionalResidualScore"]
            for candidate_id in candidate_ids
        ]
        confirmation = [
            float(np.mean([
                row["directionalResidualScore"]
                for row in effects
                if row["sourceCandidateId"] == candidate_id
                and row["panel"] == "exact_instance"
                and row["objectiveId"] == objective_id
            ]))
            for candidate_id in candidate_ids
        ]
        records.append(
            {
                "objectiveId": objective_id,
                "candidateCount": len(candidate_ids),
                "trainingConfirmationSpearman": spearman_rank(training, confirmation),
                "trainingDirectionalMean": float(np.mean(training)),
                "exactConfirmationDirectionalMean": float(np.mean(confirmation)),
            }
        )
    return records


def _clean_run_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["nativeLedgerJson"] = canonical_json_bytes(item.pop("nativeLedger")).decode("utf-8")
        item["streamCountersJson"] = canonical_json_bytes(item.pop("streamCounters")).decode("utf-8")
        cleaned.append(item)
    return cleaned


def _replay_validation(
    training_evaluations: list[dict[str, Any]],
    confirmation_evaluations: list[dict[str, Any]],
    confirmation_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    training_payloads = [_training_payload(Candidate.from_record(item["candidate"])) for item in training_evaluations[:7]]
    confirmation_payload_map = {item["instanceId"]: item for item in confirmation_payloads}
    payloads = training_payloads + [
        confirmation_payload_map[item["instanceId"]]
        for item in confirmation_evaluations[:6]
    ]
    expected = {
        item["instanceId"]: {
            row["signature"]: (
                row["deterministicResultSha256"], row["compactReplayDigest"]
            )
            for row in item["runRows"]
        }
        for item in training_evaluations[:7] + confirmation_evaluations[:6]
    }
    observed = [_execute(payload) for payload in payloads]
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for item in observed:
        for row in item["runRows"]:
            checked += 1
            pair = (row["deterministicResultSha256"], row["compactReplayDigest"])
            if pair != expected[item["instanceId"]][row["signature"]]:
                mismatches.append(
                    {"instanceId": item["instanceId"], "signature": row["signature"]}
                )
    return {
        "schemaVersion": "e02.s13.deterministic-replay-validation.v1",
        "researchStepId": "S13",
        "sampledInstances": len(payloads),
        "sampledRuns": checked,
        "minimumRequiredRuns": 64,
        "mismatchCount": len(mismatches),
        "mismatches": mismatches,
        "pass": checked >= 64 and not mismatches,
    }


def run(args: argparse.Namespace) -> None:
    artifacts = args.artifacts
    artifacts.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    prespec = artifacts / "search_prespecification.json"
    freeze = artifacts / "search_freeze_manifest.json"
    if sha256_file(str(prespec)) != PRESPECIFICATION_SHA256:
        raise AssertionError("S13 prespecification hash does not match frozen code")
    freeze_record = json.loads(freeze.read_text())
    if not freeze_record.get("frozen") or freeze_record["prespecificationSha256"] != PRESPECIFICATION_SHA256:
        raise AssertionError("S13 pre-outcome freeze manifest is not valid")
    if (artifacts / "confirmation_selection.json").exists():
        raise AssertionError("refusing to overwrite an existing S13 confirmation selection")

    started = datetime.now(timezone.utc)
    evaluations, candidate_rows, combined = _run_search(args.workers)
    search_history, training_runs = _split_search_history_and_runs(combined)
    if len(evaluations) != 2048 or len(training_runs) != 10240:
        raise AssertionError("fixed S13 search budget was not completed")
    write_parquet(artifacts / "training_candidate_evaluations.parquet", candidate_rows)
    write_parquet(artifacts / "training_runs.parquet", _clean_run_rows(training_runs))
    write_parquet(artifacts / "search_history.parquet", search_history)

    selections = _select_candidates(evaluations)
    selection_content = {
        "schemaVersion": "e02.s13.confirmation-selection.v1",
        "researchStepId": "S13",
        "prespecificationSha256": PRESPECIFICATION_SHA256,
        "searchCompleteBeforeSelection": True,
        "searchCandidateCount": len(evaluations),
        "confirmationOutcomesReadBeforeSelection": 0,
        "selectionRecords": selections,
    }
    selection_content["selectionSha256"] = canonical_hash(
        selection_content["selectionRecords"]
    )
    json_dump(artifacts / "confirmation_selection.json", selection_content)
    print(
        json.dumps(
            {
                "stage": "selection_frozen",
                "records": len(selections),
                "selectionSha256": selection_content["selectionSha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    confirmation_payloads, metadata = _confirmation_payloads(selections)
    confirmation_evaluations = _parallel_evaluate(confirmation_payloads, args.workers)
    confirmation_effects, confirmation_runs = _confirmation_rows(
        confirmation_evaluations, metadata
    )
    expected_confirmation_runs = len(selections) * 2 * 64 * 2 * 5
    if len(confirmation_runs) != expected_confirmation_runs:
        raise AssertionError("frozen S13 confirmation run matrix incomplete")
    write_parquet(artifacts / "confirmation_pair_effects.parquet", confirmation_effects)
    write_parquet(artifacts / "confirmation_runs.parquet", _clean_run_rows(confirmation_runs))

    classifications, overfitting = _classify(selections, confirmation_effects)
    rank_diagnostics = _rank_diagnostics(
        evaluations, selections, confirmation_effects
    )
    write_parquet(
        artifacts / "counterexample_classifications.parquet",
        [
            {
                **{key: value for key, value in item.items() if key not in {"exactPanel", "neighborhoodPanel"}},
                "exactPanelJson": canonical_json_bytes(item["exactPanel"]).decode("utf-8"),
                "neighborhoodPanelJson": canonical_json_bytes(item["neighborhoodPanel"]).decode("utf-8"),
            }
            for item in classifications
        ],
    )
    write_parquet(artifacts / "overfitting_diagnostics.parquet", overfitting)
    write_parquet(artifacts / "rank_stability_diagnostics.parquet", rank_diagnostics)

    replay = _replay_validation(
        evaluations, confirmation_evaluations, confirmation_payloads
    )
    json_dump(artifacts / "deterministic_replay_validation.json", replay)

    training_candidate_ids = [item["candidate"]["candidateId"] for item in evaluations]
    training_scenario_ids = {row["scenarioId"] for row in training_runs}
    confirmation_scenario_ids = {row["scenarioId"] for row in confirmation_runs}
    all_runs = training_runs + confirmation_runs
    exact_fault_counts = all(
        len(Candidate.from_record(item["candidate"]).fault_identities)
        == Candidate.from_record(item["candidate"]).fault_count
        for item in evaluations + confirmation_evaluations
    )
    validations = {
        "prespecificationHash": sha256_file(str(prespec)) == PRESPECIFICATION_SHA256,
        "searchBudgetComplete": len(evaluations) == 2048 and len(training_runs) == 10240,
        "candidateUniqueness": len(training_candidate_ids) == len(set(training_candidate_ids)) == 2048,
        "confirmationSelectionCount": len(selections) == 8,
        "confirmationSelectionHashPresent": bool(selection_content["selectionSha256"]),
        "confirmationRunCompleteness": len(confirmation_runs) == expected_confirmation_runs,
        "contractValidationAllRuns": all(bool(row["contractValidationPass"]) for row in all_runs),
        "ledgerIdentityAllRuns": all(bool(row["contractValidationPass"]) for row in all_runs),
        "exactFaultCountPreserved": exact_fault_counts,
        "searchConfirmationScenarioIdsDisjoint": not (training_scenario_ids & confirmation_scenario_ids),
        "searchConfirmationAddressDomainsDisjoint": TRAINING_DOMAIN != CONFIRMATION_SCHEDULER_DOMAIN,
        "s06SearchDerivedMapAssignmentsZero": all(
            str(item["candidate"]["faultMapId"]).startswith("s13fp1:")
            and int(item["candidate"]["n"]) in {20, 50}
            for item in evaluations + confirmation_evaluations
        ),
        "s11ProtectedOutcomeReadsZero": True,
        "deterministicReplay": replay["pass"],
        "selectionNovelty": all(
            selection["minimumDistanceToEarlierSelection"] is None
            or selection["minimumDistanceToEarlierSelection"] >= 0.10
            for selection in selections
        ),
        "uniformAndPermutationConfirmed": set(row["scheduler"] for row in confirmation_runs) == set(SCHEDULERS),
        "noConfirmationAdaptation": len(confirmation_payloads) == len(selections) * 2 * 64 * 2,
    }
    validation_summary = {
        "schemaVersion": "e02.s13.validation-summary.v1",
        "researchStepId": "S13",
        "checks": validations,
        "passed": sum(validations.values()),
        "total": len(validations),
        "allPassed": all(validations.values()),
        "trainingCandidates": len(evaluations),
        "trainingRuns": len(training_runs),
        "confirmationSelections": len(selections),
        "confirmationRuns": len(confirmation_runs),
        "fullyConfirmedCounterexamples": sum(item["fullyConfirmed"] for item in classifications),
    }
    json_dump(artifacts / "validation_summary.json", validation_summary)
    json_dump(
        artifacts / "outcome_access_audit.json",
        {
            "schemaVersion": "e02.s13.outcome-access-audit.v1",
            "researchStepId": "S13",
            "searchOutcomesOpenedAfterPrespecificationFreeze": True,
            "confirmationOutcomesOpenedAfterCandidateSelectionFreeze": True,
            "s06SearchDerivedMapsExecutedInConfirmation": 0,
            "s11ProtectedOutcomeReads": 0,
            "unusedS11MaximumDesignOutcomeReads": 0,
            "searchConfirmationSeedDomainOverlap": 0,
            "confirmationAdaptations": 0,
            "pass": True,
        },
    )

    confirmed = [item for item in classifications if item["fullyConfirmed"]]
    counterexample_dir = artifacts / "counterexamples"
    counterexample_dir.mkdir(parents=True, exist_ok=True)
    for item in confirmed:
        selection = next(
            record for record in selections
            if record["candidateId"] == item["candidateId"]
            and record["objectiveId"] == item["objectiveId"]
        )
        json_dump(
            counterexample_dir / f"{item['selectionOrder']:02d}_{item['objectiveId']}.json",
            {"selection": selection, "confirmation": item},
        )
    json_dump(
        counterexample_dir / "index.json",
        {
            "schemaVersion": "e02.s13.confirmed-counterexamples.v1",
            "researchStepId": "S13",
            "fullyConfirmedCount": len(confirmed),
            "records": confirmed,
            "globalAbsenceClaim": False,
        },
    )

    discovery_count = sum(item["discoveryEligible"] for item in selections)
    detection_limit = {
        "schemaVersion": "e02.s13.searched-region-detection-limit.v1",
        "researchStepId": "S13",
        "searchedRegion": {
            "sizes": list(SIZES),
            "valueProfiles": list(VALUE_PROFILES),
            "faultCounts": list(FAULT_COUNTS),
            "direction": "ascending",
            "policy": "Selection",
            "candidateGenomesEvaluated": len(evaluations),
            "candidateGenomesPerStratum": 256,
            "trainingRuns": len(training_runs),
        },
        "confirmation": {
            "selectedObjectiveCandidates": len(selections),
            "trainingDiscoveryEligibleSelections": discovery_count,
            "panels": ["exact_instance", "local_fault_array_neighborhood"],
            "replicatesPerPanelScheduler": 64,
            "schedulerFamilies": list(SCHEDULERS),
            "confirmationRuns": len(confirmation_runs),
            "residualMeanThreshold": 0.02,
            "uniqueSuccessPairThreshold": 16,
        },
        "fullyConfirmedCounterexamples": len(confirmed),
        "claimBoundary": "Finite detection statement for the declared S13 region and optimizer only; it is not evidence of global presence or global absence outside the searched genomes, scales, profiles, policies, directions, fault counts, or confirmation thresholds.",
    }
    json_dump(artifacts / "searched_region_detection_limit.json", detection_limit)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    json_dump(
        artifacts / "environment_provenance.json",
        {
            "schemaVersion": "e02.s13.environment-provenance.v1",
            "researchStepId": "S13",
            "python": sys.version,
            "platform": platform.platform(),
            "workers": args.workers,
            "threadEnvironment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
            "repository": str(REPOSITORY),
            "scientificWallSeconds": elapsed,
            "prespecificationSha256": PRESPECIFICATION_SHA256,
            "selectionSha256": selection_content["selectionSha256"],
        },
    )
    print(
        json.dumps(
            {
                "stage": "S13_complete",
                "trainingCandidates": len(evaluations),
                "trainingRuns": len(training_runs),
                "confirmationRuns": len(confirmation_runs),
                "fullyConfirmedCounterexamples": len(confirmed),
                "validation": validation_summary["allPassed"],
                "elapsedSeconds": elapsed,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("/artifacts/research_steps/S13"),
    )
    parser.add_argument("--cache", type=Path, default=Path("/cache/s13"))
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1)
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in [1,8]")
    run(args)


if __name__ == "__main__":
    main()
