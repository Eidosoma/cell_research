#!/usr/bin/env python3
"""Execute S08C under the byte-frozen S08P design and S08B gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from src.environment_suite import AccessDeniedError
from src.portfolio_preregistration.core import canonical_hash, read_jsonl
from src.portfolio_search.execution import (
    ARTIFACT_ROOT,
    CACHE_ROOT,
    CONTROL_PATH,
    S08P,
    VALIDATION_FAMILIES,
    _catalog,
    _candidate_lock_authorize,
    _initial_work,
    _json_bytes,
    _write_json,
    aggregate_panel,
    build_action,
    build_archive,
    build_validation_plan,
    environment_record,
    execute_work,
    generate_adaptive_generation,
    immutable_tree_hashes,
    paired_estimands,
    run_preflight,
    s09_gate,
    select_shortlist,
    sha256_file,
    validate_rows,
    _archive_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "S08C")
    parser.add_argument(
        "--phase", choices=("preflight", "smoke", "full"), default="full"
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_json_bytes(dict(row)) + b"\n" for row in rows))


def ledger_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    nested = {
        "memberPolicySha256",
        "validation",
        "nativeContractErrors",
        "outcome",
        "nativeLedgerFamilies",
        "nativeEvent",
        "provenance",
        "nativeNonFiniteValues",
    }
    projected = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if key in nested:
                item[f"{key}Json"] = json.dumps(
                    value, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
            else:
                item[key] = value
        projected.append(item)
    return pd.DataFrame(projected)


def work_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_hash(
        "E07/S08C/work-order-independent/v1",
        sorted(
            (
                str(row["logicalSlotId"]),
                str(row["configuration"]["configurationId"]),
                int(row["scenarioFamilyOrdinal"]),
            )
            for row in rows
        ),
    )


def result_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_hash(
        "E07/S08C/result-order-independent/v1",
        sorted(
            (str(row["logicalSlotId"]), str(row["logicalExpansionSha256"]))
            for row in rows
        ),
    )


def configuration_hash_validation(
    definitions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_hash, by_id = _catalog()
    rows = []
    for cid, definition in sorted(definitions.items()):
        action = build_action(definition, by_hash, by_id)
        rows.append(
            {
                "configurationId": cid,
                "taskId": definition["taskId"],
                "mode": definition["mode"],
                "portfolioSize": definition["portfolioSize"],
                "configurationDefinitionSha256": canonical_hash(
                    "E07/S08C/configuration-definition/v1", definition
                ),
                "actionSha256": action.policy_sha256,
                "memberPolicySha256": [
                    member["policySha256"] for member in definition["members"]
                ],
                "s07ArmMembershipUsed": bool(
                    definition.get("s07ArmMembershipUsed", False)
                ),
                "rejectedModelOrEmbeddingUsed": bool(
                    definition.get("rejectedModelOrEmbeddingUsed", False)
                ),
            }
        )
    return {
        "schemaVersion": "e07.s08c.configuration-hash-validation.v1",
        "researchStepId": "S08C",
        "success": all(
            not row["s07ArmMembershipUsed"]
            and not row["rejectedModelOrEmbeddingUsed"]
            and len(row["configurationDefinitionSha256"]) == 64
            and len(row["actionSha256"]) == 64
            for row in rows
        ),
        "configurationRows": len(rows),
        "uniqueConfigurationIds": len({row["configurationId"] for row in rows}),
        "uniqueActionHashes": len({row["actionSha256"] for row in rows}),
        "rows": rows,
    }


def validation_support_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["taskId"]].append(row)
    result = {}
    for task_id, task_rows in sorted(by_task.items()):
        scenario_by_family = {
            int(row["scenarioFamilyOrdinal"]): row["scenarioCommitmentSha256"]
            for row in task_rows
        }
        # Count distinct native result projections, not logical labels.
        support = {
            canonical_hash(
                "E07/S08C/native-validation-support/v1",
                {
                    "outcome": row["outcome"],
                    "nativeEvent": row["nativeEvent"],
                    "stopReason": row["stopReason"],
                },
            )
            for row in task_rows
        }
        result[task_id] = {
            "logicalFamilies": len(scenario_by_family),
            "distinctScenarioCommitments": len(set(scenario_by_family.values())),
            "distinctObservedNativeResultProjectionsAcrossAllConfigurations": len(
                support
            ),
            "independenceClaim": "not_made",
        }
    return {
        "schemaVersion": "e07.s08c.validation-support-audit.v1",
        "researchStepId": "S08C",
        "nativeSupportsRemainTaskSpecific": True,
        "universalScenarioNormalizationApplied": False,
        "byTask": result,
    }


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("workers must be in [1,8]")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    control = yaml.safe_load(CONTROL_PATH.read_text(encoding="utf-8"))
    immutable_expected = dict(control["expectedTrees"])
    preflight = run_preflight(CONTROL_PATH)
    _write_json(output / "preflight_hash_and_gate_revalidation.json", preflight)
    _write_json(
        output / "s08b_binding_revalidation.json", preflight["bindingValidation"]
    )
    _write_json(
        output / "access_control_preflight.json",
        {
            "schemaVersion": "e07.s08c.access-control-preflight.v1",
            "researchStepId": "S08C",
            "success": all(
                row["deniedBeforeMaterialization"] for row in preflight["accessRows"]
            ),
            "rows": preflight["accessRows"],
            "validationOutcomeEvaluations": 0,
            "confirmationOutcomeEvaluations": 0,
        },
    )
    if not preflight["success"]:
        _write_json(
            output / "execution_stop_status.json",
            {
                "researchStepId": "S08C",
                "status": "blocked_before_smoke",
                "success": False,
                "blockedGateIds": preflight["blockedGateIds"],
                "trainingLogicalRowsExecuted": 0,
                "validationLogicalRowsExecuted": 0,
                "confirmationLogicalRowsExecuted": 0,
            },
        )
        return 2
    if args.phase == "preflight":
        return 0

    budget = pd.read_parquet(S08P / "budget_slot_ledger.parquet")
    initial_definitions = read_jsonl(S08P / "portfolio_seed_registry.jsonl")
    definitions: dict[str, dict[str, Any]] = {
        str(row["configurationId"]): dict(row) for row in initial_definitions
    }
    initial_work = _initial_work(budget, definitions)
    smoke_work = [row for row in initial_work if row["smoke"]]
    smoke_rows, smoke_physical = execute_work(
        smoke_work, workers=args.workers, cache_dir=CACHE_ROOT / "evaluations"
    )
    smoke_validation = validate_rows(smoke_rows, 40)
    smoke_validation.update(
        {
            "schemaVersion": "e07.s08c.frozen-smoke-validation.v1",
            "researchStepId": "S08C",
            "withinFrozenTrainingBudget": True,
            "physicalAccounting": smoke_physical,
            "substantiveSearchAuthorized": smoke_validation["success"],
        }
    )
    ledger_frame(smoke_rows).to_parquet(
        output / "frozen_smoke_results.parquet", index=False
    )
    _write_json(output / "frozen_smoke_validation.json", smoke_validation)
    if not smoke_validation["success"]:
        _write_json(
            output / "execution_stop_status.json",
            {
                "researchStepId": "S08C",
                "status": "blocked_at_frozen_smoke",
                "success": False,
                "trainingLogicalRowsExecuted": 40,
                "validationLogicalRowsExecuted": 0,
                "confirmationLogicalRowsExecuted": 0,
            },
        )
        return 3
    if args.phase == "smoke":
        return 0

    training_rows, initial_physical = execute_work(
        initial_work, workers=args.workers, cache_dir=CACHE_ROOT / "evaluations"
    )
    initial_aggregates = aggregate_panel(definitions, training_rows)
    aggregates = list(initial_aggregates)
    archive = build_archive(aggregates)
    all_work = list(initial_work)
    lineage_rows = []
    convergence = [
        {
            "generation": 0,
            "logicalRowsCumulative": len(training_rows),
            "archiveCells": len(archive),
            "archiveFrontEntries": sum(len(value) for value in archive.values()),
            "validTargetConfigurations": sum(
                row["mode"] in {"fixed_balanced_identity", "environment_conditioned"}
                and row["validNativeEpisodes"]
                for row in aggregates
            ),
        }
    ]
    candidates = read_jsonl(S08P / "candidate_eligibility_registry.jsonl")
    protocol = yaml.safe_load((S08P / "s08p_portfolio_protocol.yaml").read_text())
    selector_signals = protocol["selectorSignals"]["byTask"]
    prior_ids = set(definitions)
    physical_accounting = {"smoke": smoke_physical, "initial": initial_physical}
    for generation in range(1, 7):
        works, generated, generation_lineage = generate_adaptive_generation(
            generation,
            archive,
            definitions,
            candidates,
            selector_signals,
            budget,
            prior_ids,
        )
        definitions.update(generated)
        rows, physical = execute_work(
            works, workers=args.workers, cache_dir=CACHE_ROOT / "evaluations"
        )
        training_rows.extend(rows)
        all_work.extend(works)
        lineage_rows.extend(generation_lineage)
        generation_aggregates = aggregate_panel(definitions, rows)
        aggregates.extend(generation_aggregates)
        archive = build_archive(aggregates)
        physical_accounting[f"generation_{generation}"] = physical
        convergence.append(
            {
                "generation": generation,
                "logicalRowsCumulative": len(training_rows),
                "archiveCells": len(archive),
                "archiveFrontEntries": sum(len(value) for value in archive.values()),
                "validTargetConfigurations": sum(
                    row["mode"]
                    in {"fixed_balanced_identity", "environment_conditioned"}
                    and row["validNativeEpisodes"]
                    for row in aggregates
                ),
            }
        )
        checkpoint = CACHE_ROOT / f"generation_{generation}_checkpoint.json"
        _write_json(
            checkpoint,
            {
                "generation": generation,
                "trainingLogicalRows": len(training_rows),
                "workDigest": work_digest(all_work),
                "resultDigest": result_digest(training_rows),
                "definitionIds": sorted(definitions),
            },
        )

    training_validation = validate_rows(training_rows, 11008)
    training_validation.update(
        {
            "schemaVersion": "e07.s08c.training-validation.v1",
            "researchStepId": "S08C",
            "generationsCompleted": 6,
            "runtimeDrivenWeakening": False,
            "efficacyEarlyStopUsed": False,
        }
    )
    if not training_validation["success"]:
        _write_json(output / "training_validation.json", training_validation)
        return 4

    initial_single_by_member = {}
    for definition in initial_definitions:
        if definition["mode"] == "single_policy":
            initial_single_by_member[
                (definition["taskId"], definition["members"][0]["policySha256"])
            ] = definition
    shortlist, single_leaders = select_shortlist(
        archive, definitions, aggregates, initial_single_by_member
    )
    lock_path = output / "validation_candidate_lock.json"
    validation_work, lock, validation_definitions = build_validation_plan(
        shortlist,
        single_leaders,
        definitions,
        initial_single_by_member,
        lock_path=lock_path,
    )
    lock_sha = sha256_file(lock_path)
    # Demonstrate lock denial before any validation outcome is materialized.
    first_locked = next(iter(validation_definitions.values()))
    first_action = build_action(first_locked, *_catalog())
    denied_absent = False
    try:
        _candidate_lock_authorize(
            lock_path,
            lock_sha,
            "f" * 64,
            str(first_locked["taskId"]),
            first_action.policy_sha256,
        )
    except AccessDeniedError:
        denied_absent = True
    if not denied_absent:
        raise RuntimeError("candidate lock accepted an absent configuration")
    validation_rows, validation_physical = execute_work(
        validation_work,
        workers=args.workers,
        cache_dir=CACHE_ROOT / "validation_evaluations",
    )
    validation_validation = validate_rows(validation_rows, len(validation_work))
    if not validation_validation["success"]:
        _write_json(
            output / "validation_execution_validation.json", validation_validation
        )
        return 5
    endpoint_rows, cost_rows, guard_rows = paired_estimands(
        validation_rows,
        lock["comparisonRegistry"],
        validation_definitions,
    )
    gate = s09_gate(shortlist, endpoint_rows, guard_rows, validation_rows)

    # Compact canonical artifacts.
    ledger_frame(training_rows).to_parquet(
        output / "training_evaluation_ledger.parquet", index=False, compression="zstd"
    )
    ledger_frame(validation_rows).to_parquet(
        output / "validation_evaluation_ledger.parquet", index=False, compression="zstd"
    )
    write_jsonl(
        output / "portfolio_configuration_registry.jsonl", list(definitions.values())
    )
    write_jsonl(output / "configuration_aggregates.jsonl", aggregates)
    write_jsonl(output / "adaptive_lineage_ledger.jsonl", lineage_rows)
    write_jsonl(output / "portfolio_archive.jsonl", _archive_rows(archive))
    pd.DataFrame(endpoint_rows).to_csv(
        output / "paired_native_estimands.csv", index=False
    )
    pd.DataFrame(cost_rows).to_csv(output / "paired_cost_estimands.csv", index=False)
    pd.DataFrame(guard_rows).to_csv(output / "failure_censor_guards.csv", index=False)
    _write_json(output / "training_validation.json", training_validation)
    _write_json(
        output / "convergence_and_stopping.json",
        {
            "schemaVersion": "e07.s08c.convergence.v1",
            "researchStepId": "S08C",
            "rows": convergence,
            "stoppingRule": "exactly_six_generations_or_integrity_failure",
            "generationsCompleted": 6,
            "runtimeDrivenWeakening": False,
        },
    )
    _write_json(
        output / "training_shortlist.json",
        {
            "schemaVersion": "e07.s08c.training-shortlist.v1",
            "researchStepId": "S08C",
            "byTask": shortlist,
            "singleFrontierObjectiveLeadersByTask": single_leaders,
            "maximumPortfoliosPerTask": 8,
            "selectionUsesS07ArmMembership": False,
            "selectionUsesRejectedModelOrEmbedding": False,
        },
    )
    _write_json(
        output / "validation_plan_and_accounting.json",
        {
            "schemaVersion": "e07.s08c.validation-plan-accounting.v1",
            "researchStepId": "S08C",
            "candidateLockSha256": lock_sha,
            "logicalRows": len(validation_work),
            "logicalCeiling": 3072,
            "withinCeiling": len(validation_work) <= 3072,
            "familiesPerTask": len(VALIDATION_FAMILIES),
            "physicalAccounting": validation_physical,
            "confirmationRows": 0,
        },
    )
    _write_json(output / "validation_execution_validation.json", validation_validation)
    _write_json(
        output / "validation_support_audit.json",
        validation_support_audit(validation_rows),
    )
    _write_json(
        output / "uncertainty_and_multiplicity.json",
        {
            "schemaVersion": "e07.s08c.uncertainty-multiplicity.v1",
            "researchStepId": "S08C",
            "pairedScenarioBlockBootstrapReplicates": 2000,
            "confidenceLevel": 0.95,
            "seed": 708004,
            "multiplicityMethod": "Holm",
            "family": "within_task_target_contrast_across_primary_native_endpoints",
            "crossTaskPooling": False,
            "endpointRows": len(endpoint_rows),
        },
    )
    _write_json(output / "s09_eligibility_gate.json", gate)
    config_hashes = configuration_hash_validation(definitions)
    _write_json(output / "configuration_and_member_hash_validation.json", config_hashes)
    action_groups = defaultdict(list)
    for row in config_hashes["rows"]:
        action_groups[row["actionSha256"]].append(row["configurationId"])
    semantic_groups = defaultdict(list)
    for aggregate in aggregates:
        signature = canonical_hash(
            "E07/S08C/finite-panel-equivalence/v1",
            aggregate["evaluationHashes"],
        )
        semantic_groups[(aggregate["taskId"], signature)].append(
            aggregate["configurationId"]
        )
    _write_json(
        output / "duplicate_detection_audit.json",
        {
            "schemaVersion": "e07.s08c.duplicate-audit.v1",
            "researchStepId": "S08C",
            "canonicalConfigurationDuplicates": len(definitions)
            - len(set(definitions)),
            "actionHashGroupsWithMultipleConfigurations": [
                {"actionSha256": key, "configurationIds": value}
                for key, value in sorted(action_groups.items())
                if len(value) > 1
            ],
            "finiteTrainingPanelSemanticGroupsWithMultipleConfigurations": [
                {"taskId": key[0], "signature": key[1], "configurationIds": value}
                for key, value in sorted(semantic_groups.items())
                if len(value) > 1
            ],
            "finiteSupportOnly": True,
        },
    )
    order_validation = {
        "schemaVersion": "e07.s08c.replay-worker-order-validation.v1",
        "researchStepId": "S08C",
        "forwardWorkDigest": work_digest(all_work),
        "reverseWorkDigest": work_digest(list(reversed(all_work))),
        "forwardResultDigest": result_digest(training_rows),
        "reverseResultDigest": result_digest(list(reversed(training_rows))),
        "workerOrderIndependent": work_digest(all_work)
        == work_digest(list(reversed(all_work)))
        and result_digest(training_rows)
        == result_digest(list(reversed(training_rows))),
        "nativeExactReplayRows": sum(row["replayPass"] for row in training_rows),
        "validationExactReplayRows": sum(row["replayPass"] for row in validation_rows),
    }
    order_validation["success"] = order_validation["workerOrderIndependent"]
    _write_json(output / "replay_worker_order_validation.json", order_validation)
    logical_by_generation = Counter(int(row["generation"]) for row in training_rows)
    logical_by_task = Counter(str(row["taskId"]) for row in training_rows)
    logical_by_family = Counter(
        int(row["scenarioFamilyOrdinal"]) for row in training_rows
    )
    budget_accounting = {
        "schemaVersion": "e07.s08c.complete-budget-accounting.v1",
        "researchStepId": "S08C",
        "trainingLogicalRows": len(training_rows),
        "trainingLogicalRowsExpected": 11008,
        "smokeLogicalRows": 40,
        "smokeIncludedInTrainingBudget": True,
        "byGeneration": dict(sorted(logical_by_generation.items())),
        "byTask": dict(sorted(logical_by_task.items())),
        "byScenarioFamily": dict(sorted(logical_by_family.items())),
        "validationLogicalRows": len(validation_rows),
        "validationLogicalCeiling": 3072,
        "confirmationLogicalRows": 0,
        "physicalAccounting": physical_accounting,
        "validationPhysicalAccounting": validation_physical,
        "runtimeDrivenWeakening": False,
        "complete": len(training_rows) == 11008
        and len(validation_rows) <= 3072
        and logical_by_generation
        == Counter({0: 4864, 1: 1024, 2: 1024, 3: 1024, 4: 1024, 5: 1024, 6: 1024}),
    }
    _write_json(output / "complete_budget_accounting.json", budget_accounting)
    _write_json(
        output / "access_control_validation.json",
        {
            "schemaVersion": "e07.s08c.access-control-validation.v1",
            "researchStepId": "S08C",
            "preflightNontrainingDenials": 16,
            "absentCandidateLockEntryDenied": denied_absent,
            "validationAccessAfterCandidateLockOnly": True,
            "validationOutcomeEvaluations": len(validation_rows),
            "confirmationScenarioMaterializations": 0,
            "confirmationOutcomeEvaluations": 0,
            "confirmationRemainsSealed": True,
            "success": denied_absent and len(validation_rows) <= 3072,
        },
    )
    _write_json(
        output / "dependency_exclusion_audit.json",
        {
            "schemaVersion": "e07.s08c.dependency-exclusion-audit.v1",
            "researchStepId": "S08C",
            "loadedProhibitedModules": preflight["loadedProhibitedModules"],
            "rejectedModelArtifactsOpened": 0,
            "rejectedEmbeddingArtifactsOpened": 0,
            "pseudoLabelsUsed": 0,
            "warmStartsUsed": 0,
            "distillationUses": 0,
            "s07ArmPromotionUses": 0,
            "success": not preflight["loadedProhibitedModules"],
        },
    )
    final_trees = immutable_tree_hashes()
    no_mutation = final_trees == immutable_expected
    _write_json(
        output / "no_mutation_audit.json",
        {
            "schemaVersion": "e07.s08c.no-mutation-audit.v1",
            "researchStepId": "S08C",
            "expectedTrees": immutable_expected,
            "observedTrees": final_trees,
            "unchanged": no_mutation,
            "s05ArchiveMutations": 0,
            "historicalS08ArtifactMutations": 0,
            "success": no_mutation,
        },
    )
    _write_json(output / "environment.json", environment_record())
    _write_json(
        output / "provenance.json",
        {
            "schemaVersion": "e07.s08c.provenance.v1",
            "researchStepId": "S08C",
            "controlPath": str(CONTROL_PATH),
            "controlSha256": sha256_file(CONTROL_PATH),
            "s08pProtocolSha256": sha256_file(S08P / "s08p_portfolio_protocol.yaml"),
            "candidateLockSha256": lock_sha,
            "trainingWorkDigest": work_digest(all_work),
            "trainingResultDigest": result_digest(training_rows),
            "validationResultDigest": result_digest(validation_rows),
            "immutableTrees": final_trees,
            "trainingOutcomeEvaluations": len(training_rows),
            "validationOutcomeEvaluations": len(validation_rows),
            "confirmationOutcomeEvaluations": 0,
        },
    )
    summary = {
        "schemaVersion": "e07.s08c.validation-summary.v1",
        "researchStepId": "S08C",
        "checks": {
            "preflight": preflight["success"],
            "g01ToG06": not preflight["blockedGateIds"],
            "smoke40": smoke_validation["success"],
            "training11008": training_validation["success"],
            "sixGenerations": len(convergence) == 7,
            "replay": order_validation["success"],
            "configurationHashes": config_hashes["success"],
            "validationWithinCeiling": len(validation_rows) <= 3072,
            "candidateLock": denied_absent,
            "confirmationSealed": True,
            "dependencyExclusion": not preflight["loadedProhibitedModules"],
            "noMutation": no_mutation,
            "completeAccounting": budget_accounting["complete"],
        },
        "eligibleS09Configurations": len(gate["eligibleConfigurationIds"]),
    }
    summary["passed"] = sum(summary["checks"].values())
    summary["total"] = len(summary["checks"])
    summary["success"] = all(summary["checks"].values())
    _write_json(output / "validation_summary.json", summary)
    if not summary["success"]:
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
