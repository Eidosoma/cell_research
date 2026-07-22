"""S08C execution of the byte-frozen S08P portfolio design.

The module consumes only S02/S04/S05/S08P/S08A/S08B commitments.  It has no
import path to either rejected surrogate package.  Training search is strictly
task-local; validation is reachable only after a byte-hashed candidate lock;
confirmation has no materializer or execution path.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.environment_suite import (
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
    EvaluationAction,
    Split,
    SuiteValidationError,
    portfolio_action,
)
from src.environment_suite.access import AccessBroker
from src.environment_suite.contracts import ScenarioRecord, canonical_sha256
from src.environment_suite.runners import RUNNERS, locked_dsl_validation_scope
from src.environment_suite.suite import EnvironmentSuite
from src.policy_dsl import compile_policy
from src.portfolio_preregistration.core import (
    CHIMERA_TASK,
    TASK_IDS,
    _member_is_compatible,
    _portfolio_costs,
    candidate_commitment,
    canonical_hash,
    checked_protocol,
    plan_digest,
    read_jsonl,
    validate_portfolio_registry,
    verify_frozen_inputs,
)
from src.portfolio_search.preflight import (
    _manifest_validation,
    sha256_file,
    tree_digest,
    validate_executable_bindings,
)
from src.quality_diversity.core import (
    BASE_SCENARIOS,
    OBJECTIVES,
    _aggregate_descriptors,
    _get_path,
    _json_safe,
    _numeric_leaves,
    dominates,
)


REPOSITORY = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path("/artifacts/research_steps")
S05 = ARTIFACT_ROOT / "S05"
S08P = ARTIFACT_ROOT / "S08P"
S08A = ARTIFACT_ROOT / "S08A"
S08 = ARTIFACT_ROOT / "S08"
S08B = ARTIFACT_ROOT / "S08B"
TASK_REGISTRY = REPOSITORY / "configs/environment_suite/task_registry.yaml"
SPLIT_MANIFEST = REPOSITORY / "configs/environment_suite/split_manifest.json"
CONTROL_PATH = REPOSITORY / "configs/portfolio/s08c_execution_continuation.yaml"
CACHE_ROOT = Path("/cache/e07-s08c")
TRAIN_FAMILIES = (300, 301, 302, 303)
VALIDATION_FAMILIES = tuple(range(400, 408))
MODES = (
    "single_policy",
    "fixed_balanced_identity",
    "random_static_identity",
    "random_dynamic_opportunity",
    "environment_conditioned",
)
TARGET_MODES = {"fixed_balanced_identity", "environment_conditioned"}
MUTATION_OPERATORS = (
    "replace_one_member_same_native_carrier",
    "add_one_member_same_native_carrier",
    "drop_one_member_preserving_task_carrier_contract",
    "rotate_fixed_identity_assignment",
    "toggle_permitted_selector_signal",
    "flip_selector_branch_assignment",
)
EXPECTED_E05_FAILURE_FLAG = "developmentBudgetRespected"


class FailAtomicBatchError(RuntimeError):
    """A worker batch failed before any new result row was published."""

    def __init__(self, accounting: Mapping[str, Any]):
        self.accounting = dict(accounting)
        failed = int(self.accounting.get("failedPhysicalRows", 0))
        super().__init__(f"fail-atomic batch rejected {failed} failed physical row(s)")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _exception_record(exc: BaseException) -> dict[str, str]:
    error_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
    message = str(exc)
    return {
        "errorType": error_type,
        "errorMessage": message,
        "errorSha256": canonical_sha256(
            "E07/S08D/batch-error/v1",
            {"errorType": error_type, "errorMessage": message},
        ),
    }


def execute_fail_atomic_batch(
    keys: Sequence[str],
    items: Sequence[Mapping[str, Any]],
    *,
    worker: Callable[[Mapping[str, Any]], dict[str, Any]],
    workers: int,
    executor_factory: Callable[..., Executor] = ProcessPoolExecutor,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Run one physical batch and either return every row or publish none.

    The position-indexed ledger is terminal and exact: every submitted row is
    classified as succeeded, failed, or cancelled after all futures settle.
    Successful rows remain in memory until the complete batch has passed.
    """

    if len(keys) != len(items) or len(set(keys)) != len(keys):
        raise ValueError("fail-atomic batch requires one distinct key per item")
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1,8]")
    started = time.perf_counter()
    statuses = [
        {"position": position, "physicalKey": key, "status": "submitted"}
        for position, key in enumerate(keys)
    ]
    results: dict[str, dict[str, Any]] = {}
    failure_seen = False
    with executor_factory(max_workers=workers) as executor:
        futures = {
            executor.submit(worker, item): position
            for position, item in enumerate(items)
        }
        for future in as_completed(futures):
            position = futures[future]
            if future.cancelled():
                statuses[position]["status"] = "cancelled_before_execution"
                continue
            try:
                result = future.result()
            except BaseException as exc:  # worker failures must settle the batch
                statuses[position].update(
                    {"status": "failed", **_exception_record(exc)}
                )
                if not failure_seen:
                    failure_seen = True
                    for other in futures:
                        if other is not future and not other.done():
                            other.cancel()
            else:
                statuses[position]["status"] = "succeeded"
                results[keys[position]] = result
    succeeded = sum(row["status"] == "succeeded" for row in statuses)
    failed = sum(row["status"] == "failed" for row in statuses)
    cancelled = sum(row["status"] == "cancelled_before_execution" for row in statuses)
    if succeeded + failed + cancelled != len(items):
        raise RuntimeError("fail-atomic batch accounting did not reach terminal state")
    success = failed == 0 and cancelled == 0 and succeeded == len(items)
    accounting = {
        "schemaVersion": "e07.s08d.fail-atomic-batch-accounting.v1",
        "researchStepId": "S08D",
        "success": success,
        "physicalRowsSubmitted": len(items),
        "succeededPhysicalRows": succeeded,
        "failedPhysicalRows": failed,
        "cancelledPhysicalRows": cancelled,
        "attemptedPhysicalRows": succeeded + failed,
        "newResultRowsPublished": len(items) if success else 0,
        "newCacheRowsPublished": len(items) if success else 0,
        "publicationAuthorized": success,
        "positions": statuses,
        "batchKeyCommitmentSha256": canonical_sha256(
            "E07/S08D/fail-atomic-batch-keys/v1", sorted(keys)
        ),
        "resultCommitmentSha256": (
            canonical_sha256(
                "E07/S08D/fail-atomic-batch-results/v1",
                {key: results[key] for key in sorted(results)},
            )
            if success
            else None
        ),
        "wallSeconds": time.perf_counter() - started,
    }
    if not success:
        raise FailAtomicBatchError(accounting)
    return results, accounting


def _counter_u64(*parts: object, stream: str) -> int:
    payload = "\x1f".join(str(item) for item in parts).encode("utf-8")
    digest = hashlib.sha256(
        b"E07/S08/portfolio-search/v1\x00" + stream.encode("ascii") + b"\x00" + payload
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _catalog() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = read_jsonl(S05 / "policy_catalog.jsonl")
    return (
        {str(row["policySha256"]): dict(row["document"]) for row in rows},
        {str(row["policyId"]): dict(row["document"]) for row in rows},
    )


def _single_action(
    task_id: str,
    configuration: Mapping[str, Any],
    by_hash: Mapping[str, Mapping[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]],
) -> EvaluationAction:
    from src.portfolio_search.preflight import _single_action as historical_single

    return historical_single(task_id, configuration, by_hash, by_id)


def build_action(
    configuration: Mapping[str, Any],
    by_hash: Mapping[str, Mapping[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]],
) -> EvaluationAction:
    task_id = str(configuration["taskId"])
    if configuration["mode"] == "single_policy":
        return _single_action(task_id, configuration, by_hash, by_id)
    documents = [
        by_hash[str(item["policySha256"])] for item in configuration["members"]
    ]
    return portfolio_action(documents, configuration)


def _base_records() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    train = {task: suite.records[BASE_SCENARIOS[task]] for task in TASK_IDS}
    validation = {
        record.task_id: record
        for record in suite.records.values()
        if record.split is Split.VALIDATION
    }
    return suite.tasks, train, validation


def derive_record(base: ScenarioRecord, family: int, *, split: Split) -> ScenarioRecord:
    """Derive an outcome-blind native record without crossing split support."""

    if base.split is not split or base.protected:
        raise SuiteValidationError(
            "scenario derivation requires matching unprotected base"
        )
    if split is Split.TRAIN and family not in TRAIN_FAMILIES:
        raise SuiteValidationError("training family outside frozen S08P ordinals")
    if split is Split.VALIDATION and family not in VALIDATION_FAMILIES:
        raise SuiteValidationError(
            "validation family outside frozen post-lock ordinals"
        )
    task_id = base.task_id
    parameters = dict(base.public_parameters)
    if "seed" in parameters:
        parameters["seed"] = 1 + _counter_u64(
            task_id, split.value, family, stream="scenario-seed"
        ) % (2**31 - 2)
    if task_id == "e07_s02_faults_1d":
        parameters["faultIndex"] = _counter_u64(
            task_id, split.value, family, stream="fault-index"
        ) % len(parameters["values"])
    elif task_id == "e07_s02_chimera_1d":
        parameters["replicate"] = int(family)
    elif task_id == "e07_s02_regeneration_1d":
        # E05 exposes four validated fresh-core fixtures.  Eight logical
        # validation families therefore reuse them in prespecified pairs;
        # they are never represented as eight independent physical supports.
        parameters["replicateOrdinal"] = (
            family - TRAIN_FAMILIES[0]
            if split is Split.TRAIN
            else (family - VALIDATION_FAMILIES[0]) % 4
        )
    elif task_id.startswith("e07_s02_spatial2d_") and split is Split.TRAIN:
        parameters["counterScheduleKey"] = (
            f"E07/S08/{task_id}/train-family/{family:03d}"
        )
    label = task_id.removeprefix("e07_s02_").replace("_", "-")
    scenario_id = f"e07s08c:{label}:{split.value}:{family:03d}"
    return ScenarioRecord(
        scenario_id=scenario_id,
        task_id=task_id,
        split=split,
        materializer_id=f"s08c_{split.value}_{family:03d}",
        public_parameters=parameters,
        protected=False,
        outcome_access="development" if split is Split.TRAIN else "validation",
        predecessor_partition=(
            f"S08C_outcome_blind_{split.value}_derivative_of_{base.scenario_id}"
        ),
    )


def scenario_commitment(record: ScenarioRecord) -> str:
    return canonical_sha256(
        "E07/S08C/scenario/v1",
        {
            "scenarioId": record.scenario_id,
            "taskId": record.task_id,
            "split": record.split.value,
            "publicParameters": dict(record.public_parameters),
            "predecessorPartition": record.predecessor_partition,
        },
    )


def _result_valid(result: Any, task_id: str) -> tuple[bool, list[str]]:
    false_flags = sorted(key for key, value in result.validation.items() if not value)
    allowed_source_terminal = (
        task_id == "e07_s02_regeneration_1d"
        and result.failed
        and result.stop_reason == "source_terminal"
        and false_flags == [EXPECTED_E05_FAILURE_FLAG]
    )
    errors = []
    if not result.replay_pass:
        errors.append("replay")
    if false_flags and not allowed_source_terminal:
        errors.append("native_validation")
    if not result.stop_reason or not result.native_costs or not result.native_event:
        errors.append("native_contract_fields")
    return not errors, errors


def _candidate_lock_authorize(
    lock_path: str | Path,
    expected_sha256: str,
    configuration_id: str,
    task_id: str,
    action_sha256: str,
) -> dict[str, Any]:
    path = Path(lock_path)
    if sha256_file(path) != expected_sha256:
        raise AccessDeniedError("candidate lock bytes changed")
    lock = json.loads(path.read_text(encoding="utf-8"))
    entries = {
        (str(row["taskId"]), str(row["configurationId"]), str(row["actionSha256"]))
        for row in lock["entries"]
    }
    if (task_id, configuration_id, action_sha256) not in entries:
        raise AccessDeniedError("action is absent from frozen candidate lock")
    return lock


def evaluate_work_item(work: Mapping[str, Any]) -> dict[str, Any]:
    """Run one complete native episode. Safe for process workers and cache replay."""

    configuration = dict(work["configuration"])
    task_id = str(configuration["taskId"])
    family = int(work["scenarioFamilyOrdinal"])
    split = Split(str(work["split"]))
    by_hash, by_id = _catalog()
    action = build_action(configuration, by_hash, by_id)
    tasks, train_bases, validation_bases = _base_records()
    record = derive_record(
        train_bases[task_id] if split is Split.TRAIN else validation_bases[task_id],
        family,
        split=split,
    )
    task = tasks[task_id]
    if split is Split.TRAIN:
        if work.get("candidateLockPath") or work.get("candidateLockSha256"):
            raise AccessDeniedError("training work cannot carry a validation lock")
        started = time.perf_counter()
        result = RUNNERS[task.runner_id](record, action)
        elapsed = time.perf_counter() - started
        access = "development_training"
    else:
        lock_path = str(work["candidateLockPath"])
        lock_sha = str(work["candidateLockSha256"])
        _candidate_lock_authorize(
            lock_path,
            lock_sha,
            str(configuration["configurationId"]),
            task_id,
            action.policy_sha256,
        )
        grant = AccessGrant(AccessPhase.VALIDATION, lock_sha)
        broker = AccessBroker({record.scenario_id: record}, confirmation_unsealed=False)
        broker.authorize_scenario(record.scenario_id, grant)
        started = time.perf_counter()
        with locked_dsl_validation_scope(record, action, lock_sha):
            result = RUNNERS[task.runner_id](record, action)
        elapsed = time.perf_counter() - started
        broker.authorize_outcome(record.scenario_id, grant)
        access = "candidate_locked_validation"
    valid_contract, contract_errors = _result_valid(result, task_id)
    outcome, outcome_nonfinite = _json_safe(dict(result.native_outcome))
    costs, cost_nonfinite = _json_safe(
        {key: dict(value) for key, value in result.native_costs.items()}
    )
    event, event_nonfinite = _json_safe(dict(result.native_event))
    provenance, provenance_nonfinite = _json_safe(dict(result.provenance))
    stable = {
        "schemaVersion": "e07.s08c.evaluation-row.v1",
        "researchStepId": "S08C",
        "stage": str(work["stage"]),
        "generation": int(work["generation"]),
        "taskId": task_id,
        "split": split.value,
        "scenarioFamilyOrdinal": family,
        "scenarioId": record.scenario_id,
        "scenarioCommitmentSha256": scenario_commitment(record),
        "logicalSlotId": str(work["logicalSlotId"]),
        "reservedConfigurationSlotId": str(work["reservedConfigurationSlotId"]),
        "configurationRole": str(work["configurationRole"]),
        "configurationId": str(configuration["configurationId"]),
        "mode": str(configuration["mode"]),
        "memberSetId": str(configuration["memberSetId"]),
        "memberPolicySha256": [
            str(row["policySha256"]) for row in configuration["members"]
        ],
        "configurationDefinitionSha256": canonical_hash(
            "E07/S08C/configuration-definition/v1", configuration
        ),
        "actionSha256": action.policy_sha256,
        "nativeUnit": task.horizon.native_unit,
        "stopReason": result.stop_reason,
        "censored": bool(result.censored),
        "failed": bool(result.failed),
        "replayPass": bool(result.replay_pass),
        "validation": {key: bool(value) for key, value in result.validation.items()},
        "nativeContractPass": valid_contract,
        "nativeContractErrors": contract_errors,
        "outcome": outcome,
        "nativeLedgerFamilies": costs,
        "nativeEvent": event,
        "provenance": provenance,
        "claimBoundary": task.claim_boundary,
        "accessBoundary": access,
        "nativeNonFiniteValues": (
            outcome_nonfinite + cost_nonfinite + event_nonfinite + provenance_nonfinite
        ),
    }
    stable["stableEvaluationSha256"] = canonical_sha256(
        "E07/S08C/evaluation/v1", stable
    )
    return {**stable, "elapsedSeconds": elapsed, "workerPid": os.getpid()}


def _physical_key(work: Mapping[str, Any]) -> str:
    config = work["configuration"]
    _, train_bases, validation_bases = _base_records()
    split = Split(str(work["split"]))
    task_id = str(config["taskId"])
    record = derive_record(
        train_bases[task_id] if split is Split.TRAIN else validation_bases[task_id],
        int(work["scenarioFamilyOrdinal"]),
        split=split,
    )
    by_hash, by_id = _catalog()
    action = build_action(config, by_hash, by_id)
    return canonical_sha256(
        "E07/S08C/physical-work/v1",
        {
            "split": split.value,
            "taskId": task_id,
            "scenario": scenario_commitment(record),
            "actionSha256": action.policy_sha256,
        },
    )


def execute_work(
    work: Sequence[Mapping[str, Any]],
    *,
    workers: int = 8,
    cache_dir: Path = CACHE_ROOT / "evaluations",
    failure_accounting_path: Path | None = None,
    executor_factory: Callable[..., Executor] = ProcessPoolExecutor,
    atomic_new_cache: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute unique physical rows and expand them to the exact logical roster."""

    if atomic_new_cache and cache_dir.exists():
        raise FileExistsError(
            f"fail-atomic batch requires a new cache directory: {cache_dir}"
        )
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if not atomic_new_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
    unique: dict[str, Mapping[str, Any]] = {}
    logical_keys = []
    for row in work:
        key = _physical_key(row)
        unique.setdefault(key, row)
        logical_keys.append(key)
    results: dict[str, dict[str, Any]] = {}
    pending = []
    pending_keys = []
    for key, row in unique.items():
        path = cache_dir / f"{key}.json"
        if path.exists():
            results[key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            pending_keys.append(key)
            pending.append(row)
    batch_accounting: dict[str, Any] | None = None
    if pending:
        try:
            batch_results, batch_accounting = execute_fail_atomic_batch(
                pending_keys,
                pending,
                worker=evaluate_work_item,
                workers=workers,
                executor_factory=executor_factory,
            )
        except FailAtomicBatchError as exc:
            accounting = {
                **exc.accounting,
                "logicalRows": len(work),
                "uniquePhysicalRows": len(unique),
                "cacheHits": len(unique) - len(pending),
                "preexistingCacheRowsUnchanged": True,
            }
            if failure_accounting_path is not None:
                _write_json(failure_accounting_path, accounting)
            raise FailAtomicBatchError(accounting) from exc
        results.update(batch_results)
        # No new cache row is visible until every worker row has succeeded.
        if atomic_new_cache:
            staging = Path(
                tempfile.mkdtemp(
                    prefix=f".{cache_dir.name}.s08d-staging-", dir=cache_dir.parent
                )
            )
            try:
                for key in pending_keys:
                    _write_json(staging / f"{key}.json", batch_results[key])
                os.replace(staging, cache_dir)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        else:
            for key in pending_keys:
                _write_json(cache_dir / f"{key}.json", batch_results[key])
    expanded = []
    for logical, key in zip(work, logical_keys, strict=True):
        row = dict(results[key])
        row.update(
            {
                "stage": str(logical["stage"]),
                "generation": int(logical["generation"]),
                "logicalSlotId": str(logical["logicalSlotId"]),
                "reservedConfigurationSlotId": str(
                    logical["reservedConfigurationSlotId"]
                ),
                "configurationRole": str(logical["configurationRole"]),
                "scenarioFamilyOrdinal": int(logical["scenarioFamilyOrdinal"]),
            }
        )
        row["logicalExpansionSha256"] = canonical_sha256(
            "E07/S08C/logical-expansion/v1",
            {
                "stableEvaluationSha256": row["stableEvaluationSha256"],
                "logicalSlotId": row["logicalSlotId"],
                "reservedConfigurationSlotId": row["reservedConfigurationSlotId"],
            },
        )
        expanded.append(row)
    return expanded, {
        "logicalRows": len(work),
        "uniquePhysicalRows": len(unique),
        "cacheHits": len(unique) - len(pending),
        "physicalRowsExecutedNow": len(pending),
        "failAtomic": True,
        "atomicNewCache": atomic_new_cache,
        "batchAccounting": batch_accounting,
    }


def publish_parquet_fail_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Publish a complete Parquet result as one same-directory rename."""

    if path.exists():
        raise FileExistsError(f"refusing to replace existing publication: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = canonical_sha256(
        "E07/S08D/parquet-publication/v1",
        {"path": path.name, "rows": len(frame), "columns": list(frame.columns)},
    )[:16]
    staging = path.with_name(f".{path.name}.{digest}.{os.getpid()}.staging")
    try:
        frame.to_parquet(staging, index=False, compression="zstd")
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def _initial_work(
    budget: pd.DataFrame, configurations: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for item in budget[budget["stage"] == "initial"].to_dict("records"):
        cid = str(item["configurationId"])
        rows.append(
            {
                "stage": "initial",
                "generation": 0,
                "taskId": str(item["taskId"]),
                "split": "train",
                "scenarioFamilyOrdinal": int(item["scenarioFamilyOrdinal"]),
                "logicalSlotId": str(item["logicalSlotId"]),
                "reservedConfigurationSlotId": cid,
                "configurationRole": str(item["configurationRole"]),
                "configuration": dict(configurations[cid]),
                "smoke": bool(item["smoke"]),
            }
        )
    return rows


def aggregate_configuration(
    configuration: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    task_id = str(configuration["taskId"])
    objectives: dict[str, float] = {}
    directions: dict[str, str] = {}
    for objective_id, path, direction in OBJECTIVES[task_id]:
        values = [float(_get_path(row["outcome"], path)) for row in rows]
        objectives[objective_id] = float(np.mean(values))
        directions[objective_id] = direction
    cost_rows = [_numeric_leaves(row["nativeLedgerFamilies"]) for row in rows]
    fields = sorted({field for row in cost_rows for field in row})
    costs = {
        field: float(np.mean([row.get(field, 0.0) for row in cost_rows]))
        for field in fields
        if not field.endswith("licensedLongRangeMaximumRequestedDistance")
    }
    quality = {
        **{
            f"performance.{key}": -value if directions[key] == "maximize" else value
            for key, value in objectives.items()
        },
        **{f"cost.{key}": value for key, value in costs.items()},
    }
    descriptors = _aggregate_descriptors(task_id, rows)
    valid = all(
        row["nativeContractPass"] and row["replayPass"] and not row["failed"]
        for row in rows
    )
    return {
        "schemaVersion": "e07.s08c.configuration-aggregate.v1",
        "researchStepId": "S08C",
        "taskId": task_id,
        "configurationId": configuration["configurationId"],
        "policySha256": configuration["configurationId"],
        "mode": configuration["mode"],
        "portfolioSize": int(configuration["portfolioSize"]),
        "memberSetId": configuration["memberSetId"],
        "memberPolicySha256": [row["policySha256"] for row in configuration["members"]],
        "scenarioFamilyOrdinals": sorted(
            int(row["scenarioFamilyOrdinal"]) for row in rows
        ),
        "objectives": objectives,
        "objectiveDirections": directions,
        "costVector": costs,
        "qualityMinimization": quality,
        "descriptors": descriptors,
        "failedCount": sum(bool(row["failed"]) for row in rows),
        "censoredCount": sum(bool(row["censored"]) for row in rows),
        "validNativeEpisodes": valid,
        "evaluationHashes": [
            row["stableEvaluationSha256"]
            for row in sorted(rows, key=lambda item: item["scenarioFamilyOrdinal"])
        ],
    }


def aggregate_panel(
    configurations: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["configurationId"])].append(row)
    return [
        aggregate_configuration(configurations[cid], grouped[cid])
        for cid in sorted(grouped)
    ]


def build_archive(
    aggregates: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, tuple[int, ...]], list[Mapping[str, Any]]]:
    archive: dict[tuple[str, str, tuple[int, ...]], list[Mapping[str, Any]]] = {}
    for candidate in sorted(aggregates, key=lambda row: row["configurationId"]):
        if (
            candidate["mode"] not in TARGET_MODES
            or not candidate["validNativeEpisodes"]
        ):
            continue
        for descriptor in candidate["descriptors"]:
            key = (
                str(candidate["taskId"]),
                str(descriptor["archiveId"]),
                tuple(int(value) for value in descriptor["cell"]),
            )
            front = list(archive.get(key, []))
            if any(dominates(item, candidate) for item in front):
                continue
            survivors = [item for item in front if not dominates(candidate, item)]
            equal = [
                item
                for item in survivors
                if item["qualityMinimization"] == candidate["qualityMinimization"]
            ]
            if (
                equal
                and min(item["configurationId"] for item in equal)
                < candidate["configurationId"]
            ):
                continue
            survivors = [
                item
                for item in survivors
                if item["qualityMinimization"] != candidate["qualityMinimization"]
                or candidate["configurationId"] < item["configurationId"]
            ]
            survivors.append(candidate)
            archive[key] = sorted(
                {item["configurationId"]: item for item in survivors}.values(),
                key=lambda item: item["configurationId"],
            )
    return archive


def _parent_order(
    archive: Mapping[tuple[str, str, tuple[int, ...]], Sequence[Mapping[str, Any]]],
    task_id: str,
) -> list[Mapping[str, Any]]:
    result = []
    for key in sorted(key for key in archive if key[0] == task_id):
        front = sorted(
            archive[key],
            key=lambda row: (int(row["portfolioSize"]), str(row["configurationId"])),
        )
        result.extend(front)
    dedup = {}
    for row in result:
        dedup.setdefault(str(row["configurationId"]), row)
    return list(dedup.values())


def _selector(task_id: str, signal: str, *, flipped: bool = False) -> dict[str, Any]:
    return {
        "profile": "bounded_memoryless_two_branch_v1",
        "signal": signal,
        "operator": "gt" if signal == "repair.nudge_count" else "eq",
        "threshold": 0 if signal == "repair.nudge_count" else True,
        "trueBranch": (
            "current_or_first_compatible_member"
            if flipped
            else "next_compatible_member"
        ),
        "falseBranch": (
            "next_compatible_member"
            if flipped
            else "current_or_first_compatible_member"
        ),
        "persistentMemoryBits": 0,
    }


def _enrich_members(
    task_id: str,
    members: Sequence[Mapping[str, Any]],
    candidates: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [candidates[(task_id, str(row["policySha256"]))] for row in members]


def _legal_members(task_id: str, members: Sequence[Mapping[str, Any]]) -> bool:
    if not 2 <= len(members) <= 4:
        return False
    if any(
        not _member_is_compatible(item, members[:index])
        for index, item in enumerate(members)
    ):
        return False
    carriers = {str(row["nativeCarrier"]) for row in members}
    return (
        {"Bubble", "Insertion"}.issubset(carriers)
        if task_id == CHIMERA_TASK
        else len(carriers) == 1
    )


def _adaptive_definition(
    task_id: str,
    mode: str,
    members: Sequence[Mapping[str, Any]],
    *,
    selector: Mapping[str, Any] | None,
    assignment_rotation: int = 0,
) -> dict[str, Any]:
    ordered = sorted(
        members, key=lambda row: (row["nativeCarrier"], row["policySha256"])
    )
    member_rows = [
        {
            "policyId": row["policyId"],
            "policySha256": row["policySha256"],
            "policyBodySha256": row["policyBodySha256"],
            "nativeCarrier": row["nativeCarrier"],
            "boundedSemanticGroup": row["boundedSemanticGroup"],
        }
        for row in ordered
    ]
    member_set_id = canonical_hash(
        "E07/S08C/member-set/v1",
        [task_id, [row["policySha256"] for row in member_rows]],
    )
    payload = {
        "schemaVersion": "e07.s08c.adaptive-portfolio.v1",
        "researchStepId": "S08C",
        "taskId": task_id,
        "mode": mode,
        "memberSetId": member_set_id,
        "members": member_rows,
        "selector": dict(selector) if selector is not None else None,
        "assignmentCounterDomain": f"E07/S08/{mode}/identity-opportunity/v1",
        "assignmentRotation": int(assignment_rotation),
        "portfolioSize": len(member_rows),
        "portfolioStructuralCosts": _portfolio_costs(ordered, selector),
        "s07ArmMembershipUsed": False,
        "rejectedModelOrEmbeddingUsed": False,
    }
    config_id = canonical_hash("E07/S08C/adaptive-configuration/v1", payload)
    return {
        **payload,
        "configurationId": config_id,
        "evaluationAuthorized": True,
        "outcomeMaterialized": False,
    }


def mutate_configuration(
    parent: Mapping[str, Any],
    *,
    generation: int,
    ordinal: int,
    retry: int,
    candidates: Mapping[tuple[str, str], Mapping[str, Any]],
    selector_signals: Mapping[str, Sequence[str]],
) -> tuple[dict[str, Any], str]:
    task_id = str(parent["taskId"])
    members = _enrich_members(task_id, parent["members"], candidates)
    operator = MUTATION_OPERATORS[
        (generation * 16 + ordinal + retry) % len(MUTATION_OPERATORS)
    ]
    mode = str(parent["mode"])
    selector = deepcopy(parent.get("selector"))
    rotation = int(parent.get("assignmentRotation", 0))

    def index(stream: str, size: int) -> int:
        return _counter_u64(task_id, generation, ordinal, retry, stream=stream) % size

    if operator == "replace_one_member_same_native_carrier":
        slot = index("replace-slot", len(members))
        carrier = str(members[slot]["nativeCarrier"])
        picked = members[:slot] + members[slot + 1 :]
        options = sorted(
            (
                row
                for (candidate_task, _), row in candidates.items()
                if candidate_task == task_id
                and row["nativeCarrier"] == carrier
                and _member_is_compatible(row, picked)
            ),
            key=lambda row: (
                _counter_u64(
                    row["policySha256"],
                    task_id,
                    generation,
                    ordinal,
                    retry,
                    stream="replace-choice",
                ),
                row["policySha256"],
            ),
        )
        if not options:
            raise ValueError("no legal replacement")
        members[slot] = options[0]
    elif operator == "add_one_member_same_native_carrier":
        if len(members) >= 4:
            raise ValueError("portfolio already has four members")
        carriers = sorted({str(row["nativeCarrier"]) for row in members})
        carrier = carriers[index("add-carrier", len(carriers))]
        options = sorted(
            (
                row
                for (candidate_task, _), row in candidates.items()
                if candidate_task == task_id
                and row["nativeCarrier"] == carrier
                and _member_is_compatible(row, members)
            ),
            key=lambda row: (
                _counter_u64(
                    row["policySha256"],
                    task_id,
                    generation,
                    ordinal,
                    retry,
                    stream="add-choice",
                ),
                row["policySha256"],
            ),
        )
        if not options:
            raise ValueError("no legal added member")
        members.append(options[0])
    elif operator == "drop_one_member_preserving_task_carrier_contract":
        if len(members) <= 2:
            raise ValueError("portfolio cannot drop below two")
        slots = list(range(len(members)))
        slots.sort(key=lambda slot: index(f"drop-{slot}", 2**31))
        retained = next(
            (
                members[:slot] + members[slot + 1 :]
                for slot in slots
                if _legal_members(task_id, members[:slot] + members[slot + 1 :])
            ),
            None,
        )
        if retained is None:
            raise ValueError("no legal drop")
        members = retained
    elif operator == "rotate_fixed_identity_assignment":
        mode = "fixed_balanced_identity"
        selector = None
        rotation = (rotation + 1) % len(members)
    elif operator == "toggle_permitted_selector_signal":
        signals = list(selector_signals[task_id])
        if mode != "environment_conditioned":
            mode = "environment_conditioned"
            selector = _selector(task_id, signals[index("selector", len(signals))])
        elif len(signals) > 1:
            current = signals.index(str(selector["signal"]))
            selector = _selector(task_id, signals[(current + 1) % len(signals)])
        else:
            mode = "fixed_balanced_identity"
            selector = None
    else:
        signals = list(selector_signals[task_id])
        mode = "environment_conditioned"
        signal = str(selector["signal"]) if selector is not None else signals[0]
        already_flipped = (
            selector is not None
            and selector.get("trueBranch") == "current_or_first_compatible_member"
        )
        selector = _selector(task_id, signal, flipped=not already_flipped)
    if not _legal_members(task_id, members):
        raise ValueError("mutation violates frozen member legality")
    definition = _adaptive_definition(
        task_id,
        mode,
        members,
        selector=selector,
        assignment_rotation=rotation % len(members),
    )
    return definition, operator


def matched_random_definition(target: Mapping[str, Any]) -> dict[str, Any]:
    task_id = str(target["taskId"])
    candidate_rows = [
        {
            **row,
            "complexity": compile_policy(
                _catalog()[0][str(row["policySha256"])]
            ).complexity.to_dict(),
        }
        for row in target["members"]
    ]
    # _adaptive_definition only needs the frozen member and complexity fields.
    return _adaptive_definition(
        task_id,
        "random_dynamic_opportunity",
        candidate_rows,
        selector=None,
        assignment_rotation=0,
    )


def generate_adaptive_generation(
    generation: int,
    archive: Mapping[tuple[str, str, tuple[int, ...]], Sequence[Mapping[str, Any]]],
    definitions: Mapping[str, Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    selector_signals: Mapping[str, Sequence[str]],
    budget: pd.DataFrame,
    prior_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    candidates = {
        (str(row["taskId"]), str(row["policySha256"])): row for row in candidate_rows
    }
    generated: dict[str, dict[str, Any]] = {}
    lineage = []
    works = []
    generation_budget = budget[
        (budget["stage"] == "adaptive") & (budget["generation"] == generation)
    ]
    for task_id in TASK_IDS:
        parents = _parent_order(archive, task_id)
        if not parents:
            raise RuntimeError(f"no valid task-local portfolio parent for {task_id}")
        pair_groups = generation_budget[generation_budget["taskId"] == task_id].groupby(
            "pairedSlotId", sort=True
        )
        for ordinal, (pair_id, pair_rows) in enumerate(pair_groups):
            parent_aggregate = parents[ordinal % len(parents)]
            parent = definitions[str(parent_aggregate["configurationId"])]
            target = None
            operator = None
            used_retry = None
            for retry in range(128):
                try:
                    proposal, candidate_operator = mutate_configuration(
                        parent,
                        generation=generation,
                        ordinal=ordinal,
                        retry=retry,
                        candidates=candidates,
                        selector_signals=selector_signals,
                    )
                except ValueError:
                    continue
                if (
                    proposal["configurationId"] in prior_ids
                    or proposal["configurationId"] in generated
                ):
                    continue
                target, operator, used_retry = proposal, candidate_operator, retry
                break
            if target is None:
                raise RuntimeError(
                    f"adaptive duplicate exhaustion: {task_id} generation {generation} ordinal {ordinal}"
                )
            comparator = matched_random_definition(target)
            generated[target["configurationId"]] = target
            generated.setdefault(comparator["configurationId"], comparator)
            prior_ids.update((target["configurationId"], comparator["configurationId"]))
            lineage.append(
                {
                    "schemaVersion": "e07.s08c.adaptive-lineage-row.v1",
                    "researchStepId": "S08C",
                    "generation": generation,
                    "taskId": task_id,
                    "pairedSlotId": str(pair_id),
                    "parentConfigurationId": parent["configurationId"],
                    "targetConfigurationId": target["configurationId"],
                    "matchedRandomConfigurationId": comparator["configurationId"],
                    "mutationOperator": operator,
                    "retryOrdinal": used_retry,
                    "memberPolicySha256": [
                        row["policySha256"] for row in target["members"]
                    ],
                    "s07ArmMembershipUsed": False,
                    "rejectedModelOrEmbeddingUsed": False,
                }
            )
            by_role = {
                role: frame for role, frame in pair_rows.groupby("configurationRole")
            }
            for role, definition in (
                ("target_portfolio", target),
                ("matched_random_dynamic", comparator),
            ):
                frame = by_role[role]
                for slot in frame.to_dict("records"):
                    works.append(
                        {
                            "stage": "adaptive",
                            "generation": generation,
                            "taskId": task_id,
                            "split": "train",
                            "scenarioFamilyOrdinal": int(slot["scenarioFamilyOrdinal"]),
                            "logicalSlotId": str(slot["logicalSlotId"]),
                            "reservedConfigurationSlotId": str(slot["configurationId"]),
                            "configurationRole": role,
                            "configuration": definition,
                            "smoke": False,
                        }
                    )
    if len(works) != 1024 or not 128 <= len(generated) <= 256 or len(lineage) != 128:
        raise RuntimeError("adaptive generation accounting diverged from frozen budget")
    return works, generated, lineage


def _archive_rows(
    archive: Mapping[tuple[str, str, tuple[int, ...]], Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for (task_id, archive_id, cell), front in sorted(archive.items()):
        for rank, item in enumerate(front):
            rows.append(
                {
                    "taskId": task_id,
                    "archiveId": archive_id,
                    "cell": list(cell),
                    "frontRankByHash": rank,
                    "configurationId": item["configurationId"],
                    "mode": item["mode"],
                    "portfolioSize": item["portfolioSize"],
                    "qualityMinimization": item["qualityMinimization"],
                    "objectives": item["objectives"],
                    "costVector": item["costVector"],
                }
            )
    return rows


def _single_frontier_leaders(
    task_id: str,
    aggregates: Sequence[Mapping[str, Any]],
) -> list[str]:
    singles = [
        row
        for row in aggregates
        if row["taskId"] == task_id
        and row["mode"] == "single_policy"
        and row["validNativeEpisodes"]
    ]
    selected = set()
    for objective_id, _, direction in OBJECTIVES[task_id]:
        ordered = sorted(
            singles,
            key=lambda row: (
                -float(row["objectives"][objective_id])
                if direction == "maximize"
                else float(row["objectives"][objective_id]),
                row["configurationId"],
            ),
        )
        if ordered:
            selected.add(str(ordered[0]["configurationId"]))
    return sorted(selected)


def select_shortlist(
    archive: Mapping[tuple[str, str, tuple[int, ...]], Sequence[Mapping[str, Any]]],
    definitions: Mapping[str, Mapping[str, Any]],
    aggregates: Sequence[Mapping[str, Any]],
    initial_single_by_member: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    shortlist: dict[str, list[str]] = {}
    leaders: dict[str, list[str]] = {}
    for task_id in TASK_IDS:
        leaders[task_id] = _single_frontier_leaders(task_id, aggregates)
        cells = [key for key in sorted(archive) if key[0] == task_id]
        candidates = []
        for key in cells:
            candidates.extend(
                sorted(
                    archive[key],
                    key=lambda row: (int(row["portfolioSize"]), row["configurationId"]),
                )
            )
        picked = []
        seen = set()
        component_ids = set(leaders[task_id])
        for aggregate in candidates:
            cid = str(aggregate["configurationId"])
            if cid in seen:
                continue
            definition = definitions[cid]
            proposed_components = component_ids | {
                str(
                    initial_single_by_member[(task_id, str(row["policySha256"]))][
                        "configurationId"
                    ]
                )
                for row in definition["members"]
            }
            # Eight validation families and at most 48 configurations per task
            # give the exact frozen 3,072-row global ceiling.
            projected_config_count = 2 * (len(picked) + 1) + len(proposed_components)
            if projected_config_count > 48:
                continue
            picked.append(cid)
            component_ids = proposed_components
            seen.add(cid)
            if len(picked) == 8:
                break
        shortlist[task_id] = picked
    return shortlist, leaders


def build_validation_plan(
    shortlist: Mapping[str, Sequence[str]],
    leaders: Mapping[str, Sequence[str]],
    definitions: Mapping[str, Mapping[str, Any]],
    initial_single_by_member: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    lock_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    validation_definitions: dict[str, dict[str, Any]] = {}
    comparison_rows = []
    by_hash, by_id = _catalog()
    for task_id in TASK_IDS:
        for target_id in shortlist[task_id]:
            target = definitions[target_id]
            validation_definitions[target_id] = dict(target)
            comparator_mode = (
                "random_static_identity"
                if target["mode"] == "fixed_balanced_identity"
                else "random_dynamic_opportunity"
            )
            candidate_rows = [
                {
                    **row,
                    "complexity": compile_policy(
                        by_hash[str(row["policySha256"])]
                    ).complexity.to_dict(),
                }
                for row in target["members"]
            ]
            comparator = _adaptive_definition(
                task_id, comparator_mode, candidate_rows, selector=None
            )
            validation_definitions[comparator["configurationId"]] = comparator
            component_ids = {
                str(
                    initial_single_by_member[(task_id, str(row["policySha256"]))][
                        "configurationId"
                    ]
                )
                for row in target["members"]
            }
            relevant_singles = sorted(component_ids | set(leaders[task_id]))
            for cid in relevant_singles:
                validation_definitions[cid] = dict(definitions[cid])
            comparison_rows.append(
                {
                    "taskId": task_id,
                    "targetConfigurationId": target_id,
                    "targetMode": target["mode"],
                    "matchedRandomConfigurationId": comparator["configurationId"],
                    "matchedRandomMode": comparator_mode,
                    "componentSingleConfigurationIds": sorted(component_ids),
                    "relevantSingleFrontierConfigurationIds": list(leaders[task_id]),
                }
            )
    lock_entries = []
    for cid, definition in sorted(validation_definitions.items()):
        action = build_action(definition, by_hash, by_id)
        lock_entries.append(
            {
                "taskId": definition["taskId"],
                "configurationId": cid,
                "configurationDefinitionSha256": canonical_hash(
                    "E07/S08C/configuration-definition/v1", definition
                ),
                "actionSha256": action.policy_sha256,
                "memberPolicySha256": [
                    row["policySha256"] for row in definition["members"]
                ],
                "mode": definition["mode"],
            }
        )
    lock = {
        "schemaVersion": "e07.s08c.validation-candidate-lock.v1",
        "researchStepId": "S08C",
        "trainingSelectionOnly": True,
        "validationAuthorized": True,
        "confirmationAuthorized": False,
        "entries": lock_entries,
        "shortlist": {key: list(value) for key, value in shortlist.items()},
        "comparisonRegistry": comparison_rows,
    }
    _write_json(lock_path, lock)
    lock_sha = sha256_file(lock_path)
    work = []
    ordinal = 0
    for task_id in TASK_IDS:
        task_config_ids = sorted(
            cid
            for cid, definition in validation_definitions.items()
            if definition["taskId"] == task_id
        )
        for cid in task_config_ids:
            for family in VALIDATION_FAMILIES:
                logical_id = canonical_hash(
                    "E07/S08C/validation-logical-slot/v1", [task_id, cid, family]
                )
                work.append(
                    {
                        "stage": "validation",
                        "generation": 7,
                        "taskId": task_id,
                        "split": "validation",
                        "scenarioFamilyOrdinal": family,
                        "logicalSlotId": logical_id,
                        "reservedConfigurationSlotId": cid,
                        "configurationRole": "locked_validation",
                        "configuration": validation_definitions[cid],
                        "candidateLockPath": str(lock_path),
                        "candidateLockSha256": lock_sha,
                        "logicalSlotOrdinal": ordinal,
                    }
                )
                ordinal += 1
    if len(work) > 3072:
        raise RuntimeError("validation plan exceeds frozen 3,072-row ceiling")
    return work, lock, validation_definitions


def _margin(objective_id: str) -> float:
    if any(
        token in objective_id
        for token in ("completion", "success", "departure", "acceptance")
    ):
        return 0.0
    if any(token in objective_id for token in ("time", "restricted")):
        return 1.0
    return 0.01


def _bootstrap_mean(
    values: np.ndarray, seed_parts: Sequence[str], replicates: int = 2000
) -> np.ndarray:
    n = len(values)
    samples = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        indices = [
            _counter_u64(
                *seed_parts, replicate, draw, 708004, stream="paired-bootstrap"
            )
            % n
            for draw in range(n)
        ]
        samples[replicate] = float(np.mean(values[indices]))
    return samples


def _holm(rows: list[dict[str, Any]], field: str, output: str) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: float(item[1][field]))
    running = 0.0
    m = len(rows)
    for rank, (index, row) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * float(row[field])))
        rows[index][output] = running


def paired_estimands(
    validation_rows: Sequence[Mapping[str, Any]],
    comparison_registry: Sequence[Mapping[str, Any]],
    definitions: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_config_family = {
        (str(row["configurationId"]), int(row["scenarioFamilyOrdinal"])): row
        for row in validation_rows
    }
    endpoint_rows = []
    cost_rows = []
    guard_rows = []
    for comparison in comparison_registry:
        task_id = str(comparison["taskId"])
        target_id = str(comparison["targetConfigurationId"])
        comparator_sets = [
            ("matched_random", [str(comparison["matchedRandomConfigurationId"])]),
            (
                "component_single",
                [str(value) for value in comparison["componentSingleConfigurationIds"]],
            ),
            (
                "single_frontier",
                [
                    str(value)
                    for value in comparison["relevantSingleFrontierConfigurationIds"]
                ],
            ),
        ]
        for contrast, comparator_ids in comparator_sets:
            for comparator_id in sorted(set(comparator_ids)):
                target_panel = [
                    by_config_family[(target_id, family)]
                    for family in VALIDATION_FAMILIES
                ]
                comparator_panel = [
                    by_config_family[(comparator_id, family)]
                    for family in VALIDATION_FAMILIES
                ]
                family_key = f"{target_id}:{contrast}"
                for objective_id, path, direction in OBJECTIVES[task_id]:
                    left = np.array(
                        [float(_get_path(row["outcome"], path)) for row in target_panel]
                    )
                    right = np.array(
                        [
                            float(_get_path(row["outcome"], path))
                            for row in comparator_panel
                        ]
                    )
                    benefit = left - right if direction == "maximize" else right - left
                    bootstrap = _bootstrap_mean(
                        benefit,
                        [task_id, target_id, contrast, comparator_id, objective_id],
                    )
                    margin = _margin(objective_id)
                    endpoint_rows.append(
                        {
                            "taskId": task_id,
                            "targetConfigurationId": target_id,
                            "targetMode": definitions[target_id]["mode"],
                            "contrastFamily": contrast,
                            "comparatorConfigurationId": comparator_id,
                            "objectiveId": objective_id,
                            "direction": direction,
                            "benefitPositiveDifference": float(np.mean(benefit)),
                            "margin": margin,
                            "ciLower": float(np.quantile(bootstrap, 0.025)),
                            "ciUpper": float(np.quantile(bootstrap, 0.975)),
                            "benefitP": float(
                                (1 + np.sum(bootstrap <= margin)) / (len(bootstrap) + 1)
                            ),
                            "harmP": float(
                                (1 + np.sum(bootstrap >= -margin))
                                / (len(bootstrap) + 1)
                            ),
                            "pairedFamilies": len(benefit),
                            "multiplicityFamily": family_key,
                        }
                    )
                target_costs = [
                    _numeric_leaves(row["nativeLedgerFamilies"]) for row in target_panel
                ]
                comparator_costs = [
                    _numeric_leaves(row["nativeLedgerFamilies"])
                    for row in comparator_panel
                ]
                fields = sorted(
                    {field for row in target_costs + comparator_costs for field in row}
                )
                for field in fields:
                    differences = np.array(
                        [
                            target_costs[index].get(field, 0.0)
                            - comparator_costs[index].get(field, 0.0)
                            for index in range(len(VALIDATION_FAMILIES))
                        ]
                    )
                    cost_rows.append(
                        {
                            "taskId": task_id,
                            "targetConfigurationId": target_id,
                            "contrastFamily": contrast,
                            "comparatorConfigurationId": comparator_id,
                            "costField": field,
                            "targetMinusComparatorMean": float(np.mean(differences)),
                            "crossFamilySumApplied": False,
                        }
                    )
                for endpoint in ("failed", "censored"):
                    differences = np.array(
                        [
                            float(target_panel[index][endpoint])
                            - float(comparator_panel[index][endpoint])
                            for index in range(len(VALIDATION_FAMILIES))
                        ]
                    )
                    guard_rows.append(
                        {
                            "taskId": task_id,
                            "targetConfigurationId": target_id,
                            "contrastFamily": contrast,
                            "comparatorConfigurationId": comparator_id,
                            "endpoint": endpoint,
                            "riskDifferenceTargetMinusComparator": float(
                                np.mean(differences)
                            ),
                            "guardPass": float(np.mean(differences)) <= 0.0,
                        }
                    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in endpoint_rows:
        grouped[(row["taskId"], row["multiplicityFamily"])].append(row)
    for rows in grouped.values():
        _holm(rows, "benefitP", "holmBenefitP")
        _holm(rows, "harmP", "holmHarmP")
    return endpoint_rows, cost_rows, guard_rows


def s09_gate(
    shortlist: Mapping[str, Sequence[str]],
    endpoint_rows: Sequence[Mapping[str, Any]],
    guard_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for task_id in TASK_IDS:
        for cid in shortlist[task_id]:
            endpoints = [
                row for row in endpoint_rows if row["targetConfigurationId"] == cid
            ]
            guards = [row for row in guard_rows if row["targetConfigurationId"] == cid]
            episode_rows = [
                row for row in validation_rows if row["configurationId"] == cid
            ]
            no_harm = not any(
                row["benefitPositiveDifference"] < -row["margin"]
                and row["holmHarmP"] < 0.05
                for row in endpoints
            )
            matched_benefit = any(
                row["contrastFamily"] == "matched_random"
                and row["benefitPositiveDifference"] > row["margin"]
                and row["holmBenefitP"] < 0.05
                for row in endpoints
            )
            single_benefit = any(
                row["contrastFamily"] in {"component_single", "single_frontier"}
                and row["benefitPositiveDifference"] > row["margin"]
                and row["holmBenefitP"] < 0.05
                for row in endpoints
            )
            guard = bool(guards) and all(row["guardPass"] for row in guards)
            replay = len(episode_rows) == 8 and all(
                row["replayPass"] and row["nativeContractPass"] for row in episode_rows
            )
            eligible = (
                no_harm and matched_benefit and single_benefit and guard and replay
            )
            rows.append(
                {
                    "taskId": task_id,
                    "configurationId": cid,
                    "completePairedValidation": bool(endpoints) and replay,
                    "noHolmSignificantHarmBeyondMargin": no_harm,
                    "holmSignificantBenefitVersusMatchedRandom": matched_benefit,
                    "holmSignificantBenefitVersusRelevantSingles": single_benefit,
                    "failureCensorGuardPass": guard,
                    "exactReplayAndContractPass": replay,
                    "separatePortfolioCostsDisclosed": True,
                    "frozenHashesPass": True,
                    "s09Eligible": eligible,
                }
            )
    return {
        "schemaVersion": "e07.s08c.s09-eligibility-gate.v1",
        "researchStepId": "S08C",
        "rows": rows,
        "eligibleConfigurationIds": sorted(
            row["configurationId"] for row in rows if row["s09Eligible"]
        ),
        "ifNone": "bounded_null_and_limit_S09_to_single_policy_ablation_or_stop",
    }


def run_preflight(control_path: str | Path = CONTROL_PATH) -> dict[str, Any]:
    control = yaml.safe_load(Path(control_path).read_text(encoding="utf-8"))
    schema_version = control.get("schemaVersion")
    if schema_version not in {
        "e07.s08c.execution-continuation.v1",
        "e07.s08e.execution-continuation.v1",
    }:
        raise RuntimeError("unexpected S08 execution continuation control")
    research_step_id = str(control.get("researchStepId", "S08C"))
    immutable_steps = tuple(
        control.get("immutableSteps", ("S05", "S08P", "S08A", "S08", "S08B"))
    )
    roots = {step: ARTIFACT_ROOT / step for step in immutable_steps}
    before = {step: tree_digest(root) for step, root in roots.items()}
    tree_pass = before == control["expectedTrees"]
    file_checks = []
    for path_text, expected in sorted(control["expectedFiles"].items()):
        path = Path(path_text)
        actual = sha256_file(path)
        file_checks.append(
            {
                "path": str(path),
                "expectedSha256": str(expected),
                "actualSha256": actual,
                "pass": actual == str(expected),
            }
        )
    protocol = checked_protocol(S08P / "s08p_portfolio_protocol.yaml")
    frozen_input_checks = verify_frozen_inputs(protocol)
    manifest_steps = tuple(control.get("manifestSteps", ("S08P", "S08A", "S08B")))
    manifests = [_manifest_validation(ARTIFACT_ROOT / step) for step in manifest_steps]
    candidates = read_jsonl(S08P / "candidate_eligibility_registry.jsonl")
    members = read_jsonl(S08P / "portfolio_member_sets.jsonl")
    configs = read_jsonl(S08P / "portfolio_seed_registry.jsonl")
    budget = pd.read_parquet(S08P / "budget_slot_ledger.parquet")
    legality = validate_portfolio_registry(protocol, candidates, members, configs)
    freeze = json.loads((S08P / "preregistration_freeze.json").read_text())
    candidate_hash = candidate_commitment(candidates)
    complete_plan = plan_digest(candidates, members, configs, budget.to_dict("records"))
    smoke_ids = set(budget.loc[budget["smoke"], "configurationId"].astype(str))
    bindings = validate_executable_bindings(configs, smoke_ids)
    qualification_gate_path = Path(
        control.get(
            "sourceQualification",
            S08B / "s08_execution_eligibility_gate.json",
        )
    )
    qualification_gate = json.loads(qualification_gate_path.read_text())
    gate_source_pass = (
        qualification_gate.get("technicalEligibilityPass") is True
        and not qualification_gate.get("blockedGateIds")
        and all(row["status"] == "pass" for row in qualification_gate["rows"])
    )
    fail_atomic_evidence_path = control.get("failAtomicEvidence")
    if fail_atomic_evidence_path is None:
        fail_atomic_pass = True
        fail_atomic_evidence = None
    else:
        fail_atomic_evidence = json.loads(
            Path(fail_atomic_evidence_path).read_text(encoding="utf-8")
        )
        fail_atomic_pass = bool(
            fail_atomic_evidence.get("success")
            and fail_atomic_evidence.get("sameResultCommitmentAcrossWorkerOrders")
            and fail_atomic_evidence.get("existingPublicationReplacementDenied")
        )
    candidate_pass = (
        len(candidates) == 512
        and candidate_hash == freeze["candidateEligibilityCommitmentSha256"]
        and all(
            row["eligible"]
            and not row["eligibilityUsesS07ArmMembership"]
            and not row["eligibilityUsesRejectedModelOrEmbedding"]
            and not row["promotionEvidence"]
            for row in candidates
        )
    )
    accounting_pass = (
        len(budget) == 11008
        and int(budget["smoke"].sum()) == 40
        and set(budget["split"]) == {"train"}
        and budget.groupby("generation").size().to_dict()
        == {0: 4864, 1: 1024, 2: 1024, 3: 1024, 4: 1024, 5: 1024, 6: 1024}
    )
    loaded_prohibited = sorted(
        name
        for name in sys.modules
        if name.startswith("src.surrogate_models")
        or name.startswith("src.surrogate_remediation")
    )
    access_rows = []
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    by_hash, by_id = _catalog()
    config_by_task = {
        task: min(
            (
                row
                for row in configs
                if row["taskId"] == task and row["mode"] == "single_policy"
            ),
            key=lambda row: row["configurationId"],
        )
        for task in TASK_IDS
    }
    for task_id, configuration in config_by_task.items():
        action = build_action(configuration, by_hash, by_id)
        for record in suite.records.values():
            if record.task_id != task_id or record.split is Split.TRAIN:
                continue
            try:
                suite.open_for_action(
                    task_id,
                    record.scenario_id,
                    AccessGrant(
                        AccessPhase.VALIDATION
                        if record.split is Split.VALIDATION
                        else AccessPhase.CONFIRMATION,
                        "0" * 64 if record.split is Split.CONFIRMATION else None,
                    ),
                    action,
                )
                denied = False
            except AccessDeniedError:
                denied = True
            access_rows.append(
                {
                    "taskId": task_id,
                    "split": record.split.value,
                    "deniedBeforeMaterialization": denied,
                }
            )
    access_pass = len(access_rows) == 16 and all(
        row["deniedBeforeMaterialization"] for row in access_rows
    )
    hash_pass = (
        tree_pass
        and all(row["pass"] for row in file_checks)
        and all(row["success"] for row in manifests)
        and candidate_hash == freeze["candidateEligibilityCommitmentSha256"]
        and complete_plan == freeze["completePlanSha256"]
    )
    gate_rows = [
        {
            "gateId": "G01",
            "status": "pass" if hash_pass else "blocked",
            "requirement": "frozen inputs and historical evidence unchanged",
        },
        {
            "gateId": "G02",
            "status": "pass" if candidate_pass else "blocked",
            "requirement": "512 canonical candidates with no S07-arm or rejected-model promotion",
        },
        {
            "gateId": "G03",
            "status": "pass" if legality["illegalCompositions"] == 0 else "blocked",
            "requirement": "frozen compositions, selectors, costs, and paired plan legal",
        },
        {
            "gateId": "G04",
            "status": "pass" if bindings["success"] and gate_source_pass else "blocked",
            "requirement": "live qualification and bindings cover all 1,216 configurations and 40 smoke rows",
        },
        {
            "gateId": "G05",
            "status": "pass" if access_pass and not loaded_prohibited else "blocked",
            "requirement": "protected denial and rejected dependency exclusion",
        },
        {
            "gateId": "G06",
            "status": "pass" if accounting_pass and fail_atomic_pass else "blocked",
            "requirement": "exact accounting and qualified fail-atomic publication",
        },
    ]
    blocked = [row["gateId"] for row in gate_rows if row["status"] != "pass"]
    after = {step: tree_digest(root) for step, root in roots.items()}
    return {
        "schemaVersion": f"e07.{research_step_id.lower()}.preflight.v1",
        "researchStepId": research_step_id,
        "success": not blocked and before == after,
        "status": "eligible_for_frozen_smoke"
        if not blocked
        else "blocked_before_smoke",
        "blockedGateIds": blocked,
        "gateRows": gate_rows,
        "treeHashesBefore": before,
        "treeHashesAfter": after,
        "fileChecks": file_checks,
        "frozenInputChecks": frozen_input_checks,
        "manifestChecks": manifests,
        "candidateCommitmentExpected": freeze["candidateEligibilityCommitmentSha256"],
        "candidateCommitmentActual": candidate_hash,
        "completePlanExpected": freeze["completePlanSha256"],
        "completePlanActual": complete_plan,
        "bindingValidation": bindings,
        "qualificationGatePath": str(qualification_gate_path),
        "qualificationGatePass": gate_source_pass,
        "failAtomicEvidencePath": fail_atomic_evidence_path,
        "failAtomicEvidence": fail_atomic_evidence,
        "failAtomicEvidencePass": fail_atomic_pass,
        "accessRows": access_rows,
        "loadedProhibitedModules": loaded_prohibited,
        "accounting": {
            "trainingLogicalRows": len(budget),
            "smokeLogicalRows": int(budget["smoke"].sum()),
            "validationLogicalCeiling": 3072,
            "confirmationLogicalRows": 0,
        },
        "noMutation": before == after,
    }


def validate_rows(rows: Sequence[Mapping[str, Any]], expected: int) -> dict[str, Any]:
    hashes = [row["logicalExpansionSha256"] for row in rows]
    unique_slots = {row["logicalSlotId"] for row in rows}
    errors = []
    if len(rows) != expected or len(unique_slots) != expected:
        errors.append("logical_accounting")
    if not all(row["nativeContractPass"] and row["replayPass"] for row in rows):
        errors.append("native_contract_or_replay")
    if len(hashes) != len(rows):
        errors.append("hash_accounting")
    return {
        "success": not errors,
        "expectedRows": expected,
        "observedRows": len(rows),
        "uniqueLogicalSlots": len(unique_slots),
        "failedRowsRetained": sum(bool(row["failed"]) for row in rows),
        "censoredRowsRetained": sum(bool(row["censored"]) for row in rows),
        "replayPassRows": sum(bool(row["replayPass"]) for row in rows),
        "nativeContractPassRows": sum(bool(row["nativeContractPass"]) for row in rows),
        "errors": errors,
    }


def immutable_tree_hashes(
    steps: Sequence[str] = ("S05", "S08P", "S08A", "S08", "S08B"),
) -> dict[str, str]:
    return {step: tree_digest(ARTIFACT_ROOT / step) for step in steps}


def environment_record() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": sys.platform,
        "cpuCount": os.cpu_count(),
        "workers": 8,
        "numericThreadsPerWorker": 1,
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
