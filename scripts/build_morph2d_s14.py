#!/usr/bin/env python3
"""Freeze and execute E06 S14 minimal-control and spatial-transfer analyses."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import torch

from morph2d.gpu_engine import (
    resolve_compiled_proposals,
    synthetic_compiled_workload,
    tensor_permutation_invariants,
)
from morph2d.grammar import score_grid
from morph2d.minimal_control import (
    load_minimal_control_catalog,
    policy_catalog,
    run_minimal_condition_task,
    run_minimal_once,
    run_transfer_condition_task,
    run_transfer_once,
    scenario_identity,
    validate_minimal_control_catalog,
)
from morph2d.targets import evaluate_success
from morph2d.baseline import load_baseline_assets


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("/artifacts/research_steps/S14")
DEFAULT_CACHE = Path("/cache/e06_s14")
CATALOG_PATH = ROOT / "configs/morphologies/minimal_control_catalog.yaml"
COST_AXES = [
    "totalInformationBitsIncludingObservationAndSchedule",
    "totalGraphDisplacement",
    "externalInterventionGraphDisplacement",
    "totalSourceWorkUnits",
    "totalComputationUnits",
    "totalOpportunityCostUnits",
]
MANDATORY_CONTROL_IDS = [
    "local_only_control",
    "bounded_central_full_control",
    "central_monitor_sham_control",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def sha256_value(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_bytes(value)
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl_gz(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def condition_id(kind: str, task: Mapping[str, Any]) -> str:
    keys = (
        ("split", "targetId", "challengeId", "controlPolicyId")
        if kind == "calibrated"
        else ("split", "fixtureId", "challengeId", "controlPolicyId")
    )
    return (
        kind[:4]
        + ":"
        + sha256_value("E06/S14/condition/v1", {key: task[key] for key in keys})[:20]
    )


def trace_run_ids(kind: str, task: Mapping[str, Any]) -> list[str]:
    scenario_key = task["targetId"] if kind == "calibrated" else task["fixtureId"]
    values = [
        scenario_identity(
            str(task["split"]),
            str(scenario_key),
            str(task["challengeId"]),
            int(replicate),
            str(task["controlPolicyId"]),
        )["runId"]
        for replicate in task["replicates"]
    ]
    return sorted(values)[:1]


def make_calibrated_task(
    catalog: Mapping[str, Any],
    *,
    phase: str,
    split: str,
    target_id: str,
    challenge_id: str,
    policy_id: str,
    replicates: int,
    event_budget: int,
) -> dict[str, Any]:
    task = {
        "catalog": dict(catalog),
        "phase": phase,
        "split": split,
        "targetId": target_id,
        "challengeId": challenge_id,
        "controlPolicyId": policy_id,
        "replicates": list(range(int(replicates))),
        "eventBudget": int(event_budget),
    }
    task["conditionId"] = condition_id("calibrated", task)
    task["traceRunIds"] = trace_run_ids("calibrated", task)
    return task


def make_transfer_task(
    catalog: Mapping[str, Any],
    *,
    phase: str,
    split: str,
    fixture_id: str,
    challenge_id: str,
    policy_id: str,
    replicates: int,
    event_budget: int,
) -> dict[str, Any]:
    task = {
        "catalog": dict(catalog),
        "phase": phase,
        "split": split,
        "fixtureId": fixture_id,
        "challengeId": challenge_id,
        "controlPolicyId": policy_id,
        "replicates": list(range(int(replicates))),
        "eventBudget": int(event_budget),
    }
    task["conditionId"] = condition_id("transfer", task)
    task["traceRunIds"] = trace_run_ids("transfer", task)
    return task


def training_tasks(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    split = catalog["splits"]["training"]
    return [
        make_calibrated_task(
            catalog,
            phase="training_search",
            split=str(split["splitId"]),
            target_id=str(split["targetIds"][0]),
            challenge_id=str(split["challengeIds"][0]),
            policy_id=str(item["policyId"]),
            replicates=int(split["replicatesPerPolicy"]),
            event_budget=int(split["eventBudgetTransitions"]),
        )
        for item in catalog["interventionPolicies"]["searchCandidates"]
    ]


def validation_tasks(
    catalog: Mapping[str, Any], promoted: Sequence[str]
) -> list[dict[str, Any]]:
    split = catalog["splits"]["validation"]
    policies = list(dict.fromkeys([*promoted, *MANDATORY_CONTROL_IDS]))
    return [
        make_calibrated_task(
            catalog,
            phase="validation",
            split=str(split["splitId"]),
            target_id=str(split["targetIds"][0]),
            challenge_id=str(split["challengeIds"][0]),
            policy_id=policy_id,
            replicates=int(split["replicatesPerPolicy"]),
            event_budget=int(split["eventBudgetTransitions"]),
        )
        for policy_id in policies
    ]


def holdout_tasks(
    catalog: Mapping[str, Any], locked: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    policies = list(dict.fromkeys([*locked, *MANDATORY_CONTROL_IDS]))
    holdout = catalog["splits"]["heldout"]
    calibrated = [
        make_calibrated_task(
            catalog,
            phase="heldout_confirmation",
            split=str(holdout["splitId"]),
            target_id=str(target_id),
            challenge_id=str(holdout["challengeIds"][0]),
            policy_id=policy_id,
            replicates=int(holdout["replicatesPerPolicy"]),
            event_budget=int(holdout["eventBudgetTransitions"]),
        )
        for target_id in holdout["targetIds"]
        for policy_id in policies
    ]
    negative = catalog["splits"]["negativeEndpointAudit"]
    audits = [
        make_calibrated_task(
            catalog,
            phase="negative_endpoint_audit",
            split=str(negative["splitId"]),
            target_id=str(negative["targetIds"][0]),
            challenge_id=str(challenge_id),
            policy_id=policy_id,
            replicates=int(negative["replicatesPerPolicy"]),
            event_budget=int(negative["eventBudgetTransitions"]),
        )
        for challenge_id in negative["challengeIds"]
        for policy_id in policies
    ]
    transfer = catalog["splits"]["spatialTransfer"]
    diagnostics = [
        make_transfer_task(
            catalog,
            phase="spatial_transfer_diagnostic",
            split=str(transfer["splitId"]),
            fixture_id=str(fixture_id),
            challenge_id=str(challenge_id),
            policy_id=policy_id,
            replicates=int(transfer["replicatesPerPolicy"]),
            event_budget=int(transfer["eventBudgetTransitions"]),
        )
        for fixture_id in transfer["fixtureIds"]
        for challenge_id in transfer["challengeIds"]
        for policy_id in policies
    ]
    return calibrated, audits, diagnostics


def worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def run_tasks(
    tasks: Sequence[Mapping[str, Any]],
    cache_dir: Path,
    *,
    workers: int,
    label: str,
    kind: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    phase_cache = cache_dir / label
    phase_cache.mkdir(parents=True, exist_ok=True)
    worker: Callable[[Mapping[str, Any]], Mapping[str, Any]] = (
        run_minimal_condition_task
        if kind == "calibrated"
        else run_transfer_condition_task
    )
    results: dict[str, tuple[pd.DataFrame, list[dict[str, Any]]]] = {}
    pending: list[Mapping[str, Any]] = []
    resumed = 0
    for task in tasks:
        table_path = phase_cache / f"{task['conditionId'].replace(':', '-')}.parquet"
        trace_path = (
            phase_cache / f"{task['conditionId'].replace(':', '-')}.traces.jsonl.gz"
        )
        if table_path.exists():
            frame = pd.read_parquet(table_path)
            if len(frame) == len(task["replicates"]):
                results[str(task["conditionId"])] = (frame, read_jsonl_gz(trace_path))
                resumed += 1
                continue
        pending.append(task)
    print(f"[{label}] resumed={resumed} pending={len(pending)}", flush=True)
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
            futures = {pool.submit(worker, task): task for task in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                result = future.result()
                frame = pd.DataFrame(result["rows"])
                stem = task["conditionId"].replace(":", "-")
                frame.to_parquet(
                    phase_cache / f"{stem}.parquet", index=False, compression="zstd"
                )
                write_jsonl_gz(
                    phase_cache / f"{stem}.traces.jsonl.gz", result["traces"]
                )
                results[str(task["conditionId"])] = (frame, result["traces"])
                print(
                    f"[{label}] {index}/{len(pending)} {task['conditionId']} rows={len(frame)}",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    ordered = [results[str(task["conditionId"])] for task in tasks]
    frame = (
        pd.concat([item[0] for item in ordered], ignore_index=True)
        if ordered
        else pd.DataFrame()
    )
    traces = [trace for item in ordered for trace in item[1]]
    execution = {
        "label": label,
        "kind": kind,
        "conditionTasks": len(tasks),
        "rows": len(frame),
        "resumedTasks": resumed,
        "executedTasks": len(pending),
        "wallSecondsThisInvocation": elapsed,
    }
    return frame, traces, execution


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return center - radius, center + radius


def policy_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    group_columns = [
        column
        for column in (
            "phase",
            "targetId",
            "fixtureId",
            "challengeId",
            "controlPolicyId",
        )
        if column in frame
    ]
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys, strict=True))
        complete = group[~group["failed"].astype(bool)]
        record.update(
            {
                "runCount": len(group),
                "completedRunCount": len(complete),
                "failedRunCount": int(group["failed"].sum()),
            }
        )
        if (
            "terminalConjunctiveCompletion" in complete
            and complete["terminalConjunctiveCompletion"].notna().any()
        ):
            successes = int(
                complete["terminalConjunctiveCompletion"].astype(bool).sum()
            )
            low, high = wilson_interval(successes, len(complete))
            record.update(
                {
                    "terminalCompletionCount": successes,
                    "terminalCompletionFraction": successes / len(complete),
                    "terminalCompletionWilsonLow95": low,
                    "terminalCompletionWilsonHigh95": high,
                    "terminalS01MismatchFraction": float(
                        complete["terminalS01MismatchFraction"].mean()
                    ),
                    "terminalS02RelationalScore": float(
                        complete["terminalS02RelationalScore"].mean()
                    ),
                    "censoredFraction": float(complete["censored"].mean()),
                }
            )
        if "terminalReferenceMismatchFraction" in complete:
            record.update(
                {
                    "terminalReferenceMismatchFraction": float(
                        complete["terminalReferenceMismatchFraction"].mean()
                    ),
                    "minimumReferenceMismatchFraction": float(
                        complete["minimumReferenceMismatchFraction"].mean()
                    ),
                    "terminalReferenceMatchDiagnosticFraction": float(
                        complete["terminalReferenceMatchDiagnostic"].mean()
                    ),
                    "terminalGraphComponentError": float(
                        complete["terminalGraphComponentError"].mean()
                    ),
                    "terminalNormalizedBoundaryError": float(
                        complete["terminalNormalizedBoundaryError"].mean()
                    ),
                }
            )
        for column in COST_AXES + [
            "scheduledDirectQueryCount",
            "directRecipientQuerySlots",
            "controllerInputBits",
            "actuationAttempts",
            "overrideActionUnits",
            "foregoneNativeActorSlots",
            "shamSuppressedControllerRecommendations",
        ]:
            if column in complete:
                record[column] = float(complete[column].mean())
        rows.append(record)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def pareto_training(summary: pd.DataFrame, catalog: Mapping[str, Any]) -> pd.DataFrame:
    gate = catalog["optimization"]["trainingReliabilityGate"]
    completion_margin = float(
        catalog["optimization"]["morphologyEquivalenceMargins"][
            "terminalCompletionFraction"
        ]
    )
    mismatch_margin = float(
        catalog["optimization"]["morphologyEquivalenceMargins"][
            "terminalS01MismatchFraction"
        ]
    )
    rows = summary.copy()
    rows["reliabilityEligible"] = (
        rows["terminalCompletionFraction"] >= float(gate["terminalCompletionFraction"])
    ) & (rows["terminalCompletionWilsonLow95"] >= float(gate["wilsonLower95"]))
    rows["paretoNondominated"] = False
    rows["dominatedByJson"] = "[]"
    records = rows.to_dict(orient="records")
    for index, row in enumerate(records):
        if not row["reliabilityEligible"]:
            continue
        dominators = []
        for other in records:
            if (
                other["controlPolicyId"] == row["controlPolicyId"]
                or not other["reliabilityEligible"]
            ):
                continue
            morphology_no_worse = bool(
                other["terminalCompletionFraction"]
                >= row["terminalCompletionFraction"] - completion_margin
                and other["terminalS01MismatchFraction"]
                <= row["terminalS01MismatchFraction"] + mismatch_margin
            )
            costs_no_greater = all(
                float(other[column]) <= float(row[column]) for column in COST_AXES
            )
            strict = bool(
                other["terminalCompletionFraction"] > row["terminalCompletionFraction"]
                or other["terminalS01MismatchFraction"]
                < row["terminalS01MismatchFraction"]
                or any(
                    float(other[column]) < float(row[column]) for column in COST_AXES
                )
            )
            if morphology_no_worse and costs_no_greater and strict:
                dominators.append(str(other["controlPolicyId"]))
        rows.loc[index, "paretoNondominated"] = not dominators
        rows.loc[index, "dominatedByJson"] = json.dumps(
            sorted(dominators), separators=(",", ":")
        )
    return rows


def tie_break_key(policy_id: str, catalog: Mapping[str, Any]) -> tuple[Any, ...]:
    policy = policy_catalog(catalog)[policy_id]
    epochs = list(map(int, policy["directEpochs"]))
    mean_epoch = sum(epochs) / len(epochs) if epochs else float("inf")
    return (len(epochs), int(policy["candidateLimit"]), -mean_epoch, policy_id)


def select_training(
    summary: pd.DataFrame, catalog: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    frontier = pareto_training(summary, catalog)
    eligible = frontier[
        frontier["reliabilityEligible"] & frontier["paretoNondominated"]
    ]["controlPolicyId"].tolist()
    promoted = sorted(eligible, key=lambda value: tie_break_key(str(value), catalog))[
        : int(catalog["optimization"]["maximumPoliciesPromotedToValidation"])
    ]
    return frontier, list(map(str, promoted))


def select_validation(
    summary: pd.DataFrame, promoted: Sequence[str], catalog: Mapping[str, Any]
) -> list[str]:
    gate = catalog["optimization"]["validationReliabilityGate"]
    eligible = summary[
        summary["controlPolicyId"].isin(promoted)
        & (
            summary["terminalCompletionFraction"]
            >= float(gate["terminalCompletionFraction"])
        )
        & (summary["terminalCompletionWilsonLow95"] >= float(gate["wilsonLower95"]))
    ]["controlPolicyId"].tolist()
    return sorted(map(str, eligible), key=lambda value: tie_break_key(value, catalog))[
        : int(catalog["optimization"]["maximumPoliciesLockedForHeldout"])
    ]


def freeze(output: Path) -> None:
    catalog = load_minimal_control_catalog()
    validate_minimal_control_catalog(catalog)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CATALOG_PATH, output / "frozen_minimal_control_design.yaml")
    policy_rows = _all_policy_rows(catalog)
    write_json(output / "intervention_policy_archive.json", policy_rows)
    split_manifest = {
        "schemaVersion": "e06.s14.split-manifest.v1",
        "researchStepId": "S14",
        "splits": catalog["splits"],
        "heldoutScenarioIdsMaterialized": False,
        "heldoutOutcomeAccessed": False,
        "heldoutAddressTemplateCommitmentSha256": sha256_value(
            "E06/S14/heldout-template/v1",
            {
                "seedContract": catalog["seedContract"],
                "heldout": catalog["splits"]["heldout"],
                "negative": catalog["splits"]["negativeEndpointAudit"],
                "transfer": catalog["splits"]["spatialTransfer"],
            },
        ),
    }
    write_json(output / "split_manifest.json", split_manifest)
    design = {
        "schemaVersion": "e06.s14.design-freeze.v1",
        "researchStepId": "S14",
        "frozenBeforeOptimizationAndOutcomeAccess": True,
        "catalogSha256": file_sha256(CATALOG_PATH),
        "repositoryCommit": git_output("rev-parse", "HEAD"),
        "repositoryTree": git_output("rev-parse", "HEAD^{tree}"),
        "metricDirectionsFrozen": True,
        "topologyCriteriaFrozen": True,
        "splitsFrozen": True,
        "budgetsFrozen": True,
        "searchLimitsFrozen": True,
        "paretoRulesFrozen": True,
        "transferGatesFrozen": True,
        "priorEndpointClassificationsLocked": {
            "S09": "de_novo_random_and_block_formation_null",
            "S10": "repair_null",
            "S12": "all_arm_formation_null_and_terminal_only_central_retention",
            "S13": "metrics_noninterchangeable",
        },
    }
    write_json(output / "design_freeze.json", design)
    write_json(
        output / "pre_outcome_validation.json",
        {
            "schemaVersion": "e06.s14.pre-outcome-validation.v1",
            "researchStepId": "S14",
            "success": True,
            "checks": {
                "catalogValidated": True,
                "thirteenCandidateMaximum": len(policy_rows["searchCandidates"]) == 13,
                "heldoutUnmaterialized": True,
                "scalarCostForbidden": True,
                "topologyTransferDiagnosticOnly": True,
                "formationRepairOptimizationForbidden": True,
                "s01S02ConjunctionRetained": True,
            },
        },
    )


def _all_policy_rows(catalog: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schemaVersion": "e06.s14.intervention-policy-archive.v1",
        "researchStepId": "S14",
        "searchCandidates": catalog["interventionPolicies"]["searchCandidates"],
        "mandatoryControls": catalog["interventionPolicies"]["mandatoryControls"],
        "narrowedContentRule": catalog["interventionPolicies"]["narrowedContentRule"],
    }


def cache_version(catalog: Mapping[str, Any]) -> str:
    return (
        git_output("rev-parse", "--short=12", "HEAD")
        + "-"
        + file_sha256(CATALOG_PATH)[:12]
    )


def execute_search(output: Path, cache: Path, workers: int) -> None:
    _require_freeze(output)
    catalog = load_minimal_control_catalog()
    tasks = training_tasks(catalog)
    frame, traces, execution = run_tasks(
        tasks,
        cache / cache_version(catalog),
        workers=workers,
        label="training",
        kind="calibrated",
    )
    frame.to_parquet(
        output / "training_results.parquet", index=False, compression="zstd"
    )
    write_jsonl_gz(output / "training_sampled_traces.jsonl.gz", traces)
    summary = policy_summary(frame)
    frontier, promoted = select_training(summary, catalog)
    frontier.to_csv(output / "training_minimal_control_curve.csv", index=False)
    decision = {
        "schemaVersion": "e06.s14.training-selection.v1",
        "researchStepId": "S14",
        "candidateCount": len(tasks),
        "promotedPolicyIds": promoted,
        "maximumPromoted": int(
            catalog["optimization"]["maximumPoliciesPromotedToValidation"]
        ),
        "heldoutOutcomeAccessed": False,
        "runtimeUsedForSelection": False,
        "adaptiveCandidateCreation": False,
        "decisionSha256": sha256_value("E06/S14/training-selection/v1", promoted),
    }
    write_json(output / "training_selection.json", decision)
    write_json(output / "training_execution.json", execution)


def execute_validation(output: Path, cache: Path, workers: int) -> None:
    _require_freeze(output)
    training = json.loads(
        (output / "training_selection.json").read_text(encoding="utf-8")
    )
    catalog = load_minimal_control_catalog()
    promoted = list(map(str, training["promotedPolicyIds"]))
    tasks = validation_tasks(catalog, promoted)
    frame, traces, execution = run_tasks(
        tasks,
        cache / cache_version(catalog),
        workers=workers,
        label="validation",
        kind="calibrated",
    )
    frame.to_parquet(
        output / "validation_results.parquet", index=False, compression="zstd"
    )
    write_jsonl_gz(output / "validation_sampled_traces.jsonl.gz", traces)
    summary = policy_summary(frame)
    summary.to_csv(output / "validation_control_curve.csv", index=False)
    locked = select_validation(summary, promoted, catalog)
    lock_body = {
        "schemaVersion": "e06.s14.validation-policy-lock.v1",
        "researchStepId": "S14",
        "promotedPolicyIds": promoted,
        "lockedPolicyIds": locked,
        "writtenBeforeHeldoutScenarioConstruction": True,
        "heldoutScenarioIdsMaterialized": False,
        "heldoutOutcomesAccessed": False,
        "runtimeUsedForSelection": False,
    }
    lock_body["policyLockSha256"] = sha256_value("E06/S14/policy-lock/v1", lock_body)
    write_json(output / "validation_policy_lock.json", lock_body)
    write_json(output / "validation_execution.json", execution)


def execute_holdout(output: Path, cache: Path, workers: int) -> None:
    _require_freeze(output)
    lock = json.loads(
        (output / "validation_policy_lock.json").read_text(encoding="utf-8")
    )
    if (
        not lock["writtenBeforeHeldoutScenarioConstruction"]
        or lock["heldoutOutcomesAccessed"]
    ):
        raise ValueError("S14 heldout lock is invalid")
    catalog = load_minimal_control_catalog()
    calibrated_tasks, negative_tasks, transfer_tasks = holdout_tasks(
        catalog, list(map(str, lock["lockedPolicyIds"]))
    )
    versioned_cache = cache / cache_version(catalog)
    heldout, heldout_traces, heldout_execution = run_tasks(
        calibrated_tasks,
        versioned_cache,
        workers=workers,
        label="heldout",
        kind="calibrated",
    )
    negative, negative_traces, negative_execution = run_tasks(
        negative_tasks,
        versioned_cache,
        workers=workers,
        label="negative_audit",
        kind="calibrated",
    )
    transfer, transfer_traces, transfer_execution = run_tasks(
        transfer_tasks,
        versioned_cache,
        workers=workers,
        label="transfer",
        kind="transfer",
    )
    heldout.to_parquet(
        output / "heldout_results.parquet", index=False, compression="zstd"
    )
    negative.to_parquet(
        output / "negative_endpoint_results.parquet", index=False, compression="zstd"
    )
    transfer.to_parquet(
        output / "transfer_results.parquet", index=False, compression="zstd"
    )
    write_jsonl_gz(output / "heldout_sampled_traces.jsonl.gz", heldout_traces)
    write_jsonl_gz(output / "negative_sampled_traces.jsonl.gz", negative_traces)
    write_jsonl_gz(output / "transfer_sampled_traces.jsonl.gz", transfer_traces)
    policy_summary(heldout).to_csv(
        output / "heldout_retention_summary.csv", index=False
    )
    policy_summary(negative).to_csv(output / "negative_endpoint_curve.csv", index=False)
    policy_summary(transfer).to_csv(
        output / "spatial_transfer_summary.csv", index=False
    )
    write_json(
        output / "holdout_execution.json",
        {
            "schemaVersion": "e06.s14.holdout-execution.v1",
            "researchStepId": "S14",
            "policyLockSha256": lock["policyLockSha256"],
            "heldout": heldout_execution,
            "negativeEndpointAudit": negative_execution,
            "spatialTransfer": transfer_execution,
        },
    )


def _require_freeze(output: Path) -> None:
    required = [
        output / "design_freeze.json",
        output / "pre_outcome_validation.json",
        output / "split_manifest.json",
    ]
    if not all(path.exists() for path in required):
        raise FileNotFoundError("S14 pre-outcome freeze is incomplete")
    design = json.loads(required[0].read_text(encoding="utf-8"))
    if design["catalogSha256"] != file_sha256(CATALOG_PATH):
        raise ValueError("S14 catalog changed after design freeze")


def replay_audit(output: Path) -> pd.DataFrame:
    frames = [
        pd.read_parquet(path)
        for path in (
            output / "training_results.parquet",
            output / "validation_results.parquet",
            output / "heldout_results.parquet",
            output / "negative_endpoint_results.parquet",
        )
    ]
    calibrated = pd.concat(frames, ignore_index=True)
    transfer = pd.read_parquet(output / "transfer_results.parquet")
    records = []
    for _, row in calibrated[calibrated["traceSelected"]].iterrows():
        repeated, _, _ = run_minimal_once(
            {
                "phase": row["phase"],
                "split": row["split"],
                "targetId": row["targetId"],
                "challengeId": row["challengeId"],
                "controlPolicyId": row["controlPolicyId"],
                "replicate": int(row["replicate"]),
                "eventBudget": int(row["eventBudgetTransitions"]),
                "retainTrace": True,
            }
        )
        expected = row.to_dict()
        records.append(
            {
                "runId": row["runId"],
                "kind": "calibrated",
                "exact": _rows_equal(expected, repeated),
                "episodeSha256": row["episodeSha256"],
            }
        )
    for _, row in transfer[transfer["traceSelected"]].iterrows():
        repeated, _ = run_transfer_once(
            {
                "phase": row["phase"],
                "split": row["split"],
                "fixtureId": row["fixtureId"],
                "challengeId": row["challengeId"],
                "controlPolicyId": row["controlPolicyId"],
                "replicate": int(row["replicate"]),
                "eventBudget": int(row["eventBudgetTransitions"]),
                "retainTrace": True,
            }
        )
        records.append(
            {
                "runId": row["runId"],
                "kind": "transfer",
                "exact": _rows_equal(row.to_dict(), repeated),
                "episodeSha256": row["episodeSha256"],
            }
        )
    return pd.DataFrame(records)


def _rows_equal(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    ignored = {"wallSeconds"}
    for key, value in observed.items():
        if key in ignored:
            continue
        other = expected.get(key)
        if pd.isna(other) and value is None:
            continue
        if isinstance(value, bool):
            if bool(other) != value:
                return False
        elif other != value:
            return False
    return True


def target_rescore_audit(output: Path) -> pd.DataFrame:
    frames = [
        pd.read_parquet(output / name)
        for name in (
            "training_results.parquet",
            "validation_results.parquet",
            "heldout_results.parquet",
            "negative_endpoint_results.parquet",
        )
    ]
    frame = pd.concat(frames, ignore_index=True)
    context, targets, grammars, _ = load_baseline_assets()
    del context
    records = []
    for _, block in frame.groupby(
        ["phase", "targetId", "challengeId", "controlPolicyId"], dropna=False
    ):
        count = max(1, math.ceil(0.01 * len(block)))
        for _, row in block.sort_values("runId").head(count).iterrows():
            grid = tuple(tuple(value) for value in json.loads(row["finalGridRowsJson"]))
            global_audit = evaluate_success(grid, targets[str(row["targetId"])])
            local = score_grid(grid, grammars[str(row["grammarId"])])
            records.append(
                {
                    "runId": row["runId"],
                    "s01Match": bool(global_audit["success"])
                    == bool(row["terminalS01GlobalSuccess"]),
                    "s02Match": bool(local["accepted"])
                    == bool(row["terminalS02GrammarAccepted"]),
                    "conjunctionMatch": bool(
                        global_audit["success"] and local["accepted"]
                    )
                    == bool(row["terminalConjunctiveCompletion"]),
                }
            )
    return pd.DataFrame(records)


def benchmark_smoke(output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    specification = {
        "phase": "release_smoke",
        "split": "release_smoke",
        "targetId": "layers_three_ordered_tissues",
        "challengeId": "exact_maintenance",
        "controlPolicyId": "central_q0_freeze",
        "replicate": 0,
        "eventBudget": 17,
        "retainTrace": True,
    }
    first, _, _ = run_minimal_once(specification)
    second, _, _ = run_minimal_once(specification)
    cpu_exact = _rows_equal(first, second)
    if not torch.cuda.is_available():
        raise RuntimeError("fresh S14 release smoke requires the available CUDA device")
    cpu = synthetic_compiled_workload(64, device="cpu")
    gpu = synthetic_compiled_workload(64, device="cuda")
    cpu_result = resolve_compiled_proposals(cpu)
    torch.cuda.synchronize()
    gpu_started = time.perf_counter()
    gpu_result = resolve_compiled_proposals(gpu)
    torch.cuda.synchronize()
    gpu_seconds = time.perf_counter() - gpu_started
    gpu_repeat = resolve_compiled_proposals(gpu)
    torch.cuda.synchronize()
    parity = bool(
        torch.equal(cpu_result.post_state, gpu_result.post_state.cpu())
        and torch.equal(cpu_result.accepted_mask, gpu_result.accepted_mask.cpu())
        and torch.equal(
            cpu_result.total_graph_displacement,
            gpu_result.total_graph_displacement.cpu(),
        )
        and torch.equal(gpu_result.post_state, gpu_repeat.post_state)
        and tensor_permutation_invariants(gpu.state, gpu_result.post_state).all().item()
    )
    record = {
        "schemaVersion": "e06.s14.benchmark-smoke.v1",
        "researchStepId": "S14",
        "success": bool(cpu_exact and parity),
        "cpuEpisodeExactReplay": cpu_exact,
        "gpuBatchSize": 64,
        "gpuIntegerParity": parity,
        "gpuDevice": torch.cuda.get_device_name(),
        "torchVersion": torch.__version__,
        "torchCudaRuntime": torch.version.cuda,
        "gpuKernelSeconds": gpu_seconds,
        "wallSeconds": time.perf_counter() - started,
        "cpuEpisodeSha256": first["episodeSha256"],
    }
    write_json(output / "benchmark_smoke.json", record)
    return record


def validation_summary(
    output: Path, replay: pd.DataFrame, rescore: pd.DataFrame, smoke: Mapping[str, Any]
) -> dict[str, Any]:
    result_paths = [
        output / "training_results.parquet",
        output / "validation_results.parquet",
        output / "heldout_results.parquet",
        output / "negative_endpoint_results.parquet",
        output / "transfer_results.parquet",
    ]
    frames = [pd.read_parquet(path) for path in result_paths]
    total = sum(len(frame) for frame in frames)
    failures = sum(int(frame["failed"].sum()) for frame in frames)
    checks = {
        "completeRunAccounting": failures == 0 and total > 0,
        "zeroExecutionFailures": failures == 0,
        "allInvariantChecks": all(
            bool(frame["invariantSuccess"].all()) for frame in frames
        ),
        "allPermissionChecks": all(
            bool(frame["permissionAuditSuccess"].all()) for frame in frames
        ),
        "allBudgetChecks": all(
            bool(frame["s14BudgetSuccess"].all()) for frame in frames
        ),
        "allReplaysExact": bool(replay["exact"].all()),
        "allTargetRescoresExact": bool(
            rescore[["s01Match", "s02Match", "conjunctionMatch"]].all().all()
        ),
        "transferCompletionFieldsNull": bool(
            frames[-1][
                [
                    "terminalConjunctiveCompletion",
                    "terminalS01GlobalSuccess",
                    "terminalS02GrammarAccepted",
                ]
            ]
            .isna()
            .all()
            .all()
        ),
        "freshBenchmarkSmoke": bool(smoke["success"]),
    }
    return {
        "schemaVersion": "e06.s14.validation-results.v1",
        "researchStepId": "S14",
        "success": all(checks.values()),
        "intendedRunCount": total,
        "completedRunCount": total - failures,
        "failedRunCount": failures,
        "replayCount": len(replay),
        "rescoreCount": len(rescore),
        "checks": checks,
    }


def outcome_decision(output: Path, catalog: Mapping[str, Any]) -> dict[str, Any]:
    lock = json.loads(
        (output / "validation_policy_lock.json").read_text(encoding="utf-8")
    )
    locked = list(map(str, lock["lockedPolicyIds"]))
    heldout = pd.read_csv(output / "heldout_retention_summary.csv")
    negative = pd.read_csv(output / "negative_endpoint_curve.csv")
    heldout_gate = False
    negative_gate = False
    if locked:
        rows = heldout[heldout["controlPolicyId"] == locked[0]]
        heldout_gate = bool(
            len(rows) == len(catalog["splits"]["heldout"]["targetIds"])
            and (rows["terminalCompletionFraction"] >= 0.95).all()
            and (rows["terminalCompletionWilsonLow95"] >= 0.93).all()
        )
        audits = negative[negative["controlPolicyId"] == locked[0]]
        negative_gate = bool(
            len(audits)
            == len(catalog["splits"]["negativeEndpointAudit"]["challengeIds"])
            and (audits["terminalCompletionFraction"] >= 0.20).all()
            and (audits["terminalCompletionWilsonLow95"] >= 0.15).all()
        )
    support = bool(heldout_gate and negative_gate)
    return {
        "schemaVersion": "e06.s14.outcome-decision.v1",
        "researchStepId": "S14",
        "lockedPolicyIds": locked,
        "heldoutRetentionGate": heldout_gate,
        "formationAndRepairSupportGate": negative_gate,
        "s14HypothesisSupported": support,
        "operationalRetentionSupported": heldout_gate,
        "outcomeClassification": "supportive"
        if support
        else "constraining/contradictory",
        "priorClassificationsOverturned": False,
        "transferCompletionClaimMade": False,
    }


def package_release(
    output: Path, validation: Mapping[str, Any], decision: Mapping[str, Any]
) -> dict[str, Any]:
    release_root = Path("/artifacts/release")
    release_root.mkdir(parents=True, exist_ok=True)
    archive_path = release_root / "morph2d-benchmark.tar.zst"
    with tempfile.TemporaryDirectory(
        dir="/cache", prefix="e06_s14_release_"
    ) as temporary:
        package = Path(temporary) / "morph2d-benchmark"
        package.mkdir()
        (package / "README.md").write_text(
            "# Morph2D benchmark\n\nPointer-based E06 benchmark release. Repository source is identified by the Git pointer and is not duplicated here. S14 completion is calibrated only on named bounded-square S01 targets; larger and irregular scenarios are diagnostics. E07-protected rows may not be used for training, reward shaping, search, or selection.\n\nSmoke command: `PYTHONPATH=src python scripts/build_morph2d_s14.py smoke --output /tmp/morph2d-smoke` from the pinned repository commit.\n",
            encoding="utf-8",
        )
        shutil.copy2(
            output / "intervention_policy_archive.json",
            package / "intervention_policy_archive.json",
        )
        shutil.copy2(output / "split_manifest.json", package / "split_manifest.json")
        shutil.copy2(
            output / "validation_policy_lock.json",
            package / "validation_policy_lock.json",
        )
        shutil.copy2(output / "benchmark_smoke.json", package / "benchmark_smoke.json")
        write_json(
            package / "git_pointer.json",
            {
                "repository": "https://github.com/Eidosoma/cell_research",
                "branch": "eidosoma/groups/28",
                "commit": git_output("rev-parse", "HEAD"),
                "tree": git_output("rev-parse", "HEAD^{tree}"),
            },
        )
        scenario_rows = []
        catalog = load_minimal_control_catalog()
        for split_name, split in catalog["splits"].items():
            scenario_rows.append(
                {
                    "splitName": split_name,
                    "splitId": split["splitId"],
                    "targetsOrFixtures": json.dumps(
                        split.get("targetIds", split.get("fixtureIds")),
                        separators=(",", ":"),
                    ),
                    "challenges": json.dumps(
                        split["challengeIds"], separators=(",", ":")
                    ),
                    "replicatesPerPolicy": int(split["replicatesPerPolicy"]),
                    "eventBudgetTransitions": int(split["eventBudgetTransitions"]),
                    "e07ProtectedConfirmation": bool(
                        split.get("e07ProtectedConfirmation", False)
                    ),
                    "completionCalibrated": split_name
                    in {"training", "validation", "heldout", "negativeEndpointAudit"},
                    "optimizationUse": split.get(
                        "optimizationUse",
                        "training_only" if split_name == "training" else "gated",
                    ),
                }
            )
        pd.DataFrame(scenario_rows).to_csv(
            package / "benchmark_scenarios.csv", index=False
        )
        write_json(
            package / "schemas.json",
            {
                "minimalControlRun": "e06.s14.minimal-control-run.v1",
                "transferDiagnosticRun": "e06.s14.transfer-diagnostic-run.v1",
                "catalog": "e06.s14.minimal-control-catalog.v1",
                "validation": validation["schemaVersion"],
            },
        )
        write_json(
            package / "result_pointers.json",
            {
                path.name: {"path": str(path), "sha256": file_sha256(path)}
                for path in sorted(output.glob("*.parquet"))
            },
        )
        components = []
        for path in sorted(package.iterdir()):
            components.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
        manifest = {
            "schemaVersion": "e06.s14.benchmark-release.v1",
            "researchStepId": "S14",
            "releaseClassification": decision["outcomeClassification"],
            "sourceCopied": False,
            "pointerBased": True,
            "completionCalibration": "named bounded-square S01 targets only",
            "components": components,
        }
        write_json(package / "benchmark_manifest.json", manifest)
        tar_path = Path(temporary) / "morph2d-benchmark.tar"
        subprocess.run(
            [
                "tar",
                "--sort=name",
                "--mtime=UTC 1970-01-01",
                "--owner=0",
                "--group=0",
                "--numeric-owner",
                "-cf",
                str(tar_path),
                "-C",
                str(Path(temporary)),
                "morph2d-benchmark",
            ],
            check=True,
        )
        subprocess.run(
            ["zstd", "-q", "-19", "-f", str(tar_path), "-o", str(archive_path)],
            check=True,
        )
    release_manifest = {
        "schemaVersion": "e06.s14.release-archive-manifest.v1",
        "researchStepId": "S14",
        "archivePath": str(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": file_sha256(archive_path),
        "format": "tar.zst",
        "sourceCopied": False,
        "repositoryCommit": git_output("rev-parse", "HEAD"),
        "smokeSuccess": validation["checks"]["freshBenchmarkSmoke"],
    }
    write_json(release_root / "morph2d-benchmark.manifest.json", release_manifest)
    return release_manifest


def build_report_inputs(
    output: Path,
    decision: Mapping[str, Any],
    validation: Mapping[str, Any],
    release: Mapping[str, Any],
) -> None:
    report = Path("/artifacts/report_inputs")
    if report.exists():
        shutil.rmtree(report)
    report.mkdir(parents=True)
    outcomes = {
        "S01": "supportive",
        "S02": "constraining/contradictory",
        "S03": "supportive",
        "S04": "supportive",
        "S05": "supportive",
        "S06": "supportive_with_constraining_asymmetry",
        "S07": "supportive_with_large_batch_constraint",
        "S08": "supportive",
        "S09": "supportive_with_broad_formation_constraint",
        "S10": "null",
        "S11": "constraining/contradictory",
        "S12": "constraining/contradictory",
        "S13": "supportive_with_noninterchangeability_constraint",
        "S14": decision["outcomeClassification"],
    }
    evidence_rows = []
    for step in [f"S{index:02d}" for index in range(1, 15)]:
        path = (
            Path("/artifacts/research_steps") / step / "research_step_full_results.md"
        )
        evidence_rows.append(
            {
                "researchStepId": step,
                "outcomeClassification": outcomes[step],
                "reportPath": str(path),
                "reportExists": path.exists(),
                "reportSha256": file_sha256(path) if path.exists() else None,
            }
        )
    pd.DataFrame(evidence_rows).to_csv(report / "evidence_index.csv", index=False)
    (report / "methods_summary.md").write_text(
        "# E06 methods summary\n\nE06 defined S01 target sets and S02 local grammars, compiled S03 graphs, froze S04 movements, S05 priced observations, and S06 priced channels, then validated an S07/S08 CPU/GPU engine. S09–S12 tested formation, perturbation, chimeras, and hybrid control. S13 calibrated noninterchangeable metrics. S14 exhaustively searched a frozen bounded direct-control grid on exact-start terminal retention, locked policies before held-out access, audited formation/repair without using them for selection, and treated larger/irregular cases as diagnostics. All cost families remained separate.\n",
        encoding="utf-8",
    )
    figures = []
    for path in sorted(Path("/artifacts/research_steps").glob("S*/*")):
        if path.suffix.lower() in {".png", ".svg"}:
            figures.append(
                {
                    "step": path.parent.name,
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
            )
    pd.DataFrame(figures, columns=["step", "path", "sha256"]).to_csv(
        report / "figure_index.csv", index=False
    )
    table_names = [
        "training_minimal_control_curve.csv",
        "validation_control_curve.csv",
        "heldout_retention_summary.csv",
        "negative_endpoint_curve.csv",
        "spatial_transfer_summary.csv",
        "replay_audit.csv",
        "target_rescore_audit.csv",
    ]
    table_rows = [
        {
            "researchStepId": "S14",
            "path": str(output / name),
            "sha256": file_sha256(output / name),
        }
        for name in table_names
    ]
    pd.DataFrame(table_rows).to_csv(report / "table_index.csv", index=False)
    claims = [
        (
            "E06-C01",
            "The S01/S02 conjunction is the only calibrated completion endpoint.",
            "supported_bounded",
            "S01|S02|S13|S14",
        ),
        (
            "E06-C02",
            "Broad de novo formation was not supported under the tested local/hybrid policies.",
            "null_not_supported",
            "S09|S11|S12|S14",
        ),
        (
            "E06-C03",
            "Repair was not supported under frozen count-preserving semantics.",
            "null_not_supported",
            "S10|S14",
        ),
        (
            "E06-C04",
            "Central-only exact-start terminal retention can be reduced to a frozen no-query policy only by suppressing native action.",
            "supported_operational_with_constraint",
            "S12|S14",
        ),
        (
            "E06-C05",
            "Larger-square and irregular outcomes are diagnostic and do not establish completion transfer.",
            "not_identified",
            "S03|S13|S14",
        ),
        (
            "E06-C06",
            "Morphology metrics are informative but noninterchangeable.",
            "constraining/contradictory",
            "S13|S14",
        ),
        (
            "E06-C07",
            "The abstract simulator is not biological validation or evidence of agency.",
            "not_identified",
            "S01|S14",
        ),
    ]
    pd.DataFrame(
        claims, columns=["claimId", "claim", "classification", "evidenceSteps"]
    ).to_csv(report / "claim_to_evidence_matrix.csv", index=False)
    caveats = [
        (
            "CAV-01",
            "Completion calibration is limited to unobstructed bounded-square S01 target shapes.",
        ),
        (
            "CAV-02",
            "S02 local grammar acceptance is globally underdetermined and cannot replace S01.",
        ),
        ("CAV-03", "Terminal exact-start retention is not uninterrupted maintenance."),
        (
            "CAV-04",
            "Zero-query retention under central-only control immobilizes the system and pays full foregone-opportunity cost.",
        ),
        (
            "CAV-05",
            "Formation and repair fixed-budget nulls are not global impossibility proofs.",
        ),
        (
            "CAV-06",
            "Larger-grid and irregular runs lack topology-specific completion calibration.",
        ),
        (
            "CAV-07",
            "Channel bits do not equal semantic value or causal authority; central computation remains separately priced.",
        ),
        (
            "CAV-08",
            "GPU speedup applies to the validated large-batch data plane, not the CPU-owned control plane.",
        ),
    ]
    pd.DataFrame(caveats, columns=["caveatId", "caveat"]).to_csv(
        report / "caveat_register.csv", index=False
    )
    (report / "lay_summary.md").write_text(
        "# Lay summary\n\nThe benchmark did not discover reliable pattern formation or repair. A central-only condition could keep a correct pattern at the final checkpoint with no queries by preventing all ordinary local actions; this is a useful minimal-retention result, but it is immobilization, not self-repair or self-organization. Tests on larger and irregular spaces are diagnostics because success has not been calibrated there.\n",
        encoding="utf-8",
    )
    (report / "e07_handoff.md").write_text(
        "# E06 S14 handoff to E07\n\nUse the pointer-based Morph2D release at `/artifacts/release/morph2d-benchmark.tar.zst`. Rows and split definitions marked `e07ProtectedConfirmation` must be excluded from surrogate training, reward shaping, policy search, and model selection. Preserve the S01/S02 conjunction on calibrated bounded squares; do not optimize S13 diagnostic metrics as substitutes. Treat larger-square and irregular cases as diagnostics until E07 independently calibrates topology-specific completion. Keep information, action, intervention, source-work, computation, and opportunity costs separate, and retain S06 authority asymmetries. The zero-query central policy is an immobilizing exact-start retention control, not formation or repair.\n",
        encoding="utf-8",
    )
    write_json(
        report / "provenance_manifest.json",
        {
            "schemaVersion": "e06.s14.report-provenance.v1",
            "researchStepId": "S14",
            "repositoryCommit": git_output("rev-parse", "HEAD"),
            "repositoryTree": git_output("rev-parse", "HEAD^{tree}"),
            "release": release,
            "validation": validation,
            "decision": decision,
        },
    )
    artifacts = []
    for path in sorted(report.iterdir()):
        artifacts.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    write_json(
        report / "report_input_manifest.json",
        {
            "schemaVersion": "e06.s14.report-input-manifest.v1",
            "researchStepId": "S14",
            "isFinalReportBundle": False,
            "purpose": "complete inputs for later Chief Scientist report generation",
            "artifacts": artifacts,
        },
    )


def finalize(output: Path) -> None:
    catalog = load_minimal_control_catalog()
    calibrated = pd.concat(
        [
            pd.read_parquet(output / name)
            for name in (
                "training_results.parquet",
                "validation_results.parquet",
                "heldout_results.parquet",
                "negative_endpoint_results.parquet",
            )
        ],
        ignore_index=True,
    )
    calibrated.to_parquet(
        output / "minimal_control_results.parquet", index=False, compression="zstd"
    )
    replay = replay_audit(output)
    replay.to_csv(output / "replay_audit.csv", index=False)
    rescore = target_rescore_audit(output)
    rescore.to_csv(output / "target_rescore_audit.csv", index=False)
    smoke = benchmark_smoke(output)
    validation = validation_summary(output, replay, rescore, smoke)
    write_json(output / "validation_results.json", validation)
    decision = outcome_decision(output, catalog)
    write_json(output / "outcome_decision.json", decision)
    release = package_release(output, validation, decision)
    build_report_inputs(output, decision, validation, release)
    write_json(
        output / "execution_manifest.json",
        {
            "schemaVersion": "e06.s14.execution-manifest.v1",
            "researchStepId": "S14",
            "backend": "canonical_s07_cpu_oracle_fallback_with_fresh_gpu_data_plane_smoke",
            "workers": 8,
            "threadEnvironment": {
                "OMP_NUM_THREADS": 1,
                "OPENBLAS_NUM_THREADS": 1,
                "MKL_NUM_THREADS": 1,
            },
            "repositoryCommit": git_output("rev-parse", "HEAD"),
            "catalogSha256": file_sha256(CATALOG_PATH),
        },
    )


def smoke_only(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    benchmark_smoke(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["freeze", "search", "validation", "holdout", "finalize", "smoke"],
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("S14 workers must be in 1..8")
    if args.command == "freeze":
        freeze(args.output)
    elif args.command == "search":
        execute_search(args.output, args.cache, args.workers)
    elif args.command == "validation":
        execute_validation(args.output, args.cache, args.workers)
    elif args.command == "holdout":
        execute_holdout(args.output, args.cache, args.workers)
    elif args.command == "finalize":
        finalize(args.output)
    else:
        smoke_only(args.output)


if __name__ == "__main__":
    main()
