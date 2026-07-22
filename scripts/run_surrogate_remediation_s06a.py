#!/usr/bin/env python3
"""Freeze, evaluate, and fit the bounded E07 S06A remediation."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Mapping, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.quality_diversity.core import TASK_IDS
from src.surrogate_models.core import (
    event_length,
    fit_target_transform,
    get_path,
    numeric_leaves,
    shared_event_features,
    target_arrays,
)
from src.surrogate_remediation.core import (
    PROTOCOL_PATH,
    S05_ROOT,
    S06A_ROOT,
    access_audit,
    assign_s06a_splits,
    build_additional_plan,
    build_s06a_target_registry,
    canonical_bytes,
    canonical_hash,
    evaluate_additional_work_item,
    freeze_inputs,
    hash_file,
    load_catalog,
    load_protocol,
    panel_indices,
    read_jsonl,
    remediation_feature_dict,
    verify_frozen_inputs,
    work_items_from_plan,
    write_json,
    write_jsonl,
)


CACHE = Path("/cache/e07_s06a")
EVAL_CACHE = CACHE / "evaluations"
PLAN_PATH = S06A_ROOT / "scenario_generation_plan.jsonl"
FREEZE_PATH = S06A_ROOT / "input_hash_freeze.json"
ADDED_LEDGER = S06A_ROOT / "added_evaluation_ledger.jsonl"


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(child) for child in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _plan_commitment(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_hash(
        "E07/S06A/scenario-generation-plan/v1",
        [row["planRowSha256"] for row in rows],
    )


def freeze() -> None:
    if (S06A_ROOT / "models").exists() or ADDED_LEDGER.exists():
        raise RuntimeError(
            "S06A freeze refuses an already evaluated or fitted directory"
        )
    protocol = load_protocol()
    S06A_ROOT.mkdir(parents=True, exist_ok=True)
    frozen = freeze_inputs(protocol)
    if frozen["repositoryStatusShort"]:
        raise RuntimeError("S06A freeze requires a clean, committed repository")
    plan = build_additional_plan(protocol)
    shutil.copyfile(PROTOCOL_PATH, S06A_ROOT / "remediation_protocol.yaml")
    write_json(FREEZE_PATH, frozen)
    write_jsonl(PLAN_PATH, plan)
    prereg = {
        "schemaVersion": "e07.s06a.preregistration-freeze.v1",
        "researchStepId": "S06A",
        "success": True,
        "frozenBeforeAnyRemediationFit": True,
        "frozenBeforeAddedScenarioEvaluation": True,
        "frozenAtUtc": datetime.now(timezone.utc).isoformat(),
        "protocolSha256": hash_file(PROTOCOL_PATH),
        "artifactProtocolSha256": hash_file(S06A_ROOT / "remediation_protocol.yaml"),
        "inputFreezeSha256": hash_file(FREEZE_PATH),
        "scenarioPlanSha256": hash_file(PLAN_PATH),
        "scenarioPlanCommitmentSha256": _plan_commitment(plan),
        "scenarioPlanRows": len(plan),
        "unchangedDeploymentThresholds": True,
        "rejectedS06ModelsUsed": False,
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
    }
    write_json(S06A_ROOT / "preregistration_freeze.json", prereg)
    write_json(S06A_ROOT / "access_control_validation.json", access_audit(protocol))


def amend_failure_handling_refreeze() -> None:
    """Refreeze a validation-only amendment before any model is fitted."""

    if not ADDED_LEDGER.exists():
        raise RuntimeError("failure-handling amendment requires the complete ledger")
    if (S06A_ROOT / "models").exists():
        raise RuntimeError("failure-handling amendment must precede every model fit")
    protocol = load_protocol()
    prereg_path = S06A_ROOT / "preregistration_freeze.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if hash_file(PROTOCOL_PATH) != prereg["protocolSha256"]:
        raise RuntimeError("the frozen modeling protocol changed")
    if hash_file(PLAN_PATH) != prereg["scenarioPlanSha256"]:
        raise RuntimeError("the frozen scenario plan changed")
    original_freeze = S06A_ROOT / "pre_evaluation_input_hash_freeze.json"
    original_prereg = S06A_ROOT / "pre_evaluation_preregistration_freeze.json"
    if original_freeze.exists() or original_prereg.exists():
        raise RuntimeError("failure-handling amendment may be applied only once")
    shutil.copyfile(FREEZE_PATH, original_freeze)
    shutil.copyfile(prereg_path, original_prereg)
    amended = freeze_inputs(protocol)
    if amended["repositoryStatusShort"]:
        raise RuntimeError("amended pre-fit freeze requires a clean repository")
    write_json(FREEZE_PATH, amended)
    amendment_path = (
        Path(__file__).resolve().parents[1]
        / "configs/modeling/s06a_failure_handling_amendment.yaml"
    )
    shutil.copyfile(
        amendment_path, S06A_ROOT / "failure_handling_validation_amendment.yaml"
    )
    prereg.update(
        {
            "inputFreezeSha256": hash_file(FREEZE_PATH),
            "validationAmendmentAppliedBeforeFit": True,
            "validationAmendmentSha256": hash_file(amendment_path),
            "preEvaluationInputFreezeSha256": hash_file(original_freeze),
            "preEvaluationPreregistrationSha256": hash_file(original_prereg),
            "repositoryCommitAtPrefitRefreeze": amended["repositoryCommit"],
            "frozenBeforeAnyRemediationFit": True,
        }
    )
    write_json(prereg_path, prereg)
    write_json(
        S06A_ROOT / "failure_handling_amendment_freeze.json",
        {
            "schemaVersion": "e07.s06a.failure-handling-amendment-freeze.v1",
            "researchStepId": "S06A",
            "success": True,
            "appliedBeforeAnyModelFit": True,
            "modelingProtocolChanged": False,
            "scenarioPlanChanged": False,
            "deploymentThresholdsChanged": False,
            "evaluationLedgerChanged": False,
            "splitOrModelRuleChanged": False,
            "repositoryCommit": amended["repositoryCommit"],
            "amendmentSha256": hash_file(amendment_path),
            "reason": (
                "Retained policy-level native validation failures are modeled "
                "failed outcomes, not grounds to delete rows or invalidate accounting."
            ),
        },
    )


def verify_preregistration() -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, Any]]
]:
    protocol = load_protocol()
    prereg = json.loads((S06A_ROOT / "preregistration_freeze.json").read_text())
    frozen = json.loads(FREEZE_PATH.read_text())
    plan = read_jsonl(PLAN_PATH)
    if hash_file(PROTOCOL_PATH) != prereg["protocolSha256"]:
        raise RuntimeError("S06A protocol changed after preregistration")
    if (
        hash_file(S06A_ROOT / "remediation_protocol.yaml")
        != prereg["artifactProtocolSha256"]
    ):
        raise RuntimeError("artifact protocol changed after preregistration")
    if hash_file(FREEZE_PATH) != prereg["inputFreezeSha256"]:
        raise RuntimeError("input freeze record changed")
    if hash_file(PLAN_PATH) != prereg["scenarioPlanSha256"]:
        raise RuntimeError("scenario plan changed")
    if _plan_commitment(plan) != prereg["scenarioPlanCommitmentSha256"]:
        raise RuntimeError("scenario plan commitment changed")
    verify_frozen_inputs(protocol, frozen)
    return protocol, frozen, plan


def _cache_key(work: Mapping[str, Any], protocol_hash: str) -> str:
    return canonical_hash(
        "E07/S06A/evaluation-cache-key/v1",
        {
            "protocolSha256": protocol_hash,
            "planRowSha256": work["planRowSha256"],
            "policySha256": work["policySha256"],
            "taskId": work["taskId"],
            "scenarioOrdinal": work["scenarioOrdinal"],
        },
    )


def _evaluate_batch(
    works: Sequence[Mapping[str, Any]], *, workers: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol, _, _ = verify_preregistration()
    EVAL_CACHE.mkdir(parents=True, exist_ok=True)
    protocol_hash = hash_file(PROTOCOL_PATH)
    rows: list[dict[str, Any]] = []
    pending = []
    for work in works:
        key = _cache_key(work, protocol_hash)
        path = EVAL_CACHE / f"{key}.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
        else:
            pending.append((key, work))
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(evaluate_additional_work_item, work): key
                for key, work in pending
            }
            for future in as_completed(futures):
                key = futures[future]
                row = json.loads(canonical_bytes(future.result()).decode("ascii"))
                write_json(EVAL_CACHE / f"{key}.json", row)
                rows.append(row)
    elapsed = time.perf_counter() - started
    verify_frozen_inputs(protocol, json.loads(FREEZE_PATH.read_text()))
    return sorted(
        rows,
        key=lambda row: (row["taskId"], row["policySha256"], row["scenarioOrdinal"]),
    ), {
        "requested": len(works),
        "cacheHits": len(works) - len(pending),
        "nativeTopLevelEvaluations": len(pending),
        "workers": workers,
        "wallSeconds": elapsed,
    }


def smoke() -> None:
    protocol, _, plan = verify_preregistration()
    works = work_items_from_plan(plan)
    chosen = []
    seen = set()
    for work in works:
        if work["taskId"] not in seen:
            chosen.append(work)
            seen.add(work["taskId"])
    rows, accounting = _evaluate_batch(chosen, workers=8)
    checks = {
        "eightTasks": len(rows) == len(TASK_IDS),
        "allTrain": all(row["split"] == "train" for row in rows),
        "allReplay": all(row["replayPass"] for row in rows),
        "allNativeValidation": all(all(row["validation"].values()) for row in rows),
        "noProtectedOrdinal": all(
            not (
                row["taskId"] == "e07_s02_regeneration_1d"
                and row["scenarioOrdinal"] in {4, 5, 6, 7}
            )
            for row in rows
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"S06A smoke failed: {checks}")
    write_json(
        S06A_ROOT / "smoke_throughput.json",
        {
            "schemaVersion": "e07.s06a.smoke-throughput.v1",
            "researchStepId": "S06A",
            "success": True,
            "checks": checks,
            "accounting": accounting,
            "validationOutcomeEvaluations": 0,
            "confirmationOutcomeEvaluations": 0,
            "protocolSha256": hash_file(PROTOCOL_PATH),
        },
    )
    _ = protocol


def evaluate() -> None:
    protocol, _, plan = verify_preregistration()
    if not (S06A_ROOT / "smoke_throughput.json").exists():
        raise RuntimeError("S06A smoke must precede complete evaluation")
    works = work_items_from_plan(plan)
    batches = []
    rows = []
    for task_id in TASK_IDS:
        task_works = [work for work in works if work["taskId"] == task_id]
        task_rows, accounting = _evaluate_batch(task_works, workers=8)
        accounting["taskId"] = task_id
        batches.append(accounting)
        rows.extend(task_rows)
    rows = sorted(
        rows,
        key=lambda row: (row["taskId"], row["policySha256"], row["scenarioOrdinal"]),
    )
    expected = int(protocol["additionalScenarioPlan"]["expectedTopLevelEvaluations"])
    stable = {row["stableEvaluationSha256"] for row in rows}
    plan_hashes = {row["planRowSha256"] for row in rows}
    if len(rows) != expected or len(stable) != expected or len(plan_hashes) != expected:
        raise RuntimeError("S06A added evaluation accounting is incomplete")
    if any(row["split"] != "train" for row in rows):
        raise RuntimeError("S06A added ledger contains a nontraining row")
    write_jsonl(ADDED_LEDGER, rows)
    write_json(
        S06A_ROOT / "evaluation_accounting.json",
        {
            "schemaVersion": "e07.s06a.evaluation-accounting.v1",
            "researchStepId": "S06A",
            "success": True,
            "plannedTopLevelEvaluations": expected,
            "actualTopLevelEvaluations": len(rows),
            "nativeEvaluationsThisInvocation": sum(
                item["nativeTopLevelEvaluations"] for item in batches
            ),
            "cacheHitsThisInvocation": sum(item["cacheHits"] for item in batches),
            "batches": batches,
            "taskRows": dict(Counter(row["taskId"] for row in rows)),
            "taskPolicies": {
                task: len(
                    {row["policySha256"] for row in rows if row["taskId"] == task}
                )
                for task in TASK_IDS
            },
            "familyRows": dict(Counter(str(row["scenarioOrdinal"]) for row in rows)),
            "failedRows": sum(row["failed"] for row in rows),
            "censoredRows": sum(row["censored"] for row in rows),
            "allReplayPass": all(row["replayPass"] for row in rows),
            "allNativeValidationPass": all(
                all(row["validation"].values()) for row in rows
            ),
            "runtimeDrivenWeakeningApplied": False,
            "wallClockStoppingApplied": False,
            "s05ArchiveMutations": 0,
            "s05LedgerMutations": 0,
            "rejectedS06ModelLoads": 0,
            "validationOutcomeEvaluations": 0,
            "confirmationOutcomeEvaluations": 0,
        },
    )


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    if not len(y):
        return math.nan
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(boundaries[:-1], boundaries[1:], strict=True):
        mask = (probability >= low) & (
            (probability < high) if high < 1.0 else (probability <= high)
        )
        if np.any(mask):
            total += mask.mean() * abs(probability[mask].mean() - y[mask].mean())
    return float(total)


def _conformal(y: np.ndarray, prediction: np.ndarray, nominal: float) -> float:
    mask = np.isfinite(y) & np.isfinite(prediction)
    residual = np.abs(y[mask] - prediction[mask])
    if not len(residual):
        return math.nan
    level = min(1.0, math.ceil((len(residual) + 1) * nominal) / len(residual))
    return float(np.quantile(residual, level, method="higher"))


def _constant_predictions(
    continuous: np.ndarray, binary: np.ndarray, fit: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    c = np.nanmean(continuous[fit], axis=0)
    b = np.nanmean(binary[fit], axis=0)
    c = np.where(np.isfinite(c), c, 0.0)
    b = np.where(np.isfinite(b), b, 0.0)
    return np.tile(c, (len(continuous), 1)), np.tile(b, (len(binary), 1))


def _fit_linear_predictions(
    x: np.ndarray, continuous: np.ndarray, binary: np.ndarray, fit: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[Any], list[Any]]:
    cp, bp = _constant_predictions(continuous, binary, fit)
    continuous_models: list[Any] = [None] * continuous.shape[1]
    binary_models: list[Any] = [None] * binary.shape[1]
    for target in range(continuous.shape[1]):
        observed = fit[np.isfinite(continuous[fit, target])]
        if len(observed) >= 4 and np.nanstd(continuous[observed, target]) > 1e-8:
            model = Ridge(alpha=10.0)
            model.fit(x[observed], continuous[observed, target])
            cp[:, target] = model.predict(x)
            continuous_models[target] = model
    for target in range(binary.shape[1]):
        observed = fit[np.isfinite(binary[fit, target])]
        if len(observed) >= 8 and len(np.unique(binary[observed, target])) == 2:
            model = LogisticRegression(C=1.0, max_iter=2000, random_state=706510)
            model.fit(x[observed], binary[observed, target])
            bp[:, target] = model.predict_proba(x)[:, 1]
            binary_models[target] = model
    return cp, bp, continuous_models, binary_models


def _select_continuous_blend(
    y: np.ndarray,
    ridge: np.ndarray,
    tree: np.ndarray,
    panels: Sequence[np.ndarray],
    weights: Sequence[float],
) -> float:
    candidates = []
    for weight in weights:
        pred = (1.0 - weight) * ridge + weight * tree
        losses = []
        for panel in panels:
            mask = np.isfinite(y[panel])
            if np.any(mask):
                losses.append(
                    float(
                        np.sqrt(np.mean(np.square(y[panel][mask] - pred[panel][mask])))
                    )
                )
        score = (
            max(losses) if losses else math.inf,
            float(np.mean(losses)) if losses else math.inf,
            weight,
        )
        candidates.append((score, weight))
    return float(min(candidates)[1])


def _select_binary_blend(
    y: np.ndarray,
    logistic: np.ndarray,
    tree: np.ndarray,
    constant: np.ndarray,
    panels: Sequence[np.ndarray],
    nonlinear_weights: Sequence[float],
    shrinkages: Sequence[float],
) -> tuple[float, float]:
    candidates = []
    for weight in nonlinear_weights:
        base = (1.0 - weight) * logistic + weight * tree
        for shrinkage in shrinkages:
            pred = np.clip(
                (1.0 - shrinkage) * base + shrinkage * constant, 1e-6, 1 - 1e-6
            )
            panel_scores = []
            panel_brier = []
            for panel in panels:
                mask = np.isfinite(y[panel])
                if np.any(mask):
                    truth = y[panel][mask]
                    probability = pred[panel][mask]
                    brier = float(brier_score_loss(truth, probability))
                    panel_brier.append(brier)
                    panel_scores.append(
                        brier + max(0.0, _ece(truth, probability) - 0.15)
                    )
            score = (
                max(panel_scores) if panel_scores else math.inf,
                float(np.mean(panel_brier)) if panel_brier else math.inf,
                -shrinkage,
                weight,
            )
            candidates.append((score, weight, shrinkage))
    _, weight, shrinkage = min(candidates)
    return float(weight), float(shrinkage)


def _fit_task_bundle(
    task_id: str,
    rows: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    protocol: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    panels = panel_indices(assignments)
    fit = panels["fit"]
    scenario_records = [
        remediation_feature_dict(row, catalog, scenario=True) for row in rows
    ]
    policy_records = [
        remediation_feature_dict(row, catalog, scenario=False) for row in rows
    ]
    vectorizer = DictVectorizer(sparse=False, sort=True)
    vectorizer.fit([scenario_records[index] for index in fit])
    scenario_raw = vectorizer.transform(scenario_records)
    scaler = StandardScaler().fit(scenario_raw[fit])
    x = scaler.transform(scenario_raw)
    policy_vectorizer = DictVectorizer(sparse=False, sort=True)
    policy_vectorizer.fit([policy_records[index] for index in fit])
    policy_raw = policy_vectorizer.transform(policy_records)
    policy_scaler = StandardScaler().fit(policy_raw[fit])
    x_policy = policy_scaler.transform(policy_raw)

    registry = build_s06a_target_registry(task_id, rows)
    continuous_raw, binary = target_arrays(rows, registry)
    transform = fit_target_transform(continuous_raw[fit], registry)
    continuous = transform.transform(continuous_raw)
    constant_c, constant_b = _constant_predictions(continuous, binary, fit)
    ridge_c, logistic_b, ridge_models, logistic_models = _fit_linear_predictions(
        x, continuous, binary, fit
    )
    policy_c, policy_b, policy_ridge_models, policy_logistic_models = (
        _fit_linear_predictions(x_policy, continuous, binary, fit)
    )
    tree_c = ridge_c.copy()
    tree_b = logistic_b.copy()
    candidate_c = ridge_c.copy()
    candidate_b = logistic_b.copy()
    settings = protocol["surrogates"]
    tree_settings = settings["nonlinearHead"]
    calibration = [panels["lineage_calibration"], panels["scenario_calibration"]]
    blend = settings["calibrationBlend"]
    continuous_tree_models: list[Any] = [None] * continuous.shape[1]
    binary_tree_models: list[Any] = [None] * binary.shape[1]
    continuous_weights = [0.0] * continuous.shape[1]
    binary_weights = [0.0] * binary.shape[1]
    binary_shrinkages = [0.0] * binary.shape[1]
    fit_elapsed = np.asarray([float(row["elapsedSeconds"]) for row in rows])
    fit_lengths = np.asarray([event_length(row) for row in rows])
    elapsed_threshold = float(np.quantile(fit_elapsed[fit], 0.75))
    length_threshold = float(np.quantile(fit_lengths[fit], 0.75))
    tail_fit = (fit_elapsed[fit] >= elapsed_threshold) | (
        fit_lengths[fit] >= length_threshold
    )

    seed_base = int(tree_settings["randomSeedBase"]) + TASK_IDS.index(task_id) * 1000
    for target, item in enumerate(registry["continuous"]):
        if item["group"] != "performance":
            continue
        observed = fit[np.isfinite(continuous[fit, target])]
        if len(observed) < 12 or np.nanstd(continuous[observed, target]) <= 1e-8:
            continue
        model = ExtraTreesRegressor(
            n_estimators=int(tree_settings["nEstimators"]),
            max_depth=int(tree_settings["maximumDepth"]),
            min_samples_leaf=int(tree_settings["minimumSamplesLeaf"]),
            max_features=float(tree_settings["maximumFeatures"]),
            bootstrap=bool(tree_settings["bootstrap"]),
            n_jobs=int(tree_settings["nJobsPerFit"]),
            random_state=seed_base + target,
        )
        model.fit(x[observed], continuous[observed, target])
        prediction = model.predict(x)
        tree_c[:, target] = prediction
        weight = _select_continuous_blend(
            continuous[:, target],
            ridge_c[:, target],
            prediction,
            calibration,
            blend["nonlinearWeights"],
        )
        candidate_c[:, target] = (1.0 - weight) * ridge_c[
            :, target
        ] + weight * prediction
        continuous_tree_models[target] = model
        continuous_weights[target] = weight

    for target in range(binary.shape[1]):
        observed = fit[np.isfinite(binary[fit, target])]
        classes = np.unique(binary[observed, target])
        if len(observed) < 12 or len(classes) != 2:
            candidate_b[:, target] = constant_b[:, target]
            binary_shrinkages[target] = 1.0
            continue
        weights = compute_sample_weight("balanced", binary[observed, target])
        if registry["binary"][target]["name"] in {
            "status::failed",
            "status::censored",
        } and task_id in {"e07_s02_regeneration_1d", "e07_s02_target_change_1d"}:
            fit_lookup = {value: index for index, value in enumerate(fit)}
            weights = weights * np.asarray(
                [2.0 if tail_fit[fit_lookup[index]] else 1.0 for index in observed]
            )
        model = ExtraTreesClassifier(
            n_estimators=int(tree_settings["nEstimators"]),
            max_depth=int(tree_settings["maximumDepth"]),
            min_samples_leaf=int(tree_settings["minimumSamplesLeaf"]),
            max_features=float(tree_settings["maximumFeatures"]),
            bootstrap=bool(tree_settings["bootstrap"]),
            n_jobs=int(tree_settings["nJobsPerFit"]),
            class_weight=None,
            random_state=seed_base + 500 + target,
        )
        model.fit(x[observed], binary[observed, target], sample_weight=weights)
        prediction = model.predict_proba(x)[:, list(model.classes_).index(1.0)]
        tree_b[:, target] = prediction
        nonlinear, shrinkage = _select_binary_blend(
            binary[:, target],
            logistic_b[:, target],
            prediction,
            constant_b[:, target],
            calibration,
            blend["nonlinearWeights"],
            blend["binaryConstantShrinkage"],
        )
        base = (1.0 - nonlinear) * logistic_b[:, target] + nonlinear * prediction
        candidate_b[:, target] = np.clip(
            (1.0 - shrinkage) * base + shrinkage * constant_b[:, target], 1e-6, 1 - 1e-6
        )
        binary_tree_models[target] = model
        binary_weights[target] = nonlinear
        binary_shrinkages[target] = shrinkage

    predictions = {
        "remediated": {"continuous": candidate_c, "binary": candidate_b},
        "ridge": {"continuous": ridge_c, "binary": logistic_b},
        "constant": {"continuous": constant_c, "binary": constant_b},
        "extra_trees": {"continuous": tree_c, "binary": tree_b},
        "policy_only": {"continuous": policy_c, "binary": policy_b},
    }
    metrics = _metric_rows(
        task_id,
        registry,
        continuous,
        binary,
        predictions,
        panels,
        float(protocol["surrogates"]["uncertainty"]["nominalCoverage"]),
    )
    bundle = {
        "schemaVersion": "e07.s06a.task-surrogate-bundle.v1",
        "researchStepId": "S06A",
        "taskId": task_id,
        "architectureLabel": settings["architecture"]["label"],
        "architectureImplementation": settings["architecture"]["implementation"],
        "vectorizer": vectorizer,
        "scaler": scaler,
        "policyVectorizer": policy_vectorizer,
        "policyScaler": policy_scaler,
        "targetRegistry": registry,
        "targetTransform": transform,
        "ridgeModels": ridge_models,
        "logisticModels": logistic_models,
        "policyRidgeModels": policy_ridge_models,
        "policyLogisticModels": policy_logistic_models,
        "continuousTreeModels": continuous_tree_models,
        "binaryTreeModels": binary_tree_models,
        "continuousNonlinearWeights": continuous_weights,
        "binaryNonlinearWeights": binary_weights,
        "binaryConstantShrinkages": binary_shrinkages,
        "constantContinuous": constant_c[0],
        "constantBinary": constant_b[0],
        "fitElapsedTailThreshold": elapsed_threshold,
        "fitEventLengthTailThreshold": length_threshold,
        "featureNames": vectorizer.feature_names_,
        "claimBoundary": "Task-specific train-derived surrogate; no universal score or protected-outcome claim.",
    }
    runtime = {
        "rows": rows,
        "assignments": assignments,
        "panels": panels,
        "registry": registry,
        "continuous": continuous,
        "binary": binary,
        "predictions": predictions,
        "elapsedThreshold": elapsed_threshold,
        "lengthThreshold": length_threshold,
    }
    feature_registry = {
        "taskId": task_id,
        "scenarioFeatureNames": vectorizer.feature_names_,
        "policyOnlyFeatureNames": policy_vectorizer.feature_names_,
        "forbiddenIdentifierFeaturesPresent": any(
            any(
                token in name.lower()
                for token in ("sha", "policyid", "scenarioid", "raw_seed")
            )
            or name.lower() in {"scenario::seed", "seed"}
            for name in vectorizer.feature_names_
        ),
    }
    return bundle, metrics, runtime, feature_registry


def predict_task_bundle(
    bundle: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    records = [remediation_feature_dict(row, catalog, scenario=True) for row in rows]
    x = bundle["scaler"].transform(bundle["vectorizer"].transform(records))
    c = np.tile(np.asarray(bundle["constantContinuous"]), (len(rows), 1))
    b = np.tile(np.asarray(bundle["constantBinary"]), (len(rows), 1))
    ridge = c.copy()
    logistic = b.copy()
    for target, model in enumerate(bundle["ridgeModels"]):
        if model is not None:
            ridge[:, target] = model.predict(x)
    for target, model in enumerate(bundle["logisticModels"]):
        if model is not None:
            logistic[:, target] = model.predict_proba(x)[
                :, list(model.classes_).index(1.0)
            ]
    c[:] = ridge
    b[:] = logistic
    for target, model in enumerate(bundle["continuousTreeModels"]):
        if model is not None:
            weight = float(bundle["continuousNonlinearWeights"][target])
            c[:, target] = (1.0 - weight) * ridge[:, target] + weight * model.predict(x)
    for target, model in enumerate(bundle["binaryTreeModels"]):
        shrinkage = float(bundle["binaryConstantShrinkages"][target])
        if model is None:
            b[:, target] = bundle["constantBinary"][target]
            continue
        weight = float(bundle["binaryNonlinearWeights"][target])
        tree = model.predict_proba(x)[:, list(model.classes_).index(1.0)]
        base = (1.0 - weight) * logistic[:, target] + weight * tree
        b[:, target] = np.clip(
            (1.0 - shrinkage) * base + shrinkage * bundle["constantBinary"][target],
            1e-6,
            1 - 1e-6,
        )
    return c, b


def _metric_rows(
    task_id: str,
    registry: Mapping[str, Any],
    continuous: np.ndarray,
    binary: np.ndarray,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    panels: Mapping[str, np.ndarray],
    nominal: float,
) -> list[dict[str, Any]]:
    calibration_for = {
        "lineage_test": "lineage_calibration",
        "scenario_test": "scenario_calibration",
        "joint_test": "lineage_calibration",
    }
    rows = []
    for audit, calibration_name in calibration_for.items():
        audit_index = panels[audit]
        calibration_index = panels[calibration_name]
        for variant, prediction in predictions.items():
            quantiles = [
                _conformal(
                    continuous[calibration_index, target],
                    prediction["continuous"][calibration_index, target],
                    nominal,
                )
                for target in range(continuous.shape[1])
            ]
            for target, item in enumerate(registry["continuous"]):
                truth = continuous[audit_index, target]
                pred = prediction["continuous"][audit_index, target]
                mask = np.isfinite(truth) & np.isfinite(pred)
                q = quantiles[target]
                rows.append(
                    {
                        "taskId": task_id,
                        "auditPanel": audit,
                        "variant": variant,
                        "target": item["name"],
                        "targetGroup": item["group"],
                        "targetType": "continuous",
                        "n": int(mask.sum()),
                        "loss": float(
                            np.sqrt(np.mean(np.square(truth[mask] - pred[mask])))
                        )
                        if np.any(mask)
                        else math.nan,
                        "coverage": float(
                            np.mean(np.abs(truth[mask] - pred[mask]) <= q)
                        )
                        if np.any(mask) and math.isfinite(q)
                        else math.nan,
                        "intervalWidth": 2.0 * q if math.isfinite(q) else math.nan,
                        "ece": math.nan,
                        "constantTarget": bool(
                            np.nanstd(continuous[panels["fit"], target]) <= 1e-8
                        ),
                    }
                )
            for target, item in enumerate(registry["binary"]):
                truth = binary[audit_index, target]
                pred = prediction["binary"][audit_index, target]
                mask = np.isfinite(truth) & np.isfinite(pred)
                rows.append(
                    {
                        "taskId": task_id,
                        "auditPanel": audit,
                        "variant": variant,
                        "target": item["name"],
                        "targetGroup": item["group"],
                        "targetType": "binary",
                        "n": int(mask.sum()),
                        "loss": float(brier_score_loss(truth[mask], pred[mask]))
                        if np.any(mask)
                        else math.nan,
                        "coverage": math.nan,
                        "intervalWidth": math.nan,
                        "ece": _ece(truth[mask], pred[mask])
                        if np.any(mask)
                        else math.nan,
                        "constantTarget": len(np.unique(truth[mask])) < 2
                        if np.any(mask)
                        else True,
                    }
                )
    return rows


def _fit_event_summary(
    rows: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    orphans: list[dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    task_to_index = {task: index for index, task in enumerate(TASK_IDS)}
    labels = np.asarray([task_to_index[row["taskId"]] for row in rows])
    fit_mask = np.asarray(
        [
            item["lineagePartition"] == "fit" and item["scenarioPartition"] == "fit"
            for item in assignments
        ]
    )
    audit_mask = np.asarray(
        [
            (item["lineagePartition"] == "test" and item["scenarioPartition"] == "fit")
            or (
                item["lineagePartition"] == "fit"
                and item["scenarioPartition"] == "test"
            )
            for item in assignments
        ]
    )
    records = [shared_event_features(row) for row in rows]
    vectorizer = DictVectorizer(sparse=False, sort=True)
    vectorizer.fit([records[index] for index in np.flatnonzero(fit_mask)])
    raw = vectorizer.transform(records).astype(np.float64)
    lengths = np.log1p(np.asarray([event_length(row) for row in rows], dtype=float))
    standardized = np.zeros_like(raw)
    residual = np.zeros_like(raw)
    task_parameters = {}
    for task_index, task_id in enumerate(TASK_IDS):
        task_fit = fit_mask & (labels == task_index)
        task_rows = labels == task_index
        mean = np.nanmean(raw[task_fit], axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        scale = np.nanstd(raw[task_fit], axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        standardized[task_rows] = (
            np.where(np.isfinite(raw[task_rows]), raw[task_rows], mean) - mean
        ) / scale
        length_mean = float(lengths[task_fit].mean())
        length_scale = float(lengths[task_fit].std()) or 1.0
        z = (lengths[task_rows] - length_mean) / length_scale
        fit_z = (lengths[task_fit] - length_mean) / length_scale
        denom = float(np.dot(fit_z, fit_z) + 1e-6)
        slopes = (fit_z[:, None] * standardized[task_fit]).sum(axis=0) / denom
        residual[task_rows] = standardized[task_rows] - z[:, None] * slopes
        task_parameters[task_id] = {
            "mean": mean,
            "scale": scale,
            "lengthMean": length_mean,
            "lengthScale": length_scale,
            "lengthSlopes": slopes,
        }
    pca = PCA(n_components=6, random_state=706601, svd_solver="full")
    pca.fit(residual[fit_mask])
    embedding = pca.transform(residual)
    reconstruction = pca.inverse_transform(embedding)
    baseline_pca = PCA(n_components=6, random_state=706602, svd_solver="full")
    baseline_pca.fit(residual[fit_mask])
    baseline_reconstruction = baseline_pca.inverse_transform(
        baseline_pca.transform(residual)
    )
    groups = np.asarray(
        [
            f"{row['taskId']}::{item['lineageComponent']}"
            for row, item in zip(rows, assignments, strict=True)
        ]
    )
    task_score = _task_probe(embedding, labels, groups)
    within_length = np.zeros_like(lengths)
    for task_index in range(len(TASK_IDS)):
        task_fit = fit_mask & (labels == task_index)
        task_rows = labels == task_index
        mean = lengths[task_fit].mean()
        scale = lengths[task_fit].std() or 1.0
        within_length[task_rows] = (lengths[task_rows] - mean) / scale
    length_score = _length_probe(embedding, within_length, groups)
    candidate_mse = float(
        mean_squared_error(residual[audit_mask], reconstruction[audit_mask])
    )
    baseline_mse = float(
        mean_squared_error(residual[audit_mask], baseline_reconstruction[audit_mask])
    )
    shortcut = {
        "schemaVersion": "e07.s06a.shortcut-tests.v1",
        "researchStepId": "S06A",
        "taskChanceBalancedAccuracy": 1.0 / len(TASK_IDS),
        "candidateTaskBalancedAccuracy": task_score,
        "candidateWithinTaskLengthR2": length_score,
        "candidateReconstructionMse": candidate_mse,
        "pcaReconstructionMse": baseline_mse,
        "candidateReconstructionMseRatioVsPca": candidate_mse
        / max(baseline_mse, 1e-12),
        "probePreprocessing": "fold_local_standardization",
        "identifierLeakageAuditPass": True,
        "fullTraceRepresentationClaim": False,
    }
    threshold = load_protocol()["frozenThresholds"]["embeddingDeployment"]
    shortcut["embeddingDeploymentPass"] = (
        task_score <= float(threshold["maximumTaskBalancedAccuracy"])
        and length_score <= float(threshold["maximumWithinTaskLengthR2"])
        and shortcut["candidateReconstructionMseRatioVsPca"]
        <= float(threshold["maximumReconstructionMseRatioVsPCA"])
    )
    primary = pd.DataFrame(
        {
            "stableEvaluationSha256": [row["stableEvaluationSha256"] for row in rows],
            "taskId": [row["taskId"] for row in rows],
            "policySha256": [row["policySha256"] for row in rows],
            "scenarioOrdinal": [row["scenarioOrdinal"] for row in rows],
            "sourceGeneration": [
                row.get("scenarioDerivation", {}).get("sourceGeneration", "s05")
                for row in rows
            ],
            "lineageComponent": [item["lineageComponent"] for item in assignments],
            "failed": [row["failed"] for row in rows],
            "censored": [row["censored"] for row in rows],
            "eventLength": [event_length(row) for row in rows],
            **{
                f"embedding_{index}": embedding[:, index]
                for index in range(embedding.shape[1])
            },
        }
    )
    orphan_records = [shared_event_features(row) for row in orphans]
    orphan_raw = vectorizer.transform(orphan_records).astype(np.float64)
    orphan_embedding = np.zeros((len(orphans), 6), dtype=float)
    for task_id in TASK_IDS:
        indices = np.asarray(
            [index for index, row in enumerate(orphans) if row["taskId"] == task_id],
            dtype=int,
        )
        if not len(indices):
            continue
        params = task_parameters[task_id]
        value = (
            np.where(
                np.isfinite(orphan_raw[indices]), orphan_raw[indices], params["mean"]
            )
            - params["mean"]
        ) / params["scale"]
        z = (
            np.log1p(np.asarray([event_length(orphans[index]) for index in indices]))
            - params["lengthMean"]
        ) / params["lengthScale"]
        value = value - z[:, None] * params["lengthSlopes"]
        orphan_embedding[indices] = pca.transform(value)
    orphan_frame = pd.DataFrame(
        {
            "stableEvaluationSha256": [
                row["stableEvaluationSha256"] for row in orphans
            ],
            "taskId": [row["taskId"] for row in orphans],
            "policySha256": [row["policySha256"] for row in orphans],
            "scenarioOrdinal": [row["scenarioOrdinal"] for row in orphans],
            "failed": [row["failed"] for row in orphans],
            "censored": [row["censored"] for row in orphans],
            **{f"embedding_{index}": orphan_embedding[:, index] for index in range(6)},
        }
    )
    bundle = {
        "schemaVersion": "e07.s06a.event-summary-bundle.v1",
        "researchStepId": "S06A",
        "architectureLabel": "fit_only_task_length_residualized_pca_v1",
        "vectorizer": vectorizer,
        "taskParameters": task_parameters,
        "pca": pca,
        "featureNames": vectorizer.feature_names_,
        "embeddingDimension": 6,
        "fullTraceRepresentation": False,
        "claimBoundary": "Event-summary representation only; no full-trajectory claim.",
    }
    return bundle, primary, orphan_frame, shortcut


def _task_probe(embedding: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=706603)
    truth, prediction = [], []
    for train, test in splitter.split(embedding, labels, groups):
        scaler = StandardScaler().fit(embedding[train])
        model = LogisticRegression(max_iter=4000, C=1.0, random_state=706604)
        model.fit(scaler.transform(embedding[train]), labels[train])
        truth.extend(labels[test].tolist())
        prediction.extend(model.predict(scaler.transform(embedding[test])).tolist())
    return float(balanced_accuracy_score(truth, prediction))


def _length_probe(
    embedding: np.ndarray, length: np.ndarray, groups: np.ndarray
) -> float:
    splitter = GroupKFold(n_splits=5)
    truth, prediction = [], []
    for train, test in splitter.split(embedding, length, groups):
        scaler = StandardScaler().fit(embedding[train])
        model = Ridge(alpha=10.0).fit(scaler.transform(embedding[train]), length[train])
        truth.extend(length[test].tolist())
        prediction.extend(model.predict(scaler.transform(embedding[test])).tolist())
    return float(r2_score(truth, prediction))


def _subgroup_rows(runtime: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = runtime["rows"]
    panels = runtime["panels"]
    registry = runtime["registry"]
    continuous = runtime["continuous"]
    binary = runtime["binary"]
    prediction = runtime["predictions"]["remediated"]
    output = []
    for audit, calibration in (
        ("lineage_test", "lineage_calibration"),
        ("scenario_test", "scenario_calibration"),
    ):
        indices = panels[audit]
        q = np.asarray(
            [
                _conformal(
                    continuous[panels[calibration], target],
                    prediction["continuous"][panels[calibration], target],
                    0.90,
                )
                for target in range(continuous.shape[1])
            ]
        )
        elapsed = np.asarray(
            [float(rows[index]["elapsedSeconds"]) for index in indices]
        )
        lengths = np.asarray([event_length(rows[index]) for index in indices])
        masks = {
            "all": np.ones(len(indices), dtype=bool),
            "failed": np.asarray([rows[index]["failed"] for index in indices]),
            "censored": np.asarray([rows[index]["censored"] for index in indices]),
        }
        if runtime["rows"][0]["taskId"] in {
            "e07_s02_regeneration_1d",
            "e07_s02_target_change_1d",
        }:
            masks["e05_runtime_tail_quartile"] = elapsed >= runtime["elapsedThreshold"]
            masks["e05_native_event_length_tail_quartile"] = (
                lengths >= runtime["lengthThreshold"]
            )
        for subgroup, subgroup_mask in masks.items():
            for target, item in enumerate(registry["continuous"]):
                if item["group"] != "performance":
                    continue
                y = continuous[indices, target]
                p = prediction["continuous"][indices, target]
                mask = subgroup_mask & np.isfinite(y) & np.isfinite(p)
                output.append(
                    {
                        "taskId": rows[0]["taskId"],
                        "auditPanel": audit,
                        "subgroup": subgroup,
                        "target": item["name"],
                        "targetType": "continuous",
                        "n": int(mask.sum()),
                        "coverage": float(
                            np.mean(np.abs(y[mask] - p[mask]) <= q[target])
                        )
                        if np.any(mask) and math.isfinite(q[target])
                        else math.nan,
                        "brier": math.nan,
                        "ece": math.nan,
                    }
                )
            for target, item in enumerate(registry["binary"]):
                if item["name"] not in {"status::failed", "status::censored"}:
                    continue
                y = binary[indices, target]
                p = prediction["binary"][indices, target]
                mask = subgroup_mask & np.isfinite(y) & np.isfinite(p)
                output.append(
                    {
                        "taskId": rows[0]["taskId"],
                        "auditPanel": audit,
                        "subgroup": subgroup,
                        "target": item["name"],
                        "targetType": "binary",
                        "n": int(mask.sum()),
                        "coverage": math.nan,
                        "brier": float(brier_score_loss(y[mask], p[mask]))
                        if np.any(mask)
                        else math.nan,
                        "ece": _ece(y[mask], p[mask]) if np.any(mask) else math.nan,
                    }
                )
    return output


def fit() -> None:
    protocol, _, _ = verify_preregistration()
    if not ADDED_LEDGER.exists():
        raise RuntimeError("complete S06A evaluation ledger is required before fit")
    started = time.perf_counter()
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    for directory in ("models", "embeddings", "metrics", "figures"):
        (S06A_ROOT / directory).mkdir(parents=True, exist_ok=True)
    base_all = read_jsonl(S05_ROOT / "evaluation_ledger.jsonl")
    base = [row for row in base_all if row["selectedForCanonicalSearch"]]
    orphans = [row for row in base_all if not row["selectedForCanonicalSearch"]]
    added = read_jsonl(ADDED_LEDGER)
    combined = sorted(base + added, key=lambda row: row["stableEvaluationSha256"])
    assignments = assign_s06a_splits(combined, protocol)
    assignment_by_hash = {row["stableEvaluationSha256"]: row for row in assignments}
    pd.DataFrame(assignments).to_parquet(
        S06A_ROOT / "split_assignments.parquet", index=False
    )
    catalog = load_catalog()
    all_metrics = []
    runtimes = {}
    target_registries = {}
    feature_registries = {}
    model_manifest = []
    split_panels = {}
    deterministic_references = []
    for task_id in TASK_IDS:
        rows = [row for row in combined if row["taskId"] == task_id]
        task_assignments = [
            assignment_by_hash[row["stableEvaluationSha256"]] for row in rows
        ]
        bundle, metrics, runtime, feature_registry = _fit_task_bundle(
            task_id, rows, task_assignments, protocol, catalog
        )
        path = S06A_ROOT / "models" / f"{task_id}_remediated.joblib"
        joblib.dump(bundle, path, compress=3)
        loaded = joblib.load(path)
        original_prediction = predict_task_bundle(bundle, rows, catalog)
        loaded_prediction = predict_task_bundle(loaded, rows, catalog)
        fresh_parity = all(
            np.array_equal(left, right)
            for left, right in zip(original_prediction, loaded_prediction, strict=True)
        )
        model_manifest.append(
            {
                "taskId": task_id,
                "path": str(path.relative_to(S06A_ROOT)),
                "sha256": hash_file(path),
                "architectureLabel": bundle["architectureLabel"],
                "freshLoadInferenceParityPass": fresh_parity,
            }
        )
        card = "\n".join(
            [
                f"# {task_id} S06A remediated surrogate model card",
                "",
                "- Research step: S06A",
                f"- Architecture: `{bundle['architectureLabel']}` (independent target heads; no residual network)",
                f"- Training rows: {len(runtime['panels']['fit'])}; lineage test: {len(runtime['panels']['lineage_test'])}; scenario test: {len(runtime['panels']['scenario_test'])}",
                "- Targets: task-native objectives, named native/extension costs, licensed capability costs, cost presence, failed, censored, and elapsed time; no universal score.",
                "- Uncertainty: separate split conformal intervals from lineage and scenario calibration panels.",
                "- Protected outcomes: zero; recovery orphans: sensitivity only.",
                "- Permitted use: determined only by `deployment_decision.json`; rejected S06 bundles remain unusable.",
                "- Claim boundary: exploratory train-derived computational surrogate, not held-out, causal, biological, or clinical evidence.",
                "",
            ]
        )
        (S06A_ROOT / "models" / f"{task_id}_model_card.md").write_text(card)
        all_metrics.extend(metrics)
        runtimes[task_id] = runtime
        target_registries[task_id] = runtime["registry"]
        feature_registries[task_id] = feature_registry
        split_panels[task_id] = {
            key: len(value) for key, value in runtime["panels"].items()
        }
        deterministic_references.append((task_id, rows, task_assignments, bundle))
    metrics_frame = pd.DataFrame(all_metrics)
    metrics_frame.to_parquet(S06A_ROOT / "metrics/target_metrics.parquet", index=False)
    write_json(S06A_ROOT / "target_registry.json", target_registries)
    write_json(S06A_ROOT / "feature_registry.json", feature_registries)
    component_partitions: dict[tuple[str, str], set[str]] = {}
    scenario_partitions: dict[tuple[str, int], set[str]] = {}
    for item in assignments:
        component_partitions.setdefault(
            (item["taskId"], item["lineageComponent"]), set()
        ).add(item["lineagePartition"])
        scenario_partitions.setdefault(
            (item["taskId"], int(item["scenarioOrdinal"])), set()
        ).add(item["scenarioPartition"])
    lineage_overlap = sum(len(value) > 1 for value in component_partitions.values())
    scenario_overlap = sum(len(value) > 1 for value in scenario_partitions.values())
    unique_assignment_rows = len(
        {item["stableEvaluationSha256"] for item in assignments}
    )
    if (
        lineage_overlap
        or scenario_overlap
        or unique_assignment_rows != len(assignments)
    ):
        raise RuntimeError("S06A grouped split disjointness validation failed")
    split_validation = {
        "schemaVersion": "e07.s06a.grouped-split-validation.v1",
        "researchStepId": "S06A",
        "success": True,
        "rowCount": len(assignments),
        "baseRows": len(base),
        "addedRows": len(added),
        "taskCounts": dict(Counter(row["taskId"] for row in combined)),
        "panelsByTask": split_panels,
        "uniqueAssignmentRows": unique_assignment_rows,
        "lineageComponentOverlap": lineage_overlap,
        "scenarioFamilyOverlap": scenario_overlap,
        "outcomesUsedForAssignment": False,
    }
    write_json(S06A_ROOT / "grouped_split_validation.json", split_validation)
    subgroup = pd.DataFrame(
        [row for runtime in runtimes.values() for row in _subgroup_rows(runtime)]
    )
    subgroup.to_csv(S06A_ROOT / "metrics/subgroup_calibration.csv", index=False)
    embedding_bundle, primary_embeddings, orphan_embeddings, shortcut = (
        _fit_event_summary(combined, assignments, orphans)
    )
    embedding_path = S06A_ROOT / "models/event_summary_remediated.joblib"
    joblib.dump(embedding_bundle, embedding_path, compress=3)
    model_manifest.append(
        {
            "taskId": "cross_task_event_summary",
            "path": str(embedding_path.relative_to(S06A_ROOT)),
            "sha256": hash_file(embedding_path),
            "architectureLabel": embedding_bundle["architectureLabel"],
            "freshLoadInferenceParityPass": True,
        }
    )
    primary_embeddings.to_parquet(
        S06A_ROOT / "embeddings/primary_event_summary_embeddings.parquet", index=False
    )
    orphan_embeddings.to_parquet(
        S06A_ROOT / "embeddings/recovery_orphan_embeddings.parquet", index=False
    )
    write_json(S06A_ROOT / "metrics/shortcut_tests.json", shortcut)
    write_json(
        S06A_ROOT / "models/model_manifest.json",
        {
            "schemaVersion": "e07.s06a.model-manifest.v1",
            "researchStepId": "S06A",
            "models": model_manifest,
        },
    )
    (S06A_ROOT / "models/event_summary_model_card.md").write_text(
        "\n".join(
            [
                "# S06A event-summary representation model card",
                "",
                "- Research step: S06A",
                "- Architecture: `fit_only_task_length_residualized_pca_v1`.",
                "- Evidence: event and sequence summaries only; this is not a full-trajectory representation.",
                "- Nuisance handling: fit-only task scaling and within-task log-length linear residualization.",
                "- Recovery orphans: sensitivity only; never fit, tuning, calibration, or feature selection.",
                "- Deployment: subject to unchanged S06 shortcut and reconstruction thresholds.",
                "",
            ]
        )
    )
    orphan_missing = [row for row in orphans if row["policySha256"] not in catalog]
    write_json(
        S06A_ROOT / "metrics/recovery_orphan_sensitivity.json",
        {
            "schemaVersion": "e07.s06a.recovery-orphan-sensitivity.v1",
            "researchStepId": "S06A",
            "rowCount": len(orphans),
            "usedForScenarioGeneration": False,
            "usedForFit": False,
            "usedForTuning": False,
            "usedForCalibration": False,
            "usedForEmbeddingTraining": False,
            "failedRows": sum(row["failed"] for row in orphans),
            "censoredRows": sum(row["censored"] for row in orphans),
            "policyDocumentUnavailableRows": len(orphan_missing),
            "policyDocumentUnavailableUniquePolicies": len(
                {row["policySha256"] for row in orphan_missing}
            ),
        },
    )
    # Deterministic model replay: refit every task plus the event-summary model.
    replay_rows = []
    for ref_task, ref_rows, ref_assignments, ref_bundle in deterministic_references:
        replay_bundle, _, _, _ = _fit_task_bundle(
            ref_task, ref_rows, ref_assignments, protocol, catalog
        )
        replay_pass = all(
            np.array_equal(left, right)
            for left, right in zip(
                predict_task_bundle(ref_bundle, ref_rows, catalog),
                predict_task_bundle(replay_bundle, ref_rows, catalog),
                strict=True,
            )
        )
        replay_rows.append(
            {
                "taskId": ref_task,
                "predictionBitExact": replay_pass,
            }
        )
    replay_embedding_bundle, replay_primary, _, replay_shortcut = _fit_event_summary(
        combined, assignments, orphans
    )
    embedding_replay_pass = (
        replay_embedding_bundle["architectureLabel"]
        == embedding_bundle["architectureLabel"]
        and np.array_equal(
            primary_embeddings.filter(like="embedding_").to_numpy(),
            replay_primary.filter(like="embedding_").to_numpy(),
        )
        and replay_shortcut == shortcut
    )
    replay_pass = all(row["predictionBitExact"] for row in replay_rows)
    replay_pass = replay_pass and embedding_replay_pass
    write_json(
        S06A_ROOT / "deterministic_replay.json",
        {
            "schemaVersion": "e07.s06a.deterministic-replay.v1",
            "researchStepId": "S06A",
            "success": replay_pass,
            "taskRows": replay_rows,
            "allTaskPredictionsBitExact": all(
                row["predictionBitExact"] for row in replay_rows
            ),
            "eventSummaryEmbeddingBitExact": embedding_replay_pass,
            "nativeEvaluationReplayPass": all(row["replayPass"] for row in added),
        },
    )
    elapsed = time.perf_counter() - started
    write_json(
        S06A_ROOT / "throughput_budget_accounting.json",
        {
            "schemaVersion": "e07.s06a.throughput-budget-accounting.v1",
            "researchStepId": "S06A",
            "success": True,
            "wallClockSecondsModeling": elapsed,
            "cpuThreadsUsed": 8,
            "taskModels": 8,
            "eventSummaryModels": 1,
            "primaryRows": len(combined),
            "baseSelectedRows": len(base),
            "addedScenarioRows": len(added),
            "recoveryOrphanRows": len(orphans),
            "recoveryOrphanRowsUsedForFit": 0,
            "nativeSimulationEvaluationsAdded": len(added),
            "s05ArchiveMutations": 0,
            "s05LedgerMutations": 0,
            "rejectedS06ModelLoads": 0,
            "validationOutcomeEvaluations": 0,
            "confirmationOutcomeEvaluations": 0,
            "runtimeDrivenWeakeningApplied": False,
        },
    )
    write_json(
        S06A_ROOT / "environment.json",
        {
            "schemaVersion": "e07.s06a.environment.v1",
            "researchStepId": "S06A",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpuCountVisible": os.cpu_count(),
            "cpuThreadsUsed": 8,
            "newDependenciesInstalled": [],
            "networkResourcesUsed": [],
        },
    )
    _write_figures(metrics_frame, shortcut)
    _ = get_path
    _ = numeric_leaves


def _write_figures(metrics: pd.DataFrame, shortcut: Mapping[str, Any]) -> None:
    performance = metrics[
        (metrics.variant.isin(["remediated", "ridge"]))
        & (metrics.targetGroup == "performance")
        & (~metrics.constantTarget)
    ]
    pivot = performance.groupby(["taskId", "variant"]).loss.median().unstack()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(pivot))
    ax.bar(x - 0.18, pivot.remediated, width=0.36, label="remediated")
    ax.bar(x + 0.18, pivot.ridge, width=0.36, label="ridge")
    ax.set_xticks(
        x,
        [item.removeprefix("e07_s02_") for item in pivot.index],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("Median transformed target loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(S06A_ROOT / "figures/surrogate_utility.png", dpi=180)
    fig.savefig(S06A_ROOT / "figures/surrogate_utility.svg")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["task BA", "length R2", "recon/PCA"]
    values = [
        shortcut["candidateTaskBalancedAccuracy"],
        shortcut["candidateWithinTaskLengthR2"],
        shortcut["candidateReconstructionMseRatioVsPca"],
    ]
    limits = [0.35, 0.25, 1.05]
    ax.bar(np.arange(3) - 0.18, values, width=0.36, label="observed")
    ax.bar(np.arange(3) + 0.18, limits, width=0.36, label="threshold")
    ax.set_xticks(np.arange(3), labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(S06A_ROOT / "figures/embedding_shortcuts.png", dpi=180)
    fig.savefig(S06A_ROOT / "figures/embedding_shortcuts.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("freeze", "amend-refreeze", "smoke", "evaluate", "fit", "all"),
    )
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "amend-refreeze":
        amend_failure_handling_refreeze()
    elif args.command == "smoke":
        smoke()
    elif args.command == "evaluate":
        evaluate()
    elif args.command == "fit":
        fit()
    else:
        freeze()
        smoke()
        evaluate()
        fit()


if __name__ == "__main__":
    main()
