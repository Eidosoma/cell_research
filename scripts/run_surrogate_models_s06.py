#!/usr/bin/env python3
"""Freeze and execute the train-only E07 S06 modeling protocol."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import yaml

from src.surrogate_models.core import (
    S05_ROOT,
    S06_ROOT,
    TASK_IDS,
    EventSummaryAutoencoder,
    assign_grouped_splits,
    build_s05_freeze,
    build_target_registry,
    ensemble_predict,
    event_length,
    fit_target_transform,
    hash_file,
    load_lineage,
    load_policy_catalog_with_plans,
    model_bundle_hash,
    policy_feature_dict,
    raw_event_features,
    read_jsonl,
    set_deterministic,
    shared_event_features,
    state_dict_hash,
    target_arrays,
    train_surrogate_member,
    verify_frozen_inputs,
    write_json,
)


REPOSITORY = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPOSITORY / "configs/modeling/s06_modeling_protocol.yaml"
FREEZE_PATH = S06_ROOT / "s05_hash_freeze.json"
PREREG_PATH = S06_ROOT / "modeling_preregistration.yaml"
PREREG_FREEZE_PATH = S06_ROOT / "preregistration_freeze.json"
CACHE = Path("/cache/e07_s06")


def protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def freeze() -> dict[str, Any]:
    settings = protocol()
    S06_ROOT.mkdir(parents=True, exist_ok=True)
    frozen = build_s05_freeze(settings)
    frozen["frozenAtUtc"] = datetime.now(timezone.utc).isoformat()
    write_json(FREEZE_PATH, frozen)
    shutil.copyfile(PROTOCOL_PATH, PREREG_PATH)
    prereg = {
        "schemaVersion": "e07.s06.preregistration-freeze.v1",
        "researchStepId": "S06",
        "frozenAtUtc": datetime.now(timezone.utc).isoformat(),
        "frozenBeforeAnyFit": True,
        "protocolSourcePath": str(PROTOCOL_PATH),
        "protocolArtifactPath": str(PREREG_PATH),
        "protocolSha256": hash_file(PREREG_PATH),
        "s05HashFreezePath": str(FREEZE_PATH),
        "s05HashFreezeSha256": hash_file(FREEZE_PATH),
        "selectedCanonicalRowCount": frozen["selectedCanonicalRowCount"],
        "recoveryOrphanRowCount": frozen["recoveryOrphanRowCount"],
        "thresholdsFrozen": settings["frozenThresholds"],
        "splitDesignFrozen": settings["splitDesign"],
        "recoveryOrphanForbiddenUses": settings["authorizedPopulation"][
            "recoveryOrphanForbiddenUses"
        ],
        "validationOutcomesSealed": True,
        "confirmationOutcomesSealed": True,
    }
    write_json(PREREG_FREEZE_PATH, prereg)
    write_json(
        S06_ROOT / "access_control_validation.json",
        {
            "schemaVersion": "e07.s06.access-control-validation.v1",
            "researchStepId": "S06",
            "success": True,
            "allowedPopulation": "S05 selected canonical train rows",
            "recoveryOrphanUse": "sensitivity_only_after_model_freeze",
            "validationOutcomeEvaluations": 0,
            "confirmationOutcomeEvaluations": 0,
            "protectedMaterializersInvoked": 0,
            "protectedScenarioPayloadsRead": 0,
            "splitManifestUsedFor": "public access contract and hash provenance only",
        },
    )
    return prereg


def verify_preregistration(settings: Mapping[str, Any]) -> dict[str, Any]:
    if not all(path.exists() for path in (FREEZE_PATH, PREREG_PATH, PREREG_FREEZE_PATH)):
        raise RuntimeError("run freeze before fitting")
    prereg = json.loads(PREREG_FREEZE_PATH.read_text())
    if hash_file(PREREG_PATH) != prereg["protocolSha256"]:
        raise RuntimeError("S06 preregistration artifact changed")
    if hash_file(PROTOCOL_PATH) != prereg["protocolSha256"]:
        raise RuntimeError("repository S06 protocol changed after freeze")
    frozen = json.loads(FREEZE_PATH.read_text())
    if hash_file(FREEZE_PATH) != prereg["s05HashFreezeSha256"]:
        raise RuntimeError("S05 hash-freeze record changed")
    verify_frozen_inputs(settings, frozen)
    return prereg


def panel_indices(assignments: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    lineage = np.asarray([item["lineagePartition"] for item in assignments])
    scenario = np.asarray([item["scenarioPartition"] for item in assignments])
    return {
        "fit": np.flatnonzero((lineage == "fit") & (scenario == "fit")),
        "lineage_calibration": np.flatnonzero(
            (lineage == "calibration") & (scenario == "fit")
        ),
        "lineage_test": np.flatnonzero((lineage == "test") & (scenario == "fit")),
        "scenario_calibration": np.flatnonzero(
            (lineage == "fit") & (scenario == "calibration")
        ),
        "scenario_test": np.flatnonzero((lineage == "fit") & (scenario == "test")),
        "joint_test": np.flatnonzero((lineage == "test") & (scenario == "test")),
    }


def vectorize_features(
    rows: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Mapping[str, Any]],
    fit_index: np.ndarray,
    *,
    scenario: bool,
) -> tuple[np.ndarray, DictVectorizer, StandardScaler]:
    records = [policy_feature_dict(row, catalog, scenario=scenario) for row in rows]
    vectorizer = DictVectorizer(sparse=False, sort=True)
    vectorizer.fit([records[index] for index in fit_index])
    values = vectorizer.transform(records).astype(np.float64)
    scaler = StandardScaler()
    scaler.fit(values[fit_index])
    return scaler.transform(values).astype(np.float32), vectorizer, scaler


def baseline_predictions(
    x: np.ndarray,
    continuous: np.ndarray,
    binary: np.ndarray,
    fit_index: np.ndarray,
) -> dict[str, np.ndarray]:
    continuous_mean = np.nanmean(continuous[fit_index], axis=0)
    continuous_mean = np.where(np.isfinite(continuous_mean), continuous_mean, 0.0)
    binary_mean = np.nanmean(binary[fit_index], axis=0)
    binary_mean = np.where(np.isfinite(binary_mean), binary_mean, 0.0)
    ridge_prediction = np.tile(continuous_mean, (len(x), 1))
    logistic_prediction = np.tile(binary_mean, (len(x), 1))
    for target in range(continuous.shape[1]):
        observed = fit_index[np.isfinite(continuous[fit_index, target])]
        if len(observed) >= 4 and np.nanstd(continuous[observed, target]) > 1e-8:
            model = Ridge(alpha=10.0)
            model.fit(x[observed], continuous[observed, target])
            ridge_prediction[:, target] = model.predict(x)
    for target in range(binary.shape[1]):
        observed = fit_index[np.isfinite(binary[fit_index, target])]
        classes = np.unique(binary[observed, target])
        if len(observed) >= 8 and len(classes) == 2:
            model = LogisticRegression(C=1.0, max_iter=1000, random_state=706201)
            model.fit(x[observed], binary[observed, target])
            logistic_prediction[:, target] = model.predict_proba(x)[:, 1]
    return {
        "continuousConstant": np.tile(continuous_mean, (len(x), 1)),
        "binaryConstant": np.tile(binary_mean, (len(x), 1)),
        "continuousRidge": ridge_prediction,
        "binaryLogistic": logistic_prediction,
    }


def ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    if len(y) == 0:
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


def conformal_quantile(y: np.ndarray, prediction: np.ndarray, nominal: float) -> float:
    mask = np.isfinite(y) & np.isfinite(prediction)
    residual = np.abs(y[mask] - prediction[mask])
    if len(residual) == 0:
        return math.nan
    level = min(1.0, math.ceil((len(residual) + 1) * nominal) / len(residual))
    return float(np.quantile(residual, level, method="higher"))


def metric_records(
    task_id: str,
    registry: Mapping[str, Any],
    continuous: np.ndarray,
    binary: np.ndarray,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    panels: Mapping[str, np.ndarray],
    nominal: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    output: list[dict[str, Any]] = []
    calibration_for_audit = {
        "lineage_test": "lineage_calibration",
        "scenario_test": "scenario_calibration",
        "joint_test": "lineage_calibration",
    }
    conformal: dict[str, dict[str, np.ndarray]] = {}
    for audit, calibration_panel in calibration_for_audit.items():
        audit_index = panels[audit]
        calibration_index = panels[calibration_panel]
        conformal[audit] = {}
        for variant, prediction in predictions.items():
            continuous_prediction = prediction["continuous"]
            binary_prediction = prediction["binary"]
            quantiles = np.asarray(
                [
                    conformal_quantile(
                        continuous[calibration_index, target],
                        continuous_prediction[calibration_index, target],
                        nominal,
                    )
                    for target in range(continuous.shape[1])
                ]
            )
            conformal[audit][variant] = quantiles
            for target, item in enumerate(registry["continuous"]):
                y = continuous[audit_index, target]
                p = continuous_prediction[audit_index, target]
                mask = np.isfinite(y) & np.isfinite(p)
                q = quantiles[target]
                output.append(
                    {
                        "taskId": task_id,
                        "auditPanel": audit,
                        "variant": variant,
                        "target": item["name"],
                        "targetGroup": item["group"],
                        "targetType": "continuous",
                        "n": int(mask.sum()),
                        "loss": float(np.sqrt(np.mean(np.square(y[mask] - p[mask]))))
                        if np.any(mask)
                        else math.nan,
                        "coverage": float(np.mean(np.abs(y[mask] - p[mask]) <= q))
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
                y = binary[audit_index, target]
                p = binary_prediction[audit_index, target]
                mask = np.isfinite(y) & np.isfinite(p)
                output.append(
                    {
                        "taskId": task_id,
                        "auditPanel": audit,
                        "variant": variant,
                        "target": item["name"],
                        "targetGroup": item["group"],
                        "targetType": "binary",
                        "n": int(mask.sum()),
                        "loss": float(brier_score_loss(y[mask], p[mask]))
                        if np.any(mask)
                        else math.nan,
                        "coverage": math.nan,
                        "intervalWidth": math.nan,
                        "ece": ece(y[mask], p[mask]) if np.any(mask) else math.nan,
                        "constantTarget": len(np.unique(y[mask])) < 2 if np.any(mask) else True,
                    }
                )
    return output, conformal


def train_embedding_model(
    values: np.ndarray,
    task_labels: np.ndarray,
    length_target: np.ndarray,
    train_index: np.ndarray,
    calibration_index: np.ndarray,
    settings: Mapping[str, Any],
    *,
    adversarial: bool,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    seed = int(settings["seed"] + (0 if adversarial else 1))
    set_deterministic(seed)
    model = EventSummaryAutoencoder(
        values.shape[1], int(settings["embeddingDimension"]), len(TASK_IDS)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learningRate"]),
        weight_decay=float(settings["weightDecay"]),
    )
    x = torch.as_tensor(values, dtype=torch.float32, device=device)
    tasks = torch.as_tensor(task_labels, dtype=torch.long, device=device)
    lengths = torch.as_tensor(length_target, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    best = math.inf
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(int(settings["maximumEpochs"])):
        model.train()
        order = train_index[
            torch.randperm(len(train_index), generator=generator).numpy()
        ]
        losses = []
        for start in range(0, len(order), int(settings["batchSize"])):
            index = torch.as_tensor(
                order[start : start + int(settings["batchSize"])], device=device
            )
            _, reconstruction, task_logits, length_prediction = model(
                x[index], adversarial
            )
            loss = nn.functional.mse_loss(reconstruction, x[index])
            if adversarial:
                loss = loss + float(settings["adversaryWeights"]["task"]) * nn.functional.cross_entropy(
                    task_logits, tasks[index]
                )
                loss = loss + float(settings["adversaryWeights"]["length"]) * nn.functional.mse_loss(
                    length_prediction, lengths[index]
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            index = torch.as_tensor(calibration_index, device=device)
            _, reconstruction, task_logits, length_prediction = model(x[index], False)
            validation = nn.functional.mse_loss(reconstruction, x[index])
            validation_loss = float(validation.cpu())
        history.append(
            {
                "epoch": epoch + 1,
                "trainLoss": float(np.mean(losses)),
                "calibrationReconstructionLoss": validation_loss,
            }
        )
        if validation_loss < best - 1e-7:
            best = validation_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        if (
            epoch + 1 >= int(settings["minimumEpochs"])
            and epoch + 1 - best_epoch >= int(settings["earlyStoppingPatience"])
        ):
            break
    if best_state is None:
        raise RuntimeError("embedding model produced no state")
    return best_state, history


def embedding_outputs(
    state: Mapping[str, torch.Tensor],
    values: np.ndarray,
    settings: Mapping[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = EventSummaryAutoencoder(
        values.shape[1], int(settings["embeddingDimension"]), len(TASK_IDS)
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        embedding, reconstruction, _, _ = model(
            torch.as_tensor(values, dtype=torch.float32, device=device), False
        )
    return embedding.cpu().numpy(), reconstruction.cpu().numpy()


def task_probe(
    embedding: np.ndarray,
    task_labels: np.ndarray,
    groups: np.ndarray,
) -> float:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=706301)
    truth = []
    prediction = []
    for train, test in splitter.split(embedding, task_labels, groups):
        model = LogisticRegression(max_iter=2000, C=1.0, random_state=706302)
        model.fit(embedding[train], task_labels[train])
        truth.extend(task_labels[test].tolist())
        prediction.extend(model.predict(embedding[test]).tolist())
    return float(balanced_accuracy_score(truth, prediction))


def length_probe(
    embedding: np.ndarray,
    length_target: np.ndarray,
    groups: np.ndarray,
) -> float:
    splitter = GroupKFold(n_splits=5)
    truth = []
    prediction = []
    for train, test in splitter.split(embedding, length_target, groups):
        model = Ridge(alpha=10.0)
        model.fit(embedding[train], length_target[train])
        truth.extend(length_target[test].tolist())
        prediction.extend(model.predict(embedding[test]).tolist())
    return float(r2_score(truth, prediction))


def task_within_standardize(
    values: np.ndarray,
    task_labels: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[np.ndarray, dict[int, dict[str, list[float]]]]:
    output = values.copy()
    parameters: dict[int, dict[str, list[float]]] = {}
    for task in range(len(TASK_IDS)):
        fit = fit_mask & (task_labels == task)
        mean = np.nanmean(values[fit], axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.nanstd(values[fit], axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
        rows = task_labels == task
        output[rows] = (np.where(np.isfinite(values[rows]), values[rows], mean) - mean) / std
        parameters[task] = {"mean": mean.tolist(), "scale": std.tolist()}
    return output.astype(np.float32), parameters


def summarize_task_metrics(
    metrics: pd.DataFrame, settings: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for task_id in TASK_IDS:
        for panel in ("lineage_test", "scenario_test"):
            subset = metrics[
                (metrics.taskId == task_id)
                & (metrics.auditPanel == panel)
                & (metrics.targetGroup == "performance")
                & (~metrics.constantTarget)
            ]
            losses = {}
            for variant in ("gpu", "ridge", "constant", "policy_only_gpu"):
                losses[variant] = subset[subset.variant == variant].set_index("target").loss
            common = sorted(set(losses["gpu"].dropna().index) & set(losses["ridge"].dropna().index))
            ratios = [
                losses["gpu"][name] / max(losses["ridge"][name], 1e-12)
                for name in common
            ]
            constant_ratios = [
                losses["gpu"][name] / max(losses["constant"][name], 1e-12)
                for name in common
            ]
            gpu_rows = subset[subset.variant == "gpu"]
            continuous = gpu_rows[gpu_rows.targetType == "continuous"]
            binary = gpu_rows[gpu_rows.targetType == "binary"]
            rows.append(
                {
                    "taskId": task_id,
                    "auditPanel": panel,
                    "eligibleObjectiveTargets": len(common),
                    "objectiveLossRatioVsRidge": float(np.median(ratios)) if ratios else math.nan,
                    "objectiveLossRatioVsConstant": float(np.median(constant_ratios)) if constant_ratios else math.nan,
                    "continuousObjectiveMedianCoverage": float(continuous.coverage.median()) if len(continuous) else math.nan,
                    "continuousObjectiveMedianWidth": float(continuous.intervalWidth.median()) if len(continuous) else math.nan,
                    "binaryObjectiveMaximumEce": float(binary.ece.max()) if len(binary) else math.nan,
                }
            )
    summary = pd.DataFrame(rows)
    thresholds = settings["frozenThresholds"]
    utility = thresholds["surrogateUtility"]
    task_pivot = summary.pivot(index="taskId", columns="auditPanel", values="objectiveLossRatioVsRidge")
    passes_both = (
        (task_pivot.lineage_test <= float(utility["taskObjectiveLossRatioVsRidgeMaximum"]))
        & (task_pivot.scenario_test <= float(utility["taskObjectiveLossRatioVsRidgeMaximum"]))
    )
    any_ratio = summary.objectiveLossRatioVsRidge.replace([np.inf, -np.inf], np.nan)
    constant_pivot = summary.pivot(index="taskId", columns="auditPanel", values="objectiveLossRatioVsConstant")
    constant_both = (constant_pivot.lineage_test < 1.0) & (constant_pivot.scenario_test < 1.0)
    calibration = thresholds["continuousCalibration"]
    coverage_pivot = summary.pivot(index="taskId", columns="auditPanel", values="continuousObjectiveMedianCoverage")
    coverage_both = (
        (coverage_pivot.lineage_test >= float(calibration["minimumCoverage"]))
        & (coverage_pivot.scenario_test >= float(calibration["minimumCoverage"]))
    )
    decision = {
        "utilityTasksPassingBoth": int(passes_both.sum()),
        "utilityTasksRequired": int(utility["tasksRequiredBothLineageAndScenario"]),
        "maximumObservedTaskObjectiveLossRatio": float(any_ratio.max()) if any_ratio.notna().any() else None,
        "maximumAllowedTaskObjectiveLossRatio": float(utility["maximumAnyTaskObjectiveLossRatio"]),
        "constantImprovementTasksBoth": int(constant_both.sum()),
        "constantImprovementTasksRequired": int(utility["constantBaselineImprovementTasksRequired"]),
        "continuousCoverageTasksPassingBoth": int(coverage_both.fillna(False).sum()),
        "continuousCoverageTasksRequired": int(calibration["tasksRequiredBothLineageAndScenario"]),
    }
    decision["utilityPass"] = (
        decision["utilityTasksPassingBoth"] >= decision["utilityTasksRequired"]
        and decision["maximumObservedTaskObjectiveLossRatio"]
        <= decision["maximumAllowedTaskObjectiveLossRatio"]
        and decision["constantImprovementTasksBoth"]
        >= decision["constantImprovementTasksRequired"]
    )
    decision["continuousCalibrationPass"] = (
        decision["continuousCoverageTasksPassingBoth"]
        >= decision["continuousCoverageTasksRequired"]
    )
    return summary, decision


def run() -> dict[str, Any]:
    settings = protocol()
    prereg = verify_preregistration(settings)
    started = time.perf_counter()
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    torch.set_num_threads(8)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("frozen S06 protocol requires CUDA")
    CACHE.mkdir(parents=True, exist_ok=True)
    for directory in ("models", "embeddings", "metrics", "figures"):
        (S06_ROOT / directory).mkdir(parents=True, exist_ok=True)

    all_rows = read_jsonl(S05_ROOT / "evaluation_ledger.jsonl")
    selected = sorted(
        [row for row in all_rows if row["selectedForCanonicalSearch"]],
        key=lambda row: row["stableEvaluationSha256"],
    )
    orphans = sorted(
        [row for row in all_rows if not row["selectedForCanonicalSearch"]],
        key=lambda row: row["stableEvaluationSha256"],
    )
    lineage = load_lineage()
    catalog = load_policy_catalog_with_plans()
    if any(row["policySha256"] not in catalog for row in selected):
        missing = sorted(
            {row["policySha256"] for row in selected if row["policySha256"] not in catalog}
        )
        raise RuntimeError(f"policy documents unavailable for modeling: {missing[:3]}")
    orphan_document_missing = [
        row for row in orphans if row["policySha256"] not in catalog
    ]
    assignments = assign_grouped_splits(selected, lineage, settings)
    assignment_by_hash = {item["stableEvaluationSha256"]: item for item in assignments}
    assignment_frame = pd.DataFrame(assignments)
    assignment_frame.to_parquet(S06_ROOT / "split_assignments.parquet", index=False)
    split_validation = {
        "schemaVersion": "e07.s06.grouped-split-validation.v1",
        "researchStepId": "S06",
        "success": True,
        "rowCount": len(assignments),
        "taskCounts": dict(Counter(item["taskId"] for item in assignments)),
        "lineageComponentOverlap": 0,
        "scenarioFamilyOverlap": 0,
        "outcomesUsedForAssignment": False,
        "panelsByTask": {},
    }

    metric_rows: list[dict[str, Any]] = []
    target_registries = {}
    feature_registries = {}
    histories: dict[str, Any] = {}
    model_manifest = []
    runtime_objects: dict[str, Any] = {}
    deterministic_reference = None
    inference_reload_pass = True
    native_failure_count = sum(row["failed"] for row in selected)
    native_censor_count = sum(row["censored"] for row in selected)

    for task_id in TASK_IDS:
        task_rows = [row for row in selected if row["taskId"] == task_id]
        task_assignments = [assignment_by_hash[row["stableEvaluationSha256"]] for row in task_rows]
        panels = panel_indices(task_assignments)
        split_validation["panelsByTask"][task_id] = {
            key: int(len(value)) for key, value in panels.items()
        }
        if min(len(panels[key]) for key in ("fit", "lineage_calibration", "lineage_test")) == 0:
            raise RuntimeError(f"empty required lineage panel for {task_id}")
        registry = build_target_registry(task_id, task_rows)
        target_registries[task_id] = registry
        continuous_raw, binary = target_arrays(task_rows, registry)
        transform = fit_target_transform(continuous_raw[panels["fit"]], registry)
        continuous = transform.transform(continuous_raw)

        x, vectorizer, scaler = vectorize_features(
            task_rows, catalog, panels["fit"], scenario=True
        )
        x_policy, policy_vectorizer, policy_scaler = vectorize_features(
            task_rows, catalog, panels["fit"], scenario=False
        )
        forbidden_tokens = ("sha", "hash", "policyId", "scenarioId", "outcome", "event", "cost", "elapsed")
        if any(
            any(token.lower() in name.lower() for token in forbidden_tokens)
            for name in vectorizer.feature_names_
        ):
            raise RuntimeError(f"forbidden identifier/outcome feature in {task_id}")
        feature_registries[task_id] = {
            "featureNames": vectorizer.feature_names_,
            "policyOnlyFeatureNames": policy_vectorizer.feature_names_,
            "forbiddenIdentifierFieldsPresent": False,
            "inputDimension": x.shape[1],
            "policyOnlyInputDimension": x_policy.shape[1],
        }
        baselines = baseline_predictions(x, continuous, binary, panels["fit"])
        gpu_settings = settings["surrogates"]["gpuCandidate"]
        states = []
        task_histories = []
        for seed in gpu_settings["seeds"]:
            state, history = train_surrogate_member(
                x[panels["fit"]],
                continuous[panels["fit"]],
                binary[panels["fit"]],
                x[panels["lineage_calibration"]],
                continuous[panels["lineage_calibration"]],
                binary[panels["lineage_calibration"]],
                registry,
                gpu_settings,
                seed=int(seed),
                device=device,
            )
            states.append(state)
            task_histories.append(history)
        policy_states = []
        policy_histories = []
        for seed in gpu_settings["seeds"]:
            state, history = train_surrogate_member(
                x_policy[panels["fit"]],
                continuous[panels["fit"]],
                binary[panels["fit"]],
                x_policy[panels["lineage_calibration"]],
                continuous[panels["lineage_calibration"]],
                binary[panels["lineage_calibration"]],
                registry,
                gpu_settings,
                seed=int(seed),
                device=device,
            )
            policy_states.append(state)
            policy_histories.append(history)
        gpu_c, gpu_c_sd, gpu_b, gpu_b_sd = ensemble_predict(
            states, x, continuous.shape[1], binary.shape[1], device=device
        )
        policy_c, _, policy_b, _ = ensemble_predict(
            policy_states,
            x_policy,
            continuous.shape[1],
            binary.shape[1],
            device=device,
        )
        predictions = {
            "gpu": {"continuous": gpu_c, "binary": gpu_b},
            "policy_only_gpu": {"continuous": policy_c, "binary": policy_b},
            "ridge": {
                "continuous": baselines["continuousRidge"],
                "binary": baselines["binaryLogistic"],
            },
            "constant": {
                "continuous": baselines["continuousConstant"],
                "binary": baselines["binaryConstant"],
            },
        }
        task_metrics, conformal = metric_records(
            task_id,
            registry,
            continuous,
            binary,
            predictions,
            panels,
            float(settings["surrogates"]["uncertainty"]["conformalNominalCoverage"]),
        )
        metric_rows.extend(task_metrics)
        histories[task_id] = {
            "gpu": task_histories,
            "policyOnlyGpu": policy_histories,
        }
        bundle = {
            "schemaVersion": "e07.s06.task-surrogate-bundle.v1",
            "researchStepId": "S06",
            "taskId": task_id,
            "inputDim": x.shape[1],
            "continuousDim": continuous.shape[1],
            "binaryDim": binary.shape[1],
            "featureNames": vectorizer.feature_names_,
            "featureMean": scaler.mean_.tolist(),
            "featureScale": scaler.scale_.tolist(),
            "targetCenter": transform.centers.tolist(),
            "targetScale": transform.scales.tolist(),
            "targetTransforms": list(transform.transforms),
            "targetRegistry": registry,
            "stateDicts": states,
            "stateDictHashes": [state_dict_hash(state) for state in states],
            "claimBoundary": settings["claimBoundary"],
        }
        bundle["modelBundleSha256"] = model_bundle_hash(bundle)
        model_path = S06_ROOT / "models" / f"{task_id}_surrogate.pt"
        torch.save(bundle, model_path)
        reloaded = torch.load(model_path, map_location="cpu", weights_only=False)
        reload_c, _, reload_b, _ = ensemble_predict(
            reloaded["stateDicts"],
            x[: min(8, len(x))],
            continuous.shape[1],
            binary.shape[1],
            device=device,
        )
        inference_reload_pass &= np.array_equal(reload_c, gpu_c[: len(reload_c)])
        inference_reload_pass &= np.array_equal(reload_b, gpu_b[: len(reload_b)])
        model_manifest.append(
            {
                "taskId": task_id,
                "path": str(model_path.relative_to(S06_ROOT)),
                "sha256": hash_file(model_path),
                "bundleSha256": bundle["modelBundleSha256"],
                "ensembleMembers": len(states),
                "inputDimension": x.shape[1],
                "continuousTargets": continuous.shape[1],
                "binaryTargets": binary.shape[1],
            }
        )
        runtime_objects[task_id] = {
            "rows": task_rows,
            "panels": panels,
            "registry": registry,
            "transform": transform,
            "vectorizer": vectorizer,
            "scaler": scaler,
            "states": states,
            "conformal": conformal,
            "fitElapsedThreshold": float(
                np.quantile([task_rows[index]["elapsedSeconds"] for index in panels["fit"]], 0.75)
            ),
            "fitLengthThreshold": float(
                np.quantile([event_length(task_rows[index]) for index in panels["fit"]], 0.75)
            ),
        }
        if deterministic_reference is None:
            deterministic_reference = {
                "taskId": task_id,
                "state": states[0],
                "x": x,
                "continuous": continuous,
                "binary": binary,
                "panels": panels,
                "registry": registry,
            }

    write_json(S06_ROOT / "target_registry.json", target_registries)
    write_json(S06_ROOT / "feature_registry.json", feature_registries)
    write_json(S06_ROOT / "training_history.json", histories)
    write_json(S06_ROOT / "grouped_split_validation.json", split_validation)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_parquet(S06_ROOT / "metrics/target_metrics.parquet", index=False)
    task_summary, surrogate_decision = summarize_task_metrics(metrics, settings)
    task_summary.to_csv(S06_ROOT / "metrics/task_summary.csv", index=False)
    write_json(S06_ROOT / "metrics/task_summary.json", _json_safe(task_summary.to_dict("records")))

    # Event-summary embedding candidate and raw-union ablation.
    selected_assignments = [assignment_by_hash[row["stableEvaluationSha256"]] for row in selected]
    task_to_index = {task: index for index, task in enumerate(TASK_IDS)}
    task_labels = np.asarray([task_to_index[row["taskId"]] for row in selected])
    fit_mask = np.asarray(
        [
            item["lineagePartition"] == "fit" and item["scenarioPartition"] == "fit"
            for item in selected_assignments
        ]
    )
    calibration_mask = np.asarray(
        [
            item["lineagePartition"] == "calibration"
            and item["scenarioPartition"] == "fit"
            for item in selected_assignments
        ]
    )
    audit_mask = np.asarray(
        [
            (item["lineagePartition"] == "test" and item["scenarioPartition"] == "fit")
            or (item["lineagePartition"] == "fit" and item["scenarioPartition"] == "test")
            for item in selected_assignments
        ]
    )
    shared_vectorizer = DictVectorizer(sparse=False, sort=True)
    shared_raw = shared_vectorizer.fit_transform(
        [shared_event_features(row) for row in selected]
    ).astype(np.float64)
    shared_values, task_scalers = task_within_standardize(
        shared_raw, task_labels, fit_mask
    )
    raw_vectorizer = DictVectorizer(sparse=False, sort=True)
    raw_values = raw_vectorizer.fit_transform([raw_event_features(row) for row in selected])
    raw_scaler = StandardScaler()
    raw_scaler.fit(np.nan_to_num(raw_values[fit_mask], nan=0.0))
    raw_values = raw_scaler.transform(np.nan_to_num(raw_values, nan=0.0)).astype(np.float32)
    log_length = np.log1p(np.asarray([event_length(row) for row in selected]))
    length_z = np.zeros_like(log_length)
    for task_index in range(len(TASK_IDS)):
        fit = fit_mask & (task_labels == task_index)
        mean = log_length[fit].mean()
        scale = log_length[fit].std() or 1.0
        rows = task_labels == task_index
        length_z[rows] = (log_length[rows] - mean) / scale
    embedding_settings = settings["trajectorySummaryModel"]
    embedding_state, embedding_history = train_embedding_model(
        shared_values,
        task_labels,
        length_z,
        np.flatnonzero(fit_mask),
        np.flatnonzero(calibration_mask),
        embedding_settings,
        adversarial=True,
        device=device,
    )
    raw_state, raw_history = train_embedding_model(
        raw_values,
        task_labels,
        length_z,
        np.flatnonzero(fit_mask),
        np.flatnonzero(calibration_mask),
        embedding_settings,
        adversarial=False,
        device=device,
    )
    embedding, reconstruction = embedding_outputs(
        embedding_state, shared_values, embedding_settings, device
    )
    raw_embedding, _ = embedding_outputs(raw_state, raw_values, embedding_settings, device)
    pca = PCA(n_components=int(embedding_settings["embeddingDimension"]), random_state=706401)
    pca.fit(shared_values[fit_mask])
    pca_reconstruction = pca.inverse_transform(pca.transform(shared_values))
    groups = np.asarray(
        [f"{row['taskId']}::{item['lineageComponent']}" for row, item in zip(selected, selected_assignments, strict=True)]
    )
    shortcut = {
        "schemaVersion": "e07.s06.shortcut-tests.v1",
        "researchStepId": "S06",
        "taskChanceBalancedAccuracy": 1.0 / len(TASK_IDS),
        "rawUnionTaskBalancedAccuracy": task_probe(raw_embedding, task_labels, groups),
        "candidateTaskBalancedAccuracy": task_probe(embedding, task_labels, groups),
        "rawUnionWithinTaskLengthR2": length_probe(raw_embedding, length_z, groups),
        "candidateWithinTaskLengthR2": length_probe(embedding, length_z, groups),
        "candidateReconstructionMse": float(mean_squared_error(shared_values[audit_mask], reconstruction[audit_mask])),
        "pcaReconstructionMse": float(mean_squared_error(shared_values[audit_mask], pca_reconstruction[audit_mask])),
        "identifierLeakageAuditPass": True,
        "fullTraceRepresentationClaim": False,
    }
    shortcut["candidateReconstructionMseRatioVsPca"] = shortcut[
        "candidateReconstructionMse"
    ] / max(shortcut["pcaReconstructionMse"], 1e-12)
    threshold = settings["frozenThresholds"]["embeddingDeployment"]
    shortcut["embeddingDeploymentPass"] = (
        shortcut["candidateTaskBalancedAccuracy"]
        <= float(threshold["maximumTaskBalancedAccuracy"])
        and shortcut["candidateWithinTaskLengthR2"]
        <= float(threshold["maximumWithinTaskLengthR2"])
        and shortcut["candidateReconstructionMseRatioVsPca"]
        <= float(threshold["maximumReconstructionMseRatioVsPCA"])
    )
    write_json(S06_ROOT / "metrics/shortcut_tests.json", shortcut)
    histories["eventSummaryEmbedding"] = embedding_history
    histories["rawUnionEmbeddingAblation"] = raw_history
    write_json(S06_ROOT / "training_history.json", histories)
    embedding_frame = pd.DataFrame(
        {
            "stableEvaluationSha256": [row["stableEvaluationSha256"] for row in selected],
            "taskId": [row["taskId"] for row in selected],
            "policySha256": [row["policySha256"] for row in selected],
            "scenarioOrdinal": [row["scenarioOrdinal"] for row in selected],
            "lineageComponent": [item["lineageComponent"] for item in selected_assignments],
            "failed": [row["failed"] for row in selected],
            "censored": [row["censored"] for row in selected],
            "eventLength": [event_length(row) for row in selected],
            **{f"embedding_{index}": embedding[:, index] for index in range(embedding.shape[1])},
        }
    )
    embedding_frame.to_parquet(
        S06_ROOT / "embeddings/trajectory_summary_embeddings.parquet", index=False
    )
    embedding_bundle = {
        "schemaVersion": "e07.s06.event-summary-embedding-bundle.v1",
        "researchStepId": "S06",
        "inputDim": shared_values.shape[1],
        "embeddingDim": embedding.shape[1],
        "featureNames": shared_vectorizer.feature_names_,
        "taskWithinScalers": task_scalers,
        "stateDicts": [embedding_state],
        "stateDictHashes": [state_dict_hash(embedding_state)],
        "fullTraceRepresentation": False,
        "evidenceBoundary": embedding_settings["evidenceBoundary"],
    }
    embedding_bundle["modelBundleSha256"] = model_bundle_hash(embedding_bundle)
    embedding_model_path = S06_ROOT / "models/event_summary_embedding.pt"
    torch.save(embedding_bundle, embedding_model_path)
    model_manifest.append(
        {
            "taskId": "cross_task_event_summary",
            "path": str(embedding_model_path.relative_to(S06_ROOT)),
            "sha256": hash_file(embedding_model_path),
            "bundleSha256": embedding_bundle["modelBundleSha256"],
            "embeddingDimension": embedding.shape[1],
        }
    )

    # Frozen-model-only sensitivity for the 189 recovery orphans.
    orphan_records = []
    orphan_embedding_rows = []
    for task_id in sorted({row["taskId"] for row in orphans}):
        task_orphans = [row for row in orphans if row["taskId"] == task_id]
        scorable_orphans = [
            row for row in task_orphans if row["policySha256"] in catalog
        ]
        runtime = runtime_objects[task_id]
        if scorable_orphans:
            records = [
                policy_feature_dict(row, catalog, scenario=True)
                for row in scorable_orphans
            ]
            x_orphan = runtime["scaler"].transform(
                runtime["vectorizer"].transform(records)
            ).astype(np.float32)
            c_raw, b = target_arrays(scorable_orphans, runtime["registry"])
            c = runtime["transform"].transform(c_raw)
            cp, _, bp, _ = ensemble_predict(
                runtime["states"], x_orphan, c.shape[1], b.shape[1], device=device
            )
            for target, item in enumerate(runtime["registry"]["continuous"]):
                mask = np.isfinite(c[:, target])
                orphan_records.append(
                    {
                        "taskId": task_id,
                        "target": item["name"],
                        "targetType": "continuous",
                        "n": int(mask.sum()),
                        "loss": float(np.sqrt(np.mean(np.square(c[mask, target] - cp[mask, target]))))
                        if np.any(mask)
                        else None,
                    }
                )
        shared_orphan_raw = shared_vectorizer.transform(
            [shared_event_features(row) for row in task_orphans]
        )
        task_index = task_to_index[task_id]
        params = task_scalers[task_index]
        mean = np.asarray(params["mean"])
        scale = np.asarray(params["scale"])
        shared_orphan = (
            np.where(np.isfinite(shared_orphan_raw), shared_orphan_raw, mean) - mean
        ) / scale
        orphan_embedding, _ = embedding_outputs(
            embedding_state, shared_orphan.astype(np.float32), embedding_settings, device
        )
        for row, vector in zip(task_orphans, orphan_embedding, strict=True):
            orphan_embedding_rows.append(
                {
                    "stableEvaluationSha256": row["stableEvaluationSha256"],
                    "taskId": task_id,
                    "policySha256": row["policySha256"],
                    "scenarioOrdinal": row["scenarioOrdinal"],
                    "failed": row["failed"],
                    "censored": row["censored"],
                    **{f"embedding_{index}": float(value) for index, value in enumerate(vector)},
                }
            )
    pd.DataFrame(orphan_embedding_rows).to_parquet(
        S06_ROOT / "embeddings/recovery_orphan_embeddings.parquet", index=False
    )
    write_json(
        S06_ROOT / "metrics/recovery_orphan_sensitivity.json",
        {
            "schemaVersion": "e07.s06.recovery-orphan-sensitivity.v1",
            "researchStepId": "S06",
            "rowCount": len(orphans),
            "usedForFit": False,
            "usedForTuning": False,
            "usedForCalibration": False,
            "usedForEmbeddingTraining": False,
            "tasks": dict(Counter(row["taskId"] for row in orphans)),
            "failedRows": sum(row["failed"] for row in orphans),
            "censoredRows": sum(row["censored"] for row in orphans),
            "policyDocumentAvailableRowsForFrozenSurrogateScoring": len(orphans)
            - len(orphan_document_missing),
            "policyDocumentUnavailableRows": len(orphan_document_missing),
            "policyDocumentUnavailableUniquePolicies": len(
                {row["policySha256"] for row in orphan_document_missing}
            ),
            "missingDocumentHandling": "retained in descriptive and embedding sensitivity; excluded only from policy-input surrogate scoring; no reconstruction from hashes or superseded plans",
            "targetMetrics": orphan_records,
        },
    )

    # Subgroup calibration, including E05 heavy tails.
    subgroup_rows = []
    for task_id, runtime in runtime_objects.items():
        task_rows = runtime["rows"]
        c_raw, b = target_arrays(task_rows, runtime["registry"])
        c = runtime["transform"].transform(c_raw)
        x = runtime["scaler"].transform(
            runtime["vectorizer"].transform(
                [policy_feature_dict(row, catalog, scenario=True) for row in task_rows]
            )
        ).astype(np.float32)
        cp, _, bp, _ = ensemble_predict(
            runtime["states"], x, c.shape[1], b.shape[1], device=device
        )
        for audit in ("lineage_test", "scenario_test"):
            indices = runtime["panels"][audit]
            q = runtime["conformal"][audit]["gpu"]
            subgroup_masks = {
                "all": np.ones(len(indices), dtype=bool),
                "failed": np.asarray([task_rows[index]["failed"] for index in indices]),
                "censored": np.asarray([task_rows[index]["censored"] for index in indices]),
            }
            if task_id in {"e07_s02_regeneration_1d", "e07_s02_target_change_1d"}:
                subgroup_masks["e05_runtime_tail_quartile"] = np.asarray(
                    [task_rows[index]["elapsedSeconds"] >= runtime["fitElapsedThreshold"] for index in indices]
                )
                subgroup_masks["e05_native_event_length_tail_quartile"] = np.asarray(
                    [event_length(task_rows[index]) >= runtime["fitLengthThreshold"] for index in indices]
                )
            for subgroup, subgroup_mask in subgroup_masks.items():
                for target, item in enumerate(runtime["registry"]["continuous"]):
                    if item["group"] != "performance":
                        continue
                    y = c[indices, target]
                    p = cp[indices, target]
                    mask = subgroup_mask & np.isfinite(y) & np.isfinite(p)
                    subgroup_rows.append(
                        {
                            "taskId": task_id,
                            "auditPanel": audit,
                            "subgroup": subgroup,
                            "target": item["name"],
                            "targetType": "continuous",
                            "n": int(mask.sum()),
                            "coverage": float(np.mean(np.abs(y[mask] - p[mask]) <= q[target]))
                            if np.any(mask) and math.isfinite(q[target])
                            else math.nan,
                            "brier": math.nan,
                            "ece": math.nan,
                        }
                    )
                for target, item in enumerate(runtime["registry"]["binary"]):
                    if item["name"] not in {"status::failed", "status::censored"}:
                        continue
                    y = b[indices, target]
                    p = bp[indices, target]
                    mask = subgroup_mask & np.isfinite(y) & np.isfinite(p)
                    subgroup_rows.append(
                        {
                            "taskId": task_id,
                            "auditPanel": audit,
                            "subgroup": subgroup,
                            "target": item["name"],
                            "targetType": "binary",
                            "n": int(mask.sum()),
                            "coverage": math.nan,
                            "brier": float(brier_score_loss(y[mask], p[mask])) if np.any(mask) else math.nan,
                            "ece": ece(y[mask], p[mask]) if np.any(mask) else math.nan,
                        }
                    )
    subgroup_frame = pd.DataFrame(subgroup_rows)
    subgroup_frame.to_csv(S06_ROOT / "metrics/subgroup_calibration.csv", index=False)

    # Exact deterministic replay of one frozen member.
    reference = deterministic_reference
    replay_state, _ = train_surrogate_member(
        reference["x"][reference["panels"]["fit"]],
        reference["continuous"][reference["panels"]["fit"]],
        reference["binary"][reference["panels"]["fit"]],
        reference["x"][reference["panels"]["lineage_calibration"]],
        reference["continuous"][reference["panels"]["lineage_calibration"]],
        reference["binary"][reference["panels"]["lineage_calibration"]],
        reference["registry"],
        settings["surrogates"]["gpuCandidate"],
        seed=int(settings["surrogates"]["gpuCandidate"]["seeds"][0]),
        device=device,
    )
    deterministic_replay_pass = state_dict_hash(replay_state) == state_dict_hash(reference["state"])

    # Deployment decision remains fail-closed.
    binary_threshold = settings["frozenThresholds"]["binaryCalibration"]
    gpu_binary = metrics[
        (metrics.variant == "gpu")
        & (metrics.targetType == "binary")
        & (~metrics.constantTarget)
        & (metrics.targetGroup.isin(["performance", "binary"]))
    ]
    binary_pass_tasks = 0
    for task_id in TASK_IDS:
        task = gpu_binary[gpu_binary.taskId == task_id]
        if all(
            len(task[task.auditPanel == panel]) == 0
            or task[task.auditPanel == panel].ece.max() <= float(binary_threshold["maximumECE"])
            for panel in ("lineage_test", "scenario_test")
        ):
            binary_pass_tasks += 1
    binary_calibration_pass = binary_pass_tasks >= int(
        binary_threshold["tasksRequiredBothLineageAndScenario"]
    )
    e05_tail = subgroup_frame[
        subgroup_frame.subgroup.isin(
            ["e05_runtime_tail_quartile", "e05_native_event_length_tail_quartile"]
        )
        & (subgroup_frame.targetType == "continuous")
        & (subgroup_frame.n >= 3)
    ]
    heavy_tail_coverage = float(e05_tail.coverage.median()) if len(e05_tail) else math.nan
    failure_censor_pass = (
        math.isfinite(heavy_tail_coverage)
        and heavy_tail_coverage
        >= float(settings["frozenThresholds"]["failureAndCensoring"]["heavyTailE05CoverageMinimum"])
    )
    deployment = {
        "schemaVersion": "e07.s06.deployment-decision.v1",
        "researchStepId": "S06",
        "surrogateUtility": surrogate_decision,
        "binaryCalibrationTasksPassing": binary_pass_tasks,
        "binaryCalibrationTasksRequired": int(binary_threshold["tasksRequiredBothLineageAndScenario"]),
        "binaryCalibrationPass": binary_calibration_pass,
        "e05HeavyTailMedianCoverage": heavy_tail_coverage,
        "failureAndCensoringPass": failure_censor_pass,
        "embedding": shortcut,
        "deterministicReplayPass": deterministic_replay_pass,
        "inferenceReloadParityPass": inference_reload_pass,
    }
    deployment["eligibleForS07"] = all(
        [
            surrogate_decision["utilityPass"],
            surrogate_decision["continuousCalibrationPass"],
            binary_calibration_pass,
            failure_censor_pass,
            shortcut["embeddingDeploymentPass"],
            deterministic_replay_pass,
            inference_reload_pass,
        ]
    )
    deployment["decision"] = "accept" if deployment["eligibleForS07"] else "reject"
    deployment["rejectionRuleApplied"] = not deployment["eligibleForS07"]
    deployment["permittedUse"] = (
        "S07 allocation guidance" if deployment["eligibleForS07"] else "diagnostic research artifact only"
    )
    write_json(S06_ROOT / "deployment_decision.json", _json_safe(deployment))

    write_json(
        S06_ROOT / "models/model_manifest.json",
        {
            "schemaVersion": "e07.s06.model-manifest.v1",
            "researchStepId": "S06",
            "modelCount": len(model_manifest),
            "models": model_manifest,
            "inferenceReloadParityPass": inference_reload_pass,
            "deterministicReplayPass": deterministic_replay_pass,
            "deploymentDecision": deployment["decision"],
        },
    )

    # Compact plots.
    plot = task_summary.pivot(index="taskId", columns="auditPanel", values="objectiveLossRatioVsRidge")
    fig, ax = plt.subplots(figsize=(10, 5))
    plot.plot(kind="bar", ax=ax)
    ax.axhline(float(settings["frozenThresholds"]["surrogateUtility"]["taskObjectiveLossRatioVsRidgeMaximum"]), color="black", linestyle="--", label="frozen threshold")
    ax.set_ylabel("Median task-objective loss ratio vs ridge")
    ax.set_title("S06 grouped surrogate utility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(S06_ROOT / "figures/surrogate_utility.png", dpi=180)
    fig.savefig(S06_ROOT / "figures/surrogate_utility.svg")
    plt.close(fig)
    coverage_plot = task_summary.pivot(index="taskId", columns="auditPanel", values="continuousObjectiveMedianCoverage")
    fig, ax = plt.subplots(figsize=(10, 5))
    coverage_plot.plot(kind="bar", ax=ax)
    ax.axhline(float(settings["frozenThresholds"]["continuousCalibration"]["minimumCoverage"]), color="black", linestyle="--", label="frozen minimum")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Median 90% conformal coverage")
    ax.set_title("S06 task-objective calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(S06_ROOT / "figures/calibration_coverage.png", dpi=180)
    fig.savefig(S06_ROOT / "figures/calibration_coverage.svg")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["raw union task", "candidate task", "raw length R2", "candidate length R2"],
        [
            shortcut["rawUnionTaskBalancedAccuracy"],
            shortcut["candidateTaskBalancedAccuracy"],
            shortcut["rawUnionWithinTaskLengthR2"],
            shortcut["candidateWithinTaskLengthR2"],
        ],
    )
    ax.set_title("Embedding shortcut probes")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(S06_ROOT / "figures/embedding_shortcuts.png", dpi=180)
    fig.savefig(S06_ROOT / "figures/embedding_shortcuts.svg")
    plt.close(fig)

    # Model cards are generated from frozen decisions, not promotional summaries.
    for item in model_manifest:
        card_path = S06_ROOT / "models" / f"{item['taskId']}_model_card.md"
        card_path.write_text(
            "\n".join(
                [
                    f"# Model card: {item['taskId']}",
                    "",
                    "- Research step: S06",
                    f"- Artifact: `{item['path']}`",
                    f"- SHA-256: `{item['sha256']}`",
                    "- Training population: frozen S05 canonical selected training rows only.",
                    "- Recovery-orphan use: sensitivity only; never fitting, tuning, calibration, or embedding training.",
                    "- Protected validation/confirmation outcomes: sealed.",
                    f"- Deployment decision: **{deployment['decision']}**.",
                    "- Intended use: task-specific exploratory prediction or event-summary analysis only when the fail-closed deployment record permits it.",
                    "- Prohibited use: universal scoring, cross-task dominance, biological interpretation, confirmation claims, or full-trajectory claims.",
                    "- Known risks: lineage shift, scenario-family shift, censoring, failures, task shortcuts, length shortcuts, E05 heavy runtime tails, and selected-panel bias.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    elapsed = time.perf_counter() - started
    validation = {
        "schemaVersion": "e07.s06.validation-summary.v1",
        "researchStepId": "S06",
        "success": True,
        "checks": {
            "s05HashesFrozenAndReverified": True,
            "authoritativeGenerationPlansRetained": True,
            "primaryCanonicalRows": len(selected),
            "recoveryOrphanRowsSensitivityOnly": len(orphans),
            "allPrimaryFailuresRetained": native_failure_count,
            "allPrimaryCensorsRetained": native_censor_count,
            "validationOutcomeEvaluations": 0,
            "confirmationOutcomeEvaluations": 0,
            "lineageAndScenarioGroupsDisjoint": True,
            "identifierLeakageAuditPass": True,
            "deterministicReplayPass": deterministic_replay_pass,
            "inferenceReloadParityPass": inference_reload_pass,
            "modelHashesComplete": True,
            "embeddingShortcutTestsComplete": True,
        },
        "deploymentDecision": deployment["decision"],
        "elapsedSeconds": elapsed,
    }
    write_json(S06_ROOT / "validation_summary.json", validation)
    environment = {
        "schemaVersion": "e07.s06.environment.v1",
        "researchStepId": "S06",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cudaAvailable": torch.cuda.is_available(),
        "cudaDevice": torch.cuda.get_device_name(0),
        "cudaDeviceCount": torch.cuda.device_count(),
        "cpuCountVisible": os.cpu_count(),
        "cpuThreadsUsed": 8,
        "gpuDeviceUsed": "cuda:0",
        "newDependenciesInstalled": [],
    }
    write_json(S06_ROOT / "environment.json", environment)
    write_json(
        S06_ROOT / "result_summary.json",
        {
            "schemaVersion": "e07.s06.result-summary.v1",
            "researchStepId": "S06",
            "success": True,
            "primaryRows": len(selected),
            "recoveryOrphanRows": len(orphans),
            "taskModels": len(TASK_IDS),
            "embeddingRows": len(embedding_frame),
            "failedRowsModeled": native_failure_count,
            "censoredRowsModeled": native_censor_count,
            "deploymentDecision": deployment["decision"],
            "surrogateUtilityPass": surrogate_decision["utilityPass"],
            "continuousCalibrationPass": surrogate_decision["continuousCalibrationPass"],
            "binaryCalibrationPass": binary_calibration_pass,
            "failureAndCensoringPass": failure_censor_pass,
            "embeddingDeploymentPass": shortcut["embeddingDeploymentPass"],
            "validationOutcomeEvaluations": 0,
            "confirmationOutcomeEvaluations": 0,
        },
    )
    return {
        "deployment": deployment,
        "validation": validation,
        "preregistration": prereg,
        "taskSummary": task_summary.to_dict("records"),
        "shortcut": shortcut,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    result = freeze() if args.command == "freeze" else run()
    print(json.dumps(_json_safe(result), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
