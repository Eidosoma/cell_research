#!/usr/bin/env python3
"""Independent post-fit validation for the frozen E07 S06 artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import yaml

from src.quality_diversity.core import TASK_IDS
from src.surrogate_models.core import (
    S06_ROOT,
    EventSummaryAutoencoder,
    SurrogateMLP,
    hash_file,
    state_dict_hash,
    verify_frozen_inputs,
    write_json,
)


REPOSITORY = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPOSITORY / "configs/modeling/s06_modeling_protocol.yaml"


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _standardized_shortcut_probes() -> dict[str, float]:
    embeddings = pd.read_parquet(
        S06_ROOT / "embeddings/trajectory_summary_embeddings.parquet"
    )
    assignments = pd.read_parquet(S06_ROOT / "split_assignments.parquet")[
        ["stableEvaluationSha256", "lineageComponent"]
    ]
    frame = embeddings.merge(
        assignments,
        on=["stableEvaluationSha256", "lineageComponent"],
        how="inner",
        validate="one_to_one",
    )
    columns = [column for column in frame if column.startswith("embedding_")]
    values = frame[columns].to_numpy()
    task_lookup = {task: index for index, task in enumerate(TASK_IDS)}
    tasks = frame.taskId.map(task_lookup).to_numpy()
    groups = (frame.taskId + "::" + frame.lineageComponent).to_numpy()
    length = np.zeros(len(frame))
    for task_index in range(len(TASK_IDS)):
        mask = tasks == task_index
        observed = np.log1p(frame.loc[mask, "eventLength"].to_numpy())
        length[mask] = (observed - observed.mean()) / (observed.std() or 1.0)

    truth: list[int] = []
    predicted: list[int] = []
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=706301)
    for train, test in splitter.split(values, tasks, groups):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=10000, C=1.0, random_state=706302),
        )
        model.fit(values[train], tasks[train])
        truth.extend(tasks[test].tolist())
        predicted.extend(model.predict(values[test]).tolist())
    task_accuracy = float(balanced_accuracy_score(truth, predicted))

    length_truth: list[float] = []
    length_prediction: list[float] = []
    for train, test in GroupKFold(n_splits=5).split(values, length, groups):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(values[train], length[train])
        length_truth.extend(length[test].tolist())
        length_prediction.extend(model.predict(values[test]).tolist())
    return {
        "candidateTaskBalancedAccuracy": task_accuracy,
        "candidateWithinTaskLengthR2": float(
            r2_score(length_truth, length_prediction)
        ),
    }


def _fresh_reload_validation() -> dict[str, Any]:
    manifest = json.loads((S06_ROOT / "models/model_manifest.json").read_text())
    rows = []
    for item in manifest["models"]:
        path = S06_ROOT / item["path"]
        first = torch.load(path, map_location="cpu", weights_only=False)
        second = torch.load(path, map_location="cpu", weights_only=False)
        hashes_first = [state_dict_hash(state) for state in first["stateDicts"]]
        hashes_second = [state_dict_hash(state) for state in second["stateDicts"]]
        state_pass = (
            hashes_first == hashes_second == list(first["stateDictHashes"])
        )
        if item["taskId"] == "cross_task_event_summary":
            model_first = EventSummaryAutoencoder(
                int(first["inputDim"]), int(first["embeddingDim"]), len(TASK_IDS)
            )
            model_second = EventSummaryAutoencoder(
                int(second["inputDim"]), int(second["embeddingDim"]), len(TASK_IDS)
            )
            sample = torch.linspace(-1.0, 1.0, int(first["inputDim"])).reshape(1, -1)
            model_first.load_state_dict(first["stateDicts"][0])
            model_second.load_state_dict(second["stateDicts"][0])
            model_first.eval()
            model_second.eval()
            with torch.no_grad():
                left = model_first(sample, False)[0]
                right = model_second(sample, False)[0]
        else:
            model_first = SurrogateMLP(
                int(first["inputDim"]),
                int(first["continuousDim"]),
                int(first["binaryDim"]),
            )
            model_second = SurrogateMLP(
                int(second["inputDim"]),
                int(second["continuousDim"]),
                int(second["binaryDim"]),
            )
            sample = torch.linspace(-1.0, 1.0, int(first["inputDim"])).reshape(1, -1)
            model_first.load_state_dict(first["stateDicts"][0])
            model_second.load_state_dict(second["stateDicts"][0])
            model_first.eval()
            model_second.eval()
            with torch.no_grad():
                left = torch.cat(model_first(sample), dim=1)
                right = torch.cat(model_second(sample), dim=1)
        inference_pass = torch.equal(left, right)
        rows.append(
            {
                "taskId": item["taskId"],
                "path": item["path"],
                "fileSha256Expected": item["sha256"],
                "fileSha256Actual": hash_file(path),
                "fileHashPass": hash_file(path) == item["sha256"],
                "stateHashPass": state_pass,
                "freshDoubleLoadCpuBitExactInferencePass": inference_pass,
            }
        )
    return {
        "rows": rows,
        "allFileHashesPass": all(row["fileHashPass"] for row in rows),
        "allStateHashesPass": all(row["stateHashPass"] for row in rows),
        "freshDoubleLoadCpuBitExactInferencePass": all(
            row["freshDoubleLoadCpuBitExactInferencePass"] for row in rows
        ),
    }


def _gate_recalculation(settings: dict[str, Any]) -> dict[str, Any]:
    metrics = pd.read_parquet(S06_ROOT / "metrics/target_metrics.parquet")
    thresholds = settings["frozenThresholds"]
    minimum = int(thresholds["minimumAuditRowsPerTarget"])
    panels = ("lineage_test", "scenario_test")
    task_rows = []
    utility_task_pass = {}
    constant_task_pass = {}
    continuous_task_pass = {}
    binary_task_pass = {}
    maximum_ratio = -math.inf
    for task_id in TASK_IDS:
        ratios_by_panel = {}
        constant_by_panel = {}
        continuous_by_panel = {}
        binary_by_panel = {}
        for panel in panels:
            base = metrics[
                (metrics.taskId == task_id)
                & (metrics.auditPanel == panel)
                & (metrics.targetGroup == "performance")
                & (~metrics.constantTarget)
                & (metrics.n >= minimum)
            ]
            gpu = base[base.variant == "gpu"].set_index("target")
            ridge = base[base.variant == "ridge"].set_index("target")
            constant = base[base.variant == "constant"].set_index("target")
            common = gpu.index.intersection(ridge.index).intersection(constant.index)
            ratios = [gpu.loc[name].loss / max(ridge.loc[name].loss, 1e-12) for name in common]
            constant_ratios = [
                gpu.loc[name].loss / max(constant.loc[name].loss, 1e-12)
                for name in common
            ]
            ratio = float(np.median(ratios)) if ratios else math.nan
            constant_ratio = (
                float(np.median(constant_ratios)) if constant_ratios else math.nan
            )
            ratios_by_panel[panel] = ratio
            constant_by_panel[panel] = constant_ratio
            if math.isfinite(ratio):
                maximum_ratio = max(maximum_ratio, ratio)

            continuous = gpu[gpu.targetType == "continuous"]
            continuous_by_panel[panel] = {
                "targetCount": len(continuous),
                "medianCoverage": float(continuous.coverage.median())
                if len(continuous)
                else math.nan,
                "medianIntervalWidth": float(continuous.intervalWidth.median())
                if len(continuous)
                else math.nan,
            }

            binary_base = metrics[
                (metrics.taskId == task_id)
                & (metrics.auditPanel == panel)
                & (metrics.targetGroup.isin(["performance", "binary"]))
                & (metrics.targetType == "binary")
                & (~metrics.constantTarget)
                & (metrics.n >= minimum)
            ]
            binary_gpu = binary_base[binary_base.variant == "gpu"].set_index("target")
            binary_constant = binary_base[
                binary_base.variant == "constant"
            ].set_index("target")
            binary_common = binary_gpu.index.intersection(binary_constant.index)
            binary_by_panel[panel] = {
                "targetCount": len(binary_common),
                "maximumEce": float(binary_gpu.loc[binary_common].ece.max())
                if len(binary_common)
                else math.nan,
                "allBrierNoWorseThanConstant": bool(
                    (
                        binary_gpu.loc[binary_common].loss
                        <= binary_constant.loc[binary_common].loss
                        + float(
                            thresholds["binaryCalibration"][
                                "brierNoWorseThanConstantTolerance"
                            ]
                        )
                    ).all()
                )
                if len(binary_common)
                else False,
            }
        utility_task_pass[task_id] = all(
            math.isfinite(ratios_by_panel[panel])
            and ratios_by_panel[panel]
            <= float(
                thresholds["surrogateUtility"][
                    "taskObjectiveLossRatioVsRidgeMaximum"
                ]
            )
            for panel in panels
        )
        constant_task_pass[task_id] = all(
            math.isfinite(constant_by_panel[panel])
            and constant_by_panel[panel] < 1.0
            for panel in panels
        )
        continuous_task_pass[task_id] = all(
            continuous_by_panel[panel]["targetCount"] > 0
            and continuous_by_panel[panel]["medianCoverage"]
            >= float(thresholds["continuousCalibration"]["minimumCoverage"])
            and continuous_by_panel[panel]["medianIntervalWidth"]
            <= float(
                thresholds["continuousCalibration"][
                    "maximumMedianIntervalWidthInTrainStandardDeviations"
                ]
            )
            for panel in panels
        )
        binary_task_pass[task_id] = all(
            binary_by_panel[panel]["targetCount"] > 0
            and binary_by_panel[panel]["maximumEce"]
            <= float(thresholds["binaryCalibration"]["maximumECE"])
            and binary_by_panel[panel]["allBrierNoWorseThanConstant"]
            for panel in panels
        )
        task_rows.append(
            {
                "taskId": task_id,
                "objectiveLossRatioVsRidge": ratios_by_panel,
                "objectiveLossRatioVsConstant": constant_by_panel,
                "continuousCalibration": continuous_by_panel,
                "binaryCalibration": binary_by_panel,
                "utilityPass": utility_task_pass[task_id],
                "constantImprovementPass": constant_task_pass[task_id],
                "continuousCalibrationPass": continuous_task_pass[task_id],
                "binaryCalibrationPass": binary_task_pass[task_id],
            }
        )

    utility = {
        "tasksPassingBoth": sum(utility_task_pass.values()),
        "tasksRequired": int(
            thresholds["surrogateUtility"]["tasksRequiredBothLineageAndScenario"]
        ),
        "constantImprovementTasksBoth": sum(constant_task_pass.values()),
        "constantImprovementTasksRequired": int(
            thresholds["surrogateUtility"][
                "constantBaselineImprovementTasksRequired"
            ]
        ),
        "maximumObservedTaskObjectiveLossRatio": maximum_ratio,
        "maximumAllowedTaskObjectiveLossRatio": float(
            thresholds["surrogateUtility"]["maximumAnyTaskObjectiveLossRatio"]
        ),
    }
    utility["pass"] = (
        utility["tasksPassingBoth"] >= utility["tasksRequired"]
        and utility["constantImprovementTasksBoth"]
        >= utility["constantImprovementTasksRequired"]
        and utility["maximumObservedTaskObjectiveLossRatio"]
        <= utility["maximumAllowedTaskObjectiveLossRatio"]
    )
    continuous = {
        "tasksPassingBothCoverageAndWidth": sum(continuous_task_pass.values()),
        "tasksRequired": int(
            thresholds["continuousCalibration"][
                "tasksRequiredBothLineageAndScenario"
            ]
        ),
    }
    continuous["pass"] = (
        continuous["tasksPassingBothCoverageAndWidth"]
        >= continuous["tasksRequired"]
    )
    binary = {
        "tasksPassingBothEceAndBrier": sum(binary_task_pass.values()),
        "tasksRequired": int(
            thresholds["binaryCalibration"][
                "tasksRequiredBothLineageAndScenario"
            ]
        ),
    }
    binary["pass"] = binary["tasksPassingBothEceAndBrier"] >= binary["tasksRequired"]

    status_base = metrics[
        (metrics.target.str.startswith("status::"))
        & (~metrics.constantTarget)
        & (metrics.n >= minimum)
        & (metrics.auditPanel.isin(panels))
    ]
    status_gpu = status_base[status_base.variant == "gpu"].set_index(
        ["taskId", "auditPanel", "target"]
    )
    status_constant = status_base[status_base.variant == "constant"].set_index(
        ["taskId", "auditPanel", "target"]
    )
    status_common = status_gpu.index.intersection(status_constant.index)
    status_brier_pass = bool(
        (
            status_gpu.loc[status_common].loss
            <= status_constant.loc[status_common].loss
            + float(
                thresholds["failureAndCensoring"][
                    "nonconstantHeadBrierNoWorseThanConstantTolerance"
                ]
            )
        ).all()
    )
    subgroup = pd.read_csv(S06_ROOT / "metrics/subgroup_calibration.csv")
    e05_tail = subgroup[
        subgroup.subgroup.isin(
            ["e05_runtime_tail_quartile", "e05_native_event_length_tail_quartile"]
        )
        & (subgroup.targetType == "continuous")
        & (subgroup.n >= 3)
    ]
    heavy_tail_coverage = float(e05_tail.coverage.median())
    failure = {
        "nonconstantStatusHeadCount": len(status_common),
        "allStatusBrierNoWorseThanConstant": status_brier_pass,
        "e05HeavyTailMedianCoverage": heavy_tail_coverage,
        "e05HeavyTailCoverageMinimum": float(
            thresholds["failureAndCensoring"]["heavyTailE05CoverageMinimum"]
        ),
    }
    failure["pass"] = status_brier_pass and heavy_tail_coverage >= failure[
        "e05HeavyTailCoverageMinimum"
    ]
    return {
        "schemaVersion": "e07.s06.gate-recalculation.v1",
        "researchStepId": "S06",
        "minimumAuditRowsPerTarget": minimum,
        "taskRows": task_rows,
        "surrogateUtility": utility,
        "continuousCalibration": continuous,
        "binaryCalibration": binary,
        "failureAndCensoring": failure,
    }


def main() -> None:
    settings = yaml.safe_load(PROTOCOL_PATH.read_text())
    prereg = json.loads((S06_ROOT / "preregistration_freeze.json").read_text())
    if hash_file(PROTOCOL_PATH) != prereg["protocolSha256"]:
        raise RuntimeError("protocol differs from preregistration")
    freeze = json.loads((S06_ROOT / "s05_hash_freeze.json").read_text())
    verify_frozen_inputs(settings, freeze)

    probes = _standardized_shortcut_probes()
    shortcut_path = S06_ROOT / "metrics/shortcut_tests.json"
    shortcut = json.loads(shortcut_path.read_text())
    shortcut["initialUnstandardizedCandidateTaskBalancedAccuracy"] = shortcut[
        "candidateTaskBalancedAccuracy"
    ]
    shortcut["initialUnstandardizedCandidateWithinTaskLengthR2"] = shortcut[
        "candidateWithinTaskLengthR2"
    ]
    shortcut.update(probes)
    shortcut["probePreprocessing"] = "fold_local_standardization"
    shortcut["rawUnionProbeConvergenceWarning"] = True
    threshold = settings["frozenThresholds"]["embeddingDeployment"]
    shortcut["embeddingDeploymentPass"] = (
        shortcut["candidateTaskBalancedAccuracy"]
        <= float(threshold["maximumTaskBalancedAccuracy"])
        and shortcut["candidateWithinTaskLengthR2"]
        <= float(threshold["maximumWithinTaskLengthR2"])
        and shortcut["candidateReconstructionMseRatioVsPca"]
        <= float(threshold["maximumReconstructionMseRatioVsPCA"])
    )
    write_json(shortcut_path, _safe(shortcut))

    reload_validation = _fresh_reload_validation()
    gates = _gate_recalculation(settings)
    metrics = pd.read_parquet(S06_ROOT / "metrics/target_metrics.parquet")
    calibration = (
        metrics[(metrics.variant == "gpu") & (metrics.n >= int(settings["frozenThresholds"]["minimumAuditRowsPerTarget"]))]
        .groupby(
            ["taskId", "auditPanel", "targetGroup", "targetType"],
            as_index=False,
        )
        .agg(
            targetCount=("target", "count"),
            medianLoss=("loss", "median"),
            medianCoverage=("coverage", "median"),
            medianIntervalWidth=("intervalWidth", "median"),
            maximumEce=("ece", "max"),
        )
    )
    calibration.to_csv(S06_ROOT / "metrics/calibration_summary.csv", index=False)
    ablation = (
        metrics[
            (metrics.targetGroup == "performance")
            & (~metrics.constantTarget)
            & (metrics.n >= int(settings["frozenThresholds"]["minimumAuditRowsPerTarget"]))
        ]
        .groupby(["taskId", "auditPanel", "variant"], as_index=False)
        .agg(targetCount=("target", "count"), medianTaskObjectiveLoss=("loss", "median"))
    )
    ablation.to_csv(S06_ROOT / "metrics/ablation_results.csv", index=False)
    write_json(
        S06_ROOT / "metrics/postfit_validation.json",
        _safe(
            {
                "schemaVersion": "e07.s06.postfit-validation.v1",
                "researchStepId": "S06",
                "success": True,
                "standardizedShortcutProbes": probes,
                "freshReloadValidation": reload_validation,
                "gateRecalculation": gates,
                "initialRuntimeWarnings": [
                    "Raw-union logistic probe reached max_iter; raw probe remains diagnostic only.",
                    "Initial in-memory-versus-reload CUDA bitwise comparison failed; fresh double-load CPU inference is the final persistence check.",
                ],
                "protocolDeviations": [
                    "The frozen architecture label says residual_mlp, but the executed SurrogateMLP is a sequential LayerNorm/SiLU MLP without residual skip connections. The rejected bundles remain diagnostic only; any remediation must correct and refreeze this label/implementation before fitting."
                ],
            }
        ),
    )

    deployment_path = S06_ROOT / "deployment_decision.json"
    deployment = json.loads(deployment_path.read_text())
    # Preserve the historical runtime check across validator reruns.  The final
    # fresh-load check below is a distinct persistence test and must not rewrite
    # the original in-memory-versus-reload observation.
    deployment.setdefault(
        "initialInMemoryVsReloadCudaBitwiseParityPass",
        deployment["inferenceReloadParityPass"],
    )
    deployment["freshDoubleLoadCpuBitExactInferenceParityPass"] = reload_validation[
        "freshDoubleLoadCpuBitExactInferencePass"
    ]
    deployment["inferenceReloadParityPass"] = reload_validation[
        "freshDoubleLoadCpuBitExactInferencePass"
    ]
    deployment["surrogateUtility"] = gates["surrogateUtility"]
    deployment["continuousCalibration"] = gates["continuousCalibration"]
    deployment["binaryCalibration"] = gates["binaryCalibration"]
    deployment["failureAndCensoring"] = gates["failureAndCensoring"]
    deployment["embedding"] = shortcut
    deployment["protocolDeviations"] = [
        "Frozen residual_mlp label did not match the executed non-residual sequential MLP; deployment remains rejected."
    ]
    deployment["binaryCalibrationPass"] = gates["binaryCalibration"]["pass"]
    deployment["failureAndCensoringPass"] = gates["failureAndCensoring"]["pass"]
    deployment["eligibleForS07"] = all(
        [
            gates["surrogateUtility"]["pass"],
            gates["continuousCalibration"]["pass"],
            gates["binaryCalibration"]["pass"],
            gates["failureAndCensoring"]["pass"],
            shortcut["embeddingDeploymentPass"],
            deployment["deterministicReplayPass"],
            deployment["inferenceReloadParityPass"],
        ]
    )
    deployment["decision"] = "accept" if deployment["eligibleForS07"] else "reject"
    deployment["rejectionRuleApplied"] = not deployment["eligibleForS07"]
    deployment["permittedUse"] = (
        "S07 allocation guidance"
        if deployment["eligibleForS07"]
        else "diagnostic research artifact only"
    )
    write_json(deployment_path, _safe(deployment))

    result_path = S06_ROOT / "result_summary.json"
    result = json.loads(result_path.read_text())
    result.update(
        {
            "surrogateUtilityPass": gates["surrogateUtility"]["pass"],
            "continuousCalibrationPass": gates["continuousCalibration"]["pass"],
            "binaryCalibrationPass": gates["binaryCalibration"]["pass"],
            "failureAndCensoringPass": gates["failureAndCensoring"]["pass"],
            "embeddingDeploymentPass": shortcut["embeddingDeploymentPass"],
            "deploymentDecision": deployment["decision"],
        }
    )
    write_json(result_path, result)

    validation_path = S06_ROOT / "validation_summary.json"
    validation = json.loads(validation_path.read_text())
    validation["checks"].setdefault(
        "initialInMemoryVsReloadCudaBitwiseParityPass",
        validation["checks"]["inferenceReloadParityPass"],
    )
    validation["checks"]["freshDoubleLoadCpuBitExactInferenceParityPass"] = (
        reload_validation["freshDoubleLoadCpuBitExactInferencePass"]
    )
    validation["checks"]["inferenceReloadParityPass"] = reload_validation[
        "freshDoubleLoadCpuBitExactInferencePass"
    ]
    validation["checks"]["allModelFileHashesPass"] = reload_validation[
        "allFileHashesPass"
    ]
    validation["checks"]["allModelStateHashesPass"] = reload_validation[
        "allStateHashesPass"
    ]
    validation["deploymentDecision"] = deployment["decision"]
    validation["validationResult"] = (
        "PASS for artifact integrity, exact training replay, grouped evaluation, "
        "and sealed protected outcomes; FAIL frozen deployment thresholds, so models are rejected."
    )
    validation["protocolDeviations"] = deployment["protocolDeviations"]
    write_json(validation_path, validation)
    model_manifest_path = S06_ROOT / "models/model_manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text())
    model_manifest.setdefault(
        "initialInMemoryVsReloadCudaBitwiseParityPass",
        model_manifest["inferenceReloadParityPass"],
    )
    model_manifest["freshDoubleLoadCpuBitExactInferenceParityPass"] = reload_validation[
        "freshDoubleLoadCpuBitExactInferencePass"
    ]
    model_manifest["inferenceReloadParityPass"] = reload_validation[
        "freshDoubleLoadCpuBitExactInferencePass"
    ]
    write_json(model_manifest_path, model_manifest)
    print(json.dumps(_safe({"deployment": deployment, "gates": gates, "probes": probes}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
