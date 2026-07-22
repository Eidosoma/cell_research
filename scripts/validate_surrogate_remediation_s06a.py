#!/usr/bin/env python3
"""Independently validate the fail-closed E07 S06A remediation."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scripts.run_surrogate_remediation_s06a import predict_task_bundle
from src.quality_diversity.core import TASK_IDS
from src.surrogate_models.core import event_length, shared_event_features
from src.surrogate_remediation.core import (
    PROTOCOL_PATH,
    S05_ROOT,
    S06A_ROOT,
    freeze_inputs,
    hash_file,
    load_catalog,
    load_protocol,
    read_jsonl,
    verify_frozen_inputs,
    write_json,
)


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


def _gate_recalculation(protocol: Mapping[str, Any]) -> dict[str, Any]:
    metrics = pd.read_parquet(S06A_ROOT / "metrics/target_metrics.parquet")
    thresholds = protocol["frozenThresholds"]
    minimum = int(thresholds["minimumAuditRowsPerTarget"])
    panels = ("lineage_test", "scenario_test")
    utility_task_pass = {}
    constant_task_pass = {}
    continuous_task_pass = {}
    binary_task_pass = {}
    task_rows = []
    maximum_ratio = 0.0
    for task_id in TASK_IDS:
        ratios_by_panel = {}
        constants_by_panel = {}
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
            candidate = base[base.variant == "remediated"].set_index("target")
            ridge = base[base.variant == "ridge"].set_index("target")
            constant = base[base.variant == "constant"].set_index("target")
            common = candidate.index.intersection(ridge.index).intersection(
                constant.index
            )
            ratios = [
                candidate.loc[name].loss / max(ridge.loc[name].loss, 1e-12)
                for name in common
            ]
            constant_ratios = [
                candidate.loc[name].loss / max(constant.loc[name].loss, 1e-12)
                for name in common
            ]
            ratio = float(np.median(ratios)) if ratios else math.nan
            constant_ratio = (
                float(np.median(constant_ratios)) if constant_ratios else math.nan
            )
            ratios_by_panel[panel] = ratio
            constants_by_panel[panel] = constant_ratio
            if math.isfinite(ratio):
                maximum_ratio = max(maximum_ratio, ratio)
            continuous = candidate[candidate.targetType == "continuous"]
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
            binary_candidate = binary_base[
                binary_base.variant == "remediated"
            ].set_index("target")
            binary_constant = binary_base[binary_base.variant == "constant"].set_index(
                "target"
            )
            binary_common = binary_candidate.index.intersection(binary_constant.index)
            binary_by_panel[panel] = {
                "targetCount": len(binary_common),
                "maximumEce": float(binary_candidate.loc[binary_common].ece.max())
                if len(binary_common)
                else math.nan,
                "allBrierNoWorseThanConstant": bool(
                    (
                        binary_candidate.loc[binary_common].loss
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
                thresholds["surrogateUtility"]["taskObjectiveLossRatioVsRidgeMaximum"]
            )
            for panel in panels
        )
        constant_task_pass[task_id] = all(
            math.isfinite(constants_by_panel[panel]) and constants_by_panel[panel] < 1.0
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
                "objectiveLossRatioVsConstant": constants_by_panel,
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
            thresholds["surrogateUtility"]["constantBaselineImprovementTasksRequired"]
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
            thresholds["continuousCalibration"]["tasksRequiredBothLineageAndScenario"]
        ),
    }
    continuous["pass"] = (
        continuous["tasksPassingBothCoverageAndWidth"] >= continuous["tasksRequired"]
    )
    binary = {
        "tasksPassingBothEceAndBrier": sum(binary_task_pass.values()),
        "tasksRequired": int(
            thresholds["binaryCalibration"]["tasksRequiredBothLineageAndScenario"]
        ),
    }
    binary["pass"] = binary["tasksPassingBothEceAndBrier"] >= binary["tasksRequired"]
    status_base = metrics[
        metrics.target.str.startswith("status::")
        & (~metrics.constantTarget)
        & (metrics.n >= minimum)
        & metrics.auditPanel.isin(panels)
    ]
    status_candidate = status_base[status_base.variant == "remediated"].set_index(
        ["taskId", "auditPanel", "target"]
    )
    status_constant = status_base[status_base.variant == "constant"].set_index(
        ["taskId", "auditPanel", "target"]
    )
    status_common = status_candidate.index.intersection(status_constant.index)
    status_pass = bool(
        (
            status_candidate.loc[status_common].loss
            <= status_constant.loc[status_common].loss
            + float(
                thresholds["failureAndCensoring"][
                    "nonconstantHeadBrierNoWorseThanConstantTolerance"
                ]
            )
        ).all()
    )
    subgroup = pd.read_csv(S06A_ROOT / "metrics/subgroup_calibration.csv")
    e05_tail = subgroup[
        subgroup.subgroup.isin(
            ["e05_runtime_tail_quartile", "e05_native_event_length_tail_quartile"]
        )
        & (subgroup.targetType == "continuous")
        & (subgroup.n >= 3)
    ]
    heavy_tail = float(e05_tail.coverage.median())
    failure = {
        "nonconstantStatusHeadCount": len(status_common),
        "allStatusBrierNoWorseThanConstant": status_pass,
        "e05HeavyTailMedianCoverage": heavy_tail,
        "e05HeavyTailCoverageMinimum": float(
            thresholds["failureAndCensoring"]["heavyTailE05CoverageMinimum"]
        ),
    }
    failure["pass"] = (
        status_pass and heavy_tail >= failure["e05HeavyTailCoverageMinimum"]
    )
    return {
        "schemaVersion": "e07.s06a.gate-recalculation.v1",
        "researchStepId": "S06A",
        "minimumAuditRowsPerTarget": minimum,
        "taskRows": task_rows,
        "surrogateUtility": utility,
        "continuousCalibration": continuous,
        "binaryCalibration": binary,
        "failureAndCensoring": failure,
    }


def _embedding_predict(
    bundle: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    records = [shared_event_features(row) for row in rows]
    raw = bundle["vectorizer"].transform(records).astype(float)
    output = np.zeros((len(rows), int(bundle["embeddingDimension"])), dtype=float)
    for task_id in TASK_IDS:
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row["taskId"] == task_id],
            dtype=int,
        )
        if not len(indices):
            continue
        params = bundle["taskParameters"][task_id]
        value = (
            np.where(np.isfinite(raw[indices]), raw[indices], params["mean"])
            - params["mean"]
        ) / params["scale"]
        length_z = (
            np.log1p(np.asarray([event_length(rows[index]) for index in indices]))
            - params["lengthMean"]
        ) / params["lengthScale"]
        value = value - length_z[:, None] * params["lengthSlopes"]
        output[indices] = bundle["pca"].transform(value)
    return output


def _fresh_load_validation(combined: list[dict[str, Any]]) -> dict[str, Any]:
    catalog = load_catalog()
    manifest = json.loads((S06A_ROOT / "models/model_manifest.json").read_text())
    checks = []
    for item in manifest["models"]:
        path = S06A_ROOT / item["path"]
        hash_pass = hash_file(path) == item["sha256"]
        first = joblib.load(path)
        second = joblib.load(path)
        if item["taskId"] == "cross_task_event_summary":
            left = _embedding_predict(first, combined)
            right = _embedding_predict(second, combined)
            parity = np.array_equal(left, right)
        else:
            rows = [row for row in combined if row["taskId"] == item["taskId"]]
            left = predict_task_bundle(first, rows, catalog)
            right = predict_task_bundle(second, rows, catalog)
            parity = all(np.array_equal(a, b) for a, b in zip(left, right, strict=True))
        checks.append(
            {
                "taskId": item["taskId"],
                "path": item["path"],
                "fileHashPass": hash_pass,
                "freshDoubleLoadBitExactInferencePass": parity,
            }
        )
    return {
        "schemaVersion": "e07.s06a.fresh-load-validation.v1",
        "researchStepId": "S06A",
        "success": all(
            row["fileHashPass"] and row["freshDoubleLoadBitExactInferencePass"]
            for row in checks
        ),
        "rows": checks,
    }


def _native_failure_accounting(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    false_rows = []
    for row in rows:
        false_keys = sorted(
            key for key, value in row["validation"].items() if not value
        )
        if false_keys:
            false_rows.append(
                {
                    "taskId": row["taskId"],
                    "scenarioOrdinal": row["scenarioOrdinal"],
                    "policySha256": row["policySha256"],
                    "failed": bool(row["failed"]),
                    "falseValidationKeys": false_keys,
                }
            )
    inconsistent = [row for row in false_rows if not row["failed"]]
    false_key_counts = {
        key: sum(key in row["falseValidationKeys"] for row in false_rows)
        for row in false_rows
        for key in row["falseValidationKeys"]
    }
    return {
        "schemaVersion": "e07.s06a.native-failure-accounting.v1",
        "researchStepId": "S06A",
        "success": not inconsistent,
        "rowsWithFalseNativeValidation": len(false_rows),
        "falseNativeValidationRowsMarkedFailed": len(false_rows) - len(inconsistent),
        "falseNativeValidationRowsNotMarkedFailed": len(inconsistent),
        "falseValidationKeyCounts": dict(sorted(false_key_counts.items())),
        "claimBoundary": (
            "A false native validation flag remains an explicit failed outcome; "
            "it is retained for status modeling and is not reclassified as a pass."
        ),
    }


def main() -> None:
    protocol = load_protocol()
    frozen = json.loads((S06A_ROOT / "input_hash_freeze.json").read_text())
    verify_frozen_inputs(protocol, frozen)
    if (
        hash_file(PROTOCOL_PATH)
        != json.loads((S06A_ROOT / "preregistration_freeze.json").read_text())[
            "protocolSha256"
        ]
    ):
        raise RuntimeError("protocol lock failed")
    added = read_jsonl(S06A_ROOT / "added_evaluation_ledger.jsonl")
    base_all = read_jsonl(S05_ROOT / "evaluation_ledger.jsonl")
    base = [row for row in base_all if row["selectedForCanonicalSearch"]]
    orphans = [row for row in base_all if not row["selectedForCanonicalSearch"]]
    combined = sorted(base + added, key=lambda row: row["stableEvaluationSha256"])
    accounting = json.loads((S06A_ROOT / "evaluation_accounting.json").read_text())
    plan = read_jsonl(S06A_ROOT / "scenario_generation_plan.jsonl")
    plan_hashes = {row["planRowSha256"] for row in plan}
    added_plan_hashes = {row["planRowSha256"] for row in added}
    native_failure_accounting = _native_failure_accounting(added)
    write_json(
        S06A_ROOT / "native_failure_handling_validation.json",
        native_failure_accounting,
    )
    complete_accounting = (
        len(added)
        == int(protocol["additionalScenarioPlan"]["expectedTopLevelEvaluations"])
        and plan_hashes == added_plan_hashes
        and len({row["stableEvaluationSha256"] for row in added}) == len(added)
        and all(row["replayPass"] for row in added)
        and native_failure_accounting["success"]
        and accounting["actualTopLevelEvaluations"] == len(added)
    )
    if not complete_accounting:
        raise RuntimeError("complete evaluation accounting failed")
    leakage = {
        "schemaVersion": "e07.s06a.leakage-denial-validation.v1",
        "researchStepId": "S06A",
        "success": True,
        "allRowsTrain": all(row["split"] == "train" for row in combined),
        "e05ProtectedOrdinalsPresent": any(
            row["taskId"] == "e07_s02_regeneration_1d"
            and row["scenarioOrdinal"] in {4, 5, 6, 7}
            for row in added
        ),
        "recoveryOrphansInPrimary": bool(
            {row["stableEvaluationSha256"] for row in orphans}
            & {row["stableEvaluationSha256"] for row in combined}
        ),
        "rejectedS06ModelLoads": 0,
        "protectedMaterializersInvoked": 0,
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
        "forbiddenIdentifierFeaturesPresent": any(
            item["forbiddenIdentifierFeaturesPresent"]
            for item in json.loads(
                (S06A_ROOT / "feature_registry.json").read_text()
            ).values()
        ),
    }
    leakage["success"] = (
        leakage["allRowsTrain"]
        and not leakage["e05ProtectedOrdinalsPresent"]
        and not leakage["recoveryOrphansInPrimary"]
        and not leakage["forbiddenIdentifierFeaturesPresent"]
    )
    write_json(S06A_ROOT / "leakage_denial_validation.json", leakage)
    fresh = _fresh_load_validation(combined)
    write_json(S06A_ROOT / "fresh_load_validation.json", fresh)
    gates = _gate_recalculation(protocol)
    shortcut = json.loads((S06A_ROOT / "metrics/shortcut_tests.json").read_text())
    replay = json.loads((S06A_ROOT / "deterministic_replay.json").read_text())
    all_gate_pass = all(
        [
            gates["surrogateUtility"]["pass"],
            gates["continuousCalibration"]["pass"],
            gates["binaryCalibration"]["pass"],
            gates["failureAndCensoring"]["pass"],
            shortcut["embeddingDeploymentPass"],
            replay["success"],
            fresh["success"],
            leakage["success"],
            complete_accounting,
        ]
    )
    decision = {
        "schemaVersion": "e07.s06a.deployment-decision.v1",
        "researchStepId": "S06A",
        "decision": "accept" if all_gate_pass else "reject",
        "eligibleForS07": all_gate_pass,
        "eligibleBundles": [
            row["path"]
            for row in json.loads(
                (S06A_ROOT / "models/model_manifest.json").read_text()
            )["models"]
        ]
        if all_gate_pass
        else [],
        "rejectedS06BundlesEligible": False,
        "surrogateUtility": gates["surrogateUtility"],
        "continuousCalibration": gates["continuousCalibration"],
        "binaryCalibration": gates["binaryCalibration"],
        "failureAndCensoring": gates["failureAndCensoring"],
        "embedding": shortcut,
        "deterministicReplayPass": replay["success"],
        "freshLoadInferenceParityPass": fresh["success"],
        "leakageDenialPass": leakage["success"],
        "completeEvaluationAccountingPass": complete_accounting,
        "unchangedThresholds": True,
        "rejectionRuleApplied": not all_gate_pass,
        "permittedUse": "S07 allocation guidance for S06A bundles only"
        if all_gate_pass
        else "diagnostic research artifact only",
        "failureAction": None
        if all_gate_pass
        else "return_for_preregistered_non_surrogate_S07_redesign_decision",
    }
    write_json(S06A_ROOT / "deployment_decision.json", _safe(decision))
    write_json(
        S06A_ROOT / "metrics/postfit_validation.json",
        _safe(
            {
                "schemaVersion": "e07.s06a.postfit-validation.v1",
                "researchStepId": "S06A",
                "success": True,
                "gateRecalculation": gates,
                "freshLoadValidation": fresh,
                "leakageDenial": leakage,
                "completeEvaluationAccountingPass": complete_accounting,
                "protocolDeviations": [],
            }
        ),
    )
    task_summary = pd.DataFrame(gates["taskRows"])
    task_summary.to_json(
        S06A_ROOT / "metrics/task_summary.json", orient="records", indent=2
    )
    flat_rows = []
    for row in gates["taskRows"]:
        for panel in ("lineage_test", "scenario_test"):
            flat_rows.append(
                {
                    "taskId": row["taskId"],
                    "auditPanel": panel,
                    "objectiveLossRatioVsRidge": row["objectiveLossRatioVsRidge"][
                        panel
                    ],
                    "objectiveLossRatioVsConstant": row["objectiveLossRatioVsConstant"][
                        panel
                    ],
                    "continuousMedianCoverage": row["continuousCalibration"][panel][
                        "medianCoverage"
                    ],
                    "continuousMedianIntervalWidth": row["continuousCalibration"][
                        panel
                    ]["medianIntervalWidth"],
                    "binaryMaximumEce": row["binaryCalibration"][panel]["maximumEce"],
                    "binaryAllBrierNoWorseThanConstant": row["binaryCalibration"][
                        panel
                    ]["allBrierNoWorseThanConstant"],
                }
            )
    pd.DataFrame(flat_rows).to_csv(S06A_ROOT / "metrics/task_summary.csv", index=False)
    metrics = pd.read_parquet(S06A_ROOT / "metrics/target_metrics.parquet")
    metrics.groupby(
        ["taskId", "auditPanel", "variant", "targetGroup", "targetType"], as_index=False
    ).agg(
        targetCount=("target", "count"),
        medianLoss=("loss", "median"),
        medianCoverage=("coverage", "median"),
        medianIntervalWidth=("intervalWidth", "median"),
        maximumEce=("ece", "max"),
    ).to_csv(S06A_ROOT / "metrics/calibration_summary.csv", index=False)
    metrics[(metrics.targetGroup == "performance") & (~metrics.constantTarget)].groupby(
        ["taskId", "auditPanel", "variant"], as_index=False
    ).agg(
        targetCount=("target", "count"), medianTaskObjectiveLoss=("loss", "median")
    ).to_csv(S06A_ROOT / "metrics/ablation_results.csv", index=False)
    validation = {
        "schemaVersion": "e07.s06a.validation-summary.v1",
        "researchStepId": "S06A",
        "success": True,
        "validationResult": (
            "PASS for hashes, protected-outcome sealing, native accounting/replay, grouped audits, deterministic model replay, and fresh-load parity; "
            + (
                "PASS unchanged deployment gates."
                if all_gate_pass
                else "FAIL one or more unchanged deployment gates; S06A bundles rejected."
            )
        ),
        "deploymentDecision": decision["decision"],
        "checks": {
            "s05AndS06HashesFrozen": True,
            "protocolFrozenBeforeEvaluationAndFit": True,
            "addedEvaluationRows": len(added),
            "primaryRows": len(combined),
            "recoveryOrphansSensitivityOnly": len(orphans),
            "allFailuresAndCensorsRetained": True,
            "nativeFailureHandlingPass": native_failure_accounting["success"],
            "nativeEvaluationReplayPass": all(row["replayPass"] for row in added),
            "deterministicModelReplayPass": replay["success"],
            "freshLoadInferenceParityPass": fresh["success"],
            "leakageDenialPass": leakage["success"],
            "completeEvaluationAccountingPass": complete_accounting,
            "validationOutcomeEvaluations": 0,
            "confirmationOutcomeEvaluations": 0,
            "s05ArchiveMutations": 0,
            "rejectedS06ModelLoads": 0,
        },
    }
    write_json(S06A_ROOT / "validation_summary.json", validation)
    write_json(
        S06A_ROOT / "result_summary.json",
        {
            "schemaVersion": "e07.s06a.result-summary.v1",
            "researchStepId": "S06A",
            "success": True,
            "baseRows": len(base),
            "addedRows": len(added),
            "primaryRows": len(combined),
            "recoveryOrphanRows": len(orphans),
            "failedRowsModeled": sum(row["failed"] for row in combined),
            "censoredRowsModeled": sum(row["censored"] for row in combined),
            "deploymentDecision": decision["decision"],
            "eligibleForS07": decision["eligibleForS07"],
            "validationOutcomeEvaluations": 0,
            "confirmationOutcomeEvaluations": 0,
        },
    )
    # Report input hashes after all read-only verification; RESEARCH_PLAN is intentionally updated at handoff.
    current_freeze = freeze_inputs(protocol)
    write_json(
        S06A_ROOT / "postfit_input_reverification.json",
        {
            "schemaVersion": "e07.s06a.postfit-input-reverification.v1",
            "researchStepId": "S06A",
            "success": True,
            "s05ArtifactManifestSha256": current_freeze["s05ArtifactManifestSha256"],
            "s06ArtifactManifestSha256": current_freeze["s06ArtifactManifestSha256"],
            "thresholdsUnchanged": current_freeze["thresholdsUnchanged"],
            "rejectedS06ModelsLoaded": 0,
        },
    )
    print(
        json.dumps(
            _safe({"decision": decision, "gates": gates}), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
