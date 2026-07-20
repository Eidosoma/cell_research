#!/usr/bin/env python3
"""Execute the frozen E06 S10 perturbation screen and held-out confirmation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from morph2d.baseline import load_baseline_assets
from morph2d.grammar import score_grid
from morph2d.perturbations import (
    PERTURBATION_CATALOG_VERSION,
    count_changing_feasibility_records,
    run_condition_task,
    run_perturbation_once,
    scenario_identity,
    sha256_value,
    simulated_target_lesion_pairs,
)
from morph2d.targets import evaluate_success


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = Path("/artifacts/research_steps/S10")
DEFAULT_CACHE = Path("/cache/e06_s10")
CATALOG_PATH = ROOT / "configs/morphologies/perturbation_catalog.yaml"
JSON_INDENT = 2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=JSON_INDENT, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_catalog(catalog: Mapping[str, Any]) -> list[dict[str, str]]:
    if catalog["schemaVersion"] != PERTURBATION_CATALOG_VERSION:
        raise ValueError("S10 perturbation catalog schema mismatch")
    if catalog["researchStepId"] != "S10":
        raise ValueError("catalog is not scoped to S10")
    pairs = simulated_target_lesion_pairs(catalog)
    simulation = catalog["simulation"]
    expected_pairs = int(simulation["simulatedTargetLesionPairs"])
    arm_conditions = len(pairs) * len(catalog["severities"]) * len(simulation["arms"])
    expected_arm_conditions = int(simulation["exploratoryConditionArmCount"])
    expected_runs = int(simulation["exploratoryRunCount"])
    replicates = int(simulation["exploratoryReplicatesPerConditionArm"])
    if len(pairs) != expected_pairs or arm_conditions != expected_arm_conditions:
        raise ValueError("expanded S10 matrix does not equal 33 pairs/132 arms")
    if arm_conditions * replicates != expected_runs:
        raise ValueError("expanded S10 matrix does not equal 33,000 screen runs")
    if simulation["stopping"] != "fixed_event_budget_only":
        raise ValueError("S10 stopping contract changed")
    if catalog["backend"]["production"] != "canonical_s07_cpu_oracle_fallback":
        raise ValueError("S10 may not widen the validated GPU data plane")
    if catalog["promotion"]["runtimeAndWallClockForbiddenFromSelection"] is not True:
        raise ValueError("runtime must be excluded from S10 promotion")
    if int(simulation["maximumConfirmationRunCount"]) != 2 * int(
        simulation["maximumPromotedContrasts"]
    ) * int(simulation["confirmationPairsPerPromotedContrast"]):
        raise ValueError("confirmation maximum is inconsistent")
    _context, targets, grammars, environments = load_baseline_assets()
    target_matrix = {item["targetId"]: item for item in catalog["targetPolicyMatrix"]}
    if set(target_matrix) != set(targets):
        raise ValueError("S10 target matrix must cover all seven S01 targets")
    grammar_target = {item.target_id: item.grammar_id for item in grammars.values()}
    for target_id, item in target_matrix.items():
        if item["grammarId"] != grammar_target[target_id]:
            raise ValueError("S10 target/grammar binding changed")
        environment = environments[target_id]
        if environment.geometry != "square" or environment.boundary_mode != "bounded":
            raise ValueError("S10 completion is bounded-square only")
    infeasible = count_changing_feasibility_records(catalog)
    if len(infeasible) != 28 or any(item["targetFeasible"] for item in infeasible):
        raise ValueError("count-changing feasibility matrix changed")
    return pairs


def condition_id(specification: Mapping[str, Any], phase: str) -> str:
    digest = sha256_value(
        "E06/S10/condition/v1",
        {
            "phase": phase,
            "targetId": specification["targetId"],
            "lesionId": specification["lesionId"],
            "severity": specification["severity"],
            "arm": specification["arm"],
        },
    )
    return f"{phase[:4]}-{digest[:20]}"


def trace_run_ids(
    split: str,
    target_id: str,
    lesion_id: str,
    severity: str,
    arm: str,
    replicates: Iterable[int],
    fraction: float,
) -> list[str]:
    run_ids = [
        scenario_identity(split, target_id, lesion_id, severity, int(replicate), arm)[
            "runId"
        ]
        for replicate in replicates
    ]
    return sorted(run_ids)[: max(1, math.ceil(fraction * len(run_ids)))]


def build_task(
    catalog: Mapping[str, Any],
    specification: Mapping[str, Any],
    *,
    phase: str,
    split: str,
    replicate_count: int,
    event_budget: int,
) -> dict[str, Any]:
    replicates = list(range(replicate_count))
    task = {
        "catalog": dict(catalog),
        "phase": phase,
        "split": split,
        **dict(specification),
        "replicates": replicates,
        "eventBudget": int(event_budget),
    }
    task["conditionId"] = condition_id(task, phase)
    task["traceRunIds"] = trace_run_ids(
        split,
        str(task["targetId"]),
        str(task["lesionId"]),
        str(task["severity"]),
        str(task["arm"]),
        replicates,
        float(catalog["traceAndReplay"]["traceSampleFraction"]),
    )
    return task


def screen_tasks(
    catalog: Mapping[str, Any], pairs: list[dict[str, str]]
) -> list[dict[str, Any]]:
    simulation = catalog["simulation"]
    tasks = []
    for pair in pairs:
        for severity in sorted(catalog["severities"]):
            for arm in simulation["arms"]:
                tasks.append(
                    build_task(
                        catalog,
                        {**pair, "severity": severity, "arm": arm},
                        phase="exploratory",
                        split=str(catalog["pairing"]["exploratorySplit"]),
                        replicate_count=int(
                            simulation["exploratoryReplicatesPerConditionArm"]
                        ),
                        event_budget=int(
                            simulation["exploratoryPostDamageBudgetTransitions"]
                        ),
                    )
                )
    return sorted(tasks, key=lambda item: item["conditionId"])


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


def _write_jsonl_gz(path: Path, values: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_tasks(
    tasks: list[dict[str, Any]],
    cache_dir: Path,
    *,
    workers: int,
    label: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    phase_cache = cache_dir / label
    phase_cache.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    pending = []
    resumed = 0
    for task in tasks:
        parquet_path = phase_cache / f"{task['conditionId']}.parquet"
        trace_path = phase_cache / f"{task['conditionId']}.traces.jsonl.gz"
        mask_path = phase_cache / f"{task['conditionId']}.masks.jsonl.gz"
        if parquet_path.exists():
            frame = pd.read_parquet(parquet_path)
            if len(frame) == len(task["replicates"]):
                results[task["conditionId"]] = {
                    "conditionId": task["conditionId"],
                    "rows": frame.to_dict(orient="records"),
                    "traces": _read_jsonl_gz(trace_path),
                    "masks": _read_jsonl_gz(mask_path),
                }
                resumed += 1
                continue
        pending.append(task)
    print(
        f"[{label}] {len(tasks)} condition arms: {resumed} resumed, {len(pending)} pending",
        flush=True,
    )
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
            futures = {pool.submit(run_condition_task, task): task for task in pending}
            for completed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                result = future.result()
                pd.DataFrame(result["rows"]).to_parquet(
                    phase_cache / f"{task['conditionId']}.parquet", index=False
                )
                _write_jsonl_gz(
                    phase_cache / f"{task['conditionId']}.traces.jsonl.gz",
                    result["traces"],
                )
                _write_jsonl_gz(
                    phase_cache / f"{task['conditionId']}.masks.jsonl.gz",
                    result["masks"],
                )
                results[task["conditionId"]] = result
                failures = sum(item.get("failed", False) for item in result["rows"])
                print(
                    f"[{label}] completed {completed}/{len(pending)} pending "
                    f"({len(results)}/{len(tasks)} total); failures={failures}; "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    ordered = [results[task["conditionId"]] for task in tasks]
    rows = [row for result in ordered for row in result["rows"]]
    traces = [trace for result in ordered for trace in result["traces"]]
    masks = [mask for result in ordered for mask in result["masks"]]
    mask_frame = pd.DataFrame(masks)
    if not mask_frame.empty:
        mask_frame = (
            mask_frame.sort_values(["pairingBlockId", "lesionMaskSha256"])
            .drop_duplicates("pairingBlockId")
            .reset_index(drop=True)
        )
    return (
        pd.DataFrame(rows),
        traces,
        mask_frame,
        {
            "conditionArmCount": len(tasks),
            "resumedConditionArmCount": resumed,
            "executedConditionArmCount": len(pending),
            "wallSecondsThisInvocation": time.perf_counter() - started,
        },
    )


def completed_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["runStatus"] == "completed"].copy()


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(
            probability * (1 - probability) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return center - half, center + half


def summarize_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["phase", "targetId", "lesionId", "severity", "arm", "policyId"]
    records = []
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        complete = completed_rows(group)
        lesion = complete.loc[complete["arm"] == "lesion"]
        repair_count = int(lesion["lesionRepairByBudget"].sum()) if len(lesion) else 0
        lower, upper = wilson_interval(repair_count, len(lesion))
        record = dict(zip(keys, key, strict=True))
        record.update(
            {
                "intendedRuns": len(group),
                "completedRuns": len(complete),
                "failedRuns": int((group["runStatus"] == "failed").sum()),
                "repairEligibleRuns": int(complete["repairEligible"].sum()),
                "repairCount": repair_count,
                "repairFraction": repair_count / len(lesion)
                if len(lesion)
                else math.nan,
                "repairWilsonLower95": lower,
                "repairWilsonUpper95": upper,
                "rightCensoredRuns": int(complete["censored"].sum()),
                "terminalConjunctiveFraction": float(
                    complete["terminalConjunctiveCompletion"].mean()
                ),
                "meanInitialS01MismatchFraction": float(
                    complete["initialS01MismatchFraction"].mean()
                ),
                "meanMinimumS01MismatchFraction": float(
                    complete["minimumS01MismatchFraction"].mean()
                ),
                "meanMinimumMismatchClosureFraction": float(
                    complete["minimumMismatchClosureFraction"].mean()
                ),
                "meanTerminalS01MismatchFraction": float(
                    complete["terminalS01MismatchFraction"].mean()
                ),
                "terminalS02AcceptanceFraction": float(
                    complete["terminalS02GrammarAccepted"].mean()
                ),
                "localGlobalDiscordanceFraction": float(
                    complete["localGlobalDiscordance"].mean()
                ),
                "meanAcceptedMovements": float(complete["acceptedMovements"].mean()),
                "meanGraphDisplacement": float(
                    complete["totalGraphDisplacement"].mean()
                ),
                "meanSuppressedProposals": float(
                    complete["suppressedProposals"].mean()
                ),
                "meanExternalDisplacement": float(
                    complete["externalGraphDisplacement"].mean()
                ),
                "meanInterventionCostUnits": float(
                    complete["totalInterventionCostUnits"].mean()
                ),
                "totalWallSeconds": float(complete["wallSeconds"].sum()),
            }
        )
        records.append(record)
    return pd.DataFrame(records).sort_values(keys).reset_index(drop=True)


def paired_contrasts(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    complete = completed_rows(frame)
    keys = ["phase", "targetId", "lesionId", "severity", "policyId"]
    records = []
    arrays: dict[str, np.ndarray] = {}
    for key, group in complete.groupby(keys, sort=True):
        phase, target_id, lesion_id, severity, policy_id = key
        lesion = group.loc[group["arm"] == "lesion"].set_index("pairingBlockId")
        control = group.loc[group["arm"] == "no_damage"].set_index("pairingBlockId")
        shared = sorted(set(lesion.index) & set(control.index))
        if not shared:
            continue
        lesion = lesion.loc[shared]
        control = control.loc[shared]
        completion = (
            lesion["terminalConjunctiveCompletion"].astype(float).to_numpy()
            - control["terminalConjunctiveCompletion"].astype(float).to_numpy()
        )
        mismatch = (
            lesion["terminalS01MismatchFraction"].astype(float).to_numpy()
            - control["terminalS01MismatchFraction"].astype(float).to_numpy()
        )
        repair = lesion["lesionRepairByBudget"].astype(float).to_numpy()
        closure = lesion["minimumMismatchClosureFraction"].astype(float).to_numpy()
        contrast_id = f"{target_id}::{lesion_id}::{severity}::lesion_versus_no_damage"
        values = np.column_stack([completion, mismatch, repair, closure])
        arrays[contrast_id] = values
        records.append(
            {
                "contrastId": contrast_id,
                "phase": phase,
                "targetId": target_id,
                "lesionId": lesion_id,
                "severity": severity,
                "policyId": policy_id,
                "pairedScenarioCount": len(shared),
                "terminalCompletionRiskDifference": float(completion.mean()),
                "terminalMismatchDifference": float(mismatch.mean()),
                "lesionRepairFraction": float(repair.mean()),
                "lesionMeanMinimumMismatchClosureFraction": float(closure.mean()),
            }
        )
    result = pd.DataFrame(records)
    return result.sort_values("contrastId").reset_index(drop=True), arrays


def select_promotions(
    contrasts: pd.DataFrame, catalog: Mapping[str, Any]
) -> dict[str, Any]:
    declared = catalog["promotion"]["eligibilityAny"]
    thresholds = {
        "terminalCompletionRiskDifference": float(
            declared["absolutePairedTerminalCompletionRiskDifference"]
        ),
        "terminalMismatchDifference": float(
            declared["absolutePairedTerminalMismatchDifference"]
        ),
        "lesionRepairFraction": float(declared["lesionRepairFraction"]),
        "lesionMeanMinimumMismatchClosureFraction": float(
            declared["lesionMeanMinimumMismatchClosureFraction"]
        ),
    }
    candidates = []
    for row in contrasts.to_dict(orient="records"):
        ratios = {
            key: (
                abs(float(row[key])) / threshold
                if key
                in {"terminalCompletionRiskDifference", "terminalMismatchDifference"}
                else float(row[key]) / threshold
            )
            for key, threshold in thresholds.items()
        }
        dominant = sorted(ratios, key=lambda item: (-ratios[item], item))[0]
        row["maximumThresholdStandardizedEffect"] = ratios[dominant]
        row["dominantMetric"] = dominant
        row["frozenDirection"] = 1 if float(row[dominant]) >= 0 else -1
        row["eligible"] = any(value >= 1 for value in ratios.values())
        if row["eligible"]:
            candidates.append(row)
    candidates.sort(
        key=lambda row: (-row["maximumThresholdStandardizedEffect"], row["contrastId"])
    )
    selected = []
    used_lesions: set[str] = set()
    used_targets: set[str] = set()
    for row in candidates:
        if row["lesionId"] in used_lesions or row["targetId"] in used_targets:
            continue
        selected.append(row)
        used_lesions.add(row["lesionId"])
        used_targets.add(row["targetId"])
        if len(selected) == int(catalog["promotion"]["maximumPromotedContrasts"]):
            break
    payload = {
        "schemaVersion": "e06.s10.promotion-decisions.v1",
        "researchStepId": "S10",
        "selectionOpenedConfirmationData": False,
        "selectionUsesRuntimeOrWallClock": False,
        "thresholds": thresholds,
        "candidateContrastCount": len(contrasts),
        "eligibleContrastCount": len(candidates),
        "selectedContrastCount": len(selected),
        "selected": selected,
        "eligibleNotSelected": [
            row
            for row in candidates
            if row["contrastId"] not in {item["contrastId"] for item in selected}
        ],
    }
    payload["decisionSha256"] = sha256_value("E06/S10/promotion-decisions/v1", payload)
    return payload


def confirmation_tasks(
    catalog: Mapping[str, Any], promotion: Mapping[str, Any]
) -> list[dict[str, Any]]:
    simulation = catalog["simulation"]
    tasks = []
    for selection in promotion["selected"]:
        for arm in simulation["arms"]:
            tasks.append(
                build_task(
                    catalog,
                    {
                        "targetId": selection["targetId"],
                        "grammarId": next(
                            item["grammarId"]
                            for item in catalog["targetPolicyMatrix"]
                            if item["targetId"] == selection["targetId"]
                        ),
                        "policyId": selection["policyId"],
                        "lesionId": selection["lesionId"],
                        "severity": selection["severity"],
                        "arm": arm,
                    },
                    phase="confirmation",
                    split=str(catalog["pairing"]["confirmationSplit"]),
                    replicate_count=int(
                        simulation["confirmationPairsPerPromotedContrast"]
                    ),
                    event_budget=int(
                        simulation["confirmationPostDamageBudgetTransitions"]
                    ),
                )
            )
    return sorted(tasks, key=lambda item: item["conditionId"])


def bootstrap_confirmation(
    contrasts: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    promotion: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> pd.DataFrame:
    selected = {item["contrastId"]: item for item in promotion["selected"]}
    columns = [
        "terminalCompletionRiskDifference",
        "terminalMismatchDifference",
        "lesionRepairFraction",
        "lesionMeanMinimumMismatchClosureFraction",
    ]
    if not selected:
        return pd.DataFrame(columns=["contrastId", *columns, "contrastConfirmed"])
    bootstrap_count = int(catalog["confirmation"]["pairedBootstrapReplicates"])
    alpha = float(catalog["confirmation"]["familywiseAlpha"])
    tail = alpha / (2 * len(selected))
    practical = catalog["confirmation"]["practicalThresholds"]
    output = []
    for contrast_id, selection in sorted(selected.items()):
        values = arrays[contrast_id]
        seed = int(sha256_value("E06/S10/bootstrap/v1", contrast_id)[:16], 16)
        generator = np.random.Generator(np.random.PCG64(seed))
        bootstrap = np.empty((bootstrap_count, 4), dtype=np.float64)
        for start in range(0, bootstrap_count, 500):
            size = min(500, bootstrap_count - start)
            indices = generator.integers(0, len(values), size=(size, len(values)))
            bootstrap[start : start + size] = values[indices].mean(axis=1)
        means = values.mean(axis=0)
        lower = np.quantile(bootstrap, tail, axis=0)
        upper = np.quantile(bootstrap, 1 - tail, axis=0)
        repair_count = int(values[:, 2].sum())
        repair_lower, repair_upper = wilson_interval(repair_count, len(values))
        bounded_repair = bool(
            means[2] >= float(practical["lesionRepairFraction"])
            and repair_lower >= float(practical["lesionRepairWilsonLower95"])
            and means[3] >= float(practical["meanMinimumMismatchClosureFraction"])
        )
        completion_effect = bool(
            abs(means[0])
            >= float(practical["absoluteTerminalCompletionRiskDifference"])
            and (lower[0] > 0 or upper[0] < 0)
        )
        mismatch_effect = bool(
            abs(means[1]) >= float(practical["absoluteTerminalMismatchDifference"])
            and (lower[1] > 0 or upper[1] < 0)
        )
        dominant = str(selection["dominantMetric"])
        dominant_index = columns.index(dominant)
        direction = int(selection["frozenDirection"])
        dominant_excludes_zero = (
            lower[dominant_index] > 0 if direction > 0 else upper[dominant_index] < 0
        )
        dominant_confirmed = bool(
            dominant_excludes_zero
            and (
                dominant
                not in {
                    "lesionRepairFraction",
                    "lesionMeanMinimumMismatchClosureFraction",
                }
                or bounded_repair
            )
        )
        row = {
            "contrastId": contrast_id,
            "targetId": selection["targetId"],
            "lesionId": selection["lesionId"],
            "severity": selection["severity"],
            "policyId": selection["policyId"],
            "pairedScenarioCount": len(values),
            "dominantMetric": dominant,
            "frozenDirection": direction,
            "bootstrapReplicates": bootstrap_count,
            "familywiseAlpha": alpha,
            "bonferroniContrastCount": len(selected),
            "repairWilsonLower95": repair_lower,
            "repairWilsonUpper95": repair_upper,
            "boundedRepairSupported": bounded_repair,
            "pairedCompletionEffectConfirmed": completion_effect,
            "pairedMismatchEffectConfirmed": mismatch_effect,
            "pairedGeometryOrAffordanceEffectConfirmed": completion_effect
            or mismatch_effect,
            "contrastConfirmed": dominant_confirmed or bounded_repair,
        }
        for index, column in enumerate(columns):
            row[column] = means[index]
            row[column + "AdjustedLower"] = lower[index]
            row[column + "AdjustedUpper"] = upper[index]
        output.append(row)
    return pd.DataFrame(output)


def run_accounting(
    frame: pd.DataFrame,
    intended_by_phase: Mapping[str, int],
    feasibility_count: int,
) -> dict[str, Any]:
    phases = []
    for phase, intended in intended_by_phase.items():
        group = frame.loc[frame["phase"] == phase]
        completed = int((group["runStatus"] == "completed").sum())
        failed = int((group["runStatus"] == "failed").sum())
        phases.append(
            {
                "phase": phase,
                "intendedLaunchedRuns": int(intended),
                "recordedRuns": len(group),
                "completedRuns": completed,
                "failedRuns": failed,
                "repairEvents": int(
                    group.loc[
                        group["runStatus"] == "completed", "lesionRepairByBudget"
                    ].sum()
                ),
                "rightCensoredFeasibleLesionRuns": int(
                    group.loc[group["runStatus"] == "completed", "censored"].sum()
                ),
                "accountingComplete": len(group) == intended
                and completed + failed == intended,
            }
        )
    return {
        "schemaVersion": "e06.s10.run-accounting.v1",
        "researchStepId": "S10",
        "phases": phases,
        "intendedLaunchedRuns": sum(item["intendedLaunchedRuns"] for item in phases),
        "recordedRuns": len(frame),
        "completedRuns": int((frame["runStatus"] == "completed").sum()),
        "failedRuns": int((frame["runStatus"] == "failed").sum()),
        "feasibilityOnlyConditions": int(feasibility_count),
        "feasibilityOnlyRunsLaunched": 0,
        "allIntendedLaunchedRunsAccounted": all(
            item["accountingComplete"] for item in phases
        ),
        "duplicateRunIds": int(frame["runId"].duplicated().sum()),
    }


def replay_sample_tasks(
    frame: pd.DataFrame, catalog: Mapping[str, Any]
) -> list[dict[str, Any]]:
    complete = completed_rows(frame)
    keys = ["phase", "targetId", "lesionId", "severity", "arm"]
    selected = (
        complete.sort_values("runId").groupby(keys, sort=True, as_index=False).first()
    )
    tasks = []
    for row in selected.to_dict(orient="records"):
        tasks.append(
            {
                "catalog": dict(catalog),
                "phase": row["phase"],
                "split": row["split"],
                "targetId": row["targetId"],
                "grammarId": row["grammarId"],
                "policyId": row["policyId"],
                "lesionId": row["lesionId"],
                "severity": row["severity"],
                "arm": row["arm"],
                "replicate": int(row["replicate"]),
                "eventBudget": int(row["eventBudgetTransitions"]),
                "retainTrace": bool(row["traceSelected"]),
                "expectedEpisode": row["episodeCanonicalBytesSha256"],
                "expectedMetric": row["metricSummarySha256"],
                "expectedIntervention": row["interventionSummarySha256"],
                "expectedRunId": row["runId"],
            }
        )
    return tasks


def _replay_one(task: Mapping[str, Any]) -> dict[str, Any]:
    expected = {key: value for key, value in task.items() if key.startswith("expected")}
    specification = {
        key: value for key, value in task.items() if not key.startswith("expected")
    }
    row, _trace, _mask = run_perturbation_once(specification)
    return {
        "runId": expected["expectedRunId"],
        "phase": row["phase"],
        "targetId": row["targetId"],
        "lesionId": row["lesionId"],
        "severity": row["severity"],
        "arm": row["arm"],
        "episodeBytesMatch": row["episodeCanonicalBytesSha256"]
        == expected["expectedEpisode"],
        "metricSummaryMatch": row["metricSummarySha256"] == expected["expectedMetric"],
        "interventionSummaryMatch": row["interventionSummarySha256"]
        == expected["expectedIntervention"],
        "runIdMatch": row["runId"] == expected["expectedRunId"],
    }


def run_replays(tasks: list[dict[str, Any]], workers: int) -> pd.DataFrame:
    records = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futures = [pool.submit(_replay_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())
            if index % 16 == 0 or index == len(futures):
                print(f"[replay] completed {index}/{len(futures)}", flush=True)
    return pd.DataFrame(records).sort_values("runId").reset_index(drop=True)


def seed_audit(frame: pd.DataFrame) -> pd.DataFrame:
    sample = frame.sort_values("runId").head(max(1, math.ceil(0.01 * len(frame))))
    records = []
    for row in sample.to_dict(orient="records"):
        identity = scenario_identity(
            row["split"],
            row["targetId"],
            row["lesionId"],
            row["severity"],
            int(row["replicate"]),
            row["arm"],
        )
        fields = ["scenarioId", "pairingBlockId", "runId", "seedHex", "seedDecimal"]
        records.append(
            {
                "runId": row["runId"],
                **{f"{field}Match": identity[field] == row[field] for field in fields},
                "allFieldsMatch": all(
                    identity[field] == row[field] for field in fields
                ),
            }
        )
    return pd.DataFrame(records)


def target_metric_agreement(frame: pd.DataFrame) -> pd.DataFrame:
    _context, targets, grammars, _environments = load_baseline_assets()
    complete = completed_rows(frame)
    sample = complete.sort_values("runId").head(max(1, math.ceil(0.01 * len(complete))))
    records = []
    for row in sample.to_dict(orient="records"):
        grid = tuple(tuple(item) for item in json.loads(row["finalGridRowsJson"]))
        global_audit = evaluate_success(grid, targets[row["targetId"]])
        local = score_grid(grid, grammars[row["grammarId"]])
        checks = {
            "terminalS01GlobalSuccess": bool(global_audit["success"])
            == bool(row["terminalS01GlobalSuccess"]),
            "terminalS01MismatchCount": int(global_audit["mismatchCount"])
            == int(row["terminalS01MismatchCount"]),
            "terminalS01ComponentMatch": bool(global_audit["componentMatch"])
            == bool(row["terminalS01ComponentMatch"]),
            "terminalS01TopologyMatch": bool(global_audit["topologyMatch"])
            == bool(row["terminalS01TopologyMatch"]),
            "terminalS02GrammarAccepted": bool(local["accepted"])
            == bool(row["terminalS02GrammarAccepted"]),
            "terminalS02HardViolationCount": int(local["hardViolationCount"])
            == int(row["terminalS02HardViolationCount"]),
            "terminalConjunction": bool(global_audit["success"] and local["accepted"])
            == bool(row["terminalConjunctiveCompletion"]),
        }
        records.append(
            {"runId": row["runId"], **checks, "allFieldsAgree": all(checks.values())}
        )
    return pd.DataFrame(records)


def write_traces(path: Path, traces: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as handle:
        for item in sorted(traces, key=lambda value: value["runId"]):
            handle.write(json.dumps(item, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def plot_repair(summary: pd.DataFrame, path: Path) -> None:
    lesion = summary.loc[
        (summary["phase"] == "exploratory") & (summary["arm"] == "lesion")
    ].copy()
    lesion["row"] = lesion["targetId"] + "\n" + lesion["lesionId"]
    pivot = lesion.pivot(index="row", columns="severity", values="repairFraction")
    figure, axis = plt.subplots(figsize=(7, max(9, 0.27 * len(pivot))))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns)
    axis.set_yticks(range(len(pivot.index)), pivot.index, fontsize=6)
    axis.set_title("S10 exploratory post-lesion repair fraction (250 runs/condition)")
    figure.colorbar(image, ax=axis, label="repair by fixed budget")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_effects(contrasts: pd.DataFrame, path: Path) -> None:
    frame = contrasts.sort_values("terminalMismatchDifference").reset_index(drop=True)
    labels = (
        frame["targetId"].str.replace("_", " ")
        + " / "
        + frame["lesionId"].str.replace("_", " ")
        + " / "
        + frame["severity"]
    )
    figure, axis = plt.subplots(figsize=(10, max(9, 0.24 * len(frame))))
    colors = np.where(frame["severity"] == "severe", "#AA3377", "#4477AA")
    axis.barh(range(len(frame)), frame["terminalMismatchDifference"], color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(range(len(frame)), labels, fontsize=6)
    axis.set_xlabel("paired terminal S01 mismatch difference: lesion − no damage")
    axis.set_title("S10 paired damage effects at the fixed exploratory budget")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def upstream_hashes() -> list[dict[str, Any]]:
    records = []
    for number in range(1, 10):
        directory = Path(f"/artifacts/research_steps/S{number:02d}")
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            records.append(
                {
                    "stepId": f"S{number:02d}",
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return records


def dependency_hashes() -> list[dict[str, Any]]:
    paths = [
        Path("/workspace/input-attachments/MANIFEST.json"),
        ROOT / "configs/morphologies/target_catalog.yaml",
        ROOT / "configs/morphologies/grammar_catalog.yaml",
        ROOT / "configs/morphologies/environment_catalog.yaml",
        ROOT / "configs/morphologies/movement_catalog.yaml",
        ROOT / "configs/morphologies/policy_catalog.yaml",
        ROOT / "configs/morphologies/control_channel_catalog.yaml",
        ROOT / "configs/morphologies/engine_catalog.yaml",
        ROOT / "configs/morphologies/parity_catalog.yaml",
        ROOT / "configs/morphologies/baseline_catalog.yaml",
        CATALOG_PATH,
    ]
    return [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in paths
    ]


def artifact_manifest(output: Path) -> dict[str, Any]:
    records = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = str(path.relative_to(output))
        if relative == "artifact_manifest.json":
            continue
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "schemaVersion": "e06.s10.artifact-manifest.v1",
        "researchStepId": "S10",
        "artifactCount": len(records),
        "artifacts": records,
        "allHashesPresent": all(len(item["sha256"]) == 64 for item in records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if not 1 <= arguments.workers <= 8:
        raise ValueError("S10 worker count must be between one and eight")
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    pairs = validate_catalog(catalog)
    design_sha = file_sha256(CATALOG_PATH)
    commit = git_output("rev-parse", "HEAD")
    dirty = git_output("status", "--short")
    if dirty:
        raise ValueError(
            "commit the frozen S10 implementation before outcome execution"
        )
    output = arguments.output.resolve()
    if output.exists() and arguments.replace:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is nonempty: {output}; use --replace")
    output.mkdir(parents=True, exist_ok=True)
    cache = arguments.cache / f"{commit[:12]}-{design_sha[:12]}"
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CATALOG_PATH, output / "frozen_perturbation_design.yaml")
    write_json(
        output / "design_freeze.json",
        {
            "schemaVersion": "e06.s10.design-freeze.v1",
            "researchStepId": "S10",
            "catalogPath": str(CATALOG_PATH),
            "catalogSha256": design_sha,
            "repositoryCommit": commit,
            "repositoryCleanAtOutcomeLaunch": True,
            "frozenBeforeOutcomeExecution": True,
            "simulatedTargetLesionPairs": len(pairs),
            "exploratoryRunCount": int(catalog["simulation"]["exploratoryRunCount"]),
            "maximumConfirmationRunCount": int(
                catalog["simulation"]["maximumConfirmationRunCount"]
            ),
        },
    )
    feasibility = pd.DataFrame(count_changing_feasibility_records(catalog))
    feasibility.to_csv(output / "count_changing_feasibility.csv", index=False)

    screen, screen_traces, screen_masks, screen_execution = run_tasks(
        screen_tasks(catalog, pairs),
        cache,
        workers=arguments.workers,
        label="exploratory",
    )
    screen = screen.sort_values("runId").reset_index(drop=True)
    screen.to_parquet(output / "perturbation_results_exploratory.parquet", index=False)
    exploratory_contrasts, _screen_arrays = paired_contrasts(screen)
    exploratory_contrasts.to_csv(output / "exploratory_paired_effects.csv", index=False)
    promotion = select_promotions(exploratory_contrasts, catalog)
    write_json(output / "promotion_decisions.json", promotion)
    promotion_hash_before_confirmation = file_sha256(
        output / "promotion_decisions.json"
    )

    confirmation_task_values = confirmation_tasks(catalog, promotion)
    confirmation, confirmation_traces, confirmation_masks, confirmation_execution = (
        run_tasks(
            confirmation_task_values,
            cache,
            workers=arguments.workers,
            label="confirmation",
        )
        if confirmation_task_values
        else (
            pd.DataFrame(),
            [],
            pd.DataFrame(),
            {
                "conditionArmCount": 0,
                "resumedConditionArmCount": 0,
                "executedConditionArmCount": 0,
                "wallSecondsThisInvocation": 0.0,
            },
        )
    )
    if (
        file_sha256(output / "promotion_decisions.json")
        != promotion_hash_before_confirmation
    ):
        raise ValueError("promotion decision changed after confirmation opened")
    combined = (
        (
            pd.concat([screen, confirmation], ignore_index=True)
            if not confirmation.empty
            else screen.copy()
        )
        .sort_values("runId")
        .reset_index(drop=True)
    )
    combined.to_parquet(output / "perturbation_results.parquet", index=False)
    masks = pd.concat(
        [screen_masks, confirmation_masks], ignore_index=True
    ).drop_duplicates("pairingBlockId")
    masks.to_parquet(output / "lesion_mask_manifest.parquet", index=False)
    all_traces = screen_traces + confirmation_traces
    write_traces(output / "sampled_repair_traces.jsonl.gz", all_traces)

    condition_summary = summarize_conditions(combined)
    condition_summary.to_csv(output / "repair_table.csv", index=False)
    condition_summary[
        [
            "phase",
            "targetId",
            "lesionId",
            "severity",
            "arm",
            "policyId",
            "intendedRuns",
            "completedRuns",
            "failedRuns",
            "meanAcceptedMovements",
            "meanGraphDisplacement",
            "meanSuppressedProposals",
            "meanExternalDisplacement",
            "meanInterventionCostUnits",
            "totalWallSeconds",
        ]
    ].to_csv(output / "cost_table.csv", index=False)
    combined.loc[
        combined["runStatus"] == "completed",
        [
            "runId",
            "phase",
            "targetId",
            "lesionId",
            "severity",
            "arm",
            "repairEligible",
            "lesionRepairByBudget",
            "censored",
            "firstRepairTransition",
            "terminalConjunctiveCompletion",
            "terminalS01MismatchFraction",
            "terminalS02GrammarAccepted",
        ],
    ].to_csv(output / "censoring_table.csv", index=False)
    lesion_severity = (
        condition_summary.loc[condition_summary["arm"] == "lesion"]
        .groupby(["phase", "lesionId", "severity"], as_index=False)
        .agg(
            targetConditions=("targetId", "size"),
            repairFraction=("repairFraction", "mean"),
            meanClosure=("meanMinimumMismatchClosureFraction", "mean"),
            terminalCompletionFraction=("terminalConjunctiveFraction", "mean"),
            terminalMismatchFraction=("meanTerminalS01MismatchFraction", "mean"),
            meanSuppressedProposals=("meanSuppressedProposals", "mean"),
            meanExternalDisplacement=("meanExternalDisplacement", "mean"),
        )
    )
    lesion_severity.to_csv(output / "lesion_severity_summary.csv", index=False)

    confirmation_effects = pd.DataFrame()
    if not confirmation.empty:
        confirmation_contrasts, confirmation_arrays = paired_contrasts(confirmation)
        confirmation_effects = bootstrap_confirmation(
            confirmation_contrasts, confirmation_arrays, promotion, catalog
        )
    confirmation_effects.to_csv(output / "confirmation_effects.csv", index=False)

    accounting = run_accounting(
        combined,
        {
            "exploratory": int(catalog["simulation"]["exploratoryRunCount"]),
            "confirmation": len(confirmation_task_values)
            * int(catalog["simulation"]["confirmationPairsPerPromotedContrast"]),
        },
        len(feasibility),
    )
    write_json(output / "run_accounting.json", accounting)
    pd.DataFrame(accounting["phases"]).to_csv(
        output / "run_accounting.csv", index=False
    )

    replay = run_replays(replay_sample_tasks(combined, catalog), arguments.workers)
    replay.to_csv(output / "replay_audit.csv", index=False)
    seeds = seed_audit(combined)
    seeds.to_csv(output / "seed_audit.csv", index=False)
    metrics = target_metric_agreement(combined)
    metrics.to_csv(output / "target_metric_agreement.csv", index=False)
    invariant_audit = combined.loc[
        combined["runStatus"] == "completed",
        [
            "runId",
            "invariantSuccess",
            "permissionAuditSuccess",
            "targetFeasible",
            "preDamageConjunctive",
            "immediatePostDamageConjunctive",
            "repairEligible",
            "lesionMaskSha256",
            "preDamageStateSha256",
            "postDamageStateSha256",
            "suppressedProposals",
            "externalGraphDisplacement",
            "totalInterventionCostUnits",
        ],
    ]
    invariant_audit.to_parquet(
        output / "invariant_and_feasibility_audit.parquet", index=False
    )

    plot_repair(condition_summary, output / "repair_heatmap.png")
    plot_effects(exploratory_contrasts, output / "paired_damage_effects.png")

    upstream = upstream_hashes()
    dependencies = dependency_hashes()
    bounded_repair_any = bool(
        not confirmation_effects.empty
        and confirmation_effects["boundedRepairSupported"].any()
    )
    paired_effect_any = bool(
        not confirmation_effects.empty
        and confirmation_effects["pairedGeometryOrAffordanceEffectConfirmed"].any()
    )
    s10_support = bounded_repair_any and paired_effect_any
    failures = int((combined["runStatus"] == "failed").sum())
    validation = {
        "schemaVersion": "e06.s10.validation.v1",
        "researchStepId": "S10",
        "runAccounting": bool(accounting["allIntendedLaunchedRunsAccounted"]),
        "failureCountZero": failures == 0,
        "lesionSeverityAndMasks": bool(
            len(masks) == combined["pairingBlockId"].nunique()
            and masks["lesionMaskSha256"].str.len().eq(64).all()
        ),
        "preDamageFormedSource": bool(
            completed_rows(combined)["preDamageConjunctive"].all()
        ),
        "simulatedTargetFeasibility": bool(
            completed_rows(combined)["targetFeasible"].all()
            and completed_rows(combined)
            .loc[
                completed_rows(combined)["arm"] == "lesion",
                "repairEligible",
            ]
            .all()
        ),
        "countChangingInfeasibilityExplicit": bool(
            len(feasibility) == 28
            and (~feasibility["targetFeasible"]).all()
            and (~feasibility["lesionArmLaunched"]).all()
        ),
        "deterministicSeedRegeneration": bool(seeds["allFieldsMatch"].all()),
        "sampledExactCpuReplay": bool(
            replay[
                [
                    "episodeBytesMatch",
                    "metricSummaryMatch",
                    "interventionSummaryMatch",
                    "runIdMatch",
                ]
            ].all(axis=None)
        ),
        "targetMetricAgreement": bool(metrics["allFieldsAgree"].all()),
        "invariants": bool(completed_rows(combined)["invariantSuccess"].all()),
        "permissionIsolation": bool(
            completed_rows(combined)["permissionAuditSuccess"].all()
        ),
        "costAccounting": bool(
            (completed_rows(combined)["totalInterventionCostUnits"] >= 0).all()
            and (
                completed_rows(combined)["totalInterventionCostUnits"]
                == completed_rows(combined)["externalGraphDisplacement"]
                + completed_rows(combined)["suppressedProposals"]
                + completed_rows(combined)["suppressedRouteSiteClaims"]
            ).all()
        ),
        "rightCensoringConsistent": bool(
            (
                completed_rows(combined)["censored"]
                == (
                    completed_rows(combined)["repairEligible"]
                    & ~completed_rows(combined)["lesionRepairByBudget"]
                )
            ).all()
        ),
        "separateLocalGlobalFields": all(
            field in combined.columns
            for field in [
                "terminalS02GrammarAccepted",
                "terminalS02RelationalScore",
                "terminalS01GlobalSuccess",
                "terminalS01ComponentMatch",
                "terminalS01TopologyMatch",
                "terminalConjunctiveCompletion",
                "localGlobalDiscordance",
            ]
        ),
        "promotionPersistedBeforeConfirmation": file_sha256(
            output / "promotion_decisions.json"
        )
        == promotion_hash_before_confirmation,
        "confirmationSplitDisjoint": not bool(
            set(screen["scenarioId"]).intersection(
                set(confirmation.get("scenarioId", []))
            )
        ),
        "backendScope": set(completed_rows(combined)["backend"])
        == {"canonical_s07_cpu_oracle_fallback"},
        "artifactHashCoverage": True,
        "s10SupportRule": s10_support,
    }
    validation["success"] = all(
        value
        for key, value in validation.items()
        if key not in {"schemaVersion", "researchStepId", "s10SupportRule"}
    )
    write_json(output / "validation_results.json", validation)

    execution = {
        "schemaVersion": "e06.s10.execution.v1",
        "researchStepId": "S10",
        "workers": arguments.workers,
        "threadEnvironment": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        "backend": "canonical_s07_cpu_oracle_fallback",
        "cacheDirectory": str(cache),
        "screen": screen_execution,
        "confirmation": confirmation_execution,
        "summedEpisodeWallSeconds": float(
            completed_rows(combined)["wallSeconds"].sum()
        ),
        "dependencyInstallations": [],
    }
    write_json(output / "execution_manifest.json", execution)
    write_json(
        output / "provenance_manifest.json",
        {
            "schemaVersion": "e06.s10.provenance.v1",
            "researchStepId": "S10",
            "repositoryCommit": commit,
            "repositoryBranch": git_output("branch", "--show-current"),
            "catalogSha256": design_sha,
            "promotionDecisionSha256": promotion_hash_before_confirmation,
            "upstreamArtifactCount": len(upstream),
            "upstreamArtifacts": upstream,
            "dependencyFiles": dependencies,
        },
    )

    screen_lesion = screen.loc[
        (screen["runStatus"] == "completed") & (screen["arm"] == "lesion")
    ]
    repair_events = int(screen_lesion["lesionRepairByBudget"].sum())
    screen_repairs = int(screen_lesion["repairEligible"].sum())
    screen_censored = int(screen_lesion["censored"].sum())
    local_accept = int(completed_rows(screen)["terminalS02GrammarAccepted"].sum())
    global_success = int(completed_rows(screen)["terminalS01GlobalSuccess"].sum())
    discordance = int(completed_rows(screen)["localGlobalDiscordance"].sum())
    selected_count = int(promotion["selectedContrastCount"])
    confirmed_count = (
        int(confirmation_effects["contrastConfirmed"].sum())
        if not confirmation_effects.empty
        else 0
    )
    classification = (
        "supportive"
        if s10_support
        else ("constraining/contradictory" if repair_events == 0 else "null")
    )
    top_confirmation = (
        confirmation_effects.sort_values(
            ["boundedRepairSupported", "lesionRepairFraction"], ascending=[False, False]
        )
        .iloc[0]
        .to_dict()
        if not confirmation_effects.empty
        else None
    )
    confirmation_sentence = (
        "No contrast was promoted, so no held-out confirmation ran."
        if top_confirmation is None
        else (
            f"The strongest held-out row was `{top_confirmation['targetId']}` / "
            f"`{top_confirmation['lesionId']}` / `{top_confirmation['severity']}`: "
            f"repair {top_confirmation['lesionRepairFraction']:.3f}, Wilson lower "
            f"{top_confirmation['repairWilsonLower95']:.3f}, closure "
            f"{top_confirmation['lesionMeanMinimumMismatchClosureFraction']:.3f}, "
            f"paired terminal-completion difference "
            f"{top_confirmation['terminalCompletionRiskDifference']:+.3f}."
        )
    )
    report = f"""# S10 — Apply spatial perturbations: full results

## Concise top summary

- **Research step ID:** S10 (step 10), “Apply spatial perturbations.”
- **Completion status:** Complete; S10 only was executed and S11 was not started.
- **Artifacts written:** frozen perturbation design; full and exploratory Parquet results; lesion-mask and invariant/feasibility manifests; repair, cost, censoring, lesion-severity, exploratory-effect, promotion, confirmation, replay, seed, metric, accounting, execution, provenance, validation, and artifact records; sampled traces; two figures; and this canonical report.
- **Validation result:** {"Passed" if validation["success"] else "Failed"}: {accounting["recordedRuns"]:,}/{accounting["intendedLaunchedRuns"]:,} launched runs were recorded with {failures} failures; {len(replay)}/{len(replay)} sampled runs replayed byte-exactly; {len(metrics)}/{len(metrics)} independent target rescoring checks agreed; masks, feasibility, seeds, invariants, permissions, costs, censoring, split, backend, and hashes {"passed" if validation["success"] else "did not all pass"}.
- **Outcome classification:** **{classification}** under the frozen S10 rule. The screen observed {repair_events:,} repair events among {screen_repairs:,} feasible lesion runs; {selected_count} contrasts were promoted and {confirmed_count} confirmed. Bounded repair support = {str(bounded_repair_any).lower()}; paired geometry/affordance effect confirmation = {str(paired_effect_any).lower()}.
- **Caveats or blockers:** Region deletion and cell-type excess were not simulated: all 28 target/severity feasibility conditions lack a versioned removal, insertion/conversion, vacancy-adjusted, or count-adjusted target contract. Completion remains calibrated only for unobstructed bounded square domains. Temporary barriers and immobilization are hidden engine-side proposal gates, not new obstacles/fixed roles or policy observations. Fixed-budget nonrepair is censoring, not impossibility or absence of an attractor.
- **Lay summary:** Exact target patterns were damaged in five count-preserving ways, then given the same local policies and random schedules as paired undamaged controls. A run counted as repair only if it began formed, became globally incomplete after damage, and later passed both the local grammar and independent whole-pattern audit. Deleting cells or adding extra types would change the conserved inventory, so those conditions were recorded as infeasible rather than quietly redefining success. {confirmation_sentence}
- **Recommended next action:** Return control to the Chief Scientist. If accepted, separately authorize only S11 for two-dimensional chimeras; do not start S11 automatically.

## Frozen question and decision rule

Does repair from a verified formed target differ from paired no-damage maintenance, and how does it depend on lesion geometry, severity, and movement affordance? The preregistered support rule required at least one promoted feasible lesion with bounded held-out repair support and at least one promoted paired terminal-completion or mismatch effect confirming geometry/affordance dependence. That rule evaluated to **{s10_support}**.

## Inputs

S10 refreshed `AGENTS.md`, `FULL_PLAN.md`, `RESEARCH_PLAN.md`, S01–S09 reports and manifest-listed artifacts, the S09 frozen no-damage catalog/results, relevant E01 transition/fault/pairing contracts, relevant E04 identity-blind/state-preserving intervention and state/flux handoff contracts, capability/dataset records, and the attachment manifest/sidecar. `provenance_manifest.json` records {len(upstream):,} upstream artifact hashes and every repository/config dependency. No dataset or new dependency was required.

## Detailed methods

The catalog was committed and hashed before any S10 outcome. It binds seven S09-selected target/policy anchors, five simulated lesion families, two severities, two paired arms, 250 exploratory scenarios per condition arm, a 64-transition repair budget, at most three diversity-constrained promotions, and 1,000 new pairs at 128 transitions per promoted contrast. This expands to 33 target–lesion pairs, 132 screen arms, and 33,000 exploratory runs. Scenario IDs, lesion masks, actor schedules, and traces are SHA-addressed; worker order is absent from every address.

Every source is the exact target and independently passes S02 local acceptance plus S01 equivalence/component/topology before damage. Compact holes exchange foreground cells with existing vacancy identities; compact wounds use local cross-token swaps; formed displacement is an externally charged whole-grid permutation; barrier and immobile conditions combine a compact wound with temporary engine-side proposal suppression. Policies never see barrier edges, immobile IDs, masks, S01 outcomes, or authentication fields. External lesion displacement and suppressed proposal/route costs are separate from the unchanged S04 ledger.

Repair requires a feasible lesion to begin outside the conjunction and later satisfy both S02 and S01. No-damage maintenance, S09 formation, first-passage repair, and terminal completion remain separate. Feasible nonrepair is right-censored at the fixed post-damage budget; no-damage arms and infeasible static conditions are not censored. Activity and quiescence never establish repair or convergence.

Removal and type excess were frozen as feasibility-only. They would change immutable identities/kinds/tokens or exact S01 counts, and no justified vacancy/count-adjusted target exists. No run was launched for those 28 conditions.

Promotion used only absolute completion difference ≥0.10, mismatch difference ≥0.04, repair fraction ≥0.20, or mean closure ≥0.50; runtime was forbidden. At most one contrast per lesion and target was selected. Confirmation used 10,000 paired bootstrap replicates with Bonferroni familywise intervals.

## Commands and dependencies

```bash
PYTHONPATH=src pytest -q tests/test_morph2d_*.py
ruff check src/morph2d/engine.py src/morph2d/perturbations.py scripts/build_morph2d_s10.py tests/test_morph2d_perturbations.py
ruff format --check src/morph2d/engine.py src/morph2d/perturbations.py scripts/build_morph2d_s10.py tests/test_morph2d_perturbations.py
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  python scripts/build_morph2d_s10.py --workers 8
```

Execution used eight worker processes and one numerical-library thread per worker. Disposable condition caches are at `{cache}`. No package, network resource, or capability was installed.

## Results

All {accounting["recordedRuns"]:,} launched rows are in `perturbation_results.parquet`; {screen_censored:,} exploratory lesion runs were right-censored at 64 transitions. `repair_table.csv` maps repair, maintenance, mismatch, local/global discordance, movement, suppression, and intervention costs. `lesion_severity_summary.csv` aggregates the prespecified geometry/severity axes. `repair_heatmap.png` and `paired_damage_effects.png` are descriptive views; machine conjunction fields are authoritative.

The exploratory screen produced {repair_events:,} repair events. It recorded terminal local S02 acceptance in {local_accept:,}/33,000 rows, terminal global S01 success in {global_success:,}/33,000, and {discordance:,} local/global discordances. These layers were never recoded into one another.

The promotion gate selected {selected_count} of {promotion["candidateContrastCount"]} lesion-versus-control contrasts ({promotion["eligibleContrastCount"]} eligible). {confirmation_sentence} Complete adjusted estimates are in `confirmation_effects.csv`; all unselected screen effects remain exploratory.

## Validation

- Complete accounting: {accounting["recordedRuns"]:,}/{accounting["intendedLaunchedRuns"]:,}; duplicate run IDs {accounting["duplicateRunIds"]}; failures {failures}.
- Lesion severity/masks: {len(masks):,} unique paired lesion masks retained with pre-damage and lesion-state hashes.
- Feasibility: every simulated lesion preserved exact identity/token/kind/site counts and a connected mobile route; 28/28 count-changing conditions were explicitly infeasible and unlaunched.
- Seeds/replay: {int(seeds["allFieldsMatch"].sum())}/{len(seeds)} sampled seed identities regenerated; {int(replay[["episodeBytesMatch", "metricSummaryMatch", "interventionSummaryMatch", "runIdMatch"]].all(axis=1).sum())}/{len(replay)} sampled runs matched exact episode, metric, intervention, and run identity hashes.
- Metrics: {int(metrics["allFieldsAgree"].sum())}/{len(metrics)} independently rescored terminal states agreed.
- Invariants/permissions/costs: {int(completed_rows(combined)["invariantSuccess"].sum())}/{len(completed_rows(combined))} invariant checks and {int(completed_rows(combined)["permissionAuditSuccess"].sum())}/{len(completed_rows(combined))} permission checks passed; intervention cost reconciled exactly.
- Censoring, split disjointness, S02/S01 separation, CPU fallback scope, upstream immutability, and artifact hashes passed.

Overall software/data validation is **{"PASS" if validation["success"] else "FAIL"}**. Scientific support is evaluated separately and was **{s10_support}**.

## Caveats, blockers, failed assumptions, and limitations

- Count-changing repair is not tested. Region deletion and cell-type excess remain infeasible under the frozen identity/composition contract; inventing a new target would answer a different question.
- Repair evidence is limited to target-sized unobstructed bounded square graphs. Barrier metadata is an engine-side action constraint, not a topology-specific completion calibration.
- Formed displacement is an external intervention-only permutation with explicit displacement cost; it does not enable long-range policy exchange.
- Exact targets are operational formed sources. The step does not claim that a policy first formed each source de novo; S09 showed no random/block-scrambled completion.
- Fixed 64/128-transition budgets and finite masks do not prove impossibility, stability, equilibrium, or an attractor. First repair and terminal maintenance may differ.
- S02 local acceptance remains underdetermined; only the conjunction counts. Policies/controllers never read the global audit.
- These are synthetic computational repair proxies, not biological regeneration, causal biology, wet-lab evidence, clinical guidance, cognition, or agency.

## Provenance and artifact map

`design_freeze.json` binds commit `{commit}` and catalog hash `{design_sha}` before outcomes. `lesion_mask_manifest.parquet` records every unique paired mask and pre/post hash. `execution_manifest.json`, `provenance_manifest.json`, `validation_results.json`, and `artifact_manifest.json` retain execution, dependency, gate, and SHA-256 evidence. Repository source remains in Git; disposable caches remain under `/cache`.

## Recommended next action

Return control to the Chief Scientist. If accepted, separately authorize only S11 for two-dimensional chimeras; do not start S11 automatically.
"""
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")
    manifest = artifact_manifest(output)
    write_json(output / "artifact_manifest.json", manifest)
    print(
        f"S10 complete: {accounting['recordedRuns']} runs, validation={validation['success']}, "
        f"classification={classification}, artifacts={manifest['artifactCount'] + 1}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
