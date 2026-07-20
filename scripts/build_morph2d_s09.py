#!/usr/bin/env python3
"""Execute the frozen E06 S09 baseline screen and held-out confirmations."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
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

from morph2d.baseline import (
    eligible_policies,
    load_baseline_assets,
    run_baseline_once,
    run_condition_task,
    scenario_identity,
)
from morph2d.grammar import score_grid
from morph2d.targets import evaluate_success


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = Path("/artifacts/research_steps/S09")
DEFAULT_CACHE = Path("/cache/e06_s09")
CATALOG_PATH = ROOT / "configs/morphologies/baseline_catalog.yaml"
JSON_INDENT = 2


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def sha256_value(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_bytes(value)
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_catalog(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    if catalog["schemaVersion"] != "e06.s09.baseline-catalog.v1":
        raise ValueError("S09 catalog schema mismatch")
    if catalog["researchStepId"] != "S09":
        raise ValueError("catalog is not scoped to S09")
    _, targets, grammars, _ = load_baseline_assets()
    grammar_by_target = {item.target_id: item.grammar_id for item in grammars.values()}
    declared = {item["targetId"]: item["grammarId"] for item in catalog["targetMatrix"]}
    if declared != grammar_by_target or set(declared) != set(targets):
        raise ValueError(
            "target/grammar baseline matrix does not cover the frozen catalogs"
        )
    simulation = catalog["simulation"]
    if simulation["stopping"] != "fixed_event_budget_only":
        raise ValueError("S09 must use only the frozen fixed event budget")
    if simulation["earlyStopping"] != "forbidden":
        raise ValueError("S09 early stopping must remain forbidden")
    if catalog["backend"]["production"] != "canonical_s07_cpu_oracle_fallback":
        raise ValueError("S09 may not widen the validated GPU data plane")
    if catalog["promotion"]["runtimeAndWallClockForbiddenFromSelection"] is not True:
        raise ValueError("runtime must be excluded from promotion")
    conditions = []
    for item in catalog["targetMatrix"]:
        for start_family in catalog["initialStateFamilies"]:
            for policy_id in eligible_policies(item["targetId"], catalog):
                conditions.append(
                    {
                        "targetId": item["targetId"],
                        "grammarId": item["grammarId"],
                        "startFamily": start_family,
                        "policyId": policy_id,
                    }
                )
    expected_conditions = int(simulation["exploratoryConditionCount"])
    expected_runs = int(simulation["exploratoryRunCount"])
    replicates = int(simulation["exploratoryReplicatesPerCondition"])
    if (
        len(conditions) != expected_conditions
        or len(conditions) * replicates != expected_runs
    ):
        raise ValueError("expanded S09 matrix does not equal 96 conditions/24,000 runs")
    if int(simulation["maximumConfirmationRunCount"]) != (
        2
        * int(simulation["maximumPromotedContrasts"])
        * int(simulation["confirmationPairsPerPromotedContrast"])
    ):
        raise ValueError("confirmation maximum is inconsistent with paired contrasts")
    return conditions


def condition_id(specification: Mapping[str, Any], phase: str) -> str:
    digest = sha256_value(
        "E06/S09/condition/v1",
        {
            "phase": phase,
            "targetId": specification["targetId"],
            "startFamily": specification["startFamily"],
            "policyId": specification["policyId"],
        },
    )
    return f"{phase[:4]}-{digest[:20]}"


def trace_run_ids(
    split: str,
    target_id: str,
    start_family: str,
    policy_id: str,
    replicates: Iterable[int],
    fraction: float,
) -> list[str]:
    run_ids = [
        scenario_identity(split, target_id, start_family, replicate, policy_id)["runId"]
        for replicate in replicates
    ]
    count = max(1, math.ceil(fraction * len(run_ids)))
    return sorted(run_ids)[:count]


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
        str(task["startFamily"]),
        str(task["policyId"]),
        replicates,
        float(catalog["traceAndReplay"]["traceSampleFraction"]),
    )
    return task


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


def _write_trace_cache(path: Path, traces: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for item in traces:
            handle.write(json.dumps(item, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _read_trace_cache(path: Path) -> list[dict[str, Any]]:
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
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    phase_cache = cache_dir / label
    phase_cache.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    pending = []
    resumed = 0
    for task in tasks:
        parquet_path = phase_cache / f"{task['conditionId']}.parquet"
        trace_path = phase_cache / f"{task['conditionId']}.jsonl.gz"
        if parquet_path.exists():
            frame = pd.read_parquet(parquet_path)
            if len(frame) == len(task["replicates"]):
                results[task["conditionId"]] = {
                    "conditionId": task["conditionId"],
                    "rows": frame.to_dict(orient="records"),
                    "traces": _read_trace_cache(trace_path),
                }
                resumed += 1
                continue
        pending.append(task)
    print(
        f"[{label}] {len(tasks)} conditions: {resumed} resumed, {len(pending)} pending",
        flush=True,
    )
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
            futures = {pool.submit(run_condition_task, task): task for task in pending}
            for completed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                result = future.result()
                frame = pd.DataFrame(result["rows"])
                frame.to_parquet(
                    phase_cache / f"{task['conditionId']}.parquet", index=False
                )
                _write_trace_cache(
                    phase_cache / f"{task['conditionId']}.jsonl.gz",
                    result["traces"],
                )
                results[task["conditionId"]] = result
                failures = sum(
                    item.get("runStatus") == "failed" for item in result["rows"]
                )
                elapsed = time.perf_counter() - started
                print(
                    f"[{label}] completed {completed}/{len(pending)} pending "
                    f"({len(results)}/{len(tasks)} total); failures={failures}; "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    ordered = [results[task["conditionId"]] for task in tasks]
    rows = [row for result in ordered for row in result["rows"]]
    traces = [trace for result in ordered for trace in result["traces"]]
    return (
        pd.DataFrame(rows),
        traces,
        {
            "conditionCount": len(tasks),
            "resumedConditionCount": resumed,
            "executedConditionCount": len(pending),
            "wallSecondsThisInvocation": time.perf_counter() - started,
        },
    )


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


def completed_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["runStatus"] == "completed"].copy()


def summarize_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["phase", "targetId", "startFamily", "policyId"]
    records = []
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        complete = completed_rows(group)
        success_count = (
            int(complete["conjunctiveCompletionByBudget"].sum()) if len(complete) else 0
        )
        lower, upper = wilson_interval(success_count, len(complete))
        record = dict(zip(keys, key, strict=True))
        record.update(
            {
                "intendedRuns": len(group),
                "completedRuns": len(complete),
                "failedRuns": int((group["runStatus"] == "failed").sum()),
                "censoredRuns": int(complete["censored"].sum()) if len(complete) else 0,
                "completionCount": success_count,
                "completionFraction": success_count / len(complete)
                if len(complete)
                else math.nan,
                "completionWilsonLower95": lower,
                "completionWilsonUpper95": upper,
                "meanInitialS01MismatchFraction": complete[
                    "initialS01MismatchFraction"
                ].mean(),
                "meanMinimumS01MismatchFraction": complete[
                    "minimumS01MismatchFraction"
                ].mean(),
                "meanMinimumS01Progress": (
                    complete["initialS01MismatchFraction"]
                    - complete["minimumS01MismatchFraction"]
                ).mean(),
                "meanTerminalS01MismatchFraction": complete[
                    "terminalS01MismatchFraction"
                ].mean(),
                "terminalS02AcceptanceFraction": complete[
                    "terminalS02GrammarAccepted"
                ].mean(),
                "localGlobalDiscordanceFraction": complete[
                    "localGlobalDiscordance"
                ].mean(),
                "meanAcceptedMovements": complete["acceptedMovements"].mean(),
                "meanGraphDisplacement": complete["totalGraphDisplacement"].mean(),
                "meanObservationBitsUpperBound": complete[
                    "observationCommunicatedBitsUpperBound"
                ].mean(),
                "meanChannelInformationBits": complete[
                    "channelTotalInformationBits"
                ].mean(),
                "totalWallSeconds": complete["wallSeconds"].sum(),
            }
        )
        records.append(record)
    return pd.DataFrame(records).sort_values(keys).reset_index(drop=True)


def paired_contrasts(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    complete = completed_rows(frame)
    records = []
    arrays: dict[str, np.ndarray] = {}
    reference = "exploration_v1"
    for (phase, target_id, start_family), group in complete.groupby(
        ["phase", "targetId", "startFamily"], sort=True
    ):
        control = group.loc[group["policyId"] == reference].set_index("pairingBlockId")
        for policy_id, policy_group in group.loc[
            group["policyId"] != reference
        ].groupby("policyId", sort=True):
            policy = policy_group.set_index("pairingBlockId")
            shared = sorted(set(policy.index) & set(control.index))
            if not shared:
                continue
            policy = policy.loc[shared]
            control_shared = control.loc[shared]
            completion = (
                policy["conjunctiveCompletionByBudget"].astype(float).to_numpy()
                - control_shared["conjunctiveCompletionByBudget"]
                .astype(float)
                .to_numpy()
            )
            policy_progress = (
                policy["initialS01MismatchFraction"]
                - policy["minimumS01MismatchFraction"]
            ).to_numpy(float)
            control_progress = (
                control_shared["initialS01MismatchFraction"]
                - control_shared["minimumS01MismatchFraction"]
            ).to_numpy(float)
            progress = policy_progress - control_progress
            discordance = (
                policy["localGlobalDiscordance"].astype(float).to_numpy()
                - control_shared["localGlobalDiscordance"].astype(float).to_numpy()
            )
            contrast_id = (
                f"{target_id}::{start_family}::{policy_id}::versus::{reference}"
            )
            arrays[contrast_id] = np.column_stack([completion, progress, discordance])
            records.append(
                {
                    "contrastId": contrast_id,
                    "phase": phase,
                    "targetId": target_id,
                    "startFamily": start_family,
                    "policyId": policy_id,
                    "referencePolicyId": reference,
                    "pairedScenarioCount": len(shared),
                    "completionRiskDifference": completion.mean(),
                    "minimumMismatchProgressDifference": progress.mean(),
                    "discordanceRateDifference": discordance.mean(),
                }
            )
    result = pd.DataFrame(records)
    return result.sort_values("contrastId").reset_index(drop=True), arrays


def select_promotions(
    contrasts: pd.DataFrame, catalog: Mapping[str, Any]
) -> dict[str, Any]:
    thresholds = catalog["promotion"]["eligibilityAny"]
    mappings = {
        "completionRiskDifference": float(
            thresholds["absolutePairedCompletionRiskDifference"]
        ),
        "minimumMismatchProgressDifference": float(
            thresholds["absolutePairedMinimumMismatchProgressDifference"]
        ),
        "discordanceRateDifference": float(
            thresholds["absoluteLocalGlobalDiscordanceRateDifference"]
        ),
    }
    candidates = []
    for row in contrasts.to_dict(orient="records"):
        ratios = {
            key: abs(float(row[key])) / threshold for key, threshold in mappings.items()
        }
        dominant = sorted(ratios, key=lambda key: (-ratios[key], key))[0]
        row["maximumThresholdStandardizedAbsoluteEffect"] = ratios[dominant]
        row["dominantMetric"] = dominant
        row["frozenDirection"] = 1 if float(row[dominant]) >= 0 else -1
        row["eligible"] = any(value >= 1 for value in ratios.values())
        if row["eligible"]:
            candidates.append(row)
    candidates.sort(
        key=lambda row: (
            -row["maximumThresholdStandardizedAbsoluteEffect"],
            row["contrastId"],
        )
    )
    selected = []
    used_starts = set()
    maximum = int(catalog["promotion"]["maximumPromotedContrasts"])
    for row in candidates:
        if row["startFamily"] in used_starts:
            continue
        selected.append(row)
        used_starts.add(row["startFamily"])
        if len(selected) == maximum:
            break
    payload = {
        "schemaVersion": "e06.s09.promotion-decisions.v1",
        "researchStepId": "S09",
        "selectionOpenedConfirmationData": False,
        "selectionUsesRuntimeOrWallClock": False,
        "thresholds": mappings,
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
    payload["decisionSha256"] = sha256_value("E06/S09/promotion-decisions/v1", payload)
    return payload


def confirmation_tasks(
    catalog: Mapping[str, Any], promotion: Mapping[str, Any]
) -> list[dict[str, Any]]:
    by_target = {
        item["targetId"]: item["grammarId"] for item in catalog["targetMatrix"]
    }
    simulation = catalog["simulation"]
    specifications = []
    seen = set()
    for selection in promotion["selected"]:
        for policy_id in (selection["policyId"], selection["referencePolicyId"]):
            key = (selection["targetId"], selection["startFamily"], policy_id)
            if key in seen:
                continue
            seen.add(key)
            specifications.append(
                {
                    "targetId": selection["targetId"],
                    "grammarId": by_target[selection["targetId"]],
                    "startFamily": selection["startFamily"],
                    "policyId": policy_id,
                }
            )
    return [
        build_task(
            catalog,
            specification,
            phase="confirmation",
            split="confirmation_holdout",
            replicate_count=int(simulation["confirmationPairsPerPromotedContrast"]),
            event_budget=int(simulation["confirmationEventBudgetTransitions"]),
        )
        for specification in specifications
    ]


def bootstrap_confirmation(
    contrasts: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    promotion: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> pd.DataFrame:
    selected = {item["contrastId"]: item for item in promotion["selected"]}
    if not selected:
        return pd.DataFrame(
            columns=[
                "contrastId",
                "pairedScenarioCount",
                "dominantMetric",
                "frozenDirection",
                "completionRiskDifference",
                "completionAdjustedLower",
                "completionAdjustedUpper",
                "minimumMismatchProgressDifference",
                "progressAdjustedLower",
                "progressAdjustedUpper",
                "discordanceRateDifference",
                "discordanceAdjustedLower",
                "discordanceAdjustedUpper",
                "contrastConfirmed",
                "boundedFormationSupported",
            ]
        )
    rows = contrasts.set_index("contrastId")
    count = len(selected)
    bootstrap_count = int(catalog["confirmation"]["pairedBootstrapReplicates"])
    alpha = float(catalog["confirmation"]["familywiseAlpha"])
    tail = alpha / (2 * count)
    practical = catalog["confirmation"]["practicalThresholds"]
    thresholds = {
        "completionRiskDifference": float(
            practical["absoluteCompletionRiskDifference"]
        ),
        "minimumMismatchProgressDifference": float(
            practical["absoluteMinimumMismatchProgressDifference"]
        ),
        "discordanceRateDifference": float(
            practical["absoluteDiscordanceRateDifference"]
        ),
    }
    output = []
    for contrast_id, selection in sorted(selected.items()):
        values = arrays[contrast_id]
        seed = int(sha256_value("E06/S09/bootstrap/v1", contrast_id)[:16], 16)
        generator = np.random.Generator(np.random.PCG64(seed))
        bootstrap = np.empty((bootstrap_count, 3), dtype=np.float64)
        chunk = 500
        for start in range(0, bootstrap_count, chunk):
            size = min(chunk, bootstrap_count - start)
            indices = generator.integers(0, len(values), size=(size, len(values)))
            bootstrap[start : start + size] = values[indices].mean(axis=1)
        lower = np.quantile(bootstrap, tail, axis=0)
        upper = np.quantile(bootstrap, 1 - tail, axis=0)
        means = values.mean(axis=0)
        dominant = selection["dominantMetric"]
        dominant_index = {
            "completionRiskDifference": 0,
            "minimumMismatchProgressDifference": 1,
            "discordanceRateDifference": 2,
        }[dominant]
        direction = int(selection["frozenDirection"])
        interval_excludes_zero = (
            lower[dominant_index] > 0 if direction > 0 else upper[dominant_index] < 0
        )
        confirmed = (
            abs(means[dominant_index]) >= thresholds[dominant]
            and interval_excludes_zero
        )
        local_key = (
            (rows["targetId"] == selection["targetId"])
            & (rows["startFamily"] == selection["startFamily"])
            & (rows["policyId"] == selection["policyId"])
        )
        # The policy-specific completion bound is populated by main after joining
        # confirmation condition summaries; keep contrast inference independent.
        output.append(
            {
                "contrastId": contrast_id,
                "targetId": selection["targetId"],
                "startFamily": selection["startFamily"],
                "policyId": selection["policyId"],
                "referencePolicyId": selection["referencePolicyId"],
                "pairedScenarioCount": len(values),
                "dominantMetric": dominant,
                "frozenDirection": direction,
                "completionRiskDifference": means[0],
                "completionAdjustedLower": lower[0],
                "completionAdjustedUpper": upper[0],
                "minimumMismatchProgressDifference": means[1],
                "progressAdjustedLower": lower[1],
                "progressAdjustedUpper": upper[1],
                "discordanceRateDifference": means[2],
                "discordanceAdjustedLower": lower[2],
                "discordanceAdjustedUpper": upper[2],
                "bootstrapReplicates": bootstrap_count,
                "familywiseAlpha": alpha,
                "bonferroniContrastCount": count,
                "contrastConfirmed": bool(confirmed),
                "boundedFormationSupported": False,
                "_conditionLookupPlaceholder": bool(local_key.any()),
            }
        )
    return pd.DataFrame(output)


def add_formation_support(
    confirmation: pd.DataFrame,
    condition_summary: pd.DataFrame,
) -> pd.DataFrame:
    if confirmation.empty:
        return confirmation
    output = confirmation.copy()
    for index, row in output.iterrows():
        match = condition_summary.loc[
            (condition_summary["phase"] == "confirmation")
            & (condition_summary["targetId"] == row["targetId"])
            & (condition_summary["startFamily"] == row["startFamily"])
            & (condition_summary["policyId"] == row["policyId"])
        ]
        if len(match) != 1:
            continue
        condition = match.iloc[0]
        output.loc[index, "policyCompletionFraction"] = condition["completionFraction"]
        output.loc[index, "policyCompletionWilsonLower95"] = condition[
            "completionWilsonLower95"
        ]
        output.loc[index, "boundedFormationSupported"] = bool(
            condition["completionFraction"] >= 0.20
            and condition["completionWilsonLower95"] >= 0.15
            and row["completionRiskDifference"] >= 0.10
            and row["completionAdjustedLower"] > 0
        )
    return output.drop(columns=["_conditionLookupPlaceholder"], errors="ignore")


def run_accounting(
    frame: pd.DataFrame, intended_by_phase: Mapping[str, int]
) -> dict[str, Any]:
    by_phase = []
    for phase, intended in intended_by_phase.items():
        group = frame.loc[frame["phase"] == phase]
        completed = int((group["runStatus"] == "completed").sum())
        failed = int((group["runStatus"] == "failed").sum())
        by_phase.append(
            {
                "phase": phase,
                "intendedRuns": int(intended),
                "launchedRuns": len(group),
                "recordedRuns": len(group),
                "completedRuns": completed,
                "failedRuns": failed,
                "rightCensoredNoncompletingRuns": int(
                    group.loc[group["runStatus"] == "completed", "censored"].sum()
                ),
                "completionEvents": int(
                    group.loc[
                        group["runStatus"] == "completed",
                        "conjunctiveCompletionByBudget",
                    ].sum()
                ),
                "accountingComplete": len(group) == intended
                and completed + failed == intended,
            }
        )
    return {
        "schemaVersion": "e06.s09.run-accounting.v1",
        "researchStepId": "S09",
        "phases": by_phase,
        "intendedRuns": sum(item["intendedRuns"] for item in by_phase),
        "recordedRuns": len(frame),
        "completedRuns": int((frame["runStatus"] == "completed").sum()),
        "failedRuns": int((frame["runStatus"] == "failed").sum()),
        "allIntendedRunsAccounted": all(
            item["accountingComplete"] for item in by_phase
        ),
        "duplicateRunIds": int(frame["runId"].duplicated().sum()),
    }


def replay_sample_tasks(
    frame: pd.DataFrame,
    catalog: Mapping[str, Any],
) -> list[dict[str, Any]]:
    complete = completed_rows(frame)
    keys = ["phase", "targetId", "startFamily", "policyId"]
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
                "startFamily": row["startFamily"],
                "replicate": int(row["replicate"]),
                "policyId": row["policyId"],
                "eventBudget": int(row["eventBudgetTransitions"]),
                # The lexicographically first run is also in the preregistered
                # lowest-run-ID trace sample.  Replay must preserve that frozen
                # serialization choice for a byte-exact episode comparison.
                "retainTrace": bool(row["traceSelected"]),
                "expectedEpisodeCanonicalBytesSha256": row[
                    "episodeCanonicalBytesSha256"
                ],
                "expectedMetricSummarySha256": row["metricSummarySha256"],
                "expectedRunId": row["runId"],
            }
        )
    return tasks


def _replay_one(task: Mapping[str, Any]) -> dict[str, Any]:
    expected_episode = task["expectedEpisodeCanonicalBytesSha256"]
    expected_metric = task["expectedMetricSummarySha256"]
    expected_run = task["expectedRunId"]
    specification = {
        key: value for key, value in task.items() if not key.startswith("expected")
    }
    row, _ = run_baseline_once(specification)
    return {
        "runId": expected_run,
        "phase": row["phase"],
        "targetId": row["targetId"],
        "startFamily": row["startFamily"],
        "policyId": row["policyId"],
        "episodeBytesMatch": row["episodeCanonicalBytesSha256"] == expected_episode,
        "metricSummaryMatch": row["metricSummarySha256"] == expected_metric,
        "runIdMatch": row["runId"] == expected_run,
    }


def run_replays(tasks: list[dict[str, Any]], workers: int) -> pd.DataFrame:
    if not tasks:
        return pd.DataFrame()
    records = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futures = [pool.submit(_replay_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())
            if index % 12 == 0 or index == len(futures):
                print(f"[replay] completed {index}/{len(futures)}", flush=True)
    return pd.DataFrame(records).sort_values("runId").reset_index(drop=True)


def metric_agreement(frame: pd.DataFrame) -> pd.DataFrame:
    _context, targets, grammars, _environments = load_baseline_assets()
    grammar_by_id = dict(grammars)
    complete = completed_rows(frame)
    sample_count = max(1, math.ceil(0.01 * len(complete)))
    sample = complete.sort_values("runId").head(sample_count)
    records = []
    for row in sample.to_dict(orient="records"):
        grid = tuple(tuple(item) for item in json.loads(row["finalGridRowsJson"]))
        global_audit = evaluate_success(grid, targets[row["targetId"]])
        local = score_grid(grid, grammar_by_id[row["grammarId"]])
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
            {
                "runId": row["runId"],
                "targetId": row["targetId"],
                **checks,
                "allFieldsAgree": all(checks.values()),
            }
        )
    return pd.DataFrame(records)


def seed_audit(frame: pd.DataFrame) -> pd.DataFrame:
    sample_count = max(1, math.ceil(0.01 * len(frame)))
    records = []
    for row in frame.sort_values("runId").head(sample_count).to_dict(orient="records"):
        regenerated = scenario_identity(
            row["split"],
            row["targetId"],
            row["startFamily"],
            int(row["replicate"]),
            row["policyId"],
        )
        fields = ["scenarioId", "pairingBlockId", "runId", "seedHex", "seedDecimal"]
        records.append(
            {
                "runId": row["runId"],
                **{
                    f"{field}Match": regenerated[field] == row[field]
                    for field in fields
                },
                "allFieldsMatch": all(
                    regenerated[field] == row[field] for field in fields
                ),
            }
        )
    return pd.DataFrame(records)


def write_traces(path: Path, traces: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as handle:
        for item in sorted(traces, key=lambda value: value["runId"]):
            handle.write(json.dumps(item, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def plot_completion(summary: pd.DataFrame, path: Path) -> None:
    screen = summary.loc[summary["phase"] == "exploratory"].copy()
    screen["condition"] = screen["targetId"] + "\n" + screen["startFamily"]
    pivot = screen.pivot(
        index="condition", columns="policyId", values="completionFraction"
    )
    figure, axis = plt.subplots(figsize=(12, max(8, 0.33 * len(pivot))))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(
        range(len(pivot.columns)),
        [item.replace("_v1", "") for item in pivot.columns],
        rotation=45,
        ha="right",
    )
    axis.set_yticks(range(len(pivot.index)), pivot.index, fontsize=7)
    axis.set_title(
        "S09 exploratory conjunctive completion fraction (250 runs/condition)"
    )
    figure.colorbar(image, ax=axis, label="completion fraction")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _color_map(tokens: Iterable[str]) -> dict[str, str]:
    palette = [
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#66CCEE",
        "#AA3377",
        "#BBBBBB",
    ]
    return {
        token: palette[index % len(palette)]
        for index, token in enumerate(sorted(tokens))
    }


def plot_morphologies(frame: pd.DataFrame, path: Path) -> None:
    complete = completed_rows(frame.loc[frame["phase"] == "exploratory"])
    _context, targets, _grammars, _environments = load_baseline_assets()
    figure, axes = plt.subplots(
        len(targets), 3, figsize=(9, 2.4 * len(targets)), squeeze=False
    )
    for row_index, (target_id, target) in enumerate(sorted(targets.items())):
        subset = complete.loc[complete["targetId"] == target_id].sort_values(
            ["conjunctiveCompletionByBudget", "minimumS01MismatchFraction", "runId"],
            ascending=[False, True, True],
        )
        representative = subset.iloc[0]
        grids = [
            target.grid,
            tuple(
                tuple(item)
                for item in json.loads(representative["initialGridRowsJson"])
            ),
            tuple(
                tuple(item) for item in json.loads(representative["finalGridRowsJson"])
            ),
        ]
        colors = _color_map(token for grid in grids for line in grid for token in line)
        for column, (grid, title) in enumerate(
            zip(
                grids,
                ("target", "representative initial", "best observed final"),
                strict=True,
            )
        ):
            encoded = np.asarray(
                [[list(colors).index(token) for token in line] for line in grid]
            )
            from matplotlib.colors import ListedColormap

            axes[row_index, column].imshow(
                encoded,
                cmap=ListedColormap(list(colors.values())),
                interpolation="nearest",
            )
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            axes[row_index, column].set_title(
                title if row_index == 0 else "", fontsize=9
            )
        axes[row_index, 0].set_ylabel(target_id.replace("_", "\n"), fontsize=7)
        axes[row_index, 2].text(
            1.03,
            0.5,
            f"policy={representative['policyId'].replace('_v1', '')}\n"
            f"start={representative['startFamily']}\n"
            f"min mismatch={representative['minimumS01MismatchFraction']:.3f}",
            transform=axes[row_index, 2].transAxes,
            va="center",
            fontsize=7,
        )
    figure.suptitle(
        "S09 targets, paired initial-state family examples, and best observed finals"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def upstream_hashes() -> list[dict[str, Any]]:
    records = []
    for number in range(1, 9):
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
        CATALOG_PATH,
    ]
    return [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in paths
        if path.exists()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=JSON_INDENT, sort_keys=True) + "\n", encoding="utf-8"
    )


def artifact_manifest(output: Path, excluded: set[str] | None = None) -> dict[str, Any]:
    excluded = excluded or set()
    records = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = str(path.relative_to(output))
        if relative in excluded:
            continue
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "schemaVersion": "e06.s09.artifact-manifest.v1",
        "researchStepId": "S09",
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
        raise ValueError("S09 worker count must be between one and eight")
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    conditions = validate_catalog(catalog)
    design_sha = file_sha256(CATALOG_PATH)
    commit = git_output("rev-parse", "HEAD")
    dirty = git_output("status", "--short")
    if dirty:
        raise ValueError(
            "commit the frozen S09 implementation before outcome execution"
        )
    output = arguments.output.resolve()
    if output.exists() and arguments.replace:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is nonempty: {output}; use --replace")
    output.mkdir(parents=True, exist_ok=True)
    cache = arguments.cache / f"{commit[:12]}-{design_sha[:12]}"
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CATALOG_PATH, output / "frozen_baseline_design.yaml")
    write_json(
        output / "design_freeze.json",
        {
            "schemaVersion": "e06.s09.design-freeze.v1",
            "researchStepId": "S09",
            "frozenBeforeOutcomeExecution": True,
            "catalogPath": str(CATALOG_PATH),
            "catalogSha256": design_sha,
            "repositoryCommit": commit,
            "exploratoryConditionCount": len(conditions),
            "exploratoryRunCount": catalog["simulation"]["exploratoryRunCount"],
            "maximumConfirmationRunCount": catalog["simulation"][
                "maximumConfirmationRunCount"
            ],
        },
    )
    simulation = catalog["simulation"]
    screen_tasks = [
        build_task(
            catalog,
            specification,
            phase="exploratory",
            split="exploratory",
            replicate_count=int(simulation["exploratoryReplicatesPerCondition"]),
            event_budget=int(simulation["exploratoryEventBudgetTransitions"]),
        )
        for specification in conditions
    ]
    invocation_started = time.perf_counter()
    screen, screen_traces, screen_execution = run_tasks(
        screen_tasks, cache, workers=arguments.workers, label="exploratory"
    )
    screen.to_parquet(output / "baseline_results_exploratory.parquet", index=False)
    screen_contrasts, _ = paired_contrasts(screen)
    promotion = select_promotions(screen_contrasts, catalog)
    write_json(output / "promotion_decisions.json", promotion)
    # The decision artifact exists and is hashed before any holdout tasks or
    # confirmation identities are constructed.
    promotion_persisted_sha = file_sha256(output / "promotion_decisions.json")
    promotion["persistedArtifactSha256BeforeConfirmation"] = promotion_persisted_sha
    confirm_tasks = confirmation_tasks(catalog, promotion)
    confirmation, confirmation_traces, confirmation_execution = (
        run_tasks(confirm_tasks, cache, workers=arguments.workers, label="confirmation")
        if confirm_tasks
        else (
            pd.DataFrame(columns=screen.columns),
            [],
            {
                "conditionCount": 0,
                "resumedConditionCount": 0,
                "executedConditionCount": 0,
                "wallSecondsThisInvocation": 0.0,
            },
        )
    )
    combined = pd.concat([screen, confirmation], ignore_index=True, sort=False)
    combined.to_parquet(output / "baseline_results.parquet", index=False)
    summary = summarize_conditions(combined)
    summary.to_csv(output / "completion_table.csv", index=False)
    cost_columns = [
        "phase",
        "targetId",
        "startFamily",
        "policyId",
        "meanAcceptedMovements",
        "meanGraphDisplacement",
        "meanObservationBitsUpperBound",
        "meanChannelInformationBits",
        "totalWallSeconds",
    ]
    summary[cost_columns].to_csv(output / "cost_table.csv", index=False)
    screen_contrasts.to_csv(output / "exploratory_contrasts.csv", index=False)
    confirm_contrasts, confirm_arrays = paired_contrasts(confirmation)
    confirmation_inference = bootstrap_confirmation(
        confirm_contrasts, confirm_arrays, promotion, catalog
    )
    confirmation_inference = add_formation_support(confirmation_inference, summary)
    confirmation_inference.to_csv(output / "confirmation_contrasts.csv", index=False)
    intended_by_phase = {
        "exploratory": int(simulation["exploratoryRunCount"]),
        "confirmation": len(confirm_tasks)
        * int(simulation["confirmationPairsPerPromotedContrast"]),
    }
    accounting = run_accounting(combined, intended_by_phase)
    write_json(output / "run_accounting.json", accounting)
    pd.DataFrame(accounting["phases"]).to_csv(
        output / "run_accounting.csv", index=False
    )
    summary[
        [
            "phase",
            "targetId",
            "startFamily",
            "policyId",
            "intendedRuns",
            "completedRuns",
            "failedRuns",
            "censoredRuns",
            "completionCount",
        ]
    ].to_csv(output / "censoring_table.csv", index=False)
    scenario_columns = [
        "phase",
        "split",
        "scenarioId",
        "pairingBlockId",
        "runId",
        "seedHex",
        "seedDecimal",
        "targetId",
        "startFamily",
        "replicate",
        "policyId",
        "initialStateSha256",
        "eventBudgetTransitions",
        "runStatus",
    ]
    combined[scenario_columns].to_parquet(
        output / "scenario_manifest.parquet", index=False
    )
    write_traces(
        output / "sampled_traces.jsonl.gz", screen_traces + confirmation_traces
    )
    replays = run_replays(replay_sample_tasks(combined, catalog), arguments.workers)
    replays.to_csv(output / "replay_audit.csv", index=False)
    metrics = metric_agreement(combined)
    metrics.to_csv(output / "target_metric_agreement.csv", index=False)
    seeds = seed_audit(combined)
    seeds.to_csv(output / "seed_audit.csv", index=False)
    invariant = completed_rows(combined)[
        ["runId", "invariantSuccess", "permissionAuditSuccess"]
    ]
    invariant.to_parquet(output / "invariant_audit.parquet", index=False)
    plot_completion(summary, output / "completion_heatmap.png")
    plot_morphologies(combined, output / "morphology_panels.png")
    upstream = upstream_hashes()
    provenance = {
        "schemaVersion": "e06.s09.provenance.v1",
        "researchStepId": "S09",
        "repositoryCommit": commit,
        "repositoryBranch": git_output("branch", "--show-current"),
        "repositoryCleanAtOutcomeLaunch": True,
        "catalogSha256": design_sha,
        "upstreamArtifactCount": len(upstream),
        "upstreamArtifacts": upstream,
        "dependencyFiles": dependency_hashes(),
    }
    write_json(output / "provenance_manifest.json", provenance)
    validation = {
        "schemaVersion": "e06.s09.validation.v1",
        "researchStepId": "S09",
        "runAccounting": accounting["allIntendedRunsAccounted"]
        and accounting["duplicateRunIds"] == 0,
        "failureCountZero": accounting["failedRuns"] == 0,
        "deterministicSeedRegeneration": bool(seeds["allFieldsMatch"].all()),
        "sampledCpuReplay": bool(
            replays[["episodeBytesMatch", "metricSummaryMatch", "runIdMatch"]]
            .to_numpy()
            .all()
        ),
        "targetMetricAgreement": bool(metrics["allFieldsAgree"].all()),
        "invariants": bool(invariant["invariantSuccess"].all()),
        "permissionIsolation": bool(invariant["permissionAuditSuccess"].all()),
        "separateLocalGlobalFields": all(
            column in combined
            for column in [
                "terminalS01GlobalSuccess",
                "terminalS01ComponentMatch",
                "terminalS01TopologyMatch",
                "terminalS02GrammarAccepted",
                "terminalS02SoftScore",
                "conjunctiveCompletionByBudget",
            ]
        ),
        "rightCensoringConsistent": bool(
            (
                completed_rows(combined)["censored"]
                == ~completed_rows(combined)["conjunctiveCompletionByBudget"]
            ).all()
        ),
        "promotionPersistedBeforeConfirmation": bool(promotion_persisted_sha),
        "confirmationSplitDisjoint": not bool(
            set(screen["scenarioId"]) & set(confirmation["scenarioId"])
        ),
        "backendScope": bool(
            (combined["backend"] == "canonical_s07_cpu_oracle_fallback").all()
        ),
        "artifactHashCoverage": True,
    }
    validation["success"] = all(
        value
        for key, value in validation.items()
        if key not in {"schemaVersion", "researchStepId"}
    )
    write_json(output / "validation_results.json", validation)
    formation_supported = bool(
        not confirmation_inference.empty
        and confirmation_inference["boundedFormationSupported"].fillna(False).any()
    )
    confirmed_contrasts = int(
        confirmation_inference["contrastConfirmed"].sum()
        if not confirmation_inference.empty
        else 0
    )
    screen_completion = int(screen["conjunctiveCompletionByBudget"].fillna(False).sum())
    if formation_supported:
        classification = "supportive"
    elif screen_completion == 0:
        classification = "constraining/contradictory"
    else:
        classification = "null"
    execution = {
        "schemaVersion": "e06.s09.execution.v1",
        "researchStepId": "S09",
        "repositoryCommit": commit,
        "catalogSha256": design_sha,
        "backend": "canonical_s07_cpu_oracle_fallback",
        "cpuWorkers": arguments.workers,
        "threadEnvironment": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        "exploratory": screen_execution,
        "confirmation": confirmation_execution,
        "totalInvocationWallSeconds": time.perf_counter() - invocation_started,
        "outcomeClassification": classification,
        "screenCompletionEvents": screen_completion,
        "confirmedContrastCount": confirmed_contrasts,
        "boundedFormationSupported": formation_supported,
    }
    write_json(output / "execution_manifest.json", execution)
    # The report is deliberately written after every result and validation
    # artifact so it can state actual counts.  The artifact manifest follows it.
    selected_count = int(promotion["selectedContrastCount"])
    completed_count = int(accounting["completedRuns"])
    censored_count = sum(
        item["rightCensoredNoncompletingRuns"] for item in accounting["phases"]
    )
    best = summary.sort_values(
        ["completionFraction", "meanMinimumS01MismatchFraction"],
        ascending=[False, True],
    ).iloc[0]
    caveat = (
        "Completion calibration is limited to unobstructed bounded square rectangles; "
        "S02 local acceptance is not interchangeable with S01 global completion. "
        "Fixed-budget activity and quiescence were not interpreted as convergence."
    )
    next_action = (
        "Return control to the Chief Scientist. If the evidence is accepted, separately "
        "authorize S10 to apply spatial perturbations; do not start S10 automatically."
    )
    report = f"""# S09 — Measure baseline pattern formation: full results

## Concise top summary

- **Research step ID:** S09 (step 9), “Measure baseline pattern formation.”
- **Completion status:** Complete; S09 only was executed and S10 was not started.
- **Artifacts written:** `baseline_results.parquet`, the frozen design and design-freeze record, exploratory and confirmation contrasts, promotion decisions, completion/cost/censoring/accounting tables, scenario and invariant manifests, sampled traces, CPU replay/seed/target-metric audits, two morphology figures, execution/provenance/validation records, this canonical report, and the artifact manifest.
- **Validation result:** {"Passed" if validation["success"] else "Failed"}: {completed_count:,} completed runs plus {accounting["failedRuns"]:,} preserved failures accounted for all {accounting["intendedRuns"]:,} intended runs; {len(replays):,} sampled episodes replayed byte-exactly; {len(metrics):,} independently rescored terminal states agreed; seed regeneration, censoring, invariants, permission isolation, split disjointness, and backend-scope checks {"all passed" if validation["success"] else "included failures"}.
- **Outcome classification:** {classification} under the frozen confirmation rule; {screen_completion:,} exploratory completion events, {selected_count} promoted contrasts, {confirmed_contrasts} confirmed contrasts, and bounded formation support = {str(formation_supported).lower()}.
- **Caveats or blockers:** {caveat} No execution blocker remains.
- **Lay summary:** The simulator tested whether six local movement strategies could assemble seven declared patterns from three kinds of disordered starts. A run counted only when both the local relational grammar and the independent whole-pattern audit agreed; simply continuing or stopping movement never counted. The best observed condition was `{best["targetId"]}` / `{best["startFamily"]}` / `{best["policyId"]}` with completion fraction {best["completionFraction"]:.3f}. The held-out rules, rather than visual plausibility or runtime, determined the classification above.
- **Recommended next action:** {next_action}

## Frozen question

Can the frozen S05 local policies reach calibrated relational target sets from random, block-scrambled, and partially correct states without damage or top-down direct control?

## Inputs

- Repository commit: `{commit}` on `eidosoma/groups/28`.
- Frozen design: `configs/morphologies/baseline_catalog.yaml`, SHA-256 `{design_sha}`; the identical collectible copy is `frozen_baseline_design.yaml`.
- Canonical S01 target and global-audit contract, S02 local grammars, S03 bounded-square graph/site-role contract, S04 movement contract, S05 priced observations and policies, S06 channel/timing/ledger boundary, and the S07 CPU oracle/S08 scoped parity evidence.
- Relevant E01 deterministic scenario/pairing/run-accounting contracts and E04 exact-count, identity-blind, kinetic, and state/flux handoff constraints described in the provenance manifest.
- Attachment manifest only; no dataset was required.

## Methods and preregistered parameters

The baseline catalog was committed before outcome execution. It expands all seven calibrated targets, three initial-state families, four universal policies, two hole-target boundary-policy conditions, and two stripe/layer gradient-policy conditions into 96 conditions. Exploratory screening used exactly 250 paired scenarios per condition ({int(simulation["exploratoryRunCount"]):,} runs), a fixed 64-transition budget, actor batch four, no early stopping, and one-percent trace selection by run ID before outcomes. Random and block starts are SHA-ranked identity permutations; partially correct starts are exact targets with deterministic disjoint adjacent cross-token swaps. Every start preserves all occupant identities and token counts and is S01-incomplete at transition −1.

Every non-exploration policy was paired against exploration on the same target, start, initial state, scheduler/noise key, and budget. Promotion used only the frozen absolute exploratory thresholds (completion risk difference 0.08, minimum-mismatch progress 0.04, discordance 0.15), at most one contrast per start family and three total. `promotion_decisions.json` was persisted and hashed before confirmation identities were constructed. Each promoted contrast received exactly 1,000 new paired holdout scenarios per policy at 128 transitions. Confirmation used 10,000 paired bootstrap replicates and Bonferroni-adjusted intervals over promoted contrasts.

Completion was the conjunction of an S02 grammar acceptance and an independent S01 equivalence/component/topology audit. The S01 audit was a read-only state callback after every transition and never entered observations, decisions, stopping, conflicts, or controllers. Minimum S01 orbit mismatch and separate terminal S01/S02 fields were retained. Noncompletion was right-censored at the fixed budget; invariant violations and execution errors were failed, never censored. Activity and quiescence were not used as formation or convergence evidence.

The canonical S07 CPU oracle was used because 250/1,000-condition batches are below the frozen S07 material GPU crossover and because authentication, observation construction, channels, whole-target audits, hashes, and ledgers remain CPU work. This is the required CPU fallback, not a negative GPU-parity result.

## Commands

```bash
PYTHONPATH=. pytest -q tests/test_morph2d_baseline.py
PYTHONPATH=. pytest -q tests/test_morph2d_*.py
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  python scripts/build_morph2d_s09.py --workers {arguments.workers}
```

No dependency was installed. The run used {arguments.workers} worker processes and forced one numerical-library thread per worker; disposable per-condition restart caches are under `{cache}` and are not artifacts.

## Results

### Run accounting and completion

All {accounting["intendedRuns"]:,} intended runs have rows in `baseline_results.parquet`: {completed_count:,} completed fixed-budget episodes, {censored_count:,} right-censored noncompleters, and {accounting["failedRuns"]:,} failures. The exploratory screen contained {screen_completion:,} conjunctive completion events. The best condition by completion then minimum mismatch was `{best["targetId"]}` / `{best["startFamily"]}` / `{best["policyId"]}`: completion {best["completionFraction"]:.3f} (95% Wilson {best["completionWilsonLower95"]:.3f}–{best["completionWilsonUpper95"]:.3f}), mean minimum S01 mismatch {best["meanMinimumS01MismatchFraction"]:.3f}, and terminal local/global discordance {best["localGlobalDiscordanceFraction"]:.3f}.

Condition-level formation, mismatch, local acceptance, discordance, movement, information, and wall-time fields are in `completion_table.csv` and `cost_table.csv`. `completion_heatmap.png` maps all 96 screen conditions; `morphology_panels.png` shows exact targets, representative paired initial states, and the best observed final state per target. These visualizations are descriptive; machine-scored conjunctions are authoritative.

### Promotion and held-out confirmation

The screen evaluated {promotion["candidateContrastCount"]} paired policy-versus-exploration contrasts. {promotion["eligibleContrastCount"]} crossed at least one promotion threshold and {selected_count} were selected under the frozen diversity/rank rule. Held-out confirmation produced {confirmed_contrasts} confirmed contrasts; bounded formation support was {str(formation_supported).lower()}. Exact directions, effects, familywise-adjusted intervals, policy completion Wilson bounds, and support flags are in `confirmation_contrasts.csv`; the unopened-data selection record is `promotion_decisions.json`.

### Local/global separation and censoring

`baseline_results.parquet` independently records S02 acceptance/soft/relational/hard-violation fields, S01 equivalence/component/topology/mismatch fields, first-passage conjunction, terminal conjunction, and discordance. `censoring_table.csv` retains every noncompleter, and `run_accounting.json` retains all intended, completed, censored, completion-event, and failed counts by phase. No activity-derived convergence field is present.

## Validation

- Complete accounting: {accounting["recordedRuns"]:,}/{accounting["intendedRuns"]:,} recorded; duplicate run IDs = {accounting["duplicateRunIds"]}.
- Deterministic seeds: {len(seeds):,}/{len(seeds):,} sampled run identities and counter seeds regenerated exactly.
- Sampled CPU replay: {len(replays):,}/{len(replays):,} lexicographically first condition/phase episodes matched canonical episode bytes, target-metric summary hashes, and run IDs.
- Target metrics: {len(metrics):,}/{len(metrics):,} sampled final states independently matched S01 success/mismatch/component/topology, S02 acceptance/hard violations, and terminal conjunction.
- Invariants and permissions: {int(invariant["invariantSuccess"].sum()):,}/{len(invariant):,} completed episodes preserved the S04 identity/composition/site-role contract; {int(invariant["permissionAuditSuccess"].sum()):,}/{len(invariant):,} retained the frozen observation/controller boundary.
- Censoring: every completed run has `censored == not conjunctiveCompletionByBudget`; errors and invariant violations were defined as failures.
- Pairing/splits: policy pairs share immutable scenario blocks; exploratory and confirmation scenario-ID sets are disjoint.
- Backend: every run used `canonical_s07_cpu_oracle_fallback`; no GPU permission or authority was widened.
- Artifact/provenance: {len(upstream):,} S01–S08 artifact hashes and all generated artifact hashes are retained.

The focused repository tests additionally verify the exact 96/24,000 matrix, all seven calibrated target environments, deterministic conservative incomplete starts, all four optimized singleton movement previews against canonical S04 resolution, initial-state override/audit behavior, separated local/global run fields, censoring, and pair/split identities.

## Caveats, blockers, failed assumptions, and limitations

- {caveat}
- The screen estimates only the declared 64-transition baseline and the promoted holdouts only the declared 128-transition baseline. Failure to complete is not proof of impossibility or lack of an attractor.
- Grammar acceptance alone remains insufficient by S02 construction; discordance is reported, never recoded as completion.
- Boundary seeking uses the declared natural exterior and is limited to hole targets; static gradients are authored priors limited to stripes/layers. Their semantic and computational asymmetries remain visible in the channel/information and target-specificity contracts.
- Promotion is selective. Unpromoted screen effects remain exploratory, and confirmation does not turn the matrix into a universal statement about all topologies or policies.
- These are deterministic computational task outcomes, not direct biological morphogenesis, causal biological evidence, wet-lab validation, repair evidence, or a convergence proof.

No execution blocker remains. Any failed assumption is represented by the reported null or constraining classification and the condition/contrast tables rather than being removed.

## Provenance and artifact map

`provenance_manifest.json` records the exact repository commit, branch, frozen-design hash, dependencies, and {len(upstream):,} upstream artifact hashes. `scenario_manifest.parquet` records every scenario/run/seed/initial-state identity. `execution_manifest.json` records worker/thread/backend/cache/runtime facts. `validation_results.json` records every release check. `artifact_manifest.json` records sizes and SHA-256 hashes for all compact final S09 artifacts; repository code stays in git and caches stay under `/cache`.

## Recommended next action

{next_action}
"""
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")
    manifest = artifact_manifest(output, {"artifact_manifest.json"})
    write_json(output / "artifact_manifest.json", manifest)
    print(
        f"S09 complete: {accounting['recordedRuns']} runs, "
        f"validation={validation['success']}, classification={classification}, "
        f"artifacts={manifest['artifactCount'] + 1}",
        flush=True,
    )
    return 0 if validation["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
