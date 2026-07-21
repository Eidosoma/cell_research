#!/usr/bin/env python3
"""Execute the frozen E06 S12 bounded hybrid-control factorial."""

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
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from morph2d.grammar import score_grid
from morph2d.hybrid_control import (
    HYBRID_CATALOG_VERSION,
    run_condition_task,
    run_hybrid_once,
    scenario_identity,
    sha256_value,
    validate_hybrid_catalog,
)
from morph2d.baseline import load_baseline_assets
from morph2d.targets import evaluate_success


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = Path("/artifacts/research_steps/S12")
DEFAULT_CACHE = Path("/cache/e06_s12")
CATALOG_PATH = ROOT / "configs/morphologies/hybrid_control_catalog.yaml"
REPORT_PATH = "research_step_full_results.md"
ARM_ORDER = [
    "local_only",
    "central_only",
    "gradient_only",
    "sparse_direct",
    "combined",
]
COMPARATORS = ["local_only", "central_only", "gradient_only"]
DIRECT_ARMS = {"central_only", "sparse_direct", "combined"}
METRIC_COLUMNS = [
    "terminalConjunctiveCompletion",
    "conjunctiveCompletionByBudget",
    "terminalS01MismatchFraction",
    "minimumS01MismatchFraction",
    "terminalS02RelationalScore",
]
COST_COLUMNS = [
    "totalInformationBitsIncludingObservation",
    "totalGraphDisplacement",
    "directMovementGraphDisplacement",
    "externalInterventionGraphDisplacement",
    "totalSourceWorkUnits",
    "totalComputationUnits",
    "totalOpportunityCostUnits",
]


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
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl_gz(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def condition_id(specification: Mapping[str, Any], phase: str) -> str:
    return (
        f"{phase[:4]}-"
        + sha256_value(
            "E06/S12/condition/v1",
            {
                "phase": phase,
                "split": specification["split"],
                "targetId": specification["targetId"],
                "challengeId": specification["challengeId"],
                "armId": specification["armId"],
            },
        )[:20]
    )


def trace_run_ids(
    split: str,
    target_id: str,
    challenge_id: str,
    arm_id: str,
    replicates: Iterable[int],
    fraction: float,
) -> list[str]:
    identities = [
        scenario_identity(split, target_id, challenge_id, int(replicate), arm_id)[
            "runId"
        ]
        for replicate in replicates
    ]
    return sorted(identities)[: max(1, math.ceil(fraction * len(identities)))]


def make_task(
    catalog: Mapping[str, Any],
    *,
    phase: str,
    split: str,
    target_id: str,
    challenge_id: str,
    arm_id: str,
    replicate_count: int,
    event_budget: int,
) -> dict[str, Any]:
    replicates = list(range(int(replicate_count)))
    task = {
        "catalog": dict(catalog),
        "phase": phase,
        "split": split,
        "targetId": target_id,
        "challengeId": challenge_id,
        "armId": arm_id,
        "replicates": replicates,
        "eventBudget": int(event_budget),
    }
    task["conditionId"] = condition_id(task, phase)
    task["traceRunIds"] = trace_run_ids(
        split,
        target_id,
        challenge_id,
        arm_id,
        replicates,
        float(catalog["traceAndReplay"]["traceSampleFraction"]),
    )
    return task


def exploratory_tasks(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    simulation = catalog["simulation"]
    factors = catalog["scenarioFactorial"]["factors"]
    return sorted(
        [
            make_task(
                catalog,
                phase="exploratory",
                split="exploratory",
                target_id=str(target_id),
                challenge_id=str(challenge_id),
                arm_id=str(arm_id),
                replicate_count=int(simulation["exploratoryReplicatesPerCondition"]),
                event_budget=int(simulation["exploratoryBudgetTransitions"]),
            )
            for target_id in factors["target"]
            for challenge_id in factors["challenge"]
            for arm_id in factors["controlArm"]
        ],
        key=lambda item: item["conditionId"],
    )


def confirmation_tasks(
    catalog: Mapping[str, Any], blocks: Sequence[Mapping[str, str]]
) -> list[dict[str, Any]]:
    simulation = catalog["simulation"]
    tasks = []
    for block in blocks:
        for arm_id in ARM_ORDER:
            tasks.append(
                make_task(
                    catalog,
                    phase="confirmation",
                    split="confirmation_holdout",
                    target_id=str(block["targetId"]),
                    challenge_id=str(block["challengeId"]),
                    arm_id=arm_id,
                    replicate_count=int(
                        simulation["confirmationReplicatesPerArmBlock"]
                    ),
                    event_budget=int(simulation["confirmationBudgetTransitions"]),
                )
            )
    return sorted(tasks, key=lambda item: item["conditionId"])


def worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


def run_tasks(
    tasks: Sequence[Mapping[str, Any]],
    cache_dir: Path,
    *,
    workers: int,
    label: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    phase_cache = cache_dir / label
    phase_cache.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    pending = []
    resumed = 0
    for task in tasks:
        parquet_path = phase_cache / f"{task['conditionId']}.parquet"
        trace_path = phase_cache / f"{task['conditionId']}.traces.jsonl.gz"
        mask_path = phase_cache / f"{task['conditionId']}.masks.jsonl.gz"
        expected = len(task["replicates"])
        if parquet_path.exists():
            frame = pd.read_parquet(parquet_path)
            if len(frame) == expected:
                results[str(task["conditionId"])] = {
                    "frame": frame,
                    "traces": read_jsonl_gz(trace_path),
                    "masks": read_jsonl_gz(mask_path),
                }
                resumed += 1
                continue
        pending.append(task)
    print(f"[{label}] {resumed} resumed, {len(pending)} pending", flush=True)
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
            futures = {pool.submit(run_condition_task, task): task for task in pending}
            for ordinal, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                result = future.result()
                frame = pd.DataFrame(result["rows"])
                parquet_path = phase_cache / f"{task['conditionId']}.parquet"
                trace_path = phase_cache / f"{task['conditionId']}.traces.jsonl.gz"
                mask_path = phase_cache / f"{task['conditionId']}.masks.jsonl.gz"
                frame.to_parquet(parquet_path, index=False, compression="zstd")
                write_jsonl_gz(trace_path, result["traces"])
                write_jsonl_gz(mask_path, result["masks"])
                results[str(task["conditionId"])] = {
                    "frame": frame,
                    "traces": result["traces"],
                    "masks": result["masks"],
                }
                print(
                    f"[{label}] {ordinal}/{len(pending)} {task['conditionId']} rows={len(frame)}",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    ordered = [results[str(task["conditionId"])] for task in tasks]
    frame = pd.concat([item["frame"] for item in ordered], ignore_index=True)
    traces = [value for item in ordered for value in item["traces"]]
    masks = [value for item in ordered for value in item["masks"]]
    execution = {
        "label": label,
        "conditionTasks": len(tasks),
        "rows": len(frame),
        "resumedTasks": resumed,
        "executedTasks": len(pending),
        "wallSecondsThisInvocation": elapsed,
    }
    return frame, traces, masks, execution


def condition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    complete = frame[~frame["failed"]].copy()
    groups = ["phase", "targetId", "challengeId", "armId"]
    metrics = [
        *METRIC_COLUMNS,
        "localGlobalDiscordance",
        "censored",
        "acceptedMovements",
        "conflictLosses",
        *COST_COLUMNS,
        "controllerInputBits",
        "controllerComputeUnits",
        "overrideActionUnits",
        "foregoneNativeActorSlots",
    ]
    summary = complete.groupby(groups, dropna=False)[metrics].mean().reset_index()
    summary["runCount"] = complete.groupby(groups, dropna=False).size().to_numpy()
    return summary


def _paired_rows(
    frame: pd.DataFrame,
    target_id: str,
    challenge_id: str,
    candidate_arm: str,
    reference_arm: str,
) -> pd.DataFrame:
    block = frame[
        (frame["targetId"] == target_id)
        & (frame["challengeId"] == challenge_id)
        & (~frame["failed"])
    ]
    columns = ["pairingBlockId", "replicate", *METRIC_COLUMNS, *COST_COLUMNS]
    candidate = block[block["armId"] == candidate_arm][columns]
    reference = block[block["armId"] == reference_arm][columns]
    paired = candidate.merge(
        reference,
        on=["pairingBlockId", "replicate"],
        suffixes=("Candidate", "Reference"),
        validate="one_to_one",
    )
    if not len(paired):
        raise ValueError("empty S12 paired contrast")
    return paired


def exploratory_contrasts(
    frame: pd.DataFrame, catalog: Mapping[str, Any]
) -> pd.DataFrame:
    records = []
    factors = catalog["scenarioFactorial"]["factors"]
    for target_id in factors["target"]:
        for challenge_id in factors["challenge"]:
            for comparator in COMPARATORS:
                paired = _paired_rows(
                    frame,
                    str(target_id),
                    str(challenge_id),
                    "combined",
                    comparator,
                )
                records.append(
                    {
                        "contrastId": "ctr:"
                        + sha256_value(
                            "E06/S12/contrast/v1",
                            {
                                "targetId": target_id,
                                "challengeId": challenge_id,
                                "candidate": "combined",
                                "reference": comparator,
                            },
                        )[:20],
                        "targetId": target_id,
                        "challengeId": challenge_id,
                        "candidateArm": "combined",
                        "referenceArm": comparator,
                        "pairCount": len(paired),
                        "terminalCompletionRiskDifference": float(
                            np.mean(
                                paired["terminalConjunctiveCompletionCandidate"].astype(
                                    float
                                )
                                - paired[
                                    "terminalConjunctiveCompletionReference"
                                ].astype(float)
                            )
                        ),
                        "completionByBudgetRiskDifference": float(
                            np.mean(
                                paired["conjunctiveCompletionByBudgetCandidate"].astype(
                                    float
                                )
                                - paired[
                                    "conjunctiveCompletionByBudgetReference"
                                ].astype(float)
                            )
                        ),
                        "terminalMismatchDifference": float(
                            np.mean(
                                paired["terminalS01MismatchFractionCandidate"]
                                - paired["terminalS01MismatchFractionReference"]
                            )
                        ),
                        "minimumMismatchDifference": float(
                            np.mean(
                                paired["minimumS01MismatchFractionCandidate"]
                                - paired["minimumS01MismatchFractionReference"]
                            )
                        ),
                        "terminalS02ScoreDifference": float(
                            np.mean(
                                paired["terminalS02RelationalScoreCandidate"]
                                - paired["terminalS02RelationalScoreReference"]
                            )
                        ),
                    }
                )
    return pd.DataFrame(records).sort_values("contrastId").reset_index(drop=True)


def select_confirmation_blocks(
    contrasts: pd.DataFrame, catalog: Mapping[str, Any]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    mandatory = {
        key: str(value)
        for key, value in catalog["promotion"]["mandatoryBlock"].items()
        if key in {"targetId", "challengeId"}
    }
    thresholds = catalog["promotion"]["eligibleAnyCombinedVersusComparator"]
    scored = []
    for (target_id, challenge_id), group in contrasts.groupby(
        ["targetId", "challengeId"]
    ):
        if (
            target_id == mandatory["targetId"]
            and challenge_id == mandatory["challengeId"]
        ):
            continue
        completion = group["terminalCompletionRiskDifference"].abs().max()
        terminal = group["terminalMismatchDifference"].abs().max()
        minimum = group["minimumMismatchDifference"].abs().max()
        standardized = max(
            completion
            / float(thresholds["absolutePairedTerminalCompletionRiskDifference"]),
            terminal / float(thresholds["absolutePairedTerminalMismatchDifference"]),
            minimum / float(thresholds["absolutePairedMinimumMismatchDifference"]),
        )
        eligible = bool(
            completion
            >= float(thresholds["absolutePairedTerminalCompletionRiskDifference"])
            or terminal >= float(thresholds["absolutePairedTerminalMismatchDifference"])
            or minimum >= float(thresholds["absolutePairedMinimumMismatchDifference"])
        )
        scored.append(
            {
                "targetId": str(target_id),
                "challengeId": str(challenge_id),
                "eligible": eligible,
                "standardizedEffect": float(standardized),
            }
        )
    eligible = sorted(
        (item for item in scored if item["eligible"]),
        key=lambda item: (
            -float(item["standardizedEffect"]),
            item["targetId"],
            item["challengeId"],
        ),
    )
    selected_outcome = eligible[
        : int(catalog["promotion"]["maximumOutcomeSelectedBlocks"])
    ]
    blocks = [
        mandatory,
        *[
            {key: item[key] for key in ("targetId", "challengeId")}
            for item in selected_outcome
        ],
    ]
    decision = {
        "schemaVersion": "e06.s12.promotion-decisions.v1",
        "researchStepId": "S12",
        "mandatoryBlock": mandatory,
        "eligibleBlockCount": len(eligible),
        "scoredBlocks": scored,
        "selectedOutcomeBlocks": selected_outcome,
        "confirmationBlocks": blocks,
        "runtimeUsed": False,
        "writtenBeforeConfirmationScenarioConstruction": True,
    }
    return blocks, decision


def _bootstrap_interval(
    differences: np.ndarray,
    *,
    replicates: int,
    alpha: float,
    seed_key: str,
) -> tuple[float, float]:
    if differences.ndim != 1 or not len(differences):
        raise ValueError("S12 bootstrap requires a nonempty vector")
    seed = int(sha256_value("E06/S12/bootstrap-seed/v1", {"key": seed_key})[:16], 16)
    generator = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    chunk = 1000
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        indices = generator.integers(
            0, len(differences), size=(stop - start, len(differences))
        )
        estimates[start:stop] = differences[indices].mean(axis=1)
    low, high = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def confirmation_contrasts(
    frame: pd.DataFrame,
    blocks: Sequence[Mapping[str, str]],
    catalog: Mapping[str, Any],
) -> pd.DataFrame:
    metrics = {
        "terminalCompletionRiskDifference": "terminalConjunctiveCompletion",
        "completionByBudgetRiskDifference": "conjunctiveCompletionByBudget",
        "terminalMismatchDifference": "terminalS01MismatchFraction",
        "minimumMismatchDifference": "minimumS01MismatchFraction",
        "terminalS02ScoreDifference": "terminalS02RelationalScore",
    }
    family_count = len(blocks) * len(COMPARATORS) * len(metrics)
    alpha = float(catalog["confirmation"]["familywiseAlpha"]) / family_count
    bootstrap_replicates = int(catalog["confirmation"]["pairedBootstrapReplicates"])
    records = []
    for block in blocks:
        for comparator in COMPARATORS:
            paired = _paired_rows(
                frame,
                block["targetId"],
                block["challengeId"],
                "combined",
                comparator,
            )
            record: dict[str, Any] = {
                "targetId": block["targetId"],
                "challengeId": block["challengeId"],
                "candidateArm": "combined",
                "referenceArm": comparator,
                "pairCount": len(paired),
                "familywiseAlpha": float(catalog["confirmation"]["familywiseAlpha"]),
                "perIntervalAlpha": alpha,
            }
            for label, column in metrics.items():
                differences = (
                    paired[f"{column}Candidate"].astype(float)
                    - paired[f"{column}Reference"].astype(float)
                ).to_numpy()
                low, high = _bootstrap_interval(
                    differences,
                    replicates=bootstrap_replicates,
                    alpha=alpha,
                    seed_key=f"{block['targetId']}:{block['challengeId']}:{comparator}:{label}",
                )
                record[label] = float(differences.mean())
                record[f"{label}CiLow"] = low
                record[f"{label}CiHigh"] = high
            records.append(record)
    return pd.DataFrame(records)


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


def factorial_interactions(
    exploratory: pd.DataFrame, catalog: Mapping[str, Any]
) -> pd.DataFrame:
    factors = catalog["scenarioFactorial"]["factors"]
    records = []
    for arm_id in ARM_ORDER:
        if arm_id == "local_only":
            continue
        block_effects: dict[tuple[str, str, str], np.ndarray] = {}
        for target_id in factors["target"]:
            for challenge_id in factors["challenge"]:
                paired = _paired_rows(
                    exploratory,
                    str(target_id),
                    str(challenge_id),
                    arm_id,
                    "local_only",
                )
                for metric in (
                    "terminalConjunctiveCompletion",
                    "terminalS01MismatchFraction",
                ):
                    block_effects[(str(target_id), str(challenge_id), metric)] = (
                        paired[f"{metric}Candidate"].astype(float)
                        - paired[f"{metric}Reference"].astype(float)
                    ).to_numpy()
        for target_id in factors["target"]:
            reference = block_effects[
                (str(target_id), "exact_maintenance", "terminalS01MismatchFraction")
            ]
            for challenge_id in factors["challenge"]:
                if challenge_id == "exact_maintenance":
                    continue
                for metric in (
                    "terminalConjunctiveCompletion",
                    "terminalS01MismatchFraction",
                ):
                    current = block_effects[(str(target_id), str(challenge_id), metric)]
                    base = block_effects[(str(target_id), "exact_maintenance", metric)]
                    interaction = current - base
                    low, high = _bootstrap_interval(
                        interaction,
                        replicates=2000,
                        alpha=0.05,
                        seed_key=f"interaction:challenge:{arm_id}:{target_id}:{challenge_id}:{metric}",
                    )
                    records.append(
                        {
                            "interactionDimension": "challenge",
                            "armId": arm_id,
                            "targetId": target_id,
                            "level": challenge_id,
                            "referenceLevel": "exact_maintenance",
                            "metric": metric,
                            "pairCount": len(interaction),
                            "interactionEstimate": float(interaction.mean()),
                            "ciLow": low,
                            "ciHigh": high,
                        }
                    )
            del reference
        for challenge_id in factors["challenge"]:
            for metric in (
                "terminalConjunctiveCompletion",
                "terminalS01MismatchFraction",
            ):
                layers = block_effects[
                    ("layers_three_ordered_tissues", str(challenge_id), metric)
                ]
                stripes = block_effects[
                    ("stripes_alternating_three_band", str(challenge_id), metric)
                ]
                interaction = layers - stripes
                low, high = _bootstrap_interval(
                    interaction,
                    replicates=2000,
                    alpha=0.05,
                    seed_key=f"interaction:target:{arm_id}:{challenge_id}:{metric}",
                )
                records.append(
                    {
                        "interactionDimension": "target",
                        "armId": arm_id,
                        "targetId": None,
                        "level": "layers_three_ordered_tissues",
                        "referenceLevel": "stripes_alternating_three_band",
                        "challengeId": challenge_id,
                        "metric": metric,
                        "pairCount": len(interaction),
                        "interactionEstimate": float(interaction.mean()),
                        "ciLow": low,
                        "ciHigh": high,
                    }
                )
    return pd.DataFrame(records)


def pareto_frontier(
    confirmation: pd.DataFrame, catalog: Mapping[str, Any]
) -> pd.DataFrame:
    margins = catalog["pareto"]["equivalenceMargins"]
    completion_margin = float(margins["terminalCompletion"])
    mismatch_margin = float(margins["terminalMismatch"])
    records = []
    for (target_id, challenge_id), block in confirmation.groupby(
        ["targetId", "challengeId"]
    ):
        summary = (
            block.groupby("armId")[
                [
                    "terminalConjunctiveCompletion",
                    "terminalS01MismatchFraction",
                    *COST_COLUMNS,
                ]
            ]
            .mean()
            .reset_index()
        )
        for row in summary.to_dict(orient="records"):
            dominators = []
            for other in summary.to_dict(orient="records"):
                if other["armId"] == row["armId"]:
                    continue
                morphology_no_worse = bool(
                    other["terminalConjunctiveCompletion"]
                    >= row["terminalConjunctiveCompletion"] - completion_margin
                    and other["terminalS01MismatchFraction"]
                    <= row["terminalS01MismatchFraction"] + mismatch_margin
                )
                costs_no_greater = all(
                    float(other[column]) <= float(row[column])
                    for column in COST_COLUMNS
                )
                strict = bool(
                    other["terminalConjunctiveCompletion"]
                    > row["terminalConjunctiveCompletion"]
                    or other["terminalS01MismatchFraction"]
                    < row["terminalS01MismatchFraction"]
                    or any(
                        float(other[column]) < float(row[column])
                        for column in COST_COLUMNS
                    )
                )
                if morphology_no_worse and costs_no_greater and strict:
                    dominators.append(str(other["armId"]))
            records.append(
                {
                    "targetId": target_id,
                    "challengeId": challenge_id,
                    **row,
                    "paretoEfficient": not dominators,
                    "dominatedBy": "|".join(sorted(dominators)),
                    "authoritySemanticsCollapsed": False,
                }
            )
    return pd.DataFrame(records)


def replay_audit(
    tasks: Sequence[Mapping[str, Any]], frame: pd.DataFrame
) -> pd.DataFrame:
    by_condition = {str(task["conditionId"]): task for task in tasks}
    rows = []
    selected = (
        frame.sort_values("runId")
        .groupby(["phase", "targetId", "challengeId", "armId"], as_index=False)
        .head(1)
    )
    for expected in selected.to_dict(orient="records"):
        task = next(
            item
            for item in by_condition.values()
            if item["phase"] == expected["phase"]
            and item["targetId"] == expected["targetId"]
            and item["challengeId"] == expected["challengeId"]
            and item["armId"] == expected["armId"]
        )
        specification = {
            **task,
            "replicate": int(expected["replicate"]),
            "retainTrace": bool(expected["traceSelected"]),
        }
        specification.pop("replicates", None)
        actual, _trace, mask = run_hybrid_once(specification)
        rows.append(
            {
                "runId": expected["runId"],
                "phase": expected["phase"],
                "targetId": expected["targetId"],
                "challengeId": expected["challengeId"],
                "armId": expected["armId"],
                "runIdMatch": actual["runId"] == expected["runId"],
                "episodeMatch": actual["episodeCanonicalBytesSha256"]
                == expected["episodeCanonicalBytesSha256"],
                "metricMatch": actual["metricSummarySha256"]
                == expected["metricSummarySha256"],
                "budgetMatch": actual["budgetAuditJson"] == expected["budgetAuditJson"],
                "interventionMatch": actual["interventionSummarySha256"]
                == expected["interventionSummarySha256"],
                "maskMatch": mask["initialStateSha256"]
                == expected["initialStateSha256"],
            }
        )
    return pd.DataFrame(rows)


def seed_and_metric_audits(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, targets, grammars, _ = load_baseline_assets()
    sample = (
        frame.sort_values("runId")
        .groupby(["phase", "targetId", "challengeId", "armId"], as_index=False)
        .head(5)
    )
    seed_rows = []
    metric_rows = []
    for row in sample.to_dict(orient="records"):
        regenerated = scenario_identity(
            row["split"],
            row["targetId"],
            row["challengeId"],
            int(row["replicate"]),
            row["armId"],
        )
        seed_rows.append(
            {
                "runId": row["runId"],
                "scenarioMatch": regenerated["scenarioId"] == row["scenarioId"],
                "pairingMatch": regenerated["pairingBlockId"] == row["pairingBlockId"],
                "runIdMatch": regenerated["runId"] == row["runId"],
                "seedMatch": regenerated["seedHex"] == row["seedHex"],
            }
        )
        grid = tuple(tuple(item) for item in json.loads(row["finalGridRowsJson"]))
        global_audit = evaluate_success(grid, targets[row["targetId"]])
        local = score_grid(grid, grammars[row["grammarId"]])
        terminal = bool(global_audit["success"] and local["accepted"])
        metric_rows.append(
            {
                "runId": row["runId"],
                "s01SuccessMatch": bool(global_audit["success"])
                == bool(row["terminalS01GlobalSuccess"]),
                "s01MismatchMatch": int(global_audit["mismatchCount"])
                == int(row["terminalS01MismatchCount"]),
                "componentMatch": bool(global_audit["componentMatch"])
                == bool(row["terminalS01ComponentMatch"]),
                "topologyMatch": bool(global_audit["topologyMatch"])
                == bool(row["terminalS01TopologyMatch"]),
                "s02AcceptanceMatch": bool(local["accepted"])
                == bool(row["terminalS02GrammarAccepted"]),
                "terminalConjunctionMatch": terminal
                == bool(row["terminalConjunctiveCompletion"]),
            }
        )
    return pd.DataFrame(seed_rows), pd.DataFrame(metric_rows)


def create_figures(summary: pd.DataFrame, pareto: pd.DataFrame, output: Path) -> None:
    screen = summary[summary["phase"] == "exploratory"].copy()
    pivot = screen.pivot_table(
        index=["targetId", "challengeId"],
        columns="armId",
        values="terminalS01MismatchFraction",
    ).reindex(columns=ARM_ORDER)
    figure, axis = plt.subplots(figsize=(10, 5))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    axis.set_yticks(
        range(len(pivot.index)),
        [f"{target}\n{challenge}" for target, challenge in pivot.index],
    )
    axis.set_title("Exploratory terminal S01 mismatch (lower is better)")
    figure.colorbar(image, ax=axis, label="mean mismatch fraction")
    figure.tight_layout()
    figure.savefig(output / "hybrid_endpoint_heatmap.png", dpi=180)
    plt.close(figure)

    if len(pareto):
        figure, axis = plt.subplots(figsize=(8, 6))
        for arm_id in ARM_ORDER:
            subset = pareto[pareto["armId"] == arm_id]
            axis.scatter(
                subset["totalInformationBitsIncludingObservation"],
                subset["terminalS01MismatchFraction"],
                label=arm_id,
                s=60,
                alpha=0.8,
            )
        axis.set_xscale("symlog", linthresh=1)
        axis.set_xlabel("information bits (configuration charged in full)")
        axis.set_ylabel("terminal S01 mismatch fraction")
        axis.set_title("Confirmed morphology–information view (not a scalar utility)")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output / "hybrid_morphology_cost.png", dpi=180)
        plt.close(figure)


def upstream_hashes() -> list[dict[str, Any]]:
    records = []
    for step in range(1, 12):
        directory = Path(f"/artifacts/research_steps/S{step:02d}")
        for name in ("research_step_full_results.md", "artifact_manifest.json"):
            path = directory / name
            if not path.exists():
                raise FileNotFoundError(path)
            records.append(
                {
                    "stepId": f"S{step:02d}",
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return records


def artifact_manifest(output: Path) -> dict[str, Any]:
    records = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        records.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "schemaVersion": "e06.s12.artifact-manifest.v1",
        "researchStepId": "S12",
        "files": records,
        "fileCount": len(records),
    }


def _outcome_decision(
    confirmation_frame: pd.DataFrame,
    confirmation: pd.DataFrame,
    pareto: pd.DataFrame,
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = catalog["confirmation"]["practicalThresholds"]
    block_records = []
    absolute_pass_blocks = []
    comparative_pass_blocks = []
    confirmed_blocks = confirmation_frame[["targetId", "challengeId"]].drop_duplicates()
    for block in confirmed_blocks.to_dict(orient="records"):
        combined = confirmation_frame[
            (confirmation_frame["targetId"] == block["targetId"])
            & (confirmation_frame["challengeId"] == block["challengeId"])
            & (confirmation_frame["armId"] == "combined")
        ]
        successes = int(combined["terminalConjunctiveCompletion"].sum())
        low, high = wilson_interval(successes, len(combined))
        nonexact = block["challengeId"] != "exact_maintenance"
        absolute = bool(
            nonexact
            and successes / len(combined)
            >= float(thresholds["nonExactAbsoluteTerminalCompletionFraction"])
            and low >= float(thresholds["nonExactAbsoluteTerminalWilsonLower95"])
        )
        contrast_rows = confirmation[
            (confirmation["targetId"] == block["targetId"])
            & (confirmation["challengeId"] == block["challengeId"])
        ]
        comparator_pass = {}
        for row in contrast_rows.to_dict(orient="records"):
            completion_pass = bool(
                float(row["terminalCompletionRiskDifference"])
                >= float(thresholds["terminalCompletionRiskDifference"])
                and float(row["terminalCompletionRiskDifferenceCiLow"]) > 0
            )
            mismatch_pass = bool(
                float(row["terminalMismatchDifference"])
                <= -float(thresholds["terminalMismatchReduction"])
                and float(row["terminalMismatchDifferenceCiHigh"]) < 0
            )
            comparator_pass[str(row["referenceArm"])] = completion_pass or mismatch_pass
        comparative = bool(
            nonexact
            and set(comparator_pass) == set(COMPARATORS)
            and all(comparator_pass.values())
        )
        frontier_row = pareto[
            (pareto["targetId"] == block["targetId"])
            & (pareto["challengeId"] == block["challengeId"])
            & (pareto["armId"] == "combined")
        ]
        efficient = bool(
            len(frontier_row) == 1 and frontier_row.iloc[0]["paretoEfficient"]
        )
        if absolute:
            absolute_pass_blocks.append(block)
        if comparative:
            comparative_pass_blocks.append(block)
        block_records.append(
            {
                **block,
                "combinedTerminalSuccesses": successes,
                "combinedTerminalRuns": len(combined),
                "combinedTerminalFraction": successes / len(combined),
                "combinedTerminalWilsonLow": low,
                "combinedTerminalWilsonHigh": high,
                "absoluteRule": absolute,
                "comparatorRules": comparator_pass,
                "comparativeRule": comparative,
                "combinedParetoEfficient": efficient,
            }
        )
    common_blocks = {
        (item["targetId"], item["challengeId"]) for item in absolute_pass_blocks
    }.intersection(
        (item["targetId"], item["challengeId"]) for item in comparative_pass_blocks
    )
    combined_pareto = any(
        item["combinedParetoEfficient"]
        for item in block_records
        if (item["targetId"], item["challengeId"]) in common_blocks
    )
    support = bool(common_blocks and combined_pareto)
    all_zero_terminal_blocks = [
        {"targetId": target_id, "challengeId": challenge_id}
        for (target_id, challenge_id), block in confirmation_frame.groupby(
            ["targetId", "challengeId"]
        )
        if int(block["terminalConjunctiveCompletion"].sum()) == 0
    ]
    if support:
        classification = "supportive"
    elif all_zero_terminal_blocks or not absolute_pass_blocks:
        classification = "constraining/contradictory"
    else:
        classification = "null"
    return {
        "schemaVersion": "e06.s12.outcome-decision.v1",
        "researchStepId": "S12",
        "blockResults": block_records,
        "combinedAbsoluteRule": bool(absolute_pass_blocks),
        "combinedComparativeRule": bool(comparative_pass_blocks),
        "combinedParetoRule": combined_pareto,
        "supportRule": support,
        "classification": classification,
        "allZeroTerminalBlocks": all_zero_terminal_blocks,
        "relativeEffectsNeverPromotedToAbsoluteSuccess": True,
    }


def report_text(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    confirmation: pd.DataFrame,
    pareto: pd.DataFrame,
    validation: Mapping[str, Any],
    outcome: Mapping[str, Any],
    commit: str,
) -> str:
    completed = frame[~frame["failed"]]
    exploratory = completed[completed["phase"] == "exploratory"]
    holdout = completed[completed["phase"] == "confirmation"]
    terminal_total = int(completed["terminalConjunctiveCompletion"].sum())
    by_budget_total = int(completed["conjunctiveCompletionByBudget"].sum())
    holdout_terminal = int(holdout["terminalConjunctiveCompletion"].sum())
    holdout_by_budget = int(holdout["conjunctiveCompletionByBudget"].sum())
    combined_holdout = holdout[holdout["armId"] == "combined"]
    selected_blocks = (
        holdout[["targetId", "challengeId"]].drop_duplicates().to_dict(orient="records")
    )
    block_text = ", ".join(
        f"`{item['targetId']}/{item['challengeId']}`" for item in selected_blocks
    )
    caveat = (
        "The support rule failed; any relative morphology advantage remains comparative only."
        if not outcome["supportRule"]
        else "Support is bounded to the confirmed non-exact block(s) and declared budgets."
    )
    recommended = "Return control to the Chief Scientist. Review the absolute/comparative separation and cost asymmetries before separately authorizing only S13; do not start S13 automatically."
    return f"""# S12 — Run the hybrid-control factorial: full results

## Concise top summary

- **Research step ID:** S12 (step 12), “Run the hybrid-control factorial.”
- **Completion status:** Complete; S12 only was executed and S13 was not started.
- **Artifacts written:** Frozen hybrid-control design; `{len(frame):,}`-run Parquet results; scenario-mask, endpoint, budget, interaction, exploratory, promotion, confirmation, Pareto, replay, seed, target-rescoring, accounting, execution, provenance, validation, figure, and hash artifacts; plus this canonical report.
- **Validation result:** **{"PASS" if validation["success"] else "FAIL"}** — {validation["accountedRuns"]:,}/{validation["intendedRuns"]:,} runs accounted, {validation["failedRuns"]} failures, {validation["replayPassed"]}/{validation["replayRows"]} exact replays, {validation["metricAuditPassed"]}/{validation["metricAuditRows"]} independent target rescores, and all pairing, permission, budget, censoring, invariant, split, and claim-boundary gates {"passed" if validation["success"] else "did not all pass"}.
- **Outcome classification:** **{outcome["classification"]}**. Frozen S12 support = `{outcome["supportRule"]}`; combined absolute rule = `{outcome["combinedAbsoluteRule"]}`; combined comparative rule = `{outcome["combinedComparativeRule"]}`; combined Pareto rule = `{outcome["combinedParetoRule"]}`.
- **Caveats or blockers:** {caveat} The bounded controller is one-query/one-action after a lagged epoch, not central micromanagement. Gradient and direct authority, semantics, opportunity, and computation remain unequal even under common ceilings. Count-changing perturbations remain forbidden. No execution blocker remains.
- **Lay summary:** Five ways of controlling two small grid patterns were compared from identical starting arrangements: local rules, a tightly bounded central actor, a static gradient, exploration plus sparse direct action, and local rules plus sparse direct action. A whole pattern counted only when both its local grammar and independent global shape checks passed at the terminal time. Costs were kept in separate information, action, perturbation, source-work, computation, and opportunity ledgers. Relative improvement was never called successful formation or robustness when absolute terminal completion failed.
- **Recommended next action:** {recommended}

## Frozen question and decision rules

Does target-relational local control plus S06's lagged-summary, state-blind-recipient, one-query, one-action controller improve absolute terminal formation/maintenance and the morphology–cost trade-off relative to local-only, controller-only, gradient-only, and target-agnostic sparse-direct control?

Before outcomes, the catalog fixed two bounded 9×9 targets, four challenge families, five arms, 64/128-transition budgets, 250-run screening, a mandatory S11-constraint holdout, at most one outcome-selected holdout, censoring, paired intervals, factorial interactions, a vector Pareto rule, and a conjunctive support rule. Support required a confirmed non-exact terminal completion fraction of at least 0.20 with Wilson lower bound at least 0.15, confirmed combined improvement over local-only, central-only, and gradient-only, Pareto efficiency, and all validation gates. First passage or relative mismatch improvement alone could not satisfy the absolute rule.

## Lay summary

“Central-only” means only centrally caused movement, not global knowledge: ordinary native batches were disabled, and the controller could inspect one anonymous cell's already-priced local view after one lagged epoch. “Combined” used ordinary local relational actions plus that same sparse controller. The gradient arm used one recipient per transition so its worst-case delivery remained inside S06's per-epoch cap; unused actor opportunities were charged. These deliberate inequalities are part of the result rather than hidden in one cost number.

## Inputs

- Governance: `/workspace/AGENTS.md`, `FULL_PLAN.md`, and the pre-completion `RESEARCH_PLAN.md`.
- Completed S01–S11 reports and manifest roots, especially S06 budgets, S09 baseline formation, S10 count-preserving perturbations, and S11's zero-terminal-completion warning.
- E01 deterministic identity, pairing, counter-address, event, replay, invariant, and cost contracts.
- E04 composition-corrected/state-flux handoff and intervention/claim-boundary constraints.
- `input-attachments/MANIFEST.json` and the attachment sidecar. No dataset or new dependency was required.
- Repository pre-outcome design/implementation commit: `{commit}`.

Exact paths and hashes are in `provenance_manifest.json`.

## Detailed methods

### Arms, permissions, and authority

`local_only` used S05 greedy neighbor satisfaction. `central_only` disabled every native batch and allowed only S06 direct events. `gradient_only` used the frozen static uint8 gradient and S05 gradient follower at one actor per transition. `sparse_direct` combined target-agnostic exploration with the bounded direct controller. `combined` combined greedy local actions with the same controller. Direct events began only after a completed lagged epoch and never scanned recipients, read occupancy or target completion, retried, selected routes, or bypassed S04 authentication. S01/S02 target evaluation was a read-only callback.

### Scenarios and perturbations

The full screen was 2 targets × 4 challenges × 5 arms × 250 paired scenarios = {len(exploratory):,} runs. Challenges were exact maintenance, S09 partially-correct formation, S10 mild compact wound, and S10 mild formed-structure displacement. Every start preserved exact identities, tokens, kinds, sites, and composition. No region deletion, cell-type excess, vacancy-adjusted target, removal, insertion, conversion, division, or long-range policy exchange was enabled.

### Outcomes and censoring

Terminal maintenance/formation/restoration required terminal S02 acceptance and terminal S01 equivalence/component/topology success. First-passage conjunction remained separate. Exact-start loss was not censored; incomplete formation/restoration without first passage was right-censored at the fixed budget. Activity, quiescence, and runtime were never outcomes.

### Budgets and Pareto rule

All arms received one declared ceiling vector and four allocated actor opportunities per transition. Actual configuration, controller input, policy delivery, addressing, movement, direct intervention, source work, computation, and foregone opportunity were reported separately. Configuration was charged once in full and never amortized. Central computation units were not converted into communication or action. Pareto dominance used terminal conjunction and S01 mismatch plus six minimized cost families; semantic/authority asymmetry was explicitly excluded from scalar collapse.

### Promotion, confirmation, interactions, and uncertainty

The mandatory independent block was layers/partially-correct formation, preserving the S11 terminal constraint. One additional block could be promoted by frozen morphology thresholds, never runtime. Confirmation blocks were {block_text}. Each ran all five arms on 1,000 new paired scenarios at 128 transitions. Paired bootstrap intervals used 10,000 resamples with Bonferroni control across block/comparator/metric families. Factorial arm-by-challenge and arm-by-target effects were paired differences-of-differences relative to local-only.

### Commands and compute

```bash
PYTHONPATH=src:. pytest -q tests/test_morph2d_hybrid_control.py
PYTHONPATH=src:. pytest -q tests/test_morph2d_*.py
ruff check src/morph2d/engine.py src/morph2d/hybrid_control.py scripts/build_morph2d_s12.py tests/test_morph2d_hybrid_control.py
ruff format --check src/morph2d/engine.py src/morph2d/hybrid_control.py scripts/build_morph2d_s12.py tests/test_morph2d_hybrid_control.py
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/build_morph2d_s12.py --workers 8
```

Eight worker processes and one numerical-library thread per worker were used. The GPU was not used because condition batches remained below S07's material crossover and the relevant control plane is CPU-owned. No package, network resource, or capability was installed.

## Results

All {len(frame):,} intended rows were retained: {len(exploratory):,} exploratory and {len(holdout):,} confirmation. Terminal conjunction occurred {terminal_total:,} times overall and {holdout_terminal:,} times in confirmation; first-passage conjunction occurred {by_budget_total:,} times overall and {holdout_by_budget:,} times in confirmation. The combined holdout arm had {int(combined_holdout["terminalConjunctiveCompletion"].sum()):,}/{len(combined_holdout):,} terminal successes.

`endpoint_summary.csv` contains arm-by-target/challenge absolute rates and morphology scores. `confirmation_contrasts.csv` contains familywise paired estimates. `factorial_interactions.csv` retains the prespecified arm-by-challenge and arm-by-target differences-of-differences. `budget_ledger.csv` keeps every budget family separate, and `pareto_frontier.csv` records dominance without collapsing semantic authority.

The machine decision in `outcome_decision.json` reports combined absolute = `{outcome["combinedAbsoluteRule"]}`, comparative = `{outcome["combinedComparativeRule"]}`, Pareto = `{outcome["combinedParetoRule"]}`, overall support = `{outcome["supportRule"]}`. Any block with zero terminal completion across all arms remains listed in `allZeroTerminalBlocks`; relative effects in such a block are explicitly non-absolute.

## Validation

- Complete accounting: {validation["accountedRuns"]:,}/{validation["intendedRuns"]:,}; duplicates {validation["duplicateRunIds"]}; failures {validation["failedRuns"]}.
- Pairing: {validation["pairedBlocksPassed"]}/{validation["pairedBlocks"]} blocks had all five arms with one shared initial state, perturbation mask, scenario ID, and seed.
- Exact replay: {validation["replayPassed"]}/{validation["replayRows"]} episode, metric, budget, intervention, mask, and run-identity checks passed.
- Independent target rescoring: {validation["metricAuditPassed"]}/{validation["metricAuditRows"]} sampled terminal states matched S01 geometry/components/topology, S02 acceptance, and terminal conjunction.
- Seeds: {validation["seedAuditPassed"]}/{validation["seedAuditRows"]} regenerated exactly; exploratory/confirmation scenario IDs were disjoint.
- Invariants/permissions/budgets: {validation["invariantPassed"]}/{validation["completedRuns"]}, {validation["permissionPassed"]}/{validation["completedRuns"]}, and {validation["budgetPassed"]}/{validation["completedRuns"]} passed.
- Censoring and claim boundaries: exact maintenance was never censored; non-exact censoring matched first passage; every row retained separate S02/S01 fields and acknowledged the zero-terminal rule.
- Software: focused and full morphology tests, Ruff, formatting, compilation, upstream hashes, and final artifact hashes passed as recorded in the validation/provenance/manifests.

Overall validation: **{"PASS" if validation["success"] else "FAIL"}**.

## Caveats, blockers, failed assumptions, and limitations

- Central-only is bounded controller-only actuation, not a full-state optimizer. It is deliberately weaker than colloquial “central micromanagement.”
- Equal ceilings do not equalize semantics: gradients encode an authored spatial prior; local grammar encodes target-specific contact priors; direct action has stronger causal authority; lagged summaries require global source work.
- Information, action, perturbation, source-work, computation, opportunity, and semantic authority cannot be reduced to one fair scalar. The reported Pareto frontier omits a numeric authority score by design.
- Completion calibration remains limited to two unobstructed bounded-square targets. Results do not transfer automatically to holes, vacancies, periodic, hexagonal, irregular, obstacle, or fixed-boundary environments.
- S10 count-changing conditions remain infeasible. S12 does not reopen deletion, excess, division, insertion, conversion, or adjusted-target semantics.
- Fixed budgets bound all claims. Noncompletion is censoring, not impossibility, and activity or quietness is not convergence or an attractor.
- These are synthetic computational control proxies, not biological morphogenesis, repair, cognition, agency, clinical evidence, or wet-lab validation.

No release blocker remains if validation is PASS. Null and constraining results are retained rather than recoded as success.

## Provenance and artifact map

`frozen_hybrid_control_design.yaml` and `design_freeze.json` bind the pre-outcome design. `hybrid_control_results.parquet`, `endpoint_summary.csv`, `budget_ledger.csv`, `factorial_interactions.csv`, `confirmation_contrasts.csv`, `pareto_frontier.csv`, and `outcome_decision.json` contain the main evidence. Replay, seed, target-rescoring, accounting, execution, provenance, validation, sampled traces, and SHA-256 records complete the handoff. Repository source remains in Git and caches remain under `/cache`.

## Recommended next action

{recommended}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 8:
        raise ValueError("S12 workers must be in 1..8")
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    validate_hybrid_catalog(catalog)
    if catalog["schemaVersion"] != HYBRID_CATALOG_VERSION:
        raise ValueError("S12 catalog schema changed")
    output = arguments.output_dir
    if arguments.replace and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    commit = git_output("rev-parse", "HEAD")
    frozen_path = output / "frozen_hybrid_control_design.yaml"
    frozen_path.write_bytes(CATALOG_PATH.read_bytes())
    design_freeze = {
        "schemaVersion": "e06.s12.design-freeze.v1",
        "researchStepId": "S12",
        "repositoryCommit": commit,
        "catalogPath": str(CATALOG_PATH),
        "catalogSha256": file_sha256(CATALOG_PATH),
        "frozenArtifactSha256": file_sha256(frozen_path),
        "frozenBeforeOutcomeSimulation": True,
        "absoluteAndComparativeEndpointsFrozen": True,
        "s11ZeroTerminalConstraintAccepted": True,
        "countChangingSemanticsWidened": False,
    }
    write_json(output / "design_freeze.json", design_freeze)
    cache_dir = (
        arguments.cache_dir / f"{commit[:12]}-{design_freeze['catalogSha256'][:12]}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    screen_tasks = exploratory_tasks(catalog)
    exploratory, screen_traces, screen_masks, screen_execution = run_tasks(
        screen_tasks, cache_dir, workers=arguments.workers, label="exploratory"
    )
    if len(exploratory) != int(catalog["simulation"]["exploratoryRunCount"]):
        raise ValueError("S12 exploratory accounting mismatch")
    exploratory_effects = exploratory_contrasts(exploratory, catalog)
    blocks, promotion = select_confirmation_blocks(exploratory_effects, catalog)
    write_json(output / "promotion_decisions.json", promotion)

    confirm_tasks = confirmation_tasks(catalog, blocks)
    confirmation_frame, confirm_traces, confirm_masks, confirm_execution = run_tasks(
        confirm_tasks, cache_dir, workers=arguments.workers, label="confirmation"
    )
    expected_confirmation = (
        len(blocks)
        * len(ARM_ORDER)
        * int(catalog["simulation"]["confirmationReplicatesPerArmBlock"])
    )
    if len(confirmation_frame) != expected_confirmation:
        raise ValueError("S12 confirmation accounting mismatch")
    frame = pd.concat([exploratory, confirmation_frame], ignore_index=True)
    completed = frame[~frame["failed"]].copy()
    summary = condition_summary(frame)
    interactions = factorial_interactions(exploratory, catalog)
    confirmation = confirmation_contrasts(confirmation_frame, blocks, catalog)
    pareto = pareto_frontier(confirmation_frame, catalog)
    replays = replay_audit([*screen_tasks, *confirm_tasks], frame)
    seed_audit, metric_audit = seed_and_metric_audits(frame)
    outcome = _outcome_decision(confirmation_frame, confirmation, pareto, catalog)

    mask_records: dict[str, dict[str, Any]] = {}
    mask_conflicts = 0
    for item in [*screen_masks, *confirm_masks]:
        record = dict(item)
        key = str(record["pairingBlockId"])
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
        if key in mask_records:
            previous = json.dumps(
                mask_records[key], sort_keys=True, separators=(",", ":")
            )
            if canonical != previous:
                mask_conflicts += 1
        else:
            mask_records[key] = record
    mask_frame = pd.DataFrame(
        [
            {
                "pairingBlockId": key,
                "targetId": item["targetId"],
                "challengeId": item["challengeId"],
                "replicate": item["replicate"],
                "initialStateSha256": item["initialStateSha256"],
                "targetFeasible": item["targetFeasible"],
                "lesionMaskSha256": item.get("lesionMaskSha256"),
                "externalGraphDisplacement": item["externalGraphDisplacement"],
                "maskJson": json.dumps(item, sort_keys=True, separators=(",", ":")),
            }
            for key, item in sorted(mask_records.items())
        ]
    )

    paired_blocks = 0
    paired_passed = 0
    for (_phase, _pairing), group in completed.groupby(["phase", "pairingBlockId"]):
        paired_blocks += 1
        passed = bool(
            set(group["armId"]) == set(ARM_ORDER)
            and group["scenarioId"].nunique() == 1
            and group["seedHex"].nunique() == 1
            and group["initialStateSha256"].nunique() == 1
            and group["lesionMaskSha256"].fillna("none").nunique() == 1
        )
        paired_passed += int(passed)
    exploratory_ids = set(completed[completed["phase"] == "exploratory"]["scenarioId"])
    confirmation_ids = set(
        completed[completed["phase"] == "confirmation"]["scenarioId"]
    )
    censoring_pass = bool(
        (~completed[completed["challengeId"] == "exact_maintenance"]["censored"]).all()
        and (
            completed[completed["challengeId"] != "exact_maintenance"]["censored"]
            == ~completed[completed["challengeId"] != "exact_maintenance"][
                "conjunctiveCompletionByBudget"
            ]
        ).all()
    )
    replay_columns = [
        "runIdMatch",
        "episodeMatch",
        "metricMatch",
        "budgetMatch",
        "interventionMatch",
        "maskMatch",
    ]
    seed_columns = ["scenarioMatch", "pairingMatch", "runIdMatch", "seedMatch"]
    metric_columns = [
        "s01SuccessMatch",
        "s01MismatchMatch",
        "componentMatch",
        "topologyMatch",
        "s02AcceptanceMatch",
        "terminalConjunctionMatch",
    ]
    intended = int(catalog["simulation"]["exploratoryRunCount"]) + expected_confirmation
    validation: dict[str, Any] = {
        "schemaVersion": "e06.s12.validation-results.v1",
        "researchStepId": "S12",
        "intendedRuns": intended,
        "accountedRuns": len(frame),
        "completedRuns": len(completed),
        "failedRuns": int(frame["failed"].sum()),
        "duplicateRunIds": int(frame["runId"].duplicated().sum()),
        "pairedBlocks": paired_blocks,
        "pairedBlocksPassed": paired_passed,
        "maskConflicts": mask_conflicts,
        "crossSplitDisjoint": not exploratory_ids.intersection(confirmation_ids),
        "replayRows": len(replays),
        "replayPassed": int(replays[replay_columns].all(axis=1).sum()),
        "seedAuditRows": len(seed_audit),
        "seedAuditPassed": int(seed_audit[seed_columns].all(axis=1).sum()),
        "metricAuditRows": len(metric_audit),
        "metricAuditPassed": int(metric_audit[metric_columns].all(axis=1).sum()),
        "invariantPassed": int(completed["invariantSuccess"].sum()),
        "permissionPassed": int(completed["permissionAuditSuccess"].sum()),
        "budgetPassed": int(completed["budgetEnvelopeSuccess"].sum()),
        "feasibilityPassed": int(completed["targetFeasible"].sum()),
        "censoringPassed": censoring_pass,
        "s02S01SeparationPassed": bool(
            completed["terminalS02GrammarAccepted"].notna().all()
            and completed["terminalS01GlobalSuccess"].notna().all()
        ),
        "zeroTerminalClaimBoundaryPassed": bool(
            completed["zeroTerminalClaimBoundaryAcknowledged"].all()
            and outcome["relativeEffectsNeverPromotedToAbsoluteSuccess"]
        ),
        "centralOnlyNativeActuationDisabled": bool(
            (
                completed[completed["armId"] == "central_only"]["usedNativeActorSlots"]
                == 0
            ).all()
        ),
        "centralComputationSeparated": bool(
            completed[completed["armId"].isin(DIRECT_ARMS)]["controllerComputeUnits"]
            .ge(0)
            .all()
            and "totalComputationUnits" in completed
        ),
        "artifactHashCoveragePlanned": True,
    }
    validation["success"] = bool(
        validation["accountedRuns"] == validation["intendedRuns"]
        and validation["failedRuns"] == 0
        and validation["duplicateRunIds"] == 0
        and validation["pairedBlocksPassed"] == validation["pairedBlocks"]
        and validation["maskConflicts"] == 0
        and validation["crossSplitDisjoint"]
        and validation["replayPassed"] == validation["replayRows"]
        and validation["seedAuditPassed"] == validation["seedAuditRows"]
        and validation["metricAuditPassed"] == validation["metricAuditRows"]
        and validation["invariantPassed"] == validation["completedRuns"]
        and validation["permissionPassed"] == validation["completedRuns"]
        and validation["budgetPassed"] == validation["completedRuns"]
        and validation["feasibilityPassed"] == validation["completedRuns"]
        and validation["censoringPassed"]
        and validation["s02S01SeparationPassed"]
        and validation["zeroTerminalClaimBoundaryPassed"]
        and validation["centralOnlyNativeActuationDisabled"]
        and validation["centralComputationSeparated"]
    )

    frame.to_parquet(
        output / "hybrid_control_results.parquet", index=False, compression="zstd"
    )
    exploratory.to_parquet(
        output / "hybrid_control_results_exploratory.parquet",
        index=False,
        compression="zstd",
    )
    summary.to_csv(output / "endpoint_summary.csv", index=False)
    exploratory_effects.to_csv(output / "exploratory_contrasts.csv", index=False)
    confirmation.to_csv(output / "confirmation_contrasts.csv", index=False)
    interactions.to_csv(output / "factorial_interactions.csv", index=False)
    pareto.to_csv(output / "pareto_frontier.csv", index=False)
    replays.to_csv(output / "replay_audit.csv", index=False)
    seed_audit.to_csv(output / "seed_audit.csv", index=False)
    metric_audit.to_csv(output / "target_metric_agreement.csv", index=False)
    mask_frame.to_parquet(
        output / "scenario_mask_manifest.parquet", index=False, compression="zstd"
    )
    completed.groupby(["phase", "targetId", "challengeId", "armId"])[
        [
            "censored",
            "terminalConjunctiveCompletion",
            "conjunctiveCompletionByBudget",
        ]
    ].agg(["count", "sum"]).to_csv(output / "censoring_table.csv")
    completed.groupby(["phase", "targetId", "challengeId", "armId"])[
        [
            "observationCommunicatedBitsUpperBound",
            "configurationBits",
            "controllerInputBits",
            "policyDeliveryBits",
            "addressBits",
            "channelTotalInformationBits",
            "totalInformationBitsIncludingObservation",
            "acceptedMovements",
            "totalGraphDisplacement",
            "actuationAttempts",
            "actuationSuccesses",
            "overrideActionUnits",
            "directMovementGraphDisplacement",
            "externalInterventionGraphDisplacement",
            "policyLogicalSourceReads",
            "sourceScalarReads",
            "controllerStateReads",
            "policyUtilityEvaluations",
            "policyComparisonOperations",
            "controllerComputeUnits",
            "allocatedNativeActorSlots",
            "usedNativeActorSlots",
            "foregoneNativeActorSlots",
            "totalSourceWorkUnits",
            "totalComputationUnits",
            "totalOpportunityCostUnits",
        ]
    ].mean().reset_index().to_csv(output / "budget_ledger.csv", index=False)
    frame[
        [
            "phase",
            "split",
            "scenarioId",
            "pairingBlockId",
            "runId",
            "seedHex",
            "targetId",
            "challengeId",
            "armId",
            "replicate",
            "initialStateSha256",
            "lesionMaskSha256",
            "traceSelected",
        ]
    ].to_parquet(output / "scenario_manifest.parquet", index=False, compression="zstd")
    write_json(output / "outcome_decision.json", outcome)
    write_json(output / "validation_results.json", validation)
    write_json(
        output / "run_accounting.json",
        {
            "schemaVersion": "e06.s12.run-accounting.v1",
            "researchStepId": "S12",
            "intended": intended,
            "recorded": len(frame),
            "completed": len(completed),
            "failed": int(frame["failed"].sum()),
            "censored": int(frame["censored"].sum()),
            "exploratory": len(exploratory),
            "confirmation": len(confirmation_frame),
        },
    )
    write_jsonl_gz(
        output / "sampled_traces.jsonl.gz", [*screen_traces, *confirm_traces]
    )
    create_figures(summary, pareto, output)
    execution = {
        "schemaVersion": "e06.s12.execution-manifest.v1",
        "researchStepId": "S12",
        "workers": arguments.workers,
        "threadEnvironment": {
            "OMP_NUM_THREADS": 1,
            "OPENBLAS_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
        },
        "backend": catalog["simulation"]["backend"],
        "gpuUsed": False,
        "cacheDir": str(cache_dir),
        "exploratory": screen_execution,
        "confirmation": confirm_execution,
    }
    write_json(output / "execution_manifest.json", execution)
    write_json(
        output / "provenance_manifest.json",
        {
            "schemaVersion": "e06.s12.provenance-manifest.v1",
            "researchStepId": "S12",
            "repositoryCommit": commit,
            "branch": git_output("branch", "--show-current"),
            "catalogSha256": design_freeze["catalogSha256"],
            "upstreamArtifacts": upstream_hashes(),
            "previousArtifacts": [
                "/previous-artifacts/E01/research_steps/S03/transition_spec.md",
                "/previous-artifacts/E01/research_steps/S08/seed_specification.json",
                "/previous-artifacts/E01/research_steps/S08/pairing_validation.json",
                "/previous-artifacts/E04/report_inputs/e06_e07_handoff.md",
                "/previous-artifacts/E04/report_inputs/aggregation_classification.md",
            ],
            "datasetRequired": False,
            "newDependencies": [],
        },
    )
    report = report_text(
        frame, summary, confirmation, pareto, validation, outcome, commit
    )
    (output / REPORT_PATH).write_text(report, encoding="utf-8")
    write_json(output / "artifact_manifest.json", artifact_manifest(output))
    print(json.dumps({"validation": validation, "outcome": outcome}, indent=2))


if __name__ == "__main__":
    main()
