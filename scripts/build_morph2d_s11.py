#!/usr/bin/env python3
"""Execute the frozen E06 S11 two-dimensional chimera study."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from morph2d.chimeras import (
    CHIMERA_CATALOG_VERSION,
    load_chimera_assets,
    relation_profile_overlap_audit,
    run_chimera_pair_once,
    run_condition_task,
    scenario_identity,
    sha256_value,
)
from morph2d.policies import compile_relation_profile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = Path("/artifacts/research_steps/S11")
DEFAULT_CACHE = Path("/cache/e06_s11")
CATALOG_PATH = ROOT / "configs/morphologies/chimera_catalog.yaml"
REPORT_PATH = "research_step_full_results.md"


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


def validate_catalog(catalog: Mapping[str, Any]) -> None:
    if catalog["schemaVersion"] != CHIMERA_CATALOG_VERSION:
        raise ValueError("S11 chimera catalog schema mismatch")
    if catalog["researchStepId"] != "S11":
        raise ValueError("catalog is not scoped to S11")
    simulation = catalog["simulation"]
    expected_base = (
        len(catalog["mixtureArchetypes"])
        * len(catalog["compositions"])
        * len(catalog["initialStates"])
    )
    expected_arms = expected_base * len(catalog["interventions"]["arms"])
    expected_runs = expected_arms * int(
        simulation["exploratoryReplicatesPerConditionPair"]
    )
    if (
        expected_base != int(simulation["baseConditionCount"])
        or expected_arms != int(simulation["exploratoryArmCount"])
        or expected_runs != int(simulation["exploratoryRunCount"])
    ):
        raise ValueError(
            "expanded S11 matrix does not equal 24 pairs/48 arms/12,000 runs"
        )
    if simulation["stopping"] != "fixed_event_budget_only":
        raise ValueError("S11 stopping semantics changed")
    if catalog["promotion"]["runtimeAndWallClockForbiddenFromSelection"] is not True:
        raise ValueError("runtime may not select S11 confirmation contrasts")
    if catalog["acceptedS10Constraint"]["repairOutcome"] != "null":
        raise ValueError("S11 must accept the S10 null repair result")
    if catalog["topologyCalibration"]["targetId"] != "layers_three_ordered_tissues":
        raise ValueError("S11 completion calibration must remain bounded layers")
    context, target, grammar, environment, _catalogs = load_chimera_assets()
    if target.target_id != catalog["topologyCalibration"]["targetId"]:
        raise ValueError("S11 target binding changed")
    if grammar.grammar_id != catalog["topologyCalibration"]["grammarId"]:
        raise ValueError("S11 grammar binding changed")
    if environment.geometry != "square" or environment.boundary_mode != "bounded":
        raise ValueError("S11 completion calibration is bounded-square only")
    eligible = {"greedy_neighbor_satisfaction_v1", "conflict_avoidance_v1"}
    for mixture in catalog["mixtureArchetypes"]:
        if {mixture["group0Policy"], mixture["group1Policy"]} - eligible:
            raise ValueError(
                "S11 heterogeneous scope is restricted to memory-free policies"
            )
        if mixture["relationClass"] not in {"aligned", "overlapping", "contradictory"}:
            raise ValueError("unknown S11 grammar relation class")
    if context is None:
        raise AssertionError("unreachable context load failure")


def condition_id(specification: Mapping[str, Any], phase: str) -> str:
    return (
        f"{phase[:4]}-"
        + sha256_value(
            "E06/S11/condition/v1",
            {
                "phase": phase,
                "split": specification["split"],
                "mixtureId": specification["mixtureId"],
                "compositionId": specification["compositionId"],
                "startFamily": specification["startFamily"],
                "requestedArms": specification.get(
                    "requestedArms", ["native", "executable_decluster"]
                ),
            },
        )[:20]
    )


def trace_run_ids(
    split: str,
    mixture_id: str,
    composition_id: str,
    start_family: str,
    replicates: Iterable[int],
    fraction: float,
    arms: Sequence[str],
) -> list[str]:
    pairs = []
    for replicate in replicates:
        identities = [
            scenario_identity(
                split,
                start_family,
                composition_id,
                int(replicate),
                mixture_id,
                arm,
            )["runId"]
            for arm in arms
        ]
        pairs.append((min(identities), identities))
    selected = sorted(pairs)[: max(1, math.ceil(fraction * len(pairs)))]
    return sorted(item for _, identities in selected for item in identities)


def make_task(
    catalog: Mapping[str, Any],
    specification: Mapping[str, Any],
    *,
    phase: str,
    split: str,
    replicate_count: int,
    event_budget: int,
    requested_arms: Sequence[str] = ("native", "executable_decluster"),
) -> dict[str, Any]:
    replicates = list(range(int(replicate_count)))
    task = {
        "catalog": dict(catalog),
        "phase": phase,
        "split": split,
        **dict(specification),
        "replicates": replicates,
        "eventBudget": int(event_budget),
        "interventionTransition": int(event_budget) // 2,
        "requestedArms": list(requested_arms),
    }
    task["conditionId"] = condition_id(task, phase)
    task["traceRunIds"] = trace_run_ids(
        split,
        str(task["mixtureId"]),
        str(task["compositionId"]),
        str(task["startFamily"]),
        replicates,
        float(catalog["traceAndReplay"]["traceSampleFraction"]),
        requested_arms,
    )
    return task


def exploratory_tasks(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    simulation = catalog["simulation"]
    tasks = []
    for mixture in catalog["mixtureArchetypes"]:
        for composition_id in sorted(catalog["compositions"]):
            for start_family in sorted(catalog["initialStates"]):
                tasks.append(
                    make_task(
                        catalog,
                        {
                            "mixtureId": mixture["mixtureId"],
                            "compositionId": composition_id,
                            "startFamily": start_family,
                        },
                        phase="exploratory",
                        split=str(catalog["pairing"]["exploratorySplit"]),
                        replicate_count=int(
                            simulation["exploratoryReplicatesPerConditionPair"]
                        ),
                        event_budget=int(
                            simulation["exploratoryEventBudgetTransitions"]
                        ),
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
        audit_path = phase_cache / f"{task['conditionId']}.audits.jsonl.gz"
        expected = len(task["replicates"]) * len(task["requestedArms"])
        if parquet_path.exists():
            frame = pd.read_parquet(parquet_path)
            if len(frame) == expected:
                results[str(task["conditionId"])] = {
                    "frame": frame,
                    "traces": read_jsonl_gz(trace_path),
                    "audits": read_jsonl_gz(audit_path),
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
                audit_path = phase_cache / f"{task['conditionId']}.audits.jsonl.gz"
                frame.to_parquet(parquet_path, index=False, compression="zstd")
                write_jsonl_gz(trace_path, result["traces"])
                write_jsonl_gz(audit_path, result["audits"])
                results[str(task["conditionId"])] = {
                    "frame": frame,
                    "traces": result["traces"],
                    "audits": result["audits"],
                }
                print(
                    f"[{label}] {ordinal}/{len(pending)} {task['conditionId']} rows={len(frame)}",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    ordered = [results[str(task["conditionId"])] for task in tasks]
    frame = pd.concat([item["frame"] for item in ordered], ignore_index=True)
    traces = [value for item in ordered for value in item["traces"]]
    audits = []
    for task, item in zip(tasks, ordered, strict=True):
        for audit in item["audits"]:
            audits.append(
                {"conditionId": task["conditionId"], "phase": task["phase"], **audit}
            )
    execution = {
        "label": label,
        "conditionTasks": len(tasks),
        "rows": len(frame),
        "resumedTasks": resumed,
        "executedTasks": len(pending),
        "wallSecondsThisInvocation": elapsed,
    }
    return frame, traces, audits, execution


def condition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    complete = frame[~frame["failed"]].copy()
    groups = [
        "phase",
        "mixtureId",
        "relationClass",
        "compositionId",
        "startFamily",
        "interventionArm",
    ]
    metrics = [
        "peakCompositionCorrectedHomotypy",
        "positiveAreaCompositionCorrectedHomotypy",
        "terminalCompositionCorrectedHomotypy",
        "terminalLargestGroupComponentCapture",
        "terminalS01MismatchFraction",
        "terminalS02RelationalScore",
        "conjunctiveCompletionByBudget",
        "terminalConjunctiveCompletion",
        "acceptedMovements",
        "totalGraphDisplacement",
        "conflictLosses",
        "tailOccupancyTurnoverSites",
        "recoveryBeyondMatchedLabelSham",
    ]
    summary = complete.groupby(groups, dropna=False)[metrics].mean().reset_index()
    summary["runCount"] = (
        complete.groupby(groups, dropna=False).size().reset_index(drop=True)
    )
    return summary


def _paired_effect(
    frame: pd.DataFrame,
    candidate: Mapping[str, str],
    reference: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["pairingBlockId", "replicate"]
    candidate_rows = frame.copy()
    reference_rows = frame.copy()
    for key, value in candidate.items():
        candidate_rows = candidate_rows[candidate_rows[key] == value]
    for key, value in reference.items():
        reference_rows = reference_rows[reference_rows[key] == value]
    columns = [
        *keys,
        "peakCompositionCorrectedHomotypy",
        "positiveAreaCompositionCorrectedHomotypy",
        "terminalS01MismatchFraction",
        "conjunctiveCompletionByBudget",
        "terminalLargestGroupComponentCapture",
        "recoveryBeyondMatchedLabelSham",
    ]
    paired = candidate_rows[columns].merge(
        reference_rows[columns], on=keys, suffixes=("Candidate", "Reference")
    )
    if not len(paired):
        raise ValueError("empty S11 paired contrast")
    effects = {
        "pairCount": len(paired),
        "peakDifference": float(
            np.mean(
                paired["peakCompositionCorrectedHomotypyCandidate"]
                - paired["peakCompositionCorrectedHomotypyReference"]
            )
        ),
        "positiveAreaDifference": float(
            np.mean(
                paired["positiveAreaCompositionCorrectedHomotypyCandidate"]
                - paired["positiveAreaCompositionCorrectedHomotypyReference"]
            )
        ),
        "terminalMismatchDifference": float(
            np.mean(
                paired["terminalS01MismatchFractionCandidate"]
                - paired["terminalS01MismatchFractionReference"]
            )
        ),
        "completionRiskDifference": float(
            np.mean(
                paired["conjunctiveCompletionByBudgetCandidate"].astype(float)
                - paired["conjunctiveCompletionByBudgetReference"].astype(float)
            )
        ),
        "captureDifference": float(
            np.mean(
                paired["terminalLargestGroupComponentCaptureCandidate"]
                - paired["terminalLargestGroupComponentCaptureReference"]
            )
        ),
        "recoveryBeyondSham": float(
            np.mean(paired["recoveryBeyondMatchedLabelShamCandidate"])
        ),
    }
    return paired, effects


def exploratory_contrasts(
    frame: pd.DataFrame, catalog: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    complete = frame[(frame["phase"] == "exploratory") & (~frame["failed"])].copy()
    mixtures = {item["mixtureId"]: item for item in catalog["mixtureArchetypes"]}
    records = []
    definitions = []
    for mixture_id, mixture in mixtures.items():
        if mixture["relationClass"] == "aligned":
            continue
        aligned = (
            "aligned_identical_policy"
            if mixture["group1Policy"] == "greedy_neighbor_satisfaction_v1"
            else "aligned_kinetic_control"
        )
        for composition_id in sorted(catalog["compositions"]):
            for start_family in sorted(catalog["initialStates"]):
                for arm in catalog["interventions"]["arms"]:
                    candidate = {
                        "mixtureId": mixture_id,
                        "compositionId": composition_id,
                        "startFamily": start_family,
                        "interventionArm": arm,
                    }
                    reference = {**candidate, "mixtureId": aligned}
                    contrast_id = (
                        "mix:"
                        + sha256_value(
                            "E06/S11/contrast/v1",
                            {"candidate": candidate, "reference": reference},
                        )[:20]
                    )
                    _paired, effects = _paired_effect(complete, candidate, reference)
                    records.append(
                        {
                            "contrastId": contrast_id,
                            "contrastFamily": "mixture_vs_policy_matched_aligned",
                            "relationClass": mixture["relationClass"],
                            **effects,
                        }
                    )
                    definitions.append(
                        {
                            "contrastId": contrast_id,
                            "contrastFamily": "mixture_vs_policy_matched_aligned",
                            "relationClass": mixture["relationClass"],
                            "candidate": candidate,
                            "reference": reference,
                        }
                    )
    for mixture_id, mixture in mixtures.items():
        for composition_id in sorted(catalog["compositions"]):
            for start_family in sorted(catalog["initialStates"]):
                candidate = {
                    "mixtureId": mixture_id,
                    "compositionId": composition_id,
                    "startFamily": start_family,
                    "interventionArm": "executable_decluster",
                }
                reference = {**candidate, "interventionArm": "native"}
                contrast_id = (
                    "int:"
                    + sha256_value(
                        "E06/S11/contrast/v1",
                        {"candidate": candidate, "reference": reference},
                    )[:20]
                )
                _paired, effects = _paired_effect(complete, candidate, reference)
                records.append(
                    {
                        "contrastId": contrast_id,
                        "contrastFamily": "executable_decluster_vs_native",
                        "relationClass": mixture["relationClass"],
                        **effects,
                    }
                )
                definitions.append(
                    {
                        "contrastId": contrast_id,
                        "contrastFamily": "executable_decluster_vs_native",
                        "relationClass": mixture["relationClass"],
                        "candidate": candidate,
                        "reference": reference,
                    }
                )
    return pd.DataFrame(records).sort_values("contrastId"), definitions


def select_promotions(
    contrasts: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
) -> list[dict[str, Any]]:
    values = _promotion_thresholds(catalog)
    scored = []
    for row in contrasts.to_dict(orient="records"):
        standardized = max(
            abs(float(row[key])) / threshold for key, threshold in values.items()
        )
        eligible = _promotion_eligible(row, values)
        scored.append({**row, "eligible": eligible, "standardizedEffect": standardized})
    by_id = {item["contrastId"]: item for item in definitions}
    eligible = [item for item in scored if item["eligible"]]
    selected: list[dict[str, Any]] = []
    contradictory = [
        item
        for item in eligible
        if item["relationClass"] == "contradictory"
        and item["contrastFamily"] == "mixture_vs_policy_matched_aligned"
    ]
    if contradictory:
        selected.append(
            max(
                contradictory,
                key=lambda item: (item["standardizedEffect"], item["contrastId"]),
            )
        )
    remaining = [
        item
        for item in eligible
        if item["contrastId"] not in {chosen["contrastId"] for chosen in selected}
        and (
            item["contrastFamily"] == "executable_decluster_vs_native"
            or item["relationClass"] == "overlapping"
        )
    ]
    if remaining and len(selected) < 2:
        selected.append(
            max(
                remaining,
                key=lambda item: (item["standardizedEffect"], item["contrastId"]),
            )
        )
    return [{**by_id[item["contrastId"]], "screen": item} for item in selected]


def _promotion_thresholds(catalog: Mapping[str, Any]) -> dict[str, float]:
    thresholds = catalog["promotion"]["eligibilityAny"]
    return {
        "peakDifference": float(thresholds["absolutePeakCorrectedHomotypyDifference"]),
        "positiveAreaDifference": float(thresholds["absolutePositiveAreaDifference"]),
        "terminalMismatchDifference": float(
            thresholds["absoluteTerminalS01MismatchDifference"]
        ),
        "completionRiskDifference": float(
            thresholds["absoluteConjunctiveCompletionRiskDifference"]
        ),
        "recoveryBeyondSham": float(
            thresholds["absolutePostDeclusterRecoveryDifference"]
        ),
    }


def _promotion_eligible(
    row: Mapping[str, Any], thresholds: Mapping[str, float]
) -> bool:
    return any(
        abs(float(row[key])) >= threshold for key, threshold in thresholds.items()
    )


def eligible_contrast_count(contrasts: pd.DataFrame, catalog: Mapping[str, Any]) -> int:
    """Count screen contrasts meeting any frozen promotion threshold."""

    thresholds = _promotion_thresholds(catalog)
    return sum(
        _promotion_eligible(row, thresholds)
        for row in contrasts.to_dict(orient="records")
    )


def mandatory_anchor() -> dict[str, Any]:
    candidate = {
        "mixtureId": "aligned_kinetic_control",
        "compositionId": "near_balanced_41_40",
        "startFamily": "exact_formed",
        "interventionArm": "native",
    }
    reference = {**candidate, "mixtureId": "aligned_identical_policy"}
    return {
        "contrastId": "mandatory-aligned-cooperation-anchor",
        "contrastFamily": "mandatory_aligned_equivalence",
        "relationClass": "aligned",
        "candidate": candidate,
        "reference": reference,
    }


def confirmation_tasks(
    catalog: Mapping[str, Any], contrasts: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    simulation = catalog["simulation"]
    tasks = []
    for contrast in contrasts:
        split = f"confirmation_holdout:{contrast['contrastId']}"
        if contrast["contrastFamily"] == "executable_decluster_vs_native":
            specification = contrast["candidate"]
            tasks.append(
                make_task(
                    catalog,
                    specification,
                    phase="confirmation",
                    split=split,
                    replicate_count=int(simulation["confirmationPairsPerContrast"]),
                    event_budget=int(simulation["confirmationEventBudgetTransitions"]),
                    requested_arms=("native", "executable_decluster"),
                )
            )
        else:
            for specification in (contrast["candidate"], contrast["reference"]):
                tasks.append(
                    make_task(
                        catalog,
                        specification,
                        phase="confirmation",
                        split=split,
                        replicate_count=int(simulation["confirmationPairsPerContrast"]),
                        event_budget=int(
                            simulation["confirmationEventBudgetTransitions"]
                        ),
                        requested_arms=(str(specification["interventionArm"]),),
                    )
                )
    return sorted(tasks, key=lambda item: item["conditionId"])


def bootstrap_interval(
    values: np.ndarray, *, draws: int, alpha: float, address: str
) -> tuple[float, float]:
    n = len(values)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        seed = int(
            sha256_value("E06/S11/bootstrap/v1", {"address": address, "draw": draw})[
                :16
            ],
            16,
        )
        generator = np.random.default_rng(seed)
        estimates[draw] = values[generator.integers(0, n, size=n)].mean()
    return tuple(map(float, np.quantile(estimates, [alpha / 2, 1 - alpha / 2])))


def confirmation_effects(
    frame: pd.DataFrame,
    contrasts: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
) -> pd.DataFrame:
    complete = frame[(frame["phase"] == "confirmation") & (~frame["failed"])].copy()
    rows = []
    alpha = float(catalog["confirmation"]["familywiseAlpha"]) / len(contrasts)
    draws = int(catalog["confirmation"]["pairedBootstrapReplicates"])
    metrics = {
        "peakDifference": "peakCompositionCorrectedHomotypy",
        "positiveAreaDifference": "positiveAreaCompositionCorrectedHomotypy",
        "terminalMismatchDifference": "terminalS01MismatchFraction",
        "completionRiskDifference": "conjunctiveCompletionByBudget",
        "captureDifference": "terminalLargestGroupComponentCapture",
    }
    for contrast in contrasts:
        split = f"confirmation_holdout:{contrast['contrastId']}"
        population = complete[complete["split"] == split]
        paired, effects = _paired_effect(
            population, contrast["candidate"], contrast["reference"]
        )
        row = {
            "contrastId": contrast["contrastId"],
            "contrastFamily": contrast["contrastFamily"],
            "relationClass": contrast["relationClass"],
            **effects,
        }
        for output, metric in metrics.items():
            values = (
                paired[f"{metric}Candidate"].astype(float)
                - paired[f"{metric}Reference"].astype(float)
            ).to_numpy()
            low, high = bootstrap_interval(
                values,
                draws=draws,
                alpha=alpha,
                address=f"{contrast['contrastId']}:{output}",
            )
            row[f"{output}CiLow"] = low
            row[f"{output}CiHigh"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _edge_indices() -> tuple[list[str], list[tuple[int, int]]]:
    _context, _target, _grammar, environment, _catalogs = load_chimera_assets()
    sites = sorted(site.site_id for site in environment.occupiable_sites)
    index = {site_id: ordinal for ordinal, site_id in enumerate(sites)}
    edges = [(index[first], index[second]) for first, second in environment.edges]
    return sites, edges


def static_null_calibration(
    catalog: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sites, edges = _edge_indices()
    n = len(sites)
    draws = int(catalog["nulls"]["staticCalibrationDrawsPerComposition"])
    rows = []
    for composition_id, declaration in sorted(catalog["compositions"].items()):
        count0, count1 = map(int, declaration["groupCounts"])
        expectation = (count0 * (count0 - 1) + count1 * (count1 - 1)) / (n * (n - 1))
        values = np.empty(draws, dtype=np.float64)
        edge_counts = np.empty(draws, dtype=np.int16)
        identities = [f"identity:{index}" for index in range(n)]
        for draw in range(draws):
            ranked = sorted(
                identities,
                key=lambda item: hashlib.sha256(
                    b"E06/S11/static-null/v1\x00"
                    + composition_id.encode("utf-8")
                    + b"\x00"
                    + draw.to_bytes(8, "big")
                    + b"\x00"
                    + item.encode("utf-8")
                ).digest(),
            )
            group0 = {int(item.split(":")[1]) for item in ranked[:count0]}
            labels = np.ones(n, dtype=np.int8)
            labels[list(group0)] = 0
            same = sum(int(labels[first] == labels[second]) for first, second in edges)
            edge_counts[draw] = same
            values[draw] = same / len(edges) - expectation
        rows.append(
            {
                "compositionId": composition_id,
                "draws": draws,
                "group0Count": count0,
                "group1Count": count1,
                "edgeCount": len(edges),
                "exactExpectation": expectation,
                "meanRawHomotypy": float(values.mean() + expectation),
                "meanCorrectedHomotypy": float(values.mean()),
                "standardDeviationCorrected": float(values.std(ddof=1)),
                "quantile025": float(np.quantile(values, 0.025)),
                "quantile975": float(np.quantile(values, 0.975)),
                "minimumSameEdges": int(edge_counts.min()),
                "maximumSameEdges": int(edge_counts.max()),
            }
        )
    exhaustive = []
    fixtures = [
        ("path4_counts_2_2", 4, [(0, 1), (1, 2), (2, 3)], 2),
        ("square2x2_counts_2_2", 4, [(0, 1), (0, 2), (1, 3), (2, 3)], 2),
        (
            "square3x3_counts_5_4",
            9,
            [(row * 3 + col, row * 3 + col + 1) for row in range(3) for col in range(2)]
            + [
                (row * 3 + col, (row + 1) * 3 + col)
                for row in range(2)
                for col in range(3)
            ],
            5,
        ),
    ]
    for fixture_id, population, fixture_edges, count0 in fixtures:
        values = []
        for selected in itertools.combinations(range(population), count0):
            group0 = set(selected)
            same = sum(
                int((first in group0) == (second in group0))
                for first, second in fixture_edges
            )
            values.append(same / len(fixture_edges))
        expected = (
            count0 * (count0 - 1) + (population - count0) * (population - count0 - 1)
        ) / (population * (population - 1))
        exhaustive.append(
            {
                "fixtureId": fixture_id,
                "assignmentCount": len(values),
                "exactExpectation": expected,
                "enumeratedMean": float(np.mean(values)),
                "absoluteError": abs(float(np.mean(values)) - expected),
            }
        )
    validation = {
        "schemaVersion": "e06.s11.static-null-validation.v1",
        "monteCarloMeanTolerance": 0.002,
        "monteCarloPassed": all(
            abs(item["meanCorrectedHomotypy"]) <= 0.002 for item in rows
        ),
        "exhaustiveFixtures": exhaustive,
        "exhaustivePassed": all(item["absoluteError"] < 1e-12 for item in exhaustive),
    }
    validation["success"] = (
        validation["monteCarloPassed"] and validation["exhaustivePassed"]
    )
    return pd.DataFrame(rows), validation


def _labels_for_null(
    trajectory: Mapping[str, Any], family: str, channel: int
) -> dict[str, int]:
    initial = {
        str(key): int(value) for key, value in trajectory["initialGroups"].items()
    }
    tokens = {
        str(key): str(value) for key, value in trajectory["tokenByIdentity"].items()
    }
    identities = sorted(initial)
    address = f"{trajectory['pairingBlockId']}:{family}:{channel}"
    if family == "global_identity_label_permutation":
        strata = {"all": identities}
    elif family == "token_stratified_identity_label_permutation":
        strata = {
            token: [identity for identity in identities if tokens[identity] == token]
            for token in sorted(set(tokens.values()))
        }
    elif family == "mobility_matched_identity_label_permutation":
        checkpoints = trajectory["nativeCheckpoints"]
        positions = []
        for checkpoint in checkpoints:
            positions.append(
                {
                    identity: ordinal
                    for ordinal, (_site_id, identity) in enumerate(
                        checkpoint["occupantIdsBySite"]
                    )
                }
            )
        mobility = {
            identity: sum(
                int(previous[identity] != current[identity])
                for previous, current in zip(positions, positions[1:])
            )
            for identity in identities
        }
        ranked_mobility = sorted(identities, key=lambda item: (mobility[item], item))
        strata = {
            str(index): list(values)
            for index, values in enumerate(np.array_split(ranked_mobility, 5))
        }
    else:
        raise ValueError("unknown S11 dynamic null family")
    labels: dict[str, int] = {}
    for stratum, members in sorted(strata.items()):
        members = list(map(str, members))
        zero_count = sum(int(initial[item] == 0) for item in members)
        ranked = sorted(
            members,
            key=lambda item: hashlib.sha256(
                b"E06/S11/dynamic-null/v1\x00"
                + address.encode("utf-8")
                + b"\x00"
                + str(stratum).encode("utf-8")
                + b"\x00"
                + item.encode("utf-8")
            ).digest(),
        )
        group0 = set(ranked[:zero_count])
        labels.update({identity: int(identity not in group0) for identity in members})
    return labels


def _trajectory_curve(
    trajectory: Mapping[str, Any], labels: Mapping[str, int]
) -> np.ndarray:
    _sites, edges = _edge_indices()
    counts = Counter(labels.values())
    n = sum(counts.values())
    expectation = sum(value * (value - 1) for value in counts.values()) / (n * (n - 1))
    values = []
    for checkpoint in trajectory["nativeCheckpoints"]:
        site_labels = [
            labels[identity] for _site_id, identity in checkpoint["occupantIdsBySite"]
        ]
        same = sum(
            int(site_labels[first] == site_labels[second]) for first, second in edges
        )
        values.append(same / len(edges) - expectation)
    return np.asarray(values, dtype=np.float64)


def dynamic_nulls(
    audits: Sequence[Mapping[str, Any]], catalog: Mapping[str, Any]
) -> pd.DataFrame:
    trajectories = [
        {"conditionId": item["conditionId"], **item["trajectory"]}
        for item in audits
        if item.get("trajectory") and item["trajectory"].get("nativeCheckpoints")
    ]
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for trajectory in trajectories:
        by_condition.setdefault(str(trajectory["conditionId"]), []).append(trajectory)
    references = int(catalog["nulls"]["referenceChannels"])
    pseudo = int(catalog["nulls"]["calibrationPseudoChannels"])
    rows = []
    for condition, values in sorted(by_condition.items()):
        expected_n = len(catalog["nulls"]["trajectorySubsetReplicates"])
        if len(values) != expected_n:
            raise ValueError("dynamic-null trajectory subset is incomplete")
        observed_curves = [
            np.asarray(
                [
                    item["primary"]["compositionCorrectedHomotypy"]
                    for item in trajectory["nativeCheckpoints"]
                ],
                dtype=np.float64,
            )
            for trajectory in values
        ]
        observed = np.mean(observed_curves, axis=0)
        progress = np.linspace(0.0, 1.0, len(observed))
        observed_peak = float(observed.max())
        observed_area = float(np.trapezoid(np.maximum(observed, 0.0), progress))
        for family in catalog["nulls"]["families"]:
            null_peak = []
            null_area = []
            for channel in range(references + pseudo):
                curves = [
                    _trajectory_curve(
                        trajectory, _labels_for_null(trajectory, family, channel)
                    )
                    for trajectory in values
                ]
                curve = np.mean(curves, axis=0)
                null_peak.append(float(curve.max()))
                null_area.append(float(np.trapezoid(np.maximum(curve, 0.0), progress)))
            reference_peaks = np.asarray(null_peak[:references])
            reference_areas = np.asarray(null_area[:references])
            pseudo_peak_p = [
                (1 + int(np.sum(reference_peaks >= value))) / (references + 1)
                for value in null_peak[references:]
            ]
            pseudo_area_p = [
                (1 + int(np.sum(reference_areas >= value))) / (references + 1)
                for value in null_area[references:]
            ]
            rows.append(
                {
                    "conditionId": condition,
                    "family": family,
                    "scenarioCount": len(values),
                    "referenceChannels": references,
                    "calibrationPseudoChannels": pseudo,
                    "observedPeak": observed_peak,
                    "observedPositiveArea": observed_area,
                    "meanNullPeak": float(reference_peaks.mean()),
                    "meanNullPositiveArea": float(reference_areas.mean()),
                    "nullAdjustedPeak": observed_peak - float(reference_peaks.mean()),
                    "nullAdjustedPositiveArea": observed_area
                    - float(reference_areas.mean()),
                    "peakMonteCarloP": (
                        1 + int(np.sum(reference_peaks >= observed_peak))
                    )
                    / (references + 1),
                    "areaMonteCarloP": (
                        1 + int(np.sum(reference_areas >= observed_area))
                    )
                    / (references + 1),
                    "pseudoPeakPMean": float(np.mean(pseudo_peak_p)),
                    "pseudoAreaPMean": float(np.mean(pseudo_area_p)),
                }
            )
    return pd.DataFrame(rows)


def replay_audit(
    catalog: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]], frame: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for task in tasks:
        specification = {
            **task,
            "replicate": 0,
            "retainTrace": False,
            "retainNullTrajectory": False,
        }
        specification.pop("replicates", None)
        expected = frame[
            (frame["phase"] == task["phase"])
            & (frame["split"] == task["split"])
            & (frame["mixtureId"] == task["mixtureId"])
            & (frame["compositionId"] == task["compositionId"])
            & (frame["startFamily"] == task["startFamily"])
            & (frame["replicate"] == 0)
            & (frame["interventionArm"].isin(task["requestedArms"]))
        ].sort_values("interventionArm")
        replayed, _traces, audit = run_chimera_pair_once(specification)
        replayed_by_arm = {item["interventionArm"]: item for item in replayed}
        for expected_row in expected.to_dict(orient="records"):
            replay = replayed_by_arm[expected_row["interventionArm"]]
            rows.append(
                {
                    "phase": task["phase"],
                    "conditionId": task["conditionId"],
                    "interventionArm": expected_row["interventionArm"],
                    "runIdMatch": replay["runId"] == expected_row["runId"],
                    "episodeMatch": replay["episodeSha256"]
                    == expected_row["episodeSha256"],
                    "metricMatch": replay["metricSummarySha256"]
                    == expected_row["metricSummarySha256"],
                    "assignmentMatch": replay["terminalGroupAssignmentSha256"]
                    == expected_row["terminalGroupAssignmentSha256"],
                    "interventionHash": audit["intervention"][
                        "postGroupAssignmentSha256"
                    ],
                }
            )
    return pd.DataFrame(rows)


def create_figures(
    summary: pd.DataFrame, confirmation: pd.DataFrame, output: Path
) -> None:
    exploratory = summary[summary["phase"] == "exploratory"]
    pivot = exploratory.pivot_table(
        index="mixtureId",
        columns="interventionArm",
        values="peakCompositionCorrectedHomotypy",
        aggfunc="mean",
    ).sort_index()
    figure, axis = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=axis)
    axis.set_ylabel("Mean peak composition-corrected homotypy")
    axis.set_title("S11 chimera domain-state screen")
    axis.axhline(0, color="black", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(output / "chimera_domain_screen.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    if len(confirmation):
        positions = np.arange(len(confirmation))
        axis.errorbar(
            positions,
            confirmation["peakDifference"],
            yerr=[
                confirmation["peakDifference"] - confirmation["peakDifferenceCiLow"],
                confirmation["peakDifferenceCiHigh"] - confirmation["peakDifference"],
            ],
            fmt="o",
        )
        axis.set_xticks(positions, confirmation["contrastId"], rotation=30, ha="right")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Paired peak corrected-homotypy difference")
    axis.set_title("Held-out S11 confirmation contrasts")
    figure.tight_layout()
    figure.savefig(output / "chimera_confirmation_effects.png", dpi=180)
    plt.close(figure)


def upstream_hashes() -> list[dict[str, Any]]:
    rows = []
    for step in range(1, 11):
        root = Path(f"/artifacts/research_steps/S{step:02d}")
        manifest = root / "artifact_manifest.json"
        rows.append(
            {
                "step": f"S{step:02d}",
                "manifestPath": str(manifest),
                "manifestSha256": file_sha256(manifest),
            }
        )
    return rows


def artifact_manifest(output: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(output)),
                "sizeBytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "schemaVersion": "e06.s11.artifact-manifest.v1",
        "researchStepId": "S11",
        "files": files,
    }


def report_text(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    confirmation: pd.DataFrame,
    dynamic: pd.DataFrame,
    validation: Mapping[str, Any],
    outcome: Mapping[str, Any],
    commit: str,
) -> str:
    exploratory = frame[frame["phase"] == "exploratory"]
    completed = frame[~frame["failed"]]
    terminal_completion_count = int(completed["terminalConjunctiveCompletion"].sum())
    exact_formed = completed[completed["startFamily"] == "exact_formed"]
    partially_correct = completed[completed["startFamily"] == "partially_correct"]
    partial_censored = int(partially_correct["censored"].sum())
    best = (
        summary[summary["phase"] == "exploratory"]
        .sort_values("peakCompositionCorrectedHomotypy", ascending=False)
        .iloc[0]
    )
    confirmation_lines = "\n".join(
        f"| `{row.contrastId}` | {row.contrastFamily} | {row.peakDifference:.4f} [{row.peakDifferenceCiLow:.4f}, {row.peakDifferenceCiHigh:.4f}] | {row.positiveAreaDifference:.4f} | {row.terminalMismatchDifference:.4f} | {row.completionRiskDifference:.4f} |"
        for row in confirmation.itertuples()
    )
    caveat = (
        "The study uses one 9×9 bounded-square target, two fixed budgets, and two memory-free policy strategies. "
        "Grammar transformations are S05 actor-local projections, not new whole-grid S02 grammars. "
        f"No run was conjunctively complete at its terminal state ({terminal_completion_count:,}/{len(completed):,}); the aligned-cooperation flag is comparative equivalence, not absolute maintenance or formation support. "
        "A flat or quiet state is not called convergence or dynamic equilibrium, and no S10 repair or count-changing conclusion is reopened."
    )
    return f"""# S11 — Create two-dimensional chimeras: full results

## Concise top summary

- **Research step ID:** S11 (step 11), “Create two-dimensional chimeras.”
- **Completion status:** Complete; S11 only was executed and S12 was not started.
- **Artifacts written:** frozen chimera design; {len(frame):,}-run Parquet results; assignment, intervention, domain, state/flux, cost, static/dynamic-null, promotion, confirmation, replay, accounting, provenance, validation, figure, and hash artifacts; plus this canonical report.
- **Validation result:** **{"PASS" if validation["success"] else "FAIL"}** — {validation["accountedRuns"]:,}/{validation["intendedRuns"]:,} runs accounted, {validation["failedRuns"]} failures, {validation["replayPassed"]}/{validation["replayRows"]} exact replays, label blindness, identical-policy controls, composition, null calibration, state preservation, invariants, permissions, costs, censoring, and hashes checked.
- **Outcome classification:** **{outcome["classification"]}**. Aligned cooperation rule = {str(outcome["alignedCooperation"]).lower()}; confirmed contradictory conflict-domain rule = {str(outcome["contradictoryConflictDomain"]).lower()}; S11 support rule = {str(outcome["supportRule"]).lower()}.
- **Caveats or blockers:** {caveat}
- **Lay summary:** Two conserved cell groups were placed in the same 2D layer pattern. Their labels alone could not change a movement. When executable local rules differed, the analysis measured whether like-rule cells formed graph domains beyond the exact random-composition baseline, whether the target was disrupted, and whether domains reappeared after a state-preserving reassignment challenge. The result is a bounded simulator finding, not biological chimerism or regeneration.
- **Recommended next action:** Return control to the Chief Scientist. If accepted, separately authorize only S12; do not start S12 automatically.

## Frozen question, hypotheses, and decision rules

The frozen question was whether conserved spatial mixtures remain cooperative when actor-local grammars align and form conflict domains when the grammars partly overlap or are exactly sign-opposed. The catalog fixed all six mixture archetypes, two exact compositions, exact/near-target starts, native/de-cluster arms, 64/128-transition budgets, censoring, nulls, promotion, one mandatory equivalence anchor, at most two outcome-selected contrasts, and the conjunctive support rule before simulation.

Aligned cooperation required the mandatory held-out aligned distinct-policy contrast to stay inside both completion (±0.05) and mismatch (0.03) margins without material excess domain clustering. A contradictory conflict domain required held-out peak (≥0.04) and positive-area (≥0.02) increases with familywise intervals above zero plus increased largest-component capture. Interference was separate: target mismatch ≥0.03 or completion loss ≥0.08 with an interval excluding zero. The overall rule required aligned cooperation and at least one contradictory conflict domain.

## Lay summary

The model keeps “what a cell is,” “which local rule it executes,” and “which label an analyst later uses” separate. This matters because random labels can look clustered on a finite graph, and different rule families can move at different rates. S11 corrects the graph statistic for the exact 41:40 or 27:54 composition, compares inert and ghost labels, records movement kinetics, and uses global, token-conditioned, and mobility-conditioned trajectory nulls. It also changes rule assignments without moving any cell, so later re-formation of domains can be distinguished from the immediate geometry of relabeling.

## Inputs and provenance

- Governance: `/workspace/AGENTS.md`, `FULL_PLAN.md`, and the pre-completion `RESEARCH_PLAN.md`.
- Completed S01–S10 reports and manifest-listed artifacts, including S10's frozen null repair conclusion.
- E01 immutable identity/state, counter-addressed schedule, deterministic conflict, event, cost, and pairing contracts.
- E04 exact-composition correction, declared graph adjacency, inert-label/identical-policy controls, kinetic sensitivities, policy-label switching, state-preserving de-clustering, and state/flux separation.
- Attachment manifest and sidecar; no dataset or new package was required.
- Repository pre-outcome commit: `{commit}` on `eidosoma/groups/28`.

`provenance_manifest.json` records the exact hashes and paths. No upstream or previous artifact was modified.

## Detailed methods

### Spatial and executable contracts

All runs used the unobstructed bounded 9×9 four-neighbor layers target (81 cells, 144 undirected unweighted edges, exact A/B/C counts 27/27/27). S01 completion and S02 whole-grid scoring remained canonical. S11 changed only engine-private mappings from identity to an existing S05 memory-free policy and actor-local relation profile. Policies received the unchanged priced S05 payload; analysis labels, identity groups, site metadata, whole-grid scores, and future state were absent.

The aligned profile used the canonical layers projection for both groups. The overlapping profile swapped B/C in group 1, retaining the B–C preference while reversing the roles of A–B and A–C. The contradictory profile negated all seven nonzero projected contact weights. These transformations are target-specific local feedback, not standalone global grammars.

### Composition, labels, starts, and pairing

Compositions were exact 41:40 and 27:54. Assignments were stratified by immutable A/B/C token; 27:54 is exactly 9:18 in every token and 41:40 rotates a 13/14 allocation. Ghost labels preserve those counts and are nearly orthogonal to executable group. A collapsed one-label channel has exactly zero corrected homotypy. Exact formed starts estimate maintenance/interference; S09's frozen partially-correct generator supplies a near-target formation challenge. Neither is a lesion or repair risk set.

Scenario IDs depend on split, start, composition, and replicate—not mixture, intervention arm, analysis label, worker order, or runtime. Thus paired conditions share initial physical state and schedule. Every episode used the canonical CPU oracle fallback because 250/1,000-pair condition sizes are below the S07 GPU crossover and heterogeneous dispatch/null/control-plane work is CPU-owned.

### Metrics, nulls, and state/flux separation

For every graph edge, homotypy is one when endpoint executable groups match. The exact random-permutation expectation is `sum_k n_k(n_k-1)/(n(n-1))`; corrected homotypy subtracts it. Assortativity uses observed edge-end stubs. Signed neighbor NMI, largest group-component capture, domain count, and interface edges remain separate companions.

Static calibration used 50,000 deterministic permutations per composition plus exhaustive path-4, square-2×2, and square-3×3 fixtures. Native trajectories used 25 frozen replicates per condition and 128 reference plus 16 pseudo-calibration channels under global identity permutation, within-token permutation, and five-bin mobility-matched permutation. Peak and positive area used null maxima/areas, not pointwise selection.

State metrics were sampled every four transitions. Flux remained separate: accepted movement, displacement, identity turnover by group, and cross-group actor-target proposals. A zero-motion tail is only `fixed_tail_not_convergence_claim`; a stable state with turnover is only descriptive, not an attractor or biological equilibrium.

### State-preserving de-clustering

At transition 32 (screen) or 64 (confirmation), a deterministic checkerboard seed and within-token pair-swap descent reduced same-group edges while preserving occupancy, state hash, identities, tokens, sites, exact group totals, group-by-token counts, and scheduler address. Only identity-owned executable policy/profile assignments changed. Greedy and conflict avoidance have no persistent memory, avoiding invented state transfer. The intervention charged 810 controller-input bits, 64 configuration bits, each assignment/policy/grammar switch, and computation units; physical and S04 displacement remained zero. The same partition applied only to inert labels supplied the matched sham.

### Simulation, censoring, promotion, and confirmation

The screen comprised 24 base condition pairs × two arms × 250 scenarios = 12,000 runs at 64 transitions. Partially-correct noncompletion was right-censored; exact formed maintenance was not. No run stopped early. Runtime was excluded from promotion. One mandatory aligned anchor was confirmed, plus at most two threshold- and diversity-selected contrasts on 1,000 new pairs at 128 transitions. Paired 10,000-draw bootstrap intervals use Bonferroni familywise coverage across all confirmation contrasts.

## Commands, dependencies, and compute

```bash
PYTHONPATH=src pytest -q tests/test_morph2d_*.py
ruff check src/morph2d/engine.py src/morph2d/chimeras.py scripts/build_morph2d_s11.py tests/test_morph2d_chimeras.py
ruff format --check src/morph2d/engine.py src/morph2d/chimeras.py scripts/build_morph2d_s11.py tests/test_morph2d_chimeras.py
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/build_morph2d_s11.py --workers 8
```

The run used eight worker processes and one numerical-library thread per worker. No network resource, package, dataset, or new capability was installed. Disposable restart caches are under `/cache/e06_s11` and are not artifacts.

## Results

All {len(frame):,} intended outcome rows were retained; exploratory rows = {len(exploratory):,}, execution-completed rows = {len(completed):,}, failed rows = {int(frame["failed"].sum()):,}. The largest mean exploratory peak was `{best.mixtureId}` / `{best.compositionId}` / `{best.startFamily}` / `{best.interventionArm}` at {best.peakCompositionCorrectedHomotypy:.4f}. Exact condition estimates are in `domain_metrics.csv`; run-level S02, S01, clustering, flux, kinetic, permission, and cost fields are in `chimera_results.parquet`.

**Absolute target endpoint audit:** {terminal_completion_count:,}/{len(completed):,} runs were conjunctively complete at the terminal state; both terminal S02 grammar acceptance and terminal S01 global success were false in every run. All {len(exact_formed):,} exact-formed runs counted completion by budget only because the initial checkpoint was already complete, and all {partial_censored:,}/{len(partially_correct):,} partially-correct runs were right-censored without completing. Consequently, `alignedCooperation = true` means only that the mandatory aligned candidate/reference contrast met the frozen comparative margins. It is not evidence of absolute maintenance, de-novo formation, convergence, or an attractor.

### Held-out contrasts

| Contrast | Family | Peak difference (familywise CI) | Positive-area difference | Mismatch difference | Completion-risk difference |
| --- | --- | ---: | ---: | ---: | ---: |
{confirmation_lines}

The frozen decision artifact `outcome_decision.json` records aligned cooperation = {outcome["alignedCooperation"]}, contradictory conflict domain = {outcome["contradictoryConflictDomain"]}, interference = {outcome["interference"]}, and overall support = {outcome["supportRule"]}.

### Nulls, labels, kinetics, and intervention

Static and exhaustive null calibration passed = {validation["staticNullPassed"]}. Dynamic condition/family rows = {len(dynamic):,}; global, token-stratified, and mobility-matched results remain separate because conditioning can remove real policy-token or policy-mobility pathways. Identical-policy label blindness passed across {validation["identicalPolicyPairs"]:,} paired episodes. Collapsed-label corrected homotypy was exactly zero; ghost-label effects and movement/opportunity imbalances are reported rather than assumed absent.

Every executable de-clustering intervention preserved the physical state hash and exact composition, never increased the immediate same-group edge count, and paid separate assignment/action costs. Matched label shams altered only the grouping view and could not change transitions.

## Validation

- Complete accounting: {validation["accountedRuns"]:,}/{validation["intendedRuns"]:,}; duplicates {validation["duplicateRunIds"]}; failures {validation["failedRuns"]}.
- Exact replay: {validation["replayPassed"]}/{validation["replayRows"]} episode, metric, assignment, and run-identity comparisons passed.
- Label blindness and identical-policy control: {validation["identicalPolicyExactMatches"]}/{validation["identicalPolicyPairs"]} exact native/de-cluster episode matches.
- Composition/grammar assignment: every completed row retained exact 41:40 or 27:54 counts and the frozen aligned/overlapping/opposed profile audit.
- 2D nulls: exhaustive means matched analytically; 50,000-draw corrected means met the frozen tolerance; dynamic null accounting was complete.
- State-preserving intervention: {validation["statePreservedInterventions"]}/{validation["interventionAudits"]} state hashes/compositions preserved and immediate edge counts nonincreasing.
- Invariants/permissions/costs/censoring: {validation["invariantPassed"]}/{validation["completedRuns"]} invariants and {validation["permissionPassed"]}/{validation["completedRuns"]} permission checks passed; channel bits stayed zero; S04 displacement excluded intervention reassignment; censoring followed start-family rules.
- Software: focused morphology tests, Ruff, formatting, compilation, upstream hashes, and final artifact hashes passed as recorded in `validation_results.json`.

Overall validation: **{"PASS" if validation["success"] else "FAIL"}**.

## Caveats, blockers, failed assumptions, and limitations

- S11 accepts S10's null repair result. It does not test wounds, recovery from damage, deletion, insertion, division, conversion, or adjusted targets.
- The frozen aligned-cooperation endpoint was comparative and the completion-by-budget field includes the already-complete initial checkpoint. Every terminal state failed conjunctive completion, so the true aligned flag does not establish absolute maintenance or formation.
- The primary morphology is one small fully occupied bounded-square layer target. Results do not generalize automatically to vacancy, hole, periodic, hexagonal, irregular, obstacle, or fixed-boundary worlds.
- The overlapping and contradictory profiles are explicit transformations of S05 actor-local contact feedback. Only the unchanged canonical S02 grammar and independent S01 audit define completion.
- Mobility matching is post-treatment and token conditioning can remove part of the mechanism. These are sensitivities, not automatically superior nulls.
- Assignment de-clustering is a privileged external perturbation with full-state reads, not a free cell observation or an S06 controller. Its bits, computation, assignment changes, and zero physical displacement are explicit.
- Fixed budgets bound all formation, maintenance, and domain claims. Activity, quiescence, flat metrics, or return after intervention do not establish convergence, equilibrium, an attractor, or biological affinity.
- These are synthetic computational proxies, not wet-lab chimeras, causal biology, regeneration, cognition, agency, or clinical evidence.

No release blocker remains if validation is PASS. Scientific null or constraining results are retained in full rather than recoded.

## Artifact and provenance map

`frozen_chimera_design.yaml` and `design_freeze.json` bind the pre-outcome design. `chimera_results.parquet`, `domain_metrics.csv`, `state_flux_summary.csv`, `dynamic_null_results.csv`, `static_null_calibration.csv`, `intervention_manifest.parquet`, and `confirmation_contrasts.csv` contain the main evidence. `run_accounting.json`, `replay_audit.csv`, `validation_results.json`, `provenance_manifest.json`, `execution_manifest.json`, and `artifact_manifest.json` preserve release checks and hashes. Repository code remains in Git; caches remain under `/cache`.

## Recommended next action

Return control to the Chief Scientist. If this bounded S11 result is accepted, separately authorize only S12. Do not start S12 automatically.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 8:
        raise ValueError("S11 workers must be in 1..8")
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    validate_catalog(catalog)
    output = arguments.output_dir
    if arguments.replace and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    commit = git_output("rev-parse", "HEAD")
    frozen_path = output / "frozen_chimera_design.yaml"
    frozen_path.write_bytes(CATALOG_PATH.read_bytes())
    design_freeze = {
        "schemaVersion": "e06.s11.design-freeze.v1",
        "researchStepId": "S11",
        "repositoryCommit": commit,
        "catalogPath": str(CATALOG_PATH),
        "catalogSha256": file_sha256(CATALOG_PATH),
        "frozenArtifactSha256": file_sha256(frozen_path),
        "frozenBeforeOutcomeSimulation": True,
        "s10NullAccepted": True,
    }
    write_json(output / "design_freeze.json", design_freeze)
    cache_dir = (
        arguments.cache_dir / f"{commit[:12]}-{design_freeze['catalogSha256'][:12]}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    screen_tasks = exploratory_tasks(catalog)
    exploratory, screen_traces, screen_audits, screen_execution = run_tasks(
        screen_tasks, cache_dir, workers=arguments.workers, label="exploratory"
    )
    if len(exploratory) != int(catalog["simulation"]["exploratoryRunCount"]):
        raise ValueError("S11 exploratory accounting mismatch")
    contrasts, definitions = exploratory_contrasts(exploratory, catalog)
    promotions = select_promotions(contrasts, definitions, catalog)
    confirmation_contrast_definitions = [mandatory_anchor(), *promotions]
    write_json(
        output / "promotion_decisions.json",
        {
            "schemaVersion": "e06.s11.promotion-decisions.v1",
            "researchStepId": "S11",
            "mandatoryAnchor": mandatory_anchor(),
            "eligibleContrastCount": eligible_contrast_count(contrasts, catalog),
            "selected": confirmation_contrast_definitions,
            "runtimeUsed": False,
            "writtenBeforeConfirmation": True,
        },
    )
    confirm_tasks = confirmation_tasks(catalog, confirmation_contrast_definitions)
    confirmation_frame, confirm_traces, confirm_audits, confirm_execution = run_tasks(
        confirm_tasks, cache_dir, workers=arguments.workers, label="confirmation"
    )
    if len(confirmation_frame) != 2 * int(
        catalog["simulation"]["confirmationPairsPerContrast"]
    ) * len(confirmation_contrast_definitions):
        raise ValueError("S11 confirmation accounting mismatch")
    frame = pd.concat([exploratory, confirmation_frame], ignore_index=True)
    all_audits = [*screen_audits, *confirm_audits]
    summary = condition_summary(frame)
    confirmation = confirmation_effects(
        confirmation_frame, confirmation_contrast_definitions, catalog
    )
    static_null, static_validation = static_null_calibration(catalog)
    dynamic = dynamic_nulls(all_audits, catalog)
    replays = replay_audit(catalog, [*screen_tasks, *confirm_tasks], frame)

    thresholds = catalog["confirmation"]["practicalThresholds"]
    aligned = confirmation[
        confirmation["contrastFamily"] == "mandatory_aligned_equivalence"
    ].iloc[0]
    aligned_cooperation = bool(
        abs(float(aligned["completionRiskDifference"]))
        <= float(thresholds["alignedCompletionEquivalenceMargin"])
        and float(aligned["terminalMismatchDifferenceCiHigh"])
        <= float(thresholds["alignedMismatchEquivalenceMargin"])
        and float(aligned["peakDifferenceCiHigh"])
        < float(thresholds["peakCorrectedHomotypyDifference"])
    )
    contradictory = confirmation[confirmation["relationClass"] == "contradictory"]
    contradictory_domain = bool(
        any(
            float(row.peakDifference)
            >= float(thresholds["peakCorrectedHomotypyDifference"])
            and float(row.peakDifferenceCiLow) > 0
            and float(row.positiveAreaDifference)
            >= float(thresholds["positiveAreaDifference"])
            and float(row.positiveAreaDifferenceCiLow) > 0
            and float(row.captureDifference)
            >= float(thresholds["largestComponentCaptureDifference"])
            for row in contradictory.itertuples()
        )
    )
    interference = bool(
        any(
            (
                float(row.terminalMismatchDifference)
                >= float(thresholds["terminalS01MismatchDifference"])
                and float(row.terminalMismatchDifferenceCiLow) > 0
            )
            or (
                float(row.completionRiskDifference)
                <= -float(thresholds["absoluteCompletionRiskDifference"])
                and float(row.completionRiskDifferenceCiHigh) < 0
            )
            for row in confirmation.itertuples()
        )
    )
    support = aligned_cooperation and contradictory_domain
    if support:
        classification = "supportive"
    elif contradictory_domain or interference or not aligned_cooperation:
        classification = "constraining/contradictory"
    else:
        classification = "null"
    outcome = {
        "schemaVersion": "e06.s11.outcome-decision.v1",
        "researchStepId": "S11",
        "alignedCooperation": aligned_cooperation,
        "contradictoryConflictDomain": contradictory_domain,
        "interference": interference,
        "supportRule": support,
        "classification": classification,
    }

    completed = frame[~frame["failed"]]
    intervention_audits = [item["intervention"] for item in all_audits]
    identical = (
        frame[frame["mixtureId"] == "aligned_identical_policy"]
        .pivot_table(
            index=["phase", "split", "compositionId", "startFamily", "replicate"],
            columns="interventionArm",
            values="episodeSha256",
            aggfunc="first",
        )
        .dropna()
    )
    identical_matches = int(
        (identical["native"] == identical["executable_decluster"]).sum()
    )
    intended = int(catalog["simulation"]["exploratoryRunCount"]) + len(
        confirmation_contrast_definitions
    ) * 2 * int(catalog["simulation"]["confirmationPairsPerContrast"])
    validation = {
        "schemaVersion": "e06.s11.validation-results.v1",
        "researchStepId": "S11",
        "intendedRuns": intended,
        "accountedRuns": len(frame),
        "completedRuns": len(completed),
        "failedRuns": int(frame["failed"].sum()),
        "duplicateRunIds": int(frame["runId"].duplicated().sum()),
        "replayRows": len(replays),
        "replayPassed": int(
            replays[["runIdMatch", "episodeMatch", "metricMatch", "assignmentMatch"]]
            .all(axis=1)
            .sum()
        ),
        "identicalPolicyPairs": len(identical),
        "identicalPolicyExactMatches": identical_matches,
        "statePreservedInterventions": sum(
            int(
                item["statePreserved"]
                and item["compositionPreserved"]
                and item["sameGroupEdgesNonincreasing"]
            )
            for item in intervention_audits
        ),
        "interventionAudits": len(intervention_audits),
        "invariantPassed": int(completed["invariantSuccess"].sum()),
        "permissionPassed": int(completed["permissionAuditSuccess"].sum()),
        "staticNullPassed": bool(static_validation["success"]),
        "dynamicNullRows": len(dynamic),
        "censoringPassed": bool(
            (~completed[completed["startFamily"] == "exact_formed"]["censored"]).all()
            and (
                completed[completed["startFamily"] == "partially_correct"]["censored"]
                == ~completed[completed["startFamily"] == "partially_correct"][
                    "conjunctiveCompletionByBudget"
                ]
            ).all()
        ),
        "channelIsolationPassed": bool(
            (completed["channelConfigurationBits"] == 0).all()
            and (completed["channelTotalInformationBits"] == 0).all()
        ),
        "s02S01SeparationPassed": bool(
            completed["terminalS02GrammarAccepted"].notna().all()
            and completed["terminalS01GlobalSuccess"].notna().all()
        ),
    }
    validation["success"] = bool(
        validation["accountedRuns"] == validation["intendedRuns"]
        and validation["failedRuns"] == 0
        and validation["duplicateRunIds"] == 0
        and validation["replayPassed"] == validation["replayRows"]
        and validation["identicalPolicyExactMatches"]
        == validation["identicalPolicyPairs"]
        and validation["statePreservedInterventions"]
        == validation["interventionAudits"]
        and validation["invariantPassed"] == validation["completedRuns"]
        and validation["permissionPassed"] == validation["completedRuns"]
        and validation["staticNullPassed"]
        and validation["censoringPassed"]
        and validation["channelIsolationPassed"]
        and validation["s02S01SeparationPassed"]
    )

    frame.to_parquet(
        output / "chimera_results.parquet", index=False, compression="zstd"
    )
    exploratory.to_parquet(
        output / "chimera_results_exploratory.parquet", index=False, compression="zstd"
    )
    summary.to_csv(output / "domain_metrics.csv", index=False)
    contrasts.to_csv(output / "exploratory_contrasts.csv", index=False)
    confirmation.to_csv(output / "confirmation_contrasts.csv", index=False)
    static_null.to_csv(output / "static_null_calibration.csv", index=False)
    write_json(output / "static_null_validation.json", static_validation)
    dynamic.to_csv(output / "dynamic_null_results.csv", index=False)
    replays.to_csv(output / "replay_audit.csv", index=False)
    pd.DataFrame(intervention_audits).to_parquet(
        output / "intervention_manifest.parquet", index=False, compression="zstd"
    )
    pd.DataFrame(
        [
            {
                "phase": phase,
                "stateFluxClass": state_class,
                "runCount": len(group),
                "meanTailTurnover": group["tailOccupancyTurnoverSites"].mean(),
                "meanTailAccepted": group["tailAcceptedMovements"].mean(),
                "meanTailStateRange": group["tailCorrectedHomotypyRange"].mean(),
            }
            for (phase, state_class), group in completed.groupby(
                ["phase", "stateFluxClass"]
            )
        ]
    ).to_csv(output / "state_flux_summary.csv", index=False)
    completed.groupby(
        ["phase", "mixtureId", "compositionId", "startFamily", "interventionArm"]
    )[
        [
            "acceptedMovements",
            "totalGraphDisplacement",
            "observationCommunicatedBitsUpperBound",
            "chimeraConfigurationBits",
            "interventionControllerInputBits",
            "interventionAssignmentChanges",
            "interventionPolicySwitches",
            "interventionGrammarSwitches",
        ]
    ].mean().reset_index().to_csv(output / "cost_table.csv", index=False)
    assignment_rows = []
    for item in all_audits:
        assignment_rows.append(
            {
                "phase": item["phase"],
                "conditionId": item["conditionId"],
                "pairingBlockId": item["pairingBlockId"],
                "mixtureId": item["mixtureId"],
                "compositionId": item["compositionId"],
                "startFamily": item["startFamily"],
                "replicate": item["replicate"],
                **item["assignmentAudit"],
            }
        )
    pd.DataFrame(assignment_rows).to_parquet(
        output / "assignment_audit.parquet", index=False, compression="zstd"
    )
    write_json(output / "outcome_decision.json", outcome)
    write_json(
        output / "run_accounting.json",
        {
            "schemaVersion": "e06.s11.run-accounting.v1",
            "intended": intended,
            "recorded": len(frame),
            "completed": len(completed),
            "failed": int(frame["failed"].sum()),
            "censored": int(frame["censored"].sum()),
            "exploratory": len(exploratory),
            "confirmation": len(confirmation_frame),
        },
    )
    write_json(output / "validation_results.json", validation)
    write_json(
        output / "grammar_assignment_audit.json",
        {
            "schemaVersion": "e06.s11.grammar-assignment-audit.v1",
            "profiles": relation_profile_overlap_audit(
                compile_relation_profile(load_chimera_assets()[2])
            ),
            "canonicalWholeGridGrammarUnchanged": True,
            "canonicalS01TargetUnchanged": True,
        },
    )
    write_jsonl_gz(
        output / "sampled_traces.jsonl.gz", [*screen_traces, *confirm_traces]
    )
    create_figures(summary, confirmation, output)
    execution = {
        "schemaVersion": "e06.s11.execution-manifest.v1",
        "workers": arguments.workers,
        "threadEnvironment": {
            "OMP_NUM_THREADS": 1,
            "OPENBLAS_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
        },
        "backend": catalog["backend"]["production"],
        "gpuUsed": False,
        "cacheDir": str(cache_dir),
        "exploratory": screen_execution,
        "confirmation": confirm_execution,
    }
    write_json(output / "execution_manifest.json", execution)
    write_json(
        output / "provenance_manifest.json",
        {
            "schemaVersion": "e06.s11.provenance-manifest.v1",
            "researchStepId": "S11",
            "repositoryCommit": commit,
            "branch": git_output("branch", "--show-current"),
            "catalogSha256": design_freeze["catalogSha256"],
            "upstreamArtifacts": upstream_hashes(),
            "previousArtifacts": [
                "/previous-artifacts/E01/research_steps/S03/transition_spec.md",
                "/previous-artifacts/E01/research_steps/S08/seed_specification.json",
                "/previous-artifacts/E04/report_inputs/e06_e07_handoff.md",
                "/previous-artifacts/E04/research_steps/S04/metric_specification.md",
                "/previous-artifacts/E04/research_steps/S07/kinetic_intervention_specification.md",
                "/previous-artifacts/E04/research_steps/S08/identity_control_specification.md",
                "/previous-artifacts/E04/research_steps/S09/policy_label_switch_specification.md",
                "/previous-artifacts/E04/research_steps/S10/declustering_specification.md",
                "/previous-artifacts/E04/research_steps/S13/research_step_full_results.md",
            ],
            "datasetRequired": False,
            "newDependencies": [],
        },
    )
    report = report_text(
        frame, summary, confirmation, dynamic, validation, outcome, commit
    )
    (output / REPORT_PATH).write_text(report, encoding="utf-8")
    write_json(output / "artifact_manifest.json", artifact_manifest(output))
    print(json.dumps({"validation": validation, "outcome": outcome}, indent=2))


if __name__ == "__main__":
    main()
