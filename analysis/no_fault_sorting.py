"""S09 no-fault replication runner and analysis helpers.

This module deliberately keeps three evidence layers separate:

* paper conditions and the paper Sortedness metric;
* the nondeterministic, frozen-public-commit cell-view comparator; and
* the deterministic clean-room reference semantics.

The S08 split gate is enforced before scenarios are materialized.  Full event
payloads are not generated for the large summary population; exact replay uses
the ordinary digest-emitting reference path on a preregistered sample.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from reference_simulator.engine import (
    evaluate_terminal,
    execute_batch,
    initial_state,
    run as reference_run,
)
from reference_simulator.model import LEDGER_FIELDS, Scenario, canonical_json_bytes, state_hash
from scenario_bank.core import ConditionSpec, derive_seed, materialize_scenario
from historical_backend.legacy_adapter import verify_source_tree


REPOSITORY = Path(__file__).resolve().parents[1]
S08_DIR = Path("/artifacts/research_steps/S08")
S02_MANIFEST = Path("/artifacts/research_steps/S02/source_tree_checksums.sha256")
HISTORICAL_SOURCE = Path("/cache/e01_s02/historical-worktree")
HISTORICAL_PYTHON = Path("/cache/e01_s04/venv311/bin/python")
PREREGISTRATION = REPOSITORY / "analysis" / "s09_confirmatory_preregistration.json"
ALLOWED_CONDITIONS = {
    "C-UNQ-PURE-CV-BUB-ASC",
    "C-UNQ-PURE-CV-INS-ASC",
    "C-UNQ-PURE-CV-SEL-ASC",
    "C-UNQ-PURE-TRAD-BUB-ASC",
    "C-UNQ-PURE-TRAD-INS-ASC",
    "C-UNQ-PURE-TRAD-SEL-ASC",
}
ALLOWED_SPLITS = {"paper_scale": 100, "confirmatory_holdout": 1000}
FORBIDDEN_SPLITS = {"exploratory", "policy_search_holdout"}
GRID = tuple(index / 200 for index in range(201))
S09_SCHEMA = "e01.s09.no_fault_run.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def condition_from_dict(value: Mapping[str, Any]) -> ConditionSpec:
    return ConditionSpec(
        condition_id=str(value["conditionId"]),
        family=str(value["family"]),
        input_profile=str(value["inputProfile"]),
        architecture=str(value["architecture"]),
        policies=tuple(value["policies"]),
        assignment_profile=str(value["assignmentProfile"]),
        direction_profile=str(value["directionProfile"]),
        direction_map=tuple(sorted(dict(value["directionMap"]).items())),
        fault_mode=str(value["faultMode"]),
        requested_fault_count=int(value["requestedFaultCount"]),
        placement_profile=str(value["placementProfile"]),
        analysis_label_profile=str(value["analysisLabelProfile"]),
        profile_role=str(value["profileRole"]),
        max_activations=int(value["maxActivations"]),
        historical_eligibility=str(value["historicalEligibility"]),
        paper_condition_note=str(value["paperConditionNote"]),
    )


def sortedness_percent(values: Sequence[int | float]) -> float:
    """Paper Figure 3 strict-rise Sortedness, converted to percent."""
    if not values:
        raise ValueError("Sortedness requires at least one value")
    rises = 1 + sum(right > left for left, right in zip(values, values[1:]))
    return 100.0 * rises / len(values)


def normalized_curve(points: Sequence[Sequence[float]]) -> list[float]:
    """Linearly interpolate swap-index points onto the frozen 201-point grid."""
    if not points:
        raise ValueError("trajectory has no points")
    final_index = float(points[-1][0])
    if final_index == 0:
        return [float(points[-1][1])] * len(GRID)
    x = np.asarray([float(item[0]) / final_index for item in points], dtype=np.float64)
    y = np.asarray([float(item[1]) for item in points], dtype=np.float64)
    return np.interp(np.asarray(GRID), x, y).tolist()


def _values(scenario: Scenario, occupancy: Sequence[str]) -> list[int | float]:
    cells = scenario.cell_map
    return [cells[cell_id].value for cell_id in occupancy]


def run_reference_summary(scenario: Scenario, *, retain_raw_curve: bool) -> dict[str, Any]:
    """Execute R transitions without per-activation trace serialization."""
    scenario.validate()
    state = initial_state(scenario)
    state.terminal = evaluate_terminal(scenario, state)
    initial_values = _values(scenario, state.occupancy)
    points: list[list[float]] = [[0, sortedness_percent(initial_values)]]
    start = time.perf_counter()
    while state.terminal is None:
        before = state.ledger["acceptedSwaps"]
        execute_batch(
            scenario, state, retain_events=False, emit_event_records=False
        )
        after = state.ledger["acceptedSwaps"]
        if after > before:
            if after != before + 1:
                raise AssertionError("serial S09 scenario accepted multiple swaps in one batch")
            points.append([after, sortedness_percent(_values(scenario, state.occupancy))])
    elapsed = time.perf_counter() - start
    final_values = _values(scenario, state.occupancy)
    if points[-1][0] != state.ledger["acceptedSwaps"]:
        points.append([state.ledger["acceptedSwaps"], sortedness_percent(final_values)])
    result: dict[str, Any] = {
        "backend_profile": "R-clean-room-reference-E01-v1",
        "publication_snapshot_claimed": False,
        "scenario_id": scenario.scenario_id,
        "architecture": scenario.architecture.value,
        "policy": (
            scenario.traditional_policy.value
            if scenario.traditional_policy is not None
            else scenario.cells[0].policy.value
        ),
        "scheduler": scenario.scheduler,
        "sequence_basis": "activation",
        "historical_random_stream_status": "not_applicable_reference_counter_addressed",
        "stop_reason": state.terminal,
        "completed": state.terminal == "complete",
        "timed_out": False,
        "activation_count": state.activation_count,
        "successful_swap_count": state.ledger["acceptedSwaps"],
        "initial_sortedness_percent": points[0][1],
        "final_sortedness_percent": sortedness_percent(final_values),
        "final_nonstrictly_sorted": all(a <= b for a, b in zip(final_values, final_values[1:])),
        "value_multiset_conserved": sorted(initial_values) == sorted(final_values),
        "initial_values_sha256": json_hash(initial_values),
        "final_values_sha256": json_hash(final_values),
        "final_state_hash": state_hash(scenario.scenario_id, state),
        "observable_final_hash": json_hash(final_values),
        "trace_sha256": None,
        "elapsed_seconds": elapsed,
        "adapter_safety_timeout_seconds": None,
        "thread_shutdown_clean": None,
        "curve_grid": normalized_curve(points),
        "raw_curve": points if retain_raw_curve else None,
    }
    for key in LEDGER_FIELDS:
        result[f"ledger_{key}"] = state.ledger[key]
    return result


def _reference_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    condition = condition_from_dict(task["condition"])
    scenario, metadata = materialize_scenario(condition, task["base"])
    if scenario.scenario_id != task["scenario_row"]["scenarioId"]:
        raise ValueError("S08 normalized scenario failed exact materialization")
    result = run_reference_summary(
        scenario, retain_raw_curve=bool(task["retain_raw_curve"])
    )
    result.update(_task_identity(task, metadata))
    return result


def _task_identity(task: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    row = task["scenario_row"]
    return {
        "schema_version": S09_SCHEMA,
        "research_step_id": "S09",
        "condition_id": row["conditionId"],
        "base_draw_id": row["baseDrawId"],
        "pairing_block_id": row["pairingBlockId"],
        "split": row["split"],
        "protected": bool(row["protected"]),
        "replicate_ordinal": int(row["replicateOrdinal"]),
        "runtime_seed": row["runtimeSeed"],
        "generation_key": row["generationKey"],
        "event_budget": int(row["maxActivations"]),
        "scenario_json_sha256": (
            metadata["scenarioJsonSha256"] if metadata else row["scenarioJsonSha256"]
        ),
    }


def _selected_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(str(row["conditionId"]), []).append(str(row["scenarioId"]))
    return {min(values) for values in groups.values()}


def _verify_preregistration(split: str, supplied_hash: str | None) -> str:
    actual = sha256_file(PREREGISTRATION)
    if split == "confirmatory_holdout" and supplied_hash != actual:
        raise PermissionError(
            "confirmatory holdout requires the exact S09 preregistration SHA-256"
        )
    return actual


def load_tasks(split: str, *, preregistration_sha256: str | None = None) -> list[dict[str, Any]]:
    """Read only the declared S08 logical rows and enforce split/condition gates."""
    if split in FORBIDDEN_SPLITS or split not in ALLOWED_SPLITS:
        raise PermissionError(f"S09 may not access split {split!r}")
    _verify_preregistration(split, preregistration_sha256)
    scenario_table = pq.read_table(
        S08_DIR / "paired_scenario_bank.parquet",
        filters=[("conditionFamily", "=", "unique_pure"), ("split", "=", split)],
    )
    rows = scenario_table.to_pylist()
    expected = 6 * ALLOWED_SPLITS[split]
    if len(rows) != expected:
        raise ValueError(f"expected {expected} S09 scenario rows, found {len(rows)}")
    if {row["conditionId"] for row in rows} != ALLOWED_CONDITIONS:
        raise ValueError("S09 condition set differs from the six frozen no-fault conditions")
    for row in rows:
        if (
            row["inputProfile"] != "unique_1_100"
            or row["faultMode"] != "none"
            or int(row["requestedFaultCount"]) != 0
            or int(row["realizedFaultCount"]) != 0
            or bool(row["protected"]) != (split == "confirmatory_holdout")
        ):
            raise ValueError("S09 row violates the frozen no-fault population")

    base_rows = pq.read_table(
        S08_DIR / "base_draw_bank.parquet",
        filters=[("inputProfile", "=", "unique_1_100"), ("split", "=", split)],
    ).to_pylist()
    bases = {row["baseDrawId"]: row for row in base_rows}
    if len(bases) != ALLOWED_SPLITS[split]:
        raise ValueError("base-draw count does not match frozen split size")

    condition_rows = pq.read_table(
        S08_DIR / "condition_catalog.parquet",
        filters=[("family", "=", "unique_pure")],
    ).to_pylist()
    conditions = {
        row["conditionId"]: json.loads(row["conditionJson"])
        for row in condition_rows
        if row["conditionId"] in ALLOWED_CONDITIONS
    }
    if set(conditions) != ALLOWED_CONDITIONS:
        raise ValueError("condition catalog is missing an S09 condition")

    selected = _selected_ids(rows)
    tasks = []
    for row in sorted(rows, key=lambda item: (item["conditionId"], item["replicateOrdinal"])):
        tasks.append(
            {
                "scenario_row": row,
                "condition": conditions[row["conditionId"]],
                "base": bases[row["baseDrawId"]],
                "retain_raw_curve": split == "paper_scale" or row["scenarioId"] in selected,
                "trace_selection_reason": (
                    "all_paper_scale_figure_trajectory"
                    if split == "paper_scale"
                    else (
                        "preregistered_lexicographic_scenario_id"
                        if row["scenarioId"] in selected
                        else None
                    )
                ),
            }
        )
    return tasks


def _read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                results[row["scenario_id"]] = row
    return results


def run_reference_population(
    tasks: Sequence[Mapping[str, Any]], checkpoint: Path, *, workers: int = 8
) -> list[dict[str, Any]]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1, 8]")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_checkpoint(checkpoint)
    pending = [task for task in tasks if task["scenario_row"]["scenarioId"] not in completed]
    with checkpoint.open("a", encoding="utf-8") as handle:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_reference_worker, task): task for task in pending}
            for future in as_completed(futures):
                result = future.result()
                result["trace_selection_reason"] = futures[future]["trace_selection_reason"]
                handle.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                completed[result["scenario_id"]] = result
                if len(completed) % 100 == 0 or len(completed) == len(tasks):
                    print(
                        f"reference progress {len(completed)}/{len(tasks)}",
                        file=sys.stderr,
                        flush=True,
                    )
    expected = {task["scenario_row"]["scenarioId"] for task in tasks}
    if set(completed) != expected:
        raise ValueError("reference checkpoint run accounting mismatch")
    return [completed[key] for key in sorted(completed)]


def _initial_values(task: Mapping[str, Any]) -> list[int]:
    base = task["base"]
    return [int(base["valuesById"][index]) for index in base["initialOccupancyIndices"]]


def _historical_subprocess(task: Mapping[str, Any], timeout_seconds: float) -> dict[str, Any]:
    payload = {
        "source_root": str(HISTORICAL_SOURCE),
        "checksum_manifest": str(S02_MANIFEST),
        "values": _initial_values(task),
        "policy": task["condition"]["policies"][0].lower(),
        "seed": int(task["scenario_row"]["runtimeSeed"]),
        "timeout_seconds": timeout_seconds,
        "retain_raw_curve": bool(task["retain_raw_curve"]),
    }
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [str(HISTORICAL_PYTHON), str(REPOSITORY / "scripts" / "s09_historical_worker.py")]
    process = subprocess.run(
        command,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout_seconds + 30,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"historical worker failed ({process.returncode}): {process.stderr[-4000:]}"
        )
    result = json.loads(process.stdout)
    result.update(_task_identity(task))
    result["scenario_id"] = task["scenario_row"]["scenarioId"]
    result["architecture"] = "cell_view"
    result["policy"] = task["condition"]["policies"][0]
    result["trace_selection_reason"] = task["trace_selection_reason"]
    return result


def run_historical_population(
    tasks: Sequence[Mapping[str, Any]],
    checkpoint: Path,
    *,
    workers: int = 2,
    timeout_seconds: float = 120.0,
) -> list[dict[str, Any]]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1, 8]")
    historical = [task for task in tasks if task["condition"]["architecture"] == "cell_view"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_checkpoint(checkpoint)
    pending = [task for task in historical if task["scenario_row"]["scenarioId"] not in completed]
    with checkpoint.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_historical_subprocess, task, timeout_seconds): task
                for task in pending
            }
            for future in as_completed(futures):
                result = future.result()
                handle.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                completed[result["scenario_id"]] = result
                if len(completed) % 100 == 0 or len(completed) == len(historical):
                    print(
                        f"historical progress {len(completed)}/{len(historical)}",
                        file=sys.stderr,
                        flush=True,
                    )
    expected = {
        task["scenario_row"]["scenarioId"]
        for task in historical
    }
    if set(completed) != expected:
        raise ValueError("historical checkpoint run accounting mismatch")
    return [completed[key] for key in sorted(completed)]


def _replay_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    condition = condition_from_dict(task["condition"])
    scenario, _ = materialize_scenario(condition, task["base"])
    first = reference_run(scenario, trace_mode="digest")
    second = reference_run(scenario, trace_mode="digest")
    fast = run_reference_summary(scenario, retain_raw_curve=False)
    byte_exact = first.to_json_bytes() == second.to_json_bytes()
    fast_equal = (
        fast["final_state_hash"] == first.final_state_hash
        and fast["activation_count"] == first.summary["activationCount"]
        and all(
            fast[f"ledger_{key}"] == first.summary["ledger"][key]
            for key in LEDGER_FIELDS
        )
    )
    return {
        "scenarioId": scenario.scenario_id,
        "conditionId": task["scenario_row"]["conditionId"],
        "split": task["scenario_row"]["split"],
        "byteExactDigestReplay": byte_exact,
        "summaryPathMatchesDigestPath": fast_equal,
        "eventDigest": first.event_digest,
        "finalStateHash": first.final_state_hash,
        "activationCount": first.summary["activationCount"],
        "successfulSwapCount": first.summary["ledger"]["acceptedSwaps"],
    }


def run_exact_replay_samples(
    paper_tasks: Sequence[Mapping[str, Any]],
    confirmatory_tasks: Sequence[Mapping[str, Any]],
    *,
    workers: int = 8,
) -> list[dict[str, Any]]:
    selected = [
        task
        for task in [*paper_tasks, *confirmatory_tasks]
        if task["trace_selection_reason"] == "preregistered_lexicographic_scenario_id"
        or (
            task["scenario_row"]["split"] == "paper_scale"
            and task["scenario_row"]["scenarioId"]
            in _selected_ids([item["scenario_row"] for item in paper_tasks])
        )
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_replay_worker, selected))
    return sorted(results, key=lambda row: (row["split"], row["conditionId"]))


def run_historical_variability(
    paper_tasks: Sequence[Mapping[str, Any]], *, repeats: int = 5
) -> list[dict[str, Any]]:
    selected_ids = _selected_ids(
        [task["scenario_row"] for task in paper_tasks if task["condition"]["architecture"] == "cell_view"]
    )
    selected = [
        task for task in paper_tasks
        if task["scenario_row"]["scenarioId"] in selected_ids
    ]
    results = []
    for task in selected:
        for repeat in range(repeats):
            row = _historical_subprocess(task, 120.0)
            results.append(
                {
                    "conditionId": row["condition_id"],
                    "scenarioId": row["scenario_id"],
                    "repeat": repeat,
                    "stopReason": row["stop_reason"],
                    "successfulSwapCount": row["successful_swap_count"],
                    "comparisonCount": row["ledger_valueComparisons"],
                    "traceSha256": row["trace_sha256"],
                    "elapsedSeconds": row["elapsed_seconds"],
                    "completed": row["completed"],
                }
            )
    return results


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def _bootstrap_mean(values: np.ndarray, seed: int, draws: int = 10_000) -> dict[str, float]:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("bootstrap input must be a nonempty vector")
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 250):
        count = min(250, draws - start)
        indices = generator.integers(0, len(values), size=(count, len(values)))
        samples[start : start + count] = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "ci9875_low": float(np.quantile(samples, 0.00625)),
        "ci9875_high": float(np.quantile(samples, 0.99375)),
        "bootstrap_draws": draws,
    }


def _summary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"curve_grid", "raw_curve"}}


def _write_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(
        table, path, compression="zstd", compression_level=9,
        use_dictionary=True, write_statistics=True, version="2.6",
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _trajectory_rows(results: Sequence[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    for result in results:
        for index, value in enumerate(result["curve_grid"]):
            yield {
                "backend_profile": result["backend_profile"],
                "scenario_id": result["scenario_id"],
                "condition_id": result["condition_id"],
                "pairing_block_id": result["pairing_block_id"],
                "split": result["split"],
                "architecture": result["architecture"],
                "policy": result["policy"],
                "grid_index": index,
                "normalized_swap_progress": GRID[index],
                "sortedness_percent": value,
            }


def _raw_trajectory_rows(results: Sequence[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    for result in results:
        if result.get("raw_curve") is None:
            continue
        for index, value in result["raw_curve"]:
            yield {
                "backend_profile": result["backend_profile"],
                "scenario_id": result["scenario_id"],
                "condition_id": result["condition_id"],
                "split": result["split"],
                "architecture": result["architecture"],
                "policy": result["policy"],
                "successful_swap_index": int(index),
                "sortedness_percent": float(value),
                "sequence_basis": (
                    "accepted_swap_derived_from_activation"
                    if result["backend_profile"].startswith("R-")
                    else "historical_recorded_swap"
                ),
            }


def _run_accounting(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame([_summary_row(row) for row in results])
    rows = []
    for keys, group in frame.groupby(
        ["backend_profile", "split", "architecture", "policy"], dropna=False
    ):
        backend, split, architecture, policy = keys
        successes = int(group["completed"].sum())
        low, high = _wilson(successes, len(group))
        rows.append(
            {
                "backendProfile": backend,
                "split": split,
                "architecture": architecture,
                "policy": policy,
                "intended": int(len(group)),
                "launched": int(len(group)),
                "recorded": int(len(group)),
                "completed": successes,
                "failed": int((~group["completed"]).sum()),
                "timedOut": int(group["timed_out"].sum()),
                "excluded": 0,
                "completionFraction": successes / len(group),
                "completionWilson95Low": low,
                "completionWilson95High": high,
                "stopReasons": {
                    str(key): int(value)
                    for key, value in group["stop_reason"].value_counts().items()
                },
            }
        )
    return rows


def _paired_contrasts(reference: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = {}
    for row in reference:
        records[(row["split"], row["policy"], row["architecture"], row["pairing_block_id"])] = row
    output = []
    paired_arrays: dict[str, Any] = {}
    for split in ALLOWED_SPLITS:
        for policy in ("Bubble", "Insertion", "Selection"):
            pairs = []
            blocks = sorted({
                row["pairing_block_id"] for row in reference
                if row["split"] == split and row["policy"] == policy
            })
            for block in blocks:
                cv = records[(split, policy, "cell_view", block)]
                traditional = records[(split, policy, "traditional", block)]
                gap = float(np.trapezoid(
                    np.abs(np.asarray(cv["curve_grid"]) - np.asarray(traditional["curve_grid"])),
                    np.asarray(GRID),
                ))
                pairs.append(
                    {
                        "trajectory_iad_pp": gap,
                        "swap_difference": cv["successful_swap_count"] - traditional["successful_swap_count"],
                        "swap_ratio": cv["successful_swap_count"] / max(traditional["successful_swap_count"], 1),
                        "completion_difference": int(cv["completed"]) - int(traditional["completed"]),
                    }
                )
            paired_arrays[f"{split}/{policy}"] = pairs
            for metric in pairs[0]:
                values = np.asarray([item[metric] for item in pairs], dtype=np.float64)
                seed = derive_seed("S09_paired_bootstrap", split, policy, metric)
                interval = _bootstrap_mean(values, seed)
                output.append(
                    {
                        "contrast_family": "reference_cell_view_minus_traditional",
                        "split": split,
                        "policy": policy,
                        "metric": metric,
                        "pair_count": len(pairs),
                        **interval,
                    }
                )
    return output, paired_arrays


def _historical_contrasts(
    reference: Sequence[Mapping[str, Any]], historical: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    reference_by_id = {
        row["scenario_id"]: row for row in reference if row["architecture"] == "cell_view"
    }
    output = []
    for split in ALLOWED_SPLITS:
        for policy in ("Bubble", "Insertion", "Selection"):
            pairs = []
            for hist in historical:
                if hist["split"] != split or hist["policy"] != policy:
                    continue
                ref = reference_by_id[hist["scenario_id"]]
                pairs.append(
                    {
                        "trajectory_iad_pp": float(np.trapezoid(
                            np.abs(np.asarray(hist["curve_grid"]) - np.asarray(ref["curve_grid"])),
                            np.asarray(GRID),
                        )),
                        "swap_difference": hist["successful_swap_count"] - ref["successful_swap_count"],
                        "completion_difference": int(hist["completed"]) - int(ref["completed"]),
                    }
                )
            for metric in pairs[0]:
                values = np.asarray([item[metric] for item in pairs], dtype=np.float64)
                seed = derive_seed("S09_historical_bootstrap", split, policy, metric)
                output.append(
                    {
                        "contrast_family": "historical_C_minus_reference_R_cell_view",
                        "split": split,
                        "policy": policy,
                        "metric": metric,
                        "pair_count": len(pairs),
                        **_bootstrap_mean(values, seed),
                    }
                )
    return output


def _trajectory_envelopes(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted({
        (row["backend_profile"], row["split"], row["architecture"], row["policy"])
        for row in results
    })
    for backend, split, architecture, policy in keys:
        matrix = np.asarray([
            row["curve_grid"] for row in results
            if (row["backend_profile"], row["split"], row["architecture"], row["policy"])
            == (backend, split, architecture, policy)
        ], dtype=np.float64)
        mean = matrix.mean(axis=0)
        se = matrix.std(axis=0, ddof=1) / math.sqrt(len(matrix))
        low, high = np.quantile(matrix, [0.025, 0.975], axis=0)
        for index in range(len(GRID)):
            output.append(
                {
                    "backend_profile": backend,
                    "split": split,
                    "architecture": architecture,
                    "policy": policy,
                    "n": len(matrix),
                    "grid_index": index,
                    "normalized_swap_progress": GRID[index],
                    "mean_sortedness_percent": mean[index],
                    "mean_ci95_low": mean[index] - 1.959963984540054 * se[index],
                    "mean_ci95_high": mean[index] + 1.959963984540054 * se[index],
                    "run_envelope_2_5": low[index],
                    "run_envelope_97_5": high[index],
                }
            )
    return output


def _claim_classifications(
    reference: Sequence[Mapping[str, Any]], contrasts: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    confirm = [row for row in reference if row["split"] == "confirmatory_holdout"]
    all_endpoints = all(
        row["completed"]
        and row["final_sortedness_percent"] == 100.0
        and row["final_nonstrictly_sorted"]
        and row["value_multiset_conserved"]
        for row in confirm
    )
    lookup = {
        (row["policy"], row["metric"]): row
        for row in contrasts
        if row["split"] == "confirmatory_holdout"
        and row["contrast_family"] == "reference_cell_view_minus_traditional"
    }
    bubble = lookup[("Bubble", "trajectory_iad_pp")]
    insertion = lookup[("Insertion", "trajectory_iad_pp")]
    selection_swaps = lookup[("Selection", "swap_difference")]
    return [
        {
            "claim_id": "F03-A-TRAJECTORY",
            "endpoint_classification": "supportive" if all_endpoints else "constraining/contradictory",
            "qualitative_classification": (
                "supportive" if bubble["ci9875_low"] > 5.0 else "null"
            ),
            "decision_rule": "all endpoints plus adjusted lower trajectory-IAD CI > 5 pp",
            "estimate": bubble["estimate"],
            "adjusted_ci_low": bubble["ci9875_low"],
            "adjusted_ci_high": bubble["ci9875_high"],
        },
        {
            "claim_id": "F03-B-TRAJECTORY",
            "endpoint_classification": "supportive" if all_endpoints else "constraining/contradictory",
            "qualitative_classification": (
                "supportive" if insertion["ci9875_high"] < 5.0 else "constraining/contradictory"
            ),
            "decision_rule": "all endpoints plus adjusted upper trajectory-IAD CI < 5 pp",
            "estimate": insertion["estimate"],
            "adjusted_ci_low": insertion["ci9875_low"],
            "adjusted_ci_high": insertion["ci9875_high"],
        },
        {
            "claim_id": "F03-C-SWAP-BURDEN",
            "endpoint_classification": None,
            "qualitative_classification": (
                "supportive" if selection_swaps["ci9875_low"] > 0 else "constraining/contradictory"
            ),
            "decision_rule": "adjusted lower paired swap-difference CI > 0",
            "estimate": selection_swaps["estimate"],
            "adjusted_ci_low": selection_swaps["ci9875_low"],
            "adjusted_ci_high": selection_swaps["ci9875_high"],
        },
        {
            "claim_id": "F03-C-TRAJECTORY",
            "endpoint_classification": "supportive" if all_endpoints else "constraining/contradictory",
            "qualitative_classification": (
                "supportive" if selection_swaps["ci9875_low"] > 0 else "constraining/contradictory"
            ),
            "decision_rule": "all endpoints plus adjusted lower paired swap-difference CI > 0",
            "estimate": selection_swaps["estimate"],
            "adjusted_ci_low": selection_swaps["ci9875_low"],
            "adjusted_ci_high": selection_swaps["ci9875_high"],
        },
    ]


def _plot_figures(
    results: Sequence[Mapping[str, Any]], envelopes: Sequence[Mapping[str, Any]], output: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference_paper = [
        row for row in results
        if row["backend_profile"].startswith("R-") and row["split"] == "paper_scale"
    ]
    policies = ("Bubble", "Insertion", "Selection")
    architectures = ("traditional", "cell_view")
    figure, axes = plt.subplots(3, 2, figsize=(12, 13), constrained_layout=True)
    for row_index, policy in enumerate(policies):
        for column_index, architecture in enumerate(architectures):
            axis = axes[row_index, column_index]
            subset = [
                row for row in reference_paper
                if row["policy"] == policy and row["architecture"] == architecture
            ]
            for row in subset:
                curve = np.asarray(row["raw_curve"], dtype=float)
                axis.plot(curve[:, 0], curve[:, 1], color="#2b6cb0", alpha=0.10, linewidth=0.65)
            axis.axhline(100, color="black", linewidth=0.7, linestyle=":")
            axis.set_title(f"{policy} — {'Traditional R' if architecture == 'traditional' else 'Cell-view R'}")
            axis.set_xlabel("Successful swaps")
            axis.set_ylabel("Sortedness (%)")
            axis.set_ylim(40, 101)
            axis.grid(alpha=0.15)
    figure.suptitle("Figure 3 reconstruction: 100 S08 paper-scale trajectories per condition", fontsize=14)
    figure.savefig(output / "figure3_reconstruction.png", dpi=180)
    figure.savefig(output / "figure3_reconstruction.svg")
    plt.close(figure)

    envelope_frame = pd.DataFrame(envelopes)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for axis, policy in zip(axes, policies):
        for backend, color, label in (
            ("R-clean-room-reference-E01-v1", "#2b6cb0", "R clean-room cell-view"),
            ("C-frozen-public-commit", "#c05621", "C frozen public commit"),
        ):
            subset = envelope_frame[
                (envelope_frame.backend_profile == backend)
                & (envelope_frame.split == "paper_scale")
                & (envelope_frame.architecture == "cell_view")
                & (envelope_frame.policy == policy)
            ].sort_values("grid_index")
            x = subset.normalized_swap_progress.to_numpy(dtype=float)
            mean = subset.mean_sortedness_percent.to_numpy(dtype=float)
            low = subset.mean_ci95_low.to_numpy(dtype=float)
            high = subset.mean_ci95_high.to_numpy(dtype=float)
            axis.plot(x, mean, color=color, label=label)
            axis.fill_between(x, low, high, color=color, alpha=0.18)
        axis.set_title(policy)
        axis.set_xlabel("Normalized successful-swap progress")
        axis.set_ylabel("Sortedness (%)")
        axis.set_ylim(40, 101)
        axis.grid(alpha=0.15)
    axes[0].legend(fontsize=8)
    figure.suptitle("Historical/reference overlay (paired inputs; scheduler streams not paired)")
    figure.savefig(output / "historical_reference_overlay.png", dpi=180)
    figure.savefig(output / "historical_reference_overlay.svg")
    plt.close(figure)


def build_artifacts(
    reference: Sequence[Mapping[str, Any]],
    historical: Sequence[Mapping[str, Any]],
    replay: Sequence[Mapping[str, Any]],
    variability: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    all_results = [dict(row) for row in [*reference, *historical]]
    for row in all_results:
        prefix = "reference" if row["backend_profile"].startswith("R-") else "historical-primary"
        row["run_id"] = f"{prefix}:{row['scenario_id']}"
        for field in (
            "source_precheck_valid",
            "historical_swap_probe",
            "historical_frozen_swap_attempts",
            "runtime_python",
        ):
            row.setdefault(field, None)
    _write_parquet([_summary_row(row) for row in all_results], output / "no_fault_runs.parquet")
    _write_parquet(list(_trajectory_rows(all_results)), output / "trajectory_grid.parquet")
    _write_parquet(list(_raw_trajectory_rows(all_results)), output / "paper_and_selected_swap_trajectories.parquet")

    accounting = _run_accounting(all_results)
    _write_json(output / "run_accounting.json", accounting)
    reference_contrasts, _ = _paired_contrasts(reference)
    historical_contrasts = _historical_contrasts(reference, historical)
    contrasts = [*reference_contrasts, *historical_contrasts]
    _write_parquet(contrasts, output / "paired_contrasts.parquet")
    envelopes = _trajectory_envelopes(all_results)
    _write_parquet(envelopes, output / "trajectory_envelopes.parquet")
    classifications = _claim_classifications(reference, reference_contrasts)
    pd.DataFrame(classifications).to_csv(output / "figure3_claim_classifications.csv", index=False)

    selected_ids = {
        min(
            row["scenario_id"] for row in all_results
            if (
                row["backend_profile"], row["split"], row["condition_id"]
            ) == group
        )
        for group in {
            (row["backend_profile"], row["split"], row["condition_id"])
            for row in all_results
        }
    }
    selected = [
        {
            "traceKind": "derived_successful_swap_sortedness",
            "sequenceBasis": (
                "accepted_swap_derived_from_activation"
                if row["backend_profile"].startswith("R-")
                else "historical_recorded_swap"
            ),
            "scenarioId": row["scenario_id"],
            "conditionId": row["condition_id"],
            "split": row["split"],
            "backendProfile": row["backend_profile"],
            "selectionReason": "preregistered_lexicographic_scenario_id",
            "points": row["raw_curve"],
        }
        for row in all_results
        if row["scenario_id"] in selected_ids
    ]
    selected_path = output / "selected_traces.jsonl.zst"
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in selected)
    with pa.OSFile(str(selected_path), "wb") as raw:
        with pa.CompressedOutputStream(raw, "zstd") as compressed:
            compressed.write(payload)
    _write_json(
        output / "selected_trace_manifest.json",
        {
            "schemaVersion": "e01.s09.selected_trajectory_manifest.v1",
            "recordCount": len(selected),
            "path": str(selected_path),
            "bytes": selected_path.stat().st_size,
            "sha256": sha256_file(selected_path),
            "note": "Derived swap-level trajectories, not shared-schema activation events; C remains recorded_swap and lossy.",
        },
    )
    _write_json(output / "exact_replay_samples.json", list(replay))
    _write_json(output / "historical_scheduler_variability.json", list(variability))
    _plot_figures(all_results, envelopes, output)

    final_validation = {
        "schemaVersion": "e01.s09.final_state_validation.v1",
        "runCount": len(all_results),
        "referenceRunCount": len(reference),
        "historicalRunCount": len(historical),
        "allCompleted": all(row["completed"] for row in all_results),
        "allFinalSortedness100": all(row["final_sortedness_percent"] == 100.0 for row in all_results),
        "allFinalNonStrictlySorted": all(row["final_nonstrictly_sorted"] for row in all_results),
        "allValueMultisetsConserved": all(row["value_multiset_conserved"] for row in all_results),
        "allReferenceReplaysByteExact": all(row["byteExactDigestReplay"] for row in replay),
        "allReferenceSummaryPathsMatch": all(row["summaryPathMatchesDigestPath"] for row in replay),
        "historicalExactReplayRequired": False,
        "historicalExactReplayReason": "C uses an OS thread scheduler; variability is evidence, not a replay failure.",
    }
    _write_json(output / "final_state_validation.json", final_validation)
    replication_summary = [
        {
            "research_step_id": "S09",
            "figure": 3,
            "claim_id": row["claim_id"],
            "endpoint_classification": row["endpoint_classification"],
            "qualitative_classification": row["qualitative_classification"],
            "outcome_classification": (
                "supportive"
                if row["qualitative_classification"] == "supportive"
                and row["endpoint_classification"] in {None, "supportive"}
                else "constraining/contradictory"
            ),
            "estimate": row["estimate"],
            "adjusted_ci_low": row["adjusted_ci_low"],
            "adjusted_ci_high": row["adjusted_ci_high"],
            "analysis_split": "confirmatory_holdout",
        }
        for row in classifications
    ]
    _write_parquet(replication_summary, output / "replication_summary.parquet")
    _write_parquet(replication_summary, Path("/artifacts/results/replication_summary.parquet"))
    return {
        "accounting": accounting,
        "contrasts": contrasts,
        "classifications": classifications,
        "finalValidation": final_validation,
    }


def validate_artifacts(output: Path) -> dict[str, Any]:
    """Run the frozen S09 release checks against compact final artifacts."""
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise AssertionError(f"{name}: {detail}")

    runs = pd.read_parquet(output / "no_fault_runs.parquet")
    check("run_count", len(runs) == 9900, len(runs))
    check("run_id_unique", runs.run_id.nunique() == 9900, runs.run_id.nunique())
    check("allowed_splits_only", set(runs.split) == set(ALLOWED_SPLITS), sorted(set(runs.split)))
    check(
        "forbidden_splits_absent",
        not set(runs.split) & FORBIDDEN_SPLITS,
        sorted(set(runs.split) & FORBIDDEN_SPLITS),
    )
    check("condition_set", set(runs.condition_id) == ALLOWED_CONDITIONS, sorted(set(runs.condition_id)))
    expected_groups = {
        ("R-clean-room-reference-E01-v1", "paper_scale", architecture, policy): 100
        for architecture in ("cell_view", "traditional")
        for policy in ("Bubble", "Insertion", "Selection")
    }
    expected_groups.update({
        ("R-clean-room-reference-E01-v1", "confirmatory_holdout", architecture, policy): 1000
        for architecture in ("cell_view", "traditional")
        for policy in ("Bubble", "Insertion", "Selection")
    })
    expected_groups.update({
        ("C-frozen-public-commit", "paper_scale", "cell_view", policy): 100
        for policy in ("Bubble", "Insertion", "Selection")
    })
    expected_groups.update({
        ("C-frozen-public-commit", "confirmatory_holdout", "cell_view", policy): 1000
        for policy in ("Bubble", "Insertion", "Selection")
    })
    observed_groups = {
        tuple(key): int(len(group))
        for key, group in runs.groupby(["backend_profile", "split", "architecture", "policy"])
    }
    check(
        "complete_group_accounting",
        observed_groups == expected_groups,
        [
            {
                "backendProfile": key[0], "split": key[1],
                "architecture": key[2], "policy": key[3], "count": value,
            }
            for key, value in sorted(observed_groups.items())
        ],
    )
    check("all_completed", bool(runs.completed.all()), int(runs.completed.sum()))
    check("no_timeouts", not bool(runs.timed_out.any()), int(runs.timed_out.sum()))
    check("all_sortedness_100", bool((runs.final_sortedness_percent == 100).all()), None)
    check("all_final_sorted", bool(runs.final_nonstrictly_sorted.all()), None)
    check("all_multisets_conserved", bool(runs.value_multiset_conserved.all()), None)
    check(
        "historical_source_prechecks",
        bool(runs[runs.backend_profile == "C-frozen-public-commit"].source_precheck_valid.all()),
        3300,
    )

    reference = runs[runs.backend_profile == "R-clean-room-reference-E01-v1"]
    for split, count in ALLOWED_SPLITS.items():
        for policy in ("Bubble", "Insertion", "Selection"):
            subset = reference[(reference.split == split) & (reference.policy == policy)]
            sizes = subset.groupby("pairing_block_id").size()
            check(
                f"reference_pairing_{split}_{policy}",
                len(sizes) == count and bool((sizes == 2).all()),
                {"blocks": len(sizes), "sizeValues": sorted(sizes.unique().tolist())},
            )

    grid = pq.read_table(output / "trajectory_grid.parquet")
    check("trajectory_grid_rows", grid.num_rows == 9900 * 201, grid.num_rows)
    grid_frame = grid.select(["backend_profile", "scenario_id", "grid_index"]).to_pandas()
    grid_sizes = grid_frame.groupby(["backend_profile", "scenario_id"]).size()
    check("trajectory_grid_complete", len(grid_sizes) == 9900 and bool((grid_sizes == 201).all()), len(grid_sizes))

    raw = pd.read_parquet(output / "paper_and_selected_swap_trajectories.parquet")
    raw_ids = raw[["backend_profile", "scenario_id"]].drop_duplicates()
    check("raw_trajectory_selection_count", len(raw_ids) == 909, len(raw_ids))

    selected_path = output / "selected_traces.jsonl.zst"
    with pa.memory_map(str(selected_path), "r") as source:
        with pa.CompressedInputStream(source, "zstd") as compressed:
            selected_payload = compressed.read()
    selected = [json.loads(line) for line in selected_payload.splitlines()]
    check("selected_trace_count", len(selected) == 18, len(selected))
    check(
        "selected_trace_unique",
        len({(row["backendProfile"], row["scenarioId"]) for row in selected}) == 18,
        len(selected),
    )

    replay = json.loads((output / "exact_replay_samples.json").read_text())
    check("exact_replay_sample_count", len(replay) == 12, len(replay))
    check("exact_replay_byte_equality", all(row["byteExactDigestReplay"] for row in replay), None)
    check("summary_path_parity", all(row["summaryPathMatchesDigestPath"] for row in replay), None)
    variability = json.loads((output / "historical_scheduler_variability.json").read_text())
    check("scheduler_variability_sample_count", len(variability) == 15, len(variability))
    check("scheduler_variability_completed", all(row["completed"] for row in variability), None)
    by_condition = {
        condition: [row for row in variability if row["conditionId"] == condition]
        for condition in {row["conditionId"] for row in variability}
    }
    check(
        "scheduler_variability_observed",
        any(len({row["traceSha256"] for row in rows}) > 1 for rows in by_condition.values()),
        {key: len({row["traceSha256"] for row in rows}) for key, rows in by_condition.items()},
    )

    contrasts = pd.read_parquet(output / "paired_contrasts.parquet")
    reference_confirm = contrasts[
        (contrasts.contrast_family == "reference_cell_view_minus_traditional")
        & (contrasts.split == "confirmatory_holdout")
    ]
    check("paired_contrast_rows", len(reference_confirm) == 12, len(reference_confirm))
    check("paired_contrast_pair_counts", bool((reference_confirm.pair_count == 1000).all()), None)
    claims = pd.read_csv(output / "figure3_claim_classifications.csv")
    check("figure3_claim_count", len(claims) == 4, len(claims))
    check(
        "figure3_claims_supportive",
        bool((claims.qualitative_classification == "supportive").all()),
        claims.qualitative_classification.tolist(),
    )

    check(
        "preregistration_hash",
        sha256_file(output / "confirmatory_preregistration.json") == sha256_file(PREREGISTRATION),
        sha256_file(output / "confirmatory_preregistration.json"),
    )
    frozen_check = verify_source_tree(HISTORICAL_SOURCE, S02_MANIFEST)
    check("frozen_source_still_valid", frozen_check["contentValid"], frozen_check)
    for figure in ("figure3_reconstruction.png", "historical_reference_overlay.png"):
        check(f"figure_nonempty_{figure}", (output / figure).stat().st_size > 10_000, (output / figure).stat().st_size)

    result = {
        "schemaVersion": "e01.s09.validation.v1",
        "researchStepId": "S09",
        "passed": all(item["passed"] for item in checks),
        "checkCount": len(checks),
        "checks": checks,
        "unexplainedFailures": 0,
    }
    _write_json(output / "validation_summary.json", result)
    return result
