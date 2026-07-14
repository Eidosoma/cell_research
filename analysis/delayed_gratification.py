"""S12 delayed-gratification reconstruction under the frozen paper metric.

The module keeps four layers explicit: paper wording, the exact pure function
found at the frozen public commit, the S03 reconciled interpretation, and
clean-room trajectories generated from S08 scenarios.  It never imports the
historical analysis module, whose top-level code loads unavailable author-local
arrays.  Validation instead AST-extracts only its pure metric functions from
the read-only quarantine and compares them with this implementation.
"""

from __future__ import annotations

import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.frozen_cell_results import (
    HISTORICAL_SOURCE,
    S02_CHECKSUMS,
    _execute_pure_cell_view_activation,
    _insertion_prefix_cache,
    condition_from_dict,
    load_tasks as load_s11_tasks,
    verify_source_tree,
)
from analysis.no_fault_sorting import load_tasks as load_s09_tasks
from reference_simulator.engine import (
    evaluate_terminal,
    execute_batch,
    execute_serial_summary_activation,
    initial_state,
)
from reference_simulator.model import Scenario, canonical_json_bytes, state_hash
from scenario_bank.core import materialize_scenario


REPOSITORY = Path(__file__).resolve().parents[1]
S01_CLAIMS = Path("/artifacts/research_steps/S01/claim_registry.parquet")
S08_DIR = Path("/artifacts/research_steps/S08")
S09_DIR = Path("/artifacts/research_steps/S09")
S11_DIR = Path("/artifacts/research_steps/S11")
OUTPUT_DIR = Path("/artifacts/research_steps/S12")
CACHE_DIR = Path("/cache/e01_s12")
PREREGISTRATION = REPOSITORY / "analysis" / "s12_delayed_gratification_preregistration.json"
FROZEN_DG_SOURCE = HISTORICAL_SOURCE / "analysis" / "delay_gratification_analysis_for_not_move.py"
S12_SCHEMA = "e01.s12.original_dg_run.v1"
PRIMARY_METRIC = "historical_dg_net_commit_exact_v1"
RECONCILED_METRIC = "historical_dg_net_s03_reconciled_v1"
ARXIV_METRIC = "arxiv_recovery_ratio_v1"
PREREGISTRATION_SHA256 = "8458227702637780e91b17d72c6b80326ff74a157820bbac8ba812aed539a984"
FIGURE7_RUNS = 4_200
FIGURE6_RUNS = 2_160
BOOTSTRAP_DRAWS = 10_000
PLACEMENTS = ("legacy_with_replacement", "reference_without_replacement")
POLICIES = ("Bubble", "Insertion", "Selection")
ARCHITECTURES = ("cell_view", "traditional")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(rows: Sequence[Mapping[str, Any]] | pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(rows, preserve_index=False) if isinstance(rows, pd.DataFrame) else pa.Table.from_pylist(list(rows))
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )


def collapse_plateaus(values: Sequence[int | float]) -> list[int | float]:
    result: list[int | float] = []
    for value in values:
        if not result or result[-1] != value:
            result.append(value)
    return result


def signed_runs(values: Sequence[int | float]) -> list[float]:
    """Reproduce frozen get_discrepency_arr, including its singleton [0]."""
    collapsed = collapse_plateaus(values)
    differences = [float(right - left) for left, right in zip(collapsed, collapsed[1:])]
    if not differences:
        return [0.0]
    result: list[float] = []
    current = differences[0]
    for previous, value in zip(differences, differences[1:]):
        if previous * value > 0:
            current += value
        if previous * value <= 0:
            result.append(current)
            current = value
    result.append(current)
    return result


def dg_metrics(error_trajectory: Sequence[int | float]) -> dict[str, Any]:
    """Compute exact-commit primary plus the two preregistered sensitivities."""
    runs = signed_runs(error_trajectory)
    first_positive = next((index for index, value in enumerate(runs) if value > 0), len(runs))
    pairs: list[dict[str, float]] = []
    index = first_positive
    terminal_early_return = False
    exact_sum = 0.0
    if index >= len(runs) - 1:
        terminal_early_return = index < len(runs) and runs[index] > 0
        exact_value = 0.0
    else:
        while index < len(runs):
            follower = index + 1
            if follower >= len(runs) or runs[follower] > 0:
                terminal_early_return = True
                exact_value = exact_sum
                break
            worsening = float(runs[index])
            recovery = float(-runs[follower])
            net = (recovery - worsening) / worsening
            ratio = recovery / worsening
            pairs.append(
                {
                    "worsening": worsening,
                    "recovery": recovery,
                    "net": net,
                    "recoveryRatio": ratio,
                }
            )
            exact_sum += net
            index += 2
        else:
            exact_value = exact_sum / len(pairs) if pairs else 0.0
    reconciled = float(np.mean([pair["net"] for pair in pairs])) if pairs else 0.0
    recovery_ratio = float(np.mean([pair["recoveryRatio"] for pair in pairs])) if pairs else 0.0
    return {
        "primary": float(exact_value),
        "reconciled": reconciled,
        "recoveryRatio": recovery_ratio,
        "signedRuns": runs,
        "pairNetScores": [pair["net"] for pair in pairs],
        "pairRecoveryRatios": [pair["recoveryRatio"] for pair in pairs],
        "pairCount": len(pairs),
        "worseningRunCount": sum(value > 0 for value in runs),
        "terminalEarlyReturn": terminal_early_return,
        "plateauObservationsRemoved": max(len(error_trajectory) - len(collapse_plateaus(error_trajectory)), 0),
    }


def monotonicity_error(values: Sequence[int | float]) -> int:
    return sum(right < left for left, right in zip(values, values[1:]))


def values_for(scenario: Scenario, occupancy: Sequence[str]) -> list[int | float]:
    cells = scenario.cell_map
    return [cells[cell_id].value for cell_id in occupancy]


class SwapErrorRecorder:
    """Update strict adjacent-inversion error in O(1) after an arbitrary swap."""

    def __init__(self, scenario: Scenario, occupancy: Sequence[str]) -> None:
        self.values = {cell.cell_id: cell.value for cell in scenario.cells}
        self.current = monotonicity_error([self.values[cell_id] for cell_id in occupancy])
        self.errors: list[int] = []

    def __call__(self, state: Any, left_position: int, right_position: int) -> None:
        occupancy = state.occupancy
        affected = {
            edge
            for position in (left_position, right_position)
            for edge in (position - 1, position)
            if 0 <= edge < len(occupancy) - 1
        }

        def before_id(position: int) -> str:
            if position == left_position:
                return occupancy[right_position]
            if position == right_position:
                return occupancy[left_position]
            return occupancy[position]

        before = sum(
            self.values[before_id(edge + 1)] < self.values[before_id(edge)]
            for edge in affected
        )
        after = sum(
            self.values[occupancy[edge + 1]] < self.values[occupancy[edge]]
            for edge in affected
        )
        self.current += int(after - before)
        self.errors.append(self.current)


def run_optimized_trajectory(scenario: Scenario) -> dict[str, Any]:
    """Replay exact R transitions and retain only the post-accepted-swap proxy."""
    scenario.validate()
    state = initial_state(scenario)
    initial_values = values_for(scenario, state.occupancy)
    recorder = SwapErrorRecorder(scenario, state.occupancy)
    state.terminal = evaluate_terminal(scenario, state)
    positions = {cell_id: index for index, cell_id in enumerate(state.occupancy)}
    insertion_cache: list[Any] = list(_insertion_prefix_cache(scenario, state))
    start = time.perf_counter()
    while state.terminal is None:
        if scenario.architecture.value == "cell_view":
            changed = _execute_pure_cell_view_activation(
                scenario,
                state,
                positions,
                insertion_cache,
                on_accepted_swap=recorder,
            )
        else:
            changed = execute_serial_summary_activation(
                scenario, state, on_accepted_swap=recorder
            )
        if changed:
            state.terminal = evaluate_terminal(scenario, state)
    elapsed = time.perf_counter() - start
    final_values = values_for(scenario, state.occupancy)
    if len(recorder.errors) != state.ledger["acceptedSwaps"]:
        raise AssertionError("one error observation is required per accepted swap")
    if recorder.current != monotonicity_error(final_values):
        raise AssertionError("incremental monotonicity error diverged from full recomputation")
    primary = dg_metrics(recorder.errors)
    initial_included = dg_metrics([monotonicity_error(initial_values), *recorder.errors])
    return {
        "initialError": monotonicity_error(initial_values),
        "finalError": recorder.current,
        "postSwapErrors": recorder.errors,
        "trajectorySha256": canonical_hash(recorder.errors),
        "initialIncludedTrajectorySha256": canonical_hash([monotonicity_error(initial_values), *recorder.errors]),
        "metrics": primary,
        "initialIncludedMetrics": initial_included,
        "stopReason": state.terminal,
        "completed": state.terminal == "complete",
        "censored": state.terminal in {"event_budget", "invariant_error"},
        "activationCount": state.activation_count,
        "acceptedSwaps": state.ledger["acceptedSwaps"],
        "finalStateHash": state_hash(scenario.scenario_id, state),
        "ledger": dict(state.ledger),
        "elapsedSeconds": elapsed,
        "initialValues": initial_values,
        "finalValues": final_values,
    }


def run_ordinary_trajectory(scenario: Scenario) -> dict[str, Any]:
    """Replay through the ordinary event-producing engine for validation."""
    scenario.validate()
    state = initial_state(scenario)
    initial_values = values_for(scenario, state.occupancy)
    state.terminal = evaluate_terminal(scenario, state)
    errors: list[int] = []
    while state.terminal is None:
        before = state.ledger["acceptedSwaps"]
        execute_batch(scenario, state, retain_events=False, emit_event_records=True)
        if state.ledger["acceptedSwaps"] > before:
            if state.ledger["acceptedSwaps"] != before + 1:
                raise AssertionError("S12 ordinary replay requires serial one-swap batches")
            errors.append(monotonicity_error(values_for(scenario, state.occupancy)))
    return {
        "postSwapErrors": errors,
        "trajectorySha256": canonical_hash(errors),
        "metrics": dg_metrics(errors),
        "initialIncludedMetrics": dg_metrics([monotonicity_error(initial_values), *errors]),
        "stopReason": state.terminal,
        "activationCount": state.activation_count,
        "acceptedSwaps": state.ledger["acceptedSwaps"],
        "finalStateHash": state_hash(scenario.scenario_id, state),
        "ledger": dict(state.ledger),
    }


def _catalog() -> dict[str, dict[str, Any]]:
    rows = pq.read_table(S08_DIR / "condition_catalog.parquet").to_pylist()
    return {row["conditionId"]: json.loads(row["conditionJson"]) for row in rows}


def load_figure7_tasks() -> list[dict[str, Any]]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise PermissionError("S12 preregistration changed after freezing")
    tasks: list[dict[str, Any]] = []
    for task in load_s09_tasks("paper_scale"):
        copied = dict(task)
        copied["population"] = "figure7_paper_scale"
        tasks.append(copied)
    for task in load_s11_tasks("paper_scale", "reference"):
        if task["condition"]["faultMode"] != "stuck":
            continue
        copied = dict(task)
        copied["population"] = "figure7_paper_scale"
        tasks.append(copied)
    tasks.sort(key=lambda task: (task["scenario_row"]["conditionId"], int(task["scenario_row"]["replicateOrdinal"])))
    if len(tasks) != FIGURE7_RUNS:
        raise ValueError(f"expected {FIGURE7_RUNS} Figure 7 tasks, found {len(tasks)}")
    if any(bool(task["scenario_row"]["protected"]) for task in tasks):
        raise PermissionError("Figure 7 population escaped unprotected paper_scale")
    selected: dict[str, str] = {}
    for task in tasks:
        row = task["scenario_row"]
        selected[row["conditionId"]] = min(selected.get(row["conditionId"], row["scenarioId"]), row["scenarioId"])
    for task in tasks:
        task["retain_trajectory"] = task["scenario_row"]["scenarioId"] == selected[task["scenario_row"]["conditionId"]]
    return tasks


def load_figure6_tasks() -> list[dict[str, Any]]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise PermissionError("S12 preregistration changed after freezing")
    rows = pq.read_table(
        S08_DIR / "paired_scenario_bank.parquet",
        filters=[("conditionFamily", "=", "worked_example_ambiguity_envelope"), ("split", "=", "ambiguity_envelope")],
    ).to_pylist()
    catalog = _catalog()
    bases = {
        row["baseDrawId"]: row
        for row in pq.read_table(
            S08_DIR / "base_draw_bank.parquet",
            filters=[("inputProfile", "=", "worked_1_6_exhaustive"), ("split", "=", "ambiguity_envelope")],
        ).to_pylist()
    }
    tasks = [
        {
            "scenario_row": row,
            "condition": catalog[row["conditionId"]],
            "base": bases[row["baseDrawId"]],
            "population": "figure6_ambiguity_envelope",
            "retain_trajectory": True,
        }
        for row in sorted(rows, key=lambda item: (item["conditionId"], int(item["replicateOrdinal"])))
    ]
    if len(tasks) != FIGURE6_RUNS:
        raise ValueError(f"expected {FIGURE6_RUNS} Figure 6 tasks, found {len(tasks)}")
    if any(bool(task["scenario_row"]["protected"]) for task in tasks):
        raise PermissionError("Figure 6 envelope must be unprotected")
    return tasks


def task_run_id(task: Mapping[str, Any]) -> str:
    row = task["scenario_row"]
    return "s12r:" + hashlib.sha256(f"{task['population']}|{row['scenarioId']}".encode()).hexdigest()


def _task_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    condition = condition_from_dict(task["condition"])
    scenario, metadata = materialize_scenario(condition, task["base"])
    row = task["scenario_row"]
    if scenario.scenario_id != row["scenarioId"]:
        raise ValueError("S08 scenario failed exact rematerialization")
    result = run_optimized_trajectory(scenario)
    metrics = result["metrics"]
    initial_metrics = result["initialIncludedMetrics"]
    retain_detail = bool(task["retain_trajectory"])
    policy = scenario.traditional_policy.value if scenario.traditional_policy else scenario.cells[0].policy.value
    output = {
        "schema_version": S12_SCHEMA,
        "research_step_id": "S12",
        "run_id": task_run_id(task),
        "population": task["population"],
        "scenario_id": scenario.scenario_id,
        "condition_id": row["conditionId"],
        "base_draw_id": row["baseDrawId"],
        "pairing_block_id": row["pairingBlockId"],
        "split": row["split"],
        "protected": bool(row["protected"]),
        "replicate_ordinal": int(row["replicateOrdinal"]),
        "architecture": scenario.architecture.value,
        "policy": policy,
        "fault_mode": row["faultMode"],
        "requested_fault_count": int(row["requestedFaultCount"]),
        "realized_fault_count": int(row["realizedFaultCount"]),
        "placement_profile": row["placementProfile"],
        "fault_map_id": row["faultMapId"],
        "event_budget": int(row["maxActivations"]),
        "scenario_json_sha256": metadata["scenarioJsonSha256"],
        "trajectory_sampling_profile": "post_accepted_swap_no_initial_v1",
        "metric_profile": PRIMARY_METRIC,
        "historical_dg_net": metrics["primary"],
        "s03_reconciled_dg_net": metrics["reconciled"],
        "arxiv_recovery_ratio": metrics["recoveryRatio"],
        "initial_included_historical_dg_net": initial_metrics["primary"],
        "initial_included_s03_reconciled_dg_net": initial_metrics["reconciled"],
        "initial_included_arxiv_recovery_ratio": initial_metrics["recoveryRatio"],
        "usable_pair_count": metrics["pairCount"],
        "worsening_run_count": metrics["worseningRunCount"],
        "terminal_early_return": metrics["terminalEarlyReturn"],
        "plateau_observations_removed": metrics["plateauObservationsRemoved"],
        "signed_runs": metrics["signedRuns"] if retain_detail else None,
        "pair_net_scores": metrics["pairNetScores"] if retain_detail else None,
        "pair_recovery_ratios": metrics["pairRecoveryRatios"] if retain_detail else None,
        "initial_monotonicity_error": result["initialError"],
        "final_monotonicity_error": result["finalError"],
        "post_swap_observation_count": len(result["postSwapErrors"]),
        "trajectory_sha256": result["trajectorySha256"],
        "stop_reason": result["stopReason"],
        "completed": result["completed"],
        "censored": result["censored"],
        "activation_count": result["activationCount"],
        "accepted_swaps": result["acceptedSwaps"],
        "final_state_hash": result["finalStateHash"],
        "elapsed_seconds": result["elapsedSeconds"],
        "post_swap_errors": result["postSwapErrors"] if task["retain_trajectory"] else None,
        "initial_values": result["initialValues"] if task["population"] == "figure6_ambiguity_envelope" else None,
        "final_values": result["finalValues"] if task["population"] == "figure6_ambiguity_envelope" else None,
    }
    for key, value in result["ledger"].items():
        output[f"ledger_{key}"] = value
    return output


def _read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["run_id"]] = row
    return rows


def run_population(tasks: Sequence[Mapping[str, Any]], checkpoint: Path, *, workers: int = 8) -> list[dict[str, Any]]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1,8]")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_checkpoint(checkpoint)
    missing = [task for task in tasks if task_run_id(task) not in completed]
    if missing:
        with checkpoint.open("a", encoding="utf-8") as handle, ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_task_worker, task): task for task in missing}
            for future in as_completed(futures):
                row = future.result()
                handle.write(canonical_json_bytes(row).decode("utf-8") + "\n")
                handle.flush()
                completed[row["run_id"]] = row
    ordered = [completed[task_run_id(task)] for task in tasks]
    if len({row["run_id"] for row in ordered}) != len(tasks):
        raise AssertionError("checkpoint contains duplicate requested run IDs")
    return ordered


def frozen_code_metric_oracle() -> Any:
    """Extract only the frozen pure metric helpers without importing the module."""
    source = FROZEN_DG_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(FROZEN_DG_SOURCE))
    required = {"dedup", "get_discrepency_arr", "get_first_pos", "avg_wandering_range"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in required
    ]
    if {node.name for node in nodes} != required:
        raise ValueError("frozen DG source is missing a required pure function")
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(FROZEN_DG_SOURCE), "exec"), namespace)
    return namespace["avg_wandering_range"]


def hand_calculated_fixtures() -> dict[str, Any]:
    fixtures = [
        {"id": "empty", "trajectory": [], "primary": 0.0, "reconciled": 0.0, "ratio": 0.0},
        {"id": "all_plateau", "trajectory": [3, 3, 3], "primary": 0.0, "reconciled": 0.0, "ratio": 0.0},
        {"id": "improvement_only", "trajectory": [5, 4, 3, 2], "primary": 0.0, "reconciled": 0.0, "ratio": 0.0},
        {"id": "unrecovered_worsening", "trajectory": [1, 2, 3], "primary": 0.0, "reconciled": 0.0, "ratio": 0.0},
        {"id": "figure6_panel", "trajectory": [4, 4, 2, 2, 3, 2], "primary": 0.0, "reconciled": 0.0, "ratio": 1.0},
        {"id": "net_positive", "trajectory": [5, 3, 4, 1], "primary": 2.0, "reconciled": 2.0, "ratio": 3.0},
        {"id": "net_negative_retained", "trajectory": [5, 3, 5, 4], "primary": -0.5, "reconciled": -0.5, "ratio": 0.5},
        {"id": "two_complete_pairs", "trajectory": [8, 6, 7, 5, 7, 3], "primary": 1.0, "reconciled": 1.0, "ratio": 2.0},
        {"id": "terminal_sum_quirk", "trajectory": [8, 6, 7, 5, 7, 3, 6], "primary": 2.0, "reconciled": 1.0, "ratio": 2.0},
    ]
    oracle = frozen_code_metric_oracle()
    rows = []
    for fixture in fixtures:
        observed = dg_metrics(fixture["trajectory"])
        frozen = float(oracle(list(fixture["trajectory"])))
        checks = {
            "primary": math.isclose(observed["primary"], fixture["primary"], abs_tol=1e-12),
            "reconciled": math.isclose(observed["reconciled"], fixture["reconciled"], abs_tol=1e-12),
            "ratio": math.isclose(observed["recoveryRatio"], fixture["ratio"], abs_tol=1e-12),
            "frozenCode": math.isclose(observed["primary"], frozen, abs_tol=1e-12),
        }
        rows.append({**fixture, "observed": observed, "frozenCodePrimary": frozen, "checks": checks, "success": all(checks.values())})
    return {
        "schemaVersion": "e01.s12.hand_fixtures.v1",
        "researchStepId": "S12",
        "fixtureCount": len(rows),
        "success": all(row["success"] for row in rows),
        "fixtures": rows,
    }


def frozen_code_randomized_comparison(samples: int = 1_000) -> dict[str, Any]:
    oracle = frozen_code_metric_oracle()
    rng = random.Random(0xE011200000000001)
    failures = []
    for index in range(samples):
        length = rng.randint(0, 120)
        trajectory = [rng.randint(0, 99) for _ in range(length)]
        expected = float(oracle(trajectory))
        observed = dg_metrics(trajectory)["primary"]
        if not math.isclose(expected, observed, rel_tol=0.0, abs_tol=1e-12):
            failures.append({"sample": index, "trajectory": trajectory, "frozen": expected, "cleanRoom": observed})
            if len(failures) >= 10:
                break
    return {
        "schemaVersion": "e01.s12.frozen_code_comparison.v1",
        "researchStepId": "S12",
        "frozenCommit": "1fd2bd5921c1f6b423a71f691d5189106a8a1020",
        "sourceRelativePath": "analysis/delay_gratification_analysis_for_not_move.py",
        "sourceSha256": sha256_file(FROZEN_DG_SOURCE),
        "method": "AST-extracted pure functions only; full module not imported or redistributed",
        "randomSeed": "0xE011200000000001",
        "samples": samples,
        "failureCount": len(failures),
        "failures": failures,
        "success": not failures,
    }


def _bootstrap_interval(values: np.ndarray, seed: int, confidence: float) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for start in range(0, BOOTSTRAP_DRAWS, 1_000):
        size = min(1_000, BOOTSTRAP_DRAWS - start)
        means[start : start + size] = rng.choice(values, size=(size, len(values)), replace=True).mean(axis=1)
    alpha = 1.0 - confidence
    return tuple(float(value) for value in np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def _analysis_view(frame: pd.DataFrame, placement: str) -> pd.DataFrame:
    view = frame[
        (frame.requested_fault_count == 0)
        | ((frame.requested_fault_count > 0) & (frame.placement_profile == placement))
    ].copy()
    view["analysis_placement"] = placement
    return view


def figure7_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for placement in PLACEMENTS:
        view = _analysis_view(frame, placement)
        for keys, group in view.groupby(["architecture", "policy", "requested_fault_count"], sort=True):
            values = group.historical_dg_net.to_numpy(dtype=float)
            rows.append(
                {
                    "analysis_placement": placement,
                    "architecture": keys[0],
                    "policy": keys[1],
                    "fault_count": int(keys[2]),
                    "runs": len(group),
                    "mean_historical_dg_net": float(values.mean()),
                    "population_sd": float(values.std(ddof=0)),
                    "median": float(np.median(values)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "mean_s03_reconciled": float(group.s03_reconciled_dg_net.mean()),
                    "mean_arxiv_ratio": float(group.arxiv_recovery_ratio.mean()),
                    "mean_initial_included_primary": float(group.initial_included_historical_dg_net.mean()),
                    "runs_with_usable_pair": int((group.usable_pair_count > 0).sum()),
                    "terminal_early_returns": int(group.terminal_early_return.sum()),
                    "completed": int(group.completed.sum()),
                    "quiescent": int((group.stop_reason == "quiescent").sum()),
                    "censored": int(group.censored.sum()),
                }
            )
    return pd.DataFrame(rows)


def panel_mean_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    claims = pd.read_parquet(S01_CLAIMS)
    claims = claims[(claims.figure == 7) & (claims.claim_kind == "panel_mean")]
    primary = summary[summary.analysis_placement == "legacy_with_replacement"]
    rows = []
    for claim in claims.itertuples(index=False):
        match = primary[
            (primary.architecture == claim.architecture)
            & (primary.policy == claim.algorithm)
            & (primary.fault_count == int(claim.fault_count))
        ]
        if len(match) != 1:
            raise AssertionError(f"expected one panel match for {claim.claim_id}")
        row = match.iloc[0]
        paper = float(claim.reported_mean)
        estimate = float(row.mean_historical_dg_net)
        difference = estimate - paper
        margin = max(0.05, 0.10 * abs(paper))
        if round(estimate, 2) == round(paper, 2):
            classification = "reproduced_to_display_precision"
        elif abs(difference) <= margin:
            classification = "approximately_reproduced_within_frozen_margin"
        else:
            classification = "not_reproduced"
        rows.append(
            {
                "claim_id": claim.claim_id,
                "panel": claim.panel,
                "architecture": claim.architecture,
                "policy": claim.algorithm,
                "fault_count": int(claim.fault_count),
                "paper_mean": paper,
                "generated_mean": estimate,
                "difference": difference,
                "frozen_approximation_margin": margin,
                "classification": classification,
                "paper_value_provenance": "readable_arxiv_v1_Figure7_bar_label; raw arrays unavailable",
                "analysis_placement": "legacy_with_replacement",
            }
        )
    return pd.DataFrame(rows)


def architecture_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    claims = pd.read_parquet(S01_CLAIMS)
    claims = claims[(claims.figure == 7) & (claims.claim_kind == "numerical_contrast")]
    rows = []
    for placement in PLACEMENTS:
        view = _analysis_view(frame, placement)
        for policy in POLICIES:
            subset = view[view.policy == policy]
            pivot = subset.pivot(index=["pairing_block_id", "requested_fault_count"], columns="architecture", values="historical_dg_net")
            if len(pivot) != 400 or pivot.isna().any().any():
                raise AssertionError(f"incomplete Figure 7 architecture pairing for {placement}/{policy}")
            differences = (pivot.cell_view - pivot.traditional).to_numpy(dtype=float)
            low, high = _bootstrap_interval(differences, int(canonical_hash(["contrast", placement, policy])[:16], 16), 0.95)
            claim = claims[claims.algorithm == policy].iloc[0]
            paper_magnitude = float(claim.reported_mean)
            target = -paper_magnitude if "< traditional" in str(claim.reported_effect) else paper_magnitude
            margin = max(0.05, 0.10 * abs(target))
            estimate = float(differences.mean())
            if round(estimate, 2) == round(target, 2):
                endpoint = "reproduced_to_display_precision"
            elif abs(estimate - target) <= margin:
                endpoint = "approximately_reproduced_within_frozen_margin"
            else:
                endpoint = "not_reproduced"
            expected = str(claim.reported_effect)
            if "approximately equals" in expected:
                qualitative = "supportive" if abs(estimate) <= margin else "contradictory"
            elif "< traditional" in expected:
                qualitative = "supportive" if high < 0 else ("contradictory" if low > 0 else "inconclusive")
            else:
                qualitative = "supportive" if low > 0 else ("contradictory" if high < 0 else "inconclusive")
            rows.append(
                {
                    "claim_id": claim.claim_id,
                    "analysis_placement": placement,
                    "policy": policy,
                    "pairs": len(differences),
                    "estimate_cell_view_minus_traditional": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "paper_target_signed": target,
                    "frozen_approximation_margin": margin,
                    "endpoint_classification": endpoint,
                    "qualitative_classification": qualitative,
                    "historical_z": claim.reported_statistic,
                    "historical_p": claim.reported_p_value,
                }
            )
    return pd.DataFrame(rows)


def fault_count_trends(frame: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    confidence = 1.0 - 0.05 / 6.0
    for placement in PLACEMENTS:
        view = _analysis_view(frame, placement)
        for architecture in ARCHITECTURES:
            for policy in POLICIES:
                subset = view[(view.architecture == architecture) & (view.policy == policy)]
                means = [float(subset[subset.requested_fault_count == count].historical_dg_net.mean()) for count in range(4)]
                f0 = subset[subset.requested_fault_count == 0].set_index("pairing_block_id").historical_dg_net
                f3 = subset[subset.requested_fault_count == 3].set_index("pairing_block_id").historical_dg_net
                joined = pd.concat([f0.rename("f0"), f3.rename("f3")], axis=1, join="inner")
                if len(joined) != 100 or joined.isna().any().any():
                    raise AssertionError(f"incomplete f0/f3 pairing for {placement}/{architecture}/{policy}")
                differences = (joined.f3 - joined.f0).to_numpy(dtype=float)
                low, high = _bootstrap_interval(
                    differences,
                    int(canonical_hash(["trend", placement, architecture, policy])[:16], 16),
                    confidence,
                )
                monotone = all(right >= left for left, right in zip(means, means[1:]))
                if policy in {"Bubble", "Insertion"}:
                    classification = "supportive" if monotone and low > 0 else ("contradictory" if high < 0 else "not_confirmed")
                    expected = "monotone_increase"
                else:
                    classification = "consistent_not_confirmatory" if (not monotone or low <= 0 <= high) else "contradictory_clear_increase"
                    expected = "no_clear_trend"
                rows.append(
                    {
                        "analysis_placement": placement,
                        "architecture": architecture,
                        "policy": policy,
                        "f0_mean": means[0],
                        "f1_mean": means[1],
                        "f2_mean": means[2],
                        "f3_mean": means[3],
                        "monotone_nondecreasing": monotone,
                        "paired_f3_minus_f0": float(differences.mean()),
                        "adjusted_confidence": confidence,
                        "adjusted_ci_low": low,
                        "adjusted_ci_high": high,
                        "paper_expectation": expected,
                        "classification": classification,
                    }
                )
    return pd.DataFrame(rows)


def _contains_contiguous(values: Sequence[int], target: Sequence[int]) -> bool:
    return any(list(values[index : index + len(target)]) == list(target) for index in range(max(len(values) - len(target) + 1, 0)))


def _contains_subsequence(values: Sequence[int], target: Sequence[int]) -> bool:
    iterator = iter(values)
    return all(any(value == wanted for value in iterator) for wanted in target)


def figure6_analysis(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    target = [2, 2, 4, 4, 3, 4]
    displayed_initial = [2, 5, 4, 1, 6, 3]
    augmented = frame.copy()
    augmented["sortedness_counts"] = augmented.apply(
        lambda row: [6 - int(row.initial_monotonicity_error)] + [6 - int(value) for value in row.post_swap_errors],
        axis=1,
    )
    augmented["local_drop_count"] = augmented.sortedness_counts.map(
        lambda values: sum(right < left for left, right in zip(values, values[1:]))
    )
    augmented["multiple_local_drops"] = augmented.local_drop_count >= 2
    augmented["displayed_initial_order"] = augmented.initial_values.map(lambda values: list(values) == displayed_initial)
    augmented["panel_sequence_contiguous"] = augmented.sortedness_counts.map(lambda values: _contains_contiguous(values, target))
    augmented["panel_sequence_subsequence"] = augmented.sortedness_counts.map(lambda values: _contains_subsequence(values, target))
    summary_rows = []
    for policy, group in augmented.groupby("policy", sort=True):
        summary_rows.append(
            {
                "policy": policy,
                "runs": len(group),
                "runs_with_any_drop": int((group.local_drop_count >= 1).sum()),
                "runs_with_multiple_drops": int(group.multiple_local_drops.sum()),
                "mean_drop_count": float(group.local_drop_count.mean()),
                "mean_historical_dg_net": float(group.historical_dg_net.mean()),
                "mean_usable_pairs": float(group.usable_pair_count.mean()),
                "displayed_initial_order_runs": int(group.displayed_initial_order.sum()),
                "contiguous_panel_sequence_matches": int(group.panel_sequence_contiguous.sum()),
                "subsequence_panel_sequence_matches": int(group.panel_sequence_subsequence.sum()),
                "completed": int(group.completed.sum()),
                "quiescent": int((group.stop_reason == "quiescent").sum()),
                "censored": int(group.censored.sum()),
            }
        )
    exact_rows = augmented[augmented.displayed_initial_order].copy()
    selected_ids = set(exact_rows.scenario_id)
    for policy, group in augmented[augmented.multiple_local_drops].groupby("policy"):
        selected_ids.add(group.sort_values("scenario_id").iloc[0].scenario_id)
    selected = augmented[augmented.scenario_id.isin(selected_ids)][
        [
            "scenario_id", "condition_id", "policy", "initial_values", "final_values",
            "sortedness_counts", "post_swap_errors", "historical_dg_net", "s03_reconciled_dg_net",
            "arxiv_recovery_ratio", "local_drop_count", "displayed_initial_order",
            "panel_sequence_contiguous", "panel_sequence_subsequence", "stop_reason",
        ]
    ].copy()
    fixture = {
        "schemaVersion": "e01.s12.figure6_fixture.v1",
        "researchStepId": "S12",
        "provenance": "hand-transcribed from readable official arXiv v1 Figure 6 panel B; simulator event sequence remains unavailable",
        "displayedStates": [
            [2, 5, 4, 1, 6, 3],
            [1, 5, 4, 2, 6, 3],
            [1, 2, 4, 5, 6, 3],
            [1, 2, 4, 5, 6, 3],
            [1, 2, 4, 3, 6, 5],
            [1, 2, 4, 3, 5, 6],
        ],
        "stuckValue": 4,
        "sortednessCounts": target,
        "monotonicityErrors": [4, 4, 2, 2, 3, 2],
        "primaryDg": dg_metrics([4, 4, 2, 2, 3, 2])["primary"],
        "s03ReconciledDg": dg_metrics([4, 4, 2, 2, 3, 2])["reconciled"],
        "arxivRecoveryRatio": dg_metrics([4, 4, 2, 2, 3, 2])["recoveryRatio"],
        "interpretation": "The shown drop is fully recovered but not exceeded, so normalized net DG is zero while recovery/drop is one.",
        "exactHistoricalAlgorithmInputStreamRecovered": False,
        "ambiguityEnvelopeRuns": len(augmented),
    }
    return pd.DataFrame(summary_rows), selected, fixture, augmented


def rank_classifications(summary: pd.DataFrame) -> pd.DataFrame:
    claims = pd.read_parquet(S01_CLAIMS)
    claims = claims[(claims.figure == 7) & (claims.claim_kind == "algorithm_rank")]
    rows = []
    for placement in PLACEMENTS:
        values = (
            summary[summary.analysis_placement == placement]
            .groupby(["architecture", "policy"], as_index=False)
            .mean(numeric_only=True)
        )
        pivot = values.pivot(index="architecture", columns="policy", values="mean_historical_dg_net")
        insertion_over_bubble = bool((pivot.Insertion > pivot.Bubble).all())
        selection_over_both = bool(((pivot.Selection > pivot.Insertion) & (pivot.Selection > pivot.Bubble)).all())
        for claim in claims.itertuples(index=False):
            if claim.claim_id == "F07-AB-INSERTION-BUBBLE-RANK":
                supportive = insertion_over_bubble
                estimate = float((pivot.Insertion - pivot.Bubble).min())
            else:
                supportive = selection_over_both
                estimate = float((pivot.Selection - pivot[["Bubble", "Insertion"]].max(axis=1)).min())
            rows.append(
                {
                    "claim_id": claim.claim_id,
                    "analysis_placement": placement,
                    "estimate_min_architecture_margin": estimate,
                    "classification": "supportive" if supportive else "contradictory",
                    "cell_view_bubble": float(pivot.loc["cell_view", "Bubble"]),
                    "cell_view_insertion": float(pivot.loc["cell_view", "Insertion"]),
                    "cell_view_selection": float(pivot.loc["cell_view", "Selection"]),
                    "traditional_bubble": float(pivot.loc["traditional", "Bubble"]),
                    "traditional_insertion": float(pivot.loc["traditional", "Insertion"]),
                    "traditional_selection": float(pivot.loc["traditional", "Selection"]),
                }
            )
    return pd.DataFrame(rows)


def draw_figure7(summary: pd.DataFrame, output_dir: Path) -> None:
    paper = panel_mean_comparison(summary)
    primary = summary[summary.analysis_placement == "legacy_with_replacement"]
    colors = {"traditional": "#4C78A8", "cell_view": "#E45756"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, policy in zip(axes, POLICIES):
        subset = primary[primary.policy == policy]
        x = np.arange(4)
        width = 0.36
        for offset, architecture in [(-width / 2, "traditional"), (width / 2, "cell_view")]:
            bars = subset[subset.architecture == architecture].sort_values("fault_count")
            axis.bar(
                x + offset,
                bars.mean_historical_dg_net,
                width,
                yerr=bars.population_sd,
                capsize=3,
                color=colors[architecture],
                alpha=0.78,
                label=architecture.replace("_", " "),
            )
            targets = paper[(paper.policy == policy) & (paper.architecture == architecture)].sort_values("fault_count")
            axis.scatter(x + offset, targets.paper_mean, marker="D", s=24, color="black", zorder=4)
        axis.set_title(policy)
        axis.set_xticks(x, [f"f={value}" for value in x])
        axis.set_xlabel("requested stuck-cell count")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("historical DG net (mean ± population SD)")
    axes[0].legend(frameon=False)
    fig.suptitle("Figure 7 reconstruction — legacy placement; diamonds are paper bar labels", fontsize=13)
    fig.savefig(output_dir / "figure7_reconstruction.png", dpi=220)
    fig.savefig(output_dir / "figure7_reconstruction.svg")
    plt.close(fig)


def _normalized_curve(values: Sequence[int], points: int = 101) -> np.ndarray:
    if len(values) == 1:
        return np.repeat(float(values[0]), points)
    source = np.linspace(0.0, 1.0, len(values))
    return np.interp(np.linspace(0.0, 1.0, points), source, np.asarray(values, dtype=float))


def draw_figure6(envelope: pd.DataFrame, fixture: Mapping[str, Any], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    counts = fixture["sortednessCounts"]
    axes[0].plot(range(len(counts)), counts, marker="o", color="#4C78A8", linewidth=2)
    axes[0].axvspan(3.5, 4.5, color="#E45756", alpha=0.16, label="temporary drop")
    axes[0].set_title("Panel B hand reconstruction")
    axes[0].set_xlabel("displayed state")
    axes[0].set_ylabel("Sortedness count (of 6)")
    axes[0].set_ylim(0, 6.2)
    axes[0].legend(frameon=False)
    colors = {"Bubble": "#4C78A8", "Insertion": "#F2CF5B", "Selection": "#54A24B"}
    for policy, group in envelope.groupby("policy", sort=True):
        curves = np.stack([_normalized_curve(values) for values in group.sortedness_counts])
        x = np.linspace(0, 100, curves.shape[1])
        axes[1].plot(x, curves.mean(axis=0), color=colors[policy], label=policy)
        axes[1].fill_between(x, np.quantile(curves, 0.10, axis=0), np.quantile(curves, 0.90, axis=0), color=colors[policy], alpha=0.15)
    axes[1].set_title("All 2,160 ambiguity-envelope runs")
    axes[1].set_xlabel("accepted-swap progress (%)")
    axes[1].set_ylabel("Sortedness count (mean; 10–90% band)")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    fig.suptitle("Figure 6 reconstruction — panel facts separated from clean-room envelope", fontsize=13)
    fig.savefig(output_dir / "figure6_reconstruction.png", dpi=220)
    fig.savefig(output_dir / "figure6_reconstruction.svg")
    plt.close(fig)


def replay_validation(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [
        task
        for task in tasks
        if int(task["scenario_row"]["requestedFaultCount"]) == 0
        or task["scenario_row"]["placementProfile"] == "legacy_with_replacement"
    ]
    selected: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for task in candidates:
        condition = task["condition"]
        key = (condition["architecture"], condition["policies"][0], int(condition["requestedFaultCount"]))
        if key not in selected or task["scenario_row"]["scenarioId"] < selected[key]["scenario_row"]["scenarioId"]:
            selected[key] = task
    if len(selected) != 24:
        raise AssertionError(f"expected 24 replay profiles, found {len(selected)}")
    rows = []
    for key, task in sorted(selected.items()):
        condition = condition_from_dict(task["condition"])
        scenario, _ = materialize_scenario(condition, task["base"])
        first = run_optimized_trajectory(scenario)
        second = run_optimized_trajectory(scenario)
        ordinary = run_ordinary_trajectory(scenario)
        checks = {
            "optimizedByteExact": canonical_json_bytes(first["metrics"]) == canonical_json_bytes(second["metrics"]),
            "optimizedTrajectoryHash": first["trajectorySha256"] == second["trajectorySha256"],
            "ordinaryTrajectoryHash": first["trajectorySha256"] == ordinary["trajectorySha256"],
            "ordinaryMetric": canonical_json_bytes(first["metrics"]) == canonical_json_bytes(ordinary["metrics"]),
            "finalStateHash": first["finalStateHash"] == second["finalStateHash"] == ordinary["finalStateHash"],
            "ledger": first["ledger"] == second["ledger"] == ordinary["ledger"],
        }
        rows.append(
            {
                "architecture": key[0],
                "policy": key[1],
                "faultCount": key[2],
                "scenarioId": scenario.scenario_id,
                "trajectorySha256": first["trajectorySha256"],
                "metricSha256": canonical_hash(first["metrics"]),
                "postSwapObservations": len(first["postSwapErrors"]),
                "checks": checks,
                "success": all(checks.values()),
            }
        )
    return {
        "schemaVersion": "e01.s12.replay_validation.v1",
        "researchStepId": "S12",
        "samples": len(rows),
        "success": all(row["success"] for row in rows),
        "rows": rows,
    }


def s09_no_fault_path_comparison(frame: pd.DataFrame) -> dict[str, Any]:
    selected = frame[
        (frame.requested_fault_count == 0) & frame.post_swap_errors.notna()
    ]
    if len(selected) != 6:
        raise AssertionError(f"expected six retained no-fault trajectories, found {len(selected)}")
    ids = selected.scenario_id.tolist()
    prior = pq.read_table(
        S09_DIR / "paper_and_selected_swap_trajectories.parquet",
        filters=[("scenario_id", "in", ids)],
    ).to_pandas()
    # S09 co-locates frozen-C comparator and clean-room reference curves.
    # S12 trajectories are reference trajectories, so this identity gate must
    # compare against the corresponding backend only.
    prior = prior[prior.backend_profile == "R-clean-room-reference-E01-v1"]
    rows = []
    for row in selected.itertuples(index=False):
        curve = prior[prior.scenario_id == row.scenario_id].sort_values("successful_swap_index")
        if curve.empty or int(curve.successful_swap_index.iloc[0]) != 0:
            raise AssertionError("S09 curve lacks its initial observation")
        s09_errors = [int(round(100 - value)) for value in curve.sortedness_percent.iloc[1:]]
        current = [int(value) for value in row.post_swap_errors]
        checks = {
            "postSwapCount": len(s09_errors) == len(current),
            "trajectory": s09_errors == current,
            "hash": canonical_hash(s09_errors) == row.trajectory_sha256,
        }
        rows.append(
            {
                "scenarioId": row.scenario_id,
                "architecture": row.architecture,
                "policy": row.policy,
                "observations": len(current),
                "checks": checks,
                "success": all(checks.values()),
            }
        )
    return {
        "schemaVersion": "e01.s12.s09_path_comparison.v1",
        "researchStepId": "S12",
        "samples": len(rows),
        "success": all(row["success"] for row in rows),
        "rows": rows,
    }


def run_accounting(figure7: pd.DataFrame, figure6: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([figure7, figure6], ignore_index=True)
    return (
        combined.groupby(
            ["population", "split", "architecture", "policy", "fault_mode", "requested_fault_count", "placement_profile", "stop_reason"],
            dropna=False,
        )
        .agg(
            runs=("run_id", "size"),
            completed=("completed", "sum"),
            censored=("censored", "sum"),
            accepted_swaps=("accepted_swaps", "sum"),
            post_swap_observations=("post_swap_observation_count", "sum"),
        )
        .reset_index()
    )


def metric_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["exact_minus_s03_reconciled"] = result.mean_historical_dg_net - result.mean_s03_reconciled
    result["arxiv_minus_exact"] = result.mean_arxiv_ratio - result.mean_historical_dg_net
    result["initial_included_minus_primary"] = result.mean_initial_included_primary - result.mean_historical_dg_net
    return result[
        [
            "analysis_placement", "architecture", "policy", "fault_count", "runs",
            "mean_historical_dg_net", "mean_s03_reconciled", "mean_arxiv_ratio",
            "mean_initial_included_primary", "exact_minus_s03_reconciled",
            "arxiv_minus_exact", "initial_included_minus_primary", "terminal_early_returns",
        ]
    ]


def claim_classifications(
    panel: pd.DataFrame,
    contrasts: pd.DataFrame,
    ranks: pd.DataFrame,
    figure6_summary: pd.DataFrame,
    figure6_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for row in panel.itertuples(index=False):
        supportive = row.classification != "not_reproduced"
        rows.append(
            {
                "figure": 7,
                "claim_id": row.claim_id,
                "endpoint_classification": row.classification,
                "qualitative_classification": "supportive" if supportive else "contradictory",
                "outcome_classification": "supportive" if supportive else "constraining/contradictory",
                "estimate": row.generated_mean,
                "adjusted_ci_low": np.nan,
                "adjusted_ci_high": np.nan,
                "analysis_split": "paper_scale_legacy_with_replacement",
                "caveat": "Paper raw arrays/error-bar construction unavailable; target is a transcribed bar label.",
            }
        )
    primary_contrasts = contrasts[contrasts.analysis_placement == "legacy_with_replacement"]
    for row in primary_contrasts.itertuples(index=False):
        supportive = row.qualitative_classification == "supportive"
        rows.append(
            {
                "figure": 7,
                "claim_id": row.claim_id,
                "endpoint_classification": row.endpoint_classification,
                "qualitative_classification": row.qualitative_classification,
                "outcome_classification": "supportive" if supportive else "constraining/contradictory",
                "estimate": row.estimate_cell_view_minus_traditional,
                "adjusted_ci_low": row.ci95_low,
                "adjusted_ci_high": row.ci95_high,
                "analysis_split": "paper_scale_legacy_with_replacement",
                "caveat": "Paired clean-room interval; historical z construction unavailable and not reproduced as inference.",
            }
        )
    primary_ranks = ranks[ranks.analysis_placement == "legacy_with_replacement"]
    for row in primary_ranks.itertuples(index=False):
        supportive = row.classification == "supportive"
        rows.append(
            {
                "figure": 7,
                "claim_id": row.claim_id,
                "endpoint_classification": "rank_reproduced" if supportive else "rank_not_reproduced",
                "qualitative_classification": row.classification,
                "outcome_classification": "supportive" if supportive else "constraining/contradictory",
                "estimate": row.estimate_min_architecture_margin,
                "adjusted_ci_low": np.nan,
                "adjusted_ci_high": np.nan,
                "analysis_split": "paper_scale_legacy_with_replacement",
                "caveat": "Rank uses regenerated paper-scale means; historical pooled z contrast is unavailable.",
            }
        )
    multiple = int((figure6_frame.local_drop_count >= 2).sum())
    figure6_rows = [
        {
            "figure": 6,
            "claim_id": "F06-B-FROZEN-CELL-EXAMPLE",
            "endpoint_classification": "hand_fixture_reproduced_metric_zero",
            "qualitative_classification": "supportive_with_ambiguity",
            "outcome_classification": "supportive",
            "estimate": 0.0,
            "adjusted_ci_low": np.nan,
            "adjusted_ci_high": np.nan,
            "analysis_split": "hand_fixture_plus_ambiguity_envelope",
            "caveat": "Displayed states are hand-transcribed; exact algorithm, scheduler, and omitted event sequence are not recovered.",
        },
        {
            "figure": 6,
            "claim_id": "F06-C-MULTIPLE-LOCAL-DROPS",
            "endpoint_classification": "qualitatively_reproduced_in_envelope",
            "qualitative_classification": "supportive",
            "outcome_classification": "supportive",
            "estimate": float(multiple),
            "adjusted_ci_low": np.nan,
            "adjusted_ci_high": np.nan,
            "analysis_split": "ambiguity_envelope",
            "caveat": "Envelope is exhaustive at f=1; f=2/f=3 context is assessed separately in Figure 7 trends.",
        },
        {
            "figure": 6,
            "claim_id": "F06-D-DG-DEFINITION",
            "endpoint_classification": "implementation_sensitive_formula_conflict",
            "qualitative_classification": "constraining",
            "outcome_classification": "constraining/contradictory",
            "estimate": 0.0,
            "adjusted_ci_low": np.nan,
            "adjusted_ci_high": np.nan,
            "analysis_split": "metric_reconciliation",
            "caveat": "Journal net formula, arXiv ratio wording, and frozen terminal-return behavior are distinct named metrics.",
        },
    ]
    rows.extend(figure6_rows)
    result = pd.DataFrame(rows).sort_values(["figure", "claim_id"]).reset_index(drop=True)
    if len(result) != 32 or result.claim_id.nunique() != 32:
        raise AssertionError(f"expected 32 empirical/metric Figure 6-7 claim rows, found {len(result)}")
    return result


def stable_replication_summary(claims: pd.DataFrame) -> pd.DataFrame:
    path = Path("/artifacts/results/replication_summary.parquet")
    existing = pd.read_parquet(path)
    existing = existing[existing.research_step_id != "S12"]
    additions = claims[
        [
            "figure", "claim_id", "endpoint_classification", "qualitative_classification",
            "outcome_classification", "estimate", "adjusted_ci_low", "adjusted_ci_high", "analysis_split",
        ]
    ].copy()
    additions.insert(0, "research_step_id", "S12")
    result = pd.concat([existing, additions], ignore_index=True)
    result = result.sort_values(["research_step_id", "figure", "claim_id"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(result, path)
    return result


def validate_populations(
    figure7: pd.DataFrame,
    figure6: pd.DataFrame,
    summary: pd.DataFrame,
    panel: pd.DataFrame,
) -> dict[str, Any]:
    combined = pd.concat([figure7, figure6], ignore_index=True)
    checks = {
        "figure7Runs": len(figure7) == FIGURE7_RUNS,
        "figure6Runs": len(figure6) == FIGURE6_RUNS,
        "uniqueRunIds": combined.run_id.nunique() == len(combined),
        "uniqueScenarioWithinPopulation": not combined.duplicated(["population", "scenario_id"]).any(),
        "onlyDeclaredSplits": set(combined.split) == {"paper_scale", "ambiguity_envelope"},
        "noProtectedRows": not bool(combined.protected.any()),
        "allPostSwapCounts": bool((combined.post_swap_observation_count == combined.accepted_swaps).all()),
        "allMetricsFinite": bool(np.isfinite(combined[["historical_dg_net", "s03_reconciled_dg_net", "arxiv_recovery_ratio"]].to_numpy()).all()),
        "allScenarioHashes": bool(combined.scenario_json_sha256.str.fullmatch(r"[0-9a-f]{64}").all()),
        "allTrajectoryHashes": bool(combined.trajectory_sha256.str.fullmatch(r"[0-9a-f]{64}").all()),
        "allCensoringAccounted": int(combined.censored.sum()) == int((combined.stop_reason.isin(["event_budget", "invariant_error"])).sum()),
        "figure7ProfileCounts": bool((figure7.groupby("condition_id").size() == 100).all()) and figure7.condition_id.nunique() == 42,
        "figure6ProfileCounts": bool((figure6.groupby("condition_id").size() == 720).all()) and figure6.condition_id.nunique() == 3,
        "summaryCounts": len(summary) == 48 and bool((summary.runs == 100).all()),
        "panelClaims": len(panel) == 24 and panel.claim_id.nunique() == 24,
        "reportedMeansRecomputed": True,
    }
    for row in panel.itertuples(index=False):
        match = summary[
            (summary.analysis_placement == "legacy_with_replacement")
            & (summary.architecture == row.architecture)
            & (summary.policy == row.policy)
            & (summary.fault_count == row.fault_count)
        ]
        if len(match) != 1 or not math.isclose(float(match.iloc[0].mean_historical_dg_net), row.generated_mean, abs_tol=1e-12):
            checks["reportedMeansRecomputed"] = False
            break
    return {
        "schemaVersion": "e01.s12.population_validation.v1",
        "researchStepId": "S12",
        "checks": checks,
        "success": all(checks.values()),
        "observed": {
            "figure7Runs": len(figure7),
            "figure6Runs": len(figure6),
            "censoredRuns": int(combined.censored.sum()),
            "stopReasons": {str(key): int(value) for key, value in combined.stop_reason.value_counts().items()},
        },
    }


def formula_reconciliation() -> dict[str, Any]:
    return {
        "schemaVersion": "e01.s12.formula_reconciliation.v1",
        "researchStepId": "S12",
        "decisionFrozenBeforeOutcomes": True,
        "preregistrationSha256": sha256_file(PREREGISTRATION),
        "journal": {
            "formula": "(recovery-worsening)/worsening",
            "source": "supplied journal Methods 3.4 and Figure 6 caption",
            "role": "agrees with the frozen code's per-pair net formula",
        },
        "officialArxivV1": {
            "formula": "recovery/worsening",
            "source": "/cache/e01_s01/arxiv-2401.05375.pdf pages 29-30",
            "role": "named sensitivity only",
        },
        "figure6Example": {
            "sortednessCounts": [2, 2, 4, 4, 3, 4],
            "netFormulaValue": 0.0,
            "recoveryRatioValue": 1.0,
            "observation": "temporary backtracking is visible, but recovery does not exceed the preceding drop",
        },
        "frozenCode": {
            "commit": "1fd2bd5921c1f6b423a71f691d5189106a8a1020",
            "relativePath": "analysis/delay_gratification_analysis_for_not_move.py",
            "sha256": sha256_file(FROZEN_DG_SOURCE),
            "function": "avg_wandering_range",
            "formula": "-(negativeRecoveryRun + positiveWorseningRun) / positiveWorseningRun",
            "edgeBehavior": "terminal unmatched worsening returns accumulated sum before averaging",
        },
        "s03": {
            "profile": "historical_dg_net",
            "decision": "average usable net pairs, zero if none; retain arXiv ratio separately",
            "s12Treatment": "reported as historical_dg_net_s03_reconciled_v1 beside the exact frozen-code primary",
        },
        "primary": PRIMARY_METRIC,
        "sensitivities": [RECONCILED_METRIC, ARXIV_METRIC, "initial_state_included"],
        "claimBoundary": "All scores are trajectory proxies. None demonstrates foresight or replaces Experiment 3 global-distance analysis.",
    }


def environment_record(workers: int) -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "pyarrow", "matplotlib", "scipy", "pytest"):
        try:
            from importlib.metadata import version

            packages[name] = version(name)
        except Exception:
            packages[name] = None
    return {
        "schemaVersion": "e01.s12.environment.v1",
        "researchStepId": "S12",
        "python": sys.version,
        "platform": platform.platform(),
        "cpuCountVisible": os.cpu_count(),
        "workers": workers,
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "packages": packages,
        "newDependenciesInstalled": [],
    }


def write_edge_case_note(path: Path) -> None:
    text = """# S12 delayed-gratification edge-case contract

This is a metric contract artifact, not a separate handoff report. The binding preregistration is
`analysis/s12_delayed_gratification_preregistration.json` at SHA-256
`8458227702637780e91b17d72c6b80326ff74a157820bbac8ba812aed539a984`.

The primary trajectory contains monotonicity error after each accepted swap, excluding the initial
state, no-ops, rejections, activations, and memory-only updates. Consecutive equal values are removed;
same-sign differences are merged. Leading improvements are discarded. Each positive worsening run
is paired with its immediate negative recovery run and scored `(recovery-worsening)/worsening`.
Negative and zero scores are retained. No pair means zero. A positive run is the only denominator,
so zero denominators cannot occur.

The exact frozen public function has one terminal quirk: an unmatched final worsening returns the
accumulated sum before division by the number of prior pairs. S12 preserves that behavior in
`historical_dg_net_commit_exact_v1` and reports the always-average S03 interpretation separately as
`historical_dg_net_s03_reconciled_v1`. The official arXiv `recovery/worsening` wording is the named
`arxiv_recovery_ratio_v1` sensitivity. None of these metrics establishes planning or foresight.
"""
    path.write_text(text, encoding="utf-8")


def analyze(figure7_rows: Sequence[Mapping[str, Any]], figure6_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure7 = pd.DataFrame(figure7_rows).sort_values(["condition_id", "replicate_ordinal"]).reset_index(drop=True)
    figure6 = pd.DataFrame(figure6_rows).sort_values(["condition_id", "replicate_ordinal"]).reset_index(drop=True)
    write_parquet(figure7, output_dir / "original_dg_results.parquet")
    write_parquet(figure6, output_dir / "figure6_ambiguity_envelope.parquet")

    summary = figure7_summary(figure7)
    panel = panel_mean_comparison(summary)
    contrasts = architecture_contrasts(figure7)
    trends = fault_count_trends(figure7, summary)
    ranks = rank_classifications(summary)
    fig6_summary, fig6_selected, fig6_fixture, fig6_augmented = figure6_analysis(figure6)
    accounting = run_accounting(figure7, figure6)
    sensitivity = metric_sensitivity(summary)
    claims = claim_classifications(panel, contrasts, ranks, fig6_summary, fig6_augmented)

    summary.to_csv(output_dir / "figure7_summary.csv", index=False)
    panel.to_csv(output_dir / "paper_mean_comparison.csv", index=False)
    contrasts.to_csv(output_dir / "architecture_contrasts.csv", index=False)
    trends.to_csv(output_dir / "fault_count_trends.csv", index=False)
    ranks.to_csv(output_dir / "algorithm_rank_classifications.csv", index=False)
    fig6_summary.to_csv(output_dir / "figure6_envelope_summary.csv", index=False)
    write_parquet(fig6_selected, output_dir / "figure6_selected_trajectories.parquet")
    write_json(output_dir / "figure6_panel_fixture.json", fig6_fixture)
    accounting.to_csv(output_dir / "run_accounting.csv", index=False)
    sensitivity.to_csv(output_dir / "metric_sensitivity.csv", index=False)
    claims.to_csv(output_dir / "figure6_7_claim_classifications.csv", index=False)
    write_json(output_dir / "formula_reconciliation.json", formula_reconciliation())
    write_edge_case_note(output_dir / "edge_case_rules.md")
    draw_figure7(summary, output_dir)
    draw_figure6(fig6_augmented, fig6_fixture, output_dir)
    stable = stable_replication_summary(claims)

    population_validation = validate_populations(figure7, figure6, summary, panel)
    write_json(output_dir / "population_validation.json", population_validation)
    return {
        "figure7Runs": len(figure7),
        "figure6Runs": len(figure6),
        "panelClaims": len(panel),
        "panelReproduced": int(panel.classification.str.startswith("reproduced").sum()),
        "panelApproximatelyReproduced": int(panel.classification.str.startswith("approximately").sum()),
        "panelNotReproduced": int((panel.classification == "not_reproduced").sum()),
        "supportivePrimaryContrasts": int((contrasts[contrasts.analysis_placement == "legacy_with_replacement"].qualitative_classification == "supportive").sum()),
        "stableS12Claims": int((stable.research_step_id == "S12").sum()),
        "populationValidation": population_validation["success"],
        "censoredRuns": int(pd.concat([figure7, figure6]).censored.sum()),
    }
