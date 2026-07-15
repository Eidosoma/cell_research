#!/usr/bin/env python3
"""Build the frozen E03 S12 larger-n paired trajectory corpus."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from reference_simulator.engine import evaluate_terminal, execute_batch, initial_state
from reference_simulator.model import (
    Architecture,
    Cell,
    Direction,
    FaultMode,
    Policy,
    Scenario,
    canonical_json_bytes,
)
from reference_simulator.scheduler import ScheduledOpportunity
from src.detours.context_sensitivity import (
    LEDGER_NAMES,
    METRIC_NAMES,
    POLICY_NAMES,
    STOP_COMPLETE,
    STOP_EVENT_BUDGET,
    STOP_QUIESCENT,
    bounded,
    counter_draw,
    metric_profile,
    result_dict,
    simulate_population,
    stream_root,
)
from src.detours.distances import distance_profile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/artifacts/research_steps/S12")
CACHE = Path("/cache/e03_s12")
CONTRACT = ROOT / "analysis/s12_context_sensitivity_contract.json"
S03 = Path("/artifacts/research_steps/S03")
S07 = Path("/artifacts/research_steps/S07")
S08 = Path("/artifacts/research_steps/S08")
S09 = Path("/artifacts/research_steps/S09")
S10 = Path("/artifacts/research_steps/S10")
S11 = Path("/artifacts/research_steps/S11")

SIZES = (12, 18, 24)
POLICY_PROFILES: dict[str, tuple[int, ...]] = {
    "pure_Bubble": (0,),
    "pure_Insertion": (1,),
    "pure_Selection": (2,),
    "alternating_Bubble_Insertion": (0, 1),
    "alternating_Bubble_Selection": (0, 2),
    "alternating_Insertion_Selection": (1, 2),
    "cyclic_Bubble_Insertion_Selection": (0, 1, 2),
}
SCHEDULERS = {"serial_external": 1, "batch2_external": 2, "batch4_external": 4}
DIRECTIONS = {"ascending": 0, "descending": 1}
DISORDER_PROFILES = ("low", "medium", "high")
BASE_FAULT_COUNTS = (1, 2)
PLACEMENTS = {
    1: ("boundary_low", "boundary_high", "interior_low", "interior_high"),
    2: ("boundary_pair", "interior_contiguous", "interior_dispersed", "mixed_boundary_interior"),
}
INTERVENTIONS = ("baseline", "remove", "move", "add")
REPLICATES = 8
EXPECTED_BLOCKS = 24_192
EXPECTED_RUNS = 96_768
TRACE_RUNS = math.ceil(EXPECTED_RUNS * 0.01)
STOP_LABELS = {STOP_COMPLETE: "complete", STOP_QUIESCENT: "quiescent", STOP_EVENT_BUDGET: "event_budget"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(path: Path, frame: pd.DataFrame, schema: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {b"schemaVersion": schema.encode(), b"researchStepId": b"S12"}
    )
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )


def input_paths() -> list[Path]:
    paths = [
        Path("/workspace/AGENTS.md"),
        Path("/workspace/FULL_PLAN.md"),
        Path("/workspace/RESEARCH_PLAN.md"),
        Path("/workspace/PREVIOUS_ARTIFACTS.md"),
        Path("/workspace/PREVIOUS_ARTIFACTS.json"),
        Path("/workspace/DATASETS.md"),
        Path("/workspace/DATASET_CATALOG.json"),
        Path("/workspace/DATASET_AVAILABILITY.json"),
        Path("/workspace/input-attachments/MANIFEST.json"),
        S03 / "research_step_full_results.md",
        S03 / "metric_disagreement_atlas.parquet",
        S07 / "research_step_full_results.md",
        S07 / "path_solutions.parquet",
        S08 / "research_step_full_results.md",
        S09 / "research_step_full_results.md",
        S09 / "barrier_interventions.parquet",
        S10 / "research_step_full_results.md",
        S10 / "action_suppression_results.parquet",
        S11 / "research_step_full_results.md",
        S11 / "null_results.parquet",
        S11 / "null_metric_results.parquet",
        S11 / "action_rate_matching.parquet",
        CONTRACT,
        ROOT / "scripts/build_s12_context_sensitivity.py",
        ROOT / "src/detours/context_sensitivity.py",
        ROOT / "reference_simulator/engine.py",
        ROOT / "reference_simulator/model.py",
        ROOT / "reference_simulator/policies.py",
        ROOT / "reference_simulator/scheduler.py",
        ROOT / "reference_simulator/transition_primitives.py",
    ]
    paths.extend(sorted(Path("/workspace/input-attachments").glob("*/_metadata/ATTACHMENT.md")))
    return [path for path in paths if path.exists()]


def input_hashes() -> dict[str, str]:
    return {str(path): sha256_file(path) for path in input_paths()}


def seed_from_text(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def policy_codes(profile: str, n: int) -> np.ndarray:
    cycle = POLICY_PROFILES[profile]
    return np.asarray([cycle[index % len(cycle)] for index in range(n)], dtype=np.int8)


def initial_occupancy(n: int, direction: int, disorder: str, key: str) -> np.ndarray:
    goal = np.arange(n, dtype=np.int16)
    if direction == 1:
        goal = goal[::-1].copy()
    rng = np.random.default_rng(seed_from_text(key))
    if disorder == "medium":
        result = goal.copy()
        rng.shuffle(result)
        return result
    result = goal.copy() if disorder == "low" else goal[::-1].copy()
    swaps = max(2, n // 2)
    for _ in range(swaps):
        left = int(rng.integers(0, n - 1))
        result[left], result[left + 1] = result[left + 1], result[left]
    return result


def identity_from_goal_rank(rank: int, n: int, direction: int) -> int:
    return rank if direction == 0 else n - 1 - rank


def baseline_fault_ranks(n: int, count: int, placement: str) -> tuple[int, ...]:
    if count == 1:
        values = {
            "boundary_low": (0,),
            "boundary_high": (n - 1,),
            "interior_low": (n // 3,),
            "interior_high": (2 * n // 3,),
        }
    else:
        values = {
            "boundary_pair": (0, n - 1),
            "interior_contiguous": (n // 2 - 1, n // 2),
            "interior_dispersed": (n // 3, 2 * n // 3),
            "mixed_boundary_interior": (0, n // 2),
        }
    return tuple(sorted(set(values[placement])))


def intervention_ranks(
    n: int, baseline: tuple[int, ...], intervention: str
) -> tuple[int, ...]:
    if intervention == "baseline":
        return baseline
    if intervention == "remove":
        return baseline[1:]
    if intervention == "move":
        moved: list[int] = []
        for rank in baseline:
            candidate = (rank + n // 2) % n
            while candidate in moved:
                candidate = (candidate + 1) % n
            moved.append(candidate)
        return tuple(sorted(moved))
    candidates = (n // 2, n // 4, 3 * n // 4, 0, n - 1)
    added = list(baseline)
    for rank in candidates:
        if rank not in added:
            added.append(rank)
            break
    return tuple(sorted(added))


def location_class(ranks: tuple[int, ...], n: int) -> str:
    if not ranks:
        return "none"
    boundary = [rank for rank in ranks if rank <= 1 or rank >= n - 2]
    if len(boundary) == len(ranks):
        return "boundary"
    if len(ranks) > 1 and max(np.diff(sorted(ranks)), default=n) <= 1:
        return "contiguous_interior"
    if boundary:
        return "mixed_boundary_interior"
    if len(ranks) > 1 and max(ranks) - min(ranks) >= n // 3:
        return "dispersed_interior"
    return "interior"


def build_design() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    max_n = max(SIZES)
    rows: list[dict[str, Any]] = []
    policies: list[np.ndarray] = []
    faults: list[np.ndarray] = []
    occupancies: list[np.ndarray] = []
    roots: list[np.uint64] = []
    block_count = 0
    for n in SIZES:
        for policy_profile in POLICY_PROFILES:
            policy = policy_codes(policy_profile, n)
            for scheduler, batch_width in SCHEDULERS.items():
                for direction_name, direction in DIRECTIONS.items():
                    for disorder in DISORDER_PROFILES:
                        for baseline_count in BASE_FAULT_COUNTS:
                            for placement in PLACEMENTS[baseline_count]:
                                for replicate in range(REPLICATES):
                                    block_key = (
                                        f"n={n}|policy={policy_profile}|scheduler={scheduler}|"
                                        f"direction={direction_name}|disorder={disorder}|"
                                        f"basefault={baseline_count}|placement={placement}|rep={replicate}"
                                    )
                                    block_id = "s12b:" + hashlib.sha256(block_key.encode()).hexdigest()
                                    root = stream_root(block_id)
                                    occupancy = initial_occupancy(n, direction, disorder, block_id)
                                    initial_metrics = metric_profile(occupancy, n, direction)
                                    normalized_disorder = float(initial_metrics[1] / (n * (n - 1) / 2))
                                    base_ranks = baseline_fault_ranks(n, baseline_count, placement)
                                    for intervention in INTERVENTIONS:
                                        ranks = intervention_ranks(n, base_ranks, intervention)
                                        identities = tuple(
                                            identity_from_goal_rank(rank, n, direction) for rank in ranks
                                        )
                                        fault_row = np.zeros(max_n, dtype=np.int8)
                                        for identity in identities:
                                            fault_row[identity] = 1
                                        policy_row = np.full(max_n, -1, dtype=np.int8)
                                        policy_row[:n] = policy
                                        occupancy_row = np.full(max_n, -1, dtype=np.int16)
                                        occupancy_row[:n] = occupancy
                                        run_key = f"{block_id}|{intervention}"
                                        run_id = "s12r:" + hashlib.sha256(run_key.encode()).hexdigest()
                                        rows.append(
                                            {
                                                "schema_version": "e03.s12.design.v1",
                                                "research_step_id": "S12",
                                                "run_id": run_id,
                                                "scenario_block_id": block_id,
                                                "replicate_index": replicate,
                                                "n": n,
                                                "architecture": "cell_view",
                                                "policy_profile": policy_profile,
                                                "scheduler_profile": scheduler,
                                                "batch_width": batch_width,
                                                "direction": direction_name,
                                                "direction_code": direction,
                                                "initial_disorder_profile": disorder,
                                                "initial_inversion_fraction": normalized_disorder,
                                                "baseline_fault_count": baseline_count,
                                                "baseline_placement_profile": placement,
                                                "intervention_type": intervention,
                                                "fault_mode": "stuck" if identities else "none",
                                                "fault_count": len(identities),
                                                "fault_identity_indices": list(identities),
                                                "fault_goal_ranks": list(ranks),
                                                "fault_location_class": location_class(ranks, n),
                                                "event_budget": 8 * n * n,
                                                "stream_root_hex": f"{int(root):016x}",
                                                "stream_profile": "sha256_root_splitmix64_counter_external_schedule_v1",
                                                "exact_structural_reachability": "out_of_exact_domain_large_n",
                                            }
                                        )
                                        policies.append(policy_row)
                                        faults.append(fault_row)
                                        occupancies.append(occupancy_row)
                                        roots.append(root)
                                    block_count += 1
    frame = pd.DataFrame(rows)
    if block_count != EXPECTED_BLOCKS or len(frame) != EXPECTED_RUNS:
        raise AssertionError(f"design accounting drift: {block_count} blocks, {len(frame)} runs")
    if frame.groupby("scenario_block_id").size().ne(4).any():
        raise AssertionError("every scenario block must have four intervention arms")
    selection = sorted(
        frame.index,
        key=lambda index: hashlib.sha256(frame.loc[index, "run_id"].encode()).digest(),
    )[:TRACE_RUNS]
    frame["trace_retained"] = False
    frame.loc[selection, "trace_retained"] = True
    arrays = {
        "ns": frame.n.to_numpy(np.int16),
        "policies": np.asarray(policies, dtype=np.int8),
        "faults": np.asarray(faults, dtype=np.int8),
        "occupancies": np.asarray(occupancies, dtype=np.int16),
        "directions": frame.direction_code.to_numpy(np.int8),
        "batch_widths": frame.batch_width.to_numpy(np.int8),
        "roots": np.asarray(roots, dtype=np.uint64),
        "budgets": frame.event_budget.to_numpy(np.int32),
    }
    return frame, arrays


def run_kernel(arrays: Mapping[str, np.ndarray], *, audit: tuple[np.ndarray, ...] | None = None, budgets: np.ndarray | None = None, checkpoints: np.ndarray | None = None) -> dict[str, np.ndarray]:
    count = len(arrays["ns"])
    if audit is None:
        target_swaps = np.full(count, -1, dtype=np.int32)
        target_memory = np.full(count, -1, dtype=np.int32)
        target_denominators = np.full(count, -1, dtype=np.int32)
    else:
        target_swaps, target_memory, target_denominators = audit
    result = simulate_population(
        arrays["ns"], arrays["policies"], arrays["faults"], arrays["occupancies"],
        arrays["directions"], arrays["batch_widths"], arrays["roots"],
        arrays["budgets"] if budgets is None else budgets,
        target_swaps, target_memory, target_denominators,
        np.zeros(count, dtype=np.int32) if checkpoints is None else checkpoints,
    )
    return result_dict(result)


def verify_audit_immutability(native: Mapping[str, np.ndarray], audited: Mapping[str, np.ndarray]) -> None:
    excluded = {"requested", "deficits", "checkpoint_stop", "checkpoint_fingerprint", "checkpoint_ledger", "checkpoint_levels"}
    mismatches = []
    for key in native:
        if key in excluded:
            continue
        if not np.array_equal(native[key], audited[key]):
            mismatches.append(key)
    if mismatches:
        raise AssertionError(f"shadow feasibility audit mutated native results: {mismatches}")


def subset_arrays(arrays: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[indices] for key, value in arrays.items()}


def build_long_horizon(
    design: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    native: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray]:
    eligible = np.flatnonzero(
        (design.replicate_index.to_numpy() == 0)
        & (native["stop"] == STOP_EVENT_BUDGET)
    )
    long_arrays = subset_arrays(arrays, eligible)
    primary_budgets = arrays["budgets"][eligible]
    long_budgets = primary_budgets * 4
    long_result = run_kernel(long_arrays, budgets=long_budgets, checkpoints=primary_budgets)
    checks = (
        (long_result["checkpoint_fingerprint"] == native["fingerprint"][eligible])
        & np.all(long_result["checkpoint_ledger"] == native["ledger"][eligible], axis=1)
        & np.all(long_result["checkpoint_levels"] == native["final"][eligible], axis=1)
    )
    if not checks.all():
        raise AssertionError("long-horizon prefix mismatch")
    rows = design.iloc[eligible][
        ["run_id", "scenario_block_id", "replicate_index", "n", "policy_profile", "scheduler_profile", "direction", "intervention_type"]
    ].copy()
    rows["primary_event_budget"] = primary_budgets
    rows["long_event_budget"] = long_budgets
    rows["prefix_match"] = checks
    rows["long_stop_reason"] = [STOP_LABELS[int(value)] for value in long_result["stop"]]
    rows["long_event_count"] = long_result["events"]
    rows["long_state_fingerprint_u64"] = [f"{int(value):016x}" for value in long_result["fingerprint"]]
    for metric_index, metric in enumerate(METRIC_NAMES):
        rows[f"long_final_{metric}"] = long_result["final"][:, metric_index]
        rows[f"long_max_depth_{metric}"] = long_result["max_depth"][:, metric_index]
        rows[f"long_open_episode_{metric}"] = long_result["open_episode"][:, metric_index].astype(bool)
    return rows.reset_index(drop=True), eligible


class CommonSchedule:
    def __init__(self, scenario: Scenario, root: int) -> None:
        self.scenario = scenario
        self.root = np.uint64(root)

    def __call__(self, event_index: int, remaining: int) -> tuple[ScheduledOpportunity, ...]:
        width = min(self.scenario.batch_width, remaining)
        result = []
        for ordinal in range(width):
            event = event_index + ordinal
            actor_draw = counter_draw(self.root, event, 1)
            actor_index = ((int(actor_draw) >> 32) * len(self.scenario.cells)) >> 32
            actor_id = f"cell-{actor_index:04d}"
            actor = self.scenario.cell_map[actor_id]
            draws = [("s12_actor", event, 0, int(actor_draw))]
            consumption = [("s12_actor", 1)]
            bubble_side = None
            if actor.policy == Policy.BUBBLE and actor.fault == FaultMode.NORMAL:
                side_draw = counter_draw(self.root, event, 2)
                side_index = ((int(side_draw) >> 32) * 2) >> 32
                bubble_side = "left" if side_index == 0 else "right"
                draws.append(("bubble_side", event, 0, int(side_draw)))
                consumption.append(("bubble_side", 1))
            result.append(
                ScheduledOpportunity(
                    actor_id=actor_id,
                    random_draws=tuple(draws),
                    stream_consumption=tuple(consumption),
                    bubble_side=bubble_side,
                )
            )
        return tuple(result)


class CommonPriority:
    def __init__(self, root: int) -> None:
        self.root = np.uint64(root)

    def prepare(self, proposal, event_index: int):
        return replace(proposal, priority=int(counter_draw(self.root, event_index, 3))), ()

    def outcome(self, proposal, validation, event_index: int):
        return validation

    def after_batch(self, scenario, state, proposals, decisions, batch_start_index: int) -> None:
        return None


def reference_task_payload(row: Mapping[str, Any], policy_row: np.ndarray, fault_row: np.ndarray, occupancy: np.ndarray) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "n": int(row["n"]),
        "policy": policy_row[: int(row["n"])].tolist(),
        "fault": fault_row[: int(row["n"])].tolist(),
        "occupancy": occupancy[: int(row["n"])].tolist(),
        "direction": int(row["direction_code"]),
        "batch_width": int(row["batch_width"]),
        "root": int(row["stream_root_hex"], 16),
        "event_budget": int(row["event_budget"]),
    }


def reference_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    n = int(task["n"])
    direction = Direction.ASCENDING if int(task["direction"]) == 0 else Direction.DESCENDING
    cells = tuple(
        Cell(
            cell_id=f"cell-{identity:04d}",
            value=identity,
            policy=Policy(POLICY_NAMES[int(task["policy"][identity])]),
            direction=direction,
            fault=FaultMode.STUCK if int(task["fault"][identity]) else FaultMode.NORMAL,
        )
        for identity in range(n)
    )
    scenario = Scenario.create(
        cells,
        initial_occupancy=tuple(f"cell-{identity:04d}" for identity in task["occupancy"]),
        seed=0,
        max_activations=int(task["event_budget"]),
        architecture=Architecture.CELL_VIEW,
        batch_width=int(task["batch_width"]),
        generation_key=f"E03/S12/reference/{task['run_id']}",
        fault_placement="explicit",
        requested_fault_count=sum(int(value) for value in task["fault"]),
    )
    state = initial_state(scenario)
    state.terminal = evaluate_terminal(scenario, state)
    schedule = CommonSchedule(scenario, int(task["root"]))
    priority = CommonPriority(int(task["root"]))
    trace = []
    checkpoint_index = 0
    values = [scenario.cell_map[cell_id].value for cell_id in state.occupancy]
    initial_profile = distance_profile(values, direction=direction.value)
    trace.append(
        {
            "run_id": task["run_id"],
            "checkpoint_index": checkpoint_index,
            "event_count": 0,
            **{f"distance_{metric}": int(getattr(initial_profile, metric)) for metric in METRIC_NAMES},
        }
    )
    while state.terminal is None:
        before = state.ledger["acceptedSwaps"]
        execute_batch(
            scenario,
            state,
            retain_events=False,
            emit_event_records=False,
            schedule_factory=schedule,
            execution_interceptor=priority,
        )
        if state.ledger["acceptedSwaps"] > before:
            checkpoint_index += 1
            values = [scenario.cell_map[cell_id].value for cell_id in state.occupancy]
            profile = distance_profile(values, direction=direction.value)
            trace.append(
                {
                    "run_id": task["run_id"],
                    "checkpoint_index": checkpoint_index,
                    "event_count": state.activation_count,
                    **{f"distance_{metric}": int(getattr(profile, metric)) for metric in METRIC_NAMES},
                }
            )
    if trace[-1]["event_count"] != state.activation_count:
        checkpoint_index += 1
        values = [scenario.cell_map[cell_id].value for cell_id in state.occupancy]
        profile = distance_profile(values, direction=direction.value)
        trace.append(
            {
                "run_id": task["run_id"],
                "checkpoint_index": checkpoint_index,
                "event_count": state.activation_count,
                "terminal_marker": True,
                **{f"distance_{metric}": int(getattr(profile, metric)) for metric in METRIC_NAMES},
            }
        )
    final_occ = [int(identity.split("-")[1]) for identity in state.occupancy]
    final_cursor = {
        int(identity.split("-")[1]): int(cursor) for identity, cursor in state.selection_cursors.items()
    }
    return {
        "run_id": task["run_id"],
        "stop_reason": state.terminal,
        "event_count": state.activation_count,
        "ledger": dict(state.ledger),
        "final_occupancy": final_occ,
        "final_cursors": final_cursor,
        "trace": trace,
    }


def build_reference_traces(
    design: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    native: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indices = np.flatnonzero(design.trace_retained.to_numpy())
    tasks = [
        reference_task_payload(
            design.iloc[index], arrays["policies"][index], arrays["faults"][index], arrays["occupancies"][index]
        )
        for index in indices
    ]
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reference_worker, tasks, chunksize=4))
    validations = []
    traces = []
    for index, result in zip(indices, results, strict=True):
        n = int(arrays["ns"][index])
        expected_ledger = native["ledger"][index]
        ledger = result["ledger"]
        checks = {
            "stop": result["stop_reason"] == STOP_LABELS[int(native["stop"][index])],
            "events": int(result["event_count"]) == int(native["events"][index]),
            "ledger": all(
                int(ledger[name]) == int(expected_ledger[position])
                for position, name in enumerate(LEDGER_NAMES)
            ),
            "occupancy": result["final_occupancy"] == native["final_occupancy"][index, :n].tolist(),
            "cursors": all(
                int(native["final_cursors"][index, identity]) == cursor
                for identity, cursor in result["final_cursors"].items()
            ),
            "metrics": all(
                int(result["trace"][-1][f"distance_{metric}"]) == int(native["final"][index, metric_index])
                for metric_index, metric in enumerate(METRIC_NAMES)
            ),
        }
        validations.append(
            {
                "run_id": result["run_id"],
                **{f"{name}_match": value for name, value in checks.items()},
                "all_match": all(checks.values()),
            }
        )
        traces.extend(result["trace"])
    validation = pd.DataFrame(validations)
    if len(validation) != TRACE_RUNS or not validation.all_match.all():
        failed = validation[~validation.all_match].head().to_dict("records")
        raise AssertionError(f"authoritative reference parity failed: {failed}")
    return pd.DataFrame(traces), validation


def run_frame(
    design: pd.DataFrame,
    native: Mapping[str, np.ndarray],
    audited: Mapping[str, np.ndarray],
    long_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = design.drop(columns=["direction_code"]).copy()
    runs["stop_reason"] = [STOP_LABELS[int(value)] for value in native["stop"]]
    runs["completed"] = native["stop"] == STOP_COMPLETE
    runs["quiescent"] = native["stop"] == STOP_QUIESCENT
    runs["primary_active_censored"] = native["stop"] == STOP_EVENT_BUDGET
    runs["event_count"] = native["events"]
    for index, name in enumerate(LEDGER_NAMES):
        runs[f"ledger_{name}"] = native["ledger"][:, index]
    runs["ledger_proposals"] = runs.ledger_activations
    runs["ledger_displacedCells"] = 2 * runs.ledger_acceptedSwaps
    runs["full_ledger_unit_cost"] = (
        runs[[column for column in runs if column.startswith("ledger_")]].sum(axis=1)
    )
    runs["state_fingerprint_u64"] = [f"{int(value):016x}" for value in native["fingerprint"]]
    runs["opportunity_rate"] = np.divide(
        runs.ledger_acceptedSwaps,
        runs.event_count,
        out=np.zeros(len(runs), dtype=float),
        where=runs.event_count.to_numpy() > 0,
    )
    runs["opportunity_rate_bin"] = pd.cut(
        runs.opportunity_rate,
        bins=[-1, 0, 0.05, 0.15, 0.30, np.inf],
        labels=["zero_movement", "low_le_0.05", "mid_le_0.15", "high_le_0.30", "very_high_gt_0.30"],
    ).astype(str)
    runs.loc[runs.event_count.eq(0), "opportunity_rate_bin"] = "terminal_zero_events"
    for index, category in enumerate(("swap", "memory", "unchanged")):
        runs[f"shadow_requested_{category}"] = audited["requested"][:, index]
        runs[f"shadow_infeasible_{category}"] = audited["deficits"][:, index]
    runs["s11_shadow_rate_feasibility_deficit"] = audited["deficits"].sum(axis=1) > 0
    runs["s11_rate_feasibility_stratum"] = np.where(
        runs.s11_shadow_rate_feasibility_deficit,
        "native_path_shadow_deficit",
        "native_path_shadow_feasible",
    )
    long_map = long_frame.set_index("run_id").long_stop_reason.to_dict()
    runs["recurrent_active_status"] = "not_active_primary"
    primary_active = runs.primary_active_censored
    runs.loc[primary_active, "recurrent_active_status"] = "not_assessed_long_horizon"
    for run_id, status in long_map.items():
        label = {
            "event_budget": "recurrent_active_4x",
            "complete": "completed_by_4x",
            "quiescent": "quiescent_by_4x",
        }[status]
        runs.loc[runs.run_id.eq(run_id), "recurrent_active_status"] = label

    metric_parts = []
    for metric_index, metric in enumerate(METRIC_NAMES):
        part = runs[
            [
                "run_id", "scenario_block_id", "replicate_index", "n", "architecture",
                "policy_profile", "scheduler_profile", "batch_width", "direction",
                "initial_disorder_profile", "initial_inversion_fraction", "baseline_fault_count",
                "baseline_placement_profile", "intervention_type", "fault_mode", "fault_count",
                "fault_location_class", "event_budget", "stop_reason", "completed", "quiescent",
                "primary_active_censored", "event_count", "ledger_acceptedSwaps",
                "ledger_memoryUpdates", "full_ledger_unit_cost", "opportunity_rate_bin",
                "s11_shadow_rate_feasibility_deficit", "s11_rate_feasibility_stratum",
                "recurrent_active_status", "exact_structural_reachability", "trace_retained",
            ]
        ].copy()
        part["metric"] = metric
        part["start_metric_level"] = native["initial"][:, metric_index]
        part["final_metric_level"] = native["final"][:, metric_index]
        part["peak_metric_level"] = native["peak"][:, metric_index]
        part["initial_level_excursion"] = native["max_initial_excursion"][:, metric_index]
        part["running_min_episode_count"] = native["episode_count"][:, metric_index]
        part["recovered_episode_count"] = native["recovered_count"][:, metric_index]
        part["open_censored_episode"] = native["open_episode"][:, metric_index].astype(bool)
        part["any_detour_episode"] = native["episode_count"][:, metric_index] > 0
        part["max_episode_depth"] = native["max_depth"][:, metric_index]
        part["sum_recovered_episode_depth"] = native["sum_depth"][:, metric_index]
        part["max_episode_duration_opportunities"] = native["max_duration"][:, metric_index]
        part["sum_recovered_episode_duration_opportunities"] = native["sum_duration"][:, metric_index]
        part["open_episode_duration_opportunities"] = native["open_duration"][:, metric_index]
        part["total_episode_time_at_risk_opportunities"] = (
            native["sum_duration"][:, metric_index] + native["open_duration"][:, metric_index]
        )
        part["positive_run_count"] = native["positive_runs"][:, metric_index]
        part["worsening_accepted_proposal_count"] = native["worsening_actions"][:, metric_index]
        part["s10_strict_filter_exposed"] = native["worsening_actions"][:, metric_index] > 0
        for threshold_index, threshold in enumerate((0, 1, 2, 4)):
            part[f"worsening_proposals_gt_{threshold}"] = native["threshold_counts"][:, metric_index, threshold_index]
        metric_parts.append(part)
    metrics = pd.concat(metric_parts, ignore_index=True)
    return runs, metrics


def paired_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    outcomes = [
        "completed", "any_detour_episode", "max_episode_depth",
        "max_episode_duration_opportunities", "final_metric_level",
        "full_ledger_unit_cost", "s10_strict_filter_exposed",
    ]
    baseline = metrics[metrics.intervention_type.eq("baseline")].set_index(
        ["scenario_block_id", "metric"]
    )
    active = metrics[~metrics.intervention_type.eq("baseline")].copy()
    active_index = pd.MultiIndex.from_frame(active[["scenario_block_id", "metric"]])
    reference = baseline.loc[active_index]
    for outcome in outcomes:
        active[f"difference_{outcome}"] = (
            active[outcome].astype(float).to_numpy() - reference[outcome].astype(float).to_numpy()
        )
        active[f"baseline_{outcome}"] = reference[outcome].to_numpy()
    return active[
        [
            "scenario_block_id", "metric", "n", "policy_profile", "scheduler_profile",
            "direction", "initial_disorder_profile", "baseline_fault_count",
            "baseline_placement_profile", "intervention_type", "fault_count",
            "fault_location_class", "s11_rate_feasibility_stratum", "recurrent_active_status",
            *[f"baseline_{outcome}" for outcome in outcomes],
            *[f"difference_{outcome}" for outcome in outcomes],
        ]
    ].reset_index(drop=True)


def small_n_bridge() -> pd.DataFrame:
    metric = pq.read_table(S11 / "null_metric_results.parquet").to_pandas()
    rate = pq.read_table(S11 / "action_rate_matching.parquet").to_pandas()
    rate["rate_feasibility_deficit"] = (
        rate[[
            "infeasible_swap_request_count",
            "infeasible_memory_request_count",
            "infeasible_unchanged_request_count",
        ]].sum(axis=1) > 0
    )
    metric = metric.merge(
        rate[["null_run_id", "rate_feasibility_deficit"]],
        on="null_run_id",
        how="left",
        validate="many_to_one",
    )
    metric["rate_feasibility_stratum"] = np.where(
        metric.null_family.eq("rate_matched_random"),
        np.where(metric.rate_feasibility_deficit.fillna(False), "rate_deficit", "rate_feasible"),
        "not_rate_matched",
    )
    metric["any_detour"] = metric.worsening_event_count.gt(0)
    grouped = (
        metric.groupby(
            [
                "n", "policy_profile", "metric", "s09_exact_start_status",
                "s10_strict_filter_exposed", "rate_feasibility_stratum",
                "intervention_type", "s09_recurrent_active", "opportunity_rate_bin",
                "null_family",
            ],
            dropna=False,
            observed=True,
        )
        .agg(
            runs=("null_run_id", "size"),
            completed=("completed", "sum"),
            completion_rate=("completed", "mean"),
            detour_runs=("any_detour", "sum"),
            detour_rate=("any_detour", "mean"),
            mean_excursion=("observed_excursion", "mean"),
            mean_final_metric=("final_metric_level", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "evidence_layer", "exact_small_n_structural_bridge")
    return grouped


def factor_accounting(design: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "n", "policy_profile", "scheduler_profile", "direction",
        "initial_disorder_profile", "baseline_fault_count",
        "baseline_placement_profile", "intervention_type",
    ]
    rows = []
    for field in fields:
        counts = design.groupby(field, observed=True).size()
        for level, count in counts.items():
            rows.append({"factor": field, "level": str(level), "runs": int(count)})
    return pd.DataFrame(rows)


def main() -> int:
    if Path("/artifacts/research_steps/S13").exists():
        raise RuntimeError("S13 artifact exists before S12 execution")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    before_hashes = input_hashes()
    write_json(OUTPUT / "input_immutability_pre.json", {"researchStepId": "S12", "inputs": before_hashes})
    started = time.perf_counter()

    design, arrays = build_design()
    write_parquet(OUTPUT / "larger_n_design.parquet", design, "e03.s12.larger_n_design.v1")
    write_parquet(OUTPUT / "factor_accounting.parquet", factor_accounting(design), "e03.s12.factor_accounting.v1")
    write_json(
        OUTPUT / "design_summary.json",
        {
            "researchStepId": "S12",
            "contractSha256": sha256_file(CONTRACT),
            "scenarioBlocks": int(design.scenario_block_id.nunique()),
            "runs": len(design),
            "traceRetainedRuns": int(design.trace_retained.sum()),
            "factorialComplete": True,
            "outcomesOpenedAtWrite": False,
        },
    )

    native_started = time.perf_counter()
    native = run_kernel(arrays)
    native_seconds = time.perf_counter() - native_started
    audit_started = time.perf_counter()
    audited = run_kernel(
        arrays,
        audit=(
            native["ledger"][:, 6].astype(np.int32),
            native["ledger"][:, 5].astype(np.int32),
            native["events"].astype(np.int32),
        ),
    )
    audit_seconds = time.perf_counter() - audit_started
    verify_audit_immutability(native, audited)

    long_frame, long_indices = build_long_horizon(design, arrays, native)
    traces, parity = build_reference_traces(design, arrays, native)
    runs, metrics = run_frame(design, native, audited, long_frame)
    effects = paired_effects(metrics)
    bridge = small_n_bridge()

    write_parquet(OUTPUT / "larger_n_trajectories.parquet", runs, "e03.s12.larger_n_trajectories.v1")
    write_parquet(OUTPUT / "larger_n_metric_outcomes.parquet", metrics, "e03.s12.larger_n_metric_outcomes.v1")
    write_parquet(OUTPUT / "paired_context_effects.parquet", effects, "e03.s12.paired_context_effects.v1")
    write_parquet(OUTPUT / "long_horizon_sensitivity.parquet", long_frame, "e03.s12.long_horizon_sensitivity.v1")
    write_parquet(OUTPUT / "retained_metric_traces.parquet", traces, "e03.s12.retained_metric_traces.v1")
    write_parquet(OUTPUT / "reference_engine_parity.parquet", parity, "e03.s12.reference_engine_parity.v1")
    write_parquet(OUTPUT / "small_n_bridge_strata.parquet", bridge, "e03.s12.small_n_bridge_strata.v1")

    after_hashes = input_hashes()
    immutable = before_hashes == after_hashes
    write_json(
        OUTPUT / "input_immutability.json",
        {
            "researchStepId": "S12",
            "success": immutable,
            "inputs": [
                {"path": path, "sha256Before": digest, "sha256After": after_hashes.get(path), "unchanged": after_hashes.get(path) == digest}
                for path, digest in before_hashes.items()
            ],
        },
    )
    if not immutable:
        raise AssertionError("S12 execution input mutated")
    build = {
        "researchStepId": "S12",
        "success": True,
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "scenarioBlocks": EXPECTED_BLOCKS,
        "runs": EXPECTED_RUNS,
        "runMetricRows": len(metrics),
        "pairedEffectRows": len(effects),
        "traceRuns": int(design.trace_retained.sum()),
        "traceRows": len(traces),
        "referenceParityRows": len(parity),
        "longHorizonRuns": len(long_frame),
        "smallNBridgeRows": len(bridge),
        "nativeSeconds": native_seconds,
        "shadowAuditSeconds": audit_seconds,
        "totalSeconds": time.perf_counter() - started,
        "nativeChargedOpportunities": int(native["events"].sum()),
        "shadowAuditChargedOpportunities": int(audited["events"].sum()),
        "longHorizonEligible": int(len(long_indices)),
        "longHorizonPrefixMatches": int(long_frame.prefix_match.sum()),
        "threadEnvironment": {
            "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS", "1"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
        },
        "host": {"platform": platform.platform(), "python": sys.version},
    }
    write_json(OUTPUT / "build_summary.json", build)
    print(json.dumps(build, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
