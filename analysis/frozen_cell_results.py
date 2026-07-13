"""S11 passive/stuck frozen-cell replication and analysis.

The module enforces the S08 split contract, keeps the frozen-public comparator
loss-explicit, and executes the S03/S05 reference transition rules.  Large R
populations use a proof-equivalent summary path: terminal predicates are
re-evaluated after every accepted state change, while unchanged activations
reuse the prior nonterminal proof.  Selected ordinary digest replays validate
that optimization before release.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import platform
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
from scipy import stats
import statsmodels.api as sm

from historical_backend.legacy_adapter import verify_source_tree
from reference_simulator.engine import (
    evaluate_terminal,
    execute_serial_summary_activation,
    initial_state,
    run as reference_run,
)
from reference_simulator.model import LEDGER_FIELDS, Scenario, canonical_json_bytes, state_hash
from reference_simulator.scheduler import scheduled_actor, scheduled_side
from scenario_bank.core import ConditionSpec, derive_seed, materialize_scenario


REPOSITORY = Path(__file__).resolve().parents[1]
S08_DIR = Path("/artifacts/research_steps/S08")
S01_CLAIMS = Path("/artifacts/research_steps/S01/claim_registry.parquet")
S02_CHECKSUMS = Path("/artifacts/research_steps/S02/source_tree_checksums.sha256")
HISTORICAL_SOURCE = Path("/cache/e01_s02/historical-worktree")
HISTORICAL_PYTHON = Path("/cache/e01_s04/venv311/bin/python")
PREREGISTRATION = REPOSITORY / "analysis" / "s11_frozen_cell_preregistration.json"
OUTPUT_SCHEMA = "e01.s11.frozen_cell_run.v1"
R_PROFILE = "R-clean-room-reference-E01-v1"
C_PROFILE = "C-frozen-public-commit"
PAPER_SPLIT = "paper_scale"
CONFIRM_SPLIT = "confirmatory_holdout"
FORBIDDEN_SPLITS = {"exploratory", "policy_search_holdout"}
POLICIES = ("Bubble", "Insertion", "Selection")
FAULT_MODES = ("passive", "stuck")
PLACEMENTS = ("reference_without_replacement", "legacy_with_replacement")
PAPER_R_COUNT = 7_200
CONFIRM_R_COUNT = 36_000
PAPER_C_COUNT = 900
PRIMARY_FAMILY = 18
BOOTSTRAP_DRAWS = 10_000


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


def monotonicity_error(values: Sequence[int | float]) -> int:
    return sum(right < left for left, right in zip(values, values[1:]))


def _values(scenario: Scenario, occupancy: Sequence[str]) -> list[int | float]:
    cells = scenario.cell_map
    return [cells[cell_id].value for cell_id in occupancy]


def _fault_vectors(scenario: Scenario, initial: Sequence[str], final: Sequence[str]) -> dict[str, list[Any]]:
    identities = sorted(
        (cell.cell_id for cell in scenario.cells if cell.fault.value != "normal"),
        key=lambda value: int(value.rsplit("-", 1)[1]),
    )
    initial_position = {cell_id: index for index, cell_id in enumerate(initial)}
    final_position = {cell_id: index for index, cell_id in enumerate(final)}
    return {
        "fault_cell_ids": identities,
        "fault_values": [scenario.cell_map[cell_id].value for cell_id in identities],
        "fault_positions_initial": [initial_position[cell_id] for cell_id in identities],
        "fault_positions_final": [final_position[cell_id] for cell_id in identities],
    }


def _insertion_prefix_cache(scenario: Scenario, state: Any) -> tuple[int | None, list[int]]:
    """Return earliest invalid normal edge and prefix normal-edge counts."""
    cells = scenario.cell_map
    first_invalid = None
    cumulative = [0] * len(state.occupancy)
    for right in range(1, len(state.occupancy)):
        left_cell = cells[state.occupancy[right - 1]]
        right_cell = cells[state.occupancy[right]]
        normal_edge = left_cell.fault.value == "normal" and right_cell.fault.value == "normal"
        cumulative[right] = cumulative[right - 1] + int(normal_edge)
        if normal_edge and first_invalid is None and left_cell.value > right_cell.value:
            first_invalid = right
    return first_invalid, cumulative


def _execute_pure_cell_view_activation(
    scenario: Scenario,
    state: Any,
    positions: dict[str, int],
    insertion_cache: list[Any],
) -> bool:
    """Exact batch-width-one transition specialized for pure S11 cell-view runs."""
    if state.activation_count >= scenario.max_activations:
        state.terminal = "event_budget"
        return False
    event_index = state.activation_count
    actor_id, _, consumed = scheduled_actor(scenario, event_index, include_draws=False)
    state.stream_counters["actor_activation"] = state.stream_counters.get("actor_activation", 0) + consumed
    actor = scenario.cell_map[actor_id]
    position = positions[actor_id]
    reads = 0
    comparisons = 0
    outcome = "noop"
    target_position: int | None = None
    new_cursor: int | None = None

    if actor.fault.value != "normal":
        reads = 1
    elif actor.policy.value == "Bubble":
        side, _ = scheduled_side(scenario, event_index)
        state.stream_counters["bubble_side"] = state.stream_counters.get("bubble_side", 0) + 1
        target_position = position + (-1 if side == "left" else 1)
        if not 0 <= target_position < len(state.occupancy):
            reads = 1
            target_position = None
        else:
            reads, comparisons = 2, 1
            target = scenario.cell_map[state.occupancy[target_position]]
            inversion = actor.value < target.value if side == "left" else actor.value > target.value
            if inversion:
                outcome = "reject" if target.fault.value == "stuck" else "swap"
    elif actor.policy.value == "Insertion":
        if position == 0:
            reads = 1
        else:
            first_invalid, cumulative = insertion_cache
            if first_invalid is not None and first_invalid < position:
                reads = first_invalid + 2
                comparisons = cumulative[first_invalid]
            else:
                reads = position + 2
                comparisons = cumulative[position - 1] + 1
                target_position = position - 1
                target = scenario.cell_map[state.occupancy[target_position]]
                if actor.value < target.value:
                    outcome = "reject" if target.fault.value == "stuck" else "swap"
    else:
        cursor = state.selection_cursors[actor_id]
        if not 0 <= cursor < len(state.occupancy) or cursor == position:
            reads = 1
        else:
            reads = 2
            target_position = cursor
            target = scenario.cell_map[state.occupancy[cursor]]
            if target.fault.value == "stuck":
                outcome = "memory"
                new_cursor = cursor + 1
            elif target.value <= actor.value:
                comparisons = 1
                outcome = "memory"
                new_cursor = cursor + 1
            else:
                comparisons = 1
                outcome = "swap"

    ledger = state.ledger
    ledger["activations"] += 1
    ledger["observationReads"] += reads
    ledger["valueComparisons"] += comparisons
    ledger["proposals"] += 1
    changed = False
    if outcome == "noop":
        ledger["noOps"] += 1
    elif outcome == "reject":
        ledger["rejections"] += 1
    elif outcome == "memory":
        ledger["memoryUpdates"] += 1
        state.selection_cursors[actor_id] = new_cursor
        changed = True
    else:
        assert target_position is not None
        target_id = state.occupancy[target_position]
        state.occupancy[position], state.occupancy[target_position] = target_id, actor_id
        positions[actor_id], positions[target_id] = target_position, position
        ledger["acceptedSwaps"] += 1
        ledger["displacedCells"] += 2
        changed = True
        if actor.policy.value == "Insertion":
            insertion_cache[:] = _insertion_prefix_cache(scenario, state)
    state.activation_count += 1
    if state.activation_count >= scenario.max_activations:
        state.terminal = "event_budget"
    return changed


def run_reference_summary(scenario: Scenario) -> dict[str, Any]:
    """Run exact R transitions while avoiding redundant terminal scans.

    A complete/quiescent/invariant predicate cannot change after a NoOp or a
    rejected serial proposal.  The summary path therefore reevaluates the
    ordered terminal predicate after every accepted swap/memory update and
    relies on the event-budget check for unchanged activations.
    """
    scenario.validate()
    state = initial_state(scenario)
    initial_occupancy = list(state.occupancy)
    initial_values = _values(scenario, state.occupancy)
    initial_hash = state_hash(scenario.scenario_id, state)
    state.terminal = evaluate_terminal(scenario, state)
    positions = {cell_id: index for index, cell_id in enumerate(state.occupancy)}
    insertion_cache: list[Any] = list(_insertion_prefix_cache(scenario, state))
    start = time.perf_counter()
    while state.terminal is None:
        if scenario.architecture.value == "cell_view":
            changed = _execute_pure_cell_view_activation(
                scenario, state, positions, insertion_cache
            )
        else:
            changed = execute_serial_summary_activation(scenario, state)
        if changed:
            state.terminal = evaluate_terminal(scenario, state)
    elapsed = time.perf_counter() - start
    final_values = _values(scenario, state.occupancy)
    ledger = dict(state.ledger)
    target_calculations = ledger["proposals"]
    rejected_actions = ledger["rejections"] + ledger["conflictLosses"]
    publication_cost = ledger["acceptedSwaps"] + ledger["valueComparisons"]
    unit_weight = (
        ledger["activations"]
        + ledger["observationReads"]
        + ledger["valueComparisons"]
        + target_calculations
        + ledger["proposals"]
        + ledger["noOps"]
        + ledger["rejections"]
        + ledger["memoryUpdates"]
        + ledger["acceptedSwaps"]
        + ledger["displacedCells"]
        + ledger["conflictLosses"]
    )
    result: dict[str, Any] = {
        "backend_profile": R_PROFILE,
        "publication_snapshot_claimed": False,
        "evidence_layer": "clean_room_reference",
        "scenario_id": scenario.scenario_id,
        "architecture": scenario.architecture.value,
        "policy": scenario.traditional_policy.value if scenario.traditional_policy else scenario.cells[0].policy.value,
        "scheduler": scenario.scheduler,
        "sequence_basis": "activation",
        "historical_random_stream_status": "not_applicable_reference_counter_addressed",
        "stop_reason": state.terminal,
        "completed": state.terminal == "complete",
        "censored": state.terminal in {"event_budget", "invariant_error"},
        "timed_out": False,
        "activation_count": state.activation_count,
        "monotonicity_error": monotonicity_error(final_values),
        "final_nonstrictly_sorted": all(a <= b for a, b in zip(final_values, final_values[1:])),
        "value_multiset_conserved": sorted(initial_values) == sorted(final_values),
        "initial_values_sha256": canonical_hash(initial_values),
        "final_values_sha256": canonical_hash(final_values),
        "initial_state_hash": initial_hash,
        "final_state_hash": state_hash(scenario.scenario_id, state),
        "observable_final_hash": canonical_hash(final_values),
        "trace_sha256": None,
        "elapsed_seconds": elapsed,
        "adapter_safety_timeout_seconds": None,
        "thread_shutdown_clean": None,
        "final_values": final_values,
        "swap_only_cost": ledger["acceptedSwaps"],
        "publication_swap_plus_comparison_cost": publication_cost,
        "target_calculations": target_calculations,
        "rejected_actions": rejected_actions,
        "unit_weight_full_ledger": unit_weight,
        "historical_compare_and_swap_probe": None,
        "historical_frozen_swap_attempts": None,
        "cost_profile": "complete_reference_S03_plus_S10_fault_extension",
    }
    for field in LEDGER_FIELDS:
        result[f"ledger_{field}"] = ledger[field]
    result.update(_fault_vectors(scenario, initial_occupancy, state.occupancy))
    return result


def _identity_fields(task: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    row = task["scenario_row"]
    condition = task["condition"]
    return {
        "schema_version": OUTPUT_SCHEMA,
        "research_step_id": "S11",
        "condition_id": row["conditionId"],
        "base_draw_id": row["baseDrawId"],
        "pairing_block_id": row["pairingBlockId"],
        "split": row["split"],
        "protected": bool(row["protected"]),
        "replicate_ordinal": int(row["replicateOrdinal"]),
        "profile_role": condition["profileRole"],
        "placement_profile": row["placementProfile"],
        "fault_mode": row["faultMode"],
        "requested_fault_count": int(row["requestedFaultCount"]),
        "realized_fault_count": int(row["realizedFaultCount"]),
        "fault_draw_indices": [int(value) for value in row["faultDrawIndices"]],
        "fault_distinct_indices": [int(value) for value in row["faultDistinctIndices"]],
        "fault_map_id": row["faultMapId"],
        "runtime_seed": row["runtimeSeed"],
        "generation_key": row["generationKey"],
        "event_budget": int(row["maxActivations"]),
        "historical_eligibility": condition["historicalEligibility"],
        "scenario_json_sha256": metadata["scenarioJsonSha256"] if metadata else row["scenarioJsonSha256"],
    }


def _reference_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    condition = condition_from_dict(task["condition"])
    scenario, metadata = materialize_scenario(condition, task["base"])
    if scenario.scenario_id != task["scenario_row"]["scenarioId"]:
        raise ValueError("S08 scenario identity failed exact materialization")
    result = run_reference_summary(scenario)
    result.update(_identity_fields(task, metadata))
    result["run_id"] = "s11r:" + hashlib.sha256(f"{R_PROFILE}|{scenario.scenario_id}".encode()).hexdigest()
    return result


def preregistration_sha256() -> str:
    return sha256_file(PREREGISTRATION)


def _catalog() -> dict[str, dict[str, Any]]:
    rows = pq.read_table(
        S08_DIR / "condition_catalog.parquet", filters=[("family", "=", "unique_fault")]
    ).to_pylist()
    return {row["conditionId"]: json.loads(row["conditionJson"]) for row in rows}


def load_tasks(split: str, backend: str, *, preregistration_hash: str | None = None) -> list[dict[str, Any]]:
    if split in FORBIDDEN_SPLITS or split not in {PAPER_SPLIT, CONFIRM_SPLIT}:
        raise PermissionError(f"S11 may not access split {split!r}")
    actual_prereg = preregistration_sha256()
    if split == CONFIRM_SPLIT and preregistration_hash != actual_prereg:
        raise PermissionError("confirmatory S11 access requires the exact frozen preregistration hash")

    filters: list[tuple[str, str, Any]] = [
        ("conditionFamily", "=", "unique_fault"),
        ("split", "=", split),
    ]
    if split == CONFIRM_SPLIT:
        filters.append(("placementProfile", "=", "reference_without_replacement"))
    rows = pq.read_table(S08_DIR / "paired_scenario_bank.parquet", filters=filters).to_pylist()
    catalog = _catalog()
    if backend == "historical":
        rows = [
            row
            for row in rows
            if catalog[row["conditionId"]]["historicalEligibility"]
            == "C_adapter_endpoint_supported_with_bank_seed_override"
        ]
    elif backend != "reference":
        raise ValueError(f"unknown backend {backend!r}")

    expected = {
        (PAPER_SPLIT, "reference"): PAPER_R_COUNT,
        (CONFIRM_SPLIT, "reference"): CONFIRM_R_COUNT,
        (PAPER_SPLIT, "historical"): PAPER_C_COUNT,
    }.get((split, backend))
    if expected is None:
        raise PermissionError(f"backend/split population is not declared: {backend}/{split}")
    if len(rows) != expected:
        raise ValueError(f"expected {expected} {backend}/{split} rows, found {len(rows)}")
    if split == CONFIRM_SPLIT and not all(bool(row["protected"]) for row in rows):
        raise ValueError("confirmatory rows must all be protected")
    if split == PAPER_SPLIT and any(bool(row["protected"]) for row in rows):
        raise ValueError("paper-scale rows must be unprotected")

    base_rows = pq.read_table(
        S08_DIR / "base_draw_bank.parquet",
        filters=[("inputProfile", "=", "unique_1_100"), ("split", "=", split)],
    ).to_pylist()
    bases = {row["baseDrawId"]: row for row in base_rows}
    tasks = []
    for row in sorted(rows, key=lambda value: (value["conditionId"], value["replicateOrdinal"])):
        condition = catalog[row["conditionId"]]
        if condition["faultMode"] not in FAULT_MODES or condition["policies"][0] not in POLICIES:
            raise ValueError("task escaped frozen fault/policy profiles")
        if int(condition["requestedFaultCount"]) not in {1, 2, 3}:
            raise ValueError("task escaped frozen f=1..3 profiles")
        tasks.append({"scenario_row": row, "condition": condition, "base": bases[row["baseDrawId"]]})
    return tasks


def _read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["run_id"]] = row
    return rows


def run_reference_population(tasks: Sequence[Mapping[str, Any]], checkpoint: Path, *, workers: int = 8) -> list[dict[str, Any]]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1,8]")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_checkpoint(checkpoint)
    expected_by_scenario = {task["scenario_row"]["scenarioId"] for task in tasks}
    done_scenarios = {row["scenario_id"] for row in completed.values()}
    pending = [task for task in tasks if task["scenario_row"]["scenarioId"] not in done_scenarios]
    with checkpoint.open("a", encoding="utf-8") as handle:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_reference_worker, task): task for task in pending}
            for future in as_completed(futures):
                row = future.result()
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                completed[row["run_id"]] = row
                if len(completed) % 250 == 0 or len(completed) == len(tasks):
                    print(f"S11 reference progress {len(completed)}/{len(tasks)}", file=sys.stderr, flush=True)
    if {row["scenario_id"] for row in completed.values()} != expected_by_scenario:
        raise ValueError("reference checkpoint run accounting mismatch")
    return sorted(completed.values(), key=lambda row: row["run_id"])


def _historical_payload(task: Mapping[str, Any], timeout_seconds: float) -> dict[str, Any]:
    base = task["base"]
    occupancy = [int(value) for value in base["initialOccupancyIndices"]]
    values_by_id = [int(value) for value in base["valuesById"]]
    values = [values_by_id[index] for index in occupancy]
    fault_ids = [int(value) for value in task["scenario_row"]["faultDistinctIndices"]]
    fault_positions = [occupancy.index(index) for index in fault_ids]
    return {
        "source_root": str(HISTORICAL_SOURCE),
        "checksum_manifest": str(S02_CHECKSUMS),
        "values": values,
        "policy": task["condition"]["policies"][0].lower(),
        "seed": int(task["scenario_row"]["runtimeSeed"]),
        "timeout_seconds": timeout_seconds,
        "frozen_positions": fault_positions,
        "fault_cell_ids": [f"cell-{index:04d}" for index in fault_ids],
        "fault_values": [values_by_id[index] for index in fault_ids],
    }


def _historical_subprocess(task: Mapping[str, Any], timeout_seconds: float) -> dict[str, Any]:
    payload = _historical_payload(task, timeout_seconds)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [str(HISTORICAL_PYTHON), str(REPOSITORY / "scripts" / "s11_historical_worker.py")]
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
        raise RuntimeError(f"historical S11 worker failed ({process.returncode}): {process.stderr[-4000:]}")
    result = json.loads(process.stdout)
    result.update(_identity_fields(task))
    result["scenario_id"] = task["scenario_row"]["scenarioId"]
    result["architecture"] = "cell_view"
    result["policy"] = task["condition"]["policies"][0]
    result["run_id"] = "s11c:" + hashlib.sha256(f"{C_PROFILE}|{result['scenario_id']}".encode()).hexdigest()
    return result


def run_historical_population(
    tasks: Sequence[Mapping[str, Any]],
    checkpoint: Path,
    *,
    workers: int = 2,
    timeout_seconds: float = 120.0,
) -> list[dict[str, Any]]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1,8]")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_checkpoint(checkpoint)
    expected_by_scenario = {task["scenario_row"]["scenarioId"] for task in tasks}
    done_scenarios = {row["scenario_id"] for row in completed.values()}
    pending = [task for task in tasks if task["scenario_row"]["scenarioId"] not in done_scenarios]
    with checkpoint.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_historical_subprocess, task, timeout_seconds): task
                for task in pending
            }
            for future in as_completed(futures):
                row = future.result()
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                completed[row["run_id"]] = row
                if len(completed) % 50 == 0 or len(completed) == len(tasks):
                    print(f"S11 historical progress {len(completed)}/{len(tasks)}", file=sys.stderr, flush=True)
    if {row["scenario_id"] for row in completed.values()} != expected_by_scenario:
        raise ValueError("historical checkpoint run accounting mismatch")
    return sorted(completed.values(), key=lambda row: row["run_id"])


def _replay_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    condition = condition_from_dict(task["condition"])
    scenario, _ = materialize_scenario(condition, task["base"])
    first = reference_run(scenario, trace_mode="digest")
    second = reference_run(scenario, trace_mode="digest")
    fast = run_reference_summary(scenario)
    ledger_match = all(
        fast[f"ledger_{field}"] == first.summary["ledger"][field] for field in LEDGER_FIELDS
    )
    return {
        "scenarioId": scenario.scenario_id,
        "conditionId": task["scenario_row"]["conditionId"],
        "architecture": task["condition"]["architecture"],
        "policy": task["condition"]["policies"][0],
        "faultMode": task["condition"]["faultMode"],
        "requestedFaultCount": task["condition"]["requestedFaultCount"],
        "placementProfile": task["condition"]["placementProfile"],
        "byteExactDigestReplay": first.to_json_bytes() == second.to_json_bytes(),
        "fastSummaryMatchesDigest": (
            fast["final_state_hash"] == first.final_state_hash
            and fast["activation_count"] == first.summary["activationCount"]
            and fast["stop_reason"] == first.summary["stopReason"]
            and ledger_match
        ),
        "eventDigest": first.event_digest,
        "finalStateHash": first.final_state_hash,
        "activationCount": first.summary["activationCount"],
        "monotonicityError": monotonicity_error(first.summary["finalValues"]),
    }


def run_exact_replay_samples(paper_tasks: Sequence[Mapping[str, Any]], *, workers: int = 8) -> list[dict[str, Any]]:
    corrected = [
        task
        for task in paper_tasks
        if task["condition"]["placementProfile"] == "reference_without_replacement"
    ]
    by_condition: dict[str, list[Mapping[str, Any]]] = {}
    for task in corrected:
        by_condition.setdefault(task["scenario_row"]["conditionId"], []).append(task)
    selected = [
        min(tasks, key=lambda item: item["scenario_row"]["scenarioId"])
        for tasks in by_condition.values()
    ]
    if len(selected) != 36:
        raise ValueError(f"expected one replay sample for 36 corrected profiles, found {len(selected)}")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_replay_worker, selected))
    return sorted(results, key=lambda row: row["conditionId"])


def _bootstrap_mean(values: np.ndarray, seed: int, confidence: float = 0.95) -> dict[str, float]:
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    estimates = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for start in range(0, BOOTSTRAP_DRAWS, 250):
        count = min(250, BOOTSTRAP_DRAWS - start)
        indexes = generator.integers(0, len(values), size=(count, len(values)))
        estimates[start : start + count] = values[indexes].mean(axis=1)
    alpha = (1 - confidence) / 2
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1 - alpha)),
    }


def _mean_t_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    mean = float(values.mean())
    if len(values) < 2 or float(values.std(ddof=1)) == 0:
        return mean, mean
    half = float(stats.t.ppf((1 + confidence) / 2, len(values) - 1)) * float(values.std(ddof=1)) / math.sqrt(len(values))
    return mean - half, mean + half


def summarize_runs(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "backend_profile", "split", "placement_profile", "fault_mode", "policy",
        "architecture", "requested_fault_count", "realized_fault_count",
    ]
    rows = []
    for values, group in frame.groupby(keys, dropna=False, sort=True):
        error = group.monotonicity_error.to_numpy(dtype=float)
        ci_low, ci_high = _mean_t_interval(error)
        rows.append(
            dict(zip(keys, values))
            | {
                "n": len(group),
                "mean_monotonicity_error": float(error.mean()),
                "sd_monotonicity_error": float(error.std(ddof=1)) if len(error) > 1 else 0.0,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "completion_rate": float(group.completed.mean()),
                "censored_count": int(group.censored.sum()),
                "mean_swap_only_cost": float(group.swap_only_cost.mean()),
                "mean_publication_swap_plus_comparison_cost": float(group.publication_swap_plus_comparison_cost.mean()),
                "mean_wall_time_seconds": float(group.elapsed_seconds.mean()),
            }
        )
    return pd.DataFrame(rows)


def primary_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[
        frame.backend_profile.eq(R_PROFILE)
        & frame.split.eq(CONFIRM_SPLIT)
        & frame.placement_profile.eq("reference_without_replacement")
    ]
    rows = []
    adjusted = 1 - 0.05 / PRIMARY_FAMILY
    for (policy, fault, requested), group in selected.groupby(
        ["policy", "fault_mode", "requested_fault_count"], sort=True
    ):
        wide = group.pivot(index="pairing_block_id", columns="architecture", values="monotonicity_error")
        if list(sorted(wide.columns)) != ["cell_view", "traditional"] or len(wide) != 1000 or wide.isna().any().any():
            raise ValueError(f"incomplete primary pairing for {policy}/{fault}/f={requested}")
        difference = (wide.cell_view - wide.traditional).to_numpy(dtype=float)
        seed = derive_seed("S11_primary_bootstrap", policy, fault, int(requested))
        nominal = _bootstrap_mean(difference, seed, 0.95)
        adjusted_ci = _bootstrap_mean(difference, seed, adjusted)
        if adjusted_ci["ci_high"] < 0:
            classification = "supportive"
        elif adjusted_ci["ci_low"] > 0:
            classification = "contradictory"
        else:
            classification = "inconclusive"
        rows.append(
            {
                "policy": policy,
                "fault_mode": fault,
                "requested_fault_count": int(requested),
                "placement_profile": "reference_without_replacement",
                "split": CONFIRM_SPLIT,
                "pairs": len(difference),
                "mean_cell_view_minus_traditional": float(difference.mean()),
                "median_difference": float(np.median(difference)),
                "paired_sd": float(difference.std(ddof=1)),
                "paired_dz": float(difference.mean() / difference.std(ddof=1)) if difference.std(ddof=1) else np.nan,
                "ci95_low": nominal["ci_low"],
                "ci95_high": nominal["ci_high"],
                "adjusted_confidence": adjusted,
                "adjusted_ci_low": adjusted_ci["ci_low"],
                "adjusted_ci_high": adjusted_ci["ci_high"],
                "claim_classification": classification,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != PRIMARY_FAMILY:
        raise ValueError(f"expected {PRIMARY_FAMILY} primary contrasts, found {len(result)}")
    return result


def sampling_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame[frame.backend_profile.eq(R_PROFILE) & frame.split.eq(PAPER_SPLIT)]
    rows = []
    for keys, group in source.groupby(
        ["architecture", "policy", "fault_mode", "requested_fault_count"], sort=True
    ):
        wide = group.pivot(index="pairing_block_id", columns="placement_profile", values="monotonicity_error")
        if list(sorted(wide.columns)) != ["legacy_with_replacement", "reference_without_replacement"] or len(wide) != 100:
            raise ValueError(f"incomplete placement pairing for {keys}")
        difference = (wide.legacy_with_replacement - wide.reference_without_replacement).to_numpy(dtype=float)
        seed = derive_seed("S11_sampling_bootstrap", *keys)
        ci = _bootstrap_mean(difference, seed)
        rows.append(
            {
                "architecture": keys[0],
                "policy": keys[1],
                "fault_mode": keys[2],
                "requested_fault_count": int(keys[3]),
                "pairs": len(difference),
                "mean_legacy_minus_corrected_error": float(difference.mean()),
                "ci95_low": ci["ci_low"],
                "ci95_high": ci["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def cost_effects(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame[
        frame.backend_profile.eq(R_PROFILE)
        & frame.split.eq(CONFIRM_SPLIT)
        & frame.placement_profile.eq("reference_without_replacement")
    ]
    metrics = (
        "swap_only_cost",
        "publication_swap_plus_comparison_cost",
        "ledger_observationReads",
        "ledger_valueComparisons",
        "target_calculations",
        "rejected_actions",
        "ledger_noOps",
        "ledger_memoryUpdates",
        "unit_weight_full_ledger",
    )
    rows = []
    for keys, group in source.groupby(["policy", "fault_mode", "requested_fault_count"], sort=True):
        for metric in metrics:
            wide = group.pivot(index="pairing_block_id", columns="architecture", values=metric)
            difference = (wide.cell_view - wide.traditional).to_numpy(dtype=float)
            low, high = _mean_t_interval(difference)
            rows.append(
                {
                    "policy": keys[0],
                    "fault_mode": keys[1],
                    "requested_fault_count": int(keys[2]),
                    "metric": metric,
                    "pairs": len(difference),
                    "mean_cell_view": float(wide.cell_view.mean()),
                    "mean_traditional": float(wide.traditional.mean()),
                    "mean_difference": float(difference.mean()),
                    "median_difference": float(np.median(difference)),
                    "paired_ci95_low": low,
                    "paired_ci95_high": high,
                    "ratio_of_means": float(wide.cell_view.mean() / wide.traditional.mean()) if float(wide.traditional.mean()) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def reported_mean_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    claims = pd.read_parquet(S01_CLAIMS)
    claims = claims[(claims.figure == 5) & claims.claim_kind.eq("panel_mean")].copy()
    rows = []
    profiles = [
        (R_PROFILE, "reference_without_replacement"),
        (R_PROFILE, "legacy_with_replacement"),
        (C_PROFILE, "legacy_with_replacement"),
    ]
    for claim in claims.to_dict(orient="records"):
        for backend, placement in profiles:
            selected = frame[
                frame.backend_profile.eq(backend)
                & frame.split.eq(PAPER_SPLIT)
                & frame.placement_profile.eq(placement)
                & frame.fault_mode.eq(claim["fault_type"])
                & frame.policy.eq(claim["algorithm"])
                & frame.architecture.eq(claim["architecture"])
                & frame.requested_fault_count.eq(int(claim["fault_count"]))
            ]
            base = {
                "claim_id": claim["claim_id"],
                "backend_profile": backend,
                "placement_profile": placement,
                "fault_mode": claim["fault_type"],
                "policy": claim["algorithm"],
                "architecture": claim["architecture"],
                "requested_fault_count": int(claim["fault_count"]),
                "paper_reported_mean": float(claim["reported_mean"]),
                "equivalence_margin_errors": 1.0,
            }
            if selected.empty:
                rows.append(base | {"n": 0, "simulated_mean": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "classification": "not_testable_backend_unavailable"})
                continue
            values = selected.monotonicity_error.to_numpy(dtype=float)
            seed = derive_seed("S11_paper_mean", backend, placement, claim["claim_id"])
            ci = _bootstrap_mean(values, seed)
            target = float(claim["reported_mean"])
            if ci["ci_low"] >= target - 1 and ci["ci_high"] <= target + 1:
                classification = "reproduced_within_frozen_margin"
            elif ci["ci_low"] <= target <= ci["ci_high"]:
                classification = "approximately_reproduced"
            else:
                classification = "not_reproduced"
            rows.append(base | {"n": len(values), "simulated_mean": float(values.mean()), "ci95_low": ci["ci_low"], "ci95_high": ci["ci_high"], "classification": classification})
    return pd.DataFrame(rows)


def fault_count_accounting(frame: pd.DataFrame) -> pd.DataFrame:
    unique = frame.drop_duplicates(
        ["backend_profile", "split", "placement_profile", "fault_map_id", "requested_fault_count"]
    )
    return (
        unique.groupby(
            ["backend_profile", "split", "placement_profile", "requested_fault_count", "realized_fault_count"],
            sort=True,
        )
        .size()
        .rename("unique_fault_maps")
        .reset_index()
    )


def fault_position_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run in frame.to_dict(orient="records"):
        for ordinal, (cell_id, value, initial, final) in enumerate(
            zip(
                run["fault_cell_ids"],
                run["fault_values"],
                run["fault_positions_initial"],
                run["fault_positions_final"],
            )
        ):
            rows.append(
                {
                    "run_id": run["run_id"],
                    "backend_profile": run["backend_profile"],
                    "split": run["split"],
                    "placement_profile": run["placement_profile"],
                    "architecture": run["architecture"],
                    "policy": run["policy"],
                    "fault_mode": run["fault_mode"],
                    "requested_fault_count": run["requested_fault_count"],
                    "realized_fault_count": run["realized_fault_count"],
                    "fault_ordinal": ordinal,
                    "fault_cell_id": cell_id,
                    "fault_value": value,
                    "initial_position": initial,
                    "final_position": final,
                    "initial_position_decile": min(int(initial) // 10, 9),
                    "monotonicity_error": run["monotonicity_error"],
                    "completed": run["completed"],
                }
            )
    return pd.DataFrame(rows)


def fault_position_dependence(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = frame[
        frame.backend_profile.eq(R_PROFILE)
        & frame.placement_profile.eq("reference_without_replacement")
        & frame.split.isin([PAPER_SPLIT, CONFIRM_SPLIT])
    ].copy()
    features = []
    for row in source.to_dict(orient="records"):
        positions = np.asarray(row["fault_positions_initial"], dtype=float)
        values = np.asarray(row["fault_values"], dtype=float)
        features.append(
            {
                "run_id": row["run_id"],
                "architecture": row["architecture"],
                "policy": row["policy"],
                "fault_mode": row["fault_mode"],
                "monotonicity_error": row["monotonicity_error"],
                "realized_fault_count": row["realized_fault_count"],
                "mean_initial_position_norm": float(positions.mean() / 99),
                "minimum_edge_distance_norm": float(np.minimum(positions, 99 - positions).min() / 49.5),
                "position_span_norm": float((positions.max() - positions.min()) / 99) if len(positions) > 1 else 0.0,
                "mean_fault_value_norm": float(((values - 1) / 99).mean()),
            }
        )
    design = pd.DataFrame(features)
    rows = []
    predictors = [
        "realized_fault_count",
        "mean_initial_position_norm",
        "minimum_edge_distance_norm",
        "position_span_norm",
        "mean_fault_value_norm",
    ]
    for keys, group in design.groupby(["architecture", "policy", "fault_mode"], sort=True):
        x = sm.add_constant(group[predictors].astype(float), has_constant="add")
        fit = sm.OLS(group.monotonicity_error.astype(float), x).fit(cov_type="HC3")
        for term in x.columns:
            rows.append(
                {
                    "architecture": keys[0],
                    "policy": keys[1],
                    "fault_mode": keys[2],
                    "n": len(group),
                    "term": term,
                    "coefficient": float(fit.params[term]),
                    "hc3_standard_error": float(fit.bse[term]),
                    "ci95_low": float(fit.conf_int().loc[term, 0]),
                    "ci95_high": float(fit.conf_int().loc[term, 1]),
                    "p_value_descriptive": float(fit.pvalues[term]),
                    "r_squared": float(fit.rsquared),
                }
            )
    long = fault_position_table(source)
    deciles = (
        long.groupby(
            ["architecture", "policy", "fault_mode", "initial_position_decile"], sort=True
        )
        .agg(
            fault_instances=("run_id", "size"),
            mean_monotonicity_error=("monotonicity_error", "mean"),
            completion_rate=("completed", "mean"),
        )
        .reset_index()
    )
    return pd.DataFrame(rows), deciles


def stop_reason_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            ["backend_profile", "split", "placement_profile", "architecture", "policy", "fault_mode", "requested_fault_count", "stop_reason", "completed", "censored"],
            dropna=False,
            sort=True,
        )
        .size()
        .rename("runs")
        .reset_index()
    )


def architecture_claim_table(primary: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in primary.to_dict(orient="records"):
        rows.append(
            row
            | {
                "backend_profile": R_PROFILE,
                "comparison_availability": "paired_confirmatory",
                "paper_blanket_claim": "cell_view_lower_error",
            }
        )
    legacy = frame[
        frame.backend_profile.eq(R_PROFILE)
        & frame.split.eq(PAPER_SPLIT)
        & frame.placement_profile.eq("legacy_with_replacement")
    ]
    for keys, group in legacy.groupby(["policy", "fault_mode", "requested_fault_count"]):
        wide = group.pivot(index="pairing_block_id", columns="architecture", values="monotonicity_error")
        diff = (wide.cell_view - wide.traditional).to_numpy(dtype=float)
        low, high = _mean_t_interval(diff)
        classification = "supportive" if high < 0 else "contradictory" if low > 0 else "inconclusive"
        rows.append(
            {
                "policy": keys[0], "fault_mode": keys[1], "requested_fault_count": int(keys[2]),
                "placement_profile": "legacy_with_replacement", "split": PAPER_SPLIT,
                "pairs": len(diff), "mean_cell_view_minus_traditional": float(diff.mean()),
                "median_difference": float(np.median(diff)), "paired_sd": float(diff.std(ddof=1)),
                "paired_dz": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) else np.nan,
                "ci95_low": low, "ci95_high": high, "adjusted_confidence": np.nan,
                "adjusted_ci_low": np.nan, "adjusted_ci_high": np.nan,
                "claim_classification": classification, "backend_profile": R_PROFILE,
                "comparison_availability": "paired_paper_scale_sensitivity",
                "paper_blanket_claim": "cell_view_lower_error",
            }
        )
    for policy in POLICIES:
        for fault in FAULT_MODES:
            for requested in (1, 2, 3):
                rows.append(
                    {
                        "policy": policy, "fault_mode": fault, "requested_fault_count": requested,
                        "placement_profile": "legacy_with_replacement", "split": PAPER_SPLIT,
                        "pairs": 0, "mean_cell_view_minus_traditional": np.nan,
                        "median_difference": np.nan, "paired_sd": np.nan, "paired_dz": np.nan,
                        "ci95_low": np.nan, "ci95_high": np.nan, "adjusted_confidence": np.nan,
                        "adjusted_ci_low": np.nan, "adjusted_ci_high": np.nan,
                        "claim_classification": "not_testable_historical_traditional_unavailable",
                        "backend_profile": C_PROFILE, "comparison_availability": "unavailable",
                        "paper_blanket_claim": "cell_view_lower_error",
                    }
                )
    return pd.DataFrame(rows)


def rank_claims(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (backend, placement, split, fault), group in frame[
        frame.architecture.eq("cell_view")
    ].groupby(["backend_profile", "placement_profile", "split", "fault_mode"], sort=True):
        means = group.groupby("policy").monotonicity_error.mean().to_dict()
        if set(means) != set(POLICIES):
            continue
        if fault == "passive":
            passed = means["Bubble"] < means["Insertion"] < means["Selection"]
            claim = "F05-B-PASSIVE-RANK"
            rule = "Bubble < Insertion < Selection"
        else:
            passed = means["Selection"] < means["Bubble"] and means["Selection"] < means["Insertion"]
            claim = "F05-C-STUCK-RANK"
            rule = "Selection < Bubble and Selection < Insertion"
        rows.append(
            {
                "claim_id": claim,
                "backend_profile": backend,
                "placement_profile": placement,
                "split": split,
                "fault_mode": fault,
                "mean_bubble": means["Bubble"],
                "mean_insertion": means["Insertion"],
                "mean_selection": means["Selection"],
                "frozen_rule": rule,
                "classification": "supportive" if passed else "contradictory",
            }
        )
    return pd.DataFrame(rows)


def draw_figure5(frame: pd.DataFrame, output_dir: Path) -> None:
    paper = pd.read_parquet(S01_CLAIMS)
    paper = paper[(paper.figure == 5) & paper.claim_kind.eq("panel_mean")]
    source = frame[
        frame.backend_profile.eq(R_PROFILE)
        & frame.split.eq(PAPER_SPLIT)
        & frame.placement_profile.eq("reference_without_replacement")
    ]
    legacy = frame[
        frame.backend_profile.eq(R_PROFILE)
        & frame.split.eq(PAPER_SPLIT)
        & frame.placement_profile.eq("legacy_with_replacement")
    ]
    historical = frame[frame.backend_profile.eq(C_PROFILE)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharey="row")
    colors = {"traditional": "#3A86B8", "cell_view": "#E05A7A"}
    for row_index, fault in enumerate(FAULT_MODES):
        for column, policy in enumerate(POLICIES):
            axis = axes[row_index, column]
            x = np.arange(1, 4)
            for offset, architecture in ((-0.18, "traditional"), (0.18, "cell_view")):
                means, errors = [], []
                for requested in (1, 2, 3):
                    values = source[
                        source.fault_mode.eq(fault)
                        & source.policy.eq(policy)
                        & source.architecture.eq(architecture)
                        & source.requested_fault_count.eq(requested)
                    ].monotonicity_error.to_numpy(dtype=float)
                    low, high = _mean_t_interval(values)
                    means.append(values.mean())
                    errors.append([values.mean() - low, high - values.mean()])
                axis.bar(x + offset, means, width=0.34, color=colors[architecture], alpha=0.82, label=architecture.replace("_", " ") if row_index == 0 and column == 0 else None)
                axis.errorbar(x + offset, means, yerr=np.asarray(errors).T, fmt="none", ecolor="#222222", capsize=2, linewidth=0.9)
                reported = paper[
                    paper.fault_type.eq(fault)
                    & paper.algorithm.eq(policy)
                    & paper.architecture.eq(architecture)
                ].sort_values("fault_count")
                axis.scatter(x + offset, reported.reported_mean, marker="x", s=35, color="#111111", zorder=4, label="paper panel mean" if row_index == 0 and column == 0 and architecture == "traditional" else None)
                legacy_means = [
                    legacy[
                        legacy.fault_mode.eq(fault)
                        & legacy.policy.eq(policy)
                        & legacy.architecture.eq(architecture)
                        & legacy.requested_fault_count.eq(requested)
                    ].monotonicity_error.mean()
                    for requested in (1, 2, 3)
                ]
                axis.plot(x + offset, legacy_means, color=colors[architecture], linestyle=":", marker=".", linewidth=1)
            if fault == "passive":
                c_means = [
                    historical[
                        historical.policy.eq(policy)
                        & historical.requested_fault_count.eq(requested)
                    ].monotonicity_error.mean()
                    for requested in (1, 2, 3)
                ]
                axis.scatter(x, c_means, facecolors="none", edgecolors="#6A3D9A", s=45, label="frozen C cell-view" if column == 0 else None, zorder=5)
            axis.set_title(f"{fault.capitalize()} — {policy}")
            axis.set_xticks(x, ["f=1", "f=2", "f=3"])
            axis.grid(axis="y", alpha=0.2)
            if column == 0:
                axis.set_ylabel("Final adjacent monotonicity error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Figure 5 reconstruction: corrected R bars, legacy R dotted means, paper ×, frozen C ○",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    fig.savefig(output_dir / "figure5_reconstruction.png", dpi=180)
    fig.savefig(output_dir / "figure5_reconstruction.svg")
    plt.close(fig)


def hand_counted_fault_toys() -> dict[str, Any]:
    from reference_simulator.api import create_scenario
    from reference_simulator.policies import cell_view_proposal, traditional_proposal

    cases = []
    passive_actor = create_scenario([2, 1], policy="Bubble", faults={0: "passive"}, generation_key="S11/toy/passive-actor", permute=False)
    state = initial_state(passive_actor)
    proposal = cell_view_proposal(passive_actor, state, "cell-0000", side="right")
    cases.append({"id": "passive_actor_noop", "passed": proposal.kind.value == "NoOp" and proposal.reason == "actor_fault"})

    passive_target = create_scenario([2, 1], policy="Bubble", faults={1: "passive"}, generation_key="S11/toy/passive-target", permute=False, seed=11, max_activations=200)
    result = reference_run(passive_target, trace_mode="full")
    moved = any(event["ledgerDelta"]["acceptedSwaps"] == 1 for event in result.events)
    cases.append({"id": "passive_target_movable", "passed": moved})

    stuck_target = create_scenario([2, 1], policy="Bubble", faults={1: "stuck"}, generation_key="S11/toy/stuck-target", permute=False, seed=11, max_activations=200)
    state = initial_state(stuck_target)
    proposal = cell_view_proposal(stuck_target, state, "cell-0000", side="right")
    cases.append({"id": "stuck_target_swap_proposed", "passed": proposal.kind.value == "Swap"})
    result = reference_run(stuck_target, trace_mode="full")
    cases.append({"id": "stuck_target_never_displaced", "passed": result.summary["finalOccupancy"].index("cell-0001") == 1 and result.summary["stopReason"] == "quiescent"})

    traditional_passive = create_scenario([2, 1], policy="Bubble", architecture="traditional", faults={1: "passive"}, generation_key="S11/toy/trad-passive", permute=False)
    cases.append({"id": "traditional_passive_primary_action", "passed": traditional_proposal(traditional_passive, initial_state(traditional_passive)).kind.value == "Swap"})
    traditional_stuck = create_scenario([2, 1], policy="Bubble", architecture="traditional", faults={1: "stuck"}, generation_key="S11/toy/trad-stuck", permute=False)
    result = reference_run(traditional_stuck, trace_mode="full")
    cases.append({"id": "traditional_stuck_primary_block_quiescent", "passed": result.summary["stopReason"] == "quiescent" and result.summary["ledger"]["acceptedSwaps"] == 0})
    return {"cases": cases, "passed": sum(item["passed"] for item in cases), "total": len(cases), "success": all(item["passed"] for item in cases)}


def validate_run_table(frame: pd.DataFrame) -> dict[str, Any]:
    r = frame[frame.backend_profile.eq(R_PROFILE)]
    c = frame[frame.backend_profile.eq(C_PROFILE)]
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, value: Any) -> None:
        array = np.asarray(value, dtype=bool)
        checks[name] = {"passed": bool(array.all()), "checked": int(array.size)}

    add("total_run_count", len(frame) == PAPER_R_COUNT + CONFIRM_R_COUNT + PAPER_C_COUNT)
    add("run_ids_unique", frame.run_id.nunique() == len(frame))
    add("allowed_splits_only", frame.split.isin([PAPER_SPLIT, CONFIRM_SPLIT]))
    add("no_policy_search_or_exploratory", ~frame.split.isin(FORBIDDEN_SPLITS))
    add("R_row_count", len(r) == PAPER_R_COUNT + CONFIRM_R_COUNT)
    add("C_row_count", len(c) == PAPER_C_COUNT)
    add("confirmatory_is_R_corrected_protected", (r[r.split.eq(CONFIRM_SPLIT)].protected & r[r.split.eq(CONFIRM_SPLIT)].placement_profile.eq("reference_without_replacement")))
    corrected = frame[frame.placement_profile.eq("reference_without_replacement")]
    add("corrected_realized_equals_requested", corrected.realized_fault_count.eq(corrected.requested_fault_count))
    legacy = frame[frame.placement_profile.eq("legacy_with_replacement")]
    add("legacy_realized_bounded", legacy.realized_fault_count.le(legacy.requested_fault_count) & legacy.realized_fault_count.ge(1))
    add("final_error_recomputed", frame.apply(lambda row: monotonicity_error(row.final_values) == row.monotonicity_error, axis=1))
    add("completion_recomputed", frame.apply(lambda row: all(a <= b for a, b in zip(row.final_values, row.final_values[1:])) == row.final_nonstrictly_sorted, axis=1))
    add("value_multiset_conserved", frame.value_multiset_conserved)
    add("R_activation_proposal_identity", r.ledger_activations.eq(r.ledger_proposals))
    add("R_displacement_identity", r.ledger_displacedCells.eq(2 * r.ledger_acceptedSwaps))
    add("R_swap_cost_identity", r.swap_only_cost.eq(r.ledger_acceptedSwaps))
    add("R_publication_cost_identity", r.publication_swap_plus_comparison_cost.eq(r.ledger_acceptedSwaps + r.ledger_valueComparisons))
    add("R_target_calculation_identity", r.target_calculations.eq(r.ledger_proposals))
    add("R_rejected_action_identity", r.rejected_actions.eq(r.ledger_rejections + r.ledger_conflictLosses))
    add("R_proposal_partition", r.ledger_proposals.eq(r.ledger_noOps + r.ledger_rejections + r.ledger_memoryUpdates + r.ledger_acceptedSwaps + r.ledger_conflictLosses))
    add("C_historical_scope", c.architecture.eq("cell_view") & c.fault_mode.eq("passive") & c.placement_profile.eq("legacy_with_replacement"))
    unavailable = [
        "ledger_activations", "ledger_observationReads", "ledger_valueComparisons", "ledger_proposals",
        "ledger_noOps", "ledger_rejections", "ledger_memoryUpdates", "ledger_acceptedSwaps", "ledger_conflictLosses",
        "target_calculations", "rejected_actions", "unit_weight_full_ledger",
    ]
    add("C_unavailable_fields_null", c[unavailable].isna().all(axis=1))
    add("C_threads_shutdown", c.thread_shutdown_clean)
    add("no_invariant_errors", ~frame.stop_reason.eq("invariant_error"))
    add("no_unexplained_failure", ~frame.stop_reason.eq("adapter_error"))
    return {
        "checks": checks,
        "success": all(item["passed"] for item in checks.values()),
        "totalRuns": len(frame),
        "referenceRuns": len(r),
        "historicalRuns": len(c),
    }


def analyze(frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_runs(frame)
    primary = primary_contrasts(frame)
    sensitivity = sampling_sensitivity(frame)
    costs = cost_effects(frame)
    reported = reported_mean_comparison(frame)
    count_accounting = fault_count_accounting(frame)
    positions = fault_position_table(frame)
    position_models, position_deciles = fault_position_dependence(frame)
    stops = stop_reason_table(frame)
    claims = architecture_claim_table(primary, frame)
    ranks = rank_claims(frame)

    summary.to_csv(output_dir / "figure5_summary.csv", index=False)
    primary.to_csv(output_dir / "primary_architecture_contrasts.csv", index=False)
    sensitivity.to_csv(output_dir / "sampling_sensitivity.csv", index=False)
    costs.to_csv(output_dir / "paired_cost_effects.csv", index=False)
    reported.to_csv(output_dir / "reported_mean_comparison.csv", index=False)
    count_accounting.to_csv(output_dir / "fault_count_accounting.csv", index=False)
    write_parquet(positions, output_dir / "fault_position_table.parquet")
    position_models.to_csv(output_dir / "fault_position_dependence.csv", index=False)
    position_deciles.to_csv(output_dir / "fault_position_deciles.csv", index=False)
    stops.to_csv(output_dir / "stop_reason_accounting.csv", index=False)
    claims.to_csv(output_dir / "figure5_claim_classifications.csv", index=False)
    ranks.to_csv(output_dir / "figure5_rank_classifications.csv", index=False)
    draw_figure5(frame, output_dir)

    passive_insertion = primary[
        primary.policy.eq("Insertion")
        & primary.fault_mode.eq("passive")
        & primary.requested_fault_count.isin([2, 3])
    ].copy()
    passive_insertion["paper_panel_cell_minus_traditional"] = passive_insertion.requested_fault_count.map({2: 0.63, 3: 3.58})
    passive_insertion["blanket_lower_error_claim"] = "contradicted_if_positive"
    passive_insertion.to_csv(output_dir / "passive_insertion_contradiction.csv", index=False)

    validation = validate_run_table(frame)
    write_json(output_dir / "run_validation.json", validation)
    analysis_summary = {
        "schemaVersion": "e01.s11.analysis_summary.v1",
        "researchStepId": "S11",
        "runCount": len(frame),
        "referenceRunCount": int(frame.backend_profile.eq(R_PROFILE).sum()),
        "historicalRunCount": int(frame.backend_profile.eq(C_PROFILE).sum()),
        "primaryContrasts": len(primary),
        "primaryClassifications": primary.claim_classification.value_counts().sort_index().to_dict(),
        "passiveInsertionF2F3": passive_insertion[
            ["requested_fault_count", "mean_cell_view_minus_traditional", "adjusted_ci_low", "adjusted_ci_high", "claim_classification", "paper_panel_cell_minus_traditional"]
        ].to_dict(orient="records"),
        "paperMeanClassifications": reported.classification.value_counts().sort_index().to_dict(),
        "rankClassifications": ranks.classification.value_counts().sort_index().to_dict(),
        "completionRate": float(frame.completed.mean()),
        "censoredCount": int(frame.censored.sum()),
        "stopReasons": frame.stop_reason.value_counts().sort_index().to_dict(),
        "validationSuccess": validation["success"],
    }
    write_json(output_dir / "analysis_summary.json", analysis_summary)
    return analysis_summary


def environment_record(workers: Mapping[str, int]) -> dict[str, Any]:
    return {
        "researchStepId": "S11",
        "timestampUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpuCount": os.cpu_count(),
        "workers": dict(workers),
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "preregistrationSha256": preregistration_sha256(),
        "historicalPython": str(HISTORICAL_PYTHON),
        "historicalSource": str(HISTORICAL_SOURCE),
    }
