#!/usr/bin/env python3
"""Replay and align the canonical E01 retained trajectories for E03 S02."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.chimeric_replication import AdmissibilityTracker, _execute_s13_activation
from analysis.no_fault_sorting import condition_from_dict
from reference_simulator.engine import evaluate_terminal, execute_batch, initial_state
from reference_simulator.model import canonical_json_bytes, state_hash
from scenario_bank.core import materialize_scenario
from shared_events.schema import observable_state_hash
from src.detours.replay import (
    assert_numeric_series_equal,
    canonical_hash,
    goal_directions,
    profile_rows,
    scheduler_span_total,
    stable_trace_id,
    values_hash,
)


E01 = Path("/previous-artifacts/E01")
S08 = E01 / "research_steps/S08"
S06 = E01 / "research_steps/S06"
S09 = E01 / "research_steps/S09"
S11 = E01 / "research_steps/S11"
S12 = E01 / "research_steps/S12"
S13 = E01 / "research_steps/S13"
RELEASE = E01 / "release/baseline"
SCENARIO_BANK = E01 / "scenarios/paired_scenario_bank.parquet"
BASE_BANK = S08 / "base_draw_bank.parquet"
CONDITION_CATALOG = S08 / "condition_catalog.parquet"
SCHEMA_VERSION = "e03.s02.replayed_distance.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_zstd_jsonl(path: Path) -> list[dict[str, Any]]:
    output = subprocess.check_output(["zstd", "-dc", str(path)])
    return [json.loads(line) for line in output.decode("utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(list(rows)),
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )


def _base_output(
    task: Mapping[str, Any],
    *,
    checkpoint_kind: str,
    swap_index: int,
    activation_count: int,
    scheduler_start: int | None,
    scheduler_end: int | None,
    scheduler_count: int,
    native_hash: str | None,
    values: Sequence[int | float],
    stop_reason: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "research_step_id": "S02",
        "source_step_id": task["source_step_id"],
        "source_artifact": task["source_artifact"],
        "logical_trace_id": task["logical_trace_id"],
        "scenario_id": task["scenario_id"],
        "condition_id": task.get("condition_id"),
        "split": task.get("split"),
        "backend_profile": task["backend_profile"],
        "evidence_layer": task["evidence_layer"],
        "replay_mode": task["replay_mode"],
        "sequence_basis": task["sequence_basis"],
        "checkpoint_kind": checkpoint_kind,
        "accepted_swap_index": swap_index,
        "activation_count": activation_count,
        "scheduler_checkpoint_start": scheduler_start,
        "scheduler_checkpoint_end": scheduler_end,
        "scheduler_checkpoint_count": scheduler_count,
        "native_state_hash": native_hash,
        "ordered_values_sha256": values_hash(values),
        "stop_reason": stop_reason,
    }


def _emit_profiles(
    rows: list[dict[str, Any]],
    task: Mapping[str, Any],
    values: Sequence[int | float],
    direction_values: Sequence[str],
    **checkpoint: Any,
) -> None:
    base = _base_output(task, values=values, **checkpoint)
    mixed = len(set(direction_values)) > 1
    for profile in profile_rows(values, direction_values):
        rows.append(
            {
                **base,
                "goal_direction_scope": (
                    "mixed_candidate_projection" if mixed else "native_homogeneous_goal"
                ),
                **profile,
            }
        )


def _materialize(task: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    scenario, metadata = materialize_scenario(
        condition_from_dict(task["condition"]), task["base"]
    )
    if scenario.scenario_id != task["scenario_id"]:
        raise AssertionError("scenario ID changed during rematerialization")
    if metadata["scenarioJsonSha256"] != task["scenario_row"]["scenarioJsonSha256"]:
        raise AssertionError("scenario JSON hash changed during rematerialization")
    return scenario, metadata


def _validate_expected(
    task: Mapping[str, Any],
    *,
    initial_values: Sequence[int | float],
    final_values: Sequence[int | float],
    final_hash: str,
    activation_count: int,
    accepted_swaps: int,
    stop_reason: str | None,
    paper_curve: Sequence[float],
    adjacent_curve: Sequence[int],
    source_curve: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, bool]:
    expected = task["expected"]
    checks: dict[str, bool] = {}
    for field, actual in (
        ("initial_values_sha256", values_hash(initial_values)),
        ("final_values_sha256", values_hash(final_values)),
        ("final_state_hash", final_hash),
        ("activation_count", activation_count),
        ("accepted_swaps", accepted_swaps),
        ("stop_reason", stop_reason),
    ):
        wanted = expected.get(field)
        if wanted is not None and wanted != actual:
            raise AssertionError(f"{task['logical_trace_id']} {field}: {actual!r} != {wanted!r}")
        checks[field] = wanted is None or wanted == actual
    if expected.get("initial_values") is not None:
        if list(initial_values) != list(expected["initial_values"]):
            raise AssertionError("initial ordered values differ from retained trajectory")
        checks["initial_values"] = True
    if expected.get("final_values") is not None:
        if list(final_values) != list(expected["final_values"]):
            raise AssertionError("final ordered values differ from retained trajectory")
        checks["final_values"] = True
    if expected.get("paper_curve") is not None:
        assert_numeric_series_equal(
            paper_curve, expected["paper_curve"], label="retained paper Sortedness curve"
        )
        checks["paper_curve"] = True
    if expected.get("adjacent_curve") is not None:
        assert_numeric_series_equal(
            adjacent_curve,
            expected["adjacent_curve"],
            label="retained post-swap adjacent-descents curve",
        )
        checks["adjacent_curve"] = True
    if source_curve is not None:
        observed = [float(row["paper_sortedness_percent"]) for row in source_curve]
        assert_numeric_series_equal(
            paper_curve, observed, label="retained S13 paper Sortedness curve"
        )
        checks["paper_curve"] = True
    return checks


def replay_standard(task: Mapping[str, Any], scratch: Path) -> dict[str, Any]:
    scenario, _ = _materialize(task)
    direction_values = [cell.direction.value for cell in scenario.cells]
    state = initial_state(scenario)
    initial_native_hash = state_hash(scenario.scenario_id, state)
    initial_values = [scenario.cell_map[cell].value for cell in state.occupancy]
    state.terminal = evaluate_terminal(scenario, state)
    rows: list[dict[str, Any]] = []
    _emit_profiles(
        rows,
        task,
        initial_values,
        direction_values,
        checkpoint_kind="initial",
        swap_index=0,
        activation_count=0,
        scheduler_start=None,
        scheduler_end=None,
        scheduler_count=0,
        native_hash=initial_native_hash,
        stop_reason=state.terminal,
    )
    paper_curve = [100.0 * profile_rows(initial_values, ["ascending"])[0]["paper_sortedness_strict"]]
    adjacent_curve: list[int] = []
    last_emitted_activation = 0
    while state.terminal is None:
        before = state.ledger["acceptedSwaps"]
        execute_batch(scenario, state, retain_events=False, emit_event_records=False)
        after = state.ledger["acceptedSwaps"]
        if after > before:
            if scenario.batch_width == 1 and after != before + 1:
                raise AssertionError("serial replay accepted more than one action")
            values = [scenario.cell_map[cell].value for cell in state.occupancy]
            _emit_profiles(
                rows,
                task,
                values,
                direction_values,
                checkpoint_kind="accepted_action",
                swap_index=int(after),
                activation_count=int(state.activation_count),
                scheduler_start=last_emitted_activation,
                scheduler_end=int(state.activation_count) - 1,
                scheduler_count=int(state.activation_count) - last_emitted_activation,
                native_hash=state_hash(scenario.scenario_id, state),
                stop_reason=state.terminal,
            )
            last_emitted_activation = int(state.activation_count)
            ascending = profile_rows(values, ["ascending"])[0]
            paper_curve.append(100.0 * float(ascending["paper_sortedness_strict"]))
            adjacent_curve.append(int(ascending["adjacent_descents"]))
    final_values = [scenario.cell_map[cell].value for cell in state.occupancy]
    if last_emitted_activation < state.activation_count:
        _emit_profiles(
            rows,
            task,
            final_values,
            direction_values,
            checkpoint_kind="terminal_scheduler_span",
            swap_index=int(state.ledger["acceptedSwaps"]),
            activation_count=int(state.activation_count),
            scheduler_start=last_emitted_activation,
            scheduler_end=int(state.activation_count) - 1,
            scheduler_count=int(state.activation_count) - last_emitted_activation,
            native_hash=state_hash(scenario.scenario_id, state),
            stop_reason=state.terminal,
        )
    final_hash = state_hash(scenario.scenario_id, state)
    checks = _validate_expected(
        task,
        initial_values=initial_values,
        final_values=final_values,
        final_hash=final_hash,
        activation_count=int(state.activation_count),
        accepted_swaps=int(state.ledger["acceptedSwaps"]),
        stop_reason=state.terminal,
        paper_curve=paper_curve,
        adjacent_curve=adjacent_curve,
    )
    if scheduler_span_total(rows[:: len(goal_directions(direction_values))]) != state.activation_count:
        raise AssertionError("scheduler checkpoint run-length coverage is incomplete")
    output = scratch / f"{task['logical_trace_id'].replace(':', '_')}.parquet"
    write_parquet(output, rows)
    return {
        "logicalTraceId": task["logical_trace_id"],
        "scenarioId": scenario.scenario_id,
        "sourceStepId": task["source_step_id"],
        "sourceArtifact": task["source_artifact"],
        "success": True,
        "rowCount": len(rows),
        "directionProfiles": len(goal_directions(direction_values)),
        "acceptedSwaps": int(state.ledger["acceptedSwaps"]),
        "activationCount": int(state.activation_count),
        "stopReason": state.terminal,
        "initialStateHash": initial_native_hash,
        "finalStateHash": final_hash,
        "checks": checks,
        "scratchParquet": str(output),
    }


def replay_s13(task: Mapping[str, Any], scratch: Path) -> dict[str, Any]:
    scenario, _ = _materialize(task)
    direction_values = [cell.direction.value for cell in scenario.cells]
    state = initial_state(scenario)
    initial_native_hash = state_hash(scenario.scenario_id, state)
    initial_values = [scenario.cell_map[cell].value for cell in state.occupancy]
    state.terminal = evaluate_terminal(scenario, state)
    admissibility = AdmissibilityTracker(scenario, state)
    rows: list[dict[str, Any]] = []
    _emit_profiles(
        rows,
        task,
        initial_values,
        direction_values,
        checkpoint_kind="initial",
        swap_index=0,
        activation_count=0,
        scheduler_start=None,
        scheduler_end=None,
        scheduler_count=0,
        native_hash=initial_native_hash,
        stop_reason=state.terminal,
    )
    source_curve = task["expected"]["source_curve"]
    paper_curve = [100.0 * profile_rows(initial_values, ["ascending"])[0]["paper_sortedness_strict"]]
    adjacent_curve: list[int] = []
    last_emitted_activation = 0
    while state.terminal is None:
        committed: list[tuple[int, int]] = []

        def on_swap(current: Any, first: int, second: int) -> None:
            admissibility.after_swap(current, first, second)
            committed.append((first, second))

        outcome = _execute_s13_activation(scenario, state, admissibility, on_swap=on_swap)
        if outcome != "noop":
            if outcome == "memory":
                admissibility.after_memory_update(state)
            state.terminal = admissibility.terminal_after_change(state)
        if committed:
            values = [scenario.cell_map[cell].value for cell in state.occupancy]
            _emit_profiles(
                rows,
                task,
                values,
                direction_values,
                checkpoint_kind="accepted_action",
                swap_index=int(state.ledger["acceptedSwaps"]),
                activation_count=int(state.activation_count),
                scheduler_start=last_emitted_activation,
                scheduler_end=int(state.activation_count) - 1,
                scheduler_count=int(state.activation_count) - last_emitted_activation,
                native_hash=state_hash(scenario.scenario_id, state),
                stop_reason=state.terminal,
            )
            last_emitted_activation = int(state.activation_count)
            ascending = profile_rows(values, ["ascending"])[0]
            paper_curve.append(100.0 * float(ascending["paper_sortedness_strict"]))
            adjacent_curve.append(int(ascending["adjacent_descents"]))
    official_terminal = evaluate_terminal(scenario, state)
    if official_terminal != state.terminal:
        raise AssertionError("S13 optimized replay terminal differs from ordinary authority")
    final_values = [scenario.cell_map[cell].value for cell in state.occupancy]
    if last_emitted_activation < state.activation_count:
        _emit_profiles(
            rows,
            task,
            final_values,
            direction_values,
            checkpoint_kind="terminal_scheduler_span",
            swap_index=int(state.ledger["acceptedSwaps"]),
            activation_count=int(state.activation_count),
            scheduler_start=last_emitted_activation,
            scheduler_end=int(state.activation_count) - 1,
            scheduler_count=int(state.activation_count) - last_emitted_activation,
            native_hash=state_hash(scenario.scenario_id, state),
            stop_reason=state.terminal,
        )
    final_hash = state_hash(scenario.scenario_id, state)
    checks = _validate_expected(
        task,
        initial_values=initial_values,
        final_values=final_values,
        final_hash=final_hash,
        activation_count=int(state.activation_count),
        accepted_swaps=int(state.ledger["acceptedSwaps"]),
        stop_reason=state.terminal,
        paper_curve=paper_curve,
        adjacent_curve=adjacent_curve,
        source_curve=source_curve,
    )
    if scheduler_span_total(rows[:: len(goal_directions(direction_values))]) != state.activation_count:
        raise AssertionError("S13 scheduler checkpoint run-length coverage is incomplete")
    output = scratch / f"{task['logical_trace_id'].replace(':', '_')}.parquet"
    write_parquet(output, rows)
    return {
        "logicalTraceId": task["logical_trace_id"],
        "scenarioId": scenario.scenario_id,
        "sourceStepId": "S13",
        "sourceArtifact": task["source_artifact"],
        "success": True,
        "rowCount": len(rows),
        "directionProfiles": len(goal_directions(direction_values)),
        "acceptedSwaps": int(state.ledger["acceptedSwaps"]),
        "activationCount": int(state.activation_count),
        "stopReason": state.terminal,
        "initialStateHash": initial_native_hash,
        "finalStateHash": final_hash,
        "checks": checks,
        "scratchParquet": str(output),
    }


def align_s06_reference(output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(S06 / "examples/reference_trace.manifest.json")
    events = pq.read_table(S06 / "examples/reference_trace.parquet").to_pylist()
    summary = manifest["sourceRunSummary"]
    final_occupancy = summary["summary"]["finalOccupancy"]
    final_values = summary["summary"]["finalValues"]
    value_map = dict(zip(final_occupancy, final_values))
    occupancy = sorted(value_map, key=lambda value: int(value.rsplit("-", 1)[1]))
    task = {
        "source_step_id": "S06",
        "source_artifact": "research_steps/S06/examples/reference_trace.jsonl.zst",
        "logical_trace_id": stable_trace_id("S06/reference_trace", manifest["scenarioId"]),
        "scenario_id": manifest["scenarioId"],
        "condition_id": None,
        "split": None,
        "backend_profile": "R-clean-room-reference-E01-v1",
        "evidence_layer": "clean_room_reference_retained_event",
        "replay_mode": "retained_event_state_reconstruction",
        "sequence_basis": "activation",
    }
    rows: list[dict[str, Any]] = []
    initial = [value_map[cell] for cell in occupancy]
    _emit_profiles(
        rows,
        task,
        initial,
        ["ascending"],
        checkpoint_kind="initial",
        swap_index=0,
        activation_count=0,
        scheduler_start=None,
        scheduler_end=None,
        scheduler_count=0,
        native_hash=events[0]["native_pre_hash"].removeprefix("sha256:"),
        stop_reason=None,
    )
    accepted = 0
    previous_native = events[0]["native_pre_hash"]
    for ordinal, row in enumerate(events):
        event = json.loads(row["event_json"])
        source = event["backendPayload"]["sourceEvent"]
        values_before = [value_map[cell] for cell in occupancy]
        if row["native_pre_hash"] != previous_native:
            raise AssertionError("S06 native state hash chain is discontinuous")
        if row["observable_pre_hash"] != observable_state_hash(values_before):
            raise AssertionError("S06 observable pre-state hash differs")
        if row["accepted"]:
            first = int(source["proposal"]["actorPos"])
            second = int(source["proposal"]["targetPos"])
            occupancy[first], occupancy[second] = occupancy[second], occupancy[first]
            accepted += 1
        values_after = [value_map[cell] for cell in occupancy]
        if row["observable_post_hash"] != observable_state_hash(values_after):
            raise AssertionError("S06 observable post-state hash differs")
        _emit_profiles(
            rows,
            task,
            values_after,
            ["ascending"],
            checkpoint_kind="accepted_action" if row["accepted"] else "scheduler_checkpoint",
            swap_index=accepted,
            activation_count=ordinal + 1,
            scheduler_start=ordinal,
            scheduler_end=ordinal,
            scheduler_count=1,
            native_hash=row["native_post_hash"].removeprefix("sha256:"),
            stop_reason=row["stop_reason"],
        )
        previous_native = row["native_post_hash"]
    if previous_native.removeprefix("sha256:") != summary["finalStateHash"]:
        raise AssertionError("S06 final state hash differs from source summary")
    if accepted != summary["summary"]["ledger"]["acceptedSwaps"]:
        raise AssertionError("S06 accepted-swap count differs")
    if [value_map[cell] for cell in occupancy] != final_values:
        raise AssertionError("S06 final ordered values differ")
    write_parquet(output, rows)
    return rows, {
        "logicalTraceId": task["logical_trace_id"],
        "scenarioId": manifest["scenarioId"],
        "sourceStepId": "S06",
        "sourceArtifact": task["source_artifact"],
        "success": True,
        "rowCount": len(rows),
        "directionProfiles": 1,
        "acceptedSwaps": accepted,
        "activationCount": len(events),
        "stopReason": manifest["stop"]["reason"],
        "initialStateHash": summary["initialStateHash"],
        "finalStateHash": summary["finalStateHash"],
        "checks": {
            "native_hash_chain": True,
            "observable_hash_chain": True,
            "event_count": True,
            "final_values": True,
        },
        "scratchParquet": str(output),
    }


def align_s06_historical(output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(S06 / "examples/historical_trace.manifest.json")
    event = pq.read_table(S06 / "examples/historical_trace.parquet").to_pylist()[0]
    source = manifest["sourceRunSummary"]
    initial, final = source["inputValues"], source["finalValues"]
    if event["observable_pre_hash"] != observable_state_hash(initial):
        raise AssertionError("historical observable pre-state hash differs")
    if event["observable_post_hash"] != observable_state_hash(final):
        raise AssertionError("historical observable post-state hash differs")
    task = {
        "source_step_id": "S06",
        "source_artifact": "research_steps/S06/examples/historical_trace.jsonl.zst",
        "logical_trace_id": stable_trace_id("S06/historical_trace", manifest["scenarioId"]),
        "scenario_id": manifest["scenarioId"],
        "condition_id": None,
        "split": None,
        "backend_profile": "C-frozen-public-commit",
        "evidence_layer": "frozen_public_commit_observable_snapshot",
        "replay_mode": "retained_observable_state_only",
        "sequence_basis": "recorded_swap",
    }
    rows: list[dict[str, Any]] = []
    for values, kind, swap, activation, start, count, stop in (
        (initial, "initial", 0, 0, None, 0, None),
        (final, "accepted_action", 1, 1, 0, 1, event["stop_reason"]),
    ):
        _emit_profiles(
            rows,
            task,
            values,
            ["ascending"],
            checkpoint_kind=kind,
            swap_index=swap,
            activation_count=activation,
            scheduler_start=start,
            scheduler_end=start,
            scheduler_count=count,
            native_hash=None,
            stop_reason=stop,
        )
    write_parquet(output, rows)
    return rows, {
        "logicalTraceId": task["logical_trace_id"],
        "scenarioId": manifest["scenarioId"],
        "sourceStepId": "S06",
        "sourceArtifact": task["source_artifact"],
        "success": True,
        "supplemental": True,
        "rowCount": 2,
        "directionProfiles": 1,
        "acceptedSwaps": 1,
        "activationCount": None,
        "stopReason": manifest["stop"]["reason"],
        "initialStateHash": None,
        "finalStateHash": None,
        "checks": {"observable_hash_chain": True, "native_hash_unavailable": True},
        "scratchParquet": str(output),
    }


def build_tasks() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected_s09 = read_zstd_jsonl(S09 / "selected_traces.jsonl.zst")
    s11_samples = read_json(S11 / "exact_replay_samples.json")
    s12_selected = pq.read_table(S12 / "figure6_selected_trajectories.parquet").to_pylist()
    s13_curve_rows = pq.read_table(S13 / "selected_swap_traces.parquet").to_pylist()
    s13_curves: dict[str, list[dict[str, Any]]] = {}
    for row in s13_curve_rows:
        s13_curves.setdefault(row["scenario_id"], []).append(row)
    for rows in s13_curves.values():
        rows.sort(key=lambda row: int(row["swap_index"]))

    selected_reference_s09 = [
        row for row in selected_s09 if row["backendProfile"] == "R-clean-room-reference-E01-v1"
    ]
    historical_s09 = [
        row for row in selected_s09 if row["backendProfile"] == "C-frozen-public-commit"
    ]
    target_ids = {
        *(row["scenarioId"] for row in selected_reference_s09),
        *(row["scenarioId"] for row in s11_samples),
        *(row["scenario_id"] for row in s12_selected),
        *s13_curves.keys(),
    }
    scenario_rows = {
        row["scenarioId"]: row
        for row in pq.read_table(
            SCENARIO_BANK, filters=[("scenarioId", "in", sorted(target_ids))]
        ).to_pylist()
    }
    if set(scenario_rows) != target_ids:
        raise AssertionError(f"scenario bank is missing {sorted(target_ids - set(scenario_rows))}")
    base_ids = {row["baseDrawId"] for row in scenario_rows.values()}
    bases = {
        row["baseDrawId"]: row
        for row in pq.read_table(BASE_BANK).to_pylist()
        if row["baseDrawId"] in base_ids
    }
    condition_ids = {row["conditionId"] for row in scenario_rows.values()}
    conditions = {
        row["conditionId"]: json.loads(row["conditionJson"])
        for row in pq.read_table(CONDITION_CATALOG).to_pylist()
        if row["conditionId"] in condition_ids
    }
    no_fault = {
        (row["backend_profile"], row["scenario_id"]): row
        for row in pq.read_table(S09 / "no_fault_runs.parquet").to_pylist()
    }
    frozen_results = {
        row["scenario_id"]: row
        for row in pq.read_table(
            S11 / "frozen_cell_results.parquet",
            filters=[("scenario_id", "in", [row["scenarioId"] for row in s11_samples])],
        ).to_pylist()
        if row["backend_profile"] == "R-clean-room-reference-E01-v1"
    }
    figure6_results = {
        row["scenario_id"]: row
        for row in pq.read_table(
            S12 / "figure6_ambiguity_envelope.parquet",
            filters=[("scenario_id", "in", [row["scenario_id"] for row in s12_selected])],
        ).to_pylist()
    }
    s13_results = {
        row["scenario_id"]: row
        for row in pq.read_table(
            S13 / "chimeric_results.parquet",
            filters=[("scenario_id", "in", sorted(s13_curves))],
        ).to_pylist()
    }

    tasks: list[dict[str, Any]] = []

    def identity(source_artifact: str, scenario_id: str) -> str:
        return stable_trace_id(source_artifact, scenario_id)

    for record in selected_reference_s09:
        sid = record["scenarioId"]
        row = scenario_rows[sid]
        source = no_fault[(record["backendProfile"], sid)]
        tasks.append(
            {
                "engine": "standard",
                "source_step_id": "S09",
                "source_artifact": "research_steps/S09/selected_traces.jsonl.zst",
                "logical_trace_id": identity("S09/selected_traces", sid),
                "scenario_id": sid,
                "condition_id": row["conditionId"],
                "split": row["split"],
                "backend_profile": record["backendProfile"],
                "evidence_layer": "clean_room_reference",
                "replay_mode": "deterministic_reference_replay_of_retained_metric_curve",
                "sequence_basis": record["sequenceBasis"],
                "scenario_row": row,
                "condition": conditions[row["conditionId"]],
                "base": bases[row["baseDrawId"]],
                "expected": {
                    "initial_values_sha256": source["initial_values_sha256"],
                    "final_values_sha256": source["final_values_sha256"],
                    "final_state_hash": source["final_state_hash"],
                    "activation_count": int(source["activation_count"]),
                    "accepted_swaps": int(source["ledger_acceptedSwaps"]),
                    "stop_reason": source["stop_reason"],
                    "paper_curve": [float(point[1]) for point in record["points"]],
                },
            }
        )
    for sample in s11_samples:
        sid = sample["scenarioId"]
        row = scenario_rows[sid]
        source = frozen_results[sid]
        tasks.append(
            {
                "engine": "standard",
                "source_step_id": "S11",
                "source_artifact": "research_steps/S11/exact_replay_samples.json",
                "logical_trace_id": identity("S11/exact_replay_samples", sid),
                "scenario_id": sid,
                "condition_id": row["conditionId"],
                "split": row["split"],
                "backend_profile": "R-clean-room-reference-E01-v1",
                "evidence_layer": "clean_room_reference",
                "replay_mode": "deterministic_reference_replay_of_summary_only_sample",
                "sequence_basis": "activation",
                "scenario_row": row,
                "condition": conditions[row["conditionId"]],
                "base": bases[row["baseDrawId"]],
                "expected": {
                    "initial_values_sha256": source["initial_values_sha256"],
                    "final_values_sha256": source["final_values_sha256"],
                    "final_state_hash": sample["finalStateHash"],
                    "activation_count": int(sample["activationCount"]),
                    "accepted_swaps": int(source["ledger_acceptedSwaps"]),
                    "stop_reason": source["stop_reason"],
                },
            }
        )
    for selected in s12_selected:
        sid = selected["scenario_id"]
        row = scenario_rows[sid]
        source = figure6_results[sid]
        tasks.append(
            {
                "engine": "standard",
                "source_step_id": "S12",
                "source_artifact": "research_steps/S12/figure6_selected_trajectories.parquet",
                "logical_trace_id": identity("S12/figure6_selected_trajectories", sid),
                "scenario_id": sid,
                "condition_id": row["conditionId"],
                "split": row["split"],
                "backend_profile": "R-clean-room-reference-E01-v1",
                "evidence_layer": "clean_room_reference",
                "replay_mode": "deterministic_reference_replay_of_retained_metric_curve",
                "sequence_basis": "accepted_swap_derived_from_activation",
                "scenario_row": row,
                "condition": conditions[row["conditionId"]],
                "base": bases[row["baseDrawId"]],
                "expected": {
                    "initial_values": selected["initial_values"],
                    "final_values": selected["final_values"],
                    "final_state_hash": source["final_state_hash"],
                    "activation_count": int(source["activation_count"]),
                    "accepted_swaps": int(source["accepted_swaps"]),
                    "stop_reason": source["stop_reason"],
                    "adjacent_curve": selected["post_swap_errors"],
                },
            }
        )
    s13_tasks: list[dict[str, Any]] = []
    for sid, curve in s13_curves.items():
        row = scenario_rows[sid]
        source = s13_results[sid]
        s13_tasks.append(
            {
                "engine": "s13",
                "source_step_id": "S13",
                "source_artifact": "research_steps/S13/selected_swap_traces.parquet",
                "logical_trace_id": identity("S13/selected_swap_traces", sid),
                "scenario_id": sid,
                "condition_id": row["conditionId"],
                "split": row["split"],
                "backend_profile": "R-clean-room-reference-E01-v1",
                "evidence_layer": "clean_room_reference",
                "replay_mode": "deterministic_reference_replay_of_retained_metric_curve",
                "sequence_basis": "accepted_swap_derived_from_activation",
                "scenario_row": row,
                "condition": conditions[row["conditionId"]],
                "base": bases[row["baseDrawId"]],
                "expected": {
                    "initial_values_sha256": source["initial_values_sha256"],
                    "final_values_sha256": source["final_values_sha256"],
                    "final_state_hash": source["final_state_hash"],
                    "activation_count": int(source["activation_count"]),
                    "accepted_swaps": int(source["successful_swap_count"]),
                    "stop_reason": source["stop_reason"],
                    "source_curve": curve,
                },
            }
        )
    gaps = [
        {
            "logicalTraceId": identity("S09/selected_traces", row["scenarioId"]),
            "scenarioId": row["scenarioId"],
            "sourceStepId": "S09",
            "sourceArtifact": "research_steps/S09/selected_traces.jsonl.zst",
            "backendProfile": row["backendProfile"],
            "status": "coverage_gap",
            "reasonCode": "historical_sortedness_curve_without_ordered_state_snapshots",
            "detail": (
                "The frozen-C scheduler is nondeterministic and the retained record contains only "
                "swap index plus paper Sortedness. Global S01 distances and native state hashes "
                "cannot be recovered without inventing a trajectory."
            ),
            "retainedPointCount": len(row["points"]),
        }
        for row in historical_s09
    ]
    return tasks, s13_tasks, gaps


def artifact_audit() -> list[dict[str, str]]:
    return [
        {"artifact": "release/baseline/selectedTraces.json", "classification": "primary_denominator", "disposition": "five release-selected artifact contracts audited"},
        {"artifact": "research_steps/S06/examples/reference_trace.jsonl.zst", "classification": "primary_full_activation_trace", "disposition": "aligned directly at every scheduler checkpoint"},
        {"artifact": "research_steps/S06/examples/reference_trace.parquet", "classification": "duplicate_encoding", "disposition": "used for columnar validation; not a second logical trace"},
        {"artifact": "research_steps/S06/examples/historical_trace.jsonl.zst", "classification": "supplemental_observable_trace", "disposition": "two ordered snapshots aligned; native hashes unavailable"},
        {"artifact": "research_steps/S09/selected_traces.jsonl.zst", "classification": "primary_selected_metric_traces", "disposition": "reference traces replayed; historical state gaps explicit"},
        {"artifact": "research_steps/S09/paper_and_selected_swap_trajectories.parquet", "classification": "parent_metric_trajectory_product", "disposition": "audited as parent of release selection; not counted again as independent release traces"},
        {"artifact": "research_steps/S09/trajectory_grid.parquet", "classification": "resampled_derivative", "disposition": "excluded from logical state-trace denominator"},
        {"artifact": "research_steps/S09/trajectory_envelopes.parquet", "classification": "aggregate_derivative", "disposition": "excluded from logical state-trace denominator"},
        {"artifact": "research_steps/S11/exact_replay_samples.json", "classification": "primary_summary_only_replay_set", "disposition": "all 36 samples deterministically replayed"},
        {"artifact": "research_steps/S11/frozen_cell_results.parquet", "classification": "run_summary_parent", "disposition": "used for final validation, not treated as retained trajectories"},
        {"artifact": "research_steps/S12/figure6_selected_trajectories.parquet", "classification": "primary_selected_metric_traces", "disposition": "all six deterministically replayed"},
        {"artifact": "research_steps/S12/figure6_ambiguity_envelope.parquet", "classification": "run_summary_parent_with_metric_lists", "disposition": "used for validation of the six release-selected trajectories"},
        {"artifact": "research_steps/S12/original_dg_results.parquet", "classification": "run_summary_parent", "disposition": "not an independent release-selected trace set"},
        {"artifact": "research_steps/S13/selected_swap_traces.parquet", "classification": "primary_selected_metric_traces", "disposition": "all 29 deterministically replayed"},
        {"artifact": "research_steps/S13/chimeric_trajectories.parquet", "classification": "101_point_resampled_derivative", "disposition": "excluded from logical state-trace denominator"},
        {"artifact": "research_steps/S13/chimeric_results.parquet", "classification": "run_summary_parent", "disposition": "used for state-hash and final validation"},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S02"))
    parser.add_argument("--scratch", type=Path, default=Path("/cache/e03_s02/replays"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in [1,8]")
    args.output.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    source_paths = [
        RELEASE / "selectedTraces.json", SCENARIO_BANK, BASE_BANK, CONDITION_CATALOG,
        S06 / "examples/reference_trace.manifest.json", S06 / "examples/reference_trace.parquet",
        S06 / "examples/historical_trace.manifest.json", S06 / "examples/historical_trace.parquet",
        S09 / "selected_traces.jsonl.zst", S09 / "no_fault_runs.parquet",
        S11 / "exact_replay_samples.json", S11 / "frozen_cell_results.parquet",
        S12 / "figure6_selected_trajectories.parquet", S12 / "figure6_ambiguity_envelope.parquet",
        S13 / "selected_swap_traces.parquet", S13 / "chimeric_results.parquet",
    ]
    pre_hashes = {str(path): sha256_file(path) for path in source_paths}
    start = time.perf_counter()
    tasks, s13_tasks, gaps = build_tasks()
    diagnostics: list[dict[str, Any]] = []
    _, reference_diag = align_s06_reference(args.scratch / "s06_reference.parquet")
    _, historical_diag = align_s06_historical(args.scratch / "s06_historical.parquet")
    diagnostics.extend([reference_diag, historical_diag])
    all_jobs = [(replay_standard, task) for task in tasks] + [(replay_s13, task) for task in s13_tasks]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(function, task, args.scratch): task for function, task in all_jobs}
        for future in as_completed(futures):
            diagnostics.append(future.result())
            print(f"completed {len(diagnostics) - 2}/{len(all_jobs)}", flush=True)
    diagnostics.sort(key=lambda row: (row["sourceStepId"], row["scenarioId"]))

    tables = [pq.read_table(row["scratchParquet"]) for row in diagnostics]
    combined = pa.concat_tables(tables, promote_options="default")
    pq.write_table(
        combined,
        args.output / "replayed_distances.parquet",
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )
    post_hashes = {str(path): sha256_file(path) for path in source_paths}
    immutable = pre_hashes == post_hashes
    if not immutable:
        raise AssertionError("one or more E01 source artifacts changed during S02")

    primary_diagnostics = [row for row in diagnostics if not row.get("supplemental")]
    coverage = {
        "schemaVersion": "e03.s02.trace_coverage.v1",
        "researchStepId": "S02",
        "primaryDenominator": str(RELEASE / "selectedTraces.json"),
        "primaryLogicalTraces": len(primary_diagnostics) + len(gaps),
        "alignedPrimaryLogicalTraces": len(primary_diagnostics),
        "coverageGapLogicalTraces": len(gaps),
        "supplementalAlignedLogicalTraces": 1,
        "coverageFraction": len(primary_diagnostics) / (len(primary_diagnostics) + len(gaps)),
        "aligned": [
            {
                "logicalTraceId": row["logicalTraceId"],
                "scenarioId": row["scenarioId"],
                "sourceStepId": row["sourceStepId"],
                "sourceArtifact": row["sourceArtifact"],
                "status": "aligned",
                "supplemental": bool(row.get("supplemental", False)),
                "rowCount": row["rowCount"],
            }
            for row in diagnostics
        ],
        "gaps": gaps,
        "artifactAudit": artifact_audit(),
    }
    write_json(args.output / "trace_coverage_manifest.json", coverage)
    with (args.output / "trajectory_artifact_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["artifact", "classification", "disposition"])
        writer.writeheader()
        writer.writerows(artifact_audit())
    replay_summary = {
        "schemaVersion": "e03.s02.replay_diagnostics.v1",
        "researchStepId": "S02",
        "success": True,
        "workerCount": args.workers,
        "elapsedSeconds": time.perf_counter() - start,
        "outputRows": combined.num_rows,
        "alignedPrimaryLogicalTraces": len(primary_diagnostics),
        "coverageGapLogicalTraces": len(gaps),
        "supplementalAlignedLogicalTraces": 1,
        "stateHashValidation": "passed_for_all_available_native_hashes",
        "finalDistanceValidation": "passed_direct_recomputation",
        "traceCompletenessValidation": "passed_for_all_aligned_traces",
        "sourceImmutabilityValidation": "passed" if immutable else "failed",
        "diagnostics": diagnostics,
    }
    write_json(args.output / "replay_diagnostics.json", replay_summary)
    write_json(
        args.output / "source_immutability.json",
        {
            "schemaVersion": "e03.s02.source_immutability.v1",
            "researchStepId": "S02",
            "success": immutable,
            "preRunSha256": pre_hashes,
            "postRunSha256": post_hashes,
        },
    )
    print(json.dumps({key: replay_summary[key] for key in (
        "success", "outputRows", "alignedPrimaryLogicalTraces", "coverageGapLogicalTraces",
        "supplementalAlignedLogicalTraces", "elapsedSeconds")}, indent=2))


if __name__ == "__main__":
    main()
