#!/usr/bin/env python3
"""Build the frozen E03 S13 capability and ablation corpora."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from reference_simulator.model import Direction, FaultMode, Policy, canonical_json_bytes
from scripts.build_s12_context_sensitivity import initial_occupancy, policy_codes
from src.detours.context_sensitivity import LEDGER_NAMES, METRIC_NAMES
from src.detours.memory_information import (
    ARM_NAMES,
    CAPABILITY_AUDIT_NAMES,
    STOP_COMPLETE,
    STOP_EVENT_BUDGET,
    STOP_QUIESCENT,
    complexity_rows,
    result_dict,
    simulate_population,
    stream_root,
)
from src.detours.state_space import FamilySpec, StructuralState


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/artifacts/research_steps/S13")
CACHE = Path("/cache/e03_s13")
CONTRACT = ROOT / "analysis/s13_memory_information_contract.json"
S04 = Path("/artifacts/research_steps/S04")
S07 = Path("/artifacts/research_steps/S07")
S08 = Path("/artifacts/research_steps/S08")
S09 = Path("/artifacts/research_steps/S09")
S10 = Path("/artifacts/research_steps/S10")
S11 = Path("/artifacts/research_steps/S11")
S12 = Path("/artifacts/research_steps/S12")

MAX_N = 24
LARGE_EXPECTED = 241_920
SMALL_EXPECTED = 15_480
TOTAL_EXPECTED = LARGE_EXPECTED + SMALL_EXPECTED
RUN_METRIC_EXPECTED = TOTAL_EXPECTED * 4
STOP_LABELS = {
    STOP_COMPLETE: "complete",
    STOP_QUIESCENT: "quiescent",
    STOP_EVENT_BUDGET: "event_budget",
}
POLICY_CODES = {Policy.BUBBLE: 0, Policy.INSERTION: 1, Policy.SELECTION: 2}
CONTRASTS = (
    ("single_failure_bit", "failure_bit", "native"),
    ("single_counter_2bit", "counter_2bit", "native"),
    ("single_recent_direction", "recent_direction", "native"),
    ("single_radius2", "radius2", "native"),
    ("full_vs_native", "full", "native"),
    ("ablate_failure", "full", "full_minus_failure"),
    ("ablate_counter", "full", "full_minus_counter"),
    ("ablate_recent", "full", "full_minus_recent"),
    ("ablate_radius2", "full", "full_minus_radius2"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(path: Path, frame: pd.DataFrame, schema: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {b"schemaVersion": schema.encode(), b"researchStepId": b"S13"}
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
        Path("/workspace/CAPABILITIES.md"),
        Path("/workspace/CAPABILITY_AVAILABILITY.json"),
        Path("/workspace/input-attachments/MANIFEST.json"),
        S07 / "research_step_full_results.md",
        S07 / "path_solutions.parquet",
        S08 / "research_step_full_results.md",
        S09 / "research_step_full_results.md",
        S09 / "necessary_start_coverage.parquet",
        S10 / "research_step_full_results.md",
        S11 / "research_step_full_results.md",
        S12 / "research_step_full_results.md",
        S12 / "larger_n_design.parquet",
        S12 / "larger_n_trajectories.parquet",
        S12 / "larger_n_metric_outcomes.parquet",
        S12 / "cross_layer_directional_comparison.parquet",
        S12 / "metric_outcome_summary.parquet",
        CONTRACT,
        ROOT / "scripts/build_s13_memory_information.py",
        ROOT / "src/detours/memory_information.py",
        ROOT / "src/detours/context_sensitivity.py",
        ROOT / "reference_simulator/engine.py",
        ROOT / "reference_simulator/model.py",
        ROOT / "reference_simulator/policies.py",
        ROOT / "reference_simulator/scheduler.py",
        ROOT / "reference_simulator/transition_primitives.py",
    ]
    paths.extend(sorted(Path("/workspace/input-attachments").glob("*/_metadata/ATTACHMENT.md")))
    return [path for path in paths if path.exists()]


def execution_cache_key() -> str:
    # Cache only the expensive frozen simulation layer.  Post-processing and
    # reporting code may be repaired without invalidating scientifically
    # identical kernel outputs; the full builder is still hashed by
    # ``input_paths`` for the final provenance and immutability audit.
    digest = hashlib.sha256(
        b"E03/S13/primary-cache/v2:257400-runs:ten-arms:checkpoint-v1\x00"
    )
    for path in (
        CONTRACT,
        ROOT / "src/detours/memory_information.py",
        S12 / "larger_n_design.parquet",
        S09 / "necessary_start_coverage.parquet",
    ):
        digest.update(str(path).encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def load_primary_cache(expected_key: str) -> dict[str, np.ndarray] | None:
    metadata_path = CACHE / "primary_results.meta.json"
    array_path = CACHE / "primary_results.npz"
    if not metadata_path.exists() or not array_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("cacheKey") != expected_key:
        return None
    with np.load(array_path, allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    if len(result.get("stop", ())) != TOTAL_EXPECTED:
        return None
    return result


def save_primary_cache(cache_key: str, result: dict[str, np.ndarray]) -> None:
    np.savez(CACHE / "primary_results.npz", **result)
    (CACHE / "primary_results.meta.json").write_text(
        json.dumps(
            {
                "schemaVersion": "e03.s13.primary_cache.v1",
                "cacheKey": cache_key,
                "runs": len(result["stop"]),
            },
            sort_keys=True,
        )
        + "\n"
    )


def pre_state_digest(
    n: int,
    policies: np.ndarray,
    faults: np.ndarray,
    occupancy: np.ndarray,
    cursors: np.ndarray,
    direction: int,
) -> str:
    digest = hashlib.sha256(b"E03/S13/pre-state/v1\x00")
    digest.update(bytes((n, direction)))
    digest.update(policies[:n].tobytes())
    digest.update(faults[:n].tobytes())
    digest.update(occupancy[:n].tobytes())
    digest.update(cursors[:n].tobytes())
    return digest.hexdigest()


def empty_arrays() -> dict[str, list[np.ndarray] | list[int]]:
    return {
        "ns": [],
        "policies": [],
        "faults": [],
        "occupancies": [],
        "cursors": [],
        "directions": [],
        "batch_widths": [],
        "roots": [],
        "budgets": [],
        "arm_codes": [],
    }


def append_arrays(
    arrays: dict[str, list[np.ndarray] | list[int]],
    *,
    n: int,
    policies: np.ndarray,
    faults: np.ndarray,
    occupancy: np.ndarray,
    cursors: np.ndarray,
    direction: int,
    batch_width: int,
    root: int,
    budget: int,
    arm_code: int,
) -> None:
    arrays["ns"].append(n)  # type: ignore[union-attr]
    arrays["policies"].append(policies.copy())  # type: ignore[union-attr]
    arrays["faults"].append(faults.copy())  # type: ignore[union-attr]
    arrays["occupancies"].append(occupancy.copy())  # type: ignore[union-attr]
    arrays["cursors"].append(cursors.copy())  # type: ignore[union-attr]
    arrays["directions"].append(direction)  # type: ignore[union-attr]
    arrays["batch_widths"].append(batch_width)  # type: ignore[union-attr]
    arrays["roots"].append(root)  # type: ignore[union-attr]
    arrays["budgets"].append(budget)  # type: ignore[union-attr]
    arrays["arm_codes"].append(arm_code)  # type: ignore[union-attr]


def finalize_arrays(values: dict[str, list[np.ndarray] | list[int]]) -> dict[str, np.ndarray]:
    return {
        "ns": np.asarray(values["ns"], dtype=np.int16),
        "policies": np.stack(values["policies"]).astype(np.int8),
        "faults": np.stack(values["faults"]).astype(np.int8),
        "occupancies": np.stack(values["occupancies"]).astype(np.int16),
        "cursors": np.stack(values["cursors"]).astype(np.int16),
        "directions": np.asarray(values["directions"], dtype=np.int8),
        "batch_widths": np.asarray(values["batch_widths"], dtype=np.int8),
        "roots": np.asarray(values["roots"], dtype=np.uint64),
        "budgets": np.asarray(values["budgets"], dtype=np.int32),
        "arm_codes": np.asarray(values["arm_codes"], dtype=np.int8),
    }


def build_design() -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    arrays = empty_arrays()
    rows: list[dict[str, Any]] = []

    s12 = pq.read_table(S12 / "larger_n_design.parquet").to_pandas()
    source = s12[s12.replicate_index.isin([0, 1])].copy()
    if len(source) != 24_192 or source.scenario_block_id.nunique() != 6_048:
        raise AssertionError("large-tier outcome-blind S12 selection drifted")
    for source_row in source.itertuples(index=False):
        n = int(source_row.n)
        policy = np.full(MAX_N, -1, dtype=np.int8)
        policy[:n] = policy_codes(str(source_row.policy_profile), n)
        faults = np.zeros(MAX_N, dtype=np.int8)
        for identity in source_row.fault_identity_indices:
            faults[int(identity)] = 1
        occupancy = np.full(MAX_N, -1, dtype=np.int16)
        occupancy[:n] = initial_occupancy(
            n,
            int(source_row.direction_code),
            str(source_row.initial_disorder_profile),
            str(source_row.scenario_block_id),
        )
        cursors = np.full(MAX_N, -32768, dtype=np.int16)
        root = int(source_row.stream_root_hex, 16)
        pair_id = f"large:{source_row.run_id}"
        digest = pre_state_digest(
            n, policy, faults, occupancy, cursors, int(source_row.direction_code)
        )
        for arm_code, arm in enumerate(ARM_NAMES):
            run_id = "s13r:" + hashlib.sha256(f"{pair_id}|{arm}".encode()).hexdigest()
            rows.append(
                {
                    "schema_version": "e03.s13.capability_design.v1",
                    "research_step_id": "S13",
                    "run_id": run_id,
                    "physical_pair_id": pair_id,
                    "evidence_tier": "empirical_large_n",
                    "source_s12_run_id": source_row.run_id,
                    "source_family_ordinal": -1,
                    "source_state_ordinal": -1,
                    "source_necessary_signature": None,
                    "replicate_index": int(source_row.replicate_index),
                    "n": n,
                    "architecture": "cell_view",
                    "policy_profile": source_row.policy_profile,
                    "scheduler_profile": source_row.scheduler_profile,
                    "batch_width": int(source_row.batch_width),
                    "direction": source_row.direction,
                    "direction_code": int(source_row.direction_code),
                    "initial_disorder_profile": source_row.initial_disorder_profile,
                    "intervention_type": source_row.intervention_type,
                    "fault_count": int(source_row.fault_count),
                    "fault_location_class": source_row.fault_location_class,
                    "capability_arm": arm,
                    "capability_arm_code": arm_code,
                    "event_budget": int(source_row.event_budget),
                    "stream_root_hex": f"{root:016x}",
                    "pre_physical_state_sha256": digest,
                    "exact_support": "out_of_exact_domain_large_n",
                }
            )
            append_arrays(
                arrays,
                n=n,
                policies=policy,
                faults=faults,
                occupancy=occupancy,
                cursors=cursors,
                direction=int(source_row.direction_code),
                batch_width=int(source_row.batch_width),
                root=root,
                budget=int(source_row.event_budget),
                arm_code=arm_code,
            )

    inventory = pq.read_table(S04 / "state_family_inventory.parquet").to_pandas()
    family_by_ordinal = {
        int(row.family_ordinal): FamilySpec.from_canonical_dict(
            json.loads(row.canonical_family_json)
        )
        for row in inventory.itertuples(index=False)
    }
    labels = pq.read_table(S09 / "necessary_start_coverage.parquet").to_pandas()
    coverage = labels.copy()
    coverage["capability_applicability"] = np.where(
        coverage.architecture.eq("cell_view"),
        "eligible_local_cell_policy",
        "not_applicable_traditional_global_controller",
    )
    starts = (
        labels[
            [
                "family_ordinal",
                "state_ordinal",
                "n",
                "architecture",
                "direction",
                "policy_profile",
                "source_necessary_signature",
            ]
        ]
        .drop_duplicates()
        .sort_values(["family_ordinal", "state_ordinal"])
    )
    if len(starts) != 407 or starts.architecture.eq("cell_view").sum() != 387:
        raise AssertionError("exact-start coverage drifted")
    for source_row in starts[starts.architecture.eq("cell_view")].itertuples(index=False):
        family = family_by_ordinal[int(source_row.family_ordinal)]
        state = StructuralState(
            family,
            int(source_row.state_ordinal) % math.factorial(family.n),
            int(source_row.state_ordinal) // math.factorial(family.n),
        )
        n = family.n
        policy = np.full(MAX_N, -1, dtype=np.int8)
        fault = np.zeros(MAX_N, dtype=np.int8)
        occupancy = np.full(MAX_N, -1, dtype=np.int16)
        cursors = np.full(MAX_N, -32768, dtype=np.int16)
        for identity in range(n):
            policy[identity] = POLICY_CODES[family.policies[identity]]
            fault[identity] = 1 if family.faults[identity] == FaultMode.STUCK else 0
        occupancy[:n] = np.asarray(state.occupancy, dtype=np.int16)
        for identity, cursor in state.selection_cursors.items():
            cursors[identity] = cursor
        direction_code = 0 if family.direction == Direction.ASCENDING else 1
        digest = pre_state_digest(n, policy, fault, occupancy, cursors, direction_code)
        for replicate in range(4):
            pair_id = (
                f"small:{int(source_row.family_ordinal)}:"
                f"{int(source_row.state_ordinal)}:{replicate}"
            )
            root = int(stream_root(pair_id))
            for arm_code, arm in enumerate(ARM_NAMES):
                run_id = "s13r:" + hashlib.sha256(f"{pair_id}|{arm}".encode()).hexdigest()
                rows.append(
                    {
                        "schema_version": "e03.s13.capability_design.v1",
                        "research_step_id": "S13",
                        "run_id": run_id,
                        "physical_pair_id": pair_id,
                        "evidence_tier": "exact_small_n_anchor_empirical_path",
                        "source_s12_run_id": None,
                        "source_family_ordinal": int(source_row.family_ordinal),
                        "source_state_ordinal": int(source_row.state_ordinal),
                        "source_necessary_signature": source_row.source_necessary_signature,
                        "replicate_index": replicate,
                        "n": n,
                        "architecture": "cell_view",
                        "policy_profile": source_row.policy_profile,
                        "scheduler_profile": "serial_external",
                        "batch_width": 1,
                        "direction": source_row.direction,
                        "direction_code": direction_code,
                        "initial_disorder_profile": "exact_start",
                        "intervention_type": "baseline_one_stuck",
                        "fault_count": 1,
                        "fault_location_class": "identity_specific_exact_start",
                        "capability_arm": arm,
                        "capability_arm_code": arm_code,
                        "event_budget": 2048,
                        "stream_root_hex": f"{root:016x}",
                        "pre_physical_state_sha256": digest,
                        "exact_support": "exact_s07_label_for_unmodified_start_only",
                    }
                )
                append_arrays(
                    arrays,
                    n=n,
                    policies=policy,
                    faults=fault,
                    occupancy=occupancy,
                    cursors=cursors,
                    direction=direction_code,
                    batch_width=1,
                    root=root,
                    budget=2048,
                    arm_code=arm_code,
                )

    design = pd.DataFrame(rows)
    if len(design) != TOTAL_EXPECTED:
        raise AssertionError(f"expected {TOTAL_EXPECTED} S13 runs, got {len(design)}")
    if design.groupby("physical_pair_id").capability_arm.nunique().ne(10).any():
        raise AssertionError("every eligible physical pair must have all ten arms")
    if design.groupby("physical_pair_id").stream_root_hex.nunique().ne(1).any():
        raise AssertionError("paired raw stream roots differ")
    if design.groupby("physical_pair_id").pre_physical_state_sha256.nunique().ne(1).any():
        raise AssertionError("paired pre-states differ")

    design["trace_retained"] = False
    for tier, minimum in (
        ("empirical_large_n", 2420),
        ("exact_small_n_anchor_empirical_path", 774),
    ):
        candidates = design.index[design.evidence_tier.eq(tier)].tolist()
        selected = sorted(
            candidates,
            key=lambda index: hashlib.sha256(design.at[index, "run_id"].encode()).digest(),
        )[:minimum]
        design.loc[selected, "trace_retained"] = True
    return design, finalize_arrays(arrays), coverage


def build_run_frame(
    design: pd.DataFrame,
    result: dict[str, np.ndarray],
    long_frame: pd.DataFrame,
) -> pd.DataFrame:
    runs = design.drop(columns=["direction_code", "capability_arm_code"]).copy()
    runs["stop_reason"] = [STOP_LABELS[int(value)] for value in result["stop"]]
    runs["completed"] = result["stop"] == STOP_COMPLETE
    runs["quiescent"] = result["stop"] == STOP_QUIESCENT
    runs["primary_active_censored"] = result["stop"] == STOP_EVENT_BUDGET
    runs["event_count"] = result["events"]
    for index, name in enumerate(LEDGER_NAMES):
        runs[f"ledger_{name}"] = result["ledger"][:, index]
    runs["ledger_proposals"] = runs.ledger_activations
    runs["ledger_displacedCells"] = 2 * runs.ledger_acceptedSwaps
    for index, name in enumerate(CAPABILITY_AUDIT_NAMES):
        runs[name] = result["capability_audit"][:, index]
    runs["full_ledger_unit_cost"] = runs[
        [column for column in runs if column.startswith("ledger_")]
    ].sum(axis=1) + runs.capability_memory_writes
    runs["physical_state_fingerprint_u64"] = [
        f"{int(value):016x}" for value in result["physical_fingerprint"]
    ]
    runs["full_state_fingerprint_u64"] = [
        f"{int(value):016x}" for value in result["full_fingerprint"]
    ]
    runs["trajectory_digest_u64"] = [
        f"{int(value):016x}" for value in result["trajectory_digest"]
    ]
    runs["accepted_checkpoint_count"] = result["state_checkpoints"]
    runs["maximum_radius_read"] = result["maximum_radius_read"]
    runs["first_action_override_event"] = result["first_action_override_event"]
    runs["first_override_pre_physical_fingerprint_u64"] = [
        f"{int(value):016x}"
        if int(event) >= 0
        else None
        for value, event in zip(
            result["first_override_pre_physical_fingerprint"],
            result["first_action_override_event"],
            strict=True,
        )
    ]
    runs["opportunity_rate"] = np.divide(
        runs.ledger_acceptedSwaps,
        runs.event_count,
        out=np.zeros(len(runs), dtype=float),
        where=runs.event_count.to_numpy() > 0,
    )
    runs["recurrent_active_status"] = "not_active_primary"
    runs.loc[runs.primary_active_censored, "recurrent_active_status"] = "not_assessed_long_horizon"
    if not long_frame.empty:
        labels = {
            "event_budget": "recurrent_active_4x",
            "complete": "completed_by_4x",
            "quiescent": "quiescent_by_4x",
        }
        for row in long_frame.itertuples(index=False):
            runs.loc[runs.run_id.eq(row.run_id), "recurrent_active_status"] = labels[
                row.long_stop_reason
            ]
    return runs


def build_metric_frame(
    runs: pd.DataFrame,
    design: pd.DataFrame,
    result: dict[str, np.ndarray],
    exact_labels: pd.DataFrame,
) -> pd.DataFrame:
    shared = [
        "run_id",
        "physical_pair_id",
        "evidence_tier",
        "source_s12_run_id",
        "source_family_ordinal",
        "source_state_ordinal",
        "source_necessary_signature",
        "replicate_index",
        "n",
        "architecture",
        "policy_profile",
        "scheduler_profile",
        "direction",
        "initial_disorder_profile",
        "intervention_type",
        "fault_count",
        "fault_location_class",
        "capability_arm",
        "event_budget",
        "exact_support",
        "stop_reason",
        "completed",
        "quiescent",
        "primary_active_censored",
        "recurrent_active_status",
        "event_count",
        "ledger_acceptedSwaps",
        "ledger_memoryUpdates",
        "full_ledger_unit_cost",
        "trace_retained",
    ]
    parts = []
    for metric_index, metric in enumerate(METRIC_NAMES):
        part = runs[shared].copy()
        part["metric"] = metric
        part["start_metric_level"] = result["initial"][:, metric_index]
        part["final_metric_level"] = result["final"][:, metric_index]
        part["peak_metric_level"] = result["peak"][:, metric_index]
        part["initial_level_excursion"] = result["max_initial_excursion"][:, metric_index]
        part["running_min_episode_count"] = result["episode_count"][:, metric_index]
        part["recovered_episode_count"] = result["recovered_count"][:, metric_index]
        part["open_censored_episode"] = result["open_episode"][:, metric_index].astype(bool)
        part["max_episode_depth"] = result["max_depth"][:, metric_index]
        part["sum_recovered_episode_depth"] = result["sum_depth"][:, metric_index]
        part["max_episode_duration_opportunities"] = result["max_duration"][:, metric_index]
        part["sum_recovered_episode_duration_opportunities"] = result["sum_duration"][:, metric_index]
        part["open_episode_duration_opportunities"] = result["open_duration"][:, metric_index]
        part["worsening_action_count"] = result["worsening_actions"][:, metric_index]
        part["positive_run_count"] = result["positive_runs"][:, metric_index]
        part["any_detour_episode"] = part.running_min_episode_count > 0
        part["s10_strict_filter_exposed"] = part.worsening_action_count > 0
        parts.append(part)
    metrics = pd.concat(parts, ignore_index=True)
    label_columns = exact_labels[
        [
            "family_ordinal",
            "state_ordinal",
            "metric",
            "classification",
            "minimum_excursion",
        ]
    ].rename(
        columns={
            "family_ordinal": "source_family_ordinal",
            "state_ordinal": "source_state_ordinal",
            "classification": "s07_exact_start_classification",
            "minimum_excursion": "s07_exact_minimum_excursion",
        }
    )
    metrics = metrics.merge(
        label_columns,
        on=["source_family_ordinal", "source_state_ordinal", "metric"],
        how="left",
        validate="many_to_one",
    )
    metrics.loc[
        metrics.evidence_tier.eq("empirical_large_n"), "s07_exact_start_classification"
    ] = "out_of_exact_domain_large_n"
    return metrics


def build_paired_effects(metrics: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "completed",
        "quiescent",
        "primary_active_censored",
        "any_detour_episode",
        "running_min_episode_count",
        "max_episode_depth",
        "max_episode_duration_opportunities",
        "open_censored_episode",
        "final_metric_level",
        "worsening_action_count",
        "event_count",
        "ledger_acceptedSwaps",
        "ledger_memoryUpdates",
        "full_ledger_unit_cost",
    ]
    base_columns = [
        "physical_pair_id",
        "evidence_tier",
        "replicate_index",
        "n",
        "policy_profile",
        "scheduler_profile",
        "intervention_type",
        "source_family_ordinal",
        "source_state_ordinal",
        "source_necessary_signature",
        "metric",
        "s07_exact_start_classification",
        "capability_arm",
    ]
    parts = []
    for contrast_id, first_arm, second_arm in CONTRASTS:
        first = metrics[metrics.capability_arm.eq(first_arm)][base_columns + fields].copy()
        second = metrics[metrics.capability_arm.eq(second_arm)][
            ["physical_pair_id", "metric", "capability_arm"] + fields
        ].copy()
        first = first.rename(columns={field: f"first_{field}" for field in fields})
        second = second.rename(columns={field: f"second_{field}" for field in fields})
        paired = first.merge(
            second,
            on=["physical_pair_id", "metric"],
            validate="one_to_one",
            suffixes=("", "_second"),
        )
        paired["contrast_id"] = contrast_id
        paired["first_arm"] = first_arm
        paired["second_arm"] = second_arm
        paired["both_completed"] = paired.first_completed & paired.second_completed
        for field in fields:
            paired[f"difference_{field}"] = (
                paired[f"first_{field}"].astype(float)
                - paired[f"second_{field}"].astype(float)
            )
        for field in (
            "event_count",
            "ledger_acceptedSwaps",
            "ledger_memoryUpdates",
            "full_ledger_unit_cost",
        ):
            paired.loc[~paired.both_completed, f"successful_pair_difference_{field}"] = np.nan
            paired.loc[paired.both_completed, f"successful_pair_difference_{field}"] = paired.loc[
                paired.both_completed, f"difference_{field}"
            ]
        parts.append(paired)
    result = pd.concat(parts, ignore_index=True)
    expected = (24_192 + 1_548) * len(CONTRASTS) * 4
    if len(result) != expected:
        raise AssertionError(f"paired-effect accounting drifted: {len(result)} != {expected}")
    return result


def run_long_horizon(
    design: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    primary: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, np.ndarray] | None]:
    indices = np.flatnonzero(
        (design.replicate_index.to_numpy() == 0)
        & (primary["stop"] == STOP_EVENT_BUDGET)
    )
    if len(indices) == 0:
        return pd.DataFrame(), None
    extended_budgets = arrays["budgets"][indices] * 4
    result = result_dict(
        simulate_population(
            arrays["ns"][indices],
            arrays["policies"][indices],
            arrays["faults"][indices],
            arrays["occupancies"][indices],
            arrays["cursors"][indices],
            arrays["directions"][indices],
            arrays["batch_widths"][indices],
            arrays["roots"][indices],
            extended_budgets,
            arrays["arm_codes"][indices],
            arrays["budgets"][indices],
        )
    )
    rows = []
    for local, index in enumerate(indices):
        prefix_match = bool(
            int(result["checkpoint_physical_fingerprint"][local])
            == int(primary["physical_fingerprint"][index])
            and int(result["checkpoint_full_fingerprint"][local])
            == int(primary["full_fingerprint"][index])
            and np.array_equal(result["checkpoint_ledger"][local], primary["ledger"][index])
            and np.array_equal(
                result["checkpoint_capability_audit"][local],
                primary["capability_audit"][index],
            )
            and np.array_equal(result["checkpoint_levels"][local], primary["final"][index])
        )
        rows.append(
            {
                "run_id": design.at[index, "run_id"],
                "physical_pair_id": design.at[index, "physical_pair_id"],
                "evidence_tier": design.at[index, "evidence_tier"],
                "capability_arm": design.at[index, "capability_arm"],
                "n": int(design.at[index, "n"]),
                "primary_budget": int(arrays["budgets"][index]),
                "long_budget": int(extended_budgets[local]),
                "prefix_match": prefix_match,
                "long_stop_reason": STOP_LABELS[int(result["stop"][local])],
                "long_event_count": int(result["events"][local]),
                "long_physical_fingerprint_u64": f"{int(result['physical_fingerprint'][local]):016x}",
                "long_full_fingerprint_u64": f"{int(result['full_fingerprint'][local]):016x}",
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.prefix_match.all():
        raise AssertionError("long-horizon primary prefix mismatch")
    return frame, result


def native_s12_parity(
    design: pd.DataFrame,
    result: dict[str, np.ndarray],
) -> pd.DataFrame:
    indices = np.flatnonzero(
        design.evidence_tier.eq("empirical_large_n").to_numpy()
        & design.capability_arm.eq("native").to_numpy()
    )
    source_runs = pq.read_table(S12 / "larger_n_trajectories.parquet").to_pandas().set_index("run_id")
    source_metrics = pq.read_table(S12 / "larger_n_metric_outcomes.parquet").to_pandas()
    source_metrics = source_metrics.set_index(["run_id", "metric"])
    rows = []
    for index in indices:
        source_id = design.at[index, "source_s12_run_id"]
        source = source_runs.loc[source_id]
        checks = {
            "stop": STOP_LABELS[int(result["stop"][index])] == source.stop_reason,
            "events": int(result["events"][index]) == int(source.event_count),
            "ledger": all(
                int(result["ledger"][index, position]) == int(source[f"ledger_{name}"])
                for position, name in enumerate(LEDGER_NAMES)
            ),
            "fingerprint": f"{int(result['physical_fingerprint'][index]):016x}"
            == str(source.state_fingerprint_u64),
        }
        metric_match = True
        for metric_index, metric in enumerate(METRIC_NAMES):
            expected = source_metrics.loc[(source_id, metric)]
            metric_match &= (
                int(result["initial"][index, metric_index]) == int(expected.start_metric_level)
                and int(result["final"][index, metric_index]) == int(expected.final_metric_level)
                and int(result["peak"][index, metric_index]) == int(expected.peak_metric_level)
                and int(result["episode_count"][index, metric_index])
                == int(expected.running_min_episode_count)
                and int(result["worsening_actions"][index, metric_index])
                == int(expected.worsening_accepted_proposal_count)
            )
        checks["metrics"] = bool(metric_match)
        rows.append(
            {
                "run_id": design.at[index, "run_id"],
                "source_s12_run_id": source_id,
                **{f"{name}_match": value for name, value in checks.items()},
                "all_match": all(checks.values()),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 24_192 or not frame.all_match.all():
        raise AssertionError("S13 native arm does not reproduce selected S12 outcomes")
    return frame


def retained_trace_index(
    design: pd.DataFrame,
    result: dict[str, np.ndarray],
) -> pd.DataFrame:
    indices = np.flatnonzero(design.trace_retained.to_numpy())
    rows = []
    for index in indices:
        for role, event_count, levels, physical, full in (
            (
                "initial",
                0,
                result["initial"][index],
                None,
                None,
            ),
            (
                "terminal",
                int(result["events"][index]),
                result["final"][index],
                f"{int(result['physical_fingerprint'][index]):016x}",
                f"{int(result['full_fingerprint'][index]):016x}",
            ),
        ):
            rows.append(
                {
                    "run_id": design.at[index, "run_id"],
                    "evidence_tier": design.at[index, "evidence_tier"],
                    "capability_arm": design.at[index, "capability_arm"],
                    "checkpoint_role": role,
                    "event_count": event_count,
                    "accepted_checkpoint_count": int(result["state_checkpoints"][index]),
                    "trajectory_digest_u64": f"{int(result['trajectory_digest'][index]):016x}",
                    "physical_fingerprint_u64": physical,
                    "full_fingerprint_u64": full,
                    **{
                        f"distance_{metric}": int(levels[metric_index])
                        for metric_index, metric in enumerate(METRIC_NAMES)
                    },
                    "stop_reason": (
                        STOP_LABELS[int(result["stop"][index])] if role == "terminal" else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def deterministic_replay(
    arrays: dict[str, np.ndarray],
    primary: dict[str, np.ndarray],
) -> dict[str, Any]:
    replay = result_dict(
        simulate_population(
            arrays["ns"],
            arrays["policies"],
            arrays["faults"],
            arrays["occupancies"],
            arrays["cursors"],
            arrays["directions"],
            arrays["batch_widths"],
            arrays["roots"],
            arrays["budgets"],
            arrays["arm_codes"],
            np.zeros(len(arrays["ns"]), dtype=np.int32),
        )
    )
    keys = (
        "stop",
        "events",
        "ledger",
        "capability_audit",
        "initial",
        "final",
        "peak",
        "episode_count",
        "recovered_count",
        "open_episode",
        "max_depth",
        "max_duration",
        "worsening_actions",
        "state_checkpoints",
        "trajectory_digest",
        "physical_fingerprint",
        "full_fingerprint",
        "final_occupancy",
        "final_cursors",
        "final_failure",
        "final_counter",
        "final_recent",
    )
    matches = {key: bool(np.array_equal(primary[key], replay[key])) for key in keys}
    if not all(matches.values()):
        raise AssertionError(f"full deterministic replay mismatch: {matches}")
    return {
        "researchStepId": "S13",
        "runsReplayed": len(arrays["ns"]),
        "fieldsCompared": list(keys),
        "matches": matches,
        "success": True,
    }


def main() -> None:
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    CACHE.mkdir(parents=True, exist_ok=True)
    before = {str(path): sha256_file(path) for path in input_paths()}
    write_json(
        OUTPUT / "input_immutability_pre.json",
        {"researchStepId": "S13", "inputs": before},
    )
    write_parquet(
        OUTPUT / "complexity_table.parquet",
        pd.DataFrame(complexity_rows()),
        "e03.s13.complexity_table.v1",
    )

    design, arrays, coverage = build_design()
    write_parquet(OUTPUT / "capability_design.parquet", design, "e03.s13.capability_design.v1")
    write_parquet(
        OUTPUT / "exact_start_coverage.parquet",
        coverage,
        "e03.s13.exact_start_coverage.v1",
    )

    compute_started = time.time()
    cache_key = execution_cache_key()
    primary = load_primary_cache(cache_key)
    primary_cache_used = primary is not None
    if primary is None:
        primary = result_dict(
            simulate_population(
                arrays["ns"],
                arrays["policies"],
                arrays["faults"],
                arrays["occupancies"],
                arrays["cursors"],
                arrays["directions"],
                arrays["batch_widths"],
                arrays["roots"],
                arrays["budgets"],
                arrays["arm_codes"],
                np.zeros(len(design), dtype=np.int32),
            )
        )
        save_primary_cache(cache_key, primary)
    compute_seconds = time.time() - compute_started

    long_frame, _ = run_long_horizon(design, arrays, primary)
    runs = build_run_frame(design, primary, long_frame)
    exact_labels = pq.read_table(S09 / "necessary_start_coverage.parquet").to_pandas()
    metrics = build_metric_frame(runs, design, primary, exact_labels)
    effects = build_paired_effects(metrics, runs)
    parity = native_s12_parity(design, primary)
    traces = retained_trace_index(design, primary)

    write_parquet(
        OUTPUT / "capability_runs.parquet", runs, "e03.s13.capability_runs.v1"
    )
    write_parquet(
        OUTPUT / "memory_information_results.parquet",
        metrics,
        "e03.s13.memory_information_results.v1",
    )
    write_parquet(
        OUTPUT / "paired_ablation_effects.parquet",
        effects,
        "e03.s13.paired_ablation_effects.v1",
    )
    write_parquet(
        OUTPUT / "long_horizon_sensitivity.parquet",
        long_frame,
        "e03.s13.long_horizon_sensitivity.v1",
    )
    write_parquet(
        OUTPUT / "native_s12_parity.parquet",
        parity,
        "e03.s13.native_s12_parity.v1",
    )
    write_parquet(
        OUTPUT / "retained_metric_trace_index.parquet",
        traces,
        "e03.s13.retained_metric_trace_index.v1",
    )

    replay = deterministic_replay(arrays, primary)
    write_json(OUTPUT / "deterministic_replay_validation.json", replay)

    after = {str(path): sha256_file(path) for path in input_paths()}
    immutability = {
        "researchStepId": "S13",
        "success": before == after,
        "inputs": [
            {
                "path": path,
                "sha256Before": before[path],
                "sha256After": after.get(path),
                "unchanged": before[path] == after.get(path),
            }
            for path in before
        ],
    }
    if not immutability["success"]:
        raise AssertionError("S13 execution input changed")
    write_json(OUTPUT / "input_immutability.json", immutability)

    build = {
        "researchStepId": "S13",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "runs": len(runs),
        "runMetricRows": len(metrics),
        "pairedEffectRows": len(effects),
        "largeRuns": int(runs.evidence_tier.eq("empirical_large_n").sum()),
        "smallRuns": int(runs.evidence_tier.eq("exact_small_n_anchor_empirical_path").sum()),
        "physicalPairs": int(runs.physical_pair_id.nunique()),
        "chargedOpportunities": int(runs.event_count.sum()),
        "longHorizonRuns": len(long_frame),
        "longPrefixMatches": int(long_frame.prefix_match.sum()) if len(long_frame) else 0,
        "nativeS12ParityRows": len(parity),
        "nativeS12ParityMatches": int(parity.all_match.sum()),
        "traceRetainedRuns": int(design.trace_retained.sum()),
        "traceIndexRows": len(traces),
        "computeSeconds": compute_seconds,
        "primaryCacheUsed": primary_cache_used,
        "totalSeconds": time.time() - started,
        "threadEnvironment": {
            name: os.environ.get(name)
            for name in (
                "NUMBA_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "host": {"python": platform.python_version(), "platform": platform.platform()},
        "success": True,
    }
    write_json(OUTPUT / "build_summary.json", build)
    print(json.dumps(build, indent=2))


if __name__ == "__main__":
    main()
