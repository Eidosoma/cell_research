"""E04 S01 baseline chimeric-mixture replication and aggregation analysis.

The module consumes the read-only E01 paper-scale scenario bank, executes only
same-direction chimeras, and writes compact E04 baseline evidence.  The E01
scenario identities are retained for direct parity checks.  The repeated-value
three-policy condition required by E04 S01 is generated with the frozen E01
scenario/seed semantics and is explicitly marked as an E04 extension.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.chimeric_replication import (
    GRID,
    MetricTracker,
    canonical_hash,
    run_reference_summary,
)
from analysis.no_fault_sorting import condition_from_dict
from reference_simulator.model import Cell, Policy, Scenario, canonical_json_bytes
from scenario_bank.core import ConditionSpec, materialize_scenario


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
UPSTREAM = Path("/previous-artifacts/E01")
S08_DIR = UPSTREAM / "research_steps/S08"
S13_DIR = UPSTREAM / "research_steps/S13"
OUTPUT_SCHEMA = "e04.s01.original_mixture.v1"
RUN_PROFILE = "R-clean-room-reference-E01-v1"
EXPECTED_CONDITIONS = 16
EXPECTED_RUNS = 1_600
EXPECTED_SHARED_RUNS = 1_400
BOOTSTRAP_DRAWS = 10_000

POLICY_LABELS = {
    "BUB-INS-SEL": "Bubble+Insertion+Selection",
    "BUB-INS": "Bubble+Insertion",
    "BUB-SEL": "Bubble+Selection",
    "INS-SEL": "Insertion+Selection",
}

PAPER_TARGETS = {
    "C-UNQ-CHIM-BUB-INS-EXACT-ASC": (0.65, 21),
    "C-UNQ-CHIM-BUB-SEL-EXACT-ASC": (0.72, 42),
    "C-UNQ-CHIM-INS-SEL-EXACT-ASC": (0.69, 19),
    "C-UNQ-CHIM-BUB-INS-SEL-EXACT-ASC": (0.62, 22),
    "C-REP-CHIM-BUB-INS-EXACT-ASC": (0.63, 13),
    "C-REP-CHIM-BUB-SEL-EXACT-ASC": (0.69, 100),
    "C-REP-CHIM-INS-SEL-EXACT-ASC": (0.71, 100),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT
    ).strip()


def condition_label(condition_id: str) -> str:
    for token, label in POLICY_LABELS.items():
        if token in condition_id:
            return label
    raise ValueError(f"unrecognized mixture condition {condition_id}")


def _repeated_three_way(assignment: str) -> ConditionSpec:
    assign_code = "EXACT" if assignment == "balanced_exact" else "RANDOM"
    return ConditionSpec(
        condition_id=f"C-REP-CHIM-BUB-INS-SEL-{assign_code}-ASC",
        family="same_direction_repeated",
        input_profile="repeated_1_10_x10",
        architecture="cell_view",
        policies=("Bubble", "Insertion", "Selection"),
        assignment_profile=assignment,
        direction_profile="consensus_ascending",
        direction_map=(
            ("Bubble", "ascending"),
            ("Insertion", "ascending"),
            ("Selection", "ascending"),
        ),
        fault_mode="none",
        requested_fault_count=0,
        placement_profile="not_applicable",
        analysis_label_profile="none",
        profile_role=(
            "reference_primary" if assignment == "balanced_exact" else "assignment_sensitivity"
        ),
        max_activations=1_000_000,
        historical_eligibility="reference_only_e04_plan_required_extension",
        paper_condition_note=(
            "E04 S01 requires a repeated-value three-way baseline; the paper reports "
            "no numeric target for this condition. Frozen E01 assignment and seed semantics apply."
        ),
    )


def _task_from_existing(
    row: Mapping[str, Any], condition: Mapping[str, Any], base: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "scenario_row": dict(row),
        "condition": dict(condition),
        "base": dict(base),
        "upstream_scenario": True,
        "retain_raw_trace": int(row["replicateOrdinal"]) == 0,
        "trace_selection_reason": (
            "replicate_ordinal_zero_per_condition"
            if int(row["replicateOrdinal"]) == 0
            else None
        ),
    }


def _task_from_extension(
    condition: ConditionSpec, base: Mapping[str, Any]
) -> dict[str, Any]:
    scenario, metadata = materialize_scenario(condition, base)
    condition_dict = condition.to_dict()
    row = {
        "scenarioId": scenario.scenario_id,
        "conditionId": condition.condition_id,
        "conditionFamily": condition.family,
        "profileRole": condition.profile_role,
        "assignmentProfile": condition.assignment_profile,
        "inputProfile": condition.input_profile,
        "policySet": list(condition.policies),
        "directionProfile": condition.direction_profile,
        "baseDrawId": base["baseDrawId"],
        "pairingBlockId": base["pairingBlockId"],
        "split": base["split"],
        "protected": bool(base["protected"]),
        "replicateOrdinal": int(base["replicateOrdinal"]),
        "runtimeSeed": metadata["runtimeSeed"],
        "generationKey": f"E01/S08/{condition.condition_id}/{base['baseDrawId']}",
        "maxActivations": condition.max_activations,
        "scenarioJsonSha256": metadata["scenarioJsonSha256"],
        "backendEligibility": condition.historical_eligibility,
        "historicalRandomStreamStatus": "unavailable_not_invented",
    }
    return {
        "scenario_row": row,
        "condition": condition_dict,
        "base": dict(base),
        "upstream_scenario": False,
        "retain_raw_trace": int(base["replicateOrdinal"]) == 0,
        "trace_selection_reason": (
            "replicate_ordinal_zero_per_condition"
            if int(base["replicateOrdinal"]) == 0
            else None
        ),
    }


def load_tasks() -> list[dict[str, Any]]:
    """Load the frozen unprotected S01 population without opening holdouts."""
    scenario_rows = pq.read_table(
        S08_DIR / "paired_scenario_bank.parquet",
        filters=[("split", "=", "paper_scale")],
    ).to_pylist()
    scenario_rows = [
        row
        for row in scenario_rows
        if row["conditionFamily"] in {"same_direction_unique", "same_direction_repeated"}
    ]
    if len(scenario_rows) != EXPECTED_SHARED_RUNS:
        raise ValueError(f"expected {EXPECTED_SHARED_RUNS} E01 mixture rows, found {len(scenario_rows)}")
    if any(bool(row["protected"]) for row in scenario_rows):
        raise PermissionError("S01 population unexpectedly contains protected rows")

    base_rows = pq.read_table(
        S08_DIR / "base_draw_bank.parquet",
        filters=[("split", "=", "paper_scale")],
    ).to_pylist()
    bases = {
        row["baseDrawId"]: row
        for row in base_rows
        if row["inputProfile"] in {"unique_1_100", "repeated_1_10_x10"}
    }
    if len(bases) != 200:
        raise ValueError(f"expected 200 paper-scale base draws, found {len(bases)}")

    catalog_rows = pq.read_table(S08_DIR / "condition_catalog.parquet").to_pylist()
    required = {str(row["conditionId"]) for row in scenario_rows}
    conditions = {
        str(row["conditionId"]): json.loads(row["conditionJson"])
        for row in catalog_rows
        if row["conditionId"] in required
    }
    if set(conditions) != required:
        raise ValueError("E01 condition catalog does not cover the S01 rows")

    tasks = [
        _task_from_existing(row, conditions[str(row["conditionId"])], bases[row["baseDrawId"]])
        for row in scenario_rows
    ]
    repeated_bases = sorted(
        (row for row in bases.values() if row["inputProfile"] == "repeated_1_10_x10"),
        key=lambda row: int(row["replicateOrdinal"]),
    )
    for assignment in ("balanced_exact", "independent_random"):
        condition = _repeated_three_way(assignment)
        tasks.extend(_task_from_extension(condition, base) for base in repeated_bases)

    tasks.sort(key=lambda task: (task["scenario_row"]["conditionId"], task["scenario_row"]["replicateOrdinal"]))
    counts = pd.Series([task["scenario_row"]["conditionId"] for task in tasks]).value_counts()
    if len(tasks) != EXPECTED_RUNS or len(counts) != EXPECTED_CONDITIONS or not counts.eq(100).all():
        raise ValueError(f"S01 population mismatch: runs={len(tasks)}, counts={counts.to_dict()}")
    if sum(bool(task["retain_raw_trace"]) for task in tasks) != EXPECTED_CONDITIONS:
        raise ValueError("S01 must retain one trace per condition")
    return tasks


def _linear_curve(points: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    final_swap = float(points[-1]["swap_index"])
    if final_swap <= 0:
        return [
            {
                "source_swap_index": 0.0,
                "paper_sortedness_percent": float(points[-1]["paper_sortedness_percent"]),
                "publication_aggregation": float(points[-1]["publication_aggregation"]),
            }
            for _ in GRID
        ]
    x = np.asarray([float(point["swap_index"]) / final_swap for point in points])
    target = np.asarray(GRID, dtype=np.float64)
    sortedness = np.interp(
        target, x, np.asarray([float(point["paper_sortedness_percent"]) for point in points])
    )
    aggregation = np.interp(
        target, x, np.asarray([float(point["publication_aggregation"]) for point in points])
    )
    return [
        {
            "source_swap_index": float(progress * final_swap),
            "paper_sortedness_percent": float(sortedness[index]),
            "publication_aggregation": float(aggregation[index]),
        }
        for index, progress in enumerate(GRID)
    ]


def _identity(task: Mapping[str, Any]) -> dict[str, Any]:
    row = task["scenario_row"]
    return {
        "condition_id": row["conditionId"],
        "condition_label": condition_label(str(row["conditionId"])),
        "condition_family": row["conditionFamily"],
        "profile_role": row["profileRole"],
        "assignment_profile": row["assignmentProfile"],
        "input_profile": row["inputProfile"],
        "policy_set_json": json.dumps(row["policySet"], separators=(",", ":")),
        "direction_profile": row["directionProfile"],
        "base_draw_id": row["baseDrawId"],
        "pairing_block_id": row["pairingBlockId"],
        "split": row["split"],
        "protected": bool(row["protected"]),
        "replicate_ordinal": int(row["replicateOrdinal"]),
        "runtime_seed": row["runtimeSeed"],
        "generation_key": row["generationKey"],
        "event_budget": int(row["maxActivations"]),
        "scenario_json_sha256": row["scenarioJsonSha256"],
        "backend_eligibility": row["backendEligibility"],
        "historical_random_stream_status": row["historicalRandomStreamStatus"],
        "upstream_scenario": bool(task["upstream_scenario"]),
        "trace_selection_reason": task["trace_selection_reason"],
    }


def _worker(task: Mapping[str, Any]) -> dict[str, Any]:
    condition = condition_from_dict(task["condition"])
    scenario, metadata = materialize_scenario(condition, task["base"])
    row = task["scenario_row"]
    if scenario.scenario_id != row["scenarioId"]:
        raise ValueError("scenario identity failed deterministic materialization")
    if metadata["scenarioJsonSha256"] != row["scenarioJsonSha256"]:
        raise ValueError("scenario JSON hash failed deterministic materialization")
    result = run_reference_summary(
        scenario,
        family=str(row["conditionFamily"]),
        retain_raw_trace=True,
    )
    raw_trace = result["raw_trace"]
    result["linear_curve_grid"] = _linear_curve(raw_trace)
    if not task["retain_raw_trace"]:
        result["raw_trace"] = None
    result.update(_identity(task))
    result["schema_version"] = OUTPUT_SCHEMA
    result["research_step_id"] = "S01"
    result["evidence_layer"] = "clean_room_reference"
    result["publication_snapshot_claimed"] = False
    result["run_id"] = "e04s01:" + hashlib.sha256(
        f"{RUN_PROFILE}|{scenario.scenario_id}".encode()
    ).hexdigest()
    result["linear_trajectory_sha256"] = canonical_hash(result["linear_curve_grid"])
    return result


def _read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["scenario_id"]] = row
    return rows


def run_population(
    tasks: Sequence[Mapping[str, Any]], checkpoint: Path, *, workers: int = 8
) -> list[dict[str, Any]]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1,8]")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_checkpoint(checkpoint)
    expected = {str(task["scenario_row"]["scenarioId"]) for task in tasks}
    if not set(completed) <= expected:
        raise ValueError("checkpoint contains rows outside E04 S01")
    pending = [task for task in tasks if task["scenario_row"]["scenarioId"] not in completed]
    with checkpoint.open("a", encoding="utf-8") as handle:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker, task): task for task in pending}
            for future in as_completed(futures):
                result = future.result()
                handle.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                completed[result["scenario_id"]] = result
                count = len(completed)
                if count % 50 == 0 or count == len(tasks):
                    print(f"E04 S01 progress {count}/{len(tasks)}", file=sys.stderr, flush=True)
    if set(completed) != expected:
        raise ValueError("E04 S01 run accounting mismatch")
    return [completed[key] for key in sorted(completed)]


def replay_selected(
    tasks: Sequence[Mapping[str, Any]], *, workers: int = 8
) -> list[dict[str, Any]]:
    selected = [task for task in tasks if task["retain_raw_trace"]]
    if len(selected) != EXPECTED_CONDITIONS:
        raise ValueError("deterministic replay selection mismatch")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_worker, selected))


def _run_frame(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            key: value
            for key, value in result.items()
            if key not in {"curve_grid", "linear_curve_grid", "raw_trace"}
        }
        for result in results
    ]
    frame = pd.DataFrame(rows).sort_values(["condition_id", "replicate_ordinal"]).reset_index(drop=True)
    if len(frame) != EXPECTED_RUNS or frame.run_id.duplicated().any():
        raise ValueError("E04 S01 run frame accounting mismatch")
    return frame


def _trajectory_frame(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        for grid_index, (floor, linear) in enumerate(
            zip(result["curve_grid"], result["linear_curve_grid"])
        ):
            rows.append(
                {
                    "schema_version": OUTPUT_SCHEMA,
                    "research_step_id": "S01",
                    "run_id": result["run_id"],
                    "scenario_id": result["scenario_id"],
                    "condition_id": result["condition_id"],
                    "condition_label": result["condition_label"],
                    "condition_family": result["condition_family"],
                    "assignment_profile": result["assignment_profile"],
                    "profile_role": result["profile_role"],
                    "input_profile": result["input_profile"],
                    "policy_set_json": result["policy_set_json"],
                    "pairing_block_id": result["pairing_block_id"],
                    "replicate_ordinal": result["replicate_ordinal"],
                    "upstream_scenario": result["upstream_scenario"],
                    "grid_index": grid_index,
                    "normalized_swap_progress": GRID[grid_index],
                    "progress_normalization": "accepted_swap_floor_no_interpolation",
                    "source_swap_index": int(floor["swap_index"]),
                    "paper_sortedness_percent": floor["paper_sortedness_percent"],
                    "reference_sortedness_percent": floor["reference_sortedness_percent"],
                    "publication_aggregation": floor["publication_aggregation"],
                    "reference_aggregation": floor["reference_aggregation"],
                    "duplicate_same_algotype_edge_rate": floor["duplicate_same_algotype_edge_rate"],
                    "equal_value_edge_count": floor["equal_value_edge_count"],
                    "linear_source_swap_index": linear["source_swap_index"],
                    "linear_paper_sortedness_percent": linear["paper_sortedness_percent"],
                    "linear_publication_aggregation": linear["publication_aggregation"],
                    "stop_reason": result["stop_reason"],
                    "completed": result["completed"],
                    "censored": result["censored"],
                    "event_budget": result["event_budget"],
                    "activation_count": result["activation_count"],
                    "successful_swap_count": result["successful_swap_count"],
                    "policy_counts_json": result["policy_counts_json"],
                    "publication_aggregation_null": result["publication_aggregation_null"],
                    "trajectory_sha256": result["trajectory_sha256"],
                    "linear_trajectory_sha256": result["linear_trajectory_sha256"],
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_RUNS * len(GRID):
        raise ValueError("E04 S01 trajectory accounting mismatch")
    return frame


def _trace_frame(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result["raw_trace"] is None:
            continue
        final_swaps = int(result["successful_swap_count"])
        for point in result["raw_trace"]:
            rows.append(
                {
                    "scenario_id": result["scenario_id"],
                    "condition_id": result["condition_id"],
                    "assignment_profile": result["assignment_profile"],
                    "input_profile": result["input_profile"],
                    "swap_index": int(point["swap_index"]),
                    "normalized_swap_progress": (
                        float(point["swap_index"]) / final_swaps if final_swaps else 0.0
                    ),
                    **{key: value for key, value in point.items() if key != "swap_index"},
                }
            )
    frame = pd.DataFrame(rows)
    if frame.scenario_id.nunique() != EXPECTED_CONDITIONS:
        raise ValueError("selected trace coverage mismatch")
    return frame


def _bootstrap_peak(trajectories: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition_id, group in trajectories.groupby("condition_id", sort=True):
        floor = group.pivot(
            index="scenario_id", columns="grid_index", values="publication_aggregation"
        ).sort_index().to_numpy(dtype=np.float64)
        linear = group.pivot(
            index="scenario_id", columns="grid_index", values="linear_publication_aggregation"
        ).sort_index().to_numpy(dtype=np.float64)
        if floor.shape != (100, 101) or linear.shape != (100, 101):
            raise ValueError(f"peak matrix mismatch for {condition_id}")
        seed = int.from_bytes(
            hashlib.sha256(f"E04/S01/peak-bootstrap/{condition_id}".encode()).digest()[:16],
            "big",
        )
        generator = np.random.Generator(np.random.PCG64DXSM(seed))
        counts = generator.multinomial(100, [0.01] * 100, size=BOOTSTRAP_DRAWS)
        floor_curves = counts @ floor / 100.0
        linear_curves = counts @ linear / 100.0
        floor_mean = floor.mean(axis=0)
        linear_mean = linear.mean(axis=0)
        floor_peaks = floor_curves.max(axis=1)
        linear_peaks = linear_curves.max(axis=1)
        rows.append(
            {
                "condition_id": condition_id,
                "condition_label": condition_label(condition_id),
                "assignment_profile": group.assignment_profile.iloc[0],
                "input_profile": group.input_profile.iloc[0],
                "runs": 100,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "floor_peak": float(floor_mean.max()),
                "floor_peak_progress_percent": int(floor_mean.argmax()),
                "floor_peak_ci95_low": float(np.quantile(floor_peaks, 0.025)),
                "floor_peak_ci95_high": float(np.quantile(floor_peaks, 0.975)),
                "floor_progress_ci95_low": float(np.quantile(floor_curves.argmax(axis=1), 0.025)),
                "floor_progress_ci95_high": float(np.quantile(floor_curves.argmax(axis=1), 0.975)),
                "linear_peak": float(linear_mean.max()),
                "linear_peak_progress_percent": int(linear_mean.argmax()),
                "linear_peak_ci95_low": float(np.quantile(linear_peaks, 0.025)),
                "linear_peak_ci95_high": float(np.quantile(linear_peaks, 0.975)),
                "linear_minus_floor_peak": float(linear_mean.max() - floor_mean.max()),
                "linear_minus_floor_peak_progress_percent": int(
                    linear_mean.argmax() - floor_mean.argmax()
                ),
            }
        )
    return pd.DataFrame(rows)


def _condition_summary(runs: pd.DataFrame, peaks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    peak_map = peaks.set_index("condition_id")
    for condition_id, group in runs.groupby("condition_id", sort=True):
        rows.append(
            {
                "condition_id": condition_id,
                "condition_label": condition_label(condition_id),
                "input_profile": group.input_profile.iloc[0],
                "assignment_profile": group.assignment_profile.iloc[0],
                "profile_role": group.profile_role.iloc[0],
                "runs": len(group),
                "completed_runs": int(group.completed.sum()),
                "completion_fraction": float(group.completed.mean()),
                "stop_reason_counts_json": json.dumps(
                    group.stop_reason.value_counts().sort_index().to_dict(), separators=(",", ":")
                ),
                "mean_initial_publication_aggregation": float(group.initial_publication_aggregation.mean()),
                "mean_final_publication_aggregation": float(group.final_publication_aggregation.mean()),
                "mean_initial_paper_sortedness_percent": float(group.initial_paper_sortedness_percent.mean()),
                "mean_final_paper_sortedness_percent": float(group.final_paper_sortedness_percent.mean()),
                "mean_successful_swap_count": float(group.successful_swap_count.mean()),
                "sd_successful_swap_count": float(group.successful_swap_count.std(ddof=1)),
                "mean_activation_count": float(group.activation_count.mean()),
                "mean_final_duplicate_same_algotype_edge_rate": float(
                    group.final_duplicate_same_algotype_edge_rate.mean()
                ),
                "floor_peak_publication_aggregation": float(peak_map.loc[condition_id, "floor_peak"]),
                "floor_peak_progress_percent": int(
                    peak_map.loc[condition_id, "floor_peak_progress_percent"]
                ),
                "linear_peak_publication_aggregation": float(peak_map.loc[condition_id, "linear_peak"]),
                "linear_peak_progress_percent": int(
                    peak_map.loc[condition_id, "linear_peak_progress_percent"]
                ),
                "upstream_scenario": bool(group.upstream_scenario.all()),
            }
        )
    return pd.DataFrame(rows)


def _composition_audit(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = runs[
        [
            "run_id",
            "scenario_id",
            "condition_id",
            "input_profile",
            "assignment_profile",
            "replicate_ordinal",
            "policy_counts_json",
            "clustering_label_counts_json",
            "publication_aggregation_null",
            "reference_aggregation_null",
            "initial_publication_aggregation",
            "upstream_scenario",
        ]
    ].copy()
    audit["n"] = audit.policy_counts_json.map(lambda value: sum(json.loads(value).values()))
    audit["policy_count_sum_valid"] = audit.n.eq(100)
    audit["initial_minus_composition_expectation"] = (
        audit.initial_publication_aggregation - audit.publication_aggregation_null
    )
    audit["composition_counts_sorted_json"] = audit.policy_counts_json.map(
        lambda value: json.dumps(sorted(json.loads(value).values()), separators=(",", ":"))
    )
    summary_rows = []
    for condition_id, group in audit.groupby("condition_id", sort=True):
        counts = [json.loads(value) for value in group.policy_counts_json]
        policies = sorted(set().union(*(item.keys() for item in counts)))
        summary_rows.append(
            {
                "condition_id": condition_id,
                "condition_label": condition_label(condition_id),
                "input_profile": group.input_profile.iloc[0],
                "assignment_profile": group.assignment_profile.iloc[0],
                "runs": len(group),
                "count_ranges_json": json.dumps(
                    {policy: [min(item.get(policy, 0) for item in counts), max(item.get(policy, 0) for item in counts)] for policy in policies},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "mean_publication_expectation": float(group.publication_aggregation_null.mean()),
                "mean_initial_publication_aggregation": float(group.initial_publication_aggregation.mean()),
                "mean_initial_minus_expectation": float(
                    group.initial_minus_composition_expectation.mean()
                ),
                "max_abs_initial_minus_expectation": float(
                    group.initial_minus_composition_expectation.abs().max()
                ),
            }
        )
    return audit, pd.DataFrame(summary_rows)


def _paper_agreement(summary: pd.DataFrame, peaks: pd.DataFrame) -> pd.DataFrame:
    exact = summary[summary.assignment_profile.eq("balanced_exact")]
    peak_map = peaks.set_index("condition_id")
    rows = []
    for item in exact.itertuples(index=False):
        target = PAPER_TARGETS.get(item.condition_id)
        target_peak, target_time = target if target is not None else (math.nan, math.nan)
        observed_peak = float(item.floor_peak_publication_aggregation)
        observed_time = int(item.floor_peak_progress_percent)
        peak_agrees = bool(target is not None and abs(observed_peak - target_peak) <= 0.03)
        time_agrees = bool(target is not None and abs(observed_time - target_time) <= 5)
        rows.append(
            {
                "condition_id": item.condition_id,
                "condition_label": item.condition_label,
                "input_profile": item.input_profile,
                "paper_peak": target_peak,
                "paper_peak_progress_percent": target_time,
                "observed_floor_peak": observed_peak,
                "observed_floor_peak_ci95_low": float(peak_map.loc[item.condition_id, "floor_peak_ci95_low"]),
                "observed_floor_peak_ci95_high": float(peak_map.loc[item.condition_id, "floor_peak_ci95_high"]),
                "observed_floor_peak_progress_percent": observed_time,
                "peak_tolerance": 0.03,
                "progress_tolerance_percentage_points": 5,
                "peak_agreement": peak_agrees if target is not None else None,
                "timing_agreement": time_agrees if target is not None else None,
                "baseline_classification": (
                    "agreement" if peak_agrees and time_agrees
                    else "partial_agreement" if peak_agrees or time_agrees
                    else "discrepancy" if target is not None
                    else "plan_required_extension_no_paper_target"
                ),
                "all_runs_completed": int(item.completed_runs) == int(item.runs),
            }
        )
    return pd.DataFrame(rows)


def _e01_parity(runs: pd.DataFrame) -> pd.DataFrame:
    upstream = pd.read_parquet(S13_DIR / "chimeric_results.parquet")
    upstream = upstream[
        upstream.condition_family.isin({"same_direction_unique", "same_direction_repeated"})
    ]
    current = runs[runs.upstream_scenario]
    fields = (
        "stop_reason",
        "activation_count",
        "successful_swap_count",
        "final_state_hash",
        "trajectory_sha256",
        "policy_counts_json",
        "initial_publication_aggregation",
        "final_publication_aggregation",
        "initial_paper_sortedness_percent",
        "final_paper_sortedness_percent",
    )
    merged = current.merge(
        upstream[["scenario_id", *fields]], on="scenario_id", suffixes=("_e04", "_e01"), validate="one_to_one"
    )
    rows = []
    for item in merged.to_dict("records"):
        checks = {field: item[f"{field}_e04"] == item[f"{field}_e01"] for field in fields}
        rows.append(
            {
                "scenario_id": item["scenario_id"],
                "all_scientific_fields_equal": all(checks.values()),
                **{f"equal_{field}": passed for field, passed in checks.items()},
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_SHARED_RUNS:
        raise ValueError("E01 parity accounting mismatch")
    return result


def _replay_record(
    results: Sequence[Mapping[str, Any]], replays: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    original = {row["scenario_id"]: row for row in results}
    fields = (
        "stop_reason",
        "activation_count",
        "successful_swap_count",
        "final_state_hash",
        "trajectory_sha256",
        "linear_trajectory_sha256",
        "initial_publication_aggregation",
        "final_publication_aggregation",
    )
    samples = []
    for replay in replays:
        source = original[replay["scenario_id"]]
        checks = {field: replay[field] == source[field] for field in fields}
        samples.append(
            {
                "conditionId": replay["condition_id"],
                "scenarioId": replay["scenario_id"],
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return {
        "schema": "e04.s01.deterministic_replay.v1",
        "researchStepId": "S01",
        "samples": samples,
        "allPassed": len(samples) == EXPECTED_CONDITIONS and all(row["passed"] for row in samples),
    }


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, compression="zstd")


def _plot_profile(trajectories: pd.DataFrame, output: Path, input_profile: str) -> None:
    source = trajectories[
        trajectories.input_profile.eq(input_profile)
        & trajectories.assignment_profile.eq("balanced_exact")
    ]
    conditions = sorted(source.condition_id.unique(), key=lambda value: condition_label(value))
    if len(conditions) != 4:
        raise ValueError(f"expected four exact conditions for {input_profile}")
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), sharex=True, constrained_layout=True)
    for column, condition_id in enumerate(conditions):
        group = source[source.condition_id.eq(condition_id)]
        agg = group.groupby("grid_index").publication_aggregation.agg(["mean", "sem"])
        sortedness = group.groupby("grid_index").paper_sortedness_percent.agg(["mean", "sem"])
        x = np.asarray(GRID) * 100
        axes[0, column].plot(x, agg["mean"], color="#b33f3f", label="floor/no interpolation")
        axes[0, column].fill_between(
            x,
            agg["mean"] - 1.96 * agg["sem"],
            agg["mean"] + 1.96 * agg["sem"],
            color="#b33f3f",
            alpha=0.18,
            label="pointwise 95% mean band",
        )
        linear = group.groupby("grid_index").linear_publication_aggregation.mean()
        axes[0, column].plot(x, linear, color="#333333", linestyle=":", linewidth=1, label="linear sensitivity")
        axes[1, column].plot(x, sortedness["mean"], color="#286a9b")
        axes[1, column].fill_between(
            x,
            sortedness["mean"] - 1.96 * sortedness["sem"],
            sortedness["mean"] + 1.96 * sortedness["sem"],
            color="#286a9b",
            alpha=0.18,
        )
        axes[0, column].set_title(condition_label(condition_id))
        axes[1, column].set_xlabel("Accepted-swap progress (%)")
        for row in (0, 1):
            axes[row, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel("Publication aggregation (/n)")
    axes[1, 0].set_ylabel("Paper Sortedness (%)")
    axes[0, 0].legend(fontsize=7)
    title = "Unique values 1–100" if input_profile == "unique_1_100" else "Repeated values 1–10 × 10"
    fig.suptitle(f"E04 S01 baseline trajectories — {title} — exact composition", fontsize=14)
    stem = "baseline_unique_trajectories" if input_profile == "unique_1_100" else "baseline_repeated_trajectories"
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=180)
    plt.close(fig)


def build_artifacts(
    results: Sequence[Mapping[str, Any]],
    replays: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    runs = _run_frame(results)
    trajectories = _trajectory_frame(results)
    traces = _trace_frame(results)
    peaks = _bootstrap_peak(trajectories)
    summary = _condition_summary(runs, peaks)
    composition, composition_summary = _composition_audit(runs)
    agreement = _paper_agreement(summary, peaks)
    parity = _e01_parity(runs)
    replay = _replay_record(results, replays)

    _write_parquet(trajectories, output / "original_mixtures.parquet")
    _write_parquet(runs, output / "run_summary.parquet")
    _write_parquet(traces, output / "selected_swap_traces.parquet")
    summary.to_csv(output / "condition_summary.csv", index=False)
    peaks.to_csv(output / "aggregation_peak_uncertainty.csv", index=False)
    composition.to_csv(output / "composition_audit.csv", index=False)
    composition_summary.to_csv(output / "composition_summary.csv", index=False)
    agreement.to_csv(output / "baseline_agreement.csv", index=False)
    parity.to_csv(output / "e01_parity.csv", index=False)
    peaks[
        [
            "condition_id",
            "condition_label",
            "assignment_profile",
            "input_profile",
            "floor_peak",
            "floor_peak_progress_percent",
            "linear_peak",
            "linear_peak_progress_percent",
            "linear_minus_floor_peak",
            "linear_minus_floor_peak_progress_percent",
        ]
    ].to_csv(output / "progress_alignment_audit.csv", index=False)
    write_json(output / "deterministic_replay.json", replay)
    write_json(
        output / "run_accounting.json",
        {
            "schema": "e04.s01.run_accounting.v1",
            "researchStepId": "S01",
            "executedRuns": len(runs),
            "trajectoryRows": len(trajectories),
            "conditions": int(runs.condition_id.nunique()),
            "sharedE01Scenarios": int(runs.upstream_scenario.sum()),
            "planRequiredRepeatedThreeWayExtensions": int((~runs.upstream_scenario).sum()),
            "selectedTraceScenarios": int(traces.scenario_id.nunique()),
            "selectedTraceRows": len(traces),
            "byInputProfile": runs.input_profile.value_counts().sort_index().to_dict(),
            "byAssignmentProfile": runs.assignment_profile.value_counts().sort_index().to_dict(),
            "byStopReason": runs.stop_reason.value_counts().sort_index().to_dict(),
            "protectedRowsOpened": int(runs.protected.sum()),
            "forbiddenSplitRowsOpened": int((runs.split != "paper_scale").sum()),
        },
    )
    _plot_profile(trajectories, output, "unique_1_100")
    _plot_profile(trajectories, output, "repeated_1_10_x10")
    return {
        "runs": len(runs),
        "trajectoryRows": len(trajectories),
        "conditions": int(runs.condition_id.nunique()),
    }


def _metric_fixture() -> dict[str, Any]:
    scenario = Scenario.create(
        (
            Cell("a", 3, Policy.BUBBLE, analysis_label="x"),
            Cell("b", 1, Policy.BUBBLE, analysis_label="y"),
            Cell("c", 1, Policy.BUBBLE, analysis_label="y"),
            Cell("d", 2, Policy.BUBBLE, analysis_label="x"),
        ),
        initial_occupancy=("a", "b", "c", "d"),
        generation_key="E04/S01/metric-fixture",
    )
    metrics = MetricTracker(scenario, control=True).metrics()
    expected = {
        "paper_sortedness_percent": 50.0,
        "reference_sortedness_percent": 75.0,
        "publication_aggregation": 0.25,
        "reference_aggregation": 1 / 3,
        "duplicate_same_algotype_edge_rate": 1.0,
        "equal_value_edge_count": 1.0,
    }
    checks = {
        key: math.isclose(float(metrics[key]), value, abs_tol=1e-12)
        for key, value in expected.items()
    }
    return {"passed": all(checks.values()), "checks": checks, "observed": metrics, "expected": expected}


def validate_artifacts(output: Path) -> dict[str, Any]:
    trajectories = pd.read_parquet(output / "original_mixtures.parquet")
    runs = pd.read_parquet(output / "run_summary.parquet")
    traces = pd.read_parquet(output / "selected_swap_traces.parquet")
    composition = pd.read_csv(output / "composition_audit.csv")
    composition_summary = pd.read_csv(output / "composition_summary.csv")
    peaks = pd.read_csv(output / "aggregation_peak_uncertainty.csv")
    parity = pd.read_csv(output / "e01_parity.csv")
    replay = json.loads((output / "deterministic_replay.json").read_text())
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    check("run_count", len(runs) == EXPECTED_RUNS, len(runs))
    check("run_ids_unique", runs.run_id.nunique() == EXPECTED_RUNS, runs.run_id.nunique())
    check("condition_count", runs.condition_id.nunique() == EXPECTED_CONDITIONS, runs.condition_id.nunique())
    condition_counts = runs.groupby("condition_id").size()
    check("one_hundred_runs_per_condition", condition_counts.eq(100).all(), condition_counts.value_counts().to_dict())
    check(
        "trajectory_accounting",
        len(trajectories) == EXPECTED_RUNS * 101 and trajectories.scenario_id.nunique() == EXPECTED_RUNS,
        [len(trajectories), trajectories.scenario_id.nunique()],
    )
    check("trajectory_grid_complete", trajectories.groupby("scenario_id").size().eq(101).all(), None)
    check("selected_trace_coverage", traces.scenario_id.nunique() == EXPECTED_CONDITIONS, traces.scenario_id.nunique())
    check("paper_scale_only", set(runs.split) == {"paper_scale"}, sorted(runs.split.unique()))
    check("protected_rows_zero", not runs.protected.any(), int(runs.protected.sum()))
    check("e01_shared_and_extension_accounting", int(runs.upstream_scenario.sum()) == 1400 and int((~runs.upstream_scenario).sum()) == 200, [int(runs.upstream_scenario.sum()), int((~runs.upstream_scenario).sum())])

    parsed = composition.policy_counts_json.map(json.loads)
    check("all_compositions_sum_to_100", all(sum(item.values()) == 100 for item in parsed), None)
    exact_pairs = composition[
        composition.assignment_profile.eq("balanced_exact")
        & composition.policy_counts_json.map(lambda value: len(json.loads(value)) == 2)
    ]
    check("exact_pairwise_50_50", all(sorted(item.values()) == [50, 50] for item in exact_pairs.policy_counts_json.map(json.loads)), len(exact_pairs))
    exact_three = composition[
        composition.assignment_profile.eq("balanced_exact")
        & composition.policy_counts_json.map(lambda value: len(json.loads(value)) == 3)
    ]
    check("exact_three_way_34_33_33", len(exact_three) == 200 and all(sorted(item.values()) == [33, 33, 34] for item in exact_three.policy_counts_json.map(json.loads)), len(exact_three))
    random_rows = composition[composition.assignment_profile.eq("independent_random")]
    random_nonexact = sum(sorted(item.values()) not in ([50, 50], [33, 33, 34]) for item in random_rows.policy_counts_json.map(json.loads))
    check("random_assignments_retained", random_nonexact > 0, int(random_nonexact))
    check("composition_count_sums_valid", composition.policy_count_sum_valid.all(), int((~composition.policy_count_sum_valid).sum()))

    check("all_runs_complete", runs.completed.all(), runs.stop_reason.value_counts().to_dict())
    check("stop_reason_complete", set(runs.stop_reason) == {"complete"}, runs.stop_reason.value_counts().to_dict())
    check("final_consensus_ordered", runs.final_consensus_ordered.all(), int((~runs.final_consensus_ordered).sum()))
    check("value_multiset_conserved", runs.value_multiset_conserved.all(), int((~runs.value_multiset_conserved).sum()))
    check("event_budget_metadata_complete", runs.event_budget.eq(1_000_000).all() and (runs.activation_count <= runs.event_budget).all(), [int(runs.event_budget.min()), int(runs.event_budget.max()), int(runs.activation_count.max())])

    start = trajectories[trajectories.grid_index.eq(0)].set_index("scenario_id")
    end = trajectories[trajectories.grid_index.eq(100)].set_index("scenario_id")
    indexed = runs.set_index("scenario_id")
    check("trajectory_start_aggregation", np.allclose(start.loc[indexed.index].publication_aggregation, indexed.initial_publication_aggregation), None)
    check("trajectory_end_aggregation", np.allclose(end.loc[indexed.index].publication_aggregation, indexed.final_publication_aggregation), None)
    check("trajectory_end_sortedness", np.allclose(end.loc[indexed.index].paper_sortedness_percent, indexed.final_paper_sortedness_percent), None)

    check("deterministic_replay", replay["allPassed"] and len(replay["samples"]) == EXPECTED_CONDITIONS, [replay["allPassed"], len(replay["samples"])])
    fixture = _metric_fixture()
    check("metric_fixture", fixture["passed"], fixture["checks"])
    check("e01_scientific_parity", len(parity) == EXPECTED_SHARED_RUNS and parity.all_scientific_fields_equal.all(), [len(parity), int((~parity.all_scientific_fields_equal).sum())])
    check("start_state_expectation", composition_summary.mean_initial_minus_expectation.abs().le(0.025).all(), float(composition_summary.mean_initial_minus_expectation.abs().max()))
    check("peak_bootstrap_accounting", len(peaks) == EXPECTED_CONDITIONS and peaks.runs.eq(100).all() and peaks.bootstrap_draws.eq(BOOTSTRAP_DRAWS).all(), len(peaks))
    check("progress_height_robust", peaks.linear_minus_floor_peak.abs().le(0.01).all(), float(peaks.linear_minus_floor_peak.abs().max()))
    check("progress_timing_robust", peaks.linear_minus_floor_peak_progress_percent.abs().le(2).all(), int(peaks.linear_minus_floor_peak_progress_percent.abs().max()))
    figure_paths = [
        output / f"baseline_{profile}_trajectories.{suffix}"
        for profile in ("unique", "repeated")
        for suffix in ("png", "svg")
    ]
    check("baseline_figures_present", all(path.is_file() and path.stat().st_size > 0 for path in figure_paths), [path.name for path in figure_paths])

    failed = [name for name, value in checks.items() if not value["passed"]]
    value = {
        "schema": "e04.s01.validation_summary.v1",
        "researchStepId": "S01",
        "success": not failed,
        "checkCount": len(checks),
        "failedChecks": failed,
        "checks": checks,
    }
    write_json(output / "validation_summary.json", value)
    if failed:
        raise AssertionError(f"E04 S01 validation failed: {failed}")
    return value


def write_provenance(output: Path) -> dict[str, Any]:
    inputs = {
        "researchPlan": WORKSPACE / "RESEARCH_PLAN.md",
        "fullPlan": WORKSPACE / "FULL_PLAN.md",
        "agents": WORKSPACE / "AGENTS.md",
        "previousArtifactsMarkdown": WORKSPACE / "PREVIOUS_ARTIFACTS.md",
        "previousArtifactsJson": WORKSPACE / "PREVIOUS_ARTIFACTS.json",
        "attachmentManifest": WORKSPACE / "input-attachments/MANIFEST.json",
        "attachmentSidecar": WORKSPACE / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md",
        "paperMarkdown": WORKSPACE / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/pdf-markdown.md",
        "e01ScenarioBank": S08_DIR / "paired_scenario_bank.parquet",
        "e01ConditionCatalog": S08_DIR / "condition_catalog.parquet",
        "e01BaseDrawBank": S08_DIR / "base_draw_bank.parquet",
        "e01ChimericResults": S13_DIR / "chimeric_results.parquet",
        "e01ChimericTrajectories": S13_DIR / "chimeric_trajectories.parquet",
        "e01ReleaseManifest": UPSTREAM / "release/reference_simulator/release_manifest.json",
    }
    missing = [name for name, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"provenance inputs missing: {missing}")
    value = {
        "schema": "e04.s01.provenance.v1",
        "researchStepId": "S01",
        "createdUtc": datetime.now(timezone.utc).isoformat(),
        "repositoryHeadAtPackaging": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "backendProfile": RUN_PROFILE,
        "publicationSnapshotClaimed": False,
        "historicalMixedDriverExecuted": False,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in inputs.items()
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workers": 8,
            "threadEnvironment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "packages": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
                "matplotlib": matplotlib.__version__,
            },
        },
    }
    write_json(output / "provenance.json", value)
    write_json(output / "environment.json", value["environment"])
    return value


def write_artifact_manifest(output: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    value = {
        "schema": "e04.s01.artifact_manifest.v1",
        "researchStepId": "S01",
        "artifactCount": len(files),
        "artifacts": files,
    }
    write_json(output / "artifact_manifest.json", value)
    return value
