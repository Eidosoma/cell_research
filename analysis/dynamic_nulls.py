"""E04 S06 temporally coherent and peak-corrected trajectory nulls.

The scientific assumptions are frozen in ``s06_dynamic_null_contract.json``.
This module deliberately stops at S06: it replays the frozen S03/S04 population,
constructs four trajectory-level null families, and never modifies simulator
kinetics or starts S07.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import base64
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

from analysis.chimeric_replication import (
    AdmissibilityTracker,
    GRID,
    MetricTracker,
    _execute_s13_activation,
    canonical_hash,
)
from analysis.composition_sweep import (
    SweepCondition,
    build_conditions,
    load_base_draws,
    materialize_sweep_scenario,
)
from reference_simulator.engine import evaluate_terminal, initial_state
from reference_simulator.model import canonical_json_bytes, state_hash


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY / "analysis/s06_dynamic_null_contract.json"
S01_DIR = Path("/artifacts/research_steps/S01")
S02_DIR = Path("/artifacts/research_steps/S02")
S03_DIR = Path("/artifacts/research_steps/S03")
S04_DIR = Path("/artifacts/research_steps/S04")
S05_DIR = Path("/artifacts/research_steps/S05")
S07_DIR = Path("/artifacts/research_steps/S07")
EXPECTED_MANIFEST_HASHES = {
    "S01": "df9860285fc5b1ed9f7442fb68ea5e9ed9e407918d90988d0b3592f4b3ef8079",
    "S02": "9e632eedac59f6b08ff6b619ce2451a99b49c952319af2747076fd7bb3411b54",
    "S03": "85371d82a53096764b0de52029472a7978261ca5d4857f4d392f441db076f9e9",
    "S04": "90db3cb1afbb8e1ff53fb1d4fe0e04a4d21fb66ccd3a0b922738bade82d05641",
    "S05": "8a9528c5f14abb3598cc7e1f41a145978f7f7d47717c23497623f5c00579b4ce",
}
S04_TABLE = S04_DIR / "aggregation_metrics.parquet"
S04_TABLE_BYTES = 75_335_939
S04_TABLE_SHA256 = "97ce5985d9dc9b666f8c16345a2c678535037c53ad58867fc8305c975e6db647"
OUTPUT_SCHEMA = "e04.s06.dynamic_nulls.v1"
REPLAY_SCHEMA = "e04.s06.trajectory_replay.v1"
NULL_FAMILIES = (
    "label_permuted_global",
    "label_permuted_value_stratified",
    "mobility_matched",
    "policy_neutral_transport",
)
OUTCOMES = ("peak", "duration", "positive_area")
AUDIT_REPLICATES = tuple(range(0, 250, 10))
REFERENCE_DRAWS = 500
CALIBRATION_DRAWS = 12
TOTAL_DRAWS = REFERENCE_DRAWS + CALIBRATION_DRAWS
RETAINED_DRAWS = 50
EXPECTED_CONDITIONS = 186
EXPECTED_SCENARIOS = 4_650
EXPECTED_GRID_ROWS = EXPECTED_SCENARIOS * len(GRID)
EXPECTED_GROUP_TRAJECTORIES = (
    EXPECTED_CONDITIONS * len(NULL_FAMILIES) * TOTAL_DRAWS
)
POLICY_CODES = {"bubble": 0, "insertion": 1, "selection": 2}
POLICY_NAMES = {value: key for key, value in POLICY_CODES.items()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(_json_native(value)) + b"\n")


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd"
    )


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT
    ).strip()


def derive_seed(stream: str, *address: Any) -> int:
    payload = {
        "namespace": "E04/S06/dynamic_nulls/v1",
        "stream": stream,
        "address": address,
    }
    return int.from_bytes(
        hashlib.sha256(canonical_json_bytes(payload)).digest()[:16], "big"
    )


def _verify_upstream_manifest(directory: Path, expected_hash: str) -> dict[str, Any]:
    manifest_path = directory / "artifact_manifest.json"
    observed_manifest = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    checks = []
    for artifact in manifest["artifacts"]:
        path = directory / artifact["path"]
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            {
                "path": artifact["path"],
                "expectedSha256": artifact["sha256"],
                "observedSha256": observed,
                "passed": observed == artifact["sha256"],
            }
        )
    return {
        "manifestPath": str(manifest_path),
        "expectedManifestSha256": expected_hash,
        "observedManifestSha256": observed_manifest,
        "manifestPassed": observed_manifest == expected_hash,
        "artifactChecks": checks,
        "allPassed": observed_manifest == expected_hash
        and all(item["passed"] for item in checks),
    }


def _specification_markdown(contract_hash: str) -> str:
    return f"""# S06 frozen dynamic-null specification

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | S06 |
| Completion status | Design frozen before S06 trajectory replay; simulation not yet started at freeze time |
| Artifacts written | `preregistration.json`, `freeze_record.json`, and this specification |
| Validation result | Pre-simulation structural validation passed for 186 conditions, 4,650 paired scenarios, four null families, 512 trajectories per family/condition, and a 101-point grid |
| Outcome classification | Pending S06 execution |
| Caveats or blockers | Conditioning on value or mobility can remove part of the mechanism; the policy-neutral family is a transport-operator surrogate, not a new sorting policy |
| Recommended next action | Execute the frozen S06 replay and null simulation only; do not start S07 |

Contract SHA-256: `{contract_hash}`.

## Frozen estimand

The sole confirmatory metric is publication-denominator same-policy adjacency minus
the exact S02 composition expectation. The observed unit is a condition mean over
the 25 deterministic replicate ordinals `0,10,...,240`, evaluated at all 101
accepted-swap progress points. No upstream baseline is tuned to the paper.

For each condition and null family, draws 0--499 are inference references and
draws 500--511 are exchangeable pseudo-observations reserved for calibration.
Draws are paired by index across the same 25 scenarios. Full curves are retained
for reference draws 0--49; scalar endpoints are retained for every draw.

## Null hypotheses and exchangeability

1. **Global identity-label permutation.** Immutable policy labels are permuted
   across complete identity trajectories. Exact counts, values, positions,
   transport chronology, and length remain fixed; policy-value and policy-mobility
   associations are broken.
2. **Value-stratified identity-label permutation.** The same operation is limited
   to exact-value strata for repeated inputs and frozen value deciles for unique
   inputs. This is the S05 sensitivity and can over-condition away a real pathway.
3. **Mobility-matched identity-label permutation.** Labels are permuted within five
   equal-size identity strata ranked by grid-resolved cumulative displacement,
   intervals moved, and identity index. Mobility is post-treatment, so this is a
   sensitivity null rather than an automatically superior baseline.
4. **Policy-neutral transport surrogate.** One random position permutation is
   drawn per scenario trajectory. Every observed grid-to-grid unlabeled transport
   operator is conjugated by that same permutation and applied to the observed
   initial labels. This preserves operator chronology, cycle structure, fixed-point
   counts, moved-fraction sequence, and trajectory length while breaking the
   alignment of policy with spatial transport placement.

## Peak and persistence outcomes

- Peak is the maximum of the complete 101-point condition-mean corrected curve.
- Duration is the normalized progress measure above zero for the piecewise-linear
  curve, using interpolated zero crossings.
- Positive area is the integral of the positive part of that curve.
- Peak-adjusted effect is observed peak minus the mean of null maxima; its interval
  is observed minus the 97.5th and 2.5th null quantiles.
- Monte Carlo upper-tail p-values use `(1 + # null >= observed) / 501`.
- BH adjustment is across 186 conditions within family/outcome; a global all-test
  BH sensitivity is also reported. Null maxima supply within-trajectory
  multiplicity control.

## Validation and success gates

S06 requires exact S03 replay, exact agreement with all 469,650 selected S04 grid
values, 101 points for every observed and null trajectory, zero preservation and
pairing violations, exact transport moved-fraction/autocorrelation preservation,
calibrated pseudo-observation ranks, complete accounting, and deterministic rerun
checks. A supportive outcome additionally requires at least 4/6 balanced absent-
association anchors to retain a global-null peak q <= 0.05 with adjusted effect
above 0.02, at least 4/6 to retain duration or area q <= 0.05, and positive
maximum-selection inflation in at least 95% of global-null conditions.
"""


def freeze_design(output: Path, cache: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint = cache / "trajectory_replay.jsonl"
    null_dir = cache / "condition_nulls"
    if checkpoint.exists() or (null_dir.exists() and any(null_dir.iterdir())):
        raise FileExistsError("cannot freeze S06 after a replay/null checkpoint exists")
    if S07_DIR.exists():
        raise AssertionError("S07 artifacts exist before S06 freeze")
    contract_hash = sha256_file(CONTRACT_PATH)
    contract = json.loads(CONTRACT_PATH.read_text())
    if contract["scope"]["conditions"] != EXPECTED_CONDITIONS:
        raise AssertionError("contract condition count changed")
    tasks = build_replay_tasks()
    upstream = {
        step: _verify_upstream_manifest(
            Path(f"/artifacts/research_steps/{step}"), expected
        )
        for step, expected in EXPECTED_MANIFEST_HASHES.items()
    }
    if not all(item["allPassed"] for item in upstream.values()):
        raise AssertionError("upstream artifact immutability failed before freeze")
    s04_gate = {
        "path": str(S04_TABLE),
        "bytes": S04_TABLE.stat().st_size,
        "sha256": sha256_file(S04_TABLE),
    }
    s04_gate["passed"] = (
        s04_gate["bytes"] == S04_TABLE_BYTES
        and s04_gate["sha256"] == S04_TABLE_SHA256
    )
    if not s04_gate["passed"]:
        raise AssertionError("canonical S04 table failed freeze gate")
    record = {
        "schema": "e04.s06.freeze_record.v1",
        "researchStepId": "S06",
        "frozenAt": datetime.now(timezone.utc).isoformat(),
        "frozenBeforeSimulation": True,
        "contractPath": str(CONTRACT_PATH),
        "contractSha256": contract_hash,
        "conditionCount": EXPECTED_CONDITIONS,
        "scenarioCount": len(tasks),
        "gridPoints": len(GRID),
        "nullFamilies": list(NULL_FAMILIES),
        "referenceDraws": REFERENCE_DRAWS,
        "calibrationDraws": CALIBRATION_DRAWS,
        "totalGroupNullTrajectories": EXPECTED_GROUP_TRAJECTORIES,
        "upstream": upstream,
        "s04CanonicalTableGate": s04_gate,
        "s07Absent": not S07_DIR.exists(),
    }
    write_json(output / "preregistration.json", contract)
    write_json(output / "freeze_record.json", record)
    (output / "dynamic_null_specification.md").write_text(
        _specification_markdown(contract_hash), encoding="utf-8"
    )
    return record


def assert_frozen(output: Path) -> dict[str, Any]:
    record = json.loads((output / "freeze_record.json").read_text())
    if record["contractSha256"] != sha256_file(CONTRACT_PATH):
        raise AssertionError("S06 frozen contract changed")
    if not record["frozenBeforeSimulation"]:
        raise AssertionError("S06 was not frozen before simulation")
    if record["scenarioCount"] != EXPECTED_SCENARIOS:
        raise AssertionError("S06 frozen population changed")
    if S07_DIR.exists():
        raise AssertionError("S07 artifacts appeared during S06")
    return record


def _selected_expected() -> dict[str, dict[str, Any]]:
    columns = [
        "scenario_id",
        "run_id",
        "stop_reason",
        "successful_swap_count",
        "activation_count",
        "final_state_hash",
        "trajectory_sha256",
        "scenario_json_sha256",
        "condition_id",
        "input_profile",
        "policy_set_label",
        "composition_class",
        "composition_profile",
        "first_policy_count",
        "rare_policy",
        "correlation_profile",
        "base_draw_id",
        "pairing_block_id",
        "replicate_ordinal",
        "policy_counts_json",
        "publication_aggregation_null",
    ]
    rows = pq.read_table(S03_DIR / "composition_sweep.parquet", columns=columns).to_pylist()
    selected = {
        str(row["scenario_id"]): row
        for row in rows
        if int(row["replicate_ordinal"]) in AUDIT_REPLICATES
    }
    if len(selected) != EXPECTED_SCENARIOS:
        raise AssertionError("selected S03 population has wrong cardinality")
    return selected


def build_replay_tasks() -> list[dict[str, Any]]:
    bases = load_base_draws()
    tasks: list[dict[str, Any]] = []
    for condition in build_conditions():
        for replicate in AUDIT_REPLICATES:
            base = bases[(condition.input_profile, replicate)]
            scenario, metadata = materialize_sweep_scenario(condition, base)
            tasks.append(
                {
                    "condition": condition.to_dict(),
                    "base": dict(base),
                    "scenarioId": scenario.scenario_id,
                    "scenarioJsonSha256": metadata["scenarioJsonSha256"],
                    "metadata": metadata,
                }
            )
    tasks.sort(
        key=lambda item: (
            item["condition"]["conditionId"],
            int(item["base"]["replicateOrdinal"]),
        )
    )
    if len(tasks) != EXPECTED_SCENARIOS:
        raise AssertionError("S06 replay task count changed")
    return tasks


def _encode_array(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _decode_array(value: str, dtype: np.dtype[Any], shape: tuple[int, ...]) -> np.ndarray:
    array = np.frombuffer(base64.b64decode(value), dtype=dtype)
    if array.size != math.prod(shape):
        raise ValueError("encoded array size does not match expected shape")
    return array.reshape(shape).copy()


def _replay_worker(task: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    condition = SweepCondition.from_dict(task["condition"])
    scenario, metadata = materialize_sweep_scenario(condition, task["base"])
    if scenario.scenario_id != task["scenarioId"]:
        raise AssertionError("S06 scenario rematerialization changed ID")
    if metadata["scenarioJsonSha256"] != expected["scenario_json_sha256"]:
        raise AssertionError("S06 scenario JSON differs from S03")
    final_swaps = int(expected["successful_swap_count"])
    targets: dict[int, list[int]] = defaultdict(list)
    for grid_index, progress in enumerate(GRID):
        targets[math.floor(progress * final_swaps)].append(grid_index)
    state = initial_state(scenario)
    state.terminal = evaluate_terminal(scenario, state)
    tracker = MetricTracker(scenario, False)
    admissibility = AdmissibilityTracker(scenario, state)
    original_points: list[dict[str, Any]] = [{"swap_index": 0, **tracker.metrics()}]
    occupancy = np.empty((len(GRID), len(scenario.cells)), dtype=np.uint8)
    swap_indices = np.empty(len(GRID), dtype=np.uint32)
    captured = np.zeros(len(GRID), dtype=bool)
    id_to_index = {cell.cell_id: index for index, cell in enumerate(scenario.cells)}

    def capture(swap_index: int) -> None:
        encoded = np.fromiter(
            (id_to_index[cell_id] for cell_id in state.occupancy),
            dtype=np.uint8,
            count=len(scenario.cells),
        )
        for grid_index in targets.get(swap_index, ()):
            occupancy[grid_index] = encoded
            swap_indices[grid_index] = swap_index
            captured[grid_index] = True

    capture(0)
    swap_committed = False

    def on_swap(current: Any, first: int, second: int) -> None:
        nonlocal swap_committed
        swap_committed = True
        tracker.after_swap(current, first, second)
        admissibility.after_swap(current, first, second)
        swap_index = int(current.ledger["acceptedSwaps"]) + 1
        original_points.append({"swap_index": swap_index, **tracker.metrics()})
        if swap_index in targets:
            capture(swap_index)

    started = time.perf_counter()
    while state.terminal is None:
        swap_committed = False
        outcome = _execute_s13_activation(
            scenario, state, admissibility, on_swap=on_swap
        )
        if outcome != "noop":
            if not swap_committed:
                admissibility.after_memory_update(state)
            state.terminal = admissibility.terminal_after_change(state)
    elapsed = time.perf_counter() - started
    official = evaluate_terminal(scenario, state)
    if official != state.terminal:
        raise AssertionError("incremental and official S06 terminals differ")
    if not captured.all():
        raise AssertionError("S06 failed to capture all 101 grid points")
    observed = {
        "stop_reason": state.terminal,
        "successful_swap_count": int(state.ledger["acceptedSwaps"]),
        "activation_count": int(state.activation_count),
        "final_state_hash": state_hash(scenario.scenario_id, state),
        "trajectory_sha256": canonical_hash(original_points),
    }
    replay_checks = {
        key: observed[key] == expected[key]
        for key in (
            "stop_reason",
            "successful_swap_count",
            "activation_count",
            "final_state_hash",
            "trajectory_sha256",
        )
    }
    if not all(replay_checks.values()):
        raise AssertionError(f"S06/S03 replay mismatch: {replay_checks}")
    labels = np.asarray(
        [POLICY_CODES[cell.policy.value.lower()] for cell in scenario.cells],
        dtype=np.uint8,
    )
    values = np.asarray([cell.value for cell in scenario.cells], dtype=np.int16)
    baseline = float(expected["publication_aggregation_null"])
    placed = labels[occupancy]
    corrected = (placed[:, :-1] == placed[:, 1:]).sum(axis=1) / len(labels) - baseline
    metadata_fields = {
        key: expected[key]
        for key in (
            "run_id",
            "condition_id",
            "input_profile",
            "policy_set_label",
            "composition_class",
            "composition_profile",
            "first_policy_count",
            "rare_policy",
            "correlation_profile",
            "base_draw_id",
            "pairing_block_id",
            "replicate_ordinal",
            "policy_counts_json",
        )
    }
    return {
        "schema": REPLAY_SCHEMA,
        "scenario_id": scenario.scenario_id,
        "scenario_json_sha256": metadata["scenarioJsonSha256"],
        "metadata": metadata_fields,
        "stop_reason": state.terminal,
        "successful_swap_count": observed["successful_swap_count"],
        "activation_count": observed["activation_count"],
        "final_state_hash": observed["final_state_hash"],
        "original_trajectory_sha256": observed["trajectory_sha256"],
        "replay_checks": replay_checks,
        "baseline": baseline,
        "identity_labels_b64": _encode_array(labels),
        "identity_values_b64": _encode_array(values),
        "occupancy_b64": _encode_array(occupancy),
        "swap_indices_b64": _encode_array(swap_indices),
        "observed_corrected_b64": _encode_array(corrected.astype(np.float64)),
        "elapsed_seconds": elapsed,
    }


def _read_replay_ids(checkpoint: Path) -> set[str]:
    if not checkpoint.exists():
        return set()
    result: set[str] = set()
    with checkpoint.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record["schema"] != REPLAY_SCHEMA:
                raise ValueError("S06 replay checkpoint schema changed")
            scenario_id = str(record["scenario_id"])
            if scenario_id in result:
                raise ValueError(f"duplicate replay row at line {line_number}")
            result.add(scenario_id)
    return result


def run_replay(
    tasks: Sequence[Mapping[str, Any]], checkpoint: Path, *, workers: int = 8
) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1,8]")
    expected = _selected_expected()
    task_ids = {str(task["scenarioId"]) for task in tasks}
    if task_ids != set(expected):
        raise AssertionError("S06 task population differs from frozen S03 subset")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    complete = _read_replay_ids(checkpoint)
    if not complete <= task_ids:
        raise ValueError("S06 replay checkpoint contains out-of-scope scenario")
    pending = [task for task in tasks if str(task["scenarioId"]) not in complete]
    iterator = iter(pending)
    started = time.perf_counter()
    written = 0
    with checkpoint.open("a", encoding="utf-8") as handle:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            active: dict[Any, Mapping[str, Any]] = {}
            for _ in range(min(workers * 3, len(pending))):
                task = next(iterator)
                scenario_id = str(task["scenarioId"])
                active[executor.submit(_replay_worker, task, expected[scenario_id])] = task
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    active.pop(future)
                    result = future.result()
                    handle.write(
                        json.dumps(_json_native(result), sort_keys=True, separators=(",", ":"))
                        + "\n"
                    )
                    handle.flush()
                    complete.add(str(result["scenario_id"]))
                    written += 1
                    if written % 250 == 0:
                        print(
                            json.dumps(
                                {
                                    "completed": len(complete),
                                    "pending": EXPECTED_SCENARIOS - len(complete),
                                    "elapsedSeconds": time.perf_counter() - started,
                                }
                            ),
                            flush=True,
                        )
                    try:
                        task = next(iterator)
                    except StopIteration:
                        continue
                    scenario_id = str(task["scenarioId"])
                    active[executor.submit(_replay_worker, task, expected[scenario_id])] = task
    if len(complete) != EXPECTED_SCENARIOS:
        raise AssertionError("S06 replay checkpoint incomplete")
    return {
        "scenarioCount": len(complete),
        "newScenarios": written,
        "gridRows": len(complete) * len(GRID),
        "workers": workers,
        "elapsedSeconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
    }


def replay_records(checkpoint: Path) -> Iterable[dict[str, Any]]:
    with checkpoint.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _decode_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **record,
        "labels": _decode_array(record["identity_labels_b64"], np.dtype("uint8"), (100,)),
        "values": _decode_array(record["identity_values_b64"], np.dtype("int16"), (100,)),
        "occupancy": _decode_array(
            record["occupancy_b64"], np.dtype("uint8"), (len(GRID), 100)
        ),
        "swap_indices": _decode_array(
            record["swap_indices_b64"], np.dtype("uint32"), (len(GRID),)
        ),
        "observed_corrected": _decode_array(
            record["observed_corrected_b64"], np.dtype("float64"), (len(GRID),)
        ),
    }


def package_replay_audit(checkpoint: Path, output: Path) -> dict[str, Any]:
    s04 = pq.read_table(
        S04_TABLE,
        columns=[
            "scenario_id",
            "replicate_ordinal",
            "grid_index",
            "swap_index",
            "corrected_publication_aggregation",
        ],
        filters=[("replicate_ordinal", "in", list(AUDIT_REPLICATES))],
    ).to_pandas()
    expected_by_scenario = {
        scenario: group.sort_values("grid_index")
        for scenario, group in s04.groupby("scenario_id", sort=False)
    }
    rows = []
    max_metric_error = 0.0
    max_swap_error = 0
    replay_passes = 0
    pairing: dict[str, list[str]] = defaultdict(list)
    movement_rows = []
    for raw in replay_records(checkpoint):
        record = _decode_record(raw)
        canonical = expected_by_scenario[str(record["scenario_id"])]
        if len(canonical) != len(GRID):
            raise AssertionError("selected S04 scenario lacks 101 points")
        metric_error = float(
            np.max(
                np.abs(
                    canonical.corrected_publication_aggregation.to_numpy(dtype=float)
                    - record["observed_corrected"]
                )
            )
        )
        swap_error = int(
            np.max(
                np.abs(
                    canonical.swap_index.to_numpy(dtype=np.int64)
                    - record["swap_indices"].astype(np.int64)
                )
            )
        )
        max_metric_error = max(max_metric_error, metric_error)
        max_swap_error = max(max_swap_error, swap_error)
        replay_passes += int(all(record["replay_checks"].values()))
        condition_id = str(record["metadata"]["condition_id"])
        pairing[condition_id].append(str(record["scenario_id"]))
        positions = np.argsort(record["occupancy"], axis=1)
        moved_fraction = np.mean(np.diff(positions, axis=0) != 0, axis=1)
        movement_rows.append(
            {
                "condition_id": condition_id,
                "scenario_id": record["scenario_id"],
                "moved_fraction_mean": float(np.mean(moved_fraction)),
                "moved_fraction_lag1_acf": _acf_1d(moved_fraction, 1),
                "moved_fraction_lag5_acf": _acf_1d(moved_fraction, 5),
            }
        )
        rows.append(
            {
                "schema_version": "e04.s06.replay_audit.v1",
                "research_step_id": "S06",
                "scenario_id": record["scenario_id"],
                **record["metadata"],
                "successful_swap_count": record["successful_swap_count"],
                "activation_count": record["activation_count"],
                "trajectory_length": len(GRID),
                "all_s03_replay_checks_passed": all(record["replay_checks"].values()),
                "s04_max_metric_error": metric_error,
                "s04_max_swap_index_error": swap_error,
                "baseline": record["baseline"],
            }
        )
    frame = pd.DataFrame(rows).sort_values(["condition_id", "replicate_ordinal"])
    _write_parquet(frame, output / "trajectory_replay_audit.parquet")
    _write_parquet(
        pd.DataFrame(movement_rows).sort_values(["condition_id", "scenario_id"]),
        output / "movement_autocorrelation_audit.parquet",
    )
    pairing_hashes = {
        condition: hashlib.sha256(canonical_json_bytes(sorted(scenarios))).hexdigest()
        for condition, scenarios in sorted(pairing.items())
    }
    summary = {
        "schema": "e04.s06.replay_summary.v1",
        "scenarioCount": len(frame),
        "conditionCount": int(frame.condition_id.nunique()),
        "runsPerConditionMin": int(frame.groupby("condition_id").size().min()),
        "runsPerConditionMax": int(frame.groupby("condition_id").size().max()),
        "gridRows": len(frame) * len(GRID),
        "s03ReplayPasses": replay_passes,
        "s04MatchedGridRows": int(len(frame) * len(GRID)),
        "maxS04MetricError": max_metric_error,
        "maxS04SwapIndexError": max_swap_error,
        "pairingHashes": pairing_hashes,
        "allPassed": bool(
            len(frame) == EXPECTED_SCENARIOS
            and frame.condition_id.nunique() == EXPECTED_CONDITIONS
            and frame.groupby("condition_id").size().eq(len(AUDIT_REPLICATES)).all()
            and replay_passes == EXPECTED_SCENARIOS
            and max_metric_error == 0
            and max_swap_error == 0
        ),
    }
    if not summary["allPassed"]:
        raise AssertionError("S06 replay/S04 audit failed")
    write_json(output / "trajectory_replay_summary.json", summary)
    return summary


def _acf_1d(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= lag:
        return float("nan")
    left = values[:-lag]
    right = values[lag:]
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    if denominator == 0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def _acf_rows(values: np.ndarray, lag: int) -> np.ndarray:
    left = values[:, :-lag].astype(np.float64, copy=False)
    right = values[:, lag:].astype(np.float64, copy=False)
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
    numerator = np.sum(left * right, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(values), np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def trajectory_outcomes(curves: np.ndarray) -> dict[str, np.ndarray]:
    curves = np.atleast_2d(np.asarray(curves, dtype=np.float64))
    if curves.shape[1] != len(GRID):
        raise ValueError("trajectory outcomes require the frozen 101-point grid")
    y0 = curves[:, :-1]
    y1 = curves[:, 1:]
    dt = float(GRID[1] - GRID[0])
    both_positive = (y0 > 0) & (y1 > 0)
    start_positive = (y0 > 0) & (y1 <= 0)
    end_positive = (y0 <= 0) & (y1 > 0)
    duration = both_positive.sum(axis=1, dtype=np.float64) * dt
    duration += np.where(
        start_positive,
        dt * np.divide(y0, y0 - y1, out=np.zeros_like(y0), where=(y0 - y1) != 0),
        0,
    ).sum(axis=1)
    duration += np.where(
        end_positive,
        dt * np.divide(y1, y1 - y0, out=np.zeros_like(y1), where=(y1 - y0) != 0),
        0,
    ).sum(axis=1)
    area = np.where(both_positive, dt * (y0 + y1) / 2, 0).sum(axis=1)
    area += np.where(
        start_positive,
        0.5
        * y0
        * dt
        * np.divide(y0, y0 - y1, out=np.zeros_like(y0), where=(y0 - y1) != 0),
        0,
    ).sum(axis=1)
    area += np.where(
        end_positive,
        0.5
        * y1
        * dt
        * np.divide(y1, y1 - y0, out=np.zeros_like(y1), where=(y1 - y0) != 0),
        0,
    ).sum(axis=1)
    return {
        "peak": np.max(curves, axis=1),
        "duration": duration,
        "positive_area": area,
        "lag1_acf": _acf_rows(curves, 1),
        "lag5_acf": _acf_rows(curves, 5),
    }


def _value_strata(values: np.ndarray, input_profile: str) -> np.ndarray:
    if input_profile == "repeated_1_10_x10":
        return values.astype(np.int16)
    if input_profile == "unique_1_100":
        return ((values.astype(np.int16) - 1) // 10).astype(np.int16)
    raise ValueError(f"unknown input profile {input_profile}")


def mobility_strata(occupancy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.argsort(occupancy, axis=1)
    differences = np.diff(positions.astype(np.int16), axis=0)
    displacement = np.sum(np.abs(differences), axis=0, dtype=np.int64)
    intervals_moved = np.sum(differences != 0, axis=0, dtype=np.int64)
    identity = np.arange(occupancy.shape[1])
    order = np.lexsort((identity, intervals_moved, displacement))
    strata = np.empty(occupancy.shape[1], dtype=np.uint8)
    strata[order] = np.arange(occupancy.shape[1], dtype=np.uint8) // 20
    return strata, displacement, intervals_moved


def _permuted_labels(labels: np.ndarray, strata: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    output = np.broadcast_to(labels, (TOTAL_DRAWS, len(labels))).copy()
    if strata is None:
        return rng.permuted(output, axis=1)
    for stratum in np.unique(strata):
        indices = np.flatnonzero(strata == stratum)
        output[:, indices] = rng.permuted(output[:, indices], axis=1)
    return output


def _label_curves(
    labels_by_identity: np.ndarray, occupancy: np.ndarray, baseline: float
) -> np.ndarray:
    placed = labels_by_identity[:, occupancy]
    counts = np.sum(placed[:, :, :-1] == placed[:, :, 1:], axis=2, dtype=np.int16)
    return counts.astype(np.float64) / occupancy.shape[1] - baseline


def _policy_neutral_curves(
    labels: np.ndarray,
    occupancy: np.ndarray,
    baseline: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, float]:
    n = len(labels)
    permutations = rng.permuted(
        np.broadcast_to(np.arange(n, dtype=np.uint8), (TOTAL_DRAWS, n)).copy(),
        axis=1,
    )
    inverse = np.empty_like(permutations)
    inverse[np.arange(TOTAL_DRAWS)[:, None], permutations] = np.arange(n, dtype=np.uint8)
    positions = np.argsort(occupancy, axis=1).astype(np.uint8)
    initial_ids = occupancy[0]
    initial_labels = labels[initial_ids]
    rows = np.arange(TOTAL_DRAWS)[:, None]
    result = np.empty((TOTAL_DRAWS, len(GRID)), dtype=np.float64)
    cycle_violations = 0
    for grid_index in range(len(GRID)):
        cumulative = positions[grid_index, initial_ids]
        original_sources = inverse
        original_targets = cumulative[original_sources]
        conjugated_targets = permutations[rows, original_targets]
        states = np.empty((TOTAL_DRAWS, n), dtype=np.uint8)
        states[rows, conjugated_targets] = initial_labels
        result[:, grid_index] = (
            np.sum(states[:, :-1] == states[:, 1:], axis=1, dtype=np.int16) / n
            - baseline
        )
    moved_fraction_max_error = 0.0
    for grid_index in range(1, len(GRID)):
        transition = positions[grid_index, occupancy[grid_index - 1]]
        original_targets = transition[inverse[0]]
        conjugated = permutations[0, original_targets]
        original_moved = float(np.mean(transition != np.arange(n)))
        conjugated_moved = float(np.mean(conjugated != np.arange(n)))
        moved_fraction_max_error = max(
            moved_fraction_max_error, abs(original_moved - conjugated_moved)
        )
        cycle_violations += int(
            np.sum(conjugated == np.arange(n))
            != np.sum(transition == np.arange(n))
        )
        if grid_index in (1, 25, 50, 75, 100):
            cycle_violations += int(
                _cycle_signature(conjugated) != _cycle_signature(transition)
            )
    return result, cycle_violations, moved_fraction_max_error


def _cycle_signature(mapping: np.ndarray) -> tuple[int, ...]:
    mapping = np.asarray(mapping, dtype=np.int64)
    seen = np.zeros(len(mapping), dtype=bool)
    lengths: list[int] = []
    for start in range(len(mapping)):
        if seen[start]:
            continue
        cursor = start
        length = 0
        while not seen[cursor]:
            seen[cursor] = True
            cursor = int(mapping[cursor])
            length += 1
        lengths.append(length)
    return tuple(sorted(lengths))


def _support_log(labels: np.ndarray, strata: np.ndarray) -> tuple[float, int]:
    total = 0.0
    mutable = 0
    for stratum in np.unique(strata):
        subset = labels[strata == stratum]
        _, counts = np.unique(subset, return_counts=True)
        total += math.lgamma(len(subset) + 1) - sum(
            math.lgamma(int(c) + 1) for c in counts
        )
        mutable += int(len(counts) > 1)
    return total, mutable


def _condition_file(null_dir: Path, condition_id: str) -> Path:
    digest = hashlib.sha256(condition_id.encode()).hexdigest()[:20]
    return null_dir / f"{digest}.npz"


def _condition_null_worker(
    condition_id: str,
    raw_records: Sequence[Mapping[str, Any]],
    null_dir: Path,
) -> dict[str, Any]:
    records = [_decode_record(record) for record in raw_records]
    records.sort(key=lambda item: int(item["metadata"]["replicate_ordinal"]))
    if len(records) != len(AUDIT_REPLICATES):
        raise AssertionError("condition null group lacks 25 scenarios")
    if any(str(item["metadata"]["condition_id"]) != condition_id for item in records):
        raise AssertionError("condition null group contains mixed conditions")
    family_curves = {
        family: np.zeros((TOTAL_DRAWS, len(GRID)), dtype=np.float64)
        for family in NULL_FAMILIES
    }
    observed = np.mean(
        np.stack([item["observed_corrected"] for item in records], axis=0), axis=0
    )
    value_support = []
    mobility_support = []
    cycle_violations = 0
    count_violations = 0
    moved_fraction_errors = []
    for item in records:
        labels = item["labels"]
        values = item["values"]
        occupancy = item["occupancy"]
        baseline = float(item["baseline"])
        input_profile = str(item["metadata"]["input_profile"])
        value_strata = _value_strata(values, input_profile)
        movement_strata, _, _ = mobility_strata(occupancy)
        value_log, value_mutable = _support_log(labels, value_strata)
        mobility_log, mobility_mutable = _support_log(labels, movement_strata)
        value_support.append((value_log, value_mutable))
        mobility_support.append((mobility_log, mobility_mutable))

        for family, strata in (
            ("label_permuted_global", None),
            ("label_permuted_value_stratified", value_strata),
            ("mobility_matched", movement_strata),
        ):
            rng = np.random.Generator(
                np.random.PCG64DXSM(derive_seed("labels", condition_id, item["scenario_id"], family))
            )
            permuted = _permuted_labels(labels, strata, rng)
            expected_counts = np.bincount(labels, minlength=3)
            observed_counts = np.stack(
                [np.sum(permuted == code, axis=1) for code in range(3)], axis=1
            )
            count_violations += int(np.sum(observed_counts != expected_counts))
            if strata is not None:
                for stratum in np.unique(strata):
                    indices = np.flatnonzero(strata == stratum)
                    target = np.bincount(labels[indices], minlength=3)
                    realized = np.stack(
                        [
                            np.sum(permuted[:, indices] == code, axis=1)
                            for code in range(3)
                        ],
                        axis=1,
                    )
                    count_violations += int(np.sum(realized != target))
            family_curves[family] += _label_curves(permuted, occupancy, baseline)

        rng = np.random.Generator(
            np.random.PCG64DXSM(
                derive_seed(
                    "transport", condition_id, item["scenario_id"], "policy_neutral_transport"
                )
            )
        )
        neutral, violations, moved_error = _policy_neutral_curves(
            labels, occupancy, baseline, rng
        )
        family_curves["policy_neutral_transport"] += neutral
        cycle_violations += violations
        positions = np.argsort(occupancy, axis=1)
        moved = np.mean(np.diff(positions, axis=0) != 0, axis=1)
        # Conjugation preserves fixed points and therefore interval moved fraction exactly.
        moved_fraction_errors.append(
            max(moved_error, 0.0 if len(moved) == 100 else 1.0)
        )

    for family in NULL_FAMILIES:
        family_curves[family] /= len(records)
    output_path = _condition_file(null_dir, condition_id)
    null_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        condition_id=np.asarray(condition_id),
        families=np.asarray(NULL_FAMILIES),
        curves=np.stack([family_curves[family] for family in NULL_FAMILIES]),
        observed=observed,
        scenario_ids=np.asarray([item["scenario_id"] for item in records]),
        replicate_ordinals=np.asarray(
            [item["metadata"]["replicate_ordinal"] for item in records], dtype=np.int16
        ),
        metadata_json=np.asarray(json.dumps(records[0]["metadata"], sort_keys=True)),
        value_support=np.asarray(value_support, dtype=np.float64),
        mobility_support=np.asarray(mobility_support, dtype=np.float64),
        cycle_violations=np.asarray(cycle_violations, dtype=np.int64),
        count_violations=np.asarray(count_violations, dtype=np.int64),
        moved_fraction_max_error=np.asarray(max(moved_fraction_errors), dtype=np.float64),
    )
    return {
        "conditionId": condition_id,
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "scenarioCount": len(records),
        "groupTrajectories": len(NULL_FAMILIES) * TOTAL_DRAWS,
        "countViolations": count_violations,
        "cycleViolations": cycle_violations,
        "movedFractionMaxError": max(moved_fraction_errors),
    }


def _group_replay_records(checkpoint: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in replay_records(checkpoint):
        grouped[str(record["metadata"]["condition_id"])].append(record)
    if len(grouped) != EXPECTED_CONDITIONS:
        raise AssertionError("replay checkpoint has wrong condition count")
    return grouped


def run_condition_nulls(
    checkpoint: Path, null_dir: Path, *, workers: int = 8
) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1,8]")
    grouped = _group_replay_records(checkpoint)
    expected_paths = {condition: _condition_file(null_dir, condition) for condition in grouped}
    pending = [condition for condition, path in expected_paths.items() if not path.exists()]
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _condition_null_worker, condition, grouped[condition], null_dir
            ): condition
            for condition in pending
        }
        completed = len(grouped) - len(pending)
        for future in futures:
            result = future.result()
            results.append(result)
            completed += 1
            print(
                json.dumps(
                    {
                        "completedConditions": completed,
                        "totalConditions": EXPECTED_CONDITIONS,
                        "elapsedSeconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    files = list(expected_paths.values())
    if any(not path.is_file() for path in files):
        raise AssertionError("condition-null checkpoint incomplete")
    return {
        "conditionCount": len(files),
        "newConditions": len(results),
        "groupNullTrajectories": EXPECTED_GROUP_TRAJECTORIES,
        "scenarioNullPaths": EXPECTED_GROUP_TRAJECTORIES * len(AUDIT_REPLICATES),
        "gridMetricEvaluations": EXPECTED_GROUP_TRAJECTORIES
        * len(AUDIT_REPLICATES)
        * len(GRID),
        "workers": workers,
        "elapsedSeconds": time.perf_counter() - started,
        "nullDirectory": str(null_dir),
    }


def _bh_adjust(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    result = np.full(len(array), np.nan, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(array))
    if len(finite) == 0:
        return result
    order = finite[np.argsort(array[finite], kind="mergesort")]
    ranked = array[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _conservative_p(reference: np.ndarray, observed: float) -> float:
    return float((1 + np.sum(reference >= observed)) / (len(reference) + 1))


def _randomized_p(
    reference: np.ndarray, observed: float, family: str, condition_id: str, outcome: str, index: int
) -> float:
    greater = int(np.sum(reference > observed))
    equal = int(np.sum(reference == observed))
    rng = np.random.Generator(
        np.random.PCG64DXSM(
            derive_seed("calibration_ties", family, condition_id, outcome, index)
        )
    )
    return float((greater + rng.random() * (equal + 1)) / (len(reference) + 1))


def _condition_files(null_dir: Path) -> list[Path]:
    files = sorted(null_dir.glob("*.npz"))
    if len(files) != EXPECTED_CONDITIONS:
        raise AssertionError(
            f"expected {EXPECTED_CONDITIONS} condition null files, found {len(files)}"
        )
    return files


class _ParquetAppender:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        else:
            table = table.cast(self.writer.schema)
        self.writer.write_table(table)
        self.rows += len(frame)

    def close(self) -> None:
        if self.writer is None:
            raise AssertionError(f"no rows written to {self.path}")
        self.writer.close()


def package_dynamic_nulls(null_dir: Path, output: Path) -> dict[str, Any]:
    assert_frozen(output)
    distribution_dir = output / "null_distributions"
    distribution_dir.mkdir(parents=True, exist_ok=True)
    scalar_writer = _ParquetAppender(
        distribution_dir / "dynamic_null_statistics.parquet"
    )
    retained_writer = _ParquetAppender(
        distribution_dir / "retained_null_trajectories.parquet"
    )
    response_writer = _ParquetAppender(
        distribution_dir / "null_response_surface.parquet"
    )
    observed_writer = _ParquetAppender(output / "observed_dynamic_trajectories.parquet")
    replay = pd.read_parquet(output / "trajectory_replay_audit.parquet")
    baseline_by_condition = replay.groupby("condition_id").baseline.mean().to_dict()
    effect_rows: list[dict[str, Any]] = []
    observed_rows: list[dict[str, Any]] = []
    calibration_values: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    temporal_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    preservation_rows: list[dict[str, Any]] = []
    condition_manifest_rows: list[dict[str, Any]] = []
    scalar_count = 0
    retained_count = 0
    response_count = 0

    for path in _condition_files(null_dir):
        with np.load(path, allow_pickle=False) as data:
            condition_id = str(data["condition_id"].item())
            families = tuple(str(value) for value in data["families"].tolist())
            if families != NULL_FAMILIES:
                raise AssertionError("null family order changed")
            curves = data["curves"].astype(np.float64)
            observed = data["observed"].astype(np.float64)
            scenario_ids = [str(value) for value in data["scenario_ids"].tolist()]
            replicate_ordinals = data["replicate_ordinals"].astype(int)
            metadata = json.loads(str(data["metadata_json"].item()))
            value_support = data["value_support"].astype(float)
            mobility_support = data["mobility_support"].astype(float)
            cycle_violations = int(data["cycle_violations"].item())
            count_violations = int(data["count_violations"].item())
            moved_error = float(data["moved_fraction_max_error"].item())
        if curves.shape != (len(NULL_FAMILIES), TOTAL_DRAWS, len(GRID)):
            raise AssertionError("condition null array shape changed")
        if len(scenario_ids) != len(AUDIT_REPLICATES):
            raise AssertionError("condition pairing count changed")
        pairing_hash = hashlib.sha256(
            canonical_json_bytes(sorted(scenario_ids))
        ).hexdigest()
        meta = {
            "condition_id": condition_id,
            "input_profile": metadata["input_profile"],
            "policy_set_label": metadata["policy_set_label"],
            "composition_class": metadata["composition_class"],
            "composition_profile": metadata["composition_profile"],
            "first_policy_count": metadata["first_policy_count"],
            "rare_policy": metadata["rare_policy"] or "not_applicable",
            "correlation_profile": metadata["correlation_profile"],
        }
        observed_metrics = trajectory_outcomes(observed)
        observed_scalar = {key: float(value[0]) for key, value in observed_metrics.items()}
        observed_rows.append(
            {
                "schema_version": OUTPUT_SCHEMA,
                "research_step_id": "S06",
                **meta,
                "scenario_count": len(scenario_ids),
                "pairing_hash": pairing_hash,
                "composition_baseline": float(baseline_by_condition[condition_id]),
                **observed_scalar,
            }
        )
        observed_writer.write(
            pd.DataFrame(
                {
                    "schema_version": OUTPUT_SCHEMA,
                    "research_step_id": "S06",
                    **{key: [value] * len(GRID) for key, value in meta.items()},
                    "scenario_count": len(scenario_ids),
                    "pairing_hash": pairing_hash,
                    "grid_index": np.arange(len(GRID), dtype=np.int16),
                    "accepted_swap_progress": GRID,
                    "corrected_publication_aggregation": observed,
                }
            )
        )
        condition_manifest_rows.append(
            {
                **meta,
                "pairing_hash": pairing_hash,
                "scenario_count": len(scenario_ids),
                "replicate_ordinals_json": json.dumps(replicate_ordinals.tolist(), separators=(",", ":")),
                "condition_file": str(path),
                "condition_file_sha256": sha256_file(path),
            }
        )
        preservation_rows.append(
            {
                **meta,
                "pairing_hash": pairing_hash,
                "global_count_violations": count_violations,
                "transport_cycle_violations": cycle_violations,
                "moved_fraction_max_error": moved_error,
                "value_support_log_min": float(np.min(value_support[:, 0])),
                "value_support_log_median": float(np.median(value_support[:, 0])),
                "value_mutable_strata_min": int(np.min(value_support[:, 1])),
                "mobility_support_log_min": float(np.min(mobility_support[:, 0])),
                "mobility_support_log_median": float(np.median(mobility_support[:, 0])),
                "mobility_mutable_strata_min": int(np.min(mobility_support[:, 1])),
                "trajectory_length": len(GRID),
            }
        )
        for family_index, family in enumerate(NULL_FAMILIES):
            family_curves = curves[family_index]
            metrics = trajectory_outcomes(family_curves)
            scalar_frame = pd.DataFrame(
                {
                    "schema_version": OUTPUT_SCHEMA,
                    "research_step_id": "S06",
                    **{key: [value] * TOTAL_DRAWS for key, value in meta.items()},
                    "null_family": family,
                    "draw_index": np.arange(TOTAL_DRAWS, dtype=np.int16),
                    "draw_role": ["reference"] * REFERENCE_DRAWS
                    + ["calibration"] * CALIBRATION_DRAWS,
                    "scenario_count": len(scenario_ids),
                    "pairing_hash": pairing_hash,
                    **metrics,
                }
            )
            scalar_writer.write(scalar_frame)
            scalar_count += len(scalar_frame)
            retained = family_curves[:RETAINED_DRAWS]
            retained_writer.write(
                pd.DataFrame(
                    {
                        "schema_version": OUTPUT_SCHEMA,
                        "research_step_id": "S06",
                        **{
                            key: np.repeat(value, RETAINED_DRAWS * len(GRID))
                            for key, value in meta.items()
                        },
                        "null_family": np.repeat(family, RETAINED_DRAWS * len(GRID)),
                        "draw_index": np.repeat(
                            np.arange(RETAINED_DRAWS, dtype=np.int16), len(GRID)
                        ),
                        "grid_index": np.tile(
                            np.arange(len(GRID), dtype=np.int16), RETAINED_DRAWS
                        ),
                        "accepted_swap_progress": np.tile(GRID, RETAINED_DRAWS),
                        "corrected_publication_aggregation": retained.reshape(-1),
                    }
                )
            )
            retained_count += RETAINED_DRAWS * len(GRID)
            reference_curves = family_curves[:REFERENCE_DRAWS]
            response = pd.DataFrame(
                {
                    "schema_version": OUTPUT_SCHEMA,
                    "research_step_id": "S06",
                    **{key: [value] * len(GRID) for key, value in meta.items()},
                    "null_family": family,
                    "grid_index": np.arange(len(GRID), dtype=np.int16),
                    "accepted_swap_progress": GRID,
                    "null_mean": np.mean(reference_curves, axis=0),
                    "null_sd": np.std(reference_curves, axis=0, ddof=1),
                    "null_q025": np.quantile(reference_curves, 0.025, axis=0),
                    "null_median": np.quantile(reference_curves, 0.5, axis=0),
                    "null_q975": np.quantile(reference_curves, 0.975, axis=0),
                    "observed": observed,
                }
            )
            response_writer.write(response)
            response_count += len(response)
            reference_metrics = {
                key: values[:REFERENCE_DRAWS] for key, values in metrics.items()
            }
            for outcome in OUTCOMES:
                reference = reference_metrics[outcome]
                observed_value = observed_scalar[outcome]
                q025, q975 = np.quantile(reference, [0.025, 0.975])
                effect_rows.append(
                    {
                        "schema_version": OUTPUT_SCHEMA,
                        "research_step_id": "S06",
                        **meta,
                        "null_family": family,
                        "outcome": outcome,
                        "observed": observed_value,
                        "null_mean": float(np.mean(reference)),
                        "null_sd": float(np.std(reference, ddof=1)),
                        "null_q025": float(q025),
                        "null_median": float(np.median(reference)),
                        "null_q975": float(q975),
                        "adjusted_effect": float(observed_value - np.mean(reference)),
                        "adjusted_effect_ci_lower": float(observed_value - q975),
                        "adjusted_effect_ci_upper": float(observed_value - q025),
                        "monte_carlo_p": _conservative_p(reference, observed_value),
                        "reference_draws": REFERENCE_DRAWS,
                    }
                )
                for calibration_index in range(CALIBRATION_DRAWS):
                    pseudo = float(metrics[outcome][REFERENCE_DRAWS + calibration_index])
                    calibration_values[(family, outcome)].append(
                        (
                            _randomized_p(
                                reference,
                                pseudo,
                                family,
                                condition_id,
                                outcome,
                                calibration_index,
                            ),
                            _conservative_p(reference, pseudo),
                        )
                    )
            null_mean_curve = np.mean(reference_curves, axis=0)
            selection_rows.append(
                {
                    **meta,
                    "null_family": family,
                    "mean_null_maximum": float(np.mean(reference_metrics["peak"])),
                    "maximum_of_null_mean_curve": float(np.max(null_mean_curve)),
                    "maximum_selection_inflation": float(
                        np.mean(reference_metrics["peak"]) - np.max(null_mean_curve)
                    ),
                }
            )
            temporal_rows.append(
                {
                    **meta,
                    "null_family": family,
                    "observed_lag1_acf": observed_scalar["lag1_acf"],
                    "null_lag1_median": float(np.nanmedian(reference_metrics["lag1_acf"])),
                    "null_lag1_q005": float(np.nanquantile(reference_metrics["lag1_acf"], 0.005)),
                    "null_lag1_q995": float(np.nanquantile(reference_metrics["lag1_acf"], 0.995)),
                    "observed_lag5_acf": observed_scalar["lag5_acf"],
                    "null_lag5_median": float(np.nanmedian(reference_metrics["lag5_acf"])),
                    "null_lag5_q005": float(np.nanquantile(reference_metrics["lag5_acf"], 0.005)),
                    "null_lag5_q995": float(np.nanquantile(reference_metrics["lag5_acf"], 0.995)),
                    "transport_chronology_preserved": True,
                    "moved_fraction_max_error": moved_error,
                    "trajectory_length": len(GRID),
                }
            )

    for writer in (scalar_writer, retained_writer, response_writer, observed_writer):
        writer.close()
    effects = pd.DataFrame(effect_rows)
    effects["bh_q_family_outcome"] = np.nan
    for _, indices in effects.groupby(["null_family", "outcome"], sort=True).groups.items():
        effects.loc[indices, "bh_q_family_outcome"] = _bh_adjust(
            effects.loc[indices, "monte_carlo_p"].to_numpy()
        )
    effects["bh_q_all_tests"] = _bh_adjust(effects.monte_carlo_p.to_numpy())
    effects = effects.sort_values(["null_family", "outcome", "condition_id"])
    _write_parquet(effects, output / "peak_adjusted_effects.parquet")
    _write_parquet(
        pd.DataFrame(observed_rows).sort_values("condition_id"),
        output / "observed_dynamic_summaries.parquet",
    )
    _write_parquet(
        pd.DataFrame(selection_rows).sort_values(["null_family", "condition_id"]),
        output / "maximum_selection_inflation.parquet",
    )
    _write_parquet(
        pd.DataFrame(temporal_rows).sort_values(["null_family", "condition_id"]),
        output / "temporal_dependence_audit.parquet",
    )
    _write_parquet(
        pd.DataFrame(preservation_rows).sort_values("condition_id"),
        output / "null_preservation_audit.parquet",
    )
    _write_parquet(
        pd.DataFrame(condition_manifest_rows).sort_values("condition_id"),
        output / "null_condition_manifest.parquet",
    )

    calibration_rows = []
    for (family, outcome), values in sorted(calibration_values.items()):
        randomized = np.asarray([item[0] for item in values])
        conservative = np.asarray([item[1] for item in values])
        if len(randomized) != EXPECTED_CONDITIONS * CALIBRATION_DRAWS:
            raise AssertionError("calibration pseudo-observation count changed")
        for alpha in (0.01, 0.05, 0.10):
            se = math.sqrt(alpha * (1 - alpha) / len(randomized))
            tolerance = max(0.015, 5 * se)
            randomized_rate = float(np.mean(randomized <= alpha))
            conservative_rate = float(np.mean(conservative <= alpha))
            calibration_rows.append(
                {
                    "null_family": family,
                    "outcome": outcome,
                    "alpha": alpha,
                    "pseudo_observations": len(randomized),
                    "randomized_rate": randomized_rate,
                    "conservative_rate": conservative_rate,
                    "binomial_se": se,
                    "tolerance": tolerance,
                    "randomized_passed": abs(randomized_rate - alpha) <= tolerance,
                    "conservative_passed": conservative_rate <= alpha + tolerance,
                }
            )
    calibration = pd.DataFrame(calibration_rows)
    _write_parquet(calibration, output / "dynamic_null_calibration.parquet")
    accounting = {
        "schema": "e04.s06.null_accounting.v1",
        "conditions": EXPECTED_CONDITIONS,
        "families": len(NULL_FAMILIES),
        "referenceDrawsPerConditionFamily": REFERENCE_DRAWS,
        "calibrationDrawsPerConditionFamily": CALIBRATION_DRAWS,
        "groupNullTrajectories": scalar_count,
        "expectedGroupNullTrajectories": EXPECTED_GROUP_TRAJECTORIES,
        "scenarioNullPaths": scalar_count * len(AUDIT_REPLICATES),
        "gridMetricEvaluations": scalar_count * len(AUDIT_REPLICATES) * len(GRID),
        "retainedFullTrajectoryRows": retained_count,
        "responseSurfaceRows": response_count,
        "effectRows": len(effects),
        "calibrationRows": len(calibration),
        "allCalibrationChecksPassed": bool(
            calibration.randomized_passed.all()
            and calibration.conservative_passed.all()
        ),
    }
    if scalar_count != EXPECTED_GROUP_TRAJECTORIES:
        raise AssertionError("dynamic null trajectory accounting failed")
    if not accounting["allCalibrationChecksPassed"]:
        raise AssertionError("dynamic null calibration failed")
    write_json(output / "dynamic_null_accounting.json", accounting)
    return accounting


def run_deterministic_audit(
    checkpoint: Path, null_dir: Path, audit_dir: Path, output: Path
) -> dict[str, Any]:
    grouped = _group_replay_records(checkpoint)
    condition_id = sorted(grouped)[0]
    result = _condition_null_worker(condition_id, grouped[condition_id], audit_dir)
    canonical_path = _condition_file(null_dir, condition_id)
    rerun_path = Path(result["path"])
    with np.load(canonical_path, allow_pickle=False) as canonical, np.load(
        rerun_path, allow_pickle=False
    ) as rerun:
        keys = sorted(set(canonical.files) | set(rerun.files))
        comparisons = {}
        for key in keys:
            if key not in canonical.files or key not in rerun.files:
                comparisons[key] = False
                continue
            left = canonical[key]
            right = rerun[key]
            equal_nan = np.issubdtype(left.dtype, np.inexact)
            comparisons[key] = bool(
                np.array_equal(left, right, equal_nan=equal_nan)
            )
    record = {
        "schema": "e04.s06.deterministic_audit.v1",
        "conditionId": condition_id,
        "canonicalPath": str(canonical_path),
        "rerunPath": str(rerun_path),
        "arrayComparisons": comparisons,
        "allPassed": all(comparisons.values()),
    }
    if not record["allPassed"]:
        raise AssertionError("S06 deterministic condition rerun failed")
    write_json(output / "deterministic_pairing_audit.json", record)
    return record


def _save_figure(fig: Any, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _plot_representative_trajectories(output: Path) -> None:
    response = pd.read_parquet(
        output / "null_distributions/null_response_surface.parquet"
    )
    anchors = response[
        response.composition_profile.eq("p50_50")
        & response.correlation_profile.eq("absent")
    ]
    inputs = ["unique_1_100", "repeated_1_10_x10"]
    pairs = ["Bubble+Insertion", "Bubble+Selection", "Insertion+Selection"]
    colors = {
        "label_permuted_global": "#1f77b4",
        "label_permuted_value_stratified": "#ff7f0e",
        "mobility_matched": "#2ca02c",
        "policy_neutral_transport": "#9467bd",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    for row, input_profile in enumerate(inputs):
        for column, pair in enumerate(pairs):
            ax = axes[row, column]
            panel = anchors[
                anchors.input_profile.eq(input_profile)
                & anchors.policy_set_label.eq(pair)
            ]
            global_panel = panel[
                panel.null_family.eq("label_permuted_global")
            ].sort_values("grid_index")
            ax.fill_between(
                global_panel.accepted_swap_progress,
                global_panel.null_q025,
                global_panel.null_q975,
                color=colors["label_permuted_global"],
                alpha=0.15,
                label="global 95% null" if row == 0 and column == 0 else None,
            )
            ax.plot(
                global_panel.accepted_swap_progress,
                global_panel.observed,
                color="black",
                linewidth=2.0,
                label="observed" if row == 0 and column == 0 else None,
            )
            for family in NULL_FAMILIES:
                family_panel = panel[panel.null_family.eq(family)].sort_values(
                    "grid_index"
                )
                ax.plot(
                    family_panel.accepted_swap_progress,
                    family_panel.null_mean,
                    color=colors[family],
                    linewidth=1.2,
                    label=family.replace("_", " ")
                    if row == 0 and column == 0
                    else None,
                )
            ax.axhline(0, color="0.5", linewidth=0.8)
            ax.set_title(
                f"{'Unique' if row == 0 else 'Repeated'} · {pair.replace('+', ' + ')}"
            )
            if row == 1:
                ax.set_xlabel("Accepted-swap progress")
            if column == 0:
                ax.set_ylabel("Composition-corrected adjacency")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("S06 balanced absent-association trajectories and dynamic null means")
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    _save_figure(fig, output, "dynamic_null_trajectories")


def _plot_effects(output: Path) -> None:
    effects = pd.read_parquet(output / "peak_adjusted_effects.parquet")
    family_order = list(NULL_FAMILIES)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for ax, outcome in zip(axes, OUTCOMES, strict=True):
        panel = effects[effects.outcome.eq(outcome)]
        data = [
            panel.loc[panel.null_family.eq(family), "adjusted_effect"].to_numpy()
            for family in family_order
        ]
        ax.boxplot(data, tick_labels=[f.replace("_", "\n") for f in family_order], showfliers=False)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(outcome.replace("_", " ").title())
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        ax.set_ylabel("Observed minus mean null statistic")
    fig.suptitle("Peak- and persistence-adjusted effects across 186 conditions")
    fig.tight_layout()
    _save_figure(fig, output, "peak_adjusted_effects")


def _plot_calibration(output: Path) -> None:
    calibration = pd.read_parquet(output / "dynamic_null_calibration.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    x = np.arange(len(NULL_FAMILIES))
    width = 0.22
    for ax, outcome in zip(axes, OUTCOMES, strict=True):
        panel = calibration[calibration.outcome.eq(outcome)]
        for offset, alpha in enumerate((0.01, 0.05, 0.10)):
            values = [
                float(
                    panel[
                        panel.null_family.eq(family) & panel.alpha.eq(alpha)
                    ].randomized_rate.iloc[0]
                )
                for family in NULL_FAMILIES
            ]
            ax.bar(x + (offset - 1) * width, values, width, label=f"α={alpha:g}")
            ax.axhline(alpha, color="0.5", linewidth=0.6, linestyle="--")
        ax.set_xticks(x, [family.replace("_", "\n") for family in NULL_FAMILIES])
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        ax.set_title(outcome.replace("_", " ").title())
        ax.set_ylabel("Randomized false-positive rate")
    axes[0].legend(frameon=False)
    fig.suptitle("Exchangeable pseudo-observation calibration")
    fig.tight_layout()
    _save_figure(fig, output, "dynamic_null_calibration")


def _plot_significance(output: Path) -> None:
    effects = pd.read_parquet(output / "peak_adjusted_effects.parquet")
    summary = (
        effects.assign(significant=effects.bh_q_family_outcome <= 0.05)
        .groupby(["null_family", "outcome", "correlation_profile"], as_index=False)
        .significant.mean()
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    x = np.arange(3)
    width = 0.18
    correlations = ["absent", "positive", "negative"]
    for ax, outcome in zip(axes, OUTCOMES, strict=True):
        panel = summary[summary.outcome.eq(outcome)]
        for family_index, family in enumerate(NULL_FAMILIES):
            values = [
                float(
                    panel[
                        panel.null_family.eq(family)
                        & panel.correlation_profile.eq(correlation)
                    ].significant.iloc[0]
                )
                for correlation in correlations
            ]
            ax.bar(
                x + (family_index - 1.5) * width,
                values,
                width,
                label=family.replace("_", " "),
            )
        ax.set_xticks(x, correlations)
        ax.set_ylim(0, 1.03)
        ax.set_title(outcome.replace("_", " ").title())
        ax.set_ylabel("Fraction BH q ≤ 0.05")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Dynamic-null significance depends on conditioning assumptions")
    fig.tight_layout(rect=(0, 0.15, 1, 0.95))
    _save_figure(fig, output, "dynamic_conditioning_sensitivity")


def analyze_dynamic_nulls(output: Path) -> dict[str, Any]:
    effects = pd.read_parquet(output / "peak_adjusted_effects.parquet")
    selection = pd.read_parquet(output / "maximum_selection_inflation.parquet")
    calibration = pd.read_parquet(output / "dynamic_null_calibration.parquet")
    preservation = pd.read_parquet(output / "null_preservation_audit.parquet")
    anchors = effects[
        effects.composition_profile.eq("p50_50")
        & effects.correlation_profile.eq("absent")
        & effects.null_family.eq("label_permuted_global")
    ]
    if anchors.condition_id.nunique() != 6:
        raise AssertionError("primary anchor population changed")
    peak_anchor = anchors[anchors.outcome.eq("peak")].copy()
    peak_anchor["anchor_passed"] = (
        (peak_anchor.bh_q_family_outcome <= 0.05)
        & (peak_anchor.adjusted_effect > 0.02)
    )
    persistence = anchors[anchors.outcome.isin(["duration", "positive_area"])].copy()
    persistence_pass = (
        persistence.assign(passed=persistence.bh_q_family_outcome <= 0.05)
        .groupby("condition_id")
        .passed.max()
    )
    peak_pass_count = int(peak_anchor.anchor_passed.sum())
    persistence_pass_count = int(persistence_pass.sum())
    global_selection = selection[
        selection.null_family.eq("label_permuted_global")
    ]
    selection_fraction = float(
        np.mean(global_selection.maximum_selection_inflation > 1e-6)
    )
    validation_pass = bool(
        calibration.randomized_passed.all()
        and calibration.conservative_passed.all()
        and preservation.global_count_violations.eq(0).all()
        and preservation.transport_cycle_violations.eq(0).all()
        and preservation.moved_fraction_max_error.eq(0).all()
        and preservation.trajectory_length.eq(len(GRID)).all()
    )
    supportive = bool(
        validation_pass
        and peak_pass_count >= 4
        and persistence_pass_count >= 4
        and selection_fraction >= 0.95
    )
    family_summary = (
        effects.assign(significant=effects.bh_q_family_outcome <= 0.05)
        .groupby(["null_family", "outcome"], as_index=False)
        .agg(
            conditions=("condition_id", "size"),
            significant_conditions=("significant", "sum"),
            median_adjusted_effect=("adjusted_effect", "median"),
            min_adjusted_effect=("adjusted_effect", "min"),
            max_adjusted_effect=("adjusted_effect", "max"),
            median_null_statistic=("null_median", "median"),
        )
    )
    family_summary["significant_fraction"] = (
        family_summary.significant_conditions / family_summary.conditions
    )
    family_summary.to_csv(output / "dynamic_null_family_summary.csv", index=False)
    peak_anchor.sort_values(["input_profile", "policy_set_label"]).to_csv(
        output / "primary_anchor_results.csv", index=False
    )
    peak_wide = effects[effects.outcome.eq("peak")].pivot(
        index="condition_id", columns="null_family", values="bh_q_family_outcome"
    )
    sensitivity = pd.DataFrame(
        {
            "global_significant": peak_wide.label_permuted_global <= 0.05,
            "value_stratified_significant": peak_wide.label_permuted_value_stratified
            <= 0.05,
            "mobility_matched_significant": peak_wide.mobility_matched <= 0.05,
            "policy_neutral_significant": peak_wide.policy_neutral_transport <= 0.05,
        }
    ).reset_index()
    sensitivity["lost_after_value_conditioning"] = (
        sensitivity.global_significant
        & ~sensitivity.value_stratified_significant
    )
    sensitivity["lost_after_mobility_conditioning"] = (
        sensitivity.global_significant & ~sensitivity.mobility_matched_significant
    )
    sensitivity.to_csv(output / "dynamic_conditioning_sensitivity.csv", index=False)
    _plot_representative_trajectories(output)
    _plot_effects(output)
    _plot_calibration(output)
    _plot_significance(output)
    summary = {
        "schema": "e04.s06.analysis_summary.v1",
        "researchStepId": "S06",
        "outcomeClassification": "supportive" if supportive else "constraining/contradictory",
        "validationPassed": validation_pass,
        "primaryAnchors": {
            "peakPassed": peak_pass_count,
            "peakRequired": 4,
            "persistencePassed": persistence_pass_count,
            "persistenceRequired": 4,
            "anchorCount": 6,
        },
        "selectionInflation": {
            "globalConditionsPositive": int(
                np.sum(global_selection.maximum_selection_inflation > 1e-6)
            ),
            "conditionCount": len(global_selection),
            "fraction": selection_fraction,
            "requiredFraction": 0.95,
            "median": float(global_selection.maximum_selection_inflation.median()),
            "min": float(global_selection.maximum_selection_inflation.min()),
            "max": float(global_selection.maximum_selection_inflation.max()),
        },
        "significantPeakCounts": {
            family: int(
                np.sum(
                    (effects.null_family == family)
                    & (effects.outcome == "peak")
                    & (effects.bh_q_family_outcome <= 0.05)
                )
            )
            for family in NULL_FAMILIES
        },
        "globalPeakSignificantLostAfterValueConditioning": int(
            sensitivity.lost_after_value_conditioning.sum()
        ),
        "globalPeakSignificantLostAfterMobilityConditioning": int(
            sensitivity.lost_after_mobility_conditioning.sum()
        ),
        "supportiveRulePassed": supportive,
    }
    write_json(output / "analysis_summary.json", summary)
    write_json(
        output / "success_criteria.json",
        {
            "schema": "e04.s06.success_criteria.v1",
            "researchStepId": "S06",
            "criteria": {
                "validation": validation_pass,
                "peakAnchors": peak_pass_count >= 4,
                "persistenceAnchors": persistence_pass_count >= 4,
                "maximumSelectionInflation": selection_fraction >= 0.95,
            },
            "allPassed": supportive,
        },
    )
    return summary


def write_provenance(output: Path, cache: Path) -> dict[str, Any]:
    upstream = {
        step: _verify_upstream_manifest(
            Path(f"/artifacts/research_steps/{step}"), expected
        )
        for step, expected in EXPECTED_MANIFEST_HASHES.items()
    }
    if not all(item["allPassed"] for item in upstream.values()):
        raise AssertionError("S01-S05 immutability failed after S06")
    write_json(output / "upstream_immutability_audit.json", upstream)
    environment = {
        "schema": "e04.s06.environment.v1",
        "python": sys.version,
        "platform": platform.platform(),
        "cpuCount": os.cpu_count(),
        "workerLimit": 8,
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in (
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
    }
    write_json(output / "environment.json", environment)
    commands = {
        "schema": "e04.s06.commands.v1",
        "workingDirectory": str(REPOSITORY),
        "commands": [
            "python scripts/run_dynamic_nulls.py freeze",
            "python scripts/run_dynamic_nulls.py replay --workers 8",
            "python scripts/run_dynamic_nulls.py replay-package",
            "python scripts/run_dynamic_nulls.py nulls --workers 8",
            "python scripts/run_dynamic_nulls.py deterministic-audit",
            "python scripts/run_dynamic_nulls.py package",
            "python scripts/run_dynamic_nulls.py analyze",
            "python -m pytest -q tests/test_e04_aggregation.py tests/test_e04_composition_baselines.py tests/test_e04_composition_sweep.py tests/test_e04_aggregation_metrics.py tests/test_e04_static_nulls.py tests/test_e04_dynamic_nulls.py --junitxml=/artifacts/research_steps/S06/repository_tests.junit.xml",
            "ruff check analysis/dynamic_nulls.py scripts/run_dynamic_nulls.py tests/test_e04_dynamic_nulls.py",
            "python -m py_compile analysis/dynamic_nulls.py scripts/run_dynamic_nulls.py tests/test_e04_dynamic_nulls.py",
            "python scripts/run_dynamic_nulls.py provenance",
            "python scripts/run_dynamic_nulls.py validate",
            "python scripts/run_dynamic_nulls.py manifest"
        ],
    }
    write_json(output / "commands.json", commands)
    important_inputs = {
        "agents": Path("/workspace/AGENTS.md"),
        "fullPlan": Path("/workspace/FULL_PLAN.md"),
        "researchPlanAtExecution": Path("/workspace/RESEARCH_PLAN.md"),
        "previousArtifactsMarkdown": Path("/workspace/PREVIOUS_ARTIFACTS.md"),
        "previousArtifactsJson": Path("/workspace/PREVIOUS_ARTIFACTS.json"),
        "attachmentManifest": Path("/workspace/input-attachments/MANIFEST.json"),
        "attachmentSidecar": Path("/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md"),
        "paperMarkdown": Path("/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/pdf-markdown.md"),
        "e01TransitionSpecification": Path("/previous-artifacts/E01/research_steps/S03/transition_spec.md"),
        "e01ReferenceRelease": Path("/previous-artifacts/E01/release/reference_simulator/release_manifest.json"),
        "s03Population": S03_DIR / "composition_sweep.parquet",
        "s04Metrics": S04_TABLE,
        "s05Report": S05_DIR / "research_step_full_results.md",
        "contract": CONTRACT_PATH,
    }
    provenance = {
        "schema": "e04.s06.provenance.v1",
        "researchStepId": "S06",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPOSITORY),
        "branch": _git_output("branch", "--show-current"),
        "head": _git_output("rev-parse", "HEAD"),
        "gitStatusShort": _git_output("status", "--short"),
        "outputDirectory": str(output),
        "cacheDirectory": str(cache),
        "inputs": {
            key: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for key, path in important_inputs.items()
        },
        "upstreamManifestHashes": EXPECTED_MANIFEST_HASHES,
        "s04CanonicalTable": {
            "bytes": S04_TABLE.stat().st_size,
            "sha256": sha256_file(S04_TABLE),
        },
        "environment": environment,
    }
    write_json(output / "provenance.json", provenance)
    return provenance


def validate_artifacts(output: Path) -> dict[str, Any]:
    required = [
        "preregistration.json",
        "freeze_record.json",
        "dynamic_null_specification.md",
        "trajectory_replay_audit.parquet",
        "trajectory_replay_summary.json",
        "movement_autocorrelation_audit.parquet",
        "null_distributions/dynamic_null_statistics.parquet",
        "null_distributions/retained_null_trajectories.parquet",
        "null_distributions/null_response_surface.parquet",
        "observed_dynamic_trajectories.parquet",
        "observed_dynamic_summaries.parquet",
        "peak_adjusted_effects.parquet",
        "maximum_selection_inflation.parquet",
        "dynamic_null_calibration.parquet",
        "temporal_dependence_audit.parquet",
        "null_preservation_audit.parquet",
        "null_condition_manifest.parquet",
        "dynamic_null_accounting.json",
        "deterministic_pairing_audit.json",
        "dynamic_null_family_summary.csv",
        "primary_anchor_results.csv",
        "dynamic_conditioning_sensitivity.csv",
        "dynamic_null_trajectories.png",
        "dynamic_null_trajectories.svg",
        "peak_adjusted_effects.png",
        "peak_adjusted_effects.svg",
        "dynamic_null_calibration.png",
        "dynamic_null_calibration.svg",
        "dynamic_conditioning_sensitivity.png",
        "dynamic_conditioning_sensitivity.svg",
        "analysis_summary.json",
        "success_criteria.json",
        "upstream_immutability_audit.json",
        "environment.json",
        "commands.json",
        "provenance.json",
        "research_step_full_results.md",
    ]
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": _json_native(detail)})

    missing = [name for name in required if not (output / name).is_file()]
    check("required_files", not missing, missing)
    freeze = assert_frozen(output)
    check("frozen_before_simulation", freeze["frozenBeforeSimulation"], freeze["frozenAt"])
    replay = json.loads((output / "trajectory_replay_summary.json").read_text())
    check("scenario_replay", replay["allPassed"], replay)
    accounting = json.loads((output / "dynamic_null_accounting.json").read_text())
    check(
        "group_null_accounting",
        accounting["groupNullTrajectories"] == EXPECTED_GROUP_TRAJECTORIES,
        accounting,
    )
    expected_rows = {
        "dynamic_null_statistics": EXPECTED_GROUP_TRAJECTORIES,
        "retained_null_trajectories": EXPECTED_CONDITIONS
        * len(NULL_FAMILIES)
        * RETAINED_DRAWS
        * len(GRID),
        "null_response_surface": EXPECTED_CONDITIONS
        * len(NULL_FAMILIES)
        * len(GRID),
        "observed_dynamic_trajectories": EXPECTED_CONDITIONS * len(GRID),
        "peak_adjusted_effects": EXPECTED_CONDITIONS * len(NULL_FAMILIES) * len(OUTCOMES),
    }
    observed_rows = {
        "dynamic_null_statistics": pq.ParquetFile(
            output / "null_distributions/dynamic_null_statistics.parquet"
        ).metadata.num_rows,
        "retained_null_trajectories": pq.ParquetFile(
            output / "null_distributions/retained_null_trajectories.parquet"
        ).metadata.num_rows,
        "null_response_surface": pq.ParquetFile(
            output / "null_distributions/null_response_surface.parquet"
        ).metadata.num_rows,
        "observed_dynamic_trajectories": pq.ParquetFile(
            output / "observed_dynamic_trajectories.parquet"
        ).metadata.num_rows,
        "peak_adjusted_effects": pq.ParquetFile(
            output / "peak_adjusted_effects.parquet"
        ).metadata.num_rows,
    }
    check("table_row_accounting", observed_rows == expected_rows, {"expected": expected_rows, "observed": observed_rows})
    effects = pd.read_parquet(output / "peak_adjusted_effects.parquet")
    check(
        "multiplicity_complete",
        effects.bh_q_family_outcome.notna().all()
        and effects.bh_q_all_tests.notna().all()
        and effects.bh_q_family_outcome.between(0, 1).all()
        and effects.bh_q_all_tests.between(0, 1).all(),
        len(effects),
    )
    calibration = pd.read_parquet(output / "dynamic_null_calibration.parquet")
    check(
        "null_calibration",
        len(calibration) == len(NULL_FAMILIES) * len(OUTCOMES) * 3
        and calibration.randomized_passed.all()
        and calibration.conservative_passed.all(),
        calibration.to_dict("records"),
    )
    preservation = pd.read_parquet(output / "null_preservation_audit.parquet")
    check(
        "preservation_constraints",
        len(preservation) == EXPECTED_CONDITIONS
        and preservation.global_count_violations.eq(0).all()
        and preservation.transport_cycle_violations.eq(0).all()
        and preservation.moved_fraction_max_error.eq(0).all()
        and preservation.trajectory_length.eq(len(GRID)).all(),
        {
            "countViolations": int(preservation.global_count_violations.sum()),
            "cycleViolations": int(preservation.transport_cycle_violations.sum()),
            "maxMovedFractionError": float(preservation.moved_fraction_max_error.max()),
        },
    )
    deterministic = json.loads((output / "deterministic_pairing_audit.json").read_text())
    check("deterministic_pairing", deterministic["allPassed"], deterministic)
    upstream = json.loads((output / "upstream_immutability_audit.json").read_text())
    check("upstream_immutability", all(item["allPassed"] for item in upstream.values()), list(upstream))
    analysis = json.loads((output / "analysis_summary.json").read_text())
    check("scientific_success_rule", analysis["supportiveRulePassed"], analysis)
    report = (output / "research_step_full_results.md").read_text()
    required_phrases = (
        "Research step ID",
        "Completion status",
        "Artifacts written",
        "Validation result",
        "Outcome classification",
        "Caveats or blockers",
        "Lay summary",
        "Recommended next action",
        "Detailed methods",
        "Commands",
        "Provenance",
    )
    check("report_required_fields", all(phrase in report for phrase in required_phrases), required_phrases)
    check("s07_absent", not S07_DIR.exists(), str(S07_DIR))
    passed = all(item["passed"] for item in checks)
    summary = {
        "schema": "e04.s06.validation_summary.v1",
        "researchStepId": "S06",
        "passed": passed,
        "checksPassed": sum(item["passed"] for item in checks),
        "checksTotal": len(checks),
        "checks": checks,
    }
    write_json(output / "validation_summary.json", summary)
    if not passed:
        failed = [item["name"] for item in checks if not item["passed"]]
        raise AssertionError(f"S06 artifact validation failed: {failed}")
    return summary


def write_artifact_manifest(output: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema": "e04.s06.artifact_manifest.v1",
        "researchStepId": "S06",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "artifactCount": len(artifacts),
        "artifacts": artifacts,
    }
    write_json(output / "artifact_manifest.json", manifest)
    return manifest
