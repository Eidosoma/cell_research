"""E04 S08 policy/analysis-label identity controls.

The label definitions, assignment channels, populations, estimands, and outcome
rules are frozen in ``analysis/s08_identity_control_contract.json``.  S08 uses
native E01 transitions only.  Analysis labels are immutable metadata and never
enter policy execution.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from datetime import datetime, timezone
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.chimeric_replication import GRID
from analysis.composition_sweep import (
    SweepCondition,
    derive_s03_seed,
    load_base_draws,
    materialize_sweep_scenario,
)
from analysis.dynamic_nulls import _label_curves, trajectory_outcomes
from analysis.kinetic_matching import (
    HOLDOUT_REPLICATES,
    POLICY_NAMES,
    RegimeParameters,
    _decode_run,
    build_tasks as build_s07_tasks,
    execute_kinetic_run,
)
from reference_simulator.model import (
    Architecture,
    Cell,
    Direction,
    FaultMode,
    Policy,
    Scenario,
    canonical_json_bytes,
)


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
CONTRACT_PATH = REPOSITORY / "analysis/s08_identity_control_contract.json"
OUTPUT_DIR = Path("/artifacts/research_steps/S08")
CACHE_DIR = Path("/cache/e04_s08")
S07_DIR = Path("/artifacts/research_steps/S07")
S09_DIR = Path("/artifacts/research_steps/S09")
UPSTREAM_DIRS = {
    f"S{index:02d}": Path(f"/artifacts/research_steps/S{index:02d}")
    for index in range(1, 8)
}
EXPECTED_MANIFEST_HASHES = {
    "S01": "df9860285fc5b1ed9f7442fb68ea5e9ed9e407918d90988d0b3592f4b3ef8079",
    "S02": "9e632eedac59f6b08ff6b619ce2451a99b49c952319af2747076fd7bb3411b54",
    "S03": "85371d82a53096764b0de52029472a7978261ca5d4857f4d392f441db076f9e9",
    "S04": "90db3cb1afbb8e1ff53fb1d4fe0e04a4d21fb66ccd3a0b922738bade82d05641",
    "S05": "8a9528c5f14abb3598cc7e1f41a145978f7f7d47717c23497623f5c00579b4ce",
    "S06": "e8137e96571fdd281629bf6d0c32e6c8bbbc081ede2953f19ca9c11f957aa786",
    "S07": "294b89d1bd38762fada95b785aa2f82450fe26b4fc522c2dedaf0f44e360dd18",
}
SEED_NAMESPACE = "E04/S08/identity_controls/v1"
CHANNELS = 512
PRIMARY_CHANNEL = 0
REFERENCE_CHANNELS = tuple(range(1, 501))
CALIBRATION_CHANNELS = tuple(range(501, 512))
CANDIDATE_CHANNELS = tuple(range(20))
OUTCOMES = ("peak", "positive_area")
POLICIES = ("Bubble", "Insertion", "Selection")
INPUT_PROFILES = ("unique_1_100", "repeated_1_10_x10")
EXPECTED_MIXED = 186 * len(HOLDOUT_REPLICATES)
EXPECTED_HOMOGENEOUS = len(POLICIES) * len(INPUT_PROFILES) * len(HOLDOUT_REPLICATES)
ANCHOR_POLICY_SETS = {"Bubble+Insertion", "Bubble+Selection", "Insertion+Selection"}
TRANSITION_FIELDS = (
    "stop_reason",
    "activation_count",
    "successful_swap_count",
    "final_state_hash",
    "trajectory_sha256",
    "occupancy_b64",
    "corrected_curve_sha256",
    "ledger_json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        compression="zstd",
    )


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT
    ).strip()


def derive_seed(stream: str, *address: Any) -> int:
    payload = {
        "namespace": SEED_NAMESPACE,
        "stream": stream,
        "address": list(address),
    }
    return int.from_bytes(
        hashlib.sha256(canonical_json_bytes(payload)).digest()[:16], "big"
    )


def _verify_manifest(step: str) -> dict[str, Any]:
    directory = UPSTREAM_DIRS[step]
    manifest_path = directory / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checks = []
    for item in manifest["artifacts"]:
        artifact = directory / item["path"]
        observed = sha256_file(artifact) if artifact.is_file() else None
        checks.append(
            {
                "path": item["path"],
                "expected": item["sha256"],
                "observed": observed,
                "passed": observed == item["sha256"],
            }
        )
    manifest_hash = sha256_file(manifest_path)
    expected_hash = EXPECTED_MANIFEST_HASHES[step]
    return {
        "step": step,
        "manifestPath": str(manifest_path),
        "expectedManifestSha256": expected_hash,
        "observedManifestSha256": manifest_hash,
        "manifestPassed": manifest_hash == expected_hash,
        "artifactChecks": checks,
        "allPassed": manifest_hash == expected_hash
        and all(item["passed"] for item in checks),
    }


def _specification_markdown(contract_hash: str) -> str:
    return f"""# S08 frozen policy/label identity-control specification

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | S08 |
| Completion status | Design and all analysis-label assignments frozen before any S08 transition or label-aggregation outcome was examined |
| Artifacts written | `preregistration.json`, `freeze_record.json`, and this specification |
| Validation result | Pre-outcome structure passed: 186 mixed conditions, six homogeneous-policy controls, 25 frozen replicates, 512 fixed label channels, exact 50/50 or 100/0 counts, and 450 planned labelled replay comparisons |
| Outcome classification | Pending S08 execution |
| Caveats or blockers | Labels correlated with value or selected after outcome inspection can create grouping without an action pathway; those assignments are diagnostic only |
| Recommended next action | Execute native trajectories and frozen identity controls, validate exact transition invariance, then stop before S09 |

Contract SHA-256: `{contract_hash}`.

## Frozen identity boundary

Cell identity carries immutable value, executable policy, direction, Selection
cursor state, and optional analysis metadata. Executable policy selects behavior.
Analysis label selects only the grouping function used after a state is produced.
Policy observation, proposal, validation, scheduler, stopping, and actuation may
not read the analysis label.

Labelled validation copies retain the source runtime seed and scenario ID. This
prevents the metadata-bearing scenario serialization from indirectly selecting
a different counter-addressed scheduler stream.

## Control assignments

The primary random assignment is channel 0 from a frozen 512-row exact-50/50
label bank per scenario. Channels 1--500 are reference draws and 501--511 are
calibration pseudo-observations. Distinct-label/identical-policy controls cover
Bubble, Insertion, and Selection on unique and repeated inputs. Same-label
controls collapse every native mixed-policy identity to one analysis label.
Orthogonal ghost labels use channel 0 on every native mixed trajectory.

A value-blocked 50/50 assignment on homogeneous-policy runs and best-of-20
selection over frozen random channels are post hoc artifact diagnostics. They
cannot enter the primary success rule.

## Metrics and outcome rule

Label adjacency uses the paper denominator `n=100` and the exact composition
expectation `sum_k n_k(n_k-1)/n^2`. Thus 50/50 labels have expectation 0.49 and
a single 100-cell label has expectation 0.99. Primary endpoints are the maximum
and positive area of complete 101-point condition-mean corrected trajectories.

Support requires exact label-blind replays; exact counts; calibrated random-label
channels; no systematic primary aggregation for arbitrary labels; identically
zero corrected collapsed-label curves with unchanged executable-policy curves;
and executable-policy effects exceeding ghost-label effects in all six balanced
absent-association anchors. S07 kinetic interventions are excluded.
"""


def freeze_design(output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("cannot freeze S08 after S08 artifacts exist")
    if cache.exists() and any(cache.iterdir()):
        raise FileExistsError("cannot freeze S08 after S08 cache exists")
    if S09_DIR.exists():
        raise AssertionError("S09 artifacts exist before S08 freeze")
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT_PATH.read_text())
    contract_hash = sha256_file(CONTRACT_PATH)
    upstream = {step: _verify_manifest(step) for step in EXPECTED_MANIFEST_HASHES}
    if not all(item["allPassed"] for item in upstream.values()):
        raise AssertionError("upstream artifact immutability failed")
    mixed_tasks = build_s07_tasks("holdout")
    if len(mixed_tasks) != EXPECTED_MIXED:
        raise AssertionError("mixed S08 source population changed")
    anchors = {
        task["condition"]["conditionId"]
        for task in mixed_tasks
        if task["condition"]["policySetLabel"] in ANCHOR_POLICY_SETS
        and task["condition"]["firstPolicyCount"] == 50
        and task["condition"]["correlationProfile"] == "absent"
    }
    if len(anchors) != 6:
        raise AssertionError("expected six mixed transition anchors")
    record = {
        "schema": "e04.s08.freeze_record.v1",
        "researchStepId": "S08",
        "frozenAt": datetime.now(timezone.utc).isoformat(),
        "frozenBeforeOutcomes": True,
        "contractPath": str(CONTRACT_PATH),
        "contractSha256": contract_hash,
        "implementationPath": str(Path(__file__).resolve()),
        "implementationSha256": sha256_file(Path(__file__).resolve()),
        "mixedConditionCount": 186,
        "mixedScenarioCount": EXPECTED_MIXED,
        "homogeneousConditionCount": 6,
        "homogeneousScenarioCount": EXPECTED_HOMOGENEOUS,
        "labelChannels": CHANNELS,
        "primaryChannel": PRIMARY_CHANNEL,
        "referenceChannels": [REFERENCE_CHANNELS[0], REFERENCE_CHANNELS[-1]],
        "calibrationChannels": [CALIBRATION_CHANNELS[0], CALIBRATION_CHANNELS[-1]],
        "candidateChannels": list(CANDIDATE_CHANNELS),
        "anchorConditionIds": sorted(anchors),
        "holdoutReplicates": list(HOLDOUT_REPLICATES),
        "upstream": upstream,
        "s07SourceRegime": "native_control",
        "excludedS07Regimes": contract["scope"]["excludedSourceRegimes"],
        "s09Absent": not S09_DIR.exists(),
    }
    write_json(output / "preregistration.json", contract)
    write_json(output / "freeze_record.json", record)
    (output / "identity_control_specification.md").write_text(
        _specification_markdown(contract_hash), encoding="utf-8"
    )
    return record


def assert_frozen(output: Path = OUTPUT_DIR) -> dict[str, Any]:
    record = json.loads((output / "freeze_record.json").read_text())
    if record["contractSha256"] != sha256_file(CONTRACT_PATH):
        raise AssertionError("S08 contract changed after freeze")
    if record["implementationSha256"] != sha256_file(Path(__file__).resolve()):
        raise AssertionError("S08 implementation changed after freeze")
    if not record["frozenBeforeOutcomes"]:
        raise AssertionError("S08 did not freeze before outcomes")
    if S09_DIR.exists():
        raise AssertionError("S09 artifacts appeared during S08")
    return record


def native_parameters() -> RegimeParameters:
    return RegimeParameters(
        "native_control",
        {policy: 1.0 for policy in POLICY_NAMES},
        {policy: 0.0 for policy in POLICY_NAMES},
        False,
    )


def homogeneous_condition_id(input_profile: str, policy: str) -> str:
    input_code = "UNQ" if input_profile == "unique_1_100" else "REP"
    policy_code = {"Bubble": "BUB", "Insertion": "INS", "Selection": "SEL"}[policy]
    return f"S08-HOM-{input_code}-{policy_code}"


def materialize_homogeneous_scenario(
    input_profile: str, policy: str, base: Mapping[str, Any]
) -> Scenario:
    values = [int(value) for value in base["valuesById"]]
    occupancy_indices = [int(value) for value in base["initialOccupancyIndices"]]
    cells = tuple(
        Cell(
            f"cell-{index:04d}",
            values[index],
            Policy(policy),
            Direction.ASCENDING,
            FaultMode.NORMAL,
        )
        for index in range(100)
    )
    runtime_seed = derive_s03_seed(
        "reference_runtime",
        input_profile,
        str(base["split"]),
        int(base["replicateOrdinal"]),
    )
    scenario = Scenario.create(
        cells,
        initial_occupancy=tuple(f"cell-{index:04d}" for index in occupancy_indices),
        seed=runtime_seed,
        max_activations=1_000_000,
        architecture=Architecture.CELL_VIEW,
        generation_key=(
            f"E04/S08/homogeneous/{input_profile}/{policy}/{base['baseDrawId']}"
        ),
        fault_placement="explicit",
        requested_fault_count=0,
    )
    scenario.validate()
    return scenario


def build_homogeneous_tasks() -> list[dict[str, Any]]:
    bases = load_base_draws()
    tasks = []
    for input_profile in INPUT_PROFILES:
        for policy in POLICIES:
            for replicate in HOLDOUT_REPLICATES:
                base = dict(bases[(input_profile, replicate)])
                scenario = materialize_homogeneous_scenario(input_profile, policy, base)
                tasks.append(
                    {
                        "population": "homogeneous",
                        "condition_id": homogeneous_condition_id(input_profile, policy),
                        "input_profile": input_profile,
                        "policy": policy,
                        "replicate_ordinal": replicate,
                        "base": base,
                        "scenario_id": scenario.scenario_id,
                        "scenario_json_sha256": hashlib.sha256(
                            scenario.to_json_bytes()
                        ).hexdigest(),
                    }
                )
    tasks.sort(key=lambda item: (item["condition_id"], item["replicate_ordinal"]))
    if len(tasks) != EXPECTED_HOMOGENEOUS:
        raise AssertionError("homogeneous population changed")
    return tasks


def _mixed_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    condition = SweepCondition.from_dict(task["condition"])
    scenario, metadata = materialize_sweep_scenario(condition, task["base"])
    if metadata["scenarioJsonSha256"] != task["metadata"]["scenario_json_sha256"]:
        raise AssertionError("mixed scenario rematerialization changed")
    result = execute_kinetic_run(scenario, native_parameters())
    result.update(task["metadata"])
    result["population"] = "mixed"
    result["run_id"] = (
        "e04s08:mixed:" + hashlib.sha256(scenario.scenario_id.encode()).hexdigest()
    )
    return result


def _homogeneous_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    scenario = materialize_homogeneous_scenario(
        str(task["input_profile"]), str(task["policy"]), task["base"]
    )
    if scenario.scenario_id != task["scenario_id"]:
        raise AssertionError("homogeneous scenario rematerialization changed")
    result = execute_kinetic_run(scenario, native_parameters())
    result.update(
        {
            "population": "homogeneous",
            "condition_id": task["condition_id"],
            "input_profile": task["input_profile"],
            "policy_set_label": task["policy"],
            "composition_class": "homogeneous",
            "composition_profile": "p100_00",
            "first_policy_count": 100,
            "rare_policy": None,
            "correlation_profile": "not_applicable",
            "replicate_ordinal": task["replicate_ordinal"],
            "base_draw_id": task["base"]["baseDrawId"],
            "pairing_block_id": task["base"]["pairingBlockId"],
            "scenario_json_sha256": task["scenario_json_sha256"],
            "scenario_id": scenario.scenario_id,
            "run_id": "e04s08:homogeneous:"
            + hashlib.sha256(scenario.scenario_id.encode()).hexdigest(),
        }
    )
    return result


def _checkpoint_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as handle:
        return {str(json.loads(line)["scenario_id"]) for line in handle if line.strip()}


def _run_checkpoint(
    tasks: Sequence[Mapping[str, Any]],
    checkpoint: Path,
    worker: Any,
    workers: int,
) -> dict[str, Any]:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _checkpoint_ids(checkpoint)
    key = "scenario_id" if tasks and "scenario_id" in tasks[0] else None
    pending = [
        task
        for task in tasks
        if str(task[key] if key else task["metadata"]["scenario_id"]) not in completed
    ]
    started = time.perf_counter()
    written = 0
    with (
        checkpoint.open("a", buffering=1) as handle,
        ProcessPoolExecutor(max_workers=workers) as executor,
    ):
        iterator = iter(pending)
        futures: dict[Any, Mapping[str, Any]] = {}
        for _ in range(min(workers * 2, len(pending))):
            task = next(iterator, None)
            if task is None:
                break
            futures[executor.submit(worker, task)] = task
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                result = future.result()
                handle.write(
                    json.dumps(
                        _json_native(result), sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
                written += 1
                task = next(iterator, None)
                if task is not None:
                    futures[executor.submit(worker, task)] = task
    return {
        "checkpoint": str(checkpoint),
        "requested": len(tasks),
        "preexisting": len(completed),
        "written": written,
        "observed": len(_checkpoint_ids(checkpoint)),
        "elapsedSeconds": time.perf_counter() - started,
    }


def run_corpus(
    output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR, workers: int = 8
) -> dict[str, Any]:
    assert_frozen(output)
    mixed = _run_checkpoint(
        build_s07_tasks("holdout"), cache / "mixed_native.jsonl", _mixed_worker, workers
    )
    homogeneous = _run_checkpoint(
        build_homogeneous_tasks(),
        cache / "homogeneous_native.jsonl",
        _homogeneous_worker,
        workers,
    )
    record = {
        "schema": "e04.s08.run_accounting.v1",
        "researchStepId": "S08",
        "mixed": mixed,
        "homogeneous": homogeneous,
        "totalExpected": EXPECTED_MIXED + EXPECTED_HOMOGENEOUS,
        "totalObserved": mixed["observed"] + homogeneous["observed"],
    }
    write_json(output / "run_accounting.json", record)
    return record


def read_checkpoint(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ghost_assignments(scenario_id: str) -> np.ndarray:
    base = np.repeat(np.arange(2, dtype=np.uint8), 50)
    output = np.broadcast_to(base, (CHANNELS, 100)).copy()
    rng = np.random.Generator(
        np.random.PCG64DXSM(derive_seed("ghost_assignments", scenario_id))
    )
    return rng.permuted(output, axis=1)


def value_block_assignment(values: np.ndarray, scenario_id: str) -> np.ndarray:
    rng = np.random.Generator(
        np.random.PCG64DXSM(derive_seed("value_block_ties", scenario_id))
    )
    tie_order = rng.permutation(100)
    tie_rank = np.empty(100, dtype=np.int16)
    tie_rank[tie_order] = np.arange(100, dtype=np.int16)
    order = np.lexsort((tie_rank, values.astype(np.int16)))
    labels = np.ones(100, dtype=np.uint8)
    labels[order[:50]] = 0
    return labels


def _labelled_copy(source: Scenario, labels: Sequence[str]) -> Scenario:
    if len(labels) != len(source.cells):
        raise ValueError("label assignment length mismatch")
    labelled_cells = tuple(
        replace(cell, analysis_label=str(labels[index]))
        for index, cell in enumerate(source.cells)
    )
    labelled = Scenario.create(
        labelled_cells,
        initial_occupancy=source.initial_occupancy,
        initial_selection_cursors=dict(source.initial_selection_cursors),
        seed=source.seed,
        max_activations=source.max_activations,
        architecture=source.architecture,
        scheduler=source.scheduler,
        batch_width=source.batch_width,
        traditional_policy=source.traditional_policy,
        generation_key=source.generation_key,
        fault_placement=source.fault_placement,
        requested_fault_count=source.requested_fault_count,
        rng_profile=source.rng_profile,
        goal_profile=source.goal_profile,
        metric_profile=source.metric_profile,
    )
    labelled.validate()
    object.__setattr__(labelled, "scenario_id", source.scenario_id)
    return labelled


def _comparison_row(
    source_result: Mapping[str, Any],
    observed: Mapping[str, Any],
    population: str,
    condition_id: str,
    assignment: str,
    replicate: int,
) -> dict[str, Any]:
    checks = {
        f"check_{field}": observed[field] == source_result[field]
        for field in TRANSITION_FIELDS
    }
    return {
        "population": population,
        "condition_id": condition_id,
        "replicate_ordinal": replicate,
        "scenario_id": source_result["scenario_id"],
        "assignment": assignment,
        **checks,
        "passed": all(checks.values()),
    }


def _transition_audit_worker(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_result = task["source_result"]
    if task["population"] == "homogeneous":
        source = materialize_homogeneous_scenario(
            str(task["input_profile"]), str(task["policy"]), task["base"]
        )
        codes = ghost_assignments(source.scenario_id)[PRIMARY_CHANNEL]
        assignments = {
            "distinct_labels_identical_policy": [
                "ghost:A" if code == 0 else "ghost:B" for code in codes
            ]
        }
    else:
        condition = SweepCondition.from_dict(task["condition"])
        source, _ = materialize_sweep_scenario(condition, task["base"])
        codes = ghost_assignments(source.scenario_id)[PRIMARY_CHANNEL]
        assignments = {
            "same_label_distinct_policies": ["merged:all"] * 100,
            "action_irrelevant_ghost_labels": [
                "ghost:A" if code == 0 else "ghost:B" for code in codes
            ],
        }
    rows = []
    for assignment, labels in assignments.items():
        labelled = _labelled_copy(source, labels)
        observed = execute_kinetic_run(
            labelled, native_parameters(), validate_scenario=False
        )
        rows.append(
            _comparison_row(
                source_result,
                observed,
                str(task["population"]),
                str(task["condition_id"]),
                assignment,
                int(task["replicate_ordinal"]),
            )
        )
    return rows


def run_transition_audit(
    output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR, workers: int = 8
) -> dict[str, Any]:
    assert_frozen(output)
    mixed_rows = read_checkpoint(cache / "mixed_native.jsonl")
    homogeneous_rows = read_checkpoint(cache / "homogeneous_native.jsonl")
    mixed_source = {row["scenario_id"]: row for row in mixed_rows}
    homogeneous_source = {row["scenario_id"]: row for row in homogeneous_rows}
    tasks: list[dict[str, Any]] = []
    for task in build_homogeneous_tasks():
        tasks.append({**task, "source_result": homogeneous_source[task["scenario_id"]]})
    anchors = set(
        json.loads((output / "freeze_record.json").read_text())["anchorConditionIds"]
    )
    for task in build_s07_tasks("holdout"):
        if task["condition"]["conditionId"] in anchors:
            tasks.append(
                {
                    "population": "mixed",
                    "condition_id": task["condition"]["conditionId"],
                    "input_profile": task["condition"]["inputProfile"],
                    "replicate_ordinal": int(task["base"]["replicateOrdinal"]),
                    "condition": task["condition"],
                    "base": task["base"],
                    "source_result": mixed_source[task["metadata"]["scenario_id"]],
                }
            )
    if len(tasks) != 300:
        raise AssertionError(
            f"expected 300 transition audit source tasks, found {len(tasks)}"
        )
    checkpoint = cache / "transition_audit.jsonl"
    existing: set[str] = set()
    if checkpoint.exists():
        for row in read_checkpoint(checkpoint):
            existing.add(f"{row['scenario_id']}|{row['assignment']}")
    started = time.perf_counter()
    written = 0
    pending = []
    for task in tasks:
        expected_assignments = (
            ("distinct_labels_identical_policy",)
            if task["population"] == "homogeneous"
            else ("same_label_distinct_policies", "action_irrelevant_ghost_labels")
        )
        source_id = task["source_result"]["scenario_id"]
        if not all(
            f"{source_id}|{assignment}" in existing
            for assignment in expected_assignments
        ):
            pending.append(task)
    with (
        checkpoint.open("a", buffering=1) as handle,
        ProcessPoolExecutor(max_workers=workers) as executor,
    ):
        iterator = iter(pending)
        futures: dict[Any, Mapping[str, Any]] = {}
        for _ in range(min(workers * 2, len(pending))):
            task = next(iterator, None)
            if task is None:
                break
            futures[executor.submit(_transition_audit_worker, task)] = task
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                for row in future.result():
                    key = f"{row['scenario_id']}|{row['assignment']}"
                    if key not in existing:
                        handle.write(
                            json.dumps(row, sort_keys=True, separators=(",", ":"))
                            + "\n"
                        )
                        existing.add(key)
                        written += 1
                task = next(iterator, None)
                if task is not None:
                    futures[executor.submit(_transition_audit_worker, task)] = task
    frame = pd.DataFrame(read_checkpoint(checkpoint)).sort_values(
        ["population", "condition_id", "replicate_ordinal", "assignment"]
    )
    _write_parquet(frame, output / "transition_identity_audit.parquet")
    record = {
        "schema": "e04.s08.transition_audit_summary.v1",
        "researchStepId": "S08",
        "expectedRows": 450,
        "observedRows": len(frame),
        "passedRows": int(frame.passed.sum()),
        "allPassed": len(frame) == 450 and bool(frame.passed.all()),
        "writtenThisInvocation": written,
        "elapsedSeconds": time.perf_counter() - started,
    }
    write_json(output / "transition_audit_summary.json", record)
    return record


def _bh_adjust(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array)
    ranked = array[order]
    adjusted = ranked * len(array) / np.arange(1, len(array) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _source_replay_audit(mixed_rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    source = pq.read_table(
        S07_DIR / "kinetic_matching.parquet",
        filters=[("regime", "=", "native_control")],
    ).to_pandas()
    source_map = source.set_index("scenario_id").to_dict("index")
    fields = (
        "condition_id",
        "replicate_ordinal",
        "stop_reason",
        "activation_count",
        "successful_swap_count",
        "final_state_hash",
        "trajectory_sha256",
        "corrected_curve_sha256",
        "ledger_json",
    )
    rows = []
    for row in mixed_rows:
        expected = source_map[str(row["scenario_id"])]
        checks = {f"check_{field}": row[field] == expected[field] for field in fields}
        rows.append(
            {
                "scenario_id": row["scenario_id"],
                "condition_id": row["condition_id"],
                "replicate_ordinal": row["replicate_ordinal"],
                **checks,
                "passed": all(checks.values()),
            }
        )
    return pd.DataFrame(rows)


def _source_policy_audit() -> dict[str, Any]:
    files = [
        REPOSITORY / "reference_simulator/policies.py",
        REPOSITORY / "reference_simulator/scheduler.py",
        REPOSITORY / "reference_simulator/engine.py",
        REPOSITORY / "reference_simulator/transition_primitives.py",
        REPOSITORY / "analysis/kinetic_matching.py",
    ]
    violations = []
    for path in files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "analysis_label":
                violations.append({"path": str(path), "line": node.lineno})
    return {
        "schema": "e04.s08.policy_observation_audit.v1",
        "researchStepId": "S08",
        "auditedFiles": [str(path) for path in files],
        "analysisLabelAttributeAccesses": violations,
        "passed": not violations,
        "note": "Cell serialization and invariant modules may retain analysis_label; behavior modules may not read it.",
    }


def _calibration_table(channel_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (population, outcome), group in channel_frame.groupby(
        ["population", "outcome"], sort=True
    ):
        pseudo = []
        for _, condition in group.groupby("condition_id", sort=True):
            lookup = condition.set_index("channel").value
            reference = lookup.loc[list(REFERENCE_CHANNELS)].to_numpy(dtype=float)
            for channel in CALIBRATION_CHANNELS:
                observed = float(lookup.loc[channel])
                greater = int(np.sum(reference > observed))
                equal = int(np.sum(reference == observed))
                rng = np.random.Generator(
                    np.random.PCG64DXSM(
                        derive_seed(
                            "calibration_tie",
                            population,
                            outcome,
                            str(condition.condition_id.iloc[0]),
                            channel,
                        )
                    )
                )
                randomized = (1 + greater + float(rng.random()) * equal) / 501
                conservative = (1 + greater + equal) / 501
                pseudo.append((randomized, conservative))
        values = np.asarray(pseudo, dtype=float)
        for alpha in (0.01, 0.05, 0.10):
            tolerance = max(
                0.015,
                3.5 * math.sqrt(alpha * (1 - alpha) / len(values)),
            )
            randomized_rate = float(np.mean(values[:, 0] <= alpha))
            conservative_rate = float(np.mean(values[:, 1] <= alpha))
            rows.append(
                {
                    "population": population,
                    "outcome": outcome,
                    "alpha": alpha,
                    "pseudo_observations": len(values),
                    "randomized_rate": randomized_rate,
                    "conservative_rate": conservative_rate,
                    "tolerance": tolerance,
                    "randomized_passed": abs(randomized_rate - alpha) <= tolerance,
                    "conservative_passed": conservative_rate <= alpha + tolerance,
                }
            )
    return pd.DataFrame(rows)


def _identity_results(channel_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (population, condition_id, outcome), group in channel_frame.groupby(
        ["population", "condition_id", "outcome"], sort=True
    ):
        lookup = group.set_index("channel").value
        observed = float(lookup.loc[PRIMARY_CHANNEL])
        reference = lookup.loc[list(REFERENCE_CHANNELS)].to_numpy(dtype=float)
        rows.append(
            {
                "population": population,
                "condition_id": condition_id,
                "control_family": (
                    "action_irrelevant_ghost_labels"
                    if population == "mixed"
                    else "distinct_labels_identical_policy"
                ),
                "label_definition": "ghost_random_exact_channel_0",
                "outcome": outcome,
                "observed_endpoint": observed,
                "reference_mean": float(np.mean(reference)),
                "reference_q025": float(np.quantile(reference, 0.025)),
                "reference_q975": float(np.quantile(reference, 0.975)),
                "adjusted_effect": observed - float(np.mean(reference)),
                "p_value": (1 + int(np.sum(reference >= observed))) / 501,
                "role": "primary",
            }
        )
    frame = pd.DataFrame(rows)
    frame["q_value"] = np.nan
    for _, indices in frame.groupby(
        ["population", "outcome"], sort=True
    ).groups.items():
        frame.loc[indices, "q_value"] = _bh_adjust(frame.loc[indices, "p_value"])
    return frame


def _plot_results(
    trajectories: pd.DataFrame,
    controls: pd.DataFrame,
    posthoc: pd.DataFrame,
    discrimination: pd.DataFrame,
    output: Path,
) -> None:
    colors = {
        "executable_policy": "#a23b72",
        "ghost_random_exact_channel_0": "#2e86ab",
        "collapsed": "#777777",
        "value_blocked": "#f18f01",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    mixed = trajectories[
        (trajectories.population == "mixed")
        & trajectories.control_id.isin(
            ["executable_policy", "ghost_random_exact_channel_0", "collapsed"]
        )
    ]
    mixed_mean = (
        mixed.groupby(["control_id", "progress"], sort=True)
        .corrected_mean.mean()
        .reset_index()
    )
    for control_id, group in mixed_mean.groupby("control_id", sort=True):
        axes[0].plot(
            group.progress,
            group.corrected_mean,
            label=control_id,
            color=colors[control_id],
        )
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set(
        title="Mixed-policy controls (186-condition mean)",
        xlabel="accepted-swap progress",
        ylabel="composition-corrected adjacency",
    )
    axes[0].legend(fontsize=8)
    homogeneous = trajectories[trajectories.population == "homogeneous"]
    homogeneous_mean = (
        homogeneous.groupby(["control_id", "progress"], sort=True)
        .corrected_mean.mean()
        .reset_index()
    )
    for control_id, group in homogeneous_mean.groupby("control_id", sort=True):
        axes[1].plot(
            group.progress,
            group.corrected_mean,
            label=control_id,
            color=colors.get(control_id),
        )
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set(
        title="Identical-policy controls (six-condition mean)",
        xlabel="accepted-swap progress",
        ylabel="composition-corrected adjacency",
    )
    axes[1].legend(fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"identity_control_trajectories.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, outcome in zip(axes, OUTCOMES):
        group = discrimination[discrimination.outcome == outcome].copy()
        x = np.arange(len(group))
        axis.bar(
            x - 0.18,
            group.policy_adjusted_effect,
            0.36,
            label="executable policy",
            color=colors["executable_policy"],
        )
        axis.bar(
            x + 0.18,
            group.ghost_adjusted_effect,
            0.36,
            label="ghost label",
            color=colors["ghost_random_exact_channel_0"],
        )
        axis.set_xticks(
            x,
            [item.replace("S03-", "") for item in group.condition_id],
            rotation=55,
            ha="right",
            fontsize=7,
        )
        axis.set_title(outcome.replace("_", " "))
        axis.axhline(0, color="black", linewidth=0.7)
    axes[0].set_ylabel("dynamic-null adjusted effect")
    axes[0].legend(fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"policy_vs_label_endpoints.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, outcome in zip(axes, OUTCOMES):
        values = posthoc[
            (posthoc.artifact == "best_of_20") & (posthoc.outcome == outcome)
        ].inflation
        axis.hist(values, bins=30, color="#6c5ce7", alpha=0.85)
        axis.axvline(
            float(values.median()),
            color="black",
            linestyle="--",
            label=f"median={values.median():.4f}",
        )
        axis.set(
            title=outcome.replace("_", " "),
            xlabel="best-of-20 minus frozen channel 0",
            ylabel="conditions",
        )
        axis.legend(fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"posthoc_label_selection.{suffix}", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    value_rows = controls[
        (controls.control_family == "value_blocked_identical_policy")
        & (controls.outcome == "peak")
    ]
    ax.bar(value_rows.condition_id, value_rows.observed_endpoint, color="#f18f01")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(
        title="Value-correlated labels on identical policies",
        ylabel="corrected trajectory peak",
        xlabel="homogeneous executable policy",
    )
    ax.tick_params(axis="x", rotation=45)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"value_correlated_label_artifact.{suffix}", dpi=180)
    plt.close(fig)


def analyze_controls(
    output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR
) -> dict[str, Any]:
    assert_frozen(output)
    mixed_raw = read_checkpoint(cache / "mixed_native.jsonl")
    homogeneous_raw = read_checkpoint(cache / "homogeneous_native.jsonl")
    if len(mixed_raw) != EXPECTED_MIXED or len(homogeneous_raw) != EXPECTED_HOMOGENEOUS:
        raise AssertionError("S08 corpus incomplete")

    replay = _source_replay_audit(mixed_raw)
    _write_parquet(replay, output / "s07_native_source_replay_audit.parquet")
    policy_audit = _source_policy_audit()
    write_json(output / "policy_observation_audit.json", policy_audit)

    all_rows = [("mixed", row) for row in mixed_raw] + [
        ("homogeneous", row) for row in homogeneous_raw
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for population, row in all_rows:
        grouped.setdefault((population, str(row["condition_id"])), []).append(row)

    channel_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    posthoc_rows: list[dict[str, Any]] = []
    diagnostic_control_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []

    for (population, condition_id), raw_group in sorted(grouped.items()):
        raw_group = sorted(raw_group, key=lambda item: int(item["replicate_ordinal"]))
        if len(raw_group) != len(HOLDOUT_REPLICATES):
            raise AssertionError(f"{condition_id} lacks 25 paired runs")
        mean_channels = np.zeros((CHANNELS, len(GRID)), dtype=np.float64)
        mean_policy = np.zeros(len(GRID), dtype=np.float64)
        mean_value_block = np.zeros(len(GRID), dtype=np.float64)
        count_violations = 0
        for raw in raw_group:
            decoded = _decode_run(raw)
            assignments = ghost_assignments(str(raw["scenario_id"]))
            counts = np.sum(assignments == 0, axis=1)
            count_violations += int(np.sum(counts != 50))
            curves = _label_curves(assignments, decoded["occupancy"], 0.49)
            mean_channels += curves / len(raw_group)
            mean_policy += decoded["corrected_curve"] / len(raw_group)
            primary = assignments[PRIMARY_CHANNEL]
            assignment_row = {
                "population": population,
                "condition_id": condition_id,
                "replicate_ordinal": int(raw["replicate_ordinal"]),
                "scenario_id": raw["scenario_id"],
                "primary_assignment_sha256": hashlib.sha256(
                    primary.tobytes()
                ).hexdigest(),
                "primary_label_a_count": int(np.sum(primary == 0)),
                "primary_label_b_count": int(np.sum(primary == 1)),
                "all_channel_count_violations": int(np.sum(counts != 50)),
                "collapsed_label_count": 100 if population == "mixed" else None,
            }
            if population == "homogeneous":
                blocked = value_block_assignment(
                    decoded["values"], str(raw["scenario_id"])
                )
                blocked_curve = _label_curves(
                    blocked[None, :], decoded["occupancy"], 0.49
                )[0]
                mean_value_block += blocked_curve / len(raw_group)
                assignment_row.update(
                    {
                        "value_block_assignment_sha256": hashlib.sha256(
                            blocked.tobytes()
                        ).hexdigest(),
                        "value_block_a_count": int(np.sum(blocked == 0)),
                        "value_block_b_count": int(np.sum(blocked == 1)),
                    }
                )
            else:
                assignment_row.update(
                    {
                        "value_block_assignment_sha256": None,
                        "value_block_a_count": None,
                        "value_block_b_count": None,
                    }
                )
            assignment_rows.append(assignment_row)
        if count_violations:
            raise AssertionError(f"ghost assignment count failure in {condition_id}")

        outcomes = trajectory_outcomes(mean_channels)
        for outcome in OUTCOMES:
            values = outcomes[outcome]
            for channel, value in enumerate(values):
                channel_rows.append(
                    {
                        "population": population,
                        "condition_id": condition_id,
                        "channel": channel,
                        "outcome": outcome,
                        "value": float(value),
                    }
                )
            posthoc_rows.append(
                {
                    "population": population,
                    "condition_id": condition_id,
                    "artifact": "best_of_20",
                    "outcome": outcome,
                    "primary_value": float(values[PRIMARY_CHANNEL]),
                    "selected_value": float(np.max(values[list(CANDIDATE_CHANNELS)])),
                    "selected_channel": int(
                        CANDIDATE_CHANNELS[
                            int(np.argmax(values[list(CANDIDATE_CHANNELS)]))
                        ]
                    ),
                    "inflation": float(
                        np.max(values[list(CANDIDATE_CHANNELS)])
                        - values[PRIMARY_CHANNEL]
                    ),
                }
            )
        for index, progress in enumerate(GRID):
            response_rows.append(
                {
                    "population": population,
                    "condition_id": condition_id,
                    "progress": progress,
                    "primary": float(mean_channels[PRIMARY_CHANNEL, index]),
                    "reference_mean": float(
                        np.mean(mean_channels[list(REFERENCE_CHANNELS), index])
                    ),
                    "reference_q025": float(
                        np.quantile(
                            mean_channels[list(REFERENCE_CHANNELS), index], 0.025
                        )
                    ),
                    "reference_q975": float(
                        np.quantile(
                            mean_channels[list(REFERENCE_CHANNELS), index], 0.975
                        )
                    ),
                }
            )
            control_values = [
                ("executable_policy", mean_policy[index]),
                ("ghost_random_exact_channel_0", mean_channels[PRIMARY_CHANNEL, index]),
            ]
            if population == "mixed":
                control_values.append(("collapsed", 0.0))
            for control_id, value in control_values:
                trajectory_rows.append(
                    {
                        "population": population,
                        "condition_id": condition_id,
                        "control_id": control_id,
                        "progress": progress,
                        "corrected_mean": float(value),
                    }
                )
            if population == "homogeneous":
                trajectory_rows.append(
                    {
                        "population": population,
                        "condition_id": condition_id,
                        "control_id": "value_blocked",
                        "progress": progress,
                        "corrected_mean": float(mean_value_block[index]),
                    }
                )
        if population == "homogeneous":
            blocked_outcomes = trajectory_outcomes(mean_value_block)
            for outcome in OUTCOMES:
                diagnostic_control_rows.append(
                    {
                        "population": population,
                        "condition_id": condition_id,
                        "control_family": "value_blocked_identical_policy",
                        "label_definition": "value_blocked_diagnostic",
                        "outcome": outcome,
                        "observed_endpoint": float(blocked_outcomes[outcome][0]),
                        "reference_mean": np.nan,
                        "reference_q025": np.nan,
                        "reference_q975": np.nan,
                        "adjusted_effect": np.nan,
                        "p_value": np.nan,
                        "q_value": np.nan,
                        "role": "diagnostic",
                    }
                )
            posthoc_rows.append(
                {
                    "population": population,
                    "condition_id": condition_id,
                    "artifact": "value_blocked",
                    "outcome": "final_corrected_adjacency",
                    "primary_value": float(mean_channels[PRIMARY_CHANNEL, -1]),
                    "selected_value": float(mean_value_block[-1]),
                    "selected_channel": -1,
                    "inflation": float(
                        mean_value_block[-1] - mean_channels[PRIMARY_CHANNEL, -1]
                    ),
                }
            )
        composition_rows.append(
            {
                "population": population,
                "condition_id": condition_id,
                "assignment": "ghost_random_exact",
                "counts_json": '{"ghost:A":50,"ghost:B":50}',
                "paper_expectation": 0.49,
                "raw_adjacency_if_perfectly_grouped": 0.98,
                "corrected_if_perfectly_grouped": 0.49,
                "count_violations": count_violations,
                "universal_half_excess_if_used": -0.01,
            }
        )
        if population == "mixed":
            composition_rows.append(
                {
                    "population": population,
                    "condition_id": condition_id,
                    "assignment": "collapsed",
                    "counts_json": '{"merged:all":100}',
                    "paper_expectation": 0.99,
                    "raw_adjacency_if_perfectly_grouped": 0.99,
                    "corrected_if_perfectly_grouped": 0.0,
                    "count_violations": 0,
                    "universal_half_excess_if_used": 0.49,
                }
            )

    channel_frame = pd.DataFrame(channel_rows)
    trajectories = pd.DataFrame(trajectory_rows)
    response = pd.DataFrame(response_rows)
    assignment = pd.DataFrame(assignment_rows)
    posthoc = pd.DataFrame(posthoc_rows)
    composition = pd.DataFrame(composition_rows)
    controls = _identity_results(channel_frame)

    s07_effects = pq.read_table(
        S07_DIR / "null_adjusted_effects.parquet",
        filters=[
            ("regime", "=", "native_control"),
            ("null_family", "=", "label_permuted_global"),
        ],
    ).to_pandas()
    policy_rows = []
    for row in s07_effects[s07_effects.outcome.isin(OUTCOMES)].itertuples(index=False):
        policy_rows.append(
            {
                "population": "mixed",
                "condition_id": row.condition_id,
                "control_family": "executable_policy_reference",
                "label_definition": "executable_policy",
                "outcome": row.outcome,
                "observed_endpoint": row.observed,
                "reference_mean": row.null_mean,
                "reference_q025": row.null_q025,
                "reference_q975": row.null_q975,
                "adjusted_effect": row.adjusted_effect,
                "p_value": row.p_value,
                "q_value": row.q_value,
                "role": "reference",
            }
        )
    collapsed_rows = []
    for condition_id in sorted({row["condition_id"] for row in mixed_raw}):
        for outcome in OUTCOMES:
            collapsed_rows.append(
                {
                    "population": "mixed",
                    "condition_id": condition_id,
                    "control_family": "same_label_distinct_policies",
                    "label_definition": "collapsed",
                    "outcome": outcome,
                    "observed_endpoint": 0.0,
                    "reference_mean": 0.0,
                    "reference_q025": 0.0,
                    "reference_q975": 0.0,
                    "adjusted_effect": 0.0,
                    "p_value": 1.0,
                    "q_value": 1.0,
                    "role": "primary_degenerate",
                }
            )
    controls = pd.concat(
        [
            controls,
            pd.DataFrame(policy_rows),
            pd.DataFrame(collapsed_rows),
            pd.DataFrame(diagnostic_control_rows),
        ],
        ignore_index=True,
    ).sort_values(["population", "condition_id", "control_family", "outcome"])

    ghost_lookup = controls[
        (controls.population == "mixed")
        & (controls.control_family == "action_irrelevant_ghost_labels")
    ][["condition_id", "outcome", "adjusted_effect"]].rename(
        columns={"adjusted_effect": "ghost_adjusted_effect"}
    )
    anchor_ids = set(
        json.loads((output / "freeze_record.json").read_text())["anchorConditionIds"]
    )
    policy_lookup = controls[
        (controls.control_family == "executable_policy_reference")
        & controls.condition_id.isin(anchor_ids)
    ][["condition_id", "outcome", "adjusted_effect"]].rename(
        columns={"adjusted_effect": "policy_adjusted_effect"}
    )
    discrimination = policy_lookup.merge(
        ghost_lookup, on=["condition_id", "outcome"], how="inner", validate="one_to_one"
    )
    discrimination["policy_minus_ghost"] = (
        discrimination.policy_adjusted_effect - discrimination.ghost_adjusted_effect
    )
    discrimination["threshold"] = discrimination.outcome.map(
        {"peak": 0.05, "positive_area": 0.02}
    )
    discrimination["passed"] = (
        discrimination.policy_minus_ghost > discrimination.threshold
    )

    calibration = _calibration_table(channel_frame)
    _write_parquet(channel_frame, output / "ghost_channel_endpoints.parquet")
    _write_parquet(
        response, output / "null_distributions/ghost_null_response_surface.parquet"
    )
    _write_parquet(trajectories, output / "identity_control_trajectories.parquet")
    _write_parquet(assignment, output / "control_assignment_manifest.parquet")
    _write_parquet(composition, output / "composition_audit.parquet")
    _write_parquet(posthoc, output / "posthoc_grouping_audit.parquet")
    _write_parquet(controls, output / "identity_controls.parquet")
    _write_parquet(calibration, output / "ghost_null_calibration.parquet")
    discrimination.to_csv(output / "policy_label_discrimination.csv", index=False)

    homogeneous_primary = controls[
        (controls.population == "homogeneous")
        & (controls.control_family == "distinct_labels_identical_policy")
    ]
    mixed_primary = controls[
        (controls.population == "mixed")
        & (controls.control_family == "action_irrelevant_ghost_labels")
    ]
    identical_rules = {}
    ghost_rules = {}
    for outcome, absolute_limit in (("peak", 0.01), ("positive_area", 0.005)):
        frame = homogeneous_primary[homogeneous_primary.outcome == outcome]
        identical_rules[outcome] = {
            "significant": int((frame.q_value <= 0.05).sum()),
            "medianAbsoluteEffect": float(frame.adjusted_effect.abs().median()),
            "limit": absolute_limit,
            "passed": int((frame.q_value <= 0.05).sum()) <= 1
            and float(frame.adjusted_effect.abs().median()) <= absolute_limit,
        }
    for outcome, absolute_limit in (("peak", 0.005), ("positive_area", 0.0025)):
        frame = mixed_primary[mixed_primary.outcome == outcome]
        ghost_rules[outcome] = {
            "significant": int((frame.q_value <= 0.05).sum()),
            "medianAbsoluteEffect": float(frame.adjusted_effect.abs().median()),
            "limit": absolute_limit,
            "passed": int((frame.q_value <= 0.05).sum()) <= 1
            and float(frame.adjusted_effect.abs().median()) <= absolute_limit,
        }
    transition = json.loads((output / "transition_audit_summary.json").read_text())
    calibration_passed = bool(
        (calibration.randomized_passed & calibration.conservative_passed).all()
    )
    collapsed_max = float(
        trajectories[
            (trajectories.population == "mixed")
            & (trajectories.control_id == "collapsed")
        ]
        .corrected_mean.abs()
        .max()
    )
    policy_replay_passed = len(replay) == EXPECTED_MIXED and bool(replay.passed.all())
    composition_passed = int(composition.count_violations.sum()) == 0 and bool(
        (assignment.primary_label_a_count == 50).all()
        and (assignment.primary_label_b_count == 50).all()
    )
    validity = (
        transition["allPassed"]
        and policy_audit["passed"]
        and calibration_passed
        and composition_passed
        and policy_replay_passed
        and collapsed_max == 0.0
    )
    identical_passed = all(item["passed"] for item in identical_rules.values())
    ghost_passed = all(item["passed"] for item in ghost_rules.values())
    discrimination_passed = len(discrimination) == 12 and bool(
        discrimination.passed.all()
    )
    supportive = (
        validity and identical_passed and ghost_passed and discrimination_passed
    )
    if supportive:
        classification = "supportive"
    elif validity:
        classification = "null"
    else:
        classification = "constraining/contradictory"
    criteria = {
        "schema": "e04.s08.success_criteria.v1",
        "researchStepId": "S08",
        "validityPassed": validity,
        "identicalPolicyRules": identical_rules,
        "ghostLabelRules": ghost_rules,
        "collapsedCorrectedMaxAbsolute": collapsed_max,
        "collapsedPassed": collapsed_max == 0.0 and policy_replay_passed,
        "policyDiscriminationRows": len(discrimination),
        "policyDiscriminationPassed": discrimination_passed,
        "calibrationPassed": calibration_passed,
        "compositionPassed": composition_passed,
        "policyReplayPassed": policy_replay_passed,
        "supportiveRulePassed": supportive,
        "classification": classification,
    }
    write_json(output / "success_criteria.json", criteria)
    summary = {
        "schema": "e04.s08.analysis_summary.v1",
        "researchStepId": "S08",
        "outcomeClassification": classification,
        "sourceRegime": "native_control",
        "excludedS07JointRegime": True,
        "mixedRuns": len(mixed_raw),
        "homogeneousRuns": len(homogeneous_raw),
        "labelChannelsPerScenario": CHANNELS,
        "channelEndpointRows": len(channel_frame),
        "transitionAuditRows": transition["observedRows"],
        "transitionAuditPassed": transition["allPassed"],
        "sourcePolicyAuditPassed": policy_audit["passed"],
        "sourceReplayPassed": policy_replay_passed,
        "allCalibrationPassed": calibration_passed,
        "identicalPolicyRules": identical_rules,
        "ghostLabelRules": ghost_rules,
        "collapsedCorrectedMaxAbsolute": collapsed_max,
        "policyDiscriminationMinimumByOutcome": {
            str(key): float(value)
            for key, value in discrimination.groupby("outcome")
            .policy_minus_ghost.min()
            .items()
        },
        "policyDiscriminationPassed": discrimination_passed,
        "bestOf20MedianInflation": {
            f"{population}|{outcome}": float(value)
            for (population, outcome), value in posthoc[
                posthoc.artifact == "best_of_20"
            ]
            .groupby(["population", "outcome"])
            .inflation.median()
            .items()
        },
        "valueBlockedMedianFinalInflation": float(
            posthoc[posthoc.artifact == "value_blocked"].inflation.median()
        ),
        "supportiveRulePassed": supportive,
    }
    write_json(output / "analysis_summary.json", summary)
    _plot_results(trajectories, controls, posthoc, discrimination, output)

    accounting = {
        "schema": "e04.s08.analysis_accounting.v1",
        "researchStepId": "S08",
        "identityControlRows": len(controls),
        "trajectoryRows": len(trajectories),
        "channelEndpointRows": len(channel_frame),
        "nullResponseRows": len(response),
        "assignmentRows": len(assignment),
        "compositionRows": len(composition),
        "posthocRows": len(posthoc),
        "calibrationRows": len(calibration),
        "policyDiscriminationRows": len(discrimination),
    }
    write_json(output / "analysis_accounting.json", accounting)
    return summary


def write_environment(output: Path = OUTPUT_DIR, workers: int = 8) -> dict[str, Any]:
    import matplotlib as mpl
    import scipy

    record = {
        "schema": "e04.s08.environment.v1",
        "researchStepId": "S08",
        "python": sys.version,
        "platform": platform.platform(),
        "cpuCount": os.cpu_count(),
        "workers": workers,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "scipy": scipy.__version__,
        "matplotlib": mpl.__version__,
        "gpuUsed": False,
        "dependenciesInstalled": [],
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    write_json(output / "environment.json", record)
    return record


def write_commands(output: Path = OUTPUT_DIR) -> dict[str, Any]:
    commands = [
        "python scripts/run_identity_controls.py freeze",
        "python scripts/run_identity_controls.py run --workers 8",
        "python scripts/run_identity_controls.py audit --workers 8",
        "python scripts/run_identity_controls.py analyze",
        "python -m pytest -q tests/test_e04_aggregation.py tests/test_e04_composition_baselines.py tests/test_e04_composition_sweep.py tests/test_e04_aggregation_metrics.py tests/test_e04_static_nulls.py tests/test_e04_dynamic_nulls.py tests/test_e04_kinetic_matching.py tests/test_e04_identity_controls.py --junitxml=/artifacts/research_steps/S08/repository_tests.junit.xml",
        "ruff check analysis/identity_controls.py scripts/run_identity_controls.py tests/test_e04_identity_controls.py",
        "python -m py_compile analysis/identity_controls.py scripts/run_identity_controls.py tests/test_e04_identity_controls.py",
        "git diff --check",
        "python scripts/run_identity_controls.py environment",
        "python scripts/run_identity_controls.py commands",
        "python scripts/run_identity_controls.py provenance",
        "python scripts/run_identity_controls.py validate",
        "python scripts/run_identity_controls.py manifest",
    ]
    record = {
        "schema": "e04.s08.commands.v1",
        "researchStepId": "S08",
        "workingDirectory": str(REPOSITORY),
        "commands": commands,
    }
    write_json(output / "commands.json", record)
    return record


def write_provenance(
    output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR
) -> dict[str, Any]:
    upstream = {step: _verify_manifest(step) for step in EXPECTED_MANIFEST_HASHES}
    if not all(item["allPassed"] for item in upstream.values()):
        raise AssertionError("upstream immutability failed during provenance")
    paths = [
        WORKSPACE / "AGENTS.md",
        WORKSPACE / "FULL_PLAN.md",
        WORKSPACE / "RESEARCH_PLAN.md",
        WORKSPACE / "PREVIOUS_ARTIFACTS.md",
        WORKSPACE / "PREVIOUS_ARTIFACTS.json",
        WORKSPACE / "input-attachments/MANIFEST.json",
        WORKSPACE
        / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md",
        WORKSPACE
        / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/pdf-markdown.md",
        Path("/previous-artifacts/E01/specification/transition_spec.md"),
        Path(
            "/previous-artifacts/E01/release/reference_simulator/release_manifest.json"
        ),
        Path(
            "/previous-artifacts/E01/research_steps/S13/research_step_full_results.md"
        ),
        CONTRACT_PATH,
        REPOSITORY / "analysis/identity_controls.py",
        REPOSITORY / "scripts/run_identity_controls.py",
        REPOSITORY / "tests/test_e04_identity_controls.py",
    ]
    inputs = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]
    audit = {
        "schema": "e04.s08.upstream_immutability.v1",
        "researchStepId": "S08",
        "steps": upstream,
        "allPassed": all(item["allPassed"] for item in upstream.values()),
    }
    write_json(output / "upstream_immutability_audit.json", audit)
    record = {
        "schema": "e04.s08.provenance.v1",
        "researchStepId": "S08",
        "repository": str(REPOSITORY),
        "branch": _git_output("branch", "--show-current"),
        "head": _git_output("rev-parse", "HEAD"),
        "gitStatusShort": _git_output("status", "--short"),
        "contractSha256": sha256_file(CONTRACT_PATH),
        "inputs": inputs,
        "upstreamImmutability": upstream,
        "outputDirectory": str(output),
        "cacheDirectory": str(cache),
        "sourceRegime": "S07 native_control only",
        "s07JointRegimeUsed": False,
        "evidenceLayer": "E01 clean-room deterministic reference simulator",
        "historicalPublicationBytesClaimed": False,
    }
    write_json(output / "provenance.json", record)
    return record


def _table_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def validate_artifacts(output: Path = OUTPUT_DIR) -> dict[str, Any]:
    required = [
        "preregistration.json",
        "freeze_record.json",
        "identity_control_specification.md",
        "run_accounting.json",
        "transition_identity_audit.parquet",
        "transition_audit_summary.json",
        "s07_native_source_replay_audit.parquet",
        "policy_observation_audit.json",
        "control_assignment_manifest.parquet",
        "composition_audit.parquet",
        "identity_controls.parquet",
        "identity_control_trajectories.parquet",
        "ghost_channel_endpoints.parquet",
        "null_distributions/ghost_null_response_surface.parquet",
        "ghost_null_calibration.parquet",
        "posthoc_grouping_audit.parquet",
        "policy_label_discrimination.csv",
        "analysis_summary.json",
        "analysis_accounting.json",
        "success_criteria.json",
        "identity_control_trajectories.png",
        "identity_control_trajectories.svg",
        "policy_vs_label_endpoints.png",
        "policy_vs_label_endpoints.svg",
        "posthoc_label_selection.png",
        "posthoc_label_selection.svg",
        "value_correlated_label_artifact.png",
        "value_correlated_label_artifact.svg",
        "environment.json",
        "commands.json",
        "provenance.json",
        "upstream_immutability_audit.json",
        "repository_tests.junit.xml",
        "research_step_full_results.md",
    ]
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append(
            {"name": name, "passed": bool(passed), "detail": _json_native(detail)}
        )

    missing = [
        item
        for item in required
        if not (output / item).is_file() or (output / item).stat().st_size == 0
    ]
    add("required_files", not missing, {"missing": missing, "required": len(required)})
    freeze = json.loads((output / "freeze_record.json").read_text())
    add(
        "frozen_before_outcomes",
        freeze["frozenBeforeOutcomes"]
        and freeze["contractSha256"] == sha256_file(CONTRACT_PATH)
        and freeze["implementationSha256"] == sha256_file(Path(__file__).resolve()),
        freeze["contractSha256"],
    )
    accounting = json.loads((output / "run_accounting.json").read_text())
    add(
        "run_accounting",
        accounting["totalObserved"] == accounting["totalExpected"] == 4800,
        accounting,
    )
    expected_rows = {
        "transition": 450,
        "replay": EXPECTED_MIXED,
        "assignments": EXPECTED_MIXED + EXPECTED_HOMOGENEOUS,
        "channel_endpoints": (186 + 6) * CHANNELS * len(OUTCOMES),
        "null_response": (186 + 6) * len(GRID),
        "calibration": 2 * len(OUTCOMES) * 3,
        "discrimination": 6 * len(OUTCOMES),
    }
    observed_rows = {
        "transition": _table_rows(output / "transition_identity_audit.parquet"),
        "replay": _table_rows(output / "s07_native_source_replay_audit.parquet"),
        "assignments": _table_rows(output / "control_assignment_manifest.parquet"),
        "channel_endpoints": _table_rows(output / "ghost_channel_endpoints.parquet"),
        "null_response": _table_rows(
            output / "null_distributions/ghost_null_response_surface.parquet"
        ),
        "calibration": _table_rows(output / "ghost_null_calibration.parquet"),
        "discrimination": len(pd.read_csv(output / "policy_label_discrimination.csv")),
    }
    add(
        "complete_table_accounting",
        observed_rows == expected_rows,
        {"expected": expected_rows, "observed": observed_rows},
    )
    transition = pq.read_table(output / "transition_identity_audit.parquet").to_pandas()
    add(
        "transition_identity",
        len(transition) == 450 and bool(transition.passed.all()),
        {"rows": len(transition), "passed": int(transition.passed.sum())},
    )
    replay = pq.read_table(
        output / "s07_native_source_replay_audit.parquet"
    ).to_pandas()
    add(
        "s07_native_replay",
        len(replay) == EXPECTED_MIXED and bool(replay.passed.all()),
        {"rows": len(replay), "passed": int(replay.passed.sum())},
    )
    policy = json.loads((output / "policy_observation_audit.json").read_text())
    add("policy_observation_boundary", policy["passed"], policy)
    assignment = pq.read_table(
        output / "control_assignment_manifest.parquet"
    ).to_pandas()
    add(
        "exact_assignment_counts",
        bool(
            (assignment.primary_label_a_count == 50).all()
            and (assignment.primary_label_b_count == 50).all()
            and (assignment.all_channel_count_violations == 0).all()
        ),
        {
            "rows": len(assignment),
            "violations": int(assignment.all_channel_count_violations.sum()),
        },
    )
    calibration = pq.read_table(output / "ghost_null_calibration.parquet").to_pandas()
    add(
        "ghost_null_calibration",
        len(calibration) == 12
        and bool(
            (calibration.randomized_passed & calibration.conservative_passed).all()
        ),
        {
            "rows": len(calibration),
            "passed": int(
                (calibration.randomized_passed & calibration.conservative_passed).sum()
            ),
        },
    )
    controls = pq.read_table(output / "identity_controls.parquet").to_pandas()
    primary = controls[controls.role == "primary"]
    add(
        "multiplicity_complete",
        bool(primary.q_value.notna().all() and primary.q_value.between(0, 1).all()),
        {"primaryRows": len(primary)},
    )
    criteria = json.loads((output / "success_criteria.json").read_text())
    add(
        "success_rule",
        criteria["supportiveRulePassed"] and criteria["classification"] == "supportive",
        criteria,
    )
    upstream = json.loads((output / "upstream_immutability_audit.json").read_text())
    add("upstream_immutability", upstream["allPassed"], list(upstream["steps"]))
    report = (output / "research_step_full_results.md").read_text()
    fields = [
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
    ]
    add("report_required_fields", all(field in report for field in fields), fields)
    add("s09_absent", not S09_DIR.exists(), str(S09_DIR))
    record = {
        "schema": "e04.s08.validation_summary.v1",
        "researchStepId": "S08",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
    write_json(output / "validation_summary.json", record)
    if not record["passed"]:
        failed = [item["name"] for item in checks if not item["passed"]]
        raise AssertionError(f"S08 artifact validation failed: {failed}")
    return record


def write_artifact_manifest(output: Path = OUTPUT_DIR) -> dict[str, Any]:
    artifacts = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    record = {
        "schema": "e04.s08.artifact_manifest.v1",
        "researchStepId": "S08",
        "artifactCount": len(artifacts),
        "artifacts": artifacts,
    }
    write_json(output / "artifact_manifest.json", record)
    return record
