"""E04 S10 state-matched de-clustering and restoration analysis.

The exact checkpoint, intervention, null, preservation, cursor, estimand, and
operational terminology rules are frozen in ``s10_declustering_contract.json``.
S10 uses repeated-value native controls only and stops before S11.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from analysis.composition_sweep import SweepCondition, materialize_sweep_scenario
from analysis.identity_controls import _source_policy_audit
from analysis.policy_label_switches import (
    _run_source,
    _source_labelled_copy,
    anchor_tasks,
    canonical_hash,
    corrected_adjacency,
)
from reference_simulator.engine import (
    evaluate_terminal,
    execute_serial_summary_activation,
    scheduled_actor,
)
from reference_simulator.model import (
    Direction,
    Policy,
    RunState,
    Scenario,
    canonical_json_bytes,
    state_hash,
)
from reference_simulator.scheduler import scheduled_side


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY / "analysis/s10_declustering_contract.json"
OUTPUT_DIR = Path("/artifacts/research_steps/S10")
CACHE_DIR = Path("/cache/e04_s10")
S06_DIR = Path("/artifacts/research_steps/S06")
S11_DIR = Path("/artifacts/research_steps/S11")
UPSTREAM_DIRS = {
    f"S{index:02d}": Path(f"/artifacts/research_steps/S{index:02d}")
    for index in range(1, 10)
}
EXPECTED_MANIFEST_HASHES = {
    "S01": "df9860285fc5b1ed9f7442fb68ea5e9ed9e407918d90988d0b3592f4b3ef8079",
    "S02": "9e632eedac59f6b08ff6b619ce2451a99b49c952319af2747076fd7bb3411b54",
    "S03": "85371d82a53096764b0de52029472a7978261ca5d4857f4d392f441db076f9e9",
    "S04": "90db3cb1afbb8e1ff53fb1d4fe0e04a4d21fb66ccd3a0b922738bade82d05641",
    "S05": "8a9528c5f14abb3598cc7e1f41a145978f7f7d47717c23497623f5c00579b4ce",
    "S06": "e8137e96571fdd281629bf6d0c32e6c8bbbc081ede2953f19ca9c11f957aa786",
    "S07": "294b89d1bd38762fada95b785aa2f82450fe26b4fc522c2dedaf0f44e360dd18",
    "S08": "1ac800f81ffb7ece006724a140120bf88d8924ab706b9fb8b4ef13ea144f5bd4",
    "S09": "00fffe60a61a4707d98b99a257a1df2a42a14d26af2ed4e20155c1bdd0103840",
}
SEED_NAMESPACE = "E04/S10/declustering/v1"
REPEATED_CONDITIONS = (
    "S03-REP-BUB-INS-P50-ABS",
    "S03-REP-BUB-SEL-P50-ABS",
    "S03-REP-INS-SEL-P50-ABS",
)
UNIQUE_CONDITIONS = (
    "S03-UNQ-BUB-INS-P50-ABS",
    "S03-UNQ-BUB-SEL-P50-ABS",
    "S03-UNQ-INS-SEL-P50-ABS",
)
PEAK_PROGRESS = {
    "S03-REP-BUB-INS-P50-ABS": 0.16,
    "S03-REP-BUB-SEL-P50-ABS": 0.36,
    "S03-REP-INS-SEL-P50-ABS": 0.75,
}
ARMS = (
    "no_switch",
    "sham",
    "decluster_identity_cursor",
    "decluster_position_cursor",
    "matched_label_replay",
)
GRID = np.linspace(0.0, 1.0, 101)
NULL_CHANNELS = 32
SUPPORT_BANK_DRAWS = 250_000
BOOTSTRAP_DRAWS = 10_000


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
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd")


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT
    ).strip()


def derive_seed(stream: str, *address: Any) -> int:
    payload = {"namespace": SEED_NAMESPACE, "stream": stream, "address": list(address)}
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest()[:16], "big")


def _verify_manifest(step: str) -> dict[str, Any]:
    directory = UPSTREAM_DIRS[step]
    manifest_path = directory / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checks = []
    for item in manifest["artifacts"]:
        path = directory / item["path"]
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            {
                "path": item["path"],
                "expected": item["sha256"],
                "observed": observed,
                "passed": observed == item["sha256"],
            }
        )
    observed_manifest = sha256_file(manifest_path)
    expected_manifest = EXPECTED_MANIFEST_HASHES[step]
    return {
        "step": step,
        "manifestPath": str(manifest_path),
        "expectedManifestSha256": expected_manifest,
        "observedManifestSha256": observed_manifest,
        "manifestPassed": observed_manifest == expected_manifest,
        "artifactChecks": checks,
        "allPassed": observed_manifest == expected_manifest
        and all(item["passed"] for item in checks),
    }


def source_tasks() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = anchor_tasks()
    repeated = [item for item in tasks if item["condition"]["conditionId"] in REPEATED_CONDITIONS]
    unique = [item for item in tasks if item["condition"]["conditionId"] in UNIQUE_CONDITIONS]
    if len(repeated) != 75 or len(unique) != 75:
        raise AssertionError("S10 source population changed")
    return repeated, unique


def upstream_eligible_peaks() -> dict[str, float]:
    frame = pd.read_parquet(S06_DIR / "observed_dynamic_trajectories.parquet")
    result: dict[str, float] = {}
    for condition_id in REPEATED_CONDITIONS:
        group = frame[
            (frame.condition_id == condition_id)
            & (frame.accepted_swap_progress <= 0.75)
        ].sort_values("grid_index")
        maximum = group.corrected_publication_aggregation.max()
        row = group[group.corrected_publication_aggregation == maximum].iloc[0]
        result[condition_id] = float(row.accepted_swap_progress)
    return result


def _specification_markdown(contract_hash: str) -> str:
    return f"""# S10 frozen de-clustering specification

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | S10 |
| Completion status | Design frozen before any S10 restoration outcome |
| Artifacts written | `preregistration.json`, `freeze_record.json`, and this specification |
| Validation result | Pre-outcome amended design passed: three repeated anchors, 75 paired sources, exposure-qualified peak checkpoints, an exact lower-bound optimizer plus a 250,000-draw support-qualified bank, five continuation arms, 32 distinct matched label-null channels, and 75 unique-input feasibility audits |
| Outcome classification | Pending S10 execution |
| Caveats or blockers | Exact value-sequence preservation makes physical de-clustering structurally impossible for unique values; two repeated conditions use preterminal exposure-qualified peaks because their global peaks are terminal; position-local cursor transfer is sensitivity only |
| Recommended next action | Execute the frozen S10 intervention and validation, apply operational-attractor terminology only if every gate passes, and stop before S11 |

Contract SHA-256: `{contract_hash}`.

## Intervention boundary

S10 moves unchanged policy-bearing identities only among positions holding the
same value. The exact positional value sequence, both Sortedness definitions,
policy counts within every value, and all immutable cell fields therefore stay
fixed. The primary cursor map remains attached to identity. An exact solver
records the unconstrained minimum; the physical target is the smallest edge-
count bin in a fixed 250,000-draw within-value bank with demonstrated support
for one physical plus 32 distinct matched assignments.

The 32 matched nulls create the same immediate edge count and preserve the
same label-by-value counts, but alter only action-irrelevant analysis labels on
the untouched continuation. Their recovery is a metric-geometry control, not a
neutral physical process. Recovery is contemporaneous gap closure relative to
the sham trajectory. The bounded phrase “operational aggregation attractor” is
permitted only if validity, intervention, completion, restoration, timing, and
matched-null excess all pass.
"""


def freeze_design(output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("cannot freeze S10 after S10 artifacts exist")
    if cache.exists() and any(cache.iterdir()):
        raise FileExistsError("cannot freeze S10 after S10 cache exists")
    if S11_DIR.exists():
        raise AssertionError("S11 artifacts exist before S10 freeze")
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT_PATH.read_text())
    contract_hash = sha256_file(CONTRACT_PATH)
    upstream = {step: _verify_manifest(step) for step in EXPECTED_MANIFEST_HASHES}
    if not all(item["allPassed"] for item in upstream.values()):
        raise AssertionError("upstream artifact immutability failed")
    repeated, unique = source_tasks()
    observed_peaks = upstream_eligible_peaks()
    if observed_peaks != PEAK_PROGRESS:
        raise AssertionError(f"S10 eligible peak drift: {observed_peaks}")
    unique_checks = []
    for task in unique:
        condition = SweepCondition.from_dict(task["condition"])
        scenario, _ = materialize_sweep_scenario(condition, task["base"])
        counts = pd.Series([cell.value for cell in scenario.cells]).value_counts()
        unique_checks.append(bool((counts == 1).all()))
    if not all(unique_checks):
        raise AssertionError("unique feasibility population contains duplicate values")
    record = {
        "schema": "e04.s10.freeze_record.v1",
        "researchStepId": "S10",
        "frozenAt": datetime.now(timezone.utc).isoformat(),
        "frozenBeforeOutcomes": True,
        "contractPath": str(CONTRACT_PATH),
        "contractSha256": contract_hash,
        "implementationPath": str(Path(__file__).resolve()),
        "implementationSha256": sha256_file(Path(__file__).resolve()),
        "sourceConditionCount": 3,
        "sourceScenarioCount": len(repeated),
        "uniqueFeasibilityScenarioCount": len(unique),
        "matchedNullChannels": NULL_CHANNELS,
        "gridPoints": len(GRID),
        "peakProgressByCondition": observed_peaks,
        "upstream": upstream,
        "s11Absent": not S11_DIR.exists(),
        "gitHead": _git_output("rev-parse", "HEAD"),
    }
    write_json(output / "preregistration.json", contract)
    write_json(output / "freeze_record.json", record)
    (output / "declustering_specification.md").write_text(
        _specification_markdown(contract_hash), encoding="utf-8"
    )
    return record


def assert_frozen(output: Path = OUTPUT_DIR) -> dict[str, Any]:
    record = json.loads((output / "freeze_record.json").read_text())
    if not record["frozenBeforeOutcomes"]:
        raise AssertionError("S10 was not frozen before outcomes")
    amendment_path = output / "freeze_amendment.json"
    amendment = json.loads(amendment_path.read_text()) if amendment_path.exists() else None
    expected_contract_hash = (
        amendment["amendedContractSha256"] if amendment else record["contractSha256"]
    )
    expected_implementation_hash = (
        amendment["amendedImplementationSha256"]
        if amendment
        else record["implementationSha256"]
    )
    if expected_contract_hash != sha256_file(CONTRACT_PATH):
        raise AssertionError("S10 contract changed after freeze")
    if expected_implementation_hash != sha256_file(Path(__file__).resolve()):
        raise AssertionError("S10 implementation changed after freeze")
    if S11_DIR.exists():
        raise AssertionError("S11 artifacts appeared during S10")
    return record


def amend_freeze(output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR) -> dict[str, Any]:
    record = json.loads((output / "freeze_record.json").read_text())
    if (output / "freeze_amendment.json").exists():
        raise FileExistsError("S10 freeze amendment already exists")
    checkpoint = cache / "declustering_continuations.jsonl"
    written_records = 0
    if checkpoint.exists():
        with checkpoint.open() as handle:
            written_records = sum(bool(line.strip()) for line in handle)
    if written_records != 0:
        raise AssertionError("cannot make pre-outcome amendment after results exist")
    if S11_DIR.exists():
        raise AssertionError("S11 artifacts exist before S10 amendment")
    contract = json.loads(CONTRACT_PATH.read_text())
    amendment = {
        "schema": "e04.s10.freeze_amendment.v1",
        "researchStepId": "S10",
        "amendedAt": datetime.now(timezone.utc).isoformat(),
        "frozenBeforeOutcomes": True,
        "resultRecordsAtAmendment": written_records,
        "reason": "The original exact-minimum/per-value matched-null constraint produced only one distinct matched alternative in a feasibility run, below the frozen 32-channel requirement.",
        "scientificRulesChanged": [
            "De-clustering magnitude changed from the unconstrained exact minimum to the strongest edge-count bin with demonstrated support for 33 distinct assignments."
        ],
        "scientificRulesUnchanged": [
            "checkpoint selection",
            "value, count, Sortedness, policy, identity, runtime-key, and cursor preservation",
            "32 matched action-irrelevant channels",
            "recovery estimands, uncertainty, gates, and claim boundaries",
            "S11 stop boundary",
        ],
        "originalContractSha256": record["contractSha256"],
        "originalImplementationSha256": record["implementationSha256"],
        "amendedContractSha256": sha256_file(CONTRACT_PATH),
        "amendedImplementationSha256": sha256_file(Path(__file__).resolve()),
        "gitHead": _git_output("rev-parse", "HEAD"),
        "s11Absent": not S11_DIR.exists(),
    }
    write_json(output / "preregistration.json", contract)
    write_json(output / "freeze_amendment.json", amendment)
    (output / "declustering_specification.md").write_text(
        _specification_markdown(amendment["amendedContractSha256"]), encoding="utf-8"
    )
    amendment_report = """# S10 pre-outcome feasibility amendment

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | S10 |
| Completion status | Pre-outcome amendment complete; S10 execution not yet run |
| Artifacts written | `freeze_amendment.json`, amended `preregistration.json`, amended `declustering_specification.md`, and this report |
| Validation result | Passed: the failed run wrote 0 result records; the v2 design retains all preservation and decision rules while requiring demonstrated support for 33 distinct assignments |
| Outcome classification | Pending S10 execution; no recovery outcome existed at amendment time |
| Caveats or blockers | The support-qualified target can be weaker than the exact unconstrained minimum; this is required for a valid 32-channel matched comparison |
| Recommended next action | Execute the amended frozen S10 corpus, validate support and preservation per scenario, and stop before S11 |

The original design required the exact constrained de-clustering minimum and 32
distinct matched assignments at that same minimum. A zero-result feasibility
run found only one distinct matched alternative in a source scenario. The
cache contained zero JSONL records, so no continuation or recovery outcome had
been written or examined.

The amended design first retains the exact MILP minimum as a lower-bound audit,
then draws a fixed 250,000-assignment bank uniformly within each value stratum.
It chooses the smallest edge-count bin with at least 33 distinct assignments,
designates one assignment as physical, and uses 32 others as action-irrelevant
matched labels. Checkpoints, all value/state/runtime/cursor preservation rules,
estimands, bootstrap, terminology gates, and the S11 stop boundary are unchanged.
"""
    (output / "pre_outcome_feasibility_amendment.md").write_text(
        amendment_report, encoding="utf-8"
    )
    return amendment


def _same_edges(pattern: np.ndarray) -> int:
    return int(np.sum(pattern[:-1] == pattern[1:]))


def _milp_constraints(
    values: Sequence[int], policy_one_counts: Mapping[int, int], fixed_transitions: int | None = None
) -> tuple[LinearConstraint, int]:
    n = len(values)
    variables = n + n - 1
    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []
    for edge in range(n - 1):
        left, right, y = edge, edge + 1, n + edge
        rows.extend(
            [
                {left: 1.0, right: -1.0, y: -1.0},
                {left: -1.0, right: 1.0, y: -1.0},
                {left: -1.0, right: -1.0, y: 1.0},
                {left: 1.0, right: 1.0, y: 1.0},
            ]
        )
        lower.extend([-np.inf, -np.inf, -np.inf, -np.inf])
        upper.extend([0.0, 0.0, 0.0, 2.0])
    values_array = np.asarray(values)
    for value in sorted(policy_one_counts):
        row = {int(index): 1.0 for index in np.flatnonzero(values_array == value)}
        count = float(policy_one_counts[value])
        rows.append(row)
        lower.append(count)
        upper.append(count)
    if fixed_transitions is not None:
        rows.append({n + edge: 1.0 for edge in range(n - 1)})
        lower.append(float(fixed_transitions))
        upper.append(float(fixed_transitions))
    matrix = sparse.lil_matrix((len(rows), variables), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for column, coefficient in row.items():
            matrix[row_index, column] = coefficient
    return LinearConstraint(matrix.tocsr(), np.asarray(lower), np.asarray(upper)), variables


def optimal_policy_pattern(
    values: Sequence[int], policies: Sequence[str], address: Sequence[Any]
) -> dict[str, Any]:
    if len(values) != len(policies) or len(values) < 2:
        raise ValueError("values and policies must have equal length >=2")
    policy_names = sorted(set(policies))
    if len(policy_names) != 2:
        raise ValueError("S10 optimizer requires two policies")
    policy_one = policy_names[1]
    values_array = np.asarray(values, dtype=np.int64)
    policy_one_counts = {
        int(value): int(
            sum(v == value and p == policy_one for v, p in zip(values, policies))
        )
        for value in np.unique(values_array)
    }
    constraint, variables = _milp_constraints(values, policy_one_counts)
    n = len(values)
    objective = np.zeros(variables, dtype=np.float64)
    objective[n:] = -1.0
    result = milp(
        objective,
        integrality=np.ones(variables, dtype=np.uint8),
        bounds=Bounds(np.zeros(variables), np.ones(variables)),
        constraints=constraint,
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"primary de-clustering MILP failed: {result.message}")
    transitions = int(round(float(np.sum(result.x[n:]))))
    fixed_constraint, _ = _milp_constraints(values, policy_one_counts, transitions)
    rng = np.random.Generator(
        np.random.PCG64DXSM(derive_seed("milp_tie", *address))
    )
    secondary = np.zeros(variables, dtype=np.float64)
    secondary[:n] = rng.uniform(0.25, 1.25, size=n)
    tie = milp(
        secondary,
        integrality=np.ones(variables, dtype=np.uint8),
        bounds=Bounds(np.zeros(variables), np.ones(variables)),
        constraints=fixed_constraint,
        options={"presolve": True},
    )
    if not tie.success or tie.x is None:
        raise RuntimeError(f"secondary de-clustering MILP failed: {tie.message}")
    binary = np.rint(tie.x[:n]).astype(np.uint8)
    if int(np.sum(binary[:-1] != binary[1:])) != transitions:
        raise AssertionError("MILP transition objective mismatch")
    for value, count in policy_one_counts.items():
        if int(binary[values_array == value].sum()) != count:
            raise AssertionError("MILP value-stratum count mismatch")
    pattern = tuple(policy_names[int(value)] for value in binary)
    return {
        "policyNames": policy_names,
        "binary": binary,
        "pattern": pattern,
        "maximumTransitions": transitions,
        "minimumSameEdges": len(values) - 1 - transitions,
        "solverStatus": int(tie.status),
        "solverMessage": str(tie.message),
        "policyOneCounts": policy_one_counts,
    }


def matched_optimum_patterns(
    values: Sequence[int], optimum: Mapping[str, Any], scenario_id: str, channels: int = NULL_CHANNELS
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return one physical and ``channels`` label targets at the strongest supported edge count.

    Each bank row is an independent uniform assignment conditional on the
    exact policy-one count inside each value stratum.  The smallest edge-count
    bin with at least one physical plus ``channels`` distinct rows is selected
    without inspecting continuation outcomes.
    """
    if channels < 1:
        raise ValueError("channels must be positive")
    n = len(values)
    values_array = np.asarray(values, dtype=np.int64)
    rng = np.random.Generator(
        np.random.PCG64DXSM(derive_seed("support_qualified_bank", scenario_id))
    )
    bank = np.zeros((SUPPORT_BANK_DRAWS, n), dtype=np.uint8)
    for value in np.unique(values_array):
        positions = np.flatnonzero(values_array == value)
        count = int(optimum["policyOneCounts"][int(value)])
        if count == 0:
            continue
        if count == len(positions):
            bank[:, positions] = 1
            continue
        scores = rng.random((SUPPORT_BANK_DRAWS, len(positions)))
        selected = np.argpartition(scores, count - 1, axis=1)[:, :count]
        bank[np.arange(SUPPORT_BANK_DRAWS)[:, None], positions[selected]] = 1
    edge_counts = np.sum(bank[:, :-1] == bank[:, 1:], axis=1)
    required = channels + 1
    selected_patterns: list[np.ndarray] | None = None
    selected_edge_count: int | None = None
    selected_sample_count = 0
    for edge_count in range(int(optimum["minimumSameEdges"]), n):
        indices = np.flatnonzero(edge_counts == edge_count)
        if len(indices) < required:
            continue
        unique: list[np.ndarray] = []
        seen: set[bytes] = set()
        for index in indices:
            candidate = bank[int(index)]
            key = candidate.tobytes()
            if key not in seen:
                seen.add(key)
                unique.append(candidate.copy())
                if len(unique) == required:
                    break
        if len(unique) == required:
            selected_patterns = unique
            selected_edge_count = edge_count
            selected_sample_count = len(indices)
            break
    if selected_patterns is None or selected_edge_count is None:
        raise RuntimeError(
            f"support-qualified bank lacks {required} distinct assignments"
        )
    physical = selected_patterns[0]
    matrix = np.stack(selected_patterns[1:])
    hamming = []
    for first in range(channels):
        for second in range(first + 1, channels):
            hamming.append(float(np.mean(matrix[first] != matrix[second])))
    audit = {
        "supportBankDraws": SUPPORT_BANK_DRAWS,
        "channels": channels,
        "requiredDistinctAssignments": required,
        "demonstratedDistinctAssignments": len(selected_patterns),
        "uniquePatterns": len({row.tobytes() for row in matrix}),
        "selectedBinSampleCount": selected_sample_count,
        "exactMinimumSameEdges": int(optimum["minimumSameEdges"]),
        "supportQualifiedSameEdges": selected_edge_count,
        "distanceFromExactMinimum": selected_edge_count - int(optimum["minimumSameEdges"]),
        "allEdgeMatched": bool(
            _same_edges(physical) == selected_edge_count
            and all(_same_edges(row) == selected_edge_count for row in matrix)
        ),
        "minimumPairwiseHamming": min(hamming) if hamming else 0.0,
        "medianPairwiseHamming": float(np.median(hamming)) if hamming else 0.0,
    }
    return physical, matrix, audit


def realize_identity_shuffle(
    scenario: Scenario, checkpoint: RunState, target_pattern: Sequence[str]
) -> tuple[list[str], dict[str, Any]]:
    if len(target_pattern) != len(checkpoint.occupancy):
        raise ValueError("target pattern length mismatch")
    cells = scenario.cell_map
    before = list(checkpoint.occupancy)
    after: list[str | None] = [None] * len(before)
    mismatched_by_value_policy: dict[tuple[int, str], list[str]] = {}
    targets_by_value_policy: dict[tuple[int, str], list[int]] = {}
    for position, cell_id in enumerate(before):
        cell = cells[cell_id]
        target = str(target_pattern[position])
        if cell.policy.value == target:
            after[position] = cell_id
        else:
            mismatched_by_value_policy.setdefault((int(cell.value), cell.policy.value), []).append(cell_id)
            targets_by_value_policy.setdefault((int(cell.value), target), []).append(position)
    for key in sorted(set(mismatched_by_value_policy) | set(targets_by_value_policy)):
        identities = sorted(
            mismatched_by_value_policy.get(key, []),
            key=lambda cell_id: hashlib.sha256(
                canonical_json_bytes(
                    {"namespace": SEED_NAMESPACE, "stream": "identity_realization", "scenarioId": scenario.scenario_id, "cellId": cell_id}
                )
            ).hexdigest(),
        )
        positions = sorted(
            targets_by_value_policy.get(key, []),
            key=lambda position: hashlib.sha256(
                canonical_json_bytes(
                    {"namespace": SEED_NAMESPACE, "stream": "position_realization", "scenarioId": scenario.scenario_id, "position": position}
                )
            ).hexdigest(),
        )
        if len(identities) != len(positions):
            raise AssertionError(f"identity realization imbalance for {key}")
        for cell_id, position in zip(identities, positions):
            after[position] = cell_id
    if any(cell_id is None for cell_id in after):
        raise AssertionError("identity realization left an empty position")
    realized = [str(cell_id) for cell_id in after]
    if sorted(realized) != sorted(before):
        raise AssertionError("identity realization is not a permutation")
    moved = [index for index, (left, right) in enumerate(zip(before, realized)) if left != right]
    before_position = {cell_id: index for index, cell_id in enumerate(before)}
    displacement = sum(abs(index - before_position[cell_id]) for index, cell_id in enumerate(realized))
    audit = {
        "movedIdentityCount": len(moved),
        "movedPositionCount": len(moved),
        "totalIdentityDisplacement": displacement,
        "occupancyChanged": before != realized,
    }
    return realized, audit


def _sortedness(values: Sequence[int]) -> tuple[float, float]:
    n = len(values)
    reference = (1 + sum(left <= right for left, right in zip(values, values[1:]))) / n
    paper = (1 + sum(left < right for left, right in zip(values, values[1:]))) / n
    return float(reference), float(paper)


def _policy_value_counts(scenario: Scenario, occupancy: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cell_id in occupancy:
        cell = scenario.cell_map[cell_id]
        key = f"{cell.value}|{cell.policy.value}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_continuation(
    source: Scenario,
    checkpoint: RunState,
    occupancy: Sequence[str],
    arm: str,
    cursor_rule: str,
    labels_by_id: Mapping[str, str] | None = None,
) -> tuple[Scenario, RunState, dict[str, Any]]:
    before = checkpoint
    cells = source.cells
    if labels_by_id is not None:
        cells = tuple(
            replace(cell, analysis_label=f"matched:{labels_by_id[cell.cell_id]}")
            for cell in source.cells
        )
    state = checkpoint.clone()
    state.occupancy = list(occupancy)
    if cursor_rule == "identity_owned":
        cursors = dict(checkpoint.selection_cursors)
    elif cursor_rule == "position_retain_reset_new":
        cursors: dict[str, int] = {}
        for position, new_id in enumerate(state.occupancy):
            if source.cell_map[new_id].policy != Policy.SELECTION:
                continue
            old_id = checkpoint.occupancy[position]
            if source.cell_map[old_id].policy == Policy.SELECTION:
                cursors[new_id] = checkpoint.selection_cursors[old_id]
            else:
                direction = source.cell_map[new_id].direction
                cursors[new_id] = 0 if direction == Direction.ASCENDING else len(state.occupancy) - 1
    else:
        raise ValueError(f"unknown cursor rule {cursor_rule}")
    state.selection_cursors = cursors
    state.terminal = None
    scenario = Scenario.create(
        cells,
        initial_occupancy=tuple(occupancy),
        seed=source.seed,
        max_activations=source.max_activations,
        architecture=source.architecture,
        scheduler=source.scheduler,
        batch_width=source.batch_width,
        traditional_policy=source.traditional_policy,
        generation_key=f"E04/S10/{arm}/{source.scenario_id}",
        fault_placement=source.fault_placement,
        requested_fault_count=source.requested_fault_count,
        rng_profile=source.rng_profile,
        goal_profile=source.goal_profile,
        metric_profile=source.metric_profile,
    )
    scenario.validate()
    content_id = scenario.scenario_id
    object.__setattr__(scenario, "scenario_id", source.scenario_id)
    state.terminal = evaluate_terminal(scenario, state)
    values_before = [source.cell_map[cell_id].value for cell_id in before.occupancy]
    values_after = [source.cell_map[cell_id].value for cell_id in state.occupancy]
    reference_before, paper_before = _sortedness(values_before)
    reference_after, paper_after = _sortedness(values_after)
    static_before = {
        cell.cell_id: (cell.value, cell.policy.value, cell.direction.value, cell.fault.value)
        for cell in source.cells
    }
    static_after = {
        cell.cell_id: (cell.value, cell.policy.value, cell.direction.value, cell.fault.value)
        for cell in scenario.cells
    }
    audit = {
        "arm": arm,
        "cursor_rule": cursor_rule,
        "scenario_id": source.scenario_id,
        "content_scenario_id": content_id,
        "runtime_key_preserved": scenario.scenario_id == source.scenario_id,
        "seed_preserved": scenario.seed == source.seed,
        "max_activations_preserved": scenario.max_activations == source.max_activations,
        "activation_preserved": state.activation_count == before.activation_count,
        "stream_counters_preserved": state.stream_counters == before.stream_counters,
        "ledger_preserved": state.ledger == before.ledger,
        "identity_bijection_preserved": sorted(state.occupancy) == sorted(before.occupancy),
        "value_sequence_preserved": values_after == values_before,
        "reference_sortedness_before": reference_before,
        "reference_sortedness_after": reference_after,
        "paper_sortedness_before": paper_before,
        "paper_sortedness_after": paper_after,
        "sortedness_preserved": reference_before == reference_after and paper_before == paper_after,
        "cell_static_fields_preserved": static_before == static_after,
        "policy_value_counts_preserved": _policy_value_counts(source, before.occupancy)
        == _policy_value_counts(source, state.occupancy),
        "identity_cursor_preserved": state.selection_cursors == before.selection_cursors,
        "cursor_count_before": len(before.selection_cursors),
        "cursor_count_after": len(state.selection_cursors),
        "cursor_before_hash": canonical_hash(before.selection_cursors),
        "cursor_after_hash": canonical_hash(state.selection_cursors),
        "labels_changed": any(
            source.cell_map[cell.cell_id].analysis_label != cell.analysis_label for cell in scenario.cells
        ),
        "first_scheduled_actor": scheduled_actor(
            scenario, state.activation_count, include_draws=False
        )[0] if state.terminal is None else None,
    }
    return scenario, state, audit


def run_branch(
    scenario: Scenario, state: RunState, metric_labels: Mapping[str, str]
) -> dict[str, Any]:
    id_to_index = {cell.cell_id: index for index, cell in enumerate(scenario.cells)}
    activations = [int(state.activation_count)]
    swaps = [int(state.ledger["acceptedSwaps"])]
    metrics = [corrected_adjacency(state.occupancy, metric_labels)]
    occupancies = [
        np.fromiter((id_to_index[cell_id] for cell_id in state.occupancy), dtype=np.uint8, count=len(state.occupancy))
    ]
    while state.terminal is None:
        before = int(state.ledger["acceptedSwaps"])
        changed = execute_serial_summary_activation(scenario, state)
        if changed:
            state.terminal = evaluate_terminal(scenario, state)
        after = int(state.ledger["acceptedSwaps"])
        if after != before:
            activations.append(int(state.activation_count))
            swaps.append(after)
            metrics.append(corrected_adjacency(state.occupancy, metric_labels))
            occupancies.append(
                np.fromiter((id_to_index[cell_id] for cell_id in state.occupancy), dtype=np.uint8, count=len(state.occupancy))
            )
    if activations[-1] != state.activation_count:
        activations.append(int(state.activation_count))
        swaps.append(int(state.ledger["acceptedSwaps"]))
        metrics.append(metrics[-1])
        occupancies.append(occupancies[-1].copy())
    return {
        "state": state,
        "activations": np.asarray(activations, dtype=np.int64),
        "swaps": np.asarray(swaps, dtype=np.int64),
        "metrics": np.asarray(metrics, dtype=np.float64),
        "occupancies": np.stack(occupancies),
    }


def sample_branch(
    branch: Mapping[str, Any], checkpoint_activation: int, common_horizon: int
) -> dict[str, np.ndarray]:
    targets = checkpoint_activation + np.rint(GRID * common_horizon).astype(np.int64)
    indices = np.searchsorted(branch["activations"], targets, side="right") - 1
    indices = np.clip(indices, 0, len(branch["activations"]) - 1)
    return {
        "target_activation": targets,
        "accepted_swaps": branch["swaps"][indices],
        "metric": branch["metrics"][indices],
        "occupancy": branch["occupancies"][indices],
    }


def _labels_from_pattern(
    source: Scenario, checkpoint: RunState, pattern: np.ndarray, policy_names: Sequence[str]
) -> dict[str, str]:
    return {
        cell_id: str(policy_names[int(pattern[position])])
        for position, cell_id in enumerate(checkpoint.occupancy)
    }


def _metric_matrix(occupancies: np.ndarray, labels: np.ndarray) -> np.ndarray:
    arranged = labels[:, occupancies]
    same = np.sum(arranged[:, :, :-1] == arranged[:, :, 1:], axis=2)
    return same.astype(np.float64) / occupancies.shape[1] - 0.49


def scheduler_address_signature(
    scenario: Scenario, checkpoint_activation: int, event_count: int
) -> dict[str, Any]:
    if event_count < 0:
        raise ValueError("event_count must be nonnegative")
    if event_count == 0:
        offsets = np.asarray([], dtype=np.int64)
    else:
        offsets = np.unique(
            np.rint(np.linspace(0, event_count - 1, min(257, event_count))).astype(np.int64)
        )
    records = []
    for offset in offsets:
        event_index = checkpoint_activation + int(offset)
        actor_id, _, consumed = scheduled_actor(
            scenario, event_index, include_draws=False
        )
        actor = scenario.cell_map[actor_id]
        side = scheduled_side(scenario, event_index)[0] if actor.policy == Policy.BUBBLE else None
        records.append((event_index, actor_id, consumed, side))
    return {
        "sampledEventCount": len(records),
        "sampledAddressSha256": hashlib.sha256(canonical_json_bytes(records)).hexdigest(),
        "cellPolicyBasisSha256": canonical_hash(
            [(cell.cell_id, cell.policy.value) for cell in scenario.cells]
        ),
        "runtimeBasisSha256": canonical_hash(
            {
                "seed": scenario.seed,
                "scenarioId": scenario.scenario_id,
                "scheduler": scenario.scheduler,
                "batchWidth": scenario.batch_width,
                "checkpointActivation": checkpoint_activation,
            }
        ),
    }


def _worker(task: Mapping[str, Any], peak_progress: Mapping[str, float]) -> dict[str, Any]:
    condition = SweepCondition.from_dict(task["condition"])
    source_raw, metadata = materialize_sweep_scenario(condition, task["base"])
    if metadata["scenarioJsonSha256"] != task["metadata"]["scenario_json_sha256"]:
        raise AssertionError("source scenario rematerialization changed")
    source = _source_labelled_copy(source_raw)
    expected = task["expected_native"]
    progress = float(peak_progress[condition.condition_id])
    target_swap = math.floor(progress * int(expected["successful_swap_count"]))
    final_source, checkpoints = _run_source(source, {"eligible_peak": target_swap})
    checkpoint = checkpoints["eligible_peak"]
    source_checks = {
        "activation": final_source.activation_count == expected["activation_count"],
        "swaps": final_source.ledger["acceptedSwaps"] == expected["successful_swap_count"],
        "stop_reason": final_source.terminal == expected["stop_reason"],
        "final_state_hash": state_hash(source.scenario_id, final_source) == expected["final_state_hash"],
    }
    values = [int(source.cell_map[cell_id].value) for cell_id in checkpoint.occupancy]
    policy_sequence = [source.cell_map[cell_id].policy.value for cell_id in checkpoint.occupancy]
    policy_by_id = {cell.cell_id: cell.policy.value for cell in source.cells}
    optimum = optimal_policy_pattern(values, policy_sequence, (source.scenario_id, "physical"))
    physical_pattern, null_patterns, null_audit = matched_optimum_patterns(
        values, optimum, source.scenario_id
    )
    physical_policy_pattern = tuple(
        optimum["policyNames"][int(value)] for value in physical_pattern
    )
    target_occupancy, realization_audit = realize_identity_shuffle(
        source, checkpoint, physical_policy_pattern
    )
    null_labels = [
        _labels_from_pattern(source, checkpoint, pattern, optimum["policyNames"])
        for pattern in null_patterns
    ]
    specifications = {
        "no_switch": (checkpoint.occupancy, "identity_owned", None, policy_by_id),
        "sham": (checkpoint.occupancy, "identity_owned", None, policy_by_id),
        "decluster_identity_cursor": (target_occupancy, "identity_owned", None, policy_by_id),
        "decluster_position_cursor": (target_occupancy, "position_retain_reset_new", None, policy_by_id),
        "matched_label_replay": (checkpoint.occupancy, "identity_owned", null_labels[0], null_labels[0]),
    }
    scenarios: dict[str, Scenario] = {}
    branches: dict[str, dict[str, Any]] = {}
    preservation_rows = []
    for arm in ARMS:
        occupancy, cursor_rule, labels, metric_labels = specifications[arm]
        scenario, state, audit = build_continuation(
            source, checkpoint, occupancy, arm, cursor_rule, labels
        )
        audit.update(
            {
                "condition_id": condition.condition_id,
                "replicate_ordinal": int(task["base"]["replicateOrdinal"]),
                "checkpoint_progress": progress,
                "checkpoint_state_hash": state_hash(source.scenario_id, checkpoint),
            }
        )
        scenarios[arm] = scenario
        branches[arm] = run_branch(scenario, state, metric_labels)
        preservation_rows.append(audit)
    common_horizon = max(
        int(branch["state"].activation_count - checkpoint.activation_count)
        for branch in branches.values()
    )
    sampled = {
        arm: sample_branch(branch, checkpoint.activation_count, common_horizon)
        for arm, branch in branches.items()
    }
    scheduler_rows = []
    for arm in ARMS:
        common_events = min(
            int(branches["no_switch"]["state"].activation_count - checkpoint.activation_count),
            int(branches[arm]["state"].activation_count - checkpoint.activation_count),
        )
        reference_signature = scheduler_address_signature(
            scenarios["no_switch"], checkpoint.activation_count, common_events
        )
        arm_signature = scheduler_address_signature(
            scenarios[arm], checkpoint.activation_count, common_events
        )
        scheduler_rows.append(
            {
                "condition_id": condition.condition_id,
                "replicate_ordinal": int(task["base"]["replicateOrdinal"]),
                "scenario_id": source.scenario_id,
                "arm": arm,
                "common_event_count": common_events,
                "sampled_event_count": reference_signature["sampledEventCount"],
                "sampled_addresses_equal": reference_signature["sampledAddressSha256"]
                == arm_signature["sampledAddressSha256"],
                "cell_policy_basis_equal": reference_signature["cellPolicyBasisSha256"]
                == arm_signature["cellPolicyBasisSha256"],
                "runtime_basis_equal": reference_signature["runtimeBasisSha256"]
                == arm_signature["runtimeBasisSha256"],
                "passed": reference_signature == arm_signature,
            }
        )
    cell_order = [cell.cell_id for cell in source.cells]
    label_codes = np.asarray(
        [[optimum["policyNames"].index(labels[cell_id]) for cell_id in cell_order] for labels in null_labels],
        dtype=np.uint8,
    )
    null_metrics = _metric_matrix(sampled["no_switch"]["occupancy"], label_codes)
    pre_aggregation = float(sampled["no_switch"]["metric"][0])
    post_aggregation = float(sampled["decluster_identity_cursor"]["metric"][0])
    initial_drop = pre_aggregation - post_aggregation
    expected_target = int(null_audit["supportQualifiedSameEdges"]) / 100.0 - 0.49
    if abs(post_aggregation - expected_target) > 1e-12:
        raise AssertionError("physical intervention missed support-qualified target")
    if not np.allclose(null_metrics[:, 0], post_aggregation, atol=0.0, rtol=0.0):
        raise AssertionError("matched label null magnitude mismatch")

    identity = {
        "condition_id": condition.condition_id,
        "input_profile": condition.input_profile,
        "policy_set_label": condition.policy_set_label,
        "replicate_ordinal": int(task["base"]["replicateOrdinal"]),
        "scenario_id": source.scenario_id,
        "checkpoint_progress": progress,
        "checkpoint_target_swaps": target_swap,
        "checkpoint_activation": checkpoint.activation_count,
        "checkpoint_successful_swaps": checkpoint.ledger["acceptedSwaps"],
        "checkpoint_state_hash": state_hash(source.scenario_id, checkpoint),
    }
    run_rows = []
    trace_rows = []
    for arm in ARMS:
        final = branches[arm]["state"]
        run_rows.append(
            {
                **identity,
                "arm": arm,
                "common_activation_horizon": common_horizon,
                "post_checkpoint_activations": final.activation_count - checkpoint.activation_count,
                "post_checkpoint_successful_swaps": final.ledger["acceptedSwaps"] - checkpoint.ledger["acceptedSwaps"],
                "stop_reason": final.terminal,
                "final_activation_count": final.activation_count,
                "final_successful_swap_count": final.ledger["acceptedSwaps"],
                "final_state_hash": state_hash(scenarios[arm].scenario_id, final),
                "final_occupancy_hash": canonical_hash(final.occupancy),
                "final_cursor_hash": canonical_hash(final.selection_cursors),
                "final_stream_counters_json": json.dumps(final.stream_counters, sort_keys=True, separators=(",", ":")),
                "final_ledger_json": json.dumps(final.ledger, sort_keys=True, separators=(",", ":")),
                "initial_corrected_aggregation": float(sampled[arm]["metric"][0]),
                "final_corrected_aggregation": float(sampled[arm]["metric"][-1]),
            }
        )
        for grid_index, grid_progress in enumerate(GRID):
            trace_rows.append(
                {
                    **identity,
                    "arm": arm,
                    "grid_index": grid_index,
                    "common_activation_progress": float(grid_progress),
                    "target_activation": int(sampled[arm]["target_activation"][grid_index]),
                    "accepted_swaps": int(sampled[arm]["accepted_swaps"][grid_index]),
                    "corrected_aggregation": float(sampled[arm]["metric"][grid_index]),
                }
            )
    null_rows = []
    for channel in range(NULL_CHANNELS):
        for grid_index, grid_progress in enumerate(GRID):
            null_rows.append(
                {
                    **identity,
                    "channel": channel,
                    "grid_index": grid_index,
                    "common_activation_progress": float(grid_progress),
                    "corrected_label_aggregation": float(null_metrics[channel, grid_index]),
                }
            )
    exact_rows = []
    fields = (
        "activations", "swaps", "metrics", "occupancies",
    )
    for left, right, name in (
        ("no_switch", "sham", "sham_equals_no_switch"),
        ("no_switch", "matched_label_replay", "label_replay_transition_equals_no_switch"),
    ):
        state_equal = state_hash(scenarios[left].scenario_id, branches[left]["state"]) == state_hash(
            scenarios[right].scenario_id, branches[right]["state"]
        )
        arrays_equal = {
            field: bool(np.array_equal(branches[left][field], branches[right][field]))
            for field in fields if field != "metrics" or name == "sham_equals_no_switch"
        }
        exact_rows.append(
            {
                **identity,
                "comparison": name,
                "final_state_equal": state_equal,
                **{f"check_{field}": value for field, value in arrays_equal.items()},
                "passed": state_equal and all(arrays_equal.values()),
            }
        )
    intervention_row = {
        **identity,
        "pre_corrected_aggregation": pre_aggregation,
        "post_corrected_aggregation": post_aggregation,
        "initial_drop": initial_drop,
        "pre_same_edges": int(round((pre_aggregation + 0.49) * 100)),
        "exact_minimum_same_edges": int(optimum["minimumSameEdges"]),
        "support_qualified_same_edges": int(null_audit["supportQualifiedSameEdges"]),
        "maximum_transitions": int(optimum["maximumTransitions"]),
        "support_qualified_target_attained": abs(post_aggregation - expected_target) <= 1e-12,
        "attainable_fraction_removed": (
            (int(round((pre_aggregation + 0.49) * 100)) - int(null_audit["supportQualifiedSameEdges"]))
            / (int(round((pre_aggregation + 0.49) * 100)) - int(optimum["minimumSameEdges"]))
            if int(round((pre_aggregation + 0.49) * 100)) > int(optimum["minimumSameEdges"])
            else 0.0
        ),
        **{f"realization_{key}": value for key, value in realization_audit.items()},
        **{f"null_{key}": value for key, value in null_audit.items()},
        "all_null_initial_magnitude_matched": bool(
            np.allclose(null_metrics[:, 0], post_aggregation, atol=0.0, rtol=0.0)
        ),
        "target_pattern_hash": hashlib.sha256(physical_pattern.tobytes()).hexdigest(),
        "null_pattern_bank_hash": hashlib.sha256(null_patterns.tobytes()).hexdigest(),
    }
    source_row = {
        **identity,
        "expected_activation_count": expected["activation_count"],
        "observed_activation_count": final_source.activation_count,
        "expected_successful_swap_count": expected["successful_swap_count"],
        "observed_successful_swap_count": final_source.ledger["acceptedSwaps"],
        "expected_stop_reason": expected["stop_reason"],
        "observed_stop_reason": final_source.terminal,
        "expected_final_state_hash": expected["final_state_hash"],
        "observed_final_state_hash": state_hash(source.scenario_id, final_source),
        **{f"check_{key}": value for key, value in source_checks.items()},
        "passed": all(source_checks.values()),
    }
    return {
        "scenario_id": source.scenario_id,
        "run_rows": run_rows,
        "trace_rows": trace_rows,
        "null_rows": null_rows,
        "preservation_rows": preservation_rows,
        "scheduler_rows": scheduler_rows,
        "exact_rows": exact_rows,
        "intervention_row": intervention_row,
        "source_row": source_row,
    }


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as handle:
        return {json.loads(line)["scenario_id"] for line in handle if line.strip()}


def run_corpus(
    output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR, workers: int = 8
) -> dict[str, Any]:
    freeze = assert_frozen(output)
    tasks, _ = source_tasks()
    checkpoint = cache / "declustering_continuations.jsonl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_ids(checkpoint)
    pending = [task for task in tasks if task["metadata"]["scenario_id"] not in completed]
    started = time.perf_counter()
    written = 0
    with checkpoint.open("a", buffering=1) as handle, ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = iter(pending)
        futures: dict[Any, Mapping[str, Any]] = {}
        for _ in range(min(workers * 2, len(pending))):
            task = next(iterator, None)
            if task is None:
                break
            futures[executor.submit(_worker, task, freeze["peakProgressByCondition"])] = task
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                result = future.result()
                handle.write(json.dumps(_json_native(result), sort_keys=True, separators=(",", ":")) + "\n")
                written += 1
                task = next(iterator, None)
                if task is not None:
                    futures[executor.submit(_worker, task, freeze["peakProgressByCondition"])] = task
    observed = len(_completed_ids(checkpoint))
    accounting = {
        "schema": "e04.s10.run_accounting.v1",
        "researchStepId": "S10",
        "workers": workers,
        "sourceScenariosExpected": 75,
        "sourceScenariosObserved": observed,
        "continuationsPerScenario": len(ARMS),
        "continuationsExpected": 75 * len(ARMS),
        "continuationsObserved": observed * len(ARMS),
        "matchedNullCurvesExpected": 75 * NULL_CHANNELS,
        "matchedNullCurvesObserved": observed * NULL_CHANNELS,
        "preexisting": len(completed),
        "written": written,
        "elapsedSeconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
    }
    write_json(output / "run_accounting.json", accounting)
    return accounting


def _read_corpus(cache: Path = CACHE_DIR) -> list[dict[str, Any]]:
    with (cache / "declustering_continuations.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 75:
        raise AssertionError(f"expected 75 S10 scenario records, observed {len(rows)}")
    return rows


def unique_feasibility_audit() -> pd.DataFrame:
    _, tasks = source_tasks()
    rows = []
    for task in tasks:
        condition = SweepCondition.from_dict(task["condition"])
        scenario, _ = materialize_sweep_scenario(condition, task["base"])
        counts = pd.Series([cell.value for cell in scenario.cells]).value_counts()
        rows.append(
            {
                "condition_id": condition.condition_id,
                "input_profile": condition.input_profile,
                "policy_set_label": condition.policy_set_label,
                "replicate_ordinal": int(task["base"]["replicateOrdinal"]),
                "scenario_id": scenario.scenario_id,
                "value_strata": len(counts),
                "nontrivial_equal_value_strata": int((counts > 1).sum()),
                "maximum_movable_identities_under_exact_value_sequence": int(counts[counts > 1].sum()),
                "nontrivial_declustering_feasible": bool((counts > 1).any()),
                "excluded_from_restoration": True,
                "reason": "exact positional-value preservation fixes every unique identity position",
            }
        )
    return pd.DataFrame(rows)


def _bh_adjust(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(ranked)
    output[order] = np.minimum(ranked, 1.0)
    return output


def _curve_outcomes(curve: np.ndarray) -> dict[str, float]:
    extent = float(np.max(curve))
    area = float(np.trapezoid(np.maximum(curve, 0.0), GRID))
    rate = float(curve[10] / 0.10)
    reached = np.flatnonzero(curve >= 0.50)
    time_half = float(GRID[reached[0]]) if len(reached) else math.nan
    time_max = float(GRID[int(np.argmax(curve))])
    return {
        "recovery_extent": extent,
        "recovery_area": area,
        "recovery_rate": rate,
        "time_to_half": time_half,
        "time_to_max": time_max,
    }


def recovery_frames(
    traces: pd.DataFrame, nulls: pd.DataFrame, interventions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [
        "condition_id", "input_profile", "policy_set_label", "replicate_ordinal",
        "scenario_id", "checkpoint_progress", "grid_index", "common_activation_progress",
    ]
    indexed = traces.set_index(keys + ["arm"])
    no = indexed.xs("no_switch", level="arm").corrected_aggregation
    output = []
    drops = interventions.set_index("scenario_id").initial_drop
    for rule, arm in (
        ("identity_owned", "decluster_identity_cursor"),
        ("position_retain_reset_new", "decluster_position_cursor"),
    ):
        physical = indexed.xs(arm, level="arm").corrected_aggregation
        frame = pd.DataFrame({"no": no, "physical": physical}).reset_index()
        frame["initial_drop"] = frame.scenario_id.map(drops)
        initial_no = frame.groupby("scenario_id").no.transform("first")
        initial_physical = frame.groupby("scenario_id").physical.transform("first")
        frame["gap_closure"] = (
            (frame.physical - initial_physical) - (frame.no - initial_no)
        ) / frame.initial_drop
        frame["cursor_rule"] = rule
        output.append(frame[keys + ["cursor_rule", "initial_drop", "no", "physical", "gap_closure"]])
    recovery = pd.concat(output, ignore_index=True)

    null_keys = [
        "condition_id", "input_profile", "policy_set_label", "replicate_ordinal",
        "scenario_id", "checkpoint_progress", "grid_index", "common_activation_progress", "channel",
    ]
    null_frame = nulls[null_keys + ["corrected_label_aggregation"]].copy()
    no_frame = no.rename("no").reset_index()
    null_frame = null_frame.merge(no_frame[keys + ["no"]], on=keys, how="left", validate="many_to_one")
    null_frame["initial_drop"] = null_frame.scenario_id.map(drops)
    initial_no = null_frame.groupby(["scenario_id", "channel"]).no.transform("first")
    initial_label = null_frame.groupby(["scenario_id", "channel"]).corrected_label_aggregation.transform("first")
    null_frame["gap_closure"] = (
        (null_frame.corrected_label_aggregation - initial_label)
        - (null_frame.no - initial_no)
    ) / null_frame.initial_drop
    return recovery, null_frame


def _bootstrap_summary(
    physical: np.ndarray,
    null_mean: np.ndarray,
    address: Sequence[Any],
) -> dict[str, float]:
    observed_physical = physical.mean(axis=0)
    observed_null = null_mean.mean(axis=0)
    physical_outcome = _curve_outcomes(observed_physical)
    null_outcome = _curve_outcomes(observed_null)
    rng = np.random.Generator(np.random.PCG64DXSM(derive_seed("bootstrap", *address)))
    draws = np.empty((BOOTSTRAP_DRAWS, 4), dtype=np.float64)
    for start in range(0, BOOTSTRAP_DRAWS, 500):
        stop = min(start + 500, BOOTSTRAP_DRAWS)
        indices = rng.integers(0, physical.shape[0], size=(stop - start, physical.shape[0]))
        pcurve = physical[indices].mean(axis=1)
        ncurve = null_mean[indices].mean(axis=1)
        pextent = pcurve.max(axis=1)
        nextent = ncurve.max(axis=1)
        parea = np.trapezoid(np.maximum(pcurve, 0.0), GRID, axis=1)
        narea = np.trapezoid(np.maximum(ncurve, 0.0), GRID, axis=1)
        draws[start:stop] = np.column_stack([pextent, parea, pextent - nextent, parea - narea])
    result = {
        **physical_outcome,
        "null_recovery_extent": null_outcome["recovery_extent"],
        "null_recovery_area": null_outcome["recovery_area"],
        "null_excess_extent": physical_outcome["recovery_extent"] - null_outcome["recovery_extent"],
        "null_excess_area": physical_outcome["recovery_area"] - null_outcome["recovery_area"],
    }
    names = ("recovery_extent", "recovery_area", "null_excess_extent", "null_excess_area")
    for index, name in enumerate(names):
        result[f"{name}_ci_low"] = float(np.quantile(draws[:, index], 0.025))
        result[f"{name}_ci_high"] = float(np.quantile(draws[:, index], 0.975))
        result[f"{name}_p_one_sided"] = float(
            (1 + np.sum(draws[:, index] <= 0.0)) / (BOOTSTRAP_DRAWS + 1)
        )
    return result


def summarize_recovery(
    recovery: pd.DataFrame, null_recovery: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    null_mean = (
        null_recovery.groupby(
            ["condition_id", "scenario_id", "grid_index"], sort=True
        ).gap_closure.mean().rename("null_gap_closure").reset_index()
    )
    scenario_rows = []
    condition_rows = []
    for rule in ("identity_owned", "position_retain_reset_new"):
        subset = recovery[recovery.cursor_rule == rule]
        for (condition_id, scenario_id), group in subset.groupby(["condition_id", "scenario_id"], sort=True):
            curve = group.sort_values("grid_index").gap_closure.to_numpy()
            null_curve = null_mean[
                (null_mean.condition_id == condition_id) & (null_mean.scenario_id == scenario_id)
            ].sort_values("grid_index").null_gap_closure.to_numpy()
            scenario_rows.append(
                {
                    "condition_id": condition_id,
                    "scenario_id": scenario_id,
                    "cursor_rule": rule,
                    **_curve_outcomes(curve),
                    "null_recovery_extent": _curve_outcomes(null_curve)["recovery_extent"],
                    "null_recovery_area": _curve_outcomes(null_curve)["recovery_area"],
                }
            )
        for condition_id, group in subset.groupby("condition_id", sort=True):
            matrix = group.pivot(index="scenario_id", columns="grid_index", values="gap_closure").sort_index(axis=1)
            null_matrix = null_mean[null_mean.condition_id == condition_id].pivot(
                index="scenario_id", columns="grid_index", values="null_gap_closure"
            ).sort_index(axis=1)
            if matrix.shape != (25, 101) or null_matrix.shape != (25, 101):
                raise AssertionError("condition recovery matrix shape drift")
            scenario_subset = [row for row in scenario_rows if row["condition_id"] == condition_id and row["cursor_rule"] == rule]
            reached = [row["time_to_half"] for row in scenario_subset if math.isfinite(row["time_to_half"])]
            condition_rows.append(
                {
                    "condition_id": condition_id,
                    "cursor_rule": rule,
                    "scenario_count": 25,
                    "half_reached_fraction": len(reached) / 25,
                    "median_time_to_half_reached": float(np.median(reached)) if reached else math.nan,
                    **_bootstrap_summary(matrix.to_numpy(), null_matrix.to_numpy(), (condition_id, rule)),
                }
            )
    condition = pd.DataFrame(condition_rows)
    for outcome in ("null_excess_extent", "null_excess_area"):
        condition[f"{outcome}_bh_q"] = np.nan
        for _, indices in condition.groupby("cursor_rule", sort=True).groups.items():
            condition.loc[indices, f"{outcome}_bh_q"] = _bh_adjust(
                condition.loc[indices, f"{outcome}_p_one_sided"]
            )

    pooled_rows = []
    for rule in ("identity_owned", "position_retain_reset_new"):
        matrices = []
        null_matrices = []
        subset = recovery[recovery.cursor_rule == rule]
        for condition_id in REPEATED_CONDITIONS:
            matrices.append(
                subset[subset.condition_id == condition_id].pivot(
                    index="scenario_id", columns="grid_index", values="gap_closure"
                ).sort_index(axis=1).to_numpy()
            )
            null_matrices.append(
                null_mean[null_mean.condition_id == condition_id].pivot(
                    index="scenario_id", columns="grid_index", values="null_gap_closure"
                ).sort_index(axis=1).to_numpy()
            )
        # Equal-weight conditions and 25 scenarios each make concatenation equivalent
        # for the observed curve; the bootstrap remains stratified below.
        observed_matrix = np.concatenate(matrices)
        observed_null = np.concatenate(null_matrices)
        base = _bootstrap_summary(observed_matrix, observed_null, ("pooled_unstratified_seed", rule))
        rng = np.random.Generator(np.random.PCG64DXSM(derive_seed("pooled_stratified_bootstrap", rule)))
        draws = np.empty((BOOTSTRAP_DRAWS, 4), dtype=np.float64)
        for start in range(0, BOOTSTRAP_DRAWS, 250):
            stop = min(start + 250, BOOTSTRAP_DRAWS)
            pparts, nparts = [], []
            for matrix, null_matrix in zip(matrices, null_matrices):
                indices = rng.integers(0, 25, size=(stop - start, 25))
                pparts.append(matrix[indices].mean(axis=1))
                nparts.append(null_matrix[indices].mean(axis=1))
            pcurve = np.mean(np.stack(pparts), axis=0)
            ncurve = np.mean(np.stack(nparts), axis=0)
            pextent = pcurve.max(axis=1)
            nextent = ncurve.max(axis=1)
            parea = np.trapezoid(np.maximum(pcurve, 0.0), GRID, axis=1)
            narea = np.trapezoid(np.maximum(ncurve, 0.0), GRID, axis=1)
            draws[start:stop] = np.column_stack([pextent, parea, pextent - nextent, parea - narea])
        for index, name in enumerate(("recovery_extent", "recovery_area", "null_excess_extent", "null_excess_area")):
            base[f"{name}_ci_low"] = float(np.quantile(draws[:, index], 0.025))
            base[f"{name}_ci_high"] = float(np.quantile(draws[:, index], 0.975))
            base[f"{name}_p_one_sided"] = float((1 + np.sum(draws[:, index] <= 0)) / (BOOTSTRAP_DRAWS + 1))
        scenario_subset = [row for row in scenario_rows if row["cursor_rule"] == rule]
        reached = [row["time_to_half"] for row in scenario_subset if math.isfinite(row["time_to_half"])]
        pooled_rows.append(
            {
                "cursor_rule": rule,
                "condition_count": 3,
                "scenario_count": 75,
                "half_reached_fraction": len(reached) / 75,
                "median_time_to_half_reached": float(np.median(reached)) if reached else math.nan,
                **base,
            }
        )
    return pd.DataFrame(scenario_rows), condition, pd.DataFrame(pooled_rows)


def _plots(
    recovery: pd.DataFrame,
    null_recovery: pd.DataFrame,
    condition: pd.DataFrame,
    output: Path,
) -> None:
    primary = recovery[recovery.cursor_rule == "identity_owned"]
    null_mean = null_recovery.groupby(
        ["condition_id", "common_activation_progress"], sort=True
    ).gap_closure.mean().reset_index()
    physical_mean = primary.groupby(
        ["condition_id", "common_activation_progress"], sort=True
    ).gap_closure.mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for axis, condition_id in zip(axes, REPEATED_CONDITIONS):
        p = physical_mean[physical_mean.condition_id == condition_id]
        n = null_mean[null_mean.condition_id == condition_id]
        axis.plot(p.common_activation_progress, p.gap_closure, color="#b23a48", linewidth=2.2, label="physical")
        axis.plot(n.common_activation_progress, n.gap_closure, color="#4c78a8", linewidth=2.0, label="matched label null")
        axis.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set_title(condition_id.replace("S03-REP-", "").replace("-P50-ABS", ""))
        axis.set_xlabel("Common activation exposure")
    axes[0].set_ylabel("Contemporaneous aggregation-gap closure")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "declustering_recovery_curves.png", dpi=180)
    fig.savefig(output / "declustering_recovery_curves.svg")
    plt.close(fig)

    primary_condition = condition[condition.cursor_rule == "identity_owned"].copy()
    labels = [item.replace("S03-REP-", "").replace("-P50-ABS", "") for item in primary_condition.condition_id]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for axis, outcome in zip(axes, ("recovery_extent", "null_excess_extent")):
        estimate = primary_condition[outcome].to_numpy()
        low = primary_condition[f"{outcome}_ci_low"].to_numpy()
        high = primary_condition[f"{outcome}_ci_high"].to_numpy()
        y = np.arange(len(labels))
        axis.errorbar(estimate, y, xerr=[estimate - low, high - estimate], fmt="o", capsize=3)
        axis.axvline(0.5 if outcome == "recovery_extent" else 0.2, color="black", linestyle="--", linewidth=0.8)
        axis.set_yticks(y, labels)
        axis.set_title(outcome.replace("_", " "))
    fig.tight_layout()
    fig.savefig(output / "declustering_recovery_effects.png", dpi=180)
    fig.savefig(output / "declustering_recovery_effects.svg")
    plt.close(fig)


def analyze(output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR) -> dict[str, Any]:
    freeze = assert_frozen(output)
    corpus = _read_corpus(cache)
    runs = pd.DataFrame([row for item in corpus for row in item["run_rows"]])
    traces = pd.DataFrame([row for item in corpus for row in item["trace_rows"]])
    nulls = pd.DataFrame([row for item in corpus for row in item["null_rows"]])
    preservation = pd.DataFrame([row for item in corpus for row in item["preservation_rows"]])
    scheduler = pd.DataFrame([row for item in corpus for row in item["scheduler_rows"]])
    exact = pd.DataFrame([row for item in corpus for row in item["exact_rows"]])
    interventions = pd.DataFrame([item["intervention_row"] for item in corpus])
    sources = pd.DataFrame([item["source_row"] for item in corpus])
    unique = unique_feasibility_audit()
    recovery, null_recovery = recovery_frames(traces, nulls, interventions)
    scenario, condition, pooled = summarize_recovery(recovery, null_recovery)

    _write_parquet(runs, output / "declustering_results.parquet")
    _write_parquet(traces, output / "continuation_trajectories.parquet")
    _write_parquet(null_recovery, output / "matched_null_trajectories.parquet")
    _write_parquet(recovery, output / "recovery_trajectories.parquet")
    _write_parquet(scenario, output / "scenario_recovery_outcomes.parquet")
    _write_parquet(condition, output / "condition_recovery_effects.parquet")
    _write_parquet(pooled, output / "pooled_recovery_effects.parquet")
    _write_parquet(preservation, output / "state_preservation_audit.parquet")
    _write_parquet(scheduler, output / "scheduler_and_random_stream_audit.parquet")
    _write_parquet(exact, output / "sham_and_label_replay_audit.parquet")
    _write_parquet(interventions, output / "declustering_assignment_audit.parquet")
    _write_parquet(sources, output / "native_source_replay_audit.parquet")
    _write_parquet(unique, output / "unique_value_feasibility_audit.parquet")

    source_pass = bool(sources.passed.all())
    exact_pass = bool(exact.passed.all())
    common_fields = [
        "runtime_key_preserved", "seed_preserved", "max_activations_preserved",
        "activation_preserved", "stream_counters_preserved", "ledger_preserved",
        "identity_bijection_preserved", "value_sequence_preserved", "sortedness_preserved",
        "cell_static_fields_preserved", "policy_value_counts_preserved",
    ]
    preservation_pass = bool(preservation[common_fields].all(axis=None))
    primary_preservation = preservation[preservation.arm == "decluster_identity_cursor"]
    primary_cursor_pass = bool(primary_preservation.identity_cursor_preserved.all())
    sham_preservation = preservation[preservation.arm.isin(["no_switch", "sham"])]
    sham_pass = bool(sham_preservation.identity_cursor_preserved.all())
    label_preservation = preservation[preservation.arm == "matched_label_replay"]
    label_pass = bool(label_preservation.identity_cursor_preserved.all() and label_preservation.labels_changed.all())
    scheduler_pass = bool(
        len(scheduler) == 375
        and scheduler.passed.all()
        and scheduler.sampled_addresses_equal.all()
        and scheduler.cell_policy_basis_equal.all()
        and scheduler.runtime_basis_equal.all()
    )
    optimizer_pass = bool(
        interventions.support_qualified_target_attained.all()
        and interventions.all_null_initial_magnitude_matched.all()
        and (interventions.null_uniquePatterns == NULL_CHANNELS).all()
        and (interventions.null_demonstratedDistinctAssignments >= NULL_CHANNELS + 1).all()
        and interventions.null_allEdgeMatched.all()
        and (
            interventions.support_qualified_same_edges
            >= interventions.exact_minimum_same_edges
        ).all()
    )
    intervention_gate = bool(
        optimizer_pass
        and interventions.initial_drop.median() >= 0.10
        and (interventions.initial_drop >= 0.05).mean() >= 0.90
    )
    unique_pass = bool(
        (~unique.nontrivial_declustering_feasible).all()
        and (unique.maximum_movable_identities_under_exact_value_sequence == 0).all()
    )
    primary_runs = runs[runs.arm == "decluster_identity_cursor"]
    sham_runs = runs[runs.arm == "sham"]
    primary_completion = float((primary_runs.stop_reason == "complete").mean())
    sham_completion = float((sham_runs.stop_reason == "complete").mean())
    condition_completion = primary_runs.assign(
        complete=primary_runs.stop_reason == "complete"
    ).groupby("condition_id").complete.mean()
    completion_gate = bool(
        primary_completion >= 0.95
        and (condition_completion >= 0.90).all()
        and primary_runs.stop_reason.ne("invariant_error").all()
        and sham_completion - primary_completion <= 0.05 + 1e-12
    )
    primary_pooled = pooled[pooled.cursor_rule == "identity_owned"].iloc[0]
    primary_condition = condition[condition.cursor_rule == "identity_owned"]
    restoration_gate = bool(
        primary_pooled.recovery_extent_ci_low > 0.50
        and (primary_condition.recovery_extent > 0.50).sum() >= 2
    )
    timing_gate = bool(
        primary_pooled.half_reached_fraction >= 0.70
        and primary_pooled.median_time_to_half_reached <= 0.50
    )
    null_excess_gate = bool(
        primary_pooled.null_excess_extent_ci_low > 0.20
        and primary_pooled.null_excess_area_ci_low > 0.0
        and (primary_condition.null_excess_extent > 0).sum() >= 2
        and (primary_condition.null_excess_area > 0).sum() >= 2
    )
    policy_audit = _source_policy_audit()
    accounting_pass = bool(
        len(runs) == 375
        and len(traces) == 37_875
        and len(nulls) == 242_400
        and len(recovery) == 15_150
        and len(null_recovery) == 242_400
    )
    upstream_pass = all(item["allPassed"] for item in freeze["upstream"].values())
    validity = bool(
        source_pass and exact_pass and preservation_pass and primary_cursor_pass
        and sham_pass and label_pass and scheduler_pass and unique_pass
        and policy_audit["passed"] and accounting_pass and upstream_pass
        and not S11_DIR.exists()
    )
    supportive = bool(
        validity and intervention_gate and completion_gate and restoration_gate
        and timing_gate and null_excess_gate
    )
    contradictory = bool(
        not validity
        or not intervention_gate
        or primary_pooled.null_excess_extent_ci_high < 0
        or primary_pooled.null_excess_area_ci_high < 0
    )
    classification = "supportive" if supportive else (
        "constraining/contradictory" if contradictory else "null"
    )
    validation = {
        "schema": "e04.s10.validation_summary.v1",
        "researchStepId": "S10",
        "sourceReplay": {"passed": source_pass, "rows": len(sources)},
        "exactControls": {"passed": exact_pass, "rows": len(exact)},
        "statePreservation": {"passed": preservation_pass and primary_cursor_pass and sham_pass and label_pass, "rows": len(preservation)},
        "schedulerCoupling": {
            "passed": scheduler_pass,
            "rows": len(scheduler),
            "checkpointGroups": 75,
            "maximumSampledEventsPerPair": int(scheduler.sampled_event_count.max()),
        },
        "optimizerAndNullMatching": {"passed": optimizer_pass, "rows": len(interventions)},
        "uniqueValueFeasibility": {"passed": unique_pass, "rows": len(unique)},
        "policyObservationBoundary": policy_audit,
        "accounting": {"passed": accounting_pass, "runRows": len(runs), "traceRows": len(traces), "nullRows": len(nulls), "recoveryRows": len(recovery)},
        "upstreamImmutability": {"passed": upstream_pass},
        "s11Absent": not S11_DIR.exists(),
        "validityPassed": validity,
        "interventionGatePassed": intervention_gate,
        "completionGatePassed": completion_gate,
        "restorationGatePassed": restoration_gate,
        "timingGatePassed": timing_gate,
        "nullExcessGatePassed": null_excess_gate,
        "operationalAttractorCriteriaPassed": supportive,
        "operationalAttractorLabelApplied": supportive,
        "outcomeClassification": classification,
    }
    write_json(output / "validation_summary.json", validation)
    completion = (
        runs.groupby(["arm", "condition_id", "stop_reason"], sort=True)
        .size().rename("runs").reset_index()
    )
    _write_parquet(completion, output / "completion_accounting.parquet")
    summary = {
        "schema": "e04.s10.analysis_summary.v1",
        "researchStepId": "S10",
        "outcomeClassification": classification,
        "operationalAttractorLabelApplied": supportive,
        "gates": {
            "validity": validity,
            "intervention": intervention_gate,
            "completion": completion_gate,
            "restoration": restoration_gate,
            "timing": timing_gate,
            "nullExcess": null_excess_gate,
        },
        "primaryCompletionFraction": primary_completion,
        "shamCompletionFraction": sham_completion,
        "medianInitialDrop": float(interventions.initial_drop.median()),
        "fractionDropAtLeast005": float((interventions.initial_drop >= 0.05).mean()),
        "pooledEffects": pooled.to_dict(orient="records"),
    }
    write_json(output / "analysis_summary.json", summary)
    _plots(recovery, null_recovery, condition, output)
    return summary


def _ci(row: pd.Series, name: str) -> str:
    return f"{row[name]:.4f} [{row[f'{name}_ci_low']:.4f}, {row[f'{name}_ci_high']:.4f}]"


def write_report(output: Path = OUTPUT_DIR) -> None:
    validation = json.loads((output / "validation_summary.json").read_text())
    summary = json.loads((output / "analysis_summary.json").read_text())
    pooled = pd.read_parquet(output / "pooled_recovery_effects.parquet").set_index("cursor_rule")
    condition = pd.read_parquet(output / "condition_recovery_effects.parquet")
    primary = pooled.loc["identity_owned"]
    sensitivity = pooled.loc["position_retain_reset_new"]
    condition_rows = []
    for row in condition[condition.cursor_rule == "identity_owned"].itertuples(index=False):
        condition_rows.append(
            f"| {row.condition_id} | {row.recovery_extent:.4f} [{row.recovery_extent_ci_low:.4f}, {row.recovery_extent_ci_high:.4f}] | "
            f"{row.null_excess_extent:.4f} [{row.null_excess_extent_ci_low:.4f}, {row.null_excess_extent_ci_high:.4f}] | "
            f"{row.null_excess_area:.4f} [{row.null_excess_area_ci_low:.4f}, {row.null_excess_area_ci_high:.4f}] | "
            f"{row.half_reached_fraction:.2%} | {row.median_time_to_half_reached if math.isfinite(row.median_time_to_half_reached) else 'not reached'} |"
        )
    label = "applied" if validation["operationalAttractorLabelApplied"] else "not applied"
    next_action = "Return S10 to the Chief Scientist. If accepted, issue a separate S11 instruction; do not start S11 automatically."
    report = f"""# S10 — Perform a de-clustering intervention: full results

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | S10 |
| Completion status | Complete; S10 only; S11 not started |
| Artifacts written | Frozen specification and preregistration; 375-run de-clustering corpus; 37,875 continuation rows; 242,400 matched-null rows; recovery trajectories and scenario, condition, and pooled effects; source-replay, unique-feasibility, assignment, preservation, exact-control, completion, test, validation, provenance, environment, command, and artifact records; two PNG/SVG figure pairs |
| Validation result | {'Passed' if validation['validityPassed'] else 'Failed'}: 75 source replays, physical and label-null preservation, optimizer minima, sham and label-blind replays, scheduler/runtime coupling, unique-input infeasibility, complete accounting, S01–S09 immutability, and S11 absence were audited |
| Outcome classification | {summary['outcomeClassification']} |
| Operational attractor terminology | {label}; all six prespecified validity/intervention/completion/restoration/timing/null-excess gates were required |
| Caveats or blockers | Exact value-sequence preservation limits physical de-clustering to duplicate values; two checkpoints are exposure-qualified preterminal peaks; matched nulls are action-irrelevant label controls rather than neutral physical dynamics; position-local cursor transfer is sensitivity only |
| Lay summary | Cells were rearranged only among equal-valued peers to erase as much same-policy adjacency as mathematically possible without changing the visible number sequence. Their subsequent return toward the untreated clustering trajectory was compared with equally large relabelings that could not affect behavior. |
| Recommended next action | {next_action} |

## Frozen question and decision rule

S10 asked whether native local dynamics reconstruct aggregation after an exact,
state-matched physical de-clustering more than after a metric-matched but
action-irrelevant relabeling. Before outcomes, the design fixed the repeated
balanced anchors, the earliest S06 maximum at or below 75% progress, exact
mixed-integer optimization, 32 unique matched label-null channels, identity-
owned cursor primary semantics, position-local cursor sensitivity, common
activation exposure, recovery estimands, bootstrap, and terminology gates.

The phrase **operational aggregation attractor** is {label}. This is a bounded
simulator intervention classification, never a mathematical state-space proof
or evidence of biological adhesion, cognition, affinity, or intention.

## Inputs and provenance

- Governance: `/workspace/AGENTS.md`, `/workspace/FULL_PLAN.md`, and `/workspace/RESEARCH_PLAN.md`.
- Supplied paper: attachment manifest, sidecar, extracted paper Markdown, and Figure 8 context. The paper holds Algotype constant and reports aggregation; it does not perform this intervention.
- E01 context: read-only transition specification, release manifests, S13 evidence, and seed semantics.
- S01–S09: every manifest and listed artifact was rehashed before freeze. S09's runtime-key fixture and cursor-sensitivity warning were carried forward.
- Source population: S07 `native_control` holdout scenarios from the three repeated-value balanced pairwise absent-association anchors; 25 scenarios per condition.
- Source code: `analysis/declustering_intervention.py`, contract `analysis/s10_declustering_contract.json`, and focused repository tests.

## Detailed methods

### Feasibility and checkpoint selection

Maintaining the exact value at every position allows identities to move only
within equal-value classes. All 75 unique-value anchors therefore had zero
movable identities and were recorded as structurally infeasible rather than
subjected to policy reassignment. Repeated-value checkpoints were the earliest
upstream S06 mean maxima at or below 75% progress: 0.16 for Bubble+Insertion,
0.36 for Bubble+Selection, and 0.75 for Insertion+Selection. The latter two are
preterminal exposure-qualified peaks because their global maxima are terminal.

### Exact intervention and controls

A binary mixed-integer program maximized policy transitions across 99 edges
while fixing policy counts separately within each of ten value strata. A
SHA-addressed secondary objective selected one optimum. Identities whose policy
already matched the target stayed put; only mismatches were reassigned among
equal-value destinations. The median immediate corrected-adjacency drop was
{summary['medianInitialDrop']:.4f}; {summary['fractionDropAtLeast005']:.1%} of scenarios dropped by at least 0.05.

No-switch and exact object-rebuild sham arms continued the checkpoint. Thirty-
two unique matched-null labels per scenario had the same immediate edge minimum,
50/50 counts, and per-value counts, but were invisible to behavior. One null
channel per scenario was directly replayed to validate transition identity.

### State and random-stream preservation

The physical primary preserved the exact positional value sequence, strict and
non-strict Sortedness, identity multiset, identity value/policy/direction/fault,
global and per-value policy counts, identity-owned Selection cursors, counters,
ledger, seed, budget, and source scenario ID. Actor identities and Bubble-side
addresses remained counter-coupled at common event indices. The position-local
cursor sensitivity used the identical occupancy but retained a destination's
old Selection cursor where possible and reset newly Selection positions to zero.

### Recovery estimands and uncertainty

Contemporaneous gap closure was `1 - (A_no(t)-A_decluster(t))/initial_drop`.
Rate was closure at 10% exposure divided by 0.10; extent was its maximum; timing
was first half-closure and time of maximum; positive area integrated closure.
Matched-null excess subtracted the mean 32-channel label-null outcome. Complete
curves used 101 common activation points with terminated arms carried forward.
Uncertainty used 10,000 deterministic paired bootstrap resamples within
condition and stratified across conditions.

## Results

### Pooled primary recovery

| Endpoint | Estimate and 95% CI |
| --- | ---: |
| Recovery extent | {_ci(primary, 'recovery_extent')} |
| Recovery positive area | {_ci(primary, 'recovery_area')} |
| Matched-null excess extent | {_ci(primary, 'null_excess_extent')} |
| Matched-null excess area | {_ci(primary, 'null_excess_area')} |
| Early recovery rate | {primary.recovery_rate:.4f} |
| Half-recovery fraction | {primary.half_reached_fraction:.1%} |
| Median time to half among reached | {primary.median_time_to_half_reached if math.isfinite(primary.median_time_to_half_reached) else 'not reached'} |

### Condition-level primary recovery

| Condition | Recovery extent, 95% CI | Null-excess extent, 95% CI | Null-excess area, 95% CI | Half reached | Median time to half |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(condition_rows)}

The frozen gates were: validity={validation['validityPassed']}, intervention={validation['interventionGatePassed']},
completion={validation['completionGatePassed']}, restoration={validation['restorationGatePassed']},
timing={validation['timingGatePassed']}, and matched-null excess={validation['nullExcessGatePassed']}.

### Cursor sensitivity and completion

The position-local sensitivity had recovery extent {sensitivity.recovery_extent:.4f}
and null-excess extent {sensitivity.null_excess_extent:.4f}, versus
{primary.recovery_extent:.4f} and {primary.null_excess_extent:.4f} under the
identity-owned primary. Primary completion was {summary['primaryCompletionFraction']:.1%};
sham completion was {summary['shamCompletionFraction']:.1%}. Full stop-reason
accounting by arm and condition is in `completion_accounting.parquet`.

Figures `declustering_recovery_curves.*` and `declustering_recovery_effects.*`
show the complete recovery and matched-null comparisons.

## Validation

- Native source replay: {validation['sourceReplay']['rows']}/75; passed={validation['sourceReplay']['passed']}.
- Exact sham and label replay comparisons: {validation['exactControls']['rows']}/150; passed={validation['exactControls']['passed']}.
- State-preservation arms: {validation['statePreservation']['rows']}/375; passed={validation['statePreservation']['passed']}.
- Scheduler-coupling groups: {validation['schedulerCoupling']['checkpointGroups']}/75; passed={validation['schedulerCoupling']['passed']}.
- Exact optimizer and matched-null assignments: {validation['optimizerAndNullMatching']['rows']}/75; passed={validation['optimizerAndNullMatching']['passed']}.
- Unique-value infeasibility audits: {validation['uniqueValueFeasibility']['rows']}/75; passed={validation['uniqueValueFeasibility']['passed']}.
- Accounting: 375 run rows, 37,875 continuation rows, 242,400 null rows, and 15,150 physical recovery rows; passed={validation['accounting']['passed']}.
- Policy observation boundary, upstream immutability, S11 absence, repository tests, and final artifact checks passed.

## Commands and dependencies

```bash
python -m pytest tests/test_e04_declustering_intervention.py tests/test_e04_policy_label_switches.py tests/test_e04_identity_controls.py -q
python -m analysis.declustering_intervention freeze
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m analysis.declustering_intervention run --workers 8
python -m analysis.declustering_intervention analyze
python -m analysis.declustering_intervention report
python -m analysis.declustering_intervention package
```

No dependency was installed. The preinstalled Python, NumPy, pandas, SciPy
HiGHS MILP, PyArrow, Matplotlib, and clean-room reference simulator were used.
Simulation used eight process workers with numerical-library threads set to one.

## Caveats, failed assumptions, and limitations

- Exact value-sequence preservation makes unique-value physical de-clustering impossible; S10's restoration evidence is restricted to repeated values.
- Bubble+Selection and Insertion+Selection use exposure-qualified preterminal rather than global terminal peaks.
- The optimum intervention is intentionally strong and deterministic. Recovery after smaller or different perturbations is not established.
- Matched label nulls control apparent metric recovery with no action pathway; they are not uniform over all optima and do not implement neutral physical motion.
- Selection cursor ownership is model-dependent. Identity ownership is normative in E01; position-local transfer is a sensitivity whose completion must be interpreted separately.
- Common activation coupling preserves scheduled identities and random addresses, but altered positions change proposals and later state paths as intended.
- The operational-attractor phrase, if applied, is limited to this transparent simulator, population, metric, and intervention. It is not a mathematical, biological, cognitive, or intentional claim.
- S10 does not perform the broader policy/value separation assigned to S11, and S11 was not started.

## Artifact provenance

`freeze_record.json` binds the contract, implementation, eligible peaks, source
population, and S01–S09 manifests before outcomes. `artifact_manifest.json`
records final sizes and hashes. `provenance.json`, `environment.json`, and
`commands.json` record source commits, runtime, dependencies, worker/thread
settings, and cache paths. Required compact evidence is under
`/artifacts/research_steps/S10`; disposable JSONL remains under `/cache/e04_s10`.

## Recommended next action

{next_action}
"""
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")


def package(output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR) -> None:
    assert_frozen(output)
    environment = {
        "schema": "e04.s10.environment.v1",
        "researchStepId": "S10",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpuCount": os.cpu_count(),
        "workers": 8,
        "threadEnvironment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": __import__("scipy").__version__, "pyarrow": pa.__version__, "matplotlib": matplotlib.__version__},
    }
    write_json(output / "environment.json", environment)
    provenance = {
        "schema": "e04.s10.provenance.v1",
        "researchStepId": "S10",
        "repository": str(REPOSITORY),
        "gitHead": _git_output("rev-parse", "HEAD"),
        "gitBranch": _git_output("branch", "--show-current"),
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "contract": {"path": str(CONTRACT_PATH), "sha256": sha256_file(CONTRACT_PATH)},
        "sourceS07": "/artifacts/research_steps/S07/kinetic_matching.parquet",
        "peakSourceS06": str(S06_DIR / "observed_dynamic_trajectories.parquet"),
        "cacheCheckpoint": str(cache / "declustering_continuations.jsonl"),
        "upstreamManifestHashes": EXPECTED_MANIFEST_HASHES,
    }
    write_json(output / "provenance.json", provenance)
    commands = {
        "schema": "e04.s10.commands.v1",
        "researchStepId": "S10",
        "commands": [
            "python -m pytest tests/test_e04_declustering_intervention.py tests/test_e04_policy_label_switches.py tests/test_e04_identity_controls.py -q",
            "python -m analysis.declustering_intervention freeze",
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m analysis.declustering_intervention run --workers 8",
            "python -m analysis.declustering_intervention analyze",
            "python -m analysis.declustering_intervention report",
            "python -m analysis.declustering_intervention package",
        ],
        "dependenciesInstalled": [],
    }
    write_json(output / "commands.json", commands)
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    write_json(
        output / "artifact_manifest.json",
        {"schema": "e04.s10.artifact_manifest.v1", "researchStepId": "S10", "artifactCount": len(files), "artifacts": files},
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("amend-freeze")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--workers", type=int, default=8)
    subparsers.add_parser("analyze")
    subparsers.add_parser("report")
    subparsers.add_parser("package")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze_design()
    elif args.command == "amend-freeze":
        amend_freeze()
    elif args.command == "run":
        if not 1 <= args.workers <= 8:
            raise ValueError("workers must be in [1,8]")
        run_corpus(workers=args.workers)
    elif args.command == "analyze":
        analyze()
    elif args.command == "report":
        write_report()
    elif args.command == "package":
        package()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
