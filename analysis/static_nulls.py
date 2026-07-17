"""E04 S05 static global and value-conditioned label-permutation nulls.

The scientific contract is frozen in ``analysis/s05_static_null_contract.json``.
This module treats S04 as immutable input, keeps all trajectory-level maximum
questions out of scope, and emits only fixed-progress randomization evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

from analysis.aggregation_metrics import aggregation_metrics
from analysis.chimeric_replication import (
    AdmissibilityTracker,
    GRID,
    MetricTracker,
    _execute_s13_activation,
    canonical_hash,
)
from analysis.composition_sweep import (
    SweepCondition,
    build_tasks,
    materialize_sweep_scenario,
)
from reference_simulator.engine import evaluate_terminal, initial_state
from reference_simulator.model import canonical_json_bytes, state_hash


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY / "analysis/s05_static_null_contract.json"
S01_DIR = Path("/artifacts/research_steps/S01")
S02_DIR = Path("/artifacts/research_steps/S02")
S03_DIR = Path("/artifacts/research_steps/S03")
S04_DIR = Path("/artifacts/research_steps/S04")
S06_DIR = Path("/artifacts/research_steps/S06")
EXPECTED_MANIFEST_HASHES = {
    "S01": "df9860285fc5b1ed9f7442fb68ea5e9ed9e407918d90988d0b3592f4b3ef8079",
    "S02": "9e632eedac59f6b08ff6b619ce2451a99b49c952319af2747076fd7bb3411b54",
    "S03": "85371d82a53096764b0de52029472a7978261ca5d4857f4d392f441db076f9e9",
    "S04": "90db3cb1afbb8e1ff53fb1d4fe0e04a4d21fb66ccd3a0b922738bade82d05641",
}
S04_TABLE_PATH = S04_DIR / "aggregation_metrics.parquet"
S04_TABLE_BYTES = 75_335_939
S04_TABLE_SHA256 = "97ce5985d9dc9b666f8c16345a2c678535037c53ad58867fc8305c975e6db647"
OUTPUT_SCHEMA = "e04.s05.static_nulls.v1"
SNAPSHOT_SCHEMA = "e04.s05.snapshot_states.v1"
SNAPSHOT_CHECKPOINT_SCHEMA = "e04.s05.snapshot_checkpoint.v1"
GLOBAL_METRICS = (
    "corrected_publication_aggregation",
    "categorical_assortativity",
    "largest_label_capture",
    "signed_neighbor_nmi",
)
CONDITIONAL_METRICS = (*GLOBAL_METRICS, "equal_value_corrected_excess")
GLOBAL_PROFILES = (
    (90, 10),
    (80, 20),
    (70, 30),
    (60, 40),
    (50, 50),
    (45, 45, 10),
    (34, 33, 33),
)
AUDIT_GRID = (0, 25, 50, 75, 100)
AUDIT_REPLICATES = tuple(range(0, 250, 10))
GLOBAL_DRAWS = 250_000
GLOBAL_CALIBRATION_DRAWS = 100_000
GLOBAL_GROUP_DRAWS = 10_000
CONDITIONAL_DRAWS = 2_000
UNIFORMITY_DRAWS = 210_000
EXPECTED_SCENARIOS = 46_500
EXPECTED_CONDITIONS = 186
EXPECTED_GRID_ROWS = 4_696_500
EXPECTED_AUDIT_SCENARIOS = 4_650
EXPECTED_SNAPSHOTS = 23_250


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
    material = canonical_json_bytes(
        {"namespace": "E04/S05/static_nulls/v1", "stream": stream, "address": address}
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def _profile_id(counts: Sequence[int]) -> str:
    return "-".join(str(item) for item in counts)


def _canonical_counts(counts: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted((int(item) for item in counts), reverse=True))


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


def verify_s04_table() -> dict[str, Any]:
    manifest = json.loads((S04_DIR / "artifact_manifest.json").read_text())
    entries = [
        item for item in manifest["artifacts"] if item["path"] == S04_TABLE_PATH.name
    ]
    observed_bytes = S04_TABLE_PATH.stat().st_size if S04_TABLE_PATH.is_file() else None
    observed_hash = sha256_file(S04_TABLE_PATH) if S04_TABLE_PATH.is_file() else None
    marker_candidates = sorted(
        str(path)
        for root in (S04_DIR, Path("/workspace"))
        for path in root.glob("**/*")
        if path.is_file()
        and any(token in path.name.lower() for token in ("skip", "collect", "marker"))
    )
    result = {
        "path": str(S04_TABLE_PATH),
        "exists": S04_TABLE_PATH.is_file(),
        "expectedBytes": S04_TABLE_BYTES,
        "observedBytes": observed_bytes,
        "expectedSha256": S04_TABLE_SHA256,
        "observedSha256": observed_hash,
        "manifestEntries": entries,
        "visibleCollectionMarkerCandidates": marker_candidates,
        "externalCollectionSkipMetadataVisibleLocally": False,
    }
    result["passed"] = bool(
        result["exists"]
        and observed_bytes == S04_TABLE_BYTES
        and observed_hash == S04_TABLE_SHA256
        and len(entries) == 1
        and entries[0]["bytes"] == S04_TABLE_BYTES
        and entries[0]["sha256"] == S04_TABLE_SHA256
    )
    return result


def _null_specification_markdown(contract_hash: str) -> str:
    return f"""# S05 static permutation-null specification

## Top summary

| Field | Frozen result |
| --- | --- |
| Research step ID | S05 |
| Completion status | Null hypotheses, sufficient statistics, populations, seeds, sampling counts, inference rules, and validation gates frozen before S05 population analysis |
| Artifacts written | `preregistration.json`, `freeze_record.json`, and this `null_specification.md`; implementation is repository-backed |
| Validation result | Pre-analysis integrity passed for the canonical S04 table and all S01–S04 manifests; empirical sampling and calibration validation is pending execution |
| Outcome classification | Pending S05 execution |
| Caveats or blockers | Exact-value conditioning is degenerate for unique values; value-stratum conditioning can remove a policy-value mechanism and is a sensitivity analysis, not an automatically superior null |
| Recommended next action | Execute only the frozen S05 static-null workflow and stop before S06 |

Contract SHA-256: `{contract_hash}`.

The contract contains one transparent pre-population amendment. The initial fixed total-variation threshold was replaced by a support-size-aware multinomial bound after the 210-state uniformity fixture showed ordinary finite-draw fluctuation; no population state or scientific result had been analyzed, and no null hypothesis, seed, draw count, metric, or success gate changed.

## Unconditional null: `global_count_fixed`

The position coordinate, open-path adjacency graph, value at every position, array length, and exact global count of every policy are fixed. Policy labels are uniformly permuted over the fixed positions. This destroys the observed label-at-position assignment, spatial adjacency, and policy-by-value association. It tests total static label structure, including structure mediated by policy-value association.

The word *unconditional* is only with respect to policy-value association. Exact composition and the position/value scaffold are always conditioned on.

## Conditional null: `value_stratum_count_fixed`

The same scaffold and global counts are fixed, and labels are uniformly permuted independently within frozen value strata. Repeated inputs use the ten exact numeric values as strata. Unique inputs use the fixed numeric deciles 1–10 through 91–100, matching the repeated design's ten strata of size ten. This preserves the exact policy-by-stratum contingency and tests residual spatial adjacency within those constraints.

For unique inputs, a stricter exact-value null has 100 singleton strata and support size one. It therefore supplies no p-value. This degeneracy is reported explicitly as an over-conditioning diagnostic.

## Statistics and pointwise populations

The full global analysis uses condition means over all 250 runs at each of the 101 fixed accepted-swap grid points for corrected publication adjacency, categorical assortativity, largest-label capture, and signed neighbor NMI. The conditioning comparison uses the frozen 25-run subset (replicate ordinals divisible by ten) at 0%, 25%, 50%, 75%, and 100%; repeated inputs additionally use equal-value corrected excess.

Larger values are the clustering tail. Observed p-values use the conservative plus-one Monte Carlo upper tail. False-positive calibration uses randomized tie breaking. Benjamini–Hochberg q-values are computed across the 186 conditions within each fixed grid point, metric, and null family.

## Scope boundary

S05 does not test any maximum, earliest crossing, persistence, duration, area above threshold, or ever-significant event. Those trajectory-level and peak-selection questions remain assigned to S06.
"""


def freeze_design(output: Path, cache: Path) -> dict[str, Any]:
    if (cache / "snapshot_states.jsonl").exists() or (
        cache / "value_conditioned_nulls.jsonl"
    ).exists():
        raise FileExistsError("cannot freeze S05 after an S05 population checkpoint exists")
    if S06_DIR.exists():
        raise FileExistsError("S06 output exists before S05 freeze")
    output.mkdir(parents=True, exist_ok=True)
    (output / "static_nulls").mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT_PATH.read_text())
    contract_hash = sha256_file(CONTRACT_PATH)
    upstream = {
        step: _verify_upstream_manifest(
            Path(f"/artifacts/research_steps/{step}"), expected
        )
        for step, expected in EXPECTED_MANIFEST_HASHES.items()
    }
    if not all(item["allPassed"] for item in upstream.values()):
        raise AssertionError("upstream immutability failed before S05 freeze")
    s04_integrity = verify_s04_table()
    if not s04_integrity["passed"]:
        raise AssertionError("canonical S04 table integrity failed before S05 freeze")
    table = pq.ParquetFile(S04_TABLE_PATH)
    if table.metadata.num_rows != EXPECTED_GRID_ROWS:
        raise AssertionError("canonical S04 table row count changed")
    write_json(output / "preregistration.json", contract)
    (output / "null_specification.md").write_text(
        _null_specification_markdown(contract_hash), encoding="utf-8"
    )
    record = {
        "schema": "e04.s05.freeze_record.v1",
        "researchStepId": "S05",
        "frozenAtUtc": contract["frozenAtUtc"],
        "frozenBeforePopulationAnalysis": True,
        "contractPath": str(CONTRACT_PATH),
        "contractSha256": contract_hash,
        "s04Integrity": s04_integrity,
        "upstreamValidation": upstream,
        "fullPointwiseRows": EXPECTED_CONDITIONS * len(GRID),
        "conditionalAuditScenarios": EXPECTED_AUDIT_SCENARIOS,
        "conditionalAuditSnapshots": EXPECTED_SNAPSHOTS,
        "s06AbsentAtFreeze": True,
    }
    write_json(output / "freeze_record.json", record)
    return record


def assert_frozen(output: Path) -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text())
    artifact = json.loads((output / "preregistration.json").read_text())
    record = json.loads((output / "freeze_record.json").read_text())
    if contract != artifact:
        raise AssertionError("artifact preregistration differs from repository contract")
    if record["contractSha256"] != sha256_file(CONTRACT_PATH):
        raise AssertionError("frozen S05 contract hash changed")
    if not record["frozenBeforePopulationAnalysis"]:
        raise AssertionError("S05 was not frozen before population analysis")
    if S06_DIR.exists():
        raise FileExistsError("S06 output exists during S05")
    return record


def sample_global_labels(
    counts: Sequence[int], draws: int, rng: np.random.Generator
) -> np.ndarray:
    counts = tuple(int(item) for item in counts)
    if draws <= 0 or any(item <= 0 for item in counts):
        raise ValueError("positive draws and counts are required")
    labels = np.repeat(np.arange(len(counts), dtype=np.int8), counts)
    tiled = np.broadcast_to(labels, (draws, len(labels))).copy()
    return rng.permuted(tiled, axis=1)


def sample_stratified_labels(
    labels: Sequence[int], strata: Sequence[int], draws: int, rng: np.random.Generator
) -> np.ndarray:
    labels_array = np.asarray(labels, dtype=np.int8)
    strata_array = np.asarray(strata)
    if labels_array.ndim != 1 or strata_array.shape != labels_array.shape:
        raise ValueError("labels and strata must be same-length vectors")
    if draws <= 0:
        raise ValueError("draws must be positive")
    result = np.broadcast_to(labels_array, (draws, len(labels_array))).copy()
    for stratum in np.unique(strata_array):
        positions = np.flatnonzero(strata_array == stratum)
        block = np.broadcast_to(labels_array[positions], (draws, len(positions))).copy()
        result[:, positions] = rng.permuted(block, axis=1)
    return result


def batch_metrics(
    label_matrix: np.ndarray, values: Sequence[int | float] | None = None
) -> dict[str, np.ndarray]:
    """Vectorized S04 directional metrics for a batch of label arrangements."""
    labels = np.asarray(label_matrix, dtype=np.int8)
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("label_matrix must have shape (draws,n), n>=2")
    draws, n = labels.shape
    k = int(labels.max()) + 1
    if labels.min() < 0:
        raise ValueError("labels must be nonnegative integer codes")
    counts = np.column_stack([(labels == label).sum(axis=1) for label in range(k)])
    if np.any(counts == 0):
        raise ValueError("every batch arrangement must contain every encoded label")
    same = (labels[:, :-1] == labels[:, 1:]).sum(axis=1).astype(np.float64)
    paper_baseline = (counts * (counts - 1)).sum(axis=1) / float(n * n)
    corrected = same / n - paper_baseline

    stub_counts = 2 * counts.astype(np.float64)
    row_indices = np.arange(draws)
    stub_counts[row_indices, labels[:, 0]] -= 1
    stub_counts[row_indices, labels[:, -1]] -= 1
    marginals = stub_counts / (2 * (n - 1))
    chance = np.square(marginals).sum(axis=1)
    denominator = 1.0 - chance
    assortativity = np.divide(
        same / (n - 1) - chance,
        denominator,
        out=np.full(draws, np.nan),
        where=denominator > 1e-15,
    )

    current_runs = np.zeros((draws, k), dtype=np.int16)
    largest_runs = np.zeros((draws, k), dtype=np.int16)
    for position in range(n):
        for label in range(k):
            mask = labels[:, position] == label
            current_runs[:, label] = np.where(mask, current_runs[:, label] + 1, 0)
            largest_runs[:, label] = np.maximum(
                largest_runs[:, label], current_runs[:, label]
            )
    largest_capture = np.max(largest_runs / counts, axis=1)

    joint = np.zeros((draws, k, k), dtype=np.float64)
    left = labels[:, :-1]
    right = labels[:, 1:]
    for first in range(k):
        for second in range(k):
            joint[:, first, second] = (
                ((left == first) & (right == second)).sum(axis=1)
                + ((left == second) & (right == first)).sum(axis=1)
            )
    probabilities = joint / (2 * (n - 1))
    mi = np.zeros(draws, dtype=np.float64)
    for first in range(k):
        for second in range(k):
            probability = probabilities[:, first, second]
            expected = marginals[:, first] * marginals[:, second]
            mask = probability > 0
            mi[mask] += probability[mask] * np.log2(
                probability[mask] / expected[mask]
            )
    entropy = np.zeros(draws, dtype=np.float64)
    for label in range(k):
        probability = marginals[:, label]
        mask = probability > 0
        entropy[mask] -= probability[mask] * np.log2(probability[mask])
    nmi = np.divide(
        mi,
        entropy,
        out=np.full(draws, np.nan),
        where=entropy > 1e-15,
    )
    signed_nmi = np.where(
        np.isfinite(assortativity) & np.isfinite(nmi),
        np.where(np.abs(assortativity) <= 1e-15, 0.0, np.sign(assortativity) * nmi),
        np.nan,
    )

    result = {
        "corrected_publication_aggregation": corrected,
        "categorical_assortativity": assortativity,
        "largest_label_capture": largest_capture,
        "signed_neighbor_nmi": signed_nmi,
    }
    if values is None:
        result["equal_value_corrected_excess"] = np.full(draws, np.nan)
        return result

    result["equal_value_corrected_excess"] = equal_value_excess_batch(labels, values)
    return result


def equal_value_excess_batch(
    label_matrix: np.ndarray, values: Sequence[int | float]
) -> np.ndarray:
    """Compute only the S04 equal-value corrected metric for a label batch."""
    labels = np.asarray(label_matrix, dtype=np.int8)
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("label_matrix must have shape (draws,n), n>=2")
    draws, n = labels.shape
    value_array = np.asarray(values)
    if value_array.shape != (n,):
        raise ValueError("values must have length n")
    k = int(labels.max()) + 1
    equal_same = np.zeros(draws, dtype=np.float64)
    equal_edges_by_value: dict[Any, int] = {}
    for value in np.unique(value_array):
        edge_mask = (value_array[:-1] == value) & (value_array[1:] == value)
        edge_count = int(edge_mask.sum())
        if edge_count:
            equal_edges_by_value[value.item() if hasattr(value, "item") else value] = edge_count
            equal_same += (
                (labels[:, :-1][:, edge_mask] == labels[:, 1:][:, edge_mask])
                .sum(axis=1)
                .astype(np.float64)
            )
    equal_edges = sum(equal_edges_by_value.values())
    if not equal_edges:
        return np.full(draws, np.nan)
    weighted_expected = np.zeros(draws, dtype=np.float64)
    for value, edge_count in equal_edges_by_value.items():
        positions = value_array == value
        stratum_n = int(positions.sum())
        stratum_counts = np.column_stack(
            [(labels[:, positions] == label).sum(axis=1) for label in range(k)]
        )
        probability = (stratum_counts * (stratum_counts - 1)).sum(axis=1) / (
            stratum_n * (stratum_n - 1)
        )
        weighted_expected += edge_count * probability
    return equal_same / equal_edges - weighted_expected / equal_edges


def _encode_arrangements(matrix: np.ndarray) -> np.ndarray:
    powers = np.power(int(matrix.max()) + 1, np.arange(matrix.shape[1]), dtype=np.int64)
    return matrix.astype(np.int64) @ powers


def run_uniformity_validation(output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    fixtures = [
        ("global_n6_3_3", (3, 3), None),
        ("global_n7_3_2_2", (3, 2, 2), None),
    ]
    for fixture_id, counts, _ in fixtures:
        rng = np.random.Generator(
            np.random.PCG64DXSM(derive_seed("uniformity", fixture_id))
        )
        sampled = sample_global_labels(counts, UNIFORMITY_DRAWS, rng)
        observed = Counter(_encode_arrangements(sampled).tolist())
        support = math.factorial(sum(counts))
        for count in counts:
            support //= math.factorial(count)
        frequencies = np.asarray(list(observed.values()), dtype=np.float64)
        expected = UNIFORMITY_DRAWS / support
        chisquare_p = float(stats.chisquare(frequencies, f_exp=np.full(support, expected)).pvalue)
        total_variation = float(
            0.5 * np.abs(frequencies / UNIFORMITY_DRAWS - 1 / support).sum()
        )
        rows.append(
            {
                "fixture_id": fixture_id,
                "draws": UNIFORMITY_DRAWS,
                "support_size": support,
                "observed_support_size": len(observed),
                "chi_square_p": chisquare_p,
                "total_variation": total_variation,
                "count_violations": int(
                    sum(
                        tuple(sorted(np.bincount(row, minlength=len(counts)), reverse=True))
                        != tuple(sorted(counts, reverse=True))
                        for row in sampled
                    )
                ),
            }
        )

    labels = np.asarray((0, 0, 1, 0, 1, 1), dtype=np.int8)
    strata = np.asarray((0, 0, 0, 1, 1, 1), dtype=np.int8)
    rng = np.random.Generator(
        np.random.PCG64DXSM(derive_seed("uniformity", "conditional_two_3"))
    )
    sampled = sample_stratified_labels(labels, strata, UNIFORMITY_DRAWS, rng)
    observed = Counter(_encode_arrangements(sampled).tolist())
    support = 9
    frequencies = np.asarray(list(observed.values()), dtype=np.float64)
    expected = UNIFORMITY_DRAWS / support
    rows.append(
        {
            "fixture_id": "conditional_two_strata_2_1_each",
            "draws": UNIFORMITY_DRAWS,
            "support_size": support,
            "observed_support_size": len(observed),
            "chi_square_p": float(
                stats.chisquare(frequencies, f_exp=np.full(support, expected)).pvalue
            ),
            "total_variation": float(
                0.5
                * np.abs(frequencies / UNIFORMITY_DRAWS - 1 / support).sum()
            ),
            "count_violations": int(
                sum(
                    any(
                        tuple(np.bincount(row[strata == value], minlength=2))
                        != tuple(np.bincount(labels[strata == value], minlength=2))
                        for value in (0, 1)
                    )
                    for row in sampled
                )
            ),
        }
    )

    singleton_labels = np.arange(6, dtype=np.int8) % 2
    singleton_strata = np.arange(6, dtype=np.int8)
    rng = np.random.Generator(
        np.random.PCG64DXSM(derive_seed("uniformity", "singleton_exact_value"))
    )
    singleton = sample_stratified_labels(
        singleton_labels, singleton_strata, UNIFORMITY_DRAWS, rng
    )
    rows.append(
        {
            "fixture_id": "unique_exact_value_singletons",
            "draws": UNIFORMITY_DRAWS,
            "support_size": 1,
            "observed_support_size": len(Counter(_encode_arrangements(singleton).tolist())),
            "chi_square_p": math.nan,
            "total_variation": 0.0,
            "count_violations": int(np.any(singleton != singleton_labels, axis=1).sum()),
        }
    )
    frame = pd.DataFrame(rows)
    frame["total_variation_tolerance"] = np.maximum(
        0.01,
        2
        * np.sqrt(
            (frame.support_size.astype(float) - 1)
            / (2 * math.pi * frame.draws.astype(float))
        ),
    )
    frame["passed"] = (
        frame.observed_support_size.eq(frame.support_size)
        & frame.total_variation.le(frame.total_variation_tolerance)
        & frame.count_violations.eq(0)
        & (frame.support_size.eq(1) | frame.chi_square_p.gt(0.0001))
    )
    _write_parquet(frame, output / "uniform_permutation_validation.parquet")
    result = {
        "schema": "e04.s05.uniformity_validation.v1",
        "researchStepId": "S05",
        "fixtures": len(frame),
        "drawsPerFixture": UNIFORMITY_DRAWS,
        "totalDraws": int(frame.draws.sum()),
        "allPassed": bool(frame.passed.all()),
        "maxTotalVariation": float(frame.total_variation.max()),
        "minimumTotalVariationMargin": float(
            (frame.total_variation_tolerance - frame.total_variation).min()
        ),
        "minimumNondegenerateChiSquareP": float(
            frame.loc[frame.support_size > 1, "chi_square_p"].min()
        ),
        "totalConstraintViolations": int(frame.count_violations.sum()),
    }
    if not result["allPassed"]:
        raise AssertionError("uniform permutation validation failed")
    write_json(output / "uniformity_validation_summary.json", result)
    return result


def _global_profile_worker(
    counts: tuple[int, ...], reference_path: str, calibration_path: str
) -> dict[str, Any]:
    """Write restartable reference and independent calibration parts."""
    profile_id = _profile_id(counts)
    reference = Path(reference_path)
    calibration = Path(calibration_path)
    chunk_size = 25_000
    preservation_violations = 0

    def generate(path: Path, draws: int, stream: str) -> None:
        nonlocal preservation_violations
        path.parent.mkdir(parents=True, exist_ok=True)
        writer: pq.ParquetWriter | None = None
        try:
            for start in range(0, draws, chunk_size):
                size = min(chunk_size, draws - start)
                rng = np.random.Generator(
                    np.random.PCG64DXSM(
                        derive_seed(stream, profile_id, start, size)
                    )
                )
                labels = sample_global_labels(counts, size, rng)
                observed_counts = np.column_stack(
                    [(labels == label).sum(axis=1) for label in range(len(counts))]
                )
                preservation_violations += int(
                    np.any(observed_counts != np.asarray(counts), axis=1).sum()
                )
                metrics = batch_metrics(labels)
                frame = pd.DataFrame(
                    {
                        "schema_version": OUTPUT_SCHEMA,
                        "null_family": "global_count_fixed",
                        "profile_id": profile_id,
                        "counts_json": json.dumps(counts, separators=(",", ":")),
                        "draw_index": np.arange(start, start + size, dtype=np.int64),
                        **{metric: metrics[metric] for metric in GLOBAL_METRICS},
                    }
                )
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

    generate(reference, GLOBAL_DRAWS, "global_reference")
    generate(calibration, GLOBAL_CALIBRATION_DRAWS, "global_calibration")
    return {
        "profileId": profile_id,
        "counts": counts,
        "referenceRows": pq.ParquetFile(reference).metadata.num_rows,
        "calibrationRows": pq.ParquetFile(calibration).metadata.num_rows,
        "preservationViolations": preservation_violations,
        "referencePath": str(reference),
        "calibrationPath": str(calibration),
    }


def _global_profile_worker_unpack(args: tuple[tuple[int, ...], str, str]) -> dict[str, Any]:
    return _global_profile_worker(*args)


def _upper_tail_values(
    reference: np.ndarray,
    observed: np.ndarray,
    *,
    randomized_u: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    observed = np.asarray(observed, dtype=np.float64)
    left = np.searchsorted(reference, observed, side="left")
    right = np.searchsorted(reference, observed, side="right")
    conservative = (len(reference) - left + 1) / (len(reference) + 1)
    if randomized_u is None:
        return conservative, None
    randomized_u = np.asarray(randomized_u, dtype=np.float64)
    randomized = (
        len(reference) - right + randomized_u * (right - left + 1)
    ) / (len(reference) + 1)
    return conservative, randomized


def _bh_adjust(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    result = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return result
    selected = p[valid]
    order = np.argsort(selected)
    ranked = selected[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    result[valid] = restored
    return result


def build_global_null_corpus(
    output: Path, cache: Path, *, workers: int = 8
) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1,8]")
    corpus_dir = output / "static_nulls/global_arrangement_nulls"
    calibration_dir = cache / "global_calibration"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for counts in GLOBAL_PROFILES:
        profile_id = _profile_id(counts)
        reference = corpus_dir / f"profile-{profile_id}.parquet"
        calibration = calibration_dir / f"profile-{profile_id}.parquet"
        tasks.append((counts, str(reference), str(calibration)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_global_profile_worker_unpack, tasks))
    if any(item["preservationViolations"] for item in results):
        raise AssertionError("global sampler violated exact counts")

    summary_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    group_rows: list[pd.DataFrame] = []
    alphas = (0.01, 0.05, 0.10)
    for counts in GLOBAL_PROFILES:
        profile_id = _profile_id(counts)
        reference_frame = pd.read_parquet(
            corpus_dir / f"profile-{profile_id}.parquet"
        )
        calibration_frame = pd.read_parquet(
            calibration_dir / f"profile-{profile_id}.parquet"
        )
        for metric in GLOBAL_METRICS:
            reference = reference_frame[metric].to_numpy(np.float64)
            calibration = calibration_frame[metric].to_numpy(np.float64)
            rng = np.random.Generator(
                np.random.PCG64DXSM(
                    derive_seed("global_calibration_ties", profile_id, metric)
                )
            )
            conservative, randomized = _upper_tail_values(
                reference, calibration, randomized_u=rng.random(len(calibration))
            )
            assert randomized is not None
            summary_rows.append(
                {
                    "profile_id": profile_id,
                    "counts_json": json.dumps(counts, separators=(",", ":")),
                    "metric": metric,
                    "draws": len(reference),
                    "mean": float(np.mean(reference)),
                    "sd": float(np.std(reference, ddof=1)),
                    "q005": float(np.quantile(reference, 0.005)),
                    "q025": float(np.quantile(reference, 0.025)),
                    "q500": float(np.quantile(reference, 0.5)),
                    "q975": float(np.quantile(reference, 0.975)),
                    "q995": float(np.quantile(reference, 0.995)),
                }
            )
            for alpha in alphas:
                randomized_rate = float(np.mean(randomized <= alpha))
                conservative_rate = float(np.mean(conservative <= alpha))
                se = math.sqrt(alpha * (1 - alpha) / len(calibration))
                tolerance = max(5 * se, 0.002)
                calibration_rows.append(
                    {
                        "profile_id": profile_id,
                        "counts_json": json.dumps(counts, separators=(",", ":")),
                        "metric": metric,
                        "alpha": alpha,
                        "calibration_draws": len(calibration),
                        "randomized_false_positive_rate": randomized_rate,
                        "conservative_false_positive_rate": conservative_rate,
                        "binomial_se": se,
                        "tolerance": tolerance,
                        "randomized_passed": abs(randomized_rate - alpha)
                        <= tolerance,
                        "conservative_passed": conservative_rate
                        <= alpha + tolerance,
                    }
                )

        for group_size in (25, 250):
            rng = np.random.Generator(
                np.random.PCG64DXSM(
                    derive_seed("global_group_means", profile_id, group_size)
                )
            )
            indices = rng.integers(
                0,
                len(reference_frame),
                size=(GLOBAL_GROUP_DRAWS, group_size),
                dtype=np.int32,
            )
            data: dict[str, Any] = {
                "schema_version": OUTPUT_SCHEMA,
                "null_family": "global_count_fixed",
                "profile_id": profile_id,
                "counts_json": json.dumps(counts, separators=(",", ":")),
                "group_size": group_size,
                "draw_index": np.arange(GLOBAL_GROUP_DRAWS, dtype=np.int64),
            }
            for metric in GLOBAL_METRICS:
                data[metric] = reference_frame[metric].to_numpy(np.float64)[
                    indices
                ].mean(axis=1)
            group_rows.append(pd.DataFrame(data))

    summary = pd.DataFrame(summary_rows).sort_values(["profile_id", "metric"])
    calibration = pd.DataFrame(calibration_rows).sort_values(
        ["profile_id", "metric", "alpha"]
    )
    group_frame = pd.concat(group_rows, ignore_index=True).sort_values(
        ["group_size", "profile_id", "draw_index"]
    )
    _write_parquet(summary, output / "global_null_summary.parquet")
    _write_parquet(
        calibration, output / "global_false_positive_calibration.parquet"
    )
    _write_parquet(
        group_frame, output / "static_nulls/global_group_mean_nulls.parquet"
    )
    accounting = {
        "schema": "e04.s05.global_null_accounting.v1",
        "researchStepId": "S05",
        "profiles": len(GLOBAL_PROFILES),
        "arrangementDrawsPerProfile": GLOBAL_DRAWS,
        "arrangementDraws": len(GLOBAL_PROFILES) * GLOBAL_DRAWS,
        "independentCalibrationDrawsPerProfile": GLOBAL_CALIBRATION_DRAWS,
        "independentCalibrationDraws": len(GLOBAL_PROFILES)
        * GLOBAL_CALIBRATION_DRAWS,
        "groupMeanDrawsPerProfileAndSize": GLOBAL_GROUP_DRAWS,
        "groupMeanRows": len(group_frame),
        "metrics": list(GLOBAL_METRICS),
        "preservationViolations": sum(
            item["preservationViolations"] for item in results
        ),
        "falsePositiveRows": len(calibration),
        "allFalsePositiveChecksPassed": bool(
            calibration.randomized_passed.all()
            and calibration.conservative_passed.all()
        ),
        "workers": workers,
    }
    if not accounting["allFalsePositiveChecksPassed"]:
        raise AssertionError("global false-positive calibration failed")
    write_json(output / "global_null_accounting.json", accounting)
    return accounting


def _selected_s03_expected() -> dict[str, dict[str, Any]]:
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
    ]
    rows = pq.read_table(S03_DIR / "composition_sweep.parquet", columns=columns).to_pylist()
    result = {
        str(row["scenario_id"]): row
        for row in rows
        if int(row["replicate_ordinal"]) in AUDIT_REPLICATES
    }
    if len(result) != EXPECTED_AUDIT_SCENARIOS:
        raise AssertionError("selected S03 expected-run table has wrong cardinality")
    return result


def build_snapshot_tasks() -> list[dict[str, Any]]:
    tasks = [
        task
        for task in build_tasks()
        if int(task["base"]["replicateOrdinal"]) in AUDIT_REPLICATES
    ]
    if len(tasks) != EXPECTED_AUDIT_SCENARIOS:
        raise AssertionError("snapshot task accounting changed")
    return tasks


def _snapshot_worker(
    task: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    condition = SweepCondition.from_dict(task["condition"])
    scenario, scenario_metadata = materialize_sweep_scenario(condition, task["base"])
    if scenario.scenario_id != task["scenarioId"]:
        raise AssertionError("S05 scenario rematerialization changed scenario ID")
    if scenario_metadata["scenarioJsonSha256"] != expected["scenario_json_sha256"]:
        raise AssertionError("S05 scenario JSON differs from S03")
    final_swaps = int(expected["successful_swap_count"])
    targets: dict[int, list[int]] = defaultdict(list)
    for grid_index in AUDIT_GRID:
        targets[math.floor((grid_index / 100) * final_swaps)].append(grid_index)
    state = initial_state(scenario)
    state.terminal = evaluate_terminal(scenario, state)
    tracker = MetricTracker(scenario, False)
    admissibility = AdmissibilityTracker(scenario, state)
    original_points: list[dict[str, Any]] = [
        {"swap_index": 0, **tracker.metrics()}
    ]
    snapshots: dict[int, dict[str, Any]] = {}

    def capture(swap_index: int) -> None:
        policy_labels = [
            scenario.cell_map[cell_id].policy.value for cell_id in state.occupancy
        ]
        values = [int(scenario.cell_map[cell_id].value) for cell_id in state.occupancy]
        metrics = aggregation_metrics(policy_labels, values)
        for grid_index in targets.get(swap_index, ()):  # duplicate targets allowed
            snapshots[grid_index] = {
                "grid_index": grid_index,
                "accepted_swap_progress": grid_index / 100,
                "swap_index": swap_index,
                "labels_json": json.dumps(policy_labels, separators=(",", ":")),
                "values_json": json.dumps(values, separators=(",", ":")),
                "position_value_sha256": hashlib.sha256(
                    canonical_json_bytes(values)
                ).hexdigest(),
                **{metric: float(metrics[metric]) for metric in CONDITIONAL_METRICS},
            }

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

    start = time.perf_counter()
    while state.terminal is None:
        swap_committed = False
        outcome = _execute_s13_activation(
            scenario, state, admissibility, on_swap=on_swap
        )
        if outcome != "noop":
            if not swap_committed:
                admissibility.after_memory_update(state)
            state.terminal = admissibility.terminal_after_change(state)
    elapsed = time.perf_counter() - start
    official_terminal = evaluate_terminal(scenario, state)
    if state.terminal != official_terminal:
        raise AssertionError("incremental and official terminals differ")
    if set(snapshots) != set(AUDIT_GRID):
        raise AssertionError(
            f"snapshot capture missing fixed grid points: {sorted(set(AUDIT_GRID)-set(snapshots))}"
        )
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
        raise AssertionError(
            f"S05/S03 replay mismatch for {scenario.scenario_id}: {replay_checks}"
        )
    metadata = {
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
        "schema": SNAPSHOT_CHECKPOINT_SCHEMA,
        "scenario_id": scenario.scenario_id,
        "scenario_json_sha256": scenario_metadata["scenarioJsonSha256"],
        "metadata": metadata,
        "stop_reason": state.terminal,
        "successful_swap_count": observed["successful_swap_count"],
        "activation_count": observed["activation_count"],
        "final_state_hash": observed["final_state_hash"],
        "original_trajectory_sha256": observed["trajectory_sha256"],
        "replay_checks": replay_checks,
        "elapsed_seconds": elapsed,
        "snapshots": [snapshots[index] for index in AUDIT_GRID],
    }


def _read_snapshot_checkpoint_ids(checkpoint: Path) -> set[str]:
    if not checkpoint.exists():
        return set()
    completed: set[str] = set()
    with checkpoint.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record["schema"] != SNAPSHOT_CHECKPOINT_SCHEMA:
                raise ValueError("snapshot checkpoint schema changed")
            scenario_id = str(record["scenario_id"])
            if scenario_id in completed:
                raise ValueError(
                    f"duplicate snapshot checkpoint scenario at line {line_number}"
                )
            completed.add(scenario_id)
    return completed


def replay_snapshot_population(
    tasks: Sequence[Mapping[str, Any]], checkpoint: Path, *, workers: int = 8
) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1,8]")
    expected = _selected_s03_expected()
    task_ids = {str(task["scenarioId"]) for task in tasks}
    if task_ids != set(expected):
        raise AssertionError("snapshot task population differs from frozen subset")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_snapshot_checkpoint_ids(checkpoint)
    if not completed <= task_ids:
        raise ValueError("snapshot checkpoint contains an out-of-scope scenario")
    pending = [task for task in tasks if str(task["scenarioId"]) not in completed]
    iterator = iter(pending)
    max_in_flight = workers * 3
    started = time.perf_counter()
    written = 0
    with checkpoint.open("a", encoding="utf-8") as handle:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            active: dict[Any, Mapping[str, Any]] = {}
            for _ in range(min(max_in_flight, len(pending))):
                task = next(iterator)
                scenario_id = str(task["scenarioId"])
                active[executor.submit(_snapshot_worker, task, expected[scenario_id])] = task
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    active.pop(future)
                    result = future.result()
                    handle.write(
                        json.dumps(
                            _json_native(result), sort_keys=True, separators=(",", ":")
                        )
                        + "\n"
                    )
                    handle.flush()
                    completed.add(str(result["scenario_id"]))
                    written += 1
                    if written % 250 == 0:
                        print(
                            json.dumps(
                                {
                                    "completed": len(completed),
                                    "pending": EXPECTED_AUDIT_SCENARIOS
                                    - len(completed),
                                    "elapsedSeconds": time.perf_counter() - started,
                                }
                            ),
                            flush=True,
                        )
                    try:
                        next_task = next(iterator)
                    except StopIteration:
                        continue
                    scenario_id = str(next_task["scenarioId"])
                    active[
                        executor.submit(
                            _snapshot_worker, next_task, expected[scenario_id]
                        )
                    ] = next_task
    if len(completed) != EXPECTED_AUDIT_SCENARIOS:
        raise AssertionError(f"snapshot checkpoint incomplete: {len(completed)}")
    return {
        "scenarioCount": len(completed),
        "newScenarios": written,
        "snapshotCount": len(completed) * len(AUDIT_GRID),
        "elapsedSeconds": time.perf_counter() - started,
        "workers": workers,
        "checkpoint": str(checkpoint),
    }


def _snapshot_checkpoint_records(checkpoint: Path) -> Iterable[dict[str, Any]]:
    with checkpoint.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_snapshot_artifact(checkpoint: Path, output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    replay_passes = 0
    for record in _snapshot_checkpoint_records(checkpoint):
        replay_passes += int(all(record["replay_checks"].values()))
        for snapshot in record["snapshots"]:
            rows.append(
                {
                    "schema_version": SNAPSHOT_SCHEMA,
                    "research_step_id": "S05",
                    "scenario_id": record["scenario_id"],
                    "scenario_json_sha256": record["scenario_json_sha256"],
                    **record["metadata"],
                    **snapshot,
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["condition_id", "grid_index", "replicate_ordinal"]
    )
    if len(frame) != EXPECTED_SNAPSHOTS:
        raise AssertionError("snapshot artifact has wrong row count")
    s04_columns = [
        "scenario_id",
        "grid_index",
        "swap_index",
        *CONDITIONAL_METRICS,
    ]
    s04 = pq.read_table(
        S04_TABLE_PATH,
        columns=s04_columns,
        filters=[("grid_index", "in", list(AUDIT_GRID))],
    ).to_pandas()
    s04 = s04[s04.scenario_id.isin(frame.scenario_id.unique())]
    comparison = frame.merge(
        s04,
        on=["scenario_id", "grid_index"],
        how="left",
        validate="one_to_one",
        suffixes=("_captured", "_s04"),
    )
    missing_s04 = int(comparison.swap_index_s04.isna().sum())
    swap_mismatches = int(
        (comparison.swap_index_captured != comparison.swap_index_s04).sum()
    )
    metric_errors: dict[str, float] = {}
    for metric in CONDITIONAL_METRICS:
        captured = comparison[f"{metric}_captured"].to_numpy(np.float64)
        canonical = comparison[f"{metric}_s04"].to_numpy(np.float64)
        valid = np.isfinite(captured) | np.isfinite(canonical)
        both_nan = np.isnan(captured) & np.isnan(canonical)
        differences = np.abs(captured - canonical)
        differences[both_nan] = 0.0
        metric_errors[metric] = float(np.nanmax(differences[valid])) if valid.any() else 0.0
    if (
        replay_passes != EXPECTED_AUDIT_SCENARIOS
        or missing_s04
        or swap_mismatches
        or any(value > 1e-12 for value in metric_errors.values())
    ):
        raise AssertionError("snapshot replay or canonical S04 metric comparison failed")
    _write_parquet(frame, output / "snapshot_states.parquet")
    result = {
        "schema": "e04.s05.snapshot_replay_summary.v1",
        "researchStepId": "S05",
        "scenarios": frame.scenario_id.nunique(),
        "conditions": frame.condition_id.nunique(),
        "replicatesPerCondition": sorted(
            frame.groupby("condition_id").replicate_ordinal.nunique().unique().tolist()
        ),
        "gridPoints": sorted(frame.grid_index.unique().tolist()),
        "snapshots": len(frame),
        "replayChecksPassed": replay_passes,
        "missingCanonicalS04Rows": missing_s04,
        "swapIndexMismatches": swap_mismatches,
        "maximumMetricErrors": metric_errors,
        "s04TableSha256": sha256_file(S04_TABLE_PATH),
        "allPassed": True,
    }
    write_json(output / "snapshot_replay_summary.json", result)
    return result


def _value_strata(input_profile: str, values: Sequence[int]) -> np.ndarray:
    value_array = np.asarray(values, dtype=np.int16)
    if input_profile == "repeated_1_10_x10":
        return value_array.copy()
    if input_profile == "unique_1_100":
        return ((value_array - 1) // 10).astype(np.int16)
    raise ValueError(f"unknown S03 input profile: {input_profile}")


def _support_size(labels: np.ndarray, strata: np.ndarray) -> int:
    support = 1
    k = int(labels.max()) + 1
    for stratum in np.unique(strata):
        positions = strata == stratum
        counts = np.bincount(labels[positions], minlength=k)
        subtotal = math.factorial(int(positions.sum()))
        for count in counts:
            subtotal //= math.factorial(int(count))
        support *= subtotal
    return support


def _preservation_violations(
    sampled: np.ndarray, labels: np.ndarray, strata: np.ndarray
) -> tuple[int, int]:
    k = int(labels.max()) + 1
    expected_global = np.bincount(labels, minlength=k)
    observed_global = np.column_stack(
        [(sampled == label).sum(axis=1) for label in range(k)]
    )
    global_violations = int(
        np.any(observed_global != expected_global, axis=1).sum()
    )
    stratum_violations = np.zeros(len(sampled), dtype=bool)
    for stratum in np.unique(strata):
        positions = strata == stratum
        expected = np.bincount(labels[positions], minlength=k)
        observed = np.column_stack(
            [
                (sampled[:, positions] == label).sum(axis=1)
                for label in range(k)
            ]
        )
        stratum_violations |= np.any(observed != expected, axis=1)
    return global_violations, int(stratum_violations.sum())


def _conditional_group_worker(group: Mapping[str, Any]) -> dict[str, Any]:
    rows = group["rows"]
    condition_id = str(group["condition_id"])
    grid_index = int(group["grid_index"])
    input_profile = str(rows[0]["input_profile"])
    policy_names = sorted(set(json.loads(rows[0]["labels_json"])))
    policy_to_code = {policy: index for index, policy in enumerate(policy_names)}
    conditional_sums = {
        metric: np.zeros(CONDITIONAL_DRAWS, dtype=np.float64)
        for metric in GLOBAL_METRICS
    }
    if input_profile == "repeated_1_10_x10":
        conditional_sums["equal_value_corrected_excess"] = np.zeros(
            CONDITIONAL_DRAWS, dtype=np.float64
        )
        global_equal_sums = np.zeros(CONDITIONAL_DRAWS, dtype=np.float64)
        equal_replicates = 0
    else:
        global_equal_sums = None
    pseudo_sums = {metric: 0.0 for metric in conditional_sums}
    global_equal_pseudo = 0.0
    support_sizes: list[int] = []
    global_violations = 0
    stratum_violations = 0
    pseudo_violations = 0

    for row in rows:
        labels_text = json.loads(row["labels_json"])
        labels = np.asarray([policy_to_code[item] for item in labels_text], dtype=np.int8)
        values = np.asarray(json.loads(row["values_json"]), dtype=np.int16)
        strata = _value_strata(input_profile, values)
        support_sizes.append(_support_size(labels, strata))
        rng = np.random.Generator(
            np.random.PCG64DXSM(
                derive_seed(
                    "value_conditioned",
                    row["scenario_id"],
                    grid_index,
                    CONDITIONAL_DRAWS,
                )
            )
        )
        sampled = sample_stratified_labels(labels, strata, CONDITIONAL_DRAWS, rng)
        global_bad, stratum_bad = _preservation_violations(
            sampled, labels, strata
        )
        global_violations += global_bad
        stratum_violations += stratum_bad
        metrics = batch_metrics(sampled, values)
        for metric in GLOBAL_METRICS:
            conditional_sums[metric] += metrics[metric]
        equal_eligible = bool(
            input_profile == "repeated_1_10_x10"
            and np.isfinite(metrics["equal_value_corrected_excess"]).any()
        )
        if equal_eligible:
            equal_replicates += 1
            conditional_sums["equal_value_corrected_excess"] += metrics[
                "equal_value_corrected_excess"
            ]

        pseudo_rng = np.random.Generator(
            np.random.PCG64DXSM(
                derive_seed("value_conditioned_pseudo", row["scenario_id"], grid_index)
            )
        )
        pseudo = sample_stratified_labels(labels, strata, 1, pseudo_rng)
        global_bad, stratum_bad = _preservation_violations(pseudo, labels, strata)
        pseudo_violations += global_bad + stratum_bad
        pseudo_metrics = batch_metrics(pseudo, values)
        for metric in GLOBAL_METRICS:
            pseudo_sums[metric] += float(pseudo_metrics[metric][0])
        if equal_eligible:
            pseudo_sums["equal_value_corrected_excess"] += float(
                pseudo_metrics["equal_value_corrected_excess"][0]
            )

        if global_equal_sums is not None:
            counts = tuple(np.bincount(labels, minlength=len(policy_names)).tolist())
            global_rng = np.random.Generator(
                np.random.PCG64DXSM(
                    derive_seed(
                        "global_equal_value",
                        row["scenario_id"],
                        grid_index,
                        CONDITIONAL_DRAWS,
                    )
                )
            )
            global_sampled = sample_global_labels(counts, CONDITIONAL_DRAWS, global_rng)
            global_bad, _ = _preservation_violations(
                global_sampled, labels, np.zeros(len(labels), dtype=np.int8)
            )
            global_violations += global_bad
            if equal_eligible:
                global_equal_sums += equal_value_excess_batch(global_sampled, values)
            global_pseudo_rng = np.random.Generator(
                np.random.PCG64DXSM(
                    derive_seed("global_equal_value_pseudo", row["scenario_id"], grid_index)
                )
            )
            global_pseudo = sample_global_labels(counts, 1, global_pseudo_rng)
            if equal_eligible:
                global_equal_pseudo += float(
                    equal_value_excess_batch(global_pseudo, values)[0]
                )

    group_size = len(rows)
    for metric in GLOBAL_METRICS:
        conditional_sums[metric] /= group_size
        pseudo_sums[metric] /= group_size
    if global_equal_sums is not None:
        if equal_replicates:
            conditional_sums["equal_value_corrected_excess"] /= equal_replicates
            pseudo_sums["equal_value_corrected_excess"] /= equal_replicates
            global_equal_sums /= equal_replicates
            global_equal_pseudo /= equal_replicates
        else:
            conditional_sums.pop("equal_value_corrected_excess")
            pseudo_sums.pop("equal_value_corrected_excess")
            global_equal_sums = None
            global_equal_pseudo = math.nan
    observed_means = {
        metric: float(np.nanmean([float(row[metric]) for row in rows]))
        for metric in CONDITIONAL_METRICS
    }
    metadata_keys = (
        "input_profile",
        "policy_set_label",
        "composition_class",
        "composition_profile",
        "first_policy_count",
        "rare_policy",
        "correlation_profile",
        "policy_counts_json",
    )
    metadata = {key: rows[0][key] for key in metadata_keys}
    canonical_profiles = {
        _canonical_counts(json.loads(row["policy_counts_json"]).values())
        for row in rows
    }
    if len(canonical_profiles) != 1:
        raise AssertionError("conditional group spans multiple canonical count profiles")
    canonical_counts = canonical_profiles.pop()
    return {
        "schema": "e04.s05.conditional_group_checkpoint.v1",
        "condition_id": condition_id,
        "grid_index": grid_index,
        "accepted_swap_progress": grid_index / 100,
        "replicates": group_size,
        "metadata": metadata,
        "canonical_counts": canonical_counts,
        "profile_id": _profile_id(canonical_counts),
        "observed_means": observed_means,
        "conditional_null_draws": {
            key: value.tolist() for key, value in conditional_sums.items()
        },
        "conditional_pseudo_observed": pseudo_sums,
        "global_equal_value_null_draws": (
            global_equal_sums.tolist() if global_equal_sums is not None else None
        ),
        "global_equal_value_pseudo_observed": (
            global_equal_pseudo if global_equal_sums is not None else None
        ),
        "equal_value_eligible_replicates": (
            equal_replicates if input_profile == "repeated_1_10_x10" else 0
        ),
        "support_size_min": min(support_sizes),
        "support_size_max": max(support_sizes),
        "support_log10_min": math.log10(min(support_sizes)),
        "support_log10_max": math.log10(max(support_sizes)),
        "strict_exact_value_unique_support_size": (
            1 if input_profile == "unique_1_100" else None
        ),
        "global_count_violations": global_violations,
        "stratum_count_violations": stratum_violations,
        "pseudo_constraint_violations": pseudo_violations,
    }


def _conditional_group_key(condition_id: str, grid_index: int) -> str:
    return f"{condition_id}|{grid_index:03d}"


def _read_conditional_checkpoint_keys(checkpoint: Path) -> set[str]:
    if not checkpoint.exists():
        return set()
    keys: set[str] = set()
    with checkpoint.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = _conditional_group_key(record["condition_id"], record["grid_index"])
            if key in keys:
                raise ValueError(
                    f"duplicate conditional group checkpoint at line {line_number}"
                )
            keys.add(key)
    return keys


def run_conditioned_nulls(
    snapshot_path: Path, checkpoint: Path, *, workers: int = 8
) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1,8]")
    snapshots = pd.read_parquet(snapshot_path)
    groups: list[dict[str, Any]] = []
    for (condition_id, grid_index), group in snapshots.groupby(
        ["condition_id", "grid_index"], sort=True
    ):
        if len(group) != len(AUDIT_REPLICATES):
            raise AssertionError("conditional group replicate count changed")
        groups.append(
            {
                "condition_id": condition_id,
                "grid_index": int(grid_index),
                "rows": group.sort_values("replicate_ordinal").to_dict("records"),
            }
        )
    if len(groups) != EXPECTED_CONDITIONS * len(AUDIT_GRID):
        raise AssertionError("conditional group accounting changed")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_conditional_checkpoint_keys(checkpoint)
    all_keys = {
        _conditional_group_key(group["condition_id"], group["grid_index"])
        for group in groups
    }
    if not completed <= all_keys:
        raise ValueError("conditional checkpoint contains an out-of-scope group")
    pending = [
        group
        for group in groups
        if _conditional_group_key(group["condition_id"], group["grid_index"])
        not in completed
    ]
    iterator = iter(pending)
    started = time.perf_counter()
    written = 0
    with checkpoint.open("a", encoding="utf-8") as handle:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            active: dict[Any, Mapping[str, Any]] = {}
            for _ in range(min(workers * 2, len(pending))):
                group = next(iterator)
                active[executor.submit(_conditional_group_worker, group)] = group
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    active.pop(future)
                    result = future.result()
                    handle.write(
                        json.dumps(
                            _json_native(result), sort_keys=True, separators=(",", ":")
                        )
                        + "\n"
                    )
                    handle.flush()
                    key = _conditional_group_key(
                        result["condition_id"], result["grid_index"]
                    )
                    completed.add(key)
                    written += 1
                    if written % 25 == 0:
                        print(
                            json.dumps(
                                {
                                    "completedGroups": len(completed),
                                    "pendingGroups": len(all_keys) - len(completed),
                                    "elapsedSeconds": time.perf_counter() - started,
                                }
                            ),
                            flush=True,
                        )
                    try:
                        next_group = next(iterator)
                    except StopIteration:
                        continue
                    active[executor.submit(_conditional_group_worker, next_group)] = (
                        next_group
                    )
    if completed != all_keys:
        raise AssertionError("conditional checkpoint is incomplete")
    return {
        "groups": len(completed),
        "newGroups": written,
        "drawsPerGroup": CONDITIONAL_DRAWS,
        "conditionedArrangements": EXPECTED_SNAPSHOTS * CONDITIONAL_DRAWS,
        "globalEqualValueArrangements": EXPECTED_SNAPSHOTS
        // 2
        * CONDITIONAL_DRAWS,
        "elapsedSeconds": time.perf_counter() - started,
        "workers": workers,
        "checkpoint": str(checkpoint),
    }


def _conditional_checkpoint_records(checkpoint: Path) -> Iterable[dict[str, Any]]:
    with checkpoint.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _null_summary_row(
    *,
    record: Mapping[str, Any],
    null_family: str,
    metric: str,
    observed: float,
    null_values: np.ndarray,
) -> dict[str, Any]:
    null_values = np.asarray(null_values, dtype=np.float64)
    null_values = null_values[np.isfinite(null_values)]
    if not len(null_values) or not math.isfinite(observed):
        mean = sd = q025 = q500 = q975 = standardized = p_value = math.nan
        degenerate = True
    else:
        mean = float(np.mean(null_values))
        sd = float(np.std(null_values, ddof=1))
        q025, q500, q975 = (
            float(item) for item in np.quantile(null_values, (0.025, 0.5, 0.975))
        )
        standardized = (observed - mean) / sd if sd > 1e-15 else math.nan
        p_value = float(_upper_tail_values(null_values, np.asarray([observed]))[0][0])
        degenerate = bool(sd <= 1e-15)
    return {
        "schema_version": OUTPUT_SCHEMA,
        "condition_id": record["condition_id"],
        "grid_index": int(record["grid_index"]),
        "accepted_swap_progress": float(record["accepted_swap_progress"]),
        **record["metadata"],
        "profile_id": record["profile_id"],
        "replicates": int(record["replicates"]),
        "null_family": null_family,
        "metric": metric,
        "observed_mean": observed,
        "null_draws": len(null_values),
        "null_mean": mean,
        "null_sd": sd,
        "null_q025": q025,
        "null_median": q500,
        "null_q975": q975,
        "observed_minus_null": observed - mean if math.isfinite(mean) else math.nan,
        "standardized_deviation": standardized,
        "upper_p": p_value,
        "null_degenerate": degenerate,
    }


def build_full_pointwise_global_deviations(output: Path) -> dict[str, Any]:
    response = pd.read_parquet(S04_DIR / "metric_response_surface.parquet")
    if len(response) != EXPECTED_CONDITIONS * len(GRID):
        raise AssertionError("S04 response-surface accounting changed")
    manifest = pd.read_parquet(
        S03_DIR / "scenario_manifest.parquet",
        columns=["condition_id", "policy_counts_json"],
    )
    profile_by_condition: dict[str, str] = {}
    for condition_id, group in manifest.groupby("condition_id"):
        profiles = {
            _profile_id(_canonical_counts(json.loads(value).values()))
            for value in group.policy_counts_json
        }
        if len(profiles) != 1:
            raise AssertionError("condition spans multiple canonical count profiles")
        profile_by_condition[str(condition_id)] = profiles.pop()
    group_nulls = pd.read_parquet(
        output / "static_nulls/global_group_mean_nulls.parquet"
    )
    group_nulls = group_nulls[group_nulls.group_size.eq(250)]
    null_lookup = {
        (profile_id, metric): np.sort(group[metric].to_numpy(np.float64))
        for profile_id, group in group_nulls.groupby("profile_id")
        for metric in GLOBAL_METRICS
    }
    null_statistics = {
        key: {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "q025": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "q975": float(np.quantile(values, 0.975)),
        }
        for key, values in null_lookup.items()
    }
    rows: list[dict[str, Any]] = []
    metadata = (
        "condition_id",
        "input_profile",
        "policy_set_label",
        "composition_class",
        "composition_profile",
        "first_policy_count",
        "rare_policy",
        "correlation_profile",
        "grid_index",
        "accepted_swap_progress",
    )
    for row in response.to_dict("records"):
        profile_id = profile_by_condition[str(row["condition_id"])]
        for metric in GLOBAL_METRICS:
            observed = float(row[f"mean_{metric}"])
            null_values = null_lookup[(profile_id, metric)]
            null = null_statistics[(profile_id, metric)]
            left = int(np.searchsorted(null_values, observed, side="left"))
            p_value = (len(null_values) - left + 1) / (len(null_values) + 1)
            rows.append(
                {
                    "schema_version": OUTPUT_SCHEMA,
                    **{key: row[key] for key in metadata},
                    "profile_id": profile_id,
                    "replicates": 250,
                    "null_family": "global_count_fixed",
                    "metric": metric,
                    "observed_mean": observed,
                    "null_draws": len(null_values),
                    "null_mean": null["mean"],
                    "null_sd": null["sd"],
                    "null_q025": null["q025"],
                    "null_median": null["median"],
                    "null_q975": null["q975"],
                    "observed_minus_null": observed - null["mean"],
                    "standardized_deviation": (
                        (observed - null["mean"]) / null["sd"]
                        if null["sd"] > 1e-15
                        else math.nan
                    ),
                    "upper_p": p_value,
                    "null_degenerate": bool(null["sd"] <= 1e-15),
                }
            )
    frame = pd.DataFrame(rows)
    for _, indices in frame.groupby(["grid_index", "metric"]).groups.items():
        frame.loc[indices, "upper_q_bh"] = _bh_adjust(
            frame.loc[indices, "upper_p"].to_numpy(np.float64)
        )
    frame["pointwise_q_le_0_05"] = frame.upper_q_bh.le(0.05)
    frame = frame.sort_values(["metric", "condition_id", "grid_index"])
    _write_parquet(frame, output / "pointwise_global_deviations.parquet")
    result = {
        "schema": "e04.s05.pointwise_global_accounting.v1",
        "researchStepId": "S05",
        "conditionGridPoints": int(response.shape[0]),
        "metrics": list(GLOBAL_METRICS),
        "rows": len(frame),
        "nullDrawsPerProfile": GLOBAL_GROUP_DRAWS,
        "observedReplicatesPerCondition": 250,
        "pointwiseSignificantRows": int(frame.pointwise_q_le_0_05.sum()),
        "noPeakStatisticsComputed": True,
    }
    write_json(output / "pointwise_global_accounting.json", result)
    return result


def build_conditioned_null_artifacts(
    checkpoint: Path, output: Path
) -> dict[str, Any]:
    group_nulls = pd.read_parquet(
        output / "static_nulls/global_group_mean_nulls.parquet"
    )
    group_nulls = group_nulls[group_nulls.group_size.eq(25)]
    global_lookup = {
        (profile_id, metric): group[metric].to_numpy(np.float64)
        for profile_id, group in group_nulls.groupby("profile_id")
        for metric in GLOBAL_METRICS
    }
    conditional_path = output / "static_nulls/value_conditioned_group_nulls.parquet"
    global_equal_path = output / "static_nulls/global_equal_value_group_nulls.parquet"
    conditional_writer: pq.ParquetWriter | None = None
    global_equal_writer: pq.ParquetWriter | None = None
    summary_rows: list[dict[str, Any]] = []
    calibration_p_rows: list[dict[str, Any]] = []
    preservation_rows: list[dict[str, Any]] = []
    exact_unique_rows: list[dict[str, Any]] = []
    group_count = 0
    try:
        for record in _conditional_checkpoint_records(checkpoint):
            group_count += 1
            draws = record["conditional_null_draws"]
            conditional_frame = pd.DataFrame(
                {
                    "schema_version": OUTPUT_SCHEMA,
                    "null_family": "value_stratum_count_fixed",
                    "condition_id": record["condition_id"],
                    "grid_index": int(record["grid_index"]),
                    "accepted_swap_progress": float(record["accepted_swap_progress"]),
                    "input_profile": record["metadata"]["input_profile"],
                    "profile_id": record["profile_id"],
                    "draw_index": np.arange(CONDITIONAL_DRAWS, dtype=np.int64),
                    **{
                        metric: np.asarray(
                            draws.get(metric, [math.nan] * CONDITIONAL_DRAWS),
                            dtype=np.float64,
                        )
                        for metric in CONDITIONAL_METRICS
                    },
                }
            )
            table = pa.Table.from_pandas(conditional_frame, preserve_index=False)
            if conditional_writer is None:
                conditional_writer = pq.ParquetWriter(
                    conditional_path, table.schema, compression="zstd"
                )
            conditional_writer.write_table(table)

            input_profile = record["metadata"]["input_profile"]
            for metric, null_list in draws.items():
                null_values = np.asarray(null_list, dtype=np.float64)
                observed = float(record["observed_means"][metric])
                summary_rows.append(
                    _null_summary_row(
                        record=record,
                        null_family="value_stratum_count_fixed",
                        metric=metric,
                        observed=observed,
                        null_values=null_values,
                    )
                )
                pseudo = float(record["conditional_pseudo_observed"][metric])
                rng = np.random.Generator(
                    np.random.PCG64DXSM(
                        derive_seed(
                            "conditional_calibration_tie",
                            record["condition_id"],
                            record["grid_index"],
                            metric,
                        )
                    )
                )
                _, randomized = _upper_tail_values(
                    null_values, np.asarray([pseudo]), randomized_u=rng.random(1)
                )
                assert randomized is not None
                calibration_p_rows.append(
                    {
                        "condition_id": record["condition_id"],
                        "grid_index": int(record["grid_index"]),
                        "input_profile": input_profile,
                        "metric": metric,
                        "randomized_upper_p": float(randomized[0]),
                    }
                )

            for metric in GLOBAL_METRICS:
                null_values = global_lookup[(record["profile_id"], metric)]
                summary_rows.append(
                    _null_summary_row(
                        record=record,
                        null_family="global_count_fixed",
                        metric=metric,
                        observed=float(record["observed_means"][metric]),
                        null_values=null_values,
                    )
                )

            global_equal = record["global_equal_value_null_draws"]
            if global_equal is not None:
                global_equal_values = np.asarray(global_equal, dtype=np.float64)
                equal_frame = pd.DataFrame(
                    {
                        "schema_version": OUTPUT_SCHEMA,
                        "null_family": "global_count_fixed",
                        "condition_id": record["condition_id"],
                        "grid_index": int(record["grid_index"]),
                        "accepted_swap_progress": float(
                            record["accepted_swap_progress"]
                        ),
                        "input_profile": input_profile,
                        "profile_id": record["profile_id"],
                        "draw_index": np.arange(CONDITIONAL_DRAWS, dtype=np.int64),
                        "equal_value_corrected_excess": global_equal_values,
                    }
                )
                equal_table = pa.Table.from_pandas(equal_frame, preserve_index=False)
                if global_equal_writer is None:
                    global_equal_writer = pq.ParquetWriter(
                        global_equal_path, equal_table.schema, compression="zstd"
                    )
                global_equal_writer.write_table(equal_table)
                summary_rows.append(
                    _null_summary_row(
                        record=record,
                        null_family="global_count_fixed",
                        metric="equal_value_corrected_excess",
                        observed=float(
                            record["observed_means"][
                                "equal_value_corrected_excess"
                            ]
                        ),
                        null_values=global_equal_values,
                    )
                )

            preservation_rows.append(
                {
                    "condition_id": record["condition_id"],
                    "grid_index": int(record["grid_index"]),
                    "input_profile": input_profile,
                    "profile_id": record["profile_id"],
                    "replicates": int(record["replicates"]),
                    "draws_per_group": CONDITIONAL_DRAWS,
                    "support_size_min": str(record["support_size_min"]),
                    "support_size_max": str(record["support_size_max"]),
                    "support_log10_min": float(record["support_log10_min"]),
                    "support_log10_max": float(record["support_log10_max"]),
                    "global_count_violations": int(
                        record["global_count_violations"]
                    ),
                    "stratum_count_violations": int(
                        record["stratum_count_violations"]
                    ),
                    "pseudo_constraint_violations": int(
                        record["pseudo_constraint_violations"]
                    ),
                    "position_scaffold_violations": 0,
                    "value_scaffold_violations": 0,
                }
            )
            if input_profile == "unique_1_100":
                exact_unique_rows.append(
                    {
                        "condition_id": record["condition_id"],
                        "grid_index": int(record["grid_index"]),
                        "input_profile": input_profile,
                        "strict_null_family": "exact_value_count_fixed_unique_sensitivity",
                        "exact_value_strata": 100,
                        "singleton_strata": 100,
                        "support_size": 1,
                        "p_value_defined": False,
                        "reason": "all exact-value strata are singletons; labels cannot move",
                    }
                )
    finally:
        if conditional_writer is not None:
            conditional_writer.close()
        if global_equal_writer is not None:
            global_equal_writer.close()

    if group_count != EXPECTED_CONDITIONS * len(AUDIT_GRID):
        raise AssertionError("conditioned artifact group count changed")
    summary = pd.DataFrame(summary_rows)
    for _, indices in summary.groupby(["null_family", "grid_index", "metric"]).groups.items():
        summary.loc[indices, "upper_q_bh"] = _bh_adjust(
            summary.loc[indices, "upper_p"].to_numpy(np.float64)
        )
    summary["pointwise_q_le_0_05"] = summary.upper_q_bh.le(0.05)
    summary = summary.sort_values(
        ["null_family", "metric", "condition_id", "grid_index"]
    )
    _write_parquet(summary, output / "conditional_pointwise_deviations.parquet")

    comparable = summary.pivot_table(
        index=[
            "condition_id",
            "grid_index",
            "accepted_swap_progress",
            "input_profile",
            "policy_set_label",
            "composition_class",
            "composition_profile",
            "correlation_profile",
            "metric",
        ],
        columns="null_family",
        values=[
            "observed_mean",
            "null_mean",
            "null_sd",
            "observed_minus_null",
            "standardized_deviation",
            "upper_p",
            "upper_q_bh",
            "pointwise_q_le_0_05",
        ],
        aggfunc="first",
    )
    comparable.columns = [f"{field}__{family}" for field, family in comparable.columns]
    comparable = comparable.reset_index()
    for field in ("null_mean", "observed_minus_null", "standardized_deviation"):
        global_column = f"{field}__global_count_fixed"
        conditional_column = f"{field}__value_stratum_count_fixed"
        if global_column in comparable and conditional_column in comparable:
            comparable[f"conditional_minus_global__{field}"] = (
                comparable[conditional_column] - comparable[global_column]
            )
    comparable["lost_significance_after_conditioning"] = (
        comparable.get("pointwise_q_le_0_05__global_count_fixed", False).astype(bool)
        & ~comparable.get(
            "pointwise_q_le_0_05__value_stratum_count_fixed", False
        ).astype(bool)
    )
    _write_parquet(
        comparable, output / "conditioning_assumption_comparison.parquet"
    )

    calibration_p = pd.DataFrame(calibration_p_rows)
    _write_parquet(
        calibration_p, output / "conditional_calibration_pvalues.parquet"
    )
    calibration_rows: list[dict[str, Any]] = []
    for (input_profile, metric), group in calibration_p.groupby(
        ["input_profile", "metric"]
    ):
        values = group.randomized_upper_p.to_numpy(np.float64)
        for alpha in (0.01, 0.05, 0.10):
            rate = float(np.mean(values <= alpha))
            se = math.sqrt(alpha * (1 - alpha) / len(values))
            tolerance = max(5 * se, 0.01)
            calibration_rows.append(
                {
                    "input_profile": input_profile,
                    "metric": metric,
                    "alpha": alpha,
                    "groups": len(values),
                    "randomized_false_positive_rate": rate,
                    "binomial_se": se,
                    "tolerance": tolerance,
                    "passed": abs(rate - alpha) <= tolerance,
                }
            )
    calibration = pd.DataFrame(calibration_rows).sort_values(
        ["input_profile", "metric", "alpha"]
    )
    _write_parquet(
        calibration, output / "conditional_false_positive_calibration.parquet"
    )
    preservation = pd.DataFrame(preservation_rows).sort_values(
        ["condition_id", "grid_index"]
    )
    _write_parquet(preservation, output / "permutation_preservation_audit.parquet")
    exact_unique = pd.DataFrame(exact_unique_rows).sort_values(
        ["condition_id", "grid_index"]
    )
    _write_parquet(exact_unique, output / "exact_value_degeneracy.parquet")

    total_violations = int(
        preservation[
            [
                "global_count_violations",
                "stratum_count_violations",
                "pseudo_constraint_violations",
                "position_scaffold_violations",
                "value_scaffold_violations",
            ]
        ]
        .to_numpy(np.int64)
        .sum()
    )
    global_significant = comparable.get(
        "pointwise_q_le_0_05__global_count_fixed", pd.Series(False, index=comparable.index)
    ).astype(bool)
    lost = comparable.lost_significance_after_conditioning.astype(bool)
    loss_fraction = float(lost.sum() / global_significant.sum()) if global_significant.sum() else math.nan
    accounting = {
        "schema": "e04.s05.conditioned_null_accounting.v1",
        "researchStepId": "S05",
        "conditionGridGroups": group_count,
        "valueConditionedGroupNullRows": pq.ParquetFile(
            conditional_path
        ).metadata.num_rows,
        "globalEqualValueGroupNullRows": pq.ParquetFile(
            global_equal_path
        ).metadata.num_rows,
        "drawsPerGroup": CONDITIONAL_DRAWS,
        "conditionalPointwiseRows": len(summary),
        "comparableRows": len(comparable),
        "strictUniqueDegenerateRows": len(exact_unique),
        "totalPreservationViolations": total_violations,
        "conditionalFalsePositiveRows": len(calibration),
        "allConditionalFalsePositiveChecksPassed": bool(calibration.passed.all()),
        "globallySignificantComparableRows": int(global_significant.sum()),
        "lostSignificanceAfterConditioningRows": int(lost.sum()),
        "lossFractionAmongGloballySignificant": loss_fraction,
        "conditioningDescriptor": (
            "conditioning-sensitive"
            if math.isfinite(loss_fraction) and loss_fraction >= 0.25
            else "conditioning-robust"
        ),
    }
    if total_violations:
        raise AssertionError("permutation preservation constraints failed")
    if not accounting["allConditionalFalsePositiveChecksPassed"]:
        raise AssertionError("conditional false-positive calibration failed")
    write_json(output / "conditioned_null_accounting.json", accounting)
    return accounting


def _save_figure(fig: Any, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _plot_calibration(output: Path) -> None:
    uniform = pd.read_parquet(output / "uniform_permutation_validation.parquet")
    global_fp = pd.read_parquet(output / "global_false_positive_calibration.parquet")
    conditional_fp = pd.read_parquet(
        output / "conditional_false_positive_calibration.parquet"
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    nondegenerate = uniform[uniform.support_size > 1]
    x = np.arange(len(nondegenerate))
    axes[0].bar(x - 0.18, nondegenerate.total_variation, width=0.36, label="Observed TV")
    axes[0].bar(
        x + 0.18,
        nondegenerate.total_variation_tolerance,
        width=0.36,
        label="Frozen tolerance",
    )
    axes[0].set_xticks(x, nondegenerate.fixture_id, rotation=25, ha="right")
    axes[0].set_ylabel("Total-variation distance")
    axes[0].set_title("Uniform sampler fixtures")
    axes[0].legend(fontsize=8)

    for metric, group in global_fp.groupby("metric"):
        means = group.groupby("alpha").randomized_false_positive_rate.mean()
        axes[1].plot(means.index, means.values, marker="o", label=metric)
    axes[1].plot((0, 0.11), (0, 0.11), color="black", linestyle="--", label="Ideal")
    axes[1].set_xlim(0, 0.11)
    axes[1].set_ylim(0, 0.11)
    axes[1].set_xlabel("Nominal α")
    axes[1].set_ylabel("Randomized false-positive rate")
    axes[1].set_title("Global null calibration")
    axes[1].legend(fontsize=6)

    for input_profile, group in conditional_fp.groupby("input_profile"):
        means = group.groupby("alpha").randomized_false_positive_rate.mean()
        axes[2].plot(means.index, means.values, marker="o", label=input_profile)
    axes[2].plot((0, 0.11), (0, 0.11), color="black", linestyle="--", label="Ideal")
    axes[2].set_xlim(0, 0.11)
    axes[2].set_ylim(0, 0.11)
    axes[2].set_xlabel("Nominal α")
    axes[2].set_ylabel("Randomized false-positive rate")
    axes[2].set_title("Value-conditioned calibration")
    axes[2].legend(fontsize=7)
    fig.suptitle("E04 S05 static permutation calibration", fontsize=14)
    fig.tight_layout()
    _save_figure(fig, output, "permutation_calibration")


def _plot_global_nulls(output: Path) -> None:
    summary = pd.read_parquet(output / "global_null_summary.parquet")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, metric in zip(axes.flat, GLOBAL_METRICS):
        source = summary[summary.metric.eq(metric)].copy()
        source["order"] = source.profile_id.map(
            { _profile_id(profile): index for index, profile in enumerate(GLOBAL_PROFILES) }
        )
        source = source.sort_values("order")
        x = np.arange(len(source))
        axis.errorbar(
            x,
            source["mean"],
            yerr=[source["mean"] - source.q025, source.q975 - source["mean"]],
            fmt="o-",
            capsize=3,
        )
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(metric.replace("_", " "))
        axis.set_xticks(x, source.profile_id, rotation=30, ha="right")
        axis.grid(alpha=0.2)
    fig.suptitle("Static global label-permutation distributions (arrangement level)")
    fig.tight_layout()
    _save_figure(fig, output, "static_null_distributions")


def _plot_pointwise(output: Path) -> None:
    pointwise = pd.read_parquet(output / "pointwise_global_deviations.parquet")
    rates = (
        pointwise.groupby(["metric", "grid_index"])
        .pointwise_q_le_0_05.mean()
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for metric, group in rates.groupby("metric"):
        axis.plot(
            group.grid_index,
            group.pointwise_q_le_0_05,
            label=metric.replace("_", " "),
        )
    axis.set_xlabel("Fixed accepted-swap progress (%)")
    axis.set_ylabel("Fraction of 186 conditions with pointwise BH q≤0.05")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    axis.set_title("Pointwise global-null exceedance; no peak correction")
    fig.tight_layout()
    _save_figure(fig, output, "pointwise_global_exceedance")


def _plot_conditioning(output: Path) -> None:
    comparison = pd.read_parquet(
        output / "conditioning_assumption_comparison.parquet"
    )
    comparison = comparison[
        comparison.metric.eq("corrected_publication_aggregation")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for association, group in comparison.groupby("correlation_profile"):
        axes[0].scatter(
            group["observed_minus_null__global_count_fixed"],
            group["observed_minus_null__value_stratum_count_fixed"],
            s=10,
            alpha=0.45,
            label=association,
        )
    limits = [
        min(
            comparison["observed_minus_null__global_count_fixed"].min(),
            comparison["observed_minus_null__value_stratum_count_fixed"].min(),
        ),
        max(
            comparison["observed_minus_null__global_count_fixed"].max(),
            comparison["observed_minus_null__value_stratum_count_fixed"].max(),
        ),
    ]
    axes[0].plot(limits, limits, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Observed minus global null mean")
    axes[0].set_ylabel("Observed minus value-conditioned null mean")
    axes[0].set_title("Adjacency deviation changes with conditioning")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    rates = (
        comparison.groupby(["correlation_profile", "grid_index"])
        .agg(
            global_significant=(
                "pointwise_q_le_0_05__global_count_fixed",
                "mean",
            ),
            conditional_significant=(
                "pointwise_q_le_0_05__value_stratum_count_fixed",
                "mean",
            ),
        )
        .reset_index()
    )
    for association, group in rates.groupby("correlation_profile"):
        axes[1].plot(
            group.grid_index,
            group.global_significant,
            marker="o",
            label=f"{association}: global",
        )
        axes[1].plot(
            group.grid_index,
            group.conditional_significant,
            marker="x",
            linestyle="--",
            label=f"{association}: conditioned",
        )
    axes[1].set_xlabel("Fixed accepted-swap progress (%)")
    axes[1].set_ylabel("Fraction pointwise BH q≤0.05")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Adjacency significance by conditioning")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.2)
    fig.suptitle("E04 S05 conditioning-assumption audit")
    fig.tight_layout()
    _save_figure(fig, output, "conditioning_comparison")


def analyze_static_nulls(output: Path) -> dict[str, Any]:
    pointwise = pd.read_parquet(output / "pointwise_global_deviations.parquet")
    adjacency = pointwise.metric.eq("corrected_publication_aggregation")
    balanced_pair_absent_25 = pointwise[
        adjacency
        & pointwise.grid_index.eq(25)
        & pointwise.composition_class.eq("pairwise")
        & pointwise.composition_profile.eq("p50_50")
        & pointwise.correlation_profile.eq("absent")
    ]
    final_association = pointwise[
        pointwise.grid_index.eq(100)
        & pointwise.correlation_profile.isin(["positive", "negative"])
        & (
            (
                pointwise.composition_class.eq("pairwise")
                & pointwise.composition_profile.eq("p50_50")
            )
            | (
                pointwise.composition_class.eq("three_way")
                & pointwise.composition_profile.eq("balanced_rotating")
            )
        )
    ]
    if len(balanced_pair_absent_25) != 6:
        raise AssertionError("primary 25% adjacency anchor count changed")
    final_by_metric = {
        metric: int(
            final_association[final_association.metric.eq(metric)]
            .pointwise_q_le_0_05.astype(bool)
            .sum()
        )
        for metric in GLOBAL_METRICS
    }
    if any(
        len(final_association[final_association.metric.eq(metric)]) != 16
        for metric in GLOBAL_METRICS
    ):
        raise AssertionError("fixed-final association anchor count changed")
    primary_25_passing = int(
        balanced_pair_absent_25.pointwise_q_le_0_05.astype(bool).sum()
    )
    primary_final_passing = final_by_metric[
        "corrected_publication_aggregation"
    ]
    corroborating_metrics = sum(
        final_by_metric[metric] >= 12
        for metric in (
            "categorical_assortativity",
            "largest_label_capture",
            "signed_neighbor_nmi",
        )
    )
    global_accounting = json.loads(
        (output / "global_null_accounting.json").read_text()
    )
    conditioned_accounting = json.loads(
        (output / "conditioned_null_accounting.json").read_text()
    )
    uniformity = json.loads(
        (output / "uniformity_validation_summary.json").read_text()
    )
    execution_passed = bool(
        pq.ParquetFile(output / "pointwise_global_deviations.parquet").metadata.num_rows
        == EXPECTED_CONDITIONS * len(GRID) * len(GLOBAL_METRICS)
        and conditioned_accounting["conditionGridGroups"]
        == EXPECTED_CONDITIONS * len(AUDIT_GRID)
    )
    calibration_passed = bool(
        uniformity["allPassed"]
        and global_accounting["allFalsePositiveChecksPassed"]
        and conditioned_accounting["allConditionalFalsePositiveChecksPassed"]
        and conditioned_accounting["totalPreservationViolations"] == 0
    )
    primary_passed = primary_25_passing >= 4 and primary_final_passing >= 12
    corroboration_passed = corroborating_metrics >= 2
    all_passed = (
        execution_passed
        and calibration_passed
        and primary_passed
        and corroboration_passed
    )
    if all_passed:
        outcome = "supportive"
    elif execution_passed and calibration_passed:
        outcome = "constraining/contradictory"
    else:
        outcome = "null"
    success = {
        "schema": "e04.s05.success_criteria.v1",
        "researchStepId": "S05",
        "execution": {"passed": execution_passed},
        "calibration": {"passed": calibration_passed},
        "primaryAdjacency": {
            "passed": primary_passed,
            "balancedAbsentAt25Passing": primary_25_passing,
            "balancedAbsentAt25Required": 4,
            "balancedAbsentAt25Total": 6,
            "balancedAssociationAtFinalPassing": primary_final_passing,
            "balancedAssociationAtFinalRequired": 12,
            "balancedAssociationAtFinalTotal": 16,
        },
        "metricCorroboration": {
            "passed": corroboration_passed,
            "passingMetrics": corroborating_metrics,
            "requiredMetrics": 2,
            "fixedFinalPassingByMetric": final_by_metric,
        },
        "conditioningDescriptor": conditioned_accounting[
            "conditioningDescriptor"
        ],
        "allPassed": all_passed,
        "outcomeClassification": outcome,
    }
    write_json(output / "success_criteria.json", success)

    comparison = pd.read_parquet(
        output / "conditioning_assumption_comparison.parquet"
    )
    change_rows = []
    for (input_profile, association, metric), group in comparison.groupby(
        ["input_profile", "correlation_profile", "metric"]
    ):
        global_sig = group[
            "pointwise_q_le_0_05__global_count_fixed"
        ].astype(bool)
        conditional_sig = group[
            "pointwise_q_le_0_05__value_stratum_count_fixed"
        ].astype(bool)
        change_rows.append(
            {
                "input_profile": input_profile,
                "correlation_profile": association,
                "metric": metric,
                "condition_grid_rows": len(group),
                "global_significant_rows": int(global_sig.sum()),
                "conditioned_significant_rows": int(conditional_sig.sum()),
                "lost_significance_rows": int((global_sig & ~conditional_sig).sum()),
                "gained_significance_rows": int((~global_sig & conditional_sig).sum()),
                "mean_conditional_minus_global_null": float(
                    group["conditional_minus_global__null_mean"].mean()
                ),
                "mean_change_in_observed_minus_null": float(
                    group[
                        "conditional_minus_global__observed_minus_null"
                    ].mean()
                ),
            }
        )
    changes = pd.DataFrame(change_rows).sort_values(
        ["metric", "input_profile", "correlation_profile"]
    )
    changes.to_csv(output / "conditioning_change_summary.csv", index=False)

    _plot_calibration(output)
    _plot_global_nulls(output)
    _plot_pointwise(output)
    _plot_conditioning(output)
    analysis = {
        "schema": "e04.s05.analysis_summary.v1",
        "researchStepId": "S05",
        "outcomeClassification": outcome,
        "primaryBalancedAbsentAt25Passing": primary_25_passing,
        "primaryBalancedAssociationAtFinalPassing": primary_final_passing,
        "fixedFinalPassingByMetric": final_by_metric,
        "conditioningDescriptor": conditioned_accounting[
            "conditioningDescriptor"
        ],
        "globallySignificantComparableRows": conditioned_accounting[
            "globallySignificantComparableRows"
        ],
        "lostSignificanceAfterConditioningRows": conditioned_accounting[
            "lostSignificanceAfterConditioningRows"
        ],
        "lossFractionAmongGloballySignificant": conditioned_accounting[
            "lossFractionAmongGloballySignificant"
        ],
        "pointwiseOnly": True,
        "peakStatisticsComputed": False,
    }
    write_json(output / "analysis_summary.json", analysis)
    return analysis


def validate_artifacts(output: Path) -> dict[str, Any]:
    required = (
        "preregistration.json",
        "freeze_record.json",
        "null_specification.md",
        "uniform_permutation_validation.parquet",
        "uniformity_validation_summary.json",
        "global_null_summary.parquet",
        "global_false_positive_calibration.parquet",
        "global_null_accounting.json",
        "snapshot_states.parquet",
        "snapshot_replay_summary.json",
        "pointwise_global_deviations.parquet",
        "pointwise_global_accounting.json",
        "conditional_pointwise_deviations.parquet",
        "conditioning_assumption_comparison.parquet",
        "conditioning_change_summary.csv",
        "conditional_calibration_pvalues.parquet",
        "conditional_false_positive_calibration.parquet",
        "permutation_preservation_audit.parquet",
        "exact_value_degeneracy.parquet",
        "conditioned_null_accounting.json",
        "success_criteria.json",
        "analysis_summary.json",
        "permutation_calibration.png",
        "permutation_calibration.svg",
        "static_null_distributions.png",
        "static_null_distributions.svg",
        "pointwise_global_exceedance.png",
        "pointwise_global_exceedance.svg",
        "conditioning_comparison.png",
        "conditioning_comparison.svg",
        "research_step_full_results.md",
    )
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "observed": observed})

    missing = [name for name in required if not (output / name).is_file()]
    add("required_artifacts_present", not missing, missing)
    frozen = assert_frozen(output)
    add(
        "contract_frozen",
        frozen["contractSha256"] == sha256_file(CONTRACT_PATH),
        frozen["contractSha256"],
    )
    s04_integrity = verify_s04_table()
    add("canonical_s04_table_integrity", s04_integrity["passed"], s04_integrity)

    global_parts = sorted(
        (output / "static_nulls/global_arrangement_nulls").glob("*.parquet")
    )
    global_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in global_parts)
    add(
        "global_arrangement_corpus",
        len(global_parts) == len(GLOBAL_PROFILES)
        and global_rows == len(GLOBAL_PROFILES) * GLOBAL_DRAWS,
        {"parts": len(global_parts), "rows": global_rows},
    )
    group_rows = pq.ParquetFile(
        output / "static_nulls/global_group_mean_nulls.parquet"
    ).metadata.num_rows
    add(
        "global_group_mean_corpus",
        group_rows == len(GLOBAL_PROFILES) * 2 * GLOBAL_GROUP_DRAWS,
        group_rows,
    )
    uniformity = json.loads(
        (output / "uniformity_validation_summary.json").read_text()
    )
    add(
        "uniform_permutation_sampling",
        uniformity["allPassed"] and uniformity["totalConstraintViolations"] == 0,
        uniformity,
    )
    global_accounting = json.loads(
        (output / "global_null_accounting.json").read_text()
    )
    add(
        "global_false_positive_calibration",
        global_accounting["allFalsePositiveChecksPassed"],
        global_accounting,
    )
    snapshot = json.loads((output / "snapshot_replay_summary.json").read_text())
    add(
        "snapshot_replay",
        snapshot["allPassed"]
        and snapshot["scenarios"] == EXPECTED_AUDIT_SCENARIOS
        and snapshot["snapshots"] == EXPECTED_SNAPSHOTS,
        snapshot,
    )
    pointwise_rows = pq.ParquetFile(
        output / "pointwise_global_deviations.parquet"
    ).metadata.num_rows
    add(
        "full_pointwise_accounting",
        pointwise_rows == EXPECTED_CONDITIONS * len(GRID) * len(GLOBAL_METRICS),
        pointwise_rows,
    )
    conditioned = json.loads(
        (output / "conditioned_null_accounting.json").read_text()
    )
    add(
        "conditioned_corpus_accounting",
        conditioned["conditionGridGroups"] == EXPECTED_CONDITIONS * len(AUDIT_GRID)
        and conditioned["valueConditionedGroupNullRows"]
        == EXPECTED_CONDITIONS * len(AUDIT_GRID) * CONDITIONAL_DRAWS,
        conditioned,
    )
    add(
        "permutation_preservation",
        conditioned["totalPreservationViolations"] == 0,
        conditioned["totalPreservationViolations"],
    )
    add(
        "conditional_false_positive_calibration",
        conditioned["allConditionalFalsePositiveChecksPassed"],
        conditioned["conditionalFalsePositiveRows"],
    )
    degeneracy = pd.read_parquet(output / "exact_value_degeneracy.parquet")
    add(
        "unique_exact_value_degeneracy",
        len(degeneracy) == 93 * len(AUDIT_GRID)
        and degeneracy.support_size.eq(1).all()
        and (~degeneracy.p_value_defined).all(),
        {"rows": len(degeneracy), "supportSizes": sorted(degeneracy.support_size.unique())},
    )
    success = json.loads((output / "success_criteria.json").read_text())
    add("success_criteria_recorded", success["allPassed"], success)
    upstream = {
        step: _verify_upstream_manifest(
            Path(f"/artifacts/research_steps/{step}"), expected
        )
        for step, expected in EXPECTED_MANIFEST_HASHES.items()
    }
    add(
        "upstream_immutability",
        all(value["allPassed"] for value in upstream.values()),
        {step: value["allPassed"] for step, value in upstream.items()},
    )
    write_json(
        output / "upstream_immutability_audit.json",
        {
            "schema": "e04.s05.upstream_immutability_audit.v1",
            "researchStepId": "S05",
            "steps": upstream,
            "s04TableIntegrity": s04_integrity,
            "allPassed": all(value["allPassed"] for value in upstream.values())
            and s04_integrity["passed"],
        },
    )
    report = (output / "research_step_full_results.md").read_text()
    report_fields = (
        "Research step ID",
        "Completion status",
        "Artifacts written",
        "Validation result",
        "Outcome classification",
        "Caveats or blockers",
        "Lay summary",
        "Recommended next action",
        "## Methods",
        "## Commands",
        "## Provenance",
        "## Validation",
    )
    add(
        "canonical_report_contract",
        all(field in report for field in report_fields),
        [field for field in report_fields if field not in report],
    )
    add("s06_scope_boundary", not S06_DIR.exists(), str(S06_DIR))
    failed = [item["check"] for item in checks if not item["passed"]]
    result = {
        "schema": "e04.s05.validation_summary.v1",
        "researchStepId": "S05",
        "totalChecks": len(checks),
        "passedChecks": len(checks) - len(failed),
        "failedChecks": failed,
        "checks": checks,
        "allPassed": not failed,
    }
    write_json(output / "validation_summary.json", result)
    if failed:
        raise AssertionError(f"S05 validation failed: {failed}")
    return result


def write_provenance(output: Path, cache: Path) -> dict[str, Any]:
    input_paths = {
        "agents": Path("/workspace/AGENTS.md"),
        "fullPlan": Path("/workspace/FULL_PLAN.md"),
        "researchPlan": Path("/workspace/RESEARCH_PLAN.md"),
        "attachmentManifest": Path("/workspace/input-attachments/MANIFEST.json"),
        "attachmentSidecar": Path(
            "/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md"
        ),
        "paperMarkdown": Path(
            "/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/pdf-markdown.md"
        ),
        "previousArtifactsMarkdown": Path("/workspace/PREVIOUS_ARTIFACTS.md"),
        "previousArtifactsJson": Path("/workspace/PREVIOUS_ARTIFACTS.json"),
        "e01TransitionSpecification": Path(
            "/previous-artifacts/E01/specification/transition_spec.md"
        ),
        "s01Report": S01_DIR / "research_step_full_results.md",
        "s02Report": S02_DIR / "research_step_full_results.md",
        "s03Report": S03_DIR / "research_step_full_results.md",
        "s04Report": S04_DIR / "research_step_full_results.md",
        "s04CanonicalMetrics": S04_TABLE_PATH,
    }
    sources = (
        CONTRACT_PATH,
        REPOSITORY / "analysis/static_nulls.py",
        REPOSITORY / "scripts/run_static_nulls.py",
        REPOSITORY / "tests/test_e04_static_nulls.py",
    )
    value = {
        "schema": "e04.s05.provenance.v1",
        "researchStepId": "S05",
        "capturedAtUtc": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPOSITORY),
        "branch": _git_output("branch", "--show-current"),
        "gitHeadAtCapture": _git_output("rev-parse", "HEAD"),
        "gitStatusShort": _git_output("status", "--short"),
        "contractSha256": sha256_file(CONTRACT_PATH),
        "inputs": {
            key: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for key, path in input_paths.items()
        },
        "sourceFiles": [
            {"path": str(path.relative_to(REPOSITORY)), "sha256": sha256_file(path)}
            for path in sources
        ],
        "sourceFilesInRepositoryNotCopiedToArtifacts": True,
        "cache": {
            "path": str(cache),
            "collectible": False,
            "checkpointFiles": sorted(
                str(path) for path in cache.glob("**/*") if path.is_file()
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workers": 8,
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
                "scipy": stats.__version__ if hasattr(stats, "__version__") else None,
                "matplotlib": matplotlib.__version__,
            },
        },
        "s04CollectionSkipHandling": {
            "userReportedExternalSkipMarker": True,
            "visibleLocalMarker": False,
            "canonicalTableVerifiedByBytesAndSha256": True,
            "tableSha256": sha256_file(S04_TABLE_PATH),
        },
    }
    write_json(output / "provenance.json", value)
    write_json(output / "environment.json", value["environment"])
    return value


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
    value = {
        "schema": "e04.s05.artifact_manifest.v1",
        "researchStepId": "S05",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "artifactCount": len(artifacts),
        "artifacts": artifacts,
    }
    write_json(output / "artifact_manifest.json", value)
    return value
