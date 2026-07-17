"""E04 S04 alternative aggregation metrics and frozen S03 replay.

Metric definitions and validation gates are frozen in
``analysis/s04_metric_contract.json``.  This module never mutates S01--S03;
it re-materializes the complete validated S03 population and samples several
spatial summaries on the same accepted-swap grid.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import itertools
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
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

from analysis.aggregation_baselines import expected_paper_aggregation
from analysis.chimeric_replication import (
    AdmissibilityTracker,
    GRID,
    MetricTracker,
    _execute_s13_activation,
    canonical_hash,
)
from analysis.composition_sweep import (
    EXPECTED_CONDITIONS,
    EXPECTED_RUNS,
    SweepCondition,
    build_tasks,
    materialize_sweep_scenario,
)
from reference_simulator.engine import evaluate_terminal, initial_state
from reference_simulator.model import canonical_json_bytes, state_hash


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY / "analysis/s04_metric_contract.json"
S01_DIR = Path("/artifacts/research_steps/S01")
S02_DIR = Path("/artifacts/research_steps/S02")
S03_DIR = Path("/artifacts/research_steps/S03")
S05_DIR = Path("/artifacts/research_steps/S05")
EXPECTED_MANIFEST_HASHES = {
    "S01": "df9860285fc5b1ed9f7442fb68ea5e9ed9e407918d90988d0b3592f4b3ef8079",
    "S02": "9e632eedac59f6b08ff6b619ce2451a99b49c952319af2747076fd7bb3411b54",
    "S03": "85371d82a53096764b0de52029472a7978261ca5d4857f4d392f441db076f9e9",
}
OUTPUT_SCHEMA = "e04.s04.aggregation_metrics.v1"
CHECKPOINT_SCHEMA = "e04.s04.metric_checkpoint.v1"
EXPECTED_GRID_ROWS = EXPECTED_RUNS * len(GRID)
MONTE_CARLO_DRAWS = 100_000
BOOTSTRAP_DRAWS = 10_000
EXHAUSTIVE_MAX_N = 9
EXHAUSTIVE_MAX_K = 4
METRIC_COLUMNS = (
    "same_label_edge_count",
    "publication_aggregation",
    "edge_aggregation",
    "publication_composition_baseline",
    "edge_composition_baseline",
    "corrected_publication_aggregation",
    "corrected_edge_aggregation",
    "categorical_assortativity",
    "run_count",
    "mean_run_length",
    "run_fraction_of_attainable",
    "largest_cluster_size",
    "largest_cluster_fraction",
    "largest_label_capture",
    "neighbor_mi_bits",
    "neighbor_nmi",
    "signed_neighbor_nmi",
    "equal_value_edge_count",
    "equal_value_same_label_edge_count",
    "equal_value_same_label_rate",
    "equal_value_conditioned_baseline",
    "equal_value_corrected_excess",
)
INDEPENDENT_METRICS = (
    "categorical_assortativity",
    "largest_label_capture",
    "signed_neighbor_nmi",
)


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
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd"
    )


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT
    ).strip()


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


def aggregation_metrics(
    labels: Sequence[str], values: Sequence[int | float]
) -> dict[str, float]:
    """Compute every frozen S04 metric for one open-path arrangement."""
    if len(labels) != len(values) or len(labels) < 2:
        raise ValueError("labels and values must have the same length n>=2")
    n = len(labels)
    counts = Counter(labels)
    if any(count <= 0 for count in counts.values()):
        raise AssertionError("invalid label count")
    same = sum(left == right for left, right in zip(labels, labels[1:]))
    paper_baseline = expected_paper_aggregation(tuple(counts.values()))
    edge_baseline = paper_baseline * n / (n - 1)
    edge_rate = same / (n - 1)

    degree = [1] + [2] * (n - 2) + [1]
    stubs: Counter[str] = Counter()
    for label, node_degree in zip(labels, degree):
        stubs[label] += node_degree
    marginals = {label: count / (2 * (n - 1)) for label, count in stubs.items()}
    assort_chance = sum(value * value for value in marginals.values())
    assort_denominator = 1.0 - assort_chance
    assortativity = (
        (edge_rate - assort_chance) / assort_denominator
        if assort_denominator > 1e-15
        else math.nan
    )

    runs: list[tuple[str, int]] = []
    current_label = labels[0]
    current_length = 1
    for label in labels[1:]:
        if label == current_label:
            current_length += 1
        else:
            runs.append((current_label, current_length))
            current_label = label
            current_length = 1
    runs.append((current_label, current_length))
    run_count = len(runs)
    largest_by_label: dict[str, int] = {label: 0 for label in counts}
    for label, length in runs:
        largest_by_label[label] = max(largest_by_label[label], length)
    largest = max(largest_by_label.values())
    largest_capture = max(
        largest_by_label[label] / counts[label] for label in largest_by_label
    )

    joint: Counter[tuple[str, str]] = Counter()
    for left, right in zip(labels, labels[1:]):
        joint[(left, right)] += 1
        joint[(right, left)] += 1
    total_directed = 2 * (n - 1)
    mi = 0.0
    for (left, right), count in joint.items():
        probability = count / total_directed
        mi += probability * math.log2(
            probability / (marginals[left] * marginals[right])
        )
    entropy = -sum(
        probability * math.log2(probability)
        for probability in marginals.values()
        if probability > 0
    )
    neighbor_nmi = mi / entropy if entropy > 1e-15 else math.nan
    if math.isnan(assortativity) or math.isnan(neighbor_nmi):
        signed_nmi = math.nan
    elif abs(assortativity) <= 1e-15:
        signed_nmi = 0.0
    else:
        signed_nmi = math.copysign(neighbor_nmi, assortativity)

    within_value_counts: dict[int | float, Counter[str]] = defaultdict(Counter)
    for value, label in zip(values, labels):
        within_value_counts[value][label] += 1
    equal_edges_by_value: Counter[int | float] = Counter()
    equal_same = 0
    for left_index in range(n - 1):
        if values[left_index] == values[left_index + 1]:
            value = values[left_index]
            equal_edges_by_value[value] += 1
            equal_same += int(labels[left_index] == labels[left_index + 1])
    equal_edges = sum(equal_edges_by_value.values())
    if equal_edges:
        weighted_expected = 0.0
        for value, edge_count in equal_edges_by_value.items():
            stratum = within_value_counts[value]
            stratum_n = sum(stratum.values())
            if stratum_n < 2:
                raise AssertionError("equal edge found in singleton value stratum")
            probability = sum(
                count * (count - 1) for count in stratum.values()
            ) / (stratum_n * (stratum_n - 1))
            weighted_expected += edge_count * probability
        equal_raw = equal_same / equal_edges
        equal_baseline = weighted_expected / equal_edges
        equal_excess = equal_raw - equal_baseline
    else:
        equal_raw = equal_baseline = equal_excess = math.nan

    result = {
        "same_label_edge_count": float(same),
        "publication_aggregation": same / n,
        "edge_aggregation": edge_rate,
        "publication_composition_baseline": paper_baseline,
        "edge_composition_baseline": edge_baseline,
        "corrected_publication_aggregation": same / n - paper_baseline,
        "corrected_edge_aggregation": edge_rate - edge_baseline,
        "categorical_assortativity": assortativity,
        "run_count": float(run_count),
        "mean_run_length": n / run_count,
        "run_fraction_of_attainable": len(counts) / run_count,
        "largest_cluster_size": float(largest),
        "largest_cluster_fraction": largest / n,
        "largest_label_capture": largest_capture,
        "neighbor_mi_bits": mi,
        "neighbor_nmi": neighbor_nmi,
        "signed_neighbor_nmi": signed_nmi,
        "equal_value_edge_count": float(equal_edges),
        "equal_value_same_label_edge_count": float(equal_same),
        "equal_value_same_label_rate": equal_raw,
        "equal_value_conditioned_baseline": equal_baseline,
        "equal_value_corrected_excess": equal_excess,
    }
    if tuple(result) != METRIC_COLUMNS:
        raise AssertionError("metric schema drift")
    return result


def _metric_specification_markdown(contract_hash: str) -> str:
    return f"""# S04 aggregation metric specification

## Top summary

| Field | Frozen result |
| --- | --- |
| Research step ID | S04 |
| Completion status | Metric, denominator, conditioning, validation, and inference conventions frozen before S04 population analysis |
| Artifacts written | `preregistration.json`, `freeze_record.json`, and this `metric_specification.md`; implementation is repository-backed |
| Validation result | Pre-analysis structural checks require the complete validated S03 population and immutable S01–S03 manifests; empirical calibration was pending at freeze time |
| Outcome classification | Not classified at freeze time; frozen classification rules are in the preregistration |
| Caveats or blockers | Run length is algebraically redundant with adjacency on an open path; unsigned MI also detects alternation; equal-value conditioning is undefined for unique inputs |
| Recommended next action | Execute only the frozen S04 validation, S03 replay, disagreement analysis, and report; do not begin S05 |

Contract SHA-256: `{contract_hash}`.

## Common conventions

The label is cell policy. The geometry is the open one-dimensional occupancy array with exactly `n-1` unordered adjacent edges. Trajectories use 101 points indexed as `floor(progress × final accepted swaps)`. Missing quantities are encoded as null/NaN, never as zero.

## Original adjacency and exact composition correction

Let `C` be the number of same-policy adjacent edges and `n_k` fixed policy counts. The paper-compatible statistic is `C/n`; the conventional edge rate is `C/(n-1)`. Their exact random-permutation expectations are, respectively, `sum_k n_k(n_k-1)/n²` and `sum_k n_k(n_k-1)/[n(n-1)]`. Corrected values subtract these expectations.

## Categorical assortativity

Every undirected edge contributes both orientations. If `o=C/(n-1)`, `s_k` is the number of edge-end stubs carrying label `k`, `a_k=s_k/[2(n-1)]`, and `c=sum_k a_k²`, then `r=(o-c)/(1-c)`. This observed-endpoint convention exposes finite endpoint effects; homogeneous arrays are undefined.

## Runs and largest clusters

A run is a maximal constant-policy block. We report run count `R`, mean length `n/R`, and attainable fraction `K/R`. Because `R=n-C`, runs are an interpretable but non-independent restatement of adjacency. We separately report the longest run `L`, its array fraction `L/n`, and the primary composition-scaled companion `max_k L_k/n_k`.

## Neighbor mutual information

The symmetric directed edge table has `2(n-1)` observations. Mutual information is reported in bits and divided by endpoint entropy to obtain NMI. Unsigned NMI treats perfect alternation as strong dependence, so the primary clustering-direction companion is `sign(r) × NMI`. Homogeneous arrays are undefined.

## Equal-value-conditioned clustering

Only adjacent equal-value edges are eligible. Raw homotypy is the same-policy fraction among those edges. For each value stratum `v`, the exact fixed-within-value-count probability is `q_v=sum_k n_vk(n_vk-1)/[n_v(n_v-1)]`; the baseline is the observed eligible-edge-weighted mean of `q_v`. The corrected statistic subtracts it. Unique inputs have no eligible edges and are structurally undefined.

## Frozen robustness rule

Independent directional evidence comprises assortativity, largest-label capture, and signed NMI. Run length is excluded because it is algebraically redundant; original adjacency is the reference; unsigned MI is direction-ambiguous; and equal-value conditioning has a narrower repeated-input estimand. Support requires exact replay and calibration, at least 12 of 16 fixed-composition association anchors significant in at least two independent metrics, and final-condition rank correlation at least 0.70 for at least two independent metrics.

## Claim boundary

These are computational spatial summaries of simulated arrangements. They are not biological aggregation, affinity, causality, intention, or attractor evidence. Random-label calibration is static and is not the S05 dynamic peak null.
"""


def freeze_design(output: Path, cache: Path) -> dict[str, Any]:
    checkpoint = cache / "aggregation_metrics.jsonl"
    if checkpoint.exists():
        raise FileExistsError("cannot freeze S04 after a population checkpoint exists")
    if S05_DIR.exists():
        raise FileExistsError("S05 output already exists; S04 scope boundary is not clean")
    output.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT_PATH.read_text())
    contract_hash = sha256_file(CONTRACT_PATH)
    upstream = {
        step: _verify_upstream_manifest(
            Path(f"/artifacts/research_steps/{step}"), expected
        )
        for step, expected in EXPECTED_MANIFEST_HASHES.items()
    }
    if not all(item["allPassed"] for item in upstream.values()):
        raise AssertionError("upstream immutability failed before S04 freeze")
    s03_runs = pq.read_table(
        S03_DIR / "composition_sweep.parquet",
        columns=["scenario_id", "condition_id", "protected", "split"],
    ).to_pandas()
    if len(s03_runs) != EXPECTED_RUNS or s03_runs.scenario_id.nunique() != EXPECTED_RUNS:
        raise AssertionError("S03 population accounting mismatch")
    if s03_runs.condition_id.nunique() != EXPECTED_CONDITIONS:
        raise AssertionError("S03 condition accounting mismatch")
    if s03_runs.protected.any() or set(s03_runs.split) != {"exploratory"}:
        raise PermissionError("S04 population is not the unprotected S03 exploratory set")
    write_json(output / "preregistration.json", contract)
    (output / "metric_specification.md").write_text(
        _metric_specification_markdown(contract_hash), encoding="utf-8"
    )
    record = {
        "schema": "e04.s04.freeze_record.v1",
        "researchStepId": "S04",
        "frozenAtUtc": contract["frozenAtUtc"],
        "frozenBeforePopulationAnalysis": True,
        "contractPath": str(CONTRACT_PATH),
        "contractSha256": contract_hash,
        "scenarioCount": len(s03_runs),
        "conditionCount": int(s03_runs.condition_id.nunique()),
        "gridRowCount": EXPECTED_GRID_ROWS,
        "upstreamValidation": upstream,
        "s05AbsentAtFreeze": True,
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
        raise AssertionError("frozen S04 contract hash changed")
    if not record["frozenBeforePopulationAnalysis"]:
        raise AssertionError("S04 was not frozen before population analysis")
    if S05_DIR.exists():
        raise FileExistsError("S05 output exists during S04")
    return record


def hand_built_validation() -> pd.DataFrame:
    fixtures = (
        ("homogeneous", tuple("AAAAA"), (1, 2, 3, 4, 5)),
        ("segregated", tuple("AAABBB"), (1, 2, 3, 4, 5, 6)),
        ("alternating", tuple("ABABAB"), (1, 2, 3, 4, 5, 6)),
        ("three_label_blocks", tuple("AABBCC"), (1, 2, 3, 4, 5, 6)),
        ("endpoint_sensitive", tuple("ABAABB"), (1, 2, 3, 4, 5, 6)),
        ("duplicate_conditioned", tuple("AABAAB"), (1, 1, 1, 2, 2, 2)),
    )
    rows = []
    for name, labels, values in fixtures:
        rows.append(
            {
                "pattern_id": name,
                "labels": "".join(labels),
                "values_json": json.dumps(values, separators=(",", ":")),
                **aggregation_metrics(labels, values),
            }
        )
    frame = pd.DataFrame(rows)
    expected = {
        "homogeneous": {
            "same_label_edge_count": 4.0,
            "publication_aggregation": 0.8,
            "run_count": 1.0,
            "largest_cluster_size": 5.0,
        },
        "segregated": {
            "same_label_edge_count": 4.0,
            "run_count": 2.0,
            "largest_cluster_size": 3.0,
        },
        "alternating": {
            "same_label_edge_count": 0.0,
            "categorical_assortativity": -1.0,
            "run_count": 6.0,
            "mean_run_length": 1.0,
            "largest_cluster_size": 1.0,
            "neighbor_nmi": 1.0,
            "signed_neighbor_nmi": -1.0,
        },
        "three_label_blocks": {
            "same_label_edge_count": 3.0,
            "run_count": 3.0,
            "largest_cluster_size": 2.0,
        },
        "duplicate_conditioned": {
            "equal_value_edge_count": 4.0,
            "equal_value_same_label_edge_count": 2.0,
            "equal_value_same_label_rate": 0.5,
            "equal_value_conditioned_baseline": 1 / 3,
            "equal_value_corrected_excess": 1 / 6,
        },
    }
    for name, checks in expected.items():
        row = frame.loc[frame.pattern_id == name].iloc[0]
        for key, target in checks.items():
            if not math.isclose(float(row[key]), target, abs_tol=1e-12):
                raise AssertionError(f"fixture {name}/{key}: {row[key]} != {target}")
    homogeneous = frame.loc[frame.pattern_id == "homogeneous"].iloc[0]
    if not (
        math.isnan(homogeneous.categorical_assortativity)
        and math.isnan(homogeneous.neighbor_nmi)
    ):
        raise AssertionError("homogeneous undefined convention failed")
    return frame


def exhaustive_validation() -> tuple[pd.DataFrame, dict[str, Any]]:
    grouped: dict[tuple[int, int, tuple[int, ...]], dict[str, Any]] = {}
    total = 0
    for n in range(2, EXHAUSTIVE_MAX_N + 1):
        for k in range(2, min(EXHAUSTIVE_MAX_K, n) + 1):
            for labels in itertools.product(range(k), repeat=n):
                counts = tuple(labels.count(label) for label in range(k))
                if 0 in counts:
                    continue
                metrics = aggregation_metrics(
                    tuple(str(label) for label in labels), tuple(range(n))
                )
                if metrics["run_count"] != n - metrics["same_label_edge_count"]:
                    raise AssertionError("open-path run/adjacency identity failed")
                if not -1 - 1e-12 <= metrics["categorical_assortativity"] <= 1 + 1e-12:
                    raise AssertionError("assortativity escaped its bounds")
                if not 0 <= metrics["neighbor_nmi"] <= 1 + 1e-12:
                    raise AssertionError("neighbor NMI escaped its bounds")
                key = (n, k, counts)
                record = grouped.setdefault(
                    key,
                    {
                        "arrangement_count": 0,
                        "sum_corrected_adjacency": 0.0,
                        "sum_assortativity": 0.0,
                        "sum_largest_capture": 0.0,
                        "sum_signed_nmi": 0.0,
                        "min_assortativity": math.inf,
                        "max_assortativity": -math.inf,
                        "min_largest_capture": math.inf,
                        "max_largest_capture": -math.inf,
                        "min_signed_nmi": math.inf,
                        "max_signed_nmi": -math.inf,
                    },
                )
                record["arrangement_count"] += 1
                record["sum_corrected_adjacency"] += metrics[
                    "corrected_publication_aggregation"
                ]
                for metric, short in (
                    ("categorical_assortativity", "assortativity"),
                    ("largest_label_capture", "largest_capture"),
                    ("signed_neighbor_nmi", "signed_nmi"),
                ):
                    value = metrics[metric]
                    record[f"sum_{short}"] += value
                    record[f"min_{short}"] = min(record[f"min_{short}"], value)
                    record[f"max_{short}"] = max(record[f"max_{short}"], value)
                total += 1
    rows = []
    for (n, k, counts), record in sorted(grouped.items()):
        arrangements = record["arrangement_count"]
        multinomial = math.factorial(n)
        for count in counts:
            multinomial //= math.factorial(count)
        rows.append(
            {
                "n": n,
                "k": k,
                "counts_json": json.dumps(counts, separators=(",", ":")),
                "arrangement_count": arrangements,
                "multinomial_count": multinomial,
                "mean_corrected_adjacency": record["sum_corrected_adjacency"]
                / arrangements,
                **{
                    f"mean_{short}": record[f"sum_{short}"] / arrangements
                    for short in ("assortativity", "largest_capture", "signed_nmi")
                },
                **{
                    f"{bound}_{short}": record[f"{bound}_{short}"]
                    for short in ("assortativity", "largest_capture", "signed_nmi")
                    for bound in ("min", "max")
                },
            }
        )
    frame = pd.DataFrame(rows)
    accounting = {
        "schema": "e04.s04.exhaustive_accounting.v1",
        "researchStepId": "S04",
        "maxN": EXHAUSTIVE_MAX_N,
        "maxK": EXHAUSTIVE_MAX_K,
        "compositionProfiles": len(frame),
        "surjectiveArrangementsEvaluated": total,
        "allMultinomialCountsMatch": bool(
            frame.arrangement_count.eq(frame.multinomial_count).all()
        ),
        "maxCorrectedAdjacencyAbsMean": float(
            frame.mean_corrected_adjacency.abs().max()
        ),
        "runIdentityChecked": True,
    }
    if len(frame) != 246 or total != 265_016:
        raise AssertionError(f"exhaustive accounting drift: {len(frame)}, {total}")
    if not accounting["allMultinomialCountsMatch"]:
        raise AssertionError("exhaustive multinomial accounting failed")
    if accounting["maxCorrectedAdjacencyAbsMean"] > 1e-15:
        raise AssertionError("exact corrected adjacency mean is not zero")
    return frame, accounting


GLOBAL_MC_PROFILES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("n6_balanced2", (3, 3)),
    ("n6_extreme2", (1, 5)),
    ("n6_balanced3", (2, 2, 2)),
    ("n10_balanced2", (5, 5)),
    ("n10_extreme2", (1, 9)),
    ("n10_near_balanced3", (4, 3, 3)),
    ("n20_balanced2", (10, 10)),
    ("n20_extreme2", (2, 18)),
    ("n20_near_balanced3", (7, 7, 6)),
    ("n100_balanced2", (50, 50)),
    ("n100_extreme2", (10, 90)),
    ("n100_near_balanced3", (34, 33, 33)),
)


def _seed(stream: str, identifier: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"E04/S04/{stream}/{identifier}".encode()).digest()[:16],
        "big",
    )


def _summarize_mc(
    profile_id: str,
    calibration_family: str,
    counts: Sequence[int],
    samples: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for metric, values in samples.items():
        valid = values[np.isfinite(values)]
        mean = float(valid.mean()) if len(valid) else math.nan
        sd = float(valid.std(ddof=1)) if len(valid) > 1 else math.nan
        se = sd / math.sqrt(len(valid)) if len(valid) > 1 else math.nan
        tolerance = max(5 * se, 0.001) if math.isfinite(se) else math.nan
        calibrated_zero = metric in {
            "corrected_publication_aggregation",
            "equal_value_corrected_excess",
        }
        rows.append(
            {
                "calibration_family": calibration_family,
                "profile_id": profile_id,
                "n": sum(counts),
                "k": len(counts),
                "counts_json": json.dumps(tuple(counts), separators=(",", ":")),
                "draws": len(values),
                "metric": metric,
                "valid_draws": len(valid),
                "undefined_rate": 1 - len(valid) / len(values),
                "mean": mean,
                "sd": sd,
                "se": se,
                "q025": float(np.quantile(valid, 0.025)) if len(valid) else math.nan,
                "median": float(np.median(valid)) if len(valid) else math.nan,
                "q975": float(np.quantile(valid, 0.975)) if len(valid) else math.nan,
                "zero_calibration_expected": calibrated_zero,
                "zero_calibration_tolerance": tolerance,
                "zero_calibration_passed": (
                    abs(mean) <= tolerance if calibrated_zero and len(valid) else None
                ),
            }
        )
    return rows


def _global_mc_worker(profile: tuple[str, tuple[int, ...]]) -> list[dict[str, Any]]:
    profile_id, counts = profile
    labels = np.concatenate(
        [np.full(count, str(index), dtype="U2") for index, count in enumerate(counts)]
    )
    values = np.arange(len(labels))
    rng = np.random.Generator(np.random.PCG64DXSM(_seed("global_mc", profile_id)))
    tracked = (
        "corrected_publication_aggregation",
        "categorical_assortativity",
        "mean_run_length",
        "largest_cluster_fraction",
        "largest_label_capture",
        "neighbor_mi_bits",
        "neighbor_nmi",
        "signed_neighbor_nmi",
    )
    samples = {metric: np.empty(MONTE_CARLO_DRAWS) for metric in tracked}
    for draw in range(MONTE_CARLO_DRAWS):
        arrangement = labels[rng.permutation(len(labels))]
        metrics = aggregation_metrics(arrangement, values)
        for metric in tracked:
            samples[metric][draw] = metrics[metric]
    return _summarize_mc(profile_id, "global_fixed_composition", counts, samples)


def _duplicate_profile(profile_id: str) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    parts = profile_id.split("_")
    n = int(parts[0][1:])
    layout, allocation = parts[1], parts[2]
    strata = 4 if n == 20 else 10
    stratum_size = n // strata
    grouped_values = np.repeat(np.arange(strata), stratum_size)
    if layout == "grouped":
        values = grouped_values
    elif layout == "interleaved":
        values = np.tile(np.arange(strata), stratum_size)
    else:
        raise ValueError(layout)
    labels = np.empty(n, dtype="U1")
    for value in range(strata):
        positions = np.flatnonzero(values == value)
        if allocation == "balanced":
            first = len(positions) // 2
        elif allocation == "heterogeneous":
            first = 1 if value % 2 == 0 else len(positions) - 1
        else:
            raise ValueError(allocation)
        labels[positions[:first]] = "A"
        labels[positions[first:]] = "B"
    counts = (int(np.sum(labels == "A")), int(np.sum(labels == "B")))
    return labels, values, counts


DUPLICATE_MC_PROFILES = tuple(
    f"n{n}_{layout}_{allocation}"
    for n in (20, 100)
    for layout in ("grouped", "interleaved")
    for allocation in ("balanced", "heterogeneous")
)


def _duplicate_mc_worker(profile_id: str) -> list[dict[str, Any]]:
    labels, values, counts = _duplicate_profile(profile_id)
    rng = np.random.Generator(np.random.PCG64DXSM(_seed("duplicate_mc", profile_id)))
    tracked = (
        "equal_value_same_label_rate",
        "equal_value_conditioned_baseline",
        "equal_value_corrected_excess",
    )
    samples = {metric: np.empty(MONTE_CARLO_DRAWS) for metric in tracked}
    value_positions = [np.flatnonzero(values == value) for value in np.unique(values)]
    for draw in range(MONTE_CARLO_DRAWS):
        arrangement = labels.copy()
        for positions in value_positions:
            arrangement[positions] = arrangement[positions][
                rng.permutation(len(positions))
            ]
        metrics = aggregation_metrics(arrangement, values)
        for metric in tracked:
            samples[metric][draw] = metrics[metric]
    return _summarize_mc(
        profile_id, "within_value_fixed_composition", counts, samples
    )


def run_validation(output: Path, *, workers: int = 8) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1,8]")
    fixtures = hand_built_validation()
    exhaustive, accounting = exhaustive_validation()
    _write_parquet(fixtures, output / "hand_built_patterns.parquet")
    _write_parquet(exhaustive, output / "exhaustive_small_n_metrics.parquet")
    write_json(output / "exhaustive_accounting.json", accounting)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_global_mc_worker, profile) for profile in GLOBAL_MC_PROFILES]
        futures.extend(
            executor.submit(_duplicate_mc_worker, profile)
            for profile in DUPLICATE_MC_PROFILES
        )
        for future in futures:
            rows.extend(future.result())
    calibration = pd.DataFrame(rows).sort_values(
        ["calibration_family", "profile_id", "metric"]
    )
    _write_parquet(calibration, output / "random_label_calibration.parquet")
    expected = calibration.loc[calibration.zero_calibration_expected]
    estimable_expected = expected.loc[expected.valid_draws > 0]
    passed = bool(estimable_expected.zero_calibration_passed.astype(bool).all())
    summary = {
        "schema": "e04.s04.static_validation_summary.v1",
        "researchStepId": "S04",
        "handBuiltPatterns": len(fixtures),
        "exhaustiveProfiles": len(exhaustive),
        "exhaustiveArrangements": accounting["surjectiveArrangementsEvaluated"],
        "monteCarloProfiles": len(GLOBAL_MC_PROFILES) + len(DUPLICATE_MC_PROFILES),
        "monteCarloDrawsPerProfile": MONTE_CARLO_DRAWS,
        "totalMonteCarloArrangements": (
            len(GLOBAL_MC_PROFILES) + len(DUPLICATE_MC_PROFILES)
        )
        * MONTE_CARLO_DRAWS,
        "zeroCalibrationRows": len(expected),
        "estimableZeroCalibrationRows": len(estimable_expected),
        "structurallyUndefinedZeroCalibrationRows": int(
            (expected.valid_draws == 0).sum()
        ),
        "allZeroCalibrationsPassed": passed,
        "maxAdjacencyAbsoluteMean": float(
            estimable_expected.loc[
                estimable_expected.metric == "corrected_publication_aggregation", "mean"
            ].abs().max()
        ),
        "maxEqualValueAbsoluteMean": float(
            estimable_expected.loc[
                estimable_expected.metric == "equal_value_corrected_excess", "mean"
            ].abs().max()
        ),
    }
    if not passed:
        raise AssertionError("random-label zero calibration failed")
    write_json(output / "static_validation_summary.json", summary)
    return summary


def _s03_expected() -> dict[str, dict[str, Any]]:
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
    result = {str(row["scenario_id"]): row for row in rows}
    if len(result) != EXPECTED_RUNS:
        raise AssertionError("S03 expected-run table has wrong cardinality")
    return result


def _trajectory_targets(final_swaps: int) -> dict[int, list[int]]:
    targets: dict[int, list[int]] = defaultdict(list)
    for grid_index, progress in enumerate(GRID):
        targets[math.floor(progress * final_swaps)].append(grid_index)
    return dict(targets)


def _population_worker(
    task: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    condition = SweepCondition.from_dict(task["condition"])
    scenario, scenario_metadata = materialize_sweep_scenario(condition, task["base"])
    if scenario.scenario_id != task["scenarioId"]:
        raise AssertionError("S04 scenario rematerialization changed scenario ID")
    if scenario_metadata["scenarioJsonSha256"] != expected["scenario_json_sha256"]:
        raise AssertionError("S04 scenario JSON differs from S03")
    final_swaps = int(expected["successful_swap_count"])
    targets = _trajectory_targets(final_swaps)
    state = initial_state(scenario)
    state.terminal = evaluate_terminal(scenario, state)
    tracker = MetricTracker(scenario, False)
    admissibility = AdmissibilityTracker(scenario, state)
    original_points: list[dict[str, Any]] = [
        {"swap_index": 0, **tracker.metrics()}
    ]
    metric_rows: list[list[Any] | None] = [None] * len(GRID)

    def capture(swap_index: int) -> None:
        labels = [scenario.cell_map[cell_id].policy.value for cell_id in state.occupancy]
        values = [scenario.cell_map[cell_id].value for cell_id in state.occupancy]
        metrics = aggregation_metrics(labels, values)
        payload = [float(metrics[column]) for column in METRIC_COLUMNS]
        for grid_index in targets.get(swap_index, ()):  # duplicate floor targets
            metric_rows[grid_index] = [grid_index, swap_index, *payload]

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
    if any(row is None for row in metric_rows):
        missing = [index for index, row in enumerate(metric_rows) if row is None]
        raise AssertionError(f"S04 trajectory capture missing grid indices {missing}")

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
            f"S04/S03 replay mismatch for {scenario.scenario_id}: {replay_checks}"
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
        "schema": CHECKPOINT_SCHEMA,
        "scenario_id": scenario.scenario_id,
        "scenario_json_sha256": scenario_metadata["scenarioJsonSha256"],
        "metadata": metadata,
        "stop_reason": state.terminal,
        "successful_swap_count": int(state.ledger["acceptedSwaps"]),
        "activation_count": int(state.activation_count),
        "final_state_hash": observed["final_state_hash"],
        "original_trajectory_sha256": observed["trajectory_sha256"],
        "replay_checks": replay_checks,
        "elapsed_seconds": elapsed,
        "metric_columns": list(METRIC_COLUMNS),
        "metric_rows": metric_rows,
    }


def _read_checkpoint_ids(checkpoint: Path) -> set[str]:
    if not checkpoint.exists():
        return set()
    completed: set[str] = set()
    with checkpoint.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            scenario_id = str(record["scenario_id"])
            if scenario_id in completed:
                raise ValueError(f"duplicate checkpoint scenario at line {line_number}")
            if record["metric_columns"] != list(METRIC_COLUMNS):
                raise ValueError("checkpoint metric schema differs from frozen schema")
            completed.add(scenario_id)
    return completed


def run_population(
    tasks: Sequence[Mapping[str, Any]], checkpoint: Path, *, workers: int = 8
) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1,8]")
    expected = _s03_expected()
    task_ids = {str(task["scenarioId"]) for task in tasks}
    if task_ids != set(expected):
        raise AssertionError("rematerialized S03 task population differs from S03 output")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_checkpoint_ids(checkpoint)
    if not completed <= task_ids:
        raise ValueError("checkpoint contains scenario outside frozen S03 population")
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
                active[executor.submit(_population_worker, task, expected[scenario_id])] = task
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    task = active.pop(future)
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
                                    "pending": EXPECTED_RUNS - len(completed),
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
                            _population_worker, next_task, expected[scenario_id]
                        )
                    ] = next_task
    if len(completed) != EXPECTED_RUNS:
        raise AssertionError(f"S04 checkpoint incomplete: {len(completed)}")
    return {
        "scenarioCount": len(completed),
        "newScenarios": written,
        "gridRowCount": len(completed) * len(GRID),
        "elapsedSeconds": time.perf_counter() - started,
        "workers": workers,
        "checkpoint": str(checkpoint),
    }


def _checkpoint_records(checkpoint: Path) -> Iterable[dict[str, Any]]:
    with checkpoint.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _record_rows(record: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = record["metadata"]
    run_id = "e04s04:" + hashlib.sha256(
        f"{record['scenario_id']}|{sha256_file(CONTRACT_PATH)}".encode()
    ).hexdigest()
    trajectory_rows = []
    metric_arrays: dict[str, list[float]] = {metric: [] for metric in METRIC_COLUMNS}
    for packed in record["metric_rows"]:
        grid_index, swap_index, *values = packed
        metric_values = dict(zip(METRIC_COLUMNS, values))
        for metric, value in metric_values.items():
            metric_arrays[metric].append(float(value) if value is not None else math.nan)
        trajectory_rows.append(
            {
                "schema_version": OUTPUT_SCHEMA,
                "research_step_id": "S04",
                "run_id": run_id,
                "s03_run_id": metadata["run_id"],
                "scenario_id": record["scenario_id"],
                "scenario_json_sha256": record["scenario_json_sha256"],
                "condition_id": metadata["condition_id"],
                "input_profile": metadata["input_profile"],
                "policy_set_label": metadata["policy_set_label"],
                "composition_class": metadata["composition_class"],
                "composition_profile": metadata["composition_profile"],
                # A numeric NaN keeps the streamed schema stable when a
                # three-way batch contains no pairwise first-policy count.
                "first_policy_count": (
                    float(metadata["first_policy_count"])
                    if metadata["first_policy_count"] is not None
                    else math.nan
                ),
                # Keep a stable string schema in streamed Parquet batches;
                # pairwise runs use the empty string rather than a null-only
                # first batch that PyArrow would infer as the null type.
                "rare_policy": metadata["rare_policy"] or "",
                "correlation_profile": metadata["correlation_profile"],
                "base_draw_id": metadata["base_draw_id"],
                "pairing_block_id": metadata["pairing_block_id"],
                "replicate_ordinal": metadata["replicate_ordinal"],
                "policy_counts_json": metadata["policy_counts_json"],
                "grid_index": grid_index,
                "accepted_swap_progress": GRID[grid_index],
                "swap_index": swap_index,
                **metric_values,
            }
        )
    summary = {
        "schema_version": "e04.s04.metric_run_summary.v1",
        "research_step_id": "S04",
        "run_id": run_id,
        "s03_run_id": metadata["run_id"],
        "scenario_id": record["scenario_id"],
        "scenario_json_sha256": record["scenario_json_sha256"],
        **{key: metadata[key] for key in metadata if key != "run_id"},
        "stop_reason": record["stop_reason"],
        "successful_swap_count": record["successful_swap_count"],
        "activation_count": record["activation_count"],
        "final_state_hash": record["final_state_hash"],
        "original_trajectory_sha256": record["original_trajectory_sha256"],
        "all_s03_replay_checks_passed": all(record["replay_checks"].values()),
        "elapsed_seconds": record["elapsed_seconds"],
    }
    for metric, values in metric_arrays.items():
        array = np.asarray(values, dtype=float)
        summary[f"initial_{metric}"] = array[0]
        summary[f"final_{metric}"] = array[-1]
        summary[f"auc_{metric}"] = (
            float(np.trapezoid(array, x=GRID)) if np.isfinite(array).all() else math.nan
        )
    return trajectory_rows, summary


def build_metric_tables(checkpoint: Path, output: Path) -> dict[str, Any]:
    if _read_checkpoint_ids(checkpoint).__len__() != EXPECTED_RUNS:
        raise AssertionError("cannot build S04 artifacts from incomplete checkpoint")
    trajectory_path = output / "aggregation_metrics.parquet"
    writer: pq.ParquetWriter | None = None
    batch: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    trajectory_count = 0
    for record in _checkpoint_records(checkpoint):
        rows, summary = _record_rows(record)
        batch.extend(rows)
        summaries.append(summary)
        if len(batch) >= 25_000:
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(
                    trajectory_path, table.schema, compression="zstd"
                )
            else:
                # Null-only columns are inferred as Arrow null in batches such
                # as unique-value runs; cast them to the established nullable
                # numeric schema before appending.
                table = table.cast(writer.schema)
            writer.write_table(table)
            trajectory_count += len(batch)
            batch = []
    if batch:
        table = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(trajectory_path, table.schema, compression="zstd")
        else:
            table = table.cast(writer.schema)
        writer.write_table(table)
        trajectory_count += len(batch)
    if writer is not None:
        writer.close()
    if trajectory_count != EXPECTED_GRID_ROWS or len(summaries) != EXPECTED_RUNS:
        raise AssertionError(
            f"S04 table accounting mismatch: {trajectory_count}, {len(summaries)}"
        )
    summary_frame = pd.DataFrame(summaries).sort_values(
        ["condition_id", "replicate_ordinal"]
    )
    _write_parquet(summary_frame, output / "metric_run_summary.parquet")
    return {
        "scenarioCount": len(summary_frame),
        "trajectoryRowCount": trajectory_count,
        "allS03ReplayChecksPassed": bool(
            summary_frame.all_s03_replay_checks_passed.all()
        ),
    }


def _paired_bootstrap(
    differences: np.ndarray, identifier: str, family_size: int
) -> tuple[float, float, float, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan, math.nan, math.nan, math.nan
    rng = np.random.Generator(np.random.PCG64DXSM(_seed("bootstrap", identifier)))
    estimates = np.empty(BOOTSTRAP_DRAWS)
    for start in range(0, BOOTSTRAP_DRAWS, 500):
        stop = min(start + 500, BOOTSTRAP_DRAWS)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[indices].mean(axis=1)
    alpha = 0.05 / family_size
    return (
        float(values.mean()),
        float(np.quantile(estimates, alpha / 2)),
        float(np.quantile(estimates, 1 - alpha / 2)),
        float(values.std(ddof=1)),
    )


def _condition_subset(
    runs: pd.DataFrame,
    *,
    input_profile: str,
    policy_set_label: str,
    composition_profile: str,
    correlation_profile: str,
) -> pd.DataFrame:
    return runs.loc[
        runs.input_profile.eq(input_profile)
        & runs.policy_set_label.eq(policy_set_label)
        & runs.composition_profile.eq(composition_profile)
        & runs.correlation_profile.eq(correlation_profile)
    ]


def _paired_values(
    target: pd.DataFrame, reference: pd.DataFrame, column: str
) -> np.ndarray:
    keys = ["pairing_block_id", "replicate_ordinal"]
    merged = target[keys + [column]].merge(
        reference[keys + [column]], on=keys, suffixes=("_target", "_reference")
    )
    if len(merged) != 250:
        raise AssertionError(f"paired contrast has {len(merged)} rather than 250 rows")
    return (
        merged[f"{column}_target"].to_numpy(float)
        - merged[f"{column}_reference"].to_numpy(float)
    )


def _association_contrasts(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ("corrected_publication_aggregation", *INDEPENDENT_METRICS)
    input_profiles = sorted(runs.input_profile.unique())
    policy_sets = (
        ("Bubble+Insertion", "p50_50"),
        ("Bubble+Selection", "p50_50"),
        ("Insertion+Selection", "p50_50"),
        ("Bubble+Insertion+Selection", "balanced_rotating"),
    )
    for input_profile in input_profiles:
        for policy_set, composition in policy_sets:
            reference = _condition_subset(
                runs,
                input_profile=input_profile,
                policy_set_label=policy_set,
                composition_profile=composition,
                correlation_profile="absent",
            )
            for target_profile in ("positive", "negative"):
                target = _condition_subset(
                    runs,
                    input_profile=input_profile,
                    policy_set_label=policy_set,
                    composition_profile=composition,
                    correlation_profile=target_profile,
                )
                contrast_id = (
                    f"association/{input_profile}/{policy_set}/{composition}/"
                    f"{target_profile}-absent"
                )
                for metric in metrics:
                    column = f"final_{metric}"
                    differences = _paired_values(target, reference, column)
                    estimate, low, high, sd = _paired_bootstrap(
                        differences, f"{contrast_id}/{metric}", 16
                    )
                    rows.append(
                        {
                            "contrast_family": "association",
                            "contrast_id": contrast_id,
                            "input_profile": input_profile,
                            "policy_set_label": policy_set,
                            "composition_profile": composition,
                            "target_profile": target_profile,
                            "reference_profile": "absent",
                            "estimand": "final_target_minus_absent",
                            "metric": metric,
                            "paired_runs": len(differences),
                            "mean_difference": estimate,
                            "paired_sd": sd,
                            "bonferroni_ci95_low": low,
                            "bonferroni_ci95_high": high,
                            "adjusted_positive": bool(low > 0),
                            "independent_directional_metric": metric
                            in INDEPENDENT_METRICS,
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.contrast_id.nunique() != 16 or len(frame) != 64:
        raise AssertionError("association anchor accounting mismatch")
    return frame


def _composition_contrasts(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ("corrected_publication_aggregation", *INDEPENDENT_METRICS)
    for input_profile in sorted(runs.input_profile.unique()):
        for policy_set in (
            "Bubble+Insertion",
            "Bubble+Selection",
            "Insertion+Selection",
        ):
            reference = _condition_subset(
                runs,
                input_profile=input_profile,
                policy_set_label=policy_set,
                composition_profile="p50_50",
                correlation_profile="absent",
            )
            for extreme in ("p10_90", "p90_10"):
                target = _condition_subset(
                    runs,
                    input_profile=input_profile,
                    policy_set_label=policy_set,
                    composition_profile=extreme,
                    correlation_profile="absent",
                )
                contrast_id = (
                    f"composition/{input_profile}/{policy_set}/{extreme}-p50_50"
                )
                for metric in metrics:
                    column = f"auc_{metric}"
                    differences = _paired_values(target, reference, column)
                    estimate, low, high, sd = _paired_bootstrap(
                        differences, f"{contrast_id}/{metric}", 12
                    )
                    rows.append(
                        {
                            "contrast_family": "composition",
                            "contrast_id": contrast_id,
                            "input_profile": input_profile,
                            "policy_set_label": policy_set,
                            "composition_profile": extreme,
                            "target_profile": extreme,
                            "reference_profile": "p50_50",
                            "estimand": "auc_extreme_minus_balanced",
                            "metric": metric,
                            "paired_runs": len(differences),
                            "mean_difference": estimate,
                            "paired_sd": sd,
                            "bonferroni_ci95_low": low,
                            "bonferroni_ci95_high": high,
                            "adjusted_positive": bool(low > 0),
                            "adjusted_negative": bool(high < 0),
                            "independent_directional_metric": metric
                            in INDEPENDENT_METRICS,
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.contrast_id.nunique() != 12 or len(frame) != 48:
        raise AssertionError("composition audit accounting mismatch")
    return frame


def _condition_summary(runs: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "condition_id",
        "input_profile",
        "policy_set_label",
        "composition_class",
        "composition_profile",
        "first_policy_count",
        "rare_policy",
        "correlation_profile",
    ]
    value_columns = [
        column
        for metric in METRIC_COLUMNS
        for column in (f"final_{metric}", f"auc_{metric}")
    ]
    summary = (
        runs.groupby(grouping, dropna=False)[value_columns]
        .mean()
        .add_prefix("mean_")
        .reset_index()
    )
    counts = runs.groupby(grouping, dropna=False).size().rename("runs").reset_index()
    summary = summary.merge(counts, on=grouping, validate="one_to_one")
    if len(summary) != EXPECTED_CONDITIONS or not summary.runs.eq(250).all():
        raise AssertionError("S04 condition summary accounting mismatch")
    return summary


CORRELATION_METRICS = (
    "corrected_publication_aggregation",
    "categorical_assortativity",
    "mean_run_length",
    "largest_cluster_fraction",
    "largest_label_capture",
    "neighbor_nmi",
    "signed_neighbor_nmi",
    "equal_value_corrected_excess",
)


def _correlation_tables(
    conditions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for estimand in ("final", "auc"):
        for first_index, first in enumerate(CORRELATION_METRICS):
            for second in CORRELATION_METRICS[first_index + 1 :]:
                first_col = f"mean_{estimand}_{first}"
                second_col = f"mean_{estimand}_{second}"
                valid = conditions[[first_col, second_col]].dropna()
                if len(valid) >= 3:
                    spearman = stats.spearmanr(valid[first_col], valid[second_col])
                    kendall = stats.kendalltau(valid[first_col], valid[second_col])
                    rho, rho_p = float(spearman.statistic), float(spearman.pvalue)
                    tau, tau_p = float(kendall.statistic), float(kendall.pvalue)
                else:
                    rho = rho_p = tau = tau_p = math.nan
                rows.append(
                    {
                        "estimand": estimand,
                        "metric_a": first,
                        "metric_b": second,
                        "conditions": len(valid),
                        "spearman_rho": rho,
                        "spearman_p": rho_p,
                        "kendall_tau": tau,
                        "kendall_p": tau_p,
                    }
                )
    long = pd.DataFrame(rows)
    final_values = conditions[
        [f"mean_final_{metric}" for metric in CORRELATION_METRICS]
    ].rename(columns=lambda value: value.removeprefix("mean_final_"))
    spearman_matrix = final_values.corr(method="spearman")

    disagreement_rows = []
    reference = conditions["mean_final_corrected_publication_aggregation"]
    reference_rank = reference.rank(pct=True, method="average")
    for metric in CORRELATION_METRICS[1:]:
        values = conditions[f"mean_final_{metric}"]
        metric_rank = values.rank(pct=True, method="average")
        sign_comparable = metric in {
            "categorical_assortativity",
            "signed_neighbor_nmi",
            "equal_value_corrected_excess",
        }
        for index, condition in conditions.iterrows():
            left, right = reference.iloc[index], values.iloc[index]
            disagreement_rows.append(
                {
                    "condition_id": condition.condition_id,
                    "input_profile": condition.input_profile,
                    "composition_profile": condition.composition_profile,
                    "correlation_profile": condition.correlation_profile,
                    "reference_metric": "corrected_publication_aggregation",
                    "alternate_metric": metric,
                    "reference_value": left,
                    "alternate_value": right,
                    "reference_rank_percentile": reference_rank.iloc[index],
                    "alternate_rank_percentile": metric_rank.iloc[index],
                    "absolute_rank_disagreement": abs(
                        reference_rank.iloc[index] - metric_rank.iloc[index]
                    ),
                    "rank_quartile_disagreement": (
                        math.floor(min(reference_rank.iloc[index], 0.999999) * 4)
                        != math.floor(min(metric_rank.iloc[index], 0.999999) * 4)
                        if math.isfinite(metric_rank.iloc[index])
                        else None
                    ),
                    "sign_comparable": sign_comparable,
                    "sign_disagreement": (
                        np.sign(left) != np.sign(right)
                        if sign_comparable and math.isfinite(right)
                        else None
                    ),
                }
            )
    disagreements = pd.DataFrame(disagreement_rows)
    return long, spearman_matrix, disagreements


def _response_surface(output: Path) -> pl.DataFrame:
    grouping = [
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
    ]
    result = (
        pl.scan_parquet(output / "aggregation_metrics.parquet")
        .group_by(grouping)
        .agg([pl.col(metric).mean().alias(f"mean_{metric}") for metric in METRIC_COLUMNS])
        .sort(["condition_id", "grid_index"])
        .collect(engine="streaming")
    )
    if result.height != EXPECTED_CONDITIONS * len(GRID):
        raise AssertionError("S04 response-surface accounting mismatch")
    result.write_parquet(output / "metric_response_surface.parquet", compression="zstd")
    return result


def _plot_correlation_matrix(matrix: pd.DataFrame, output: Path) -> None:
    labels = [value.replace("_", " ") for value in matrix.columns]
    fig, axis = plt.subplots(figsize=(9.2, 7.8))
    image = axis.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(labels)), labels, rotation=48, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = matrix.iloc[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}" if math.isfinite(value) else "NA",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if math.isfinite(value) and abs(value) > 0.55 else "black",
            )
    axis.set_title("S04 final condition-mean Spearman correlations")
    fig.colorbar(image, ax=axis, label="Spearman rho")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"metric_correlation_matrix.{suffix}", dpi=220)
    plt.close(fig)


def _plot_disagreement(conditions: pd.DataFrame, output: Path) -> None:
    reference = conditions["mean_final_corrected_publication_aggregation"]
    plot_metrics = (
        "categorical_assortativity",
        "largest_label_capture",
        "signed_neighbor_nmi",
        "equal_value_corrected_excess",
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
    for axis, metric in zip(axes.flat, plot_metrics):
        alternate = conditions[f"mean_final_{metric}"]
        for input_profile, color, marker in (
            ("unique_1_100", "#386cb0", "o"),
            ("repeated_1_10_x10", "#f0027f", "s"),
        ):
            mask = conditions.input_profile.eq(input_profile) & alternate.notna()
            axis.scatter(
                reference[mask],
                alternate[mask],
                s=20,
                alpha=0.72,
                color=color,
                marker=marker,
                label=input_profile if axis is axes.flat[0] else None,
            )
        axis.axvline(0, color="0.6", linewidth=0.8)
        if metric in {
            "categorical_assortativity",
            "signed_neighbor_nmi",
            "equal_value_corrected_excess",
        }:
            axis.axhline(0, color="0.6", linewidth=0.8)
        axis.set_xlabel("corrected original adjacency (final mean)")
        axis.set_ylabel(metric.replace("_", " "))
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("S04 metric agreement and deliberate disagreements")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"metric_disagreement_atlas.{suffix}", dpi=220)
    plt.close(fig)


def _plot_duplicate_conditioning(conditions: pd.DataFrame, output: Path) -> None:
    frame = conditions.loc[conditions.input_profile.eq("repeated_1_10_x10")].copy()
    grouped = (
        frame.groupby("correlation_profile")
        .agg(
            original=("mean_final_corrected_publication_aggregation", "mean"),
            equal_value=("mean_final_equal_value_corrected_excess", "mean"),
        )
        .reindex(["absent", "positive", "negative"])
    )
    x = np.arange(len(grouped))
    width = 0.34
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    axis.bar(x - width / 2, grouped.original, width, label="global corrected adjacency")
    axis.bar(x + width / 2, grouped.equal_value, width, label="equal-value corrected")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, grouped.index)
    axis.set_ylabel("mean final condition-level metric")
    axis.set_title("Repeated inputs: conditioning changes the estimand")
    axis.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"duplicate_conditioning.{suffix}", dpi=220)
    plt.close(fig)


def analyze_metrics(output: Path) -> dict[str, Any]:
    runs = pd.read_parquet(output / "metric_run_summary.parquet")
    if len(runs) != EXPECTED_RUNS:
        raise AssertionError("S04 run summary is incomplete")
    conditions = _condition_summary(runs)
    _write_parquet(conditions, output / "metric_condition_summary.parquet")
    response = _response_surface(output)
    association = _association_contrasts(runs)
    composition = _composition_contrasts(runs)
    contrasts = pd.concat([association, composition], ignore_index=True)
    contrasts.to_csv(output / "metric_contrasts.csv", index=False)

    correlations, matrix, disagreements = _correlation_tables(conditions)
    correlations.to_csv(output / "metric_correlations.csv", index=False)
    matrix.to_csv(output / "metric_correlation_matrix.csv")
    disagreements.to_csv(output / "metric_disagreement_analysis.csv", index=False)

    independent = association.loc[association.independent_directional_metric]
    anchor_counts = (
        independent.groupby("contrast_id").adjusted_positive.sum().astype(int)
    )
    association_passes = int((anchor_counts >= 2).sum())
    rank_rows = correlations.loc[
        correlations.estimand.eq("final")
        & correlations.metric_a.eq("corrected_publication_aggregation")
        & correlations.metric_b.isin(INDEPENDENT_METRICS)
    ].copy()
    rank_passes = int((rank_rows.spearman_rho >= 0.70).sum())
    static = json.loads((output / "static_validation_summary.json").read_text())
    execution_pass = bool(runs.all_s03_replay_checks_passed.all())
    calibration_pass = bool(static["allZeroCalibrationsPassed"])
    association_pass = association_passes >= 12
    rank_pass = rank_passes >= 2
    supportive = execution_pass and calibration_pass and association_pass and rank_pass
    outcome = "supportive" if supportive else "constraining/contradictory"
    criteria = {
        "schema": "e04.s04.success_criteria.v1",
        "researchStepId": "S04",
        "execution": {
            "passed": execution_pass,
            "scenarioCount": len(runs),
            "trajectoryRowCount": EXPECTED_GRID_ROWS,
        },
        "calibration": {
            "passed": calibration_pass,
            "estimableRows": static["estimableZeroCalibrationRows"],
            "structurallyUndefinedRows": static[
                "structurallyUndefinedZeroCalibrationRows"
            ],
        },
        "associationRobustness": {
            "passed": association_pass,
            "requiredAnchors": 12,
            "passingAnchors": association_passes,
            "totalAnchors": 16,
            "requiredIndependentMetricsPerAnchor": 2,
        },
        "rankRobustness": {
            "passed": rank_pass,
            "requiredMetrics": 2,
            "passingMetrics": rank_passes,
            "threshold": 0.70,
            "rows": rank_rows.to_dict("records"),
        },
        "allPassed": supportive,
        "outcomeClassification": outcome,
    }
    write_json(output / "success_criteria.json", criteria)

    sign_comparable = disagreements.loc[disagreements.sign_comparable]
    summary = {
        "schema": "e04.s04.analysis_summary.v1",
        "researchStepId": "S04",
        "outcomeClassification": outcome,
        "conditionCount": len(conditions),
        "responseSurfaceRows": response.height,
        "associationAnchorsPassing": association_passes,
        "associationAnchorsTotal": 16,
        "rankMetricsPassing": rank_passes,
        "rankMetricsTotal": 3,
        "medianAbsoluteRankDisagreement": {
            metric: float(group.absolute_rank_disagreement.median())
            for metric, group in disagreements.groupby("alternate_metric")
        },
        "rankQuartileDisagreementRate": {
            metric: float(group.rank_quartile_disagreement.dropna().mean())
            for metric, group in disagreements.groupby("alternate_metric")
        },
        "signDisagreementRateWhereComparable": {
            metric: float(group.sign_disagreement.dropna().astype(bool).mean())
            for metric, group in sign_comparable.groupby("alternate_metric")
        },
        "uniqueEqualValueDefinedConditionMeans": int(
            conditions.loc[
                conditions.input_profile.eq("unique_1_100"),
                "mean_final_equal_value_corrected_excess",
            ].notna().sum()
        ),
        "repeatedEqualValueDefinedConditionMeans": int(
            conditions.loc[
                conditions.input_profile.eq("repeated_1_10_x10"),
                "mean_final_equal_value_corrected_excess",
            ].notna().sum()
        ),
    }
    write_json(output / "analysis_summary.json", summary)
    _plot_correlation_matrix(matrix, output)
    _plot_disagreement(conditions, output)
    _plot_duplicate_conditioning(conditions, output)
    return summary


def validate_artifacts(output: Path) -> dict[str, Any]:
    required = (
        "preregistration.json",
        "freeze_record.json",
        "metric_specification.md",
        "hand_built_patterns.parquet",
        "exhaustive_small_n_metrics.parquet",
        "exhaustive_accounting.json",
        "random_label_calibration.parquet",
        "static_validation_summary.json",
        "aggregation_metrics.parquet",
        "metric_run_summary.parquet",
        "metric_condition_summary.parquet",
        "metric_response_surface.parquet",
        "metric_contrasts.csv",
        "metric_correlations.csv",
        "metric_correlation_matrix.csv",
        "metric_disagreement_analysis.csv",
        "success_criteria.json",
        "analysis_summary.json",
        "metric_correlation_matrix.png",
        "metric_correlation_matrix.svg",
        "metric_disagreement_atlas.png",
        "metric_disagreement_atlas.svg",
        "duplicate_conditioning.png",
        "duplicate_conditioning.svg",
    )
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "observed": observed})

    missing = [name for name in required if not (output / name).is_file()]
    add("required_artifacts_present", not missing, missing)
    if missing:
        raise FileNotFoundError(f"missing S04 artifacts: {missing}")
    frozen = assert_frozen(output)
    add("contract_frozen", True, frozen["contractSha256"])

    metrics_file = pq.ParquetFile(output / "aggregation_metrics.parquet")
    add(
        "aggregation_metric_row_count",
        metrics_file.metadata.num_rows == EXPECTED_GRID_ROWS,
        metrics_file.metadata.num_rows,
    )
    schema_names = set(metrics_file.schema_arrow.names)
    add("aggregation_metric_schema", set(METRIC_COLUMNS) <= schema_names, sorted(schema_names))
    runs = pd.read_parquet(output / "metric_run_summary.parquet")
    add("run_count", len(runs) == EXPECTED_RUNS, len(runs))
    add("unique_scenario_count", runs.scenario_id.nunique() == EXPECTED_RUNS, runs.scenario_id.nunique())
    add(
        "s03_replay_all_rows",
        bool(runs.all_s03_replay_checks_passed.all()),
        int(runs.all_s03_replay_checks_passed.sum()),
    )
    add("all_runs_complete", set(runs.stop_reason) == {"complete"}, sorted(runs.stop_reason.unique()))
    conditions = pd.read_parquet(output / "metric_condition_summary.parquet")
    add("condition_count", len(conditions) == EXPECTED_CONDITIONS, len(conditions))
    add("replication_count", bool(conditions.runs.eq(250).all()), sorted(conditions.runs.unique()))
    response_rows = pq.ParquetFile(output / "metric_response_surface.parquet").metadata.num_rows
    add(
        "response_surface_count",
        response_rows == EXPECTED_CONDITIONS * len(GRID),
        response_rows,
    )
    algebra = pl.scan_parquet(output / "aggregation_metrics.parquet").select(
        [
            (
                pl.col("run_count")
                - (100 - pl.col("same_label_edge_count"))
            )
            .abs()
            .max()
            .alias("max_run_identity_error"),
            (
                pl.col("publication_aggregation")
                - pl.col("same_label_edge_count") / 100
            )
            .abs()
            .max()
            .alias("max_paper_denominator_error"),
            (
                pl.col("edge_aggregation")
                - pl.col("same_label_edge_count") / 99
            )
            .abs()
            .max()
            .alias("max_edge_denominator_error"),
            (
                pl.col("corrected_publication_aggregation")
                - (
                    pl.col("publication_aggregation")
                    - pl.col("publication_composition_baseline")
                )
            )
            .abs()
            .max()
            .alias("max_correction_identity_error"),
        ]
    ).collect(engine="streaming").to_dicts()[0]
    for name, value in algebra.items():
        add(name, float(value) <= 1e-15, float(value))
    unique_conditional = pl.scan_parquet(output / "aggregation_metrics.parquet").filter(
        pl.col("input_profile") == "unique_1_100"
    ).select(
        pl.col("equal_value_corrected_excess").is_not_null().sum().alias("defined")
    ).collect(engine="streaming")["defined"][0]
    add("unique_equal_value_structurally_undefined", unique_conditional == 0, unique_conditional)
    repeated_final = conditions.loc[
        conditions.input_profile.eq("repeated_1_10_x10"),
        "mean_final_equal_value_corrected_excess",
    ]
    add("repeated_final_equal_value_estimable", bool(repeated_final.notna().all()), int(repeated_final.notna().sum()))

    static = json.loads((output / "static_validation_summary.json").read_text())
    exhaustive = json.loads((output / "exhaustive_accounting.json").read_text())
    criteria = json.loads((output / "success_criteria.json").read_text())
    add("static_calibration", static["allZeroCalibrationsPassed"], static)
    add(
        "exhaustive_accounting",
        exhaustive["surjectiveArrangementsEvaluated"] == 265_016
        and exhaustive["compositionProfiles"] == 246
        and exhaustive["allMultinomialCountsMatch"],
        exhaustive,
    )
    add("success_criteria_recorded", "outcomeClassification" in criteria, criteria)

    upstream = {
        step: _verify_upstream_manifest(
            Path(f"/artifacts/research_steps/{step}"), expected
        )
        for step, expected in EXPECTED_MANIFEST_HASHES.items()
    }
    upstream_passed = all(value["allPassed"] for value in upstream.values())
    add("upstream_immutability", upstream_passed, {key: value["allPassed"] for key, value in upstream.items()})
    add("s05_scope_boundary", not S05_DIR.exists(), str(S05_DIR))
    write_json(
        output / "upstream_immutability_audit.json",
        {
            "schema": "e04.s04.upstream_immutability_audit.v1",
            "researchStepId": "S04",
            "allPassed": upstream_passed,
            "steps": upstream,
        },
    )
    passed = sum(item["passed"] for item in checks)
    summary = {
        "schema": "e04.s04.validation_summary.v1",
        "researchStepId": "S04",
        "allPassed": passed == len(checks),
        "passedChecks": passed,
        "totalChecks": len(checks),
        "checks": checks,
    }
    write_json(output / "validation_summary.json", summary)
    if not summary["allPassed"]:
        failures = [item for item in checks if not item["passed"]]
        raise AssertionError(f"S04 validation failed: {failures}")
    return summary


def write_provenance(output: Path) -> dict[str, Any]:
    source_paths = (
        CONTRACT_PATH,
        REPOSITORY / "analysis/aggregation_metrics.py",
        REPOSITORY / "scripts/run_aggregation_metrics.py",
        REPOSITORY / "tests/test_e04_aggregation_metrics.py",
    )
    environment = {
        "schema": "e04.s04.environment.v1",
        "researchStepId": "S04",
        "capturedAtUtc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpuCount": os.cpu_count(),
        "workers": 8,
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "polars": pl.__version__,
            "scipy": stats.__version__ if hasattr(stats, "__version__") else None,
            "matplotlib": matplotlib.__version__,
        },
    }
    write_json(output / "environment.json", environment)
    commands = {
        "schema": "e04.s04.commands.v1",
        "researchStepId": "S04",
        "workingDirectory": str(REPOSITORY),
        "commands": [
            "python -m unittest tests.test_e04_aggregation_metrics -v",
            "python scripts/run_aggregation_metrics.py freeze",
            "python scripts/run_aggregation_metrics.py static-validation --workers 8",
            "python scripts/run_aggregation_metrics.py run --workers 8",
            "python scripts/run_aggregation_metrics.py analyze",
            "python scripts/run_aggregation_metrics.py validate",
            "python scripts/run_aggregation_metrics.py provenance",
            "python scripts/run_aggregation_metrics.py manifest",
        ],
        "checkpoint": "/cache/e04_s04/aggregation_metrics.jsonl",
    }
    write_json(output / "commands.json", commands)
    provenance = {
        "schema": "e04.s04.provenance.v1",
        "researchStepId": "S04",
        "capturedAtUtc": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPOSITORY),
        "branch": _git_output("branch", "--show-current"),
        "gitHeadAtCapture": _git_output("rev-parse", "HEAD"),
        "gitStatusShort": _git_output("status", "--short"),
        "contractSha256": sha256_file(CONTRACT_PATH),
        "sourceFiles": [
            {"path": str(path.relative_to(REPOSITORY)), "sha256": sha256_file(path)}
            for path in source_paths
        ],
        "upstreamManifestSha256": EXPECTED_MANIFEST_HASHES,
        "s03PopulationArtifact": {
            "path": str(S03_DIR / "composition_sweep.parquet"),
            "sha256": sha256_file(S03_DIR / "composition_sweep.parquet"),
        },
        "checkpoint": {
            "path": "/cache/e04_s04/aggregation_metrics.jsonl",
            "collectible": False,
            "purpose": "restartable per-run trajectory checkpoint",
        },
        "sourceFilesInRepositoryNotCopiedToArtifacts": True,
    }
    write_json(output / "provenance.json", provenance)
    return provenance


def write_artifact_manifest(output: Path) -> dict[str, Any]:
    paths = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    artifacts = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    manifest = {
        "schema": "e04.s04.artifact_manifest.v1",
        "researchStepId": "S04",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "artifactCount": len(artifacts),
        "artifacts": artifacts,
    }
    write_json(output / "artifact_manifest.json", manifest)
    return manifest
