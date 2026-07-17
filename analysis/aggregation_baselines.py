"""E04 S02 composition-conditioned aggregation baselines.

This module is deliberately read-only with respect to S01.  It derives exact
expectations for fixed label counts, validates them by exhaustive enumeration
and Monte Carlo permutation, and writes derived corrected trajectories without
changing any S01 value or file.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
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
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from reference_simulator.model import canonical_json_bytes


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
S01_DIR = Path("/artifacts/research_steps/S01")
UPSTREAM = Path("/previous-artifacts/E01")
OUTPUT_SCHEMA = "e04.s02.composition_corrected_aggregation.v1"
S01_MANIFEST_SHA256 = "df9860285fc5b1ed9f7442fb68ea5e9ed9e407918d90988d0b3592f4b3ef8079"
EXHAUSTIVE_MAX_N = 9
EXHAUSTIVE_MAX_K = 4
MONTE_CARLO_DRAWS = 250_000
BOOTSTRAP_DRAWS = 10_000

MONTE_CARLO_PROFILES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("n2_balanced_two", (1, 1)),
    ("n3_two_one", (2, 1)),
    ("n6_balanced_three", (2, 2, 2)),
    ("n10_balanced_two", (5, 5)),
    ("n10_extreme_two", (9, 1)),
    ("n10_near_balanced_three", (4, 3, 3)),
    ("n20_imbalanced_three", (14, 4, 2)),
    ("n100_balanced_two", (50, 50)),
    ("n100_ten_ninety", (90, 10)),
    ("n100_one_ninety_nine", (99, 1)),
    ("n100_rotating_three", (34, 33, 33)),
    ("n100_imbalanced_three", (70, 20, 10)),
    ("n100_balanced_four", (25, 25, 25, 25)),
    ("n100_imbalanced_four", (35, 25, 20, 20)),
    ("n100_s01_random_edge_three", (46, 32, 22)),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_native(value: Any) -> Any:
    """Convert NumPy/pandas scalar containers to canonical JSON primitives."""
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
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


def validate_counts(counts: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in counts)
    if not result or any(value <= 0 for value in result):
        raise ValueError("counts must be a nonempty sequence of positive integers")
    if sum(result) < 2:
        raise ValueError("aggregation requires n >= 2")
    return result


def exact_pair_probability(counts: Sequence[int]) -> float:
    """P(two distinct positions have the same label) under fixed counts."""
    values = validate_counts(counts)
    n = sum(values)
    return sum(value * (value - 1) for value in values) / (n * (n - 1))


def expected_linear_aggregation(counts: Sequence[int], denominator: int) -> float:
    """Expected linear same-label edges divided by an explicit denominator."""
    values = validate_counts(counts)
    n = sum(values)
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (n - 1) * exact_pair_probability(values) / denominator


def expected_paper_aggregation(counts: Sequence[int]) -> float:
    values = validate_counts(counts)
    return expected_linear_aggregation(values, sum(values))


def expected_edge_aggregation(counts: Sequence[int]) -> float:
    values = validate_counts(counts)
    return expected_linear_aggregation(values, sum(values) - 1)


def expected_cyclic_aggregation(counts: Sequence[int]) -> float:
    """Expected cyclic adjacency with n edges divided by n."""
    return exact_pair_probability(counts)


def with_replacement_pair_probability(counts: Sequence[int]) -> float:
    values = validate_counts(counts)
    n = sum(values)
    return sum((value / n) ** 2 for value in values)


def paper_with_replacement_approximation(counts: Sequence[int]) -> float:
    values = validate_counts(counts)
    n = sum(values)
    return (n - 1) * with_replacement_pair_probability(values) / n


def baseline_record(counts: Sequence[int]) -> dict[str, Any]:
    values = validate_counts(counts)
    n = sum(values)
    exact_pair = exact_pair_probability(values)
    paper = expected_paper_aggregation(values)
    replacement = with_replacement_pair_probability(values)
    return {
        "n": n,
        "k": len(values),
        "counts_json": json.dumps(list(values), separators=(",", ":")),
        "paper_denominator": n,
        "linear_edge_denominator": n - 1,
        "cyclic_edge_denominator": n,
        "paper_expectation": paper,
        "linear_edge_expectation": exact_pair,
        "cyclic_edge_expectation": exact_pair,
        "with_replacement_pair_probability": replacement,
        "paper_with_replacement_approximation": paper_with_replacement_approximation(
            values
        ),
        "paper_universal_half_bias": 0.5 - paper,
        "edge_with_replacement_minus_exact": replacement - exact_pair,
        "paper_with_replacement_minus_exact": paper_with_replacement_approximation(
            values
        )
        - paper,
        "paper_maximum": (n - 1) / n,
    }


def aggregation_numerators(labels: Sequence[int]) -> tuple[int, int]:
    if len(labels) < 2:
        raise ValueError("aggregation requires at least two labels")
    linear = sum(left == right for left, right in zip(labels, labels[1:]))
    cyclic = linear + int(labels[0] == labels[-1])
    return int(linear), int(cyclic)


def exhaustive_validation(
    max_n: int = EXHAUSTIVE_MAX_N, max_k: int = EXHAUSTIVE_MAX_K
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if max_n < 2 or max_k < 2:
        raise ValueError("exhaustive bounds require max_n,max_k >= 2")
    rows: list[dict[str, Any]] = []
    total_arrangements = 0
    total_profiles = 0
    for n in range(2, max_n + 1):
        for k in range(2, min(max_k, n) + 1):
            groups: dict[tuple[int, ...], list[int]] = {}
            for labels in itertools.product(range(k), repeat=n):
                counts = tuple(labels.count(label) for label in range(k))
                if 0 in counts:
                    continue
                linear, cyclic = aggregation_numerators(labels)
                record = groups.setdefault(counts, [0, 0, 0])
                record[0] += 1
                record[1] += linear
                record[2] += cyclic
            for counts, (arrangements, linear_sum, cyclic_sum) in sorted(
                groups.items()
            ):
                analytic = baseline_record(counts)
                mean_paper = linear_sum / (arrangements * n)
                mean_edge = linear_sum / (arrangements * (n - 1))
                mean_cyclic = cyclic_sum / (arrangements * n)
                multinomial = math.factorial(n)
                for count in counts:
                    multinomial //= math.factorial(count)
                rows.append(
                    {
                        **analytic,
                        "arrangement_count": arrangements,
                        "multinomial_count": multinomial,
                        "mean_paper_enumerated": mean_paper,
                        "mean_edge_enumerated": mean_edge,
                        "mean_cyclic_enumerated": mean_cyclic,
                        "paper_abs_error": abs(
                            mean_paper - analytic["paper_expectation"]
                        ),
                        "edge_abs_error": abs(
                            mean_edge - analytic["linear_edge_expectation"]
                        ),
                        "cyclic_abs_error": abs(
                            mean_cyclic - analytic["cyclic_edge_expectation"]
                        ),
                    }
                )
                total_arrangements += arrangements
                total_profiles += 1
    frame = (
        pd.DataFrame(rows).sort_values(["n", "k", "counts_json"]).reset_index(drop=True)
    )
    accounting = {
        "schema": "e04.s02.exhaustive_accounting.v1",
        "researchStepId": "S02",
        "maxN": max_n,
        "maxK": max_k,
        "compositionProfiles": total_profiles,
        "surjectiveArrangementsEvaluated": total_arrangements,
        "maxPaperAbsError": float(frame.paper_abs_error.max()),
        "maxEdgeAbsError": float(frame.edge_abs_error.max()),
        "maxCyclicAbsError": float(frame.cyclic_abs_error.max()),
        "allMultinomialCountsMatch": bool(
            frame.arrangement_count.eq(frame.multinomial_count).all()
        ),
    }
    return frame, accounting


def _mc_worker(profile: tuple[str, tuple[int, ...]]) -> dict[str, Any]:
    profile_id, counts = profile
    analytic = baseline_record(counts)
    n = analytic["n"]
    labels = np.concatenate(
        [np.full(count, label, dtype=np.int16) for label, count in enumerate(counts)]
    )
    seed = int.from_bytes(
        hashlib.sha256(f"E04/S02/monte-carlo/{profile_id}".encode()).digest()[:16],
        "big",
    )
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    sums = np.zeros(3, dtype=np.float64)
    sums2 = np.zeros(3, dtype=np.float64)
    processed = 0
    batch_size = 5_000
    while processed < MONTE_CARLO_DRAWS:
        batch = min(batch_size, MONTE_CARLO_DRAWS - processed)
        matrix = np.broadcast_to(labels, (batch, n)).copy()
        matrix = generator.permuted(matrix, axis=1)
        linear = (matrix[:, 1:] == matrix[:, :-1]).sum(axis=1, dtype=np.int32)
        cyclic = linear + (matrix[:, 0] == matrix[:, -1])
        metrics = np.column_stack((linear / n, linear / (n - 1), cyclic / n))
        sums += metrics.sum(axis=0)
        sums2 += np.square(metrics).sum(axis=0)
        processed += batch
    means = sums / processed
    variances = (sums2 - processed * np.square(means)) / (processed - 1)
    variances = np.maximum(variances, 0.0)
    ses = np.sqrt(variances / processed)
    expected = np.asarray(
        [
            analytic["paper_expectation"],
            analytic["linear_edge_expectation"],
            analytic["cyclic_edge_expectation"],
        ]
    )
    errors = means - expected
    tolerance = np.maximum(5 * ses, 0.001)
    passed = np.abs(errors) <= tolerance
    return {
        "profile_id": profile_id,
        **analytic,
        "draws": processed,
        "seed_uint128": str(seed),
        "mean_paper_monte_carlo": float(means[0]),
        "se_paper_monte_carlo": float(ses[0]),
        "error_paper": float(errors[0]),
        "tolerance_paper": float(tolerance[0]),
        "paper_passed": bool(passed[0]),
        "mean_edge_monte_carlo": float(means[1]),
        "se_edge_monte_carlo": float(ses[1]),
        "error_edge": float(errors[1]),
        "tolerance_edge": float(tolerance[1]),
        "edge_passed": bool(passed[1]),
        "mean_cyclic_monte_carlo": float(means[2]),
        "se_cyclic_monte_carlo": float(ses[2]),
        "error_cyclic": float(errors[2]),
        "tolerance_cyclic": float(tolerance[2]),
        "cyclic_passed": bool(passed[2]),
        "all_passed": bool(passed.all()),
    }


def monte_carlo_validation(workers: int = 8) -> pd.DataFrame:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1,8]")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(_mc_worker, MONTE_CARLO_PROFILES))
    return pd.DataFrame(rows).sort_values("profile_id").reset_index(drop=True)


def verify_s01_immutable() -> dict[str, Any]:
    manifest_path = S01_DIR / "artifact_manifest.json"
    manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    checks = []
    for artifact in manifest["artifacts"]:
        path = S01_DIR / artifact["path"]
        checks.append(
            {
                "path": artifact["path"],
                "exists": path.is_file(),
                "expectedSha256": artifact["sha256"],
                "observedSha256": sha256_file(path) if path.is_file() else None,
                "passed": path.is_file() and sha256_file(path) == artifact["sha256"],
            }
        )
    return {
        "schema": "e04.s02.s01_immutability_audit.v1",
        "researchStepId": "S02",
        "s01ManifestPath": str(manifest_path),
        "expectedManifestSha256": S01_MANIFEST_SHA256,
        "observedManifestSha256": manifest_hash,
        "manifestHashPassed": manifest_hash == S01_MANIFEST_SHA256,
        "artifactChecks": checks,
        "allPassed": manifest_hash == S01_MANIFEST_SHA256
        and all(item["passed"] for item in checks),
    }


def _counts_from_json(value: str) -> tuple[int, ...]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("S01 policy_counts_json must contain an object")
    return tuple(sorted((int(count) for count in parsed.values()), reverse=True))


def corrected_trajectories() -> tuple[pd.DataFrame, pd.DataFrame]:
    source_path = S01_DIR / "original_mixtures.parquet"
    source = pd.read_parquet(source_path)
    if len(source) != 161_600 or source.scenario_id.nunique() != 1_600:
        raise ValueError("validated S01 trajectory accounting changed")
    counts = source.policy_counts_json.map(_counts_from_json)
    unique_counts = sorted(set(counts), key=lambda item: (sum(item), len(item), item))
    records = {item: baseline_record(item) for item in unique_counts}
    paper_expected = counts.map(lambda item: records[item]["paper_expectation"])
    edge_expected = counts.map(lambda item: records[item]["linear_edge_expectation"])
    paper_max = counts.map(lambda item: records[item]["paper_maximum"])
    result = source.copy()
    result["schema_version"] = OUTPUT_SCHEMA
    result["research_step_id"] = "S02"
    result["s01_source_schema_version"] = source.schema_version
    result["counts_vector_json"] = counts.map(
        lambda item: json.dumps(list(item), separators=(",", ":"))
    )
    result["cell_count"] = counts.map(sum)
    result["paper_composition_expectation"] = paper_expected
    result["linear_edge_composition_expectation"] = edge_expected
    result["cyclic_composition_expectation"] = edge_expected
    result["with_replacement_pair_probability"] = counts.map(
        lambda item: records[item]["with_replacement_pair_probability"]
    )
    result["paper_with_replacement_approximation"] = counts.map(
        lambda item: records[item]["paper_with_replacement_approximation"]
    )
    result["paper_excess_over_composition"] = (
        result.publication_aggregation - paper_expected
    )
    result["linear_edge_excess_over_composition"] = (
        result.reference_aggregation - edge_expected
    )
    result["paper_excess_over_universal_half"] = result.publication_aggregation - 0.5
    result["universal_half_minus_composition_expectation"] = 0.5 - paper_expected
    result["paper_normalized_excess_to_max"] = result.paper_excess_over_composition / (
        paper_max - paper_expected
    )
    result["linear_edge_normalized_excess_to_max"] = (
        result.linear_edge_excess_over_composition / (1.0 - edge_expected)
    )
    baseline_table = (
        pd.DataFrame(records.values())
        .sort_values(["n", "k", "counts_json"])
        .reset_index(drop=True)
    )
    return result, baseline_table


def _bootstrap_corrected_peaks(trajectories: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition_id, group in trajectories.groupby("condition_id", sort=True):
        raw = (
            group.pivot(
                index="scenario_id",
                columns="grid_index",
                values="publication_aggregation",
            )
            .sort_index()
            .to_numpy(dtype=np.float64)
        )
        corrected = (
            group.pivot(
                index="scenario_id",
                columns="grid_index",
                values="paper_excess_over_composition",
            )
            .sort_index()
            .to_numpy(dtype=np.float64)
        )
        if raw.shape != (100, 101) or corrected.shape != (100, 101):
            raise ValueError(f"S02 peak matrix mismatch for {condition_id}")
        seed = int.from_bytes(
            hashlib.sha256(
                f"E04/S02/corrected-bootstrap/{condition_id}".encode()
            ).digest()[:16],
            "big",
        )
        generator = np.random.Generator(np.random.PCG64DXSM(seed))
        weights = generator.multinomial(100, [0.01] * 100, size=BOOTSTRAP_DRAWS)
        curves = weights @ corrected / 100.0
        observed_raw = raw.mean(axis=0)
        observed_corrected = corrected.mean(axis=0)
        peaks = curves.max(axis=1)
        locations = curves.argmax(axis=1)
        expected = float(group.paper_composition_expectation.mean())
        rows.append(
            {
                "condition_id": condition_id,
                "condition_label": group.condition_label.iloc[0],
                "input_profile": group.input_profile.iloc[0],
                "assignment_profile": group.assignment_profile.iloc[0],
                "runs": 100,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "mean_paper_composition_expectation": expected,
                "mean_raw_start": float(observed_raw[0]),
                "mean_corrected_start": float(observed_corrected[0]),
                "mean_raw_peak": float(observed_raw.max()),
                "mean_raw_peak_progress_percent": int(observed_raw.argmax()),
                "mean_corrected_peak": float(observed_corrected.max()),
                "mean_corrected_peak_progress_percent": int(
                    observed_corrected.argmax()
                ),
                "corrected_peak_ci95_low": float(np.quantile(peaks, 0.025)),
                "corrected_peak_ci95_high": float(np.quantile(peaks, 0.975)),
                "corrected_peak_progress_ci95_low": float(
                    np.quantile(locations, 0.025)
                ),
                "corrected_peak_progress_ci95_high": float(
                    np.quantile(locations, 0.975)
                ),
                "mean_raw_final": float(observed_raw[-1]),
                "mean_corrected_final": float(observed_corrected[-1]),
                "raw_peak_excess_over_half": float(observed_raw.max() - 0.5),
                "universal_half_understates_peak_excess_by": float(0.5 - expected),
            }
        )
    frame = pd.DataFrame(rows)
    frame["raw_peak_rank"] = (
        frame.groupby(["input_profile", "assignment_profile"])
        .mean_raw_peak.rank(method="min", ascending=False)
        .astype(int)
    )
    frame["corrected_peak_rank"] = (
        frame.groupby(["input_profile", "assignment_profile"])
        .mean_corrected_peak.rank(method="min", ascending=False)
        .astype(int)
    )
    frame["rank_change_after_correction"] = (
        frame.raw_peak_rank - frame.corrected_peak_rank
    )
    frame["raw_peak_above_half"] = frame.mean_raw_peak > 0.5
    frame["corrected_peak_positive"] = frame.mean_corrected_peak > 0
    frame["corrected_peak_ci_excludes_zero"] = frame.corrected_peak_ci95_low > 0
    frame["raw_final_near_half_0_02"] = (frame.mean_raw_final - 0.5).abs() <= 0.02
    frame["corrected_final_near_zero_0_02"] = frame.mean_corrected_final.abs() <= 0.02
    return frame.sort_values(
        ["input_profile", "assignment_profile", "condition_label"]
    ).reset_index(drop=True)


def _plot_corrected(
    trajectories: pd.DataFrame, output: Path, input_profile: str
) -> None:
    source = trajectories[
        trajectories.input_profile.eq(input_profile)
        & trajectories.assignment_profile.eq("balanced_exact")
    ]
    conditions = sorted(
        source.condition_id.unique(),
        key=lambda value: source[source.condition_id.eq(value)].condition_label.iloc[0],
    )
    if len(conditions) != 4:
        raise ValueError(f"expected four exact S01 conditions for {input_profile}")
    fig, axes = plt.subplots(
        2, 4, figsize=(17, 7), sharex=True, constrained_layout=True
    )
    x = np.arange(101)
    for column, condition_id in enumerate(conditions):
        group = source[source.condition_id.eq(condition_id)]
        label = group.condition_label.iloc[0]
        raw = group.groupby("grid_index").publication_aggregation.agg(["mean", "sem"])
        corrected = group.groupby("grid_index").paper_excess_over_composition.agg(
            ["mean", "sem"]
        )
        baseline = float(group.paper_composition_expectation.iloc[0])
        axes[0, column].plot(x, raw["mean"], color="#a63838", label="S01 mean")
        axes[0, column].fill_between(
            x,
            raw["mean"] - 1.96 * raw["sem"],
            raw["mean"] + 1.96 * raw["sem"],
            color="#a63838",
            alpha=0.16,
        )
        axes[0, column].axhline(
            baseline,
            color="#1e6b52",
            linestyle="--",
            label=f"composition baseline {baseline:.4f}",
        )
        axes[0, column].axhline(
            0.5, color="#444444", linestyle=":", label="universal 0.5"
        )
        axes[1, column].plot(x, corrected["mean"], color="#654a9b")
        axes[1, column].fill_between(
            x,
            corrected["mean"] - 1.96 * corrected["sem"],
            corrected["mean"] + 1.96 * corrected["sem"],
            color="#654a9b",
            alpha=0.16,
        )
        axes[1, column].axhline(0, color="#1e6b52", linestyle="--")
        axes[0, column].set_title(label)
        axes[1, column].set_xlabel("Accepted-swap progress (%)")
        for row in (0, 1):
            axes[row, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel("Publication aggregation (/n)")
    axes[1, 0].set_ylabel("Excess over exact composition null")
    axes[0, 0].legend(fontsize=7)
    title = (
        "Unique values 1–100"
        if input_profile == "unique_1_100"
        else "Repeated values 1–10 × 10"
    )
    fig.suptitle(f"E04 S02 composition-corrected trajectories — {title}", fontsize=14)
    stem = (
        "corrected_unique_trajectories"
        if input_profile == "unique_1_100"
        else "corrected_repeated_trajectories"
    )
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=180)
    plt.close(fig)


def write_exhaustive_artifacts(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    frame, accounting = exhaustive_validation()
    _write_parquet(frame, output / "exhaustive_small_n_validation.parquet")
    write_json(output / "exhaustive_accounting.json", accounting)
    fixture_labels = ((0, 0, 1), (0, 1, 0), (0, 1, 1))
    fixtures = {
        "schema": "e04.s02.baseline_fixtures.v1",
        "researchStepId": "S02",
        "analyticProfiles": {
            "balanced_two_n100": baseline_record((50, 50)),
            "balanced_three_n100": baseline_record((34, 33, 33)),
            "imbalanced_three_n100": baseline_record((70, 20, 10)),
            "balanced_four_n100": baseline_record((25, 25, 25, 25)),
        },
        "arrangementFixtures": [
            {
                "labels": list(labels),
                "linearNumerator": aggregation_numerators(labels)[0],
                "cyclicNumerator": aggregation_numerators(labels)[1],
                "paperAggregation": aggregation_numerators(labels)[0] / len(labels),
                "edgeAggregation": aggregation_numerators(labels)[0]
                / (len(labels) - 1),
                "cyclicAggregation": aggregation_numerators(labels)[1] / len(labels),
            }
            for labels in fixture_labels
        ],
    }
    write_json(output / "baseline_fixtures.json", fixtures)
    return accounting


def write_monte_carlo_artifacts(output: Path, workers: int = 8) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    frame = monte_carlo_validation(workers)
    _write_parquet(frame, output / "monte_carlo_calibration.parquet")
    value = {
        "schema": "e04.s02.monte_carlo_accounting.v1",
        "researchStepId": "S02",
        "profiles": len(frame),
        "drawsPerProfile": MONTE_CARLO_DRAWS,
        "totalPermutations": int(frame.draws.sum()),
        "workers": workers,
        "allPassed": bool(frame.all_passed.all()),
        "maxAbsoluteError": float(
            frame[["error_paper", "error_edge", "error_cyclic"]].abs().to_numpy().max()
        ),
    }
    write_json(output / "monte_carlo_accounting.json", value)
    return value


def write_corrected_artifacts(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    immutability = verify_s01_immutable()
    if not immutability["allPassed"]:
        raise AssertionError("S01 immutability audit failed before correction")
    trajectories, baselines = corrected_trajectories()
    summary = _bootstrap_corrected_peaks(trajectories)
    _write_parquet(trajectories, output / "corrected_trajectories.parquet")
    _write_parquet(baselines, output / "composition_baselines.parquet")
    summary.to_csv(output / "interpretation_changes.csv", index=False)
    write_json(output / "s01_immutability_audit.json", immutability)
    _plot_corrected(trajectories, output, "unique_1_100")
    _plot_corrected(trajectories, output, "repeated_1_10_x10")
    return {
        "trajectoryRows": len(trajectories),
        "runs": int(trajectories.scenario_id.nunique()),
        "conditions": int(trajectories.condition_id.nunique()),
        "distinctCountVectors": len(baselines),
    }


def validate_artifacts(output: Path) -> dict[str, Any]:
    exhaustive = pd.read_parquet(output / "exhaustive_small_n_validation.parquet")
    exhaustive_accounting = json.loads(
        (output / "exhaustive_accounting.json").read_text()
    )
    monte = pd.read_parquet(output / "monte_carlo_calibration.parquet")
    monte_accounting = json.loads((output / "monte_carlo_accounting.json").read_text())
    fixtures = json.loads((output / "baseline_fixtures.json").read_text())
    trajectories = pd.read_parquet(output / "corrected_trajectories.parquet")
    baselines = pd.read_parquet(output / "composition_baselines.parquet")
    changes = pd.read_csv(output / "interpretation_changes.csv")
    immutability = verify_s01_immutable()
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    check(
        "s01_immutability",
        immutability["allPassed"],
        [
            immutability["manifestHashPassed"],
            sum(not item["passed"] for item in immutability["artifactChecks"]),
        ],
    )
    check(
        "exhaustive_n_coverage",
        set(exhaustive.n) == set(range(2, EXHAUSTIVE_MAX_N + 1)),
        sorted(exhaustive.n.unique()),
    )
    check(
        "exhaustive_k_coverage",
        set(exhaustive.k) == {2, 3, 4},
        sorted(exhaustive.k.unique()),
    )
    check(
        "exhaustive_profile_accounting",
        len(exhaustive) == exhaustive_accounting["compositionProfiles"],
        len(exhaustive),
    )
    check(
        "exhaustive_arrangement_accounting",
        int(exhaustive.arrangement_count.sum())
        == exhaustive_accounting["surjectiveArrangementsEvaluated"],
        int(exhaustive.arrangement_count.sum()),
    )
    check(
        "exhaustive_multinomial_counts",
        exhaustive.arrangement_count.eq(exhaustive.multinomial_count).all(),
        int((exhaustive.arrangement_count != exhaustive.multinomial_count).sum()),
    )
    check(
        "exhaustive_paper_exact",
        exhaustive.paper_abs_error.le(1e-15).all(),
        float(exhaustive.paper_abs_error.max()),
    )
    check(
        "exhaustive_edge_exact",
        exhaustive.edge_abs_error.le(1e-15).all(),
        float(exhaustive.edge_abs_error.max()),
    )
    check(
        "exhaustive_cyclic_exact",
        exhaustive.cyclic_abs_error.le(1e-15).all(),
        float(exhaustive.cyclic_abs_error.max()),
    )
    check(
        "baseline_fixtures",
        math.isclose(
            fixtures["analyticProfiles"]["balanced_two_n100"]["paper_expectation"],
            0.49,
            abs_tol=1e-15,
        )
        and math.isclose(
            fixtures["analyticProfiles"]["balanced_three_n100"]["paper_expectation"],
            0.3234,
            abs_tol=1e-15,
        )
        and len(fixtures["arrangementFixtures"]) == 3,
        None,
    )
    check("monte_carlo_profiles", len(monte) == len(MONTE_CARLO_PROFILES), len(monte))
    check(
        "monte_carlo_draws",
        monte.draws.eq(MONTE_CARLO_DRAWS).all()
        and int(monte.draws.sum()) == monte_accounting["totalPermutations"],
        int(monte.draws.sum()),
    )
    check(
        "monte_carlo_calibration",
        monte.all_passed.all() and monte_accounting["allPassed"],
        int((~monte.all_passed).sum()),
    )
    check(
        "corrected_trajectory_accounting",
        len(trajectories) == 161_600
        and trajectories.scenario_id.nunique() == 1_600
        and trajectories.condition_id.nunique() == 16,
        [
            len(trajectories),
            trajectories.scenario_id.nunique(),
            trajectories.condition_id.nunique(),
        ],
    )
    check(
        "corrected_grid_complete",
        trajectories.groupby("scenario_id").size().eq(101).all(),
        None,
    )
    check(
        "s01_null_matches_derived",
        np.allclose(
            trajectories.publication_aggregation_null,
            trajectories.paper_composition_expectation,
            atol=1e-15,
        ),
        float(
            (
                trajectories.publication_aggregation_null
                - trajectories.paper_composition_expectation
            )
            .abs()
            .max()
        ),
    )
    denominator_factor = (trajectories.cell_count - 1) / trajectories.cell_count
    check(
        "paper_edge_expectation_relation",
        np.allclose(
            trajectories.paper_composition_expectation,
            trajectories.linear_edge_composition_expectation * denominator_factor,
            atol=1e-15,
        ),
        float(
            (
                trajectories.paper_composition_expectation
                - trajectories.linear_edge_composition_expectation * denominator_factor
            )
            .abs()
            .max()
        ),
    )
    check(
        "observed_denominator_relation",
        np.allclose(
            trajectories.publication_aggregation,
            trajectories.reference_aggregation * denominator_factor,
            atol=1e-15,
        ),
        float(
            (
                trajectories.publication_aggregation
                - trajectories.reference_aggregation * denominator_factor
            )
            .abs()
            .max()
        ),
    )
    check(
        "paper_correction_identity",
        np.allclose(
            trajectories.paper_excess_over_composition,
            trajectories.publication_aggregation
            - trajectories.paper_composition_expectation,
            atol=1e-15,
        ),
        None,
    )
    check(
        "edge_correction_identity",
        np.allclose(
            trajectories.linear_edge_excess_over_composition,
            trajectories.reference_aggregation
            - trajectories.linear_edge_composition_expectation,
            atol=1e-15,
        ),
        None,
    )
    check(
        "baseline_table_complete",
        len(baselines) == trajectories.counts_vector_json.nunique(),
        [len(baselines), trajectories.counts_vector_json.nunique()],
    )
    check(
        "balanced_pair_paper_baseline",
        math.isclose(expected_paper_aggregation((50, 50)), 0.49, abs_tol=1e-15),
        expected_paper_aggregation((50, 50)),
    )
    check(
        "rotating_three_paper_baseline",
        math.isclose(expected_paper_aggregation((34, 33, 33)), 0.3234, abs_tol=1e-15),
        expected_paper_aggregation((34, 33, 33)),
    )
    check(
        "interpretation_rows",
        len(changes) == 16 and changes.condition_id.nunique() == 16,
        [len(changes), changes.condition_id.nunique()],
    )
    check(
        "bootstrap_accounting",
        changes.runs.eq(100).all()
        and changes.bootstrap_draws.eq(BOOTSTRAP_DRAWS).all(),
        None,
    )
    check(
        "corrected_peak_ci_positive",
        changes.corrected_peak_ci_excludes_zero.all(),
        int((~changes.corrected_peak_ci_excludes_zero).sum()),
    )
    exact_three = changes[
        changes.assignment_profile.eq("balanced_exact")
        & changes.condition_label.eq("Bubble+Insertion+Selection")
    ]
    check(
        "three_way_rank_reversal",
        len(exact_three) == 2
        and (exact_three.corrected_peak_rank < exact_three.raw_peak_rank).all(),
        exact_three[["input_profile", "raw_peak_rank", "corrected_peak_rank"]].to_dict(
            "records"
        ),
    )
    figure_paths = [
        output / f"corrected_{profile}_trajectories.{suffix}"
        for profile in ("unique", "repeated")
        for suffix in ("png", "svg")
    ]
    check(
        "corrected_figures",
        all(path.is_file() and path.stat().st_size > 0 for path in figure_paths),
        [path.name for path in figure_paths],
    )

    failed = [name for name, value in checks.items() if not value["passed"]]
    value = {
        "schema": "e04.s02.validation_summary.v1",
        "researchStepId": "S02",
        "success": not failed,
        "checkCount": len(checks),
        "failedChecks": failed,
        "checks": checks,
    }
    write_json(output / "validation_summary.json", value)
    if failed:
        raise AssertionError(f"E04 S02 validation failed: {failed}")
    return value


def write_provenance(output: Path) -> dict[str, Any]:
    inputs = {
        "researchPlan": WORKSPACE / "RESEARCH_PLAN.md",
        "fullPlan": WORKSPACE / "FULL_PLAN.md",
        "agents": WORKSPACE / "AGENTS.md",
        "attachmentManifest": WORKSPACE / "input-attachments/MANIFEST.json",
        "attachmentSidecar": WORKSPACE
        / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md",
        "paperMarkdown": WORKSPACE
        / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/pdf-markdown.md",
        "previousArtifactsMarkdown": WORKSPACE / "PREVIOUS_ARTIFACTS.md",
        "previousArtifactsJson": WORKSPACE / "PREVIOUS_ARTIFACTS.json",
        "e01PaperMetricDefinitions": UPSTREAM
        / "research_steps/S01/paper_metric_definitions.md",
        "e01TransitionSpecification": UPSTREAM
        / "research_steps/S03/transition_spec.md",
        "e01S13Report": UPSTREAM / "research_steps/S13/research_step_full_results.md",
        "e01ReleaseManifest": UPSTREAM
        / "release/reference_simulator/release_manifest.json",
        "s01ArtifactManifest": S01_DIR / "artifact_manifest.json",
        "s01OriginalMixtures": S01_DIR / "original_mixtures.parquet",
        "s01RunSummary": S01_DIR / "run_summary.parquet",
        "s01FullReport": S01_DIR / "research_step_full_results.md",
    }
    missing = [name for name, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"provenance inputs missing: {missing}")
    value = {
        "schema": "e04.s02.provenance.v1",
        "researchStepId": "S02",
        "createdUtc": datetime.now(timezone.utc).isoformat(),
        "repositoryHeadAtPackaging": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "s01Modified": False,
        "s01ImmutabilityAudit": verify_s01_immutable(),
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
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
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    value = {
        "schema": "e04.s02.artifact_manifest.v1",
        "researchStepId": "S02",
        "artifactCount": len(files),
        "artifacts": files,
    }
    write_json(output / "artifact_manifest.json", value)
    return value
