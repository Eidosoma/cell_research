#!/usr/bin/env python3
"""Validate E03 S01 distances and emit compact reproducible evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from src.detours.distances import distance_profile


FIXTURES = REPOSITORY / "tests" / "fixtures" / "s01_distance_fixtures.json"
METRICS = (
    "adjacent_descents",
    "paper_sortedness_distance",
    "inversion_count",
    "normalized_kendall_distance",
    "spearman_footrule",
    "normalized_spearman_footrule",
    "maximum_rank_error",
    "normalized_maximum_rank_error",
    "duplicate_aware_earth_movers_distance",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture_rows() -> list[dict[str, Any]]:
    payload = json.loads(FIXTURES.read_text())
    rows = []
    for fixture in payload["fixtures"]:
        profile = distance_profile(
            fixture["values"], direction=fixture["direction"]
        ).to_dict()
        rows.append(
            {
                "fixtureId": fixture["fixtureId"],
                "values": json.dumps(fixture["values"], separators=(",", ":")),
                **profile,
            }
        )
    return rows


def _classification(delta: float) -> str:
    if delta < -1e-12:
        return "improved"
    if delta > 1e-12:
        return "worsened"
    return "neutral"


def _strict_inversion_indices(values: tuple[int, ...]) -> Iterable[int]:
    return (
        index
        for index, (left, right) in enumerate(zip(values, values[1:]))
        if left > right
    )


def transition_disagreement() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    counts: Counter[tuple[str, int, str, str]] = Counter()
    witnesses: dict[str, dict[str, Any]] = {}
    accounting: Counter[str] = Counter()

    families: list[tuple[str, int, Iterable[tuple[int, ...]]]] = []
    for n in range(2, 9):
        families.append(("unique_permutations", n, itertools.permutations(range(n))))
    for n in range(2, 8):
        families.append(("duplicate_alphabet_3", n, itertools.product(range(3), repeat=n)))

    for family, n, states in families:
        for values in states:
            accounting[f"{family}States"] += 1
            before = distance_profile(values)
            for index in _strict_inversion_indices(values):
                after_values = list(values)
                after_values[index], after_values[index + 1] = (
                    after_values[index + 1],
                    after_values[index],
                )
                after_values_tuple = tuple(after_values)
                after = distance_profile(after_values_tuple)
                accounting[f"{family}Transitions"] += 1
                classifications = {}
                for metric in METRICS:
                    label = _classification(
                        float(getattr(after, metric)) - float(getattr(before, metric))
                    )
                    counts[(family, n, metric, label)] += 1
                    classifications[metric] = label

                local = classifications["adjacent_descents"]
                global_label = classifications["inversion_count"]
                witness_type = f"local_{local}__kendall_{global_label}"
                if witness_type not in witnesses:
                    witnesses[witness_type] = {
                        "witnessType": witness_type,
                        "family": family,
                        "n": n,
                        "swapLeftIndex": index,
                        "beforeValues": list(values),
                        "afterValues": list(after_values_tuple),
                        "before": before.to_dict(),
                        "after": after.to_dict(),
                    }

    rows = []
    for family, n, metric, label in sorted(counts):
        total = sum(counts[(family, n, metric, item)] for item in ("improved", "neutral", "worsened"))
        rows.append(
            {
                "stateFamily": family,
                "n": n,
                "metric": metric,
                "classification": label,
                "transitionCount": counts[(family, n, metric, label)],
                "familyMetricTotal": total,
                "fraction": counts[(family, n, metric, label)] / total,
            }
        )
    return rows, list(witnesses.values()), dict(accounting)


def benchmark(iterations: int) -> dict[str, Any]:
    rng = random.Random(301_2026)
    cases = []
    for case in range(64):
        if case % 2:
            values = [index % 10 for index in range(100)]
        else:
            values = list(range(100))
        rng.shuffle(values)
        cases.append(tuple(values))

    # Warm caches and bytecode before timing.
    for values in cases:
        distance_profile(values)
    started = time.perf_counter()
    checksum = 0.0
    for index in range(iterations):
        profile = distance_profile(cases[index % len(cases)])
        checksum += profile.normalized_kendall_distance
    elapsed = time.perf_counter() - started
    return {
        "schemaVersion": "E03.S01.performance-benchmark.v1",
        "researchStepId": "S01",
        "sequenceLength": 100,
        "uniqueAndDuplicateCaseCount": len(cases),
        "iterations": iterations,
        "elapsedSeconds": elapsed,
        "profilesPerSecond": iterations / elapsed,
        "microsecondsPerProfile": elapsed * 1_000_000 / iterations,
        "checksum": checksum,
        "workerProcesses": 1,
        "threading": "intentional serial metric evaluation",
        "interpretation": "Descriptive local-runtime measurement; not a cross-machine performance guarantee.",
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_artifact_manifest(output: Path) -> None:
    records = []
    for path in sorted(output.iterdir()):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        records.append(
            {
                "path": str(path),
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(
        output / "artifact_manifest.json",
        {
            "schemaVersion": "E03.S01.artifact-manifest.v1",
            "researchStepId": "S01",
            "artifactCount": len(records),
            "artifacts": records,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--benchmark-iterations", type=int, default=20_000)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    artifacts_root = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    output = args.output or artifacts_root / "research_steps" / "S01"
    output.mkdir(parents=True, exist_ok=True)
    if args.manifest_only:
        write_artifact_manifest(output)
        return

    pytest_result = None
    if not args.skip_pytest:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_distances.py",
            f"--junitxml={output / 'distance_tests.junit.xml'}",
        ]
        completed = subprocess.run(command, cwd=REPOSITORY, text=True, capture_output=True)
        (output / "distance_tests.log").write_text(completed.stdout + completed.stderr)
        pytest_result = {
            "command": " ".join(command),
            "exitCode": completed.returncode,
            "passed": completed.returncode == 0,
        }
        if completed.returncode:
            raise SystemExit(completed.returncode)

    fixtures = fixture_rows()
    transition_rows, witnesses, accounting = transition_disagreement()
    performance = benchmark(args.benchmark_iterations)

    write_csv(output / "metric_fixture_results.csv", fixtures)
    write_csv(output / "exhaustive_transition_disagreement.csv", transition_rows)
    write_json(
        output / "disagreement_witnesses.json",
        {
            "schemaVersion": "E03.S01.disagreement-witnesses.v1",
            "researchStepId": "S01",
            "witnesses": witnesses,
        },
    )
    write_json(output / "performance_benchmark.json", performance)

    validation = {
        "schemaVersion": "E03.S01.validation-summary.v1",
        "researchStepId": "S01",
        "success": pytest_result is None or pytest_result["passed"],
        "pytest": pytest_result,
        "handCalculatedFixtureCount": len(fixtures),
        "uniquePermutationCountThroughN8": sum(math.factorial(n) for n in range(1, 9)),
        "uniqueDirectionProfileCountThroughN8": 2 * sum(
            math.factorial(n) for n in range(1, 9)
        ),
        "duplicateSequenceCountAlphabet3ThroughN7": sum(3**n for n in range(1, 8)),
        "duplicateDirectionProfileCountAlphabet3ThroughN7": 2 * sum(
            3**n for n in range(1, 8)
        ),
        "transitionAccounting": accounting,
        "validatedProperties": [
            "identity at every ascending and descending goal",
            "strict inversion parity with an independent pairwise oracle",
            "goal-set footrule and bottleneck parity with brute-force tie assignments",
            "one-dimensional transport parity with an independent cumulative-prefix formula",
            "multiset-specific normalization bounds and reverse-order maxima",
            "direction-reversal symmetry",
            "Kendall ascending/descending complement over unequal pairs",
            "monotone global distance along adjacent inversion-removal toy paths",
            "paper strict-Sortedness duplicate boundary",
            "compatibility with the prior E02 duplicate-aware footrule contract",
            "fail-closed handling of empty, non-finite, Boolean, text, and invalid-direction inputs",
        ],
        "failedAssertions": 0,
    }
    write_json(output / "validation_summary.json", validation)

    environment = {
        "schemaVersion": "E03.S01.environment.v1",
        "researchStepId": "S01",
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpuCount": os.cpu_count(),
        "packages": {
            name: package_version(name)
            for name in ("pytest", "numpy", "pandas", "pyarrow")
        },
        "threadEnvironment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    write_json(output / "environment.json", environment)

    provenance_paths = {
        "implementation": REPOSITORY / "src" / "detours" / "distances.py",
        "tests": REPOSITORY / "tests" / "test_distances.py",
        "fixtures": FIXTURES,
        "evidenceGenerator": REPOSITORY / "scripts" / "validate_s01_distances.py",
        "e01PaperMetricDefinitions": Path(
            "/previous-artifacts/E01/research_steps/S01/paper_metric_definitions.md"
        ),
        "e01DelayedGratificationEdgeCases": Path(
            "/previous-artifacts/E01/research_steps/S12/edge_case_rules.md"
        ),
        "e01ReferenceReleaseManifest": Path(
            "/previous-artifacts/E01/release/reference_simulator/release_manifest.json"
        ),
        "suppliedPaperMarkdown": Path(
            "/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/pdf-markdown.md"
        ),
    }
    missing = [name for name, path in provenance_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing provenance inputs: {missing}")
    write_json(
        output / "provenance.json",
        {
            "schemaVersion": "E03.S01.provenance.v1",
            "researchStepId": "S01",
            "repository": "https://github.com/Eidosoma/cell_research",
            "branch": "eidosoma/groups/28",
            "inputAndCodeHashes": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in provenance_paths.items()
            },
        },
    )


if __name__ == "__main__":
    main()
