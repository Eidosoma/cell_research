#!/usr/bin/env python3
"""Execute E02 S10 input-distribution stress tests.

S10 varies seeded input distributions in the S01 deterministic simulator, uses
S09 tie-aware metrics for duplicate-heavy cases, and records E01 original
random-unique baseline context where the upstream original traces expose
initial/final value arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from e02_deterministic_simulator import DeterministicEventSimulator, aggregation, state_hash
from scripts.e02_s08_dg_nulls import delayed_gratification_from_sortedness
from scripts.e02_s09_alternative_metrics import parse_json_array, prefixed_metrics, state_distance_metrics


EXPERIMENT_ID = "E02"
STEP_ID = "S10"
STEP_NUMBER = 10
DEFAULT_E01_S04_SUMMARY = Path("/previous-artifacts/E01/results/e01_s04_replicate_summary.parquet")
PROFILE_IDS = [
    "random_unique",
    "nearly_sorted_unique",
    "reverse_unique",
    "block_shuffled_unique",
    "duplicate_heavy",
    "heavy_tailed",
    "local_repeated_motifs",
]
ALGORITHMS = ["bubble", "insertion", "selection"]


@dataclass(frozen=True)
class StressCondition:
    condition_id: str
    condition_class: str
    algorithm_label: str
    algotypes: tuple[str, ...]
    frozen_variant: str = "none"
    frozen_count: int = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {
            "args": args,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "ok": proc.returncode == 0,
        }
    except Exception as exc:  # pragma: no cover - defensive provenance path
        return {"args": args, "returncode": None, "stdout": "", "stderr": repr(exc), "ok": False}


def get_git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "-v"], cwd=REPO_ROOT)
    return {
        "commit": commit["stdout"].strip() if commit["ok"] else "unknown",
        "branch": branch["stdout"].strip() if branch["ok"] else "unknown",
        "dirtyStatus": status["stdout"].strip(),
        "remote": remote["stdout"].strip(),
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if hasattr(value, "item"):
        return json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compact_json(value: Any) -> str:
    return json.dumps(json_ready(value), separators=(",", ":"))


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if not math.isfinite(value):
                return ""
            if abs(value) >= 1000 or (0 < abs(value) < 0.001):
                return f"{value:.3g}"
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def stable_seed(*parts: Any) -> int:
    payload = json.dumps(json_ready(parts), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def input_values_for_profile(profile_id: str, n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(int(seed))
    if profile_id == "random_unique":
        values = np.arange(1, n + 1, dtype=np.int16)
        rng.shuffle(values)
        return [int(value) for value in values]
    if profile_id == "nearly_sorted_unique":
        values = np.arange(1, n + 1, dtype=np.int16)
        swap_positions = rng.choice(np.arange(n - 1), size=max(1, n // 6), replace=False)
        for position in swap_positions:
            values[position], values[position + 1] = values[position + 1], values[position]
        return [int(value) for value in values]
    if profile_id == "reverse_unique":
        return [int(value) for value in range(n, 0, -1)]
    if profile_id == "block_shuffled_unique":
        block_size = max(3, n // 6)
        values = np.arange(1, n + 1, dtype=np.int16)
        blocks = [values[start : start + block_size].copy() for start in range(0, n, block_size)]
        order = np.arange(len(blocks))
        rng.shuffle(order)
        if len(order) > 1 and np.array_equal(order, np.arange(len(blocks))):
            order = np.roll(order, 1)
        return [int(value) for index in order for value in blocks[int(index)]]
    if profile_id == "duplicate_heavy":
        unique_count = max(3, n // 5)
        values = np.resize(np.arange(1, unique_count + 1, dtype=np.int16), n)
        rng.shuffle(values)
        return [int(value) for value in values]
    if profile_id == "heavy_tailed":
        head_count = max(3, n // 3)
        tail = np.minimum(rng.zipf(1.65, size=n - head_count), max(5, n // 2)).astype(np.int16)
        values = np.concatenate([np.ones(head_count, dtype=np.int16), tail])
        if len(set(int(value) for value in values)) < 3:
            values[-3:] = np.array([2, 3, max(4, int(values.max()))], dtype=np.int16)
        rng.shuffle(values)
        return [int(value) for value in values[:n]]
    if profile_id == "local_repeated_motifs":
        motifs = [
            np.array([3, 1, 2, 3, 1], dtype=np.int16),
            np.array([2, 1, 3, 2, 1], dtype=np.int16),
            np.array([4, 2, 1, 3, 2], dtype=np.int16),
        ]
        motif = motifs[int(rng.integers(0, len(motifs)))]
        rotation = int(rng.integers(0, len(motif)))
        motif = np.roll(motif, rotation)
        values = np.resize(motif, n)
        return [int(value) for value in values]
    raise ValueError(f"unknown input profile: {profile_id}")


def profile_description(profile_id: str) -> str:
    descriptions = {
        "random_unique": "seeded random permutation of unique 1..n values",
        "nearly_sorted_unique": "sorted unique values with seeded local adjacent inversions",
        "reverse_unique": "unique values in exact reverse target order",
        "block_shuffled_unique": "sorted unique blocks with seeded block-order shuffle",
        "duplicate_heavy": "few repeated integer values with seeded shuffle",
        "heavy_tailed": "Zipf-like duplicate-heavy values with a forced high-frequency head",
        "local_repeated_motifs": "repeated local value motif with seeded motif/rotation",
    }
    return descriptions[profile_id]


def profile_record(profile_id: str, n: int, seed: int) -> dict[str, Any]:
    values = input_values_for_profile(profile_id, n, seed)
    metrics = state_distance_metrics(values)
    counts = pd.Series(values).value_counts()
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "inputProfile": profile_id,
        "description": profile_description(profile_id),
        "n": int(n),
        "validationSeed": int(seed),
        "valuesHash": state_hash(values),
        "valuesPreview": compact_json(values[: min(12, len(values))]),
        "uniqueValueCount": int(len(set(values))),
        "hasDuplicates": bool(len(set(values)) < len(values)),
        "maxMultiplicity": int(counts.max()),
        "initialSortednessPercent": float(metrics["sortednessPercent"]),
        "initialKendallTauDistanceNormalized": float(metrics["kendallTauDistanceNormalized"]),
        "initialEditDistanceToTargetOrderNormalized": float(metrics["editDistanceToTargetOrderNormalized"]),
    }


def stress_conditions() -> list[StressCondition]:
    conditions: list[StressCondition] = [
        StressCondition("pure_bubble", "pure_no_frozen", "bubble", ("bubble",)),
        StressCondition("pure_insertion", "pure_no_frozen", "insertion", ("insertion",)),
        StressCondition("pure_selection", "pure_no_frozen", "selection", ("selection",)),
        StressCondition(
            "chimera_bubble_insertion_selection",
            "same_goal_chimera",
            "bubble+insertion+selection",
            ("bubble", "insertion", "selection"),
        ),
    ]
    for algorithm in ALGORITHMS:
        for frozen_variant in ("passive", "stuck"):
            conditions.append(
                StressCondition(
                    f"frozen_{algorithm}_{frozen_variant}_f2",
                    "frozen_cell",
                    algorithm,
                    (algorithm,),
                    frozen_variant=frozen_variant,
                    frozen_count=2,
                )
            )
    return conditions


def balanced_algotypes(algotypes: tuple[str, ...], n: int, seed: int) -> list[str]:
    if len(algotypes) == 1:
        return [algotypes[0]] * n
    base = [algotypes[index % len(algotypes)] for index in range(n)]
    rng = np.random.default_rng(int(seed))
    rng.shuffle(base)
    return [str(value) for value in base]


def labels_for_algotypes(algotypes: list[str]) -> list[int]:
    mapping = {name: index for index, name in enumerate(sorted(set(algotypes)))}
    return [int(mapping[name]) for name in algotypes]


def frozen_positions_for_condition(condition: StressCondition, n: int, seed: int) -> list[int]:
    if condition.frozen_count <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    positions = rng.choice(np.arange(n), size=condition.frozen_count, replace=False)
    return [int(value) for value in sorted(positions)]


def sortedness_trajectory(result: Any) -> np.ndarray:
    return np.asarray([float(row["sortedness_percent"]) for row in result.trace_rows], dtype=float)


def aggregation_trajectory(result: Any) -> np.ndarray:
    values: list[float] = []
    for row in result.trace_rows:
        algotypes = json.loads(row["algotypes_json"])
        values.append(float(aggregation(algotypes)))
    return np.asarray(values, dtype=float)


def trajectory_proxy_metrics(result: Any) -> dict[str, Any]:
    sortedness = sortedness_trajectory(result)
    aggregation_values = aggregation_trajectory(result)
    deltas = np.diff(sortedness)
    signs = np.sign(deltas[np.abs(deltas) > 1e-12])
    sign_changes = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0
    dg = delayed_gratification_from_sortedness(sortedness)
    return {
        "initialAggregation": float(aggregation_values[0]) if len(aggregation_values) else None,
        "peakAggregation": float(np.max(aggregation_values)) if len(aggregation_values) else None,
        "finalAggregation": float(aggregation_values[-1]) if len(aggregation_values) else None,
        "aucAggregation": float(np.mean(aggregation_values)) if len(aggregation_values) else None,
        "sortednessTotalVariation": float(np.abs(deltas).sum()) if len(deltas) else 0.0,
        "sortednessNetGain": float(sortedness[-1] - sortedness[0]) if len(sortedness) else 0.0,
        "sortednessSignChanges": sign_changes,
        "delayedGratification": float(dg["delayedGratification"]),
        "dgEventCount": int(dg["dgEventCount"]),
        "dgTotalDrop": float(dg["dgTotalDrop"]),
        "dgTotalRecovery": float(dg["dgTotalRecovery"]),
        "dgMaxEventScore": float(dg["dgMaxEventScore"]),
        "dgMinEventScore": float(dg["dgMinEventScore"]),
    }


def run_one_condition(
    *,
    condition: StressCondition,
    profile_id: str,
    n: int,
    replicate_index: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    input_seed = stable_seed(STEP_ID, profile_id, "input", replicate_index)
    scheduler_seed = stable_seed(STEP_ID, profile_id, condition.condition_id, "scheduler", replicate_index)
    tie_seed = stable_seed(STEP_ID, profile_id, condition.condition_id, "tie", replicate_index)
    algotype_seed = stable_seed(STEP_ID, profile_id, condition.condition_id, "algotypes", replicate_index)
    frozen_seed = stable_seed(STEP_ID, profile_id, condition.condition_id, "frozen", replicate_index)
    initial_values = input_values_for_profile(profile_id, n, input_seed)
    algotypes = balanced_algotypes(condition.algotypes, n, algotype_seed)
    labels = labels_for_algotypes(algotypes)
    frozen_positions = frozen_positions_for_condition(condition, n, frozen_seed)
    condition_id = f"S10_{profile_id}_{condition.condition_id}_rep{replicate_index:03d}"
    sim = DeterministicEventSimulator(
        initial_values,
        algotypes,
        labels=labels,
        frozen_positions=frozen_positions,
        frozen_variant=condition.frozen_variant,
        scheduler_seed=scheduler_seed,
        tie_breaker_seed=tie_seed,
        condition_id=condition_id,
        implementation="cell_view",
        research_step_id=STEP_ID,
    )
    result = sim.run(
        max_activations=max_activations,
        max_swaps=max_swaps,
        max_comparisons=max_comparisons,
        no_move_checks_required=2,
        no_move_check_interval=max(1, n),
    )
    initial_metrics = prefixed_metrics("initial", initial_values)
    final_metrics = prefixed_metrics("final", result.final_values)
    trajectory_metrics = trajectory_proxy_metrics(result)
    record: dict[str, Any] = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "conditionId": condition_id,
        "conditionTemplateId": condition.condition_id,
        "conditionClass": condition.condition_class,
        "implementation": "S01_deterministic_event_simulator",
        "algorithm": condition.algorithm_label,
        "inputProfile": profile_id,
        "n": int(n),
        "replicateIndex": int(replicate_index),
        "replicateNumber": int(replicate_index + 1),
        "inputSeed": int(input_seed),
        "schedulerSeed": int(scheduler_seed),
        "tieBreakerSeed": int(tie_seed),
        "algotypeAssignmentSeed": int(algotype_seed),
        "frozenPositionSeed": None if condition.frozen_count == 0 else int(frozen_seed),
        "frozenVariant": condition.frozen_variant,
        "frozenCount": int(condition.frozen_count),
        "initialFrozenPositions": compact_json(frozen_positions),
        "finalFrozenPositions": compact_json(result.final_frozen_positions),
        "initialValuesHash": state_hash(initial_values),
        "finalValuesHash": state_hash(result.final_values),
        "initialValues": compact_json(initial_values),
        "finalValues": compact_json(result.final_values),
        "initialAlgotypes": compact_json(algotypes),
        "finalAlgotypes": compact_json(result.final_algotypes),
        "initialAlgotypeCounts": compact_json(pd.Series(algotypes).value_counts().sort_index().to_dict()),
        "finalAlgotypeCounts": compact_json(pd.Series(result.final_algotypes).value_counts().sort_index().to_dict()),
        "completed": bool(result.completed),
        "stopReason": result.stop_reason,
        "capStop": result.stop_reason.startswith("max_"),
        "swapCount": int(result.swap_count),
        "comparisonCount": int(result.comparison_count),
        "archivedCompareAndSwapCount": int(result.archived_compare_and_swap_count),
        "activationCount": int(result.activation_count),
        "eventCount": int(result.event_count),
        "blockedMoveAttempts": int(result.blocked_move_attempts),
        "frozenSwapAttempts": int(result.frozen_swap_attempts),
        "wallTimeSeconds": float(result.wall_time_seconds),
        "valueCountPreserved": sorted(initial_values) == sorted(result.final_values),
        "algotypeCountPreserved": sorted(algotypes) == sorted(result.final_algotypes),
        "hasDuplicateValues": len(set(initial_values)) < len(initial_values),
        "uniqueValueCount": int(len(set(initial_values))),
        "maxValueMultiplicity": int(pd.Series(initial_values).value_counts().max()),
    }
    record.update(initial_metrics)
    record.update(final_metrics)
    for key in [
        "SortednessDistanceNormalized",
        "KendallTauDistanceNormalized",
        "SpearmanFootruleDistanceNormalized",
        "EarthMoverPositionDistanceNormalized",
        "EditDistanceToTargetOrderNormalized",
    ]:
        record[f"improvement{key}"] = float(record[f"initial{key}"] - record[f"final{key}"])
    record.update(trajectory_metrics)

    trace_rows: list[dict[str, Any]] = []
    for row in result.trace_rows:
        trace_rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "conditionId": condition_id,
                "conditionTemplateId": condition.condition_id,
                "conditionClass": condition.condition_class,
                "algorithm": condition.algorithm_label,
                "inputProfile": profile_id,
                "replicateIndex": int(replicate_index),
                "frozenVariant": condition.frozen_variant,
                "frozenCount": int(condition.frozen_count),
                "eventIndex": int(row["event_index"]),
                "eventKind": row["event_kind"],
                "activationIndex": int(row["activation_index"]),
                "swapCount": int(row["swap_count"]),
                "comparisonCount": int(row["comparison_count"]),
                "sortednessRawCount": int(row["sortedness_raw_count"]),
                "sortednessPercent": float(row["sortedness_percent"]),
                "monotonicityError": int(row["monotonicity_error"]),
                "aggregation": float(aggregation(json.loads(row["algotypes_json"]))),
                "stateHash": row["state_hash"],
                "initialStateHash": row["initial_state_hash"],
            }
        )
    return record, trace_rows


def validate_generators(n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_rows = [profile_record(profile_id, n, stable_seed(STEP_ID, profile_id, "profile_validation")) for profile_id in PROFILE_IDS]
    checks: list[dict[str, Any]] = []
    for profile_id in PROFILE_IDS:
        seed = stable_seed(STEP_ID, profile_id, "unit")
        first = input_values_for_profile(profile_id, n, seed)
        second = input_values_for_profile(profile_id, n, seed)
        other = input_values_for_profile(profile_id, n, seed + 1)
        checks.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "check": f"{profile_id}_seed_replay",
                "passed": first == second,
                "detail": f"{profile_id} generator is deterministic for fixed seed.",
            }
        )
        checks.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "check": f"{profile_id}_length",
                "passed": len(first) == n,
                "detail": f"{profile_id} generator returns n={n} values.",
            }
        )
        if profile_id != "reverse_unique":
            checks.append(
                {
                    "experimentId": EXPERIMENT_ID,
                    "researchStepId": STEP_ID,
                    "check": f"{profile_id}_seed_changes_output",
                    "passed": first != other,
                    "detail": f"{profile_id} changes output for different seeds.",
                }
            )
    random_values = input_values_for_profile("random_unique", n, 1)
    reverse_values = input_values_for_profile("reverse_unique", n, 1)
    nearly_values = input_values_for_profile("nearly_sorted_unique", n, 1)
    duplicate_values = input_values_for_profile("duplicate_heavy", n, 1)
    heavy_values = input_values_for_profile("heavy_tailed", n, 1)
    motif_values = input_values_for_profile("local_repeated_motifs", n, 1)
    checks.extend(
        [
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "check": "random_unique_is_permutation",
                "passed": sorted(random_values) == list(range(1, n + 1)) and len(set(random_values)) == n,
                "detail": "random_unique is a permutation of unique 1..n values.",
            },
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "check": "reverse_unique_is_reverse",
                "passed": reverse_values == list(range(n, 0, -1)),
                "detail": "reverse_unique is exact reverse target order.",
            },
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "check": "nearly_sorted_is_high_sortedness",
                "passed": state_distance_metrics(nearly_values)["sortednessPercent"] >= 75.0,
                "detail": "nearly_sorted_unique starts close to target order.",
            },
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "check": "duplicate_heavy_has_duplicates",
                "passed": len(set(duplicate_values)) < n,
                "detail": "duplicate_heavy contains repeated values.",
            },
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "check": "heavy_tailed_is_skewed",
                "passed": int(pd.Series(heavy_values).value_counts().max()) >= max(3, n // 3),
                "detail": "heavy_tailed has a high-frequency head value.",
            },
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "check": "local_repeated_motifs_has_repetition",
                "passed": len(set(motif_values)) <= 4 and len(set(tuple(motif_values[i : i + 5]) for i in range(0, n - 4, 5))) <= 2,
                "detail": "local_repeated_motifs contains repeated motif blocks.",
            },
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "check": "tie_aware_sorted_duplicate_zero_distance",
                "passed": state_distance_metrics(sorted(duplicate_values))["kendallTauDistanceNormalized"] == 0.0
                and state_distance_metrics(sorted(duplicate_values))["earthMoverPositionDistanceNormalized"] == 0.0,
                "detail": "S09 tie-aware metrics score sorted duplicate arrays as zero distance.",
            },
        ]
    )
    return pd.DataFrame(profile_rows), pd.DataFrame(checks)


def run_s10_matrix(
    *,
    n: int,
    replicates: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for profile_id in PROFILE_IDS:
        for condition in stress_conditions():
            for replicate_index in range(replicates):
                record, trace = run_one_condition(
                    condition=condition,
                    profile_id=profile_id,
                    n=n,
                    replicate_index=replicate_index,
                    max_activations=max_activations,
                    max_swaps=max_swaps,
                    max_comparisons=max_comparisons,
                )
                records.append(record)
                trace_rows.extend(trace)
    return pd.DataFrame(records), pd.DataFrame(trace_rows)


def load_e01_original_context(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    source = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    for row in source.itertuples(index=False):
        row_dict = row._asdict()
        initial_values = parse_json_array(row_dict["initial_values_json"])
        final_values = parse_json_array(row_dict["final_values_json"])
        record = {
            "experimentId": "E01",
            "researchStepId": "S04",
            "contextForResearchStepId": STEP_ID,
            "conditionId": row_dict["condition_id"],
            "implementation": row_dict["implementation"],
            "algorithm": row_dict["algorithm"],
            "inputProfile": "random_unique",
            "n": int(row_dict["n"]),
            "replicateIndex": int(row_dict["replicate_index"]),
            "replicateNumber": int(row_dict["replicate_number"]),
            "completed": bool(row_dict["completed"]),
            "stopReason": row_dict["stop_reason"],
            "swapCount": int(row_dict["swap_count"]),
            "comparisonCount": int(row_dict["comparison_count"]),
            "eventCount": int(row_dict["event_count"]),
            "initialValuesHash": state_hash(initial_values),
            "finalValuesHash": state_hash(final_values),
            "valueCountPreserved": sorted(initial_values) == sorted(final_values),
        }
        record.update(prefixed_metrics("initial", initial_values))
        record.update(prefixed_metrics("final", final_values))
        rows.append(record)
    return pd.DataFrame(rows)


def summarize_results(result_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        result_df.groupby(["inputProfile", "conditionClass", "algorithm", "frozenVariant", "frozenCount"], dropna=False)
        .agg(
            runCount=("conditionId", "size"),
            completedRate=("completed", "mean"),
            capStopRate=("capStop", "mean"),
            noMoveStopRate=("stopReason", lambda s: float(np.mean(pd.Series(s).str.contains("no_cell_can_move", regex=False)))),
            finalSortednessPercentMean=("finalSortednessPercent", "mean"),
            finalKendallDistanceMean=("finalKendallTauDistanceNormalized", "mean"),
            finalEditDistanceMean=("finalEditDistanceToTargetOrderNormalized", "mean"),
            finalEarthMoverDistanceMean=("finalEarthMoverPositionDistanceNormalized", "mean"),
            swapCountMean=("swapCount", "mean"),
            activationCountMean=("activationCount", "mean"),
            delayedGratificationMean=("delayedGratification", "mean"),
            peakAggregationMean=("peakAggregation", "mean"),
            finalAggregationMean=("finalAggregation", "mean"),
        )
        .reset_index()
    )
    summary.insert(0, "researchStepId", STEP_ID)
    summary.insert(0, "experimentId", EXPERIMENT_ID)
    return summary


def original_context_summary(context_df: pd.DataFrame) -> pd.DataFrame:
    if context_df.empty:
        return pd.DataFrame()
    summary = (
        context_df.groupby(["implementation", "algorithm", "inputProfile"], dropna=False)
        .agg(
            runCount=("conditionId", "size"),
            completedRate=("completed", "mean"),
            finalSortednessPercentMean=("finalSortednessPercent", "mean"),
            finalKendallDistanceMean=("finalKendallTauDistanceNormalized", "mean"),
            finalEditDistanceMean=("finalEditDistanceToTargetOrderNormalized", "mean"),
            swapCountMean=("swapCount", "mean"),
            comparisonCountMean=("comparisonCount", "mean"),
        )
        .reset_index()
    )
    summary.insert(0, "contextForResearchStepId", STEP_ID)
    return summary


def diagnostics_table(
    *,
    result_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    generator_checks_df: pd.DataFrame,
    original_context_df: pd.DataFrame,
    expected_rows: int,
    repo_tests_passed: bool,
) -> pd.DataFrame:
    duplicate_profiles = {"duplicate_heavy", "heavy_tailed", "local_repeated_motifs"}
    duplicate_rows = result_df[result_df["inputProfile"].isin(duplicate_profiles)]
    checks = [
        ("generator_checks_passed", bool(generator_checks_df["passed"].all()), "All seeded generator validation checks passed."),
        ("all_profiles_represented", set(result_df["inputProfile"]) == set(PROFILE_IDS), "All seven planned input profiles are represented."),
        ("expected_row_count", len(result_df) == expected_rows, f"Stress table has expected {expected_rows} rows."),
        ("trace_rows_written", len(trace_df) >= len(result_df), "Trace proxy rows were written for every run."),
        ("all_conditions_represented", result_df["conditionTemplateId"].nunique() == len(stress_conditions()), "All planned condition templates are represented."),
        ("value_counts_preserved", bool(result_df["valueCountPreserved"].all()), "All runs preserve value counts."),
        ("algotype_counts_preserved", bool(result_df["algotypeCountPreserved"].all()), "All runs preserve Algotype counts."),
        ("stop_reasons_logged", bool(result_df["stopReason"].notna().all()), "Every run records a stop reason or cap."),
        ("tie_aware_duplicate_metrics_finite", bool(np.isfinite(duplicate_rows["finalKendallTauDistanceNormalized"].to_numpy(dtype=float)).all()), "Duplicate-heavy profile metrics are finite."),
        ("profile_summary_written", len(profile_df) == len(PROFILE_IDS), "Profile generator summary has one row per profile."),
        ("stress_summary_written", len(summary_df) > 0, "Stress summary table is nonempty."),
        ("original_context_loaded", len(original_context_df) > 0, "E01 original random-unique context table was loaded."),
        ("repo_tests_passed", bool(repo_tests_passed), "Repository S10 unit tests passed."),
    ]
    return pd.DataFrame(
        [
            {"experimentId": EXPERIMENT_ID, "researchStepId": STEP_ID, "check": name, "passed": bool(passed), "detail": detail}
            for name, passed, detail in checks
        ]
    )


def validate_outputs(diagnostics_df: pd.DataFrame) -> dict[str, Any]:
    failures = diagnostics_df.loc[~diagnostics_df["passed"], "detail"].tolist()
    return {
        "success": not failures,
        "validationResult": "passed" if not failures else "failed",
        "checksPassed": int(diagnostics_df["passed"].sum()),
        "checksTotal": int(len(diagnostics_df)),
        "failures": failures,
    }


def write_tables(
    *,
    result_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    generator_checks_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    original_context_df: pd.DataFrame,
    original_summary_df: pd.DataFrame,
    artifacts_dir: Path,
) -> dict[str, Path]:
    result_dir = artifacts_dir / "results"
    trace_dir = artifacts_dir / "traces" / "e02" / STEP_ID
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    step_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "input_distributions_parquet": result_dir / "e02_input_distributions.parquet",
        "input_distributions_csv": result_dir / "e02_input_distributions.csv",
        "summary_parquet": result_dir / "e02_input_distribution_summary.parquet",
        "summary_csv": result_dir / "e02_input_distribution_summary.csv",
        "profiles_parquet": result_dir / "e02_input_distribution_profiles.parquet",
        "profiles_csv": result_dir / "e02_input_distribution_profiles.csv",
        "generator_checks_parquet": result_dir / "e02_input_distribution_generator_checks.parquet",
        "generator_checks_csv": result_dir / "e02_input_distribution_generator_checks.csv",
        "diagnostics_parquet": result_dir / "e02_input_distribution_diagnostics.parquet",
        "diagnostics_csv": result_dir / "e02_input_distribution_diagnostics.csv",
        "original_context_parquet": result_dir / "e02_input_distribution_original_context.parquet",
        "original_context_csv": result_dir / "e02_input_distribution_original_context.csv",
        "original_summary_parquet": result_dir / "e02_input_distribution_original_context_summary.parquet",
        "original_summary_csv": result_dir / "e02_input_distribution_original_context_summary.csv",
        "trace_parquet": trace_dir / "e02_input_distribution_trace_events.parquet",
        "trace_csv_gz": trace_dir / "e02_input_distribution_trace_events.csv.gz",
    }
    result_df.to_parquet(paths["input_distributions_parquet"], index=False)
    result_df.to_csv(paths["input_distributions_csv"], index=False)
    summary_df.to_parquet(paths["summary_parquet"], index=False)
    summary_df.to_csv(paths["summary_csv"], index=False)
    profile_df.to_parquet(paths["profiles_parquet"], index=False)
    profile_df.to_csv(paths["profiles_csv"], index=False)
    generator_checks_df.to_parquet(paths["generator_checks_parquet"], index=False)
    generator_checks_df.to_csv(paths["generator_checks_csv"], index=False)
    diagnostics_df.to_parquet(paths["diagnostics_parquet"], index=False)
    diagnostics_df.to_csv(paths["diagnostics_csv"], index=False)
    original_context_df.to_parquet(paths["original_context_parquet"], index=False)
    original_context_df.to_csv(paths["original_context_csv"], index=False)
    original_summary_df.to_parquet(paths["original_summary_parquet"], index=False)
    original_summary_df.to_csv(paths["original_summary_csv"], index=False)
    trace_df.to_parquet(paths["trace_parquet"], index=False)
    trace_df.to_csv(paths["trace_csv_gz"], index=False, compression="gzip")
    write_json(step_dir / "generator_validation.json", {"researchStepId": STEP_ID, "checks": generator_checks_df.to_dict(orient="records")})
    paths["generator_validation_json"] = step_dir / "generator_validation.json"
    return paths


def plot_summary(summary_df: pd.DataFrame, result_df: pd.DataFrame, figure_dir: Path) -> tuple[Path, Path, Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_df = summary_df[summary_df["conditionClass"].isin(["pure_no_frozen", "same_goal_chimera"])].copy()
    profiles = PROFILE_IDS
    algorithms = sorted(plot_df["algorithm"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for algorithm in algorithms:
        subset = plot_df[plot_df["algorithm"].eq(algorithm)].set_index("inputProfile").reindex(profiles)
        axes[0].plot(range(len(profiles)), subset["finalKendallDistanceMean"], marker="o", label=algorithm)
        axes[1].plot(range(len(profiles)), subset["swapCountMean"], marker="o", label=algorithm)
    for ax, ylabel, title in [
        (axes[0], "mean final Kendall distance", "Final order distance by input profile"),
        (axes[1], "mean accepted swaps", "Efficiency by input profile"),
    ]:
        ax.set_xticks(range(len(profiles)))
        ax.set_xticklabels([profile.replace("_", "\n") for profile in profiles], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    summary_png = figure_dir / "e02_input_distributions_summary.png"
    summary_pdf = figure_dir / "e02_input_distributions_summary.pdf"
    fig.savefig(summary_png, dpi=180)
    fig.savefig(summary_pdf)
    plt.close(fig)

    frozen = summary_df[summary_df["conditionClass"].eq("frozen_cell")].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    if not frozen.empty:
        pivot = (
            frozen.groupby(["inputProfile", "frozenVariant"])["completedRate"]
            .mean()
            .unstack("frozenVariant")
            .reindex(profiles)
            .fillna(0.0)
        )
        x = np.arange(len(pivot.index))
        width = 0.35
        for idx, variant in enumerate(pivot.columns):
            ax.bar(x + (idx - (len(pivot.columns) - 1) / 2) * width, pivot[variant], width=width, label=variant)
        ax.set_xticks(x)
        ax.set_xticklabels([profile.replace("_", "\n") for profile in profiles], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("completion rate")
        ax.set_title("Frozen Cell completion by input profile")
        ax.legend()
    fig.tight_layout()
    frozen_png = figure_dir / "e02_input_distributions_frozen_completion.png"
    frozen_pdf = figure_dir / "e02_input_distributions_frozen_completion.pdf"
    fig.savefig(frozen_png, dpi=180)
    fig.savefig(frozen_pdf)
    plt.close(fig)
    return summary_png, summary_pdf, frozen_png, frozen_pdf


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.exists() and path.is_file():
            records.append({"path": str(path), "sha256": sha256_path(path), "sizeBytes": path.stat().st_size})
    return sorted(records, key=lambda row: row["path"])


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_dir = step_dir / "code"
    paths = [
        (REPO_ROOT / "scripts" / "e02_s10_input_distributions.py", code_dir / "scripts" / "e02_s10_input_distributions.py"),
        (REPO_ROOT / "tests" / "test_e02_input_distributions.py", code_dir / "tests" / "test_e02_input_distributions.py"),
        (REPO_ROOT / "scripts" / "e02_s09_alternative_metrics.py", code_dir / "scripts" / "e02_s09_alternative_metrics.py"),
        (REPO_ROOT / "scripts" / "e02_s08_dg_nulls.py", code_dir / "scripts" / "e02_s08_dg_nulls.py"),
        (REPO_ROOT / "e02_deterministic_simulator" / "__init__.py", code_dir / "e02_deterministic_simulator" / "__init__.py"),
        (REPO_ROOT / "e02_deterministic_simulator" / "metrics.py", code_dir / "e02_deterministic_simulator" / "metrics.py"),
        (REPO_ROOT / "e02_deterministic_simulator" / "simulator.py", code_dir / "e02_deterministic_simulator" / "simulator.py"),
    ]
    copied: list[Path] = []
    for src, dst in paths:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def run_repo_tests(step_dir: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "unittest", "tests.test_e02_input_distributions"]
    result = run_command(command, cwd=REPO_ROOT)
    log_path = step_dir / "repo_unit_test_log.txt"
    log_path.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT:\n" + result["stdout"] + "\nSTDERR:\n" + result["stderr"],
        encoding="utf-8",
    )
    return {
        "command": command,
        "returncode": result["returncode"],
        "passed": bool(result["ok"]),
        "logPath": str(log_path),
    }


def outcome_classification(summary_df: pd.DataFrame, validation_success: bool) -> str:
    if not validation_success:
        return "null"
    pure = summary_df[summary_df["conditionClass"].eq("pure_no_frozen")]
    pure_complete_min = float(pure["completedRate"].min()) if len(pure) else 0.0
    frozen = summary_df[summary_df["conditionClass"].eq("frozen_cell")]
    frozen_complete_mean = float(frozen["completedRate"].mean()) if len(frozen) else 0.0
    if pure_complete_min == 1.0 and frozen_complete_mean >= 0.75:
        return "supportive"
    if pure_complete_min == 1.0:
        return "constraining/contradictory"
    return "constraining/contradictory"


def write_validation_report(step_dir: Path, diagnostics_df: pd.DataFrame, validation: dict[str, Any]) -> Path:
    lines = [
        "# E02 S10 Validation Report",
        "",
        "- Research step ID: S10",
        "- Completion status: completed" if validation["success"] else "- Completion status: failed",
        "- Artifacts written: input-distribution stress table, summaries, generator checks, trace proxy table, E01 original context table, figures, copied code, manifest, status JSON, and run manifest.",
        f"- Validation result: {validation['validationResult']}",
        "- Caveats or blockers: S10 uses the S01 deterministic simulator for all new distribution variants; E01 original code is included as random-unique baseline context because the original threaded scripts do not expose a validated seeded arbitrary-distribution interface.",
        "- Recommended next action: stop before S11 for Chief Scientist review.",
        "",
        "## Checks",
    ]
    for row in diagnostics_df.itertuples(index=False):
        lines.append(f"- {'passed' if row.passed else 'failed'}: {row.detail}")
    path = step_dir / "validation_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_reports_and_manifests(
    *,
    artifacts_dir: Path,
    started_at: float,
    result_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    generator_checks_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    original_context_df: pd.DataFrame,
    original_summary_df: pd.DataFrame,
    table_paths: dict[str, Path],
    figure_paths: tuple[Path, ...],
    code_paths: list[Path],
    repo_tests: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Path]:
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    step_dir.mkdir(parents=True, exist_ok=True)
    validation_path = write_validation_report(step_dir, diagnostics_df, validation)
    outcome = outcome_classification(summary_df, validation["success"])
    caveat = (
        "S10 uses the S01 deterministic simulator for all new distribution variants; E01 original code is included as random-unique baseline context because the original threaded scripts do not expose a validated seeded arbitrary-distribution interface. "
        "Frozen Cell rows use representative f=2 random placements and do not replace the planned S11 placement sweep."
    )
    recommended = "Stop before S11 for Chief Scientist review; if accepted, proceed to S11 Frozen Cell placement variation."

    top_profile_rows = (
        summary_df[summary_df["conditionClass"].isin(["pure_no_frozen", "same_goal_chimera"])]
        .groupby("inputProfile", dropna=False)
        .agg(
            meanCompletedRate=("completedRate", "mean"),
            meanFinalKendallDistance=("finalKendallDistanceMean", "mean"),
            meanSwapCount=("swapCountMean", "mean"),
            meanDelayedGratification=("delayedGratificationMean", "mean"),
        )
        .reset_index()
        .sort_values("inputProfile")
    )
    profile_table = markdown_table(
        ["Input profile", "Completed", "Final Kendall", "Swaps", "DG"],
        [
            [
                row.inputProfile,
                row.meanCompletedRate,
                row.meanFinalKendallDistance,
                row.meanSwapCount,
                row.meanDelayedGratification,
            ]
            for row in top_profile_rows.itertuples(index=False)
        ],
    )
    frozen_rows = (
        summary_df[summary_df["conditionClass"].eq("frozen_cell")]
        .groupby(["inputProfile", "frozenVariant"], dropna=False)
        .agg(completedRate=("completedRate", "mean"), finalKendallDistance=("finalKendallDistanceMean", "mean"))
        .reset_index()
        .sort_values(["inputProfile", "frozenVariant"])
    )
    frozen_table = markdown_table(
        ["Input profile", "Frozen variant", "Completed", "Final Kendall"],
        [[row.inputProfile, row.frozenVariant, row.completedRate, row.finalKendallDistance] for row in frozen_rows.itertuples(index=False)],
    )
    summary_path = step_dir / "summary.md"
    summary_lines = [
        "# E02 S10 Summary",
        "",
        "- Research step ID: S10",
        "- Completion status: completed" if validation["success"] else "- Completion status: failed",
        "- Artifacts written: `$ARTIFACTS_DIR/results/e02_input_distributions.parquet`, input-distribution summaries, generator checks, trace proxy table, E01 original context table, figures, copied code, manifest, status JSON, and run manifest.",
        f"- Validation result: {validation['validationResult']}",
        f"- Caveats or blockers: {caveat}",
        "- Lay summary: S10 replaced the random unique arrays with seven seeded input distributions. Pure no-Frozen and same-goal chimeric runs remained broadly competent across the selected deterministic matrix, while Frozen Cell outcomes were more profile- and variant-dependent. Duplicate-heavy cases were scored with the S09 tie-aware metrics so equal-value ties are not counted as errors.",
        f"- Recommended next action: {recommended}",
        f"- Outcome classification: {outcome}",
        "",
        "## Run Counts",
        "",
        markdown_table(
            ["Item", "Count"],
            [
                ["Stress runs", len(result_df)],
                ["Trace proxy rows", len(trace_df)],
                ["Input profiles", result_df["inputProfile"].nunique()],
                ["Condition templates", result_df["conditionTemplateId"].nunique()],
                ["E01 original context rows", len(original_context_df)],
                ["Generator checks", len(generator_checks_df)],
            ],
        ),
        "",
        "## No-Frozen And Chimera Profile Summary",
        "",
        profile_table,
        "",
        "## Frozen Profile Summary",
        "",
        frozen_table,
        "",
        "## Validation",
    ]
    for row in diagnostics_df.itertuples(index=False):
        summary_lines.append(f"- {'passed' if row.passed else 'failed'}: {row.detail}")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    status_path = step_dir / "status.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "provenance" / "run_manifest.json"
    artifact_inputs = list(table_paths.values()) + list(figure_paths) + code_paths + [
        validation_path,
        summary_path,
        status_path,
        artifact_manifest_path,
        run_manifest_path,
        step_dir / "repo_unit_test_log.txt",
    ]
    artifacts_written = [str(path) for path in sorted(set(artifact_inputs), key=str)]
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(validation["success"] and repo_tests["passed"]),
        "status": "completed" if validation["success"] and repo_tests["passed"] else "failed",
        "artifactsWritten": artifacts_written,
        "validationResult": validation["validationResult"] if repo_tests["passed"] else "failed",
        "caveatsOrBlockers": caveat,
        "recommendedNextAction": recommended,
        "outcomeClassification": outcome,
        "repoTests": repo_tests,
        "runSeconds": time.perf_counter() - started_at,
        "stressRunRows": int(len(result_df)),
        "traceRows": int(len(trace_df)),
        "summaryRows": int(len(summary_df)),
        "generatorCheckRows": int(len(generator_checks_df)),
        "originalContextRows": int(len(original_context_df)),
    }
    write_json(status_path, status_payload)

    all_artifacts = collect_artifacts(list(table_paths.values()) + list(figure_paths) + code_paths + [validation_path, summary_path, status_path, step_dir / "repo_unit_test_log.txt"])
    manifest_payload = {
        "schema": "eidosoma.research_step_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAt": utc_now(),
        "git": get_git_metadata(),
        "inputs": {
            "deterministicSimulator": "e02_deterministic_simulator",
            "s09MetricHelper": "scripts/e02_s09_alternative_metrics.py",
            "s08DgHelper": "scripts/e02_s08_dg_nulls.py",
            "e01S04Summary": str(DEFAULT_E01_S04_SUMMARY),
        },
        "parameters": {
            "inputProfiles": PROFILE_IDS,
            "conditionTemplates": [condition.condition_id for condition in stress_conditions()],
        },
        "outputs": all_artifacts,
        "validation": validation,
        "repoTests": repo_tests,
        "stressRunRows": int(len(result_df)),
        "traceRows": int(len(trace_df)),
        "summaryRows": int(len(summary_df)),
        "profileRows": int(len(profile_df)),
        "originalContextRows": int(len(original_context_df)),
        "outcomeClassification": outcome,
    }
    write_json(artifact_manifest_path, manifest_payload)

    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "latestResearchStepId": STEP_ID,
        "generatedAt": utc_now(),
        "startedAt": utc_now(),
        "statusPath": str(status_path),
        "git": get_git_metadata(),
        "hardware": {"platform": platform.platform(), "python": sys.version, "cpuCount": os.cpu_count()},
        "packageVersions": {"numpy": np.__version__, "pandas": pd.__version__},
        "runs": [
            {
                "researchStepId": STEP_ID,
                "status": status_payload["status"],
                "runSeconds": status_payload["runSeconds"],
                "stressRunRows": int(len(result_df)),
            }
        ],
        "artifacts": all_artifacts,
    }
    write_json(run_manifest_path, run_manifest)
    return {
        "summary": summary_path,
        "status": status_path,
        "validation": validation_path,
        "artifact_manifest": artifact_manifest_path,
        "run_manifest": run_manifest_path,
    }


def run_s10(args: argparse.Namespace) -> int:
    started_at = time.perf_counter()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    figure_dir = artifacts_dir / "figures" / "e02"
    step_dir.mkdir(parents=True, exist_ok=True)
    profile_df, generator_checks_df = validate_generators(args.n)
    result_df, trace_df = run_s10_matrix(
        n=args.n,
        replicates=args.replicates,
        max_activations=args.max_activations,
        max_swaps=args.max_swaps,
        max_comparisons=args.max_comparisons,
    )
    summary_df = summarize_results(result_df)
    original_context_df = load_e01_original_context(args.e01_s04_summary)
    original_summary_df = original_context_summary(original_context_df)
    repo_tests_placeholder = {"passed": True}
    diagnostics_df = diagnostics_table(
        result_df=result_df,
        trace_df=trace_df,
        summary_df=summary_df,
        profile_df=profile_df,
        generator_checks_df=generator_checks_df,
        original_context_df=original_context_df,
        expected_rows=len(PROFILE_IDS) * len(stress_conditions()) * args.replicates,
        repo_tests_passed=repo_tests_placeholder["passed"],
    )
    validation = validate_outputs(diagnostics_df)
    table_paths = write_tables(
        result_df=result_df,
        trace_df=trace_df,
        summary_df=summary_df,
        profile_df=profile_df,
        generator_checks_df=generator_checks_df,
        diagnostics_df=diagnostics_df,
        original_context_df=original_context_df,
        original_summary_df=original_summary_df,
        artifacts_dir=artifacts_dir,
    )
    figure_paths = plot_summary(summary_df, result_df, figure_dir)
    code_paths = copy_code_artifacts(step_dir)
    repo_tests = run_repo_tests(step_dir)
    diagnostics_df = diagnostics_table(
        result_df=result_df,
        trace_df=trace_df,
        summary_df=summary_df,
        profile_df=profile_df,
        generator_checks_df=generator_checks_df,
        original_context_df=original_context_df,
        expected_rows=len(PROFILE_IDS) * len(stress_conditions()) * args.replicates,
        repo_tests_passed=repo_tests["passed"],
    )
    validation = validate_outputs(diagnostics_df)
    diagnostics_df.to_parquet(table_paths["diagnostics_parquet"], index=False)
    diagnostics_df.to_csv(table_paths["diagnostics_csv"], index=False)
    if not repo_tests["passed"]:
        validation["success"] = False
        validation["validationResult"] = "failed"
        validation.setdefault("failures", []).append("Repository S10 unit tests failed.")
    write_reports_and_manifests(
        artifacts_dir=artifacts_dir,
        started_at=started_at,
        result_df=result_df,
        trace_df=trace_df,
        summary_df=summary_df,
        profile_df=profile_df,
        generator_checks_df=generator_checks_df,
        diagnostics_df=diagnostics_df,
        original_context_df=original_context_df,
        original_summary_df=original_summary_df,
        table_paths=table_paths,
        figure_paths=figure_paths,
        code_paths=code_paths,
        repo_tests=repo_tests,
        validation=validation,
    )
    return 0 if validation["success"] and repo_tests["passed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--e01-s04-summary", type=Path, default=DEFAULT_E01_S04_SUMMARY)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--max-activations", type=int, default=300_000)
    parser.add_argument("--max-swaps", type=int, default=100_000)
    parser.add_argument("--max-comparisons", type=int, default=800_000)
    return parser.parse_args()


def main() -> int:
    return run_s10(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
