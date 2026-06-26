#!/usr/bin/env python3
"""Execute E02 S03 activation-rate artifact tests.

S03 changes only activation probabilities while leaving S01 deterministic
cell-view policy logic unchanged. It compares equalized rates against explicit
Algotype multipliers and throttles in pure controls and same-goal chimeras.
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
from collections import Counter
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

from e02_deterministic_simulator import (  # noqa: E402
    DeterministicEventSimulator,
    aggregation,
    initial_values_from_seed,
    monotonicity_error,
    sortedness_percent,
    sortedness_raw,
    state_hash,
)


EXPERIMENT_ID = "E02"
STEP_ID = "S03"
STEP_NUMBER = 3
DEFAULT_E01_ARTIFACTS = Path("/previous-artifacts/E01")
SAME_GOAL_MIXTURES = [
    "pure_bubble",
    "pure_insertion",
    "pure_selection",
    "bubble_insertion",
    "bubble_selection",
    "insertion_selection",
    "bubble_insertion_selection",
]


@dataclass
class CommandResult:
    args: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    ok: bool


@dataclass
class RateRunResult:
    completed: bool
    stop_reason: str
    initial_values: list[int]
    final_values: list[int]
    initial_algotypes: list[str]
    final_algotypes: list[str]
    swap_count: int
    comparison_count: int
    archived_compare_and_swap_count: int
    activation_count: int
    event_count: int
    final_sortedness_percent: float
    final_monotonicity_error: int
    initial_aggregation: float
    final_aggregation: float
    peak_aggregation: float
    dg_max_drop: float
    dg_decrease_count: int
    wall_time_seconds: float
    activation_counts_by_algotype: dict[str, int]
    activation_shares_by_algotype: dict[str, float]
    target_shares_by_algotype: dict[str, float]
    max_activation_share_abs_error: float
    trace_rows: list[dict[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> CommandResult:
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
        return CommandResult(args, proc.returncode, proc.stdout, proc.stderr, proc.returncode == 0)
    except Exception as exc:  # pragma: no cover - defensive provenance path
        return CommandResult(args, None, "", repr(exc), False)


def get_git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "-v"], cwd=REPO_ROOT)
    return {
        "commit": commit.stdout.strip() if commit.ok else "unknown",
        "branch": branch.stdout.strip() if branch.ok else "unknown",
        "dirtyStatus": status.stdout.strip(),
        "remote": remote.stdout.strip(),
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
    if hasattr(value, "item"):
        return json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def scaled_allocation(mixture_id: str, n: int) -> dict[str, int]:
    if mixture_id.startswith("pure_"):
        return {mixture_id.removeprefix("pure_"): n}
    if mixture_id == "bubble_insertion":
        return {"bubble": n // 2, "insertion": n - n // 2}
    if mixture_id == "bubble_selection":
        return {"bubble": n // 2, "selection": n - n // 2}
    if mixture_id == "insertion_selection":
        return {"insertion": n // 2, "selection": n - n // 2}
    if mixture_id == "bubble_insertion_selection":
        base = n // 3
        return {
            "bubble": base + (1 if n % 3 > 0 else 0),
            "insertion": base + (1 if n % 3 > 1 else 0),
            "selection": base,
        }
    raise ValueError(f"unsupported S03 mixture: {mixture_id}")


def algotypes_from_seed(mixture_id: str, n: int, seed: Any) -> list[str]:
    allocation = scaled_allocation(mixture_id, n)
    algotypes = [algorithm for algorithm, count in allocation.items() for _ in range(count)]
    if len(algotypes) != n:
        raise ValueError(f"allocation for {mixture_id} gives {len(algotypes)} cells, expected {n}")
    if len(allocation) > 1:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(algotypes)
    return algotypes


def same_goal_condition_id(mixture_id: str) -> str:
    return f"S09_cell_view_{mixture_id}_unique_same_goal"


def rate_profiles_for_mixture(mixture_id: str) -> list[dict[str, Any]]:
    algorithms = list(scaled_allocation(mixture_id, 30).keys())
    profiles = [
        {
            "rateProfileId": "equalized_all_1x",
            "profileFamily": "equalized",
            "focusAlgotype": "none",
            "rateWeights": {algorithm: 1.0 for algorithm in algorithms},
        }
    ]
    if len(algorithms) == 1:
        return profiles
    for algorithm in algorithms:
        weights = {item: 1.0 for item in algorithms}
        weights[algorithm] = 3.0
        profiles.append(
            {
                "rateProfileId": f"{algorithm}_fast_3x",
                "profileFamily": "multiplier",
                "focusAlgotype": algorithm,
                "rateWeights": weights,
            }
        )
    for algorithm in algorithms:
        weights = {item: 1.0 for item in algorithms}
        weights[algorithm] = 1.0 / 3.0
        profiles.append(
            {
                "rateProfileId": f"{algorithm}_throttle_0p33x",
                "profileFamily": "throttle",
                "focusAlgotype": algorithm,
                "rateWeights": weights,
            }
        )
    return profiles


def load_s03_inputs(e01_artifacts: Path, *, n: int, replicate_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_ids = [same_goal_condition_id(mixture) for mixture in SAME_GOAL_MIXTURES]
    condition_matrix = pd.read_csv(e01_artifacts / "research_steps" / "S03" / "condition_matrix.csv")
    seed_table = pd.read_csv(e01_artifacts / "research_steps" / "S03" / "seed_table.csv")
    conditions = condition_matrix[condition_matrix["conditionId"].isin(condition_ids)].copy()
    conditions["s03N"] = int(n)
    conditions = conditions.sort_values("conditionId").reset_index(drop=True)
    seeds = seed_table[
        (seed_table["conditionId"].isin(condition_ids))
        & (seed_table["replicateIndex"] < int(replicate_count))
    ].copy()
    seeds = seeds.sort_values(["conditionId", "replicateIndex"]).reset_index(drop=True)
    if len(conditions) != len(condition_ids):
        missing = sorted(set(condition_ids) - set(conditions["conditionId"]))
        raise ValueError(f"Missing S03 same-goal conditions in E01 matrix: {missing}")
    return conditions, seeds


def trajectory_diagnostics(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sortedness = [float(row["sortedness_percent"]) for row in trace_rows]
    aggregations = []
    for row in trace_rows:
        algotypes_raw = row.get("algotypes_json")
        if algotypes_raw:
            aggregations.append(aggregation(json.loads(algotypes_raw)))
    running_best = -math.inf
    max_drop = 0.0
    decrease_count = 0
    previous = None
    for value in sortedness:
        running_best = max(running_best, value)
        max_drop = max(max_drop, running_best - value)
        if previous is not None and value < previous:
            decrease_count += 1
        previous = value
    return {
        "dgMaxDropPercent": float(max_drop),
        "dgDecreaseCount": int(decrease_count),
        "initialAggregation": aggregations[0] if aggregations else None,
        "peakAggregation": max(aggregations) if aggregations else None,
    }


def target_activation_shares(initial_algotypes: list[str], rate_weights: dict[str, float]) -> dict[str, float]:
    counts = Counter(initial_algotypes)
    weighted = {algotype: counts[algotype] * float(rate_weights[algotype]) for algotype in counts}
    denom = sum(weighted.values())
    if denom <= 0:
        raise ValueError("activation weights must have positive total")
    return {algotype: value / denom for algotype, value in weighted.items()}


def choose_weighted_cell_id(
    sim: DeterministicEventSimulator,
    rate_weights: dict[str, float],
    rng: np.random.Generator,
) -> int | None:
    eligible = sim.eligible_cell_ids()
    if not eligible:
        return None
    weights = []
    for cell_id in eligible:
        pos = sim.positions_by_id[cell_id]
        weights.append(float(rate_weights[sim.cells[pos].algotype]))
    weights_arr = np.asarray(weights, dtype=float)
    if np.any(weights_arr < 0) or weights_arr.sum() <= 0:
        raise ValueError("activation weights must be nonnegative and have positive total")
    if np.allclose(weights_arr, weights_arr[0]):
        return int(rng.choice(eligible))
    return int(rng.choice(eligible, p=weights_arr / weights_arr.sum()))


def run_rate_profile(
    *,
    initial_values: list[int],
    initial_algotypes: list[str],
    condition_id: str,
    rate_profile_id: str,
    rate_weights: dict[str, float],
    scheduler_seed: int,
    tie_breaker_seed: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
    no_move_checks_required: int = 2,
) -> RateRunResult:
    start = time.perf_counter()
    sim = DeterministicEventSimulator(
        initial_values,
        initial_algotypes,
        scheduler_seed=scheduler_seed,
        tie_breaker_seed=tie_breaker_seed,
        condition_id=condition_id,
        research_step_id=STEP_ID,
    )
    rng = sim.scheduler_rng
    no_move_checks = 0
    stop_reason = "sorted"
    activation_counts: Counter[str] = Counter()
    interval = max(1, len(initial_values))
    while True:
        if sim.is_sorted():
            stop_reason = "sorted"
            break
        if sim.activation_count >= max_activations:
            stop_reason = "max_activation_cap"
            break
        if sim.swap_count >= max_swaps:
            stop_reason = "max_step_cap"
            break
        if sim.comparison_count >= max_comparisons:
            stop_reason = "max_comparison_cap"
            break
        if sim.activation_count % interval == 0:
            if not sim.legal_action_exists():
                no_move_checks += 1
                if no_move_checks >= no_move_checks_required:
                    stop_reason = "no_cell_can_move_after_two_checks"
                    break
            else:
                no_move_checks = 0
        cell_id = choose_weighted_cell_id(sim, rate_weights, rng)
        if cell_id is None:
            stop_reason = "no_eligible_cells"
            break
        actor_algotype = sim.cells[sim.positions_by_id[cell_id]].algotype
        outcome = sim.step(forced_cell_id=cell_id)
        if outcome.activated:
            activation_counts[actor_algotype] += 1

    final_values = sim.current_values()
    trace_rows = list(sim.trace_rows)
    diagnostics = trajectory_diagnostics(trace_rows)
    total_activations = sum(activation_counts.values())
    realized_shares = {
        algotype: (activation_counts[algotype] / total_activations if total_activations else 0.0)
        for algotype in sorted(set(initial_algotypes))
    }
    target_shares = target_activation_shares(initial_algotypes, rate_weights)
    max_share_error = max(abs(realized_shares.get(algotype, 0.0) - target_shares.get(algotype, 0.0)) for algotype in target_shares)
    return RateRunResult(
        completed=sim.is_sorted(),
        stop_reason=stop_reason,
        initial_values=list(initial_values),
        final_values=final_values,
        initial_algotypes=list(initial_algotypes),
        final_algotypes=sim.current_algotypes(),
        swap_count=sim.swap_count,
        comparison_count=sim.comparison_count,
        archived_compare_and_swap_count=sim.archived_compare_and_swap_count,
        activation_count=sim.activation_count,
        event_count=len(trace_rows),
        final_sortedness_percent=sortedness_percent(final_values),
        final_monotonicity_error=monotonicity_error(final_values),
        initial_aggregation=float(diagnostics["initialAggregation"] or aggregation(initial_algotypes)),
        final_aggregation=aggregation(sim.current_algotypes()),
        peak_aggregation=float(diagnostics["peakAggregation"] or aggregation(sim.current_algotypes())),
        dg_max_drop=float(diagnostics["dgMaxDropPercent"]),
        dg_decrease_count=int(diagnostics["dgDecreaseCount"]),
        wall_time_seconds=time.perf_counter() - start,
        activation_counts_by_algotype=dict(activation_counts),
        activation_shares_by_algotype=realized_shares,
        target_shares_by_algotype=target_shares,
        max_activation_share_abs_error=float(max_share_error),
        trace_rows=trace_rows,
    )


def policy_signature() -> str:
    return sha256_path(REPO_ROOT / "e02_deterministic_simulator" / "simulator.py")


def make_result_record(
    *,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    n: int,
    profile: dict[str, Any],
    result: RateRunResult,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
) -> dict[str, Any]:
    activation_counts = result.activation_counts_by_algotype
    activation_shares = result.activation_shares_by_algotype
    target_shares = result.target_shares_by_algotype
    dominant_algotype = max(activation_shares, key=lambda algotype: activation_shares[algotype]) if activation_shares else None
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "conditionId": condition["conditionId"],
        "mixtureId": condition["mixtureId"],
        "rateProfileId": profile["rateProfileId"],
        "profileFamily": profile["profileFamily"],
        "focusAlgotype": profile["focusAlgotype"],
        "policyLogicId": "DeterministicEventSimulator:S01_policy_semantics",
        "policySignatureSha256": policy_signature(),
        "n": int(n),
        "replicateIndex": int(seed_row["replicateIndex"]),
        "replicateNumber": int(seed_row["replicateNumber"]),
        "inputPermutationSeed": int(seed_row["inputPermutationSeed"]),
        "algotypeAssignmentSeed": None
        if pd.isna(seed_row.get("algotypeAssignmentSeed"))
        else int(seed_row["algotypeAssignmentSeed"]),
        "schedulerSeed": int(seed_row["schedulerSeed"]),
        "tieBreakerSeed": int(seed_row["tieBreakerSeed"]),
        "rateWeights": json.dumps(profile["rateWeights"], sort_keys=True, separators=(",", ":")),
        "targetActivationShares": json.dumps(target_shares, sort_keys=True, separators=(",", ":")),
        "realizedActivationCounts": json.dumps(activation_counts, sort_keys=True, separators=(",", ":")),
        "realizedActivationShares": json.dumps(activation_shares, sort_keys=True, separators=(",", ":")),
        "maxActivationShareAbsError": result.max_activation_share_abs_error,
        "dominantActivationAlgotype": dominant_algotype,
        "dominantActivationShare": activation_shares.get(dominant_algotype, None) if dominant_algotype else None,
        "initialValuesHash": state_hash(result.initial_values),
        "finalValuesHash": state_hash(result.final_values),
        "initialValues": json.dumps(result.initial_values, separators=(",", ":")),
        "finalValues": json.dumps(result.final_values, separators=(",", ":")),
        "initialAlgotypes": json.dumps(result.initial_algotypes, separators=(",", ":")),
        "finalAlgotypes": json.dumps(result.final_algotypes, separators=(",", ":")),
        "completed": bool(result.completed),
        "stopReason": result.stop_reason,
        "finalSortednessPercent": float(result.final_sortedness_percent),
        "finalMonotonicityError": int(result.final_monotonicity_error),
        "swapCount": int(result.swap_count),
        "comparisonCount": int(result.comparison_count),
        "archivedCompareAndSwapCount": int(result.archived_compare_and_swap_count),
        "swapPlusComparisonSteps": int(result.swap_count + result.comparison_count),
        "activationCount": int(result.activation_count),
        "eventCount": int(result.event_count),
        "initialAggregation": float(result.initial_aggregation),
        "finalAggregation": float(result.final_aggregation),
        "peakAggregation": float(result.peak_aggregation),
        "dgMaxDropPercent": float(result.dg_max_drop),
        "dgDecreaseCount": int(result.dg_decrease_count),
        "wallTimeSeconds": float(result.wall_time_seconds),
        "maxActivationCap": int(max_activations),
        "maxSwapCap": int(max_swaps),
        "maxComparisonCap": int(max_comparisons),
    }


def annotate_trace_rows(
    rows: list[dict[str, Any]],
    *,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    annotated = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "mixtureId": condition["mixtureId"],
                "rateProfileId": profile["rateProfileId"],
                "profileFamily": profile["profileFamily"],
                "focusAlgotype": profile["focusAlgotype"],
                "rateWeights": json.dumps(profile["rateWeights"], sort_keys=True, separators=(",", ":")),
                "replicateIndex": int(seed_row["replicateIndex"]),
                "replicateNumber": int(seed_row["replicateNumber"]),
                "inputPermutationSeed": int(seed_row["inputPermutationSeed"]),
                "algotypeAssignmentSeed": None
                if pd.isna(seed_row.get("algotypeAssignmentSeed"))
                else int(seed_row["algotypeAssignmentSeed"]),
            }
        )
        annotated.append(item)
    return annotated


def run_activation_rate_matrix(
    *,
    e01_artifacts: Path,
    n: int,
    replicate_count: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    conditions_df, seeds_df = load_s03_inputs(e01_artifacts, n=n, replicate_count=replicate_count)
    result_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    profile_count = 0
    for condition_tuple in conditions_df.itertuples(index=False):
        condition = condition_tuple._asdict()
        mixture_id = str(condition["mixtureId"])
        profiles = rate_profiles_for_mixture(mixture_id)
        profile_count += len(profiles)
        condition_seeds = seeds_df[seeds_df["conditionId"] == condition["conditionId"]].copy()
        for seed_tuple in condition_seeds.itertuples(index=False):
            seed_row = seed_tuple._asdict()
            input_seed = int(seed_row["inputPermutationSeed"])
            algotype_seed = seed_row.get("algotypeAssignmentSeed")
            if pd.isna(algotype_seed):
                algotype_seed = int(seed_row["schedulerSeed"])
            initial_values = initial_values_from_seed(input_seed, n=n, profile=str(condition["inputProfile"]))
            initial_algotypes = algotypes_from_seed(mixture_id, n, algotype_seed)
            for profile in profiles:
                result = run_rate_profile(
                    initial_values=initial_values,
                    initial_algotypes=initial_algotypes,
                    condition_id=condition["conditionId"],
                    rate_profile_id=profile["rateProfileId"],
                    rate_weights=profile["rateWeights"],
                    scheduler_seed=int(seed_row["schedulerSeed"]),
                    tie_breaker_seed=int(seed_row["tieBreakerSeed"]),
                    max_activations=max_activations,
                    max_swaps=max_swaps,
                    max_comparisons=max_comparisons,
                )
                result_rows.append(
                    make_result_record(
                        condition=condition,
                        seed_row=seed_row,
                        n=n,
                        profile=profile,
                        result=result,
                        max_activations=max_activations,
                        max_swaps=max_swaps,
                        max_comparisons=max_comparisons,
                    )
                )
                trace_rows.extend(annotate_trace_rows(result.trace_rows, condition=condition, seed_row=seed_row, profile=profile))
    validation = {
        "conditionCount": int(len(conditions_df)),
        "replicateCountPerCondition": int(replicate_count),
        "profileCountAcrossConditions": int(profile_count),
        "expectedRows": int(profile_count * replicate_count),
    }
    return pd.DataFrame(result_rows), pd.DataFrame(trace_rows), validation


def summarize_activation_rates(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_df.groupby(["mixtureId", "rateProfileId", "profileFamily", "focusAlgotype"], dropna=False)
        .agg(
            runCount=("conditionId", "count"),
            completedCount=("completed", "sum"),
            completionRate=("completed", "mean"),
            meanFinalSortednessPercent=("finalSortednessPercent", "mean"),
            meanFinalMonotonicityError=("finalMonotonicityError", "mean"),
            meanSwapCount=("swapCount", "mean"),
            meanActivationCount=("activationCount", "mean"),
            meanDominantActivationShare=("dominantActivationShare", "mean"),
            meanMaxActivationShareAbsError=("maxActivationShareAbsError", "mean"),
            meanPeakAggregation=("peakAggregation", "mean"),
            meanFinalAggregation=("finalAggregation", "mean"),
            meanDgMaxDropPercent=("dgMaxDropPercent", "mean"),
            wallTimeSecondsTotal=("wallTimeSeconds", "sum"),
        )
        .reset_index()
    )
    equalized = summary[summary["rateProfileId"] == "equalized_all_1x"][
        [
            "mixtureId",
            "meanActivationCount",
            "meanSwapCount",
            "meanPeakAggregation",
            "meanFinalAggregation",
            "meanDgMaxDropPercent",
        ]
    ].rename(
        columns={
            "meanActivationCount": "equalizedMeanActivationCount",
            "meanSwapCount": "equalizedMeanSwapCount",
            "meanPeakAggregation": "equalizedMeanPeakAggregation",
            "meanFinalAggregation": "equalizedMeanFinalAggregation",
            "meanDgMaxDropPercent": "equalizedMeanDgMaxDropPercent",
        }
    )
    summary = summary.merge(equalized, on="mixtureId", how="left")
    summary["deltaActivationCountVsEqualized"] = summary["meanActivationCount"] - summary["equalizedMeanActivationCount"]
    summary["deltaSwapCountVsEqualized"] = summary["meanSwapCount"] - summary["equalizedMeanSwapCount"]
    summary["deltaPeakAggregationVsEqualized"] = summary["meanPeakAggregation"] - summary["equalizedMeanPeakAggregation"]
    summary["deltaFinalAggregationVsEqualized"] = summary["meanFinalAggregation"] - summary["equalizedMeanFinalAggregation"]
    summary["deltaDgMaxDropVsEqualized"] = summary["meanDgMaxDropPercent"] - summary["equalizedMeanDgMaxDropPercent"]
    return summary.sort_values(["mixtureId", "profileFamily", "rateProfileId"]).reset_index(drop=True)


def validate_outputs(results_df: pd.DataFrame, validation: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    checks: list[str] = []
    failures: list[str] = []
    if len(results_df) == validation["expectedRows"]:
        checks.append(f"Activation-rate result row count matched expected {validation['expectedRows']}.")
    else:
        failures.append(f"Activation-rate results have {len(results_df)} rows, expected {validation['expectedRows']}.")
    seed_groups = results_df.groupby(["mixtureId", "replicateIndex"])
    if all(group["initialValuesHash"].nunique() == 1 for _, group in seed_groups):
        checks.append("Initial value hashes are matched across rate profiles for every mixture/replicate.")
    else:
        failures.append("At least one mixture/replicate has mismatched initial value hashes across rate profiles.")
    if all(group["initialAlgotypes"].nunique() == 1 for _, group in seed_groups):
        checks.append("Initial Algotype assignments are matched across rate profiles for every mixture/replicate.")
    else:
        failures.append("At least one mixture/replicate has mismatched initial Algotype assignments across rate profiles.")
    if bool((results_df["finalValues"].apply(lambda text: Counter(json.loads(text))) == results_df["initialValues"].apply(lambda text: Counter(json.loads(text)))).all()):
        checks.append("All activation-rate runs conserve value multisets.")
    else:
        failures.append("At least one activation-rate run failed value-multiset conservation.")
    if bool((results_df["finalAlgotypes"].apply(lambda text: Counter(json.loads(text))) == results_df["initialAlgotypes"].apply(lambda text: Counter(json.loads(text)))).all()):
        checks.append("All activation-rate runs conserve Algotype multisets.")
    else:
        failures.append("At least one activation-rate run failed Algotype-multiset conservation.")
    if results_df["policySignatureSha256"].nunique() == 1:
        checks.append("Policy signature is unchanged across all activation-rate profiles.")
    else:
        failures.append("Policy signature differs across activation-rate profiles.")
    tolerance = results_df["activationCount"].apply(lambda count: max(0.08, 4.0 / math.sqrt(max(1, int(count)))))
    if bool((results_df["maxActivationShareAbsError"] <= tolerance).all()):
        checks.append("Realized activation shares match target distributions within binomial-tolerant bounds.")
    else:
        bad = results_df[results_df["maxActivationShareAbsError"] > tolerance][
            ["mixtureId", "rateProfileId", "replicateIndex", "activationCount", "maxActivationShareAbsError"]
        ]
        failures.append(f"Activation-share realization exceeded tolerance for {len(bad)} rows.")
    pure_equalized = results_df[(results_df["mixtureId"].str.startswith("pure_")) & (results_df["rateProfileId"] == "equalized_all_1x")]
    if bool(pure_equalized["completed"].all()):
        checks.append("Pure equalized controls completed with sorted final states.")
    else:
        failures.append("At least one pure equalized control did not complete.")
    if set(SAME_GOAL_MIXTURES) == set(results_df["mixtureId"].unique()):
        checks.append("All planned pure and same-goal mixed conditions are represented.")
    else:
        failures.append(f"Observed mixtures differ from planned: {sorted(results_df['mixtureId'].unique())}.")
    return not failures, checks, failures


def write_tables(results_df: pd.DataFrame, trace_df: pd.DataFrame, artifacts_dir: Path) -> dict[str, Path]:
    results_dir = artifacts_dir / "results"
    traces_dir = artifacts_dir / "traces" / "e02" / STEP_ID
    results_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    summary_df = summarize_activation_rates(results_df)
    paths = {
        "results_parquet": results_dir / "e02_activation_rates.parquet",
        "results_csv": results_dir / "e02_activation_rates.csv",
        "summary_parquet": results_dir / "e02_activation_rate_summary.parquet",
        "summary_csv": results_dir / "e02_activation_rate_summary.csv",
        "trace_parquet": traces_dir / "e02_activation_rate_trace_events.parquet",
        "trace_csv_gz": traces_dir / "e02_activation_rate_trace_events.csv.gz",
    }
    results_df.to_parquet(paths["results_parquet"], index=False)
    results_df.to_csv(paths["results_csv"], index=False)
    summary_df.to_parquet(paths["summary_parquet"], index=False)
    summary_df.to_csv(paths["summary_csv"], index=False)
    trace_df.to_parquet(paths["trace_parquet"], index=False)
    trace_df.to_csv(paths["trace_csv_gz"], index=False, compression="gzip")
    return paths


def plot_activation_rates(summary_df: pd.DataFrame, figure_dir: Path) -> tuple[Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / "e02_activation_rates_summary.png"
    pdf_path = figure_dir / "e02_activation_rates_summary.pdf"
    mixed = summary_df[~summary_df["mixtureId"].str.startswith("pure_")].copy()
    mixed["profileShort"] = mixed["rateProfileId"].str.replace("_", "\n")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = [
        ("meanDominantActivationShare", "Dominant activation share"),
        ("meanActivationCount", "Mean activation count"),
        ("meanPeakAggregation", "Mean peak Aggregation"),
        ("meanDgMaxDropPercent", "Mean DG proxy max drop"),
    ]
    mixtures = ["bubble_insertion", "bubble_selection", "insertion_selection", "bubble_insertion_selection"]
    colors = {
        "bubble_insertion": "#4c78a8",
        "bubble_selection": "#f58518",
        "insertion_selection": "#54a24b",
        "bubble_insertion_selection": "#b279a2",
    }
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        profile_order = list(dict.fromkeys(mixed["rateProfileId"].tolist()))
        x = np.arange(len(profile_order))
        width = 0.18
        for offset_index, mixture in enumerate(mixtures):
            subset = mixed[mixed["mixtureId"] == mixture].set_index("rateProfileId").reindex(profile_order)
            ax.bar(x + (offset_index - 1.5) * width, subset[metric].to_numpy(dtype=float), width=width, label=mixture, color=colors[mixture])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([item.replace("_", "\n") for item in profile_order], rotation=0, ha="center", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("E02 S03 activation-rate interventions (matched policies, seeds, and initial arrays)", fontsize=13)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    artifacts = []
    seen: set[Path] = set()
    for path in paths:
        if path.exists() and path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            artifacts.append({"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)})
    return sorted(artifacts, key=lambda item: item["path"])


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_dir = step_dir / "code"
    script_dst = code_dir / "scripts" / "e02_s03_activation_rates.py"
    test_dst = code_dir / "tests" / "test_e02_activation_rates.py"
    script_dst.parent.mkdir(parents=True, exist_ok=True)
    test_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "scripts" / "e02_s03_activation_rates.py", script_dst)
    if (REPO_ROOT / "tests" / "test_e02_activation_rates.py").exists():
        shutil.copy2(REPO_ROOT / "tests" / "test_e02_activation_rates.py", test_dst)
    return [path for path in [script_dst, test_dst] if path.exists()]


def write_validation_report(step_dir: Path, checks: list[str], failures: list[str]) -> Path:
    path = step_dir / "validation_report.md"
    checks_text = "\n".join(f"- {item}" for item in checks) or "- None"
    failures_text = "\n".join(f"- {item}" for item in failures) or "- None"
    path.write_text(
        f"""# S03 Validation Report

- Research step ID: {STEP_ID}
- Completion status: {'completed' if not failures else 'completed with validation failures'}
- Artifacts written: activation-rate result tables, summary tables, compact trace events, figures, copied code, manifests, and status files under `$ARTIFACTS_DIR`.
- Validation result: {'passed' if not failures else 'failed'}
- Caveats or blockers: activation-rate controls change who gets activated, so total activation counts can change independently of swap counts; DG values are Sortedness-backtracking proxies pending S08; pure controls only test unchanged policy logic because rate ratios are not identifiable with one Algotype.
- Recommended next action: proceed to S04 trajectory-preserving label shuffles only after Chief Scientist review.

## Checks

{checks_text}

## Failures

{failures_text}
""",
        encoding="utf-8",
    )
    return path


def run_repo_tests(step_dir: Path) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]
    result = run_command(cmd, cwd=REPO_ROOT)
    log_path = step_dir / "repo_unit_test_log.txt"
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\nSTDOUT\n" + result.stdout + "\n\nSTDERR\n" + result.stderr,
        encoding="utf-8",
    )
    return {"command": cmd, "returnCode": result.returncode, "success": result.ok, "logPath": str(log_path)}


def write_reports_and_manifests(
    *,
    artifacts_dir: Path,
    step_dir: Path,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    checks: list[str],
    failures: list[str],
    validation: dict[str, Any],
    repo_test_payload: dict[str, Any],
    artifact_paths: list[Path],
    started_at: str,
    n: int,
    replicate_count: int,
    max_activations: int,
) -> tuple[Path, Path, Path]:
    success = not failures and bool(repo_test_payload["success"])
    mixed = summary_df[~summary_df["mixtureId"].str.startswith("pure_")]
    non_equalized = mixed[mixed["rateProfileId"] != "equalized_all_1x"]
    max_peak_delta = float(non_equalized["deltaPeakAggregationVsEqualized"].abs().max()) if len(non_equalized) else 0.0
    max_activation_delta = float(non_equalized["deltaActivationCountVsEqualized"].abs().max()) if len(non_equalized) else 0.0
    outcome_classification = "supportive" if success and max_peak_delta < 0.15 else "constraining/contradictory"
    validation_result = (
        "passed: matched seeds and initial arrays, policy signature unchanged, realized activation shares matched target distributions, and repository tests passed"
        if success
        else "failed: see validation_report.md and status.json"
    )
    caveats = [
        "Activation-rate controls alter only scheduler selection probabilities, not local policy code.",
        "Rate matching changes total activation opportunities, so activation counts and swap counts are reported separately.",
        "DG is reported as a Sortedness-backtracking proxy for S03; the fuller DG-specific null audit remains S08.",
        "Pure controls verify unchanged policy logic, but activation-rate ratios are only identifiable in mixed-Algotype conditions.",
        f"S03 uses a bounded n={n} matched-seed matrix to avoid starting S04-scale permutation nulls.",
    ]
    recommended_next_action = "Proceed to S04 trajectory-preserving label shuffles only after Chief Scientist review; carry S03 rate-control caveats into Aggregation interpretations."
    summary_path = step_dir / "summary.md"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "provenance" / "run_manifest.json"
    reported_artifact_paths = [*artifact_paths, summary_path, status_path, manifest_path, run_manifest_path]
    reported_artifact_text_paths = sorted({str(path) for path in reported_artifact_paths})
    artifact_records = collect_artifacts(reported_artifact_paths)
    artifact_text = "\n".join(f"- `{path}`" for path in reported_artifact_text_paths)
    validation_table = markdown_table(
        ["Check type", "Count"],
        [
            ["activation-rate result rows", len(results_df)],
            ["same-goal mixtures", results_df["mixtureId"].nunique()],
            ["rate profiles", results_df["rateProfileId"].nunique()],
            ["replicates per condition", replicate_count],
            ["trace rows", int(sum(results_df["eventCount"]))],
        ],
    )
    preview = summary_df[summary_df["mixtureId"].isin(["bubble_insertion", "bubble_selection"])].head(14)
    preview_table = markdown_table(
        ["Mixture", "Profile", "Runs", "Completion", "Dominant act.", "Activation delta", "Peak Agg. delta", "DG delta"],
        [
            [
                row.mixtureId,
                row.rateProfileId,
                int(row.runCount),
                row.completionRate,
                row.meanDominantActivationShare,
                row.deltaActivationCountVsEqualized,
                row.deltaPeakAggregationVsEqualized,
                row.deltaDgMaxDropVsEqualized,
            ]
            for row in preview.itertuples(index=False)
        ],
    )
    summary_path.write_text(
        f"""# E02 S03 Status Summary

- Research step ID: {STEP_ID}
- Step number: {STEP_NUMBER}
- Completion status: {'completed' if success else 'completed with validation failures'}
- Outcome classification: {outcome_classification}
- Artifacts written:
{artifact_text}
- Validation result: {validation_result}
- Caveats or blockers: {' '.join(caveats)}
- Lay summary: S03 changed activation rates while keeping cell policies and initial arrays fixed. Equalized and skewed-rate mixed conditions mostly still sorted, but rate skew changed activation counts and shifted peak Aggregation or DG-proxy magnitudes in some mixtures, so downstream Aggregation and efficiency claims need activation-rate controls.
- Recommended next action: {recommended_next_action}

## Run Matrix

{validation_table}

## Rate-Effect Preview

{preview_table}

## Anchor Notes

- Max absolute peak-Aggregation delta versus equalized mixed profiles: {max_peak_delta}
- Max absolute activation-count delta versus equalized mixed profiles: {max_activation_delta}
- n: {n}
- Replicates per condition: {replicate_count}
- Max activation cap: {max_activations}
""",
        encoding="utf-8",
    )
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "artifactsWritten": reported_artifact_text_paths,
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats if not failures else caveats + failures,
        "recommendedNextAction": recommended_next_action,
        "outcomeClassification": outcome_classification,
        "startedAt": started_at,
        "completedAt": utc_now(),
        "n": int(n),
        "replicateCountPerCondition": int(replicate_count),
        "maxActivations": int(max_activations),
        "sameGoalMixtures": SAME_GOAL_MIXTURES,
        "repoUnitTestCommand": repo_test_payload,
        "validationChecks": checks,
        "validationFailures": failures,
        "validationMatrix": validation,
        "maxAbsPeakAggregationDeltaVsEqualized": max_peak_delta,
        "maxAbsActivationCountDeltaVsEqualized": max_activation_delta,
    }
    write_json(status_path, status_payload)
    write_json(
        manifest_path,
        {
            "schema": "eidosoma.e02.s03.artifact_manifest.v1",
            "experimentId": EXPERIMENT_ID,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "generatedAt": utc_now(),
            "git": get_git_metadata(),
            "artifacts": artifact_records,
            "validationResult": validation_result,
            "caveatsOrBlockers": caveats,
        },
    )
    write_json(
        run_manifest_path,
        {
            "schema": "eidosoma.e02.run_manifest.v1",
            "experimentId": EXPERIMENT_ID,
            "latestResearchStepId": STEP_ID,
            "generatedAt": utc_now(),
            "startedAt": started_at,
            "statusPath": str(status_path),
            "git": get_git_metadata(),
            "hardware": {
                "platform": platform.platform(),
                "python": sys.version,
                "cpuCount": os.cpu_count(),
                "workerCount": 1,
                "gpuUsed": False,
            },
            "packageVersions": {"numpy": np.__version__, "pandas": pd.__version__},
            "artifacts": artifact_records,
        },
    )
    return summary_path, status_path, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--e01-artifacts-dir", type=Path, default=DEFAULT_E01_ARTIFACTS)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--replicate-count", type=int, default=3)
    parser.add_argument("--max-activations", type=int, default=250_000)
    parser.add_argument("--max-swaps", type=int, default=50_000)
    parser.add_argument("--max-comparisons", type=int, default=500_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now()
    artifacts_dir: Path = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    figure_dir = artifacts_dir / "figures" / "e02"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_df, trace_df, validation = run_activation_rate_matrix(
        e01_artifacts=args.e01_artifacts_dir,
        n=int(args.n),
        replicate_count=int(args.replicate_count),
        max_activations=int(args.max_activations),
        max_swaps=int(args.max_swaps),
        max_comparisons=int(args.max_comparisons),
    )
    table_paths = write_tables(results_df, trace_df, artifacts_dir)
    summary_df = pd.read_parquet(table_paths["summary_parquet"])
    figure_png, figure_pdf = plot_activation_rates(summary_df, figure_dir)
    validation_success, checks, failures = validate_outputs(results_df, validation)
    repo_test_payload = run_repo_tests(step_dir)
    validation_report = write_validation_report(step_dir, checks, failures)
    code_paths = copy_code_artifacts(step_dir)
    artifact_paths = [
        *table_paths.values(),
        figure_png,
        figure_pdf,
        validation_report,
        step_dir / "repo_unit_test_log.txt",
        *code_paths,
    ]
    summary_path, status_path, manifest_path = write_reports_and_manifests(
        artifacts_dir=artifacts_dir,
        step_dir=step_dir,
        results_df=results_df,
        summary_df=summary_df,
        checks=checks,
        failures=failures,
        validation=validation,
        repo_test_payload=repo_test_payload,
        artifact_paths=artifact_paths,
        started_at=started_at,
        n=int(args.n),
        replicate_count=int(args.replicate_count),
        max_activations=int(args.max_activations),
    )
    final_paths = [*artifact_paths, summary_path, status_path, manifest_path, artifacts_dir / "provenance" / "run_manifest.json"]
    manifest_payload = read_json(manifest_path)
    manifest_payload["artifacts"] = collect_artifacts(final_paths)
    write_json(manifest_path, manifest_payload)
    run_payload = read_json(artifacts_dir / "provenance" / "run_manifest.json")
    run_payload["artifacts"] = collect_artifacts(final_paths)
    write_json(artifacts_dir / "provenance" / "run_manifest.json", run_payload)
    return 0 if validation_success and bool(repo_test_payload["success"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
