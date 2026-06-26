#!/usr/bin/env python3
"""Execute E02 S13 stop-condition sensitivity tests.

S13 keeps S01 policy logic, seeds, initial arrays, Frozen Cell placements, and
opposite-goal policy assignments fixed while varying only termination rules:
no-movement windows, event caps, convergence criteria, and stable-state
definitions.
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

from e02_deterministic_simulator import DeterministicEventSimulator, aggregation, sortedness_raw, state_hash
from scripts.e02_s08_dg_nulls import delayed_gratification_from_sortedness
from scripts.e02_s09_alternative_metrics import prefixed_metrics
from scripts.e02_s10_input_distributions import (
    compact_json,
    input_values_for_profile,
    json_ready,
    markdown_table,
    sha256_path,
    stable_seed,
    write_json,
)


EXPERIMENT_ID = "E02"
STEP_ID = "S13"
STEP_NUMBER = 13
DEFAULT_S12_BEHAVIOR_PATH = Path("/artifacts/results/e02_frozen_behavior_variants.parquet")
DEFAULT_E01_OPPOSITE_PATH = Path("/previous-artifacts/E01/results/e01_opposite_direction_chimera_summary.parquet")


@dataclass(frozen=True)
class StopPolicy:
    stop_policy_id: str
    policy_family: str
    description: str
    max_activations: int
    max_swaps: int
    max_comparisons: int
    no_move_checks_required: int
    no_move_interval_factor: float
    stable_state_definition: str
    convergence_criterion: str
    accepts_stable_as_converged: bool
    stable_window_activations: int | None = None


@dataclass(frozen=True)
class S13Condition:
    condition_template_id: str
    condition_class: str
    algorithm_label: str
    algotypes: tuple[str, ...]
    goal_directions: dict[str, str]
    frozen_variant: str = "none"
    frozen_count: int = 0
    frozen_positions_mode: str = "none"


@dataclass
class StopRunResult:
    stop_reason: str
    stop_reason_detail: str
    global_increasing_sorted: bool
    global_decreasing_sorted: bool
    criterion_completed: bool
    accepted_as_converged: bool
    activation_count: int
    swap_count: int
    comparison_count: int
    archived_compare_and_swap_count: int
    blocked_move_attempts: int
    frozen_swap_attempts: int
    no_move_check_count: int
    no_move_failed_check_count: int
    stable_window_triggered: bool
    last_swap_activation: int
    last_state_change_activation: int
    final_values: list[int]
    final_algotypes: list[str]
    final_frozen_positions: list[int]
    final_frozen_cell_ids: list[int]
    trace_rows: list[dict[str, Any]]
    wall_time_seconds: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
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


def stop_policies(n: int) -> list[StopPolicy]:
    high_activation = 120_000
    high_swaps = 20_000
    high_comparisons = 250_000
    return [
        StopPolicy(
            "canonical_s01",
            "canonical",
            "S01-style sorted-or-two-no-legal-action-checks with broad event caps.",
            high_activation,
            high_swaps,
            high_comparisons,
            2,
            1.0,
            "legal_action_absent",
            "global_increasing_sorted",
            True,
        ),
        StopPolicy(
            "no_move_window_1",
            "no_movement_window",
            "Stop after one no-legal-action check.",
            high_activation,
            high_swaps,
            high_comparisons,
            1,
            1.0,
            "legal_action_absent",
            "global_increasing_sorted",
            True,
        ),
        StopPolicy(
            "no_move_window_5",
            "no_movement_window",
            "Require five no-legal-action checks before accepting stability.",
            high_activation,
            high_swaps,
            high_comparisons,
            5,
            1.0,
            "legal_action_absent",
            "global_increasing_sorted",
            True,
        ),
        StopPolicy(
            "dense_no_move_checks",
            "no_movement_window",
            "Check for legal moves twice as often as S01.",
            high_activation,
            high_swaps,
            high_comparisons,
            2,
            0.5,
            "legal_action_absent",
            "global_increasing_sorted",
            True,
        ),
        StopPolicy(
            "sparse_no_move_checks",
            "no_movement_window",
            "Check for legal moves every four array lengths.",
            high_activation,
            high_swaps,
            high_comparisons,
            2,
            4.0,
            "legal_action_absent",
            "global_increasing_sorted",
            True,
        ),
        StopPolicy(
            "sorted_only_ignore_no_move",
            "convergence_criteria",
            "Ignore no-move stability and stop only on global increasing sortedness or event caps.",
            high_activation,
            high_swaps,
            high_comparisons,
            10**9,
            1.0,
            "none",
            "global_increasing_sorted",
            False,
        ),
        StopPolicy(
            "goal_aware_completion",
            "convergence_criteria",
            "For conflict chimeras, stop when each Algotype subsequence satisfies its own goal direction.",
            high_activation,
            high_swaps,
            high_comparisons,
            2,
            1.0,
            "legal_action_absent",
            "goal_aware",
            True,
        ),
        StopPolicy(
            "no_swap_stable_window",
            "stable_state_definition",
            "Accept stability after a long activation window with no accepted swaps.",
            high_activation,
            high_swaps,
            high_comparisons,
            2,
            1.0,
            "no_swap_window",
            "global_increasing_sorted",
            True,
            stable_window_activations=3 * n,
        ),
        StopPolicy(
            "state_hash_plateau_window",
            "stable_state_definition",
            "Accept stability after a long activation window with unchanged value-state hash.",
            high_activation,
            high_swaps,
            high_comparisons,
            2,
            1.0,
            "state_hash_plateau",
            "global_increasing_sorted",
            True,
            stable_window_activations=3 * n,
        ),
        StopPolicy(
            "tight_activation_cap",
            "event_cap",
            "Low activation cap to expose cap-truncated convergence calls.",
            500,
            high_swaps,
            high_comparisons,
            2,
            1.0,
            "legal_action_absent",
            "global_increasing_sorted",
            True,
        ),
        StopPolicy(
            "tight_swap_cap",
            "event_cap",
            "Low accepted-swap cap to expose step-cap sensitivity.",
            high_activation,
            80,
            high_comparisons,
            2,
            1.0,
            "legal_action_absent",
            "global_increasing_sorted",
            True,
        ),
        StopPolicy(
            "tight_comparison_cap",
            "event_cap",
            "Low comparison cap to expose comparison-count sensitivity.",
            high_activation,
            high_swaps,
            900,
            2,
            1.0,
            "legal_action_absent",
            "global_increasing_sorted",
            True,
        ),
    ]


def s13_conditions() -> list[S13Condition]:
    conditions = [
        S13Condition("baseline_pure_bubble", "baseline", "bubble", ("bubble",), {"bubble": "increasing"}),
        S13Condition("baseline_pure_insertion", "baseline", "insertion", ("insertion",), {"insertion": "increasing"}),
        S13Condition("baseline_pure_selection", "baseline", "selection", ("selection",), {"selection": "increasing"}),
        S13Condition(
            "conflict_bubble_down_selection_up",
            "conflict_chimera",
            "bubble+selection",
            ("bubble", "selection"),
            {"bubble": "decreasing", "selection": "increasing"},
        ),
        S13Condition(
            "conflict_bubble_up_insertion_down",
            "conflict_chimera",
            "bubble+insertion",
            ("bubble", "insertion"),
            {"bubble": "increasing", "insertion": "decreasing"},
        ),
        S13Condition(
            "conflict_selection_down_insertion_up",
            "conflict_chimera",
            "selection+insertion",
            ("selection", "insertion"),
            {"selection": "decreasing", "insertion": "increasing"},
        ),
    ]
    for algorithm in ("bubble", "insertion", "selection"):
        for frozen_variant in ("passive", "stuck"):
            conditions.append(
                S13Condition(
                    f"frozen_{algorithm}_{frozen_variant}_f2_center",
                    "frozen_cell",
                    algorithm,
                    (algorithm,),
                    {algorithm: "increasing"},
                    frozen_variant=frozen_variant,
                    frozen_count=2,
                    frozen_positions_mode="center",
                )
            )
    return conditions


def balanced_algotypes(algotypes: tuple[str, ...], n: int, seed: int) -> list[str]:
    if len(algotypes) == 1:
        return [algotypes[0]] * n
    allocation = [algotypes[index % len(algotypes)] for index in range(n)]
    rng = np.random.default_rng(int(seed))
    rng.shuffle(allocation)
    return [str(value) for value in allocation]


def reverse_flags_for_algotypes(algotypes: list[str], goal_directions: dict[str, str]) -> list[bool]:
    return [goal_directions[algotype] == "decreasing" for algotype in algotypes]


def labels_for_algotypes(algotypes: list[str]) -> list[int]:
    mapping = {name: index for index, name in enumerate(sorted(set(algotypes)))}
    return [int(mapping[name]) for name in algotypes]


def frozen_positions_for_condition(condition: S13Condition, n: int) -> list[int]:
    if condition.frozen_count <= 0:
        return []
    if condition.frozen_positions_mode == "center":
        start = (n - condition.frozen_count) // 2
        return list(range(start, start + condition.frozen_count))
    raise ValueError(f"unsupported frozen mode: {condition.frozen_positions_mode}")


def condition_seeds(condition: S13Condition, replicate_index: int) -> dict[str, int]:
    return {
        "inputSeed": stable_seed(STEP_ID, condition.condition_template_id, "input", replicate_index),
        "algotypeSeed": stable_seed(STEP_ID, condition.condition_template_id, "algotypes", replicate_index),
        "schedulerSeed": stable_seed(STEP_ID, condition.condition_template_id, "scheduler", replicate_index),
        "tieBreakerSeed": stable_seed(STEP_ID, condition.condition_template_id, "tie", replicate_index),
    }


def goal_sortedness_by_algotype(
    values: list[int],
    algotypes: list[str],
    goal_directions: dict[str, str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for algotype in sorted(set(algotypes)):
        sub_values = [int(value) for value, label in zip(values, algotypes) if label == algotype]
        direction = goal_directions[algotype]
        raw = sortedness_raw(sub_values, direction=direction)
        denom = max(1, len(sub_values) - 1)
        result[algotype] = {
            "goalDirection": direction,
            "cellCount": len(sub_values),
            "raw": int(raw),
            "percent": float(100.0 * raw / denom),
            "complete": bool(raw == len(sub_values) - 1),
        }
    return result


def goal_aware_complete(sim: DeterministicEventSimulator, condition: S13Condition) -> bool:
    if condition.condition_class != "conflict_chimera":
        return sim.is_sorted()
    values = sim.current_values()
    algotypes = sim.current_algotypes()
    goal_metrics = goal_sortedness_by_algotype(values, algotypes, condition.goal_directions)
    return bool(goal_metrics and all(item["complete"] for item in goal_metrics.values()))


def convergence_reached(sim: DeterministicEventSimulator, condition: S13Condition, policy: StopPolicy) -> tuple[bool, str]:
    values = sim.current_values()
    if policy.convergence_criterion == "global_increasing_sorted":
        return sim.is_sorted(), "sorted"
    if policy.convergence_criterion == "goal_aware":
        if goal_aware_complete(sim, condition):
            reason = "sorted" if sim.is_sorted() else "goal_aware_converged"
            return True, reason
        return False, ""
    if policy.convergence_criterion == "either_global_direction":
        inc = sortedness_raw(values, "increasing") == len(values) - 1
        dec = sortedness_raw(values, "decreasing") == len(values) - 1
        return bool(inc or dec), "either_global_direction_sorted"
    raise ValueError(f"unknown convergence criterion: {policy.convergence_criterion}")


def policy_config_hash(
    initial_algotypes: list[str],
    reverse_directions: list[bool],
    frozen_positions: list[int],
    frozen_variant: str,
    goal_directions: dict[str, str],
) -> str:
    payload = {
        "initialAlgotypes": initial_algotypes,
        "reverseDirections": reverse_directions,
        "frozenPositions": frozen_positions,
        "frozenVariant": frozen_variant,
        "goalDirections": goal_directions,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def snapshot_trace_row(
    *,
    sim: DeterministicEventSimulator,
    condition: S13Condition,
    policy: StopPolicy,
    replicate_index: int,
    event_kind: str,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    values = sim.current_values()
    algotypes = sim.current_algotypes()
    inc_raw = sortedness_raw(values, "increasing")
    dec_raw = sortedness_raw(values, "decreasing")
    denom = max(1, len(values) - 1)
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "conditionTemplateId": condition.condition_template_id,
        "stopPolicyId": policy.stop_policy_id,
        "conditionClass": condition.condition_class,
        "algorithm": condition.algorithm_label,
        "replicateIndex": int(replicate_index),
        "eventIndex": int(len(sim.trace_rows) - 1 if event_kind != "final" else len(sim.trace_rows)),
        "eventKind": event_kind,
        "activationIndex": int(sim.activation_count),
        "swapCount": int(sim.swap_count),
        "comparisonCount": int(sim.comparison_count),
        "archivedCompareAndSwapCount": int(sim.archived_compare_and_swap_count),
        "increasingSortednessRaw": int(inc_raw),
        "increasingSortednessPercent": float(100.0 * inc_raw / denom),
        "decreasingSortednessRaw": int(dec_raw),
        "decreasingSortednessPercent": float(100.0 * dec_raw / denom),
        "monotonicityError": int((len(values) - 1) - inc_raw),
        "aggregation": float(aggregation(algotypes)),
        "stateHash": state_hash(values),
        "values": compact_json(values),
        "algotypes": compact_json(algotypes),
        "frozenPositions": compact_json(sim.current_frozen_positions()),
        "goalSortednessByAlgotype": compact_json(goal_sortedness_by_algotype(values, algotypes, condition.goal_directions)),
        "stopReason": stop_reason,
    }


def sortedness_trajectory(trace_rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["increasingSortednessPercent"]) for row in trace_rows], dtype=float)


def trajectory_proxy_metrics(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sortedness = sortedness_trajectory(trace_rows)
    aggregation_values = np.asarray([float(row["aggregation"]) for row in trace_rows], dtype=float)
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


def run_stop_policy(
    *,
    initial_values: list[int],
    initial_algotypes: list[str],
    reverse_directions: list[bool],
    frozen_positions: list[int],
    labels: list[int],
    condition: S13Condition,
    policy: StopPolicy,
    replicate_index: int,
    scheduler_seed: int,
    tie_seed: int,
) -> StopRunResult:
    sim = DeterministicEventSimulator(
        initial_values,
        initial_algotypes,
        labels=labels,
        reverse_directions=reverse_directions,
        frozen_positions=frozen_positions,
        frozen_variant=condition.frozen_variant,
        scheduler_seed=scheduler_seed,
        tie_breaker_seed=tie_seed,
        condition_id=f"S13_{condition.condition_template_id}_{policy.stop_policy_id}_rep{replicate_index:03d}",
        implementation="cell_view",
        research_step_id=STEP_ID,
    )
    start = time.perf_counter()
    interval = max(1, int(round(len(initial_values) * policy.no_move_interval_factor)))
    trace_rows = [snapshot_trace_row(sim=sim, condition=condition, policy=policy, replicate_index=replicate_index, event_kind="initial")]
    no_move_checks = 0
    no_move_failed_checks = 0
    last_swap_activation = 0
    last_state_change_activation = 0
    last_state_hash = state_hash(sim.current_values())
    stable_window_triggered = False
    stop_reason = "unknown"
    stop_reason_detail = ""

    while True:
        converged, convergence_reason = convergence_reached(sim, condition, policy)
        if converged:
            stop_reason = convergence_reason
            stop_reason_detail = f"convergence_criterion={policy.convergence_criterion}"
            break
        if sim.activation_count >= policy.max_activations:
            stop_reason = "max_activation_cap"
            stop_reason_detail = f"activation_count={sim.activation_count}; cap={policy.max_activations}"
            break
        if sim.swap_count >= policy.max_swaps:
            stop_reason = "max_step_cap"
            stop_reason_detail = f"swap_count={sim.swap_count}; cap={policy.max_swaps}"
            break
        if sim.comparison_count >= policy.max_comparisons:
            stop_reason = "max_comparison_cap"
            stop_reason_detail = f"comparison_count={sim.comparison_count}; cap={policy.max_comparisons}"
            break

        if policy.stable_state_definition == "legal_action_absent" and sim.activation_count % interval == 0:
            if not sim.legal_action_exists():
                no_move_checks += 1
                if no_move_checks >= policy.no_move_checks_required:
                    stop_reason = "no_cell_can_move_after_required_checks"
                    stop_reason_detail = (
                        f"no_move_checks={no_move_checks}; required={policy.no_move_checks_required}; "
                        f"interval={interval}"
                    )
                    stable_window_triggered = True
                    break
            else:
                no_move_failed_checks += 1
                no_move_checks = 0
        elif policy.stable_state_definition == "no_swap_window" and sim.activation_count > 0:
            window = int(policy.stable_window_activations or len(initial_values))
            if sim.activation_count - last_swap_activation >= window:
                stop_reason = "no_swap_window_stable"
                stop_reason_detail = f"no accepted swaps for {window} activations"
                stable_window_triggered = True
                break
        elif policy.stable_state_definition == "state_hash_plateau" and sim.activation_count > 0:
            window = int(policy.stable_window_activations or len(initial_values))
            if sim.activation_count - last_state_change_activation >= window:
                stop_reason = "state_hash_plateau_stable"
                stop_reason_detail = f"value-state hash unchanged for {window} activations"
                stable_window_triggered = True
                break

        outcome = sim.step()
        if not outcome.activated:
            stop_reason = "no_eligible_cells"
            stop_reason_detail = outcome.reason
            break
        current_hash = state_hash(sim.current_values())
        if current_hash != last_state_hash:
            last_state_hash = current_hash
            last_state_change_activation = int(sim.activation_count)
        if outcome.swapped:
            last_swap_activation = int(sim.activation_count)
            trace_rows.append(
                snapshot_trace_row(
                    sim=sim,
                    condition=condition,
                    policy=policy,
                    replicate_index=replicate_index,
                    event_kind="swap",
                )
            )

    trace_rows.append(
        snapshot_trace_row(
            sim=sim,
            condition=condition,
            policy=policy,
            replicate_index=replicate_index,
            event_kind="final",
            stop_reason=stop_reason,
        )
    )
    values = sim.current_values()
    global_inc = sortedness_raw(values, "increasing") == len(values) - 1
    global_dec = sortedness_raw(values, "decreasing") == len(values) - 1
    accepted_as_converged = stop_reason in {"sorted", "goal_aware_converged", "either_global_direction_sorted"} or (
        policy.accepts_stable_as_converged
        and stop_reason
        in {
            "no_cell_can_move_after_required_checks",
            "no_swap_window_stable",
            "state_hash_plateau_stable",
        }
    )
    criterion_completed = bool(accepted_as_converged)
    return StopRunResult(
        stop_reason=stop_reason,
        stop_reason_detail=stop_reason_detail,
        global_increasing_sorted=bool(global_inc),
        global_decreasing_sorted=bool(global_dec),
        criterion_completed=criterion_completed,
        accepted_as_converged=bool(accepted_as_converged),
        activation_count=int(sim.activation_count),
        swap_count=int(sim.swap_count),
        comparison_count=int(sim.comparison_count),
        archived_compare_and_swap_count=int(sim.archived_compare_and_swap_count),
        blocked_move_attempts=int(sim.blocked_move_attempts),
        frozen_swap_attempts=int(sim.frozen_swap_attempts),
        no_move_check_count=int(no_move_checks),
        no_move_failed_check_count=int(no_move_failed_checks),
        stable_window_triggered=bool(stable_window_triggered),
        last_swap_activation=int(last_swap_activation),
        last_state_change_activation=int(last_state_change_activation),
        final_values=values,
        final_algotypes=sim.current_algotypes(),
        final_frozen_positions=sim.current_frozen_positions(),
        final_frozen_cell_ids=sim.frozen_cell_ids(),
        trace_rows=trace_rows,
        wall_time_seconds=float(time.perf_counter() - start),
    )


def run_one_condition_policy(
    *,
    condition: S13Condition,
    policy: StopPolicy,
    n: int,
    replicate_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seeds = condition_seeds(condition, replicate_index)
    initial_values = input_values_for_profile("random_unique", n, seeds["inputSeed"])
    initial_algotypes = balanced_algotypes(condition.algotypes, n, seeds["algotypeSeed"])
    reverse_directions = reverse_flags_for_algotypes(initial_algotypes, condition.goal_directions)
    labels = labels_for_algotypes(initial_algotypes)
    frozen_positions = frozen_positions_for_condition(condition, n)
    initial_policy_hash = policy_config_hash(
        initial_algotypes,
        reverse_directions,
        frozen_positions,
        condition.frozen_variant,
        condition.goal_directions,
    )
    result = run_stop_policy(
        initial_values=initial_values,
        initial_algotypes=initial_algotypes,
        reverse_directions=reverse_directions,
        frozen_positions=frozen_positions,
        labels=labels,
        condition=condition,
        policy=policy,
        replicate_index=replicate_index,
        scheduler_seed=seeds["schedulerSeed"],
        tie_seed=seeds["tieBreakerSeed"],
    )
    initial_metrics = prefixed_metrics("initial", initial_values)
    final_metrics = prefixed_metrics("final", result.final_values)
    trajectory_metrics = trajectory_proxy_metrics(result.trace_rows)
    final_goal_metrics = goal_sortedness_by_algotype(result.final_values, result.final_algotypes, condition.goal_directions)
    initial_frozen_cell_ids = [int(pos) for pos in frozen_positions]
    record: dict[str, Any] = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "conditionId": f"S13_{condition.condition_template_id}_{policy.stop_policy_id}_rep{replicate_index:03d}",
        "conditionTemplateId": condition.condition_template_id,
        "conditionClass": condition.condition_class,
        "implementation": "S01_deterministic_event_simulator_custom_stop_loop",
        "algorithm": condition.algorithm_label,
        "inputProfile": "random_unique",
        "n": int(n),
        "replicateIndex": int(replicate_index),
        "replicateNumber": int(replicate_index + 1),
        "inputSeed": int(seeds["inputSeed"]),
        "algotypeAssignmentSeed": int(seeds["algotypeSeed"]),
        "schedulerSeed": int(seeds["schedulerSeed"]),
        "tieBreakerSeed": int(seeds["tieBreakerSeed"]),
        "initialValuesHash": state_hash(initial_values),
        "initialPolicyHash": initial_policy_hash,
        "initialAlgotypesHash": hashlib.sha256(compact_json(initial_algotypes).encode("utf-8")).hexdigest(),
        "initialReverseDirectionsHash": hashlib.sha256(compact_json(reverse_directions).encode("utf-8")).hexdigest(),
        "initialFrozenPositionsHash": hashlib.sha256(compact_json(frozen_positions).encode("utf-8")).hexdigest(),
        "initialValues": compact_json(initial_values),
        "finalValues": compact_json(result.final_values),
        "initialAlgotypes": compact_json(initial_algotypes),
        "finalAlgotypes": compact_json(result.final_algotypes),
        "reverseDirections": compact_json(reverse_directions),
        "goalDirections": compact_json(condition.goal_directions),
        "initialAlgotypeCounts": compact_json(dict(sorted(Counter(initial_algotypes).items()))),
        "finalAlgotypeCounts": compact_json(dict(sorted(Counter(result.final_algotypes).items()))),
        "frozenVariant": condition.frozen_variant,
        "frozenCount": int(condition.frozen_count),
        "initialFrozenPositions": compact_json(frozen_positions),
        "finalFrozenPositions": compact_json(result.final_frozen_positions),
        "initialFrozenCellIds": compact_json(initial_frozen_cell_ids),
        "finalFrozenCellIds": compact_json(result.final_frozen_cell_ids),
        "stopPolicyId": policy.stop_policy_id,
        "stopPolicyFamily": policy.policy_family,
        "stopPolicyDescription": policy.description,
        "maxActivationCap": int(policy.max_activations),
        "maxSwapCap": int(policy.max_swaps),
        "maxComparisonCap": int(policy.max_comparisons),
        "noMoveChecksRequired": int(policy.no_move_checks_required),
        "noMoveCheckInterval": int(max(1, round(n * policy.no_move_interval_factor))),
        "stableStateDefinition": policy.stable_state_definition,
        "stableWindowActivations": policy.stable_window_activations,
        "convergenceCriterion": policy.convergence_criterion,
        "acceptsStableAsConverged": bool(policy.accepts_stable_as_converged),
        "completed": bool(result.criterion_completed),
        "criterionCompleted": bool(result.criterion_completed),
        "acceptedAsConverged": bool(result.accepted_as_converged),
        "globalIncreasingSorted": bool(result.global_increasing_sorted),
        "globalDecreasingSorted": bool(result.global_decreasing_sorted),
        "stopReason": result.stop_reason,
        "stopReasonDetail": result.stop_reason_detail,
        "capStop": result.stop_reason.startswith("max_"),
        "activationCapHit": result.stop_reason == "max_activation_cap",
        "swapCapHit": result.stop_reason == "max_step_cap",
        "comparisonCapHit": result.stop_reason == "max_comparison_cap",
        "stableStop": result.stop_reason
        in {
            "no_cell_can_move_after_required_checks",
            "no_swap_window_stable",
            "state_hash_plateau_stable",
        },
        "stableWindowTriggered": bool(result.stable_window_triggered),
        "noMoveCheckCount": int(result.no_move_check_count),
        "noMoveFailedCheckCount": int(result.no_move_failed_check_count),
        "lastSwapActivation": int(result.last_swap_activation),
        "lastStateChangeActivation": int(result.last_state_change_activation),
        "swapCount": int(result.swap_count),
        "comparisonCount": int(result.comparison_count),
        "archivedCompareAndSwapCount": int(result.archived_compare_and_swap_count),
        "activationCount": int(result.activation_count),
        "eventCount": int(len(result.trace_rows)),
        "blockedMoveAttempts": int(result.blocked_move_attempts),
        "frozenSwapAttempts": int(result.frozen_swap_attempts),
        "finalValuesHash": state_hash(result.final_values),
        "valueCountPreserved": Counter(initial_values) == Counter(result.final_values),
        "algotypeCountPreserved": Counter(initial_algotypes) == Counter(result.final_algotypes),
        "frozenCountPreserved": len(result.final_frozen_cell_ids) == int(condition.frozen_count),
        "finalGoalSortednessByAlgotype": compact_json(final_goal_metrics),
        "wallTimeSeconds": float(result.wall_time_seconds),
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
        out = dict(row)
        out["conditionId"] = record["conditionId"]
        out["initialValuesHash"] = record["initialValuesHash"]
        out["initialPolicyHash"] = initial_policy_hash
        trace_rows.append(out)
    return record, trace_rows


def attach_canonical_sensitivity(result_df: pd.DataFrame) -> pd.DataFrame:
    df = result_df.copy()
    reference = df[df["stopPolicyId"].eq("canonical_s01")][
        [
            "conditionTemplateId",
            "replicateIndex",
            "stopReason",
            "finalValuesHash",
            "criterionCompleted",
            "globalIncreasingSorted",
            "finalSortednessPercent",
            "finalKendallTauDistanceNormalized",
            "activationCount",
            "swapCount",
            "comparisonCount",
        ]
    ].rename(
        columns={
            "stopReason": "canonicalStopReason",
            "finalValuesHash": "canonicalFinalValuesHash",
            "criterionCompleted": "canonicalCriterionCompleted",
            "globalIncreasingSorted": "canonicalGlobalIncreasingSorted",
            "finalSortednessPercent": "canonicalFinalSortednessPercent",
            "finalKendallTauDistanceNormalized": "canonicalFinalKendallTauDistanceNormalized",
            "activationCount": "canonicalActivationCount",
            "swapCount": "canonicalSwapCount",
            "comparisonCount": "canonicalComparisonCount",
        }
    )
    df = df.merge(reference, on=["conditionTemplateId", "replicateIndex"], how="left")
    df["stopReasonChangedVsCanonical"] = df["stopReason"] != df["canonicalStopReason"]
    df["finalStateChangedVsCanonical"] = df["finalValuesHash"] != df["canonicalFinalValuesHash"]
    df["criterionCompletedChangedVsCanonical"] = df["criterionCompleted"] != df["canonicalCriterionCompleted"]
    df["globalSortedChangedVsCanonical"] = df["globalIncreasingSorted"] != df["canonicalGlobalIncreasingSorted"]
    df["finalSortednessDeltaVsCanonical"] = df["finalSortednessPercent"] - df["canonicalFinalSortednessPercent"]
    df["finalKendallDistanceDeltaVsCanonical"] = (
        df["finalKendallTauDistanceNormalized"] - df["canonicalFinalKendallTauDistanceNormalized"]
    )
    df["activationDeltaVsCanonical"] = df["activationCount"] - df["canonicalActivationCount"]
    df["swapDeltaVsCanonical"] = df["swapCount"] - df["canonicalSwapCount"]
    df["comparisonDeltaVsCanonical"] = df["comparisonCount"] - df["canonicalComparisonCount"]
    return df


def run_s13_matrix(*, n: int, replicates: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    conditions = s13_conditions()
    policies = stop_policies(n)
    records: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for condition in conditions:
        for replicate_index in range(replicates):
            for policy in policies:
                record, traces = run_one_condition_policy(
                    condition=condition,
                    policy=policy,
                    n=n,
                    replicate_index=replicate_index,
                )
                records.append(record)
                trace_rows.extend(traces)
    result_df = attach_canonical_sensitivity(pd.DataFrame(records))
    return result_df, pd.DataFrame(trace_rows)


def summarize_results(result_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        result_df.groupby(["conditionClass", "conditionTemplateId", "stopPolicyId", "stopPolicyFamily"], dropna=False)
        .agg(
            runCount=("conditionId", "size"),
            completedRate=("criterionCompleted", "mean"),
            globalIncreasingSortedRate=("globalIncreasingSorted", "mean"),
            capStopRate=("capStop", "mean"),
            stableStopRate=("stableStop", "mean"),
            stopReasonChangedRate=("stopReasonChangedVsCanonical", "mean"),
            finalStateChangedRate=("finalStateChangedVsCanonical", "mean"),
            criterionCompletedChangedRate=("criterionCompletedChangedVsCanonical", "mean"),
            finalSortednessPercentMean=("finalSortednessPercent", "mean"),
            finalKendallDistanceMean=("finalKendallTauDistanceNormalized", "mean"),
            activationCountMean=("activationCount", "mean"),
            swapCountMean=("swapCount", "mean"),
            comparisonCountMean=("comparisonCount", "mean"),
            delayedGratificationMean=("delayedGratification", "mean"),
            wallTimeSecondsMean=("wallTimeSeconds", "mean"),
        )
        .reset_index()
        .sort_values(["conditionClass", "conditionTemplateId", "stopPolicyId"])
    )
    summary.insert(0, "researchStepId", STEP_ID)
    summary.insert(0, "experimentId", EXPERIMENT_ID)
    return summary


def summarize_policy_effects(result_df: pd.DataFrame) -> pd.DataFrame:
    effects = (
        result_df.groupby(["stopPolicyId", "stopPolicyFamily"], dropna=False)
        .agg(
            runCount=("conditionId", "size"),
            completedRate=("criterionCompleted", "mean"),
            globalIncreasingSortedRate=("globalIncreasingSorted", "mean"),
            capStopRate=("capStop", "mean"),
            stableStopRate=("stableStop", "mean"),
            stopReasonChangedRate=("stopReasonChangedVsCanonical", "mean"),
            finalStateChangedRate=("finalStateChangedVsCanonical", "mean"),
            criterionCompletedChangedRate=("criterionCompletedChangedVsCanonical", "mean"),
            maxAbsFinalSortednessDelta=("finalSortednessDeltaVsCanonical", lambda x: float(np.nanmax(np.abs(x)))),
            maxAbsFinalKendallDelta=("finalKendallDistanceDeltaVsCanonical", lambda x: float(np.nanmax(np.abs(x)))),
            meanActivationDelta=("activationDeltaVsCanonical", "mean"),
            meanSwapDelta=("swapDeltaVsCanonical", "mean"),
        )
        .reset_index()
        .sort_values(["stopPolicyFamily", "stopPolicyId"])
    )
    effects.insert(0, "researchStepId", STEP_ID)
    effects.insert(0, "experimentId", EXPERIMENT_ID)
    return effects


def load_context_tables(s12_path: Path, e01_opposite_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    s12 = pd.read_parquet(s12_path) if s12_path.exists() else pd.DataFrame()
    e01 = pd.read_parquet(e01_opposite_path) if e01_opposite_path.exists() else pd.DataFrame()
    if not s12.empty:
        keep = [
            "conditionId",
            "conditionTemplateId",
            "algorithm",
            "placementCategory",
            "defectBehaviorVariant",
            "completed",
            "stopReason",
            "activationCount",
            "swapCount",
            "comparisonCount",
            "finalSortednessPercent",
            "finalKendallTauDistanceNormalized",
        ]
        s12 = s12[[col for col in keep if col in s12.columns]].copy()
        s12.insert(0, "contextForResearchStepId", STEP_ID)
    if not e01.empty:
        e01 = e01.copy()
        e01.insert(0, "contextForResearchStepId", STEP_ID)
    return s12, e01


def diagnostics_table(
    *,
    result_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    policy_effect_df: pd.DataFrame,
    s12_context_df: pd.DataFrame,
    e01_context_df: pd.DataFrame,
    expected_rows: int,
    repo_tests_passed: bool,
) -> pd.DataFrame:
    seed_groups = result_df.groupby(["conditionTemplateId", "replicateIndex"], dropna=False)
    matched_initial_values = seed_groups["initialValuesHash"].nunique().max() == 1
    matched_policies = seed_groups["initialPolicyHash"].nunique().max() == 1
    canonical = result_df[result_df["stopPolicyId"].eq("canonical_s01")]
    cap_profiles = result_df[result_df["stopPolicyFamily"].eq("event_cap")]
    checks = [
        ("expected_row_count", len(result_df) == expected_rows, f"Stop-condition table has expected {expected_rows} rows."),
        ("baseline_frozen_conflict_represented", set(result_df["conditionClass"]) == {"baseline", "frozen_cell", "conflict_chimera"}, "Baseline, Frozen Cell, and conflict-chimera condition classes are represented."),
        ("all_stop_policy_families_represented", {"canonical", "no_movement_window", "event_cap", "convergence_criteria", "stable_state_definition"}.issubset(set(result_df["stopPolicyFamily"])), "No-movement windows, event caps, convergence criteria, and stable-state definitions are represented."),
        ("matched_initial_arrays", bool(matched_initial_values), "Initial arrays are matched across stop policies within every condition/replicate."),
        ("matched_policy_assignments", bool(matched_policies), "Initial Algotypes, reverse directions, Frozen Cell placements, and goal directions are matched across stop policies."),
        ("stop_reasons_logged", bool(result_df["stopReason"].notna().all()), "Every run records a stop reason."),
        ("caps_logged", bool(result_df[["maxActivationCap", "maxSwapCap", "maxComparisonCap"]].notna().all().all()), "Every run records activation, swap, and comparison caps."),
        ("event_cap_profiles_trigger_caps", bool(len(cap_profiles) > 0 and cap_profiles["capStop"].any()), "At least one event-cap stress profile triggers a cap stop."),
        ("canonical_rows_complete", len(canonical) == len(s13_conditions()) * result_df["replicateIndex"].nunique(), "Canonical reference rows exist for every condition/replicate."),
        ("value_counts_preserved", bool(result_df["valueCountPreserved"].all()), "All runs preserve value counts."),
        ("algotype_counts_preserved", bool(result_df["algotypeCountPreserved"].all()), "All runs preserve Algotype counts."),
        ("frozen_counts_preserved", bool(result_df["frozenCountPreserved"].all()), "Frozen Cell counts are preserved."),
        ("trace_rows_written", len(trace_df) > 0, "Trace rows are written."),
        ("trace_stop_policies_represented", set(result_df["stopPolicyId"]).issubset(set(trace_df["stopPolicyId"])) if len(trace_df) else False, "Trace rows cover every stop policy."),
        ("summary_written", len(summary_df) > 0, "Stop-condition summary table is nonempty."),
        ("policy_effects_written", len(policy_effect_df) > 0, "Stop-policy effect table is nonempty."),
        ("s12_context_loaded", len(s12_context_df) > 0, "S12 Frozen Cell behavior context table was loaded."),
        ("e01_opposite_context_loaded", len(e01_context_df) > 0, "E01 opposite-direction chimera context table was loaded."),
        ("repo_tests_passed", bool(repo_tests_passed), "Repository S13 unit tests passed."),
    ]
    return pd.DataFrame(
        [
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "check": check,
                "passed": bool(passed),
                "detail": detail,
            }
            for check, passed, detail in checks
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
    artifacts_dir: Path,
    result_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    policy_effect_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    s12_context_df: pd.DataFrame,
    e01_context_df: pd.DataFrame,
) -> dict[str, Path]:
    result_dir = artifacts_dir / "results"
    trace_dir = artifacts_dir / "traces" / "e02" / STEP_ID
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    step_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results_parquet": result_dir / "e02_stop_condition_sensitivity.parquet",
        "results_csv": result_dir / "e02_stop_condition_sensitivity.csv",
        "summary_parquet": result_dir / "e02_stop_condition_summary.parquet",
        "summary_csv": result_dir / "e02_stop_condition_summary.csv",
        "policy_effects_parquet": result_dir / "e02_stop_condition_policy_effects.parquet",
        "policy_effects_csv": result_dir / "e02_stop_condition_policy_effects.csv",
        "diagnostics_parquet": result_dir / "e02_stop_condition_diagnostics.parquet",
        "diagnostics_csv": result_dir / "e02_stop_condition_diagnostics.csv",
        "s12_context_parquet": result_dir / "e02_stop_condition_s12_context.parquet",
        "s12_context_csv": result_dir / "e02_stop_condition_s12_context.csv",
        "e01_context_parquet": result_dir / "e02_stop_condition_e01_opposite_context.parquet",
        "e01_context_csv": result_dir / "e02_stop_condition_e01_opposite_context.csv",
        "trace_parquet": trace_dir / "e02_stop_condition_trace_events.parquet",
        "trace_csv_gz": trace_dir / "e02_stop_condition_trace_events.csv.gz",
    }
    result_df.to_parquet(paths["results_parquet"], index=False)
    result_df.to_csv(paths["results_csv"], index=False)
    summary_df.to_parquet(paths["summary_parquet"], index=False)
    summary_df.to_csv(paths["summary_csv"], index=False)
    policy_effect_df.to_parquet(paths["policy_effects_parquet"], index=False)
    policy_effect_df.to_csv(paths["policy_effects_csv"], index=False)
    diagnostics_df.to_parquet(paths["diagnostics_parquet"], index=False)
    diagnostics_df.to_csv(paths["diagnostics_csv"], index=False)
    s12_context_df.to_parquet(paths["s12_context_parquet"], index=False)
    s12_context_df.to_csv(paths["s12_context_csv"], index=False)
    e01_context_df.to_parquet(paths["e01_context_parquet"], index=False)
    e01_context_df.to_csv(paths["e01_context_csv"], index=False)
    trace_df.to_parquet(paths["trace_parquet"], index=False)
    trace_df.to_csv(paths["trace_csv_gz"], index=False, compression="gzip")
    return paths


def plot_outputs(policy_effect_df: pd.DataFrame, summary_df: pd.DataFrame, figure_dir: Path) -> tuple[Path, Path, Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    order = [policy.stop_policy_id for policy in stop_policies(30)]
    plot_df = policy_effect_df.set_index("stopPolicyId").reindex(order).reset_index()
    labels = [value.replace("_", "\n") for value in plot_df["stopPolicyId"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    axes[0].bar(range(len(plot_df)), plot_df["stopReasonChangedRate"], color="#4c78a8")
    axes[1].bar(range(len(plot_df)), plot_df["finalStateChangedRate"], color="#f58518")
    axes[2].bar(range(len(plot_df)), plot_df["capStopRate"], color="#54a24b")
    for ax, ylabel, title in [
        (axes[0], "rate", "Stop reason changed vs canonical"),
        (axes[1], "rate", "Final state changed vs canonical"),
        (axes[2], "rate", "Cap stops"),
    ]:
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    sensitivity_png = figure_dir / "e02_stop_condition_sensitivity_summary.png"
    sensitivity_pdf = figure_dir / "e02_stop_condition_sensitivity_summary.pdf"
    fig.savefig(sensitivity_png, dpi=180)
    fig.savefig(sensitivity_pdf)
    plt.close(fig)

    class_df = (
        summary_df.groupby(["conditionClass", "stopPolicyId"], dropna=False)
        .agg(finalKendallDistanceMean=("finalKendallDistanceMean", "mean"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    for condition_class in ["baseline", "frozen_cell", "conflict_chimera"]:
        subset = class_df[class_df["conditionClass"].eq(condition_class)].set_index("stopPolicyId").reindex(order)
        ax.plot(range(len(order)), subset["finalKendallDistanceMean"], marker="o", label=condition_class)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("mean final Kendall distance")
    ax.set_title("Final order distance by condition class and stop policy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    distance_png = figure_dir / "e02_stop_condition_final_distance.png"
    distance_pdf = figure_dir / "e02_stop_condition_final_distance.pdf"
    fig.savefig(distance_png, dpi=180)
    fig.savefig(distance_pdf)
    plt.close(fig)
    return sensitivity_png, sensitivity_pdf, distance_png, distance_pdf


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.exists() and path.is_file():
            records.append({"path": str(path), "sha256": sha256_path(path), "sizeBytes": path.stat().st_size})
    return sorted(records, key=lambda row: row["path"])


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_dir = step_dir / "code"
    paths = [
        (REPO_ROOT / "scripts" / "e02_s13_stop_condition_sensitivity.py", code_dir / "scripts" / "e02_s13_stop_condition_sensitivity.py"),
        (REPO_ROOT / "tests" / "test_e02_stop_condition_sensitivity.py", code_dir / "tests" / "test_e02_stop_condition_sensitivity.py"),
        (REPO_ROOT / "scripts" / "e02_s12_frozen_behavior_variants.py", code_dir / "scripts" / "e02_s12_frozen_behavior_variants.py"),
        (REPO_ROOT / "scripts" / "e02_s10_input_distributions.py", code_dir / "scripts" / "e02_s10_input_distributions.py"),
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
    command = [sys.executable, "-m", "unittest", "tests.test_e02_stop_condition_sensitivity"]
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


def outcome_classification(policy_effect_df: pd.DataFrame, validation_success: bool) -> str:
    if not validation_success:
        return "null"
    noncanonical = policy_effect_df[~policy_effect_df["stopPolicyId"].eq("canonical_s01")]
    if noncanonical.empty:
        return "null"
    max_state_changed = float(noncanonical["finalStateChangedRate"].max())
    max_completion_changed = float(noncanonical["criterionCompletedChangedRate"].max())
    max_distance_delta = float(noncanonical["maxAbsFinalKendallDelta"].max())
    if max_state_changed > 0.1 or max_completion_changed > 0.1 or max_distance_delta > 0.02:
        return "constraining/contradictory"
    return "supportive"


def write_validation_report(step_dir: Path, diagnostics_df: pd.DataFrame, validation: dict[str, Any]) -> Path:
    lines = [
        "# E02 S13 Validation Report",
        "",
        "- Research step ID: S13",
        "- Completion status: completed" if validation["success"] else "- Completion status: completed with validation failures",
        "- Artifacts written: stop-condition sensitivity table, summary tables, policy-effect table, diagnostics, trace table, context tables, figures, copied code, manifest, status JSON, and run manifest.",
        f"- Validation result: {validation['validationResult']}",
        "- Caveats or blockers: S13 changes only stop criteria around the S01 simulator; no sorting policy logic is changed. Cap-truncated and stability-accepted rows are sensitivity probes, not direct biological or original-threaded claims.",
        "- Recommended next action: stop before S14 for Chief Scientist review.",
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
    policy_effect_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    s12_context_df: pd.DataFrame,
    e01_context_df: pd.DataFrame,
    table_paths: dict[str, Path],
    figure_paths: tuple[Path, ...],
    code_paths: list[Path],
    repo_tests: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Path]:
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    validation_path = write_validation_report(step_dir, diagnostics_df, validation)
    outcome = outcome_classification(policy_effect_df, validation["success"])
    artifact_candidates = list(table_paths.values()) + list(figure_paths) + code_paths + [validation_path, Path(repo_tests["logPath"])]
    manifest_path = step_dir / "artifact_manifest.json"
    status_path = step_dir / "status.json"
    summary_path = step_dir / "summary.md"
    run_manifest_path = artifacts_dir / "provenance" / "run_manifest.json"
    artifact_records = collect_artifacts(artifact_candidates)
    max_state_changed = float(policy_effect_df.loc[~policy_effect_df["stopPolicyId"].eq("canonical_s01"), "finalStateChangedRate"].max())
    max_completion_changed = float(policy_effect_df.loc[~policy_effect_df["stopPolicyId"].eq("canonical_s01"), "criterionCompletedChangedRate"].max())
    max_distance_delta = float(policy_effect_df.loc[~policy_effect_df["stopPolicyId"].eq("canonical_s01"), "maxAbsFinalKendallDelta"].max())
    cap_stop_rows = int(result_df["capStop"].sum())
    stable_stop_rows = int(result_df["stableStop"].sum())
    stop_reason_counts = result_df["stopReason"].value_counts().sort_index().to_dict()
    caveats = (
        "S13 uses the deterministic S01 simulator and bounded n=30 representative baseline, Frozen Cell, "
        "and opposite-goal conflict-chimera conditions. Stop policies intentionally include cap-truncated "
        "and stability-accepted definitions, so completion fields must be read with the stopPolicyId and "
        "convergenceCriterion columns."
    )
    recommended = "Stop before S14 for Chief Scientist review; if accepted, proceed to S14 stronger statistics."
    lay_summary = (
        "S13 held arrays and cell policies fixed while changing only when a run is allowed to stop. "
        "The audit shows which convergence or failure labels are robust and which are artifacts of caps, "
        "no-move windows, or accepting partial stable states."
    )
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(validation["success"]),
        "status": "completed" if validation["success"] else "completed_with_validation_failures",
        "artifactsWritten": [record["path"] for record in artifact_records],
        "validationResult": validation["validationResult"],
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended,
        "completionStatus": "completed" if validation["success"] else "completed with validation failures",
        "outcomeClassification": outcome,
        "completedAt": utc_now(),
        "durationSeconds": float(time.perf_counter() - started_at),
        "resultRows": int(len(result_df)),
        "traceRows": int(len(trace_df)),
        "summaryRows": int(len(summary_df)),
        "policyEffectRows": int(len(policy_effect_df)),
        "diagnosticRows": int(len(diagnostics_df)),
        "s12ContextRows": int(len(s12_context_df)),
        "e01OppositeContextRows": int(len(e01_context_df)),
        "capStopRows": cap_stop_rows,
        "stableStopRows": stable_stop_rows,
        "stopReasonCounts": stop_reason_counts,
        "maxFinalStateChangedRateVsCanonical": max_state_changed,
        "maxCriterionCompletedChangedRateVsCanonical": max_completion_changed,
        "maxAbsFinalKendallDeltaVsCanonical": max_distance_delta,
        "repoTests": repo_tests,
        "git": get_git_metadata(),
        "python": sys.version,
        "platform": platform.platform(),
        "environment": {
            "ARTIFACTS_DIR": str(artifacts_dir),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "laySummary": lay_summary,
    }
    write_json(status_path, status_payload)
    manifest_payload = {
        "schema": "eidosoma.artifact_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "artifacts": artifact_records,
    }
    write_json(manifest_path, manifest_payload)
    artifact_records = collect_artifacts(artifact_candidates + [manifest_path, status_path])
    manifest_payload["artifacts"] = artifact_records
    write_json(manifest_path, manifest_payload)
    status_payload["artifactsWritten"] = [record["path"] for record in artifact_records]
    write_json(status_path, status_payload)
    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "lastResearchStepId": STEP_ID,
        "lastStepNumber": STEP_NUMBER,
        "updatedAt": utc_now(),
        "git": get_git_metadata(),
        "artifacts": artifact_records,
    }
    write_json(run_manifest_path, run_manifest)

    preview = policy_effect_df.sort_values("stopPolicyId")[
        [
            "stopPolicyId",
            "completedRate",
            "capStopRate",
            "stopReasonChangedRate",
            "finalStateChangedRate",
            "maxAbsFinalKendallDelta",
        ]
    ].head(12)
    lines = [
        "# E02 S13 Summary",
        "",
        "- Research step ID: S13",
        "- Completion status: completed" if validation["success"] else "- Completion status: completed with validation failures",
        "- Artifacts written: `$ARTIFACTS_DIR/results/e02_stop_condition_sensitivity.parquet`, stop-condition summaries, policy-effect table, diagnostics, context tables, trace table, figures, copied code, manifest, status JSON, and run manifest.",
        f"- Validation result: {validation['validationResult']}",
        f"- Caveats or blockers: {caveats}",
        f"- Lay summary: {lay_summary}",
        f"- Recommended next action: {recommended}",
        f"- Outcome classification: {outcome}",
        "",
        "## Run Counts",
        "",
        markdown_table(
            ["Item", "Count"],
            [
                ["Stop-condition rows", len(result_df)],
                ["Trace rows", len(trace_df)],
                ["Summary rows", len(summary_df)],
                ["Policy-effect rows", len(policy_effect_df)],
                ["Cap-stop rows", cap_stop_rows],
                ["Stable-stop rows", stable_stop_rows],
                ["Diagnostics", len(diagnostics_df)],
            ],
        ),
        "",
        "## Stop-Policy Effects",
        "",
        markdown_table(
            ["Stop policy", "Completed", "Cap stops", "Stop changed", "State changed", "Max Kendall delta"],
            [
                [
                    row.stopPolicyId,
                    row.completedRate,
                    row.capStopRate,
                    row.stopReasonChangedRate,
                    row.finalStateChangedRate,
                    row.maxAbsFinalKendallDelta,
                ]
                for row in preview.itertuples(index=False)
            ],
        ),
        "",
        "## Validation",
    ]
    for row in diagnostics_df.itertuples(index=False):
        lines.append(f"- {'passed' if row.passed else 'failed'}: {row.detail}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact_records = collect_artifacts(artifact_candidates + [manifest_path, status_path, summary_path])
    manifest_payload["artifacts"] = artifact_records
    write_json(manifest_path, manifest_payload)
    status_payload["artifactsWritten"] = [record["path"] for record in artifact_records]
    write_json(status_path, status_payload)
    run_manifest["artifacts"] = artifact_records
    write_json(run_manifest_path, run_manifest)
    return {
        "status": status_path,
        "summary": summary_path,
        "validation": validation_path,
        "manifest": manifest_path,
        "run_manifest": run_manifest_path,
    }


def run_s13(args: argparse.Namespace) -> int:
    started_at = time.perf_counter()
    artifacts_dir = Path(args.artifacts_dir)
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    step_dir.mkdir(parents=True, exist_ok=True)
    result_df, trace_df = run_s13_matrix(n=int(args.n), replicates=int(args.replicates))
    summary_df = summarize_results(result_df)
    policy_effect_df = summarize_policy_effects(result_df)
    s12_context_df, e01_context_df = load_context_tables(Path(args.s12_context), Path(args.e01_opposite_context))
    code_paths = copy_code_artifacts(step_dir)
    repo_tests = run_repo_tests(step_dir)
    expected_rows = len(s13_conditions()) * int(args.replicates) * len(stop_policies(int(args.n)))
    diagnostics_df = diagnostics_table(
        result_df=result_df,
        trace_df=trace_df,
        summary_df=summary_df,
        policy_effect_df=policy_effect_df,
        s12_context_df=s12_context_df,
        e01_context_df=e01_context_df,
        expected_rows=expected_rows,
        repo_tests_passed=bool(repo_tests["passed"]),
    )
    validation = validate_outputs(diagnostics_df)
    table_paths = write_tables(
        artifacts_dir=artifacts_dir,
        result_df=result_df,
        trace_df=trace_df,
        summary_df=summary_df,
        policy_effect_df=policy_effect_df,
        diagnostics_df=diagnostics_df,
        s12_context_df=s12_context_df,
        e01_context_df=e01_context_df,
    )
    figure_paths = plot_outputs(policy_effect_df, summary_df, artifacts_dir / "figures" / "e02")
    report_paths = write_reports_and_manifests(
        artifacts_dir=artifacts_dir,
        started_at=started_at,
        result_df=result_df,
        trace_df=trace_df,
        summary_df=summary_df,
        policy_effect_df=policy_effect_df,
        diagnostics_df=diagnostics_df,
        s12_context_df=s12_context_df,
        e01_context_df=e01_context_df,
        table_paths=table_paths,
        figure_paths=figure_paths,
        code_paths=code_paths,
        repo_tests=repo_tests,
        validation=validation,
    )
    print(
        json.dumps(
            {
                "researchStepId": STEP_ID,
                "success": validation["success"],
                "validationResult": validation["validationResult"],
                "resultRows": len(result_df),
                "traceRows": len(trace_df),
                "statusPath": str(report_paths["status"]),
            },
            indent=2,
        )
    )
    return 0 if validation["success"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", default=os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--s12-context", default=str(DEFAULT_S12_BEHAVIOR_PATH))
    parser.add_argument("--e01-opposite-context", default=str(DEFAULT_E01_OPPOSITE_PATH))
    return parser


def main() -> int:
    return run_s13(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
