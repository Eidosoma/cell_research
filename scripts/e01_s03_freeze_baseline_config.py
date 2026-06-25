#!/usr/bin/env python3
"""Freeze the E01 S03 baseline configuration and dry-run matrix.

This is a configuration/dry-run generator only. It intentionally does not run
any S04 simulations or large sweeps.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "E01"
STEP_ID = "S03"
STEP_NUMBER = 3
STATUS = "completed"
OUTCOME_CLASSIFICATION = "supportive"
CONFIG_VERSION = "e01_baseline_config.v1"
REPEAT_COUNT = 100
UNIQUE_N = 100
ALGORITHMS = ["bubble", "insertion", "selection"]
IMPLEMENTATIONS = ["traditional", "cell_view"]


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
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - defensive provenance path
        return {
            "args": args,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
        }


def get_git_metadata(repo_root: Path) -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    branch = run_command(["git", "branch", "--show-current"], cwd=repo_root)
    status = run_command(["git", "status", "--short"], cwd=repo_root)
    remote = run_command(["git", "remote", "-v"], cwd=repo_root)
    return {
        "commit": commit["stdout"] if commit["ok"] else "unknown",
        "branch": branch["stdout"] if branch["ok"] else "unknown",
        "dirtyStatus": status["stdout"],
        "remote": remote["stdout"],
    }


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def slug(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("+", "plus")
        .replace("/", "_")
    )


def json_string(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def yaml_dump(value: Any, indent: int = 0) -> str:
    space = " " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{space}{key}:")
                lines.append(yaml_dump(item, indent + 2))
            else:
                lines.append(f"{space}{key}: {yaml_scalar(item)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{space}-")
                lines.append(yaml_dump(item, indent + 2))
            else:
                lines.append(f"{space}- {yaml_scalar(item)}")
    else:
        lines.append(f"{space}{yaml_scalar(value)}")
    return "\n".join(lines)


def seed_for(condition_id: str, replicate_index: int, stream: str) -> int:
    text = f"{CONFIG_VERSION}:{condition_id}:{replicate_index}:{stream}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def seed_for_parts(*parts: Any) -> int:
    text = ":".join([CONFIG_VERSION, *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def allocation_for_mixture(mixture_id: str, n: int = UNIQUE_N) -> dict[str, int]:
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
        return {"bubble": base + (1 if n % 3 > 0 else 0), "insertion": base + (1 if n % 3 > 1 else 0), "selection": base}
    if mixture_id == "bubble_down_selection_up":
        return {"bubble": n // 2, "selection": n - n // 2}
    if mixture_id == "bubble_up_insertion_down":
        return {"bubble": n // 2, "insertion": n - n // 2}
    if mixture_id == "selection_down_insertion_up":
        return {"selection": n // 2, "insertion": n - n // 2}
    raise ValueError(f"unknown mixture_id: {mixture_id}")


def direction_for_mixture(mixture_id: str) -> dict[str, str]:
    if mixture_id == "bubble_down_selection_up":
        return {"bubble": "decreasing", "selection": "increasing"}
    if mixture_id == "bubble_up_insertion_down":
        return {"bubble": "increasing", "insertion": "decreasing"}
    if mixture_id == "selection_down_insertion_up":
        return {"selection": "decreasing", "insertion": "increasing"}
    allocation = allocation_for_mixture(mixture_id)
    return {algorithm: "increasing" for algorithm in allocation}


def algorithms_for_mixture(mixture_id: str) -> list[str]:
    return list(allocation_for_mixture(mixture_id).keys())


def new_condition(
    *,
    producer_step: str,
    condition_id: str,
    purpose: str,
    implementation: str,
    mixture_id: str,
    input_profile: str,
    frozen_variant: str,
    frozen_count: int,
    goal_pattern: str,
    stop_policy: str,
    trace_consumers: list[str],
    wrapper_requirement: str,
    notes: str,
) -> dict[str, Any]:
    allocation = allocation_for_mixture(mixture_id)
    directions = direction_for_mixture(mixture_id)
    return {
        "conditionId": condition_id,
        "producerStep": producer_step,
        "purpose": purpose,
        "implementation": implementation,
        "mixtureId": mixture_id,
        "algorithms": algorithms_for_mixture(mixture_id),
        "algotypeAllocation": allocation,
        "goalDirections": directions,
        "inputProfile": input_profile,
        "n": UNIQUE_N,
        "repeatCount": REPEAT_COUNT,
        "frozenVariant": frozen_variant,
        "frozenCount": frozen_count,
        "goalPattern": goal_pattern,
        "stopPolicy": stop_policy,
        "traceConsumers": trace_consumers,
        "wrapperRequirement": wrapper_requirement,
        "notes": notes,
    }


def build_conditions() -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []

    for implementation in IMPLEMENTATIONS:
        for algorithm in ALGORITHMS:
            conditions.append(
                new_condition(
                    producer_step="S04",
                    condition_id=f"S04_{implementation}_{algorithm}_unique_f0_none",
                    purpose="Figure 3 trajectories and Figure 4 efficiency source traces",
                    implementation=implementation,
                    mixture_id=f"pure_{algorithm}",
                    input_profile="unique_1_100",
                    frozen_variant="none",
                    frozen_count=0,
                    goal_pattern="all_increasing",
                    stop_policy="sorted_or_cap",
                    trace_consumers=["S05", "S06"],
                    wrapper_requirement="traditional wrapper required" if implementation == "traditional" else "cell-view class wrapper required",
                    notes="Matched no-Frozen baseline. Records swap-only and comparison-inclusive counts.",
                )
            )

    for implementation in IMPLEMENTATIONS:
        for algorithm in ALGORITHMS:
            conditions.append(
                new_condition(
                    producer_step="S07",
                    condition_id=f"S07_{implementation}_{algorithm}_unique_f0_none",
                    purpose="Figure 5 Frozen Cell baseline and Figure 7 DG f=0 reference",
                    implementation=implementation,
                    mixture_id=f"pure_{algorithm}",
                    input_profile="unique_1_100",
                    frozen_variant="none",
                    frozen_count=0,
                    goal_pattern="all_increasing",
                    stop_policy="sorted_or_cap",
                    trace_consumers=["S08", "S06"],
                    wrapper_requirement="traditional wrapper required" if implementation == "traditional" else "cell-view class wrapper required",
                    notes="No-Frozen control is included once rather than duplicated under passive and stuck variants.",
                )
            )
            for frozen_variant in ["passive", "stuck"]:
                for frozen_count in [1, 2, 3]:
                    conditions.append(
                        new_condition(
                            producer_step="S07",
                            condition_id=f"S07_{implementation}_{algorithm}_unique_f{frozen_count}_{frozen_variant}",
                            purpose="Figure 5 Frozen Cell robustness and Figure 7 DG",
                            implementation=implementation,
                            mixture_id=f"pure_{algorithm}",
                            input_profile="unique_1_100",
                            frozen_variant=frozen_variant,
                            frozen_count=frozen_count,
                            goal_pattern="all_increasing",
                            stop_policy="sorted_no_move_or_cap",
                            trace_consumers=["S08", "S06"],
                            wrapper_requirement="traditional wrapper required" if implementation == "traditional" else "cell-view class wrapper required",
                            notes="Paper-declared Frozen Cell semantics override ambiguous archived-script behavior.",
                        )
                    )

    same_goal_mixtures = [
        "pure_bubble",
        "pure_insertion",
        "pure_selection",
        "bubble_insertion",
        "bubble_selection",
        "insertion_selection",
        "bubble_insertion_selection",
    ]
    for mixture_id in same_goal_mixtures:
        conditions.append(
            new_condition(
                producer_step="S09",
                condition_id=f"S09_cell_view_{mixture_id}_unique_same_goal",
                purpose="Figure 8 same-goal unique-value chimera traces",
                implementation="cell_view",
                mixture_id=mixture_id,
                input_profile="unique_1_100",
                frozen_variant="none",
                frozen_count=0,
                goal_pattern="all_increasing",
                stop_policy="sorted_no_move_or_cap",
                trace_consumers=["S10", "S06"],
                wrapper_requirement="cell-view chimera wrapper required",
                notes="Pure conditions are same-code controls; all-three uses balanced 34/33/33 allocation because n=100 is not divisible by 3.",
            )
        )

    for mixture_id in ["bubble_insertion", "bubble_selection", "insertion_selection"]:
        conditions.append(
            new_condition(
                producer_step="S11",
                condition_id=f"S11_cell_view_{mixture_id}_duplicate_same_goal",
                purpose="Figure 8 duplicate-value chimera traces",
                implementation="cell_view",
                mixture_id=mixture_id,
                input_profile="duplicate_1_10_x10",
                frozen_variant="none",
                frozen_count=0,
                goal_pattern="all_increasing",
                stop_policy="sorted_no_move_or_cap",
                trace_consumers=["S10", "S11", "S06"],
                wrapper_requirement="cell-view duplicate-value chimera wrapper required",
                notes="Uses paper baseline duplicate profile: values 1 through 10, ten copies each, n=100.",
            )
        )

    opposite_mixtures = [
        "bubble_down_selection_up",
        "bubble_up_insertion_down",
        "selection_down_insertion_up",
    ]
    for input_profile in ["unique_1_100", "duplicate_1_10_x10"]:
        for mixture_id in opposite_mixtures:
            conditions.append(
                new_condition(
                    producer_step="S12",
                    condition_id=f"S12_cell_view_{mixture_id}_{input_profile}",
                    purpose="Figures 9 and 10 opposite-direction chimera traces",
                    implementation="cell_view",
                    mixture_id=mixture_id,
                    input_profile=input_profile,
                    frozen_variant="none",
                    frozen_count=0,
                    goal_pattern="opposite_direction",
                    stop_policy="stable_no_move_or_cap",
                    trace_consumers=["S06", "S13"],
                    wrapper_requirement="cell-view opposite-direction chimera wrapper required",
                    notes="Encodes the three paper pairings explicitly instead of relying on active disorder-script defaults.",
                )
            )

    return conditions


def build_config(git_meta: dict[str, Any], s02_status: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "eidosoma.e01_baseline_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "configVersion": CONFIG_VERSION,
        "generatedAt": utc_now(),
        "generatedBy": "scripts/e01_s03_freeze_baseline_config.py",
        "sourceRefs": {
            "researchPlan": "/workspace/RESEARCH_PLAN.md",
            "fullPlan": "/workspace/FULL_PLAN.md",
            "s02CodePaperMap": "/artifacts/research_steps/S02/code_paper_map.md",
            "s02Status": "/artifacts/research_steps/S02/status.json",
            "paperMarkdown": "/workspace/input-attachments/a8ba3250-a8c7-4b7b-9a3f-edc9edb9e5f5/pdf-markdown.md",
        },
        "git": git_meta,
        "runPolicy": {
            "s03DryRunOnly": True,
            "doNotStartLargeSweeps": True,
            "nextExecutableStep": "S04 only after Chief Scientist instruction",
            "maxCpuWorkersForFutureSweeps": 8,
            "s03WorkerCount": 1,
            "gpuRequiredForS03": False,
        },
        "paperBaseline": {
            "repeatCount": REPEAT_COUNT,
            "uniqueInputProfile": {
                "id": "unique_1_100",
                "n": UNIQUE_N,
                "values": {"start": 1, "stopInclusive": 100, "copiesEach": 1},
                "ordering": "random_permutation_per_replicate",
            },
            "duplicateInputProfile": {
                "id": "duplicate_1_10_x10",
                "n": UNIQUE_N,
                "values": {"start": 1, "stopInclusive": 10, "copiesEach": 10},
                "ordering": "random_permutation_per_replicate",
            },
            "algorithms": ALGORITHMS,
            "implementations": IMPLEMENTATIONS,
            "frozenCellCounts": [0, 1, 2, 3],
            "frozenCellVariants": ["none", "passive", "stuck"],
            "sameGoalMixtures": [
                "pure_bubble",
                "pure_insertion",
                "pure_selection",
                "bubble_insertion",
                "bubble_selection",
                "insertion_selection",
                "bubble_insertion_selection",
            ],
            "duplicateSameGoalMixtures": [
                "bubble_insertion",
                "bubble_selection",
                "insertion_selection",
            ],
            "oppositeDirectionMixtures": [
                "bubble_down_selection_up",
                "bubble_up_insertion_down",
                "selection_down_insertion_up",
            ],
        },
        "implementationSources": {
            "cellView": {
                "bubble": "modules/multithread/BubbleSortCell.py",
                "insertion": "modules/multithread/InsertionSortCell.py",
                "selection": "modules/multithread/SelectionSortCell.py",
                "probe": "modules/multithread/StatusProbe.py",
                "sharedCell": "modules/multithread/MultiThreadCell.py",
            },
            "traditional": {
                "status": "wrapper_required",
                "reason": "S02 found no clean paper-setting traditional Bubble, Insertion, or Selection runner.",
                "traceConvention": "wrapper records initial array, every swap state, swap_count, comparison_count, and final stop_reason using the same schema as cell-view traces.",
            },
        },
        "semantics": {
            "frozenCells": {
                "none": "No Frozen Cells.",
                "passive": "Frozen element cannot initiate a move but may be moved by a non-frozen actor.",
                "stuck": "Frozen element cannot initiate a move and no swap involving that element is allowed.",
                "s02Ambiguity": "Archived scripts mix stopped-thread and movable-by-others behavior; S03 freezes paper-declared semantics for wrappers.",
            },
            "metrics": {
                "sortednessPercent": "100 * nondecreasing_adjacent_pairs / (n - 1), computed from the active goal direction; for descending goals use nonincreasing adjacent pairs.",
                "sortednessRawCount": "Number of adjacent pairs consistent with the active goal direction.",
                "monotonicityError": "(n - 1) - sortednessRawCount for the relevant direction.",
                "swapOnlySteps": "Number of successful swaps.",
                "comparisonPlusSwapSteps": "Wrapper convention: all recorded value comparisons plus successful swaps; archived cell-view compare_and_swap_count is preserved separately.",
                "delayedGratification": "To be ported and unit-tested in S08 from S02-mapped helper logic; S03 records required input traces only.",
                "aggregation": "Canonical paper metric: percent of cells whose directly adjacent left neighbor has the same Algotype. Store undirected/right-neighbor variants only as secondary diagnostics.",
            },
            "stopReasons": [
                "sorted",
                "no_cell_can_move_after_two_checks",
                "stable_sortedness_window",
                "max_step_cap",
                "wall_time_cap",
                "error",
            ],
            "stopPolicies": {
                "sorted_or_cap": {
                    "primary": "sorted",
                    "fallbacks": ["max_step_cap", "wall_time_cap", "error"],
                    "maxSwapEvents": 250000,
                    "maxComparisonEvents": 2000000,
                },
                "sorted_no_move_or_cap": {
                    "primary": "sorted",
                    "fallbacks": ["no_cell_can_move_after_two_checks", "max_step_cap", "wall_time_cap", "error"],
                    "noMoveChecksRequired": 2,
                    "maxSwapEvents": 500000,
                    "maxComparisonEvents": 4000000,
                },
                "stable_no_move_or_cap": {
                    "primary": "no_cell_can_move_after_two_checks",
                    "fallbacks": ["stable_sortedness_window", "max_step_cap", "wall_time_cap", "error"],
                    "noMoveChecksRequired": 2,
                    "stableWindowEvents": 10000,
                    "stableSortednessTolerance": 0.0,
                    "maxSwapEvents": 750000,
                    "maxComparisonEvents": 6000000,
                },
            },
        },
        "seedPolicy": {
            "algorithm": "sha256-derived uint32 seeds",
            "baseText": CONFIG_VERSION,
            "replicateIndexing": "0-based replicate_index, 1-based replicate_number",
            "streams": [
                "input_permutation",
                "algotype_assignment",
                "frozen_position",
                "scheduler",
                "tie_breaker",
            ],
            "matchedInputs": "Conditions sharing the same replicate_index and input_profile reuse input_permutation_seed for matched initial arrays.",
            "matchedFrozenPositions": "Frozen conditions sharing replicate_index, input_profile, and frozen_count reuse frozen_position_seed where comparisons require matched defects.",
            "conditionSpecificStreams": ["scheduler", "tie_breaker"],
            "analysisSeedTablePath": "/artifacts/research_steps/S03/analysis_seed_table.csv",
        },
        "traceSchema": {
            "conditionLevelFields": [
                "condition_id",
                "producer_step",
                "implementation",
                "mixture_id",
                "algorithms",
                "goal_directions",
                "input_profile",
                "frozen_variant",
                "frozen_count",
                "repeat_count",
            ],
            "replicateLevelFields": [
                "condition_id",
                "replicate_index",
                "replicate_number",
                "input_permutation_seed",
                "algotype_assignment_seed",
                "frozen_position_seed",
                "scheduler_seed",
                "tie_breaker_seed",
                "initial_values",
                "initial_algotypes",
                "frozen_positions",
                "stop_reason",
                "completed",
                "swap_count",
                "comparison_count",
                "final_sortedness_percent",
                "final_monotonicity_error",
            ],
            "preferredFormats": ["parquet", "npy_probe_compatibility", "json_manifest"],
        },
        "s02AmbiguitiesCarriedForward": s02_status.get("caveatsOrBlockers", []),
    }


def build_seed_rows(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        for replicate_index in range(REPEAT_COUNT):
            condition_id = condition["conditionId"]
            has_multiple_algotypes = len(condition["algorithms"]) > 1
            frozen_seed = (
                seed_for_parts(
                    "frozen_position",
                    condition["inputProfile"],
                    condition["frozenCount"],
                    replicate_index,
                )
                if condition["frozenCount"]
                else ""
            )
            algotype_seed = (
                seed_for_parts(
                    "algotype_assignment",
                    condition["inputProfile"],
                    condition["mixtureId"],
                    json_string(condition["goalDirections"]),
                    replicate_index,
                )
                if has_multiple_algotypes
                else ""
            )
            rows.append(
                {
                    "researchStepId": STEP_ID,
                    "configVersion": CONFIG_VERSION,
                    "conditionId": condition_id,
                    "producerStep": condition["producerStep"],
                    "replicateIndex": replicate_index,
                    "replicateNumber": replicate_index + 1,
                    "inputProfile": condition["inputProfile"],
                    "implementation": condition["implementation"],
                    "mixtureId": condition["mixtureId"],
                    "frozenVariant": condition["frozenVariant"],
                    "frozenCount": condition["frozenCount"],
                    "inputPermutationSeed": seed_for_parts("input_permutation", condition["inputProfile"], replicate_index),
                    "algotypeAssignmentSeed": algotype_seed,
                    "frozenPositionSeed": frozen_seed,
                    "schedulerSeed": seed_for(condition_id, replicate_index, "scheduler"),
                    "tieBreakerSeed": seed_for(condition_id, replicate_index, "tie_breaker"),
                }
            )
    return rows


def build_analysis_seed_rows() -> list[dict[str, Any]]:
    analysis_specs = [
        ("S06", "efficiency_ztests_and_bootstrap", "S04,S05", 10000),
        ("S06", "frozen_robustness_ztests_and_bootstrap", "S07", 10000),
        ("S06", "delayed_gratification_ztests_and_bootstrap", "S08", 10000),
        ("S06", "aggregation_ztests_and_bootstrap", "S09,S10,S11,S12", 10000),
        ("S08", "delayed_gratification_hand_trace_tests", "S07", 0),
        ("S10", "unique_value_aggregation_bootstrap", "S09", 10000),
        ("S11", "duplicate_value_aggregation_bootstrap", "S11", 10000),
        ("S12", "opposite_direction_summary_bootstrap", "S12", 10000),
    ]
    rows = []
    for analysis_step, analysis_id, source_steps, bootstrap_replicates in analysis_specs:
        notes = (
            "Bootstrap/statistical RNG seed for later analysis."
            if bootstrap_replicates
            else "Deterministic unit or formula validation only."
        )
        rows.append(
            {
                "researchStepId": STEP_ID,
                "configVersion": CONFIG_VERSION,
                "analysisStep": analysis_step,
                "analysisId": analysis_id,
                "sourceProducerSteps": source_steps,
                "bootstrapReplicates": bootstrap_replicates,
                "analysisSeed": seed_for_parts("analysis", analysis_step, analysis_id),
                "notes": notes,
            }
        )
    return rows


def condition_csv_rows(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for condition in conditions:
        rows.append(
            {
                "researchStepId": STEP_ID,
                "configVersion": CONFIG_VERSION,
                "conditionId": condition["conditionId"],
                "producerStep": condition["producerStep"],
                "purpose": condition["purpose"],
                "implementation": condition["implementation"],
                "mixtureId": condition["mixtureId"],
                "algorithms": ";".join(condition["algorithms"]),
                "algotypeAllocation": json_string(condition["algotypeAllocation"]),
                "goalDirections": json_string(condition["goalDirections"]),
                "inputProfile": condition["inputProfile"],
                "n": condition["n"],
                "repeatCount": condition["repeatCount"],
                "frozenVariant": condition["frozenVariant"],
                "frozenCount": condition["frozenCount"],
                "goalPattern": condition["goalPattern"],
                "stopPolicy": condition["stopPolicy"],
                "traceConsumers": ";".join(condition["traceConsumers"]),
                "wrapperRequirement": condition["wrapperRequirement"],
                "notes": condition["notes"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def validate(
    config: dict[str, Any],
    conditions: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []

    expected_by_step = {"S04": 6, "S07": 42, "S09": 7, "S11": 3, "S12": 6}
    counts_by_step = dict(Counter(condition["producerStep"] for condition in conditions))
    if counts_by_step != expected_by_step:
        errors.append(f"condition counts by step mismatch: expected {expected_by_step}, observed {counts_by_step}")

    if config["paperBaseline"]["repeatCount"] != 100:
        errors.append("repeatCount is not 100")
    if config["paperBaseline"]["uniqueInputProfile"]["values"] != {"start": 1, "stopInclusive": 100, "copiesEach": 1}:
        errors.append("unique input profile is not values 1 through 100 with one copy each")
    if config["paperBaseline"]["duplicateInputProfile"]["values"] != {"start": 1, "stopInclusive": 10, "copiesEach": 10}:
        errors.append("duplicate input profile is not values 1 through 10 with ten copies each")

    expected_seed_rows = len(conditions) * REPEAT_COUNT
    if len(seed_rows) != expected_seed_rows:
        errors.append(f"seed row mismatch: expected {expected_seed_rows}, observed {len(seed_rows)}")

    duplicate_seed_check = set()
    for row in seed_rows:
        key = (row["conditionId"], row["replicateIndex"])
        if key in duplicate_seed_check:
            errors.append(f"duplicate seed replicate key: {key}")
            break
        duplicate_seed_check.add(key)

    input_seed_by_profile_rep: dict[tuple[str, int], Any] = {}
    frozen_seed_by_profile_count_rep: dict[tuple[str, int, int], Any] = {}
    for row in seed_rows:
        input_key = (row["inputProfile"], int(row["replicateIndex"]))
        input_seed_by_profile_rep.setdefault(input_key, row["inputPermutationSeed"])
        if input_seed_by_profile_rep[input_key] != row["inputPermutationSeed"]:
            errors.append(f"input seed is not matched for {input_key}")
            break
        if int(row["frozenCount"]) > 0:
            frozen_key = (row["inputProfile"], int(row["frozenCount"]), int(row["replicateIndex"]))
            frozen_seed_by_profile_count_rep.setdefault(frozen_key, row["frozenPositionSeed"])
            if frozen_seed_by_profile_count_rep[frozen_key] != row["frozenPositionSeed"]:
                errors.append(f"frozen seed is not matched for {frozen_key}")
                break

    s07 = [condition for condition in conditions if condition["producerStep"] == "S07"]
    for implementation in IMPLEMENTATIONS:
        for algorithm in ALGORITHMS:
            matching = [
                (condition["frozenCount"], condition["frozenVariant"])
                for condition in s07
                if condition["implementation"] == implementation
                and condition["mixtureId"] == f"pure_{algorithm}"
            ]
            expected = [(0, "none")] + [
                (frozen_count, frozen_variant)
                for frozen_variant in ["passive", "stuck"]
                for frozen_count in [1, 2, 3]
            ]
            if sorted(matching) != sorted(expected):
                errors.append(f"S07 frozen matrix mismatch for {implementation}/{algorithm}: {matching}")

    s12_pairings = {
        condition["mixtureId"]
        for condition in conditions
        if condition["producerStep"] == "S12"
    }
    expected_pairings = {
        "bubble_down_selection_up",
        "bubble_up_insertion_down",
        "selection_down_insertion_up",
    }
    if s12_pairings != expected_pairings:
        errors.append(f"S12 pairings mismatch: expected {expected_pairings}, observed {s12_pairings}")

    if "doNotStartLargeSweeps" not in config["runPolicy"] or not config["runPolicy"]["doNotStartLargeSweeps"]:
        errors.append("runPolicy does not explicitly forbid S04 sweeps during S03")

    if config["implementationSources"]["traditional"]["status"] != "wrapper_required":
        errors.append("traditional implementation is not marked wrapper_required")

    if not config["s02AmbiguitiesCarriedForward"]:
        warnings.append("S02 ambiguities list is empty")

    return (
        not errors,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "success": not errors,
            "status": "passed" if not errors else "failed",
            "errors": errors,
            "warnings": warnings,
            "counts": {
                "conditionCount": len(conditions),
                "seedRowCount": len(seed_rows),
                "matchedInputSeedGroups": len(input_seed_by_profile_rep),
                "matchedFrozenSeedGroups": len(frozen_seed_by_profile_count_rep),
                "repeatCount": REPEAT_COUNT,
                "conditionsByStep": counts_by_step,
                "conditionsByImplementation": dict(Counter(condition["implementation"] for condition in conditions)),
                "conditionsByInputProfile": dict(Counter(condition["inputProfile"] for condition in conditions)),
                "conditionsByFrozenVariant": dict(Counter(condition["frozenVariant"] for condition in conditions)),
            },
            "checks": [
                "config encodes n=100 and N=100",
                "unique values are 1 through 100",
                "duplicate values are 10 copies each of 1 through 10",
                "Frozen Cell counts cover 0, 1, 2, and 3 with passive/stuck variants for f>0",
                "traditional implementations are marked wrapper_required",
                "same-goal and opposite-direction chimeras are explicit",
                "seed rows equal condition_count * repeat_count",
                "input seeds are matched by input profile and replicate index",
                "frozen-position seeds are matched by input profile, frozen count, and replicate index",
                "S03 run policy forbids S04 large sweeps",
            ],
        },
    )


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(paths):
        if path.exists() and path.is_file():
            artifacts.append(
                {
                    "path": str(path),
                    "sizeBytes": path.stat().st_size,
                    "sha256": sha256_path(path),
                }
            )
    return artifacts


def collect_step_artifacts(step_dir: Path, extra_paths: list[Path]) -> list[dict[str, Any]]:
    paths = [path for path in step_dir.rglob("*") if path.is_file()] + extra_paths
    unique_paths = sorted({path.resolve() for path in paths})
    artifacts = []
    for path in unique_paths:
        artifacts.append(
            {
                "path": str(path),
                "relativePath": path.name if path not in step_dir.rglob("*") else str(path.relative_to(step_dir)),
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    return artifacts


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def write_config_notes(
    path: Path,
    config: dict[str, Any],
    conditions: list[dict[str, Any]],
    validation: dict[str, Any],
    artifacts_written: list[str],
) -> None:
    by_step = validation["counts"]["conditionsByStep"]
    step_rows = [[step, by_step.get(step, 0), by_step.get(step, 0) * REPEAT_COUNT] for step in sorted(by_step)]
    ambiguity_rows = [[item] for item in config["s02AmbiguitiesCarriedForward"]]
    artifacts = "\n".join(f"- `{artifact}`" for artifact in artifacts_written)
    path.write_text(
        f"""# S03 Config Notes

- Research step ID: {STEP_ID}
- Step number: {STEP_NUMBER}
- Completion status: {STATUS}
- Outcome classification: {OUTCOME_CLASSIFICATION}
- Artifacts written:
{artifacts}
- Validation result: {"passed" if validation["success"] else "failed"}; {validation["counts"]["conditionCount"]} dry-run conditions and {validation["counts"]["seedRowCount"]} replicate seed rows were generated.
- Caveats or blockers: S02 ambiguities remain active configuration constraints, especially traditional wrapper reconstruction, Frozen Cell semantics, comparison counting, lock-randomization mismatch, metric naming, and missing author-local raw arrays.
- Recommended next action: Proceed to S04 only after Chief Scientist instruction, using this config as the source of paper constants and seed conventions.

## Frozen Question

The paper's main settings can be encoded as explicit, rerunnable configs. S03 freezes those settings without running S04 simulations.

## Canonical Choices

- Arrays: unique profile is values 1 through 100, one copy each; duplicate profile is values 1 through 10, ten copies each.
- Repeats: every simulation condition has N=100 planned replicates.
- Seed matching: input seeds are shared by input profile and replicate index; frozen-position seeds are shared by input profile, frozen count, and replicate index; scheduler seeds remain condition-specific.
- Traditional algorithms: Bubble, Insertion, and Selection are marked `wrapper_required` because S02 found no clean paper-setting runner.
- Cell-view algorithms: Bubble, Insertion, and Selection wrappers should call the S02-mapped classes and record normalized trace fields.
- Frozen Cells: f=0 is represented once with variant `none`; f=1,2,3 are represented for both `passive` and `stuck`.
- Passive Frozen Cell: cannot initiate a move but may be moved by a non-frozen actor.
- Stuck Frozen Cell: cannot initiate and cannot be moved by any actor.
- Aggregation: canonical metric is the paper's left-neighbor same-Algotype percentage. Other adjacency definitions are secondary diagnostics.
- All-three chimera allocation: n=100 cannot be split exactly three ways, so the config freezes a deterministic near-equal allocation of 34 Bubble, 33 Insertion, and 33 Selection cells.

## Dry-Run Matrix

{markdown_table(["Producer step", "Condition count", "Replicate seed rows"], step_rows)}

## S02 Ambiguities Carried Forward

{markdown_table(["Ambiguity or blocker"], ambiguity_rows)}

## No S04 Sweep Statement

S03 generated configuration, seed, and validation artifacts only. It did not execute sorting simulations, figure generation, statistical tests, or any large sweeps.
""",
        encoding="utf-8",
    )


def write_validation_md(path: Path, validation: dict[str, Any]) -> None:
    count_rows = [[key, value] for key, value in validation["counts"].items()]
    check_rows = [[check] for check in validation["checks"]]
    error_rows = [[error] for error in validation["errors"]] or [["None"]]
    warning_rows = [[warning] for warning in validation["warnings"]] or [["None"]]
    path.write_text(
        f"""# S03 Dry-Run Validation

- Research step ID: {STEP_ID}
- Step number: {STEP_NUMBER}
- Completion status: {STATUS}
- Validation result: {"passed" if validation["success"] else "failed"}
- Artifacts written: see `status.json` and `artifact_manifest.json`
- Caveats or blockers: validation is a dry-run matrix check only; no simulation behavior is validated here.
- Recommended next action: use the validated config in S04 after Chief Scientist instruction.

## Counts

{markdown_table(["Field", "Value"], count_rows)}

## Checks

{markdown_table(["Check"], check_rows)}

## Errors

{markdown_table(["Error"], error_rows)}

## Warnings

{markdown_table(["Warning"], warning_rows)}
""",
        encoding="utf-8",
    )


def write_summary(
    path: Path,
    artifacts_written: list[str],
    validation_result: str,
    caveats: list[str],
    recommended_next_action: str,
) -> None:
    artifacts = "\n".join(f"- `{artifact}`" for artifact in artifacts_written)
    caveat_lines = "\n".join(f"- {caveat}" for caveat in caveats)
    path.write_text(
        f"""# S03 Status Summary

- Research step ID: {STEP_ID}
- Step number: {STEP_NUMBER}
- Completion status: {STATUS}
- Outcome classification: {OUTCOME_CLASSIFICATION}
- Artifacts written:
{artifacts}
- Validation result: {validation_result}
- Caveats or blockers:
{caveat_lines}
- Lay summary: The paper baseline is now encoded as a rerunnable configuration with explicit seeds and a dry-run condition matrix. This step did not run the actual sorting sweeps; it only made future sweeps auditable and explicit.
- Recommended next action: {recommended_next_action}
""",
        encoding="utf-8",
    )


def update_run_manifest(
    artifacts_dir: Path,
    status_payload: dict[str, Any],
    step_dir: Path,
    extra_paths: list[Path],
) -> None:
    provenance_dir = artifacts_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = provenance_dir / "run_manifest.json"
    if run_manifest_path.exists():
        try:
            manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}

    manifest.setdefault("experimentId", EXPERIMENT_ID)
    manifest["generatedAt"] = utc_now()
    manifest["researchStepId"] = STEP_ID
    manifest["recommendedNextAction"] = status_payload["recommendedNextAction"]
    manifest["caveatsOrBlockers"] = status_payload["caveatsOrBlockers"]
    manifest["artifactsWritten"] = status_payload["artifactsWritten"]
    manifest["git"] = status_payload["git"]
    manifest["runtime"] = status_payload["runtime"]
    manifest.setdefault("researchSteps", {})
    artifacts = collect_step_artifacts(step_dir, extra_paths)
    manifest["researchSteps"][STEP_ID] = {
        "status": status_payload["status"],
        "success": status_payload["success"],
        "artifactCount": len(artifacts),
        "artifacts": artifacts,
        "validationResult": status_payload["validationResult"],
        "conditionCount": status_payload["conditionCount"],
        "seedRowCount": status_payload["seedRowCount"],
        "configPath": status_payload["configPath"],
        "generatedAt": status_payload["generatedAt"],
    }
    run_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    code_dir = step_dir / "code"
    config_dir = artifacts_dir / "configs"
    step_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    s02_status_path = artifacts_dir / "research_steps" / "S02" / "status.json"
    s02_status = json.loads(s02_status_path.read_text(encoding="utf-8")) if s02_status_path.exists() else {}
    git_meta = get_git_metadata(repo_root)
    config = build_config(git_meta, s02_status)
    conditions = build_conditions()
    condition_rows = condition_csv_rows(conditions)
    seed_rows = build_seed_rows(conditions)
    analysis_seed_rows = build_analysis_seed_rows()
    ok, validation = validate(config, conditions, seed_rows)

    config["dryRunConditionMatrix"] = {
        "conditionCount": len(conditions),
        "seedRowCount": len(seed_rows),
        "conditionsByStep": validation["counts"]["conditionsByStep"],
        "conditionMatrixPath": str(step_dir / "condition_matrix.csv"),
        "seedTablePath": str(step_dir / "seed_table.csv"),
        "analysisSeedTablePath": str(step_dir / "analysis_seed_table.csv"),
    }

    config_yaml_path = config_dir / "e01_baseline_config.yaml"
    config_json_path = config_dir / "e01_baseline_config.json"
    config_yaml_path.write_text(yaml_dump(config) + "\n", encoding="utf-8")
    config_json_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    write_csv(
        step_dir / "condition_matrix.csv",
        condition_rows,
        [
            "researchStepId",
            "configVersion",
            "conditionId",
            "producerStep",
            "purpose",
            "implementation",
            "mixtureId",
            "algorithms",
            "algotypeAllocation",
            "goalDirections",
            "inputProfile",
            "n",
            "repeatCount",
            "frozenVariant",
            "frozenCount",
            "goalPattern",
            "stopPolicy",
            "traceConsumers",
            "wrapperRequirement",
            "notes",
        ],
    )
    (step_dir / "condition_matrix.json").write_text(
        json.dumps(conditions, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(
        step_dir / "seed_table.csv",
        seed_rows,
        [
            "researchStepId",
            "configVersion",
            "conditionId",
            "producerStep",
            "replicateIndex",
            "replicateNumber",
            "inputProfile",
            "implementation",
            "mixtureId",
            "frozenVariant",
            "frozenCount",
            "inputPermutationSeed",
            "algotypeAssignmentSeed",
            "frozenPositionSeed",
            "schedulerSeed",
            "tieBreakerSeed",
        ],
    )
    write_csv(
        step_dir / "analysis_seed_table.csv",
        analysis_seed_rows,
        [
            "researchStepId",
            "configVersion",
            "analysisStep",
            "analysisId",
            "sourceProducerSteps",
            "bootstrapReplicates",
            "analysisSeed",
            "notes",
        ],
    )
    seed_summary = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "configVersion": CONFIG_VERSION,
        "seedRowCount": len(seed_rows),
        "analysisSeedRowCount": len(analysis_seed_rows),
        "uniqueSeedTuples": len({(row["conditionId"], row["replicateIndex"]) for row in seed_rows}),
        "streams": config["seedPolicy"]["streams"],
        "repeatCount": REPEAT_COUNT,
        "conditionCount": len(conditions),
    }
    (step_dir / "seed_table_summary.json").write_text(
        json.dumps(seed_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (step_dir / "dry_run_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_validation_md(step_dir / "dry_run_validation.md", validation)

    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)

    artifacts_written = [
        str(config_yaml_path),
        str(config_json_path),
        str(step_dir / "config_notes.md"),
        str(step_dir / "condition_matrix.csv"),
        str(step_dir / "condition_matrix.json"),
        str(step_dir / "seed_table.csv"),
        str(step_dir / "analysis_seed_table.csv"),
        str(step_dir / "seed_table_summary.json"),
        str(step_dir / "dry_run_validation.md"),
        str(step_dir / "dry_run_validation.json"),
        str(step_dir / "summary.md"),
        str(step_dir / "status.json"),
        str(step_dir / "artifact_manifest.json"),
        str(code_dir / Path(__file__).name),
    ]
    write_config_notes(step_dir / "config_notes.md", config, conditions, validation, artifacts_written)

    validation_result = (
        f"passed: generated {len(conditions)} dry-run conditions and {len(seed_rows)} seed rows; "
        "validated n=100, N=100, Frozen Cell matrix, duplicate-value profile, opposite-direction pairings, and no-S04-sweep policy."
        if ok
        else "failed: " + "; ".join(validation["errors"])
    )
    caveats = [
        "S03 freezes configuration only; no sorting simulations, figures, or large sweeps were run.",
        "Traditional Bubble, Insertion, and Selection require wrapper implementations because S02 found no clean paper-setting runners.",
        "Frozen Cell passive/stuck semantics are paper-declared wrapper semantics because archived code mixes behaviors.",
        "Comparison-inclusive efficiency uses a wrapper convention and preserves archived compare_and_swap_count separately.",
        "Delayed Gratification formula still requires S08 clean port and hand-trajectory unit tests.",
        "All-three same-goal chimera allocation is near-equal 34/33/33 because n=100 is not divisible by three.",
    ]
    recommended_next_action = (
        "Hand control back; proceed to S04 only after Chief Scientist instruction, using "
        "/artifacts/configs/e01_baseline_config.yaml and the S03 seed table as the source of constants and seeds."
    )
    runtime = {
        "pythonExecutable": sys.executable,
        "pythonVersion": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "osCpuCount": os.cpu_count(),
        "workerCount": 1,
        "gpuUsed": False,
    }
    status_payload: dict[str, Any] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": ok,
        "status": STATUS if ok else "failed",
        "artifactsWritten": artifacts_written,
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats + validation["errors"],
        "recommendedNextAction": recommended_next_action,
        "outcomeClassification": OUTCOME_CLASSIFICATION if ok else "blocked",
        "generatedAt": utc_now(),
        "experimentId": EXPERIMENT_ID,
        "conditionCount": len(conditions),
        "seedRowCount": len(seed_rows),
        "analysisSeedRowCount": len(analysis_seed_rows),
        "conditionsByStep": validation["counts"]["conditionsByStep"],
        "configPath": str(config_yaml_path),
        "configJsonPath": str(config_json_path),
        "s04SweepsStarted": False,
        "git": git_meta,
        "runtime": runtime,
    }
    write_summary(
        step_dir / "summary.md",
        artifacts_written,
        validation_result,
        caveats,
        recommended_next_action,
    )
    (step_dir / "status.json").write_text(
        json.dumps(status_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_manifest = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAt": utc_now(),
        "status": STATUS if ok else "failed",
        "success": ok,
        "validationResult": validation_result,
        "artifacts": collect_artifacts([Path(path) for path in artifacts_written if Path(path).exists()]),
    }
    (step_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    update_run_manifest(artifacts_dir, status_payload, step_dir, [config_yaml_path, config_json_path])

    print(validation_result)
    print(f"Wrote S03 artifacts under {step_dir} and config under {config_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
