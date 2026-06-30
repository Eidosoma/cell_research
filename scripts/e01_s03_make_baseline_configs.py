#!/usr/bin/env python3
"""Freeze E01 S03 baseline configuration and condition matrix.

This script does not run any sorting simulation. It writes deterministic seed
banks, paper-condition rows, and explicit assumptions needed by downstream
replication steps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_SEED = 2026063003
REPEAT_COUNT = 100
ARRAY_LENGTH = 100
ALGORITHMS = ("bubble", "insertion", "selection")
MODES = ("traditional", "cell_view")
FROZEN_SEMANTICS = ("passive", "stuck")
FROZEN_COUNTS = (0, 1, 2, 3)


CSV_FIELDS = [
    "condition_id",
    "step_scope",
    "figure_targets",
    "run_family",
    "mode",
    "baseline_source",
    "algorithm",
    "algotype_mix",
    "algotype_roles",
    "direction_profile",
    "value_distribution",
    "array_length",
    "repeat_count",
    "value_bank_id",
    "frozen_count",
    "frozen_semantics",
    "frozen_index_bank_id",
    "algotype_assignment_bank_id",
    "matched_group_id",
    "stop_rule",
    "max_steps_policy",
    "primary_metrics",
    "outputs_expected",
    "reconstruction_notes",
    "caveats",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path("/artifacts"))
    return parser.parse_args()


def git_commit(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_for(namespace: int, repeat_idx: int = 0, variant: int = 0) -> int:
    return BASE_SEED * 1_000_000 + namespace * 10_000 + variant * 1_000 + repeat_idx


def shuffled(values: list[int], seed: int) -> list[int]:
    rng = random.Random(seed)
    copy = list(values)
    rng.shuffle(copy)
    return copy


def build_value_banks() -> dict[str, Any]:
    unique_values = list(range(1, ARRAY_LENGTH + 1))
    duplicate_values = [value for value in range(1, 11) for _ in range(10)]
    return {
        "unique_1_to_100": {
            "description": "Paper baseline: random permutation from 1 to 100 without duplicates.",
            "valueDistribution": "unique_1_to_100",
            "arrayLength": ARRAY_LENGTH,
            "repeatCount": REPEAT_COUNT,
            "seeds": [seed_for(10, repeat_idx) for repeat_idx in range(REPEAT_COUNT)],
            "initialArrays": [
                shuffled(unique_values, seed_for(10, repeat_idx)) for repeat_idx in range(REPEAT_COUNT)
            ],
        },
        "duplicate_1_to_10_x10": {
            "description": "Paper duplicate-value condition: 10 repeated copies of each digit 1 through 10.",
            "valueDistribution": "duplicate_1_to_10_x10",
            "arrayLength": ARRAY_LENGTH,
            "repeatCount": REPEAT_COUNT,
            "seeds": [seed_for(11, repeat_idx) for repeat_idx in range(REPEAT_COUNT)],
            "initialArrays": [
                shuffled(duplicate_values, seed_for(11, repeat_idx)) for repeat_idx in range(REPEAT_COUNT)
            ],
        },
    }


def build_frozen_banks() -> dict[str, Any]:
    banks: dict[str, Any] = {
        "frozen_0": {
            "frozenCount": 0,
            "description": "No Frozen Cells; reused by unperturbed and f=0 DG baselines.",
            "seeds": [seed_for(20, repeat_idx, 0) for repeat_idx in range(REPEAT_COUNT)],
            "indices": [[] for _ in range(REPEAT_COUNT)],
        }
    }
    for frozen_count in (1, 2, 3):
        indices = []
        seeds = []
        for repeat_idx in range(REPEAT_COUNT):
            seed = seed_for(20, repeat_idx, frozen_count)
            rng = random.Random(seed)
            seeds.append(seed)
            indices.append(sorted(rng.sample(range(ARRAY_LENGTH), frozen_count)))
        banks[f"frozen_{frozen_count}"] = {
            "frozenCount": frozen_count,
            "description": f"Matched frozen-position bank for f={frozen_count}. Same indices are reused across algorithms, modes, and frozen semantics.",
            "seeds": seeds,
            "indices": indices,
        }
    return banks


def balanced_assignment(
    labels: list[str],
    counts: list[int],
    seed: int,
) -> list[str]:
    values: list[str] = []
    for label, count in zip(labels, counts):
        values.extend([label] * count)
    rng = random.Random(seed)
    rng.shuffle(values)
    return values


def build_algotype_assignment_banks() -> dict[str, Any]:
    pair_specs = {
        "bubble_insertion_50_50": (["bubble", "insertion"], [50, 50]),
        "bubble_selection_50_50": (["bubble", "selection"], [50, 50]),
        "insertion_selection_50_50": (["insertion", "selection"], [50, 50]),
        "bubble_bubble_label_control_50_50": (["bubble_label_a", "bubble_label_b"], [50, 50]),
    }
    banks: dict[str, Any] = {}
    for bank_idx, (bank_id, (labels, counts)) in enumerate(pair_specs.items(), start=30):
        banks[bank_id] = {
            "description": "Balanced 50/50 shuffled Algotype labels per repeat.",
            "labels": labels,
            "countsPerRepeatRule": dict(zip(labels, counts)),
            "seeds": [seed_for(bank_idx, repeat_idx) for repeat_idx in range(REPEAT_COUNT)],
            "assignments": [
                balanced_assignment(labels, counts, seed_for(bank_idx, repeat_idx))
                for repeat_idx in range(REPEAT_COUNT)
            ],
        }

    three_labels = ["bubble", "insertion", "selection"]
    three_assignments = []
    three_counts = []
    three_seeds = []
    for repeat_idx in range(REPEAT_COUNT):
        extra_label = three_labels[repeat_idx % len(three_labels)]
        counts = {label: 33 for label in three_labels}
        counts[extra_label] += 1
        seed = seed_for(34, repeat_idx)
        three_seeds.append(seed)
        three_counts.append(counts)
        three_assignments.append(
            balanced_assignment(three_labels, [counts[label] for label in three_labels], seed)
        )
    banks["bubble_insertion_selection_balanced_remainder_rotating"] = {
        "description": "Three-way 100-cell mix. Because 100 is not divisible by 3, the extra cell rotates across Bubble, Insertion, and Selection over repeats.",
        "labels": three_labels,
        "countsPerRepeatRule": "33/33/33 plus one rotating remainder label by repeat index modulo 3",
        "countsPerRepeat": three_counts,
        "seeds": three_seeds,
        "assignments": three_assignments,
    }
    return banks


def source_for_mode(mode: str) -> str:
    if mode == "traditional":
        return "reconstructed_canonical_from_paper_figure_2_due_to_missing_public_runner"
    return "public_repository_cell_view_classes"


def metrics_for_family(run_family: str) -> str:
    if run_family in {"unperturbed_baseline", "frozen_robustness"}:
        return "sortedness_percent; monotonicity_error_count; swap_steps; compare_plus_swap_steps; delayed_gratification_for_frozen_dg"
    if "chimera" in run_family or "opposite_direction" in run_family:
        return "sortedness_percent; swap_steps; aggregation_left_neighbor_primary; aggregation_right_neighbor_legacy_sensitivity"
    return "sortedness_percent; swap_steps"


def condition_row(
    condition_id: str,
    step_scope: str,
    figure_targets: str,
    run_family: str,
    mode: str,
    algorithm: str,
    algotype_mix: str,
    algotype_roles: str,
    direction_profile: str,
    value_distribution: str,
    value_bank_id: str,
    frozen_count: int,
    frozen_semantics: str,
    algotype_assignment_bank_id: str,
    matched_group_id: str,
    stop_rule: str,
    max_steps_policy: str,
    outputs_expected: str,
    reconstruction_notes: str,
    caveats: str,
) -> dict[str, Any]:
    return {
        "condition_id": condition_id,
        "step_scope": step_scope,
        "figure_targets": figure_targets,
        "run_family": run_family,
        "mode": mode,
        "baseline_source": source_for_mode(mode),
        "algorithm": algorithm,
        "algotype_mix": algotype_mix,
        "algotype_roles": algotype_roles,
        "direction_profile": direction_profile,
        "value_distribution": value_distribution,
        "array_length": ARRAY_LENGTH,
        "repeat_count": REPEAT_COUNT,
        "value_bank_id": value_bank_id,
        "frozen_count": frozen_count,
        "frozen_semantics": frozen_semantics,
        "frozen_index_bank_id": f"frozen_{frozen_count}",
        "algotype_assignment_bank_id": algotype_assignment_bank_id,
        "matched_group_id": matched_group_id,
        "stop_rule": stop_rule,
        "max_steps_policy": max_steps_policy,
        "primary_metrics": metrics_for_family(run_family),
        "outputs_expected": outputs_expected,
        "reconstruction_notes": reconstruction_notes,
        "caveats": caveats,
    }


def build_conditions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counter = 1

    def add(**kwargs: Any) -> None:
        nonlocal counter
        rows.append(condition_row(condition_id=f"E01C{counter:03d}", **kwargs))
        counter += 1

    for mode in MODES:
        for algorithm in ALGORITHMS:
            add(
                step_scope="S04,S05,S06,S07_f0,S08_f0",
                figure_targets="Figure3,Figure4,Figure7_f0",
                run_family="unperturbed_baseline",
                mode=mode,
                algorithm=algorithm,
                algotype_mix=algorithm,
                algotype_roles=f"{algorithm}:increasing",
                direction_profile="all_increasing",
                value_distribution="unique_1_to_100",
                value_bank_id="unique_1_to_100",
                frozen_count=0,
                frozen_semantics="none",
                algotype_assignment_bank_id="not_applicable",
                matched_group_id=f"unique_f0_{algorithm}",
                stop_rule="stop_when_sorted",
                max_steps_policy="no_fixed_max_for_traditional; cell_view_guard_to_be_set_in_runner_and_recorded",
                outputs_expected="raw_trace; sortedness_trajectory; swap_count; comparison_count",
                reconstruction_notes=(
                    "Traditional mode is reconstructed from paper Figure 2 because S02 found no public traditional runner."
                    if mode == "traditional"
                    else "Cell-view mode uses public repository cell class mapped in S02."
                ),
                caveats="f=0 rows are reused as DG zero-frozen baselines.",
            )

    for frozen_semantics in FROZEN_SEMANTICS:
        for frozen_count in (1, 2, 3):
            for mode in MODES:
                for algorithm in ALGORITHMS:
                    add(
                        step_scope="S07,S08",
                        figure_targets="Figure5,Figure7",
                        run_family="frozen_robustness",
                        mode=mode,
                        algorithm=algorithm,
                        algotype_mix=algorithm,
                        algotype_roles=f"{algorithm}:increasing",
                        direction_profile="all_increasing",
                        value_distribution="unique_1_to_100",
                        value_bank_id="unique_1_to_100",
                        frozen_count=frozen_count,
                        frozen_semantics=frozen_semantics,
                        algotype_assignment_bank_id="not_applicable",
                        matched_group_id=f"unique_f{frozen_count}_{frozen_semantics}_{algorithm}",
                        stop_rule="stop_when_sorted_or_no_legal_move_twice",
                        max_steps_policy="runner_must_record_timeout_or_no_legal_move; initial guard suggested as 15000 successful swaps for cell_view",
                        outputs_expected="raw_trace; sortedness_trajectory; monotonicity_error; frozen_attempts; stop_reason",
                        reconstruction_notes=(
                            "Traditional frozen mode is reconstructed because public code only references missing original frozen arrays."
                            if mode == "traditional"
                            else "Cell-view frozen mode uses public classes plus explicit passive/stuck wrapper semantics."
                        ),
                        caveats="Passive/stuck behavior is a frozen S03 assumption and must be unit-tested before S07 full runs.",
                    )

    same_goal_specs = [
        ("same_goal_bubble_insertion", "bubble_insertion_50_50", "bubble:increasing; insertion:increasing"),
        ("same_goal_bubble_selection", "bubble_selection_50_50", "bubble:increasing; selection:increasing"),
        ("same_goal_insertion_selection", "insertion_selection_50_50", "insertion:increasing; selection:increasing"),
        (
            "same_goal_bubble_insertion_selection",
            "bubble_insertion_selection_balanced_remainder_rotating",
            "bubble:increasing; insertion:increasing; selection:increasing",
        ),
        (
            "same_algorithm_bubble_label_control",
            "bubble_bubble_label_control_50_50",
            "bubble_label_a:increasing_behavior_bubble; bubble_label_b:increasing_behavior_bubble",
        ),
    ]
    for run_name, assignment_bank, roles in same_goal_specs:
        if assignment_bank == "bubble_insertion_selection_balanced_remainder_rotating":
            same_goal_caveat = "Three-way mix uses rotating 34/33/33 remainder because 100 cells cannot be split exactly three ways."
        elif assignment_bank == "bubble_bubble_label_control_50_50":
            same_goal_caveat = "Negative control uses two 50/50 shuffled labels but both labels execute Bubble behavior."
        else:
            same_goal_caveat = "Pairwise mix uses balanced 50/50 shuffled Algotype labels."
        add(
            step_scope="S09,S10",
            figure_targets="Figure8a,Figure8b,Figure8c",
            run_family="same_goal_chimera",
            mode="cell_view",
            algorithm="mixed",
            algotype_mix=run_name,
            algotype_roles=roles,
            direction_profile="all_increasing",
            value_distribution="unique_1_to_100",
            value_bank_id="unique_1_to_100",
            frozen_count=0,
            frozen_semantics="none",
            algotype_assignment_bank_id=assignment_bank,
            matched_group_id=f"chimera_unique_{run_name}",
            stop_rule="stop_when_sorted_or_no_legal_move_twice",
            max_steps_policy="runner_must_record_timeout_or_no_legal_move; guard to be calibrated in pilot without changing seeds",
            outputs_expected="raw_trace; per_step_algotype_positions; sortedness_trajectory; aggregation_trajectory; swap_count",
            reconstruction_notes="Chimeric conditions use public cell-view classes; assignment banks replace inconsistent public runner defaults.",
            caveats=same_goal_caveat,
        )

    duplicate_specs = [
        ("duplicate_bubble_insertion", "bubble_insertion_50_50", "bubble:increasing; insertion:increasing"),
        ("duplicate_bubble_selection", "bubble_selection_50_50", "bubble:increasing; selection:increasing"),
        ("duplicate_insertion_selection", "insertion_selection_50_50", "insertion:increasing; selection:increasing"),
    ]
    for run_name, assignment_bank, roles in duplicate_specs:
        add(
            step_scope="S11",
            figure_targets="Figure8d,Figure8e",
            run_family="duplicate_value_chimera",
            mode="cell_view",
            algorithm="mixed",
            algotype_mix=run_name,
            algotype_roles=roles,
            direction_profile="all_increasing",
            value_distribution="duplicate_1_to_10_x10",
            value_bank_id="duplicate_1_to_10_x10",
            frozen_count=0,
            frozen_semantics="none",
            algotype_assignment_bank_id=assignment_bank,
            matched_group_id=f"chimera_duplicate_{run_name}",
            stop_rule="stop_when_sorted_or_no_legal_move_twice",
            max_steps_policy="runner_must_record_timeout_or_no_legal_move; guard to be calibrated in pilot without changing seeds",
            outputs_expected="raw_trace; per_step_algotype_positions; sortedness_trajectory; aggregation_trajectory; final_equal_value_block_summary",
            reconstruction_notes="Duplicate-value arrays are fixed to 10 copies each of values 1..10, correcting the public active script's 200-cell 0..9 mismatch.",
            caveats="Sortedness with duplicates must allow nondecreasing ties.",
        )

    opposite_unique_specs = [
        (
            "opposite_unique_bubble_down_selection_up",
            "bubble_selection_50_50",
            "bubble:decreasing; selection:increasing",
        ),
        (
            "opposite_unique_bubble_up_insertion_down",
            "bubble_insertion_50_50",
            "bubble:increasing; insertion:decreasing",
        ),
        (
            "opposite_unique_insertion_up_selection_down",
            "insertion_selection_50_50",
            "insertion:increasing; selection:decreasing",
        ),
    ]
    for run_name, assignment_bank, roles in opposite_unique_specs:
        add(
            step_scope="S12",
            figure_targets="Figure9",
            run_family="opposite_direction_chimera_unique",
            mode="cell_view",
            algorithm="mixed",
            algotype_mix=run_name,
            algotype_roles=roles,
            direction_profile="mixed_increasing_decreasing",
            value_distribution="unique_1_to_100",
            value_bank_id="unique_1_to_100",
            frozen_count=0,
            frozen_semantics="none",
            algotype_assignment_bank_id=assignment_bank,
            matched_group_id=f"opposite_unique_{run_name}",
            stop_rule="stop_when_no_legal_move_twice_or_stable_equilibrium_guard",
            max_steps_policy="runner_must_record equilibrium_or_timeout; guard to be calibrated in pilot without changing seeds",
            outputs_expected="raw_trace; sortedness_trajectory_for_declared_goal; aggregation_trajectory; final_equilibrium_summary",
            reconstruction_notes="Opposite-direction assignment is explicit here because public disorder runner active flags are inconsistent.",
            caveats="Dominance interpretation remains proxy-only and scheduler-sensitive.",
        )

    opposite_duplicate_specs = [
        (
            "opposite_duplicate_bubble_down_selection_up",
            "bubble_selection_50_50",
            "bubble:decreasing; selection:increasing",
        ),
        (
            "opposite_duplicate_bubble_up_insertion_down",
            "bubble_insertion_50_50",
            "bubble:increasing; insertion:decreasing",
        ),
        (
            "opposite_duplicate_insertion_up_selection_down",
            "insertion_selection_50_50",
            "insertion:increasing; selection:decreasing",
        ),
    ]
    for run_name, assignment_bank, roles in opposite_duplicate_specs:
        add(
            step_scope="S12",
            figure_targets="Figure10",
            run_family="opposite_direction_chimera_duplicate",
            mode="cell_view",
            algorithm="mixed",
            algotype_mix=run_name,
            algotype_roles=roles,
            direction_profile="mixed_increasing_decreasing",
            value_distribution="duplicate_1_to_10_x10",
            value_bank_id="duplicate_1_to_10_x10",
            frozen_count=0,
            frozen_semantics="none",
            algotype_assignment_bank_id=assignment_bank,
            matched_group_id=f"opposite_duplicate_{run_name}",
            stop_rule="stop_when_no_legal_move_twice_or_stable_equilibrium_guard",
            max_steps_policy="runner_must_record equilibrium_or_timeout; guard to be calibrated in pilot without changing seeds",
            outputs_expected="raw_trace; sortedness_trajectory_for_declared_goal; aggregation_trajectory; final_equilibrium_summary",
            reconstruction_notes="Repeated opposite-direction arrays use 10 copies each of values 1..10 per paper Figure 10.",
            caveats="Dominance interpretation remains proxy-only and scheduler-sensitive.",
        )
    return rows


def build_config(repo_dir: Path) -> dict[str, Any]:
    conditions = build_conditions()
    return {
        "schema": "eidosoma.e01.s03.baseline_config.v1",
        "experimentId": "E01",
        "researchStepId": "S03",
        "stepNumber": 3,
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "repo": {
            "path": str(repo_dir),
            "branch": "eidosoma/groups/12",
            "commitAtConfigGeneration": git_commit(repo_dir),
        },
        "sourceContext": {
            "paperMarkdown": "/workspace/input-attachments/f93afdc5-f2e5-4ecc-80bb-e088f93acf3c/pdf-markdown.md",
            "s02MappingTable": "/artifacts/tables/code_paper_mapping.csv",
            "s02CodebaseMap": "/artifacts/reports/e01_codebase_map.md",
            "s02Conclusion": "Traditional public runners and original saved arrays were absent; traditional baselines are reconstructed here.",
        },
        "globalDefaults": {
            "arrayLength": ARRAY_LENGTH,
            "repeatCount": REPEAT_COUNT,
            "positions": "zero_based_positions_0_to_99",
            "paperValueConvention": "one_based_values_1_to_100_or_1_to_10_for_duplicate_conditions",
            "randomGenerator": "python_random.Random_mersenne_twister",
            "baseSeed": BASE_SEED,
            "frozenCountsDeclared": list(FROZEN_COUNTS),
            "cpuPolicy": "configuration_only_no_parallel_work",
        },
        "traditionalBaselines": {
            "sourceStatus": "reconstructed",
            "reason": "S02 found no public traditional-generation runner and no saved original arrays.",
            "bubble": {
                "paperSource": "Figure 2(a)",
                "reconstructedRule": "Top-down controller repeatedly selects the leftmost inversion and swaps that element rightward until the right neighbor is greater or equal; repeats until sorted or no legal move.",
                "comparisonCounting": "Count each adjacent value comparison considered by the controller; exact convention will be tested in S05 sensitivity if needed.",
            },
            "insertion": {
                "paperSource": "Figure 2(c)",
                "reconstructedRule": "Maintain sorted prefix; choose the leftmost unsorted element and adjacent-swap it left until prefix order is restored.",
                "comparisonCounting": "Count each adjacent comparison used in the insertion while-loop.",
            },
            "selection": {
                "paperSource": "Figure 2(e)",
                "reconstructedRule": "For each sorted boundary, find the smallest unvisited value and put it into the next sorted position using one direct exchange as the primary movement-step convention.",
                "comparisonCounting": "Count scan comparisons used to identify the selected minimum.",
                "caveat": "If S04/S05 evidence indicates the original used adjacent swaps rather than direct exchange, report as a divergence and add a sensitivity condition without changing this baseline label.",
            },
        },
        "cellViewBaselines": {
            "sourceStatus": "public_repository_classes",
            "bubble": "modules/multithread/BubbleSortCell.py",
            "insertion": "modules/multithread/InsertionSortCell.py",
            "selection": "modules/multithread/SelectionSortCell.py",
            "probe": "modules/multithread/StatusProbe.py",
            "baseCell": "modules/multithread/MultiThreadCell.py",
        },
        "frozenCellSemantics": {
            "none": "No frozen indices; f=0 rows are unperturbed baselines reused by DG.",
            "passive": {
                "definition": "Frozen cells do not initiate moves but may be displaced by legal moves initiated by non-frozen cells.",
                "cellViewAssumption": "Matches the public base swap behavior in which an initiating FREEZE cell refuses to move while a FREEZE target can be moved; Selection requires explicit validation because it has special frozen-target code.",
                "traditionalAssumption": "A selected frozen element cannot be the active mover; non-frozen active elements may exchange with it if the algorithm's move rule selects that exchange.",
            },
            "stuck": {
                "definition": "Frozen cells neither initiate moves nor change position when targeted by another move.",
                "cellViewAssumption": "Requires wrapper-level target-frozen blocking because public MultiThreadCell.swap only checks the initiator status.",
                "traditionalAssumption": "Any controller action that would change a frozen index is disallowed; the controller proceeds to the next legal action or stops if none exists.",
                "sourceStatus": "reconstructed_semantic_not_active_public_runner",
            },
        },
        "metricDefinitions": {
            "sortedness_percent_primary": {
                "definition": "For a length-n trajectory state, numerator = 1 + count of adjacent ordered pairs i=1..n-1 satisfying the designated direction, using <= for increasing/nondecreasing and >= for decreasing/nonincreasing. Sortedness percent = 100 * numerator / n.",
                "paperLines": "Methods Sortedness Value lines 126-128",
                "caveat": "Public repository uses multiple monotonicity definitions; downstream wrappers must keep primary and legacy values separate.",
            },
            "monotonicity_error_count_primary": {
                "definition": "Count of adjacent order violations for the designated direction: arr[i] < arr[i-1] for increasing/nondecreasing and arr[i] > arr[i-1] for decreasing/nonincreasing. Maximum is n-1.",
                "paperLines": "Methods monotonicity error line 122",
            },
            "aggregation_left_neighbor_primary": {
                "definition": "Percentage of cells with a directly adjacent left neighbor of the same Algotype; first cell has no left neighbor and contributes 0 unless S10 sensitivity specifies otherwise.",
                "paperLines": "Methods Aggregation Value line 144",
                "legacySensitivity": "Also compute the repository's right-neighbor implementation from analysis/cell_type_aggregation_analysis.py for divergence logging.",
            },
            "delayed_gratification_primary": {
                "definition": "Backtracking/recovery metric over Sortedness or monotonicity-error trajectories as described in Methods and Figure 6; S08 must implement import-safe code and toy-case validation.",
                "paperLines": "Methods lines 136-138; Figure 6 caption line 214",
            },
        },
        "seedBanks": {
            "valueBanks": build_value_banks(),
            "frozenIndexBanks": build_frozen_banks(),
            "algotypeAssignmentBanks": build_algotype_assignment_banks(),
        },
        "conditions": conditions,
    }


def validate_config(config: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    errors: list[str] = []
    value_banks = config["seedBanks"]["valueBanks"]
    frozen_banks = config["seedBanks"]["frozenIndexBanks"]
    algotype_banks = config["seedBanks"]["algotypeAssignmentBanks"]
    conditions = config["conditions"]

    if len(conditions) != 56:
        errors.append(f"expected 56 condition rows, found {len(conditions)}")

    unique_bank = value_banks["unique_1_to_100"]
    duplicate_bank = value_banks["duplicate_1_to_10_x10"]
    for idx, arr in enumerate(unique_bank["initialArrays"]):
        if len(arr) != ARRAY_LENGTH or sorted(arr) != list(range(1, ARRAY_LENGTH + 1)):
            errors.append(f"unique bank repeat {idx} is not a permutation of 1..100")
    for idx, arr in enumerate(duplicate_bank["initialArrays"]):
        if len(arr) != ARRAY_LENGTH:
            errors.append(f"duplicate bank repeat {idx} length is {len(arr)}")
        counts = {value: arr.count(value) for value in range(1, 11)}
        if counts != {value: 10 for value in range(1, 11)}:
            errors.append(f"duplicate bank repeat {idx} counts are {counts}")

    for frozen_count in FROZEN_COUNTS:
        bank = frozen_banks[f"frozen_{frozen_count}"]
        if len(bank["indices"]) != REPEAT_COUNT:
            errors.append(f"frozen_{frozen_count} repeat count mismatch")
        for repeat_idx, indices in enumerate(bank["indices"]):
            if len(indices) != frozen_count or len(set(indices)) != frozen_count:
                errors.append(f"frozen_{frozen_count} repeat {repeat_idx} index count mismatch")
            if any(index < 0 or index >= ARRAY_LENGTH for index in indices):
                errors.append(f"frozen_{frozen_count} repeat {repeat_idx} index out of range")

    condition_ids = [row["condition_id"] for row in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        errors.append("condition IDs are not unique")
    for row in conditions:
        if row["repeat_count"] != REPEAT_COUNT:
            errors.append(f"{row['condition_id']} repeat_count mismatch")
        if row["array_length"] != ARRAY_LENGTH:
            errors.append(f"{row['condition_id']} array_length mismatch")
        if row["value_bank_id"] not in value_banks:
            errors.append(f"{row['condition_id']} references missing value bank {row['value_bank_id']}")
        if row["frozen_index_bank_id"] not in frozen_banks:
            errors.append(f"{row['condition_id']} references missing frozen bank {row['frozen_index_bank_id']}")
        bank_id = row["algotype_assignment_bank_id"]
        if bank_id != "not_applicable" and bank_id not in algotype_banks:
            errors.append(f"{row['condition_id']} references missing algotype bank {bank_id}")
        if row["mode"] == "traditional" and "reconstructed" not in row["baseline_source"]:
            errors.append(f"{row['condition_id']} traditional row is not marked reconstructed")

    families = {row["run_family"] for row in conditions}
    required_families = {
        "unperturbed_baseline",
        "frozen_robustness",
        "same_goal_chimera",
        "duplicate_value_chimera",
        "opposite_direction_chimera_unique",
        "opposite_direction_chimera_duplicate",
    }
    missing_families = sorted(required_families - families)
    if missing_families:
        errors.append(f"missing run families: {missing_families}")

    frozen_counts_in_conditions = sorted({int(row["frozen_count"]) for row in conditions})
    if frozen_counts_in_conditions != list(FROZEN_COUNTS):
        errors.append(f"condition matrix frozen counts are {frozen_counts_in_conditions}")

    summary = {
        "conditionRows": len(conditions),
        "runFamilyCounts": {
            family: sum(1 for row in conditions if row["run_family"] == family)
            for family in sorted(families)
        },
        "traditionalRows": sum(1 for row in conditions if row["mode"] == "traditional"),
        "cellViewRows": sum(1 for row in conditions if row["mode"] == "cell_view"),
        "repetitionCount": REPEAT_COUNT,
        "arrayLength": ARRAY_LENGTH,
        "frozenCountsInConditions": frozen_counts_in_conditions,
        "valueBanks": sorted(value_banks),
        "algotypeAssignmentBanks": sorted(algotype_banks),
    }
    return not errors, errors, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in CSV_FIELDS})


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / "S03"
    config_dir = artifacts_dir / "configs"
    table_dir = artifacts_dir / "tables"
    step_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(repo_dir)
    ok, errors, summary = validate_config(config)
    config["validationSummary"] = summary
    config["validationErrors"] = errors

    config_path = config_dir / "e01_baseline_configs.json"
    matrix_path = table_dir / "e01_condition_matrix.csv"
    validation_path = step_dir / "s03_validation.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"

    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(matrix_path, config["conditions"])

    artifacts_written = [str(config_path), str(matrix_path), str(validation_path), str(artifact_manifest_path)]
    validation = {
        "researchStepId": "S03",
        "stepNumber": 3,
        "success": ok,
        "status": "completed_with_caveats" if ok else "failed_validation",
        "artifactsWritten": artifacts_written,
        "validationResult": "passed_with_caveats" if ok else "failed",
        "validationSummary": summary,
        "validationErrors": errors,
        "caveatsOrBlockers": [
            "Traditional Bubble, Insertion, and Selection baselines are reconstructed because S02 found no public traditional-generation source or saved original arrays.",
            "Passive and stuck Frozen Cell semantics are frozen assumptions and require focused implementation tests before S07.",
            "Metric formulas are frozen as primary definitions, but legacy repository definitions must be retained for divergence checks.",
        ],
        "recommendedNextAction": "Proceed to S04 only after using this config as the sole source of array seeds, condition IDs, and reconstructed-baseline labels.",
    }
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifact_manifest = {
        "researchStepId": "S03",
        "stepNumber": 3,
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "artifacts": [
            {
                "path": str(config_path),
                "kind": "baseline_config",
                "sha256": sha256_file(config_path),
                "description": "Machine-readable frozen baseline configuration and deterministic seed banks.",
            },
            {
                "path": str(matrix_path),
                "kind": "condition_matrix",
                "sha256": sha256_file(matrix_path),
                "description": "Condition-level matrix for S04-S12 downstream runs.",
            },
            {
                "path": str(validation_path),
                "kind": "validation",
                "sha256": sha256_file(validation_path),
                "description": "S03 validation summary.",
            },
        ],
        "summary": summary,
    }
    artifact_manifest_path.write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Refresh manifest entry after artifact_manifest has its own checksum.
    artifact_manifest["artifacts"].append(
        {
            "path": str(artifact_manifest_path),
            "kind": "artifact_manifest",
            "sha256": None,
            "checksumNote": "Self-referential checksum omitted here; final file hash is tracked in /artifacts/checksums/sha256sums.txt.",
            "description": "Manifest for S03 outputs.",
        }
    )
    artifact_manifest_path.write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
