"""Seeded homeostatic benchmark definitions for E04 S05."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from e02_deterministic_simulator.metrics import sortedness_raw


HOMEOSTASIS_BENCHMARK_VERSION = "e04_s05_homeostatic_tasks.v1"
PERTURBATION_TYPES = ("swap", "insert", "delete", "freeze", "recover", "damage")
TRIVIAL_CONTROLLERS = ("noop", "oracle_sort")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if hasattr(value, "item"):
        return _json_ready(value.item())
    if isinstance(value, float):
        return round(float(value), 12)
    return value


def stable_json(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def normalized_sortedness_record(
    values: Sequence[int],
    *,
    initial_length: int,
    direction: str = "increasing",
) -> dict[str, Any]:
    """Return dynamic-length sortedness metrics with explicit denominators."""

    current_length = len(values)
    current_adjacent_pairs = max(0, current_length - 1)
    current_denominator = max(1, current_adjacent_pairs)
    initial_adjacent_pairs = max(0, int(initial_length) - 1)
    initial_denominator = max(1, initial_adjacent_pairs)
    sorted_pair_count = int(sortedness_raw(values, direction=direction)) if current_length > 1 else 0
    monotonicity_error = max(0, current_adjacent_pairs - sorted_pair_count)
    sortedness_fraction = 1.0 if current_adjacent_pairs == 0 else sorted_pair_count / current_denominator
    return {
        "values": [int(value) for value in values],
        "currentLength": int(current_length),
        "initialLength": int(initial_length),
        "lengthRatio": float(current_length / max(1, int(initial_length))),
        "currentAdjacentPairCount": int(current_adjacent_pairs),
        "currentPairDenominator": int(current_denominator),
        "initialAdjacentPairCount": int(initial_adjacent_pairs),
        "initialPairDenominator": int(initial_denominator),
        "sortedPairCount": int(sorted_pair_count),
        "sortednessFraction": float(sortedness_fraction),
        "sortednessPercent": float(100.0 * sortedness_fraction),
        "sortedPairsPerInitialDenominator": float(sorted_pair_count / initial_denominator),
        "monotonicityError": int(monotonicity_error),
        "monotonicityErrorFraction": float(monotonicity_error / current_denominator),
        "normalizationVersion": HOMEOSTASIS_BENCHMARK_VERSION,
    }


def apply_perturbation(
    values: Sequence[int],
    event: Mapping[str, Any],
    *,
    frozen_cell_ids: Iterable[int] | None = None,
    damaged_cell_ids: Iterable[int] | None = None,
) -> tuple[list[int], set[int], set[int]]:
    """Apply one benchmark perturbation to a value array and simple state sets."""

    updated = [int(value) for value in values]
    frozen = set(int(value) for value in (frozen_cell_ids or ()))
    damaged = set(int(value) for value in (damaged_cell_ids or ()))
    event_type = str(event["type"])
    if event_type not in PERTURBATION_TYPES:
        raise ValueError(f"unknown perturbation type: {event_type}")
    if event_type == "swap":
        left = int(event["leftIndex"])
        right = int(event.get("rightIndex", left + 1))
        if not (0 <= left < len(updated) and 0 <= right < len(updated)):
            raise ValueError(f"swap indices out of bounds: {left}, {right}")
        updated[left], updated[right] = updated[right], updated[left]
    elif event_type == "insert":
        position = int(event["position"])
        if not 0 <= position <= len(updated):
            raise ValueError(f"insert position out of bounds: {position}")
        updated.insert(position, int(event["value"]))
    elif event_type == "delete":
        position = int(event["position"])
        if not 0 <= position < len(updated):
            raise ValueError(f"delete position out of bounds: {position}")
        del updated[position]
    elif event_type == "freeze":
        frozen.add(int(event["cellId"]))
    elif event_type == "recover":
        frozen.discard(int(event["cellId"]))
    elif event_type == "damage":
        damaged.add(int(event["cellId"]))
    return updated, frozen, damaged


def generate_seeded_schedule(
    *,
    initial_values: Sequence[int],
    schedule_seed: int,
    event_templates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expand event templates into a deterministic perturbation schedule."""

    rng = np.random.default_rng(int(schedule_seed))
    reference_values = [int(value) for value in initial_values]
    frozen: set[int] = set()
    damaged: set[int] = set()
    schedule: list[dict[str, Any]] = []
    for index, template in enumerate(sorted(event_templates, key=lambda item: (int(item["tick"]), str(item["type"])))):
        event_type = str(template["type"])
        if event_type not in PERTURBATION_TYPES:
            raise ValueError(f"unknown perturbation type: {event_type}")
        event: dict[str, Any] = {
            "eventIndex": int(index),
            "tick": int(template["tick"]),
            "type": event_type,
            "source": template.get("source", "template"),
        }
        if event_type == "swap":
            if "leftIndex" in template:
                left = int(template["leftIndex"])
            else:
                left = int(rng.integers(0, max(1, len(reference_values) - 1)))
            event["leftIndex"] = left
            event["rightIndex"] = int(template.get("rightIndex", min(left + 1, len(reference_values) - 1)))
        elif event_type == "insert":
            position = int(template.get("position", rng.integers(0, len(reference_values) + 1)))
            value = int(template.get("value", rng.integers(1, max(2, len(reference_values) * 2 + 2))))
            event["position"] = position
            event["value"] = value
        elif event_type == "delete":
            position = int(template.get("position", rng.integers(0, max(1, len(reference_values)))))
            event["position"] = position
        elif event_type in {"freeze", "recover", "damage"}:
            event["cellId"] = int(template.get("cellId", rng.integers(0, max(1, len(reference_values)))))
            if event_type == "damage":
                event["durationTicks"] = int(template.get("durationTicks", 2))
        reference_values, frozen, damaged = apply_perturbation(
            reference_values,
            event,
            frozen_cell_ids=frozen,
            damaged_cell_ids=damaged,
        )
        event["postReferenceLength"] = len(reference_values)
        event["scheduleSeed"] = int(schedule_seed)
        schedule.append(event)
    return schedule


def build_homeostatic_benchmark_config() -> dict[str, Any]:
    """Build the S05 seed benchmark suite with expanded schedules."""

    tasks = [
        {
            "taskId": "steady_sorted_no_perturbation",
            "description": "Trivial sorted control for no-op controller sanity checks.",
            "initialValues": [1, 2, 3, 4, 5, 6],
            "horizonTicks": 6,
            "scheduleSeed": 5001,
            "sortednessThresholdPercent": 100.0,
            "eventTemplates": [],
        },
        {
            "taskId": "adjacent_swap_recovery",
            "description": "Seeded adjacent swaps test recovery-time and failure-duration metrics.",
            "initialValues": [1, 2, 3, 4, 5, 6, 7, 8],
            "horizonTicks": 8,
            "scheduleSeed": 5002,
            "sortednessThresholdPercent": 95.0,
            "eventTemplates": [
                {"tick": 1, "type": "swap", "source": "seeded_random_adjacent"},
                {"tick": 4, "type": "swap", "source": "seeded_random_adjacent"},
            ],
        },
        {
            "taskId": "insertion_deletion_normalization",
            "description": "Insertion/deletion task used to validate current-length metric normalization.",
            "initialValues": [1, 2, 3, 4],
            "horizonTicks": 5,
            "scheduleSeed": 5003,
            "sortednessThresholdPercent": 95.0,
            "eventTemplates": [
                {"tick": 1, "type": "insert", "position": 0, "value": 99, "source": "fixed_normalization_probe"},
                {"tick": 3, "type": "delete", "position": 0, "source": "fixed_normalization_probe"},
            ],
        },
        {
            "taskId": "frozen_damage_mixed_events",
            "description": "Schema-level homeostasis task combining Frozen Cell recovery and damage events.",
            "initialValues": [1, 2, 3, 4, 5, 6],
            "horizonTicks": 7,
            "scheduleSeed": 5004,
            "sortednessThresholdPercent": 95.0,
            "eventTemplates": [
                {"tick": 1, "type": "freeze", "cellId": 2, "source": "fixed_freeze"},
                {"tick": 2, "type": "damage", "cellId": 3, "durationTicks": 2, "source": "fixed_damage"},
                {"tick": 4, "type": "recover", "cellId": 2, "source": "fixed_recover"},
                {"tick": 5, "type": "swap", "source": "seeded_random_adjacent"},
            ],
        },
    ]
    expanded_tasks = []
    for task in tasks:
        schedule = generate_seeded_schedule(
            initial_values=task["initialValues"],
            schedule_seed=task["scheduleSeed"],
            event_templates=task["eventTemplates"],
        )
        expanded = {key: value for key, value in task.items() if key != "eventTemplates"}
        expanded["perturbationSchedule"] = schedule
        expanded["scheduleHash"] = stable_hash(schedule)
        expanded_tasks.append(expanded)
    return {
        "schema": "eidosoma.e04_homeostatic_tasks.v1",
        "producerStep": "S05",
        "homeostasisBenchmarkVersion": HOMEOSTASIS_BENCHMARK_VERSION,
        "updateOrder": "perturb_then_score_then_trivial_controller_then_score",
        "normalization": normalization_contract(),
        "trivialControllers": [
            {"controllerId": "noop", "description": "No action after perturbation."},
            {
                "controllerId": "oracle_sort",
                "description": "Validation-only global sort after perturbation; not a local policy baseline.",
            },
        ],
        "tasks": expanded_tasks,
    }


def normalization_contract() -> dict[str, Any]:
    return {
        "sortednessDenominator": "current adjacent-pair count at each tick, using denominator 1 for length 0 or 1",
        "failureCriterion": "post-control sortednessPercent below task sortednessThresholdPercent",
        "recoveryTimeTicks": "first post-control tick at or above threshold after a perturbation that made pre-control state fail",
        "lengthChangingEvents": "insert/delete recompute all denominators from current length before thresholding",
        "energyProxy": "number of controller ticks that changed the value sequence for trivial controllers",
    }


def apply_trivial_controller(values: Sequence[int], controller_id: str) -> tuple[list[int], bool]:
    current = [int(value) for value in values]
    if controller_id == "noop":
        return current, False
    if controller_id == "oracle_sort":
        sorted_values = sorted(current)
        return sorted_values, sorted_values != current
    raise ValueError(f"unknown trivial controller: {controller_id}")


def evaluate_trivial_controller(task: Mapping[str, Any], controller_id: str) -> dict[str, Any]:
    if controller_id not in TRIVIAL_CONTROLLERS:
        raise ValueError(f"unknown trivial controller: {controller_id}")
    values = [int(value) for value in task["initialValues"]]
    initial_length = len(values)
    frozen: set[int] = set()
    damaged: set[int] = set()
    threshold = float(task["sortednessThresholdPercent"])
    records: list[dict[str, Any]] = []
    energy_proxy = 0
    schedule = list(task.get("perturbationSchedule", ()))
    for tick in range(int(task["horizonTicks"])):
        events = [event for event in schedule if int(event["tick"]) == tick]
        for event in events:
            values, frozen, damaged = apply_perturbation(values, event, frozen_cell_ids=frozen, damaged_cell_ids=damaged)
        pre = normalized_sortedness_record(values, initial_length=initial_length)
        values, changed = apply_trivial_controller(values, controller_id)
        energy_proxy += int(changed)
        post = normalized_sortedness_record(values, initial_length=initial_length)
        records.append(
            {
                "tick": int(tick),
                "events": events,
                "preSortednessPercent": pre["sortednessPercent"],
                "postSortednessPercent": post["sortednessPercent"],
                "postInRange": bool(post["sortednessPercent"] >= threshold),
                "currentLength": post["currentLength"],
                "currentPairDenominator": post["currentPairDenominator"],
                "frozenCellCount": len(frozen),
                "damagedCellCount": len(damaged),
                "values": list(values),
            }
        )
    failure_ticks = [record for record in records if not record["postInRange"]]
    recovery_times: list[int | None] = []
    for event in schedule:
        event_tick = int(event["tick"])
        pre_record = next((record for record in records if record["tick"] == event_tick), None)
        if pre_record is None or float(pre_record["preSortednessPercent"]) >= threshold:
            recovery_times.append(0)
            continue
        recovered_tick = next(
            (record["tick"] for record in records if record["tick"] >= event_tick and record["postInRange"]),
            None,
        )
        recovery_times.append(None if recovered_tick is None else int(recovered_tick - event_tick))
    finite_recovery_times = [value for value in recovery_times if value is not None]
    return {
        "taskId": str(task["taskId"]),
        "controllerId": controller_id,
        "horizonTicks": int(task["horizonTicks"]),
        "timeInRangeFraction": float((len(records) - len(failure_ticks)) / max(1, len(records))),
        "failureDurationTicks": int(len(failure_ticks)),
        "energyProxy": int(energy_proxy),
        "maxRecoveryTimeTicks": max(finite_recovery_times) if finite_recovery_times else None,
        "unrecoveredPerturbationCount": int(sum(1 for value in recovery_times if value is None)),
        "tickRecords": records,
        "recoveryTimes": recovery_times,
        "finalValues": list(values),
        "normalizationVersion": HOMEOSTASIS_BENCHMARK_VERSION,
    }
