#!/usr/bin/env python3
"""Execute E02 S12 Frozen Cell behavior-variant tests.

S12 keeps the S01 deterministic simulator policy code intact for baseline
passive/stuck controls, then layers explicit defect-behavior gates over frozen
target swaps for probabilistic, time-varying, fatigue/recovery, and directional
sticky variants. The step logs defect decisions and state transitions.
"""

from __future__ import annotations

import argparse
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

from e02_deterministic_simulator import DeterministicEventSimulator, aggregation, state_hash
from scripts.e02_s08_dg_nulls import delayed_gratification_from_sortedness
from scripts.e02_s09_alternative_metrics import prefixed_metrics
from scripts.e02_s10_input_distributions import compact_json, input_values_for_profile, json_ready, markdown_table, sha256_path, stable_seed, write_json
from scripts.e02_s11_frozen_placement import (
    frozen_positions_for_category,
    frozen_value_metadata,
    placement_label,
    validate_placement_category,
)


EXPERIMENT_ID = "E02"
STEP_ID = "S12"
STEP_NUMBER = 12
ALGORITHMS = ["bubble", "insertion", "selection"]
PLACEMENT_CATEGORIES = ["center", "random", "array_ends"]
FROZEN_COUNT = 2
DEFAULT_E01_FROZEN_PATH = Path("/previous-artifacts/E01/results/e01_frozen_cell_robustness.parquet")
DEFAULT_S11_PLACEMENT_PATH = Path("/artifacts/results/e02_frozen_placement.parquet")


@dataclass(frozen=True)
class BehaviorSpec:
    behavior_id: str
    behavior_family: str
    base_frozen_variant: str
    description: str
    block_probability: float | None = None
    phase_length: int | None = None
    fatigue_threshold: int | None = None
    recovery_window: int | None = None
    blocked_direction: str | None = None
    requires_transition_logging: bool = False


@dataclass(frozen=True)
class BehaviorCondition:
    algorithm: str
    placement_category: str
    behavior: BehaviorSpec
    frozen_count: int


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


def behavior_specs() -> list[BehaviorSpec]:
    return [
        BehaviorSpec(
            behavior_id="baseline_passive",
            behavior_family="baseline",
            base_frozen_variant="passive",
            description="Original S01 passive Frozen Cell: cannot initiate, can be moved by active cells.",
        ),
        BehaviorSpec(
            behavior_id="baseline_stuck",
            behavior_family="baseline",
            base_frozen_variant="stuck",
            description="Original S01 stuck Frozen Cell: cannot initiate and blocks swaps involving the frozen target.",
        ),
        BehaviorSpec(
            behavior_id="probabilistic_stuck_p50",
            behavior_family="probabilistic",
            base_frozen_variant="passive",
            description="Frozen target blocks each attempted incoming swap with probability 0.5.",
            block_probability=0.5,
        ),
        BehaviorSpec(
            behavior_id="time_varying_stuck_periodic",
            behavior_family="time_varying",
            base_frozen_variant="passive",
            description="Frozen target alternates between stuck and permissive phases every 12 activations.",
            phase_length=12,
            requires_transition_logging=True,
        ),
        BehaviorSpec(
            behavior_id="fatigue_recovery",
            behavior_family="fatigue_recovery",
            base_frozen_variant="passive",
            description="Frozen target becomes temporarily stuck after two incoming nudges, then recovers after eight activations.",
            fatigue_threshold=2,
            recovery_window=8,
            requires_transition_logging=True,
        ),
        BehaviorSpec(
            behavior_id="directional_sticky_from_left",
            behavior_family="directional_sticky",
            base_frozen_variant="passive",
            description="Frozen target blocks incoming swaps from the left but allows incoming swaps from the right.",
            blocked_direction="from_left",
        ),
        BehaviorSpec(
            behavior_id="directional_sticky_from_right",
            behavior_family="directional_sticky",
            base_frozen_variant="passive",
            description="Frozen target blocks incoming swaps from the right but allows incoming swaps from the left.",
            blocked_direction="from_right",
        ),
    ]


def required_behavior_ids() -> set[str]:
    return {spec.behavior_id for spec in behavior_specs()}


def behavior_conditions() -> list[BehaviorCondition]:
    return [
        BehaviorCondition(algorithm=algorithm, placement_category=category, behavior=spec, frozen_count=FROZEN_COUNT)
        for algorithm in ALGORITHMS
        for category in PLACEMENT_CATEGORIES
        for spec in behavior_specs()
    ]


def behavior_state_label(spec: BehaviorSpec, activation_count: int) -> str:
    if spec.behavior_id == "baseline_passive":
        return "permissive"
    if spec.behavior_id == "baseline_stuck":
        return "stuck"
    if spec.behavior_id == "time_varying_stuck_periodic":
        phase_length = int(spec.phase_length or 1)
        phase_index = (activation_count // phase_length) % 2
        return "stuck" if phase_index == 0 else "permissive"
    if spec.behavior_id == "fatigue_recovery":
        return "permissive"
    if spec.behavior_family in {"probabilistic", "directional_sticky"}:
        return "stochastic_gate" if spec.behavior_family == "probabilistic" else "directional_gate"
    return "permissive"


class BehaviorVariantSimulator(DeterministicEventSimulator):
    """S12 wrapper that gates swaps into frozen targets without rewriting policy logic."""

    def __init__(
        self,
        initial_values: list[int],
        algorithm: str,
        *,
        behavior: BehaviorSpec,
        behavior_seed: int,
        frozen_positions: list[int],
        scheduler_seed: int,
        tie_breaker_seed: int,
        condition_id: str,
    ) -> None:
        self.behavior = behavior
        self.behavior_seed = int(behavior_seed)
        self.behavior_rng = np.random.default_rng(self.behavior_seed)
        self.behavior_event_log: list[dict[str, Any]] = []
        self.behavior_state_by_cell_id: dict[int, dict[str, Any]] = {}
        self.behavior_decision_count = 0
        self.behavior_blocked_attempts = 0
        self.behavior_allowed_frozen_target_swaps = 0
        super().__init__(
            initial_values,
            algorithm,
            frozen_positions=frozen_positions,
            frozen_variant=behavior.base_frozen_variant,
            scheduler_seed=scheduler_seed,
            tie_breaker_seed=tie_breaker_seed,
            condition_id=condition_id,
            implementation="cell_view",
            research_step_id=STEP_ID,
        )
        for cell_id in self.frozen_cell_ids():
            self.behavior_state_by_cell_id[cell_id] = {
                "state": behavior_state_label(behavior, 0),
                "nudgeCount": 0,
                "recoveryUntilActivation": None,
            }
            self._log_behavior_event(
                event_type="initial_state",
                frozen_cell_id=cell_id,
                old_state=None,
                new_state=self.behavior_state_by_cell_id[cell_id]["state"],
                reason="initial_behavior_state",
            )

    def _log_behavior_event(
        self,
        *,
        event_type: str,
        frozen_cell_id: int | None,
        old_state: str | None,
        new_state: str | None,
        reason: str,
        actor_cell_id: int | None = None,
        actor_position: int | None = None,
        target_position: int | None = None,
        incoming_direction: str | None = None,
        blocked: bool | None = None,
        random_draw: float | None = None,
    ) -> None:
        self.behavior_event_log.append(
            {
                "eventIndex": len(self.behavior_event_log),
                "eventType": event_type,
                "activationIndex": int(self.activation_count) if hasattr(self, "activation_count") else 0,
                "swapCount": int(self.swap_count) if hasattr(self, "swap_count") else 0,
                "behaviorVariant": self.behavior.behavior_id,
                "behaviorFamily": self.behavior.behavior_family,
                "frozenCellId": frozen_cell_id,
                "oldState": old_state,
                "newState": new_state,
                "reason": reason,
                "actorCellId": actor_cell_id,
                "actorPosition": actor_position,
                "targetPosition": target_position,
                "incomingDirection": incoming_direction,
                "blocked": blocked,
                "randomDraw": random_draw,
            }
        )

    def _transition_state(self, frozen_cell_id: int, new_state: str, reason: str) -> None:
        state = self.behavior_state_by_cell_id[frozen_cell_id]
        old_state = str(state["state"])
        if old_state == new_state:
            return
        state["state"] = new_state
        self._log_behavior_event(
            event_type="transition",
            frozen_cell_id=frozen_cell_id,
            old_state=old_state,
            new_state=new_state,
            reason=reason,
        )

    def _update_time_varying_states(self) -> None:
        if self.behavior.behavior_id != "time_varying_stuck_periodic":
            return
        new_state = behavior_state_label(self.behavior, self.activation_count)
        for cell_id in self.frozen_cell_ids():
            self._transition_state(cell_id, new_state, "periodic_phase_update")

    def _update_fatigue_recovery(self, frozen_cell_id: int) -> None:
        if self.behavior.behavior_id != "fatigue_recovery":
            return
        state = self.behavior_state_by_cell_id[frozen_cell_id]
        recovery_until = state.get("recoveryUntilActivation")
        if state["state"] == "fatigued_stuck" and recovery_until is not None and self.activation_count >= int(recovery_until):
            state["nudgeCount"] = 0
            state["recoveryUntilActivation"] = None
            self._transition_state(frozen_cell_id, "recovered_permissive", "recovery_window_elapsed")

    def _incoming_direction(self, actor_pos: int, target_pos: int) -> str:
        if actor_pos < target_pos:
            return "from_left"
        if actor_pos > target_pos:
            return "from_right"
        return "same_position"

    def _target_can_eventually_allow(self, actor_pos: int, target_pos: int) -> bool:
        if not self._target_in_bounds(target_pos):
            return False
        target = self.cells[target_pos]
        if not target.frozen:
            return True
        if self.behavior.behavior_id == "baseline_stuck":
            return False
        if self.behavior.behavior_family == "directional_sticky":
            return self._incoming_direction(actor_pos, target_pos) != self.behavior.blocked_direction
        return True

    def _dynamic_blocks_target(self, actor_pos: int, target_pos: int) -> tuple[bool, str, float | None]:
        target = self.cells[target_pos]
        frozen_cell_id = target.cell_id
        incoming_direction = self._incoming_direction(actor_pos, target_pos)
        if self.behavior.behavior_id == "probabilistic_stuck_p50":
            draw = float(self.behavior_rng.random())
            blocked = draw < float(self.behavior.block_probability or 0.0)
            return blocked, "probabilistic_gate", draw
        if self.behavior.behavior_id == "time_varying_stuck_periodic":
            self._update_time_varying_states()
            state = self.behavior_state_by_cell_id[frozen_cell_id]["state"]
            return state == "stuck", f"time_varying_state_{state}", None
        if self.behavior.behavior_id == "fatigue_recovery":
            self._update_fatigue_recovery(frozen_cell_id)
            state = self.behavior_state_by_cell_id[frozen_cell_id]
            if state["state"] == "fatigued_stuck":
                return True, "fatigue_recovery_window_active", None
            state["nudgeCount"] = int(state["nudgeCount"]) + 1
            if int(state["nudgeCount"]) >= int(self.behavior.fatigue_threshold or 1):
                state["recoveryUntilActivation"] = int(self.activation_count + int(self.behavior.recovery_window or 1))
                self._transition_state(frozen_cell_id, "fatigued_stuck", "fatigue_threshold_reached")
            return False, "fatigue_permissive_nudge", None
        if self.behavior.behavior_family == "directional_sticky":
            return incoming_direction == self.behavior.blocked_direction, "directional_sticky_gate", None
        return False, "permissive_gate", None

    def _can_swap(self, actor_pos: int, target_pos: int) -> bool:
        if self.behavior.behavior_family == "baseline":
            return super()._can_swap(actor_pos, target_pos)
        if not self._target_in_bounds(target_pos):
            return False
        if self.cells[actor_pos].frozen:
            self._count_frozen_attempt(actor_pos)
            return False
        target = self.cells[target_pos]
        if not target.frozen:
            return True

        blocked, reason, random_draw = self._dynamic_blocks_target(actor_pos, target_pos)
        self.behavior_decision_count += 1
        incoming_direction = self._incoming_direction(actor_pos, target_pos)
        self._log_behavior_event(
            event_type="decision",
            frozen_cell_id=target.cell_id,
            old_state=str(self.behavior_state_by_cell_id[target.cell_id]["state"]),
            new_state=str(self.behavior_state_by_cell_id[target.cell_id]["state"]),
            reason=reason,
            actor_cell_id=self.cells[actor_pos].cell_id,
            actor_position=actor_pos,
            target_position=target_pos,
            incoming_direction=incoming_direction,
            blocked=blocked,
            random_draw=random_draw,
        )
        if blocked:
            self.blocked_move_attempts += 1
            self.behavior_blocked_attempts += 1
            return False
        self.behavior_allowed_frozen_target_swaps += 1
        return True

    def step(self, *, forced_cell_id: int | None = None, forced_direction: int | None = None):  # type: ignore[override]
        self._update_time_varying_states()
        outcome = super().step(forced_cell_id=forced_cell_id, forced_direction=forced_direction)
        self._update_time_varying_states()
        return outcome

    def legal_action_exists(self) -> bool:
        for cell in self.cells:
            if cell.frozen:
                continue
            pos = self.positions_by_id[cell.cell_id]
            if cell.algotype == "bubble":
                for direction in (-1, 1):
                    target = pos + direction
                    if not self._target_in_bounds(target):
                        continue
                    if cell.reverse_direction:
                        should_swap = cell.value < self.cells[target].value if direction == 1 else cell.value > self.cells[target].value
                    else:
                        should_swap = cell.value > self.cells[target].value if direction == 1 else cell.value < self.cells[target].value
                    if should_swap and self._target_can_eventually_allow(pos, target):
                        return True
            elif cell.algotype == "insertion":
                target = pos - 1
                if self._target_in_bounds(target) and self._insertion_prefix_enabled(pos, cell.reverse_direction):
                    if cell.reverse_direction and cell.value > self.cells[target].value and self._target_can_eventually_allow(pos, target):
                        return True
                    if not cell.reverse_direction and cell.value < self.cells[target].value and self._target_can_eventually_allow(pos, target):
                        return True
            elif cell.algotype == "selection":
                ideal = cell.ideal_position
                if ideal is not None and self._target_in_bounds(ideal) and pos != ideal and self._target_can_eventually_allow(pos, ideal):
                    return True
        return False

    def behavior_state_snapshot(self) -> dict[str, Any]:
        return {
            str(cell_id): {
                "state": state["state"],
                "nudgeCount": int(state["nudgeCount"]),
                "recoveryUntilActivation": state["recoveryUntilActivation"],
            }
            for cell_id, state in sorted(self.behavior_state_by_cell_id.items())
        }


def add_s12_trace_logging(sim: BehaviorVariantSimulator, frozen_cell_ids: list[int]) -> None:
    def enrich_last_row() -> None:
        positions = sim.position_by_cell_id()
        id_positions = {str(cell_id): int(positions[cell_id]) for cell_id in frozen_cell_ids}
        sim.trace_rows[-1]["frozen_cell_ids_json"] = compact_json(frozen_cell_ids)
        sim.trace_rows[-1]["frozen_cell_id_positions_json"] = compact_json(id_positions)
        sim.trace_rows[-1]["defect_behavior_variant"] = sim.behavior.behavior_id
        sim.trace_rows[-1]["defect_behavior_state_json"] = compact_json(sim.behavior_state_snapshot())

    enrich_last_row()
    original_append = sim._append_trace_row

    def wrapped_append(**kwargs: Any) -> None:
        original_append(**kwargs)
        enrich_last_row()

    sim._append_trace_row = wrapped_append  # type: ignore[method-assign]


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


def condition_seeds(condition: BehaviorCondition, replicate_index: int) -> dict[str, int]:
    return {
        "inputSeed": stable_seed(STEP_ID, "random_unique_input", replicate_index),
        "schedulerSeed": stable_seed(STEP_ID, condition.algorithm, condition.placement_category, condition.behavior.behavior_id, "scheduler", replicate_index),
        "tieBreakerSeed": stable_seed(STEP_ID, condition.algorithm, condition.placement_category, condition.behavior.behavior_id, "tie", replicate_index),
        "frozenPositionSeed": stable_seed(STEP_ID, condition.algorithm, condition.placement_category, "frozen", replicate_index),
        "behaviorSeed": stable_seed(STEP_ID, condition.algorithm, condition.placement_category, condition.behavior.behavior_id, "behavior", replicate_index),
    }


def run_vanilla_baseline(
    *,
    initial_values: list[int],
    condition: BehaviorCondition,
    frozen_positions: list[int],
    scheduler_seed: int,
    tie_seed: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
) -> Any:
    sim = DeterministicEventSimulator(
        initial_values,
        condition.algorithm,
        frozen_positions=frozen_positions,
        frozen_variant=condition.behavior.base_frozen_variant,
        scheduler_seed=scheduler_seed,
        tie_breaker_seed=tie_seed,
        condition_id=f"vanilla_{condition.algorithm}_{condition.placement_category}_{condition.behavior.behavior_id}",
        implementation="cell_view",
        research_step_id=STEP_ID,
    )
    return sim.run(
        max_activations=max_activations,
        max_swaps=max_swaps,
        max_comparisons=max_comparisons,
        no_move_checks_required=2,
        no_move_check_interval=max(1, len(initial_values)),
    )


def compare_baseline(custom_result: Any, vanilla_result: Any, condition_id: str, behavior_id: str) -> dict[str, Any]:
    trace_hashes_equal = [row["state_hash"] for row in custom_result.trace_rows] == [row["state_hash"] for row in vanilla_result.trace_rows]
    fields_equal = (
        custom_result.completed == vanilla_result.completed
        and custom_result.stop_reason == vanilla_result.stop_reason
        and custom_result.final_values == vanilla_result.final_values
        and custom_result.final_algotypes == vanilla_result.final_algotypes
        and custom_result.final_frozen_positions == vanilla_result.final_frozen_positions
        and custom_result.swap_count == vanilla_result.swap_count
        and custom_result.comparison_count == vanilla_result.comparison_count
        and custom_result.archived_compare_and_swap_count == vanilla_result.archived_compare_and_swap_count
        and custom_result.blocked_move_attempts == vanilla_result.blocked_move_attempts
        and custom_result.activation_count == vanilla_result.activation_count
    )
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "conditionId": condition_id,
        "defectBehaviorVariant": behavior_id,
        "baselineReferenceVariant": behavior_id.replace("baseline_", ""),
        "passed": bool(fields_equal and trace_hashes_equal),
        "traceHashesEqual": bool(trace_hashes_equal),
        "finalValuesEqual": custom_result.final_values == vanilla_result.final_values,
        "finalFrozenPositionsEqual": custom_result.final_frozen_positions == vanilla_result.final_frozen_positions,
        "countsEqual": bool(fields_equal),
        "customStopReason": custom_result.stop_reason,
        "vanillaStopReason": vanilla_result.stop_reason,
        "customSwapCount": int(custom_result.swap_count),
        "vanillaSwapCount": int(vanilla_result.swap_count),
        "customActivationCount": int(custom_result.activation_count),
        "vanillaActivationCount": int(vanilla_result.activation_count),
    }


def run_one_condition(
    *,
    condition: BehaviorCondition,
    n: int,
    replicate_index: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    seeds = condition_seeds(condition, replicate_index)
    initial_values = input_values_for_profile("random_unique", n, seeds["inputSeed"])
    frozen_positions = frozen_positions_for_category(
        condition.placement_category,
        initial_values,
        seeds["frozenPositionSeed"],
        condition.frozen_count,
    )
    condition_id = f"S12_{condition.algorithm}_{condition.placement_category}_{condition.behavior.behavior_id}_rep{replicate_index:03d}"
    sim = BehaviorVariantSimulator(
        initial_values,
        condition.algorithm,
        behavior=condition.behavior,
        behavior_seed=seeds["behaviorSeed"],
        frozen_positions=frozen_positions,
        scheduler_seed=seeds["schedulerSeed"],
        tie_breaker_seed=seeds["tieBreakerSeed"],
        condition_id=condition_id,
    )
    frozen_cell_ids = sim.frozen_cell_ids()
    initial_id_positions = {str(cell_id): int(sim.position_by_cell_id()[cell_id]) for cell_id in frozen_cell_ids}
    add_s12_trace_logging(sim, frozen_cell_ids)
    result = sim.run(
        max_activations=max_activations,
        max_swaps=max_swaps,
        max_comparisons=max_comparisons,
        no_move_checks_required=2,
        no_move_check_interval=max(1, n),
    )
    final_id_positions = {str(cell_id): int(sim.position_by_cell_id()[cell_id]) for cell_id in frozen_cell_ids}
    final_frozen_cell_ids = sim.frozen_cell_ids()
    value_meta = frozen_value_metadata(initial_values, frozen_positions)
    displacement_by_id = {
        str(cell_id): int(final_id_positions[str(cell_id)] - initial_id_positions[str(cell_id)])
        for cell_id in frozen_cell_ids
    }
    abs_displacements = [abs(value) for value in displacement_by_id.values()]
    signed_displacements = list(displacement_by_id.values())
    initial_metrics = prefixed_metrics("initial", initial_values)
    final_metrics = prefixed_metrics("final", result.final_values)
    trajectory_metrics = trajectory_proxy_metrics(result)
    baseline_check = None
    if condition.behavior.behavior_family == "baseline":
        vanilla = run_vanilla_baseline(
            initial_values=initial_values,
            condition=condition,
            frozen_positions=frozen_positions,
            scheduler_seed=seeds["schedulerSeed"],
            tie_seed=seeds["tieBreakerSeed"],
            max_activations=max_activations,
            max_swaps=max_swaps,
            max_comparisons=max_comparisons,
        )
        baseline_check = compare_baseline(result, vanilla, condition_id, condition.behavior.behavior_id)

    record: dict[str, Any] = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "conditionId": condition_id,
        "conditionTemplateId": f"{condition.algorithm}_{condition.placement_category}_{condition.behavior.behavior_id}",
        "implementation": "S01_deterministic_event_simulator_with_s12_defect_gate",
        "algorithm": condition.algorithm,
        "inputProfile": "random_unique",
        "n": int(n),
        "replicateIndex": int(replicate_index),
        "replicateNumber": int(replicate_index + 1),
        "inputSeed": int(seeds["inputSeed"]),
        "schedulerSeed": int(seeds["schedulerSeed"]),
        "tieBreakerSeed": int(seeds["tieBreakerSeed"]),
        "frozenPositionSeed": int(seeds["frozenPositionSeed"]),
        "behaviorSeed": int(seeds["behaviorSeed"]),
        "frozenCount": int(condition.frozen_count),
        "baseFrozenVariantUsed": condition.behavior.base_frozen_variant,
        "defectBehaviorVariant": condition.behavior.behavior_id,
        "defectBehaviorFamily": condition.behavior.behavior_family,
        "defectBehaviorDescription": condition.behavior.description,
        "placementCategory": condition.placement_category,
        "placementCategoryLabel": placement_label(condition.placement_category),
        "placementCategoryValid": bool(validate_placement_category(condition.placement_category, frozen_positions, initial_values, condition.frozen_count)),
        "initialFrozenPositions": compact_json(frozen_positions),
        "finalFrozenPositions": compact_json(result.final_frozen_positions),
        "initialFrozenCellIds": compact_json(frozen_cell_ids),
        "finalFrozenCellIds": compact_json(final_frozen_cell_ids),
        "initialFrozenCellIdPositions": compact_json(initial_id_positions),
        "finalFrozenCellIdPositions": compact_json(final_id_positions),
        "frozenCellDisplacements": compact_json(displacement_by_id),
        "frozenCellIdentityLogged": bool(len(frozen_cell_ids) == condition.frozen_count),
        "frozenCellIdPreserved": sorted(frozen_cell_ids) == sorted(final_frozen_cell_ids),
        "frozenPositionCountPreserved": len(result.initial_frozen_positions) == len(result.final_frozen_positions) == condition.frozen_count,
        "frozenCellsMoved": bool(any(value != 0 for value in signed_displacements)),
        "frozenMeanAbsoluteDisplacement": float(np.mean(abs_displacements)) if abs_displacements else 0.0,
        "frozenMaxAbsoluteDisplacement": int(max(abs_displacements)) if abs_displacements else 0,
        "frozenMeanSignedDisplacement": float(np.mean(signed_displacements)) if signed_displacements else 0.0,
        "initialValuesHash": state_hash(initial_values),
        "finalValuesHash": state_hash(result.final_values),
        "initialValues": compact_json(initial_values),
        "finalValues": compact_json(result.final_values),
        "initialAlgotypes": compact_json([condition.algorithm] * n),
        "finalAlgotypes": compact_json(result.final_algotypes),
        "initialAlgotypeCounts": compact_json({condition.algorithm: n}),
        "finalAlgotypeCounts": compact_json(Counter(result.final_algotypes)),
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
        "behaviorDecisionCount": int(sim.behavior_decision_count),
        "behaviorBlockedAttempts": int(sim.behavior_blocked_attempts),
        "behaviorAllowedFrozenTargetSwaps": int(sim.behavior_allowed_frozen_target_swaps),
        "behaviorTransitionCount": int(sum(1 for row in sim.behavior_event_log if row["eventType"] == "transition")),
        "behaviorEventCount": int(len(sim.behavior_event_log)),
        "reroutingEventProxy": int(result.blocked_move_attempts + result.frozen_swap_attempts),
        "wallTimeSeconds": float(result.wall_time_seconds),
        "valueCountPreserved": sorted(initial_values) == sorted(result.final_values),
        "algotypeCountPreserved": sorted([condition.algorithm] * n) == sorted(result.final_algotypes),
        "hasDuplicateValues": len(set(initial_values)) < len(initial_values),
        "uniqueValueCount": int(len(set(initial_values))),
        "maxValueMultiplicity": int(pd.Series(initial_values).value_counts().max()),
        "baselineEquivalencePassed": None if baseline_check is None else bool(baseline_check["passed"]),
    }
    record.update({key: compact_json(value) if isinstance(value, list) else value for key, value in value_meta.items()})
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
                "conditionTemplateId": record["conditionTemplateId"],
                "algorithm": condition.algorithm,
                "inputProfile": "random_unique",
                "replicateIndex": int(replicate_index),
                "defectBehaviorVariant": condition.behavior.behavior_id,
                "defectBehaviorFamily": condition.behavior.behavior_family,
                "baseFrozenVariantUsed": condition.behavior.base_frozen_variant,
                "frozenCount": int(condition.frozen_count),
                "placementCategory": condition.placement_category,
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
                "frozenPositions": row["frozen_positions_json"],
                "frozenCellIds": row.get("frozen_cell_ids_json", "[]"),
                "frozenCellIdPositions": row.get("frozen_cell_id_positions_json", "{}"),
                "defectBehaviorState": row.get("defect_behavior_state_json", "{}"),
            }
        )

    event_rows: list[dict[str, Any]] = []
    for event in sim.behavior_event_log:
        event_record = {
            "experimentId": EXPERIMENT_ID,
            "researchStepId": STEP_ID,
            "conditionId": condition_id,
            "conditionTemplateId": record["conditionTemplateId"],
            "algorithm": condition.algorithm,
            "placementCategory": condition.placement_category,
            "replicateIndex": int(replicate_index),
        }
        event_record.update(event)
        event_rows.append(event_record)
    return record, trace_rows, event_rows, baseline_check


def run_s12_matrix(
    *,
    n: int,
    replicates: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for condition in behavior_conditions():
        for replicate_index in range(replicates):
            record, traces, events, baseline_check = run_one_condition(
                condition=condition,
                n=n,
                replicate_index=replicate_index,
                max_activations=max_activations,
                max_swaps=max_swaps,
                max_comparisons=max_comparisons,
            )
            records.append(record)
            trace_rows.extend(traces)
            event_rows.extend(events)
            if baseline_check is not None:
                baseline_rows.append(baseline_check)
    return pd.DataFrame(records), pd.DataFrame(trace_rows), pd.DataFrame(event_rows), pd.DataFrame(baseline_rows)


def summarize_results(result_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        result_df.groupby(["algorithm", "placementCategory", "defectBehaviorVariant", "defectBehaviorFamily"], dropna=False)
        .agg(
            runCount=("conditionId", "size"),
            completedRate=("completed", "mean"),
            capStopRate=("capStop", "mean"),
            finalSortednessPercentMean=("finalSortednessPercent", "mean"),
            finalKendallDistanceMean=("finalKendallTauDistanceNormalized", "mean"),
            finalEditDistanceMean=("finalEditDistanceToTargetOrderNormalized", "mean"),
            swapCountMean=("swapCount", "mean"),
            activationCountMean=("activationCount", "mean"),
            blockedMoveAttemptsMean=("blockedMoveAttempts", "mean"),
            behaviorBlockedAttemptsMean=("behaviorBlockedAttempts", "mean"),
            behaviorAllowedFrozenTargetSwapsMean=("behaviorAllowedFrozenTargetSwaps", "mean"),
            behaviorTransitionCountMean=("behaviorTransitionCount", "mean"),
            delayedGratificationMean=("delayedGratification", "mean"),
            dgEventCountMean=("dgEventCount", "mean"),
        )
        .reset_index()
        .sort_values(["algorithm", "placementCategory", "defectBehaviorVariant"])
    )
    summary.insert(0, "researchStepId", STEP_ID)
    summary.insert(0, "experimentId", EXPERIMENT_ID)
    return summary


def load_context_tables(s11_path: Path, e01_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    s11 = pd.read_parquet(s11_path) if s11_path.exists() else pd.DataFrame()
    e01 = pd.read_parquet(e01_path) if e01_path.exists() else pd.DataFrame()
    if not s11.empty:
        s11 = s11[s11["placementCategory"].isin(PLACEMENT_CATEGORIES)].copy()
        s11.insert(0, "contextForResearchStepId", STEP_ID)
    if not e01.empty:
        e01 = e01[
            e01["algorithm"].isin(ALGORITHMS)
            & e01["frozenVariant"].isin(["passive", "stuck"])
            & e01["frozenCount"].eq(FROZEN_COUNT)
        ].copy()
        e01.insert(0, "contextForResearchStepId", STEP_ID)
    return s11, e01


def diagnostics_table(
    *,
    result_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    behavior_event_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    s11_context_df: pd.DataFrame,
    e01_context_df: pd.DataFrame,
    expected_rows: int,
    repo_tests_passed: bool,
) -> pd.DataFrame:
    dynamic = result_df[~result_df["defectBehaviorFamily"].eq("baseline")]
    required_transition_behaviors = {
        spec.behavior_id for spec in behavior_specs() if spec.requires_transition_logging
    }
    transition_events = behavior_event_df[behavior_event_df["eventType"].eq("transition")] if len(behavior_event_df) else pd.DataFrame()
    transition_covered = required_transition_behaviors.issubset(set(transition_events["behaviorVariant"])) if len(transition_events) else False
    decision_events = behavior_event_df[behavior_event_df["eventType"].eq("decision")] if len(behavior_event_df) else pd.DataFrame()
    matched_initial = result_df.groupby("replicateIndex")["initialValuesHash"].nunique().max() == 1
    checks = [
        ("expected_row_count", len(result_df) == expected_rows, f"Behavior-variant table has expected {expected_rows} rows."),
        ("all_behavior_variants_represented", set(result_df["defectBehaviorVariant"]) == required_behavior_ids(), "All required passive/stuck, probabilistic, time-varying, fatigue/recovery, and directional sticky variants are represented."),
        ("all_algorithms_represented", set(result_df["algorithm"]) == set(ALGORITHMS), "Bubble, Insertion, and Selection are represented."),
        ("all_selected_placements_represented", set(result_df["placementCategory"]) == set(PLACEMENT_CATEGORIES), "Selected S12 placement categories are represented."),
        ("matched_initial_arrays", bool(matched_initial), "Initial arrays are matched across all S12 conditions within each replicate."),
        ("placement_categories_valid", bool(result_df["placementCategoryValid"].all()), "Every row passes exact placement-category validation."),
        ("frozen_identities_logged", bool(result_df["frozenCellIdentityLogged"].all()), "Frozen cell identities are logged for every row."),
        ("frozen_identity_preserved", bool(result_df["frozenCellIdPreserved"].all()), "Frozen cell identities are preserved through each run."),
        ("frozen_count_preserved", bool(result_df["frozenPositionCountPreserved"].all()), "Frozen Cell counts are preserved through each run."),
        ("value_counts_preserved", bool(result_df["valueCountPreserved"].all()), "All runs preserve value counts."),
        ("algotype_counts_preserved", bool(result_df["algotypeCountPreserved"].all()), "All runs preserve Algotype counts."),
        ("baseline_equivalence_rows_written", len(baseline_df) == len(result_df[result_df["defectBehaviorFamily"].eq("baseline")]), "Passive/stuck baseline equivalence rows are written."),
        ("baseline_passive_stuck_unchanged", bool(len(baseline_df) > 0 and baseline_df["passed"].all()), "Passive and stuck baselines match the unmodified S01 simulator exactly."),
        ("behavior_events_logged", len(behavior_event_df) > 0, "Behavior event log is nonempty."),
        ("dynamic_decisions_logged", set(dynamic["defectBehaviorVariant"]).issubset(set(decision_events["behaviorVariant"])) if len(decision_events) else False, "Dynamic variants log frozen-target behavior decisions."),
        ("transition_variants_logged", bool(transition_covered), "Time-varying and fatigue/recovery variants log state transitions."),
        ("trace_behavior_snapshots_written", bool(len(trace_df) > 0 and trace_df["defectBehaviorState"].notna().all()), "Trace rows include defect behavior state snapshots."),
        ("stop_reasons_logged", bool(result_df["stopReason"].notna().all()), "Every run records a stop reason or cap."),
        ("summary_written", len(summary_df) > 0, "Behavior summary table is nonempty."),
        ("s11_context_loaded", len(s11_context_df) > 0, "S11 placement context table was loaded."),
        ("e01_context_loaded", len(e01_context_df) > 0, "E01 Frozen Cell context table was loaded."),
        ("repo_tests_passed", bool(repo_tests_passed), "Repository S12 unit tests passed."),
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
    artifacts_dir: Path,
    result_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    behavior_event_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    s11_context_df: pd.DataFrame,
    e01_context_df: pd.DataFrame,
) -> dict[str, Path]:
    result_dir = artifacts_dir / "results"
    trace_dir = artifacts_dir / "traces" / "e02" / STEP_ID
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    step_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "behavior_variants_parquet": result_dir / "e02_frozen_behavior_variants.parquet",
        "behavior_variants_csv": result_dir / "e02_frozen_behavior_variants.csv",
        "summary_parquet": result_dir / "e02_frozen_behavior_variant_summary.parquet",
        "summary_csv": result_dir / "e02_frozen_behavior_variant_summary.csv",
        "behavior_events_parquet": result_dir / "e02_frozen_behavior_variant_events.parquet",
        "behavior_events_csv": result_dir / "e02_frozen_behavior_variant_events.csv",
        "baseline_verification_parquet": result_dir / "e02_frozen_behavior_baseline_verification.parquet",
        "baseline_verification_csv": result_dir / "e02_frozen_behavior_baseline_verification.csv",
        "diagnostics_parquet": result_dir / "e02_frozen_behavior_variant_diagnostics.parquet",
        "diagnostics_csv": result_dir / "e02_frozen_behavior_variant_diagnostics.csv",
        "s11_context_parquet": result_dir / "e02_frozen_behavior_s11_context.parquet",
        "s11_context_csv": result_dir / "e02_frozen_behavior_s11_context.csv",
        "e01_context_parquet": result_dir / "e02_frozen_behavior_e01_context.parquet",
        "e01_context_csv": result_dir / "e02_frozen_behavior_e01_context.csv",
        "trace_parquet": trace_dir / "e02_frozen_behavior_trace_events.parquet",
        "trace_csv_gz": trace_dir / "e02_frozen_behavior_trace_events.csv.gz",
    }
    result_df.to_parquet(paths["behavior_variants_parquet"], index=False)
    result_df.to_csv(paths["behavior_variants_csv"], index=False)
    summary_df.to_parquet(paths["summary_parquet"], index=False)
    summary_df.to_csv(paths["summary_csv"], index=False)
    behavior_event_df.to_parquet(paths["behavior_events_parquet"], index=False)
    behavior_event_df.to_csv(paths["behavior_events_csv"], index=False)
    baseline_df.to_parquet(paths["baseline_verification_parquet"], index=False)
    baseline_df.to_csv(paths["baseline_verification_csv"], index=False)
    diagnostics_df.to_parquet(paths["diagnostics_parquet"], index=False)
    diagnostics_df.to_csv(paths["diagnostics_csv"], index=False)
    s11_context_df.to_parquet(paths["s11_context_parquet"], index=False)
    s11_context_df.to_csv(paths["s11_context_csv"], index=False)
    e01_context_df.to_parquet(paths["e01_context_parquet"], index=False)
    e01_context_df.to_csv(paths["e01_context_csv"], index=False)
    trace_df.to_parquet(paths["trace_parquet"], index=False)
    trace_df.to_csv(paths["trace_csv_gz"], index=False, compression="gzip")
    return paths


def plot_summary(summary_df: pd.DataFrame, figure_dir: Path) -> tuple[Path, Path, Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    behavior_order = [spec.behavior_id for spec in behavior_specs()]
    plot_df = summary_df.groupby(["defectBehaviorVariant", "defectBehaviorFamily"], dropna=False).agg(
        completedRate=("completedRate", "mean"),
        finalKendallDistance=("finalKendallDistanceMean", "mean"),
        blockedAttempts=("behaviorBlockedAttemptsMean", "mean"),
        delayedGratification=("delayedGratificationMean", "mean"),
    ).reset_index()
    plot_df = plot_df.set_index("defectBehaviorVariant").reindex(behavior_order).reset_index()
    labels = [value.replace("_", "\n") for value in plot_df["defectBehaviorVariant"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    axes[0].bar(range(len(plot_df)), plot_df["completedRate"], color="#4c78a8")
    axes[1].bar(range(len(plot_df)), plot_df["finalKendallDistance"], color="#f58518")
    axes[2].bar(range(len(plot_df)), plot_df["blockedAttempts"], color="#54a24b")
    for ax, ylabel, title in [
        (axes[0], "completion rate", "Completion"),
        (axes[1], "mean final Kendall distance", "Residual order distance"),
        (axes[2], "blocked attempts", "Behavior-blocked attempts"),
    ]:
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0, 1.05)
    fig.tight_layout()
    summary_png = figure_dir / "e02_frozen_behavior_variants_summary.png"
    summary_pdf = figure_dir / "e02_frozen_behavior_variants_summary.pdf"
    fig.savefig(summary_png, dpi=180)
    fig.savefig(summary_pdf)
    plt.close(fig)

    dg = summary_df.groupby(["defectBehaviorVariant", "algorithm"])["delayedGratificationMean"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    for algorithm in ALGORITHMS:
        subset = dg[dg["algorithm"].eq(algorithm)].set_index("defectBehaviorVariant").reindex(behavior_order)
        ax.plot(range(len(behavior_order)), subset["delayedGratificationMean"], marker="o", label=algorithm)
    ax.set_xticks(range(len(behavior_order)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mean DG")
    ax.set_title("E01-compatible DG proxy by behavior variant")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    dg_png = figure_dir / "e02_frozen_behavior_variants_dg.png"
    dg_pdf = figure_dir / "e02_frozen_behavior_variants_dg.pdf"
    fig.savefig(dg_png, dpi=180)
    fig.savefig(dg_pdf)
    plt.close(fig)
    return summary_png, summary_pdf, dg_png, dg_pdf


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.exists() and path.is_file():
            records.append({"path": str(path), "sha256": sha256_path(path), "sizeBytes": path.stat().st_size})
    return sorted(records, key=lambda row: row["path"])


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_dir = step_dir / "code"
    paths = [
        (REPO_ROOT / "scripts" / "e02_s12_frozen_behavior_variants.py", code_dir / "scripts" / "e02_s12_frozen_behavior_variants.py"),
        (REPO_ROOT / "tests" / "test_e02_frozen_behavior_variants.py", code_dir / "tests" / "test_e02_frozen_behavior_variants.py"),
        (REPO_ROOT / "scripts" / "e02_s11_frozen_placement.py", code_dir / "scripts" / "e02_s11_frozen_placement.py"),
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
    command = [sys.executable, "-m", "unittest", "tests.test_e02_frozen_behavior_variants"]
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
    baseline = summary_df[summary_df["defectBehaviorFamily"].eq("baseline")]
    dynamic = summary_df[~summary_df["defectBehaviorFamily"].eq("baseline")]
    if baseline.empty or dynamic.empty:
        return "null"
    baseline_completion = float(baseline["completedRate"].mean())
    dynamic_completion = float(dynamic["completedRate"].mean())
    dynamic_distance_max = float(dynamic["finalKendallDistanceMean"].max())
    if abs(dynamic_completion - baseline_completion) > 0.15 or dynamic_distance_max > 0.05:
        return "constraining/contradictory"
    return "supportive"


def write_validation_report(step_dir: Path, diagnostics_df: pd.DataFrame, validation: dict[str, Any]) -> Path:
    lines = [
        "# E02 S12 Validation Report",
        "",
        "- Research step ID: S12",
        "- Completion status: completed" if validation["success"] else "- Completion status: failed",
        "- Artifacts written: Frozen Cell behavior-variant table, summaries, behavior event/transition log, passive/stuck baseline equivalence table, S11/E01 context tables, trace table with behavior state snapshots, figures, copied code, manifest, status JSON, and run manifest.",
        f"- Validation result: {validation['validationResult']}",
        "- Caveats or blockers: S12 behavior variants are simulator extensions layered over S01 policy logic; biological realism is not claimed. Passive and stuck baselines are explicitly compared to the unmodified S01 simulator.",
        "- Recommended next action: stop before S13 for Chief Scientist review.",
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
    behavior_event_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    s11_context_df: pd.DataFrame,
    e01_context_df: pd.DataFrame,
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
        "S12 behavior variants are explicit simulator extensions for f=2 Frozen Cells on three selected S11 placements. "
        "They preserve S01 policy logic but add defect gates, so results audit robustness to defect dynamics rather than reproduce original paper variants. "
        "Passive/stuck baselines were verified unchanged against the unmodified S01 simulator."
    )
    recommended = "Stop before S13 for Chief Scientist review; if accepted, proceed to S13 stop-condition sensitivity."
    behavior_rows = (
        summary_df.groupby(["defectBehaviorVariant", "defectBehaviorFamily"], dropna=False)
        .agg(
            completedRate=("completedRate", "mean"),
            finalKendallDistance=("finalKendallDistanceMean", "mean"),
            blockedAttempts=("behaviorBlockedAttemptsMean", "mean"),
            transitions=("behaviorTransitionCountMean", "mean"),
            delayedGratification=("delayedGratificationMean", "mean"),
        )
        .reset_index()
        .sort_values("defectBehaviorVariant")
    )
    behavior_table = markdown_table(
        ["Behavior", "Family", "Completed", "Final Kendall", "Blocked", "Transitions", "DG"],
        [
            [
                row.defectBehaviorVariant,
                row.defectBehaviorFamily,
                row.completedRate,
                row.finalKendallDistance,
                row.blockedAttempts,
                row.transitions,
                row.delayedGratification,
            ]
            for row in behavior_rows.itertuples(index=False)
        ],
    )
    algorithm_rows = (
        summary_df.groupby("algorithm", dropna=False)
        .agg(
            completedRate=("completedRate", "mean"),
            finalKendallDistance=("finalKendallDistanceMean", "mean"),
            capStopRate=("capStopRate", "mean"),
        )
        .reset_index()
        .sort_values("algorithm")
    )
    algorithm_table = markdown_table(
        ["Algorithm", "Completed", "Final Kendall", "Cap stops"],
        [[row.algorithm, row.completedRate, row.finalKendallDistance, row.capStopRate] for row in algorithm_rows.itertuples(index=False)],
    )
    summary_path = step_dir / "summary.md"
    summary_lines = [
        "# E02 S12 Summary",
        "",
        "- Research step ID: S12",
        "- Completion status: completed" if validation["success"] else "- Completion status: failed",
        "- Artifacts written: `$ARTIFACTS_DIR/results/e02_frozen_behavior_variants.parquet`, behavior summaries, behavior event log, baseline verification table, S11/E01 context tables, trace table, figures, copied code, manifest, status JSON, and run manifest.",
        f"- Validation result: {validation['validationResult']}",
        f"- Caveats or blockers: {caveat}",
        "- Lay summary: S12 kept the sorting policies fixed and changed only how Frozen Cells respond when active cells try to swap with them. The run logs each dynamic defect decision and transition, and confirms the passive/stuck baselines still match the original S01 simulator.",
        f"- Recommended next action: {recommended}",
        f"- Outcome classification: {outcome}",
        "",
        "## Run Counts",
        "",
        markdown_table(
            ["Item", "Count"],
            [
                ["Behavior-variant runs", len(result_df)],
                ["Trace rows", len(trace_df)],
                ["Behavior event rows", len(behavior_event_df)],
                ["Baseline verification rows", len(baseline_df)],
                ["S11 context rows", len(s11_context_df)],
                ["E01 context rows", len(e01_context_df)],
                ["Diagnostics", len(diagnostics_df)],
            ],
        ),
        "",
        "## Behavior Summary",
        "",
        behavior_table,
        "",
        "## Algorithm Summary",
        "",
        algorithm_table,
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
        "behaviorRunRows": int(len(result_df)),
        "traceRows": int(len(trace_df)),
        "behaviorEventRows": int(len(behavior_event_df)),
        "baselineVerificationRows": int(len(baseline_df)),
        "summaryRows": int(len(summary_df)),
        "s11ContextRows": int(len(s11_context_df)),
        "e01ContextRows": int(len(e01_context_df)),
    }
    write_json(status_path, status_payload)

    all_artifacts = collect_artifacts(
        list(table_paths.values())
        + list(figure_paths)
        + code_paths
        + [validation_path, summary_path, status_path, step_dir / "repo_unit_test_log.txt"]
    )
    manifest_payload = {
        "schema": "eidosoma.research_step_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAt": utc_now(),
        "git": get_git_metadata(),
        "inputs": {
            "deterministicSimulator": "e02_deterministic_simulator",
            "s08DgHelper": "scripts/e02_s08_dg_nulls.py",
            "s09MetricHelper": "scripts/e02_s09_alternative_metrics.py",
            "s10InputGeneratorHelper": "scripts/e02_s10_input_distributions.py",
            "s11PlacementHelper": "scripts/e02_s11_frozen_placement.py",
            "s11PlacementContext": str(DEFAULT_S11_PLACEMENT_PATH),
            "e01FrozenContext": str(DEFAULT_E01_FROZEN_PATH),
        },
        "parameters": {
            "inputProfile": "random_unique",
            "n": int(result_df["n"].iloc[0]) if len(result_df) else None,
            "replicates": int(result_df["replicateIndex"].nunique()) if len(result_df) else 0,
            "algorithms": ALGORITHMS,
            "placementCategories": PLACEMENT_CATEGORIES,
            "frozenCount": FROZEN_COUNT,
            "behaviorVariants": [spec.behavior_id for spec in behavior_specs()],
        },
        "outputs": all_artifacts,
        "validation": validation,
        "repoTests": repo_tests,
        "behaviorRunRows": int(len(result_df)),
        "traceRows": int(len(trace_df)),
        "behaviorEventRows": int(len(behavior_event_df)),
        "baselineVerificationRows": int(len(baseline_df)),
        "summaryRows": int(len(summary_df)),
        "outcomeClassification": outcome,
    }
    write_json(artifact_manifest_path, manifest_payload)

    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "latestResearchStepId": STEP_ID,
        "generatedAt": utc_now(),
        "statusPath": str(status_path),
        "git": get_git_metadata(),
        "hardware": {"platform": platform.platform(), "python": sys.version, "cpuCount": os.cpu_count()},
        "packageVersions": {"numpy": np.__version__, "pandas": pd.__version__},
        "runs": [
            {
                "researchStepId": STEP_ID,
                "status": status_payload["status"],
                "runSeconds": status_payload["runSeconds"],
                "behaviorRunRows": int(len(result_df)),
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


def run_s12(args: argparse.Namespace) -> int:
    started_at = time.perf_counter()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    figure_dir = artifacts_dir / "figures" / "e02"
    step_dir.mkdir(parents=True, exist_ok=True)
    result_df, trace_df, behavior_event_df, baseline_df = run_s12_matrix(
        n=args.n,
        replicates=args.replicates,
        max_activations=args.max_activations,
        max_swaps=args.max_swaps,
        max_comparisons=args.max_comparisons,
    )
    summary_df = summarize_results(result_df)
    s11_context_df, e01_context_df = load_context_tables(args.s11_placement_context, args.e01_frozen_context)
    expected_rows = len(behavior_conditions()) * args.replicates
    diagnostics_df = diagnostics_table(
        result_df=result_df,
        trace_df=trace_df,
        behavior_event_df=behavior_event_df,
        baseline_df=baseline_df,
        summary_df=summary_df,
        s11_context_df=s11_context_df,
        e01_context_df=e01_context_df,
        expected_rows=expected_rows,
        repo_tests_passed=True,
    )
    table_paths = write_tables(
        artifacts_dir=artifacts_dir,
        result_df=result_df,
        trace_df=trace_df,
        behavior_event_df=behavior_event_df,
        baseline_df=baseline_df,
        summary_df=summary_df,
        diagnostics_df=diagnostics_df,
        s11_context_df=s11_context_df,
        e01_context_df=e01_context_df,
    )
    figure_paths = plot_summary(summary_df, figure_dir)
    code_paths = copy_code_artifacts(step_dir)
    repo_tests = run_repo_tests(step_dir)
    diagnostics_df = diagnostics_table(
        result_df=result_df,
        trace_df=trace_df,
        behavior_event_df=behavior_event_df,
        baseline_df=baseline_df,
        summary_df=summary_df,
        s11_context_df=s11_context_df,
        e01_context_df=e01_context_df,
        expected_rows=expected_rows,
        repo_tests_passed=repo_tests["passed"],
    )
    validation = validate_outputs(diagnostics_df)
    diagnostics_df.to_parquet(table_paths["diagnostics_parquet"], index=False)
    diagnostics_df.to_csv(table_paths["diagnostics_csv"], index=False)
    if not repo_tests["passed"]:
        validation["success"] = False
        validation["validationResult"] = "failed"
        validation.setdefault("failures", []).append("Repository S12 unit tests failed.")
    write_reports_and_manifests(
        artifacts_dir=artifacts_dir,
        started_at=started_at,
        result_df=result_df,
        trace_df=trace_df,
        behavior_event_df=behavior_event_df,
        baseline_df=baseline_df,
        summary_df=summary_df,
        diagnostics_df=diagnostics_df,
        s11_context_df=s11_context_df,
        e01_context_df=e01_context_df,
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
    parser.add_argument("--s11-placement-context", type=Path, default=DEFAULT_S11_PLACEMENT_PATH)
    parser.add_argument("--e01-frozen-context", type=Path, default=DEFAULT_E01_FROZEN_PATH)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--max-activations", type=int, default=300_000)
    parser.add_argument("--max-swaps", type=int, default=100_000)
    parser.add_argument("--max-comparisons", type=int, default=800_000)
    return parser.parse_args()


def main() -> int:
    return run_s12(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
