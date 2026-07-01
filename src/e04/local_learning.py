"""Local-only learning pilots for E04 S06."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.e02.deterministic_simulator import (
    EventTracingStatusProbe,
    SimulatorConfig,
    _active_positions,
    _build_cells,
    _choose_active_position,
    cell_labels,
    cell_state_signature,
    cell_values,
    frozen_positions,
    monotonicity_error_count,
    sortedness_percent,
    stable_json_sha256,
)
from src.e03.policy_interface import (
    PolicyAction,
    PolicyState,
    PolicyStepResult,
    cells_signature,
    observe_cell,
)
from src.e04.fatigue_damage import (
    RELIABILITY_MODES,
    FatigueDamageBenchmarkConfig,
    FatigueDamageController,
    delayed_gratification_from_sortedness,
)
from src.e04.homeostasis import (
    HOMEOSTATIC_TASKS,
    HomeostaticBenchmarkConfig,
    PerturbationSpec,
    _apply_perturbation,
    build_homeostatic_schedule,
)
from src.e04.memory_policies import BoundedCellMemoryBank, CellMemoryConfig
from src.e04.repairable_frozen import RepairBenchmarkConfig, RepairController, patch_repairable_swaps
from src.e04.signaling import SignalBank, SignalConfig


LEARNING_POLICY_MODES = ("nonlearning_control", "local_learning")
LEARNING_REWARD_FEATURES = (
    "local_order_delta",
    "swap_success_bonus",
    "blocked_penalty",
    "idle_with_local_disorder_penalty",
    "memory_frustration_delta",
)
LEARNING_FORBIDDEN_KEYS = (
    "sortedness",
    "global",
    "rank",
    "whole_array",
    "final",
    "target_morphology",
)


@dataclass(frozen=True)
class LocalLearningConfig:
    """Configuration for one S06 local-learning pilot run."""

    values: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    algorithm: str = "bubble"
    task_name: str = "swap_shocks"
    repair_rule: str = "nudge_repair"
    interface_mode: str = "full"
    reliability_mode: str = "no_fatigue_control"
    policy_mode: str = "local_learning"
    activation_seed: int = 0
    policy_seed: int | None = None
    schedule_seed: int = 0
    learning_seed: int = 0
    max_events: int = 120
    activation_distribution: str = "uniform_active"
    target_sortedness_percent: float = 100.0
    nudge_threshold: int = 2
    fatigue_threshold: int = 3
    recovery_events: int = 4
    damage_threshold: int = 4
    learning_rate: float = 0.25
    exploration_epsilon: float = 0.08
    order_bias: float = 0.4
    sort_direction: str = "increasing"

    def __post_init__(self) -> None:
        if self.algorithm not in {"bubble", "insertion", "selection"}:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        if self.task_name not in HOMEOSTATIC_TASKS:
            raise ValueError(f"Unsupported task_name: {self.task_name}")
        if self.repair_rule != "nudge_repair":
            raise ValueError("S06 pilot currently supports nudge_repair only")
        if self.interface_mode not in {"full", "no_memory", "no_signal"}:
            raise ValueError(f"Unsupported interface_mode: {self.interface_mode}")
        if self.reliability_mode not in RELIABILITY_MODES:
            raise ValueError(f"Unsupported reliability_mode: {self.reliability_mode}")
        if self.policy_mode not in LEARNING_POLICY_MODES:
            raise ValueError(f"Unsupported policy_mode: {self.policy_mode}")
        if self.max_events <= 0:
            raise ValueError("max_events must be positive")
        if not 0.0 <= self.exploration_epsilon <= 1.0:
            raise ValueError("exploration_epsilon must be in [0, 1]")

    @property
    def learning_enabled(self) -> bool:
        return self.policy_mode == "local_learning"

    @property
    def memory_enabled(self) -> bool:
        return self.interface_mode != "no_memory"

    @property
    def signal_enabled(self) -> bool:
        return self.interface_mode != "no_signal"

    def to_homeostatic_config(self) -> HomeostaticBenchmarkConfig:
        return HomeostaticBenchmarkConfig(
            values=self.values,
            algorithm=self.algorithm,
            task_name=self.task_name,
            repair_rule=self.repair_rule,
            interface_mode=self.interface_mode,
            reliability_mode=self.reliability_mode,
            activation_seed=self.activation_seed,
            policy_seed=self.policy_seed,
            schedule_seed=self.schedule_seed,
            max_events=self.max_events,
            activation_distribution=self.activation_distribution,
            target_sortedness_percent=self.target_sortedness_percent,
            nudge_threshold=self.nudge_threshold,
            fatigue_threshold=self.fatigue_threshold,
            recovery_events=self.recovery_events,
            damage_threshold=self.damage_threshold,
            sort_direction=self.sort_direction,
        )

    def to_repair_config(self) -> RepairBenchmarkConfig:
        return RepairBenchmarkConfig(
            values=self.values,
            algorithm=self.algorithm,
            frozen_indices=(),
            repair_rule=self.repair_rule,
            interface_mode=self.interface_mode,
            activation_seed=self.activation_seed,
            policy_seed=self.policy_seed,
            max_events=self.max_events,
            activation_distribution=self.activation_distribution,
            nudge_threshold=self.nudge_threshold,
            sort_direction=self.sort_direction,
        )

    def to_fatigue_config(self) -> FatigueDamageBenchmarkConfig:
        return FatigueDamageBenchmarkConfig(
            values=self.values,
            algorithm=self.algorithm,
            frozen_indices=(),
            repair_rule=self.repair_rule,
            interface_mode=self.interface_mode,
            reliability_mode=self.reliability_mode,
            activation_seed=self.activation_seed,
            policy_seed=self.policy_seed,
            max_events=self.max_events,
            activation_distribution=self.activation_distribution,
            nudge_threshold=self.nudge_threshold,
            fatigue_threshold=self.fatigue_threshold,
            recovery_events=self.recovery_events,
            damage_threshold=self.damage_threshold,
            sort_direction=self.sort_direction,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": list(self.values),
            "algorithm": self.algorithm,
            "task_name": self.task_name,
            "repair_rule": self.repair_rule,
            "interface_mode": self.interface_mode,
            "reliability_mode": self.reliability_mode,
            "policy_mode": self.policy_mode,
            "activation_seed": int(self.activation_seed),
            "policy_seed": int(self.policy_seed if self.policy_seed is not None else self.activation_seed),
            "schedule_seed": int(self.schedule_seed),
            "learning_seed": int(self.learning_seed),
            "max_events": int(self.max_events),
            "activation_distribution": self.activation_distribution,
            "target_sortedness_percent": float(self.target_sortedness_percent),
            "nudge_threshold": int(self.nudge_threshold),
            "fatigue_threshold": int(self.fatigue_threshold),
            "recovery_events": int(self.recovery_events),
            "damage_threshold": int(self.damage_threshold),
            "learning_rate": float(self.learning_rate),
            "exploration_epsilon": float(self.exploration_epsilon),
            "order_bias": float(self.order_bias),
            "sort_direction": self.sort_direction,
        }


@dataclass(frozen=True)
class LearningStepRecord:
    """One local learning update or nonlearning decision record."""

    event_step: int
    thread_id: int
    actor_position: int
    action: str
    target_index: int | None
    policy_mode: str
    reward: float
    weight_before: float
    weight_after: float
    updated: bool
    reward_terms: Mapping[str, float]
    accessed_indices: tuple[int, ...]
    memory_frustration_before: int
    memory_frustration_after: int
    sensed_blocked: float
    sensed_frustrated: float
    oracle_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_step": int(self.event_step),
            "thread_id": int(self.thread_id),
            "actor_position": int(self.actor_position),
            "action": self.action,
            "target_index": None if self.target_index is None else int(self.target_index),
            "policy_mode": self.policy_mode,
            "reward": float(self.reward),
            "weight_before": float(self.weight_before),
            "weight_after": float(self.weight_after),
            "updated": bool(self.updated),
            "reward_terms": {str(key): float(value) for key, value in sorted(self.reward_terms.items())},
            "accessed_indices": list(self.accessed_indices),
            "memory_frustration_before": int(self.memory_frustration_before),
            "memory_frustration_after": int(self.memory_frustration_after),
            "sensed_blocked": float(self.sensed_blocked),
            "sensed_frustrated": float(self.sensed_frustrated),
            "oracle_fields": list(self.oracle_fields),
        }


class LocalLearningController:
    """Per-cell local bandit over adjacent actions with local rewards only."""

    ACTIONS = ("swap_left", "swap_right", "idle")

    def __init__(self, config: LocalLearningConfig) -> None:
        self.config = config
        self.rng = random.Random(config.learning_seed)
        self.weights: dict[int, dict[str, float]] = defaultdict(lambda: {action: 0.0 for action in self.ACTIONS})
        self.update_log: list[LearningStepRecord] = []
        self.decision_count = 0
        self.update_count = 0
        self.action_counts: Counter[str] = Counter()

    def _target_for_action(self, actor_index: int, action: str) -> int | None:
        if action == "swap_left":
            return actor_index - 1
        if action == "swap_right":
            return actor_index + 1
        return None

    def legal_actions(self, observation: Any) -> tuple[str, ...]:
        actions = ["idle"]
        if observation.actor_status not in {"ACTIVE", "CellStatus.ACTIVE"}:
            return tuple(actions)
        if observation.actor_index > observation.left_boundary:
            left_status = observation.statuses[observation.actor_index - 1]
            if left_status in {"ACTIVE", "FREEZE", "CellStatus.ACTIVE", "CellStatus.FREEZE"}:
                actions.append("swap_left")
        if observation.actor_index < observation.right_boundary:
            right_status = observation.statuses[observation.actor_index + 1]
            if right_status in {"ACTIVE", "FREEZE", "CellStatus.ACTIVE", "CellStatus.FREEZE"}:
                actions.append("swap_right")
        return tuple(actions)

    def _pair_disordered(self, left_value: int, right_value: int, reverse: bool) -> bool:
        return left_value < right_value if reverse else left_value > right_value

    def _local_disorder_count(self, values: Sequence[int], indices: Sequence[int], reverse: bool) -> int:
        if not values:
            return 0
        selected = sorted(set(int(idx) for idx in indices if 0 <= int(idx) < len(values)))
        pairs = {
            (idx, idx + 1)
            for idx in selected
            if idx + 1 < len(values) and (idx + 1 in selected or idx in selected)
        }
        return sum(1 for left, right in pairs if self._pair_disordered(int(values[left]), int(values[right]), reverse))

    def _accessed_indices(self, actor_index: int, target_index: int | None, length: int) -> tuple[int, ...]:
        indices = {actor_index - 1, actor_index, actor_index + 1}
        if target_index is not None:
            indices.update({target_index - 1, target_index, target_index + 1})
        return tuple(sorted(idx for idx in indices if 0 <= idx < length))

    def _base_action_score(self, observation: Any, memory_state: Any, action: str, blocked_signal: float, frustrated_signal: float) -> float:
        target = self._target_for_action(observation.actor_index, action)
        score = self.weights[int(observation.actor_thread_id)][action]
        if target is not None and 0 <= target < len(observation.values):
            left = min(int(observation.actor_index), int(target))
            right = max(int(observation.actor_index), int(target))
            if self._pair_disordered(int(observation.values[left]), int(observation.values[right]), bool(observation.reverse_direction)):
                score += float(self.config.order_bias)
            if observation.statuses[target] in {"FREEZE", "CellStatus.FREEZE"}:
                score += 0.15 + 0.1 * min(1.0, float(memory_state.local_frustration) / 5.0)
        if action == "idle":
            score -= 0.05 + 0.05 * min(1.0, float(memory_state.local_frustration) / 5.0)
        if action != "idle":
            score += 0.05 * float(blocked_signal + frustrated_signal)
        return float(score)

    def select_action(self, observation: Any, memory_state: Any, sensation: Any | None) -> tuple[str, int | None, dict[str, Any]]:
        legal = self.legal_actions(observation)
        blocked_signal = 0.0
        frustrated_signal = 0.0
        if sensation is not None:
            blocked_signal = float(sensation.diffusive_fields.get("blocked", 0.0))
            frustrated_signal = float(sensation.diffusive_fields.get("frustrated", 0.0))
        if self.config.learning_enabled and self.rng.random() < self.config.exploration_epsilon:
            action = self.rng.choice(tuple(legal))
            selection_mode = "epsilon"
        else:
            scored = [
                (self._base_action_score(observation, memory_state, action, blocked_signal, frustrated_signal), action)
                for action in legal
            ]
            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            action = scored[0][1]
            selection_mode = "greedy_local"
        target = self._target_for_action(observation.actor_index, action)
        if target is not None and not (observation.left_boundary <= target <= observation.right_boundary):
            target = None
            action = "idle"
        return (
            action,
            target,
            {
                "selection_mode": selection_mode,
                "blocked_signal": blocked_signal,
                "frustrated_signal": frustrated_signal,
                "legal_actions": list(legal),
            },
        )

    def record_reward(
        self,
        *,
        event_step: int,
        observation: Any,
        action: str,
        target_index: int | None,
        values_after: Sequence[int],
        memory_before: Any,
        memory_after: Any,
        swap_delta: int,
        frozen_attempt_delta: int,
        selection_metadata: Mapping[str, Any],
    ) -> LearningStepRecord:
        thread_id = int(observation.actor_thread_id)
        accessed = self._accessed_indices(observation.actor_index, target_index, len(observation.values))
        before_disorder = self._local_disorder_count(observation.values, accessed, observation.reverse_direction)
        after_disorder = self._local_disorder_count(values_after, accessed, observation.reverse_direction)
        local_order_delta = float(before_disorder - after_disorder)
        blocked = frozen_attempt_delta > 0 or (action != "idle" and swap_delta == 0)
        frustration_delta = int(memory_after.local_frustration) - int(memory_before.local_frustration)
        reward_terms = {
            "local_order_delta": local_order_delta,
            "swap_success_bonus": 0.25 if swap_delta > 0 else 0.0,
            "blocked_penalty": -0.35 if blocked else 0.0,
            "idle_with_local_disorder_penalty": -0.08 if action == "idle" and before_disorder > 0 else 0.0,
            "memory_frustration_delta": -0.05 * max(0, frustration_delta) + 0.03 * max(0, -frustration_delta),
        }
        reward = float(sum(reward_terms.values()))
        weight_before = float(self.weights[thread_id][action])
        updated = bool(self.config.learning_enabled)
        weight_after = weight_before
        if updated:
            weight_after = max(-5.0, min(5.0, weight_before + float(self.config.learning_rate) * reward))
            self.weights[thread_id][action] = weight_after
            self.update_count += 1
        self.decision_count += 1
        self.action_counts[action] += 1
        record = LearningStepRecord(
            event_step=event_step,
            thread_id=thread_id,
            actor_position=int(observation.actor_index),
            action=action,
            target_index=target_index,
            policy_mode=self.config.policy_mode,
            reward=reward,
            weight_before=weight_before,
            weight_after=weight_after,
            updated=updated,
            reward_terms=reward_terms,
            accessed_indices=accessed,
            memory_frustration_before=int(memory_before.local_frustration),
            memory_frustration_after=int(memory_after.local_frustration),
            sensed_blocked=float(selection_metadata.get("blocked_signal", 0.0)),
            sensed_frustrated=float(selection_metadata.get("frustrated_signal", 0.0)),
        )
        self.update_log.append(record)
        return record

    def audit_no_global_oracle(self) -> dict[str, Any]:
        forbidden_hits = [
            key
            for key in LEARNING_REWARD_FEATURES
            for token in LEARNING_FORBIDDEN_KEYS
            if token in key.lower()
        ]
        logged_oracles = [
            item.to_dict()
            for item in self.update_log
            if item.oracle_fields
        ]
        return {
            "usesGlobalOracle": bool(forbidden_hits or logged_oracles),
            "forbiddenRewardFeatureHits": forbidden_hits,
            "loggedOracleFieldCount": len(logged_oracles),
            "rewardFeatures": list(LEARNING_REWARD_FEATURES),
            "allowedObservationScope": "actor, immediate neighbors, S01 memory, S02 local/diffusive signals, local action result",
            "offlineOnlyMetricsExcluded": ["sortedness_percent", "full_array_rank", "final_target"],
        }

    def stable_digest(self) -> str:
        payload = {
            "weights": {
                str(thread_id): {action: round(float(value), 12) for action, value in sorted(weights.items())}
                for thread_id, weights in sorted(self.weights.items())
            },
            "decision_count": int(self.decision_count),
            "update_count": int(self.update_count),
            "action_counts": dict(sorted(self.action_counts.items())),
        }
        return stable_json_sha256(payload)

    def summary(self) -> dict[str, Any]:
        rewards = [item.reward for item in self.update_log]
        return {
            "policy_mode": self.config.policy_mode,
            "learning_enabled": bool(self.config.learning_enabled),
            "decision_count": int(self.decision_count),
            "update_count": int(self.update_count),
            "positive_reward_count": int(sum(1 for value in rewards if value > 0)),
            "negative_reward_count": int(sum(1 for value in rewards if value < 0)),
            "mean_local_reward": float(sum(rewards) / len(rewards)) if rewards else 0.0,
            "action_counts": dict(sorted(self.action_counts.items())),
            "weight_digest": self.stable_digest(),
            "update_log": [item.to_dict() for item in self.update_log[:128]],
            "update_log_truncated": len(self.update_log) > 128,
        }


@dataclass(frozen=True)
class LocalLearningPilotResult:
    """Summary of one S06 pilot run."""

    config: LocalLearningConfig
    schedule: tuple[PerturbationSpec, ...]
    final_values: tuple[int, ...]
    final_labels: tuple[str, ...]
    final_frozen_positions: tuple[int, ...]
    event_count: int
    stop_reason: str
    swap_count: int
    comparison_count: int
    frozen_attempt_count: int
    perturbation_log: tuple[dict[str, Any], ...]
    fatigue_transition_log: tuple[dict[str, Any], ...]
    impairment_log: tuple[dict[str, Any], ...]
    repair_summary: Mapping[str, Any]
    reliability_summary: Mapping[str, Any]
    signal_audit: Mapping[str, Any]
    learning_audit: Mapping[str, Any]
    learning_summary: Mapping[str, Any]
    event_log: tuple[dict[str, Any], ...]
    memory_digest: str
    signal_digest: str
    learning_digest: str

    def _offline_metrics(self) -> dict[str, Any]:
        trajectory = [{"event_step": 0, "values": list(self.config.values)}]
        trajectory.extend(
            {"event_step": int(event["event_step"]), "values": list(event["current_values"])}
            for event in self.event_log
            if "current_values" in event
        )
        sortedness_values = [
            sortedness_percent(item["values"], self.config.sort_direction)
            for item in trajectory
        ]
        target_flags = [value >= self.config.target_sortedness_percent for value in sortedness_values]
        out_runs: list[int] = []
        current_run = 0
        for flag in target_flags:
            if flag:
                if current_run:
                    out_runs.append(current_run)
                current_run = 0
            else:
                current_run += 1
        if current_run:
            out_runs.append(current_run)
        recovery_times: list[int] = []
        unrecovered = 0
        for perturbation in self.perturbation_log:
            start_step = int(perturbation["event_step"])
            recovered_at: int | None = None
            for item, flag in zip(trajectory, target_flags, strict=True):
                if int(item["event_step"]) >= start_step and flag:
                    recovered_at = int(item["event_step"])
                    break
            if recovered_at is None:
                unrecovered += 1
            else:
                recovery_times.append(max(0, recovered_at - start_step))
        dg_metrics = delayed_gratification_from_sortedness(sortedness_values)
        damaged_transitions = sum(1 for item in self.fatigue_transition_log if item.get("to_state") == "damaged")
        return {
            "initial_sortedness_percent": float(sortedness_values[0]) if sortedness_values else 100.0,
            "final_sortedness_percent": float(sortedness_values[-1]) if sortedness_values else 100.0,
            "min_sortedness_percent": float(min(sortedness_values)) if sortedness_values else 100.0,
            "mean_sortedness_percent": float(sum(sortedness_values) / len(sortedness_values)) if sortedness_values else 100.0,
            "final_monotonicity_error_count": int(monotonicity_error_count(self.final_values, self.config.sort_direction)),
            "time_in_target_event_count": int(sum(1 for flag in target_flags if flag)),
            "time_in_target_fraction": float(sum(1 for flag in target_flags if flag) / len(target_flags)) if target_flags else 1.0,
            "longest_out_of_target_run": int(max(out_runs) if out_runs else 0),
            "recovered_perturbation_count": int(len(recovery_times)),
            "unrecovered_perturbation_count": int(unrecovered),
            "mean_recovery_events": float(sum(recovery_times) / len(recovery_times)) if recovery_times else None,
            "max_recovery_events": int(max(recovery_times)) if recovery_times else None,
            "cumulative_damage_transition_count": int(damaged_transitions),
            "cumulative_impairment_count": int(len(self.impairment_log)),
            **{key: value for key, value in dg_metrics.items() if key.startswith("dg_")},
        }

    def to_row(self) -> dict[str, Any]:
        metrics = self._offline_metrics()
        perturbation_counts = {
            kind: sum(1 for item in self.perturbation_log if item["perturbation_type"] == kind)
            for kind in ("swap", "delete_insert", "freeze")
        }
        learning_summary = dict(self.learning_summary)
        return {
            **self.config.to_dict(),
            "event_count": int(self.event_count),
            "stop_reason": self.stop_reason,
            "swap_count": int(self.swap_count),
            "comparison_count": int(self.comparison_count),
            "frozen_attempt_count": int(self.frozen_attempt_count),
            "energy_total": int(self.swap_count),
            "perturbation_count": int(len(self.perturbation_log)),
            "swap_perturbation_count": int(perturbation_counts["swap"]),
            "delete_insert_perturbation_count": int(perturbation_counts["delete_insert"]),
            "freeze_perturbation_count": int(perturbation_counts["freeze"]),
            **metrics,
            "learning_decision_count": int(learning_summary.get("decision_count", 0)),
            "learning_update_count": int(learning_summary.get("update_count", 0)),
            "positive_reward_count": int(learning_summary.get("positive_reward_count", 0)),
            "negative_reward_count": int(learning_summary.get("negative_reward_count", 0)),
            "mean_local_reward": float(learning_summary.get("mean_local_reward", 0.0)),
            "final_values_json": json.dumps(list(self.final_values), separators=(",", ":")),
            "final_labels_json": json.dumps(list(self.final_labels), separators=(",", ":")),
            "final_frozen_positions_json": json.dumps(list(self.final_frozen_positions), separators=(",", ":")),
            "schedule_json": json.dumps([item.to_dict() for item in self.schedule], sort_keys=True, separators=(",", ":"), default=str),
            "schedule_sha256": stable_json_sha256([item.to_dict() for item in self.schedule]),
            "perturbation_log_json": json.dumps(list(self.perturbation_log), sort_keys=True, separators=(",", ":"), default=str),
            "fatigue_transition_count": len(self.fatigue_transition_log),
            "fatigue_transition_log_json": json.dumps(list(self.fatigue_transition_log), sort_keys=True, separators=(",", ":"), default=str),
            "impairment_count": len(self.impairment_log),
            "impairment_log_json": json.dumps(list(self.impairment_log), sort_keys=True, separators=(",", ":"), default=str),
            "unfreeze_count": int(self.repair_summary.get("unfreeze_count", 0)),
            "repair_summary_json": json.dumps(dict(self.repair_summary), sort_keys=True, separators=(",", ":"), default=str),
            "reliability_summary_json": json.dumps(dict(self.reliability_summary), sort_keys=True, separators=(",", ":"), default=str),
            "signal_audit_json": json.dumps(dict(self.signal_audit), sort_keys=True, separators=(",", ":"), default=str),
            "learning_audit_json": json.dumps(dict(self.learning_audit), sort_keys=True, separators=(",", ":"), default=str),
            "learning_summary_json": json.dumps(dict(self.learning_summary), sort_keys=True, separators=(",", ":"), default=str),
            "uses_global_oracle": bool(
                self.signal_audit.get("usesGlobalOracle", False)
                or self.learning_audit.get("usesGlobalOracle", False)
            ),
            "event_log_sha256": stable_json_sha256(list(self.event_log)),
            "memory_digest": self.memory_digest,
            "signal_digest": self.signal_digest,
            "learning_digest": self.learning_digest,
        }


def _execute_local_action(
    actor: Any,
    action: str,
    target_index: int | None,
    probe: EventTracingStatusProbe,
) -> tuple[int, int]:
    before_swaps = int(probe.swap_count)
    before_frozen = int(probe.frozen_swap_attempts)
    if action != "idle" and target_index is not None:
        probe.record_compare_and_swap()
        actor.swap((int(target_index), int(actor.current_position[1])))
    return int(probe.swap_count) - before_swaps, int(probe.frozen_swap_attempts) - before_frozen


def simulate_local_learning_pilot(config: LocalLearningConfig) -> LocalLearningPilotResult:
    """Run one S06 local-learning pilot on an S05 homeostatic task."""

    base_config = SimulatorConfig(
        values=config.values,
        algorithm=config.algorithm,
        frozen_indices=(),
        frozen_semantics="passive",
        activation_seed=config.activation_seed,
        policy_seed=config.policy_seed,
        activation_distribution=config.activation_distribution,
        max_events=config.max_events,
        stop_when_sorted=False,
        convergence_criterion="none",
        sort_direction=config.sort_direction,
    )
    probe = EventTracingStatusProbe()
    cells, cell_status = _build_cells(base_config, probe)
    memory_bank = BoundedCellMemoryBank(CellMemoryConfig(enabled=config.memory_enabled))
    signal_bank = SignalBank(
        len(cells),
        SignalConfig(
            enabled=config.signal_enabled,
            local_radius=1,
            diffusion_enabled=config.signal_enabled,
            noise_seed=config.policy_seed or config.activation_seed,
        ),
    )
    repair_controller = RepairController(cells, cell_status, config.to_repair_config(), signal_bank)
    patch_repairable_swaps(cells, repair_controller)
    reliability = FatigueDamageController(config.to_fatigue_config())
    reliability.initialize(cells)
    learner = LocalLearningController(config)

    schedule = build_homeostatic_schedule(config.to_homeostatic_config())
    schedule_by_step: dict[int, list[PerturbationSpec]] = {}
    for spec in schedule:
        schedule_by_step.setdefault(int(spec.event_step), []).append(spec)

    activation_rng = random.Random(config.activation_seed)
    random.seed(config.policy_seed if config.policy_seed is not None else config.activation_seed)
    cursor = 0
    event_log: list[dict[str, Any]] = []
    perturbation_log: list[dict[str, Any]] = []
    next_thread_id = max(int(cell.threadID) for cell in cells) + 1
    stop_reason = "fixed_horizon_complete"

    for event_step in range(1, config.max_events + 1):
        for spec in schedule_by_step.get(event_step, []):
            perturbation_record, next_thread_id = _apply_perturbation(
                spec=spec,
                cells=cells,
                cell_status=cell_status,
                repair_controller=repair_controller,
                reliability=reliability,
                next_thread_id=next_thread_id,
            )
            perturbation_log.append(perturbation_record)
        repair_controller.before_event(event_step)
        reliability.before_event(event_step, cells)
        active_positions = _active_positions(cells, cell_status)
        if not active_positions:
            stop_reason = "no_active_cells"
            break
        position, cursor = _choose_active_position(
            active_positions,
            config.activation_distribution,
            activation_rng,
            cursor,
        )
        actor = cells[position]
        before_signature_public = cells_signature(cells)
        before_signature = cell_state_signature(cells)
        before_repair_transitions = len(repair_controller.unfreeze_log)
        before_fatigue_transitions = len(reliability.transition_log)
        probe.current_event_step = event_step
        applied_action = "impaired_skip"
        target_index: int | None = None
        reward = 0.0
        learning_updated = False
        local_accessed_indices: tuple[int, ...] = ()
        if reliability.can_act(actor):
            observation = observe_cell(actor)
            memory_before = memory_bank.get(observation.actor_thread_id)
            sensation = signal_bank.sense(observation, event_step=event_step) if config.signal_enabled else None
            action, target_index, selection_metadata = learner.select_action(observation, memory_before, sensation)
            swap_delta, frozen_attempt_delta = _execute_local_action(actor, action, target_index, probe)
            after_signature_public = cells_signature(cells)
            applied_action = "swap" if swap_delta > 0 else "blocked_swap_attempt" if frozen_attempt_delta > 0 else action
            result = PolicyStepResult(
                behavior=observation.behavior,
                observation=observation,
                state_before=PolicyState(
                    ideal_position=observation.ideal_position,
                    reverse_direction=observation.reverse_direction,
                    memory={"s06_local_learning": "local_only"},
                ),
                proposed_action=PolicyAction(action_type="swap" if target_index is not None else "idle", target_index=target_index),
                applied_action=PolicyAction(action_type=applied_action, target_index=target_index),
                signature_before=before_signature_public,
                signature_after=after_signature_public,
                comparison_delta=1 if action != "idle" and target_index is not None else 0,
                swap_delta=swap_delta,
                frozen_attempt_delta=frozen_attempt_delta,
                state_changed=after_signature_public != before_signature_public,
            )
            memory_after = memory_bank.update_from_step(result, event_step=event_step)
            blocked = applied_action == "blocked_swap_attempt" or frozen_attempt_delta > 0
            if config.signal_enabled:
                signal_bank.emit(observation, memory_after, blocked=blocked, event_step=event_step)
            values_after = cell_values(cells)
            record = learner.record_reward(
                event_step=event_step,
                observation=observation,
                action=action,
                target_index=target_index,
                values_after=values_after,
                memory_before=memory_before,
                memory_after=memory_after,
                swap_delta=swap_delta,
                frozen_attempt_delta=frozen_attempt_delta,
                selection_metadata=selection_metadata,
            )
            reward = record.reward
            learning_updated = record.updated
            local_accessed_indices = record.accessed_indices
            reliability.after_event(actor, swap_delta=swap_delta)
        else:
            reliability.record_impairment(actor, event_step)
            swap_delta = 0
        after_signature = cell_state_signature(cells)
        event_log.append(
            {
                "event_step": int(event_step),
                "activated_position": int(position),
                "activated_thread_id": int(actor.threadID),
                "activated_value_before": int(before_signature[position][1]),
                "applied_action": str(applied_action),
                "target_index": target_index,
                "swap_delta": int(swap_delta),
                "public_state_changed": bool(after_signature != before_signature),
                "unfreeze_delta": len(repair_controller.unfreeze_log) - before_repair_transitions,
                "fatigue_transition_delta": len(reliability.transition_log) - before_fatigue_transitions,
                "perturbation_count_at_event": len(schedule_by_step.get(event_step, [])),
                "learning_reward": float(reward),
                "learning_updated": bool(learning_updated),
                "local_accessed_indices": list(local_accessed_indices),
                "reliability_state_after": reliability.state_for(actor),
                "current_values": list(cell_values(cells)),
                "current_labels": list(cell_labels(cells)),
                "current_frozen_positions": list(frozen_positions(cells, cell_status)),
            }
        )

    return LocalLearningPilotResult(
        config=config,
        schedule=schedule,
        final_values=tuple(cell_values(cells)),
        final_labels=cell_labels(cells),
        final_frozen_positions=frozen_positions(cells, cell_status),
        event_count=len(event_log),
        stop_reason=stop_reason,
        swap_count=int(probe.swap_count),
        comparison_count=int(probe.compare_and_swap_count),
        frozen_attempt_count=int(probe.frozen_swap_attempts),
        perturbation_log=tuple(perturbation_log),
        fatigue_transition_log=tuple(item.to_dict() for item in reliability.transition_log),
        impairment_log=tuple(item.to_dict() for item in reliability.impairment_log),
        repair_summary=repair_controller.summary(),
        reliability_summary=reliability.summary(),
        signal_audit=signal_bank.audit_no_global_oracle(),
        learning_audit=learner.audit_no_global_oracle(),
        learning_summary=learner.summary(),
        event_log=tuple(event_log),
        memory_digest=memory_bank.stable_digest(),
        signal_digest=signal_bank.stable_digest(),
        learning_digest=learner.stable_digest(),
    )


def default_local_learning_configs(
    *,
    task_names: Sequence[str] = HOMEOSTATIC_TASKS,
    interface_modes: Sequence[str] = ("full", "no_memory", "no_signal"),
    reliability_modes: Sequence[str] = ("no_fatigue_control", "fatigue_recovery", "cumulative_damage"),
    seeds: Sequence[int] = (10_601, 10_602),
    max_events: int = 120,
) -> list[LocalLearningConfig]:
    """Return compact S06 pilot configs with matched learning/nonlearning controls."""

    configs: list[LocalLearningConfig] = []
    for seed_idx, seed in enumerate(seeds):
        for task_name in task_names:
            for interface_mode in interface_modes:
                for reliability_mode in reliability_modes:
                    for policy_mode in LEARNING_POLICY_MODES:
                        configs.append(
                            LocalLearningConfig(
                                algorithm="bubble",
                                task_name=task_name,
                                repair_rule="nudge_repair",
                                interface_mode=interface_mode,
                                reliability_mode=reliability_mode,
                                policy_mode=policy_mode,
                                activation_seed=int(seed + 100 * seed_idx),
                                policy_seed=int(seed + 10_000 + 100 * seed_idx),
                                schedule_seed=int(seed + 20_000 + 100 * seed_idx),
                                learning_seed=int(seed + 30_000 + 100 * seed_idx),
                                max_events=max_events,
                            )
                        )
    return configs


def run_local_learning_matrix(configs: Sequence[LocalLearningConfig]) -> list[LocalLearningPilotResult]:
    return [simulate_local_learning_pilot(config) for config in configs]
