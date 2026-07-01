"""Local-only evolutionary search support for E04 S08.

The search policies in this module consume only the S07
``LocalTrainingObservation`` projection.  Rich E03 observations are used by the
simulator only to build that projection and to update the existing S01/S02
memory and signal side channels after an action has already been chosen.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

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
from src.e03.gpu_batch_simulator import jax_backend_summary
from src.e03.policy_interface import PolicyAction, PolicyState, PolicyStepResult, cells_signature, observe_cell
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
    simulate_homeostatic_benchmark,
)
from src.e04.memory_policies import BoundedCellMemoryBank, CellMemoryConfig
from src.e04.no_oracle_protocol import (
    ALLOWED_TRAINING_SIGNAL_FIELDS,
    EXCLUDED_TRAINING_SIGNAL_FIELDS,
    LOCAL_ONLY_PROTOCOL_ID,
    LocalTrainingObservation,
    audit_training_feature_dict,
    project_local_training_observation,
)
from src.e04.repairable_frozen import RepairBenchmarkConfig, RepairController, patch_repairable_swaps
from src.e04.signaling import SignalBank, SignalConfig


S08_PROTOCOL_ID = f"{LOCAL_ONLY_PROTOCOL_ID}:e04_s08_evolution"
S08_POLICY_FAMILY = "s08_local_memory_signal_adjacent_policy"
ACTION_NAMES = ("idle", "swap_left", "swap_right")
PARAMETER_NAMES = (
    "idle_bias",
    "swap_left_bias",
    "swap_right_bias",
    "disorder_weight",
    "frustration_weight",
    "blocked_signal_weight",
    "frustrated_signal_weight",
    "failed_swap_weight",
    "last_failure_weight",
    "time_since_weight",
    "neighbor_count_weight",
    "illegal_penalty",
    "idle_order_bonus",
)


@dataclass(frozen=True)
class S08EvaluationConfig:
    """One train, validation, or held-out evaluation condition."""

    split: str
    values: tuple[int, ...]
    task_name: str
    reliability_mode: str
    activation_seed: int
    policy_seed: int
    schedule_seed: int
    max_events: int = 120
    algorithm: str = "bubble"
    repair_rule: str = "nudge_repair"
    interface_mode: str = "full"
    activation_distribution: str = "uniform_active"
    target_sortedness_percent: float = 100.0
    nudge_threshold: int = 2
    fatigue_threshold: int = 3
    recovery_events: int = 4
    damage_threshold: int = 4
    sort_direction: str = "increasing"

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "heldout"}:
            raise ValueError(f"Unsupported split: {self.split}")
        if self.task_name not in HOMEOSTATIC_TASKS:
            raise ValueError(f"Unsupported task_name: {self.task_name}")
        if self.reliability_mode not in RELIABILITY_MODES:
            raise ValueError(f"Unsupported reliability_mode: {self.reliability_mode}")
        if len(self.values) < 3:
            raise ValueError("values must include at least three cells")
        if self.max_events <= 0:
            raise ValueError("max_events must be positive")

    @property
    def memory_enabled(self) -> bool:
        return self.interface_mode != "no_memory"

    @property
    def signal_enabled(self) -> bool:
        return self.interface_mode != "no_signal"

    def to_homeostatic_config(self, *, algorithm: str | None = None) -> HomeostaticBenchmarkConfig:
        return HomeostaticBenchmarkConfig(
            values=self.values,
            algorithm=algorithm or self.algorithm,
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

    def split_key(self) -> tuple[Any, ...]:
        return (
            self.split,
            self.values,
            self.task_name,
            self.reliability_mode,
            self.activation_seed,
            self.policy_seed,
            self.schedule_seed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "values": list(self.values),
            "array_size": len(self.values),
            "algorithm": self.algorithm,
            "task_name": self.task_name,
            "repair_rule": self.repair_rule,
            "interface_mode": self.interface_mode,
            "reliability_mode": self.reliability_mode,
            "activation_seed": int(self.activation_seed),
            "policy_seed": int(self.policy_seed),
            "schedule_seed": int(self.schedule_seed),
            "max_events": int(self.max_events),
            "activation_distribution": self.activation_distribution,
            "target_sortedness_percent": float(self.target_sortedness_percent),
            "nudge_threshold": int(self.nudge_threshold),
            "fatigue_threshold": int(self.fatigue_threshold),
            "recovery_events": int(self.recovery_events),
            "damage_threshold": int(self.damage_threshold),
            "sort_direction": self.sort_direction,
        }


@dataclass(frozen=True)
class EvolutionGenome:
    """One candidate local-only policy genome."""

    policy_id: str
    generation: int
    population_index: int
    parent_ids: tuple[str, ...]
    mutation_seed: int
    mutation_scale: float
    parameters: Mapping[str, float]
    lineage_note: str = ""

    def vector(self) -> np.ndarray:
        return np.asarray([float(self.parameters[name]) for name in PARAMETER_NAMES], dtype=np.float32)

    def to_policy_record(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_family": S08_POLICY_FAMILY,
            "protocol_id": S08_PROTOCOL_ID,
            "generation": int(self.generation),
            "population_index": int(self.population_index),
            "parent_ids": list(self.parent_ids),
            "mutation_seed": int(self.mutation_seed),
            "mutation_scale": float(self.mutation_scale),
            "parameters": {name: float(self.parameters[name]) for name in PARAMETER_NAMES},
            "allowed_training_signal_fields": list(ALLOWED_TRAINING_SIGNAL_FIELDS),
            "excluded_training_signal_fields": list(EXCLUDED_TRAINING_SIGNAL_FIELDS),
            "policy_input_contract": "src.e04.no_oracle_protocol.LocalTrainingObservation",
            "lineage_note": self.lineage_note,
        }


@dataclass(frozen=True)
class PolicyDecision:
    """One action selected from projected local-only features."""

    action: str
    target_side: str | None
    scores: Mapping[str, float]
    feature_audit: Mapping[str, Any]


class S08LocalEvolutionPolicy:
    """Parameterized adjacent-action policy over the S07 projected view."""

    def __init__(self, genome: EvolutionGenome) -> None:
        self.genome = genome
        self.params = {name: float(genome.parameters[name]) for name in PARAMETER_NAMES}

    @staticmethod
    def _active_or_frozen(status: str | None) -> bool:
        return str(status) in {"ACTIVE", "FREEZE", "CellStatus.ACTIVE", "CellStatus.FREEZE"}

    @staticmethod
    def _left_disordered(obs: LocalTrainingObservation) -> bool:
        if not obs.left_present or obs.left_value is None:
            return False
        return int(obs.left_value) < int(obs.actor_value) if obs.reverse_direction else int(obs.left_value) > int(obs.actor_value)

    @staticmethod
    def _right_disordered(obs: LocalTrainingObservation) -> bool:
        if not obs.right_present or obs.right_value is None:
            return False
        return int(obs.actor_value) < int(obs.right_value) if obs.reverse_direction else int(obs.actor_value) > int(obs.right_value)

    def select_action(self, projected: LocalTrainingObservation) -> PolicyDecision:
        features = projected.to_feature_dict()
        audit = audit_training_feature_dict(features)
        left_disordered = self._left_disordered(projected)
        right_disordered = self._right_disordered(projected)
        left_movable = bool(projected.left_present and self._active_or_frozen(projected.left_status))
        right_movable = bool(projected.right_present and self._active_or_frozen(projected.right_status))
        frustration = min(1.0, max(0.0, float(projected.local_frustration) / 5.0))
        time_since = min(1.0, max(0.0, float(projected.time_since_movement) / 20.0))
        failed_swaps = min(1.0, max(0.0, float(projected.failed_swap_count) / 4.0))
        neighbor_count = min(1.0, max(0.0, float(projected.neighbor_identity_count) / 4.0))
        last_failure = 1.0 if projected.last_move_success is False else 0.0
        blocked_signal = min(1.0, max(0.0, float(projected.signal_blocked)))
        frustrated_signal = min(1.0, max(0.0, float(projected.signal_frustrated)))
        any_local_disorder = float(left_disordered or right_disordered)
        memory_signal_pressure = (
            self.params["frustration_weight"] * frustration
            + self.params["blocked_signal_weight"] * blocked_signal
            + self.params["frustrated_signal_weight"] * frustrated_signal
            + self.params["failed_swap_weight"] * failed_swaps
            + self.params["last_failure_weight"] * last_failure
            + self.params["time_since_weight"] * time_since
            + self.params["neighbor_count_weight"] * neighbor_count
        )
        left_score = (
            self.params["swap_left_bias"]
            + self.params["disorder_weight"] * float(left_disordered)
            + memory_signal_pressure
            - abs(self.params["illegal_penalty"]) * float(not left_movable)
        )
        right_score = (
            self.params["swap_right_bias"]
            + self.params["disorder_weight"] * float(right_disordered)
            + memory_signal_pressure
            - abs(self.params["illegal_penalty"]) * float(not right_movable)
        )
        idle_score = (
            self.params["idle_bias"]
            + self.params["idle_order_bonus"] * float(not any_local_disorder)
            - 0.35 * frustration
            - 0.15 * failed_swaps
        )
        scores = {
            "idle": float(idle_score),
            "swap_left": float(left_score),
            "swap_right": float(right_score),
        }
        ranked = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
        action = ranked[0][0]
        if action == "swap_left" and not left_movable:
            action = "idle"
        if action == "swap_right" and not right_movable:
            action = "idle"
        return PolicyDecision(
            action=action,
            target_side="left" if action == "swap_left" else "right" if action == "swap_right" else None,
            scores=scores,
            feature_audit=audit,
        )


@dataclass(frozen=True)
class S08TrajectoryResult:
    """Compact result for one evolved-policy trajectory."""

    genome: EvolutionGenome
    config: S08EvaluationConfig
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
    event_log: tuple[dict[str, Any], ...]
    memory_digest: str
    signal_digest: str
    action_counts: Mapping[str, int]
    feature_audit_summary: Mapping[str, Any]

    def _offline_metrics(self) -> dict[str, Any]:
        trajectory = [{"event_step": 0, "values": list(self.config.values)}]
        trajectory.extend(
            {"event_step": int(event["event_step"]), "values": list(event["current_values"])}
            for event in self.event_log
            if "current_values" in event
        )
        sortedness_values = [sortedness_percent(item["values"], self.config.sort_direction) for item in trajectory]
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
            "time_in_target_fraction": float(sum(1 for flag in target_flags) / len(target_flags)) if target_flags else 1.0,
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
        row = {
            **self.config.to_dict(),
            "policy_id": self.genome.policy_id,
            "policy_family": S08_POLICY_FAMILY,
            "candidate_type": "evolved_local_only",
            "baseline_type": "none",
            "generation": int(self.genome.generation),
            "population_index": int(self.genome.population_index),
            "parent_ids_json": json.dumps(list(self.genome.parent_ids), separators=(",", ":")),
            "mutation_seed": int(self.genome.mutation_seed),
            "mutation_scale": float(self.genome.mutation_scale),
            "protocol_id": S08_PROTOCOL_ID,
            "centralized_baseline": False,
            "uses_global_oracle": bool(
                self.signal_audit.get("usesGlobalOracle", False)
                or self.feature_audit_summary.get("usesGlobalOracle", False)
            ),
            "eligible_for_local_only_claims": True,
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
            "fitness_score": 0.0,
            "final_values_json": json.dumps(list(self.final_values), separators=(",", ":")),
            "final_labels_json": json.dumps(list(self.final_labels), separators=(",", ":")),
            "final_frozen_positions_json": json.dumps(list(self.final_frozen_positions), separators=(",", ":")),
            "schedule_json": json.dumps([item.to_dict() for item in self.schedule], sort_keys=True, separators=(",", ":"), default=str),
            "schedule_sha256": stable_json_sha256([item.to_dict() for item in self.schedule]),
            "perturbation_log_json": json.dumps(list(self.perturbation_log), sort_keys=True, separators=(",", ":"), default=str),
            "fatigue_transition_count": len(self.fatigue_transition_log),
            "impairment_count": len(self.impairment_log),
            "unfreeze_count": int(self.repair_summary.get("unfreeze_count", 0)),
            "repair_summary_json": json.dumps(dict(self.repair_summary), sort_keys=True, separators=(",", ":"), default=str),
            "reliability_summary_json": json.dumps(dict(self.reliability_summary), sort_keys=True, separators=(",", ":"), default=str),
            "signal_audit_json": json.dumps(dict(self.signal_audit), sort_keys=True, separators=(",", ":"), default=str),
            "feature_audit_summary_json": json.dumps(dict(self.feature_audit_summary), sort_keys=True, separators=(",", ":"), default=str),
            "action_counts_json": json.dumps(dict(sorted(self.action_counts.items())), sort_keys=True, separators=(",", ":")),
            "event_log_sha256": stable_json_sha256(list(self.event_log)),
            "memory_digest": self.memory_digest,
            "signal_digest": self.signal_digest,
            "parameter_json": json.dumps(dict(self.genome.parameters), sort_keys=True, separators=(",", ":")),
        }
        row["fitness_score"] = fitness_from_row(row)
        return row


def heuristic_parameter_vector() -> np.ndarray:
    """Return a deterministic seed policy resembling local Bubble repair."""

    values = {
        "idle_bias": -0.15,
        "swap_left_bias": 0.0,
        "swap_right_bias": 0.0,
        "disorder_weight": 2.25,
        "frustration_weight": 0.35,
        "blocked_signal_weight": 0.25,
        "frustrated_signal_weight": 0.20,
        "failed_swap_weight": 0.18,
        "last_failure_weight": -0.05,
        "time_since_weight": 0.08,
        "neighbor_count_weight": 0.02,
        "illegal_penalty": 6.0,
        "idle_order_bonus": 0.45,
    }
    return np.asarray([values[name] for name in PARAMETER_NAMES], dtype=np.float32)


def params_from_vector(vector: Sequence[float]) -> dict[str, float]:
    return {name: float(vector[index]) for index, name in enumerate(PARAMETER_NAMES)}


def policy_id_for(generation: int, population_index: int, parameters: Mapping[str, float], parent_ids: Sequence[str]) -> str:
    digest = stable_json_sha256(
        {
            "generation": generation,
            "population_index": population_index,
            "parameters": {name: round(float(parameters[name]), 8) for name in PARAMETER_NAMES},
            "parent_ids": list(parent_ids),
            "protocol_id": S08_PROTOCOL_ID,
        }
    )[:10]
    return f"e04_s08_g{generation:02d}_i{population_index:02d}_{digest}"


def _jax_normal(shape: tuple[int, ...], seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    info = jax_backend_summary()
    try:
        import jax

        key = jax.random.PRNGKey(int(seed))
        values = jax.random.normal(key, shape=shape)
        return np.asarray(values, dtype=np.float32), info
    except Exception as exc:  # pragma: no cover - environment dependent fallback
        rng = np.random.default_rng(seed)
        info = {**info, "fallbackToNumpy": True, "fallbackReason": repr(exc)}
        return rng.normal(size=shape).astype(np.float32), info


def initial_population(population_size: int, seed: int) -> tuple[list[EvolutionGenome], dict[str, Any]]:
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    noise, backend = _jax_normal((population_size, len(PARAMETER_NAMES)), seed)
    base = heuristic_parameter_vector()
    vectors = base[None, :] + 0.55 * noise
    vectors[0, :] = base
    vectors[:, PARAMETER_NAMES.index("illegal_penalty")] = np.clip(
        np.abs(vectors[:, PARAMETER_NAMES.index("illegal_penalty")]), 2.0, 9.0
    )
    genomes: list[EvolutionGenome] = []
    for index, vector in enumerate(vectors):
        params = params_from_vector(vector)
        policy_id = policy_id_for(0, index, params, ())
        genomes.append(
            EvolutionGenome(
                policy_id=policy_id,
                generation=0,
                population_index=index,
                parent_ids=(),
                mutation_seed=seed,
                mutation_scale=0.55,
                parameters=params,
                lineage_note="heuristic_seed" if index == 0 else "jax_initialized_noise_around_heuristic",
            )
        )
    return genomes, backend


def next_generation(
    *,
    elites: Sequence[EvolutionGenome],
    generation: int,
    population_size: int,
    seed: int,
    mutation_scale: float,
) -> tuple[list[EvolutionGenome], dict[str, Any]]:
    if not elites:
        raise ValueError("at least one elite is required")
    noise, backend = _jax_normal((population_size, len(PARAMETER_NAMES)), seed + generation)
    genomes: list[EvolutionGenome] = []
    for index in range(population_size):
        parent = elites[index % len(elites)]
        scale = 0.0 if index < len(elites) else float(mutation_scale)
        vector = parent.vector() + scale * noise[index]
        vector = np.clip(vector, -6.0, 6.0)
        vector[PARAMETER_NAMES.index("illegal_penalty")] = np.clip(abs(vector[PARAMETER_NAMES.index("illegal_penalty")]), 2.0, 9.0)
        params = params_from_vector(vector)
        policy_id = policy_id_for(generation, index, params, (parent.policy_id,))
        genomes.append(
            EvolutionGenome(
                policy_id=policy_id,
                generation=generation,
                population_index=index,
                parent_ids=(parent.policy_id,),
                mutation_seed=seed + generation,
                mutation_scale=scale,
                parameters=params,
                lineage_note="elite_copy" if scale == 0.0 else "jax_mutation_from_elite",
            )
        )
    return genomes, backend


def default_s08_eval_configs(
    *,
    max_events: int = 120,
    heldout_max_events: int = 140,
    train_seeds: Sequence[int] = (18001, 18002),
    validation_seeds: Sequence[int] = (18101, 18102),
    heldout_seeds: Sequence[int] = (18201, 18202),
) -> list[S08EvaluationConfig]:
    configs: list[S08EvaluationConfig] = []
    for seed in train_seeds:
        for task_name in ("swap_shocks", "frozen_damage"):
            for reliability_mode in ("no_fatigue_control", "fatigue_recovery"):
                configs.append(
                    S08EvaluationConfig(
                        split="train",
                        values=(1, 2, 3, 4, 5, 6),
                        task_name=task_name,
                        reliability_mode=reliability_mode,
                        activation_seed=seed,
                        policy_seed=seed + 10_000,
                        schedule_seed=seed + 20_000,
                        max_events=max_events,
                    )
                )
    for seed in validation_seeds:
        for task_name in ("turnover_replacement", "mixed_perturbations"):
            for reliability_mode in ("no_fatigue_control", "fatigue_recovery"):
                configs.append(
                    S08EvaluationConfig(
                        split="validation",
                        values=(1, 2, 3, 4, 5, 6),
                        task_name=task_name,
                        reliability_mode=reliability_mode,
                        activation_seed=seed,
                        policy_seed=seed + 10_000,
                        schedule_seed=seed + 20_000,
                        max_events=max_events,
                    )
                )
    for seed in heldout_seeds:
        for task_name in HOMEOSTATIC_TASKS:
            configs.append(
                S08EvaluationConfig(
                    split="heldout",
                    values=(1, 2, 3, 4, 5, 6, 7, 8),
                    task_name=task_name,
                    reliability_mode="cumulative_damage",
                    activation_seed=seed,
                    policy_seed=seed + 10_000,
                    schedule_seed=seed + 20_000,
                    max_events=heldout_max_events,
                )
            )
    return configs


def assert_split_separation(configs: Sequence[S08EvaluationConfig]) -> dict[str, Any]:
    seeds_by_split: dict[str, set[int]] = {"train": set(), "validation": set(), "heldout": set()}
    keys_by_split: dict[str, set[tuple[Any, ...]]] = {"train": set(), "validation": set(), "heldout": set()}
    for config in configs:
        seeds_by_split[config.split].update({config.activation_seed, config.policy_seed, config.schedule_seed})
        keys_by_split[config.split].add(config.split_key()[1:])
    overlaps: dict[str, list[int]] = {}
    for left in seeds_by_split:
        for right in seeds_by_split:
            if left >= right:
                continue
            overlap = sorted(seeds_by_split[left] & seeds_by_split[right])
            if overlap:
                overlaps[f"{left}_x_{right}"] = overlap
    return {
        "success": not overlaps,
        "seedOverlaps": overlaps,
        "trainConfigCount": len(keys_by_split["train"]),
        "validationConfigCount": len(keys_by_split["validation"]),
        "heldoutConfigCount": len(keys_by_split["heldout"]),
        "trainArraySizes": sorted({len(config.values) for config in configs if config.split == "train"}),
        "validationArraySizes": sorted({len(config.values) for config in configs if config.split == "validation"}),
        "heldoutArraySizes": sorted({len(config.values) for config in configs if config.split == "heldout"}),
    }


def _execute_adjacent_action(actor: Any, action: str, probe: EventTracingStatusProbe) -> tuple[int, int, int | None]:
    before_swaps = int(probe.swap_count)
    before_frozen = int(probe.frozen_swap_attempts)
    if action == "swap_left":
        target_index = int(actor.current_position[0]) - 1
    elif action == "swap_right":
        target_index = int(actor.current_position[0]) + 1
    else:
        return 0, 0, None
    if target_index < 0 or target_index >= len(actor.cells):
        return 0, 0, None
    probe.record_compare_and_swap()
    actor.swap((target_index, int(actor.current_position[1])))
    return int(probe.swap_count) - before_swaps, int(probe.frozen_swap_attempts) - before_frozen, target_index


def _feature_audit_summary(audits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    forbidden_hits: Counter[str] = Counter()
    excluded_signal_hits: Counter[str] = Counter()
    list_like_hits: Counter[str] = Counter()
    for audit in audits:
        forbidden_hits.update(str(item) for item in audit.get("forbiddenTokenHits", []))
        excluded_signal_hits.update(str(item) for item in audit.get("excludedSignalHits", []))
        list_like_hits.update(str(key) for key in dict(audit.get("listLikeFeatureValues", {})))
    return {
        "usesGlobalOracle": bool(forbidden_hits or excluded_signal_hits or list_like_hits),
        "auditCount": len(audits),
        "forbiddenTokenHits": dict(sorted(forbidden_hits.items())),
        "excludedSignalHits": dict(sorted(excluded_signal_hits.items())),
        "listLikeFeatureValueHits": dict(sorted(list_like_hits.items())),
        "allowedSignalFields": list(ALLOWED_TRAINING_SIGNAL_FIELDS),
        "excludedSignalFields": list(EXCLUDED_TRAINING_SIGNAL_FIELDS),
        "protocolId": LOCAL_ONLY_PROTOCOL_ID,
    }


def simulate_evolved_policy(config: S08EvaluationConfig, genome: EvolutionGenome) -> S08TrajectoryResult:
    """Run one evolved local-only policy on an S05 homeostatic task."""

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
            noise_seed=config.policy_seed,
        ),
    )
    repair_controller = RepairController(cells, cell_status, config.to_repair_config(), signal_bank)
    patch_repairable_swaps(cells, repair_controller)
    reliability = FatigueDamageController(config.to_fatigue_config())
    reliability.initialize(cells)
    policy = S08LocalEvolutionPolicy(genome)
    schedule = build_homeostatic_schedule(config.to_homeostatic_config())
    schedule_by_step: dict[int, list[PerturbationSpec]] = {}
    for spec in schedule:
        schedule_by_step.setdefault(int(spec.event_step), []).append(spec)

    activation_rng = random.Random(config.activation_seed)
    random.seed(config.policy_seed)
    cursor = 0
    event_log: list[dict[str, Any]] = []
    perturbation_log: list[dict[str, Any]] = []
    feature_audits: list[Mapping[str, Any]] = []
    action_counts: Counter[str] = Counter()
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
        before_signature = cell_state_signature(cells)
        before_public_signature = cells_signature(cells)
        before_repair_transitions = len(repair_controller.unfreeze_log)
        before_fatigue_transitions = len(reliability.transition_log)
        probe.current_event_step = event_step
        applied_action = "impaired_skip"
        target_index: int | None = None
        decision_scores: Mapping[str, float] = {}
        projected_feature_count = 0
        feature_audit_uses_oracle = False
        if reliability.can_act(actor):
            observation = observe_cell(actor)
            memory_before = memory_bank.get(observation.actor_thread_id)
            sensation = signal_bank.sense(observation, event_step=event_step) if config.signal_enabled else None
            projected = project_local_training_observation(observation, memory_before, sensation)
            decision = policy.select_action(projected)
            feature_audits.append(decision.feature_audit)
            feature_audit_uses_oracle = bool(decision.feature_audit.get("usesGlobalOracle", False))
            projected_feature_count = int(decision.feature_audit.get("featureCount", 0))
            decision_scores = decision.scores
            swap_delta, frozen_attempt_delta, target_index = _execute_adjacent_action(actor, decision.action, probe)
            action_counts[decision.action] += 1
            after_public_signature = cells_signature(cells)
            applied_action = "swap" if swap_delta > 0 else "blocked_swap_attempt" if frozen_attempt_delta > 0 else decision.action
            result = PolicyStepResult(
                behavior=observation.behavior,
                observation=observation,
                state_before=PolicyState(
                    ideal_position=None,
                    reverse_direction=projected.reverse_direction,
                    memory={"s08_local_only_protocol": S08_PROTOCOL_ID},
                ),
                proposed_action=PolicyAction(
                    action_type="swap" if target_index is not None else "idle",
                    target_index=target_index,
                    metadata={"target_side": decision.target_side},
                ),
                applied_action=PolicyAction(action_type=applied_action, target_index=target_index),
                signature_before=before_public_signature,
                signature_after=after_public_signature,
                comparison_delta=1 if target_index is not None else 0,
                swap_delta=swap_delta,
                frozen_attempt_delta=frozen_attempt_delta,
                state_changed=after_public_signature != before_public_signature,
            )
            memory_after = memory_bank.update_from_step(result, event_step=event_step)
            blocked = applied_action == "blocked_swap_attempt" or frozen_attempt_delta > 0
            if config.signal_enabled:
                signal_bank.emit(observation, memory_after, blocked=blocked, event_step=event_step)
            reliability.after_event(actor, swap_delta=swap_delta)
        else:
            reliability.record_impairment(actor, event_step)
            swap_delta = 0
            frozen_attempt_delta = 0
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
                "frozen_attempt_delta": int(frozen_attempt_delta),
                "public_state_changed": bool(after_signature != before_signature),
                "unfreeze_delta": len(repair_controller.unfreeze_log) - before_repair_transitions,
                "fatigue_transition_delta": len(reliability.transition_log) - before_fatigue_transitions,
                "perturbation_count_at_event": len(schedule_by_step.get(event_step, [])),
                "feature_protocol_id": LOCAL_ONLY_PROTOCOL_ID if projected_feature_count else "not_projected_impaired_or_inactive",
                "projected_feature_count": int(projected_feature_count),
                "feature_audit_uses_oracle": bool(feature_audit_uses_oracle),
                "decision_scores": {key: round(float(value), 6) for key, value in sorted(decision_scores.items())},
                "reliability_state_after": reliability.state_for(actor),
                "current_values": list(cell_values(cells)),
                "current_labels": list(cell_labels(cells)),
                "current_frozen_positions": list(frozen_positions(cells, cell_status)),
            }
        )

    return S08TrajectoryResult(
        genome=genome,
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
        event_log=tuple(event_log),
        memory_digest=memory_bank.stable_digest(),
        signal_digest=signal_bank.stable_digest(),
        action_counts=dict(action_counts),
        feature_audit_summary=_feature_audit_summary(feature_audits),
    )


def fitness_from_row(row: Mapping[str, Any]) -> float:
    """Scalar objective for S08 selection; higher is better."""

    time_target = float(row.get("time_in_target_fraction", 0.0) or 0.0)
    final_sorted = float(row.get("final_sortedness_percent", 0.0) or 0.0) / 100.0
    perturbation_count = max(1, int(row.get("perturbation_count", 0) or 0))
    recovered = float(row.get("recovered_perturbation_count", 0) or 0) / perturbation_count
    freeze_count = int(row.get("freeze_perturbation_count", 0) or 0)
    if freeze_count:
        repair = min(1.0, float(row.get("unfreeze_count", 0) or 0) / freeze_count)
    else:
        repair = recovered
    event_count = max(1, int(row.get("event_count", 0) or 0))
    energy_efficiency = 1.0 - min(1.0, float(row.get("energy_total", 0) or 0) / event_count)
    damage_efficiency = 1.0 - min(1.0, float(row.get("cumulative_impairment_count", 0) or 0) / event_count)
    oracle_penalty = 0.5 if bool(row.get("uses_global_oracle", False)) else 0.0
    return float(
        0.45 * time_target
        + 0.20 * final_sorted
        + 0.15 * recovered
        + 0.10 * repair
        + 0.07 * energy_efficiency
        + 0.03 * damage_efficiency
        - oracle_penalty
    )


def evaluate_genomes(genomes: Sequence[EvolutionGenome], configs: Sequence[S08EvaluationConfig]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for genome in genomes:
        for config in configs:
            result = simulate_evolved_policy(config, genome)
            rows.append(result.to_row())
    return rows


def summarize_policy_fitness(rows: Sequence[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("split") == split and row.get("candidate_type") == "evolved_local_only":
            grouped.setdefault(str(row["policy_id"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for policy_id, items in grouped.items():
        first = items[0]
        summary.append(
            {
                "policy_id": policy_id,
                "split": split,
                "generation": int(first["generation"]),
                "population_index": int(first["population_index"]),
                "mean_fitness": float(np.mean([float(item["fitness_score"]) for item in items])),
                "mean_time_in_target_fraction": float(np.mean([float(item["time_in_target_fraction"]) for item in items])),
                "mean_final_sortedness_percent": float(np.mean([float(item["final_sortedness_percent"]) for item in items])),
                "mean_energy_total": float(np.mean([float(item["energy_total"]) for item in items])),
                "oracle_hit_rows": int(sum(1 for item in items if bool(item.get("uses_global_oracle", False)))),
                "run_count": len(items),
            }
        )
    return sorted(summary, key=lambda item: (item["mean_fitness"], -item["generation"]), reverse=True)


def run_open_loop_baselines(
    configs: Sequence[S08EvaluationConfig],
    *,
    algorithms: Sequence[str] = ("bubble", "insertion", "selection"),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in configs:
        for algorithm in algorithms:
            result = simulate_homeostatic_benchmark(config.to_homeostatic_config(algorithm=algorithm))
            row = result.to_row()
            row.update(
                {
                    "split": config.split,
                    "array_size": len(config.values),
                    "policy_id": f"open_loop_{algorithm}",
                    "policy_family": "original_open_loop_cell_view",
                    "candidate_type": "open_loop_original",
                    "baseline_type": "original_open_loop",
                    "generation": -1,
                    "population_index": -1,
                    "parent_ids_json": "[]",
                    "mutation_seed": None,
                    "mutation_scale": 0.0,
                    "protocol_id": "not_a_training_policy_baseline",
                    "centralized_baseline": False,
                    "eligible_for_local_only_claims": True,
                    "feature_audit_summary_json": json.dumps(
                        {"usesGlobalOracle": False, "reason": "no learned training features"},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "action_counts_json": "{}",
                    "parameter_json": "{}",
                }
            )
            row["uses_global_oracle"] = bool(row.get("uses_global_oracle", False))
            row["fitness_score"] = fitness_from_row(row)
            rows.append(row)
    return rows


def select_top_genomes(
    genomes: Sequence[EvolutionGenome],
    fitness_summary: Sequence[Mapping[str, Any]],
    elite_count: int,
) -> list[EvolutionGenome]:
    by_id = {genome.policy_id: genome for genome in genomes}
    selected: list[EvolutionGenome] = []
    for record in fitness_summary:
        genome = by_id.get(str(record["policy_id"]))
        if genome is not None:
            selected.append(genome)
        if len(selected) >= elite_count:
            break
    return selected


def run_s08_search(
    *,
    population_size: int = 10,
    generations: int = 4,
    elite_count: int = 3,
    validation_count: int = 4,
    heldout_count: int = 2,
    max_events: int = 120,
    heldout_max_events: int = 140,
    seed: int = 80801,
) -> dict[str, Any]:
    """Run the compact S08 evolutionary search and return row-level results."""

    if generations < 1:
        raise ValueError("generations must be positive")
    configs = default_s08_eval_configs(max_events=max_events, heldout_max_events=heldout_max_events)
    split_audit = assert_split_separation(configs)
    train_configs = [config for config in configs if config.split == "train"]
    validation_configs = [config for config in configs if config.split == "validation"]
    heldout_configs = [config for config in configs if config.split == "heldout"]
    population, init_backend = initial_population(population_size, seed)
    backend_records = [{"stage": "initial_population", **init_backend}]
    all_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    lineage_records: list[dict[str, Any]] = []

    for generation in range(generations):
        train_rows = evaluate_genomes(population, train_configs)
        all_rows.extend(train_rows)
        train_summary = summarize_policy_fitness(train_rows, "train")
        for rank, record in enumerate(train_summary, start=1):
            progress_rows.append({**record, "rank": rank})
        for genome in population:
            matching = [record for record in train_summary if record["policy_id"] == genome.policy_id]
            lineage_records.append(
                {
                    **genome.to_policy_record(),
                    "train_mean_fitness": float(matching[0]["mean_fitness"]) if matching else None,
                    "train_rank": next((index for index, item in enumerate(train_summary, start=1) if item["policy_id"] == genome.policy_id), None),
                }
            )
        elites = select_top_genomes(population, train_summary, elite_count)
        if generation < generations - 1:
            mutation_scale = max(0.12, 0.38 * (0.72 ** generation))
            population, backend = next_generation(
                elites=elites,
                generation=generation + 1,
                population_size=population_size,
                seed=seed,
                mutation_scale=mutation_scale,
            )
            backend_records.append({"stage": f"generation_{generation + 1}_mutation", **backend})

    final_train_summary = summarize_policy_fitness(all_rows, "train")
    final_generation = max(row["generation"] for row in final_train_summary) if final_train_summary else generations - 1
    final_train_summary = [row for row in final_train_summary if row["generation"] == final_generation]
    validation_genomes = select_top_genomes(population, final_train_summary, validation_count)
    validation_rows = evaluate_genomes(validation_genomes, validation_configs)
    all_rows.extend(validation_rows)
    validation_summary = summarize_policy_fitness(validation_rows, "validation")
    heldout_genomes = select_top_genomes(validation_genomes, validation_summary, heldout_count)
    heldout_rows = evaluate_genomes(heldout_genomes, heldout_configs)
    all_rows.extend(heldout_rows)
    heldout_summary = summarize_policy_fitness(heldout_rows, "heldout")
    baseline_rows = run_open_loop_baselines([*train_configs, *validation_configs, *heldout_configs])
    all_rows.extend(baseline_rows)

    selected_records: list[dict[str, Any]] = []
    summary_by_split = {
        "train": {record["policy_id"]: record for record in final_train_summary},
        "validation": {record["policy_id"]: record for record in validation_summary},
        "heldout": {record["policy_id"]: record for record in heldout_summary},
    }
    for genome in heldout_genomes:
        record = genome.to_policy_record()
        for split, split_records in summary_by_split.items():
            if genome.policy_id in split_records:
                record[f"{split}_mean_fitness"] = float(split_records[genome.policy_id]["mean_fitness"])
                record[f"{split}_run_count"] = int(split_records[genome.policy_id]["run_count"])
        selected_records.append(record)

    drift_log = [
        {
            "item": "s07_projected_local_only_view",
            "status": "enforced",
            "detail": "evolved action selection consumes LocalTrainingObservation feature dictionaries only",
        },
        {
            "item": "target_derived_signal_fields",
            "status": "excluded_from_training",
            "detail": f"allowed={list(ALLOWED_TRAINING_SIGNAL_FIELDS)}; excluded={list(EXCLUDED_TRAINING_SIGNAL_FIELDS)}",
        },
        {
            "item": "e03_jax_dsl_batch_kernel",
            "status": "not_used_for_policy_trajectories",
            "detail": "E03 DSL batch kernel exposes ideal-position/target machinery incompatible with S07 local-only training; S08 uses the S05 Python event simulator and JAX/GPU only for vectorized population mutation/ranking metadata.",
        },
        {
            "item": "jax_gpu_backend",
            "status": "available" if any(record.get("defaultBackend") == "gpu" for record in backend_records) else "fallback_or_unavailable",
            "detail": json.dumps(backend_records[-1], sort_keys=True, default=str),
        },
    ]
    return {
        "rows": all_rows,
        "progressRows": progress_rows,
        "lineageRecords": lineage_records,
        "selectedPolicies": selected_records,
        "splitAudit": split_audit,
        "backendRecords": backend_records,
        "driftLog": drift_log,
        "configs": [config.to_dict() for config in configs],
    }
