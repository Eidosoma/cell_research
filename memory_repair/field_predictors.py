"""Seed-separated simulated field predictors for E04 S12."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from e02_deterministic_simulator.metrics import sortedness_percent

from .competence_proxies import (
    CompetencePolicySpec,
    baseline_competence_policies,
    competence_policy_from_s08_candidate,
    local_memory_signal_competence_policy,
    policy_for_competence_condition,
)
from .fatigue import FATIGUE_DAMAGE_VERSION, FatigueDamageConfig
from .homeostasis import HOMEOSTASIS_BENCHMARK_VERSION, normalized_sortedness_record, stable_hash, stable_json
from .learning import LOCAL_LEARNING_VERSION, LearningEventSimulator, apply_homeostatic_event_to_simulator
from .memory import MEMORY_REPAIR_VERSION
from .repair import REPAIR_REPAIR_VERSION, RepairRuleConfig
from .signals import SIGNAL_CHANNELS, SIGNAL_REPAIR_VERSION, SignalConfig
from .training_constraints import TRAINING_CONSTRAINT_VERSION


FIELD_PREDICTOR_VERSION = "e04_s12_tissue_field_predictors.v1"
FIELD_PROXY_SCOPE_NOTE = (
    "Simulated aggregate field proxy only; not a biological tissue field, bioelectric field, "
    "or morphogen measurement."
)
FIELD_TARGETS = (
    "futureRepairEventProxy",
    "futureFailureEventProxy",
    "futureRecoveryEventProxy",
)
FIELD_VARIANTS = (
    "observed_fields",
    "permuted_null_fields",
    "zero_null_fields",
    "gaussian_null_fields",
)
FIELD_FEATURE_PREFIXES = (
    "signal_",
    "memory_",
    "learning_",
    "repair_",
    "fatigue_",
)


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
        return round(float(value), 12) if math.isfinite(value) else None
    return value


def compact_json(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:12]}"


def config_hash(payload: Any) -> str:
    return hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()


def _loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return default
    return value


@dataclass(frozen=True)
class FieldPredictionTask:
    task_id: str
    task_family: str
    task_panel: str
    initial_values: tuple[int, ...]
    horizon: int
    frozen_positions: tuple[int, ...] = ()
    sortedness_threshold_percent: float = 95.0
    perturbation_schedule: tuple[Mapping[str, Any], ...] = ()
    activations_per_tick: int = 12
    fatigue_config: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        schedule = [dict(event) for event in self.perturbation_schedule]
        return {
            "taskId": self.task_id,
            "taskFamily": self.task_family,
            "taskPanel": self.task_panel,
            "initialValues": list(self.initial_values),
            "initialLength": int(len(self.initial_values)),
            "horizon": int(self.horizon),
            "frozenPositions": list(self.frozen_positions),
            "sortednessThresholdPercent": float(self.sortedness_threshold_percent),
            "perturbationSchedule": schedule,
            "activationsPerTick": int(self.activations_per_tick),
            "fatigueConfig": dict(self.fatigue_config or FatigueDamageConfig("none").to_dict()),
            "scheduleHash": stable_hash(schedule),
            "claimBoundary": FIELD_PROXY_SCOPE_NOTE,
            "fieldPredictorVersion": FIELD_PREDICTOR_VERSION,
        }


def build_s12_tasks() -> tuple[FieldPredictionTask, ...]:
    recovery_schedule = (
        {"eventIndex": 0, "tick": 1, "type": "swap", "leftIndex": 2, "rightIndex": 3, "source": "s12_field_swap"},
        {"eventIndex": 1, "tick": 2, "type": "freeze", "cellId": 4, "source": "s12_field_freeze"},
        {"eventIndex": 2, "tick": 3, "type": "damage", "cellId": 2, "durationTicks": 2, "source": "s12_field_damage"},
        {"eventIndex": 3, "tick": 5, "type": "recover", "cellId": 4, "source": "s12_field_recover"},
        {"eventIndex": 4, "tick": 6, "type": "swap", "leftIndex": 4, "rightIndex": 5, "source": "s12_field_swap_late"},
    )
    return (
        FieldPredictionTask(
            task_id="field_repair_signal_pair_n3",
            task_family="repair",
            task_panel="s12_repair_fields",
            initial_values=(2, 1, 3),
            frozen_positions=(1,),
            horizon=44,
        ),
        FieldPredictionTask(
            task_id="field_repair_barrier_n4",
            task_family="repair",
            task_panel="s12_repair_fields",
            initial_values=(3, 1, 2, 4),
            frozen_positions=(1,),
            horizon=56,
        ),
        FieldPredictionTask(
            task_id="field_fatigue_damage_n8",
            task_family="fatigue",
            task_panel="s12_failure_fields",
            initial_values=(8, 1, 7, 2, 6, 3, 5, 4),
            horizon=84,
            fatigue_config=FatigueDamageConfig(
                "combined",
                movement_threshold=3,
                failed_swap_threshold=2,
                frustration_threshold=2,
                recovery_activations=2,
                damage_probability=0.08,
                damage_cooldown_activations=2,
                random_seed=12701,
            ).to_dict(),
        ),
        FieldPredictionTask(
            task_id="field_homeostasis_mixed_n8",
            task_family="homeostasis",
            task_panel="s12_recovery_fields",
            initial_values=(1, 2, 3, 4, 5, 6, 7, 8),
            horizon=9,
            perturbation_schedule=recovery_schedule,
            activations_per_tick=14,
            fatigue_config=FatigueDamageConfig(
                "stochastic_damage",
                damage_probability=0.05,
                damage_cooldown_activations=2,
                random_seed=12702,
            ).to_dict(),
        ),
    )


def build_s12_policies(evolved_rows: Sequence[Mapping[str, Any]] = (), *, max_evolved: int = 2) -> tuple[CompetencePolicySpec, ...]:
    policies: list[CompetencePolicySpec] = [baseline_competence_policies()[0], local_memory_signal_competence_policy()]
    for row in evolved_rows:
        policy = competence_policy_from_s08_candidate(row)
        if policy.replayable:
            policies.append(policy)
        if sum(1 for item in policies if item.family_kind == "evolved") >= max_evolved:
            break
    return tuple(policies)


def condition_rows_for_field_policy(
    policy: CompetencePolicySpec,
    tasks: Sequence[FieldPredictionTask],
    *,
    seed_count: int = 5,
    train_seed_count: int = 3,
    seed_base: int = 14100,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not policy.replayable:
        return rows
    base_signal = SignalConfig.from_spec(policy.signal_config or "no_signal")
    base_repair = RepairRuleConfig.from_spec(policy.repair_config or RepairRuleConfig("nudge_count", nudge_threshold=2))
    for task_index, task in enumerate(tasks):
        task_payload = task.to_dict()
        fatigue = FatigueDamageConfig.from_spec(task_payload["fatigueConfig"])
        for seed_index in range(seed_count):
            scheduler_seed = int(seed_base + task_index * 100 + seed_index)
            tie_seed = int(seed_base + 5000 + task_index * 100 + seed_index)
            signal_seed = int(seed_base + 9000 + task_index * 100 + seed_index)
            signal_config = SignalConfig.from_spec({**base_signal.to_dict(), "randomSeed": signal_seed})
            split = "train" if int(seed_index) < int(train_seed_count) else "test"
            controlled = {
                "policy": policy.to_dict(),
                "task": task_payload,
                "schedulerSeed": scheduler_seed,
                "tieBreakerSeed": tie_seed,
                "signalConfig": signal_config.to_dict(),
                "repairConfig": base_repair.to_dict(),
                "fatigueConfig": fatigue.to_dict(),
            }
            rows.append(
                {
                    "conditionId": stable_id("s12_condition", [policy.policy_id, task.task_id, seed_index]),
                    "policyId": policy.policy_id,
                    "policyGroup": policy.policy_group,
                    "familyKind": policy.family_kind,
                    "sourceStep": policy.source_step,
                    "sourceCandidateId": policy.source_candidate_id,
                    "taskId": task.task_id,
                    "taskFamily": task.task_family,
                    "taskPanel": task.task_panel,
                    "seedIndex": int(seed_index),
                    "split": split,
                    "schedulerSeed": scheduler_seed,
                    "tieBreakerSeed": tie_seed,
                    "signalRandomSeed": signal_seed,
                    "trainSeedCount": int(train_seed_count),
                    "heldOutForModelEvaluation": split == "test",
                    "selectionUsed": False,
                    "oracleAccessAllowed": bool(policy.oracle_access_allowed),
                    "policyAuditSuccess": bool(policy.policy_audit_success),
                    "taskSpecJson": compact_json(task_payload),
                    "signalConfigJson": compact_json(signal_config.to_dict()),
                    "repairConfigJson": compact_json(base_repair.to_dict()),
                    "fatigueConfigJson": compact_json(fatigue.to_dict()),
                    "memoryConfigJson": compact_json(policy.memory_config),
                    "learningConfigJson": compact_json(policy.learning_config) if policy.learning_config else None,
                    "basePolicySpecJson": compact_json(policy.base_policy_spec),
                    "controlledConfigHash": config_hash(controlled),
                    "claimBoundary": FIELD_PROXY_SCOPE_NOTE,
                    "fieldPredictorVersion": FIELD_PREDICTOR_VERSION,
                    "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
                }
            )
    return rows


def _numeric_array(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    if isinstance(values, np.ndarray):
        return values.astype(float)
    if isinstance(values, (list, tuple)):
        return np.asarray([float(value) for value in values], dtype=float)
    return np.asarray([], dtype=float)


def _field_stats(prefix: str, values: Any) -> dict[str, float]:
    arr = _numeric_array(values)
    if arr.size == 0:
        return {
            f"{prefix}_sum": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_nonzero_fraction": 0.0,
            f"{prefix}_gradient_total": 0.0,
            f"{prefix}_gradient_max": 0.0,
        }
    gradients = np.abs(np.diff(arr)) if arr.size > 1 else np.asarray([], dtype=float)
    return {
        f"{prefix}_sum": float(arr.sum()),
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_nonzero_fraction": float((arr > 1e-12).mean()),
        f"{prefix}_gradient_total": float(gradients.sum()) if gradients.size else 0.0,
        f"{prefix}_gradient_max": float(gradients.max()) if gradients.size else 0.0,
    }


def _signal_features(row: Mapping[str, Any]) -> dict[str, float]:
    fields = _loads(row.get("signal_fields_json"), {})
    out: dict[str, float] = {}
    total = 0.0
    total_gradient = 0.0
    for channel in SIGNAL_CHANNELS:
        stats = _field_stats(f"signal_{channel}", fields.get(channel, []))
        out.update(stats)
        total += stats[f"signal_{channel}_sum"]
        total_gradient += stats[f"signal_{channel}_gradient_total"]
    out["signal_total_energy"] = float(total)
    out["signal_total_gradient"] = float(total_gradient)
    return out


def _memory_features(row: Mapping[str, Any]) -> dict[str, float]:
    states = _loads(row.get("memory_states_json"), [])
    memory_states = [item.get("memory_state", {}) for item in states if isinstance(item, Mapping)]
    active = [state for state in memory_states if isinstance(state, Mapping) and bool(state)]
    n = max(1, len(memory_states))
    frustrations = np.asarray([float(state.get("local_frustration", 0.0)) for state in active], dtype=float)
    failed = np.asarray([float(state.get("recent_failed_swaps", 0.0)) for state in active], dtype=float)
    ages = np.asarray([float(state.get("time_since_movement", 0.0)) for state in active], dtype=float)
    last_success = np.asarray([float(bool(state.get("last_move_success", False))) for state in active], dtype=float)
    neighbor_ids = []
    for state in active:
        ids = state.get("recent_neighbor_ids", [])
        if isinstance(ids, list):
            neighbor_ids.extend(int(value) for value in ids if value is not None)
    return {
        "memory_active_density": float(len(active) / n),
        "memory_local_frustration_mean": float(frustrations.mean()) if frustrations.size else 0.0,
        "memory_local_frustration_max": float(frustrations.max()) if frustrations.size else 0.0,
        "memory_recent_failed_swaps_sum": float(failed.sum()) if failed.size else 0.0,
        "memory_recent_failed_swaps_mean": float(failed.mean()) if failed.size else 0.0,
        "memory_time_since_movement_mean": float(ages.mean()) if ages.size else 0.0,
        "memory_last_move_success_fraction": float(last_success.mean()) if last_success.size else 0.0,
        "memory_recent_neighbor_unique_count": float(len(set(neighbor_ids))),
    }


def _learning_features(row: Mapping[str, Any]) -> dict[str, float]:
    states = _loads(row.get("learning_states_json"), [])
    learning_states = [item.get("learning_state", {}) for item in states if isinstance(item, Mapping)]
    active = [state for state in learning_states if isinstance(state, Mapping) and bool(state)]
    n = max(1, len(learning_states))
    recent_failures = np.asarray([float(state.get("recentLocalFailures", 0.0)) for state in active], dtype=float)
    rewards = np.asarray([float(state.get("lastReward", 0.0)) for state in active], dtype=float)
    updates = np.asarray([float(state.get("updateCount", 0.0)) for state in active], dtype=float)
    probs = []
    for state in active:
        payload = state.get("actionProbabilities", {})
        if isinstance(payload, Mapping):
            probs.append([float(payload.get("swap_left", 0.0)), float(payload.get("swap_right", 0.0)), float(payload.get("wait", 0.0))])
    prob_arr = np.asarray(probs, dtype=float) if probs else np.zeros((0, 3), dtype=float)
    entropy = 0.0
    if prob_arr.size:
        clipped = np.clip(prob_arr, 1e-12, 1.0)
        entropy = float((-clipped * np.log(clipped)).sum(axis=1).mean())
    return {
        "learning_active_density": float(len(active) / n),
        "learning_recent_failures_mean": float(recent_failures.mean()) if recent_failures.size else 0.0,
        "learning_recent_failures_max": float(recent_failures.max()) if recent_failures.size else 0.0,
        "learning_last_reward_mean": float(rewards.mean()) if rewards.size else 0.0,
        "learning_update_count_mean": float(updates.mean()) if updates.size else 0.0,
        "learning_action_entropy_mean": entropy,
        "learning_swap_left_probability_mean": float(prob_arr[:, 0].mean()) if prob_arr.size else 0.0,
        "learning_swap_right_probability_mean": float(prob_arr[:, 1].mean()) if prob_arr.size else 0.0,
        "learning_wait_probability_mean": float(prob_arr[:, 2].mean()) if prob_arr.size else 0.0,
    }


def _repair_features(row: Mapping[str, Any]) -> tuple[dict[str, float], int]:
    states = _loads(row.get("repair_states_json"), [])
    repair_states = [state for state in states if isinstance(state, Mapping)]
    frozen = [state for state in repair_states if bool(state.get("frozen", False))]
    exposure = np.asarray([float(state.get("signal_exposure", 0.0)) for state in repair_states], dtype=float)
    nudge = np.asarray([float(state.get("nudge_count", 0.0)) for state in repair_states], dtype=float)
    age = np.asarray([float(state.get("frozen_age_activations", 0.0)) for state in repair_states], dtype=float)
    events = _loads(row.get("recovery_events_json"), [])
    recovery_count = len(events) if isinstance(events, list) else 0
    return (
        {
            "repair_frozen_count": float(len(frozen)),
            "repair_frozen_density": float(len(frozen) / max(1, len(repair_states))) if repair_states else 0.0,
            "repair_signal_exposure_sum": float(exposure.sum()) if exposure.size else 0.0,
            "repair_signal_exposure_mean": float(exposure.mean()) if exposure.size else 0.0,
            "repair_signal_exposure_max": float(exposure.max()) if exposure.size else 0.0,
            "repair_nudge_count_sum": float(nudge.sum()) if nudge.size else 0.0,
            "repair_frozen_age_mean": float(age.mean()) if age.size else 0.0,
            "repair_frozen_age_max": float(age.max()) if age.size else 0.0,
        },
        int(recovery_count),
    )


def _fatigue_features(row: Mapping[str, Any]) -> tuple[dict[str, float], int, int]:
    states = _loads(row.get("fatigue_states_json"), [])
    fatigue_states = [state for state in states if isinstance(state, Mapping)]
    fatigued = [state for state in fatigue_states if bool(state.get("fatigued", False))]
    damaged = [state for state in fatigue_states if bool(state.get("damaged", False))]
    frustration = np.asarray([float(state.get("frustration_count", 0.0)) for state in fatigue_states], dtype=float)
    failed = np.asarray([float(state.get("failed_swap_count", 0.0)) for state in fatigue_states], dtype=float)
    movement = np.asarray([float(state.get("movement_count", 0.0)) for state in fatigue_states], dtype=float)
    events = _loads(row.get("fatigue_events_json"), [])
    event_types = [str(event.get("event_type", "")) for event in events if isinstance(event, Mapping)]
    recovery_count = sum(1 for event_type in event_types if event_type.endswith("recovered"))
    failure_count = sum(1 for event_type in event_types if event_type and not event_type.endswith("recovered"))
    return (
        {
            "fatigue_fatigued_count": float(len(fatigued)),
            "fatigue_damaged_count": float(len(damaged)),
            "fatigue_impaired_density": float((len(fatigued) + len(damaged)) / max(1, len(fatigue_states))) if fatigue_states else 0.0,
            "fatigue_frustration_sum": float(frustration.sum()) if frustration.size else 0.0,
            "fatigue_frustration_mean": float(frustration.mean()) if frustration.size else 0.0,
            "fatigue_failed_swap_sum": float(failed.sum()) if failed.size else 0.0,
            "fatigue_movement_sum": float(movement.sum()) if movement.size else 0.0,
        },
        int(failure_count),
        int(recovery_count),
    )


def extract_field_feature_rows(
    trace_rows: Sequence[Mapping[str, Any]],
    condition: Mapping[str, Any],
    *,
    lookahead_rows: int = 8,
) -> list[dict[str, Any]]:
    base_rows: list[dict[str, Any]] = []
    for trace_index, row in enumerate(trace_rows):
        repair_features, recovery_count = _repair_features(row)
        fatigue_features, fatigue_failure_count, fatigue_recovery_count = _fatigue_features(row)
        values = _loads(row.get("frozen_positions_json"), [])
        frozen_count = int(repair_features["repair_frozen_count"]) if "repair_frozen_count" in repair_features else len(values)
        feature_row = {
            "conditionId": condition["conditionId"],
            "policyId": condition["policyId"],
            "policyGroup": condition["policyGroup"],
            "familyKind": condition["familyKind"],
            "taskId": condition["taskId"],
            "taskFamily": condition["taskFamily"],
            "taskPanel": condition["taskPanel"],
            "seedIndex": int(condition["seedIndex"]),
            "split": condition["split"],
            "schedulerSeed": int(condition["schedulerSeed"]),
            "tieBreakerSeed": int(condition["tieBreakerSeed"]),
            "traceRowIndex": int(trace_index),
            "eventIndex": int(row.get("event_index", trace_index)),
            "activationIndex": int(row.get("activation_index", 0)),
            "eventKind": str(row.get("event_kind", "")),
            "currentSortednessPercent": float(row.get("sortedness_percent", 100.0)),
            "currentMonotonicityError": int(row.get("monotonicity_error", 0)),
            "currentBlockedMoveAttempts": int(row.get("blocked_move_attempts", 0)),
            "currentSwapCount": int(row.get("swap_count", 0)),
            "currentFrozenCount": int(frozen_count),
            "currentRepairRecoveryCount": int(recovery_count),
            "currentFatigueFailureCount": int(fatigue_failure_count),
            "currentFatigueRecoveryCount": int(fatigue_recovery_count),
            "claimBoundary": FIELD_PROXY_SCOPE_NOTE,
            "fieldPredictorVersion": FIELD_PREDICTOR_VERSION,
        }
        feature_row.update(_signal_features(row))
        feature_row.update(_memory_features(row))
        feature_row.update(_learning_features(row))
        feature_row.update(repair_features)
        feature_row.update(fatigue_features)
        base_rows.append(_json_ready(feature_row))

    labeled_rows: list[dict[str, Any]] = []
    for index, row in enumerate(base_rows):
        future = base_rows[index + 1 : index + 1 + int(lookahead_rows)]
        if not future:
            continue
        future_sortedness = [float(item["currentSortednessPercent"]) for item in future]
        future_repair_count = max(int(item["currentRepairRecoveryCount"]) for item in future)
        future_fatigue_failure_count = max(int(item["currentFatigueFailureCount"]) for item in future)
        future_fatigue_recovery_count = max(int(item["currentFatigueRecoveryCount"]) for item in future)
        future_blocked = max(int(item["currentBlockedMoveAttempts"]) for item in future)
        future_frozen = max(int(item["currentFrozenCount"]) for item in future)
        current_sortedness = float(row["currentSortednessPercent"])
        current_repair_count = int(row["currentRepairRecoveryCount"])
        current_fatigue_failure_count = int(row["currentFatigueFailureCount"])
        current_fatigue_recovery_count = int(row["currentFatigueRecoveryCount"])
        current_blocked = int(row["currentBlockedMoveAttempts"])
        current_frozen = int(row["currentFrozenCount"])
        repair_event = future_repair_count > current_repair_count
        failure_event = (
            future_blocked > current_blocked
            or future_fatigue_failure_count > current_fatigue_failure_count
            or future_frozen > current_frozen
            or min(future_sortedness) <= current_sortedness - 10.0
        )
        recovery_event = (
            repair_event
            or future_fatigue_recovery_count > current_fatigue_recovery_count
            or (current_sortedness < 95.0 and max(future_sortedness) >= 95.0 and max(future_sortedness) >= current_sortedness + 5.0)
        )
        row = dict(row)
        row.update(
            {
                "lookaheadRows": int(lookahead_rows),
                "futureWindowRowCount": int(len(future)),
                "futureRepairEventProxy": bool(repair_event),
                "futureFailureEventProxy": bool(failure_event),
                "futureRecoveryEventProxy": bool(recovery_event),
            }
        )
        labeled_rows.append(_json_ready(row))
    return labeled_rows


def run_field_condition(policy_spec: CompetencePolicySpec, condition: Mapping[str, Any], *, lookahead_rows: int = 8) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task = _loads(condition["taskSpecJson"], {})
    policy = policy_for_competence_condition(policy_spec)
    signal_config = SignalConfig.from_spec(_loads(condition["signalConfigJson"], {}))
    repair_config = RepairRuleConfig.from_spec(_loads(condition["repairConfigJson"], {}))
    fatigue_config = FatigueDamageConfig.from_spec(_loads(condition["fatigueConfigJson"], {}))
    sim = LearningEventSimulator(
        task["initialValues"],
        policy,
        frozen_positions=task.get("frozenPositions", ()),
        frozen_variant="stuck" if task.get("frozenPositions") else "none",
        scheduler_seed=int(condition["schedulerSeed"]),
        tie_breaker_seed=int(condition["tieBreakerSeed"]),
        signal_config=signal_config,
        repair_config=repair_config,
        fatigue_config=fatigue_config,
        condition_id=str(condition["conditionId"]),
        implementation="e04_s12_field_predictor_cpu_reference",
        research_step_id="S12",
    )
    task_family = str(task["taskFamily"])
    if task_family == "homeostasis":
        initial_length = int(task["initialLength"])
        threshold = float(task["sortednessThresholdPercent"])
        tick_in_range = 0
        for tick in range(int(task["horizon"])):
            events = [event for event in task.get("perturbationSchedule", ()) if int(event["tick"]) == tick]
            for event in events:
                apply_homeostatic_event_to_simulator(sim, event)
            for _ in range(int(task.get("activationsPerTick", 12))):
                sim.step()
            post = normalized_sortedness_record(sim.current_values(), initial_length=initial_length)
            tick_in_range += int(float(post["sortednessPercent"]) >= threshold)
        homeostatic_maintenance = float(tick_in_range / max(1, int(task["horizon"])))
    else:
        homeostatic_maintenance = None
        for _ in range(int(task["horizon"])):
            sim.step()

    final_sortedness = float(sortedness_percent(sim.current_values()))
    trace_feature_rows = extract_field_feature_rows(sim.trace_rows, condition, lookahead_rows=lookahead_rows)
    result = {
        **{key: condition[key] for key in condition.keys()},
        "traceRowCount": int(len(sim.trace_rows)),
        "featureRowCount": int(len(trace_feature_rows)),
        "finalSortednessPercent": final_sortedness,
        "homeostaticMaintenance": homeostatic_maintenance,
        "recoveredCellCount": int(len(sim.recovery_events)),
        "remainingFrozenCellCount": int(len(sim.current_frozen_positions())),
        "fatigueEventCount": int(len(getattr(sim, "fatigue_events", []))),
        "blockedMoveAttempts": int(sim.blocked_move_attempts),
        "swapCount": int(sim.swap_count),
        "comparisonCount": int(sim.comparison_count),
        "activationCount": int(sim.activation_count),
        "finalValuesJson": compact_json(sim.current_values()),
        "claimBoundary": FIELD_PROXY_SCOPE_NOTE,
        "memoryRepairVersion": MEMORY_REPAIR_VERSION,
        "signalRepairVersion": SIGNAL_REPAIR_VERSION,
        "repairRepairVersion": REPAIR_REPAIR_VERSION,
        "fatigueDamageVersion": FATIGUE_DAMAGE_VERSION,
        "localLearningVersion": LOCAL_LEARNING_VERSION,
        "homeostasisBenchmarkVersion": HOMEOSTASIS_BENCHMARK_VERSION,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
        "fieldPredictorVersion": FIELD_PREDICTOR_VERSION,
    }
    return _json_ready(result), _json_ready(trace_feature_rows)


def field_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in feature_df.columns
        if any(str(column).startswith(prefix) for prefix in FIELD_FEATURE_PREFIXES)
        and pd.api.types.is_numeric_dtype(feature_df[column])
    )


def field_feature_definitions(feature_columns: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for column in feature_columns:
        if column.startswith("signal_"):
            source = "simulated local signal field"
        elif column.startswith("memory_"):
            source = "bounded per-cell memory field"
        elif column.startswith("learning_"):
            source = "local learning-state field"
        elif column.startswith("repair_"):
            source = "repair-state field"
        elif column.startswith("fatigue_"):
            source = "fatigue/damage-state field"
        else:
            source = "simulated aggregate field"
        rows.append(
            {
                "featureId": column,
                "sourceFieldFamily": source,
                "directComputationalDefinition": f"Aggregate trace-time value extracted from {source}; column `{column}`.",
                "usedForTargets": list(FIELD_TARGETS),
                "claimBoundary": FIELD_PROXY_SCOPE_NOTE,
                "fieldPredictorVersion": FIELD_PREDICTOR_VERSION,
            }
        )
    return rows


def _variant_matrices(
    x_train: np.ndarray,
    x_test: np.ndarray,
    *,
    variant: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    if variant == "observed_fields":
        return x_train.copy(), x_test.copy()
    if variant == "zero_null_fields":
        return np.zeros_like(x_train), np.zeros_like(x_test)
    if variant == "permuted_null_fields":
        train = x_train.copy()
        test = x_test.copy()
        for col in range(train.shape[1]):
            train[:, col] = rng.permutation(train[:, col])
            test[:, col] = rng.permutation(test[:, col])
        return train, test
    if variant == "gaussian_null_fields":
        means = np.nanmean(x_train, axis=0)
        stds = np.nanstd(x_train, axis=0)
        stds = np.where(stds <= 1e-12, 0.0, stds)
        train = rng.normal(means, stds, size=x_train.shape)
        test = rng.normal(means, stds, size=x_test.shape)
        return train, test
    raise ValueError(f"unknown field variant: {variant}")


def evaluate_field_predictors(
    feature_df: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    null_seed: int = 15100,
) -> pd.DataFrame:
    feature_columns = list(feature_columns or field_feature_columns(feature_df))
    rows: list[dict[str, Any]] = []
    if not feature_columns:
        return pd.DataFrame()
    train_mask = feature_df["split"].astype(str) == "train"
    test_mask = feature_df["split"].astype(str) == "test"
    x_train_observed = feature_df.loc[train_mask, feature_columns].to_numpy(dtype=float)
    x_test_observed = feature_df.loc[test_mask, feature_columns].to_numpy(dtype=float)
    for target_index, target in enumerate(FIELD_TARGETS):
        y_train = feature_df.loc[train_mask, target].astype(int).to_numpy()
        y_test = feature_df.loc[test_mask, target].astype(int).to_numpy()
        train_classes = set(int(value) for value in y_train)
        test_classes = set(int(value) for value in y_test)
        for variant_index, variant in enumerate(FIELD_VARIANTS):
            x_train, x_test = _variant_matrices(
                x_train_observed,
                x_test_observed,
                variant=variant,
                seed=int(null_seed + target_index * 100 + variant_index),
            )
            evaluated = len(train_classes) == 2 and len(test_classes) == 2 and len(y_train) > 0 and len(y_test) > 0
            detail = "evaluated"
            metrics = {
                "testRocAuc": None,
                "testAveragePrecision": None,
                "testBalancedAccuracy": None,
                "testBrierScore": None,
            }
            if evaluated:
                try:
                    model = make_pipeline(
                        SimpleImputer(strategy="median"),
                        StandardScaler(),
                        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=0),
                    )
                    model.fit(x_train, y_train)
                    scores = model.predict_proba(x_test)[:, 1]
                    pred = (scores >= 0.5).astype(int)
                    metrics = {
                        "testRocAuc": float(roc_auc_score(y_test, scores)),
                        "testAveragePrecision": float(average_precision_score(y_test, scores)),
                        "testBalancedAccuracy": float(balanced_accuracy_score(y_test, pred)),
                        "testBrierScore": float(brier_score_loss(y_test, scores)),
                    }
                except Exception as exc:  # pragma: no cover - defensive path for degenerate panels
                    evaluated = False
                    detail = f"model_failed: {type(exc).__name__}: {exc}"
            else:
                detail = f"insufficient class variation trainClasses={sorted(train_classes)} testClasses={sorted(test_classes)}"
            rows.append(
                _json_ready(
                    {
                        "targetId": target,
                        "fieldVariant": variant,
                        "evaluated": bool(evaluated),
                        "detail": detail,
                        "trainRows": int(len(y_train)),
                        "testRows": int(len(y_test)),
                        "trainPositiveCount": int(y_train.sum()) if len(y_train) else 0,
                        "testPositiveCount": int(y_test.sum()) if len(y_test) else 0,
                        "testPrevalence": float(y_test.mean()) if len(y_test) else None,
                        "featureColumnCount": int(len(feature_columns)),
                        **metrics,
                        "claimBoundary": FIELD_PROXY_SCOPE_NOTE,
                        "fieldPredictorVersion": FIELD_PREDICTOR_VERSION,
                    }
                )
            )
    return pd.DataFrame(rows)


def summarize_field_predictors(model_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if model_df.empty:
        return pd.DataFrame()
    for target, group in model_df.groupby("targetId", dropna=False):
        observed = group[(group["fieldVariant"] == "observed_fields") & group["evaluated"].astype(bool)]
        nulls = group[(group["fieldVariant"] != "observed_fields") & group["evaluated"].astype(bool)]
        if observed.empty or nulls.empty:
            rows.append(
                {
                    "targetId": target,
                    "observedAuc": None,
                    "bestNullAuc": None,
                    "observedMinusBestNullAuc": None,
                    "observedAveragePrecision": None,
                    "bestNullAveragePrecision": None,
                    "observedMinusBestNullAveragePrecision": None,
                    "supportiveProxySignal": False,
                    "detail": "missing evaluated observed or null rows",
                    "claimBoundary": FIELD_PROXY_SCOPE_NOTE,
                    "fieldPredictorVersion": FIELD_PREDICTOR_VERSION,
                }
            )
            continue
        observed_row = observed.iloc[0]
        best_null_auc = float(nulls["testRocAuc"].max())
        best_null_ap = float(nulls["testAveragePrecision"].max())
        auc_delta = float(observed_row["testRocAuc"] - best_null_auc)
        ap_delta = float(observed_row["testAveragePrecision"] - best_null_ap)
        rows.append(
            _json_ready(
                {
                    "targetId": target,
                    "observedAuc": float(observed_row["testRocAuc"]),
                    "bestNullAuc": best_null_auc,
                    "observedMinusBestNullAuc": auc_delta,
                    "observedAveragePrecision": float(observed_row["testAveragePrecision"]),
                    "bestNullAveragePrecision": best_null_ap,
                    "observedMinusBestNullAveragePrecision": ap_delta,
                    "supportiveProxySignal": bool(auc_delta > 0.02 or ap_delta > 0.02),
                    "detail": "observed field model compared against best null-field control",
                    "claimBoundary": FIELD_PROXY_SCOPE_NOTE,
                    "fieldPredictorVersion": FIELD_PREDICTOR_VERSION,
                }
            )
        )
    return pd.DataFrame(rows)


def validate_field_predictor_outputs(
    condition_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    model_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    definition_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check_id: str, success: bool, detail: str) -> None:
        rows.append(
            {
                "checkId": check_id,
                "success": bool(success),
                "detail": detail,
                "claimBoundary": FIELD_PROXY_SCOPE_NOTE,
                "fieldPredictorVersion": FIELD_PREDICTOR_VERSION,
            }
        )

    add("condition_rows_nonempty", len(condition_df) > 0, f"conditionRows={len(condition_df)}")
    add("trace_feature_rows_nonempty", len(trace_df) > 0, f"traceRows={len(trace_df)}")
    train_seeds = set(condition_df.loc[condition_df["split"] == "train", "seedIndex"].astype(int))
    test_seeds = set(condition_df.loc[condition_df["split"] == "test", "seedIndex"].astype(int))
    add("seed_separated_train_test", bool(train_seeds and test_seeds and train_seeds.isdisjoint(test_seeds)), f"trainSeeds={sorted(train_seeds)} testSeeds={sorted(test_seeds)}")
    add("no_selection_or_global_oracle", bool((~condition_df["selectionUsed"].astype(bool)).all() and (~condition_df["oracleAccessAllowed"].astype(bool)).all()), "selectionUsed=false and oracleAccessAllowed=false")
    add("policy_audits_passed", bool(condition_df["policyAuditSuccess"].astype(bool).all()), "all replayed policies passed S07 no-oracle audit")
    feature_columns = field_feature_columns(trace_df)
    add("field_feature_columns_present", len(feature_columns) >= 12, f"featureColumnCount={len(feature_columns)}")
    target_variation_ok = True
    details = []
    for target in FIELD_TARGETS:
        train_classes = trace_df.loc[trace_df["split"] == "train", target].astype(int).nunique()
        test_classes = trace_df.loc[trace_df["split"] == "test", target].astype(int).nunique()
        details.append(f"{target}:trainClasses={train_classes},testClasses={test_classes}")
        target_variation_ok = target_variation_ok and train_classes == 2 and test_classes == 2
    add("targets_have_train_test_class_variation", bool(target_variation_ok), "; ".join(details))
    variants = set(model_df["fieldVariant"]) if not model_df.empty else set()
    add("null_field_controls_present", set(FIELD_VARIANTS).issubset(variants), f"variants={sorted(variants)}")
    evaluated_observed = model_df[(model_df["fieldVariant"] == "observed_fields") & model_df["evaluated"].astype(bool)]
    add("observed_models_evaluated", len(evaluated_observed) == len(FIELD_TARGETS), f"observedEvaluatedRows={len(evaluated_observed)}")
    add("observed_vs_null_quantified", len(summary_df) == len(FIELD_TARGETS), f"summaryRows={len(summary_df)}")
    definition_ok = len(definition_df) == len(feature_columns) and definition_df["claimBoundary"].astype(str).str.contains("Simulated aggregate field proxy only", regex=False).all()
    add("field_definitions_documented", bool(definition_ok), f"definitionRows={len(definition_df)} featureColumns={len(feature_columns)}")
    scope_ok = (
        trace_df["claimBoundary"].astype(str).str.contains("Simulated aggregate field proxy only", regex=False).all()
        and model_df["claimBoundary"].astype(str).str.contains("Simulated aggregate field proxy only", regex=False).all()
        and summary_df["claimBoundary"].astype(str).str.contains("Simulated aggregate field proxy only", regex=False).all()
    )
    add("proxy_scope_boundaries_present", bool(scope_ok), "trace, model, and summary rows carry S12 proxy scope language")
    return pd.DataFrame(rows)


def field_replay_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(stable_json(list(rows)).encode("utf-8")).hexdigest()
