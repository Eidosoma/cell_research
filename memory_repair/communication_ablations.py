"""Communication-architecture ablation helpers for E04 S10."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from e02_deterministic_simulator.metrics import sortedness_percent
from morphospace.competence import delayed_gratification_from_sortedness

from .ablations import (
    PolicyFamilySpec,
    classic_policy_families,
    evolved_policy_family_from_candidate,
    learned_policy_families,
    policy_family_from_frontier_row,
)
from .fatigue import FATIGUE_DAMAGE_VERSION, FatigueDamageConfig
from .homeostasis import HOMEOSTASIS_BENCHMARK_VERSION, build_homeostatic_benchmark_config, normalized_sortedness_record, stable_hash, stable_json
from .learning import (
    LOCAL_LEARNING_VERSION,
    LearningEventSimulator,
    LocalLearningConfig,
    LocalLearningPolicyWrapper,
    apply_homeostatic_event_to_simulator,
)
from .memory import MEMORY_REPAIR_VERSION
from .repair import REPAIR_REPAIR_VERSION, RepairRuleConfig
from .signals import SIGNAL_CHANNELS, SIGNAL_REPAIR_VERSION, SignalConfig, SignalPolicyWrapper, serialize_signal_fields
from .training_constraints import TRAINING_CONSTRAINT_VERSION


COMMUNICATION_ABLATION_VERSION = "e04_s10_communication_ablations.v1"
SIGNAL_ABLATION_AXIS = "signal_architecture"
SIGNAL_ABLATION_VARIANTS = (
    "no_signal",
    "nearest_neighbor",
    "diffusive",
    "long_range_scalar",
    "noisy_diffusive",
    "randomized_control",
)
SIGNAL_REFERENCE_VARIANT = "no_signal"


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
        return round(float(value), 12) if math.isfinite(float(value)) else None
    return value


def compact_json(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:12]}"


def config_hash(payload: Any) -> str:
    return hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CommunicationAblationTask:
    task_id: str
    task_family: str
    task_panel: str
    initial_values: tuple[int, ...]
    horizon: int
    repair_config: Mapping[str, Any]
    fatigue_config: Mapping[str, Any]
    frozen_positions: tuple[int, ...] = ()
    sortedness_threshold_percent: float = 95.0
    perturbation_schedule: tuple[Mapping[str, Any], ...] = ()
    activations_per_tick: int = 12
    transfer: bool = False

    def to_dict(self) -> dict[str, Any]:
        schedule = [dict(event) for event in self.perturbation_schedule]
        return {
            "taskId": self.task_id,
            "taskFamily": self.task_family,
            "taskPanel": self.task_panel,
            "initialValues": list(self.initial_values),
            "horizon": int(self.horizon),
            "frozenPositions": list(self.frozen_positions),
            "sortednessThresholdPercent": float(self.sortedness_threshold_percent),
            "perturbationSchedule": schedule,
            "activationsPerTick": int(self.activations_per_tick),
            "repairConfig": dict(self.repair_config),
            "fatigueConfig": dict(self.fatigue_config),
            "transfer": bool(self.transfer),
            "scheduleHash": stable_hash(schedule),
            "communicationAblationVersion": COMMUNICATION_ABLATION_VERSION,
        }


def build_s10_tasks() -> tuple[CommunicationAblationTask, ...]:
    """Return deterministic repairable, fatigue, and homeostasis tasks for S10."""

    homeostasis = {task["taskId"]: task for task in build_homeostatic_benchmark_config()["tasks"]}
    signal_repair = RepairRuleConfig("signal_threshold", signal_channel="blocked", signal_threshold=1.0, approach_direction="either").to_dict()
    harder_signal_repair = RepairRuleConfig(
        "signal_threshold",
        signal_channel="blocked",
        signal_threshold=1.4,
        approach_direction="either",
    ).to_dict()
    return (
        CommunicationAblationTask(
            task_id="repairable_signal_pair_training",
            task_family="repairable",
            task_panel="s10_repairable_training",
            initial_values=(2, 1, 3),
            frozen_positions=(1,),
            horizon=32,
            repair_config=signal_repair,
            fatigue_config=FatigueDamageConfig("none").to_dict(),
        ),
        CommunicationAblationTask(
            task_id="repairable_barrier_transfer",
            task_family="repairable",
            task_panel="s10_repairable_transfer",
            initial_values=(4, 1, 3, 2, 5),
            frozen_positions=(1,),
            horizon=48,
            repair_config=harder_signal_repair,
            fatigue_config=FatigueDamageConfig("none").to_dict(),
            transfer=True,
        ),
        CommunicationAblationTask(
            task_id="fatigue_failed_swap_training",
            task_family="fatigue",
            task_panel="s10_fatigue_training",
            initial_values=(3, 1, 2, 4),
            frozen_positions=(1,),
            horizon=48,
            repair_config=signal_repair,
            fatigue_config=FatigueDamageConfig("failed_swap", failed_swap_threshold=2, recovery_activations=2).to_dict(),
        ),
        CommunicationAblationTask(
            task_id="fatigue_combined_transfer",
            task_family="fatigue",
            task_panel="s10_fatigue_transfer",
            initial_values=(5, 1, 4, 2, 3, 6),
            frozen_positions=(2,),
            horizon=64,
            repair_config=harder_signal_repair,
            fatigue_config=FatigueDamageConfig(
                "combined",
                movement_threshold=3,
                failed_swap_threshold=2,
                frustration_threshold=2,
                recovery_activations=2,
                damage_probability=0.05,
                damage_cooldown_activations=2,
                random_seed=710,
            ).to_dict(),
            transfer=True,
        ),
        CommunicationAblationTask(
            task_id="homeostasis_frozen_damage_training",
            task_family="homeostasis",
            task_panel="s10_homeostasis_training",
            initial_values=tuple(homeostasis["frozen_damage_mixed_events"]["initialValues"]),
            horizon=int(homeostasis["frozen_damage_mixed_events"]["horizonTicks"]),
            sortedness_threshold_percent=float(homeostasis["frozen_damage_mixed_events"]["sortednessThresholdPercent"]),
            perturbation_schedule=tuple(homeostasis["frozen_damage_mixed_events"]["perturbationSchedule"]),
            activations_per_tick=12,
            repair_config=signal_repair,
            fatigue_config=FatigueDamageConfig(
                "stochastic_damage",
                damage_probability=0.05,
                damage_cooldown_activations=2,
                random_seed=711,
            ).to_dict(),
        ),
        CommunicationAblationTask(
            task_id="homeostasis_insert_delete_transfer",
            task_family="homeostasis",
            task_panel="s10_homeostasis_transfer",
            initial_values=tuple(homeostasis["insertion_deletion_normalization"]["initialValues"]),
            horizon=int(homeostasis["insertion_deletion_normalization"]["horizonTicks"]),
            sortedness_threshold_percent=float(homeostasis["insertion_deletion_normalization"]["sortednessThresholdPercent"]),
            perturbation_schedule=tuple(homeostasis["insertion_deletion_normalization"]["perturbationSchedule"]),
            activations_per_tick=12,
            repair_config=signal_repair,
            fatigue_config=FatigueDamageConfig("none").to_dict(),
            transfer=True,
        ),
    )


def signal_config_for_variant(variant: str, *, random_seed: int = 0, array_length: int = 8) -> SignalConfig:
    """Map S10 ablation labels to logged SignalConfig records."""

    if variant == "no_signal":
        return SignalConfig("no_signal", signal_range=0, random_seed=random_seed)
    if variant == "nearest_neighbor":
        return SignalConfig("nearest_neighbor", signal_range=1, diffusion_rate=0.0, decay=0.05, random_seed=random_seed)
    if variant == "diffusive":
        return SignalConfig("diffusive", signal_range=1, diffusion_rate=0.25, decay=0.05, random_seed=random_seed)
    if variant == "long_range_scalar":
        return SignalConfig(
            "diffusive",
            channels=SIGNAL_CHANNELS,
            signal_range=max(3, int(array_length)),
            diffusion_rate=0.4,
            decay=0.01,
            emission_scale=1.0,
            random_seed=random_seed,
        )
    if variant == "noisy_diffusive":
        return SignalConfig(
            "diffusive",
            signal_range=1,
            diffusion_rate=0.25,
            decay=0.05,
            noise_std=0.25,
            random_seed=random_seed,
        )
    if variant == "randomized_control":
        return SignalConfig("randomized", signal_range=1, random_seed=random_seed)
    raise ValueError(f"unknown S10 signal ablation variant: {variant}")


def condition_rows_for_communication_family(
    family: PolicyFamilySpec,
    tasks: Sequence[CommunicationAblationTask],
    *,
    seed_count: int = 3,
    seed_base: int = 10100,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not family.replayable:
        return rows
    for task_index, task in enumerate(tasks):
        task_payload = task.to_dict()
        repair_config = RepairRuleConfig.from_spec(task.repair_config)
        fatigue_config = FatigueDamageConfig.from_spec(task.fatigue_config)
        for seed_index in range(seed_count):
            scheduler_seed = int(seed_base + task_index * 100 + seed_index)
            tie_seed = int(seed_base + 5000 + task_index * 100 + seed_index)
            for variant_index, variant in enumerate(SIGNAL_ABLATION_VARIANTS):
                signal_seed = int(seed_base + 9000 + task_index * 100 + seed_index * 10 + variant_index)
                signal_config = signal_config_for_variant(variant, random_seed=signal_seed, array_length=len(task.initial_values))
                controlled = {
                    "family": family.to_dict(),
                    "task": task_payload,
                    "schedulerSeed": scheduler_seed,
                    "tieBreakerSeed": tie_seed,
                    "repairConfig": repair_config.to_dict(),
                    "fatigueConfig": fatigue_config.to_dict(),
                    "axis": SIGNAL_ABLATION_AXIS,
                }
                mechanism = {
                    "signalArchitecture": variant,
                    "signalConfig": signal_config.to_dict(),
                }
                rows.append(
                    {
                        "conditionId": stable_id("s10_condition", [family.family_id, task.task_id, seed_index, variant]),
                        "familyId": family.family_id,
                        "familyKind": family.family_kind,
                        "sourceStep": family.source_step,
                        "sourceCandidateId": family.source_candidate_id,
                        "ablationAxis": SIGNAL_ABLATION_AXIS,
                        "ablationVariant": variant,
                        "referenceVariant": SIGNAL_REFERENCE_VARIANT,
                        "taskId": task.task_id,
                        "taskFamily": task.task_family,
                        "taskPanel": task.task_panel,
                        "transfer": bool(task.transfer),
                        "seedIndex": int(seed_index),
                        "schedulerSeed": scheduler_seed,
                        "tieBreakerSeed": tie_seed,
                        "signalRandomSeed": signal_seed,
                        "signalEngineVariant": signal_config.variant,
                        "signalRange": int(signal_config.signal_range),
                        "diffusionRate": float(signal_config.diffusion_rate),
                        "decay": float(signal_config.decay),
                        "noiseStd": float(signal_config.noise_std),
                        "emissionScale": float(signal_config.emission_scale),
                        "taskSpecJson": compact_json(task_payload),
                        "signalConfigJson": compact_json(signal_config.to_dict()),
                        "repairConfigJson": compact_json(repair_config.to_dict()),
                        "fatigueConfigJson": compact_json(fatigue_config.to_dict()),
                        "basePolicySpecJson": compact_json(family.base_policy_spec),
                        "controlledConfigHash": config_hash(controlled),
                        "mechanismConfigHash": config_hash(mechanism),
                        "controlledGroupId": "|".join([family.family_id, SIGNAL_ABLATION_AXIS, task.task_id, str(seed_index)]),
                        "communicationAblationVersion": COMMUNICATION_ABLATION_VERSION,
                    }
                )
    return rows


def _policy_for_family(family: PolicyFamilySpec):
    if family.family_kind == "evolved":
        return LocalLearningPolicyWrapper("bubble", LocalLearningConfig.from_spec(family.learning_config or {}))
    if family.family_kind == "learned":
        return LocalLearningPolicyWrapper("bubble", LocalLearningConfig.from_spec(family.learning_config or "local_adaptive"))
    return family.base_policy


def _active_fatigue_counts(simulator: LearningEventSimulator) -> tuple[int, int]:
    fatigued = 0
    damaged = 0
    for state in getattr(simulator, "fatigue_states", {}).values():
        fatigued += int(int(state.get("fatigue_cooldown_remaining", 0)) > 0)
        damaged += int(int(state.get("damage_cooldown_remaining", 0)) > 0)
    return fatigued, damaged


def _trace_sortedness(simulator: LearningEventSimulator, fallback: Sequence[int]) -> list[float]:
    values = [float(row.get("sortedness_percent", np.nan)) for row in simulator.trace_rows if "sortedness_percent" in row]
    values = [value for value in values if math.isfinite(value)]
    if values:
        return values
    return [float(sortedness_percent(fallback))]


def _signal_energy_from_fields(fields: Mapping[str, np.ndarray]) -> float:
    return float(sum(float(np.asarray(values, dtype=float).sum()) for values in fields.values()))


def _signal_trace_proxies(simulator: LearningEventSimulator) -> dict[str, float]:
    cumulative_energy = 0.0
    false_alarm_energy = 0.0
    max_range_seen = 0.0
    for row in simulator.trace_rows:
        signal_fields = row.get("signal_fields_json")
        if isinstance(signal_fields, str):
            try:
                fields = json.loads(signal_fields)
            except json.JSONDecodeError:
                fields = {}
        else:
            fields = {}
        row_energy = 0.0
        for values in fields.values():
            if isinstance(values, list):
                row_energy += sum(float(value) for value in values)
        cumulative_energy += row_energy
        repair_states = row.get("repair_states_json")
        frozen_present = False
        if isinstance(repair_states, str):
            try:
                states = json.loads(repair_states)
                frozen_present = any(bool(state.get("frozen", False)) for state in states if isinstance(state, Mapping))
            except json.JSONDecodeError:
                frozen_present = False
        if row_energy > 0.0 and not frozen_present:
            false_alarm_energy += row_energy
        signal_config = row.get("signal_config_json")
        if isinstance(signal_config, str):
            try:
                payload = json.loads(signal_config)
                max_range_seen = max(max_range_seen, float(payload.get("signalRange", 0)))
            except json.JSONDecodeError:
                pass
    return {
        "cumulativeSignalEnergy": float(cumulative_energy),
        "falseAlarmSignalEnergy": float(false_alarm_energy),
        "maxSignalRangeObserved": float(max_range_seen),
    }


def _run_condition_core(
    family: PolicyFamilySpec,
    condition: Mapping[str, Any],
    *,
    explicit_no_signal_wrapper: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task = json.loads(str(condition["taskSpecJson"]))
    base_policy = _policy_for_family(family)
    signal_config = SignalConfig.from_spec(json.loads(str(condition["signalConfigJson"])))
    repair_config = RepairRuleConfig.from_spec(json.loads(str(condition["repairConfigJson"])))
    fatigue_config = FatigueDamageConfig.from_spec(json.loads(str(condition["fatigueConfigJson"])))
    policy = SignalPolicyWrapper(base_policy, SignalConfig("no_signal")) if explicit_no_signal_wrapper else base_policy
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
        auto_wrap_policies=not explicit_no_signal_wrapper,
        condition_id=str(condition["conditionId"]),
        implementation="e04_s10_communication_ablation_cpu_reference",
        research_step_id="S10",
    )
    tick_rows: list[dict[str, Any]] = []
    initial_length = len(task["initialValues"])
    if task["taskFamily"] in {"repairable", "fatigue"}:
        for _ in range(int(task["horizon"])):
            if sim.is_sorted() and not sim.current_frozen_positions():
                break
            sim.step()
    else:
        threshold = float(task["sortednessThresholdPercent"])
        for tick in range(int(task["horizon"])):
            events = [event for event in task.get("perturbationSchedule", ()) if int(event["tick"]) == tick]
            for event in events:
                apply_homeostatic_event_to_simulator(sim, event)
            pre = normalized_sortedness_record(sim.current_values(), initial_length=initial_length)
            for _ in range(int(task.get("activationsPerTick", 12))):
                sim.step()
            post = normalized_sortedness_record(sim.current_values(), initial_length=initial_length)
            tick_rows.append(
                {
                    "conditionId": condition["conditionId"],
                    "familyId": condition["familyId"],
                    "ablationVariant": condition["ablationVariant"],
                    "taskId": condition["taskId"],
                    "taskFamily": condition["taskFamily"],
                    "seedIndex": int(condition["seedIndex"]),
                    "tick": int(tick),
                    "eventsJson": compact_json(events),
                    "preSortednessPercent": float(pre["sortednessPercent"]),
                    "postSortednessPercent": float(post["sortednessPercent"]),
                    "postInRange": bool(post["sortednessPercent"] >= threshold),
                    "currentLength": int(post["currentLength"]),
                    "currentPairDenominator": int(post["currentPairDenominator"]),
                    "signalEnergyAfterTick": _signal_energy_from_fields(sim.signal_fields),
                    "valuesJson": compact_json(sim.current_values()),
                    "communicationAblationVersion": COMMUNICATION_ABLATION_VERSION,
                    "homeostasisBenchmarkVersion": HOMEOSTASIS_BENCHMARK_VERSION,
                }
            )

    sortedness_series = _trace_sortedness(sim, sim.current_values())
    dg = delayed_gratification_from_sortedness(sortedness_series)
    final_sortedness = float(sortedness_percent(sim.current_values()))
    if task["taskFamily"] == "homeostasis" and tick_rows:
        time_in_range = float(sum(row["postInRange"] for row in tick_rows) / len(tick_rows))
        completed = bool(time_in_range >= 0.75 and final_sortedness >= float(task["sortednessThresholdPercent"]))
    else:
        time_in_range = None
        completed = bool(sim.is_sorted() and not sim.current_frozen_positions())
    repair_success = bool(sim.is_sorted() and not sim.current_frozen_positions()) if task["taskFamily"] in {"repairable", "fatigue"} else None
    active_fatigued, active_damaged = _active_fatigue_counts(sim)
    signal_proxies = _signal_trace_proxies(sim)
    final_signal_energy = _signal_energy_from_fields(sim.signal_fields)
    energy_proxy = float(
        sim.comparison_count
        + sim.swap_count
        + sim.blocked_move_attempts
        + 0.01 * signal_proxies["cumulativeSignalEnergy"]
        + 0.05 * len(getattr(sim, "fatigue_events", []))
    )
    coordination_proxy = float(
        len(sim.recovery_events)
        + (0.0 if time_in_range is None else time_in_range)
        + (1.0 if repair_success else 0.0 if repair_success is not None else 0.0)
    )
    score = (
        (2.0 if completed else 0.0)
        + final_sortedness / 100.0
        + coordination_proxy
        - 0.004 * energy_proxy
        - 0.01 * signal_proxies["falseAlarmSignalEnergy"]
    )
    row = {
        **{key: condition[key] for key in condition.keys()},
        "completed": completed,
        "repairSuccess": repair_success,
        "homeostaticMaintenance": time_in_range,
        "finalSortednessPercent": final_sortedness,
        "delayedGratification": float(dg["delayedGratification"]),
        "delayedGratificationEventCount": int(dg["dgEventCount"]),
        "coordinationProxy": coordination_proxy,
        "falseAlarmProxy": float(signal_proxies["falseAlarmSignalEnergy"]),
        "signalEnergyProxy": float(signal_proxies["cumulativeSignalEnergy"]),
        "finalSignalEnergy": final_signal_energy,
        "energyProxy": energy_proxy,
        "activationCount": int(sim.activation_count),
        "swapCount": int(sim.swap_count),
        "comparisonCount": int(sim.comparison_count),
        "blockedMoveAttempts": int(sim.blocked_move_attempts),
        "recoveredCellCount": int(len(sim.recovery_events)),
        "remainingFrozenCellCount": int(len(sim.current_frozen_positions())),
        "fatigueEventCount": int(len(getattr(sim, "fatigue_events", []))),
        "activeFatiguedCellCount": int(active_fatigued),
        "activeDamagedCellCount": int(active_damaged),
        "initialValuesJson": compact_json(task["initialValues"]),
        "finalValuesJson": compact_json(sim.current_values()),
        "finalSignalFieldsJson": compact_json(serialize_signal_fields(sim.signal_fields)),
        "sortednessTraceJson": compact_json(sortedness_series),
        "score": float(score),
        "memoryRepairVersion": MEMORY_REPAIR_VERSION,
        "signalRepairVersion": SIGNAL_REPAIR_VERSION,
        "repairRepairVersion": REPAIR_REPAIR_VERSION,
        "fatigueDamageVersion": FATIGUE_DAMAGE_VERSION,
        "localLearningVersion": LOCAL_LEARNING_VERSION,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
        "communicationAblationVersion": COMMUNICATION_ABLATION_VERSION,
    }
    return _json_ready(row), _json_ready(tick_rows)


def run_communication_condition(family: PolicyFamilySpec, condition: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _run_condition_core(family, condition, explicit_no_signal_wrapper=False)


def run_no_signal_baseline_condition(family: PolicyFamilySpec, condition: Mapping[str, Any]) -> dict[str, Any]:
    if str(condition["ablationVariant"]) != SIGNAL_REFERENCE_VARIANT:
        raise ValueError("no-signal baseline reproduction only accepts the no_signal ablation arm")
    row, _ = _run_condition_core(family, condition, explicit_no_signal_wrapper=True)
    return row


def compute_communication_deltas(result_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = (
        "score",
        "finalSortednessPercent",
        "energyProxy",
        "delayedGratification",
        "coordinationProxy",
        "falseAlarmProxy",
        "signalEnergyProxy",
        "activationCount",
    )
    for group_id, group in result_df.groupby("controlledGroupId", dropna=False):
        reference = group[group["ablationVariant"] == SIGNAL_REFERENCE_VARIANT]
        if reference.empty:
            continue
        ref = reference.iloc[0]
        for _, row in group.iterrows():
            if row["ablationVariant"] == SIGNAL_REFERENCE_VARIANT:
                continue
            payload = {
                "controlledGroupId": group_id,
                "familyId": row["familyId"],
                "familyKind": row["familyKind"],
                "ablationAxis": row["ablationAxis"],
                "ablationVariant": row["ablationVariant"],
                "referenceVariant": SIGNAL_REFERENCE_VARIANT,
                "taskId": row["taskId"],
                "taskFamily": row["taskFamily"],
                "taskPanel": row["taskPanel"],
                "transfer": bool(row["transfer"]),
                "seedIndex": int(row["seedIndex"]),
                "communicationAblationVersion": COMMUNICATION_ABLATION_VERSION,
            }
            for metric in metrics:
                payload[f"{metric}Delta"] = float(row[metric]) - float(ref[metric])
            payload["completedDelta"] = int(bool(row["completed"])) - int(bool(ref["completed"]))
            if pd.notna(row.get("repairSuccess")) and pd.notna(ref.get("repairSuccess")):
                payload["repairSuccessDelta"] = int(bool(row["repairSuccess"])) - int(bool(ref["repairSuccess"]))
            else:
                payload["repairSuccessDelta"] = None
            rows.append(_json_ready(payload))
    return pd.DataFrame(rows)


def summarize_communication_effects(delta_df: pd.DataFrame) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()
    grouped = delta_df.groupby(["ablationVariant", "familyKind", "taskFamily", "transfer"], dropna=False)
    summary = grouped.agg(
        comparisonCount=("scoreDelta", "size"),
        meanScoreDelta=("scoreDelta", "mean"),
        medianScoreDelta=("scoreDelta", "median"),
        maxScoreDelta=("scoreDelta", "max"),
        minScoreDelta=("scoreDelta", "min"),
        meanRepairSuccessDelta=("repairSuccessDelta", "mean"),
        meanCoordinationDelta=("coordinationProxyDelta", "mean"),
        meanFalseAlarmDelta=("falseAlarmProxyDelta", "mean"),
        meanSignalEnergyDelta=("signalEnergyProxyDelta", "mean"),
        meanEnergyDelta=("energyProxyDelta", "mean"),
        meanDelayedGratificationDelta=("delayedGratificationDelta", "mean"),
        improvedScoreFraction=("scoreDelta", lambda values: float((values > 1e-9).mean())),
    ).reset_index()
    summary["communicationAblationVersion"] = COMMUNICATION_ABLATION_VERSION
    return summary


def communication_transfer_gaps(delta_df: pd.DataFrame) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()
    grouped = delta_df.groupby(["ablationVariant", "familyKind", "taskFamily", "transfer"], dropna=False)["scoreDelta"].mean().reset_index()
    rows: list[dict[str, Any]] = []
    for keys, group in grouped.groupby(["ablationVariant", "familyKind", "taskFamily"], dropna=False):
        training = group[group["transfer"].astype(bool) == False]["scoreDelta"]
        transfer = group[group["transfer"].astype(bool) == True]["scoreDelta"]
        if training.empty or transfer.empty:
            continue
        rows.append(
            {
                "ablationVariant": keys[0],
                "familyKind": keys[1],
                "taskFamily": keys[2],
                "trainingMeanScoreDelta": float(training.iloc[0]),
                "transferMeanScoreDelta": float(transfer.iloc[0]),
                "overfittingGap": float(training.iloc[0] - transfer.iloc[0]),
                "communicationAblationVersion": COMMUNICATION_ABLATION_VERSION,
            }
        )
    return pd.DataFrame(rows)


def no_signal_reproduction_rows(
    condition_df: pd.DataFrame,
    result_df: pd.DataFrame,
    families: Mapping[str, PolicyFamilySpec],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    no_signal = condition_df[condition_df["ablationVariant"] == SIGNAL_REFERENCE_VARIANT]
    metrics = (
        "completed",
        "repairSuccess",
        "homeostaticMaintenance",
        "finalSortednessPercent",
        "coordinationProxy",
        "falseAlarmProxy",
        "signalEnergyProxy",
        "energyProxy",
        "activationCount",
        "swapCount",
        "comparisonCount",
        "blockedMoveAttempts",
        "recoveredCellCount",
        "remainingFrozenCellCount",
        "fatigueEventCount",
        "activeFatiguedCellCount",
        "activeDamagedCellCount",
        "finalValuesJson",
        "sortednessTraceJson",
        "score",
    )
    result_by_condition = {str(row["conditionId"]): row for _, row in result_df.iterrows()}
    for _, condition in no_signal.iterrows():
        condition_id = str(condition["conditionId"])
        observed = result_by_condition.get(condition_id)
        if observed is None:
            rows.append(
                {
                    "conditionId": condition_id,
                    "success": False,
                    "detail": "missing no-signal result row",
                    "communicationAblationVersion": COMMUNICATION_ABLATION_VERSION,
                }
            )
            continue
        baseline = run_no_signal_baseline_condition(families[str(condition["familyId"])], condition)
        mismatches = []
        for metric in metrics:
            left = observed.get(metric)
            right = baseline.get(metric)
            if isinstance(left, float) or isinstance(right, float):
                left_float = float(left) if pd.notna(left) else np.nan
                right_float = float(right) if right is not None else np.nan
                equal = (pd.isna(left_float) and pd.isna(right_float)) or abs(left_float - right_float) <= 1e-12
            else:
                if pd.isna(left) and right is None:
                    equal = True
                else:
                    equal = left == right
            if not equal:
                mismatches.append(metric)
        rows.append(
            {
                "conditionId": condition_id,
                "familyId": condition["familyId"],
                "taskId": condition["taskId"],
                "seedIndex": int(condition["seedIndex"]),
                "success": len(mismatches) == 0,
                "detail": "matched explicit no-signal wrapper baseline" if not mismatches else f"mismatched metrics: {mismatches}",
                "communicationAblationVersion": COMMUNICATION_ABLATION_VERSION,
            }
        )
    return pd.DataFrame(rows)


def validate_communication_outputs(
    condition_df: pd.DataFrame,
    result_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    no_signal_df: pd.DataFrame,
    *,
    expected_evolved_min: int = 1,
    required_family_kinds: set[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    required_family_kinds = required_family_kinds or {"classic", "frontier", "learned", "evolved"}

    def add(check_id: str, success: bool, detail: str) -> None:
        rows.append(
            {
                "checkId": check_id,
                "success": bool(success),
                "detail": detail,
                "communicationAblationVersion": COMMUNICATION_ABLATION_VERSION,
            }
        )

    add("condition_rows_nonempty", len(condition_df) > 0, f"conditionRows={len(condition_df)}")
    add("result_rows_match_conditions", len(result_df) == len(condition_df), f"resultRows={len(result_df)} conditionRows={len(condition_df)}")
    group_sizes = condition_df.groupby("controlledGroupId")["ablationVariant"].nunique()
    add("paired_signal_variant_sets_complete", bool((group_sizes == len(SIGNAL_ABLATION_VARIANTS)).all()), f"groupCount={len(group_sizes)}")
    seed_ok = condition_df.groupby("controlledGroupId")[["schedulerSeed", "tieBreakerSeed"]].nunique().max().max() == 1
    add("paired_scheduler_seeds_identical", bool(seed_ok), "schedulerSeed and tieBreakerSeed are constant within each controlled group")
    schedule_ok = condition_df.groupby("controlledGroupId")["taskSpecJson"].nunique().max() == 1
    add("perturbation_schedules_identical", bool(schedule_ok), "taskSpecJson is constant within each controlled group")
    controlled_ok = condition_df.groupby("controlledGroupId")["controlledConfigHash"].nunique().max() == 1
    add("only_signal_architecture_changes", bool(controlled_ok), "controlledConfigHash is constant within each signal-ablation group")
    variants = set(condition_df["ablationVariant"])
    add("planned_signal_architectures_present", set(SIGNAL_ABLATION_VARIANTS).issubset(variants), f"variants={sorted(variants)}")
    task_families = set(condition_df["taskFamily"])
    add("repair_fatigue_homeostasis_tasks_present", {"repairable", "fatigue", "homeostasis"}.issubset(task_families), f"taskFamilies={sorted(task_families)}")
    range_noise_ok = condition_df[["signalRange", "noiseStd", "diffusionRate", "decay", "emissionScale", "signalRandomSeed"]].notna().all().all()
    add("range_noise_parameters_logged", bool(range_noise_ok), "signal range, noise, diffusion, decay, emission, and random seed columns are populated")
    no_signal_ok = len(no_signal_df) > 0 and bool(no_signal_df["success"].all())
    add("no_signal_baseline_reproduction", no_signal_ok, f"noSignalRows={len(no_signal_df)} passed={int(no_signal_df['success'].sum()) if len(no_signal_df) else 0}")
    family_kinds = set(condition_df["familyKind"])
    add("required_policy_families_present", required_family_kinds.issubset(family_kinds), f"familyKinds={sorted(family_kinds)}")
    evolved_count = condition_df.loc[condition_df["familyKind"] == "evolved", "familyId"].nunique()
    add("s08_evolved_candidates_replayable", evolved_count >= expected_evolved_min, f"evolvedFamilyCount={evolved_count}")
    add("deltas_quantified", len(delta_df) > 0, f"deltaRows={len(delta_df)}")
    transfer_rows = result_df[result_df["transfer"].astype(bool)]
    add("transfer_tasks_present", len(transfer_rows) > 0, f"transferResultRows={len(transfer_rows)}")
    metric_ok = result_df[["score", "finalSortednessPercent", "energyProxy", "coordinationProxy", "falseAlarmProxy", "signalEnergyProxy"]].notna().all().all()
    add("core_communication_metrics_present", bool(metric_ok), "score, sortedness, energy, coordination, false-alarm, and signal-energy metrics are populated")
    return pd.DataFrame(rows)


def replay_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(stable_json(list(rows)).encode("utf-8")).hexdigest()
