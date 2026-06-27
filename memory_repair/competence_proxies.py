"""Held-out competence-proxy measurements for E04 S11."""

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

from .fatigue import FATIGUE_DAMAGE_VERSION, FatigueDamageConfig
from .homeostasis import HOMEOSTASIS_BENCHMARK_VERSION, normalized_sortedness_record, stable_hash, stable_json
from .learning import (
    LOCAL_LEARNING_VERSION,
    LearningEventSimulator,
    LocalLearningConfig,
    LocalLearningPolicyWrapper,
    apply_homeostatic_event_to_simulator,
)
from .memory import MEMORY_REPAIR_VERSION, MemoryConfig, MemoryPolicyWrapper
from .repair import REPAIR_REPAIR_VERSION, RepairRuleConfig
from .signals import SIGNAL_REPAIR_VERSION, SignalConfig, serialize_signal_fields
from .training_constraints import (
    TRAINING_CONSTRAINT_VERSION,
    audit_policy_spec_for_oracle_access,
    local_only_training_protocol,
)


COMPETENCE_PROXY_VERSION = "e04_s11_competence_proxies.v1"
PROXY_SCOPE_NOTE = (
    "Direct computational proxy only; not a direct measure of biological intelligence, "
    "cognition, agency, or sentience."
)
COMPETENCE_AXES = (
    "goal_attainment_proxy",
    "multiple_routes_to_goal_proxy",
    "barrier_circumvention_proxy",
    "recovery_after_perturbation_proxy",
    "transfer_to_larger_arrays_proxy",
    "graceful_degradation_proxy",
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


def _bounded_unit(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return float(max(0.0, min(1.0, value)))


@dataclass(frozen=True)
class CompetenceProxyTask:
    task_id: str
    task_family: str
    task_panel: str
    competence_axis: str
    initial_values: tuple[int, ...]
    horizon: int
    frozen_positions: tuple[int, ...] = ()
    sortedness_threshold_percent: float = 95.0
    perturbation_schedule: tuple[Mapping[str, Any], ...] = ()
    activations_per_tick: int = 14
    fatigue_config: Mapping[str, Any] | None = None
    transfer: bool = False
    heldout: bool = True
    selection_used: bool = False
    perturbation_level: str = "none"
    perturbation_severity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        schedule = [dict(event) for event in self.perturbation_schedule]
        fatigue = dict(self.fatigue_config or FatigueDamageConfig("none").to_dict())
        return {
            "taskId": self.task_id,
            "taskFamily": self.task_family,
            "taskPanel": self.task_panel,
            "competenceAxis": self.competence_axis,
            "initialValues": list(self.initial_values),
            "initialLength": int(len(self.initial_values)),
            "horizon": int(self.horizon),
            "frozenPositions": list(self.frozen_positions),
            "sortednessThresholdPercent": float(self.sortedness_threshold_percent),
            "perturbationSchedule": schedule,
            "activationsPerTick": int(self.activations_per_tick),
            "fatigueConfig": fatigue,
            "transfer": bool(self.transfer),
            "heldOut": bool(self.heldout),
            "selectionUsed": bool(self.selection_used),
            "perturbationLevel": self.perturbation_level,
            "perturbationSeverity": float(self.perturbation_severity),
            "scheduleHash": stable_hash(schedule),
            "claimBoundary": PROXY_SCOPE_NOTE,
            "competenceProxyVersion": COMPETENCE_PROXY_VERSION,
        }


@dataclass(frozen=True)
class CompetencePolicySpec:
    policy_id: str
    policy_group: str
    family_kind: str
    source_step: str
    policy_label: str
    base_policy: Any
    base_policy_spec: Mapping[str, Any]
    learning_config: Mapping[str, Any] | None = None
    memory_config: Mapping[str, Any] | None = None
    signal_config: Mapping[str, Any] | None = None
    repair_config: Mapping[str, Any] | None = None
    source_candidate_id: str | None = None
    replayable: bool = True
    replay_blocker: str | None = None
    oracle_access_allowed: bool = False
    policy_audit_success: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "policyId": self.policy_id,
            "policyGroup": self.policy_group,
            "familyKind": self.family_kind,
            "sourceStep": self.source_step,
            "policyLabel": self.policy_label,
            "basePolicySpec": dict(self.base_policy_spec),
            "learningConfig": None if self.learning_config is None else dict(self.learning_config),
            "memoryConfig": self.memory_config or MemoryConfig("no_memory").to_dict(),
            "signalConfig": self.signal_config or SignalConfig("no_signal").to_dict(),
            "repairConfig": self.repair_config or RepairRuleConfig("nudge_count", nudge_threshold=2, approach_direction="either").to_dict(),
            "sourceCandidateId": self.source_candidate_id,
            "replayable": bool(self.replayable),
            "replayBlocker": self.replay_blocker,
            "oracleAccessAllowed": bool(self.oracle_access_allowed),
            "policyAuditSuccess": bool(self.policy_audit_success),
            "claimBoundary": PROXY_SCOPE_NOTE,
            "competenceProxyVersion": COMPETENCE_PROXY_VERSION,
            "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
        }


def competence_metric_definitions() -> list[dict[str, Any]]:
    return [
        {
            "metricId": "goal_attainment_proxy",
            "displayName": "Goal attainment proxy",
            "aggregationLevel": "condition",
            "directComputationalMetric": (
                "0.5 * final sortedness fraction plus 0.5 * task completion evidence; "
                "homeostasis completion evidence is time-in-range."
            ),
            "normalization": "Bounded to [0, 1]; higher means stronger direct task completion evidence.",
            "heldOutRequirement": "Measured only on S11 held-out perturbations and array sizes.",
            "claimBoundary": PROXY_SCOPE_NOTE,
        },
        {
            "metricId": "multiple_routes_to_goal_proxy",
            "displayName": "Multiple routes to same goal proxy",
            "aggregationLevel": "policy_profile",
            "directComputationalMetric": (
                "Unique successful value-state transition signatures divided by successful held-out route-task runs."
            ),
            "normalization": "Bounded to [0, 1]; zero when no held-out route run reaches the goal.",
            "heldOutRequirement": "Uses held-out route task seeds not used for S08/S09/S10 selection.",
            "claimBoundary": PROXY_SCOPE_NOTE,
        },
        {
            "metricId": "barrier_circumvention_proxy",
            "displayName": "Barrier circumvention proxy",
            "aggregationLevel": "condition",
            "directComputationalMetric": (
                "Repairable Frozen Cell completion evidence combining final sortedness and remaining-frozen penalty."
            ),
            "normalization": "Bounded to [0, 1]; one requires sortedness above threshold and no remaining frozen cells.",
            "heldOutRequirement": "Uses held-out barrier placements and larger arrays than S10.",
            "claimBoundary": PROXY_SCOPE_NOTE,
        },
        {
            "metricId": "recovery_after_perturbation_proxy",
            "displayName": "Recovery after perturbation proxy",
            "aggregationLevel": "condition",
            "directComputationalMetric": (
                "Mean bounded inverse latency from each perturbation tick to restored sortedness-in-range."
            ),
            "normalization": "Bounded to [0, 1]; one means immediate post-perturbation recovery for all logged events.",
            "heldOutRequirement": "Uses held-out swap, freeze, recover, damage, insert, and delete schedules.",
            "claimBoundary": PROXY_SCOPE_NOTE,
        },
        {
            "metricId": "transfer_to_larger_arrays_proxy",
            "displayName": "Transfer to larger arrays proxy",
            "aggregationLevel": "condition",
            "directComputationalMetric": "Goal-attainment evidence on held-out arrays larger than prior E04 ablation panels.",
            "normalization": "Bounded to [0, 1]; higher means stronger held-out larger-array completion evidence.",
            "heldOutRequirement": "Initial length is greater than the S10 maximum initial length of 6.",
            "claimBoundary": PROXY_SCOPE_NOTE,
        },
        {
            "metricId": "graceful_degradation_proxy",
            "displayName": "Graceful degradation proxy",
            "aggregationLevel": "policy_profile",
            "directComputationalMetric": (
                "Mean degradation-task goal evidence adjusted by the normalized drop from low to high perturbation severity."
            ),
            "normalization": "Bounded to [0, 1]; higher means less direct performance loss under stronger damage/fatigue.",
            "heldOutRequirement": "Uses paired low, medium, and high held-out degradation tasks.",
            "claimBoundary": PROXY_SCOPE_NOTE,
        },
        {
            "metricId": "energy_efficiency_proxy",
            "displayName": "Energy efficiency proxy",
            "aggregationLevel": "condition",
            "directComputationalMetric": (
                "One minus bounded activation, swap, comparison, blocked-move, fatigue-event, and signal-energy cost."
            ),
            "normalization": "Bounded to [0, 1] with horizon-scaled denominator.",
            "heldOutRequirement": "Reported for every S11 held-out run as a secondary computational proxy.",
            "claimBoundary": PROXY_SCOPE_NOTE,
        },
        {
            "metricId": "overall_competence_proxy",
            "displayName": "Overall competence proxy",
            "aggregationLevel": "policy_profile",
            "directComputationalMetric": (
                "Mean of available S11 proxy axes; it is a summary score for this computational panel only."
            ),
            "normalization": "Bounded to [0, 1]; not a biological or cognitive intelligence score.",
            "heldOutRequirement": "Aggregates only S11 held-out proxy metrics.",
            "claimBoundary": PROXY_SCOPE_NOTE,
        },
    ]


def build_s11_tasks() -> tuple[CompetenceProxyTask, ...]:
    recovery_schedule = (
        {"eventIndex": 0, "tick": 1, "type": "swap", "leftIndex": 2, "rightIndex": 3, "source": "s11_heldout_swap"},
        {"eventIndex": 1, "tick": 2, "type": "freeze", "cellId": 5, "source": "s11_heldout_freeze"},
        {"eventIndex": 2, "tick": 3, "type": "damage", "cellId": 2, "durationTicks": 2, "source": "s11_heldout_damage"},
        {"eventIndex": 3, "tick": 4, "type": "insert", "position": 0, "value": 99, "source": "s11_heldout_insert"},
        {"eventIndex": 4, "tick": 6, "type": "delete", "position": 0, "source": "s11_heldout_delete"},
        {"eventIndex": 5, "tick": 7, "type": "recover", "cellId": 5, "source": "s11_heldout_recover"},
        {"eventIndex": 6, "tick": 9, "type": "swap", "leftIndex": 4, "rightIndex": 5, "source": "s11_heldout_swap_late"},
    )
    degradation_base = (10, 1, 9, 2, 8, 3, 7, 4, 6, 5)
    return (
        CompetenceProxyTask(
            task_id="heldout_route_same_goal_n8",
            task_family="sorting",
            task_panel="s11_heldout_route",
            competence_axis="multiple_routes_to_goal_proxy",
            initial_values=(8, 1, 6, 2, 7, 3, 5, 4),
            horizon=96,
        ),
        CompetenceProxyTask(
            task_id="heldout_barrier_circumvention_n10",
            task_family="repairable",
            task_panel="s11_heldout_barrier",
            competence_axis="barrier_circumvention_proxy",
            initial_values=(8, 3, 6, 1, 10, 2, 9, 4, 7, 5),
            frozen_positions=(4, 5),
            horizon=140,
        ),
        CompetenceProxyTask(
            task_id="heldout_recovery_mixed_events_n8",
            task_family="homeostasis",
            task_panel="s11_heldout_recovery",
            competence_axis="recovery_after_perturbation_proxy",
            initial_values=(1, 2, 3, 4, 5, 6, 7, 8),
            horizon=12,
            perturbation_schedule=recovery_schedule,
            activations_per_tick=16,
            sortedness_threshold_percent=95.0,
            fatigue_config=FatigueDamageConfig(
                "stochastic_damage",
                damage_probability=0.04,
                damage_cooldown_activations=2,
                random_seed=11301,
            ).to_dict(),
        ),
        CompetenceProxyTask(
            task_id="heldout_transfer_larger_n12",
            task_family="transfer",
            task_panel="s11_heldout_transfer",
            competence_axis="transfer_to_larger_arrays_proxy",
            initial_values=(12, 1, 11, 2, 10, 3, 9, 4, 8, 5, 7, 6),
            horizon=180,
            transfer=True,
        ),
        CompetenceProxyTask(
            task_id="heldout_degradation_low_n10",
            task_family="degradation",
            task_panel="s11_heldout_degradation",
            competence_axis="graceful_degradation_proxy",
            initial_values=degradation_base,
            horizon=120,
            perturbation_level="low",
            perturbation_severity=0.0,
            fatigue_config=FatigueDamageConfig(
                "combined",
                movement_threshold=8,
                failed_swap_threshold=6,
                frustration_threshold=6,
                recovery_activations=2,
                damage_probability=0.0,
                random_seed=11401,
            ).to_dict(),
        ),
        CompetenceProxyTask(
            task_id="heldout_degradation_medium_n10",
            task_family="degradation",
            task_panel="s11_heldout_degradation",
            competence_axis="graceful_degradation_proxy",
            initial_values=degradation_base,
            horizon=120,
            perturbation_level="medium",
            perturbation_severity=0.5,
            fatigue_config=FatigueDamageConfig(
                "combined",
                movement_threshold=4,
                failed_swap_threshold=3,
                frustration_threshold=3,
                recovery_activations=2,
                damage_probability=0.04,
                damage_cooldown_activations=2,
                random_seed=11402,
            ).to_dict(),
        ),
        CompetenceProxyTask(
            task_id="heldout_degradation_high_n10",
            task_family="degradation",
            task_panel="s11_heldout_degradation",
            competence_axis="graceful_degradation_proxy",
            initial_values=degradation_base,
            horizon=120,
            perturbation_level="high",
            perturbation_severity=1.0,
            fatigue_config=FatigueDamageConfig(
                "combined",
                movement_threshold=3,
                failed_swap_threshold=2,
                frustration_threshold=2,
                recovery_activations=3,
                damage_probability=0.10,
                damage_cooldown_activations=3,
                random_seed=11403,
            ).to_dict(),
        ),
    )


def _policy_audit(policy: Any) -> dict[str, Any]:
    return audit_policy_spec_for_oracle_access(policy, local_only_training_protocol())


def baseline_competence_policies() -> tuple[CompetencePolicySpec, ...]:
    nudge = RepairRuleConfig("nudge_count", nudge_threshold=2, approach_direction="either").to_dict()
    no_signal = SignalConfig("no_signal").to_dict()
    no_memory = MemoryConfig("no_memory").to_dict()
    specs = []
    for algotype in ("bubble", "insertion"):
        policy_spec = {
            "policy_id": f"classic_{algotype}",
            "family": "classic",
            "algotype": algotype,
            "parameters": {},
        }
        audit = _policy_audit(policy_spec)
        specs.append(
            CompetencePolicySpec(
                policy_id=f"baseline_classic_{algotype}_open_loop",
                policy_group="baseline",
                family_kind="classic",
                source_step="E03_S01",
                policy_label=f"classic_{algotype}_open_loop",
                base_policy=algotype,
                base_policy_spec=policy_spec,
                memory_config=no_memory,
                signal_config=no_signal,
                repair_config=nudge,
                policy_audit_success=bool(audit["success"]),
                replayable=bool(audit["success"]),
                replay_blocker=None if audit["success"] else compact_json(audit.get("violations", [])),
            )
        )
    return tuple(specs)


def local_memory_signal_competence_policy() -> CompetencePolicySpec:
    learning = LocalLearningConfig("local_adaptive", learning_rate=0.35, random_seed=11601).to_dict()
    memory = MemoryConfig("bounded_counter", counter_max=5, neighbor_history=4).to_dict()
    base = LocalLearningPolicyWrapper("bubble", LocalLearningConfig.from_spec(learning))
    policy = MemoryPolicyWrapper(base, MemoryConfig.from_spec(memory))
    signal = SignalConfig(
        "diffusive",
        signal_range=1,
        diffusion_rate=0.25,
        decay=0.05,
        emission_scale=1.0,
        random_seed=11601,
    ).to_dict()
    repair = RepairRuleConfig("signal_threshold", signal_threshold=0.75, approach_direction="either").to_dict()
    audit = _policy_audit(policy)
    return CompetencePolicySpec(
        policy_id="enhanced_local_memory_signal_adaptive",
        policy_group="enhanced",
        family_kind="learned",
        source_step="S06_S10",
        policy_label="local adaptive bubble with bounded memory and diffusive signal",
        base_policy=policy,
        base_policy_spec=policy.to_spec().to_dict(),
        learning_config=learning,
        memory_config=memory,
        signal_config=signal,
        repair_config=repair,
        policy_audit_success=bool(audit["success"]),
        replayable=bool(audit["success"]),
        replay_blocker=None if audit["success"] else compact_json(audit.get("violations", [])),
    )


def competence_policy_from_s08_candidate(row: Mapping[str, Any]) -> CompetencePolicySpec:
    learning = json.loads(str(row["learningConfigJson"]))
    memory = json.loads(str(row["memoryConfigJson"]))
    signal = json.loads(str(row["signalConfigJson"]))
    repair = json.loads(str(row["repairConfigJson"]))
    base = LocalLearningPolicyWrapper("bubble", LocalLearningConfig.from_spec(learning))
    policy = MemoryPolicyWrapper(base, MemoryConfig.from_spec(memory))
    audit = _policy_audit(policy)
    candidate_ok = bool(row.get("auditSuccess", True)) and bool(row.get("trainingCertificationSuccess", True))
    replayable = bool(audit["success"] and candidate_ok)
    return CompetencePolicySpec(
        policy_id=f"enhanced_s08_rank_{int(row.get('rank', -1))}_{row['candidateId']}",
        policy_group="enhanced",
        family_kind="evolved",
        source_step="S08",
        policy_label=f"s08 evolved local memory-signal rank {int(row.get('rank', -1))}",
        base_policy=policy,
        base_policy_spec=policy.to_spec().to_dict(),
        learning_config=learning,
        memory_config=memory,
        signal_config=signal,
        repair_config=repair,
        source_candidate_id=str(row["candidateId"]),
        replayable=replayable,
        replay_blocker=None if replayable else compact_json(audit.get("violations", [])),
        policy_audit_success=bool(audit["success"]),
    )


def policy_for_competence_condition(policy: CompetencePolicySpec) -> Any:
    if policy.family_kind in {"learned", "evolved"} and policy.learning_config is not None:
        base = LocalLearningPolicyWrapper("bubble", LocalLearningConfig.from_spec(policy.learning_config))
        memory = MemoryConfig.from_spec(policy.memory_config or "no_memory")
        return MemoryPolicyWrapper(base, memory) if memory.variant != "no_memory" else base
    return policy.base_policy


def condition_rows_for_competence_policy(
    policy: CompetencePolicySpec,
    tasks: Sequence[CompetenceProxyTask],
    *,
    seed_count: int = 4,
    seed_base: int = 12100,
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
                    "conditionId": stable_id("s11_condition", [policy.policy_id, task.task_id, seed_index]),
                    "policyId": policy.policy_id,
                    "policyGroup": policy.policy_group,
                    "familyKind": policy.family_kind,
                    "sourceStep": policy.source_step,
                    "sourceCandidateId": policy.source_candidate_id,
                    "taskId": task.task_id,
                    "taskFamily": task.task_family,
                    "taskPanel": task.task_panel,
                    "competenceAxis": task.competence_axis,
                    "transfer": bool(task.transfer),
                    "heldOut": bool(task.heldout),
                    "selectionUsed": bool(task.selection_used),
                    "perturbationLevel": task.perturbation_level,
                    "perturbationSeverity": float(task.perturbation_severity),
                    "initialLength": int(len(task.initial_values)),
                    "seedIndex": int(seed_index),
                    "schedulerSeed": scheduler_seed,
                    "tieBreakerSeed": tie_seed,
                    "signalRandomSeed": signal_seed,
                    "oracleAccessAllowed": bool(policy.oracle_access_allowed),
                    "policyAuditSuccess": bool(policy.policy_audit_success),
                    "signalEngineVariant": signal_config.variant,
                    "signalRange": int(signal_config.signal_range),
                    "diffusionRate": float(signal_config.diffusion_rate),
                    "decay": float(signal_config.decay),
                    "noiseStd": float(signal_config.noise_std),
                    "emissionScale": float(signal_config.emission_scale),
                    "taskSpecJson": compact_json(task_payload),
                    "signalConfigJson": compact_json(signal_config.to_dict()),
                    "repairConfigJson": compact_json(base_repair.to_dict()),
                    "fatigueConfigJson": compact_json(fatigue.to_dict()),
                    "memoryConfigJson": compact_json(policy.memory_config or MemoryConfig("no_memory").to_dict()),
                    "learningConfigJson": compact_json(policy.learning_config) if policy.learning_config is not None else None,
                    "basePolicySpecJson": compact_json(policy.base_policy_spec),
                    "controlledConfigHash": config_hash(controlled),
                    "claimBoundary": PROXY_SCOPE_NOTE,
                    "competenceProxyVersion": COMPETENCE_PROXY_VERSION,
                    "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
                }
            )
    return rows


def _active_fatigue_counts(simulator: LearningEventSimulator) -> tuple[int, int]:
    fatigued = 0
    damaged = 0
    for state in getattr(simulator, "fatigue_states", {}).values():
        fatigued += int(int(state.get("fatigue_cooldown_remaining", 0)) > 0)
        damaged += int(int(state.get("damage_cooldown_remaining", 0)) > 0)
    return fatigued, damaged


def _signal_energy_from_fields(fields: Mapping[str, np.ndarray]) -> float:
    return float(sum(float(np.asarray(values, dtype=float).sum()) for values in fields.values()))


def _signal_trace_proxies(simulator: LearningEventSimulator) -> dict[str, float]:
    cumulative_energy = 0.0
    false_alarm_energy = 0.0
    for row in simulator.trace_rows:
        fields = {}
        payload = row.get("signal_fields_json")
        if isinstance(payload, str):
            try:
                fields = json.loads(payload)
            except json.JSONDecodeError:
                fields = {}
        row_energy = 0.0
        for values in fields.values():
            if isinstance(values, list):
                row_energy += sum(float(value) for value in values)
        cumulative_energy += row_energy
        repair_payload = row.get("repair_states_json")
        frozen_present = False
        if isinstance(repair_payload, str):
            try:
                states = json.loads(repair_payload)
                frozen_present = any(bool(state.get("frozen", False)) for state in states if isinstance(state, Mapping))
            except json.JSONDecodeError:
                frozen_present = False
        if row_energy > 0.0 and not frozen_present:
            false_alarm_energy += row_energy
    return {
        "cumulativeSignalEnergy": float(cumulative_energy),
        "falseAlarmSignalEnergy": float(false_alarm_energy),
    }


def _trajectory_hash(states: Sequence[Sequence[int]]) -> str:
    return hashlib.sha256(compact_json([[int(value) for value in state] for state in states]).encode("utf-8")).hexdigest()


def _transition_hash(states: Sequence[Sequence[int]]) -> str:
    transitions = []
    previous = None
    for state in states:
        current = tuple(int(value) for value in state)
        if previous is not None and current != previous:
            transitions.append([list(previous), list(current)])
        previous = current
    return hashlib.sha256(compact_json(transitions).encode("utf-8")).hexdigest()


def _recovery_proxy_from_ticks(tick_rows: Sequence[Mapping[str, Any]], horizon: int) -> float | None:
    event_ticks = []
    for row in tick_rows:
        events = row.get("eventsJson", "[]")
        if isinstance(events, str):
            try:
                payload = json.loads(events)
            except json.JSONDecodeError:
                payload = []
        else:
            payload = events
        if payload:
            event_ticks.append(int(row["tick"]))
    if not event_ticks:
        return None
    scores: list[float] = []
    for event_tick in event_ticks:
        available = [row for row in tick_rows if int(row["tick"]) >= event_tick]
        latency = None
        for row in available:
            if bool(row.get("postInRange", False)):
                latency = int(row["tick"]) - event_tick
                break
        span = max(1, int(horizon) - event_tick)
        if latency is None:
            scores.append(0.0)
        else:
            scores.append(max(0.0, 1.0 - float(latency) / float(span)))
    return float(np.mean(scores)) if scores else None


def _energy_efficiency_proxy(energy_proxy: float, *, horizon: int, initial_length: int) -> float:
    denominator = max(1.0, float(horizon) * max(1.0, float(initial_length)) * 2.0)
    return float(max(0.0, min(1.0, 1.0 - float(energy_proxy) / denominator)))


def run_competence_condition(policy_spec: CompetencePolicySpec, condition: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task = json.loads(str(condition["taskSpecJson"]))
    policy = policy_for_competence_condition(policy_spec)
    signal_config = SignalConfig.from_spec(json.loads(str(condition["signalConfigJson"])))
    repair_config = RepairRuleConfig.from_spec(json.loads(str(condition["repairConfigJson"])))
    fatigue_config = FatigueDamageConfig.from_spec(json.loads(str(condition["fatigueConfigJson"])))
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
        implementation="e04_s11_competence_proxy_cpu_reference",
        research_step_id="S11",
    )
    task_family = str(task["taskFamily"])
    threshold = float(task["sortednessThresholdPercent"])
    initial_length = int(task["initialLength"])
    value_trace: list[list[int]] = [sim.current_values()]
    tick_rows: list[dict[str, Any]] = []

    if task_family == "homeostasis":
        for tick in range(int(task["horizon"])):
            events = [event for event in task.get("perturbationSchedule", ()) if int(event["tick"]) == tick]
            for event in events:
                apply_homeostatic_event_to_simulator(sim, event)
                value_trace.append(sim.current_values())
            pre = normalized_sortedness_record(sim.current_values(), initial_length=initial_length)
            for _ in range(int(task.get("activationsPerTick", 14))):
                sim.step()
                value_trace.append(sim.current_values())
            post = normalized_sortedness_record(sim.current_values(), initial_length=initial_length)
            tick_rows.append(
                {
                    "conditionId": condition["conditionId"],
                    "policyId": condition["policyId"],
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
                    "claimBoundary": PROXY_SCOPE_NOTE,
                    "competenceProxyVersion": COMPETENCE_PROXY_VERSION,
                    "homeostasisBenchmarkVersion": HOMEOSTASIS_BENCHMARK_VERSION,
                }
            )
    else:
        fixed_horizon = task_family == "degradation"
        for _ in range(int(task["horizon"])):
            if not fixed_horizon and sim.is_sorted() and not sim.current_frozen_positions():
                break
            sim.step()
            value_trace.append(sim.current_values())

    sortedness_series = [float(sortedness_percent(state)) for state in value_trace]
    dg = delayed_gratification_from_sortedness(sortedness_series)
    final_values = sim.current_values()
    final_sortedness = float(sortedness_percent(final_values))
    if task_family == "homeostasis" and tick_rows:
        time_in_range = float(sum(row["postInRange"] for row in tick_rows) / len(tick_rows))
        completed = bool(time_in_range >= 0.75 and final_sortedness >= threshold)
    else:
        time_in_range = None
        completed = bool(final_sortedness >= threshold and not sim.current_frozen_positions())
    repair_success = (
        bool(final_sortedness >= threshold and not sim.current_frozen_positions())
        if task_family in {"repairable", "fatigue"}
        else None
    )
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
    completion_evidence = time_in_range if time_in_range is not None else (1.0 if completed else 0.0)
    goal_attainment_proxy = _bounded_unit(0.5 * (final_sortedness / 100.0) + 0.5 * float(completion_evidence))
    initial_frozen_count = len(task.get("frozenPositions", ()))
    remaining_frozen = len(sim.current_frozen_positions())
    if task_family == "repairable":
        remaining_penalty = remaining_frozen / max(1, initial_frozen_count)
        barrier_proxy = _bounded_unit((final_sortedness / 100.0) * (1.0 - 0.5 * remaining_penalty) + (0.25 if completed else 0.0))
    else:
        barrier_proxy = None
    recovery_proxy = _recovery_proxy_from_ticks(tick_rows, int(task["horizon"])) if task_family == "homeostasis" else None
    transfer_proxy = _bounded_unit(goal_attainment_proxy) if bool(task.get("transfer", False)) or task_family == "transfer" else None
    degradation_performance_proxy = _bounded_unit(goal_attainment_proxy) if task_family == "degradation" else None
    energy_efficiency = _energy_efficiency_proxy(energy_proxy, horizon=int(task["horizon"]), initial_length=initial_length)
    core_values = [
        goal_attainment_proxy,
        barrier_proxy,
        recovery_proxy,
        transfer_proxy,
        degradation_performance_proxy,
        energy_efficiency,
    ]
    composite = float(np.mean([value for value in core_values if value is not None]))

    route_state_hash = _trajectory_hash(value_trace)
    route_transition_hash = _transition_hash(value_trace)
    row = {
        **{key: condition[key] for key in condition.keys()},
        "completed": bool(completed),
        "repairSuccess": repair_success,
        "homeostaticMaintenance": time_in_range,
        "finalSortednessPercent": final_sortedness,
        "goalAttainmentProxy": goal_attainment_proxy,
        "barrierCircumventionProxy": barrier_proxy,
        "recoveryAfterPerturbationProxy": recovery_proxy,
        "transferToLargerArraysProxy": transfer_proxy,
        "degradationPerformanceProxy": degradation_performance_proxy,
        "energyEfficiencyProxy": energy_efficiency,
        "conditionCompetenceCompositeProxy": composite,
        "delayedGratification": float(dg["delayedGratification"]),
        "delayedGratificationEventCount": int(dg["dgEventCount"]),
        "falseAlarmProxy": float(signal_proxies["falseAlarmSignalEnergy"]),
        "signalEnergyProxy": float(signal_proxies["cumulativeSignalEnergy"]),
        "finalSignalEnergy": final_signal_energy,
        "energyProxy": energy_proxy,
        "activationCount": int(sim.activation_count),
        "swapCount": int(sim.swap_count),
        "comparisonCount": int(sim.comparison_count),
        "blockedMoveAttempts": int(sim.blocked_move_attempts),
        "recoveredCellCount": int(len(sim.recovery_events)),
        "remainingFrozenCellCount": int(remaining_frozen),
        "initialFrozenCellCount": int(initial_frozen_count),
        "fatigueEventCount": int(len(getattr(sim, "fatigue_events", []))),
        "activeFatiguedCellCount": int(active_fatigued),
        "activeDamagedCellCount": int(active_damaged),
        "initialValuesJson": compact_json(task["initialValues"]),
        "finalValuesJson": compact_json(final_values),
        "finalSignalFieldsJson": compact_json(serialize_signal_fields(sim.signal_fields)),
        "sortednessTraceJson": compact_json(sortedness_series),
        "stateTraceLength": int(len(value_trace)),
        "uniqueStateCount": int(len({tuple(state) for state in value_trace})),
        "routeStateTraceHash": route_state_hash,
        "routeTransitionHash": route_transition_hash,
        "stateTraceJson": compact_json(value_trace),
        "claimBoundary": PROXY_SCOPE_NOTE,
        "memoryRepairVersion": MEMORY_REPAIR_VERSION,
        "signalRepairVersion": SIGNAL_REPAIR_VERSION,
        "repairRepairVersion": REPAIR_REPAIR_VERSION,
        "fatigueDamageVersion": FATIGUE_DAMAGE_VERSION,
        "localLearningVersion": LOCAL_LEARNING_VERSION,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
        "competenceProxyVersion": COMPETENCE_PROXY_VERSION,
    }
    return _json_ready(row), _json_ready(tick_rows)


def _mean_or_none(values: pd.Series) -> float | None:
    values = values.dropna()
    return None if values.empty else float(values.mean())


def _route_diversity_proxy(group: pd.DataFrame) -> float | None:
    route = group[group["competenceAxis"] == "multiple_routes_to_goal_proxy"]
    if route.empty:
        return None
    successful = route[route["goalAttainmentProxy"] >= 0.95]
    if successful.empty:
        return 0.0
    unique = successful["routeTransitionHash"].nunique(dropna=True)
    return float(min(1.0, unique / max(1, len(successful))))


def _graceful_degradation_proxy(group: pd.DataFrame) -> float | None:
    degradation = group[group["competenceAxis"] == "graceful_degradation_proxy"]
    if degradation.empty:
        return None
    means = degradation.groupby("perturbationLevel", dropna=False)["degradationPerformanceProxy"].mean().to_dict()
    ordered = [means[level] for level in ("low", "medium", "high") if level in means and pd.notna(means[level])]
    if not ordered:
        return None
    low = float(means.get("low", ordered[0]))
    high = float(means.get("high", ordered[-1]))
    drop = max(0.0, low - high) / max(1e-9, low)
    area = float(np.mean(ordered))
    return _bounded_unit(0.5 * area + 0.5 * (1.0 - min(1.0, drop)))


def summarize_competence_profiles(result_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if result_df.empty:
        return pd.DataFrame()
    for keys, group in result_df.groupby(["policyId", "policyGroup", "familyKind", "sourceStep"], dropna=False):
        route = _route_diversity_proxy(group)
        graceful = _graceful_degradation_proxy(group)
        goal = _mean_or_none(group["goalAttainmentProxy"])
        barrier = _mean_or_none(group["barrierCircumventionProxy"])
        recovery = _mean_or_none(group["recoveryAfterPerturbationProxy"])
        transfer = _mean_or_none(group["transferToLargerArraysProxy"])
        energy = _mean_or_none(group["energyEfficiencyProxy"])
        axes = {
            "goalAttainmentProxyMean": goal,
            "multipleRoutesToGoalProxy": route,
            "barrierCircumventionProxyMean": barrier,
            "recoveryAfterPerturbationProxyMean": recovery,
            "transferToLargerArraysProxyMean": transfer,
            "gracefulDegradationProxy": graceful,
            "energyEfficiencyProxyMean": energy,
        }
        overall_values = [value for value in axes.values() if value is not None and math.isfinite(float(value))]
        overall = float(np.mean(overall_values)) if overall_values else None
        rows.append(
            _json_ready(
                {
                    "policyId": keys[0],
                    "policyGroup": keys[1],
                    "familyKind": keys[2],
                    "sourceStep": keys[3],
                    "conditionCount": int(len(group)),
                    "taskCount": int(group["taskId"].nunique()),
                    "heldOutConditionCount": int(group["heldOut"].astype(bool).sum()),
                    "completedRunCount": int(group["completed"].astype(bool).sum()),
                    "meanFinalSortednessPercent": float(group["finalSortednessPercent"].mean()),
                    "meanEnergyProxy": float(group["energyProxy"].mean()),
                    "meanDelayedGratification": float(group["delayedGratification"].mean()),
                    **axes,
                    "overallCompetenceProxy": overall,
                    "proxyAxisCount": int(len(overall_values)),
                    "claimBoundary": PROXY_SCOPE_NOTE,
                    "competenceProxyVersion": COMPETENCE_PROXY_VERSION,
                }
            )
        )
    return pd.DataFrame(rows)


def competence_metric_table(result_df: pd.DataFrame, profile_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    condition_metrics = {
        "goal_attainment_proxy": "goalAttainmentProxy",
        "barrier_circumvention_proxy": "barrierCircumventionProxy",
        "recovery_after_perturbation_proxy": "recoveryAfterPerturbationProxy",
        "transfer_to_larger_arrays_proxy": "transferToLargerArraysProxy",
        "energy_efficiency_proxy": "energyEfficiencyProxy",
    }
    for _, row in result_df.iterrows():
        for metric_id, column in condition_metrics.items():
            value = row.get(column)
            if pd.isna(value):
                continue
            rows.append(
                {
                    "recordId": row["conditionId"],
                    "aggregationLevel": "condition",
                    "metricId": metric_id,
                    "metricValue": float(value),
                    "policyId": row["policyId"],
                    "policyGroup": row["policyGroup"],
                    "taskId": row["taskId"],
                    "taskPanel": row["taskPanel"],
                    "heldOut": bool(row["heldOut"]),
                    "claimBoundary": PROXY_SCOPE_NOTE,
                    "competenceProxyVersion": COMPETENCE_PROXY_VERSION,
                }
            )
    profile_metrics = {
        "goal_attainment_proxy": "goalAttainmentProxyMean",
        "multiple_routes_to_goal_proxy": "multipleRoutesToGoalProxy",
        "barrier_circumvention_proxy": "barrierCircumventionProxyMean",
        "recovery_after_perturbation_proxy": "recoveryAfterPerturbationProxyMean",
        "transfer_to_larger_arrays_proxy": "transferToLargerArraysProxyMean",
        "graceful_degradation_proxy": "gracefulDegradationProxy",
        "energy_efficiency_proxy": "energyEfficiencyProxyMean",
        "overall_competence_proxy": "overallCompetenceProxy",
    }
    for _, row in profile_df.iterrows():
        for metric_id, column in profile_metrics.items():
            value = row.get(column)
            if pd.isna(value):
                continue
            rows.append(
                {
                    "recordId": row["policyId"],
                    "aggregationLevel": "policy_profile",
                    "metricId": metric_id,
                    "metricValue": float(value),
                    "policyId": row["policyId"],
                    "policyGroup": row["policyGroup"],
                    "taskId": None,
                    "taskPanel": None,
                    "heldOut": True,
                    "claimBoundary": PROXY_SCOPE_NOTE,
                    "competenceProxyVersion": COMPETENCE_PROXY_VERSION,
                }
            )
    return pd.DataFrame(_json_ready(rows))


def competence_group_comparison(profile_df: pd.DataFrame) -> pd.DataFrame:
    if profile_df.empty:
        return pd.DataFrame()
    metric_columns = [
        "goalAttainmentProxyMean",
        "multipleRoutesToGoalProxy",
        "barrierCircumventionProxyMean",
        "recoveryAfterPerturbationProxyMean",
        "transferToLargerArraysProxyMean",
        "gracefulDegradationProxy",
        "overallCompetenceProxy",
    ]
    rows = []
    for metric in metric_columns:
        values = profile_df.groupby("policyGroup", dropna=False)[metric].mean().to_dict()
        baseline = values.get("baseline")
        enhanced = values.get("enhanced")
        rows.append(
            {
                "metricId": metric,
                "baselineMean": None if baseline is None or pd.isna(baseline) else float(baseline),
                "enhancedMean": None if enhanced is None or pd.isna(enhanced) else float(enhanced),
                "enhancedMinusBaseline": None
                if baseline is None or enhanced is None or pd.isna(baseline) or pd.isna(enhanced)
                else float(enhanced - baseline),
                "claimBoundary": PROXY_SCOPE_NOTE,
                "competenceProxyVersion": COMPETENCE_PROXY_VERSION,
            }
        )
    return pd.DataFrame(_json_ready(rows))


def validate_competence_outputs(
    condition_df: pd.DataFrame,
    result_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    definition_df: pd.DataFrame,
    *,
    expected_evolved_min: int = 1,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check_id: str, success: bool, detail: str) -> None:
        rows.append(
            {
                "checkId": check_id,
                "success": bool(success),
                "detail": detail,
                "claimBoundary": PROXY_SCOPE_NOTE,
                "competenceProxyVersion": COMPETENCE_PROXY_VERSION,
            }
        )

    add("condition_rows_nonempty", len(condition_df) > 0, f"conditionRows={len(condition_df)}")
    add("result_rows_match_conditions", len(result_df) == len(condition_df), f"resultRows={len(result_df)} conditionRows={len(condition_df)}")
    add("metric_rows_nonempty", len(metric_df) > 0, f"metricRows={len(metric_df)}")
    add("profiles_nonempty", len(profile_df) > 0, f"profileRows={len(profile_df)}")
    add("all_conditions_held_out", bool(condition_df["heldOut"].astype(bool).all()), "all condition rows are marked heldOut=true")
    add("no_selection_or_global_oracle", bool((~condition_df["selectionUsed"].astype(bool)).all() and (~condition_df["oracleAccessAllowed"].astype(bool)).all()), "selectionUsed=false and oracleAccessAllowed=false for every condition")
    add("policy_audits_passed", bool(condition_df["policyAuditSuccess"].astype(bool).all()), "all replayed policy specs pass S07 no-oracle audit")
    groups = set(condition_df["policyGroup"])
    add("baseline_and_enhanced_present", {"baseline", "enhanced"}.issubset(groups), f"policyGroups={sorted(groups)}")
    evolved_count = condition_df.loc[condition_df["familyKind"] == "evolved", "policyId"].nunique()
    add("s08_evolved_candidates_included", evolved_count >= expected_evolved_min, f"evolvedPolicyCount={evolved_count}")
    axes = set(condition_df["competenceAxis"])
    required_task_axes = set(COMPETENCE_AXES) - {"goal_attainment_proxy"}
    add("planned_competence_axes_present", required_task_axes.issubset(axes), f"axes={sorted(axes)}")
    add("heldout_larger_sizes_present", bool((condition_df["initialLength"] > 6).any()), f"maxInitialLength={int(condition_df['initialLength'].max()) if len(condition_df) else 0}")
    bounded = metric_df["metricValue"].dropna().between(0.0, 1.0).all() if len(metric_df) else False
    add("proxy_metric_values_bounded", bool(bounded), "all reported proxy metric values are within [0, 1]")
    definition_metrics = set(definition_df["metricId"])
    observed_metrics = set(metric_df["metricId"])
    add("metric_definitions_cover_observed", observed_metrics.issubset(definition_metrics), f"observedMetrics={sorted(observed_metrics)}")
    definition_text_ok = bool(definition_df["claimBoundary"].astype(str).str.contains("Direct computational proxy only", regex=False).all())
    add("proxy_claim_boundaries_documented", definition_text_ok, "every metric definition carries the proxy-scoped claim boundary")
    direct_definitions_ok = bool(definition_df["directComputationalMetric"].astype(str).str.len().gt(20).all())
    add("direct_computational_definitions_documented", direct_definitions_ok, "all metric definitions include direct computational formulas or measurement definitions")
    route_ok = "multipleRoutesToGoalProxy" in profile_df.columns and profile_df["multipleRoutesToGoalProxy"].notna().any()
    add("route_diversity_quantified", bool(route_ok), "policy profiles include multipleRoutesToGoalProxy")
    graceful_ok = "gracefulDegradationProxy" in profile_df.columns and profile_df["gracefulDegradationProxy"].notna().any()
    add("graceful_degradation_quantified", bool(graceful_ok), "policy profiles include gracefulDegradationProxy")
    return pd.DataFrame(rows)


def competence_replay_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(stable_json(list(rows)).encode("utf-8")).hexdigest()
