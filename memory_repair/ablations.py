"""Memory-capacity ablation helpers for E04 S09."""

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
from morphospace import DSLPolicy, LocalRulePolicy
from morphospace.competence import delayed_gratification_from_sortedness
from morphospace.rule_dsl import parse_rule_program

from .evolution import EVOLUTION_SEARCH_VERSION
from .homeostasis import (
    HOMEOSTASIS_BENCHMARK_VERSION,
    build_homeostatic_benchmark_config,
    normalized_sortedness_record,
    stable_hash,
    stable_json,
)
from .learning import (
    LOCAL_LEARNING_VERSION,
    LearningEventSimulator,
    LocalLearningConfig,
    LocalLearningPolicyWrapper,
    apply_homeostatic_event_to_simulator,
)
from .memory import MEMORY_REPAIR_VERSION, MEMORY_VARIANTS, MemoryConfig, MemoryPolicyWrapper
from .repair import REPAIR_REPAIR_VERSION, RepairRuleConfig
from .signals import SIGNAL_REPAIR_VERSION, SignalConfig
from .training_constraints import TRAINING_CONSTRAINT_VERSION, audit_policy_spec_for_oracle_access, local_only_training_protocol


MEMORY_ABLATION_VERSION = "e04_s09_memory_ablations.v1"
CELL_MEMORY_VARIANTS = ("no_memory", "one_bit", "bounded_counter", "neighbor_memory")
FIELD_MEMORY_VARIANTS = ("no_signal_field_memory", "signal_field_memory")
ABLATION_AXES = ("cell_memory_capacity", "signal_field_memory_capacity")
DEFAULT_MEMORY_COUNTER_MAX = 5
DEFAULT_NEIGHBOR_HISTORY = 4


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
class MemoryAblationTask:
    task_id: str
    task_family: str
    task_panel: str
    initial_values: tuple[int, ...]
    horizon: int
    frozen_positions: tuple[int, ...] = ()
    sortedness_threshold_percent: float = 95.0
    perturbation_schedule: tuple[Mapping[str, Any], ...] = ()
    activations_per_tick: int = 12
    transfer: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "taskId": self.task_id,
            "taskFamily": self.task_family,
            "taskPanel": self.task_panel,
            "initialValues": list(self.initial_values),
            "horizon": int(self.horizon),
            "frozenPositions": list(self.frozen_positions),
            "sortednessThresholdPercent": float(self.sortedness_threshold_percent),
            "perturbationSchedule": [dict(event) for event in self.perturbation_schedule],
            "activationsPerTick": int(self.activations_per_tick),
            "transfer": bool(self.transfer),
            "scheduleHash": stable_hash([dict(event) for event in self.perturbation_schedule]),
            "memoryAblationVersion": MEMORY_ABLATION_VERSION,
        }


@dataclass(frozen=True)
class PolicyFamilySpec:
    family_id: str
    family_kind: str
    source_step: str
    policy_label: str
    base_policy: Any
    base_policy_spec: Mapping[str, Any]
    learning_config: Mapping[str, Any] | None = None
    base_signal_config: Mapping[str, Any] | None = None
    base_repair_config: Mapping[str, Any] | None = None
    source_candidate_id: str | None = None
    replayable: bool = True
    replay_blocker: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "familyId": self.family_id,
            "familyKind": self.family_kind,
            "sourceStep": self.source_step,
            "policyLabel": self.policy_label,
            "basePolicySpec": dict(self.base_policy_spec),
            "learningConfig": None if self.learning_config is None else dict(self.learning_config),
            "baseSignalConfig": self.base_signal_config or SignalConfig("no_signal").to_dict(),
            "baseRepairConfig": self.base_repair_config or RepairRuleConfig("nudge_count", nudge_threshold=2).to_dict(),
            "sourceCandidateId": self.source_candidate_id,
            "replayable": bool(self.replayable),
            "replayBlocker": self.replay_blocker,
            "memoryAblationVersion": MEMORY_ABLATION_VERSION,
        }


def build_s09_tasks() -> tuple[MemoryAblationTask, ...]:
    """Return deterministic repair, homeostasis, and transfer tasks for S09."""

    homeostasis = {task["taskId"]: task for task in build_homeostatic_benchmark_config()["tasks"]}
    transfer_schedule = (
        {"eventIndex": 0, "tick": 1, "type": "swap", "leftIndex": 2, "rightIndex": 3, "source": "s09_fixed_transfer"},
        {"eventIndex": 1, "tick": 3, "type": "insert", "position": 0, "value": 99, "source": "s09_fixed_transfer"},
        {"eventIndex": 2, "tick": 5, "type": "delete", "position": 0, "source": "s09_fixed_transfer"},
        {"eventIndex": 3, "tick": 6, "type": "freeze", "cellId": 4, "source": "s09_fixed_transfer"},
    )
    return (
        MemoryAblationTask(
            task_id="repair_pair_n2_training",
            task_family="repair",
            task_panel="s09_repair_training",
            initial_values=(2, 1),
            frozen_positions=(1,),
            horizon=32,
        ),
        MemoryAblationTask(
            task_id="repair_barrier_n5_transfer",
            task_family="repair",
            task_panel="s09_repair_transfer",
            initial_values=(4, 1, 3, 2, 5),
            frozen_positions=(1,),
            horizon=48,
            transfer=True,
        ),
        MemoryAblationTask(
            task_id="homeostasis_adjacent_swap_training",
            task_family="homeostasis",
            task_panel="s09_homeostasis_training",
            initial_values=tuple(homeostasis["adjacent_swap_recovery"]["initialValues"]),
            horizon=int(homeostasis["adjacent_swap_recovery"]["horizonTicks"]),
            sortedness_threshold_percent=float(homeostasis["adjacent_swap_recovery"]["sortednessThresholdPercent"]),
            perturbation_schedule=tuple(homeostasis["adjacent_swap_recovery"]["perturbationSchedule"]),
        ),
        MemoryAblationTask(
            task_id="homeostasis_insert_delete_transfer",
            task_family="homeostasis",
            task_panel="s09_homeostasis_transfer",
            initial_values=(1, 2, 3, 4, 5, 6, 7, 8),
            horizon=8,
            sortedness_threshold_percent=95.0,
            perturbation_schedule=transfer_schedule,
            transfer=True,
        ),
    )


def policy_family_from_frontier_row(row: Mapping[str, Any]) -> PolicyFamilySpec:
    program = parse_rule_program(row["dslProgramJson"])
    policy = DSLPolicy(program)
    spec = policy.to_spec().to_dict()
    audit = audit_policy_spec_for_oracle_access(spec, local_only_training_protocol())
    return PolicyFamilySpec(
        family_id=f"frontier_{str(row['policyId'])}",
        family_kind="frontier",
        source_step="E03_S14",
        policy_label=str(row.get("className", row["policyId"])),
        base_policy=policy,
        base_policy_spec=spec,
        replayable=bool(audit["success"]),
        replay_blocker=None if audit["success"] else compact_json(audit.get("violations", [])),
    )


def classic_policy_families() -> tuple[PolicyFamilySpec, ...]:
    return (
        PolicyFamilySpec(
            family_id="classic_bubble",
            family_kind="classic",
            source_step="E03_S01",
            policy_label="classic_bubble",
            base_policy="bubble",
            base_policy_spec={"policy_id": "classic_bubble", "family": "classic", "algotype": "bubble", "parameters": {}},
        ),
        PolicyFamilySpec(
            family_id="classic_insertion",
            family_kind="classic",
            source_step="E03_S01",
            policy_label="classic_insertion",
            base_policy="insertion",
            base_policy_spec={"policy_id": "classic_insertion", "family": "classic", "algotype": "insertion", "parameters": {}},
        ),
    )


def learned_policy_families() -> tuple[PolicyFamilySpec, ...]:
    config = LocalLearningConfig("local_adaptive", learning_rate=0.35).to_dict()
    policy = LocalLearningPolicyWrapper("bubble", config)
    return (
        PolicyFamilySpec(
            family_id="learned_local_adaptive_bubble",
            family_kind="learned",
            source_step="S06",
            policy_label="local_adaptive_bubble",
            base_policy=policy,
            base_policy_spec=policy.to_spec().to_dict(),
            learning_config=config,
            base_signal_config=SignalConfig("no_signal").to_dict(),
            base_repair_config=RepairRuleConfig("nudge_count", nudge_threshold=2, approach_direction="either").to_dict(),
        ),
    )


def evolved_policy_family_from_candidate(row: Mapping[str, Any]) -> PolicyFamilySpec:
    learning_config = json.loads(row["learningConfigJson"])
    signal_config = json.loads(row["signalConfigJson"])
    repair_config = json.loads(row["repairConfigJson"])
    policy = LocalLearningPolicyWrapper("bubble", LocalLearningConfig.from_spec(learning_config))
    audit = audit_policy_spec_for_oracle_access(policy, local_only_training_protocol())
    return PolicyFamilySpec(
        family_id=f"evolved_{row['candidateId']}",
        family_kind="evolved",
        source_step="S08",
        policy_label=f"s08_rank_{int(row.get('rank', -1))}",
        base_policy=policy,
        base_policy_spec=policy.to_spec().to_dict(),
        learning_config=learning_config,
        base_signal_config=signal_config,
        base_repair_config=repair_config,
        source_candidate_id=str(row["candidateId"]),
        replayable=bool(audit["success"] and row.get("auditSuccess", True)),
        replay_blocker=None if audit["success"] else compact_json(audit.get("violations", [])),
    )


def cell_memory_config(variant: str, *, counter_max: int = DEFAULT_MEMORY_COUNTER_MAX, neighbor_history: int = DEFAULT_NEIGHBOR_HISTORY) -> MemoryConfig:
    if variant not in CELL_MEMORY_VARIANTS:
        raise ValueError(f"unknown cell-memory variant: {variant}")
    return MemoryConfig(variant, counter_max=counter_max, neighbor_history=neighbor_history)


def condition_rows_for_family(
    family: PolicyFamilySpec,
    tasks: Sequence[MemoryAblationTask],
    *,
    seed_count: int = 3,
    seed_base: int = 9100,
) -> list[dict[str, Any]]:
    """Build paired ablation condition records for one policy family."""

    rows: list[dict[str, Any]] = []
    if not family.replayable:
        return rows
    base_signal = SignalConfig.from_spec(family.base_signal_config or "no_signal")
    base_repair = RepairRuleConfig.from_spec(family.base_repair_config or RepairRuleConfig("nudge_count", nudge_threshold=2))
    for task_index, task in enumerate(tasks):
        task_payload = task.to_dict()
        for seed_index in range(seed_count):
            scheduler_seed = int(seed_base + task_index * 100 + seed_index)
            tie_seed = int(seed_base + 5000 + task_index * 100 + seed_index)
            for variant in CELL_MEMORY_VARIANTS:
                memory_config = cell_memory_config(variant)
                signal_config = base_signal
                repair_config = base_repair
                controlled = {
                    "family": family.to_dict(),
                    "task": task_payload,
                    "schedulerSeed": scheduler_seed,
                    "tieBreakerSeed": tie_seed,
                    "signalConfig": signal_config.to_dict(),
                    "repairConfig": repair_config.to_dict(),
                    "axis": "cell_memory_capacity",
                }
                mechanism = {"memoryConfig": memory_config.to_dict()}
                rows.append(
                    {
                        "conditionId": stable_id("s09_condition", [family.family_id, task.task_id, seed_index, "cell", variant]),
                        "familyId": family.family_id,
                        "familyKind": family.family_kind,
                        "sourceStep": family.source_step,
                        "sourceCandidateId": family.source_candidate_id,
                        "ablationAxis": "cell_memory_capacity",
                        "ablationVariant": variant,
                        "referenceVariant": "no_memory",
                        "taskId": task.task_id,
                        "taskFamily": task.task_family,
                        "taskPanel": task.task_panel,
                        "transfer": bool(task.transfer),
                        "seedIndex": int(seed_index),
                        "schedulerSeed": scheduler_seed,
                        "tieBreakerSeed": tie_seed,
                        "taskSpecJson": compact_json(task_payload),
                        "memoryConfigJson": compact_json(memory_config.to_dict()),
                        "signalConfigJson": compact_json(signal_config.to_dict()),
                        "repairConfigJson": compact_json(repair_config.to_dict()),
                        "basePolicySpecJson": compact_json(family.base_policy_spec),
                        "controlledConfigHash": config_hash(controlled),
                        "mechanismConfigHash": config_hash(mechanism),
                        "controlledGroupId": "|".join([family.family_id, "cell_memory_capacity", task.task_id, str(seed_index)]),
                        "memoryAblationVersion": MEMORY_ABLATION_VERSION,
                    }
                )

            field_repair = base_repair if base_repair.variant == "signal_threshold" else RepairRuleConfig("signal_threshold", signal_threshold=0.45, approach_direction="either")
            field_signal = base_signal if base_signal.variant != "no_signal" else SignalConfig("diffusive", diffusion_rate=0.25, decay=0.05, emission_scale=1.0, random_seed=scheduler_seed)
            for variant, signal_config in (
                ("no_signal_field_memory", SignalConfig("no_signal")),
                ("signal_field_memory", field_signal),
            ):
                memory_config = MemoryConfig("no_memory")
                controlled = {
                    "family": family.to_dict(),
                    "task": task_payload,
                    "schedulerSeed": scheduler_seed,
                    "tieBreakerSeed": tie_seed,
                    "memoryConfig": memory_config.to_dict(),
                    "repairConfig": field_repair.to_dict(),
                    "axis": "signal_field_memory_capacity",
                }
                mechanism = {"signalConfig": signal_config.to_dict()}
                rows.append(
                    {
                        "conditionId": stable_id("s09_condition", [family.family_id, task.task_id, seed_index, "field", variant]),
                        "familyId": family.family_id,
                        "familyKind": family.family_kind,
                        "sourceStep": family.source_step,
                        "sourceCandidateId": family.source_candidate_id,
                        "ablationAxis": "signal_field_memory_capacity",
                        "ablationVariant": variant,
                        "referenceVariant": "no_signal_field_memory",
                        "taskId": task.task_id,
                        "taskFamily": task.task_family,
                        "taskPanel": task.task_panel,
                        "transfer": bool(task.transfer),
                        "seedIndex": int(seed_index),
                        "schedulerSeed": scheduler_seed,
                        "tieBreakerSeed": tie_seed,
                        "taskSpecJson": compact_json(task_payload),
                        "memoryConfigJson": compact_json(memory_config.to_dict()),
                        "signalConfigJson": compact_json(signal_config.to_dict()),
                        "repairConfigJson": compact_json(field_repair.to_dict()),
                        "basePolicySpecJson": compact_json(family.base_policy_spec),
                        "controlledConfigHash": config_hash(controlled),
                        "mechanismConfigHash": config_hash(mechanism),
                        "controlledGroupId": "|".join([family.family_id, "signal_field_memory_capacity", task.task_id, str(seed_index)]),
                        "memoryAblationVersion": MEMORY_ABLATION_VERSION,
                    }
                )
    return rows


def policy_for_condition(family: PolicyFamilySpec, condition: Mapping[str, Any]) -> LocalRulePolicy:
    base = family.base_policy
    if family.family_kind == "evolved":
        config = LocalLearningConfig.from_spec(family.learning_config or {})
        base = LocalLearningPolicyWrapper("bubble", config)
    elif family.family_kind == "learned":
        config = LocalLearningConfig.from_spec(family.learning_config or "local_adaptive")
        base = LocalLearningPolicyWrapper("bubble", config)
    return MemoryPolicyWrapper(base, MemoryConfig.from_spec(json.loads(str(condition["memoryConfigJson"]))))


def _trace_sortedness(simulator: LearningEventSimulator, fallback: Sequence[int]) -> list[float]:
    values = [float(row.get("sortedness_percent", np.nan)) for row in simulator.trace_rows if "sortedness_percent" in row]
    values = [value for value in values if math.isfinite(value)]
    if values:
        return values
    return [float(sortedness_percent(fallback))]


def _energy_proxy(simulator: LearningEventSimulator) -> float:
    signal_energy = 0.0
    for values in getattr(simulator, "signal_fields", {}).values():
        signal_energy += float(np.asarray(values, dtype=float).sum())
    return float(simulator.comparison_count + simulator.swap_count + simulator.blocked_move_attempts + 0.01 * signal_energy)


def run_ablation_condition(family: PolicyFamilySpec, condition: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task = json.loads(str(condition["taskSpecJson"]))
    policy = policy_for_condition(family, condition)
    signal_config = SignalConfig.from_spec(json.loads(str(condition["signalConfigJson"])))
    repair_config = RepairRuleConfig.from_spec(json.loads(str(condition["repairConfigJson"])))
    sim = LearningEventSimulator(
        task["initialValues"],
        policy,
        frozen_positions=task.get("frozenPositions", ()),
        frozen_variant="stuck" if task.get("frozenPositions") else "none",
        scheduler_seed=int(condition["schedulerSeed"]),
        tie_breaker_seed=int(condition["tieBreakerSeed"]),
        signal_config=signal_config,
        repair_config=repair_config,
        condition_id=str(condition["conditionId"]),
        implementation="e04_s09_memory_ablation_cpu_reference",
        research_step_id="S09",
    )
    tick_rows: list[dict[str, Any]] = []
    initial_length = len(task["initialValues"])
    if task["taskFamily"] == "repair":
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
                    "ablationAxis": condition["ablationAxis"],
                    "ablationVariant": condition["ablationVariant"],
                    "taskId": condition["taskId"],
                    "seedIndex": int(condition["seedIndex"]),
                    "tick": int(tick),
                    "eventsJson": compact_json(events),
                    "preSortednessPercent": float(pre["sortednessPercent"]),
                    "postSortednessPercent": float(post["sortednessPercent"]),
                    "postInRange": bool(post["sortednessPercent"] >= threshold),
                    "currentLength": int(post["currentLength"]),
                    "currentPairDenominator": int(post["currentPairDenominator"]),
                    "valuesJson": compact_json(sim.current_values()),
                    "memoryAblationVersion": MEMORY_ABLATION_VERSION,
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
    repair_success = bool(sim.is_sorted() and not sim.current_frozen_positions()) if task["taskFamily"] == "repair" else None
    score = (
        (2.0 if completed else 0.0)
        + final_sortedness / 100.0
        + (0.0 if time_in_range is None else time_in_range)
        + float(len(sim.recovery_events)) * 0.5
        - 0.004 * _energy_proxy(sim)
    )
    row = {
        **{key: condition[key] for key in condition.keys()},
        "completed": completed,
        "repairSuccess": repair_success,
        "homeostaticMaintenance": time_in_range,
        "finalSortednessPercent": final_sortedness,
        "delayedGratification": float(dg["delayedGratification"]),
        "delayedGratificationEventCount": int(dg["dgEventCount"]),
        "energyProxy": _energy_proxy(sim),
        "activationCount": int(sim.activation_count),
        "swapCount": int(sim.swap_count),
        "comparisonCount": int(sim.comparison_count),
        "blockedMoveAttempts": int(sim.blocked_move_attempts),
        "recoveredCellCount": int(len(sim.recovery_events)),
        "remainingFrozenCellCount": int(len(sim.current_frozen_positions())),
        "initialValuesJson": compact_json(task["initialValues"]),
        "finalValuesJson": compact_json(sim.current_values()),
        "sortednessTraceJson": compact_json(sortedness_series),
        "score": float(score),
        "memoryRepairVersion": MEMORY_REPAIR_VERSION,
        "signalRepairVersion": SIGNAL_REPAIR_VERSION,
        "repairRepairVersion": REPAIR_REPAIR_VERSION,
        "localLearningVersion": LOCAL_LEARNING_VERSION,
        "memoryAblationVersion": MEMORY_ABLATION_VERSION,
    }
    return _json_ready(row), _json_ready(tick_rows)


def compute_ablation_deltas(result_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ("score", "finalSortednessPercent", "energyProxy", "delayedGratification", "activationCount")
    for group_id, group in result_df.groupby("controlledGroupId", dropna=False):
        reference_variant = str(group["referenceVariant"].iloc[0])
        reference = group[group["ablationVariant"] == reference_variant]
        if reference.empty:
            continue
        ref = reference.iloc[0]
        for _, row in group.iterrows():
            if row["ablationVariant"] == reference_variant:
                continue
            payload = {
                "controlledGroupId": group_id,
                "familyId": row["familyId"],
                "familyKind": row["familyKind"],
                "ablationAxis": row["ablationAxis"],
                "ablationVariant": row["ablationVariant"],
                "referenceVariant": reference_variant,
                "taskId": row["taskId"],
                "taskFamily": row["taskFamily"],
                "taskPanel": row["taskPanel"],
                "transfer": bool(row["transfer"]),
                "seedIndex": int(row["seedIndex"]),
                "memoryAblationVersion": MEMORY_ABLATION_VERSION,
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


def summarize_ablation_effects(delta_df: pd.DataFrame) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()
    grouped = delta_df.groupby(["ablationAxis", "ablationVariant", "familyKind", "taskFamily", "transfer"], dropna=False)
    summary = grouped.agg(
        comparisonCount=("scoreDelta", "size"),
        meanScoreDelta=("scoreDelta", "mean"),
        medianScoreDelta=("scoreDelta", "median"),
        maxScoreDelta=("scoreDelta", "max"),
        minScoreDelta=("scoreDelta", "min"),
        meanSortednessDelta=("finalSortednessPercentDelta", "mean"),
        meanEnergyDelta=("energyProxyDelta", "mean"),
        meanDelayedGratificationDelta=("delayedGratificationDelta", "mean"),
        meanActivationDelta=("activationCountDelta", "mean"),
        improvedScoreFraction=("scoreDelta", lambda values: float((values > 1e-9).mean())),
    ).reset_index()
    summary["memoryAblationVersion"] = MEMORY_ABLATION_VERSION
    return summary


def validate_ablation_outputs(condition_df: pd.DataFrame, result_df: pd.DataFrame, delta_df: pd.DataFrame, *, expected_evolved_min: int = 1) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check_id: str, success: bool, detail: str) -> None:
        rows.append(
            {
                "checkId": check_id,
                "success": bool(success),
                "detail": detail,
                "memoryAblationVersion": MEMORY_ABLATION_VERSION,
            }
        )

    add("condition_rows_nonempty", len(condition_df) > 0, f"conditionRows={len(condition_df)}")
    add("result_rows_match_conditions", len(result_df) == len(condition_df), f"resultRows={len(result_df)} conditionRows={len(condition_df)}")
    group_sizes = condition_df.groupby("controlledGroupId")["ablationVariant"].nunique()
    expected_sizes = condition_df.groupby("controlledGroupId")["ablationAxis"].first().map(
        {"cell_memory_capacity": len(CELL_MEMORY_VARIANTS), "signal_field_memory_capacity": len(FIELD_MEMORY_VARIANTS)}
    )
    add("paired_variant_sets_complete", bool((group_sizes == expected_sizes).all()), f"groupCount={len(group_sizes)}")
    seed_ok = condition_df.groupby("controlledGroupId")[["schedulerSeed", "tieBreakerSeed"]].nunique().max().max() == 1
    add("paired_seeds_identical", bool(seed_ok), "schedulerSeed and tieBreakerSeed are constant within each controlled group")
    schedule_ok = condition_df.groupby("controlledGroupId")["taskSpecJson"].nunique().max() == 1
    add("perturbation_schedules_identical", bool(schedule_ok), "taskSpecJson is constant within each controlled group")
    controlled_ok = condition_df.groupby("controlledGroupId")["controlledConfigHash"].nunique().max() == 1
    add("only_memory_axis_changes", bool(controlled_ok), "controlledConfigHash is constant within each ablation group")
    cell_variants = set(condition_df.loc[condition_df["ablationAxis"] == "cell_memory_capacity", "ablationVariant"])
    add("cell_memory_variants_present", set(CELL_MEMORY_VARIANTS).issubset(cell_variants), f"cellVariants={sorted(cell_variants)}")
    field_variants = set(condition_df.loc[condition_df["ablationAxis"] == "signal_field_memory_capacity", "ablationVariant"])
    add("signal_field_memory_variant_present", set(FIELD_MEMORY_VARIANTS).issubset(field_variants), f"fieldVariants={sorted(field_variants)}")
    family_kinds = set(condition_df["familyKind"])
    add("required_policy_families_present", {"classic", "frontier", "learned", "evolved"}.issubset(family_kinds), f"familyKinds={sorted(family_kinds)}")
    evolved_count = condition_df.loc[condition_df["familyKind"] == "evolved", "familyId"].nunique()
    add("s08_evolved_candidates_replayable", evolved_count >= expected_evolved_min, f"evolvedFamilyCount={evolved_count}")
    add("deltas_quantified", len(delta_df) > 0, f"deltaRows={len(delta_df)}")
    transfer_rows = result_df[result_df["transfer"].astype(bool)]
    add("transfer_tasks_present", len(transfer_rows) > 0, f"transferResultRows={len(transfer_rows)}")
    metric_ok = result_df[["score", "finalSortednessPercent", "energyProxy", "delayedGratification"]].notna().all().all()
    add("core_metrics_present", bool(metric_ok), "score, sortedness, energy, and DG are populated")
    return pd.DataFrame(rows)


def replay_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(stable_json(list(rows)).encode("utf-8")).hexdigest()
