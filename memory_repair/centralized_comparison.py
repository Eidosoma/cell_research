"""Centralized global-oracle repair comparison for E04 S14."""

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

from .competence_proxies import (
    COMPETENCE_PROXY_VERSION,
    CompetencePolicySpec,
    baseline_competence_policies,
    competence_policy_from_s08_candidate,
    local_memory_signal_competence_policy,
    run_competence_condition,
)
from .fatigue import FATIGUE_DAMAGE_VERSION, FatigueDamageConfig
from .homeostasis import HOMEOSTASIS_BENCHMARK_VERSION, normalized_sortedness_record, stable_hash, stable_json
from .learning import LOCAL_LEARNING_VERSION
from .memory import MEMORY_REPAIR_VERSION, MemoryConfig
from .repair import REPAIR_REPAIR_VERSION, RepairRuleConfig
from .signals import SIGNAL_REPAIR_VERSION, SignalConfig
from .training_constraints import (
    ORACLE_BASELINE_ID,
    TRAINING_CONSTRAINT_VERSION,
    global_oracle_baseline_protocol,
    global_oracle_baseline_record,
)


CENTRALIZED_COMPARISON_VERSION = "e04_s14_centralized_repair_comparison.v1"
CENTRALIZED_PROXY_SCOPE_NOTE = (
    "Direct computational local-versus-global controller proxy only; not a biological tissue repair, "
    "morphogenesis, or intelligence measurement."
)
CENTRALIZED_ORACLE_CONTROLLER_ID = "centralized_global_oracle_repair_controller"
CENTRALIZED_CONTROLLER_GROUP = "global_oracle"
CONTROLLER_KINDS = ("local", "global_oracle")


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
class CentralizedComparisonTask:
    task_id: str
    task_family: str
    task_panel: str
    initial_values: tuple[int, ...]
    horizon: int
    frozen_positions: tuple[int, ...] = ()
    perturbation_schedule: tuple[Mapping[str, Any], ...] = ()
    activations_per_tick: int = 14
    sortedness_threshold_percent: float = 95.0
    fatigue_config: Mapping[str, Any] | None = None
    repair_config: Mapping[str, Any] | None = None
    fairness_block_id: str = "s14_default_paired_task_seed"

    def to_dict(self) -> dict[str, Any]:
        schedule = [dict(event) for event in self.perturbation_schedule]
        fatigue = dict(self.fatigue_config or FatigueDamageConfig("none").to_dict())
        repair = dict(self.repair_config or RepairRuleConfig("nudge_count", nudge_threshold=2, approach_direction="either").to_dict())
        task_hash_payload = {
            "initialValues": list(self.initial_values),
            "horizon": int(self.horizon),
            "frozenPositions": list(self.frozen_positions),
            "perturbationSchedule": schedule,
            "activationsPerTick": int(self.activations_per_tick),
            "fatigueConfig": fatigue,
            "repairConfig": repair,
        }
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
            "perturbationEventCount": int(len(schedule)),
            "scheduleHash": stable_hash(schedule),
            "activationsPerTick": int(self.activations_per_tick),
            "fatigueConfig": fatigue,
            "repairConfig": repair,
            "fairnessBlockId": self.fairness_block_id,
            "pairedTaskSeedRequired": True,
            "taskConfigHash": stable_hash(task_hash_payload),
            "claimBoundary": CENTRALIZED_PROXY_SCOPE_NOTE,
            "centralizedComparisonVersion": CENTRALIZED_COMPARISON_VERSION,
        }


def build_s14_tasks() -> tuple[CentralizedComparisonTask, ...]:
    homeostasis_schedule = (
        {"eventIndex": 0, "tick": 1, "type": "swap", "leftIndex": 3, "rightIndex": 4, "source": "s14_pair_swap"},
        {"eventIndex": 1, "tick": 2, "type": "freeze", "cellId": 7, "source": "s14_pair_freeze"},
        {"eventIndex": 2, "tick": 3, "type": "damage", "cellId": 4, "durationTicks": 2, "source": "s14_pair_damage"},
        {"eventIndex": 3, "tick": 5, "type": "insert", "position": 0, "value": 99, "source": "s14_pair_insert"},
        {"eventIndex": 4, "tick": 7, "type": "delete", "position": 0, "source": "s14_pair_delete"},
        {"eventIndex": 5, "tick": 8, "type": "recover", "cellId": 7, "source": "s14_pair_recover"},
        {"eventIndex": 6, "tick": 10, "type": "swap", "leftIndex": 7, "rightIndex": 8, "source": "s14_pair_late_swap"},
    )
    return (
        CentralizedComparisonTask(
            task_id="s14_sorting_pair_n8",
            task_family="sorting",
            task_panel="s14_pair_sorting",
            initial_values=(8, 1, 7, 2, 6, 3, 5, 4),
            horizon=112,
        ),
        CentralizedComparisonTask(
            task_id="s14_repair_edge_pair_n10",
            task_family="repairable",
            task_panel="s14_pair_repair",
            initial_values=(10, 1, 9, 2, 8, 3, 7, 4, 6, 5),
            frozen_positions=(1, 8),
            horizon=168,
            repair_config=RepairRuleConfig("signal_threshold", signal_threshold=0.9, nudge_threshold=3, approach_direction="either").to_dict(),
        ),
        CentralizedComparisonTask(
            task_id="s14_fatigue_damage_n10",
            task_family="fatigue",
            task_panel="s14_pair_fatigue",
            initial_values=(9, 2, 8, 1, 10, 3, 7, 4, 6, 5),
            frozen_positions=(5,),
            horizon=152,
            fatigue_config=FatigueDamageConfig(
                "combined",
                movement_threshold=4,
                failed_swap_threshold=3,
                frustration_threshold=3,
                recovery_activations=2,
                damage_probability=0.05,
                damage_cooldown_activations=2,
                random_seed=14101,
            ).to_dict(),
            repair_config=RepairRuleConfig("nudge_count", nudge_threshold=3, approach_direction="either").to_dict(),
        ),
        CentralizedComparisonTask(
            task_id="s14_homeostasis_mixed_n10",
            task_family="homeostasis",
            task_panel="s14_pair_homeostasis",
            initial_values=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            horizon=13,
            perturbation_schedule=homeostasis_schedule,
            activations_per_tick=16,
            fatigue_config=FatigueDamageConfig(
                "stochastic_damage",
                damage_probability=0.05,
                damage_cooldown_activations=2,
                random_seed=14102,
            ).to_dict(),
            repair_config=RepairRuleConfig("signal_threshold", signal_threshold=0.9, nudge_threshold=3, approach_direction="either").to_dict(),
        ),
    )


def build_s14_local_policies(evolved_rows: Sequence[Mapping[str, Any]] = (), *, max_evolved: int = 1) -> tuple[CompetencePolicySpec, ...]:
    policies: list[CompetencePolicySpec] = [baseline_competence_policies()[0], local_memory_signal_competence_policy()]
    for row in evolved_rows:
        policy = competence_policy_from_s08_candidate(row)
        if policy.replayable:
            policies.append(policy)
        if sum(1 for item in policies if item.family_kind == "evolved") >= max_evolved:
            break
    return tuple(policies)


def centralized_oracle_controller_spec() -> dict[str, Any]:
    protocol = global_oracle_baseline_protocol().to_dict()
    return {
        "policyId": CENTRALIZED_ORACLE_CONTROLLER_ID,
        "policyGroup": CENTRALIZED_CONTROLLER_GROUP,
        "familyKind": "centralized_global_oracle",
        "sourceStep": "S14",
        "policyLabel": "centralized global-oracle repair upper-bound controller",
        "controllerKind": "global_oracle",
        "oracleAllowed": True,
        "oracleBaseline": True,
        "oracleBaselineLabel": ORACLE_BASELINE_ID,
        "usesGlobalController": True,
        "usesGlobalSortednessSignal": True,
        "usesWholeArrayValues": True,
        "usesWholeArrayRanks": True,
        "usesWholeArrayTargetSignal": True,
        "usesTargetPositionOracle": True,
        "trainingProtocol": protocol,
        "claimBoundary": CENTRALIZED_PROXY_SCOPE_NOTE,
        "centralizedComparisonVersion": CENTRALIZED_COMPARISON_VERSION,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
    }


def local_condition_rows_for_s14(
    policies: Sequence[CompetencePolicySpec],
    tasks: Sequence[CentralizedComparisonTask | Mapping[str, Any]],
    *,
    seed_count: int = 3,
    seed_base: int = 18100,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        if not policy.replayable:
            continue
        base_signal = SignalConfig.from_spec(policy.signal_config or "no_signal")
        for task_index, task_obj in enumerate(tasks):
            task = task_obj.to_dict() if isinstance(task_obj, CentralizedComparisonTask) else dict(task_obj)
            repair = RepairRuleConfig.from_spec(task["repairConfig"])
            fatigue = FatigueDamageConfig.from_spec(task["fatigueConfig"])
            for seed_index in range(seed_count):
                scheduler_seed = int(seed_base + task_index * 100 + seed_index)
                tie_seed = int(seed_base + 10_000 + task_index * 100 + seed_index)
                signal_seed = int(seed_base + 20_000 + task_index * 100 + seed_index)
                signal_config = SignalConfig.from_spec({**base_signal.to_dict(), "randomSeed": signal_seed})
                pair_key = stable_id("s14_pair", [task["taskId"], seed_index])
                controlled = {
                    "policy": policy.to_dict(),
                    "task": task,
                    "schedulerSeed": scheduler_seed,
                    "tieBreakerSeed": tie_seed,
                    "signalConfig": signal_config.to_dict(),
                    "repairConfig": repair.to_dict(),
                    "fatigueConfig": fatigue.to_dict(),
                    "pairKey": pair_key,
                }
                rows.append(
                    {
                        "conditionId": stable_id("s14_local_condition", [policy.policy_id, task["taskId"], seed_index]),
                        "pairKey": pair_key,
                        "controllerKind": "local",
                        "policyId": policy.policy_id,
                        "policyGroup": policy.policy_group,
                        "familyKind": policy.family_kind,
                        "sourceStep": policy.source_step,
                        "sourceCandidateId": policy.source_candidate_id,
                        "taskId": task["taskId"],
                        "taskFamily": task["taskFamily"],
                        "taskPanel": task["taskPanel"],
                        "taskConfigHash": task["taskConfigHash"],
                        "seedIndex": int(seed_index),
                        "schedulerSeed": scheduler_seed,
                        "tieBreakerSeed": tie_seed,
                        "signalRandomSeed": signal_seed,
                        "oracleAllowed": False,
                        "oracleBaseline": False,
                        "oracleBaselineLabel": None,
                        "usesGlobalController": False,
                        "policyAuditSuccess": bool(policy.policy_audit_success),
                        "pairedTaskSeedRequired": True,
                        "fairnessBlockId": task["fairnessBlockId"],
                        "taskSpecJson": compact_json(task),
                        "signalConfigJson": compact_json(signal_config.to_dict()),
                        "repairConfigJson": compact_json(repair.to_dict()),
                        "fatigueConfigJson": compact_json(fatigue.to_dict()),
                        "memoryConfigJson": compact_json(policy.memory_config or MemoryConfig("no_memory").to_dict()),
                        "learningConfigJson": compact_json(policy.learning_config) if policy.learning_config is not None else None,
                        "basePolicySpecJson": compact_json(policy.base_policy_spec),
                        "controlledConfigHash": config_hash(controlled),
                        "claimBoundary": CENTRALIZED_PROXY_SCOPE_NOTE,
                        "centralizedComparisonVersion": CENTRALIZED_COMPARISON_VERSION,
                        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
                    }
                )
    return rows


def global_condition_rows_for_s14(
    tasks: Sequence[CentralizedComparisonTask | Mapping[str, Any]],
    *,
    seed_count: int = 3,
    seed_base: int = 18100,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    controller = centralized_oracle_controller_spec()
    for task_index, task_obj in enumerate(tasks):
        task = task_obj.to_dict() if isinstance(task_obj, CentralizedComparisonTask) else dict(task_obj)
        repair = RepairRuleConfig.from_spec(task["repairConfig"])
        fatigue = FatigueDamageConfig.from_spec(task["fatigueConfig"])
        for seed_index in range(seed_count):
            scheduler_seed = int(seed_base + task_index * 100 + seed_index)
            tie_seed = int(seed_base + 10_000 + task_index * 100 + seed_index)
            signal_seed = int(seed_base + 20_000 + task_index * 100 + seed_index)
            pair_key = stable_id("s14_pair", [task["taskId"], seed_index])
            controlled = {
                "policy": controller,
                "task": task,
                "schedulerSeed": scheduler_seed,
                "tieBreakerSeed": tie_seed,
                "signalRandomSeed": signal_seed,
                "pairKey": pair_key,
            }
            rows.append(
                {
                    "conditionId": stable_id("s14_global_condition", [CENTRALIZED_ORACLE_CONTROLLER_ID, task["taskId"], seed_index]),
                    "pairKey": pair_key,
                    "controllerKind": "global_oracle",
                    "policyId": CENTRALIZED_ORACLE_CONTROLLER_ID,
                    "policyGroup": CENTRALIZED_CONTROLLER_GROUP,
                    "familyKind": "centralized_global_oracle",
                    "sourceStep": "S14",
                    "sourceCandidateId": None,
                    "taskId": task["taskId"],
                    "taskFamily": task["taskFamily"],
                    "taskPanel": task["taskPanel"],
                    "taskConfigHash": task["taskConfigHash"],
                    "seedIndex": int(seed_index),
                    "schedulerSeed": scheduler_seed,
                    "tieBreakerSeed": tie_seed,
                    "signalRandomSeed": signal_seed,
                    "oracleAllowed": True,
                    "oracleBaseline": True,
                    "oracleBaselineLabel": ORACLE_BASELINE_ID,
                    "usesGlobalController": True,
                    "usesGlobalSortednessSignal": True,
                    "usesWholeArrayValues": True,
                    "usesWholeArrayRanks": True,
                    "usesWholeArrayTargetSignal": True,
                    "usesTargetPositionOracle": True,
                    "policyAuditSuccess": True,
                    "pairedTaskSeedRequired": True,
                    "fairnessBlockId": task["fairnessBlockId"],
                    "taskSpecJson": compact_json(task),
                    "signalConfigJson": compact_json(SignalConfig("no_signal").to_dict()),
                    "repairConfigJson": compact_json(repair.to_dict()),
                    "fatigueConfigJson": compact_json(fatigue.to_dict()),
                    "memoryConfigJson": compact_json(MemoryConfig("no_memory").to_dict()),
                    "learningConfigJson": None,
                    "basePolicySpecJson": compact_json(controller),
                    "controlledConfigHash": config_hash(controlled),
                    "claimBoundary": CENTRALIZED_PROXY_SCOPE_NOTE,
                    "centralizedComparisonVersion": CENTRALIZED_COMPARISON_VERSION,
                    "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
                }
            )
    return rows


@dataclass
class _OracleCell:
    cell_id: int
    value: int
    frozen: bool = False
    damage_cooldown: int = 0


class CentralizedOracleRepairSimulator:
    """Small global-controller reference used only as an explicit S14 oracle baseline."""

    def __init__(self, task: Mapping[str, Any], *, scheduler_seed: int = 0) -> None:
        self.task = dict(task)
        self.rng = np.random.default_rng(int(scheduler_seed))
        self.cells = [_OracleCell(index, int(value), False, 0) for index, value in enumerate(task["initialValues"])]
        for position in task.get("frozenPositions", ()):
            if 0 <= int(position) < len(self.cells):
                self.cells[int(position)].frozen = True
        self.next_cell_id = max((cell.cell_id for cell in self.cells), default=-1) + 1
        self.activation_count = 0
        self.swap_count = 0
        self.comparison_count = 0
        self.intervention_count = 0
        self.repair_intervention_count = 0
        self.damage_repair_intervention_count = 0
        self.perturbation_count = 0
        self.tick_rows: list[dict[str, Any]] = []
        self.oracle_records: list[dict[str, Any]] = []
        self.state_trace: list[list[int]] = [self.current_values()]

    def current_values(self) -> list[int]:
        return [int(cell.value) for cell in self.cells]

    def frozen_cell_ids(self) -> list[int]:
        return [int(cell.cell_id) for cell in self.cells if cell.frozen]

    def damaged_cell_ids(self) -> list[int]:
        return [int(cell.cell_id) for cell in self.cells if int(cell.damage_cooldown) > 0]

    def _positions_by_id(self) -> dict[int, int]:
        return {int(cell.cell_id): index for index, cell in enumerate(self.cells)}

    def _oracle_record(self, reason: str, *, action_type: str, target_cell_id: int | None = None, swap_index: int | None = None) -> dict[str, Any]:
        record = global_oracle_baseline_record(self.current_values(), reason=reason)
        record.update(
            {
                "actionType": action_type,
                "targetCellId": target_cell_id,
                "swapIndex": swap_index,
                "activationIndex": int(self.activation_count),
                "frozenCellIds": self.frozen_cell_ids(),
                "damagedCellIds": self.damaged_cell_ids(),
                "usesGlobalController": True,
                "claimBoundary": CENTRALIZED_PROXY_SCOPE_NOTE,
                "centralizedComparisonVersion": CENTRALIZED_COMPARISON_VERSION,
            }
        )
        return _json_ready(record)

    def apply_event(self, event: Mapping[str, Any]) -> None:
        event_type = str(event["type"])
        if event_type == "swap":
            left = int(event["leftIndex"])
            right = int(event.get("rightIndex", left + 1))
            if 0 <= left < len(self.cells) and 0 <= right < len(self.cells):
                self.cells[left], self.cells[right] = self.cells[right], self.cells[left]
                self.perturbation_count += 1
        elif event_type == "insert":
            position = max(0, min(int(event["position"]), len(self.cells)))
            self.cells.insert(position, _OracleCell(self.next_cell_id, int(event["value"]), False, 0))
            self.next_cell_id += 1
            self.perturbation_count += 1
        elif event_type == "delete":
            position = int(event["position"])
            if 0 <= position < len(self.cells):
                self.cells.pop(position)
                self.perturbation_count += 1
        elif event_type == "freeze":
            cell_id = int(event["cellId"])
            pos = self._positions_by_id().get(cell_id)
            if pos is not None:
                self.cells[pos].frozen = True
                self.perturbation_count += 1
        elif event_type == "recover":
            cell_id = int(event["cellId"])
            pos = self._positions_by_id().get(cell_id)
            if pos is not None:
                self.cells[pos].frozen = False
                self.cells[pos].damage_cooldown = 0
                self.perturbation_count += 1
        elif event_type == "damage":
            cell_id = int(event["cellId"])
            pos = self._positions_by_id().get(cell_id)
            if pos is not None:
                duration = int(event.get("durationTicks", event.get("durationActivations", 2)))
                self.cells[pos].damage_cooldown = max(1, duration)
                self.perturbation_count += 1
        self.state_trace.append(self.current_values())

    def step(self) -> dict[str, Any]:
        self.activation_count += 1
        frozen = [cell for cell in self.cells if cell.frozen]
        if frozen:
            target = frozen[0]
            target.frozen = False
            self.intervention_count += 1
            self.repair_intervention_count += 1
            record = self._oracle_record("global_oracle_unfreeze", action_type="repair_frozen", target_cell_id=target.cell_id)
            self.oracle_records.append(record)
            self.state_trace.append(self.current_values())
            return record
        damaged = [cell for cell in self.cells if int(cell.damage_cooldown) > 0]
        if damaged:
            target = damaged[0]
            target.damage_cooldown = 0
            self.intervention_count += 1
            self.damage_repair_intervention_count += 1
            record = self._oracle_record("global_oracle_repair_damage", action_type="repair_damage", target_cell_id=target.cell_id)
            self.oracle_records.append(record)
            self.state_trace.append(self.current_values())
            return record
        values = self.current_values()
        self.comparison_count += max(0, len(values) - 1)
        swap_index = None
        for index in range(len(values) - 1):
            if values[index] > values[index + 1]:
                swap_index = index
                break
        if swap_index is None:
            record = self._oracle_record("global_oracle_wait_sorted", action_type="wait")
            self.oracle_records.append(record)
            self.state_trace.append(self.current_values())
            return record
        self.cells[swap_index], self.cells[swap_index + 1] = self.cells[swap_index + 1], self.cells[swap_index]
        self.swap_count += 1
        record = self._oracle_record("global_oracle_adjacent_inversion_swap", action_type="swap_adjacent_inversion", swap_index=swap_index)
        self.oracle_records.append(record)
        self.state_trace.append(self.current_values())
        return record

    def run(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        task_family = str(self.task["taskFamily"])
        threshold = float(self.task["sortednessThresholdPercent"])
        initial_length = int(self.task["initialLength"])
        if task_family == "homeostasis":
            for tick in range(int(self.task["horizon"])):
                events = [event for event in self.task.get("perturbationSchedule", ()) if int(event["tick"]) == tick]
                for event in events:
                    self.apply_event(event)
                pre = normalized_sortedness_record(self.current_values(), initial_length=initial_length)
                for _ in range(int(self.task.get("activationsPerTick", 14))):
                    self.step()
                post = normalized_sortedness_record(self.current_values(), initial_length=initial_length)
                self.tick_rows.append(
                    {
                        "tick": int(tick),
                        "eventsJson": compact_json(events),
                        "preSortednessPercent": float(pre["sortednessPercent"]),
                        "postSortednessPercent": float(post["sortednessPercent"]),
                        "postInRange": bool(float(post["sortednessPercent"]) >= threshold),
                        "currentLength": int(post["currentLength"]),
                        "currentPairDenominator": int(post["currentPairDenominator"]),
                        "valuesJson": compact_json(self.current_values()),
                        "claimBoundary": CENTRALIZED_PROXY_SCOPE_NOTE,
                        "centralizedComparisonVersion": CENTRALIZED_COMPARISON_VERSION,
                    }
                )
        else:
            for _ in range(int(self.task["horizon"])):
                if sorted(self.current_values()) == self.current_values() and not self.frozen_cell_ids() and not self.damaged_cell_ids():
                    break
                self.step()
        return self.result_record(), [_json_ready(row) for row in self.tick_rows]

    def result_record(self) -> dict[str, Any]:
        sortedness_series = [float(sortedness_percent(state)) for state in self.state_trace]
        dg = delayed_gratification_from_sortedness(sortedness_series)
        final_sortedness = float(sortedness_percent(self.current_values()))
        task_family = str(self.task["taskFamily"])
        threshold = float(self.task["sortednessThresholdPercent"])
        initial_frozen_count = len(self.task.get("frozenPositions", ()))
        remaining_frozen = len(self.frozen_cell_ids())
        if task_family == "homeostasis" and self.tick_rows:
            time_in_range = float(sum(row["postInRange"] for row in self.tick_rows) / len(self.tick_rows))
            completed = bool(time_in_range >= 0.75 and final_sortedness >= threshold)
        else:
            time_in_range = None
            completed = bool(final_sortedness >= threshold and remaining_frozen == 0 and not self.damaged_cell_ids())
        completion_evidence = time_in_range if time_in_range is not None else (1.0 if completed else 0.0)
        goal_proxy = _bounded_unit(0.5 * (final_sortedness / 100.0) + 0.5 * float(completion_evidence))
        if task_family in {"repairable", "fatigue"}:
            remaining_penalty = remaining_frozen / max(1, initial_frozen_count)
            repair_proxy = _bounded_unit((final_sortedness / 100.0) * (1.0 - 0.5 * remaining_penalty) + (0.25 if completed else 0.0))
        else:
            repair_proxy = None
        energy_proxy = float(self.comparison_count + self.swap_count + 2.0 * self.intervention_count)
        denominator = max(1.0, float(self.task["horizon"]) * max(1.0, float(self.task["initialLength"])) * 2.0)
        energy_efficiency = _bounded_unit(1.0 - energy_proxy / denominator)
        values = [goal_proxy, repair_proxy, time_in_range, energy_efficiency]
        controller_score = float(np.mean([value for value in values if value is not None]))
        return _json_ready(
            {
                "completed": bool(completed),
                "repairSuccess": bool(completed) if task_family in {"repairable", "fatigue"} else None,
                "homeostaticMaintenance": time_in_range,
                "finalSortednessPercent": final_sortedness,
                "goalAttainmentProxy": goal_proxy,
                "repairCapabilityProxy": repair_proxy,
                "energyEfficiencyProxy": energy_efficiency,
                "controllerScore": controller_score,
                "delayedGratification": float(dg["delayedGratification"]),
                "delayedGratificationEventCount": int(dg["dgEventCount"]),
                "energyProxy": energy_proxy,
                "activationCount": int(self.activation_count),
                "swapCount": int(self.swap_count),
                "comparisonCount": int(self.comparison_count),
                "interventionCount": int(self.intervention_count),
                "repairInterventionCount": int(self.repair_intervention_count),
                "damageRepairInterventionCount": int(self.damage_repair_intervention_count),
                "blockedMoveAttempts": 0,
                "recoveredCellCount": int(self.repair_intervention_count),
                "remainingFrozenCellCount": int(remaining_frozen),
                "activeDamagedCellCount": int(len(self.damaged_cell_ids())),
                "perturbationCount": int(self.perturbation_count),
                "initialValuesJson": compact_json(self.task["initialValues"]),
                "finalValuesJson": compact_json(self.current_values()),
                "sortednessTraceJson": compact_json(sortedness_series),
                "stateTraceLength": int(len(self.state_trace)),
                "uniqueStateCount": int(len({tuple(state) for state in self.state_trace})),
                "stateTraceJson": compact_json(self.state_trace),
                "oracleRecordCount": int(len(self.oracle_records)),
                "oracleRecordsJson": compact_json(self.oracle_records),
                "oracleAllowed": True,
                "oracleBaseline": True,
                "oracleBaselineLabel": ORACLE_BASELINE_ID,
                "usesGlobalController": True,
                "usesGlobalSortednessSignal": True,
                "usesWholeArrayValues": True,
                "usesWholeArrayRanks": True,
                "usesWholeArrayTargetSignal": True,
                "usesTargetPositionOracle": True,
                "claimBoundary": CENTRALIZED_PROXY_SCOPE_NOTE,
                "centralizedComparisonVersion": CENTRALIZED_COMPARISON_VERSION,
            }
        )


def run_s14_local_condition(policy_spec: CompetencePolicySpec, condition: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result, ticks = run_competence_condition(
        policy_spec,
        condition,
        implementation="e04_s14_local_controller_cpu_reference",
        research_step_id="S14",
    )
    score = float(result.get("conditionCompetenceCompositeProxy", np.nan))
    result.update(
        {
            "pairKey": condition["pairKey"],
            "controllerKind": "local",
            "controllerScore": score,
            "interventionCount": 0,
            "repairInterventionCount": 0,
            "damageRepairInterventionCount": 0,
            "oracleAllowed": False,
            "oracleBaseline": False,
            "oracleBaselineLabel": None,
            "usesGlobalController": False,
            "claimBoundary": CENTRALIZED_PROXY_SCOPE_NOTE,
            "centralizedComparisonVersion": CENTRALIZED_COMPARISON_VERSION,
        }
    )
    updated_ticks = []
    for tick in ticks:
        row = dict(tick)
        row.update(
            {
                "pairKey": condition["pairKey"],
                "controllerKind": "local",
                "claimBoundary": CENTRALIZED_PROXY_SCOPE_NOTE,
                "centralizedComparisonVersion": CENTRALIZED_COMPARISON_VERSION,
            }
        )
        updated_ticks.append(_json_ready(row))
    return _json_ready(result), updated_ticks


def run_s14_global_condition(condition: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task = json.loads(str(condition["taskSpecJson"]))
    sim = CentralizedOracleRepairSimulator(task, scheduler_seed=int(condition["schedulerSeed"]))
    result, ticks = sim.run()
    result.update({key: condition[key] for key in condition.keys()})
    result.update(
        {
            "controllerKind": "global_oracle",
            "policyId": CENTRALIZED_ORACLE_CONTROLLER_ID,
            "policyGroup": CENTRALIZED_CONTROLLER_GROUP,
            "familyKind": "centralized_global_oracle",
            "sourceStep": "S14",
            "claimBoundary": CENTRALIZED_PROXY_SCOPE_NOTE,
            "centralizedComparisonVersion": CENTRALIZED_COMPARISON_VERSION,
            "memoryRepairVersion": MEMORY_REPAIR_VERSION,
            "signalRepairVersion": SIGNAL_REPAIR_VERSION,
            "repairRepairVersion": REPAIR_REPAIR_VERSION,
            "fatigueDamageVersion": FATIGUE_DAMAGE_VERSION,
            "localLearningVersion": LOCAL_LEARNING_VERSION,
            "homeostasisBenchmarkVersion": HOMEOSTASIS_BENCHMARK_VERSION,
            "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
        }
    )
    updated_ticks = []
    for tick in ticks:
        row = dict(tick)
        row.update(
            {
                "conditionId": condition["conditionId"],
                "pairKey": condition["pairKey"],
                "policyId": CENTRALIZED_ORACLE_CONTROLLER_ID,
                "taskId": condition["taskId"],
                "taskFamily": condition["taskFamily"],
                "seedIndex": int(condition["seedIndex"]),
                "controllerKind": "global_oracle",
                "claimBoundary": CENTRALIZED_PROXY_SCOPE_NOTE,
                "centralizedComparisonVersion": CENTRALIZED_COMPARISON_VERSION,
            }
        )
        updated_ticks.append(_json_ready(row))
    return _json_ready(result), updated_ticks


def summarize_local_global_deltas(result_df: pd.DataFrame) -> pd.DataFrame:
    if result_df.empty:
        return pd.DataFrame()
    global_rows = result_df[result_df["controllerKind"] == "global_oracle"].copy()
    local_rows = result_df[result_df["controllerKind"] == "local"].copy()
    rows: list[dict[str, Any]] = []
    global_by_pair = {str(row.pairKey): row for row in global_rows.itertuples(index=False)}
    for local in local_rows.itertuples(index=False):
        global_row = global_by_pair.get(str(local.pairKey))
        if global_row is None:
            continue
        rows.append(
            {
                "comparisonId": stable_id("s14_comparison", [local.policyId, local.pairKey]),
                "pairKey": local.pairKey,
                "taskId": local.taskId,
                "taskFamily": local.taskFamily,
                "taskPanel": local.taskPanel,
                "seedIndex": int(local.seedIndex),
                "localPolicyId": local.policyId,
                "localPolicyGroup": local.policyGroup,
                "localFamilyKind": local.familyKind,
                "globalPolicyId": global_row.policyId,
                "localControllerScore": float(local.controllerScore),
                "globalControllerScore": float(global_row.controllerScore),
                "globalMinusLocalScore": float(global_row.controllerScore - local.controllerScore),
                "localFinalSortednessPercent": float(local.finalSortednessPercent),
                "globalFinalSortednessPercent": float(global_row.finalSortednessPercent),
                "globalMinusLocalFinalSortedness": float(global_row.finalSortednessPercent - local.finalSortednessPercent),
                "localCompleted": bool(local.completed),
                "globalCompleted": bool(global_row.completed),
                "localEnergyProxy": float(local.energyProxy),
                "globalEnergyProxy": float(global_row.energyProxy),
                "globalMinusLocalEnergyProxy": float(global_row.energyProxy - local.energyProxy),
                "localInterventionCount": int(getattr(local, "interventionCount", 0)),
                "globalInterventionCount": int(global_row.interventionCount),
                "pairedByTaskAndSeed": True,
                "taskConfigHash": local.taskConfigHash,
                "globalOracleExplicitlyLabeled": bool(global_row.oracleAllowed and global_row.oracleBaseline and global_row.usesGlobalController),
                "claimBoundary": CENTRALIZED_PROXY_SCOPE_NOTE,
                "centralizedComparisonVersion": CENTRALIZED_COMPARISON_VERSION,
            }
        )
    return pd.DataFrame(_json_ready(rows))


def summarize_centralized_groups(delta_df: pd.DataFrame) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in delta_df.groupby(["localPolicyId", "localPolicyGroup", "localFamilyKind"], dropna=False):
        rows.append(
            {
                "localPolicyId": keys[0],
                "localPolicyGroup": keys[1],
                "localFamilyKind": keys[2],
                "comparisonCount": int(len(group)),
                "taskCount": int(group["taskId"].nunique()),
                "meanLocalControllerScore": float(group["localControllerScore"].mean()),
                "meanGlobalControllerScore": float(group["globalControllerScore"].mean()),
                "meanGlobalMinusLocalScore": float(group["globalMinusLocalScore"].mean()),
                "meanGlobalMinusLocalFinalSortedness": float(group["globalMinusLocalFinalSortedness"].mean()),
                "meanGlobalMinusLocalEnergyProxy": float(group["globalMinusLocalEnergyProxy"].mean()),
                "globalCompletedCount": int(group["globalCompleted"].astype(bool).sum()),
                "localCompletedCount": int(group["localCompleted"].astype(bool).sum()),
                "claimBoundary": CENTRALIZED_PROXY_SCOPE_NOTE,
                "centralizedComparisonVersion": CENTRALIZED_COMPARISON_VERSION,
            }
        )
    return pd.DataFrame(_json_ready(rows)).sort_values("meanGlobalMinusLocalScore", ascending=False, kind="mergesort").reset_index(drop=True)


def fairness_assumptions() -> list[dict[str, Any]]:
    return [
        {
            "assumptionId": "paired_task_seed",
            "assumption": "Local and global rows are paired by identical taskId, taskConfigHash, seedIndex, schedulerSeed, tieBreakerSeed, and signalRandomSeed where those seeds apply.",
            "limitation": "The centralized controller is deterministic and does not use every stochastic seed internally.",
        },
        {
            "assumptionId": "same_action_budget",
            "assumption": "The centralized controller receives one oracle action per activation: either one direct repair/damage intervention, one adjacent inversion swap, or one wait action.",
            "limitation": "A global scan is not an adjacent local observation and is counted separately through comparison and intervention cost proxies.",
        },
        {
            "assumptionId": "global_oracle_upper_bound",
            "assumption": "The centralized controller can inspect whole-array values, target order, frozen cells, and damaged cells and can directly repair one impaired cell per activation.",
            "limitation": "This is an explicit nonlocal upper-bound comparator, not a biologically local policy.",
        },
        {
            "assumptionId": "energy_proxy_not_equivalent",
            "assumption": "Energy comparisons report simulator operation counts: local swaps/comparisons/blocked moves/signals and global scans/swaps/interventions.",
            "limitation": "These energy proxies are useful for within-simulator accounting but are not physically comparable biological costs.",
        },
        {
            "assumptionId": "multi_task_panel",
            "assumption": "The panel covers sorting, repairable frozen cells, fatigue/damage, and homeostasis tasks.",
            "limitation": "The panel is intentionally small and does not exhaustively map all S09-S13 stress regimes.",
        },
    ]


def validate_centralized_outputs(
    condition_df: pd.DataFrame,
    result_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    group_df: pd.DataFrame,
    fairness_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check_id: str, success: bool, detail: str) -> None:
        rows.append(
            {
                "checkId": check_id,
                "success": bool(success),
                "detail": detail,
                "claimBoundary": CENTRALIZED_PROXY_SCOPE_NOTE,
                "centralizedComparisonVersion": CENTRALIZED_COMPARISON_VERSION,
            }
        )

    add("condition_rows_nonempty", len(condition_df) > 0, f"conditionRows={len(condition_df)}")
    add("result_rows_match_conditions", len(result_df) == len(condition_df), f"resultRows={len(result_df)} conditionRows={len(condition_df)}")
    controller_kinds = set(condition_df["controllerKind"]) if len(condition_df) else set()
    add("local_and_global_controllers_present", {"local", "global_oracle"}.issubset(controller_kinds), f"controllerKinds={sorted(controller_kinds)}")
    global_conditions = condition_df[condition_df["controllerKind"] == "global_oracle"]
    local_conditions = condition_df[condition_df["controllerKind"] == "local"]
    global_labeled = bool(
        len(global_conditions) > 0
        and global_conditions["oracleAllowed"].astype(bool).all()
        and global_conditions["oracleBaseline"].astype(bool).all()
        and global_conditions["usesGlobalController"].astype(bool).all()
        and global_conditions["oracleBaselineLabel"].eq(ORACLE_BASELINE_ID).all()
    )
    add("global_oracle_controller_explicitly_labeled", global_labeled, f"globalConditionRows={len(global_conditions)}")
    local_unlabeled = bool(
        len(local_conditions) > 0
        and (~local_conditions["oracleAllowed"].astype(bool)).all()
        and (~local_conditions["oracleBaseline"].astype(bool)).all()
        and (~local_conditions["usesGlobalController"].astype(bool)).all()
    )
    add("local_rows_not_oracle_labeled", local_unlabeled, f"localConditionRows={len(local_conditions)}")
    add("local_policy_audits_passed", bool(local_conditions["policyAuditSuccess"].astype(bool).all()), "all local policy specs pass the S07 no-oracle audit")
    required_families = {"sorting", "repairable", "fatigue", "homeostasis"}
    families = set(condition_df["taskFamily"]) if len(condition_df) else set()
    add("required_task_families_present", required_families.issubset(families), f"taskFamilies={sorted(families)}")
    pair_counts = condition_df.groupby(["pairKey", "taskId", "seedIndex"], dropna=False)["controllerKind"].nunique()
    add("paired_task_seed_controller_rows_present", bool(len(pair_counts) > 0 and pair_counts.ge(2).all()), f"pairCount={len(pair_counts)}")
    global_pair_count = result_df[result_df["controllerKind"] == "global_oracle"]["pairKey"].nunique()
    expected_delta_count = len(local_conditions)
    add("comparison_rows_for_every_local_condition", len(delta_df) == expected_delta_count, f"deltaRows={len(delta_df)} expectedLocalRows={expected_delta_count} globalPairCount={global_pair_count}")
    task_hash_ok = True
    for _, group in condition_df.groupby("pairKey", dropna=False):
        task_hash_ok = task_hash_ok and group["taskConfigHash"].nunique(dropna=False) == 1
        task_hash_ok = task_hash_ok and group["schedulerSeed"].nunique(dropna=False) == 1
    add("paired_rows_share_task_and_seed_hashes", bool(task_hash_ok), "each pairKey has one taskConfigHash and one schedulerSeed")
    add("fairness_assumptions_documented", len(fairness_df) >= 4 and fairness_df["assumption"].astype(str).str.len().gt(20).all(), f"fairnessRows={len(fairness_df)}")
    explicit_deltas = bool(not delta_df.empty and delta_df["globalOracleExplicitlyLabeled"].astype(bool).all() and delta_df["pairedByTaskAndSeed"].astype(bool).all())
    add("paired_deltas_labeled_and_quantified", explicit_deltas, f"deltaRows={len(delta_df)}")
    add("group_summary_nonempty", len(group_df) > 0, f"groupRows={len(group_df)}")
    scope_ok = bool(
        condition_df["claimBoundary"].astype(str).str.contains("Direct computational local-versus-global controller proxy", regex=False).all()
        and result_df["claimBoundary"].astype(str).str.contains("Direct computational local-versus-global controller proxy", regex=False).all()
        and delta_df["claimBoundary"].astype(str).str.contains("Direct computational local-versus-global controller proxy", regex=False).all()
    )
    add("proxy_scope_boundaries_present", scope_ok, "condition, result, and delta rows carry S14 proxy scope language")
    return pd.DataFrame(rows)


def centralized_replay_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(stable_json(list(rows)).encode("utf-8")).hexdigest()
