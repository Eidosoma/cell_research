"""Selection-versus-holdout transfer checks for E04 S13."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .competence_proxies import (
    COMPETENCE_PROXY_VERSION,
    CompetencePolicySpec,
    baseline_competence_policies,
    competence_policy_from_s08_candidate,
    local_memory_signal_competence_policy,
    run_competence_condition,
)
from .fatigue import FATIGUE_DAMAGE_VERSION, FatigueDamageConfig
from .homeostasis import HOMEOSTASIS_BENCHMARK_VERSION, stable_hash, stable_json
from .learning import LOCAL_LEARNING_VERSION
from .memory import MEMORY_REPAIR_VERSION, MemoryConfig
from .repair import REPAIR_REPAIR_VERSION, RepairRuleConfig
from .signals import SIGNAL_REPAIR_VERSION, SignalConfig
from .training_constraints import TRAINING_CONSTRAINT_VERSION


OVERFITTING_TRANSFER_VERSION = "e04_s13_overfitting_transfer.v1"
OVERFITTING_PROXY_SCOPE_NOTE = (
    "Direct computational generalization proxy only; not a biological robustness, "
    "learning, or morphogenetic generalization measurement."
)
S13_SPLIT_ROLES = ("selection", "holdout")
S13_SCORE_COLUMN = "conditionCompetenceCompositeProxy"


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


def _scheduled_cells(schedule: Sequence[Mapping[str, Any]], event_type: str) -> tuple[int, ...]:
    return tuple(sorted(int(event["cellId"]) for event in schedule if str(event.get("type")) == event_type and "cellId" in event))


def frozen_placement_signature(
    frozen_positions: Sequence[int],
    schedule: Sequence[Mapping[str, Any]],
    *,
    category: str,
) -> str:
    payload = {
        "category": category,
        "initialFrozenPositions": sorted(int(position) for position in frozen_positions),
        "scheduledFreezeCellIds": list(_scheduled_cells(schedule, "freeze")),
        "scheduledRecoverCellIds": list(_scheduled_cells(schedule, "recover")),
    }
    return stable_hash(payload)


def perturbation_signature(
    schedule: Sequence[Mapping[str, Any]],
    fatigue_config: Mapping[str, Any],
    *,
    regime: str,
    horizon: int,
    activations_per_tick: int,
) -> str:
    payload = {
        "regime": regime,
        "schedule": [dict(event) for event in schedule],
        "fatigueConfig": dict(fatigue_config),
        "horizon": int(horizon),
        "activationsPerTick": int(activations_per_tick),
    }
    return stable_hash(payload)


def recovery_threshold_signature(nudge_threshold: int, signal_threshold: float) -> str:
    return stable_hash({"nudgeThreshold": int(nudge_threshold), "signalThreshold": float(signal_threshold)})


@dataclass(frozen=True)
class OverfittingTransferTask:
    task_id: str
    split_role: str
    task_family: str
    task_panel: str
    transfer_axis: str
    initial_values: tuple[int, ...]
    horizon: int
    frozen_positions: tuple[int, ...] = ()
    perturbation_schedule: tuple[Mapping[str, Any], ...] = ()
    activations_per_tick: int = 14
    sortedness_threshold_percent: float = 95.0
    fatigue_config: Mapping[str, Any] | None = None
    perturbation_regime: str = "none"
    frozen_placement_category: str = "none"
    recovery_nudge_threshold: int = 2
    recovery_signal_threshold: float = 0.75
    signal_noise_std: float = 0.0
    size_regime: str = "selection_size"

    def __post_init__(self) -> None:
        if self.split_role not in S13_SPLIT_ROLES:
            raise ValueError(f"split_role must be one of {S13_SPLIT_ROLES}, got {self.split_role!r}")
        if int(self.recovery_nudge_threshold) < 1:
            raise ValueError("recovery_nudge_threshold must be positive")
        if float(self.recovery_signal_threshold) <= 0.0:
            raise ValueError("recovery_signal_threshold must be positive")
        if float(self.signal_noise_std) < 0.0:
            raise ValueError("signal_noise_std must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        schedule = [dict(event) for event in self.perturbation_schedule]
        fatigue = dict(self.fatigue_config or FatigueDamageConfig("none").to_dict())
        placement_signature = frozen_placement_signature(
            self.frozen_positions,
            schedule,
            category=self.frozen_placement_category,
        )
        perturb_signature = perturbation_signature(
            schedule,
            fatigue,
            regime=self.perturbation_regime,
            horizon=self.horizon,
            activations_per_tick=self.activations_per_tick,
        )
        threshold_signature = recovery_threshold_signature(
            self.recovery_nudge_threshold,
            self.recovery_signal_threshold,
        )
        return {
            "taskId": self.task_id,
            "splitRole": self.split_role,
            "heldOut": self.split_role == "holdout",
            "selectionUsed": self.split_role == "selection",
            "taskFamily": self.task_family,
            "taskPanel": self.task_panel,
            "transferAxis": self.transfer_axis,
            "initialValues": list(self.initial_values),
            "initialLength": int(len(self.initial_values)),
            "horizon": int(self.horizon),
            "frozenPositions": list(self.frozen_positions),
            "scheduledFreezeCellIds": list(_scheduled_cells(schedule, "freeze")),
            "scheduledRecoverCellIds": list(_scheduled_cells(schedule, "recover")),
            "sortednessThresholdPercent": float(self.sortedness_threshold_percent),
            "perturbationSchedule": schedule,
            "activationsPerTick": int(self.activations_per_tick),
            "fatigueConfig": fatigue,
            "perturbationRegime": self.perturbation_regime,
            "perturbationEventCount": int(len(schedule)),
            "perturbationRatePerTick": float(len(schedule) / max(1, int(self.horizon))),
            "frozenPlacementCategory": self.frozen_placement_category,
            "frozenPlacementSignature": placement_signature,
            "recoveryNudgeThreshold": int(self.recovery_nudge_threshold),
            "recoverySignalThreshold": float(self.recovery_signal_threshold),
            "recoveryThresholdSignature": threshold_signature,
            "signalNoiseStd": float(self.signal_noise_std),
            "noiseRegime": f"noise_{float(self.signal_noise_std):.3f}",
            "sizeRegime": self.size_regime,
            "transfer": self.split_role == "holdout",
            "perturbationLevel": self.split_role,
            "perturbationSeverity": float(self.signal_noise_std),
            "scheduleHash": stable_hash(schedule),
            "perturbationSignature": perturb_signature,
            "claimBoundary": OVERFITTING_PROXY_SCOPE_NOTE,
            "overfittingTransferVersion": OVERFITTING_TRANSFER_VERSION,
        }


def build_s13_tasks() -> tuple[OverfittingTransferTask, ...]:
    selection_homeostasis_schedule = (
        {"eventIndex": 0, "tick": 1, "type": "swap", "leftIndex": 2, "rightIndex": 3, "source": "s13_selection_swap"},
        {"eventIndex": 1, "tick": 3, "type": "freeze", "cellId": 4, "source": "s13_selection_freeze_center"},
        {"eventIndex": 2, "tick": 5, "type": "damage", "cellId": 3, "durationTicks": 1, "source": "s13_selection_damage"},
        {"eventIndex": 3, "tick": 7, "type": "recover", "cellId": 4, "source": "s13_selection_recover_center"},
    )
    holdout_homeostasis_schedule = (
        {"eventIndex": 0, "tick": 1, "type": "insert", "position": 0, "value": 99, "source": "s13_holdout_insert"},
        {"eventIndex": 1, "tick": 2, "type": "swap", "leftIndex": 8, "rightIndex": 9, "source": "s13_holdout_swap_edge"},
        {"eventIndex": 2, "tick": 4, "type": "freeze", "cellId": 9, "source": "s13_holdout_freeze_edge"},
        {"eventIndex": 3, "tick": 6, "type": "damage", "cellId": 7, "durationTicks": 3, "source": "s13_holdout_damage"},
        {"eventIndex": 4, "tick": 8, "type": "delete", "position": 0, "source": "s13_holdout_delete"},
        {"eventIndex": 5, "tick": 10, "type": "recover", "cellId": 9, "source": "s13_holdout_recover_edge"},
    )
    return (
        OverfittingTransferTask(
            task_id="selection_repair_center_single_n8",
            split_role="selection",
            task_family="repairable",
            task_panel="s13_selection_repair",
            transfer_axis="selection_repair_threshold",
            initial_values=(8, 1, 7, 2, 6, 3, 5, 4),
            frozen_positions=(3,),
            horizon=112,
            perturbation_regime="selection_static_center_single_frozen",
            frozen_placement_category="center_single",
            recovery_nudge_threshold=2,
            recovery_signal_threshold=0.65,
            signal_noise_std=0.0,
            size_regime="selection_mid_n8",
        ),
        OverfittingTransferTask(
            task_id="selection_homeostasis_swap_freeze_n8",
            split_role="selection",
            task_family="homeostasis",
            task_panel="s13_selection_perturbation",
            transfer_axis="selection_known_perturbations",
            initial_values=(1, 2, 3, 4, 5, 6, 7, 8),
            horizon=11,
            perturbation_schedule=selection_homeostasis_schedule,
            activations_per_tick=15,
            fatigue_config=FatigueDamageConfig(
                "stochastic_damage",
                damage_probability=0.02,
                damage_cooldown_activations=2,
                random_seed=13101,
            ).to_dict(),
            perturbation_regime="selection_swap_freeze_recover_low_damage",
            frozen_placement_category="scheduled_center_single",
            recovery_nudge_threshold=2,
            recovery_signal_threshold=0.65,
            signal_noise_std=0.02,
            size_regime="selection_mid_n8",
        ),
        OverfittingTransferTask(
            task_id="selection_degradation_low_noise_n10",
            split_role="selection",
            task_family="degradation",
            task_panel="s13_selection_noise",
            transfer_axis="selection_low_noise_fatigue",
            initial_values=(10, 1, 9, 2, 8, 3, 7, 4, 6, 5),
            frozen_positions=(4,),
            horizon=126,
            fatigue_config=FatigueDamageConfig(
                "combined",
                movement_threshold=6,
                failed_swap_threshold=5,
                frustration_threshold=5,
                recovery_activations=2,
                damage_probability=0.02,
                damage_cooldown_activations=2,
                random_seed=13102,
            ).to_dict(),
            perturbation_regime="selection_low_noise_mild_fatigue",
            frozen_placement_category="inner_single",
            recovery_nudge_threshold=2,
            recovery_signal_threshold=0.65,
            signal_noise_std=0.02,
            size_regime="selection_mid_n10",
        ),
        OverfittingTransferTask(
            task_id="holdout_repair_edge_pair_n12",
            split_role="holdout",
            task_family="repairable",
            task_panel="s13_holdout_repair",
            transfer_axis="holdout_unseen_size_edge_pair_threshold",
            initial_values=(12, 1, 11, 2, 10, 3, 9, 4, 8, 5, 7, 6),
            frozen_positions=(1, 10),
            horizon=184,
            perturbation_regime="holdout_static_edge_pair_frozen",
            frozen_placement_category="edge_pair",
            recovery_nudge_threshold=4,
            recovery_signal_threshold=1.15,
            signal_noise_std=0.08,
            size_regime="holdout_larger_n12",
        ),
        OverfittingTransferTask(
            task_id="holdout_homeostasis_insert_delete_noise_n12",
            split_role="holdout",
            task_family="homeostasis",
            task_panel="s13_holdout_perturbation",
            transfer_axis="holdout_insert_delete_edge_freeze",
            initial_values=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
            horizon=13,
            perturbation_schedule=holdout_homeostasis_schedule,
            activations_per_tick=18,
            fatigue_config=FatigueDamageConfig(
                "stochastic_damage",
                damage_probability=0.09,
                damage_cooldown_activations=3,
                random_seed=13201,
            ).to_dict(),
            perturbation_regime="holdout_insert_delete_edge_freeze_high_damage",
            frozen_placement_category="scheduled_edge_single_length_change",
            recovery_nudge_threshold=4,
            recovery_signal_threshold=1.15,
            signal_noise_std=0.08,
            size_regime="holdout_larger_n12",
        ),
        OverfittingTransferTask(
            task_id="holdout_degradation_high_noise_n14",
            split_role="holdout",
            task_family="degradation",
            task_panel="s13_holdout_noise",
            transfer_axis="holdout_high_noise_fatigue",
            initial_values=(14, 1, 13, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7),
            frozen_positions=(2, 11),
            horizon=196,
            fatigue_config=FatigueDamageConfig(
                "combined",
                movement_threshold=3,
                failed_swap_threshold=2,
                frustration_threshold=2,
                recovery_activations=3,
                damage_probability=0.12,
                damage_cooldown_activations=3,
                random_seed=13202,
            ).to_dict(),
            perturbation_regime="holdout_high_noise_strong_fatigue",
            frozen_placement_category="asymmetric_pair",
            recovery_nudge_threshold=5,
            recovery_signal_threshold=1.35,
            signal_noise_std=0.12,
            size_regime="holdout_larger_n14",
        ),
    )


def build_s13_policies(evolved_rows: Sequence[Mapping[str, Any]] = (), *, max_evolved: int = 2) -> tuple[CompetencePolicySpec, ...]:
    policies: list[CompetencePolicySpec] = list(baseline_competence_policies())
    policies.append(local_memory_signal_competence_policy())
    for row in evolved_rows:
        policy = competence_policy_from_s08_candidate(row)
        if policy.replayable:
            policies.append(policy)
        if sum(1 for item in policies if item.family_kind == "evolved") >= max_evolved:
            break
    return tuple(policies)


def _task_payload(task: OverfittingTransferTask | Mapping[str, Any]) -> dict[str, Any]:
    return task.to_dict() if isinstance(task, OverfittingTransferTask) else dict(task)


def _thresholded_repair_config(policy: CompetencePolicySpec, task: Mapping[str, Any]) -> RepairRuleConfig:
    base = RepairRuleConfig.from_spec(policy.repair_config or RepairRuleConfig("nudge_count", nudge_threshold=2))
    return RepairRuleConfig(
        variant=base.variant,
        nudge_threshold=int(task["recoveryNudgeThreshold"]),
        signal_channel=base.signal_channel,
        signal_threshold=float(task["recoverySignalThreshold"]),
        elapsed_activations=base.elapsed_activations,
        approach_direction=base.approach_direction,
    )


def condition_rows_for_transfer_policy(
    policy: CompetencePolicySpec,
    tasks: Sequence[OverfittingTransferTask | Mapping[str, Any]],
    *,
    seed_count: int = 3,
    seed_base: int = 16100,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not policy.replayable:
        return rows
    base_signal = SignalConfig.from_spec(policy.signal_config or "no_signal")
    for task_index, task_obj in enumerate(tasks):
        task = _task_payload(task_obj)
        fatigue = FatigueDamageConfig.from_spec(task["fatigueConfig"])
        repair_config = _thresholded_repair_config(policy, task)
        for replicate_index in range(seed_count):
            seed_index = int(replicate_index if task["splitRole"] == "selection" else 100 + replicate_index)
            scheduler_seed = int(seed_base + task_index * 1000 + seed_index)
            tie_seed = int(seed_base + 100_000 + task_index * 1000 + seed_index)
            signal_seed = int(seed_base + 200_000 + task_index * 1000 + seed_index)
            signal_config = SignalConfig.from_spec(
                {
                    **base_signal.to_dict(),
                    "randomSeed": signal_seed,
                    "noiseStd": float(task["signalNoiseStd"]),
                }
            )
            controlled = {
                "policy": policy.to_dict(),
                "task": task,
                "schedulerSeed": scheduler_seed,
                "tieBreakerSeed": tie_seed,
                "signalConfig": signal_config.to_dict(),
                "repairConfig": repair_config.to_dict(),
                "fatigueConfig": fatigue.to_dict(),
            }
            rows.append(
                {
                    "conditionId": stable_id("s13_condition", [policy.policy_id, task["taskId"], seed_index]),
                    "policyId": policy.policy_id,
                    "policyGroup": policy.policy_group,
                    "familyKind": policy.family_kind,
                    "sourceStep": policy.source_step,
                    "sourceCandidateId": policy.source_candidate_id,
                    "taskId": task["taskId"],
                    "taskFamily": task["taskFamily"],
                    "taskPanel": task["taskPanel"],
                    "transferAxis": task["transferAxis"],
                    "splitRole": task["splitRole"],
                    "split": task["splitRole"],
                    "heldOut": bool(task["heldOut"]),
                    "selectionUsed": bool(task["selectionUsed"]),
                    "initialLength": int(task["initialLength"]),
                    "sizeRegime": task["sizeRegime"],
                    "seedIndex": seed_index,
                    "replicateIndex": int(replicate_index),
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
                    "noiseRegime": task["noiseRegime"],
                    "emissionScale": float(signal_config.emission_scale),
                    "repairRuleVariant": repair_config.variant,
                    "recoveryNudgeThreshold": int(repair_config.nudge_threshold),
                    "recoverySignalThreshold": float(repair_config.signal_threshold),
                    "recoveryThresholdSignature": task["recoveryThresholdSignature"],
                    "perturbationRegime": task["perturbationRegime"],
                    "perturbationSignature": task["perturbationSignature"],
                    "perturbationEventCount": int(task["perturbationEventCount"]),
                    "perturbationRatePerTick": float(task["perturbationRatePerTick"]),
                    "frozenPlacementCategory": task["frozenPlacementCategory"],
                    "frozenPlacementSignature": task["frozenPlacementSignature"],
                    "taskSpecJson": compact_json(task),
                    "signalConfigJson": compact_json(signal_config.to_dict()),
                    "repairConfigJson": compact_json(repair_config.to_dict()),
                    "fatigueConfigJson": compact_json(fatigue.to_dict()),
                    "memoryConfigJson": compact_json(policy.memory_config or MemoryConfig("no_memory").to_dict()),
                    "learningConfigJson": compact_json(policy.learning_config) if policy.learning_config is not None else None,
                    "basePolicySpecJson": compact_json(policy.base_policy_spec),
                    "controlledConfigHash": config_hash(controlled),
                    "claimBoundary": OVERFITTING_PROXY_SCOPE_NOTE,
                    "overfittingTransferVersion": OVERFITTING_TRANSFER_VERSION,
                    "competenceProxyVersion": COMPETENCE_PROXY_VERSION,
                    "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
                }
            )
    return rows


def run_transfer_condition(policy_spec: CompetencePolicySpec, condition: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result, ticks = run_competence_condition(
        policy_spec,
        condition,
        implementation="e04_s13_overfitting_transfer_cpu_reference",
        research_step_id="S13",
    )
    score = float(result.get(S13_SCORE_COLUMN, np.nan))
    result.update(
        {
            "conditionTransferScore": score,
            "splitRole": str(condition["splitRole"]),
            "heldOut": bool(condition["heldOut"]),
            "selectionUsed": bool(condition["selectionUsed"]),
            "claimBoundary": OVERFITTING_PROXY_SCOPE_NOTE,
            "overfittingTransferVersion": OVERFITTING_TRANSFER_VERSION,
            "competenceProxyVersion": COMPETENCE_PROXY_VERSION,
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
    for row in ticks:
        tick_row = dict(row)
        tick_row.update(
            {
                "splitRole": str(condition["splitRole"]),
                "heldOut": bool(condition["heldOut"]),
                "selectionUsed": bool(condition["selectionUsed"]),
                "claimBoundary": OVERFITTING_PROXY_SCOPE_NOTE,
                "overfittingTransferVersion": OVERFITTING_TRANSFER_VERSION,
            }
        )
        updated_ticks.append(_json_ready(tick_row))
    return _json_ready(result), updated_ticks


def select_policies_from_selection_results(
    result_df: pd.DataFrame,
    *,
    baseline_per_group: int = 1,
    enhanced_per_group: int = 2,
) -> pd.DataFrame:
    if result_df.empty:
        return pd.DataFrame()
    selection = result_df[result_df["splitRole"] == "selection"].copy()
    if selection.empty:
        return pd.DataFrame()
    score_by_policy = (
        selection.groupby(["policyId", "policyGroup", "familyKind", "sourceStep"], dropna=False)["conditionTransferScore"]
        .mean()
        .reset_index(name="selectionMeanScore")
        .sort_values(["policyGroup", "selectionMeanScore", "policyId"], ascending=[True, False, True], kind="mergesort")
    )
    rows: list[dict[str, Any]] = []
    for group, group_df in score_by_policy.groupby("policyGroup", dropna=False):
        limit = baseline_per_group if str(group) == "baseline" else enhanced_per_group
        chosen = group_df.head(int(limit))
        for rank, row in enumerate(chosen.itertuples(index=False), start=1):
            basis = selection[selection["policyId"] == row.policyId]
            rows.append(
                {
                    "decisionId": stable_id("s13_selection_decision", [row.policyId, row.policyGroup, rank]),
                    "selectedPolicyId": row.policyId,
                    "policyGroup": row.policyGroup,
                    "familyKind": row.familyKind,
                    "sourceStep": row.sourceStep,
                    "selectionRankWithinGroup": int(rank),
                    "selectionOnlyMeanScore": float(row.selectionMeanScore),
                    "selectionConditionCount": int(len(basis)),
                    "selectionConditionIdsHash": hashlib.sha256(
                        stable_json(sorted(str(value) for value in basis["conditionId"])).encode("utf-8")
                    ).hexdigest(),
                    "decisionRule": f"top {limit} policy/policies by mean selection-panel conditionTransferScore within policyGroup={group}",
                    "holdoutResultsUsedForSelection": False,
                    "claimBoundary": OVERFITTING_PROXY_SCOPE_NOTE,
                    "overfittingTransferVersion": OVERFITTING_TRANSFER_VERSION,
                }
            )
    return pd.DataFrame(_json_ready(rows))


def summarize_transfer_gaps(result_df: pd.DataFrame, decision_df: pd.DataFrame) -> pd.DataFrame:
    if result_df.empty:
        return pd.DataFrame()
    selected_ids = set(decision_df["selectedPolicyId"]) if not decision_df.empty else set()
    grouped = (
        result_df.groupby(["policyId", "policyGroup", "familyKind", "sourceStep"], dropna=False)
        .agg(
            selectionMeanScore=("conditionTransferScore", lambda s: float(s[result_df.loc[s.index, "splitRole"] == "selection"].mean())),
            holdoutMeanScore=("conditionTransferScore", lambda s: float(s[result_df.loc[s.index, "splitRole"] == "holdout"].mean())),
            selectionConditionCount=("splitRole", lambda s: int((s == "selection").sum())),
            holdoutConditionCount=("splitRole", lambda s: int((s == "holdout").sum())),
            selectionCompletedRate=("completed", lambda s: float(s[result_df.loc[s.index, "splitRole"] == "selection"].astype(bool).mean())),
            holdoutCompletedRate=("completed", lambda s: float(s[result_df.loc[s.index, "splitRole"] == "holdout"].astype(bool).mean())),
            selectionMeanFinalSortedness=("finalSortednessPercent", lambda s: float(s[result_df.loc[s.index, "splitRole"] == "selection"].mean())),
            holdoutMeanFinalSortedness=("finalSortednessPercent", lambda s: float(s[result_df.loc[s.index, "splitRole"] == "holdout"].mean())),
            selectionMeanEnergyProxy=("energyProxy", lambda s: float(s[result_df.loc[s.index, "splitRole"] == "selection"].mean())),
            holdoutMeanEnergyProxy=("energyProxy", lambda s: float(s[result_df.loc[s.index, "splitRole"] == "holdout"].mean())),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for row in grouped.itertuples(index=False):
        selection_score = float(row.selectionMeanScore)
        holdout_score = float(row.holdoutMeanScore)
        holdout_minus_selection = holdout_score - selection_score
        relative_drop = (selection_score - holdout_score) / max(1e-9, abs(selection_score))
        rows.append(
            {
                **row._asdict(),
                "holdoutMinusSelectionScore": holdout_minus_selection,
                "selectionMinusHoldoutScore": selection_score - holdout_score,
                "relativeSelectionToHoldoutDrop": float(relative_drop),
                "selectedForHoldoutReview": str(row.policyId) in selected_ids,
                "claimBoundary": OVERFITTING_PROXY_SCOPE_NOTE,
                "overfittingTransferVersion": OVERFITTING_TRANSFER_VERSION,
            }
        )
    out = pd.DataFrame(_json_ready(rows))
    if out.empty:
        return out
    out["selectionRankOverall"] = out["selectionMeanScore"].rank(method="min", ascending=False).astype(int)
    out["holdoutRankOverall"] = out["holdoutMeanScore"].rank(method="min", ascending=False).astype(int)
    out["rankShiftHoldoutMinusSelection"] = out["holdoutRankOverall"] - out["selectionRankOverall"]
    return out.sort_values(["selectedForHoldoutReview", "selectionMeanScore"], ascending=[False, False], kind="mergesort").reset_index(drop=True)


def transfer_group_summary(gap_df: pd.DataFrame) -> pd.DataFrame:
    if gap_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for group, group_df in gap_df.groupby("policyGroup", dropna=False):
        selected = group_df[group_df["selectedForHoldoutReview"].astype(bool)]
        source = selected if not selected.empty else group_df
        rows.append(
            {
                "policyGroup": group,
                "policyCount": int(group_df["policyId"].nunique()),
                "selectedPolicyCount": int(selected["policyId"].nunique()),
                "meanSelectionScore": float(source["selectionMeanScore"].mean()),
                "meanHoldoutScore": float(source["holdoutMeanScore"].mean()),
                "meanHoldoutMinusSelectionScore": float(source["holdoutMinusSelectionScore"].mean()),
                "meanRelativeDrop": float(source["relativeSelectionToHoldoutDrop"].mean()),
                "claimBoundary": OVERFITTING_PROXY_SCOPE_NOTE,
                "overfittingTransferVersion": OVERFITTING_TRANSFER_VERSION,
            }
        )
    return pd.DataFrame(_json_ready(rows))


def validate_overfitting_outputs(
    task_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    result_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    gap_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check_id: str, success: bool, detail: str) -> None:
        rows.append(
            {
                "checkId": check_id,
                "success": bool(success),
                "detail": detail,
                "claimBoundary": OVERFITTING_PROXY_SCOPE_NOTE,
                "overfittingTransferVersion": OVERFITTING_TRANSFER_VERSION,
            }
        )

    add("task_rows_nonempty", len(task_df) > 0, f"taskRows={len(task_df)}")
    add("condition_rows_nonempty", len(condition_df) > 0, f"conditionRows={len(condition_df)}")
    add("result_rows_match_conditions", len(result_df) == len(condition_df), f"resultRows={len(result_df)} conditionRows={len(condition_df)}")
    split_roles = set(condition_df["splitRole"]) if len(condition_df) else set()
    add("selection_and_holdout_present", set(S13_SPLIT_ROLES).issubset(split_roles), f"splitRoles={sorted(split_roles)}")
    flag_ok = bool(
        (condition_df["heldOut"].astype(bool) == (condition_df["splitRole"] == "holdout")).all()
        and (condition_df["selectionUsed"].astype(bool) == (condition_df["splitRole"] == "selection")).all()
    )
    add("selection_holdout_flags_consistent", flag_ok, "selection rows have selectionUsed=true; holdout rows have heldOut=true and selectionUsed=false")
    add("no_global_oracle_access", bool((~condition_df["oracleAccessAllowed"].astype(bool)).all()), "oracleAccessAllowed=false for every condition")
    add("policy_audits_passed", bool(condition_df["policyAuditSuccess"].astype(bool).all()), "all replayed policy specs pass S07 no-oracle audit")
    selected_ids = set(decision_df["selectedPolicyId"]) if not decision_df.empty else set()
    result_ids = set(result_df["policyId"]) if not result_df.empty else set()
    decision_ok = bool(
        not decision_df.empty
        and selected_ids.issubset(result_ids)
        and (~decision_df["holdoutResultsUsedForSelection"].astype(bool)).all()
    )
    add("selection_decisions_use_selection_only", decision_ok, f"selectedPolicies={sorted(selected_ids)}")

    selection_conditions = condition_df[condition_df["splitRole"] == "selection"]
    holdout_conditions = condition_df[condition_df["splitRole"] == "holdout"]

    def disjoint(column: str) -> tuple[bool, str]:
        left = set(selection_conditions[column].dropna().map(str))
        right = set(holdout_conditions[column].dropna().map(str))
        return bool(left and right and left.isdisjoint(right)), f"selection={sorted(left)} holdout={sorted(right)}"

    add("seed_indices_separated", *disjoint("seedIndex"))
    for column in ("schedulerSeed", "tieBreakerSeed", "signalRandomSeed"):
        ok, detail = disjoint(column)
        add(f"{column}_separated", ok, detail)
    add("sizes_separated", *disjoint("initialLength"))
    if len(condition_df):
        max_selection = int(selection_conditions["initialLength"].max()) if not selection_conditions.empty else 0
        min_holdout = int(holdout_conditions["initialLength"].min()) if not holdout_conditions.empty else 0
        add("holdout_sizes_larger", bool(min_holdout > max_selection), f"maxSelectionLength={max_selection} minHoldoutLength={min_holdout}")
    add("perturbation_regimes_separated", *disjoint("perturbationSignature"))
    add("frozen_placements_separated", *disjoint("frozenPlacementSignature"))
    add("recovery_nudge_thresholds_separated", *disjoint("recoveryNudgeThreshold"))
    add("recovery_signal_thresholds_separated", *disjoint("recoverySignalThreshold"))
    add("noise_regimes_separated", *disjoint("noiseRegime"))
    if not result_df.empty and selected_ids:
        selected_holdout = result_df[(result_df["policyId"].isin(selected_ids)) & (result_df["splitRole"] == "holdout")]
        add("holdout_results_exist_for_selected_policies", bool(len(selected_holdout) > 0), f"selectedHoldoutRows={len(selected_holdout)}")
    else:
        add("holdout_results_exist_for_selected_policies", False, "selectedHoldoutRows=0")
    finite_gaps = (
        not gap_df.empty
        and gap_df["selectionMeanScore"].notna().all()
        and gap_df["holdoutMeanScore"].notna().all()
        and gap_df["holdoutMinusSelectionScore"].notna().all()
    )
    add("generalization_gaps_quantified", bool(finite_gaps), f"gapRows={len(gap_df)}")
    scope_ok = bool(
        condition_df["claimBoundary"].astype(str).str.contains("Direct computational generalization proxy only", regex=False).all()
        and result_df["claimBoundary"].astype(str).str.contains("Direct computational generalization proxy only", regex=False).all()
        and gap_df["claimBoundary"].astype(str).str.contains("Direct computational generalization proxy only", regex=False).all()
    )
    add("proxy_scope_boundaries_present", scope_ok, "condition, result, and gap rows carry S13 proxy scope language")
    return pd.DataFrame(rows)


def overfitting_replay_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(stable_json(list(rows)).encode("utf-8")).hexdigest()
