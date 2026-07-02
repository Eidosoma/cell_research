"""Communication-mode ablations for E04 S10.

S10 holds policy-visible memory at the S09 ``neighbor_memory`` setting and
varies only the allowed local signaling mode.  Policy actions still consume the
S07 ``LocalTrainingObservation`` projection after an S10 mask.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
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
    stable_json_sha256,
)
from src.e03.policy_interface import PolicyAction, PolicyState, PolicyStepResult, cells_signature, observe_cell
from src.e04.evolutionary_search import (
    EvolutionGenome,
    S08LocalEvolutionPolicy,
    S08TrajectoryResult,
    _execute_adjacent_action,
    _feature_audit_summary,
    fitness_from_row,
)
from src.e04.fatigue_damage import FatigueDamageController
from src.e04.homeostasis import PerturbationSpec, _apply_perturbation, build_homeostatic_schedule
from src.e04.memory_ablations import S09EvaluationConfig, default_memory_ablation_specs, default_s09_eval_configs
from src.e04.memory_policies import BoundedCellMemoryBank, CellMemoryState
from src.e04.no_oracle_protocol import (
    ALLOWED_TRAINING_SIGNAL_FIELDS,
    EXCLUDED_TRAINING_SIGNAL_FIELDS,
    LOCAL_ONLY_PROTOCOL_ID,
    LocalTrainingObservation,
    project_local_training_observation,
)
from src.e04.repairable_frozen import RepairController, patch_repairable_swaps
from src.e04.signaling import LocalSignalValues, SignalBank, SignalConfig, SignalSensation


S10_PROTOCOL_ID = f"{LOCAL_ONLY_PROTOCOL_ID}:e04_s10_communication_ablation"
S10_POLICY_FAMILY = "s08_selected_policy_with_s10_communication_ablation"
S10_MEMORY_BASELINE = "neighbor_memory"
REQUIRED_COMMUNICATION_ABLATIONS = (
    "memory_only_no_signal",
    "nearest_neighbor_signaling",
    "diffusive_signaling",
    "long_range_scalar_fields",
    "noisy_diffusive_signaling",
)


@dataclass(frozen=True)
class CommunicationAblationSpec:
    """One allowed signaling mode for S10."""

    name: str
    label: str
    order: int
    signal_enabled: bool
    access_mode: str
    local_radius: int
    diffusion_enabled: bool
    diffusion_rate: float
    decay: float = 0.0
    noise_std: float = 0.0
    less_local: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if self.name not in REQUIRED_COMMUNICATION_ABLATIONS:
            raise ValueError(f"Unsupported communication ablation: {self.name}")
        if self.access_mode not in {"none", "nearest_neighbor", "diffusive", "long_range", "noisy_diffusive"}:
            raise ValueError(f"Unsupported access_mode: {self.access_mode}")
        if self.local_radius < 0:
            raise ValueError("local_radius must be non-negative")
        if not 0.0 <= self.diffusion_rate <= 0.5:
            raise ValueError("diffusion_rate must be in [0, 0.5]")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")

    def to_signal_config(self, *, noise_seed: int) -> SignalConfig:
        return SignalConfig(
            enabled=self.signal_enabled,
            local_radius=self.local_radius,
            diffusion_enabled=self.diffusion_enabled,
            diffusion_rate=self.diffusion_rate,
            decay=self.decay,
            noise_std=self.noise_std,
            noise_seed=int(noise_seed),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "order": int(self.order),
            "signalEnabled": bool(self.signal_enabled),
            "accessMode": self.access_mode,
            "localRadius": int(self.local_radius),
            "diffusionEnabled": bool(self.diffusion_enabled),
            "diffusionRate": float(self.diffusion_rate),
            "decay": float(self.decay),
            "noiseStd": float(self.noise_std),
            "lessLocal": bool(self.less_local),
            "allowedSignalFields": list(ALLOWED_TRAINING_SIGNAL_FIELDS),
            "excludedSignalFields": list(EXCLUDED_TRAINING_SIGNAL_FIELDS),
            "description": self.description,
        }


def default_communication_ablation_specs() -> tuple[CommunicationAblationSpec, ...]:
    """Return the predeclared S10 signaling modes."""

    return (
        CommunicationAblationSpec(
            name="memory_only_no_signal",
            label="Memory only",
            order=0,
            signal_enabled=False,
            access_mode="none",
            local_radius=0,
            diffusion_enabled=False,
            diffusion_rate=0.0,
            description="S09 neighbor-memory policy-visible control with all signals disabled.",
        ),
        CommunicationAblationSpec(
            name="nearest_neighbor_signaling",
            label="Nearest-neighbor",
            order=1,
            signal_enabled=True,
            access_mode="nearest_neighbor",
            local_radius=1,
            diffusion_enabled=False,
            diffusion_rate=0.0,
            description="Reads allowed blocked/frustrated emissions from the immediate local window only.",
        ),
        CommunicationAblationSpec(
            name="diffusive_signaling",
            label="Diffusive",
            order=2,
            signal_enabled=True,
            access_mode="diffusive",
            local_radius=1,
            diffusion_enabled=True,
            diffusion_rate=0.25,
            description="Reads allowed blocked/frustrated scalar fields after local diffusion.",
        ),
        CommunicationAblationSpec(
            name="long_range_scalar_fields",
            label="Long-range scalar",
            order=3,
            signal_enabled=True,
            access_mode="long_range",
            local_radius=3,
            diffusion_enabled=True,
            diffusion_rate=0.45,
            less_local=True,
            description="Reads allowed diffusive scalar fields across a wider local radius; labeled less local.",
        ),
        CommunicationAblationSpec(
            name="noisy_diffusive_signaling",
            label="Noisy diffusive",
            order=4,
            signal_enabled=True,
            access_mode="noisy_diffusive",
            local_radius=1,
            diffusion_enabled=True,
            diffusion_rate=0.25,
            noise_std=0.05,
            description="Diffusive allowed fields with deterministic Gaussian sensing noise.",
        ),
    )


def neighbor_memory_spec():
    specs = {spec.name: spec for spec in default_memory_ablation_specs()}
    return specs[S10_MEMORY_BASELINE]


def default_s10_eval_configs(
    *,
    max_events: int = 120,
    heldout_max_events: int = 140,
    seeds: Sequence[int] = (19001, 19002),
) -> tuple[S09EvaluationConfig, ...]:
    """Use the S09 matched evaluation grid for direct control reuse."""

    return default_s09_eval_configs(max_events=max_events, heldout_max_events=heldout_max_events, seeds=seeds)


def apply_communication_ablation(
    projected: LocalTrainingObservation,
    spec: CommunicationAblationSpec,
) -> LocalTrainingObservation:
    """Expose S09 neighbor memory plus the selected S10 signal mode."""

    memory_spec = neighbor_memory_spec()
    return LocalTrainingObservation(
        protocol_id=LOCAL_ONLY_PROTOCOL_ID,
        behavior=projected.behavior,
        actor_value=projected.actor_value,
        actor_status=projected.actor_status,
        reverse_direction=projected.reverse_direction,
        left_present=projected.left_present,
        left_value=projected.left_value,
        left_status=projected.left_status,
        right_present=projected.right_present,
        right_value=projected.right_value,
        right_status=projected.right_status,
        at_left_boundary=projected.at_left_boundary,
        at_right_boundary=projected.at_right_boundary,
        last_move_success=projected.last_move_success,
        last_action_type=None,
        time_since_movement=min(int(projected.time_since_movement), memory_spec.time_since_capacity),
        local_frustration=min(int(projected.local_frustration), memory_spec.frustration_capacity),
        failed_swap_count=min(int(projected.failed_swap_count), memory_spec.failed_swap_capacity),
        neighbor_identity_count=min(int(projected.neighbor_identity_count), memory_spec.neighbor_capacity),
        signal_blocked=float(projected.signal_blocked) if spec.signal_enabled else 0.0,
        signal_frustrated=float(projected.signal_frustrated) if spec.signal_enabled else 0.0,
        signal_scope=projected.signal_scope if spec.signal_enabled else "none",
    )


def _local_indices(length: int, actor_position: int, radius: int) -> tuple[int, ...]:
    left = max(0, int(actor_position) - int(radius))
    right = min(int(length) - 1, int(actor_position) + int(radius))
    return tuple(range(left, right + 1))


def _aggregate_local_signals(bank: SignalBank, indices: Sequence[int]) -> dict[str, float]:
    values: dict[str, float] = {}
    for field in ALLOWED_TRAINING_SIGNAL_FIELDS:
        field_values = [bank.local_by_position[int(index)].field(field) for index in indices]
        values[field] = float(max(field_values) if field_values else 0.0)
    return values


def _aggregate_field_signals(bank: SignalBank, indices: Sequence[int], mode: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for field in ALLOWED_TRAINING_SIGNAL_FIELDS:
        field_values = [float(bank.fields[field][int(index)]) for index in indices]
        if not field_values:
            values[field] = 0.0
        elif mode == "mean":
            values[field] = float(sum(field_values) / len(field_values))
        else:
            values[field] = float(field_values[len(field_values) // 2])
    return values


def _sense_policy_signals(
    bank: SignalBank,
    observation: Any,
    *,
    event_step: int,
    spec: CommunicationAblationSpec,
) -> tuple[SignalSensation | None, dict[str, Any] | None]:
    if not spec.signal_enabled or not bank.config.enabled:
        return None, None
    actor_position = int(observation.actor_index)
    if spec.access_mode == "nearest_neighbor":
        indices = _local_indices(bank.length, actor_position, spec.local_radius)
        values = _aggregate_local_signals(bank, indices)
        access_scope = "nearest_neighbor_allowed_signal_window"
    elif spec.access_mode == "long_range":
        indices = _local_indices(bank.length, actor_position, spec.local_radius)
        values = _aggregate_field_signals(bank, indices, "mean")
        access_scope = f"less_local_allowed_scalar_fields_radius_{spec.local_radius}"
    else:
        indices = (actor_position,)
        values = _aggregate_field_signals(bank, indices, "center")
        access_scope = "explicit_allowed_diffusive_fields"

    noise_applied = False
    noise_values: dict[str, float] = {}
    if spec.noise_std > 0.0:
        noise_applied = True
        for field in ALLOWED_TRAINING_SIGNAL_FIELDS:
            noise = bank.rng.gauss(0.0, spec.noise_std)
            values[field] += noise
            noise_values[field] = float(noise)

    sensation = SignalSensation(
        event_step=event_step,
        actor_thread_id=int(observation.actor_thread_id),
        actor_position=actor_position,
        local_signals=(),
        diffusive_fields=values,
        accessed_indices=tuple(indices),
        access_scope=access_scope,
        noise_applied=noise_applied,
    )
    record = {
        "event_step": int(event_step),
        "actor_thread_id": int(observation.actor_thread_id),
        "actor_position": actor_position,
        "accessed_indices": list(indices),
        "max_access_distance": max((abs(int(index) - actor_position) for index in indices), default=0),
        "access_scope": access_scope,
        "signal_values": {field: float(values[field]) for field in ALLOWED_TRAINING_SIGNAL_FIELDS},
        "noise_applied": bool(noise_applied),
        "noise_std": float(spec.noise_std),
        "noise_values": noise_values,
    }
    return sensation, record


def _emit_allowed_signal_fields(
    bank: SignalBank,
    observation: Any,
    memory: CellMemoryState,
    *,
    blocked: bool,
    event_step: int,
) -> dict[str, Any] | None:
    if not bank.config.enabled:
        return None
    position = int(observation.actor_index)
    blocked_value = 1.0 if blocked or memory.last_move_success is False else 0.0
    frustrated_value = max(0.0, min(1.0, float(memory.local_frustration) / 255.0))
    values = LocalSignalValues(blocked=blocked_value, frustrated=frustrated_value)
    bank.set_local_signal(position, values)
    bank.deposit_field(position, values)
    if bank.config.diffusion_enabled:
        bank.diffuse_once()
    return {
        "event_step": int(event_step),
        "source_thread_id": int(observation.actor_thread_id),
        "source_position": position,
        "allowed_signal_values": {
            "blocked": float(blocked_value),
            "frustrated": float(frustrated_value),
        },
        "excluded_signal_values": {name: 0.0 for name in EXCLUDED_TRAINING_SIGNAL_FIELDS},
        "access_scope": "local_result_allowed_fields_only",
    }


def _capacity_record(masked: LocalTrainingObservation) -> dict[str, Any]:
    return {
        "lastMoveSuccessVisible": masked.last_move_success is not None,
        "lastActionTypeVisible": masked.last_action_type is not None,
        "timeSinceMovement": int(masked.time_since_movement),
        "localFrustration": int(masked.local_frustration),
        "failedSwapCount": int(masked.failed_swap_count),
        "neighborIdentityCount": int(masked.neighbor_identity_count),
        "signalBlocked": float(masked.signal_blocked),
        "signalFrustrated": float(masked.signal_frustrated),
        "signalVisible": bool(masked.signal_scope != "none" or masked.signal_blocked or masked.signal_frustrated),
    }


def _capacity_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    memory_spec = neighbor_memory_spec()
    violations: list[str] = []
    if not records:
        return {"recordCount": 0, "capacityViolations": ["no_projected_records"]}
    max_time = max(int(item["timeSinceMovement"]) for item in records)
    max_frustration = max(int(item["localFrustration"]) for item in records)
    max_failed = max(int(item["failedSwapCount"]) for item in records)
    max_neighbor = max(int(item["neighborIdentityCount"]) for item in records)
    if max_time > memory_spec.time_since_capacity:
        violations.append("time_since_movement_exceeds_neighbor_memory_capacity")
    if max_frustration > memory_spec.frustration_capacity:
        violations.append("local_frustration_exceeds_neighbor_memory_capacity")
    if max_failed > memory_spec.failed_swap_capacity:
        violations.append("failed_swap_count_exceeds_neighbor_memory_capacity")
    if max_neighbor > memory_spec.neighbor_capacity:
        violations.append("neighbor_identity_count_exceeds_neighbor_memory_capacity")
    if any(bool(item["lastActionTypeVisible"]) for item in records):
        violations.append("last_action_type_visible")
    return {
        "recordCount": len(records),
        "memoryBaseline": S10_MEMORY_BASELINE,
        "capacityLimit": memory_spec.capacity_limit(),
        "maxObserved": {
            "timeSinceMovement": max_time,
            "localFrustration": max_frustration,
            "failedSwapCount": max_failed,
            "neighborIdentityCount": max_neighbor,
            "signalBlocked": max(float(item["signalBlocked"]) for item in records),
            "signalFrustrated": max(float(item["signalFrustrated"]) for item in records),
        },
        "signalVisibleCount": int(sum(1 for item in records if bool(item["signalVisible"]))),
        "capacityViolations": violations,
    }


def _signal_summary(
    *,
    bank: SignalBank,
    spec: CommunicationAblationSpec,
    sense_records: Sequence[Mapping[str, Any]],
    emission_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    excluded_max = {
        name: max((abs(float(value)) for value in bank.fields[name]), default=0.0)
        for name in EXCLUDED_TRAINING_SIGNAL_FIELDS
    }
    allowed_values: list[float] = []
    max_access_distance = 0
    noise_values: list[float] = []
    for record in sense_records:
        max_access_distance = max(max_access_distance, int(record.get("max_access_distance", 0)))
        signal_values = dict(record.get("signal_values", {}))
        allowed_values.extend(float(signal_values.get(field, 0.0)) for field in ALLOWED_TRAINING_SIGNAL_FIELDS)
        noise_values.extend(float(value) for value in dict(record.get("noise_values", {})).values())
    finite = all(math.isfinite(value) for value in allowed_values + noise_values)
    excluded_nonzero = {name: value for name, value in excluded_max.items() if value > 1e-12}
    return {
        "usesGlobalOracle": bool(excluded_nonzero or not finite),
        "communicationAblation": spec.name,
        "signalConfig": spec.to_dict(),
        "senseRecordCount": len(sense_records),
        "emissionRecordCount": len(emission_records),
        "noiseAppliedCount": int(sum(1 for record in sense_records if bool(record.get("noise_applied", False)))),
        "configuredNoiseStd": float(spec.noise_std),
        "noiseAbsMaxObserved": max((abs(value) for value in noise_values), default=0.0),
        "maxAccessDistanceObserved": int(max_access_distance),
        "allowedSignalMinObserved": min(allowed_values, default=0.0),
        "allowedSignalMaxObserved": max(allowed_values, default=0.0),
        "allowedSignalsFinite": bool(finite),
        "allowedSignalFields": list(ALLOWED_TRAINING_SIGNAL_FIELDS),
        "excludedSignalFields": list(EXCLUDED_TRAINING_SIGNAL_FIELDS),
        "excludedSignalFieldAbsMax": excluded_max,
        "targetDerivedSignalFieldsZeroed": not excluded_nonzero,
        "lessLocal": bool(spec.less_local),
    }


def simulate_communication_ablation(
    config: S09EvaluationConfig,
    genome: EvolutionGenome,
    spec: CommunicationAblationSpec,
) -> dict[str, Any]:
    """Run one selected S08 policy with S09 neighbor memory and one signal mode."""

    memory_spec = neighbor_memory_spec()
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
    memory_bank = BoundedCellMemoryBank(memory_spec.memory_config)
    signal_bank = SignalBank(len(cells), spec.to_signal_config(noise_seed=config.policy_seed))
    repair_controller = RepairController(cells, cell_status, config.to_repair_config(), signal_bank)
    patch_repairable_swaps(cells, repair_controller)
    reliability = FatigueDamageController(config.to_fatigue_config())
    reliability.initialize(cells)
    policy = S08LocalEvolutionPolicy(genome)
    schedule = build_homeostatic_schedule(config.to_homeostatic_config())
    schedule_by_step: dict[int, list[PerturbationSpec]] = {}
    for item in schedule:
        schedule_by_step.setdefault(int(item.event_step), []).append(item)

    activation_rng = random.Random(config.activation_seed)
    random.seed(config.policy_seed)
    cursor = 0
    event_log: list[dict[str, Any]] = []
    perturbation_log: list[dict[str, Any]] = []
    feature_audits: list[Mapping[str, Any]] = []
    capacity_records: list[Mapping[str, Any]] = []
    signal_sense_records: list[Mapping[str, Any]] = []
    signal_emission_records: list[Mapping[str, Any]] = []
    action_counts: Counter[str] = Counter()
    next_thread_id = max(int(cell.threadID) for cell in cells) + 1
    stop_reason = "fixed_horizon_complete"

    for event_step in range(1, config.max_events + 1):
        for item in schedule_by_step.get(event_step, []):
            perturbation_record, next_thread_id = _apply_perturbation(
                spec=item,
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
        swap_delta = 0
        frozen_attempt_delta = 0
        if reliability.can_act(actor):
            observation = observe_cell(actor)
            memory_before = memory_bank.get(observation.actor_thread_id)
            sensation, signal_record = _sense_policy_signals(
                signal_bank,
                observation,
                event_step=event_step,
                spec=spec,
            )
            if signal_record is not None:
                signal_sense_records.append(signal_record)
            projected = project_local_training_observation(observation, memory_before, sensation)
            masked = apply_communication_ablation(projected, spec)
            capacity_records.append(_capacity_record(masked))
            decision = policy.select_action(masked)
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
                    reverse_direction=masked.reverse_direction,
                    memory={
                        "s10_local_only_protocol": S10_PROTOCOL_ID,
                        "memory_baseline": S10_MEMORY_BASELINE,
                        "communication_ablation": spec.name,
                    },
                ),
                proposed_action=PolicyAction(
                    action_type="swap" if target_index is not None else "idle",
                    target_index=target_index,
                    metadata={"target_side": decision.target_side, "communication_ablation": spec.name},
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
            emission_record = _emit_allowed_signal_fields(
                signal_bank,
                observation,
                memory_after,
                blocked=blocked,
                event_step=event_step,
            )
            if emission_record is not None:
                signal_emission_records.append(emission_record)
            reliability.after_event(actor, swap_delta=swap_delta)
        else:
            reliability.record_impairment(actor, event_step)
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

    feature_summary = _feature_audit_summary(feature_audits)
    capacity_summary = _capacity_summary(capacity_records)
    signal_summary = _signal_summary(
        bank=signal_bank,
        spec=spec,
        sense_records=signal_sense_records,
        emission_records=signal_emission_records,
    )
    result = S08TrajectoryResult(
        genome=genome,
        config=config,  # type: ignore[arg-type]
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
        signal_audit=signal_summary,
        event_log=tuple(event_log),
        memory_digest=memory_bank.stable_digest(),
        signal_digest=signal_bank.stable_digest(),
        action_counts=dict(action_counts),
        feature_audit_summary=feature_summary,
    )
    row = result.to_row()
    row.update(
        {
            "protocol_id": S10_PROTOCOL_ID,
            "policy_family": S10_POLICY_FAMILY,
            "candidate_type": "s08_selected_communication_ablation",
            "baseline_type": "none",
            "s08_source_policy_id": genome.policy_id,
            "memory_ablation": S10_MEMORY_BASELINE,
            "s09_memory_baseline": S10_MEMORY_BASELINE,
            "communication_ablation": spec.name,
            "communication_ablation_label": spec.label,
            "communication_ablation_order": int(spec.order),
            "communication_ablation_description": spec.description,
            "communication_less_local": bool(spec.less_local),
            "signal_config_json": json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":")),
            "signal_observed_json": json.dumps(signal_summary, sort_keys=True, separators=(",", ":")),
            "memory_capacity_observed_json": json.dumps(capacity_summary, sort_keys=True, separators=(",", ":")),
            "policy_input_contract": "S07 LocalTrainingObservation masked by S10 CommunicationAblationSpec",
            "allowed_training_signal_fields_json": json.dumps(list(ALLOWED_TRAINING_SIGNAL_FIELDS), separators=(",", ":")),
            "excluded_training_signal_fields_json": json.dumps(list(EXCLUDED_TRAINING_SIGNAL_FIELDS), separators=(",", ":")),
            "centralized_baseline": False,
            "eligible_for_local_only_claims": True,
            "source_reused_from_s09": False,
            "match_key_sha256": stable_json_sha256(
                {
                    "policy_id": genome.policy_id,
                    "config": config.to_dict(),
                }
            ),
        }
    )
    row["uses_global_oracle"] = bool(signal_summary.get("usesGlobalOracle", False) or feature_summary.get("usesGlobalOracle", False))
    row["fitness_score"] = fitness_from_row(row)
    return row


def run_communication_ablation_matrix(
    *,
    genomes: Sequence[EvolutionGenome],
    configs: Sequence[S09EvaluationConfig],
    specs: Sequence[CommunicationAblationSpec],
) -> list[dict[str, Any]]:
    """Evaluate selected genomes under all non-control S10 communication modes."""

    rows: list[dict[str, Any]] = []
    for genome in genomes:
        for config in configs:
            for spec in specs:
                if spec.name == "memory_only_no_signal":
                    continue
                rows.append(simulate_communication_ablation(config, genome, spec))
    return rows
