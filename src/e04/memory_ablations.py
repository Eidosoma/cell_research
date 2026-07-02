"""Memory-capacity ablations for E04 S09.

S09 reuses the selected S08 adjacent-action genomes, but masks their policy
inputs after the S07 local-only projection.  The rich E03 observation is used
only to construct that projection; policy decisions see the bounded ablation
view defined here.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
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
    PARAMETER_NAMES,
    S08LocalEvolutionPolicy,
    S08TrajectoryResult,
    _execute_adjacent_action,
    _feature_audit_summary,
    fitness_from_row,
)
from src.e04.fatigue_damage import (
    RELIABILITY_MODES,
    FatigueDamageBenchmarkConfig,
    FatigueDamageController,
)
from src.e04.homeostasis import (
    HOMEOSTATIC_TASKS,
    HomeostaticBenchmarkConfig,
    PerturbationSpec,
    _apply_perturbation,
    build_homeostatic_schedule,
)
from src.e04.memory_policies import BoundedCellMemoryBank, CellMemoryConfig, CellMemoryState
from src.e04.no_oracle_protocol import (
    ALLOWED_TRAINING_SIGNAL_FIELDS,
    EXCLUDED_TRAINING_SIGNAL_FIELDS,
    LOCAL_ONLY_PROTOCOL_ID,
    LocalTrainingObservation,
    project_local_training_observation,
)
from src.e04.repairable_frozen import RepairBenchmarkConfig, RepairController, patch_repairable_swaps
from src.e04.signaling import LocalSignalValues, SignalBank, SignalConfig, SignalSensation


S09_PROTOCOL_ID = f"{LOCAL_ONLY_PROTOCOL_ID}:e04_s09_memory_ablation"
S09_POLICY_FAMILY = "s08_selected_policy_with_s09_memory_ablation"
REQUIRED_MEMORY_ABLATIONS = (
    "no_memory",
    "one_bit_memory",
    "bounded_counter_memory",
    "neighbor_memory",
    "signal_field_memory",
)
BENCHMARK_FAMILIES = ("repair", "frozen_cell", "homeostatic")


@dataclass(frozen=True)
class MemoryAblationSpec:
    """One policy-visible memory-capacity setting."""

    name: str
    label: str
    order: int
    memory_config: CellMemoryConfig
    signal_enabled: bool
    expose_last_move_success: bool
    expose_time_since_movement: bool
    expose_local_frustration: bool
    expose_failed_swap_count: bool
    expose_neighbor_identity_count: bool
    expose_allowed_signal_fields: bool
    description: str

    def __post_init__(self) -> None:
        if self.name not in REQUIRED_MEMORY_ABLATIONS:
            raise ValueError(f"Unsupported memory ablation: {self.name}")

    @property
    def failed_swap_capacity(self) -> int:
        return int(self.memory_config.failed_swap_capacity if self.expose_failed_swap_count else 0)

    @property
    def neighbor_capacity(self) -> int:
        return int(self.memory_config.neighbor_capacity if self.expose_neighbor_identity_count else 0)

    @property
    def time_since_capacity(self) -> int:
        return int(self.memory_config.max_time_since_movement if self.expose_time_since_movement else 0)

    @property
    def frustration_capacity(self) -> int:
        return int(self.memory_config.max_frustration if self.expose_local_frustration else 0)

    def capacity_limit(self) -> dict[str, Any]:
        return {
            "memoryAblation": self.name,
            "cellMemoryEnabled": bool(self.memory_config.enabled),
            "signalFieldMemoryEnabled": bool(self.signal_enabled and self.expose_allowed_signal_fields),
            "lastMoveSuccessVisible": bool(self.expose_last_move_success),
            "lastActionTypeVisible": False,
            "timeSinceMovementMax": self.time_since_capacity,
            "localFrustrationMax": self.frustration_capacity,
            "failedSwapCapacity": self.failed_swap_capacity,
            "neighborIdentityCapacity": self.neighbor_capacity,
            "allowedSignalFieldsVisible": list(ALLOWED_TRAINING_SIGNAL_FIELDS)
            if self.expose_allowed_signal_fields
            else [],
            "excludedSignalFields": list(EXCLUDED_TRAINING_SIGNAL_FIELDS),
            "forbiddenTargetDerivedFieldsPolicyVisible": [],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "order": int(self.order),
            "memoryConfig": self.memory_config.to_dict(),
            "signalEnabled": bool(self.signal_enabled),
            "capacityLimit": self.capacity_limit(),
            "description": self.description,
        }


def default_memory_ablation_specs() -> tuple[MemoryAblationSpec, ...]:
    """Return the S09 predeclared memory-capacity ladder."""

    return (
        MemoryAblationSpec(
            name="no_memory",
            label="No memory",
            order=0,
            memory_config=CellMemoryConfig(enabled=False),
            signal_enabled=False,
            expose_last_move_success=False,
            expose_time_since_movement=False,
            expose_local_frustration=False,
            expose_failed_swap_count=False,
            expose_neighbor_identity_count=False,
            expose_allowed_signal_fields=False,
            description="No policy-visible cell memory and no persistent signal-field memory.",
        ),
        MemoryAblationSpec(
            name="one_bit_memory",
            label="One-bit memory",
            order=1,
            memory_config=CellMemoryConfig(
                enabled=True,
                failed_swap_capacity=0,
                neighbor_capacity=0,
                max_time_since_movement=0,
                max_frustration=0,
                record_neighbor_identities=False,
            ),
            signal_enabled=False,
            expose_last_move_success=True,
            expose_time_since_movement=False,
            expose_local_frustration=False,
            expose_failed_swap_count=False,
            expose_neighbor_identity_count=False,
            expose_allowed_signal_fields=False,
            description="Only the last-move success bit is policy-visible.",
        ),
        MemoryAblationSpec(
            name="bounded_counter_memory",
            label="Bounded counters",
            order=2,
            memory_config=CellMemoryConfig(
                enabled=True,
                failed_swap_capacity=2,
                neighbor_capacity=0,
                max_time_since_movement=7,
                max_frustration=3,
                record_neighbor_identities=False,
            ),
            signal_enabled=False,
            expose_last_move_success=True,
            expose_time_since_movement=True,
            expose_local_frustration=True,
            expose_failed_swap_count=True,
            expose_neighbor_identity_count=False,
            expose_allowed_signal_fields=False,
            description="Last success plus bounded local counters, with neighbor identities hidden.",
        ),
        MemoryAblationSpec(
            name="neighbor_memory",
            label="Neighbor memory",
            order=3,
            memory_config=CellMemoryConfig(
                enabled=True,
                failed_swap_capacity=2,
                neighbor_capacity=2,
                max_time_since_movement=7,
                max_frustration=3,
                record_neighbor_identities=True,
            ),
            signal_enabled=False,
            expose_last_move_success=True,
            expose_time_since_movement=True,
            expose_local_frustration=True,
            expose_failed_swap_count=True,
            expose_neighbor_identity_count=True,
            expose_allowed_signal_fields=False,
            description="Bounded counters plus two recent immediate-neighbor identity records.",
        ),
        MemoryAblationSpec(
            name="signal_field_memory",
            label="Signal-field memory",
            order=4,
            memory_config=CellMemoryConfig(enabled=False),
            signal_enabled=True,
            expose_last_move_success=False,
            expose_time_since_movement=False,
            expose_local_frustration=False,
            expose_failed_swap_count=False,
            expose_neighbor_identity_count=False,
            expose_allowed_signal_fields=True,
            description="No policy-visible cell memory; persistent allowed blocked/frustrated signal fields only.",
        ),
    )


@dataclass(frozen=True)
class S09EvaluationConfig:
    """One matched S09 memory-ablation evaluation condition."""

    benchmark_family: str
    values: tuple[int, ...]
    task_name: str
    reliability_mode: str
    activation_seed: int
    policy_seed: int
    schedule_seed: int
    max_events: int = 120
    algorithm: str = "bubble"
    repair_rule: str = "nudge_repair"
    activation_distribution: str = "uniform_active"
    target_sortedness_percent: float = 100.0
    nudge_threshold: int = 2
    fatigue_threshold: int = 3
    recovery_events: int = 4
    damage_threshold: int = 4
    sort_direction: str = "increasing"

    def __post_init__(self) -> None:
        if self.benchmark_family not in BENCHMARK_FAMILIES:
            raise ValueError(f"Unsupported benchmark_family: {self.benchmark_family}")
        if self.task_name not in HOMEOSTATIC_TASKS:
            raise ValueError(f"Unsupported task_name: {self.task_name}")
        if self.reliability_mode not in RELIABILITY_MODES:
            raise ValueError(f"Unsupported reliability_mode: {self.reliability_mode}")
        if len(self.values) < 3:
            raise ValueError("values must include at least three cells")
        if self.max_events <= 0:
            raise ValueError("max_events must be positive")

    def to_homeostatic_config(self, *, algorithm: str | None = None, interface_mode: str = "full") -> HomeostaticBenchmarkConfig:
        return HomeostaticBenchmarkConfig(
            values=self.values,
            algorithm=algorithm or self.algorithm,
            task_name=self.task_name,
            repair_rule=self.repair_rule,
            interface_mode=interface_mode,
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

    def to_repair_config(self, *, interface_mode: str = "full") -> RepairBenchmarkConfig:
        return RepairBenchmarkConfig(
            values=self.values,
            algorithm=self.algorithm,
            frozen_indices=(),
            repair_rule=self.repair_rule,
            interface_mode=interface_mode,
            activation_seed=self.activation_seed,
            policy_seed=self.policy_seed,
            max_events=self.max_events,
            activation_distribution=self.activation_distribution,
            nudge_threshold=self.nudge_threshold,
            sort_direction=self.sort_direction,
        )

    def to_fatigue_config(self, *, interface_mode: str = "full") -> FatigueDamageBenchmarkConfig:
        return FatigueDamageBenchmarkConfig(
            values=self.values,
            algorithm=self.algorithm,
            frozen_indices=(),
            repair_rule=self.repair_rule,
            interface_mode=interface_mode,
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

    def match_key(self) -> tuple[Any, ...]:
        return (
            self.benchmark_family,
            self.values,
            self.task_name,
            self.reliability_mode,
            self.activation_seed,
            self.policy_seed,
            self.schedule_seed,
            self.max_events,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": "matched_ablation",
            "benchmark_family": self.benchmark_family,
            "values": list(self.values),
            "array_size": len(self.values),
            "algorithm": self.algorithm,
            "task_name": self.task_name,
            "repair_rule": self.repair_rule,
            "interface_mode": "memory_ablation",
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


def default_s09_eval_configs(
    *,
    max_events: int = 120,
    heldout_max_events: int = 140,
    seeds: Sequence[int] = (19001, 19002),
) -> tuple[S09EvaluationConfig, ...]:
    """Return a compact matched repair, Frozen Cell, and homeostatic matrix."""

    configs: list[S09EvaluationConfig] = []
    for seed in seeds:
        configs.extend(
            [
                S09EvaluationConfig(
                    benchmark_family="repair",
                    values=(1, 2, 3, 4, 5, 6),
                    task_name="frozen_damage",
                    reliability_mode="no_fatigue_control",
                    activation_seed=seed,
                    policy_seed=seed + 10_000,
                    schedule_seed=seed + 20_000,
                    max_events=max_events,
                ),
                S09EvaluationConfig(
                    benchmark_family="frozen_cell",
                    values=(1, 2, 3, 4, 5, 6),
                    task_name="mixed_perturbations",
                    reliability_mode="fatigue_recovery",
                    activation_seed=seed + 100,
                    policy_seed=seed + 10_100,
                    schedule_seed=seed + 20_100,
                    max_events=max_events,
                ),
                S09EvaluationConfig(
                    benchmark_family="homeostatic",
                    values=(1, 2, 3, 4, 5, 6),
                    task_name="swap_shocks",
                    reliability_mode="no_fatigue_control",
                    activation_seed=seed + 200,
                    policy_seed=seed + 10_200,
                    schedule_seed=seed + 20_200,
                    max_events=max_events,
                ),
                S09EvaluationConfig(
                    benchmark_family="homeostatic",
                    values=(1, 2, 3, 4, 5, 6),
                    task_name="turnover_replacement",
                    reliability_mode="fatigue_recovery",
                    activation_seed=seed + 300,
                    policy_seed=seed + 10_300,
                    schedule_seed=seed + 20_300,
                    max_events=max_events,
                ),
                S09EvaluationConfig(
                    benchmark_family="homeostatic",
                    values=(1, 2, 3, 4, 5, 6, 7, 8),
                    task_name="mixed_perturbations",
                    reliability_mode="cumulative_damage",
                    activation_seed=seed + 400,
                    policy_seed=seed + 10_400,
                    schedule_seed=seed + 20_400,
                    max_events=heldout_max_events,
                ),
            ]
        )
    return tuple(configs)


def assert_matched_seed_design(configs: Sequence[S09EvaluationConfig]) -> dict[str, Any]:
    """Audit that each condition has stable, reusable seed keys."""

    keys = [config.match_key() for config in configs]
    duplicates = len(keys) - len(set(keys))
    seed_triples = [(config.activation_seed, config.policy_seed, config.schedule_seed) for config in configs]
    family_counts = Counter(config.benchmark_family for config in configs)
    return {
        "success": duplicates == 0 and set(family_counts) == set(BENCHMARK_FAMILIES),
        "configCount": len(configs),
        "duplicateConfigKeys": duplicates,
        "seedTriples": [list(item) for item in seed_triples],
        "benchmarkFamilyCounts": dict(sorted(family_counts.items())),
        "arraySizes": sorted({len(config.values) for config in configs}),
    }


def load_selected_s08_policies(path: Path, *, limit: int | None = None) -> tuple[EvolutionGenome, ...]:
    """Load selected S08 policy JSONL records as immutable genomes."""

    records: list[EvolutionGenome] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            parameters = {name: float(record["parameters"][name]) for name in PARAMETER_NAMES}
            records.append(
                EvolutionGenome(
                    policy_id=str(record["policy_id"]),
                    generation=int(record["generation"]),
                    population_index=int(record["population_index"]),
                    parent_ids=tuple(str(item) for item in record.get("parent_ids", ())),
                    mutation_seed=int(record["mutation_seed"]),
                    mutation_scale=float(record["mutation_scale"]),
                    parameters=parameters,
                    lineage_note=f"s08_selected_policy_jsonl_line_{line_number}",
                )
            )
            if limit is not None and len(records) >= int(limit):
                break
    if not records:
        raise ValueError(f"No selected S08 policies loaded from {path}")
    return tuple(records)


def apply_memory_ablation(projected: LocalTrainingObservation, spec: MemoryAblationSpec) -> LocalTrainingObservation:
    """Mask projected S07 features down to one S09 memory-capacity setting."""

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
        last_move_success=projected.last_move_success if spec.expose_last_move_success else None,
        last_action_type=None,
        time_since_movement=min(int(projected.time_since_movement), spec.time_since_capacity),
        local_frustration=min(int(projected.local_frustration), spec.frustration_capacity),
        failed_swap_count=min(int(projected.failed_swap_count), spec.failed_swap_capacity),
        neighbor_identity_count=min(int(projected.neighbor_identity_count), spec.neighbor_capacity),
        signal_blocked=float(projected.signal_blocked) if spec.expose_allowed_signal_fields else 0.0,
        signal_frustrated=float(projected.signal_frustrated) if spec.expose_allowed_signal_fields else 0.0,
        signal_scope=projected.signal_scope if spec.expose_allowed_signal_fields else "none",
    )


def _capacity_observation(masked: LocalTrainingObservation) -> dict[str, Any]:
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


def _capacity_summary(records: Sequence[Mapping[str, Any]], spec: MemoryAblationSpec) -> dict[str, Any]:
    if not records:
        return {"recordCount": 0, "capacityLimit": spec.capacity_limit(), "capacityViolations": ["no_projected_records"]}
    violations: list[str] = []
    max_time = max(int(item["timeSinceMovement"]) for item in records)
    max_frustration = max(int(item["localFrustration"]) for item in records)
    max_failed = max(int(item["failedSwapCount"]) for item in records)
    max_neighbor = max(int(item["neighborIdentityCount"]) for item in records)
    if max_time > spec.time_since_capacity:
        violations.append("time_since_movement_exceeds_capacity")
    if max_frustration > spec.frustration_capacity:
        violations.append("local_frustration_exceeds_capacity")
    if max_failed > spec.failed_swap_capacity:
        violations.append("failed_swap_count_exceeds_capacity")
    if max_neighbor > spec.neighbor_capacity:
        violations.append("neighbor_identity_count_exceeds_capacity")
    if any(bool(item["lastActionTypeVisible"]) for item in records):
        violations.append("last_action_type_visible")
    if not spec.expose_last_move_success and any(bool(item["lastMoveSuccessVisible"]) for item in records):
        violations.append("last_move_success_visible_when_hidden")
    if not spec.expose_allowed_signal_fields and any(bool(item["signalVisible"]) for item in records):
        violations.append("signal_fields_visible_when_hidden")
    return {
        "recordCount": len(records),
        "capacityLimit": spec.capacity_limit(),
        "maxObserved": {
            "timeSinceMovement": max_time,
            "localFrustration": max_frustration,
            "failedSwapCount": max_failed,
            "neighborIdentityCount": max_neighbor,
            "signalBlocked": max(float(item["signalBlocked"]) for item in records),
            "signalFrustrated": max(float(item["signalFrustrated"]) for item in records),
        },
        "lastMoveSuccessValueCounts": dict(
            sorted(Counter(str(item["lastMoveSuccessVisible"]) for item in records).items())
        ),
        "signalVisibleCount": int(sum(1 for item in records if bool(item["signalVisible"]))),
        "capacityViolations": violations,
    }


def _sense_allowed_signal_fields(
    bank: SignalBank,
    observation: Any,
    *,
    event_step: int,
    spec: MemoryAblationSpec,
) -> SignalSensation | None:
    if not spec.signal_enabled or not spec.expose_allowed_signal_fields or not bank.config.enabled:
        return None
    actor_position = int(observation.actor_index)
    indices = bank.local_indices_for(actor_position)
    diffusive_fields = {name: float(bank.fields[name][actor_position]) for name in ALLOWED_TRAINING_SIGNAL_FIELDS}
    return SignalSensation(
        event_step=event_step,
        actor_thread_id=int(observation.actor_thread_id),
        actor_position=actor_position,
        local_signals=(),
        diffusive_fields=diffusive_fields,
        accessed_indices=indices,
        access_scope="local_window_plus_explicit_allowed_diffusive_fields",
        noise_applied=False,
    )


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


def _signal_audit(bank: SignalBank, records: Sequence[Mapping[str, Any]], spec: MemoryAblationSpec) -> dict[str, Any]:
    excluded_max = {
        name: max((abs(float(value)) for value in bank.fields[name]), default=0.0)
        for name in EXCLUDED_TRAINING_SIGNAL_FIELDS
    }
    excluded_nonzero = {name: value for name, value in excluded_max.items() if value > 1e-12}
    return {
        "usesGlobalOracle": bool(excluded_nonzero),
        "allowedSignalFields": list(ALLOWED_TRAINING_SIGNAL_FIELDS),
        "excludedSignalFields": list(EXCLUDED_TRAINING_SIGNAL_FIELDS),
        "excludedSignalFieldAbsMax": excluded_max,
        "targetDerivedSignalFieldsZeroed": not excluded_nonzero,
        "emissionRecordCount": len(records),
        "signalFieldMemoryEnabled": bool(spec.signal_enabled and spec.expose_allowed_signal_fields),
        "diffusiveFieldsExplicitlyLabeled": bool(
            not spec.signal_enabled or spec.expose_allowed_signal_fields
        ),
    }


def simulate_memory_ablation(
    config: S09EvaluationConfig,
    genome: EvolutionGenome,
    spec: MemoryAblationSpec,
    *,
    schedule_override: Sequence[PerturbationSpec] | None = None,
) -> dict[str, Any]:
    """Run one S08 selected policy under one S09 memory-capacity ablation."""

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
    memory_bank = BoundedCellMemoryBank(spec.memory_config)
    signal_bank = SignalBank(
        len(cells),
        SignalConfig(
            enabled=spec.signal_enabled,
            local_radius=1,
            diffusion_enabled=spec.signal_enabled,
            noise_seed=config.policy_seed,
        ),
    )
    repair_controller = RepairController(cells, cell_status, config.to_repair_config(), signal_bank)
    patch_repairable_swaps(cells, repair_controller)
    reliability = FatigueDamageController(config.to_fatigue_config())
    reliability.initialize(cells)
    policy = S08LocalEvolutionPolicy(genome)
    schedule = tuple(schedule_override) if schedule_override is not None else build_homeostatic_schedule(config.to_homeostatic_config())
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
    signal_records: list[Mapping[str, Any]] = []
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
            sensation = _sense_allowed_signal_fields(signal_bank, observation, event_step=event_step, spec=spec)
            projected = project_local_training_observation(observation, memory_before, sensation)
            masked = apply_memory_ablation(projected, spec)
            capacity_records.append(_capacity_observation(masked))
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
                        "s09_local_only_protocol": S09_PROTOCOL_ID,
                        "memory_ablation": spec.name,
                    },
                ),
                proposed_action=PolicyAction(
                    action_type="swap" if target_index is not None else "idle",
                    target_index=target_index,
                    metadata={"target_side": decision.target_side, "memory_ablation": spec.name},
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
            signal_record = _emit_allowed_signal_fields(
                signal_bank,
                observation,
                memory_after,
                blocked=blocked,
                event_step=event_step,
            )
            if signal_record is not None:
                signal_records.append(signal_record)
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
    capacity_summary = _capacity_summary(capacity_records, spec)
    signal_audit = _signal_audit(signal_bank, signal_records, spec)
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
        signal_audit=signal_audit,
        event_log=tuple(event_log),
        memory_digest=memory_bank.stable_digest(),
        signal_digest=signal_bank.stable_digest(),
        action_counts=dict(action_counts),
        feature_audit_summary=feature_summary,
    )
    row = result.to_row()
    row.update(
        {
            "protocol_id": S09_PROTOCOL_ID,
            "policy_family": S09_POLICY_FAMILY,
            "candidate_type": "s08_selected_memory_ablation",
            "baseline_type": "none",
            "s08_source_policy_id": genome.policy_id,
            "memory_ablation": spec.name,
            "memory_ablation_label": spec.label,
            "memory_ablation_order": int(spec.order),
            "memory_ablation_description": spec.description,
            "memory_capacity_limit_json": json.dumps(spec.capacity_limit(), sort_keys=True, separators=(",", ":")),
            "memory_capacity_observed_json": json.dumps(capacity_summary, sort_keys=True, separators=(",", ":")),
            "policy_input_contract": "S07 LocalTrainingObservation masked by S09 MemoryAblationSpec",
            "allowed_training_signal_fields_json": json.dumps(list(ALLOWED_TRAINING_SIGNAL_FIELDS), separators=(",", ":")),
            "excluded_training_signal_fields_json": json.dumps(list(EXCLUDED_TRAINING_SIGNAL_FIELDS), separators=(",", ":")),
            "centralized_baseline": False,
            "eligible_for_local_only_claims": True,
            "match_key_sha256": stable_json_sha256(
                {
                    "policy_id": genome.policy_id,
                    "config": config.to_dict(),
                }
            ),
        }
    )
    row["uses_global_oracle"] = bool(signal_audit.get("usesGlobalOracle", False) or feature_summary.get("usesGlobalOracle", False))
    row["fitness_score"] = fitness_from_row(row)
    return row


def run_memory_ablation_matrix(
    *,
    genomes: Sequence[EvolutionGenome],
    configs: Sequence[S09EvaluationConfig],
    specs: Sequence[MemoryAblationSpec],
) -> list[dict[str, Any]]:
    """Evaluate every selected genome under every matched memory ablation."""

    rows: list[dict[str, Any]] = []
    for genome in genomes:
        for config in configs:
            for spec in specs:
                rows.append(simulate_memory_ablation(config, genome, spec))
    return rows
