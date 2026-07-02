"""Intelligence-like competency benchmarking for E04 S11.

S11 evaluates policy behavior on held-out perturbation schedules and computes
competency scores offline from event traces.  Learned/local policy decisions
consume only the S07 projected local-only view, optionally masked by the S09
memory ablation or S10 communication ablation used for the policy mode.
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
    monotonicity_error_count,
    sortedness_percent,
    stable_json_sha256,
)
from src.e03.policy_interface import PolicyAction, PolicyState, PolicyStepResult, cells_signature, observe_cell
from src.e04.communication_ablations import (
    CommunicationAblationSpec,
    _capacity_record as _s10_capacity_record,
    _capacity_summary as _s10_capacity_summary,
    _emit_allowed_signal_fields as _s10_emit_allowed_signal_fields,
    _sense_policy_signals,
    _signal_summary as _s10_signal_summary,
    apply_communication_ablation,
    default_communication_ablation_specs,
    neighbor_memory_spec,
)
from src.e04.evolutionary_search import (
    EvolutionGenome,
    S08LocalEvolutionPolicy,
    _execute_adjacent_action,
    _feature_audit_summary,
    fitness_from_row,
)
from src.e04.fatigue_damage import RELIABILITY_MODES, FatigueDamageBenchmarkConfig, FatigueDamageController
from src.e04.fatigue_damage import delayed_gratification_from_sortedness
from src.e04.homeostasis import PerturbationSpec, _apply_perturbation
from src.e04.memory_ablations import (
    MemoryAblationSpec,
    _capacity_observation as _s09_capacity_observation,
    _capacity_summary as _s09_capacity_summary,
    apply_memory_ablation,
    default_memory_ablation_specs,
)
from src.e04.memory_policies import BoundedCellMemoryBank, CellMemoryConfig
from src.e04.no_oracle_protocol import (
    ALLOWED_TRAINING_SIGNAL_FIELDS,
    EXCLUDED_TRAINING_SIGNAL_FIELDS,
    LOCAL_ONLY_PROTOCOL_ID,
    project_local_training_observation,
)
from src.e04.repairable_frozen import RepairBenchmarkConfig, RepairController, patch_repairable_swaps
from src.e04.signaling import SignalBank, SignalConfig


S11_PROTOCOL_ID = f"{LOCAL_ONLY_PROTOCOL_ID}:e04_s11_competency_metrics"
S11_POLICY_FAMILY = "s11_competency_policy_comparison"
S11_COMPETENCY_AXES = (
    "heldout_perturbations",
    "transfer_larger_array",
    "alternate_route_barrier",
    "recovery_profile",
    "graceful_degradation",
)
REQUIRED_S11_POLICY_MODES = (
    "original_open_loop_bubble",
    "s08_discovered_no_memory",
    "neighbor_memory",
    "nearest_neighbor_signaling",
    "diffusive_signaling",
    "long_range_scalar_fields",
    "noisy_diffusive_signaling",
)
S11_HELDOUT_SEEDS = (41001, 41002)
S11_DEGRADATION_LEVELS = (0, 1, 2, 3)


@dataclass(frozen=True)
class S11CompetencyConfig:
    """One held-out S11 competency condition."""

    competency_axis: str
    scenario_name: str
    values: tuple[int, ...]
    reliability_mode: str
    activation_seed: int
    policy_seed: int
    schedule_seed: int
    max_events: int
    damage_level: int = 0
    split: str = "s11_heldout"
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
        if self.competency_axis not in S11_COMPETENCY_AXES:
            raise ValueError(f"Unsupported competency_axis: {self.competency_axis}")
        if not self.scenario_name.startswith("s11_"):
            raise ValueError("S11 scenario names must be explicitly held-out and prefixed with s11_")
        if len(self.values) < 4:
            raise ValueError("S11 competency values must include at least four cells")
        if self.reliability_mode not in RELIABILITY_MODES:
            raise ValueError(f"Unsupported reliability_mode: {self.reliability_mode}")
        if self.algorithm != "bubble":
            raise ValueError("S11 uses Bubble-style matched policies only")
        if self.max_events <= 0:
            raise ValueError("max_events must be positive")
        if self.damage_level < 0:
            raise ValueError("damage_level must be non-negative")

    def to_repair_config(self, *, interface_mode: str) -> RepairBenchmarkConfig:
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

    def to_fatigue_config(self, *, interface_mode: str) -> FatigueDamageBenchmarkConfig:
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
            self.competency_axis,
            self.scenario_name,
            self.values,
            self.reliability_mode,
            self.damage_level,
            self.activation_seed,
            self.policy_seed,
            self.schedule_seed,
            self.max_events,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "benchmark_family": "competency",
            "competency_axis": self.competency_axis,
            "scenario_name": self.scenario_name,
            "values": list(self.values),
            "array_size": len(self.values),
            "algorithm": self.algorithm,
            "task_name": self.scenario_name,
            "repair_rule": self.repair_rule,
            "interface_mode": "s11_competency",
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
            "damage_level": int(self.damage_level),
            "sort_direction": self.sort_direction,
        }


@dataclass(frozen=True)
class S11PolicyMode:
    """One policy mechanism compared in S11."""

    name: str
    label: str
    order: int
    mechanism_class: str
    mode_kind: str
    memory_ablation: str | None = None
    communication_ablation: str | None = None
    communication_less_local: bool = False
    s09_s10_supported: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if self.name not in REQUIRED_S11_POLICY_MODES:
            raise ValueError(f"Unsupported S11 policy mode: {self.name}")
        if self.mode_kind not in {"open_loop", "memory_ablation", "communication_ablation"}:
            raise ValueError(f"Unsupported mode_kind: {self.mode_kind}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "order": int(self.order),
            "mechanismClass": self.mechanism_class,
            "modeKind": self.mode_kind,
            "memoryAblation": self.memory_ablation,
            "communicationAblation": self.communication_ablation,
            "communicationLessLocal": bool(self.communication_less_local),
            "s09S10Supported": bool(self.s09_s10_supported),
            "description": self.description,
        }


def default_s11_policy_modes() -> tuple[S11PolicyMode, ...]:
    """Return original, S08, S09-supported, and S10 sensitivity modes."""

    s10_specs = {spec.name: spec for spec in default_communication_ablation_specs()}
    return (
        S11PolicyMode(
            name="original_open_loop_bubble",
            label="Original open-loop",
            order=0,
            mechanism_class="original_open_loop",
            mode_kind="open_loop",
            description="Original public Bubble cell-view policy with no added memory or signaling.",
        ),
        S11PolicyMode(
            name="s08_discovered_no_memory",
            label="S08 discovered",
            order=1,
            mechanism_class="s08_discovered_no_memory",
            mode_kind="memory_ablation",
            memory_ablation="no_memory",
            description="Selected S08 adjacent-action policy with the S09 no-memory mask.",
        ),
        S11PolicyMode(
            name="neighbor_memory",
            label="Neighbor memory",
            order=2,
            mechanism_class="s09_supported_memory",
            mode_kind="communication_ablation",
            communication_ablation="memory_only_no_signal",
            s09_s10_supported=True,
            description="S09-supported neighbor-memory mechanism with S10 no-signal control.",
        ),
        S11PolicyMode(
            name="nearest_neighbor_signaling",
            label="Nearest-neighbor signal",
            order=3,
            mechanism_class="s10_signal_sensitivity",
            mode_kind="communication_ablation",
            communication_ablation="nearest_neighbor_signaling",
            communication_less_local=s10_specs["nearest_neighbor_signaling"].less_local,
            description="S10 nearest-neighbor allowed blocked/frustrated signaling sensitivity arm.",
        ),
        S11PolicyMode(
            name="diffusive_signaling",
            label="Diffusive signal",
            order=4,
            mechanism_class="s10_signal_sensitivity",
            mode_kind="communication_ablation",
            communication_ablation="diffusive_signaling",
            communication_less_local=s10_specs["diffusive_signaling"].less_local,
            description="S10 local diffusive allowed-field signaling sensitivity arm.",
        ),
        S11PolicyMode(
            name="long_range_scalar_fields",
            label="Long-range scalar",
            order=5,
            mechanism_class="s10_less_local_signal_sensitivity",
            mode_kind="communication_ablation",
            communication_ablation="long_range_scalar_fields",
            communication_less_local=s10_specs["long_range_scalar_fields"].less_local,
            description="S10 less-local long-range scalar-field sensitivity arm.",
        ),
        S11PolicyMode(
            name="noisy_diffusive_signaling",
            label="Noisy diffusive signal",
            order=6,
            mechanism_class="s10_signal_sensitivity",
            mode_kind="communication_ablation",
            communication_ablation="noisy_diffusive_signaling",
            communication_less_local=s10_specs["noisy_diffusive_signaling"].less_local,
            description="S10 noisy diffusive allowed-field signaling sensitivity arm.",
        ),
    )


def default_s11_eval_configs(
    *,
    seeds: Sequence[int] = S11_HELDOUT_SEEDS,
    max_events: int = 180,
    transfer_max_events: int = 220,
) -> tuple[S11CompetencyConfig, ...]:
    """Return held-out competency scenarios with seeds outside S08-S10 grids."""

    configs: list[S11CompetencyConfig] = []
    for seed in seeds:
        configs.extend(
            [
                S11CompetencyConfig(
                    competency_axis="heldout_perturbations",
                    scenario_name="s11_unseen_mixed_shocks",
                    values=tuple(range(1, 9)),
                    reliability_mode="fatigue_recovery",
                    activation_seed=seed,
                    policy_seed=seed + 10_000,
                    schedule_seed=seed + 20_000,
                    max_events=max_events,
                ),
                S11CompetencyConfig(
                    competency_axis="transfer_larger_array",
                    scenario_name="s11_larger_array_transfer",
                    values=tuple(range(1, 13)),
                    reliability_mode="cumulative_damage",
                    activation_seed=seed + 100,
                    policy_seed=seed + 10_100,
                    schedule_seed=seed + 20_100,
                    max_events=transfer_max_events,
                ),
                S11CompetencyConfig(
                    competency_axis="alternate_route_barrier",
                    scenario_name="s11_middle_barrier_route",
                    values=tuple(range(1, 11)),
                    reliability_mode="fatigue_recovery",
                    activation_seed=seed + 200,
                    policy_seed=seed + 10_200,
                    schedule_seed=seed + 20_200,
                    max_events=max_events,
                    nudge_threshold=3,
                ),
                S11CompetencyConfig(
                    competency_axis="recovery_profile",
                    scenario_name="s11_late_recovery_shocks",
                    values=tuple(range(1, 9)),
                    reliability_mode="fatigue_recovery",
                    activation_seed=seed + 300,
                    policy_seed=seed + 10_300,
                    schedule_seed=seed + 20_300,
                    max_events=max_events,
                ),
            ]
        )
        for level in S11_DEGRADATION_LEVELS:
            configs.append(
                S11CompetencyConfig(
                    competency_axis="graceful_degradation",
                    scenario_name="s11_damage_ladder",
                    values=tuple(range(1, 9)),
                    reliability_mode="cumulative_damage",
                    activation_seed=seed + 400 + level,
                    policy_seed=seed + 10_400 + level,
                    schedule_seed=seed + 20_400 + level,
                    max_events=max_events,
                    damage_level=level,
                )
            )
    return tuple(configs)


def assert_s11_design(configs: Sequence[S11CompetencyConfig], modes: Sequence[S11PolicyMode]) -> dict[str, Any]:
    """Validate held-out axes, seed ranges, and policy-mode coverage."""

    keys = [config.match_key() for config in configs]
    seeds: set[int] = set()
    for config in configs:
        seeds.update({config.activation_seed, config.policy_seed, config.schedule_seed})
    axis_counts = Counter(config.competency_axis for config in configs)
    degradation_levels = sorted(
        {int(config.damage_level) for config in configs if config.competency_axis == "graceful_degradation"}
    )
    mode_names = tuple(mode.name for mode in modes)
    return {
        "success": bool(
            len(keys) == len(set(keys))
            and set(axis_counts) == set(S11_COMPETENCY_AXES)
            and tuple(mode_names) == REQUIRED_S11_POLICY_MODES
            and degradation_levels == list(S11_DEGRADATION_LEVELS)
            and min(seeds) >= min(S11_HELDOUT_SEEDS)
        ),
        "configCount": len(configs),
        "duplicateConfigKeys": len(keys) - len(set(keys)),
        "competencyAxisCounts": dict(sorted(axis_counts.items())),
        "policyModes": list(mode_names),
        "arraySizes": sorted({len(config.values) for config in configs}),
        "degradationLevels": degradation_levels,
        "minSeed": min(seeds) if seeds else None,
        "heldoutSeedFloor": min(S11_HELDOUT_SEEDS),
    }


def _swap_spec(event_step: int, left: int, right: int, length: int) -> PerturbationSpec:
    left = max(0, min(length - 1, int(left)))
    right = max(0, min(length - 1, int(right)))
    if left == right:
        right = min(length - 1, left + 1) if left < length - 1 else max(0, left - 1)
    left, right = sorted((left, right))
    return PerturbationSpec(
        event_step=int(event_step),
        perturbation_type="swap",
        params={"left_position": left, "right_position": right},
        safe_representation="s11_direct_cell_reindex_fixed_length",
    )


def _freeze_spec(event_step: int, position: int, length: int) -> PerturbationSpec:
    return PerturbationSpec(
        event_step=int(event_step),
        perturbation_type="freeze",
        params={"position": max(0, min(length - 1, int(position)))},
        safe_representation="s11_mark_current_identity_frozen_for_repair_controller",
    )


def _turnover_spec(event_step: int, delete_position: int, insert_position: int, inserted_value: int, length: int) -> PerturbationSpec:
    return PerturbationSpec(
        event_step=int(event_step),
        perturbation_type="delete_insert",
        params={
            "delete_position": max(0, min(length - 1, int(delete_position))),
            "insert_position": max(0, min(length - 1, int(insert_position))),
            "inserted_value": int(inserted_value),
        },
        safe_representation="s11_fixed_size_turnover_reuses_cell_object",
    )


def build_s11_competency_schedule(config: S11CompetencyConfig) -> tuple[PerturbationSpec, ...]:
    """Build S11 held-out schedules without reading current Sortedness."""

    rng = random.Random(config.schedule_seed)
    length = len(config.values)
    high_value = max(config.values) + 5
    if config.competency_axis == "heldout_perturbations":
        specs = [
            _swap_spec(18, rng.randrange(length), rng.randrange(length), length),
            _freeze_spec(47, rng.randrange(length), length),
            _turnover_spec(86, rng.randrange(length), rng.randrange(length), high_value + rng.randrange(4), length),
            _swap_spec(129, rng.randrange(length), rng.randrange(length), length),
        ]
    elif config.competency_axis == "transfer_larger_array":
        specs = [
            _swap_spec(24, 1 + rng.randrange(length - 2), length - 1 - rng.randrange(length // 3), length),
            _freeze_spec(62, rng.randrange(1, length - 1), length),
            _turnover_spec(104, rng.randrange(length), rng.randrange(length), high_value + rng.randrange(6), length),
            _swap_spec(148, rng.randrange(length), rng.randrange(length), length),
            _freeze_spec(188, rng.randrange(1, length - 1), length),
        ]
    elif config.competency_axis == "alternate_route_barrier":
        mid = length // 2
        specs = [
            _freeze_spec(22, mid - 1, length),
            _freeze_spec(34, mid, length),
            _swap_spec(58, mid - 2, mid + 1, length),
            _turnover_spec(96, mid, max(0, mid - 2), high_value + rng.randrange(5), length),
            _swap_spec(132, rng.randrange(0, mid), rng.randrange(mid, length), length),
        ]
    elif config.competency_axis == "recovery_profile":
        specs = [
            _swap_spec(30, rng.randrange(length), rng.randrange(length), length),
            _freeze_spec(74, rng.randrange(length), length),
            _swap_spec(112, rng.randrange(length), rng.randrange(length), length),
            _turnover_spec(144, rng.randrange(length), rng.randrange(length), high_value + rng.randrange(5), length),
        ]
    elif config.competency_axis == "graceful_degradation":
        specs = [
            _swap_spec(42, rng.randrange(length), rng.randrange(length), length),
            _swap_spec(118, rng.randrange(length), rng.randrange(length), length),
        ]
        candidate_positions = list(range(length))
        rng.shuffle(candidate_positions)
        for index in range(int(config.damage_level)):
            specs.append(_freeze_spec(22 + 18 * index, candidate_positions[index % length], length))
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"Unsupported competency axis: {config.competency_axis}")
    clipped = [spec for spec in specs if spec.event_step <= config.max_events]
    return tuple(sorted(clipped, key=lambda item: (item.event_step, item.perturbation_type)))


def _memory_spec_for_mode(mode: S11PolicyMode) -> MemoryAblationSpec | None:
    if mode.mode_kind != "memory_ablation":
        return None
    specs = {spec.name: spec for spec in default_memory_ablation_specs()}
    if mode.memory_ablation not in specs:
        raise ValueError(f"Unsupported memory mode: {mode.memory_ablation}")
    return specs[str(mode.memory_ablation)]


def _communication_spec_for_mode(mode: S11PolicyMode) -> CommunicationAblationSpec | None:
    if mode.mode_kind != "communication_ablation":
        return None
    specs = {spec.name: spec for spec in default_communication_ablation_specs()}
    if mode.communication_ablation not in specs:
        raise ValueError(f"Unsupported communication mode: {mode.communication_ablation}")
    return specs[str(mode.communication_ablation)]


def _empty_signal_summary(mode: S11PolicyMode) -> dict[str, Any]:
    return {
        "usesGlobalOracle": False,
        "communicationAblation": mode.communication_ablation or "none",
        "senseRecordCount": 0,
        "emissionRecordCount": 0,
        "noiseAppliedCount": 0,
        "configuredNoiseStd": 0.0,
        "noiseAbsMaxObserved": 0.0,
        "maxAccessDistanceObserved": 0,
        "allowedSignalMinObserved": 0.0,
        "allowedSignalMaxObserved": 0.0,
        "allowedSignalsFinite": True,
        "allowedSignalFields": list(ALLOWED_TRAINING_SIGNAL_FIELDS),
        "excludedSignalFields": list(EXCLUDED_TRAINING_SIGNAL_FIELDS),
        "excludedSignalFieldAbsMax": {name: 0.0 for name in EXCLUDED_TRAINING_SIGNAL_FIELDS},
        "targetDerivedSignalFieldsZeroed": True,
        "lessLocal": bool(mode.communication_less_local),
    }


def _shannon_binary_entropy(left_count: int, right_count: int) -> float:
    total = int(left_count) + int(right_count)
    if total <= 0:
        return 0.0
    score = 0.0
    for count in (left_count, right_count):
        if count:
            p = float(count) / total
            score -= p * math.log(p, 2)
    return float(score)


def offline_metrics_from_trace(
    *,
    config: S11CompetencyConfig,
    trace_rows: Sequence[Mapping[str, Any]],
    perturbation_log: Sequence[Mapping[str, Any]],
    final_values: Sequence[int],
    fatigue_transition_log: Sequence[Mapping[str, Any]],
    impairment_log: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute S11 competency metrics only from saved trace-like rows."""

    sorted_trace = sorted(trace_rows, key=lambda item: int(item["event_step"]))
    sortedness_values = [float(item["sortedness_percent"]) for item in sorted_trace]
    target_flags = [value >= config.target_sortedness_percent for value in sortedness_values]
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
    for perturbation in perturbation_log:
        start_step = int(perturbation["event_step"])
        recovered_at: int | None = None
        for row, flag in zip(sorted_trace, target_flags, strict=True):
            if int(row["event_step"]) >= start_step and flag:
                recovered_at = int(row["event_step"])
                break
        if recovered_at is None:
            unrecovered += 1
        else:
            recovery_times.append(max(0, recovered_at - start_step))

    left_swaps = sum(1 for row in sorted_trace if row.get("swap_side") == "left" and int(row.get("swap_delta", 0)) > 0)
    right_swaps = sum(1 for row in sorted_trace if row.get("swap_side") == "right" and int(row.get("swap_delta", 0)) > 0)
    route_diversity = _shannon_binary_entropy(left_swaps, right_swaps)
    perturbation_count = max(1, len(perturbation_log))
    recovery_rate = float(len(recovery_times) / perturbation_count)
    final_sorted = float(sortedness_percent(final_values, config.sort_direction)) / 100.0
    time_target = float(sum(1 for flag in target_flags if flag) / len(target_flags)) if target_flags else 1.0
    event_count = max(1, len([row for row in sorted_trace if int(row["event_step"]) > 0]))
    energy_total = sum(int(row.get("swap_delta", 0)) for row in sorted_trace)
    impairment_count = len(impairment_log)
    energy_efficiency = 1.0 - min(1.0, float(energy_total) / event_count)
    damage_efficiency = 1.0 - min(1.0, float(impairment_count) / event_count)
    if config.competency_axis == "alternate_route_barrier":
        axis_score = 0.45 * recovery_rate + 0.25 * route_diversity + 0.30 * final_sorted
    elif config.competency_axis == "recovery_profile":
        axis_score = 0.55 * recovery_rate + 0.25 * time_target + 0.20 * final_sorted
    elif config.competency_axis == "transfer_larger_array":
        axis_score = 0.45 * final_sorted + 0.35 * time_target + 0.20 * recovery_rate
    elif config.competency_axis == "graceful_degradation":
        axis_score = 0.35 * final_sorted + 0.25 * time_target + 0.20 * recovery_rate + 0.20 * damage_efficiency
    else:
        axis_score = 0.35 * time_target + 0.25 * recovery_rate + 0.25 * final_sorted + 0.15 * route_diversity
    competency_score = (
        0.30 * time_target
        + 0.20 * final_sorted
        + 0.20 * recovery_rate
        + 0.12 * route_diversity
        + 0.10 * damage_efficiency
        + 0.08 * energy_efficiency
    )
    dg_metrics = delayed_gratification_from_sortedness(sortedness_values)
    damaged_transitions = sum(1 for item in fatigue_transition_log if item.get("to_state") == "damaged")
    return {
        "initial_sortedness_percent": float(sortedness_values[0]) if sortedness_values else 100.0,
        "final_sortedness_percent": float(sortedness_values[-1]) if sortedness_values else 100.0,
        "min_sortedness_percent": float(min(sortedness_values)) if sortedness_values else 100.0,
        "mean_sortedness_percent": float(sum(sortedness_values) / len(sortedness_values)) if sortedness_values else 100.0,
        "final_monotonicity_error_count": int(monotonicity_error_count(final_values, config.sort_direction)),
        "time_in_target_event_count": int(sum(1 for flag in target_flags if flag)),
        "time_in_target_fraction": time_target,
        "longest_out_of_target_run": int(max(out_runs) if out_runs else 0),
        "recovered_perturbation_count": int(len(recovery_times)),
        "unrecovered_perturbation_count": int(unrecovered),
        "mean_recovery_events": float(sum(recovery_times) / len(recovery_times)) if recovery_times else None,
        "max_recovery_events": int(max(recovery_times)) if recovery_times else None,
        "cumulative_damage_transition_count": int(damaged_transitions),
        "cumulative_impairment_count": int(impairment_count),
        "route_left_successful_swaps": int(left_swaps),
        "route_right_successful_swaps": int(right_swaps),
        "route_diversity_score": float(route_diversity),
        "recovery_rate": float(recovery_rate),
        "energy_efficiency_score": float(energy_efficiency),
        "damage_efficiency_score": float(damage_efficiency),
        "competency_axis_score": float(axis_score),
        "competency_score": float(competency_score),
        "competency_metrics_offline_from_trace": True,
        **{key: value for key, value in dg_metrics.items() if key.startswith("dg_")},
    }


def _trace_row(
    *,
    run_id: str,
    config: S11CompetencyConfig,
    mode: S11PolicyMode,
    source_policy_id: str,
    event_step: int,
    activated_position: int | None,
    activated_thread_id: int | None,
    applied_action: str,
    decision_action: str,
    target_index: int | None,
    swap_side: str | None,
    swap_delta: int,
    frozen_attempt_delta: int,
    public_state_changed: bool,
    unfreeze_delta: int,
    fatigue_transition_delta: int,
    perturbation_count_at_event: int,
    feature_protocol_id: str,
    projected_feature_count: int,
    feature_audit_uses_oracle: bool,
    reliability_state_after: str,
    current_values: Sequence[int],
    current_labels: Sequence[str],
    current_frozen_positions: Sequence[int],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "event_step": int(event_step),
        "split": config.split,
        "competency_axis": config.competency_axis,
        "scenario_name": config.scenario_name,
        "damage_level": int(config.damage_level),
        "array_size": len(config.values),
        "policy_mode": mode.name,
        "policy_mode_label": mode.label,
        "s08_source_policy_id": source_policy_id,
        "activated_position": activated_position,
        "activated_thread_id": activated_thread_id,
        "applied_action": applied_action,
        "decision_action": decision_action,
        "target_index": target_index,
        "swap_side": swap_side,
        "swap_delta": int(swap_delta),
        "frozen_attempt_delta": int(frozen_attempt_delta),
        "public_state_changed": bool(public_state_changed),
        "unfreeze_delta": int(unfreeze_delta),
        "fatigue_transition_delta": int(fatigue_transition_delta),
        "perturbation_count_at_event": int(perturbation_count_at_event),
        "feature_protocol_id": feature_protocol_id,
        "projected_feature_count": int(projected_feature_count),
        "feature_audit_uses_oracle": bool(feature_audit_uses_oracle),
        "reliability_state_after": reliability_state_after,
        "current_values_json": json.dumps(list(current_values), separators=(",", ":")),
        "current_labels_json": json.dumps(list(current_labels), separators=(",", ":")),
        "current_frozen_positions_json": json.dumps(list(current_frozen_positions), separators=(",", ":")),
        "sortedness_percent": float(sortedness_percent(current_values, config.sort_direction)),
        "target_met": bool(sortedness_percent(current_values, config.sort_direction) >= config.target_sortedness_percent),
    }


def _row_from_trace(
    *,
    config: S11CompetencyConfig,
    mode: S11PolicyMode,
    genome: EvolutionGenome,
    run_id: str,
    schedule: Sequence[PerturbationSpec],
    final_values: Sequence[int],
    final_labels: Sequence[str],
    final_frozen_positions: Sequence[int],
    event_count: int,
    stop_reason: str,
    probe: EventTracingStatusProbe,
    perturbation_log: Sequence[Mapping[str, Any]],
    fatigue_transition_log: Sequence[Mapping[str, Any]],
    impairment_log: Sequence[Mapping[str, Any]],
    repair_summary: Mapping[str, Any],
    reliability_summary: Mapping[str, Any],
    signal_summary: Mapping[str, Any],
    feature_summary: Mapping[str, Any],
    capacity_summary: Mapping[str, Any],
    action_counts: Mapping[str, int],
    memory_digest: str,
    signal_digest: str,
    trace_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = offline_metrics_from_trace(
        config=config,
        trace_rows=trace_rows,
        perturbation_log=perturbation_log,
        final_values=final_values,
        fatigue_transition_log=fatigue_transition_log,
        impairment_log=impairment_log,
    )
    perturbation_counts = {
        kind: sum(1 for item in perturbation_log if item["perturbation_type"] == kind)
        for kind in ("swap", "delete_insert", "freeze")
    }
    if mode.mode_kind == "open_loop":
        policy_id = "open_loop_bubble"
        policy_family = "original_open_loop_cell_view"
        candidate_type = "original_open_loop"
        baseline_type = "original_open_loop"
        generation = -1
        population_index = -1
        mutation_seed = None
        mutation_scale = 0.0
        parent_ids_json = "[]"
        parameter_json = "{}"
        policy_input_contract = "Original public BubbleSortCell.move baseline; no learned S07 feature vector."
    else:
        policy_id = genome.policy_id
        policy_family = S11_POLICY_FAMILY
        candidate_type = mode.mechanism_class
        baseline_type = "none"
        generation = int(genome.generation)
        population_index = int(genome.population_index)
        mutation_seed = int(genome.mutation_seed)
        mutation_scale = float(genome.mutation_scale)
        parent_ids_json = json.dumps(list(genome.parent_ids), separators=(",", ":"))
        parameter_json = json.dumps(dict(genome.parameters), sort_keys=True, separators=(",", ":"))
        policy_input_contract = (
            "S07 LocalTrainingObservation masked by S09 MemoryAblationSpec"
            if mode.mode_kind == "memory_ablation"
            else "S07 LocalTrainingObservation masked by S10 CommunicationAblationSpec"
        )
    row = {
        **config.to_dict(),
        "run_id": run_id,
        "policy_id": policy_id,
        "policy_family": policy_family,
        "candidate_type": candidate_type,
        "baseline_type": baseline_type,
        "generation": generation,
        "population_index": population_index,
        "parent_ids_json": parent_ids_json,
        "mutation_seed": mutation_seed,
        "mutation_scale": mutation_scale,
        "protocol_id": S11_PROTOCOL_ID,
        "centralized_baseline": False,
        "uses_global_oracle": bool(signal_summary.get("usesGlobalOracle", False) or feature_summary.get("usesGlobalOracle", False)),
        "eligible_for_local_only_claims": True,
        "event_count": int(event_count),
        "stop_reason": stop_reason,
        "swap_count": int(probe.swap_count),
        "comparison_count": int(probe.compare_and_swap_count),
        "frozen_attempt_count": int(probe.frozen_swap_attempts),
        "energy_total": int(probe.swap_count),
        "perturbation_count": int(len(perturbation_log)),
        "swap_perturbation_count": int(perturbation_counts["swap"]),
        "delete_insert_perturbation_count": int(perturbation_counts["delete_insert"]),
        "freeze_perturbation_count": int(perturbation_counts["freeze"]),
        **metrics,
        "final_values_json": json.dumps(list(final_values), separators=(",", ":")),
        "final_labels_json": json.dumps(list(final_labels), separators=(",", ":")),
        "final_frozen_positions_json": json.dumps(list(final_frozen_positions), separators=(",", ":")),
        "schedule_json": json.dumps([item.to_dict() for item in schedule], sort_keys=True, separators=(",", ":"), default=str),
        "schedule_sha256": stable_json_sha256([item.to_dict() for item in schedule]),
        "perturbation_log_json": json.dumps(list(perturbation_log), sort_keys=True, separators=(",", ":"), default=str),
        "fatigue_transition_count": len(fatigue_transition_log),
        "impairment_count": len(impairment_log),
        "unfreeze_count": int(repair_summary.get("unfreeze_count", 0)),
        "repair_summary_json": json.dumps(dict(repair_summary), sort_keys=True, separators=(",", ":"), default=str),
        "reliability_summary_json": json.dumps(dict(reliability_summary), sort_keys=True, separators=(",", ":"), default=str),
        "signal_audit_json": json.dumps(dict(signal_summary), sort_keys=True, separators=(",", ":"), default=str),
        "signal_observed_json": json.dumps(dict(signal_summary), sort_keys=True, separators=(",", ":"), default=str),
        "feature_audit_summary_json": json.dumps(dict(feature_summary), sort_keys=True, separators=(",", ":"), default=str),
        "memory_capacity_observed_json": json.dumps(dict(capacity_summary), sort_keys=True, separators=(",", ":"), default=str),
        "action_counts_json": json.dumps(dict(sorted(action_counts.items())), sort_keys=True, separators=(",", ":")),
        "event_log_sha256": stable_json_sha256(list(trace_rows)),
        "memory_digest": memory_digest,
        "signal_digest": signal_digest,
        "parameter_json": parameter_json,
        "s08_source_policy_id": genome.policy_id,
        "policy_mode": mode.name,
        "policy_mode_label": mode.label,
        "policy_mode_order": int(mode.order),
        "mechanism_class": mode.mechanism_class,
        "mode_kind": mode.mode_kind,
        "s09_s10_supported_mechanism": bool(mode.s09_s10_supported),
        "memory_ablation": mode.memory_ablation,
        "communication_ablation": mode.communication_ablation,
        "communication_less_local": bool(mode.communication_less_local),
        "policy_input_contract": policy_input_contract,
        "allowed_training_signal_fields_json": json.dumps(list(ALLOWED_TRAINING_SIGNAL_FIELDS), separators=(",", ":")),
        "excluded_training_signal_fields_json": json.dumps(list(EXCLUDED_TRAINING_SIGNAL_FIELDS), separators=(",", ":")),
        "trace_row_count": int(len(trace_rows)),
        "match_key_sha256": stable_json_sha256(
            {"source_policy_id": genome.policy_id, "config": config.to_dict(), "mode": mode.name}
        ),
    }
    row["fitness_score"] = fitness_from_row(row)
    return row


def simulate_s11_competency_run(
    config: S11CompetencyConfig,
    genome: EvolutionGenome,
    mode: S11PolicyMode,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one policy mode on one held-out S11 competency scenario."""

    memory_spec = _memory_spec_for_mode(mode)
    communication_spec = _communication_spec_for_mode(mode)
    if mode.mode_kind == "memory_ablation":
        memory_config = memory_spec.memory_config if memory_spec is not None else CellMemoryConfig(enabled=False)
        signal_config = SignalConfig(enabled=False, diffusion_enabled=False, noise_seed=config.policy_seed)
        interface_mode = "no_memory" if mode.memory_ablation == "no_memory" else "full"
    elif mode.mode_kind == "communication_ablation":
        neighbor_spec = neighbor_memory_spec()
        memory_config = neighbor_spec.memory_config
        assert communication_spec is not None
        signal_config = communication_spec.to_signal_config(noise_seed=config.policy_seed)
        interface_mode = "no_signal" if not communication_spec.signal_enabled else "full"
    else:
        memory_config = CellMemoryConfig(enabled=False)
        signal_config = SignalConfig(enabled=False, diffusion_enabled=False, noise_seed=config.policy_seed)
        interface_mode = "no_memory"

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
    memory_bank = BoundedCellMemoryBank(memory_config)
    signal_bank = SignalBank(len(cells), signal_config)
    repair_controller = RepairController(cells, cell_status, config.to_repair_config(interface_mode=interface_mode), signal_bank)
    patch_repairable_swaps(cells, repair_controller)
    reliability = FatigueDamageController(config.to_fatigue_config(interface_mode=interface_mode))
    reliability.initialize(cells)
    policy = S08LocalEvolutionPolicy(genome)
    schedule = build_s11_competency_schedule(config)
    schedule_by_step: dict[int, list[PerturbationSpec]] = {}
    for item in schedule:
        schedule_by_step.setdefault(int(item.event_step), []).append(item)

    run_id = stable_json_sha256({"config": config.to_dict(), "mode": mode.to_dict(), "sourcePolicyId": genome.policy_id})[:16]
    activation_rng = random.Random(config.activation_seed)
    random.seed(config.policy_seed)
    cursor = 0
    perturbation_log: list[dict[str, Any]] = []
    feature_audits: list[Mapping[str, Any]] = []
    capacity_records: list[Mapping[str, Any]] = []
    signal_sense_records: list[Mapping[str, Any]] = []
    signal_emission_records: list[Mapping[str, Any]] = []
    trace_rows: list[dict[str, Any]] = [
        _trace_row(
            run_id=run_id,
            config=config,
            mode=mode,
            source_policy_id=genome.policy_id,
            event_step=0,
            activated_position=None,
            activated_thread_id=None,
            applied_action="initial_state",
            decision_action="initial_state",
            target_index=None,
            swap_side=None,
            swap_delta=0,
            frozen_attempt_delta=0,
            public_state_changed=False,
            unfreeze_delta=0,
            fatigue_transition_delta=0,
            perturbation_count_at_event=0,
            feature_protocol_id="offline_trace_initial_state",
            projected_feature_count=0,
            feature_audit_uses_oracle=False,
            reliability_state_after="not_applicable",
            current_values=cell_values(cells),
            current_labels=cell_labels(cells),
            current_frozen_positions=frozen_positions(cells, cell_status),
        )
    ]
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
        before_swaps = int(probe.swap_count)
        before_comparisons = int(probe.compare_and_swap_count)
        before_frozen_attempts = int(probe.frozen_swap_attempts)
        before_position = int(actor.current_position[0])
        actor_thread_id = int(actor.threadID)
        probe.current_event_step = event_step
        applied_action = "impaired_skip"
        decision_action = "impaired_skip"
        target_index: int | None = None
        swap_side: str | None = None
        projected_feature_count = 0
        feature_audit_uses_oracle = False
        swap_delta = 0
        frozen_attempt_delta = 0
        if reliability.can_act(actor):
            if mode.mode_kind == "open_loop":
                actor.move()
                after_by_thread = {int(cell.threadID): idx for idx, cell in enumerate(cells)}
                after_position = int(after_by_thread.get(actor_thread_id, before_position))
                swap_delta = int(probe.swap_count) - before_swaps
                frozen_attempt_delta = int(probe.frozen_swap_attempts) - before_frozen_attempts
                target_index = after_position if swap_delta > 0 else None
                swap_side = "left" if after_position < before_position else "right" if after_position > before_position else None
                applied_action = "swap" if swap_delta > 0 else "blocked_swap_attempt" if frozen_attempt_delta > 0 else "wait"
                decision_action = "original_bubble_move"
                action_counts[applied_action] += 1
            else:
                observation = observe_cell(actor)
                memory_before = memory_bank.get(observation.actor_thread_id)
                if mode.mode_kind == "communication_ablation":
                    assert communication_spec is not None
                    sensation, signal_record = _sense_policy_signals(
                        signal_bank,
                        observation,
                        event_step=event_step,
                        spec=communication_spec,
                    )
                    if signal_record is not None:
                        signal_sense_records.append(signal_record)
                    projected = project_local_training_observation(observation, memory_before, sensation)
                    masked = apply_communication_ablation(projected, communication_spec)
                    capacity_records.append(_s10_capacity_record(masked))
                else:
                    assert memory_spec is not None
                    projected = project_local_training_observation(observation, memory_before, None)
                    masked = apply_memory_ablation(projected, memory_spec)
                    capacity_records.append(_s09_capacity_observation(masked))
                decision = policy.select_action(masked)
                feature_audits.append(decision.feature_audit)
                feature_audit_uses_oracle = bool(decision.feature_audit.get("usesGlobalOracle", False))
                projected_feature_count = int(decision.feature_audit.get("featureCount", 0))
                swap_delta, frozen_attempt_delta, target_index = _execute_adjacent_action(actor, decision.action, probe)
                action_counts[decision.action] += 1
                after_public_signature = cells_signature(cells)
                applied_action = "swap" if swap_delta > 0 else "blocked_swap_attempt" if frozen_attempt_delta > 0 else decision.action
                decision_action = decision.action
                swap_side = decision.target_side if swap_delta > 0 else None
                result = PolicyStepResult(
                    behavior=observation.behavior,
                    observation=observation,
                    state_before=PolicyState(
                        ideal_position=None,
                        reverse_direction=masked.reverse_direction,
                        memory={
                            "s11_local_only_protocol": S11_PROTOCOL_ID,
                            "policy_mode": mode.name,
                        },
                    ),
                    proposed_action=PolicyAction(
                        action_type="swap" if target_index is not None else "idle",
                        target_index=target_index,
                        metadata={"target_side": decision.target_side, "policy_mode": mode.name},
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
                if mode.mode_kind == "communication_ablation":
                    assert communication_spec is not None
                    emission_record = _s10_emit_allowed_signal_fields(
                        signal_bank,
                        observation,
                        memory_after,
                        blocked=blocked,
                        event_step=event_step,
                    )
                    if emission_record is not None:
                        signal_emission_records.append(emission_record)
                reliability.after_event(actor, swap_delta=swap_delta)
            if mode.mode_kind == "open_loop":
                reliability.after_event(actor, swap_delta=swap_delta)
        else:
            reliability.record_impairment(actor, event_step)
        comparison_delta = int(probe.compare_and_swap_count) - before_comparisons
        if comparison_delta < 0:  # pragma: no cover - defensive guard
            comparison_delta = 0
        after_signature = cell_state_signature(cells)
        trace_rows.append(
            _trace_row(
                run_id=run_id,
                config=config,
                mode=mode,
                source_policy_id=genome.policy_id,
                event_step=event_step,
                activated_position=int(position),
                activated_thread_id=int(actor_thread_id),
                applied_action=applied_action,
                decision_action=decision_action,
                target_index=target_index,
                swap_side=swap_side,
                swap_delta=int(swap_delta),
                frozen_attempt_delta=int(frozen_attempt_delta),
                public_state_changed=bool(after_signature != before_signature),
                unfreeze_delta=len(repair_controller.unfreeze_log) - before_repair_transitions,
                fatigue_transition_delta=len(reliability.transition_log) - before_fatigue_transitions,
                perturbation_count_at_event=len(schedule_by_step.get(event_step, [])),
                feature_protocol_id=LOCAL_ONLY_PROTOCOL_ID if projected_feature_count else (
                    "original_open_loop_no_s07_feature_vector" if mode.mode_kind == "open_loop" else "not_projected_impaired_or_inactive"
                ),
                projected_feature_count=int(projected_feature_count),
                feature_audit_uses_oracle=bool(feature_audit_uses_oracle),
                reliability_state_after=reliability.state_for(actor),
                current_values=cell_values(cells),
                current_labels=cell_labels(cells),
                current_frozen_positions=frozen_positions(cells, cell_status),
            )
        )

    feature_summary = _feature_audit_summary(feature_audits) if feature_audits else {
        "usesGlobalOracle": False,
        "auditCount": 0,
        "forbiddenTokenHits": {},
        "excludedSignalHits": {},
        "listLikeFeatureValueHits": {},
        "allowedSignalFields": list(ALLOWED_TRAINING_SIGNAL_FIELDS),
        "excludedSignalFields": list(EXCLUDED_TRAINING_SIGNAL_FIELDS),
        "protocolId": LOCAL_ONLY_PROTOCOL_ID,
    }
    if mode.mode_kind == "communication_ablation":
        assert communication_spec is not None
        capacity_summary = _s10_capacity_summary(capacity_records)
        signal_summary = _s10_signal_summary(
            bank=signal_bank,
            spec=communication_spec,
            sense_records=signal_sense_records,
            emission_records=signal_emission_records,
        )
    elif mode.mode_kind == "memory_ablation":
        assert memory_spec is not None
        capacity_summary = _s09_capacity_summary(capacity_records, memory_spec)
        signal_summary = _empty_signal_summary(mode)
    else:
        capacity_summary = {
            "recordCount": 0,
            "capacityViolations": [],
            "note": "Original open-loop baseline has no S09/S10 memory-capacity feature mask.",
        }
        signal_summary = _empty_signal_summary(mode)

    row = _row_from_trace(
        config=config,
        mode=mode,
        genome=genome,
        run_id=run_id,
        schedule=schedule,
        final_values=cell_values(cells),
        final_labels=cell_labels(cells),
        final_frozen_positions=frozen_positions(cells, cell_status),
        event_count=max(0, len(trace_rows) - 1),
        stop_reason=stop_reason,
        probe=probe,
        perturbation_log=perturbation_log,
        fatigue_transition_log=tuple(item.to_dict() for item in reliability.transition_log),
        impairment_log=tuple(item.to_dict() for item in reliability.impairment_log),
        repair_summary=repair_controller.summary(),
        reliability_summary=reliability.summary(),
        signal_summary=signal_summary,
        feature_summary=feature_summary,
        capacity_summary=capacity_summary,
        action_counts=action_counts,
        memory_digest=memory_bank.stable_digest(),
        signal_digest=signal_bank.stable_digest(),
        trace_rows=trace_rows,
    )
    return row, trace_rows


def run_s11_competency_matrix(
    *,
    genomes: Sequence[EvolutionGenome],
    configs: Sequence[S11CompetencyConfig],
    modes: Sequence[S11PolicyMode],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate all selected S08 policies under all S11 modes and configs."""

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for genome in genomes:
        for config in configs:
            for mode in modes:
                row, trace_rows = simulate_s11_competency_run(config, genome, mode)
                rows.append(row)
                traces.extend(trace_rows)
    return rows, traces
