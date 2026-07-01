"""Homeostatic perturbation benchmark support for E04 S05."""

from __future__ import annotations

import json
import random
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
from src.e04.fatigue_damage import (
    RELIABILITY_MODES,
    FatigueDamageBenchmarkConfig,
    FatigueDamageController,
    delayed_gratification_from_sortedness,
)
from src.e04.memory_policies import BoundedCellMemoryBank, CellMemoryConfig, memory_policy_for_cell
from src.e04.repairable_frozen import (
    INTERFACE_MODES,
    RepairBenchmarkConfig,
    RepairController,
    patch_repairable_swaps,
)
from src.e04.signaling import SignalBank, SignalConfig, signal_policy_for_cell


HOMEOSTATIC_TASKS = (
    "swap_shocks",
    "turnover_replacement",
    "frozen_damage",
    "mixed_perturbations",
)
HOMEOSTATIC_REPAIR_RULES = ("passive_control", "stuck_control", "nudge_repair")
PERTURBATION_TYPES = ("swap", "delete_insert", "freeze")


@dataclass(frozen=True)
class PerturbationSpec:
    """One scheduled exogenous perturbation."""

    event_step: int
    perturbation_type: str
    params: Mapping[str, Any]
    safe_representation: str

    def __post_init__(self) -> None:
        if self.event_step <= 0:
            raise ValueError("event_step must be positive")
        if self.perturbation_type not in PERTURBATION_TYPES:
            raise ValueError(f"Unsupported perturbation_type: {self.perturbation_type}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_step": int(self.event_step),
            "perturbation_type": self.perturbation_type,
            "params": dict(self.params),
            "safe_representation": self.safe_representation,
        }


@dataclass(frozen=True)
class HomeostaticBenchmarkConfig:
    """Configuration for one S05 homeostatic benchmark run."""

    values: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    algorithm: str = "bubble"
    task_name: str = "swap_shocks"
    repair_rule: str = "nudge_repair"
    interface_mode: str = "full"
    reliability_mode: str = "no_fatigue_control"
    activation_seed: int = 0
    policy_seed: int | None = None
    schedule_seed: int = 0
    max_events: int = 120
    activation_distribution: str = "uniform_active"
    target_sortedness_percent: float = 100.0
    nudge_threshold: int = 2
    fatigue_threshold: int = 3
    recovery_events: int = 4
    damage_threshold: int = 4
    sort_direction: str = "increasing"

    def __post_init__(self) -> None:
        if len(self.values) < 3:
            raise ValueError("homeostatic values must contain at least three cells")
        if self.algorithm not in {"bubble", "insertion", "selection"}:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        if self.task_name not in HOMEOSTATIC_TASKS:
            raise ValueError(f"Unsupported task_name: {self.task_name}")
        if self.repair_rule not in HOMEOSTATIC_REPAIR_RULES:
            raise ValueError(f"Unsupported repair_rule: {self.repair_rule}")
        if self.interface_mode not in INTERFACE_MODES:
            raise ValueError(f"Unsupported interface_mode: {self.interface_mode}")
        if self.reliability_mode not in RELIABILITY_MODES:
            raise ValueError(f"Unsupported reliability_mode: {self.reliability_mode}")
        if self.max_events <= 0:
            raise ValueError("max_events must be positive")
        if not 0.0 <= self.target_sortedness_percent <= 100.0:
            raise ValueError("target_sortedness_percent must be in [0, 100]")

    @property
    def memory_enabled(self) -> bool:
        return self.interface_mode != "no_memory"

    @property
    def signal_enabled(self) -> bool:
        return self.interface_mode != "no_signal"

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": list(self.values),
            "algorithm": self.algorithm,
            "task_name": self.task_name,
            "repair_rule": self.repair_rule,
            "interface_mode": self.interface_mode,
            "reliability_mode": self.reliability_mode,
            "activation_seed": int(self.activation_seed),
            "policy_seed": int(self.policy_seed if self.policy_seed is not None else self.activation_seed),
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
class HomeostaticBenchmarkResult:
    """Compact result for one homeostatic benchmark run."""

    config: HomeostaticBenchmarkConfig
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

    def _offline_metrics(self) -> dict[str, Any]:
        trajectory = [{"event_step": 0, "values": list(self.config.values)}]
        trajectory.extend(
            {"event_step": int(event["event_step"]), "values": list(event["current_values"])}
            for event in self.event_log
            if "current_values" in event
        )
        sortedness_values = [
            sortedness_percent(item["values"], self.config.sort_direction)
            for item in trajectory
        ]
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
            "time_in_target_fraction": float(sum(1 for flag in target_flags if flag) / len(target_flags)) if target_flags else 1.0,
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
            for kind in PERTURBATION_TYPES
        }
        return {
            **self.config.to_dict(),
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
            "final_values_json": json.dumps(list(self.final_values), separators=(",", ":")),
            "final_labels_json": json.dumps(list(self.final_labels), separators=(",", ":")),
            "final_frozen_positions_json": json.dumps(list(self.final_frozen_positions), separators=(",", ":")),
            "schedule_json": json.dumps([item.to_dict() for item in self.schedule], sort_keys=True, separators=(",", ":"), default=str),
            "schedule_sha256": stable_json_sha256([item.to_dict() for item in self.schedule]),
            "perturbation_log_json": json.dumps(list(self.perturbation_log), sort_keys=True, separators=(",", ":"), default=str),
            "fatigue_transition_count": len(self.fatigue_transition_log),
            "fatigue_transition_log_json": json.dumps(list(self.fatigue_transition_log), sort_keys=True, separators=(",", ":"), default=str),
            "impairment_count": len(self.impairment_log),
            "impairment_log_json": json.dumps(list(self.impairment_log), sort_keys=True, separators=(",", ":"), default=str),
            "unfreeze_count": int(self.repair_summary.get("unfreeze_count", 0)),
            "repair_summary_json": json.dumps(dict(self.repair_summary), sort_keys=True, separators=(",", ":"), default=str),
            "reliability_summary_json": json.dumps(dict(self.reliability_summary), sort_keys=True, separators=(",", ":"), default=str),
            "signal_audit_json": json.dumps(dict(self.signal_audit), sort_keys=True, separators=(",", ":"), default=str),
            "uses_global_oracle": bool(self.signal_audit.get("usesGlobalOracle", False)),
            "event_log_sha256": stable_json_sha256(list(self.event_log)),
            "memory_digest": self.memory_digest,
            "signal_digest": self.signal_digest,
        }


def build_homeostatic_schedule(config: HomeostaticBenchmarkConfig) -> tuple[PerturbationSpec, ...]:
    """Build a deterministic perturbation schedule without reading current Sortedness."""

    rng = random.Random(config.schedule_seed)
    length = len(config.values)
    max_index = length - 1

    def swap_spec(event_step: int) -> PerturbationSpec:
        left, right = sorted(rng.sample(range(length), 2))
        return PerturbationSpec(
            event_step=event_step,
            perturbation_type="swap",
            params={"left_position": left, "right_position": right},
            safe_representation="direct_cell_reindex_fixed_length",
        )

    def turnover_spec(event_step: int, offset: int) -> PerturbationSpec:
        delete_position = rng.randrange(length)
        insert_position = rng.randrange(length)
        inserted_value = int(max(config.values) + 1 + offset + rng.randrange(3))
        return PerturbationSpec(
            event_step=event_step,
            perturbation_type="delete_insert",
            params={
                "delete_position": delete_position,
                "insert_position": insert_position,
                "inserted_value": inserted_value,
            },
            safe_representation="fixed_size_delete_insert_reuses_cell_object_with_new_thread_id",
        )

    def freeze_spec(event_step: int) -> PerturbationSpec:
        return PerturbationSpec(
            event_step=event_step,
            perturbation_type="freeze",
            params={"position": rng.randrange(length)},
            safe_representation="mark_current_identity_frozen_for_s03_repair_controller",
        )

    if config.task_name == "swap_shocks":
        specs = [swap_spec(20), swap_spec(50), swap_spec(80)]
    elif config.task_name == "turnover_replacement":
        specs = [turnover_spec(20, 0), turnover_spec(55, 3), swap_spec(90)]
    elif config.task_name == "frozen_damage":
        specs = [freeze_spec(15), swap_spec(35), freeze_spec(70)]
    elif config.task_name == "mixed_perturbations":
        specs = [swap_spec(15), freeze_spec(35), turnover_spec(60, 6), swap_spec(90)]
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"Unsupported task_name: {config.task_name}")

    clipped = []
    for spec in specs:
        if spec.event_step <= config.max_events and max_index >= 0:
            clipped.append(spec)
    return tuple(sorted(clipped, key=lambda item: (item.event_step, item.perturbation_type)))


def _reindex_cells(cells: list[Any]) -> None:
    right_boundary = (len(cells) - 1, 1)
    for idx, cell in enumerate(cells):
        cell.current_position = (idx, 1)
        cell.target_position = (idx, 1)
        cell.left_boundary = (0, 1)
        cell.right_boundary = right_boundary
        if hasattr(cell, "update"):
            cell.update()


def _reset_reliability_identity(controller: FatigueDamageController, old_thread_id: int, new_thread_id: int) -> None:
    for mapping_name in ("state_by_thread", "total_moves", "moves_since_recovery", "rest_events", "energy_by_thread"):
        mapping = getattr(controller, mapping_name)
        if old_thread_id in mapping:
            del mapping[old_thread_id]
    controller.state_by_thread[int(new_thread_id)] = "healthy"


def _apply_perturbation(
    *,
    spec: PerturbationSpec,
    cells: list[Any],
    cell_status: Any,
    repair_controller: RepairController,
    reliability: FatigueDamageController,
    next_thread_id: int,
) -> tuple[dict[str, Any], int]:
    before_values = list(cell_values(cells))
    before_labels = list(cell_labels(cells))
    before_frozen = list(frozen_positions(cells, cell_status))
    params = dict(spec.params)
    if spec.perturbation_type == "swap":
        left = max(0, min(len(cells) - 1, int(params["left_position"])))
        right = max(0, min(len(cells) - 1, int(params["right_position"])))
        cells[left], cells[right] = cells[right], cells[left]
        _reindex_cells(cells)
    elif spec.perturbation_type == "freeze":
        position = max(0, min(len(cells) - 1, int(params["position"])))
        target = cells[position]
        target.status = cell_status.FREEZE
        target.previous_status = cell_status.FREEZE
        repair_controller.frozen_thread_ids.add(int(target.threadID))
        repair_controller.initial_frozen_positions.setdefault(int(target.threadID), int(position))
    elif spec.perturbation_type == "delete_insert":
        delete_position = max(0, min(len(cells) - 1, int(params["delete_position"])))
        insert_position = max(0, min(len(cells) - 1, int(params["insert_position"])))
        inserted_value = int(params["inserted_value"])
        cell = cells.pop(delete_position)
        old_thread_id = int(cell.threadID)
        repair_controller.frozen_thread_ids.discard(old_thread_id)
        repair_controller.initial_frozen_positions.pop(old_thread_id, None)
        cell.threadID = int(next_thread_id)
        next_thread_id += 1
        cell.value = inserted_value
        cell.label = str(cells[0].label if cells else cell.label)
        cell.status = cell_status.ACTIVE
        cell.previous_status = cell_status.ACTIVE
        cell.tried_to_swap_with_frozen = False
        cells.insert(insert_position, cell)
        _reset_reliability_identity(reliability, old_thread_id, int(cell.threadID))
        _reindex_cells(cells)
    else:  # pragma: no cover - guarded by PerturbationSpec
        raise ValueError(f"Unsupported perturbation_type: {spec.perturbation_type}")

    after_values = list(cell_values(cells))
    return (
        {
            **spec.to_dict(),
            "before_values": before_values,
            "after_values": after_values,
            "before_labels": before_labels,
            "after_labels": list(cell_labels(cells)),
            "before_frozen_positions": before_frozen,
            "after_frozen_positions": list(frozen_positions(cells, cell_status)),
            "values_changed": bool(after_values != before_values),
        },
        next_thread_id,
    )


def simulate_homeostatic_benchmark(config: HomeostaticBenchmarkConfig) -> HomeostaticBenchmarkResult:
    """Run one fixed-horizon homeostatic benchmark without policy access to global Sortedness."""

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
            noise_seed=config.policy_seed or config.activation_seed,
        ),
    )
    repair_controller = RepairController(cells, cell_status, config.to_repair_config(), signal_bank)
    if config.repair_rule != "passive_control":
        patch_repairable_swaps(cells, repair_controller)
    reliability = FatigueDamageController(config.to_fatigue_config())
    reliability.initialize(cells)

    schedule = build_homeostatic_schedule(config)
    schedule_by_step: dict[int, list[PerturbationSpec]] = {}
    for spec in schedule:
        schedule_by_step.setdefault(int(spec.event_step), []).append(spec)

    activation_rng = random.Random(config.activation_seed)
    random.seed(config.policy_seed if config.policy_seed is not None else config.activation_seed)
    cursor = 0
    event_log: list[dict[str, Any]] = []
    perturbation_log: list[dict[str, Any]] = []
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
        before_swaps = int(probe.swap_count)
        before_repair_transitions = len(repair_controller.unfreeze_log)
        before_fatigue_transitions = len(reliability.transition_log)
        probe.current_event_step = event_step
        applied_action = "impaired_skip"
        if reliability.can_act(actor):
            if config.signal_enabled:
                step_result = signal_policy_for_cell(actor, signal_bank, memory_bank).step(actor, event_step=event_step)
                applied_action = step_result.memory_result.applied_action.action_type
            else:
                step_result = memory_policy_for_cell(actor, memory_bank).step(actor, event_step=event_step)
                applied_action = step_result.applied_action.action_type
        else:
            reliability.record_impairment(actor, event_step)
        swap_delta = int(probe.swap_count) - before_swaps
        reliability.after_event(actor, swap_delta=swap_delta)
        after_signature = cell_state_signature(cells)
        event_log.append(
            {
                "event_step": int(event_step),
                "activated_position": int(position),
                "activated_thread_id": int(actor.threadID),
                "activated_value_before": int(before_signature[position][1]),
                "applied_action": str(applied_action),
                "swap_delta": int(swap_delta),
                "public_state_changed": bool(after_signature != before_signature),
                "unfreeze_delta": len(repair_controller.unfreeze_log) - before_repair_transitions,
                "fatigue_transition_delta": len(reliability.transition_log) - before_fatigue_transitions,
                "perturbation_count_at_event": len(schedule_by_step.get(event_step, [])),
                "reliability_state_after": reliability.state_for(actor),
                "current_values": list(cell_values(cells)),
                "current_labels": list(cell_labels(cells)),
                "current_frozen_positions": list(frozen_positions(cells, cell_status)),
            }
        )

    return HomeostaticBenchmarkResult(
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
    )


def default_homeostatic_configs(
    *,
    algorithms: Sequence[str] = ("bubble", "insertion", "selection"),
    seeds: Sequence[int] = (9501, 9502),
    max_events: int = 120,
) -> list[HomeostaticBenchmarkConfig]:
    """Return the compact S05 homeostatic baseline matrix."""

    configs: list[HomeostaticBenchmarkConfig] = []
    for algorithm in algorithms:
        for seed_idx, seed in enumerate(seeds):
            for task_name in HOMEOSTATIC_TASKS:
                for repair_rule in HOMEOSTATIC_REPAIR_RULES:
                    for interface_mode in INTERFACE_MODES:
                        for reliability_mode in RELIABILITY_MODES:
                            configs.append(
                                HomeostaticBenchmarkConfig(
                                    algorithm=algorithm,
                                    task_name=task_name,
                                    repair_rule=repair_rule,
                                    interface_mode=interface_mode,
                                    reliability_mode=reliability_mode,
                                    activation_seed=int(seed + 100 * seed_idx),
                                    policy_seed=int(seed + 10_000 + 100 * seed_idx),
                                    schedule_seed=int(seed + 20_000 + 100 * seed_idx),
                                    max_events=max_events,
                                )
                            )
    return configs


def run_homeostatic_matrix(configs: Sequence[HomeostaticBenchmarkConfig]) -> list[HomeostaticBenchmarkResult]:
    return [simulate_homeostatic_benchmark(config) for config in configs]
