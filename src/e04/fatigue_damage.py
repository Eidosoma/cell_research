"""Dynamic fatigue and damage benchmark support for E04 S04."""

from __future__ import annotations

import json
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
    monotonicity_error_count,
    sortedness_percent,
    stable_json_sha256,
)
from src.e04.memory_policies import BoundedCellMemoryBank, CellMemoryConfig, memory_policy_for_cell
from src.e04.repairable_frozen import (
    INTERFACE_MODES,
    RepairBenchmarkConfig,
    RepairController,
    patch_repairable_swaps,
    simulate_repair_benchmark,
)
from src.e04.signaling import SignalBank, SignalConfig, signal_policy_for_cell


RELIABILITY_MODES = ("no_fatigue_control", "fatigue_recovery", "cumulative_damage")
S04_REPAIR_RULES = ("passive_control", "stuck_control", "nudge_repair")


def _compact_consecutive(values: Sequence[float]) -> list[float]:
    compact: list[float] = []
    for value in values:
        value = float(value)
        if not compact or compact[-1] != value:
            compact.append(value)
    return compact


def _signed_segments(values: Sequence[float]) -> list[float]:
    compact = _compact_consecutive(values)
    if len(compact) < 2:
        return []
    segments: list[float] = []
    for idx in range(1, len(compact)):
        delta = float(compact[idx] - compact[idx - 1])
        if delta == 0.0:
            continue
        if not segments or segments[-1] * delta <= 0:
            segments.append(delta)
        else:
            segments[-1] += delta
    return segments


def delayed_gratification_from_sortedness(values: Sequence[float]) -> dict[str, Any]:
    """Compute the E01-style DG proxy from an offline Sortedness trajectory."""

    segments = _signed_segments(values)
    events: list[tuple[float, float, float]] = []
    terminal_unrecovered_drop_segments = 0
    for idx, segment in enumerate(segments):
        if segment < 0:
            if idx + 1 < len(segments) and segments[idx + 1] > 0:
                drop = -float(segment)
                recovery = float(segments[idx + 1])
                events.append((drop, recovery, (recovery - drop) / drop if drop > 0 else 0.0))
            else:
                terminal_unrecovered_drop_segments += 1
    ratios = [event[2] for event in events]
    drops = [event[0] for event in events]
    recoveries = [event[1] for event in events]
    total_drop = float(sum(drops))
    total_recovery = float(sum(recoveries))
    return {
        "dg_primary": float(sum(ratios) / len(ratios)) if ratios else 0.0,
        "dg_event_count": int(len(events)),
        "dg_total_drop": total_drop,
        "dg_total_recovery": total_recovery,
        "dg_total_net_gain": total_recovery - total_drop,
        "dg_total_ratio": float((total_recovery - total_drop) / total_drop) if total_drop > 0 else 0.0,
        "terminal_unrecovered_drop_segment_count": int(terminal_unrecovered_drop_segments),
        "trajectory_point_count": int(len(values)),
        "compact_sortedness_point_count": int(len(_compact_consecutive(values))),
        "signed_segment_count": int(len(segments)),
    }


@dataclass(frozen=True)
class FatigueDamageBenchmarkConfig:
    """Configuration for one dynamic unreliability run."""

    values: tuple[int, ...] = (4, 1, 3, 2)
    algorithm: str = "bubble"
    frozen_indices: tuple[int, ...] = (1,)
    repair_rule: str = "nudge_repair"
    interface_mode: str = "full"
    reliability_mode: str = "fatigue_recovery"
    activation_seed: int = 0
    policy_seed: int | None = None
    max_events: int = 120
    activation_distribution: str = "uniform_active"
    nudge_threshold: int = 2
    fatigue_threshold: int = 2
    recovery_events: int = 3
    damage_threshold: int = 3
    sort_direction: str = "increasing"

    def __post_init__(self) -> None:
        if self.reliability_mode not in RELIABILITY_MODES:
            raise ValueError(f"Unsupported reliability_mode: {self.reliability_mode}")
        if self.repair_rule not in S04_REPAIR_RULES:
            raise ValueError(f"Unsupported repair_rule: {self.repair_rule}")
        if self.interface_mode not in INTERFACE_MODES:
            raise ValueError(f"Unsupported interface_mode: {self.interface_mode}")
        if self.algorithm not in {"bubble", "insertion", "selection"}:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        if self.max_events <= 0:
            raise ValueError("max_events must be positive")
        if self.fatigue_threshold <= 0:
            raise ValueError("fatigue_threshold must be positive")
        if self.recovery_events <= 0:
            raise ValueError("recovery_events must be positive")
        if self.damage_threshold <= 0:
            raise ValueError("damage_threshold must be positive")

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
            frozen_indices=self.frozen_indices,
            repair_rule=self.repair_rule,
            interface_mode=self.interface_mode,
            activation_seed=self.activation_seed,
            policy_seed=self.policy_seed,
            max_events=self.max_events,
            activation_distribution=self.activation_distribution,
            nudge_threshold=self.nudge_threshold,
            sort_direction=self.sort_direction,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": list(self.values),
            "algorithm": self.algorithm,
            "frozen_indices": list(self.frozen_indices),
            "repair_rule": self.repair_rule,
            "interface_mode": self.interface_mode,
            "reliability_mode": self.reliability_mode,
            "activation_seed": int(self.activation_seed),
            "policy_seed": int(self.policy_seed if self.policy_seed is not None else self.activation_seed),
            "max_events": int(self.max_events),
            "activation_distribution": self.activation_distribution,
            "nudge_threshold": int(self.nudge_threshold),
            "fatigue_threshold": int(self.fatigue_threshold),
            "recovery_events": int(self.recovery_events),
            "damage_threshold": int(self.damage_threshold),
            "sort_direction": self.sort_direction,
        }


@dataclass(frozen=True)
class ReliabilityTransition:
    """One fatigue/damage state transition."""

    event_step: int
    thread_id: int
    position: int
    value: int
    from_state: str
    to_state: str
    reason: str
    trigger_value: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_step": int(self.event_step),
            "thread_id": int(self.thread_id),
            "position": int(self.position),
            "value": int(self.value),
            "from_state": self.from_state,
            "to_state": self.to_state,
            "reason": self.reason,
            "trigger_value": int(self.trigger_value),
        }


@dataclass(frozen=True)
class ImpairmentRecord:
    """One event where an impaired active cell could not act."""

    event_step: int
    thread_id: int
    position: int
    state: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_step": int(self.event_step),
            "thread_id": int(self.thread_id),
            "position": int(self.position),
            "state": self.state,
            "reason": self.reason,
        }


class FatigueDamageController:
    """Local dynamic unreliability controller."""

    def __init__(self, config: FatigueDamageBenchmarkConfig) -> None:
        self.config = config
        self.state_by_thread: dict[int, str] = {}
        self.total_moves: Counter[int] = Counter()
        self.moves_since_recovery: Counter[int] = Counter()
        self.rest_events: Counter[int] = Counter()
        self.energy_by_thread: Counter[int] = Counter()
        self.transition_log: list[ReliabilityTransition] = []
        self.impairment_log: list[ImpairmentRecord] = []
        self.current_event_step = 0

    def initialize(self, cells: Sequence[Any]) -> None:
        for cell in cells:
            self.state_by_thread[int(cell.threadID)] = "healthy"

    def state_for(self, cell: Any) -> str:
        return self.state_by_thread.get(int(cell.threadID), "healthy")

    def _transition(self, cell: Any, to_state: str, reason: str, trigger_value: int) -> None:
        thread_id = int(cell.threadID)
        from_state = self.state_by_thread.get(thread_id, "healthy")
        if from_state == to_state:
            return
        self.state_by_thread[thread_id] = to_state
        self.transition_log.append(
            ReliabilityTransition(
                event_step=int(self.current_event_step),
                thread_id=thread_id,
                position=int(cell.current_position[0]),
                value=int(cell.value),
                from_state=from_state,
                to_state=to_state,
                reason=reason,
                trigger_value=int(trigger_value),
            )
        )

    def before_event(self, event_step: int, cells: Sequence[Any]) -> None:
        self.current_event_step = int(event_step)
        if self.config.reliability_mode != "fatigue_recovery":
            return
        for cell in cells:
            thread_id = int(cell.threadID)
            if self.state_by_thread.get(thread_id, "healthy") != "fatigued":
                continue
            self.rest_events[thread_id] += 1
            if self.rest_events[thread_id] >= self.config.recovery_events:
                self.rest_events[thread_id] = 0
                self.moves_since_recovery[thread_id] = 0
                self._transition(cell, "healthy", "recovery_after_rest", self.config.recovery_events)

    def can_act(self, cell: Any) -> bool:
        state = self.state_for(cell)
        return state == "healthy" or self.config.reliability_mode == "no_fatigue_control"

    def record_impairment(self, cell: Any, event_step: int) -> None:
        state = self.state_for(cell)
        self.impairment_log.append(
            ImpairmentRecord(
                event_step=int(event_step),
                thread_id=int(cell.threadID),
                position=int(cell.current_position[0]),
                state=state,
                reason=f"{state}_cell_cannot_act",
            )
        )

    def after_event(self, cell: Any, *, swap_delta: int) -> None:
        if swap_delta <= 0:
            return
        thread_id = int(cell.threadID)
        self.total_moves[thread_id] += int(swap_delta)
        self.energy_by_thread[thread_id] += int(swap_delta)
        if self.config.reliability_mode == "no_fatigue_control":
            return
        if self.config.reliability_mode == "fatigue_recovery":
            if self.state_by_thread.get(thread_id, "healthy") != "healthy":
                return
            self.moves_since_recovery[thread_id] += int(swap_delta)
            if self.moves_since_recovery[thread_id] >= self.config.fatigue_threshold:
                self.rest_events[thread_id] = 0
                self._transition(cell, "fatigued", "fatigue_threshold_reached", self.moves_since_recovery[thread_id])
        elif self.config.reliability_mode == "cumulative_damage":
            if self.state_by_thread.get(thread_id, "healthy") == "damaged":
                return
            if self.total_moves[thread_id] >= self.config.damage_threshold:
                self._transition(cell, "damaged", "damage_threshold_reached", self.total_moves[thread_id])

    def summary(self) -> dict[str, Any]:
        states = Counter(self.state_by_thread.values())
        return {
            "reliability_mode": self.config.reliability_mode,
            "state_counts": dict(sorted(states.items())),
            "transition_count": len(self.transition_log),
            "impairment_count": len(self.impairment_log),
            "total_moves_by_thread": {str(k): int(v) for k, v in sorted(self.total_moves.items())},
            "energy_by_thread": {str(k): int(v) for k, v in sorted(self.energy_by_thread.items())},
            "transitions": [item.to_dict() for item in self.transition_log[:96]],
            "transitions_truncated": len(self.transition_log) > 96,
            "impairments": [item.to_dict() for item in self.impairment_log[:96]],
            "impairments_truncated": len(self.impairment_log) > 96,
        }


@dataclass(frozen=True)
class FatigueDamageBenchmarkResult:
    """Summary and compact logs for one S04 run."""

    config: FatigueDamageBenchmarkConfig
    final_values: tuple[int, ...]
    final_labels: tuple[str, ...]
    final_frozen_positions: tuple[int, ...]
    event_count: int
    swap_count: int
    comparison_count: int
    frozen_attempt_count: int
    repair_success: bool
    final_sortedness_percent: float
    final_monotonicity_error_count: int
    energy_total: int
    fatigue_transition_log: tuple[dict[str, Any], ...]
    impairment_log: tuple[dict[str, Any], ...]
    repair_summary: Mapping[str, Any]
    reliability_summary: Mapping[str, Any]
    signal_audit: Mapping[str, Any]
    event_log: tuple[dict[str, Any], ...]
    baseline_match: bool | None

    def to_row(self) -> dict[str, Any]:
        sortedness_trajectory = [sortedness_percent(self.config.values, self.config.sort_direction)]
        sortedness_trajectory.extend(
            sortedness_percent(event["current_values"], self.config.sort_direction)
            for event in self.event_log
            if "current_values" in event
        )
        dg_metrics = delayed_gratification_from_sortedness(sortedness_trajectory)
        return {
            **self.config.to_dict(),
            "event_count": int(self.event_count),
            "swap_count": int(self.swap_count),
            "comparison_count": int(self.comparison_count),
            "frozen_attempt_count": int(self.frozen_attempt_count),
            "repair_success": bool(self.repair_success),
            "final_sortedness_percent": float(self.final_sortedness_percent),
            "final_monotonicity_error_count": int(self.final_monotonicity_error_count),
            "energy_total": int(self.energy_total),
            "public_state_change_count": int(sum(1 for event in self.event_log if event.get("public_state_changed", False))),
            "dg_primary": float(dg_metrics["dg_primary"]),
            "dg_event_count": int(dg_metrics["dg_event_count"]),
            "dg_total_drop": float(dg_metrics["dg_total_drop"]),
            "dg_total_recovery": float(dg_metrics["dg_total_recovery"]),
            "dg_total_net_gain": float(dg_metrics["dg_total_net_gain"]),
            "dg_total_ratio": float(dg_metrics["dg_total_ratio"]),
            "final_values_json": json.dumps(list(self.final_values), separators=(",", ":")),
            "final_labels_json": json.dumps(list(self.final_labels), separators=(",", ":")),
            "final_frozen_positions_json": json.dumps(list(self.final_frozen_positions), separators=(",", ":")),
            "unfreeze_count": int(self.repair_summary.get("unfreeze_count", 0)),
            "fatigue_transition_count": len(self.fatigue_transition_log),
            "fatigue_transition_log_json": json.dumps(list(self.fatigue_transition_log), sort_keys=True, separators=(",", ":"), default=str),
            "impairment_count": len(self.impairment_log),
            "impairment_log_json": json.dumps(list(self.impairment_log), sort_keys=True, separators=(",", ":"), default=str),
            "repair_summary_json": json.dumps(dict(self.repair_summary), sort_keys=True, separators=(",", ":"), default=str),
            "reliability_summary_json": json.dumps(dict(self.reliability_summary), sort_keys=True, separators=(",", ":"), default=str),
            "signal_audit_json": json.dumps(dict(self.signal_audit), sort_keys=True, separators=(",", ":"), default=str),
            "uses_global_oracle": bool(self.signal_audit.get("usesGlobalOracle", False)),
            "event_log_sha256": stable_json_sha256(list(self.event_log)),
            "baseline_match": self.baseline_match,
        }


def simulate_fatigue_damage_benchmark(config: FatigueDamageBenchmarkConfig) -> FatigueDamageBenchmarkResult:
    """Run one S04 dynamic unreliability benchmark."""

    base_config = SimulatorConfig(
        values=config.values,
        algorithm=config.algorithm,
        frozen_indices=config.frozen_indices,
        frozen_semantics="passive",
        activation_seed=config.activation_seed,
        policy_seed=config.policy_seed,
        activation_distribution=config.activation_distribution,
        max_events=config.max_events,
        stop_when_sorted=False,
        sort_direction=config.sort_direction,
    )
    probe = EventTracingStatusProbe()
    cells, cell_status = _build_cells(base_config, probe)
    memory_bank = BoundedCellMemoryBank(CellMemoryConfig(enabled=config.memory_enabled))
    signal_bank = SignalBank(
        len(cells),
        SignalConfig(enabled=config.signal_enabled, local_radius=1, diffusion_enabled=config.signal_enabled, noise_seed=config.policy_seed or config.activation_seed),
    )
    repair_controller = RepairController(cells, cell_status, config.to_repair_config(), signal_bank)
    if config.repair_rule != "passive_control":
        patch_repairable_swaps(cells, repair_controller)
    reliability = FatigueDamageController(config)
    reliability.initialize(cells)
    activation_rng = random.Random(config.activation_seed)
    random.seed(config.policy_seed if config.policy_seed is not None else config.activation_seed)
    event_log: list[dict[str, Any]] = []
    cursor = 0
    for event_step in range(1, config.max_events + 1):
        repair_controller.before_event(event_step)
        reliability.before_event(event_step, cells)
        active_positions = _active_positions(cells, cell_status)
        if not active_positions:
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
                "reliability_state_after": reliability.state_for(actor),
                "current_values": list(cell_values(cells)),
                "current_frozen_positions": list(repair_controller.final_frozen_positions()),
            }
        )
    final_values = cell_values(cells)
    final_frozen_positions = repair_controller.final_frozen_positions()
    baseline_match: bool | None = None
    if config.reliability_mode == "no_fatigue_control":
        baseline = simulate_repair_benchmark(config.to_repair_config())
        baseline_match = (
            tuple(final_values) == baseline.final_values
            and final_frozen_positions == baseline.final_frozen_positions
            and int(probe.swap_count) == baseline.swap_count
            and int(probe.frozen_swap_attempts) == baseline.frozen_attempt_count
        )
    return FatigueDamageBenchmarkResult(
        config=config,
        final_values=tuple(final_values),
        final_labels=cell_labels(cells),
        final_frozen_positions=final_frozen_positions,
        event_count=len(event_log),
        swap_count=int(probe.swap_count),
        comparison_count=int(probe.compare_and_swap_count),
        frozen_attempt_count=int(probe.frozen_swap_attempts),
        repair_success=len(final_frozen_positions) == 0,
        final_sortedness_percent=sortedness_percent(final_values, config.sort_direction),
        final_monotonicity_error_count=monotonicity_error_count(final_values, config.sort_direction),
        energy_total=sum(reliability.energy_by_thread.values()),
        fatigue_transition_log=tuple(item.to_dict() for item in reliability.transition_log),
        impairment_log=tuple(item.to_dict() for item in reliability.impairment_log),
        repair_summary=repair_controller.summary(),
        reliability_summary=reliability.summary(),
        signal_audit=signal_bank.audit_no_global_oracle(),
        event_log=tuple(event_log),
        baseline_match=baseline_match,
    )


def default_fatigue_damage_configs(
    *,
    algorithms: Sequence[str] = ("bubble", "insertion", "selection"),
    seeds: Sequence[int] = (8401, 8402),
    max_events: int = 80,
) -> list[FatigueDamageBenchmarkConfig]:
    """Return compact S04 benchmark matrix preserving S03 controls."""

    configs: list[FatigueDamageBenchmarkConfig] = []
    for algorithm in algorithms:
        for seed_idx, seed in enumerate(seeds):
            for repair_rule in S04_REPAIR_RULES:
                for interface_mode in INTERFACE_MODES:
                    for reliability_mode in RELIABILITY_MODES:
                        configs.append(
                            FatigueDamageBenchmarkConfig(
                                algorithm=algorithm,
                                repair_rule=repair_rule,
                                interface_mode=interface_mode,
                                reliability_mode=reliability_mode,
                                activation_seed=int(seed + 100 * seed_idx),
                                policy_seed=int(seed + 10_000 + 100 * seed_idx),
                                max_events=max_events,
                            )
                        )
    return configs


def run_fatigue_damage_matrix(configs: Sequence[FatigueDamageBenchmarkConfig]) -> list[FatigueDamageBenchmarkResult]:
    return [simulate_fatigue_damage_benchmark(config) for config in configs]
