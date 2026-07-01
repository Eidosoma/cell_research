"""Repairable Frozen Cell benchmark support for E04 S03."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
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
    is_sorted,
    monotonicity_error_count,
    sortedness_percent,
    stable_json_sha256,
)
from src.e04.memory_policies import BoundedCellMemoryBank, CellMemoryConfig, memory_policy_for_cell
from src.e04.signaling import SignalBank, SignalConfig, signal_policy_for_cell


REPAIR_RULES = (
    "passive_control",
    "stuck_control",
    "nudge_repair",
    "signal_threshold_repair",
    "time_repair",
    "directional_repair",
)
INTERFACE_MODES = ("full", "no_memory", "no_signal")


@dataclass(frozen=True)
class RepairBenchmarkConfig:
    """Configuration for one repairable Frozen Cell benchmark run."""

    values: tuple[int, ...] = (4, 1, 3, 2)
    algorithm: str = "bubble"
    frozen_indices: tuple[int, ...] = (1,)
    repair_rule: str = "nudge_repair"
    interface_mode: str = "full"
    activation_seed: int = 0
    policy_seed: int | None = None
    max_events: int = 120
    activation_distribution: str = "uniform_active"
    nudge_threshold: int = 2
    signal_threshold: float = 0.35
    time_threshold: int = 12
    directional_required_from: str = "left"
    sort_direction: str = "increasing"

    def __post_init__(self) -> None:
        if self.repair_rule not in REPAIR_RULES:
            raise ValueError(f"Unsupported repair_rule: {self.repair_rule}")
        if self.interface_mode not in INTERFACE_MODES:
            raise ValueError(f"Unsupported interface_mode: {self.interface_mode}")
        if self.algorithm not in {"bubble", "insertion", "selection"}:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        if self.directional_required_from not in {"left", "right"}:
            raise ValueError("directional_required_from must be left or right")
        if self.max_events <= 0:
            raise ValueError("max_events must be positive")
        if self.nudge_threshold <= 0:
            raise ValueError("nudge_threshold must be positive")
        if self.time_threshold <= 0:
            raise ValueError("time_threshold must be positive")
        if not 0.0 <= self.signal_threshold <= 10.0:
            raise ValueError("signal_threshold must be in [0, 10]")

    @property
    def memory_enabled(self) -> bool:
        return self.interface_mode != "no_memory"

    @property
    def signal_enabled(self) -> bool:
        return self.interface_mode != "no_signal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": list(self.values),
            "algorithm": self.algorithm,
            "frozen_indices": list(self.frozen_indices),
            "repair_rule": self.repair_rule,
            "interface_mode": self.interface_mode,
            "activation_seed": int(self.activation_seed),
            "policy_seed": int(self.policy_seed if self.policy_seed is not None else self.activation_seed),
            "max_events": int(self.max_events),
            "activation_distribution": self.activation_distribution,
            "nudge_threshold": int(self.nudge_threshold),
            "signal_threshold": float(self.signal_threshold),
            "time_threshold": int(self.time_threshold),
            "directional_required_from": self.directional_required_from,
            "sort_direction": self.sort_direction,
        }


@dataclass(frozen=True)
class FrozenAttemptRecord:
    """One local attempt to interact with a frozen identity."""

    event_step: int
    actor_thread_id: int
    actor_position: int
    target_thread_id: int
    target_position: int
    direction_from_actor: str
    repair_rule: str
    decision: str
    trigger_value: float | int | str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_step": int(self.event_step),
            "actor_thread_id": int(self.actor_thread_id),
            "actor_position": int(self.actor_position),
            "target_thread_id": int(self.target_thread_id),
            "target_position": int(self.target_position),
            "direction_from_actor": self.direction_from_actor,
            "repair_rule": self.repair_rule,
            "decision": self.decision,
            "trigger_value": self.trigger_value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class UnfreezeRecord:
    """One logged Frozen Cell unfreezing event."""

    event_step: int
    target_thread_id: int
    target_position: int
    target_value: int
    repair_rule: str
    trigger: str
    trigger_value: float | int | str
    direction_from_actor: str | None = None
    actor_thread_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_step": int(self.event_step),
            "target_thread_id": int(self.target_thread_id),
            "target_position": int(self.target_position),
            "target_value": int(self.target_value),
            "repair_rule": self.repair_rule,
            "trigger": self.trigger,
            "trigger_value": self.trigger_value,
            "direction_from_actor": self.direction_from_actor,
            "actor_thread_id": self.actor_thread_id,
        }


class RepairController:
    """Local repair logic for frozen identities."""

    def __init__(
        self,
        cells: Sequence[Any],
        cell_status: Any,
        config: RepairBenchmarkConfig,
        signal_bank: SignalBank,
    ) -> None:
        self.cells = cells
        self.cell_status = cell_status
        self.config = config
        self.signal_bank = signal_bank
        self.frozen_thread_ids = {int(cells[idx].threadID) for idx in config.frozen_indices}
        self.initial_frozen_positions = {int(cells[idx].threadID): int(idx) for idx in config.frozen_indices}
        self.nudge_counts: Counter[int] = Counter()
        self.attempt_log: list[FrozenAttemptRecord] = []
        self.unfreeze_log: list[UnfreezeRecord] = []
        self.current_event_step = 0

    def is_frozen_identity(self, cell: Any) -> bool:
        return int(cell.threadID) in self.frozen_thread_ids

    def is_currently_frozen(self, cell: Any) -> bool:
        return self.is_frozen_identity(cell) and cell.status == self.cell_status.FREEZE

    def _direction_from_actor(self, actor: Any, target: Any) -> str:
        return "left" if int(actor.current_position[0]) < int(target.current_position[0]) else "right"

    def _count_probe_attempt_once(self, actor: Any) -> None:
        if not actor.tried_to_swap_with_frozen:
            actor.status_probe.count_frozen_cell_attempt()
            actor.tried_to_swap_with_frozen = True

    def _unfreeze(
        self,
        target: Any,
        *,
        trigger: str,
        trigger_value: float | int | str,
        actor: Any | None = None,
        direction_from_actor: str | None = None,
    ) -> None:
        target.status = self.cell_status.ACTIVE
        target.previous_status = self.cell_status.ACTIVE
        self.unfreeze_log.append(
            UnfreezeRecord(
                event_step=int(self.current_event_step),
                target_thread_id=int(target.threadID),
                target_position=int(target.current_position[0]),
                target_value=int(target.value),
                repair_rule=self.config.repair_rule,
                trigger=trigger,
                trigger_value=trigger_value,
                direction_from_actor=direction_from_actor,
                actor_thread_id=None if actor is None else int(actor.threadID),
            )
        )

    def before_event(self, event_step: int) -> None:
        self.current_event_step = int(event_step)
        if self.config.repair_rule != "time_repair":
            return
        if event_step < self.config.time_threshold:
            return
        for cell in self.cells:
            if self.is_currently_frozen(cell):
                self._unfreeze(cell, trigger="time_threshold", trigger_value=int(event_step))

    def signal_value_at(self, position: int) -> float:
        if not self.config.signal_enabled or not self.signal_bank.config.enabled:
            return 0.0
        position = max(0, min(self.signal_bank.length - 1, int(position)))
        return float(self.signal_bank.fields["blocked"][position] + self.signal_bank.fields["frustrated"][position])

    def should_allow_or_unfreeze_target(self, actor: Any, target: Any) -> bool:
        if not self.is_currently_frozen(target):
            return True
        if self.config.repair_rule == "passive_control":
            return True
        direction = self._direction_from_actor(actor, target)
        thread_id = int(target.threadID)
        self.nudge_counts[thread_id] += 1
        decision = "blocked"
        trigger_value: float | int | str | None = None
        reason = "stuck_control"
        if self.config.repair_rule == "nudge_repair":
            trigger_value = int(self.nudge_counts[thread_id])
            if self.nudge_counts[thread_id] >= self.config.nudge_threshold:
                self._unfreeze(
                    target,
                    trigger="nudge_threshold",
                    trigger_value=int(self.nudge_counts[thread_id]),
                    actor=actor,
                    direction_from_actor=direction,
                )
                decision = "unfroze_and_allowed"
                reason = "nudge_threshold_met"
        elif self.config.repair_rule == "signal_threshold_repair":
            trigger_value = self.signal_value_at(int(target.current_position[0]))
            if float(trigger_value) >= self.config.signal_threshold:
                self._unfreeze(
                    target,
                    trigger="signal_threshold",
                    trigger_value=float(trigger_value),
                    actor=actor,
                    direction_from_actor=direction,
                )
                decision = "unfroze_and_allowed"
                reason = "signal_threshold_met"
            else:
                reason = "signal_below_threshold"
        elif self.config.repair_rule == "directional_repair":
            trigger_value = direction
            if direction == self.config.directional_required_from:
                self._unfreeze(
                    target,
                    trigger="directional_contact",
                    trigger_value=direction,
                    actor=actor,
                    direction_from_actor=direction,
                )
                decision = "unfroze_and_allowed"
                reason = "direction_matched"
            else:
                reason = "direction_mismatch"
        elif self.config.repair_rule == "time_repair":
            trigger_value = int(self.current_event_step)
            reason = "time_threshold_not_yet_met"
        elif self.config.repair_rule == "stuck_control":
            reason = "stuck_control"
        else:
            reason = self.config.repair_rule
        self.attempt_log.append(
            FrozenAttemptRecord(
                event_step=int(self.current_event_step),
                actor_thread_id=int(actor.threadID),
                actor_position=int(actor.current_position[0]),
                target_thread_id=int(target.threadID),
                target_position=int(target.current_position[0]),
                direction_from_actor=direction,
                repair_rule=self.config.repair_rule,
                decision=decision,
                trigger_value=trigger_value,
                reason=reason,
            )
        )
        return decision == "unfroze_and_allowed"

    def final_frozen_positions(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                int(cell.current_position[0])
                for cell in self.cells
                if self.is_frozen_identity(cell) and cell.status == self.cell_status.FREEZE
            )
        )

    def summary(self) -> dict[str, Any]:
        return {
            "repair_rule": self.config.repair_rule,
            "initial_frozen_thread_ids": sorted(self.frozen_thread_ids),
            "initial_frozen_positions_by_thread": {str(k): int(v) for k, v in sorted(self.initial_frozen_positions.items())},
            "attempt_count": len(self.attempt_log),
            "unfreeze_count": len(self.unfreeze_log),
            "nudge_counts_by_thread": {str(k): int(v) for k, v in sorted(self.nudge_counts.items())},
            "unfreeze_log": [item.to_dict() for item in self.unfreeze_log],
            "attempt_log": [item.to_dict() for item in self.attempt_log[:64]],
            "attempt_log_truncated": len(self.attempt_log) > 64,
        }


def patch_repairable_swaps(cells: Sequence[Any], controller: RepairController) -> None:
    """Patch public cell swap methods to enforce repairable Frozen Cell rules."""

    def make_swap(cell: Any) -> Any:
        original_swap = cell.swap

        def repairable_swap(target_position: tuple[int, int], skip_stats: bool = False) -> Any:
            if controller.is_currently_frozen(cell):
                controller._count_probe_attempt_once(cell)
                return None
            target = cell.cells[int(target_position[0])]
            if controller.is_currently_frozen(target):
                if not controller.should_allow_or_unfreeze_target(cell, target):
                    controller._count_probe_attempt_once(cell)
                    return None
            cell.tried_to_swap_with_frozen = False
            target.tried_to_swap_with_frozen = False
            return original_swap(target_position, skip_stats)

        return repairable_swap

    for cell in cells:
        cell.swap = make_swap(cell)


@dataclass(frozen=True)
class RepairBenchmarkResult:
    """Summary and compact logs for one repair benchmark run."""

    config: RepairBenchmarkConfig
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
    unfreeze_log: tuple[dict[str, Any], ...]
    frozen_attempt_log: tuple[dict[str, Any], ...]
    event_log: tuple[dict[str, Any], ...]
    signal_audit: Mapping[str, Any]
    repair_summary: Mapping[str, Any]
    memory_digest: str
    signal_digest: str

    def to_row(self) -> dict[str, Any]:
        return {
            **self.config.to_dict(),
            "event_count": int(self.event_count),
            "swap_count": int(self.swap_count),
            "comparison_count": int(self.comparison_count),
            "frozen_attempt_count": int(self.frozen_attempt_count),
            "repair_success": bool(self.repair_success),
            "final_sortedness_percent": float(self.final_sortedness_percent),
            "final_monotonicity_error_count": int(self.final_monotonicity_error_count),
            "final_values_json": json.dumps(list(self.final_values), separators=(",", ":")),
            "final_labels_json": json.dumps(list(self.final_labels), separators=(",", ":")),
            "final_frozen_positions_json": json.dumps(list(self.final_frozen_positions), separators=(",", ":")),
            "unfreeze_count": len(self.unfreeze_log),
            "unfreeze_log_json": json.dumps(list(self.unfreeze_log), sort_keys=True, separators=(",", ":"), default=str),
            "frozen_attempt_count_logged": len(self.frozen_attempt_log),
            "frozen_attempt_log_json": json.dumps(list(self.frozen_attempt_log), sort_keys=True, separators=(",", ":"), default=str),
            "signal_audit_json": json.dumps(dict(self.signal_audit), sort_keys=True, separators=(",", ":"), default=str),
            "uses_global_oracle": bool(self.signal_audit.get("usesGlobalOracle", False)),
            "repair_summary_json": json.dumps(dict(self.repair_summary), sort_keys=True, separators=(",", ":"), default=str),
            "memory_digest": self.memory_digest,
            "signal_digest": self.signal_digest,
            "event_log_sha256": stable_json_sha256(list(self.event_log)),
        }


def simulate_repair_benchmark(config: RepairBenchmarkConfig) -> RepairBenchmarkResult:
    """Run one deterministic repairable Frozen Cell benchmark."""

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
    controller = RepairController(cells, cell_status, config, signal_bank)
    if config.repair_rule != "passive_control":
        patch_repairable_swaps(cells, controller)
    activation_rng = random.Random(config.activation_seed)
    random.seed(config.policy_seed if config.policy_seed is not None else config.activation_seed)
    event_log: list[dict[str, Any]] = []
    cursor = 0
    for event_step in range(1, config.max_events + 1):
        controller.before_event(event_step)
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
        before_unfreeze_count = len(controller.unfreeze_log)
        probe.current_event_step = event_step
        if config.signal_enabled:
            step_result = signal_policy_for_cell(actor, signal_bank, memory_bank).step(actor, event_step=event_step)
            applied_action = step_result.memory_result.applied_action.action_type
        else:
            step_result = memory_policy_for_cell(actor, memory_bank).step(actor, event_step=event_step)
            applied_action = step_result.applied_action.action_type
        after_signature = cell_state_signature(cells)
        public_state_changed = after_signature != before_signature
        event_log.append(
            {
                "event_step": int(event_step),
                "activated_position": int(position),
                "activated_thread_id": int(actor.threadID),
                "activated_value_before": int(before_signature[position][1]),
                "applied_action": str(applied_action),
                "swap_delta": int(probe.swap_count) - before_swaps,
                "public_state_changed": bool(public_state_changed),
                "unfreeze_delta": len(controller.unfreeze_log) - before_unfreeze_count,
                "current_values": list(cell_values(cells)),
                "current_frozen_positions": list(controller.final_frozen_positions()),
            }
        )
    final_values = cell_values(cells)
    final_frozen_positions = controller.final_frozen_positions()
    signal_audit = signal_bank.audit_no_global_oracle()
    return RepairBenchmarkResult(
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
        unfreeze_log=tuple(item.to_dict() for item in controller.unfreeze_log),
        frozen_attempt_log=tuple(item.to_dict() for item in controller.attempt_log),
        event_log=tuple(event_log),
        signal_audit=signal_audit,
        repair_summary=controller.summary(),
        memory_digest=memory_bank.stable_digest(),
        signal_digest=signal_bank.stable_digest(),
    )


def default_benchmark_configs(
    *,
    algorithms: Sequence[str] = ("bubble", "insertion", "selection"),
    seeds: Sequence[int] = (7301, 7302, 7303),
    max_events: int = 120,
) -> list[RepairBenchmarkConfig]:
    """Return the compact S03 baseline policy benchmark matrix."""

    configs: list[RepairBenchmarkConfig] = []
    for algorithm in algorithms:
        for seed_idx, seed in enumerate(seeds):
            for repair_rule in REPAIR_RULES:
                for interface_mode in INTERFACE_MODES:
                    configs.append(
                        RepairBenchmarkConfig(
                            algorithm=algorithm,
                            repair_rule=repair_rule,
                            interface_mode=interface_mode,
                            activation_seed=int(seed + 100 * seed_idx),
                            policy_seed=int(seed + 10_000 + 100 * seed_idx),
                            max_events=max_events,
                        )
                    )
    return configs


def run_benchmark_matrix(configs: Sequence[RepairBenchmarkConfig]) -> list[RepairBenchmarkResult]:
    return [simulate_repair_benchmark(config) for config in configs]
