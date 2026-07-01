"""Local and diffusive signaling extensions for E04 S02.

Signals are an external, ablatable side channel around the E03 public-policy
wrapper and the E04 S01 memory bank.  The implementation records which array
indices were read while building an emission or sensation so downstream audits
can distinguish local signals from explicitly diffusive scalar fields.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.e03.policy_interface import OriginalCellPolicyWrapper, PolicyObservation, PolicyState, cells_signature
from src.e04.memory_policies import (
    BoundedCellMemoryBank,
    CellMemoryConfig,
    CellMemoryState,
    MemoryEnabledPolicyWrapper,
    MemoryStepResult,
)


SIGNAL_FIELDS = ("blocked", "sorted", "frustrated", "target_seeking", "morphogen")
FORBIDDEN_ORACLE_KEYS = (
    "sortedness",
    "monotonicity",
    "global",
    "rank",
    "whole_array",
    "final_values",
    "all_values",
)


@dataclass(frozen=True)
class SignalConfig:
    """Configuration for local and diffusive cell signaling."""

    enabled: bool = True
    local_radius: int = 1
    diffusion_enabled: bool = True
    diffusion_rate: float = 0.25
    decay: float = 0.0
    noise_std: float = 0.0
    noise_seed: int = 0
    max_log_entries: int = 512

    def __post_init__(self) -> None:
        if int(self.local_radius) < 0:
            raise ValueError("local_radius must be non-negative")
        if int(self.max_log_entries) < 0:
            raise ValueError("max_log_entries must be non-negative")
        if not 0.0 <= float(self.diffusion_rate) <= 0.5:
            raise ValueError("diffusion_rate must be in [0, 0.5]")
        if not 0.0 <= float(self.decay) <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        if float(self.noise_std) < 0.0:
            raise ValueError("noise_std must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "local_radius": int(self.local_radius),
            "diffusion_enabled": bool(self.diffusion_enabled),
            "diffusion_rate": float(self.diffusion_rate),
            "decay": float(self.decay),
            "noise_std": float(self.noise_std),
            "noise_seed": int(self.noise_seed),
            "max_log_entries": int(self.max_log_entries),
        }


@dataclass(frozen=True)
class LocalObservationWindow:
    """A bounded local view used for signal emission and audit."""

    actor_index: int
    radius: int
    values: Mapping[int, int]
    labels: Mapping[int, str]
    statuses: Mapping[int, str]
    accessed_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_index": int(self.actor_index),
            "radius": int(self.radius),
            "values": {str(key): int(value) for key, value in sorted(self.values.items())},
            "labels": {str(key): str(value) for key, value in sorted(self.labels.items())},
            "statuses": {str(key): str(value) for key, value in sorted(self.statuses.items())},
            "accessed_indices": list(self.accessed_indices),
        }


@dataclass(frozen=True)
class LocalSignalValues:
    """Numeric signal values emitted by one cell at one position."""

    blocked: float = 0.0
    sorted: float = 0.0
    frustrated: float = 0.0
    target_seeking: float = 0.0
    morphogen: float = 0.0

    def clipped(self) -> "LocalSignalValues":
        return LocalSignalValues(
            blocked=_clip01(self.blocked),
            sorted=_clip01(self.sorted),
            frustrated=_clip01(self.frustrated),
            target_seeking=max(-1.0, min(1.0, float(self.target_seeking))),
            morphogen=_clip01(self.morphogen),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "blocked": float(self.blocked),
            "sorted": float(self.sorted),
            "frustrated": float(self.frustrated),
            "target_seeking": float(self.target_seeking),
            "morphogen": float(self.morphogen),
        }

    def field(self, name: str) -> float:
        if name not in SIGNAL_FIELDS:
            raise ValueError(f"Unsupported signal field: {name}")
        return float(getattr(self, name))


@dataclass(frozen=True)
class SignalEmission:
    """One emitted local signal record."""

    event_step: int | None
    source_thread_id: int
    source_position: int
    source_label: str
    source_behavior: str
    values: LocalSignalValues
    accessed_indices: tuple[int, ...]
    access_scope: str = "local_window"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_step": self.event_step,
            "source_thread_id": int(self.source_thread_id),
            "source_position": int(self.source_position),
            "source_label": str(self.source_label),
            "source_behavior": str(self.source_behavior),
            "values": self.values.to_dict(),
            "accessed_indices": list(self.accessed_indices),
            "access_scope": self.access_scope,
        }


@dataclass(frozen=True)
class SignalSensation:
    """Signals sensed by one actor before policy execution."""

    event_step: int | None
    actor_thread_id: int
    actor_position: int
    local_signals: tuple[tuple[int, LocalSignalValues], ...]
    diffusive_fields: Mapping[str, float]
    accessed_indices: tuple[int, ...]
    access_scope: str
    noise_applied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_step": self.event_step,
            "actor_thread_id": int(self.actor_thread_id),
            "actor_position": int(self.actor_position),
            "local_signals": [
                {"position": int(position), "values": values.to_dict()} for position, values in self.local_signals
            ],
            "diffusive_fields": {str(key): float(value) for key, value in sorted(self.diffusive_fields.items())},
            "accessed_indices": list(self.accessed_indices),
            "access_scope": self.access_scope,
            "noise_applied": bool(self.noise_applied),
        }


@dataclass(frozen=True)
class SignalPolicyObservation:
    """Policy observation with S01 memory and S02 sensed signals."""

    local: PolicyObservation
    memory: CellMemoryState
    sensation: SignalSensation
    signal_config: SignalConfig

    @property
    def state(self) -> PolicyState:
        return PolicyState(
            ideal_position=self.local.ideal_position,
            reverse_direction=self.local.reverse_direction,
            memory={
                "e04_cell_memory": self.memory.to_dict(),
                "e04_signals": self.sensation.to_dict(),
            },
        )


@dataclass(frozen=True)
class SignalStepResult:
    """Result of one memory-enabled step plus signal update."""

    memory_result: MemoryStepResult
    sensation_before: SignalSensation
    emission_after: SignalEmission | None
    signal_enabled: bool

    @property
    def behavior(self) -> str:
        return self.memory_result.behavior


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_local_window(observation: PolicyObservation, radius: int) -> LocalObservationWindow:
    """Build a bounded local view without exposing full-array metrics."""

    radius = int(radius)
    left = max(observation.left_boundary, observation.actor_index - radius)
    right = min(observation.right_boundary, observation.actor_index + radius)
    indices = tuple(range(left, right + 1))
    return LocalObservationWindow(
        actor_index=int(observation.actor_index),
        radius=radius,
        values={idx: int(observation.values[idx]) for idx in indices},
        labels={idx: str(observation.labels[idx]) for idx in indices},
        statuses={idx: str(observation.statuses[idx]) for idx in indices},
        accessed_indices=indices,
    )


def local_order_score(observation: PolicyObservation, window: LocalObservationWindow) -> float:
    """Return actor-local ordered-neighbor fraction, not global sortedness."""

    checks: list[bool] = []
    actor_idx = observation.actor_index
    actor_value = int(observation.actor_value)
    reverse = bool(observation.reverse_direction)
    left_idx = actor_idx - 1
    right_idx = actor_idx + 1
    if left_idx in window.values:
        left_value = int(window.values[left_idx])
        checks.append(left_value >= actor_value if reverse else left_value <= actor_value)
    if right_idx in window.values:
        right_value = int(window.values[right_idx])
        checks.append(actor_value >= right_value if reverse else actor_value <= right_value)
    if not checks:
        return 1.0
    return sum(1 for item in checks if item) / len(checks)


def target_seek_signal(observation: PolicyObservation) -> float:
    """Encode only direction toward the current local-policy target."""

    if observation.ideal_position is None or observation.ideal_position == observation.actor_index:
        return 0.0
    return 1.0 if observation.ideal_position > observation.actor_index else -1.0


def infer_signal_values(
    observation: PolicyObservation,
    memory: CellMemoryState,
    *,
    blocked: bool = False,
    radius: int = 1,
) -> tuple[LocalSignalValues, tuple[int, ...]]:
    """Infer emitted signals from local observation fields and S01 memory."""

    window = build_local_window(observation, radius)
    blocked_value = 1.0 if blocked or memory.last_move_success is False else 0.0
    max_frustration = 255.0
    frustration_value = _clip01(float(memory.local_frustration) / max_frustration)
    target_value = target_seek_signal(observation)
    sorted_value = local_order_score(observation, window)
    morphogen = _clip01((blocked_value + frustration_value + abs(target_value)) / 3.0)
    return (
        LocalSignalValues(
            blocked=blocked_value,
            sorted=sorted_value,
            frustrated=frustration_value,
            target_seeking=target_value,
            morphogen=morphogen,
        ).clipped(),
        window.accessed_indices,
    )


class SignalBank:
    """External signal field and local-emission store."""

    def __init__(self, length: int, config: SignalConfig | None = None) -> None:
        if int(length) <= 0:
            raise ValueError("length must be positive")
        self.length = int(length)
        self.config = config or SignalConfig()
        self.local_by_position: list[LocalSignalValues] = [LocalSignalValues() for _ in range(self.length)]
        self.fields: dict[str, list[float]] = {name: [0.0 for _ in range(self.length)] for name in SIGNAL_FIELDS}
        self.rng = random.Random(self.config.noise_seed)
        self.emission_log: list[SignalEmission] = []
        self.sensation_log: list[SignalSensation] = []
        self.diffusion_step_count = 0

    def clear(self) -> None:
        self.local_by_position = [LocalSignalValues() for _ in range(self.length)]
        self.fields = {name: [0.0 for _ in range(self.length)] for name in SIGNAL_FIELDS}
        self.emission_log.clear()
        self.sensation_log.clear()
        self.diffusion_step_count = 0

    def _check_position(self, position: int) -> int:
        position = int(position)
        if position < 0 or position >= self.length:
            raise ValueError(f"position out of range: {position}")
        return position

    def set_local_signal(self, position: int, values: LocalSignalValues) -> None:
        position = self._check_position(position)
        self.local_by_position[position] = values.clipped()

    def deposit_field(self, position: int, values: LocalSignalValues) -> None:
        position = self._check_position(position)
        if not self.config.diffusion_enabled:
            return
        clipped = values.clipped()
        for name in SIGNAL_FIELDS:
            self.fields[name][position] += clipped.field(name)

    def emit(
        self,
        observation: PolicyObservation,
        memory: CellMemoryState,
        *,
        blocked: bool = False,
        event_step: int | None = None,
    ) -> SignalEmission | None:
        if not self.config.enabled:
            return None
        values, accessed_indices = infer_signal_values(
            observation,
            memory,
            blocked=blocked,
            radius=self.config.local_radius,
        )
        position = self._check_position(observation.actor_index)
        self.set_local_signal(position, values)
        self.deposit_field(position, values)
        emission = SignalEmission(
            event_step=event_step,
            source_thread_id=int(observation.actor_thread_id),
            source_position=position,
            source_label=str(observation.actor_label),
            source_behavior=str(observation.behavior),
            values=values,
            accessed_indices=tuple(accessed_indices),
        )
        if len(self.emission_log) < self.config.max_log_entries:
            self.emission_log.append(emission)
        if self.config.diffusion_enabled:
            self.diffuse_once()
        return emission

    def diffuse_once(self) -> None:
        """Diffuse scalar fields along the 1D array with optional decay."""

        if not self.config.diffusion_enabled:
            return
        rate = float(self.config.diffusion_rate)
        retain = 1.0 - float(self.config.decay)
        updated: dict[str, list[float]] = {}
        for name, values in self.fields.items():
            new_values: list[float] = []
            for idx, value in enumerate(values):
                left = values[idx - 1] if idx > 0 else value
                right = values[idx + 1] if idx < self.length - 1 else value
                new_values.append(retain * ((1.0 - 2.0 * rate) * value + rate * left + rate * right))
            updated[name] = new_values
        self.fields = updated
        self.diffusion_step_count += 1

    def local_indices_for(self, actor_index: int) -> tuple[int, ...]:
        actor_index = self._check_position(actor_index)
        radius = int(self.config.local_radius)
        left = max(0, actor_index - radius)
        right = min(self.length - 1, actor_index + radius)
        return tuple(range(left, right + 1))

    def sense(
        self,
        observation: PolicyObservation,
        *,
        event_step: int | None = None,
    ) -> SignalSensation:
        actor_position = self._check_position(observation.actor_index)
        if not self.config.enabled:
            sensation = SignalSensation(
                event_step=event_step,
                actor_thread_id=int(observation.actor_thread_id),
                actor_position=actor_position,
                local_signals=(),
                diffusive_fields={name: 0.0 for name in SIGNAL_FIELDS},
                accessed_indices=(),
                access_scope="disabled",
                noise_applied=False,
            )
            return sensation
        indices = self.local_indices_for(actor_position)
        local_signals = tuple((idx, self.local_by_position[idx]) for idx in indices)
        diffusive_fields: dict[str, float] = {}
        for name in SIGNAL_FIELDS:
            value = float(self.fields[name][actor_position])
            if self.config.noise_std > 0.0:
                value += self.rng.gauss(0.0, float(self.config.noise_std))
            diffusive_fields[name] = value
        sensation = SignalSensation(
            event_step=event_step,
            actor_thread_id=int(observation.actor_thread_id),
            actor_position=actor_position,
            local_signals=local_signals,
            diffusive_fields=diffusive_fields,
            accessed_indices=indices,
            access_scope="local_window_plus_explicit_diffusive_fields"
            if self.config.diffusion_enabled
            else "local_window",
            noise_applied=bool(self.config.noise_std > 0.0),
        )
        if len(self.sensation_log) < self.config.max_log_entries:
            self.sensation_log.append(sensation)
        return sensation

    def audit_no_global_oracle(self) -> dict[str, Any]:
        """Audit logged signal records for locality and forbidden key names."""

        records = [item.to_dict() for item in self.emission_log] + [item.to_dict() for item in self.sensation_log]
        serialized = json.dumps(records, sort_keys=True, separators=(",", ":"), default=str).lower()
        forbidden_hits = [key for key in FORBIDDEN_ORACLE_KEYS if key in serialized]
        locality_violations: list[dict[str, Any]] = []
        radius = int(self.config.local_radius)
        for item in self.emission_log:
            allowed = set(range(max(0, item.source_position - radius), min(self.length - 1, item.source_position + radius) + 1))
            if not set(item.accessed_indices).issubset(allowed):
                locality_violations.append(item.to_dict())
        for item in self.sensation_log:
            allowed = set(range(max(0, item.actor_position - radius), min(self.length - 1, item.actor_position + radius) + 1))
            if not set(item.accessed_indices).issubset(allowed):
                locality_violations.append(item.to_dict())
        return {
            "usesGlobalOracle": bool(forbidden_hits or locality_violations),
            "forbiddenKeyHits": forbidden_hits,
            "localityViolationCount": len(locality_violations),
            "localityViolations": locality_violations[:8],
            "diffusiveFieldsExplicitlyLabeled": bool(
                not self.config.diffusion_enabled
                or all("diffusive" in item.access_scope for item in self.sensation_log)
            ),
            "localRadius": radius,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "length": int(self.length),
            "config": self.config.to_dict(),
            "local_by_position": [values.to_dict() for values in self.local_by_position],
            "fields": {name: [float(value) for value in values] for name, values in sorted(self.fields.items())},
            "diffusion_step_count": int(self.diffusion_step_count),
            "emission_log": [item.to_dict() for item in self.emission_log],
            "sensation_log": [item.to_dict() for item in self.sensation_log],
        }

    def stable_digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SignalEnabledPolicyWrapper:
    """Behavior-preserving policy wrapper with memory and local signals."""

    def __init__(
        self,
        *,
        memory_wrapper: MemoryEnabledPolicyWrapper,
        signal_bank: SignalBank,
    ) -> None:
        self.memory_wrapper = memory_wrapper
        self.signal_bank = signal_bank
        self.behavior = memory_wrapper.behavior

    @classmethod
    def from_cell(
        cls,
        cell: Any,
        *,
        signal_bank: SignalBank,
        memory_bank: BoundedCellMemoryBank | None = None,
        memory_config: CellMemoryConfig | None = None,
        label_to_behavior: Mapping[str, str] | None = None,
    ) -> "SignalEnabledPolicyWrapper":
        memory_wrapper = MemoryEnabledPolicyWrapper.from_cell(
            cell,
            memory_bank=memory_bank,
            memory_config=memory_config,
            label_to_behavior=label_to_behavior,
        )
        return cls(memory_wrapper=memory_wrapper, signal_bank=signal_bank)

    def observe(self, cell: Any, event_step: int | None = None) -> SignalPolicyObservation:
        local = OriginalCellPolicyWrapper.from_cell(cell, self.memory_wrapper.label_to_behavior).observe(cell)
        memory = self.memory_wrapper.memory_bank.get(local.actor_thread_id)
        sensation = self.signal_bank.sense(local, event_step=event_step)
        return SignalPolicyObservation(
            local=local,
            memory=memory,
            sensation=sensation,
            signal_config=self.signal_bank.config,
        )

    def state(self, cell: Any) -> PolicyState:
        return self.observe(cell).state

    def step(self, cell: Any, event_step: int | None = None) -> SignalStepResult:
        observation = self.observe(cell, event_step=event_step)
        memory_result = self.memory_wrapper.step(cell, event_step=event_step)
        blocked = (
            memory_result.base_result.applied_action.action_type == "blocked_swap_attempt"
            or memory_result.base_result.frozen_attempt_delta > 0
        )
        emission = self.signal_bank.emit(
            memory_result.base_result.observation,
            memory_result.memory_after,
            blocked=blocked,
            event_step=event_step,
        )
        return SignalStepResult(
            memory_result=memory_result,
            sensation_before=observation.sensation,
            emission_after=emission,
            signal_enabled=bool(self.signal_bank.config.enabled),
        )


def signal_policy_for_cell(
    cell: Any,
    signal_bank: SignalBank,
    memory_bank: BoundedCellMemoryBank | None = None,
    memory_config: CellMemoryConfig | None = None,
    label_to_behavior: Mapping[str, str] | None = None,
) -> SignalEnabledPolicyWrapper:
    """Return a signal-enabled wrapper for a public cell instance."""

    return SignalEnabledPolicyWrapper.from_cell(
        cell,
        signal_bank=signal_bank,
        memory_bank=memory_bank,
        memory_config=memory_config,
        label_to_behavior=label_to_behavior,
    )


def signal_signature(bank: SignalBank) -> tuple[Any, ...]:
    """Return a compact deterministic signal signature for tests."""

    payload = bank.to_dict()
    return (
        tuple(tuple(round(value, 12) for value in payload["fields"][name]) for name in SIGNAL_FIELDS),
        tuple((idx, values.to_dict()) for idx, values in enumerate(bank.local_by_position)),
        bank.stable_digest(),
    )


def public_signature(cells: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
    """Expose E03 public-state signature through the E04 module."""

    return cells_signature(cells)
