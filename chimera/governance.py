"""E06 S09 local governance-mechanism interventions for chimeric collectives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from e02_deterministic_simulator.metrics import aggregation, sortedness_percent, state_hash
from e02_deterministic_simulator.simulator import StepOutcome
from memory_repair import SignalEventSimulator
from morphospace import ProposedAction

from .dominance import (
    DEFAULT_S05_BOUNDARY_CAVEATS_PATH,
    artifact_records,
    load_boundary_caveats,
    predefined_winner_criteria,
    score_dominance_results,
)
from .goals import (
    augment_goal_results,
    condition_goal_metadata,
    policy_goal_specs,
    reverse_directions_for_assignment,
)
from .interfaces import (
    DEFAULT_S06_PAIR_SUMMARY_PATH,
    DEFAULT_S06_WINNER_CRITERIA_PATH,
    DEFAULT_S07_PRIORITY_PATH,
    INTERFACE_RULE_VERSION,
    InterfaceRuleEventSimulator,
)
from .mixtures import (
    DEFAULT_PANEL_PATH,
    compact_json,
    compact_result_csv,
    git_value,
    initial_values,
    load_ready_panel,
    parse_json_maybe,
    policy_assignment,
    simulator_backend,
    write_json,
)
from .mosaics import MOSAIC_CLASSIFIER_VERSION, REQUESTED_MOSAIC_CLASSES, classify_mosaic_formations, safe_float
from .panel import CLAIM_BOUNDARY, instantiate_panel_policy, json_ready


STEP_ID = "S09"
STEP_NUMBER = 9
GOVERNANCE_VERSION = "e06_s09_governance_mechanisms.v1"
DEFAULT_S08_RESULTS_PATH = Path("/artifacts/results/e06_interface_rules.parquet")
DEFAULT_S08_EFFECT_SUMMARY_PATH = Path("/artifacts/research_steps/S08/interface_effect_summary.parquet")
GOVERNANCE_GOAL_MODES: tuple[str, ...] = ("same_goal", "opposite_goal", "partially_compatible")
GOVERNANCE_VARIANTS: tuple[str, ...] = (
    "no_governance_control",
    "local_voting_range1",
    "leader_cells_range2",
    "pacemaker_cells_range2",
    "quorum_signal_range2",
    "conflict_resolution_range1",
    "organizer_limited_range4",
    "organizer_global_upper_bound",
)
_PANEL_LOOKUP: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class GovernanceConfig:
    """Serializable governance intervention with explicit information-audit fields."""

    variant: str
    enabled: bool
    family: str
    influence_range: int
    information_access: str
    local_information_only: bool = True
    uses_global_state: bool = False
    uses_target_map: bool = False
    uses_organizer: bool = False
    broad_control_like: bool = False
    intervention_strength: float = 1.0
    quorum_threshold: float = 0.60
    pacemaker_period: int = 5
    leader_fraction: float = 0.02
    description: str = ""

    def __post_init__(self) -> None:
        if self.influence_range < 0:
            raise ValueError("influence_range must be nonnegative")
        if not 0.0 <= float(self.quorum_threshold) <= 1.0:
            raise ValueError("quorum_threshold must be in [0, 1]")
        if self.pacemaker_period < 1:
            raise ValueError("pacemaker_period must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GOVERNANCE_VERSION,
            "variant": self.variant,
            "enabled": bool(self.enabled),
            "family": self.family,
            "influenceRange": int(self.influence_range),
            "informationAccess": self.information_access,
            "localInformationOnly": bool(self.local_information_only),
            "usesGlobalState": bool(self.uses_global_state),
            "usesTargetMap": bool(self.uses_target_map),
            "usesOrganizer": bool(self.uses_organizer),
            "broadControlLike": bool(self.broad_control_like),
            "interventionStrength": float(self.intervention_strength),
            "quorumThreshold": float(self.quorum_threshold),
            "pacemakerPeriod": int(self.pacemaker_period),
            "leaderFraction": float(self.leader_fraction),
            "description": self.description,
            "claimBoundary": CLAIM_BOUNDARY,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | str | "GovernanceConfig") -> "GovernanceConfig":
        if isinstance(payload, GovernanceConfig):
            return payload
        if isinstance(payload, str):
            try:
                parsed = parse_json_maybe(payload, None)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                payload = parsed
            else:
                return governance_config_by_variant(payload)
        return cls(
            variant=str(payload.get("variant", "no_governance_control")),
            enabled=bool(payload.get("enabled", False)),
            family=str(payload.get("family", "disabled")),
            influence_range=int(payload.get("influenceRange", payload.get("influence_range", 0))),
            information_access=str(payload.get("informationAccess", payload.get("information_access", "none"))),
            local_information_only=bool(payload.get("localInformationOnly", payload.get("local_information_only", True))),
            uses_global_state=bool(payload.get("usesGlobalState", payload.get("uses_global_state", False))),
            uses_target_map=bool(payload.get("usesTargetMap", payload.get("uses_target_map", False))),
            uses_organizer=bool(payload.get("usesOrganizer", payload.get("uses_organizer", False))),
            broad_control_like=bool(payload.get("broadControlLike", payload.get("broad_control_like", False))),
            intervention_strength=float(payload.get("interventionStrength", payload.get("intervention_strength", 1.0))),
            quorum_threshold=float(payload.get("quorumThreshold", payload.get("quorum_threshold", 0.60))),
            pacemaker_period=int(payload.get("pacemakerPeriod", payload.get("pacemaker_period", 5))),
            leader_fraction=float(payload.get("leaderFraction", payload.get("leader_fraction", 0.02))),
            description=str(payload.get("description", "")),
        )


def governance_configs() -> tuple[GovernanceConfig, ...]:
    return (
        GovernanceConfig(
            variant="no_governance_control",
            enabled=False,
            family="disabled",
            influence_range=0,
            information_access="none",
            description="No governance mechanism; preserves the S08 disabled-interface baseline when goal labels match.",
        ),
        GovernanceConfig(
            variant="local_voting_range1",
            enabled=True,
            family="local_voting",
            influence_range=1,
            information_access="actor plus nearest-neighbor values, labels, and goal directions",
            description="Range-1 cells vote by local goal direction and steer adjacent swaps that improve that local ordering.",
        ),
        GovernanceConfig(
            variant="leader_cells_range2",
            enabled=True,
            family="leader_cells",
            influence_range=2,
            information_access="nearest designated same-array leader cell direction within range 2",
            leader_fraction=0.02,
            description="Deterministic leader cells exert range-limited direction cues on nearby cells.",
        ),
        GovernanceConfig(
            variant="pacemaker_cells_range2",
            enabled=True,
            family="pacemaker_cells",
            influence_range=2,
            information_access="periodic local pulse from fixed pacemaker cells within range 2",
            pacemaker_period=5,
            description="Pacemaker cells periodically bias nearby cells toward locally increasing order.",
        ),
        GovernanceConfig(
            variant="quorum_signal_range2",
            enabled=True,
            family="quorum_signal",
            influence_range=2,
            information_access="range-2 local label quorum and goal-direction majority",
            quorum_threshold=0.60,
            description="Local quorum signal follows a range-2 policy-label majority when a clear majority exists.",
        ),
        GovernanceConfig(
            variant="conflict_resolution_range1",
            enabled=True,
            family="conflict_resolution",
            influence_range=1,
            information_access="range-1 local labels, values, and opposing goal-direction flags",
            description="Conflict-resolution signal gates adjacent swaps in mixed local-goal neighborhoods.",
        ),
        GovernanceConfig(
            variant="organizer_limited_range4",
            enabled=True,
            family="organizer_limited",
            influence_range=4,
            information_access="range-4 local organizer cue without whole-array target map",
            uses_organizer=True,
            description="Range-limited organizer cells provide local ordering cues without reading the full target map.",
        ),
        GovernanceConfig(
            variant="organizer_global_upper_bound",
            enabled=True,
            family="organizer_global_upper_bound",
            influence_range=10_000,
            information_access="whole-array target-map upper-bound controller",
            local_information_only=False,
            uses_global_state=True,
            uses_target_map=True,
            uses_organizer=True,
            broad_control_like=True,
            description="Explicit broad/global-control-like organizer baseline flagged as an upper-bound intervention.",
        ),
    )


def governance_config_by_variant(variant: str) -> GovernanceConfig:
    lookup = {config.variant: config for config in governance_configs()}
    if variant not in lookup:
        raise ValueError(f"unknown S09 governance variant: {variant}")
    return lookup[variant]


def governance_parameter_table(configs: Sequence[GovernanceConfig] | None = None) -> pd.DataFrame:
    configs = tuple(configs or governance_configs())
    rows: list[dict[str, Any]] = []
    for index, config in enumerate(configs):
        payload = config.to_dict()
        rows.append(
            {
                "governanceIndex": int(index),
                "governanceVariant": config.variant,
                "governanceEnabled": bool(config.enabled),
                "governanceFamily": config.family,
                "influenceRange": int(config.influence_range),
                "informationAccess": config.information_access,
                "localInformationOnly": bool(config.local_information_only),
                "usesGlobalState": bool(config.uses_global_state),
                "usesTargetMap": bool(config.uses_target_map),
                "usesOrganizer": bool(config.uses_organizer),
                "broadControlLike": bool(config.broad_control_like),
                "interventionStrength": float(config.intervention_strength),
                "quorumThreshold": float(config.quorum_threshold),
                "pacemakerPeriod": int(config.pacemaker_period),
                "leaderFraction": float(config.leader_fraction),
                "governanceConfigJson": compact_json(payload),
                "description": config.description,
                "governanceVersion": GOVERNANCE_VERSION,
                "claimBoundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _labels(cells: Sequence[Any]) -> list[int]:
    return [int(cell.label) for cell in cells]


def _interfaces(labels: Sequence[int]) -> int:
    return int(sum(1 for left, right in zip(labels, labels[1:]) if int(left) != int(right)))


def _window_indices(center: int, radius: int, n: int) -> list[int]:
    lo = max(0, int(center) - int(radius))
    hi = min(int(n), int(center) + int(radius) + 1)
    return list(range(lo, hi))


def _swap_copy(values: Sequence[Any], a: int, b: int) -> list[Any]:
    out = list(values)
    out[int(a)], out[int(b)] = out[int(b)], out[int(a)]
    return out


def _edge_order_score(values: Sequence[int], indices: Sequence[int], *, reverse: bool) -> int:
    valid = set(int(i) for i in indices)
    score = 0
    for left in sorted(valid):
        right = left + 1
        if right not in valid or right >= len(values):
            continue
        score += int(values[left] >= values[right] if reverse else values[left] <= values[right])
    return int(score)


def _local_order_delta(values: Sequence[int], actor_pos: int, target_pos: int, *, reverse: bool, radius: int) -> float:
    lo = min(int(actor_pos), int(target_pos))
    hi = max(int(actor_pos), int(target_pos))
    indices = list(range(max(0, lo - int(radius)), min(len(values), hi + int(radius) + 1)))
    before = _edge_order_score(values, indices, reverse=reverse)
    after_values = _swap_copy(values, actor_pos, target_pos)
    after = _edge_order_score(after_values, indices, reverse=reverse)
    return float(after - before)


def _interface_delta(labels: Sequence[int], actor_pos: int, target_pos: int) -> int:
    before = _interfaces(labels)
    after = _interfaces(_swap_copy(labels, actor_pos, target_pos))
    return int(after - before)


def _candidate_positions(actor_pos: int, n: int) -> list[int]:
    return [pos for pos in (int(actor_pos) - 1, int(actor_pos) + 1) if 0 <= pos < int(n)]


class GovernanceEventSimulator(InterfaceRuleEventSimulator):
    """Interface-rule simulator with deterministic local governance proposal hooks."""

    def __init__(
        self,
        *args: Any,
        governance_config: GovernanceConfig | Mapping[str, Any] | str | None = None,
        **kwargs: Any,
    ) -> None:
        self.governance_config = GovernanceConfig.from_payload(governance_config or "no_governance_control")
        self.governance_evaluations = 0
        self.governance_interventions = 0
        self.governance_injected_swaps = 0
        self.governance_redirected_swaps = 0
        self.governance_vetoed_swaps = 0
        self.governance_global_information_reads = 0
        self.governance_broad_control_interventions = 0
        self.governance_reason_counts: Counter[str] = Counter()
        super().__init__(*args, **kwargs)
        self._leader_cell_ids = self._select_leader_cell_ids()
        self._pacemaker_cell_ids = self._select_pacemaker_cell_ids()

    def _select_leader_cell_ids(self) -> list[int]:
        by_label: dict[int, int] = {}
        for cell in self.cells:
            by_label.setdefault(int(cell.label), int(cell.cell_id))
        return sorted(by_label.values())

    def _select_pacemaker_cell_ids(self) -> list[int]:
        if not self.cells:
            return []
        positions = sorted(set([len(self.cells) // 4, len(self.cells) // 2, (3 * len(self.cells)) // 4]))
        return sorted(int(self.cells[min(len(self.cells) - 1, max(0, pos))].cell_id) for pos in positions)

    def _near_cell_id(self, actor_pos: int, cell_ids: Sequence[int], radius: int) -> bool:
        for cell_id in cell_ids:
            pos = self.positions_by_id.get(int(cell_id))
            if pos is not None and abs(int(pos) - int(actor_pos)) <= int(radius):
                return True
        return False

    def _local_reverse_majority(self, actor_pos: int, radius: int) -> bool:
        positions = _window_indices(actor_pos, radius, len(self.cells))
        reverse_votes = sum(1 for pos in positions if self.cells[pos].reverse_direction)
        return bool(reverse_votes > len(positions) / 2.0)

    def _local_label_quorum_direction(self, actor_pos: int, radius: int, threshold: float) -> bool | None:
        positions = _window_indices(actor_pos, radius, len(self.cells))
        if not positions:
            return None
        label_counts = Counter(int(self.cells[pos].label) for pos in positions)
        label, count = label_counts.most_common(1)[0]
        if count / len(positions) < float(threshold):
            return None
        reverse_votes = [self.cells[pos].reverse_direction for pos in positions if int(self.cells[pos].label) == int(label)]
        return bool(sum(bool(value) for value in reverse_votes) > len(reverse_votes) / 2.0)

    def _global_target_direction(self, actor_pos: int) -> int:
        values = self.current_values()
        actor_value = values[int(actor_pos)]
        sorted_values = sorted(values)
        candidate_targets = [index for index, value in enumerate(sorted_values) if int(value) == int(actor_value)]
        if not candidate_targets:
            return 0
        ideal = min(candidate_targets, key=lambda index: abs(int(index) - int(actor_pos)))
        if ideal > int(actor_pos):
            return 1
        if ideal < int(actor_pos):
            return -1
        return 0

    def _score_candidate(self, actor_pos: int, target_pos: int, family: str) -> float:
        values = self.current_values()
        labels = _labels(self.cells)
        config = self.governance_config
        radius = max(1, min(int(config.influence_range), len(self.cells) - 1))
        if family == "local_voting":
            return _local_order_delta(values, actor_pos, target_pos, reverse=self._local_reverse_majority(actor_pos, radius), radius=radius)
        if family == "leader_cells":
            if not self._near_cell_id(actor_pos, self._leader_cell_ids, radius):
                return 0.0
            leader_reverse = self._local_reverse_majority(actor_pos, radius)
            return _local_order_delta(values, actor_pos, target_pos, reverse=leader_reverse, radius=radius)
        if family == "pacemaker_cells":
            if self.activation_count % int(config.pacemaker_period) != 0:
                return 0.0
            if not self._near_cell_id(actor_pos, self._pacemaker_cell_ids, radius):
                return 0.0
            return _local_order_delta(values, actor_pos, target_pos, reverse=False, radius=radius)
        if family == "quorum_signal":
            reverse = self._local_label_quorum_direction(actor_pos, radius, config.quorum_threshold)
            if reverse is None:
                return 0.0
            return _local_order_delta(values, actor_pos, target_pos, reverse=reverse, radius=radius) - 0.25 * _interface_delta(labels, actor_pos, target_pos)
        if family == "conflict_resolution":
            window = _window_indices(actor_pos, radius, len(self.cells))
            if len({int(self.cells[pos].label) for pos in window}) < 2 and len({bool(self.cells[pos].reverse_direction) for pos in window}) < 2:
                return 0.0
            inc = _local_order_delta(values, actor_pos, target_pos, reverse=False, radius=radius)
            dec = _local_order_delta(values, actor_pos, target_pos, reverse=True, radius=radius)
            return 0.5 * (inc + dec) - 0.35 * _interface_delta(labels, actor_pos, target_pos)
        if family == "organizer_limited":
            if not self._near_cell_id(actor_pos, self._pacemaker_cell_ids, radius):
                return 0.0
            return _local_order_delta(values, actor_pos, target_pos, reverse=False, radius=radius) - 0.10 * _interface_delta(labels, actor_pos, target_pos)
        if family == "organizer_global_upper_bound":
            self.governance_global_information_reads += 1
            direction = self._global_target_direction(actor_pos)
            if direction == 0 or int(target_pos) != int(actor_pos) + int(direction):
                return 0.0
            before = sortedness_percent(values, direction="increasing")
            after = sortedness_percent(_swap_copy(values, actor_pos, target_pos), direction="increasing")
            return float(after - before)
        return 0.0

    def _best_governance_target(self, actor_pos: int) -> tuple[int | None, float]:
        best_target: int | None = None
        best_score = 0.0
        for target in _candidate_positions(actor_pos, len(self.cells)):
            if not self._target_status_allows_policy_check(target):
                continue
            score = self._score_candidate(actor_pos, target, self.governance_config.family)
            if score > best_score + 1e-12:
                best_target = int(target)
                best_score = float(score)
        return best_target, best_score

    def _governance_adjust_proposal(self, actor_pos: int, proposed: ProposedAction) -> ProposedAction:
        config = self.governance_config
        if not config.enabled:
            return proposed
        self.governance_evaluations += 1
        target, best_score = self._best_governance_target(actor_pos)
        current_score = 0.0
        if proposed.action == "swap" and proposed.target_position is not None and self._target_in_bounds(int(proposed.target_position)):
            current_score = self._score_candidate(actor_pos, int(proposed.target_position), config.family)
        if target is not None and best_score > max(0.0, current_score) + 1e-12:
            self.governance_interventions += 1
            if config.broad_control_like:
                self.governance_broad_control_interventions += 1
            reason = "governance_redirected_swap" if proposed.action == "swap" else "governance_injected_swap"
            self.governance_reason_counts[f"{config.variant}:{reason}"] += 1
            if proposed.action == "swap":
                self.governance_redirected_swaps += 1
            else:
                self.governance_injected_swaps += 1
            return ProposedAction.swap(
                int(target),
                comparison_delta=int(proposed.comparison_delta),
                state_updates=proposed.state_updates,
                swapped_reason=f"{reason}:{config.variant}",
                blocked_reason=f"governance_swap_blocked:{config.variant}",
            )
        if proposed.action == "swap" and current_score < -0.25 and config.family in {"quorum_signal", "conflict_resolution"}:
            self.governance_interventions += 1
            self.governance_vetoed_swaps += 1
            self.governance_reason_counts[f"{config.variant}:governance_vetoed_swap"] += 1
            return ProposedAction.wait(
                f"governance_vetoed_swap:{config.variant}",
                target_position=proposed.target_position,
                comparison_delta=int(proposed.comparison_delta),
                state_updates=proposed.state_updates,
            )
        return proposed

    def step(
        self,
        *,
        forced_cell_id: int | None = None,
        forced_direction: int | None = None,
    ) -> StepOutcome:
        eligible = self.eligible_cell_ids()
        if not eligible:
            return StepOutcome(False, None, None, None, None, False, 0, 0, False, "no_eligible_cells")
        if forced_cell_id is None:
            actor_cell_id = int(self.scheduler_rng.choice(eligible))
        else:
            actor_cell_id = int(forced_cell_id)
            if actor_cell_id not in eligible:
                return StepOutcome(False, actor_cell_id, None, None, None, False, 0, 0, False, "forced_cell_not_eligible")

        trace_count_before = len(self.trace_rows)
        actor_pos = self.positions_by_id[actor_cell_id]
        self._set_actor_signal_state(actor_pos)
        actor = self.cells[actor_pos]
        self.activation_count += 1
        archived_delta = 1 if self._original_style_move_opportunity(actor_pos) else 0
        self.archived_compare_and_swap_count += archived_delta
        observation = actor.policy.observe(self.cells, actor_pos, actor.state, self.frozen_variant)
        proposed = actor.policy.propose_action(observation, actor.state, self.tie_rng, forced_direction=forced_direction)
        proposed = actor.policy.constrain_action(observation, actor.state, proposed)
        proposed = self._governance_adjust_proposal(actor_pos, proposed)
        self.comparison_count += int(proposed.comparison_delta)
        actor.state.update(proposed.state_updates)

        if proposed.action != "swap" or proposed.target_position is None:
            outcome = StepOutcome(
                True,
                actor.cell_id,
                actor_pos,
                actor_pos,
                proposed.target_position,
                False,
                int(proposed.comparison_delta),
                archived_delta,
                False,
                proposed.reason,
            )
            actor.policy.update_state(actor.state, observation, proposed, outcome)
            self._apply_signal_update(actor_pos, observation, proposed, outcome)
            if self.trace_signal_activations and len(self.trace_rows) == trace_count_before:
                self._append_trace_row(
                    event_kind="signal_update",
                    activation_index=self.activation_count,
                    actor_cell_id=outcome.actor_cell_id,
                    actor_algotype=actor.algotype,
                    target_position=outcome.target_position,
                )
            return outcome

        target_pos = int(proposed.target_position)
        if not self._target_status_allows_policy_check(target_pos):
            outcome = StepOutcome(
                True,
                actor.cell_id,
                actor_pos,
                actor_pos,
                target_pos,
                False,
                int(proposed.comparison_delta),
                archived_delta,
                False,
                "target_oob",
            )
            actor.policy.update_state(actor.state, observation, proposed, outcome)
            self._apply_signal_update(actor_pos, observation, proposed, outcome)
            if self.trace_signal_activations and len(self.trace_rows) == trace_count_before:
                self._append_trace_row(
                    event_kind="signal_update",
                    activation_index=self.activation_count,
                    actor_cell_id=outcome.actor_cell_id,
                    actor_algotype=actor.algotype,
                    target_position=outcome.target_position,
                )
            return outcome
        if not self._can_swap(actor_pos, target_pos):
            outcome = StepOutcome(
                True,
                actor.cell_id,
                actor_pos,
                actor_pos,
                target_pos,
                False,
                int(proposed.comparison_delta),
                archived_delta,
                True,
                proposed.blocked_reason,
            )
            actor.policy.update_state(actor.state, observation, proposed, outcome)
            self._apply_signal_update(actor_pos, observation, proposed, outcome)
            if self.trace_signal_activations and len(self.trace_rows) == trace_count_before:
                self._append_trace_row(
                    event_kind="signal_update",
                    activation_index=self.activation_count,
                    actor_cell_id=outcome.actor_cell_id,
                    actor_algotype=actor.algotype,
                    target_position=outcome.target_position,
                )
            return outcome

        self._swap(actor_pos, target_pos)
        outcome = StepOutcome(
            True,
            actor.cell_id,
            actor_pos,
            target_pos,
            target_pos,
            True,
            int(proposed.comparison_delta),
            archived_delta,
            False,
            proposed.swapped_reason,
        )
        actor.policy.update_state(actor.state, observation, proposed, outcome)
        self._apply_signal_update(target_pos, observation, proposed, outcome)
        self._append_trace_row(
            event_kind="swap",
            activation_index=self.activation_count,
            actor_cell_id=outcome.actor_cell_id,
            actor_algotype=actor.algotype,
            target_position=outcome.target_position,
        )
        return outcome

    def governance_summary(self) -> dict[str, Any]:
        return {
            "governanceConfig": self.governance_config.to_dict(),
            "governanceEvaluations": int(self.governance_evaluations),
            "governanceInterventions": int(self.governance_interventions),
            "governanceInjectedSwaps": int(self.governance_injected_swaps),
            "governanceRedirectedSwaps": int(self.governance_redirected_swaps),
            "governanceVetoedSwaps": int(self.governance_vetoed_swaps),
            "governanceGlobalInformationReads": int(self.governance_global_information_reads),
            "governanceBroadControlInterventions": int(self.governance_broad_control_interventions),
            "governanceReasonCounts": dict(self.governance_reason_counts),
        }


def load_winner_criteria(path: Path = DEFAULT_S06_WINNER_CRITERIA_PATH) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return predefined_winner_criteria(load_boundary_caveats(DEFAULT_S05_BOUNDARY_CAVEATS_PATH))


def load_s08_results(path: Path = DEFAULT_S08_RESULTS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"S08 interface-rule results missing: {path}")
    df = pd.read_parquet(path)
    required = {
        "conditionId",
        "s08SourceConditionId",
        "interfaceRuleVariant",
        "policyIdsJson",
        "arrangementPolicyIdsJson",
        "goalMode",
        "targetCountsJson",
        "valueSeed",
        "schedulerSeed",
        "tieBreakerSeed",
        "finalStateHash",
        "finalValuesJson",
        "finalPanelPolicyIdsJson",
        "s07MosaicClass",
        "finalInterfaceDensityS07",
        "finalTargetQualityScore",
        "dominanceProxyScore",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"S08 interface results missing required columns: {missing}")
    if df.empty:
        raise ValueError("S08 interface results are empty")
    return df


def select_s09_contexts(s08_results: pd.DataFrame, *, max_contexts: int = 18) -> pd.DataFrame:
    disabled = s08_results[s08_results["interfaceRuleVariant"].astype(str) == "disabled_baseline"].copy()
    if disabled.empty:
        raise ValueError("S08 results contain no disabled_baseline rows")
    enabled = s08_results[s08_results["interfaceRuleVariant"].astype(str) != "disabled_baseline"].copy()
    rows: list[dict[str, Any]] = []
    for source_id, group in enabled.groupby("s08SourceConditionId", sort=False):
        class_change = (~group["s08ClassMatchesBaseline"].map(bool)).mean() if "s08ClassMatchesBaseline" in group.columns else 0.0
        max_interface_delta = pd.to_numeric(group.get("deltaFinalInterfaceDensityS07", pd.Series(dtype=float)), errors="coerce").abs().max()
        max_target_delta = pd.to_numeric(group.get("deltaFinalTargetQualityScore", pd.Series(dtype=float)), errors="coerce").abs().max()
        max_blocks = pd.to_numeric(group.get("interfaceRuleBlockedSwaps", pd.Series(dtype=float)), errors="coerce").max()
        rows.append(
            {
                "s08SourceConditionId": str(source_id),
                "s09InterfaceSensitiveScore": float(
                    5.0 * safe_float(class_change, 0.0)
                    + 2.0 * safe_float(max_interface_delta, 0.0)
                    + safe_float(max_target_delta, 0.0)
                    + min(1.0, safe_float(max_blocks, 0.0) / 1000.0)
                ),
                "s09InterfaceSensitivityTagsJson": compact_json(
                    [
                        tag
                        for tag, flag in [
                            ("class_changed_under_interface_rules", safe_float(class_change, 0.0) > 0.0),
                            ("large_interface_density_delta", safe_float(max_interface_delta, 0.0) >= 0.05),
                            ("target_quality_shift", safe_float(max_target_delta, 0.0) >= 0.02),
                            ("many_interface_blocks", safe_float(max_blocks, 0.0) >= 500.0),
                        ]
                        if flag
                    ]
                ),
            }
        )
    sensitivity = pd.DataFrame(rows)
    selected = disabled.merge(sensitivity, on="s08SourceConditionId", how="left")
    selected["s09InterfaceSensitiveScore"] = pd.to_numeric(selected["s09InterfaceSensitiveScore"], errors="coerce").fillna(0.0)
    selected = selected.sort_values(
        ["s09InterfaceSensitiveScore", "s08ContextIndex", "conditionId"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(int(max_contexts))
    selected = selected.reset_index(drop=True)
    selected.insert(0, "s09ContextIndex", np.arange(len(selected), dtype=int))
    selected["s09SourceS08ConditionId"] = selected["conditionId"].astype(str)
    selected["s09BaselineS08ConditionId"] = selected["conditionId"].astype(str)
    selected["s09BaselineSourceConditionId"] = selected["s08SourceConditionId"].astype(str)
    selected["s09BaselineS07MosaicClass"] = selected["s07MosaicClass"].astype(str)
    selected["s09BaselineFinalStateHash"] = selected["finalStateHash"].astype(str)
    selected["s09BaselineFinalValuesJson"] = selected["finalValuesJson"].astype(str)
    selected["s09BaselineFinalPanelPolicyIdsJson"] = selected["finalPanelPolicyIdsJson"].astype(str)
    selected["s09BaselineStopReason"] = selected["stopReason"].astype(str)
    selected["s09BaselineSwapCount"] = pd.to_numeric(selected["swapCount"], errors="coerce").fillna(-1).astype(int)
    selected["s09BaselineActivationCount"] = pd.to_numeric(selected["activationCount"], errors="coerce").fillna(-1).astype(int)
    return selected


def build_governance_condition_matrix(
    selected_contexts: pd.DataFrame,
    configs: Sequence[GovernanceConfig] | None = None,
    goal_modes: Sequence[str] = GOVERNANCE_GOAL_MODES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configs = tuple(configs or governance_configs())
    parameter_df = governance_parameter_table(configs)
    rows: list[dict[str, Any]] = []
    goal_rows: list[dict[str, Any]] = []
    for context in selected_contexts.to_dict(orient="records"):
        policy_ids = [str(item) for item in parse_json_maybe(context["policyIdsJson"], [])]
        assigned = [str(item) for item in parse_json_maybe(context["arrangementPolicyIdsJson"], [])]
        for goal_index, goal_mode in enumerate(goal_modes):
            specs = policy_goal_specs(policy_ids, str(goal_mode))
            reverse = reverse_directions_for_assignment(assigned, specs)
            goal_meta = condition_goal_metadata(policy_ids, str(goal_mode))
            for governance_index, config in enumerate(configs):
                row = dict(context)
                row["conditionId"] = f"s09_c{int(context['s09ContextIndex']):03d}_{goal_mode}_{config.variant}"
                row["researchStepId"] = STEP_ID
                row["sourceResearchStepId"] = "S08"
                row["implementationPrefix"] = "e06_s09"
                row["governanceVersion"] = GOVERNANCE_VERSION
                row["governanceIndex"] = int(governance_index)
                row["governanceVariant"] = config.variant
                row["governanceEnabled"] = bool(config.enabled)
                row["governanceFamily"] = config.family
                row["governanceConfigJson"] = compact_json(config.to_dict())
                row["governanceConfigHash"] = hashlib.sha256(row["governanceConfigJson"].encode("utf-8")).hexdigest()[:16]
                row["influenceRange"] = int(config.influence_range)
                row["informationAccess"] = config.information_access
                row["localInformationOnly"] = bool(config.local_information_only)
                row["usesGlobalState"] = bool(config.uses_global_state)
                row["usesTargetMap"] = bool(config.uses_target_map)
                row["usesOrganizer"] = bool(config.uses_organizer)
                row["broadControlLike"] = bool(config.broad_control_like)
                row["goalMode"] = str(goal_mode)
                row["goalModeIndex"] = int(goal_index)
                row["goalCompatibilityClass"] = str(goal_meta["compatibilityClass"])
                row["conditionGoalMetadataJson"] = compact_json(goal_meta)
                row["policyGoalMetadataJson"] = compact_json(specs)
                row["goalReverseDirectionsJson"] = compact_json(reverse)
                row["goalDirectionByPolicyJson"] = compact_json({pid: specs[pid]["targetDirection"] for pid in policy_ids})
                row["goalBehaviorDirectionByPolicyJson"] = compact_json({pid: specs[pid]["behaviorDirection"] for pid in policy_ids})
                row["interfaceRuleVersion"] = INTERFACE_RULE_VERSION
                row["interfaceRuleVariant"] = "disabled_baseline"
                row["interfaceRuleEnabled"] = False
                row["interfaceRuleFamily"] = "disabled"
                row["interfaceRuleConfigJson"] = compact_json({"schema": INTERFACE_RULE_VERSION, "variant": "disabled_baseline", "enabled": False})
                row["conditionHash"] = hashlib.sha256(
                    compact_json(
                        {
                            "s09SourceS08ConditionId": context["conditionId"],
                            "goalMode": goal_mode,
                            "governance": config.to_dict(),
                        }
                    ).encode("utf-8")
                ).hexdigest()[:16]
                rows.append(row)
            goal_rows.append(
                {
                    "s09ContextIndex": int(context["s09ContextIndex"]),
                    "s09SourceS08ConditionId": str(context["conditionId"]),
                    "goalMode": str(goal_mode),
                    "goalCompatibilityClass": str(goal_meta["compatibilityClass"]),
                    "conditionGoalMetadataJson": compact_json(goal_meta),
                    "goalReverseDirectionsJson": compact_json(reverse),
                }
            )
    return pd.DataFrame(rows).reset_index(drop=True), parameter_df, pd.DataFrame(goal_rows)


def _governance_worker_init(panel_records: Sequence[Mapping[str, Any]]) -> None:
    global _PANEL_LOOKUP
    _PANEL_LOOKUP = {str(row["panelPolicyId"]): dict(row) for row in panel_records}


def run_governance_condition(
    condition: Mapping[str, Any],
    panel_lookup: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    max_activations: int = 6000,
    max_swaps: int = 4000,
    max_comparisons: int = 30000,
) -> dict[str, Any]:
    panel_lookup = panel_lookup or _PANEL_LOOKUP
    ids = [str(item) for item in parse_json_maybe(condition["policyIdsJson"], [])]
    records = [dict(panel_lookup[pid]) for pid in ids]
    assigned_ids, numeric_labels, arrangement_hash = policy_assignment(condition)
    policy_cache = {pid: instantiate_panel_policy(panel_lookup[pid]) for pid in ids}
    policies = [policy_cache[pid] for pid in assigned_ids]
    reverse = [bool(value) for value in parse_json_maybe(condition.get("goalReverseDirectionsJson", ""), [])]
    if len(reverse) != int(condition["n"]):
        raise ValueError("goalReverseDirectionsJson length must match n")
    values = initial_values(int(condition["valueSeed"]), int(condition["n"]), str(condition.get("valueProfile", "random_unique")))
    source_backend = simulator_backend(records)
    first_signal = next((policy for policy in policy_cache.values() if hasattr(policy, "signal_config")), None)
    signal_config = getattr(first_signal, "signal_config", "no_signal")
    signal_config_json = compact_json(signal_config.to_dict()) if hasattr(signal_config, "to_dict") else compact_json(signal_config)
    config = GovernanceConfig.from_payload(condition.get("governanceConfigJson", "no_governance_control"))
    started = time.perf_counter()
    try:
        simulator = GovernanceEventSimulator(
            values,
            policies,
            labels=numeric_labels,
            reverse_directions=reverse,
            scheduler_seed=int(condition["schedulerSeed"]),
            tie_breaker_seed=int(condition["tieBreakerSeed"]),
            condition_id=str(condition["conditionId"]),
            research_step_id=STEP_ID,
            signal_config=signal_config,
            auto_wrap_policies=False,
            trace_signal_activations=False,
            trace_memory_activations=False,
            interface_rule_config="disabled_baseline",
            governance_config=config,
            implementation=f"{condition.get('implementationPrefix', 'e06_s09')}_governance",
        )
        result = simulator.run(
            max_activations=int(max_activations),
            max_swaps=int(max_swaps),
            max_comparisons=int(max_comparisons),
            no_move_check_interval=int(condition["n"]),
        )
        final_labels = [int(cell.label) for cell in simulator.cells]
        final_ids = [ids[label] for label in final_labels]
        initial_ids = list(assigned_ids)
        final_values = simulator.current_values()
        initial_counts = Counter(initial_ids)
        final_counts = Counter(final_ids)
        norm_positions = np.linspace(0.0, 1.0, num=len(final_ids)) if len(final_ids) > 1 else np.asarray([0.0])
        mean_positions = {
            pid: float(np.mean([norm_positions[i] for i, value in enumerate(final_ids) if value == pid]))
            for pid in ids
        }
        left_id = str(condition["leftPanelPolicyId"])
        right_id = str(condition["rightPanelPolicyId"])
        left_mean = mean_positions.get(left_id)
        right_mean = mean_positions.get(right_id)
        minority_count = min(initial_counts.values()) if initial_counts else 0
        minority_ids = sorted(pid for pid, count in initial_counts.items() if count == minority_count)
        final_aggregation = aggregation(final_ids)
        initial_aggregation = aggregation(initial_ids)
        completed = sortedness_percent(final_values) >= 100.0 - 1e-9
        if completed:
            final_class = "sorted"
        elif str(result.stop_reason).startswith("max_"):
            final_class = "resource_capped_partial"
        elif result.stop_reason == "no_cell_can_move_after_two_checks":
            final_class = "no_move_partial"
        else:
            final_class = "partial"
        governance_summary = simulator.governance_summary()
        interface_summary = simulator.interface_summary()
        return {
            **{key: json_ready(value) for key, value in condition.items()},
            "runSucceeded": True,
            "runError": "",
            "simulationBackend": "governance_event",
            "sourceSimulationBackend": source_backend,
            "signalConfigJson": signal_config_json,
            "initialValuesJson": compact_json(values),
            "finalValuesJson": compact_json(final_values),
            "initialStateHash": state_hash(values),
            "finalStateHash": state_hash(final_values),
            "initialPolicyAssignmentHash": arrangement_hash,
            "initialPanelPolicyIdsJson": compact_json(initial_ids),
            "finalPanelPolicyIdsJson": compact_json(final_ids),
            "actualCountsJson": compact_json(dict(initial_counts)),
            "finalCountsJson": compact_json(dict(final_counts)),
            "actualProportionsJson": compact_json({pid: initial_counts[pid] / int(condition["n"]) for pid in ids}),
            "actualRatiosMatchTarget": compact_json(dict(initial_counts)) == str(condition["targetCountsJson"]),
            "valueCountsConserved": Counter(values) == Counter(final_values),
            "policyCountsConserved": initial_counts == final_counts,
            "completed": bool(completed),
            "stopReason": result.stop_reason,
            "finalStateClass": final_class,
            "initialSortednessPercent": sortedness_percent(values),
            "finalSortednessPercent": sortedness_percent(final_values),
            "sortednessGain": sortedness_percent(final_values) - sortedness_percent(values),
            "initialAggregation": float(initial_aggregation),
            "finalAggregation": float(final_aggregation),
            "aggregationDelta": float(final_aggregation - initial_aggregation),
            "aggregationClass": "increased" if final_aggregation > initial_aggregation + 0.05 else ("decreased" if final_aggregation < initial_aggregation - 0.05 else "stable"),
            "swapCount": int(result.swap_count),
            "comparisonCount": int(result.comparison_count),
            "activationCount": int(result.activation_count),
            "eventCount": int(result.event_count),
            "blockedMoveAttempts": int(result.blocked_move_attempts),
            "frozenSwapAttempts": int(result.frozen_swap_attempts),
            "minorityPanelPolicyIdsJson": compact_json(minority_ids),
            "minorityInitialCount": int(minority_count),
            "minorityPersistence": 1.0 if minority_count > 0 and all(final_counts[pid] == initial_counts[pid] for pid in minority_ids) else 0.0,
            "leftPolicyMeanFinalPosition": left_mean,
            "rightPolicyMeanFinalPosition": right_mean,
            "leftMinusRightPositionBias": None if left_mean is None or right_mean is None else float(right_mean - left_mean),
            "meanFinalPositionByPolicyJson": compact_json(mean_positions),
            "interfaceRuleEvaluations": int(interface_summary["interfaceRuleEvaluations"]),
            "interfaceRuleAllowedSwaps": int(interface_summary["interfaceRuleAllowedSwaps"]),
            "interfaceRuleBlockedSwaps": int(interface_summary["interfaceRuleBlockedSwaps"]),
            "interfaceCrossBoundaryAttempts": int(interface_summary["interfaceCrossBoundaryAttempts"]),
            "interfaceRuleBlockReasonCountsJson": compact_json(interface_summary["interfaceRuleBlockReasonCounts"]),
            "governanceEvaluations": int(governance_summary["governanceEvaluations"]),
            "governanceInterventions": int(governance_summary["governanceInterventions"]),
            "governanceInjectedSwaps": int(governance_summary["governanceInjectedSwaps"]),
            "governanceRedirectedSwaps": int(governance_summary["governanceRedirectedSwaps"]),
            "governanceVetoedSwaps": int(governance_summary["governanceVetoedSwaps"]),
            "governanceGlobalInformationReads": int(governance_summary["governanceGlobalInformationReads"]),
            "governanceBroadControlInterventions": int(governance_summary["governanceBroadControlInterventions"]),
            "governanceReasonCountsJson": compact_json(governance_summary["governanceReasonCounts"]),
            "runtimeSeconds": float(time.perf_counter() - started),
            "claimBoundary": CLAIM_BOUNDARY,
        }
    except Exception as exc:  # pragma: no cover - retained for artifact-level failure accounting
        return {
            **{key: json_ready(value) for key, value in condition.items()},
            "runSucceeded": False,
            "runError": f"{type(exc).__name__}: {exc}",
            "simulationBackend": "governance_event",
            "sourceSimulationBackend": source_backend,
            "runtimeSeconds": float(time.perf_counter() - started),
            "claimBoundary": CLAIM_BOUNDARY,
        }


def _run_governance_worker(condition: Mapping[str, Any], max_activations: int, max_swaps: int, max_comparisons: int) -> dict[str, Any]:
    return run_governance_condition(
        condition,
        max_activations=max_activations,
        max_swaps=max_swaps,
        max_comparisons=max_comparisons,
    )


def run_governance_conditions(
    condition_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    *,
    workers: int = 1,
    max_activations: int = 6000,
    max_swaps: int = 4000,
    max_comparisons: int = 30000,
) -> pd.DataFrame:
    records = [dict(row) for row in condition_df.to_dict(orient="records")]
    panel_records = [dict(row) for row in panel_df.to_dict(orient="records")]
    if int(workers) <= 1:
        lookup = {str(row["panelPolicyId"]): row for row in panel_records}
        return pd.DataFrame(
            [
                run_governance_condition(
                    record,
                    lookup,
                    max_activations=max_activations,
                    max_swaps=max_swaps,
                    max_comparisons=max_comparisons,
                )
                for record in records
            ]
        )
    with ProcessPoolExecutor(max_workers=int(workers), initializer=_governance_worker_init, initargs=(panel_records,)) as pool:
        futures = [
            pool.submit(_run_governance_worker, record, int(max_activations), int(max_swaps), int(max_comparisons))
            for record in records
        ]
        return pd.DataFrame([future.result() for future in futures])


def attach_governance_baseline_deltas(classified: pd.DataFrame) -> pd.DataFrame:
    baseline = classified[classified["governanceVariant"].astype(str) == "no_governance_control"].copy()
    key_cols = ["s09ContextIndex", "goalMode"]
    base_cols = [
        *key_cols,
        "conditionId",
        "s07MosaicClass",
        "finalStateHash",
        "finalValuesJson",
        "finalPanelPolicyIdsJson",
        "stopReason",
        "swapCount",
        "activationCount",
        "finalInterfaceDensityS07",
        "finalAggregationS07",
        "finalTargetQualityScore",
        "dominanceProxyScore",
        "meanPolicyGoalScore",
        "minPolicyGoalScore",
    ]
    renamed = baseline[[col for col in base_cols if col in baseline.columns]].rename(
        columns={
            "conditionId": "s09NoGovernanceConditionId",
            "s07MosaicClass": "noGovernanceMosaicClass",
            "finalStateHash": "noGovernanceFinalStateHash",
            "finalValuesJson": "noGovernanceFinalValuesJson",
            "finalPanelPolicyIdsJson": "noGovernanceFinalPanelPolicyIdsJson",
            "stopReason": "noGovernanceStopReason",
            "swapCount": "noGovernanceSwapCount",
            "activationCount": "noGovernanceActivationCount",
            "finalInterfaceDensityS07": "noGovernanceFinalInterfaceDensityS07",
            "finalAggregationS07": "noGovernanceFinalAggregationS07",
            "finalTargetQualityScore": "noGovernanceFinalTargetQualityScore",
            "dominanceProxyScore": "noGovernanceDominanceProxyScore",
            "meanPolicyGoalScore": "noGovernanceMeanPolicyGoalScore",
            "minPolicyGoalScore": "noGovernanceMinPolicyGoalScore",
        }
    )
    out = classified.merge(renamed, on=key_cols, how="left")
    out["s09MosaicTransition"] = out["noGovernanceMosaicClass"].astype(str) + "->" + out["s07MosaicClass"].astype(str)
    out["s09ClassMatchesNoGovernance"] = out["noGovernanceMosaicClass"].astype(str) == out["s07MosaicClass"].astype(str)
    out["deltaFinalInterfaceDensityS09"] = pd.to_numeric(out["finalInterfaceDensityS07"], errors="coerce") - pd.to_numeric(
        out["noGovernanceFinalInterfaceDensityS07"], errors="coerce"
    )
    out["deltaFinalAggregationS09"] = pd.to_numeric(out["finalAggregationS07"], errors="coerce") - pd.to_numeric(
        out["noGovernanceFinalAggregationS07"], errors="coerce"
    )
    out["deltaFinalTargetQualityScoreS09"] = pd.to_numeric(out["finalTargetQualityScore"], errors="coerce") - pd.to_numeric(
        out["noGovernanceFinalTargetQualityScore"], errors="coerce"
    )
    out["deltaDominanceProxyScoreS09"] = pd.to_numeric(out["dominanceProxyScore"], errors="coerce") - pd.to_numeric(
        out["noGovernanceDominanceProxyScore"], errors="coerce"
    )
    out["deltaMeanPolicyGoalScoreS09"] = pd.to_numeric(out["meanPolicyGoalScore"], errors="coerce") - pd.to_numeric(
        out["noGovernanceMeanPolicyGoalScore"], errors="coerce"
    )
    out["deltaMinPolicyGoalScoreS09"] = pd.to_numeric(out["minPolicyGoalScore"], errors="coerce") - pd.to_numeric(
        out["noGovernanceMinPolicyGoalScore"], errors="coerce"
    )
    info_score = (
        pd.to_numeric(out["influenceRange"], errors="coerce").clip(upper=1000).fillna(0.0) / 1000.0
        + out["usesGlobalState"].map(bool).astype(float) * 2.0
        + out["usesTargetMap"].map(bool).astype(float) * 2.0
        + out["usesOrganizer"].map(bool).astype(float) * 0.5
    )
    intervention_rate = pd.to_numeric(out["governanceInterventions"], errors="coerce").fillna(0.0) / pd.to_numeric(
        out["activationCount"], errors="coerce"
    ).replace(0, np.nan).fillna(1.0)
    out["governanceInterventionRate"] = intervention_rate
    out["governanceInformationAccessScore"] = info_score
    out["governanceCostProxy"] = intervention_rate * (1.0 + info_score)
    out["rescueSuccessProxy"] = (
        (pd.to_numeric(out["deltaMinPolicyGoalScoreS09"], errors="coerce").fillna(0.0) >= 5.0)
        | (pd.to_numeric(out["deltaFinalTargetQualityScoreS09"], errors="coerce").fillna(0.0) >= 0.05)
    ) & ~out["broadControlLike"].map(bool)
    out["upperBoundControlProxy"] = out["broadControlLike"].map(bool)
    return out


def s08_disabled_reproduction(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subset = scored[
        (scored["governanceVariant"].astype(str) == "no_governance_control")
        & (scored["goalMode"].astype(str) == "opposite_goal")
    ]
    for row in subset.to_dict(orient="records"):
        rows.append(
            {
                "conditionId": str(row["conditionId"]),
                "s09SourceS08ConditionId": str(row.get("s09SourceS08ConditionId", "")),
                "s09BaselineSourceConditionId": str(row.get("s09BaselineSourceConditionId", "")),
                "finalStateHashMatchesS08": bool(str(row.get("finalStateHash", "")) == str(row.get("s09BaselineFinalStateHash", ""))),
                "finalValuesMatchS08": bool(str(row.get("finalValuesJson", "")) == str(row.get("s09BaselineFinalValuesJson", ""))),
                "finalPolicyIdsMatchS08": bool(str(row.get("finalPanelPolicyIdsJson", "")) == str(row.get("s09BaselineFinalPanelPolicyIdsJson", ""))),
                "stopReasonMatchesS08": bool(str(row.get("stopReason", "")) == str(row.get("s09BaselineStopReason", ""))),
                "swapCountMatchesS08": bool(int(row.get("swapCount", -1)) == int(row.get("s09BaselineSwapCount", -2))),
                "activationCountMatchesS08": bool(int(row.get("activationCount", -1)) == int(row.get("s09BaselineActivationCount", -2))),
                "baselineS07MosaicClass": str(row.get("s09BaselineS07MosaicClass", "")),
                "s09S07MosaicClass": str(row.get("s07MosaicClass", "")),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    check_cols = [
        "finalStateHashMatchesS08",
        "finalValuesMatchS08",
        "finalPolicyIdsMatchS08",
        "stopReasonMatchesS08",
        "swapCountMatchesS08",
        "activationCountMatchesS08",
    ]
    out["s08DisabledBaselineExactReproduction"] = out[check_cols].all(axis=1)
    return out


def governance_effect_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["governanceVariant", "governanceFamily", "goalMode"]
    for keys, group in scored.groupby(group_cols, dropna=False, sort=False):
        variant, family, goal_mode = keys
        rows.append(
            {
                "governanceVariant": str(variant),
                "governanceFamily": str(family),
                "goalMode": str(goal_mode),
                "conditionCount": int(len(group)),
                "runSuccessRate": float(group["runSucceeded"].map(bool).mean()),
                "classMatchNoGovernanceRate": float(group["s09ClassMatchesNoGovernance"].map(bool).mean()),
                "rescueSuccessProxyRate": float(group["rescueSuccessProxy"].map(bool).mean()),
                "meanDeltaFinalTargetQualityScore": float(group["deltaFinalTargetQualityScoreS09"].mean()),
                "meanDeltaMinPolicyGoalScore": float(group["deltaMinPolicyGoalScoreS09"].mean()),
                "meanDeltaFinalInterfaceDensity": float(group["deltaFinalInterfaceDensityS09"].mean()),
                "meanGovernanceInterventions": float(pd.to_numeric(group["governanceInterventions"], errors="coerce").mean()),
                "meanGovernanceInterventionRate": float(pd.to_numeric(group["governanceInterventionRate"], errors="coerce").mean()),
                "meanGovernanceCostProxy": float(pd.to_numeric(group["governanceCostProxy"], errors="coerce").mean()),
                "broadControlLike": bool(group["broadControlLike"].map(bool).any()),
                "transitionCountsJson": compact_json(dict(Counter(group["s09MosaicTransition"].astype(str)))),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rankScore"] = (
        pd.to_numeric(out["rescueSuccessProxyRate"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["meanDeltaFinalTargetQualityScore"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["meanDeltaMinPolicyGoalScore"], errors="coerce").fillna(0.0) / 100.0
        - pd.to_numeric(out["meanGovernanceCostProxy"], errors="coerce").fillna(0.0)
        - out["broadControlLike"].map(bool).astype(float) * 0.25
    )
    return out.sort_values(["goalMode", "rankScore"], ascending=[True, False], kind="mergesort").reset_index(drop=True)


def governance_context_summary(scored: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["governanceVariant", "s09BaselineS07MosaicClass", "goalMode", "pairCategory", "arrangementType", "valueProfile"]
    rows: list[dict[str, Any]] = []
    for keys, group in scored.groupby(group_cols, dropna=False, sort=False):
        payload = dict(zip(group_cols, keys, strict=True))
        rows.append(
            {
                **payload,
                "conditionCount": int(len(group)),
                "classMatchNoGovernanceRate": float(group["s09ClassMatchesNoGovernance"].map(bool).mean()),
                "rescueSuccessProxyRate": float(group["rescueSuccessProxy"].map(bool).mean()),
                "meanDeltaFinalTargetQualityScore": float(group["deltaFinalTargetQualityScoreS09"].mean()),
                "meanDeltaMinPolicyGoalScore": float(group["deltaMinPolicyGoalScoreS09"].mean()),
                "meanGovernanceInterventions": float(pd.to_numeric(group["governanceInterventions"], errors="coerce").mean()),
                "transitionCountsJson": compact_json(dict(Counter(group["s09MosaicTransition"].astype(str)))),
            }
        )
    return pd.DataFrame(rows)


def governance_information_audit(parameter_df: pd.DataFrame) -> pd.DataFrame:
    out = parameter_df.copy()
    out["globalControlFlag"] = (
        out["usesGlobalState"].map(bool)
        | out["usesTargetMap"].map(bool)
        | out["broadControlLike"].map(bool)
        | (pd.to_numeric(out["influenceRange"], errors="coerce").fillna(0) > 10)
    )
    out["auditInterpretation"] = np.where(
        out["globalControlFlag"],
        "explicit upper-bound or broad-control-like intervention; not local-only evidence",
        "local or range-limited governance mechanism",
    )
    return out


def validation_checks(
    selected_contexts: pd.DataFrame,
    condition_df: pd.DataFrame,
    parameter_df: pd.DataFrame,
    scored: pd.DataFrame,
    reproduction: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> pd.DataFrame:
    no_gov = scored[scored["governanceVariant"].astype(str) == "no_governance_control"]
    enabled = scored[scored["governanceEnabled"].map(bool)]
    enabled_by_variant = (
        enabled.groupby("governanceVariant", dropna=False)
        .agg(
            totalEvaluations=("governanceEvaluations", "sum"),
            totalInterventions=("governanceInterventions", "sum"),
        )
        if not enabled.empty
        else pd.DataFrame()
    )
    reproduction_ok = bool(
        not reproduction.empty
        and reproduction["s08DisabledBaselineExactReproduction"].map(bool).all()
        and len(reproduction) == len(selected_contexts)
    )
    checks = [
        {
            "checkId": "s08_disabled_baselines_loaded",
            "success": bool(not selected_contexts.empty and selected_contexts["s09BaselineFinalStateHash"].astype(str).str.len().gt(0).all()),
            "detail": f"{len(selected_contexts)} S08 disabled-baseline contexts selected",
        },
        {
            "checkId": "condition_matrix_crosses_governance_and_goals",
            "success": bool(
                len(condition_df) == len(selected_contexts) * len(GOVERNANCE_GOAL_MODES) * len(GOVERNANCE_VARIANTS)
                and set(GOVERNANCE_VARIANTS) <= set(condition_df["governanceVariant"].astype(str))
                and set(GOVERNANCE_GOAL_MODES) <= set(condition_df["goalMode"].astype(str))
            ),
            "detail": f"{len(condition_df)} conditions across {len(GOVERNANCE_VARIANTS)} governance variants and {len(GOVERNANCE_GOAL_MODES)} goal modes",
        },
        {
            "checkId": "no_governance_controls_included",
            "success": bool(
                len(no_gov) == len(selected_contexts) * len(GOVERNANCE_GOAL_MODES)
                and no_gov.groupby(["s09ContextIndex", "goalMode"]).size().eq(1).all()
            ),
            "detail": "one no-governance control per S09 context and goal mode",
        },
        {
            "checkId": "influence_range_and_information_access_logged",
            "success": bool(
                parameter_df["informationAccess"].astype(str).str.len().gt(0).all()
                and parameter_df["governanceConfigJson"].astype(str).str.len().gt(10).all()
                and condition_df["informationAccess"].astype(str).str.len().gt(0).all()
            ),
            "detail": "mechanism parameter table and condition rows log influence range and information access",
        },
        {
            "checkId": "broad_organizer_global_control_flagged",
            "success": bool(
                audit_df[audit_df["governanceVariant"].astype(str) == "organizer_global_upper_bound"]["globalControlFlag"].map(bool).all()
                and scored[scored["governanceVariant"].astype(str) == "organizer_global_upper_bound"]["broadControlLike"].map(bool).all()
            ),
            "detail": "broad organizer/global-target-map mechanism is explicitly flagged as upper-bound control",
        },
        {
            "checkId": "s08_opposite_no_governance_reproduction",
            "success": reproduction_ok,
            "detail": (
                f"{int(reproduction['s08DisabledBaselineExactReproduction'].sum()) if not reproduction.empty else 0}/"
                f"{len(reproduction)} no-governance opposite-goal rows exactly reproduced S08 disabled baselines"
            ),
        },
        {
            "checkId": "one_result_per_condition",
            "success": bool(len(scored) == len(condition_df) and scored["conditionId"].is_unique),
            "detail": f"{len(scored)} result rows for {len(condition_df)} S09 conditions",
        },
        {
            "checkId": "runs_succeeded_and_counts_conserved",
            "success": bool(
                scored["runSucceeded"].fillna(False).map(bool).all()
                and scored["valueCountsConserved"].fillna(False).map(bool).all()
                and scored["policyCountsConserved"].fillna(False).map(bool).all()
            ),
            "detail": "all S09 runs succeeded and conserved value/policy counts",
        },
        {
            "checkId": "governance_variants_exercised",
            "success": bool(
                not enabled_by_variant.empty
                and enabled_by_variant["totalEvaluations"].gt(0).all()
                and enabled_by_variant["totalInterventions"].gt(0).all()
            ),
            "detail": f"enabled governance variants evaluated {int(pd.to_numeric(enabled.get('governanceEvaluations', pd.Series(dtype=int)), errors='coerce').sum()) if not enabled.empty else 0} activations",
        },
        {
            "checkId": "mosaic_scores_and_deltas_populated",
            "success": bool(
                scored["s07MosaicClass"].astype(str).isin(set(REQUESTED_MOSAIC_CLASSES)).all()
                and scored["deltaFinalTargetQualityScoreS09"].notna().all()
                and scored["deltaMinPolicyGoalScoreS09"].notna().all()
            ),
            "detail": f"S09 rows classified with {MOSAIC_CLASSIFIER_VERSION} and no-governance deltas populated",
        },
        {
            "checkId": "metric_boundary_caveats_retained",
            "success": bool(
                scored["metricBoundary"].astype(str).str.len().gt(0).all()
                and scored["claimBoundary"].astype(str).str.contains("Computational").all()
            ),
            "detail": "inherited metric-boundary caveats and computational claim boundary retained",
        },
    ]
    return pd.DataFrame(checks)


def write_governance_plots(effect_summary: pd.DataFrame, figure_dir: Path, step_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if effect_summary.empty:
        return paths
    plot_df = effect_summary.copy()
    variants = list(dict.fromkeys(plot_df["governanceVariant"].astype(str)))
    goals = list(dict.fromkeys(plot_df["goalMode"].astype(str)))
    x = np.arange(len(variants))
    width = 0.8 / max(1, len(goals))
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, goal in enumerate(goals):
        subset = plot_df[plot_df["goalMode"].astype(str) == goal].set_index("governanceVariant")
        values = [float(subset.loc[variant, "meanDeltaFinalTargetQualityScore"]) if variant in subset.index else 0.0 for variant in variants]
        ax.bar(x + idx * width - 0.4 + width / 2, values, width=width, label=goal)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels=variants, rotation=35, ha="right")
    ax.set_ylabel("Mean target-quality delta vs no-governance")
    ax.set_title("E06 S09 governance mechanism effects")
    ax.legend()
    fig.tight_layout()
    for path in [
        figure_dir / "e06_s09_governance_effects.png",
        figure_dir / "e06_s09_governance_effects.pdf",
        step_dir / "governance_effects.png",
        step_dir / "governance_effects.pdf",
    ]:
        fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
        paths.append(path)
    plt.close(fig)

    rank = effect_summary.groupby("governanceVariant", dropna=False)["rankScore"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(rank.index.astype(str), rank.to_numpy(dtype=float), color="#4f7c8a")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(rank)), labels=rank.index.astype(str), rotation=35, ha="right")
    ax.set_ylabel("Mean rank score")
    ax.set_title("E06 S09 governance rank score")
    fig.tight_layout()
    for path in [
        figure_dir / "e06_s09_governance_rank.png",
        figure_dir / "e06_s09_governance_rank.pdf",
        step_dir / "governance_rank.png",
        step_dir / "governance_rank.pdf",
    ]:
        fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def write_markdown_report(
    *,
    step_dir: Path,
    status: Mapping[str, Any],
    artifacts_written: Sequence[str],
    effect_summary: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> Path:
    top = effect_summary.sort_values("rankScore", ascending=False, kind="mergesort").head(8)
    effect_lines = "\n".join(
        f"- `{row.governanceVariant}` / `{row.goalMode}`: rescue proxy {row.rescueSuccessProxyRate:.2f}, "
        f"target-quality delta {row.meanDeltaFinalTargetQualityScore:.3f}, "
        f"min-goal delta {row.meanDeltaMinPolicyGoalScore:.1f}, cost {row.meanGovernanceCostProxy:.3f}, "
        f"broad-control flag {bool(row.broadControlLike)}"
        for row in top.itertuples(index=False)
    ) or "- No governance summaries available."
    text = f"""# Research Step S09: Test governance mechanisms

## Completion status

Research step ID: `S09`. {status['status']} on {status['completedAt']}. Outcome classification: {status['outcomeClassification']}.

## Artifacts written

{chr(10).join(f"- `{path}`" for path in artifacts_written)}

## Validation result

{status['validationResult']}. Ran {status['conditionCount']} governance conditions across {status['selectedContextCount']} S08 contexts, {status['goalModeCount']} goal modes, and {status['governanceVariantCount']} governance variants. Validation checks passed: {int(validation_df['success'].sum())}/{len(validation_df)}.

## Caveats or blockers

Governance mechanisms are computational local-intervention proxies. Local voting, leader, pacemaker, quorum, conflict-resolution, and limited-organizer mechanisms have explicitly logged local information ranges. The broad organizer uses whole-array target-map information and is flagged as an upper-bound/global-control-like baseline, not local self-organization evidence. Dominance and mosaic labels remain inherited computational proxies.

## Lay summary

S09 tested whether limited local governance cues could steer S08-priority chimeric contexts relative to no-governance controls. The run also included an explicitly broad organizer baseline so its extra information access could be audited rather than mixed with local-only mechanisms.

## Governance effect summary

{effect_lines}

## Recommended next action

{status['recommendedNextAction']}
"""
    path = step_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def run_s09_governance_mechanisms(
    *,
    artifacts_dir: Path | None = None,
    repo_root: Path | None = None,
    panel_path: Path = DEFAULT_PANEL_PATH,
    s08_results_path: Path = DEFAULT_S08_RESULTS_PATH,
    s08_effect_summary_path: Path = DEFAULT_S08_EFFECT_SUMMARY_PATH,
    s08_priority_path: Path = DEFAULT_S07_PRIORITY_PATH,
    s06_pair_summary_path: Path = DEFAULT_S06_PAIR_SUMMARY_PATH,
    winner_criteria_path: Path = DEFAULT_S06_WINNER_CRITERIA_PATH,
    workers: int | None = None,
    max_contexts: int = 18,
    max_activations: int = 6000,
    max_swaps: int = 4000,
    max_comparisons: int = 30000,
) -> dict[str, Any]:
    artifacts_dir = artifacts_dir or Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    workers = int(workers if workers is not None else min(8, os.cpu_count() or 1))
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e06"
    provenance_dir = artifacts_dir / "provenance"
    for path in [step_dir, results_dir, figures_dir, provenance_dir]:
        path.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    ready_panel = load_ready_panel(panel_path)
    s08_results = load_s08_results(s08_results_path)
    s08_effect_summary = pd.read_parquet(s08_effect_summary_path) if s08_effect_summary_path.exists() else pd.DataFrame()
    selected_contexts = select_s09_contexts(s08_results, max_contexts=max_contexts)
    condition_df, parameter_df, goal_metadata_df = build_governance_condition_matrix(selected_contexts)
    audit_df = governance_information_audit(parameter_df)
    criteria = load_winner_criteria(winner_criteria_path)
    pair_summary = pd.read_parquet(s06_pair_summary_path) if s06_pair_summary_path.exists() else pd.DataFrame()

    raw = run_governance_conditions(
        condition_df,
        ready_panel,
        workers=workers,
        max_activations=max_activations,
        max_swaps=max_swaps,
        max_comparisons=max_comparisons,
    )
    goal = augment_goal_results(raw)
    scored = score_dominance_results(goal, criteria)
    scored["researchStepId"] = STEP_ID
    scored["sourceResearchStepId"] = "S09"
    classified = classify_mosaic_formations(scored, pair_summary if not pair_summary.empty else None)
    classified = attach_governance_baseline_deltas(classified)
    reproduction = s08_disabled_reproduction(classified)
    effect_summary = governance_effect_summary(classified)
    context_summary = governance_context_summary(classified)
    validation_df = validation_checks(selected_contexts, condition_df, parameter_df, classified, reproduction, audit_df)

    parameter_csv = step_dir / "governance_mechanism_parameters.csv"
    parameter_parquet = step_dir / "governance_mechanism_parameters.parquet"
    audit_csv = step_dir / "governance_information_audit.csv"
    audit_parquet = step_dir / "governance_information_audit.parquet"
    selected_csv = step_dir / "selected_s08_governance_contexts.csv"
    selected_parquet = step_dir / "selected_s08_governance_contexts.parquet"
    condition_csv = step_dir / "governance_condition_matrix.csv"
    condition_parquet = step_dir / "governance_condition_matrix.parquet"
    goal_csv = step_dir / "governance_goal_metadata.csv"
    goal_parquet = step_dir / "governance_goal_metadata.parquet"
    runs_csv = step_dir / "governance_runs.csv"
    runs_parquet = step_dir / "governance_runs.parquet"
    result_csv = results_dir / "e06_governance_mechanisms.csv"
    result_parquet = results_dir / "e06_governance_mechanisms.parquet"
    combined_csv = results_dir / "e06_governance_interventions.csv"
    combined_parquet = results_dir / "e06_governance_interventions.parquet"
    effect_csv = step_dir / "governance_effect_summary.csv"
    effect_parquet = step_dir / "governance_effect_summary.parquet"
    context_csv = step_dir / "governance_context_summary.csv"
    context_parquet = step_dir / "governance_context_summary.parquet"
    reproduction_csv = step_dir / "s08_disabled_baseline_reproduction.csv"
    reproduction_parquet = step_dir / "s08_disabled_baseline_reproduction.parquet"
    validation_csv = step_dir / "validation_checks.csv"
    validation_parquet = step_dir / "validation_checks.parquet"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    classified["interventionStepId"] = STEP_ID
    classified["interventionType"] = "governance_mechanism"
    compact_classified = compact_result_csv(classified)
    for path, df in [
        (parameter_csv, parameter_df),
        (audit_csv, audit_df),
        (selected_csv, compact_result_csv(selected_contexts)),
        (condition_csv, compact_result_csv(condition_df)),
        (goal_csv, goal_metadata_df),
        (runs_csv, compact_classified),
        (result_csv, compact_classified),
        (effect_csv, effect_summary),
        (context_csv, context_summary),
        (reproduction_csv, reproduction),
        (validation_csv, validation_df),
    ]:
        df.to_csv(path, index=False)
    for path, df in [
        (parameter_parquet, parameter_df),
        (audit_parquet, audit_df),
        (selected_parquet, selected_contexts),
        (condition_parquet, condition_df),
        (goal_parquet, goal_metadata_df),
        (runs_parquet, classified),
        (result_parquet, classified),
        (effect_parquet, effect_summary),
        (context_parquet, context_summary),
        (reproduction_parquet, reproduction),
        (validation_parquet, validation_df),
    ]:
        df.to_parquet(path, index=False)

    if DEFAULT_S08_RESULTS_PATH.exists():
        s08_for_combined = pd.read_parquet(DEFAULT_S08_RESULTS_PATH)
        s08_for_combined["interventionStepId"] = "S08"
        s08_for_combined["interventionType"] = "interface_rule"
        combined = pd.concat([s08_for_combined, classified], ignore_index=True, sort=False)
    else:
        combined = classified.copy()
    compact_result_csv(combined).to_csv(combined_csv, index=False)
    combined.to_parquet(combined_parquet, index=False)

    figure_paths = write_governance_plots(effect_summary, figures_dir, step_dir)
    completed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    validation_result = "passed" if validation_df["success"].map(bool).all() else "failed"
    reproduction_rate = float(reproduction["s08DisabledBaselineExactReproduction"].mean()) if not reproduction.empty else float("nan")
    local_only = classified[~classified["broadControlLike"].map(bool) & classified["governanceEnabled"].map(bool)]
    local_rescue_rate = float(local_only["rescueSuccessProxy"].map(bool).mean()) if not local_only.empty else 0.0
    outcome = "supportive" if validation_result == "passed" and reproduction_rate >= 1.0 else "constraining/contradictory"
    artifacts = [
        parameter_csv,
        parameter_parquet,
        audit_csv,
        audit_parquet,
        selected_csv,
        selected_parquet,
        condition_csv,
        condition_parquet,
        goal_csv,
        goal_parquet,
        runs_csv,
        runs_parquet,
        result_csv,
        result_parquet,
        combined_csv,
        combined_parquet,
        effect_csv,
        effect_parquet,
        context_csv,
        context_parquet,
        reproduction_csv,
        reproduction_parquet,
        validation_csv,
        validation_parquet,
        *figure_paths,
        step_dir / "summary.md",
        status_path,
        manifest_path,
        run_manifest_path,
    ]
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": validation_result == "passed",
        "status": "completed" if validation_result == "passed" else "completed_with_validation_failure",
        "artifactsWritten": [str(path) for path in artifacts],
        "validationResult": validation_result,
        "caveatsOrBlockers": [
            "Governance mechanisms are computational local-intervention proxies, not biological governance mechanisms.",
            "Local-only claims are limited to logged influence ranges and information-access declarations.",
            "The organizer_global_upper_bound variant uses whole-array target-map information and is flagged as broad/global-control-like.",
            "Dominance, rescue, and mosaic labels remain computational proxies inherited from S06/S07/S08.",
            CLAIM_BOUNDARY,
        ],
        "recommendedNextAction": (
            "Chief Scientist review, then proceed to S10 graft experiments using S09 no-governance controls, "
            "local-governance effect summaries, and the flagged broad-organizer upper-bound baseline; do not start S10 until review is complete."
        ),
        "laySummary": (
            "S09 compared no-governance controls with local voting, leader-cell, pacemaker, quorum, conflict-resolution, "
            "limited-organizer, and broad-organizer governance variants across S08-priority contexts and three goal modes."
        ),
        "outcomeClassification": outcome,
        "selectedContextCount": int(len(selected_contexts)),
        "conditionCount": int(len(condition_df)),
        "resultCount": int(len(classified)),
        "goalModeCount": int(len(GOVERNANCE_GOAL_MODES)),
        "governanceVariantCount": int(parameter_df["governanceVariant"].nunique()),
        "s08DisabledBaselineExactReproductionRate": reproduction_rate,
        "localGovernanceRescueSuccessProxyRate": local_rescue_rate,
        "broadControlRunCount": int(classified["broadControlLike"].map(bool).sum()),
        "workerCount": int(workers),
        "maxActivations": int(max_activations),
        "maxSwaps": int(max_swaps),
        "maxComparisons": int(max_comparisons),
        "completedAt": completed_at,
        "wallTimeSeconds": float(time.perf_counter() - started),
        "newDependenciesInstalled": [],
        "sourceCodeLocation": str(repo_root),
        "sourceCodeArtifactPolicy": "repository-backed source was committed to git; source files were not copied into artifacts per workspace instructions",
    }
    summary_path = write_markdown_report(
        step_dir=step_dir,
        status=status,
        artifacts_written=[str(path) for path in artifacts],
        effect_summary=effect_summary,
        validation_df=validation_df,
    )
    if summary_path not in artifacts:
        artifacts.append(summary_path)
    status["artifactsWritten"] = [str(path) for path in artifacts]
    write_json(status_path, status)

    manifest_payload = {
        "schema": "eidosoma.step_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "governanceVersion": GOVERNANCE_VERSION,
        "inputArtifacts": {
            "s08Results": str(s08_results_path),
            "s08EffectSummary": str(s08_effect_summary_path),
            "s08PriorityContext": str(s08_priority_path),
            "s06PairSummary": str(s06_pair_summary_path),
            "winnerCriteria": str(winner_criteria_path),
            "panel": str(panel_path),
        },
        "s08EffectSummaryRows": int(len(s08_effect_summary)),
        "artifacts": artifact_records(artifacts),
    }
    write_json(manifest_path, manifest_payload)

    existing_manifest: dict[str, Any] = {}
    if run_manifest_path.exists():
        try:
            existing_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_manifest = {}
    research_steps = dict(existing_manifest.get("researchSteps", {}))
    research_steps[STEP_ID] = status
    run_manifest = {
        **existing_manifest,
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": "E06",
        "lastResearchStepId": STEP_ID,
        "lastStepNumber": STEP_NUMBER,
        "updatedAt": completed_at,
        "git": {
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"], repo_root),
            "commit": git_value(["rev-parse", "HEAD"], repo_root),
            "dirtyStatus": git_value(["status", "--short"], repo_root),
            "remote": git_value(["remote", "-v"], repo_root),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workerCount": int(workers),
            "threadEnvironment": {
                key: os.environ.get(key)
                for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
                if os.environ.get(key) is not None
            },
            "newDependenciesInstalled": [],
        },
        "researchSteps": research_steps,
        "artifacts": artifact_records(artifacts),
    }
    write_json(run_manifest_path, run_manifest)
    manifest_payload["artifacts"] = artifact_records(artifacts)
    write_json(manifest_path, manifest_payload)
    return {
        "status": status,
        "selectedContexts": selected_contexts,
        "conditions": condition_df,
        "parameters": parameter_df,
        "informationAudit": audit_df,
        "results": classified,
        "effectSummary": effect_summary,
        "contextSummary": context_summary,
        "s08BaselineReproduction": reproduction,
        "validation": validation_df,
        "artifactPaths": [str(path) for path in artifacts],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E06 S09 governance-mechanism interventions")
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--s08-results-path", type=Path, default=DEFAULT_S08_RESULTS_PATH)
    parser.add_argument("--s08-effect-summary-path", type=Path, default=DEFAULT_S08_EFFECT_SUMMARY_PATH)
    parser.add_argument("--s06-pair-summary-path", type=Path, default=DEFAULT_S06_PAIR_SUMMARY_PATH)
    parser.add_argument("--winner-criteria-path", type=Path, default=DEFAULT_S06_WINNER_CRITERIA_PATH)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-contexts", type=int, default=18)
    parser.add_argument("--max-activations", type=int, default=6000)
    parser.add_argument("--max-swaps", type=int, default=4000)
    parser.add_argument("--max-comparisons", type=int, default=30000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_s09_governance_mechanisms(
        artifacts_dir=args.artifacts_dir,
        panel_path=args.panel_path,
        s08_results_path=args.s08_results_path,
        s08_effect_summary_path=args.s08_effect_summary_path,
        s06_pair_summary_path=args.s06_pair_summary_path,
        winner_criteria_path=args.winner_criteria_path,
        workers=args.workers,
        max_contexts=args.max_contexts,
        max_activations=args.max_activations,
        max_swaps=args.max_swaps,
        max_comparisons=args.max_comparisons,
    )
    status = result["status"]
    print(
        f"{STEP_ID} {status['status']}: {status['conditionCount']} governance conditions, "
        f"S08 reproduction {status['s08DisabledBaselineExactReproductionRate']:.3f}, "
        f"validation {status['validationResult']}"
    )
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
