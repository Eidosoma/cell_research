"""Local reinforcement-like learning rules for E04 S06."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from e02_deterministic_simulator.simulator import StepOutcome
from morphospace import (
    BubblePolicy,
    LocalObservation,
    LocalRulePolicy,
    PolicyCell,
    PolicySpec,
    ProposedAction,
    policy_from_spec,
)
from morphospace.policies import CellSnapshot

from .fatigue import (
    FATIGUE_DAMAGE_VERSION,
    FatigueDamageConfig,
    FatigueDamageEventSimulator,
    initial_fatigue_state,
)
from .homeostasis import apply_perturbation
from .memory import MEMORY_REPAIR_VERSION
from .repair import REPAIR_REPAIR_VERSION, RepairRuleConfig, initial_repair_state
from .signals import SIGNAL_REPAIR_VERSION, SignalConfig


LOCAL_LEARNING_VERSION = "e04_s06_local_learning.v1"
LEARNING_STATE_KEY = "__e04_learning__"
LEARNING_PENDING_ACTION_KEY = "__e04_learning_pending_action__"
LEARNING_VARIANTS = ("fixed", "local_adaptive", "random_adaptation")
LOCAL_LEARNING_ACTIONS = ("swap_left", "swap_right", "wait")
LOCAL_REWARD_ALLOWED_INPUTS = (
    "actor_value",
    "left_value",
    "left_frozen",
    "right_value",
    "right_frozen",
    "selected_local_action",
    "outcome_swapped",
    "outcome_blocked_move_attempt",
    "outcome_reason",
    "prior_recent_local_failures",
)
LOCAL_REWARD_FORBIDDEN_INPUTS = (
    "sortedness",
    "global_sortedness",
    "whole_array_values",
    "target_array",
    "target_position",
    "ideal_position",
    "left_context",
    "future_state",
)


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
        return round(float(value), 12) if math.isfinite(float(value)) else None
    return value


def _compact_json(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256(_compact_json(list(parts)).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


@dataclass(frozen=True)
class LocalLearningConfig:
    """Bounded local action-probability adaptation configuration."""

    variant: str = "fixed"
    learning_rate: float = 0.35
    min_action_probability: float = 0.04
    max_action_probability: float = 0.92
    reward_clip: float = 1.0
    random_seed: int = 0
    successful_swap_reward: float = 0.45
    local_inversion_reward: float = 0.35
    blocked_neighbor_reward: float = 0.18
    blocked_swap_penalty: float = -0.30
    non_improving_swap_penalty: float = -0.25
    recent_failure_bonus: float = 0.10
    locally_ordered_wait_reward: float = 0.04
    disordered_wait_penalty: float = -0.06
    initial_action_probabilities: Mapping[str, float] = field(
        default_factory=lambda: {"swap_left": 0.45, "swap_right": 0.45, "wait": 0.10}
    )

    def __post_init__(self) -> None:
        if self.variant not in LEARNING_VARIANTS:
            raise ValueError(f"learning variant must be one of {LEARNING_VARIANTS}, got {self.variant!r}")
        if float(self.learning_rate) < 0.0:
            raise ValueError("learning_rate must be nonnegative")
        if not 0.0 <= float(self.min_action_probability) <= 1.0:
            raise ValueError("min_action_probability must be between 0 and 1")
        if not 0.0 <= float(self.max_action_probability) <= 1.0:
            raise ValueError("max_action_probability must be between 0 and 1")
        if float(self.min_action_probability) > float(self.max_action_probability):
            raise ValueError("min_action_probability cannot exceed max_action_probability")
        action_count = len(LOCAL_LEARNING_ACTIONS)
        if float(self.min_action_probability) * action_count > 1.0 + 1e-12:
            raise ValueError("min_action_probability is infeasible for the action set")
        if float(self.max_action_probability) * action_count < 1.0 - 1e-12:
            raise ValueError("max_action_probability is infeasible for the action set")
        if float(self.reward_clip) <= 0.0:
            raise ValueError("reward_clip must be positive")
        normalized = normalize_action_probabilities(
            self.initial_action_probabilities,
            min_probability=float(self.min_action_probability),
            max_probability=float(self.max_action_probability),
        )
        object.__setattr__(self, "initial_action_probabilities", normalized)

    @classmethod
    def from_spec(cls, payload: Mapping[str, Any] | str | "LocalLearningConfig") -> "LocalLearningConfig":
        if isinstance(payload, LocalLearningConfig):
            return payload
        if isinstance(payload, str):
            return cls(variant=payload)
        probabilities = payload.get("initialActionProbabilities", payload.get("initial_action_probabilities", None))
        return cls(
            variant=str(payload.get("variant", "fixed")),
            learning_rate=float(payload.get("learningRate", payload.get("learning_rate", 0.35))),
            min_action_probability=float(
                payload.get("minActionProbability", payload.get("min_action_probability", 0.04))
            ),
            max_action_probability=float(
                payload.get("maxActionProbability", payload.get("max_action_probability", 0.92))
            ),
            reward_clip=float(payload.get("rewardClip", payload.get("reward_clip", 1.0))),
            random_seed=int(payload.get("randomSeed", payload.get("random_seed", 0))),
            successful_swap_reward=float(
                payload.get("successfulSwapReward", payload.get("successful_swap_reward", 0.45))
            ),
            local_inversion_reward=float(
                payload.get("localInversionReward", payload.get("local_inversion_reward", 0.35))
            ),
            blocked_neighbor_reward=float(
                payload.get("blockedNeighborReward", payload.get("blocked_neighbor_reward", 0.18))
            ),
            blocked_swap_penalty=float(payload.get("blockedSwapPenalty", payload.get("blocked_swap_penalty", -0.30))),
            non_improving_swap_penalty=float(
                payload.get("nonImprovingSwapPenalty", payload.get("non_improving_swap_penalty", -0.25))
            ),
            recent_failure_bonus=float(payload.get("recentFailureBonus", payload.get("recent_failure_bonus", 0.10))),
            locally_ordered_wait_reward=float(
                payload.get("locallyOrderedWaitReward", payload.get("locally_ordered_wait_reward", 0.04))
            ),
            disordered_wait_penalty=float(
                payload.get("disorderedWaitPenalty", payload.get("disordered_wait_penalty", -0.06))
            ),
            initial_action_probabilities=probabilities
            or {"swap_left": 0.45, "swap_right": 0.45, "wait": 0.10},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "learningRate": float(self.learning_rate),
            "minActionProbability": float(self.min_action_probability),
            "maxActionProbability": float(self.max_action_probability),
            "rewardClip": float(self.reward_clip),
            "randomSeed": int(self.random_seed),
            "successfulSwapReward": float(self.successful_swap_reward),
            "localInversionReward": float(self.local_inversion_reward),
            "blockedNeighborReward": float(self.blocked_neighbor_reward),
            "blockedSwapPenalty": float(self.blocked_swap_penalty),
            "nonImprovingSwapPenalty": float(self.non_improving_swap_penalty),
            "recentFailureBonus": float(self.recent_failure_bonus),
            "locallyOrderedWaitReward": float(self.locally_ordered_wait_reward),
            "disorderedWaitPenalty": float(self.disordered_wait_penalty),
            "initialActionProbabilities": dict(self.initial_action_probabilities),
            "localLearningVersion": LOCAL_LEARNING_VERSION,
            "usesGlobalSortednessSignal": False,
            "usesWholeArrayTargetSignal": False,
        }


def normalize_action_probabilities(
    weights: Mapping[str, float],
    *,
    min_probability: float = 0.04,
    max_probability: float = 0.92,
) -> dict[str, float]:
    """Project positive action weights to a bounded normalized simplex."""

    actions = tuple(LOCAL_LEARNING_ACTIONS)
    min_probability = float(min_probability)
    max_probability = float(max_probability)
    if min_probability * len(actions) > 1.0 + 1e-12 or max_probability * len(actions) < 1.0 - 1e-12:
        raise ValueError("probability bounds are infeasible for the action set")
    raw = {action: max(1e-12, float(weights.get(action, 0.0))) for action in actions}
    free = set(actions)
    fixed: dict[str, float] = {}
    while free:
        remaining_mass = 1.0 - sum(fixed.values())
        total = sum(raw[action] for action in free)
        if total <= 0.0:
            trial = {action: remaining_mass / len(free) for action in free}
        else:
            trial = {action: remaining_mass * raw[action] / total for action in free}
        low = {action: min_probability for action, probability in trial.items() if probability < min_probability}
        high = {action: max_probability for action, probability in trial.items() if probability > max_probability}
        newly_fixed = {**low, **high}
        if not newly_fixed:
            fixed.update(trial)
            break
        fixed.update(newly_fixed)
        free.difference_update(newly_fixed)
    if len(fixed) != len(actions):
        missing = [action for action in actions if action not in fixed]
        remaining = max(0.0, 1.0 - sum(fixed.values()))
        for action in missing:
            fixed[action] = remaining / max(1, len(missing))
    total = sum(fixed.values())
    normalized = {action: fixed[action] / total for action in actions}
    drift = 1.0 - sum(normalized.values())
    normalized[actions[-1]] += drift
    return {action: float(_clip(normalized[action], min_probability, max_probability)) for action in actions}


def initial_learning_state(config: LocalLearningConfig | Mapping[str, Any] | str) -> dict[str, Any]:
    config = LocalLearningConfig.from_spec(config)
    probabilities = dict(config.initial_action_probabilities)
    return {
        "learning_schema_version": LOCAL_LEARNING_VERSION,
        "variant": config.variant,
        "actionWeights": dict(probabilities),
        "actionProbabilities": dict(probabilities),
        "updateCount": 0,
        "recentLocalFailures": 0,
        "lastSelectedAction": None,
        "lastReward": 0.0,
        "lastRewardRecord": {},
        "usesGlobalSortednessSignal": False,
        "usesWholeArrayTargetSignal": False,
    }


def serialize_learning_state(learning_state: Mapping[str, Any]) -> dict[str, Any]:
    return _json_ready(dict(learning_state))


def learning_state_for_trace(policy_state: Mapping[str, Any]) -> dict[str, Any]:
    value = policy_state.get(LEARNING_STATE_KEY, {})
    if isinstance(value, Mapping):
        return serialize_learning_state(value)
    return {}


def reset_learning_state(
    policy_state: dict[str, Any],
    config: LocalLearningConfig | Mapping[str, Any] | str,
) -> dict[str, Any]:
    policy_state[LEARNING_STATE_KEY] = initial_learning_state(config)
    policy_state.pop(LEARNING_PENDING_ACTION_KEY, None)
    return policy_state


def learning_information_boundary() -> dict[str, Any]:
    return {
        "schema": "eidosoma.e04_s06_learning_information_boundary.v1",
        "localLearningVersion": LOCAL_LEARNING_VERSION,
        "allowedRewardInputs": list(LOCAL_REWARD_ALLOWED_INPUTS),
        "forbiddenRewardInputs": list(LOCAL_REWARD_FORBIDDEN_INPUTS),
        "usesGlobalSortednessSignal": False,
        "usesWholeArrayTargetSignal": False,
        "notes": (
            "S06 reward records are computed from adjacent-neighborhood values, local frozen flags, "
            "the selected local action, outcome flags, and the actor cell's own recent local failures."
        ),
    }


def _local_snapshot_payload(snapshot: CellSnapshot | None, side: str) -> dict[str, Any]:
    if snapshot is None:
        return {f"{side}_value": None, f"{side}_frozen": None}
    return {f"{side}_value": int(snapshot.value), f"{side}_frozen": bool(snapshot.frozen)}


def _action_target(action_key: str, actor_position: int) -> int | None:
    if action_key == "swap_left":
        return int(actor_position) - 1
    if action_key == "swap_right":
        return int(actor_position) + 1
    return None


def _local_action_has_neighbor(observation: LocalObservation, action_key: str) -> bool:
    if action_key == "swap_left":
        return observation.left is not None
    if action_key == "swap_right":
        return observation.right is not None
    return True


def _local_inversion_would_improve(observation: LocalObservation, action_key: str) -> bool:
    actor = observation.actor
    if action_key == "swap_left":
        return observation.left is not None and actor.value < observation.left.value
    if action_key == "swap_right":
        return observation.right is not None and actor.value > observation.right.value
    return False


def _selected_action_from_outcome(
    observation: LocalObservation,
    action: ProposedAction,
    state: Mapping[str, Any],
) -> str:
    pending = state.get(LEARNING_PENDING_ACTION_KEY)
    if pending in LOCAL_LEARNING_ACTIONS:
        return str(pending)
    if action.action == "swap" and action.target_position is not None:
        if int(action.target_position) < int(observation.actor_position):
            return "swap_left"
        if int(action.target_position) > int(observation.actor_position):
            return "swap_right"
    return "wait"


def compute_local_learning_reward(
    observation: LocalObservation,
    state: Mapping[str, Any],
    action: ProposedAction,
    outcome: StepOutcome,
    config: LocalLearningConfig | Mapping[str, Any] | str,
) -> dict[str, Any]:
    """Return a local-only reward record for one actor update."""

    config = LocalLearningConfig.from_spec(config)
    learning_state = learning_state_for_trace(state)
    selected = _selected_action_from_outcome(observation, action, state)
    prior_failures = int(learning_state.get("recentLocalFailures", 0))
    target_snapshot = observation.left if selected == "swap_left" else observation.right if selected == "swap_right" else None
    chosen_inversion = _local_inversion_would_improve(observation, selected)
    target_frozen = bool(target_snapshot.frozen) if target_snapshot is not None else False
    reward = 0.0
    components: dict[str, float] = {}

    if outcome.swapped:
        components["successful_swap"] = float(config.successful_swap_reward)
        if chosen_inversion:
            components["local_inversion_reduced"] = float(config.local_inversion_reward)
        else:
            components["non_improving_swap_penalty"] = float(config.non_improving_swap_penalty)
        if prior_failures > 0:
            components["lower_frustration_proxy"] = float(config.recent_failure_bonus)
    elif outcome.blocked_move_attempt and selected in {"swap_left", "swap_right"}:
        if target_frozen:
            components["blocked_neighbor_recovery_proxy"] = float(config.blocked_neighbor_reward)
        else:
            components["blocked_swap_penalty"] = float(config.blocked_swap_penalty)
    elif selected in {"swap_left", "swap_right"} and not chosen_inversion:
        components["no_local_improvement_penalty"] = -0.12
    elif selected == "wait" and (observation.left is None and observation.right is None):
        components["isolated_wait"] = 0.0
    elif selected == "wait":
        locally_ordered = not _local_inversion_would_improve(observation, "swap_left") and not _local_inversion_would_improve(
            observation, "swap_right"
        )
        components["locally_ordered_wait"] = (
            float(config.locally_ordered_wait_reward) if locally_ordered else float(config.disordered_wait_penalty)
        )

    reward = float(sum(components.values()))
    reward = _clip(reward, -float(config.reward_clip), float(config.reward_clip))
    local_inputs = {
        "actor_value": int(observation.actor.value),
        **_local_snapshot_payload(observation.left, "left"),
        **_local_snapshot_payload(observation.right, "right"),
        "selected_local_action": selected,
        "outcome_swapped": bool(outcome.swapped),
        "outcome_blocked_move_attempt": bool(outcome.blocked_move_attempt),
        "outcome_reason": outcome.reason,
        "prior_recent_local_failures": prior_failures,
    }
    return {
        "reward": float(reward),
        "selectedAction": selected,
        "components": components,
        "localInputs": local_inputs,
        "inputFieldsUsed": list(LOCAL_REWARD_ALLOWED_INPUTS),
        "forbiddenFieldsUsed": [],
        "usesGlobalSortednessSignal": False,
        "usesWholeArrayTargetSignal": False,
        "localLearningVersion": LOCAL_LEARNING_VERSION,
    }


def _updated_learning_state(
    config: LocalLearningConfig,
    current: Mapping[str, Any],
    observation: LocalObservation,
    action: ProposedAction,
    outcome: StepOutcome,
    pending_state: Mapping[str, Any],
) -> dict[str, Any]:
    state = dict(initial_learning_state(config))
    state.update(dict(current))
    reward_record = compute_local_learning_reward(observation, pending_state, action, outcome, config)
    selected = str(reward_record["selectedAction"])
    reward = float(reward_record["reward"])
    update_count = int(state.get("updateCount", 0)) + 1
    weights = {action_key: float(state.get("actionWeights", {}).get(action_key, 1.0)) for action_key in LOCAL_LEARNING_ACTIONS}

    if config.variant == "local_adaptive":
        weights[selected] = max(1e-12, weights[selected] * math.exp(float(config.learning_rate) * reward))
    elif config.variant == "random_adaptation":
        rng = np.random.default_rng(_stable_seed(config.random_seed, observation.actor.cell_id, update_count))
        for action_key in LOCAL_LEARNING_ACTIONS:
            weights[action_key] = max(1e-12, weights[action_key] * math.exp(float(config.learning_rate) * float(rng.normal(0.0, 0.5))))

    probabilities = normalize_action_probabilities(
        weights,
        min_probability=config.min_action_probability,
        max_probability=config.max_action_probability,
    )
    if config.variant == "fixed":
        weights = dict(config.initial_action_probabilities)
        probabilities = dict(config.initial_action_probabilities)

    recent_failures = int(state.get("recentLocalFailures", 0))
    if outcome.swapped:
        recent_failures = 0
    elif selected in {"swap_left", "swap_right"} and reward <= 0:
        recent_failures = min(99, recent_failures + 1)

    updated = {
        "learning_schema_version": LOCAL_LEARNING_VERSION,
        "variant": config.variant,
        "actionWeights": probabilities if config.variant == "fixed" else weights,
        "actionProbabilities": probabilities,
        "updateCount": update_count,
        "recentLocalFailures": recent_failures,
        "lastSelectedAction": selected,
        "lastReward": reward,
        "lastRewardRecord": reward_record,
        "usesGlobalSortednessSignal": False,
        "usesWholeArrayTargetSignal": False,
    }
    return serialize_learning_state(updated)


class LocalLearningPolicyWrapper(LocalRulePolicy):
    """Local action-probability policy with outcome-based local updates."""

    family = "local_learning_repair"
    version = LOCAL_LEARNING_VERSION

    def __init__(
        self,
        base_policy: str | LocalRulePolicy | PolicySpec | Mapping[str, Any] = "bubble",
        learning_config: LocalLearningConfig | Mapping[str, Any] | str = "fixed",
    ) -> None:
        self.base_policy = base_policy if isinstance(base_policy, LocalRulePolicy) else policy_from_spec(base_policy)
        self.learning_config = LocalLearningConfig.from_spec(learning_config)
        self.algotype = f"learning_{self.learning_config.variant}"
        self.policy_id = f"e04_learning_{self.learning_config.variant}__{self.base_policy.policy_id}"
        super().__init__(
            base_policy=self.base_policy.to_spec().to_dict(),
            learning_config=self.learning_config.to_dict(),
        )

    def to_spec(self) -> PolicySpec:
        return PolicySpec(
            policy_id=self.policy_id,
            family=self.family,
            algotype=self.algotype,
            version=self.version,
            parameters={
                "basePolicy": self.base_policy.to_spec().to_dict(),
                "learningConfig": self.learning_config.to_dict(),
            },
        )

    def initial_state(
        self,
        *,
        cell_id: int,
        position: int,
        value: int,
        n: int,
        reverse_direction: bool = False,
    ) -> dict[str, Any]:
        return reset_learning_state({}, self.learning_config)

    def observe(
        self,
        cells: Sequence[PolicyCell],
        actor_position: int,
        state: Mapping[str, Any],
        frozen_variant: str,
    ) -> LocalObservation:
        local = BubblePolicy().observe(cells, actor_position, {}, frozen_variant)
        return LocalObservation(
            actor=local.actor,
            actor_position=local.actor_position,
            n=local.n,
            left=local.left,
            right=local.right,
            left_context=(),
            target_position=None,
            target=None,
            frozen_variant=frozen_variant,
        )

    def propose_action(
        self,
        observation: LocalObservation,
        state: Mapping[str, Any],
        rng: np.random.Generator,
        *,
        forced_direction: int | None = None,
    ) -> ProposedAction:
        if forced_direction is not None:
            selected = "swap_right" if forced_direction > 0 else "swap_left"
        else:
            learning_state = learning_state_for_trace(state) or initial_learning_state(self.learning_config)
            probabilities = learning_state.get("actionProbabilities", self.learning_config.initial_action_probabilities)
            p = [float(probabilities.get(action, 0.0)) for action in LOCAL_LEARNING_ACTIONS]
            p = np.asarray(p, dtype=float)
            p = p / p.sum()
            selected = str(rng.choice(LOCAL_LEARNING_ACTIONS, p=p))
        updates = {LEARNING_PENDING_ACTION_KEY: selected}
        if selected == "wait":
            return ProposedAction.wait("learning_wait", state_updates=updates)
        target_position = _action_target(selected, observation.actor_position)
        return ProposedAction.swap(
            int(target_position),
            comparison_delta=1,
            state_updates=updates,
            swapped_reason="learning_local_swap",
            blocked_reason="learning_swap_blocked",
        )

    def constrain_action(
        self,
        observation: LocalObservation,
        state: Mapping[str, Any],
        action: ProposedAction,
    ) -> ProposedAction:
        selected = action.state_updates.get(LEARNING_PENDING_ACTION_KEY, "wait")
        if selected == "wait":
            return action
        if selected not in {"swap_left", "swap_right"} or not _local_action_has_neighbor(observation, selected):
            return ProposedAction.wait(
                "learning_target_oob",
                target_position=action.target_position,
                comparison_delta=action.comparison_delta,
                state_updates=action.state_updates,
            )
        if not _local_inversion_would_improve(observation, selected):
            return ProposedAction.wait(
                "learning_no_local_improvement",
                target_position=action.target_position,
                comparison_delta=action.comparison_delta,
                state_updates=action.state_updates,
            )
        return action

    def update_state(
        self,
        state: dict[str, Any],
        observation: LocalObservation,
        action: ProposedAction,
        outcome: StepOutcome,
    ) -> None:
        pending_state = dict(state)
        state[LEARNING_STATE_KEY] = _updated_learning_state(
            self.learning_config,
            learning_state_for_trace(state),
            observation,
            action,
            outcome,
            pending_state,
        )
        state.pop(LEARNING_PENDING_ACTION_KEY, None)

    def legal_action_exists(self, observation: LocalObservation, state: Mapping[str, Any]) -> bool:
        return _local_inversion_would_improve(observation, "swap_left") or _local_inversion_would_improve(
            observation, "swap_right"
        )


def learning_policy_from_spec(spec: PolicySpec | Mapping[str, Any] | str | LocalRulePolicy) -> LocalRulePolicy:
    if isinstance(spec, LocalLearningPolicyWrapper):
        return spec
    if isinstance(spec, str):
        return LocalLearningPolicyWrapper("bubble", spec if spec in LEARNING_VARIANTS else "fixed")
    if isinstance(spec, PolicySpec):
        payload = spec.to_dict()
    elif isinstance(spec, LocalRulePolicy):
        return spec
    else:
        payload = dict(spec)
    if payload.get("family") != "local_learning_repair":
        return policy_from_spec(payload)
    parameters = dict(payload.get("parameters", {}))
    base_policy = parameters.get("basePolicy") or parameters.get("base_policy") or "bubble"
    learning_config = parameters.get("learningConfig") or parameters.get("learning_config") or "fixed"
    return LocalLearningPolicyWrapper(base_policy, learning_config)


def learning_policy_to_json(policy: LocalRulePolicy) -> str:
    return _compact_json(policy.to_spec().to_dict())


def learning_policy_from_json(payload: str) -> LocalRulePolicy:
    return learning_policy_from_spec(json.loads(payload))


def build_learning_variants(
    base_policy: str | LocalRulePolicy | PolicySpec | Mapping[str, Any] = "bubble",
    variants: Iterable[str] = LEARNING_VARIANTS,
    *,
    learning_rate: float = 0.35,
    random_seed: int = 0,
) -> tuple[LocalLearningPolicyWrapper, ...]:
    return tuple(
        LocalLearningPolicyWrapper(
            base_policy,
            LocalLearningConfig(variant=variant, learning_rate=learning_rate, random_seed=random_seed),
        )
        for variant in variants
    )


def _unwrap_learning_policy(policy: LocalRulePolicy) -> LocalLearningPolicyWrapper | None:
    if isinstance(policy, LocalLearningPolicyWrapper):
        return policy
    base = getattr(policy, "base_policy", None)
    if isinstance(base, LocalRulePolicy):
        return _unwrap_learning_policy(base)
    return None


class LearningEventSimulator(FatigueDamageEventSimulator):
    """Fatigue/repair simulator decorated with local-learning trace fields."""

    def __init__(
        self,
        initial_values: Sequence[int],
        policies: str | LocalRulePolicy | PolicySpec | Sequence[str | LocalRulePolicy | PolicySpec | Mapping[str, Any]],
        *,
        trace_learning_activations: bool = True,
        fatigue_config: FatigueDamageConfig | Mapping[str, Any] | str = "none",
        repair_config: RepairRuleConfig | Mapping[str, Any] | str = "permanent",
        signal_config: SignalConfig | Mapping[str, Any] | str = "no_signal",
        implementation: str = "local_learning_interface",
        research_step_id: str = "S06",
        **kwargs: Any,
    ) -> None:
        self.trace_learning_activations = bool(trace_learning_activations)
        super().__init__(
            initial_values,
            policies,
            fatigue_config=fatigue_config,
            repair_config=repair_config,
            signal_config=signal_config,
            implementation=implementation,
            research_step_id=research_step_id,
            **kwargs,
        )
        if self.trace_rows:
            self._decorate_learning_trace_row(self.trace_rows[-1])

    def _learning_state_payload(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for position, cell in enumerate(self.cells):
            learning_policy = _unwrap_learning_policy(cell.policy)
            state = learning_state_for_trace(cell.state)
            rows.append(
                {
                    "position": int(position),
                    "cell_id": int(cell.cell_id),
                    "policy_id": cell.policy.policy_id,
                    "learning_policy_id": None if learning_policy is None else learning_policy.policy_id,
                    "learning_variant": state.get("variant", "unwrapped") if state else "unwrapped",
                    "learning_state": state,
                }
            )
        return rows

    def _actor_learning_trace(self, actor_cell_id: int | None) -> dict[str, Any]:
        if actor_cell_id is None:
            return {}
        position = self.positions_by_id.get(int(actor_cell_id))
        if position is None:
            return {}
        cell = self.cells[position]
        return {
            "position": int(position),
            "cell_id": int(cell.cell_id),
            "policy_id": cell.policy.policy_id,
            "learning_state": learning_state_for_trace(cell.state),
        }

    def _decorate_learning_trace_row(self, row: dict[str, Any]) -> None:
        payload = self._learning_state_payload()
        row["learning_schema_version"] = LOCAL_LEARNING_VERSION
        row["learning_states_json"] = _compact_json(payload)
        row["actor_learning_state_json"] = _compact_json(self._actor_learning_trace(row.get("actor_cell_id")))
        row["learning_variant_counts_json"] = _compact_json(Counter(item["learning_variant"] for item in payload))
        row["learning_information_boundary_json"] = _compact_json(learning_information_boundary())

    def _append_trace_row(
        self,
        *,
        event_kind: str,
        activation_index: int,
        actor_cell_id: int | None,
        actor_algotype: str | None,
        target_position: int | None,
    ) -> None:
        super()._append_trace_row(
            event_kind=event_kind,
            activation_index=activation_index,
            actor_cell_id=actor_cell_id,
            actor_algotype=actor_algotype,
            target_position=target_position,
        )
        if hasattr(self, "trace_learning_activations"):
            self._decorate_learning_trace_row(self.trace_rows[-1])

    def step(
        self,
        *,
        forced_cell_id: int | None = None,
        forced_direction: int | None = None,
    ) -> StepOutcome:
        trace_count_before = len(self.trace_rows)
        outcome = super().step(forced_cell_id=forced_cell_id, forced_direction=forced_direction)
        if self.trace_learning_activations and outcome.activated and len(self.trace_rows) == trace_count_before:
            actor_algotype = None
            if outcome.actor_cell_id is not None and outcome.actor_cell_id in self.positions_by_id:
                actor_algotype = self.cells[self.positions_by_id[outcome.actor_cell_id]].algotype
            self._append_trace_row(
                event_kind="learning_update",
                activation_index=self.activation_count,
                actor_cell_id=outcome.actor_cell_id,
                actor_algotype=actor_algotype,
                target_position=outcome.target_position,
            )
        elif self.trace_rows:
            self._decorate_learning_trace_row(self.trace_rows[-1])
        return outcome


def _refresh_positions_and_eligible(simulator: LearningEventSimulator) -> None:
    simulator.positions_by_id = {cell.cell_id: pos for pos, cell in enumerate(simulator.cells)}
    simulator._eligible_cell_ids = sorted(cell.cell_id for cell in simulator.cells if not cell.frozen)
    for position, cell in enumerate(simulator.cells):
        if cell.cell_id in getattr(simulator, "fatigue_states", {}):
            simulator.fatigue_states[cell.cell_id]["current_position"] = int(position)
        if cell.cell_id in getattr(simulator, "repair_states", {}):
            simulator.repair_states[cell.cell_id]["current_position"] = int(position)


def _resize_signal_fields_for_insert(simulator: LearningEventSimulator, position: int) -> None:
    for channel, values in simulator.signal_fields.items():
        simulator.signal_fields[channel] = np.insert(values, int(position), 0.0)


def _resize_signal_fields_for_delete(simulator: LearningEventSimulator, position: int) -> None:
    for channel, values in simulator.signal_fields.items():
        if 0 <= int(position) < len(values):
            simulator.signal_fields[channel] = np.delete(values, int(position))


def apply_homeostatic_event_to_simulator(
    simulator: LearningEventSimulator,
    event: Mapping[str, Any],
) -> None:
    """Apply an S05 perturbation directly to a learning simulator."""

    event_type = str(event["type"])
    if event_type == "swap":
        left = int(event["leftIndex"])
        right = int(event.get("rightIndex", left + 1))
        if not (0 <= left < len(simulator.cells) and 0 <= right < len(simulator.cells)):
            raise ValueError(f"swap indices out of bounds: {left}, {right}")
        simulator.cells[left], simulator.cells[right] = simulator.cells[right], simulator.cells[left]
        _refresh_positions_and_eligible(simulator)
    elif event_type == "insert":
        position = int(event["position"])
        if not 0 <= position <= len(simulator.cells):
            raise ValueError(f"insert position out of bounds: {position}")
        prototype = simulator.cells[0].policy if simulator.cells else LocalLearningPolicyWrapper()
        cell_id = max((cell.cell_id for cell in simulator.cells), default=-1) + 1
        value = int(event["value"])
        new_cell = PolicyCell(
            cell_id=cell_id,
            value=value,
            policy=prototype,
            label=0,
            reverse_direction=False,
            frozen=False,
            state=prototype.initial_state(cell_id=cell_id, position=position, value=value, n=len(simulator.cells) + 1),
        )
        simulator.cells.insert(position, new_cell)
        simulator.fatigue_states[cell_id] = initial_fatigue_state(cell_id, position)
        _resize_signal_fields_for_insert(simulator, position)
        _refresh_positions_and_eligible(simulator)
    elif event_type == "delete":
        position = int(event["position"])
        if not 0 <= position < len(simulator.cells):
            raise ValueError(f"delete position out of bounds: {position}")
        removed = simulator.cells.pop(position)
        simulator.fatigue_states.pop(removed.cell_id, None)
        simulator.repair_states.pop(removed.cell_id, None)
        _resize_signal_fields_for_delete(simulator, position)
        _refresh_positions_and_eligible(simulator)
    elif event_type == "freeze":
        cell_id = int(event["cellId"])
        if cell_id in simulator.positions_by_id:
            position = simulator.positions_by_id[cell_id]
            simulator.cells[position].frozen = True
            simulator.repair_states.setdefault(cell_id, initial_repair_state(cell_id, position))
            _refresh_positions_and_eligible(simulator)
    elif event_type == "recover":
        cell_id = int(event["cellId"])
        if cell_id in simulator.positions_by_id:
            simulator._unfreeze_cell(cell_id, "homeostatic_recover_event")
            _refresh_positions_and_eligible(simulator)
    elif event_type == "damage":
        cell_id = int(event["cellId"])
        if cell_id in simulator.positions_by_id:
            simulator._induce_damage(cell_id, "homeostatic_damage_event")
    else:
        values, _, _ = apply_perturbation(simulator.current_values(), event)
        if len(values) != len(simulator.cells):
            raise ValueError(f"unsupported perturbation for simulator: {event_type}")
        for cell, value in zip(simulator.cells, values):
            cell.value = int(value)
    simulator._append_trace_row(
        event_kind=f"homeostatic_{event_type}",
        activation_index=simulator.activation_count,
        actor_cell_id=None,
        actor_algotype=None,
        target_position=event.get("position", event.get("leftIndex", event.get("cellId"))),
    )


def learning_summary_record(simulator: LearningEventSimulator) -> dict[str, Any]:
    states = simulator._learning_state_payload()
    probabilities = [
        item.get("learning_state", {}).get("actionProbabilities", {})
        for item in states
        if item.get("learning_state", {}).get("actionProbabilities")
    ]
    min_probability = min((min(row.values()) for row in probabilities), default=None)
    max_probability = max((max(row.values()) for row in probabilities), default=None)
    return {
        "learningVariantCountsJson": _compact_json(Counter(item["learning_variant"] for item in states)),
        "learningStateRowsJson": _compact_json(states),
        "minActionProbabilityObserved": min_probability,
        "maxActionProbabilityObserved": max_probability,
        "localLearningVersion": LOCAL_LEARNING_VERSION,
        "memoryRepairVersion": MEMORY_REPAIR_VERSION,
        "signalRepairVersion": SIGNAL_REPAIR_VERSION,
        "repairRepairVersion": REPAIR_REPAIR_VERSION,
        "fatigueDamageVersion": FATIGUE_DAMAGE_VERSION,
    }
