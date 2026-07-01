"""No-global-oracle training contract for E04 S07."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from src.e03.policy_interface import PolicyObservation
from src.e04.memory_policies import CellMemoryState
from src.e04.signaling import SIGNAL_FIELDS, SignalSensation


LOCAL_ONLY_PROTOCOL_ID = "s07_local_only_adjacent_update_v1"
LOCAL_ONLY_BASELINE_TYPES = ("local_learning", "nonlearning_no_update_control")
CENTRALIZED_BASELINE_TYPE = "centralized_global_oracle_baseline"
ALLOWED_TRAINING_SIGNAL_FIELDS = ("blocked", "frustrated")
EXCLUDED_TRAINING_SIGNAL_FIELDS = tuple(name for name in SIGNAL_FIELDS if name not in ALLOWED_TRAINING_SIGNAL_FIELDS)
FORBIDDEN_TRAINING_TOKENS = (
    "sortedness",
    "monotonicity",
    "rank",
    "full_array",
    "whole_array",
    "all_values",
    "final",
    "target_position",
    "ideal_position",
    "target_sortedness",
    "target_morphology",
)
LEGACY_OBSERVATION_ORACLE_EXPOSURES = (
    "values",
    "labels",
    "statuses",
    "ideal_position",
)


@dataclass(frozen=True)
class LocalTrainingObservation:
    """Strict future-training view with no full array or target-position fields."""

    protocol_id: str
    behavior: str
    actor_value: int
    actor_status: str
    reverse_direction: bool
    left_present: bool
    left_value: int | None
    left_status: str | None
    right_present: bool
    right_value: int | None
    right_status: str | None
    at_left_boundary: bool
    at_right_boundary: bool
    last_move_success: bool | None
    last_action_type: str | None
    time_since_movement: int
    local_frustration: int
    failed_swap_count: int
    neighbor_identity_count: int
    signal_blocked: float
    signal_frustrated: float
    signal_scope: str

    def to_feature_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "behavior": self.behavior,
            "actor_value": int(self.actor_value),
            "actor_status": self.actor_status,
            "reverse_direction": bool(self.reverse_direction),
            "left_present": bool(self.left_present),
            "left_value": self.left_value,
            "left_status": self.left_status,
            "right_present": bool(self.right_present),
            "right_value": self.right_value,
            "right_status": self.right_status,
            "at_left_boundary": bool(self.at_left_boundary),
            "at_right_boundary": bool(self.at_right_boundary),
            "last_move_success": self.last_move_success,
            "last_action_type": self.last_action_type,
            "time_since_movement": int(self.time_since_movement),
            "local_frustration": int(self.local_frustration),
            "failed_swap_count": int(self.failed_swap_count),
            "neighbor_identity_count": int(self.neighbor_identity_count),
            "signal_blocked": float(self.signal_blocked),
            "signal_frustrated": float(self.signal_frustrated),
            "signal_scope": self.signal_scope,
        }


def project_local_training_observation(
    observation: PolicyObservation,
    memory: CellMemoryState | None = None,
    sensation: SignalSensation | None = None,
) -> LocalTrainingObservation:
    """Project a rich policy observation into the S07 local-only training view."""

    memory = memory or CellMemoryState()
    left_idx = int(observation.actor_index) - 1
    right_idx = int(observation.actor_index) + 1
    left_present = left_idx >= int(observation.left_boundary)
    right_present = right_idx <= int(observation.right_boundary)
    diffusive = dict(sensation.diffusive_fields) if sensation is not None else {}
    return LocalTrainingObservation(
        protocol_id=LOCAL_ONLY_PROTOCOL_ID,
        behavior=str(observation.behavior),
        actor_value=int(observation.actor_value),
        actor_status=str(observation.actor_status),
        reverse_direction=bool(observation.reverse_direction),
        left_present=bool(left_present),
        left_value=int(observation.values[left_idx]) if left_present else None,
        left_status=str(observation.statuses[left_idx]) if left_present else None,
        right_present=bool(right_present),
        right_value=int(observation.values[right_idx]) if right_present else None,
        right_status=str(observation.statuses[right_idx]) if right_present else None,
        at_left_boundary=not left_present,
        at_right_boundary=not right_present,
        last_move_success=memory.last_move_success,
        last_action_type=memory.last_action_type,
        time_since_movement=int(memory.time_since_movement),
        local_frustration=int(memory.local_frustration),
        failed_swap_count=len(memory.recent_failed_swaps),
        neighbor_identity_count=len(memory.recent_neighbor_identities),
        signal_blocked=float(diffusive.get("blocked", 0.0)),
        signal_frustrated=float(diffusive.get("frustrated", 0.0)),
        signal_scope=str(sensation.access_scope) if sensation is not None else "none",
    )


def audit_training_feature_dict(features: Mapping[str, Any]) -> dict[str, Any]:
    """Check that a projected training feature dictionary has no oracle tokens."""

    serialized = json.dumps(dict(features), sort_keys=True, separators=(",", ":"), default=str).lower()
    forbidden_hits = [token for token in FORBIDDEN_TRAINING_TOKENS if token in serialized]
    excluded_signal_hits = [
        field
        for field in EXCLUDED_TRAINING_SIGNAL_FIELDS
        if f"signal_{field}" in serialized or f'"{field}"' in serialized
    ]
    list_like_values = {
        key: len(value)
        for key, value in features.items()
        if isinstance(value, (list, tuple)) and key not in {"protocol_id"}
    }
    return {
        "usesGlobalOracle": bool(forbidden_hits or excluded_signal_hits or list_like_values),
        "forbiddenTokenHits": forbidden_hits,
        "excludedSignalHits": excluded_signal_hits,
        "listLikeFeatureValues": list_like_values,
        "featureCount": len(features),
        "allowedSignalFields": list(ALLOWED_TRAINING_SIGNAL_FIELDS),
        "excludedSignalFields": list(EXCLUDED_TRAINING_SIGNAL_FIELDS),
    }


def legacy_policy_observation_audit() -> dict[str, Any]:
    """Document why raw E03 observations are not the future S07 training view."""

    raw_fields = tuple(PolicyObservation.__dataclass_fields__.keys())
    exposed = [field for field in LEGACY_OBSERVATION_ORACLE_EXPOSURES if field in raw_fields]
    return {
        "rawPolicyObservationFields": list(raw_fields),
        "oracleExposureFields": exposed,
        "requiresProjectionForLocalOnlyTraining": bool(exposed),
        "strictProtocolId": LOCAL_ONLY_PROTOCOL_ID,
    }


def centralized_baseline_contract() -> dict[str, Any]:
    """Return the explicit label required for any future global-controller baseline."""

    return {
        "baselineType": CENTRALIZED_BASELINE_TYPE,
        "usesGlobalOracle": True,
        "eligibleForLocalClaims": False,
        "allowedPurpose": "ceiling_or_sanity_comparison_only",
        "mustNotBeMixedWithLocalOnlyRows": True,
    }
