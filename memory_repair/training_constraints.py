"""No-global-oracle training constraints for E04 S07."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from morphospace import LocalRulePolicy, PolicySpec

from .homeostasis import HOMEOSTASIS_BENCHMARK_VERSION, apply_perturbation, normalized_sortedness_record
from .learning import LOCAL_LEARNING_VERSION, LOCAL_REWARD_ALLOWED_INPUTS, LOCAL_REWARD_FORBIDDEN_INPUTS


TRAINING_CONSTRAINT_VERSION = "e04_s07_no_global_oracle.v1"
TRAINING_PROTOCOL_MODES = ("local_only", "global_oracle_baseline")
ORACLE_BASELINE_ID = "global_oracle_sort_upper_bound"
LOCAL_ONLY_PROTOCOL_ID = "e04_s07_local_only_training_protocol"
GLOBAL_ORACLE_PROTOCOL_ID = "e04_s07_global_oracle_baseline_protocol"

LOCAL_ONLY_ALLOWED_OBSERVATION_FIELDS = (
    "actor_value",
    "actor_frozen",
    "actor_algotype",
    "actor_local_memory",
    "actor_local_signal",
    "left_value",
    "left_frozen",
    "left_algotype",
    "right_value",
    "right_frozen",
    "right_algotype",
    "local_signal_center",
    "local_signal_left",
    "local_signal_right",
    "local_signal_sum",
    "local_signal_mean",
    "local_signal_max",
    "recent_local_outcome",
    "selected_local_action",
    "action_probabilities",
    "local_tick",
)
LOCAL_ONLY_ALLOWED_REWARD_FIELDS = tuple(LOCAL_REWARD_ALLOWED_INPUTS)
ORACLE_FORBIDDEN_FIELDS = tuple(
    dict.fromkeys(
        list(LOCAL_REWARD_FORBIDDEN_INPUTS)
        + [
            "sortedness_percent",
            "sortedness_raw_count",
            "global_sortedness",
            "whole_array_values",
            "whole_array_ranks",
            "whole_array_position_ranks",
            "target_array",
            "future_target_array",
            "future_state",
            "final_values",
            "target_position",
            "ideal_position",
            "left_context",
            "right_context",
            "all_cell_values",
            "global_rank",
        ]
    )
)
ORACLE_FLAG_FIELDS = (
    "usesGlobalSortednessSignal",
    "usesWholeArrayValues",
    "usesWholeArrayRanks",
    "usesWholeArrayTargetSignal",
    "usesFutureTargetArray",
    "usesTargetPositionOracle",
    "usesGlobalController",
    "oracleAllowed",
    "oracleBaseline",
    "oracleBaselineLabel",
)
DSL_TARGET_ORACLE_OPS = (
    "target_exists",
    "compare_target",
    "swap_target",
    "estimate_target_position",
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


@dataclass(frozen=True)
class TrainingProtocolConfig:
    """Information-boundary declaration for S07 training or baselines."""

    protocol_id: str = LOCAL_ONLY_PROTOCOL_ID
    mode: str = "local_only"
    oracle_allowed: bool = False
    oracle_baseline_label: str | None = None
    allowed_observation_fields: tuple[str, ...] = LOCAL_ONLY_ALLOWED_OBSERVATION_FIELDS
    allowed_reward_fields: tuple[str, ...] = LOCAL_ONLY_ALLOWED_REWARD_FIELDS
    forbidden_fields: tuple[str, ...] = ORACLE_FORBIDDEN_FIELDS
    allow_selection_target_estimate: bool = False

    def __post_init__(self) -> None:
        if self.mode not in TRAINING_PROTOCOL_MODES:
            raise ValueError(f"mode must be one of {TRAINING_PROTOCOL_MODES}, got {self.mode!r}")
        object.__setattr__(self, "allowed_observation_fields", tuple(str(value) for value in self.allowed_observation_fields))
        object.__setattr__(self, "allowed_reward_fields", tuple(str(value) for value in self.allowed_reward_fields))
        object.__setattr__(self, "forbidden_fields", tuple(str(value) for value in self.forbidden_fields))
        if self.mode == "local_only":
            if self.oracle_allowed:
                raise ValueError("local_only protocol cannot allow oracle access")
            if self.oracle_baseline_label:
                raise ValueError("local_only protocol cannot carry an oracle baseline label")
            if self.allow_selection_target_estimate:
                raise ValueError("local_only protocol cannot allow Selection-like target-position estimates")
        if self.mode == "global_oracle_baseline":
            if not self.oracle_allowed:
                raise ValueError("global_oracle_baseline protocol must set oracle_allowed=True")
            if not self.oracle_baseline_label:
                raise ValueError("global_oracle_baseline protocol requires oracle_baseline_label")

    @classmethod
    def from_spec(cls, payload: Mapping[str, Any] | str | "TrainingProtocolConfig") -> "TrainingProtocolConfig":
        if isinstance(payload, TrainingProtocolConfig):
            return payload
        if isinstance(payload, str):
            if payload == "global_oracle_baseline":
                return global_oracle_baseline_protocol()
            return local_only_training_protocol()
        return cls(
            protocol_id=str(payload.get("protocolId", payload.get("protocol_id", LOCAL_ONLY_PROTOCOL_ID))),
            mode=str(payload.get("mode", "local_only")),
            oracle_allowed=bool(payload.get("oracleAllowed", payload.get("oracle_allowed", False))),
            oracle_baseline_label=payload.get("oracleBaselineLabel", payload.get("oracle_baseline_label")),
            allowed_observation_fields=tuple(
                payload.get(
                    "allowedObservationFields",
                    payload.get("allowed_observation_fields", LOCAL_ONLY_ALLOWED_OBSERVATION_FIELDS),
                )
            ),
            allowed_reward_fields=tuple(
                payload.get("allowedRewardFields", payload.get("allowed_reward_fields", LOCAL_ONLY_ALLOWED_REWARD_FIELDS))
            ),
            forbidden_fields=tuple(payload.get("forbiddenFields", payload.get("forbidden_fields", ORACLE_FORBIDDEN_FIELDS))),
            allow_selection_target_estimate=bool(
                payload.get("allowSelectionTargetEstimate", payload.get("allow_selection_target_estimate", False))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocolId": self.protocol_id,
            "mode": self.mode,
            "oracleAllowed": bool(self.oracle_allowed),
            "oracleBaseline": self.mode == "global_oracle_baseline",
            "oracleBaselineLabel": self.oracle_baseline_label,
            "allowedObservationFields": list(self.allowed_observation_fields),
            "allowedRewardFields": list(self.allowed_reward_fields),
            "forbiddenFields": list(self.forbidden_fields),
            "allowSelectionTargetEstimate": bool(self.allow_selection_target_estimate),
            "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
            "localLearningVersion": LOCAL_LEARNING_VERSION,
        }


def local_only_training_protocol() -> TrainingProtocolConfig:
    return TrainingProtocolConfig()


def global_oracle_baseline_protocol() -> TrainingProtocolConfig:
    return TrainingProtocolConfig(
        protocol_id=GLOBAL_ORACLE_PROTOCOL_ID,
        mode="global_oracle_baseline",
        oracle_allowed=True,
        oracle_baseline_label=ORACLE_BASELINE_ID,
        allow_selection_target_estimate=True,
        allowed_observation_fields=LOCAL_ONLY_ALLOWED_OBSERVATION_FIELDS
        + (
            "whole_array_values",
            "whole_array_ranks",
            "target_array",
            "target_position",
            "global_sortedness",
        ),
        allowed_reward_fields=LOCAL_ONLY_ALLOWED_REWARD_FIELDS
        + (
            "whole_array_values",
            "target_array",
            "global_sortedness",
        ),
    )


def training_protocol_bundle() -> dict[str, Any]:
    return {
        "schema": "eidosoma.e04_s07_training_protocols.v1",
        "producerStep": "S07",
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
        "localOnlyProtocol": local_only_training_protocol().to_dict(),
        "globalOracleBaselineProtocol": global_oracle_baseline_protocol().to_dict(),
        "oracleFlagFields": list(ORACLE_FLAG_FIELDS),
        "dslTargetOracleOps": list(DSL_TARGET_ORACLE_OPS),
        "notes": (
            "Local-only training records must not use global Sortedness, whole-array ranks, target arrays, future states, "
            "or target-position estimates. Global-oracle baselines are allowed only when explicitly labeled."
        ),
    }


def _walk_json(value: Any, path: str = "$") -> list[tuple[str, Any, Any]]:
    rows: list[tuple[str, Any, Any]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            current = f"{path}.{key}"
            rows.append((current, key, item))
            rows.extend(_walk_json(item, current))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            current = f"{path}[{index}]"
            rows.append((current, index, item))
            rows.extend(_walk_json(item, current))
    return rows


def _flag_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _field_is_forbidden(field: str, forbidden_fields: Sequence[str]) -> bool:
    text = str(field)
    normalized = text.replace("-", "_").replace(".", "_")
    return any(term == normalized or term in normalized for term in forbidden_fields)


def audit_payload_for_oracle_leakage(
    payload: Mapping[str, Any],
    protocol: TrainingProtocolConfig | Mapping[str, Any] | str = "local_only",
    *,
    payload_kind: str = "payload",
) -> dict[str, Any]:
    """Audit one structured payload for forbidden oracle fields and flags."""

    protocol = TrainingProtocolConfig.from_spec(protocol)
    oracle_flag_hits: list[dict[str, Any]] = []
    forbidden_field_hits: list[dict[str, Any]] = []
    for path, key, value in _walk_json(payload):
        key_text = str(key)
        if key_text in ORACLE_FLAG_FIELDS and key_text not in {"oracleAllowed", "oracleBaseline", "oracleBaselineLabel"}:
            if _flag_truthy(value):
                oracle_flag_hits.append({"path": path, "field": key_text, "value": _json_ready(value)})
        if _field_is_forbidden(key_text, protocol.forbidden_fields):
            forbidden_field_hits.append({"path": path, "field": key_text})
        if isinstance(value, str) and _field_is_forbidden(value, protocol.forbidden_fields):
            forbidden_field_hits.append({"path": path, "field": value})
        if isinstance(value, str) and value in DSL_TARGET_ORACLE_OPS:
            forbidden_field_hits.append({"path": path, "field": value})

    has_oracle_marker = bool(payload.get("oracleAllowed") or payload.get("oracleBaseline") or payload.get("oracleBaselineLabel"))
    if protocol.mode == "global_oracle_baseline":
        success = bool(payload.get("oracleAllowed") is True or has_oracle_marker)
    else:
        success = not oracle_flag_hits and not forbidden_field_hits and not has_oracle_marker
    return {
        "payloadKind": payload_kind,
        "protocolId": protocol.protocol_id,
        "protocolMode": protocol.mode,
        "success": bool(success),
        "oracleFlagHits": oracle_flag_hits,
        "forbiddenFieldHits": forbidden_field_hits,
        "oracleMarkerPresent": bool(has_oracle_marker),
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
    }


def audit_reward_record(
    reward_record: Mapping[str, Any],
    protocol: TrainingProtocolConfig | Mapping[str, Any] | str = "local_only",
) -> dict[str, Any]:
    """Audit a learning reward record for local-only training compliance."""

    protocol = TrainingProtocolConfig.from_spec(protocol)
    input_fields = set(str(value) for value in reward_record.get("inputFieldsUsed", ()))
    local_inputs = reward_record.get("localInputs", {})
    if not isinstance(local_inputs, Mapping):
        local_inputs = {}
    disallowed_inputs = sorted(
        field
        for field in input_fields
        if field not in set(protocol.allowed_reward_fields) or _field_is_forbidden(field, protocol.forbidden_fields)
    )
    disallowed_local_keys = sorted(
        key for key in local_inputs if key not in set(protocol.allowed_reward_fields) or _field_is_forbidden(str(key), protocol.forbidden_fields)
    )
    flag_payload = audit_payload_for_oracle_leakage(reward_record, protocol, payload_kind="reward_record")
    success = flag_payload["success"] and not disallowed_inputs and not disallowed_local_keys
    if protocol.mode == "global_oracle_baseline":
        success = bool(reward_record.get("oracleAllowed") is True or reward_record.get("oracleBaseline") is True)
    return {
        "payloadKind": "reward_record",
        "protocolId": protocol.protocol_id,
        "protocolMode": protocol.mode,
        "success": bool(success),
        "disallowedInputs": disallowed_inputs,
        "disallowedLocalInputKeys": disallowed_local_keys,
        "oracleFlagHits": flag_payload["oracleFlagHits"],
        "forbiddenFieldHits": flag_payload["forbiddenFieldHits"],
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
    }


def audit_observation_log(
    observation_payload: Mapping[str, Any],
    protocol: TrainingProtocolConfig | Mapping[str, Any] | str = "local_only",
) -> dict[str, Any]:
    protocol = TrainingProtocolConfig.from_spec(protocol)
    fields = set(str(value) for value in observation_payload.get("fields", observation_payload.keys()))
    disallowed = sorted(
        field
        for field in fields
        if field not in set(protocol.allowed_observation_fields) or _field_is_forbidden(field, protocol.forbidden_fields)
    )
    flag_payload = audit_payload_for_oracle_leakage(observation_payload, protocol, payload_kind="observation_log")
    success = flag_payload["success"] and not disallowed
    if protocol.mode == "global_oracle_baseline":
        success = bool(observation_payload.get("oracleAllowed") is True or observation_payload.get("oracleBaseline") is True)
    return {
        "payloadKind": "observation_log",
        "protocolId": protocol.protocol_id,
        "protocolMode": protocol.mode,
        "success": bool(success),
        "disallowedObservationFields": disallowed,
        "oracleFlagHits": flag_payload["oracleFlagHits"],
        "forbiddenFieldHits": flag_payload["forbiddenFieldHits"],
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
    }


def _policy_payload(policy: PolicySpec | LocalRulePolicy | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(policy, PolicySpec):
        return policy.to_dict()
    if isinstance(policy, LocalRulePolicy):
        return policy.to_spec().to_dict()
    return dict(policy)


def audit_policy_spec_for_oracle_access(
    policy: PolicySpec | LocalRulePolicy | Mapping[str, Any],
    protocol: TrainingProtocolConfig | Mapping[str, Any] | str = "local_only",
) -> dict[str, Any]:
    """Audit a policy spec or DSL program for target/global-oracle dependencies."""

    protocol = TrainingProtocolConfig.from_spec(protocol)
    payload = _policy_payload(policy)
    violations: list[dict[str, Any]] = []
    family = str(payload.get("family", ""))
    algotype = str(payload.get("algotype", payload.get("policy_id", "")))
    if algotype in {"selection", "classic_selection"} or payload.get("policy_id") == "classic_selection":
        violations.append({"path": "$.algotype", "field": "selection_target_position_semantics"})
    for path, key, value in _walk_json(payload):
        key_text = str(key)
        if key_text in {"target_position", "ideal_position", "target_array", "whole_array_values"}:
            violations.append({"path": path, "field": key_text})
        if isinstance(value, str):
            if value in DSL_TARGET_ORACLE_OPS:
                violations.append({"path": path, "field": value})
            if _field_is_forbidden(value, protocol.forbidden_fields):
                violations.append({"path": path, "field": value})
        if key_text == "target_key" and str(value) == "target_position":
            violations.append({"path": path, "field": "target_key:target_position"})
    if family == "dsl":
        parameters = payload.get("parameters", {})
        program = parameters.get("program") if isinstance(parameters, Mapping) else None
        if isinstance(program, Mapping):
            for path, key, value in _walk_json(program):
                if isinstance(value, str) and value in DSL_TARGET_ORACLE_OPS:
                    violations.append({"path": f"$.parameters.program{path[1:]}", "field": value})
    if protocol.mode == "global_oracle_baseline":
        success = True
    else:
        success = not violations
    return {
        "payloadKind": "policy_spec",
        "policyId": str(payload.get("policy_id", payload.get("policyId", "unknown"))),
        "family": family,
        "algotype": algotype,
        "protocolId": protocol.protocol_id,
        "protocolMode": protocol.mode,
        "success": bool(success),
        "violations": violations,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
    }


def certify_training_record(
    record: Mapping[str, Any],
    protocol: TrainingProtocolConfig | Mapping[str, Any] | str = "local_only",
) -> dict[str, Any]:
    """Audit a compound training record with optional policy, observation, and reward sections."""

    protocol = TrainingProtocolConfig.from_spec(protocol)
    checks: list[dict[str, Any]] = []
    if "policySpec" in record and isinstance(record["policySpec"], Mapping):
        checks.append(audit_policy_spec_for_oracle_access(record["policySpec"], protocol))
    if "observationLog" in record and isinstance(record["observationLog"], Mapping):
        checks.append(audit_observation_log(record["observationLog"], protocol))
    if "rewardRecord" in record and isinstance(record["rewardRecord"], Mapping):
        checks.append(audit_reward_record(record["rewardRecord"], protocol))
    if not checks:
        checks.append(audit_payload_for_oracle_leakage(record, protocol, payload_kind="training_record"))
    success = all(check["success"] for check in checks)
    return {
        "payloadKind": "training_record",
        "protocolId": protocol.protocol_id,
        "protocolMode": protocol.mode,
        "success": bool(success),
        "checks": checks,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
    }


def global_oracle_baseline_record(
    values: Sequence[int],
    *,
    reason: str = "oracle_sort",
) -> dict[str, Any]:
    """Return an explicitly labeled global-oracle baseline action record."""

    current = [int(value) for value in values]
    target = sorted(current)
    return {
        "baselineId": ORACLE_BASELINE_ID,
        "oracleAllowed": True,
        "oracleBaseline": True,
        "oracleBaselineLabel": ORACLE_BASELINE_ID,
        "usesGlobalSortednessSignal": True,
        "usesWholeArrayValues": True,
        "usesWholeArrayRanks": True,
        "usesWholeArrayTargetSignal": True,
        "usesFutureTargetArray": False,
        "usesTargetPositionOracle": True,
        "usesGlobalController": True,
        "whole_array_values": current,
        "target_array": target,
        "global_sortedness": normalized_sortedness_record(current, initial_length=len(current))["sortednessPercent"],
        "reason": reason,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
    }


def apply_global_oracle_sort(values: Sequence[int]) -> tuple[list[int], bool, dict[str, Any]]:
    current = [int(value) for value in values]
    record = global_oracle_baseline_record(current)
    sorted_values = sorted(current)
    return sorted_values, sorted_values != current, record


def evaluate_global_oracle_homeostatic_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a labeled global-oracle sort baseline on one S05 homeostatic task."""

    values = [int(value) for value in task["initialValues"]]
    initial_length = len(values)
    frozen: set[int] = set()
    damaged: set[int] = set()
    threshold = float(task["sortednessThresholdPercent"])
    schedule = list(task.get("perturbationSchedule", ()))
    records: list[dict[str, Any]] = []
    oracle_records: list[dict[str, Any]] = []
    energy = 0
    for tick in range(int(task["horizonTicks"])):
        events = [event for event in schedule if int(event["tick"]) == tick]
        for event in events:
            values, frozen, damaged = apply_perturbation(values, event, frozen_cell_ids=frozen, damaged_cell_ids=damaged)
        pre = normalized_sortedness_record(values, initial_length=initial_length)
        values, changed, oracle_record = apply_global_oracle_sort(values)
        oracle_record["tick"] = int(tick)
        oracle_record["taskId"] = str(task["taskId"])
        oracle_records.append(oracle_record)
        energy += int(changed)
        post = normalized_sortedness_record(values, initial_length=initial_length)
        records.append(
            {
                "tick": int(tick),
                "taskId": str(task["taskId"]),
                "eventsJson": _compact_json(events),
                "preSortednessPercent": pre["sortednessPercent"],
                "postSortednessPercent": post["sortednessPercent"],
                "postInRange": bool(post["sortednessPercent"] >= threshold),
                "currentLength": post["currentLength"],
                "currentPairDenominator": post["currentPairDenominator"],
                "valuesJson": _compact_json(values),
                "oracleRecordJson": _compact_json(oracle_record),
            }
        )
    failure_ticks = [record for record in records if not record["postInRange"]]
    return {
        "taskId": str(task["taskId"]),
        "baselineId": ORACLE_BASELINE_ID,
        "oracleAllowed": True,
        "oracleBaseline": True,
        "oracleBaselineLabel": ORACLE_BASELINE_ID,
        "usesGlobalSortednessSignal": True,
        "usesWholeArrayValues": True,
        "usesWholeArrayTargetSignal": True,
        "timeInRangeFraction": float((len(records) - len(failure_ticks)) / max(1, len(records))),
        "failureDurationTicks": int(len(failure_ticks)),
        "energyProxy": int(energy),
        "finalValuesJson": _compact_json(values),
        "tickRecords": records,
        "oracleRecords": oracle_records,
        "homeostasisBenchmarkVersion": HOMEOSTASIS_BENCHMARK_VERSION,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
    }
