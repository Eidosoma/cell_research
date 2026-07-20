"""Priced top-down communication and actuation channels for E06 S06.

The channel layer never widens S05 policy observations implicitly.  Engine
adapters may construct privileged source projections, but policies and central
controllers receive only validated, versioned payloads with explicit
information, computation, and action ledgers.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .environments import Environment, boundary_observation
from .movements import MovementState, resolve_batch
from .policies import (
    ObservationBuild,
    PolicyDecision,
    PolicyValidationError,
    materialize_decision,
    validate_policy_payload,
)


CHANNEL_CATALOG_VERSION = "e06.s06.control-channel-catalog.v1"
DELIVERY_VERSION = "e06.s06.channel-delivery.v1"
DIRECT_VIEW_VERSION = "e06.s06.direct-controller-view.v1"
DIRECT_EVENT_VERSION = "e06.s06.direct-intervention-event.v1"
SUPPORTED_CHANNEL_TYPES = {
    "static_gradient",
    "boundary_signal",
    "sparse_instruction",
    "global_summary",
    "direct_intervention",
}
INSTRUCTION_ALPHABET = (
    "noop",
    "prefer_exploration",
    "prefer_local_relation",
    "seek_natural_boundary",
    "avoid_natural_boundary",
    "ascend_gradient",
    "descend_gradient",
    "conserve_movement",
)
CHANNEL_LEDGER_FIELDS = (
    "sourceScalarReads",
    "controllerStateReads",
    "recipientDeliveries",
    "configurationBits",
    "controllerInputBits",
    "policyDeliveryBits",
    "addressBits",
    "totalInformationBits",
    "noiseDraws",
    "controllerComputeUnits",
    "actuationAttempts",
    "actuationSuccesses",
    "overrideActionUnits",
    "movementGraphDisplacement",
    "suppressedNativeActions",
    "opportunityCostUnits",
)
FORBIDDEN_CONTROLLER_KEYS = {
    "actorid",
    "actoridentity",
    "analysislabel",
    "analysislabels",
    "boundarytags",
    "candidateproposals",
    "candidatesites",
    "completion",
    "conflictpriorities",
    "conflictpriority",
    "currentbatchproposals",
    "expectedoccupants",
    "fullstate",
    "futurestate",
    "futurerandomdraws",
    "globalaudit",
    "occupancy",
    "occupantid",
    "occupantidentities",
    "proposalid",
    "rawboundarytags",
    "route",
    "routes",
    "s01globalcompletionaudit",
    "s02wholegridscore",
    "siteid",
    "siteids",
    "siteroles",
    "sitetokens",
    "statesha256",
    "targetmembership",
}


class ChannelValidationError(ValueError):
    """Raised when a channel, projection, ledger, or event is invalid."""


@dataclass(frozen=True)
class ChannelDefinition:
    channel_id: str
    channel_type: str
    source: Mapping[str, Any]
    semantic_content: str
    spatial_resolution: str
    update_schedule: str
    noise_model: Mapping[str, Any]
    bandwidth: Mapping[str, Any]
    action_cost: Mapping[str, Any]
    controller_state_permissions: Mapping[str, Any]
    target_specificity: str
    unavoidable_asymmetries: tuple[str, ...]


@dataclass(frozen=True)
class ChannelDelivery:
    channel_id: str
    channel_type: str
    epoch_index: int
    recipient_scope: str
    payload: Mapping[str, Any]
    ledger: Mapping[str, int]
    source_commitment_sha256: str
    delivery_sha256: str


@dataclass(frozen=True)
class GlobalSummarySource:
    source_epoch: int
    local_dissatisfaction_bits: tuple[int, ...]
    submitted_proposals: int
    conflict_losses: int
    active_count: int


@dataclass(frozen=True)
class DirectControllerView:
    channel_id: str
    epoch_index: int
    recipient_alias: str
    payload: Mapping[str, Any]
    ledger: Mapping[str, int]
    view_sha256: str


@dataclass(frozen=True)
class DirectControllerDecision:
    action: str
    selected_candidate_key: str | None
    reason: str
    score: int | None


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_payload(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + _canonical_json_bytes(value)
    ).hexdigest()


def _bits_for_count(maximum_inclusive: int) -> int:
    return max(1, math.ceil(math.log2(maximum_inclusive + 1)))


def empty_channel_ledger() -> dict[str, int]:
    return {field: 0 for field in CHANNEL_LEDGER_FIELDS}


def _finalize_ledger(raw: Mapping[str, int]) -> dict[str, int]:
    ledger = empty_channel_ledger()
    if set(raw) - set(ledger):
        raise ChannelValidationError("unknown channel ledger field")
    ledger.update({key: int(value) for key, value in raw.items()})
    if any(value < 0 for value in ledger.values()):
        raise ChannelValidationError("channel ledger values must be nonnegative")
    expected = (
        ledger["configurationBits"]
        + ledger["controllerInputBits"]
        + ledger["policyDeliveryBits"]
        + ledger["addressBits"]
    )
    ledger["totalInformationBits"] = expected
    return ledger


def _normalize_key(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _forbidden_key_paths(value: Any, path: str = "payload") -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _normalize_key(key) in FORBIDDEN_CONTROLLER_KEYS:
                violations.append(f"{path}.{key}")
            violations.extend(_forbidden_key_paths(item, f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            violations.extend(_forbidden_key_paths(item, f"{path}[{index}]"))
    return violations


def validate_controller_view_payload(payload: Mapping[str, Any]) -> None:
    violations = _forbidden_key_paths(payload)
    if violations:
        raise ChannelValidationError(
            f"forbidden controller fields: {', '.join(violations)}"
        )
    local = payload.get("localPolicyPayload")
    if local is not None:
        try:
            validate_policy_payload(local)
        except PolicyValidationError as error:
            raise ChannelValidationError(str(error)) from error


def validate_channel_payload(payload: Mapping[str, Any]) -> None:
    violations = _forbidden_key_paths(payload)
    if violations:
        raise ChannelValidationError(
            f"forbidden channel payload fields: {', '.join(violations)}"
        )


def validate_source_projection(
    definition: ChannelDefinition, projection: Mapping[str, Any]
) -> None:
    allowed = {str(item) for item in definition.source["allowedInputs"]}
    extra = set(projection) - allowed
    if extra:
        raise ChannelValidationError(
            f"source projection fields are not allowlisted: {sorted(extra)}"
        )


def _counter_u64(
    scenario_key: str,
    channel_id: str,
    epoch_index: int,
    recipient_key: str,
    draw_index: int,
) -> int:
    if epoch_index < 0 or draw_index < 0:
        raise ChannelValidationError("noise counter indices must be nonnegative")
    payload = (
        b"E06/S06/channel-noise/v1\x00"
        + str(scenario_key).encode("utf-8")
        + b"\x00"
        + channel_id.encode("ascii")
        + b"\x00"
        + epoch_index.to_bytes(8, "big")
        + b"\x00"
        + str(recipient_key).encode("utf-8")
        + b"\x00"
        + draw_index.to_bytes(4, "big")
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _counter_bounded(
    scenario_key: str,
    channel_id: str,
    epoch_index: int,
    recipient_key: str,
    draw_index: int,
    bound: int,
) -> tuple[int, int]:
    if bound <= 0:
        raise ChannelValidationError("bounded channel draw requires a positive bound")
    limit = ((1 << 64) // bound) * bound
    consumed = 0
    while True:
        value = _counter_u64(
            scenario_key,
            channel_id,
            epoch_index,
            recipient_key,
            draw_index + consumed,
        )
        consumed += 1
        if value < limit:
            return value % bound, consumed


def _noise_outcome(
    definition: ChannelDefinition,
    *,
    scenario_key: str,
    epoch_index: int,
    recipient_key: str,
    draw_index: int,
) -> tuple[int | bool, int]:
    kind = definition.noise_model["kind"]
    parameters = definition.noise_model["parameters"]
    if kind == "bounded_uniform_integer":
        radius = int(parameters["radius"])
        value, draws = _counter_bounded(
            scenario_key,
            definition.channel_id,
            epoch_index,
            recipient_key,
            draw_index,
            2 * radius + 1,
        )
        return value - radius, draws
    if kind in {"independent_erasure", "independent_actuation_failure"}:
        numerator = int(parameters["numerator"])
        denominator = int(parameters["denominator"])
        value, draws = _counter_bounded(
            scenario_key,
            definition.channel_id,
            epoch_index,
            recipient_key,
            draw_index,
            denominator,
        )
        return value < numerator, draws
    if kind == "independent_adjacent_bin_jitter":
        down = int(parameters["downNumerator"])
        unchanged = int(parameters["unchangedNumerator"])
        up = int(parameters["upNumerator"])
        denominator = int(parameters["denominator"])
        if down + unchanged + up != denominator:
            raise ChannelValidationError("adjacent-bin noise probabilities do not sum")
        value, draws = _counter_bounded(
            scenario_key,
            definition.channel_id,
            epoch_index,
            recipient_key,
            draw_index,
            denominator,
        )
        if value < down:
            return -1, draws
        if value < down + unchanged:
            return 0, draws
        return 1, draws
    raise ChannelValidationError(f"unsupported channel noise model: {kind}")


def _delivery_body(
    definition: ChannelDefinition,
    epoch_index: int,
    recipient_scope: str,
    payload: Mapping[str, Any],
    ledger: Mapping[str, int],
    source_commitment_sha256: str,
) -> dict[str, Any]:
    return {
        "schemaVersion": DELIVERY_VERSION,
        "channelId": definition.channel_id,
        "channelType": definition.channel_type,
        "epochIndex": epoch_index,
        "recipientScope": recipient_scope,
        "payload": payload,
        "ledger": ledger,
        "sourceCommitment": {
            "sha256EngineAudit": source_commitment_sha256,
            "disclosedToRecipient": False,
        },
        "permissionAudit": {
            "s05BasePayloadUnchanged": True,
            "rawSourceFieldsDisclosed": False,
            "fullStateDisclosed": False,
            "globalCompletionDisclosed": False,
        },
    }


def _make_delivery(
    definition: ChannelDefinition,
    *,
    epoch_index: int,
    recipient_scope: str,
    payload: Mapping[str, Any],
    ledger: Mapping[str, int],
    source_commitment: Any,
) -> ChannelDelivery:
    if epoch_index < 0:
        raise ChannelValidationError("epoch index must be nonnegative")
    validate_channel_payload(payload)
    finalized = _finalize_ledger(ledger)
    source_hash = _sha256_payload("E06/S06/source-commitment/v1", source_commitment)
    body = _delivery_body(
        definition,
        epoch_index,
        recipient_scope,
        payload,
        finalized,
        source_hash,
    )
    return ChannelDelivery(
        channel_id=definition.channel_id,
        channel_type=definition.channel_type,
        epoch_index=epoch_index,
        recipient_scope=recipient_scope,
        payload=dict(payload),
        ledger=finalized,
        source_commitment_sha256=source_hash,
        delivery_sha256=_sha256_payload("E06/S06/delivery/v1", body),
    )


def delivery_to_dict(delivery: ChannelDelivery) -> dict[str, Any]:
    definition_stub = ChannelDefinition(
        channel_id=delivery.channel_id,
        channel_type=delivery.channel_type,
        source={},
        semantic_content="",
        spatial_resolution="",
        update_schedule="",
        noise_model={},
        bandwidth={},
        action_cost={},
        controller_state_permissions={},
        target_specificity="",
        unavoidable_asymmetries=(),
    )
    body = _delivery_body(
        definition_stub,
        delivery.epoch_index,
        delivery.recipient_scope,
        delivery.payload,
        delivery.ledger,
        delivery.source_commitment_sha256,
    )
    return {**body, "deliverySha256": delivery.delivery_sha256}


def canonical_delivery_bytes(delivery: ChannelDelivery) -> bytes:
    return _canonical_json_bytes(delivery_to_dict(delivery))


def parse_delivery(raw: Mapping[str, Any]) -> ChannelDelivery:
    required = {
        "schemaVersion",
        "channelId",
        "channelType",
        "epochIndex",
        "recipientScope",
        "payload",
        "ledger",
        "sourceCommitment",
        "permissionAudit",
        "deliverySha256",
    }
    if set(raw) != required or raw["schemaVersion"] != DELIVERY_VERSION:
        raise ChannelValidationError("channel delivery schema mismatch")
    if set(raw["ledger"]) != set(CHANNEL_LEDGER_FIELDS):
        raise ChannelValidationError("channel delivery ledger schema mismatch")
    validate_channel_payload(raw["payload"])
    source = raw["sourceCommitment"]
    if set(source) != {"sha256EngineAudit", "disclosedToRecipient"}:
        raise ChannelValidationError("source commitment schema mismatch")
    delivery = ChannelDelivery(
        channel_id=str(raw["channelId"]),
        channel_type=str(raw["channelType"]),
        epoch_index=int(raw["epochIndex"]),
        recipient_scope=str(raw["recipientScope"]),
        payload=dict(raw["payload"]),
        ledger=_finalize_ledger(raw["ledger"]),
        source_commitment_sha256=str(source["sha256EngineAudit"]),
        delivery_sha256=str(raw["deliverySha256"]),
    )
    if delivery_to_dict(delivery) != dict(raw):
        raise ChannelValidationError("channel delivery content/hash mismatch")
    expected = _sha256_payload(
        "E06/S06/delivery/v1",
        {key: raw[key] for key in raw if key != "deliverySha256"},
    )
    if expected != delivery.delivery_sha256:
        raise ChannelValidationError("channel delivery hash mismatch")
    return delivery


def compile_static_gradient(
    environment: Environment,
    definition: ChannelDefinition,
    *,
    axis: str,
    direction: str,
) -> tuple[dict[str, int], dict[str, int]]:
    if definition.channel_type != "static_gradient":
        raise ChannelValidationError("wrong channel type for gradient compiler")
    if axis not in {"first", "second"} or direction not in {"up", "down"}:
        raise ChannelValidationError("gradient axis/direction invalid")
    index = 0 if axis == "first" else 1
    sites = environment.occupiable_sites
    values = [site.coordinate[index] for site in sites]
    minimum, maximum = min(values), max(values)
    if minimum == maximum:
        raise ChannelValidationError("gradient axis has no spatial variation")
    field = {
        site.site_id: round(
            255 * (site.coordinate[index] - minimum) / (maximum - minimum)
        )
        for site in sites
    }
    ledger = _finalize_ledger(
        {
            "sourceScalarReads": len(sites),
            "configurationBits": 8 * len(sites) + 1,
        }
    )
    return field, ledger


def realize_noisy_gradient(
    definition: ChannelDefinition,
    field: Mapping[str, int],
    *,
    scenario_key: str,
    epoch_index: int = 0,
) -> tuple[dict[str, int], int]:
    noisy: dict[str, int] = {}
    draws = 0
    for index, (site_id, value) in enumerate(sorted(field.items())):
        jitter, consumed = _noise_outcome(
            definition,
            scenario_key=scenario_key,
            epoch_index=epoch_index,
            recipient_key=f"gradient-site-{index}",
            draw_index=draws,
        )
        draws += consumed
        noisy[site_id] = max(0, min(255, int(value) + int(jitter)))
    return noisy, draws


def build_gradient_delivery_from_s05(
    definition: ChannelDefinition,
    observation_build: ObservationBuild,
    *,
    compiled_field: Mapping[str, int],
    direction: str,
    configuration_ledger: Mapping[str, int],
    noise_draws: int,
    epoch_index: int,
) -> ChannelDelivery:
    if definition.channel_type != "static_gradient":
        raise ChannelValidationError("wrong channel type for gradient delivery")
    payload = observation_build.observation.payload
    if "currentGradientLevel" not in payload:
        raise ChannelValidationError("S05 observation lacks gradient projection")
    channel_payload = {
        "currentGradientLevel": int(payload["currentGradientLevel"]),
        "gradientDirection": direction,
        "candidates": [
            {
                "candidateKey": item["candidateKey"],
                "gradientDelta": int(item["gradientDelta"]),
            }
            for item in payload["candidates"]
        ],
    }
    return _make_delivery(
        definition,
        epoch_index=epoch_index,
        recipient_scope="one_s05_policy_activation",
        payload=channel_payload,
        ledger={
            "sourceScalarReads": len(compiled_field)
            + observation_build.observation.budget["gradientSignalReads"],
            "recipientDeliveries": 1,
            "configurationBits": configuration_ledger["configurationBits"],
            "policyDeliveryBits": observation_build.observation.budget[
                "communicatedBitsUpperBound"
            ],
            "noiseDraws": noise_draws,
        },
        source_commitment={
            "field": dict(sorted(compiled_field.items())),
            "direction": direction,
        },
    )


def compile_boundary_levels(
    environment: Environment, definition: ChannelDefinition
) -> tuple[dict[str, int], dict[str, int]]:
    if definition.channel_type != "boundary_signal":
        raise ChannelValidationError("wrong channel type for boundary compiler")
    levels = {
        site.site_id: min(
            6, len(boundary_observation(environment, site.site_id)["boundaryTags"])
        )
        for site in environment.occupiable_sites
    }
    return levels, _finalize_ledger(
        {
            "sourceScalarReads": len(levels),
            "configurationBits": 3 * len(levels) + 2,
        }
    )


def build_boundary_delivery(
    definition: ChannelDefinition,
    *,
    current_level: int,
    candidate_levels: Mapping[str, int],
    direction: str,
    directive_applies: bool,
    configuration_bits: int,
    scenario_key: str,
    epoch_index: int,
) -> ChannelDelivery:
    if definition.channel_type != "boundary_signal":
        raise ChannelValidationError("wrong channel type for boundary delivery")
    if direction not in {"seek", "avoid"} or not 0 <= current_level <= 6:
        raise ChannelValidationError("boundary projection invalid")
    draws = 0
    current_erased, consumed = _noise_outcome(
        definition,
        scenario_key=scenario_key,
        epoch_index=epoch_index,
        recipient_key="boundary-current",
        draw_index=draws,
    )
    draws += consumed
    visible_current = 0 if current_erased else current_level
    candidates = []
    for index, (key, level) in enumerate(sorted(candidate_levels.items())):
        if not 0 <= int(level) <= 6:
            raise ChannelValidationError("candidate boundary level invalid")
        erased, consumed = _noise_outcome(
            definition,
            scenario_key=scenario_key,
            epoch_index=epoch_index,
            recipient_key=f"boundary-candidate-{index}",
            draw_index=draws,
        )
        draws += consumed
        visible = 0 if erased else int(level)
        raw_delta = visible - visible_current
        candidates.append(
            {
                "candidateKey": key,
                "naturalBoundaryDelta": raw_delta
                if direction == "seek"
                else -raw_delta,
                "signalPresent": not bool(erased),
            }
        )
    payload = {
        "currentNaturalBoundaryLevel": visible_current,
        "currentSignalPresent": not bool(current_erased),
        "boundaryDirection": direction,
        "boundaryDirectiveApplies": bool(directive_applies),
        "candidates": candidates,
    }
    return _make_delivery(
        definition,
        epoch_index=epoch_index,
        recipient_scope="one_s05_policy_activation",
        payload=payload,
        ledger={
            "sourceScalarReads": 1 + len(candidate_levels),
            "recipientDeliveries": 1,
            "configurationBits": configuration_bits,
            "policyDeliveryBits": 6 + 5 * len(candidate_levels),
            "noiseDraws": draws,
        },
        source_commitment={
            "currentLevel": current_level,
            "candidateLevels": dict(sorted(candidate_levels.items())),
            "direction": direction,
            "directiveApplies": directive_applies,
        },
    )


def build_sparse_instruction_delivery(
    definition: ChannelDefinition,
    *,
    instruction: str,
    region_alias: str,
    region_size: int,
    region_count: int,
    scenario_key: str,
    epoch_index: int,
    scheduled_instruction_count: int = 1,
) -> ChannelDelivery:
    if definition.channel_type != "sparse_instruction":
        raise ChannelValidationError("wrong channel type for instruction delivery")
    if instruction not in INSTRUCTION_ALPHABET:
        raise ChannelValidationError("instruction is outside frozen alphabet")
    if region_size < 4 or not 1 <= region_count <= 16:
        raise ChannelValidationError("instruction region is too fine or numerous")
    if scheduled_instruction_count < 1:
        raise ChannelValidationError("scheduled instruction count must be positive")
    erased, draws = _noise_outcome(
        definition,
        scenario_key=scenario_key,
        epoch_index=epoch_index,
        recipient_key=region_alias,
        draw_index=0,
    )
    payload = {
        "instruction": "noop" if erased else instruction,
        "instructionPresent": not bool(erased),
    }
    address_bits = _bits_for_count(region_count - 1)
    configuration_bits = 3 * scheduled_instruction_count + address_bits * region_count
    return _make_delivery(
        definition,
        epoch_index=epoch_index,
        recipient_scope=f"coarse_region:{region_alias}",
        payload=payload,
        ledger={
            "sourceScalarReads": 1,
            "controllerStateReads": 1,
            "recipientDeliveries": region_size,
            "configurationBits": configuration_bits,
            "policyDeliveryBits": 4 * region_size,
            "addressBits": address_bits,
            "noiseDraws": draws,
            "controllerComputeUnits": 1,
        },
        source_commitment={
            "instruction": instruction,
            "regionAlias": region_alias,
            "regionSize": region_size,
            "regionCount": region_count,
            "epochIndex": epoch_index,
        },
    )


def _fraction_bin(numerator: int, denominator: int, bins: int) -> int:
    if denominator <= 0 or not 0 <= numerator <= denominator:
        raise ChannelValidationError("aggregate numerator/denominator invalid")
    return min(bins - 1, (numerator * bins) // denominator)


def build_global_summary_delivery(
    definition: ChannelDefinition,
    source: GlobalSummarySource,
    *,
    scenario_key: str,
    epoch_index: int,
    recipient_count: int,
) -> ChannelDelivery:
    if definition.channel_type != "global_summary":
        raise ChannelValidationError("wrong channel type for global summary")
    if source.source_epoch != epoch_index - 1:
        raise ChannelValidationError("global summary must be delayed exactly one epoch")
    if source.active_count != len(source.local_dissatisfaction_bits):
        raise ChannelValidationError("active count and dissatisfaction inputs mismatch")
    if any(value not in {0, 1} for value in source.local_dissatisfaction_bits):
        raise ChannelValidationError("dissatisfaction projection must contain bits")
    if not 1 <= recipient_count <= 16:
        raise ChannelValidationError("summary recipient count exceeds epoch budget")
    dissatisfied = sum(source.local_dissatisfaction_bits)
    raw_dissatisfaction = _fraction_bin(dissatisfied, source.active_count, 8)
    raw_conflict = _fraction_bin(
        source.conflict_losses, max(1, source.submitted_proposals), 4
    )
    draws = 0
    diss_jitter, consumed = _noise_outcome(
        definition,
        scenario_key=scenario_key,
        epoch_index=epoch_index,
        recipient_key="summary-dissatisfaction",
        draw_index=draws,
    )
    draws += consumed
    conflict_jitter, consumed = _noise_outcome(
        definition,
        scenario_key=scenario_key,
        epoch_index=epoch_index,
        recipient_key="summary-conflict",
        draw_index=draws,
    )
    draws += consumed
    payload = {
        "dissatisfactionBin": max(0, min(7, raw_dissatisfaction + int(diss_jitter))),
        "laggedConflictBin": max(0, min(3, raw_conflict + int(conflict_jitter))),
        "summaryAgeEpochs": 1,
    }
    return _make_delivery(
        definition,
        epoch_index=epoch_index,
        recipient_scope="global_broadcast",
        payload=payload,
        ledger={
            "sourceScalarReads": source.active_count + source.submitted_proposals,
            "recipientDeliveries": recipient_count,
            "configurationBits": 8,
            "policyDeliveryBits": 6 * recipient_count,
            "noiseDraws": draws,
            "controllerComputeUnits": source.active_count + source.submitted_proposals,
        },
        source_commitment={
            "sourceEpoch": source.source_epoch,
            "localDissatisfactionBits": list(source.local_dissatisfaction_bits),
            "submittedProposals": source.submitted_proposals,
            "conflictLosses": source.conflict_losses,
            "activeCount": source.active_count,
        },
    )


def state_blind_recipient_alias(epoch_index: int, population_size: int) -> str:
    if epoch_index < 0 or population_size <= 0:
        raise ChannelValidationError("recipient schedule inputs invalid")
    return f"u{epoch_index % population_size:04d}"


def build_direct_controller_view(
    definition: ChannelDefinition,
    observation_build: ObservationBuild,
    *,
    epoch_index: int,
    population_size: int,
    lagged_summary: Mapping[str, int],
    remaining_information_budget: int,
    remaining_action_budget: int,
) -> DirectControllerView:
    if definition.channel_type != "direct_intervention":
        raise ChannelValidationError("wrong channel type for direct controller view")
    if min(remaining_information_budget, remaining_action_budget) < 0:
        raise ChannelValidationError("controller budgets must be nonnegative")
    if set(lagged_summary) != {
        "dissatisfactionBin",
        "laggedConflictBin",
        "summaryAgeEpochs",
    }:
        raise ChannelValidationError("direct controller summary schema mismatch")
    alias = state_blind_recipient_alias(epoch_index, population_size)
    payload = {
        "epochIndex": epoch_index,
        "recipientAlias": alias,
        "laggedGlobalSummary": dict(lagged_summary),
        "localPolicyPayload": observation_build.observation.payload,
        "remainingInformationBudget": remaining_information_budget,
        "remainingActionBudget": remaining_action_budget,
    }
    validate_controller_view_payload(payload)
    address_bits = _bits_for_count(population_size - 1)
    local_bits = observation_build.observation.budget["communicatedBitsUpperBound"]
    ledger = _finalize_ledger(
        {
            "sourceScalarReads": 1,
            "controllerStateReads": 4,
            "recipientDeliveries": 1,
            "configurationBits": 64,
            "controllerInputBits": local_bits + 6 + 24,
            "addressBits": address_bits,
            "controllerComputeUnits": len(
                observation_build.observation.payload["candidates"]
            ),
        }
    )
    body = {
        "schemaVersion": DIRECT_VIEW_VERSION,
        "channelId": definition.channel_id,
        "epochIndex": epoch_index,
        "recipientAlias": alias,
        "payload": payload,
        "ledger": ledger,
        "permissionAudit": {
            "recipientSelectedWithoutState": True,
            "oneS05PayloadOnly": True,
            "siteIdentityRouteAndAuthenticationAbsent": True,
            "globalCompletionAbsent": True,
        },
    }
    return DirectControllerView(
        channel_id=definition.channel_id,
        epoch_index=epoch_index,
        recipient_alias=alias,
        payload=payload,
        ledger=ledger,
        view_sha256=_sha256_payload("E06/S06/direct-view/v1", body),
    )


def direct_view_to_dict(view: DirectControllerView) -> dict[str, Any]:
    return {
        "schemaVersion": DIRECT_VIEW_VERSION,
        "channelId": view.channel_id,
        "epochIndex": view.epoch_index,
        "recipientAlias": view.recipient_alias,
        "payload": view.payload,
        "ledger": view.ledger,
        "permissionAudit": {
            "recipientSelectedWithoutState": True,
            "oneS05PayloadOnly": True,
            "siteIdentityRouteAndAuthenticationAbsent": True,
            "globalCompletionAbsent": True,
        },
        "viewSha256": view.view_sha256,
    }


def canonical_direct_view_bytes(view: DirectControllerView) -> bytes:
    return _canonical_json_bytes(direct_view_to_dict(view))


def decide_direct_controller(
    view_payload: Mapping[str, Any],
) -> DirectControllerDecision:
    """Pure bounded controller over one validated S05 payload and lagged bins."""

    validate_controller_view_payload(view_payload)
    if int(view_payload["remainingActionBudget"]) <= 0:
        return DirectControllerDecision("noop", None, "action_budget_exhausted", None)
    local = view_payload["localPolicyPayload"]
    candidates = list(local["candidates"])
    if not candidates:
        return DirectControllerDecision("noop", None, "no_candidate", None)

    def score(record: Mapping[str, Any]) -> int:
        value = int(record.get("localRelationDelta", 0))
        value += int(record.get("gradientDelta", 0))
        value += int(record.get("naturalBoundaryDelta", 0))
        value -= int(record.get("movementCost", 0))
        return value

    best = max(
        candidates,
        key=lambda item: (score(item), -int(str(item["candidateKey"])[1:])),
    )
    return DirectControllerDecision(
        "select_candidate",
        str(best["candidateKey"]),
        "best_disclosed_local_projection",
        score(best),
    )


def direct_decision_to_dict(decision: DirectControllerDecision) -> dict[str, Any]:
    return {
        "action": decision.action,
        "selectedCandidateKey": decision.selected_candidate_key,
        "reason": decision.reason,
        "score": decision.score,
    }


def execute_direct_intervention(
    definition: ChannelDefinition,
    environment: Environment,
    state: MovementState,
    observation_build: ObservationBuild,
    view: DirectControllerView,
    decision: DirectControllerDecision,
    *,
    scenario_key: str,
    batch_nonce: str,
) -> dict[str, Any]:
    if definition.channel_type != "direct_intervention":
        raise ChannelValidationError("wrong channel type for direct intervention")
    if view.channel_id != definition.channel_id:
        raise ChannelValidationError("direct view channel mismatch")
    failed, noise_draws = _noise_outcome(
        definition,
        scenario_key=scenario_key,
        epoch_index=view.epoch_index,
        recipient_key=view.recipient_alias,
        draw_index=0,
    )
    proposal = None
    if decision.action == "select_candidate":
        policy_decision = PolicyDecision(
            policy_id=observation_build.observation.policy_id,
            action="proposal",
            selected_candidate_key=decision.selected_candidate_key,
            reason="s06_direct_override",
            decision_score=decision.score,
        )
        if not failed:
            proposal = materialize_decision(observation_build, policy_decision)
    elif decision.action != "noop":
        raise ChannelValidationError("unsupported direct-controller action")
    batch = resolve_batch(
        environment,
        state,
        () if proposal is None else (proposal,),
        batch_nonce=batch_nonce,
    )
    succeeded = (
        proposal is not None and proposal.proposal_id in batch["acceptedProposalIds"]
    )
    ledger = dict(view.ledger)
    ledger.update(
        _finalize_ledger(
            {
                **ledger,
                "noiseDraws": noise_draws,
                "actuationAttempts": int(decision.action != "noop"),
                "actuationSuccesses": int(succeeded),
                "overrideActionUnits": int(decision.action != "noop"),
                "movementGraphDisplacement": batch["costLedger"][
                    "totalGraphDisplacement"
                ],
                "opportunityCostUnits": int(decision.action != "noop"),
            }
        )
    )
    body = {
        "schemaVersion": DIRECT_EVENT_VERSION,
        "channelId": definition.channel_id,
        "epochIndex": view.epoch_index,
        "controllerView": direct_view_to_dict(view),
        "controllerDecision": direct_decision_to_dict(decision),
        "actuationNoise": {
            "failed": bool(failed),
            "draws": noise_draws,
            "addressDisclosedToController": False,
        },
        "materializationAudit": {
            "proposalCreated": proposal is not None,
            "authenticationAndRouteAddedEngineSide": proposal is not None,
            "controllerSawProposalEnvelope": False,
        },
        "movementBatchResult": batch,
        "ledger": ledger,
        "outcome": "accepted"
        if succeeded
        else ("actuation_failure" if failed else "noop_or_rejected"),
    }
    return {**body, "eventSha256": _sha256_payload("E06/S06/direct-event/v1", body)}


def canonical_direct_event_bytes(event: Mapping[str, Any]) -> bytes:
    return _canonical_json_bytes(dict(event))


def validate_ledger_against_budget(
    definition: ChannelDefinition, ledger: Mapping[str, int]
) -> list[str]:
    if set(ledger) != set(CHANNEL_LEDGER_FIELDS):
        return ["ledger_schema"]
    failures = []
    limits = {
        "configurationBits": definition.bandwidth["maxConfigurationBitsPerScenario"],
        "controllerInputBits": definition.bandwidth["maxControllerInputBitsPerEpoch"],
        "policyDeliveryBits": definition.bandwidth["maxPolicyDeliveryBitsPerEpoch"],
        "addressBits": definition.bandwidth["maxAddressBitsPerEpoch"],
        "actuationAttempts": definition.action_cost["maxActuationAttemptsPerEpoch"],
        "overrideActionUnits": definition.action_cost["maxOverrideActionUnitsPerEpoch"],
        "movementGraphDisplacement": definition.action_cost[
            "maxMovementGraphDisplacementPerEpoch"
        ],
    }
    for field, maximum in limits.items():
        if int(ledger[field]) > int(maximum):
            failures.append(field)
    if ledger["totalInformationBits"] != (
        ledger["configurationBits"]
        + ledger["controllerInputBits"]
        + ledger["policyDeliveryBits"]
        + ledger["addressBits"]
    ):
        failures.append("totalInformationBits")
    return sorted(set(failures))


def channel_comparison_vector(
    definition: ChannelDefinition, ledger: Mapping[str, int]
) -> dict[str, Any]:
    return {
        "channelId": definition.channel_id,
        "channelType": definition.channel_type,
        "configurationBits": ledger["configurationBits"],
        "controllerInputBits": ledger["controllerInputBits"],
        "policyDeliveryBits": ledger["policyDeliveryBits"],
        "addressBits": ledger["addressBits"],
        "totalInformationBits": ledger["totalInformationBits"],
        "sourceScalarReads": ledger["sourceScalarReads"],
        "controllerComputeUnits": ledger["controllerComputeUnits"],
        "actuationAttempts": ledger["actuationAttempts"],
        "overrideActionUnits": ledger["overrideActionUnits"],
        "movementGraphDisplacement": ledger["movementGraphDisplacement"],
        "targetSpecificity": definition.target_specificity,
        "semanticContent": definition.semantic_content,
        "scalarCollapseAllowed": False,
    }


def calibrate_noise(
    definition: ChannelDefinition, *, draws: int | None = None
) -> dict[str, Any]:
    draw_count = int(draws or definition.noise_model["calibrationDraws"])
    counts: Counter[str] = Counter()
    logical_draws = 0
    for index in range(draw_count):
        outcome, consumed = _noise_outcome(
            definition,
            scenario_key="s06-noise-calibration-v1",
            epoch_index=index,
            recipient_key="calibration",
            draw_index=0,
        )
        logical_draws += consumed
        counts[str(outcome).lower()] += 1
    kind = definition.noise_model["kind"]
    parameters = definition.noise_model["parameters"]
    if kind == "bounded_uniform_integer":
        radius = int(parameters["radius"])
        expected = {
            str(value): 1 / (2 * radius + 1) for value in range(-radius, radius + 1)
        }
    elif kind in {"independent_erasure", "independent_actuation_failure"}:
        probability = int(parameters["numerator"]) / int(parameters["denominator"])
        expected = {"true": probability, "false": 1 - probability}
    elif kind == "independent_adjacent_bin_jitter":
        denominator = int(parameters["denominator"])
        expected = {
            "-1": int(parameters["downNumerator"]) / denominator,
            "0": int(parameters["unchangedNumerator"]) / denominator,
            "1": int(parameters["upNumerator"]) / denominator,
        }
    else:
        raise ChannelValidationError("unsupported calibration model")
    observed = {key: counts[key] / draw_count for key in expected}
    errors = {key: abs(observed[key] - expected[key]) for key in expected}
    tolerance = float(definition.noise_model["toleranceAbsoluteProbability"])
    return {
        "channelId": definition.channel_id,
        "noiseKind": kind,
        "sampleCount": draw_count,
        "counterDrawsConsumed": logical_draws,
        "expectedProbabilities": expected,
        "observedProbabilities": observed,
        "absoluteErrors": errors,
        "maximumAbsoluteError": max(errors.values()),
        "toleranceAbsoluteProbability": tolerance,
        "success": max(errors.values()) <= tolerance,
    }


def controller_source_forbidden_accesses() -> list[str]:
    source = inspect.getsource(decide_direct_controller).lower()
    forbidden = [
        "analysis_label",
        "actor_id",
        "site_id",
        "route",
        "occupancy",
        "state_sha",
        "proposal_id",
        "global_completion",
        "target_membership",
        "conflict_priority",
        "future_state",
    ]
    return sorted(item for item in forbidden if item in source)


def channel_to_dict(definition: ChannelDefinition) -> dict[str, Any]:
    return {
        "channelId": definition.channel_id,
        "channelType": definition.channel_type,
        "source": dict(definition.source),
        "semanticContent": definition.semantic_content,
        "spatialResolution": definition.spatial_resolution,
        "updateSchedule": definition.update_schedule,
        "noiseModel": dict(definition.noise_model),
        "bandwidth": dict(definition.bandwidth),
        "actionCost": dict(definition.action_cost),
        "controllerStatePermissions": dict(definition.controller_state_permissions),
        "targetSpecificity": definition.target_specificity,
        "unavoidableAsymmetries": list(definition.unavoidable_asymmetries),
    }


def _parse_channel(raw: Mapping[str, Any]) -> ChannelDefinition:
    required = {
        "channelId",
        "channelType",
        "source",
        "semanticContent",
        "spatialResolution",
        "updateSchedule",
        "noiseModel",
        "bandwidth",
        "actionCost",
        "controllerStatePermissions",
        "targetSpecificity",
        "unavoidableAsymmetries",
    }
    if set(raw) != required:
        raise ChannelValidationError("channel declaration keys mismatch")
    channel_type = str(raw["channelType"])
    if channel_type not in SUPPORTED_CHANNEL_TYPES:
        raise ChannelValidationError("unsupported channel type")
    source = dict(raw["source"])
    if set(source) != {
        "owner",
        "implementation",
        "allowedInputs",
        "runtimeStateReads",
    }:
        raise ChannelValidationError("channel source declaration mismatch")
    noise = dict(raw["noiseModel"])
    if set(noise) != {
        "kind",
        "parameters",
        "calibrationDraws",
        "toleranceAbsoluteProbability",
    }:
        raise ChannelValidationError("channel noise declaration mismatch")
    bandwidth = dict(raw["bandwidth"])
    if set(bandwidth) != {
        "configurationBitsFormula",
        "policyBitsFormula",
        "maxConfigurationBitsPerScenario",
        "maxControllerInputBitsPerEpoch",
        "maxPolicyDeliveryBitsPerEpoch",
        "maxAddressBitsPerEpoch",
    }:
        raise ChannelValidationError("channel bandwidth declaration mismatch")
    action = dict(raw["actionCost"])
    if set(action) != {
        "kind",
        "maxActuationAttemptsPerEpoch",
        "maxOverrideActionUnitsPerEpoch",
        "maxMovementGraphDisplacementPerEpoch",
    }:
        raise ChannelValidationError("channel action-cost declaration mismatch")
    permissions = dict(raw["controllerStatePermissions"])
    if set(permissions) != {"allowed", "persistentBits"}:
        raise ChannelValidationError("controller-state declaration mismatch")
    if any(
        int(value) < 0
        for key, value in {**bandwidth, **action}.items()
        if key.startswith("max")
    ):
        raise ChannelValidationError("channel budget maxima must be nonnegative")
    if (
        channel_type == "direct_intervention"
        and {_normalize_key(item) for item in source["allowedInputs"]}
        & FORBIDDEN_CONTROLLER_KEYS
    ):
        raise ChannelValidationError(
            "direct controller source allowlist contains hidden state"
        )
    if {
        _normalize_key(item) for item in permissions["allowed"]
    } & FORBIDDEN_CONTROLLER_KEYS:
        raise ChannelValidationError(
            "controller persistent state allowlist contains hidden state"
        )
    if int(noise["calibrationDraws"]) < 100000:
        raise ChannelValidationError("noise calibration must use at least 100000 draws")
    if not 0 < float(noise["toleranceAbsoluteProbability"]) <= 0.02:
        raise ChannelValidationError("noise calibration tolerance invalid")
    return ChannelDefinition(
        channel_id=str(raw["channelId"]),
        channel_type=channel_type,
        source=source,
        semantic_content=str(raw["semanticContent"]),
        spatial_resolution=str(raw["spatialResolution"]),
        update_schedule=str(raw["updateSchedule"]),
        noise_model=noise,
        bandwidth=bandwidth,
        action_cost=action,
        controller_state_permissions=permissions,
        target_specificity=str(raw["targetSpecificity"]),
        unavoidable_asymmetries=tuple(
            str(item) for item in raw["unavoidableAsymmetries"]
        ),
    )


def parse_channel_catalog(
    raw: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[ChannelDefinition, ...]]:
    required = {
        "schemaVersion",
        "researchStepId",
        "libraryVersion",
        "commonBudgetContract",
        "permissionContract",
        "channels",
        "instructionAlphabet",
        "fixtures",
    }
    if set(raw) != required:
        raise ChannelValidationError("channel catalog keys mismatch")
    if (
        raw["schemaVersion"] != CHANNEL_CATALOG_VERSION
        or raw["researchStepId"] != "S06"
    ):
        raise ChannelValidationError("channel catalog version mismatch")
    if tuple(raw["instructionAlphabet"]) != INSTRUCTION_ALPHABET:
        raise ChannelValidationError("instruction alphabet mismatch")
    declared_ledger = tuple(raw["commonBudgetContract"]["ledgerFields"])
    if declared_ledger != CHANNEL_LEDGER_FIELDS:
        raise ChannelValidationError("common channel ledger mismatch")
    channels = tuple(_parse_channel(item) for item in raw["channels"])
    if len(channels) != 5 or {item.channel_type for item in channels} != (
        SUPPORTED_CHANNEL_TYPES
    ):
        raise ChannelValidationError("catalog must define each S06 channel once")
    if len({item.channel_id for item in channels}) != len(channels):
        raise ChannelValidationError("channel IDs must be unique")
    fixtures = raw["fixtures"]
    if len(fixtures) != 5 or {item["channelId"] for item in fixtures} != {
        item.channel_id for item in channels
    }:
        raise ChannelValidationError("catalog must define one fixture per channel")
    metadata = {key: raw[key] for key in required - {"channels"}}
    return metadata, channels


def load_channel_catalog(
    path: str | Path,
) -> tuple[Mapping[str, Any], tuple[ChannelDefinition, ...]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ChannelValidationError("channel catalog root must be a mapping")
    return parse_channel_catalog(raw)
