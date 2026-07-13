"""Canonical JSON Schema definitions and deterministic identity helpers."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping


EVENT_SCHEMA_VERSION = "e01.shared_event.v1.0.0"
TRACE_MANIFEST_SCHEMA_VERSION = "e01.shared_trace_manifest.v1.0.0"
ADAPTER_VERSION = "e01.shared_event_adapters.v1.0.0"
EVENT_SCHEMA_URN = "urn:eidosoma:e01:shared-event:1.0.0"
TRACE_SCHEMA_URN = "urn:eidosoma:e01:shared-trace-manifest:1.0.0"

AVAILABILITY_STATUSES = (
    "observed",
    "derived_exact",
    "derived_ambiguous",
    "unavailable",
    "not_applicable",
)

# Every nullable/comparability-critical semantic field has an obligatory entry.
# This prevents null from silently meaning either absent, inapplicable, or lossy.
TRACKED_FIELDS = (
    "run.scenarioId",
    "run.eventIndex",
    "randomness.streamIdentifiers",
    "randomness.addressingProfile",
    "randomness.draws",
    "randomness.seedOverride",
    "actor.id",
    "actor.algotype",
    "actor.direction",
    "actor.prePosition",
    "actor.postPosition",
    "target.id",
    "target.prePosition",
    "target.postPosition",
    "observation.readCount",
    "observation.valueComparisonCount",
    "observation.details",
    "observation.logicalProjection",
    "proposal.kind",
    "proposal.reason",
    "proposal.targetPosition",
    "proposal.newCursor",
    "proposal.priorityUint64",
    "proposal.ordinal",
    "decision.status",
    "decision.accepted",
    "faultInteraction.actorMode",
    "faultInteraction.targetMode",
    "faultInteraction.kind",
    "displacement.count",
    "displacement.positions",
    "costDelta.activations",
    "costDelta.observationReads",
    "costDelta.valueComparisons",
    "costDelta.proposals",
    "costDelta.noOps",
    "costDelta.rejections",
    "costDelta.memoryUpdates",
    "costDelta.acceptedSwaps",
    "costDelta.displacedCells",
    "costDelta.conflictLosses",
    "state.nativePreHash",
    "state.nativePostHash",
    "state.nativeHashScope",
    "state.observablePreHash",
    "state.observablePostHash",
    "stop.reason",
    "stop.terminal",
    "stop.censored",
)

HISTORICAL_ALWAYS_LOSSY_FIELDS = frozenset(
    {
        "randomness.streamIdentifiers",
        "randomness.addressingProfile",
        "randomness.draws",
        "actor.id",
        "actor.prePosition",
        "actor.postPosition",
        "target.id",
        "target.prePosition",
        "target.postPosition",
        "observation.readCount",
        "observation.valueComparisonCount",
        "observation.details",
        "observation.logicalProjection",
        "proposal.reason",
        "proposal.targetPosition",
        "proposal.newCursor",
        "proposal.priorityUint64",
        "proposal.ordinal",
        "costDelta.activations",
        "costDelta.observationReads",
        "costDelta.valueComparisons",
        "costDelta.proposals",
        "costDelta.noOps",
        "costDelta.rejections",
        "costDelta.memoryUpdates",
        "costDelta.conflictLosses",
        "state.nativePreHash",
        "state.nativePostHash",
        "state.nativeHashScope",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def event_content_hash(event: Mapping[str, Any]) -> str:
    material = deepcopy(dict(event))
    material.pop("eventId", None)
    return sha256_json(material)


def assign_event_id(event: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(event))
    result.pop("eventId", None)
    result["eventId"] = "sha256:" + sha256_json(result)
    return result


def trace_content_hash(events: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]) -> str:
    digest = hashlib.sha256(b"E01/shared-trace/v1\x00")
    for event in events:
        digest.update(canonical_json_bytes(event))
        digest.update(b"\n")
    return digest.hexdigest()


def observable_state_hash(values: list[int | float]) -> str:
    return "sha256:" + sha256_json({"scope": "ordered_values_v1", "values": values})


def availability_record(
    status: str,
    *,
    source: str,
    method: str | None = None,
    reason_code: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    if status not in AVAILABILITY_STATUSES:
        raise ValueError(f"unsupported availability status: {status}")
    return {
        "status": status,
        "source": source,
        "method": method,
        "reasonCode": reason_code,
        "note": note,
    }


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def _strict_object(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required if required is not None else list(properties),
    }


def event_schema() -> dict[str, Any]:
    nullable_string = _nullable({"type": "string"})
    nullable_nonempty_string = _nullable({"type": "string", "minLength": 1})
    nullable_integer = _nullable({"type": "integer"})
    nullable_nonnegative_integer = _nullable({"type": "integer", "minimum": 0})
    nullable_boolean = _nullable({"type": "boolean"})
    availability = _strict_object(
        {
            "status": {"enum": list(AVAILABILITY_STATUSES)},
            "source": {"type": "string", "minLength": 1},
            "method": nullable_string,
            "reasonCode": nullable_string,
            "note": nullable_string,
        }
    )
    draw = _strict_object(
        {
            "stream": {"type": "string", "minLength": 1},
            "eventIndex": {"type": "integer", "minimum": 0},
            "drawIndex": {"type": "integer", "minimum": 0},
            "uint64": {"type": "string", "pattern": "^[0-9]+$"},
        }
    )
    cost_properties = {
        name: _nullable({"type": "integer", "minimum": 0})
        for name in (
            "activations",
            "observationReads",
            "valueComparisons",
            "proposals",
            "noOps",
            "rejections",
            "memoryUpdates",
            "acceptedSwaps",
            "displacedCells",
            "conflictLosses",
        )
    }
    field_availability = _strict_object(
        {path: {"$ref": "#/$defs/availability"} for path in TRACKED_FIELDS}
    )
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": EVENT_SCHEMA_URN,
        "title": "E01 shared backend event",
        "description": (
            "One semantic event with mandatory field-level availability/provenance. "
            "Null never implies parity: fieldAvailability states observed, exact-derived, "
            "ambiguous-derived, unavailable, or not-applicable."
        ),
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schemaVersion",
            "schemaUri",
            "eventId",
            "recordType",
            "backend",
            "run",
            "randomness",
            "actor",
            "target",
            "observation",
            "proposal",
            "decision",
            "faultInteraction",
            "displacement",
            "costDelta",
            "state",
            "stop",
            "fieldAvailability",
            "backendPayload",
        ],
        "properties": {
            "schemaVersion": {"const": EVENT_SCHEMA_VERSION},
            "schemaUri": {"const": EVENT_SCHEMA_URN},
            "eventId": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
            "recordType": {"const": "event"},
            "backend": _strict_object(
                {
                    "backendId": {
                        "enum": ["reference", "historical_frozen_public_commit"]
                    },
                    "profile": {"type": "string", "minLength": 1},
                    "evidenceLayer": {
                        "enum": ["clean_room_reference", "frozen_public_commit"]
                    },
                    "sourceCommit": nullable_string,
                    "publicationSnapshotClaimed": {"const": False},
                    "adapterVersion": {"const": ADAPTER_VERSION},
                }
            ),
            "run": _strict_object(
                {
                    "runId": {"type": "string", "minLength": 1},
                    "scenarioId": nullable_nonempty_string,
                    "eventIndex": nullable_nonnegative_integer,
                    "sequenceBasis": {"enum": ["activation", "recorded_swap"]},
                    "batch": _strict_object(
                        {
                            "batchId": {"type": "string", "minLength": 1},
                            "width": {"type": "integer", "minimum": 1, "maximum": 8},
                            "ordinal": {"type": "integer", "minimum": 0},
                        }
                    ),
                }
            ),
            "randomness": _strict_object(
                {
                    "streamIdentifiers": _nullable(
                        {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
                    ),
                    "addressingProfile": nullable_string,
                    "draws": _nullable({"type": "array", "items": draw}),
                    "seedOverride": nullable_string,
                }
            ),
            "actor": _strict_object(
                {
                    "id": nullable_nonempty_string,
                    "algotype": nullable_nonempty_string,
                    "direction": _nullable(
                        {"enum": ["ascending", "descending", "controller"]}
                    ),
                    "prePosition": nullable_nonnegative_integer,
                    "postPosition": nullable_nonnegative_integer,
                }
            ),
            "target": _strict_object(
                {
                    "id": nullable_nonempty_string,
                    "prePosition": nullable_nonnegative_integer,
                    "postPosition": nullable_nonnegative_integer,
                }
            ),
            "observation": _strict_object(
                {
                    "readCount": nullable_nonnegative_integer,
                    "valueComparisonCount": nullable_nonnegative_integer,
                    "details": _nullable({"type": "object"}),
                    "logicalProjection": _nullable({"type": "object"}),
                }
            ),
            "proposal": _strict_object(
                {
                    "kind": _nullable({"enum": ["NoOp", "Swap", "MemoryUpdate"]}),
                    "reason": nullable_string,
                    "targetPosition": nullable_nonnegative_integer,
                    "newCursor": nullable_integer,
                    "priorityUint64": _nullable({"type": "string", "pattern": "^[0-9]+$"}),
                    "ordinal": nullable_nonnegative_integer,
                }
            ),
            "decision": _strict_object(
                {
                    "status": nullable_string,
                    "accepted": nullable_boolean,
                }
            ),
            "faultInteraction": _strict_object(
                {
                    "actorMode": _nullable({"enum": ["normal", "passive", "stuck"]}),
                    "targetMode": _nullable({"enum": ["normal", "passive", "stuck"]}),
                    "kind": nullable_string,
                }
            ),
            "displacement": _strict_object(
                {
                    "count": nullable_nonnegative_integer,
                    "positions": _nullable(
                        _strict_object(
                            {
                                "changed": {
                                    "type": "array",
                                    "items": {"type": "integer", "minimum": 0},
                                    "uniqueItems": True,
                                },
                                "actorFrom": nullable_integer,
                                "actorTo": nullable_integer,
                                "targetFrom": nullable_integer,
                                "targetTo": nullable_integer,
                            }
                        )
                    ),
                }
            ),
            "costDelta": _strict_object(cost_properties),
            "state": _strict_object(
                {
                    "nativePreHash": _nullable({"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"}),
                    "nativePostHash": _nullable({"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"}),
                    "nativeHashScope": nullable_string,
                    "observablePreHash": _nullable({"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"}),
                    "observablePostHash": _nullable({"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"}),
                    "observableHashScope": {"const": "ordered_values_v1"},
                }
            ),
            "stop": _strict_object(
                {
                    "terminal": nullable_boolean,
                    "reason": nullable_string,
                    "censored": nullable_boolean,
                }
            ),
            "fieldAvailability": field_availability,
            "backendPayload": {"type": "object"},
        },
        "$defs": {"availability": availability},
    }
    return schema


def trace_manifest_schema() -> dict[str, Any]:
    nullable_string = _nullable({"type": "string"})
    format_record = _strict_object(
        {
            "format": {"enum": ["jsonl.zst", "parquet.zstd"]},
            "path": {"type": "string", "minLength": 1},
            "bytes": {"type": "integer", "minimum": 0},
            "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "compression": {"const": "zstd"},
        }
    )
    source_record = _strict_object(
        {
            "path": {"type": "string", "minLength": 1},
            "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "classification": {"type": "string", "minLength": 1},
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": TRACE_SCHEMA_URN,
        "title": "E01 shared trace manifest",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schemaVersion",
            "schemaUri",
            "traceId",
            "eventSchemaVersion",
            "adapterVersion",
            "runId",
            "scenarioId",
            "backend",
            "sequenceBasis",
            "eventCount",
            "traceContentSha256",
            "formats",
            "sourceArtifacts",
            "fieldStatusCounts",
            "unavailableFields",
            "ambiguousFields",
            "stop",
            "sourceRunSummary",
        ],
        "properties": {
            "schemaVersion": {"const": TRACE_MANIFEST_SCHEMA_VERSION},
            "schemaUri": {"const": TRACE_SCHEMA_URN},
            "traceId": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
            "eventSchemaVersion": {"const": EVENT_SCHEMA_VERSION},
            "adapterVersion": {"const": ADAPTER_VERSION},
            "runId": {"type": "string", "minLength": 1},
            "scenarioId": nullable_string,
            "backend": {"enum": ["reference", "historical_frozen_public_commit"]},
            "sequenceBasis": {"enum": ["activation", "recorded_swap"]},
            "eventCount": {"type": "integer", "minimum": 0},
            "traceContentSha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "formats": {"type": "array", "items": format_record},
            "sourceArtifacts": {"type": "array", "items": source_record},
            "fieldStatusCounts": {
                "type": "object",
                "additionalProperties": {"type": "integer", "minimum": 0},
            },
            "unavailableFields": {
                "type": "array", "items": {"type": "string"}, "uniqueItems": True
            },
            "ambiguousFields": {
                "type": "array", "items": {"type": "string"}, "uniqueItems": True
            },
            "stop": _strict_object(
                {
                    "reason": nullable_string,
                    "terminalEventIndex": _nullable({"type": "integer", "minimum": 0}),
                }
            ),
            "sourceRunSummary": {"type": "object"},
        },
    }
