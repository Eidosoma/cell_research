"""JSON Schema, identity, availability, and trace-chain validation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .model import TraceBundle
from .schema import (
    EVENT_SCHEMA_VERSION,
    HISTORICAL_ALWAYS_LOSSY_FIELDS,
    TRACKED_FIELDS,
    TRACE_MANIFEST_SCHEMA_VERSION,
    event_content_hash,
    event_schema,
    sha256_json,
    trace_content_hash,
    trace_manifest_schema,
)


class UnsupportedSchemaVersion(ValueError):
    pass


def _get_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def _schema_errors(validator: Draft202012Validator, value: Mapping[str, Any]) -> list[str]:
    return [
        f"{'/'.join(str(item) for item in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(value), key=lambda item: list(item.absolute_path))
    ]


def validate_event(event: Mapping[str, Any]) -> None:
    if event.get("schemaVersion") != EVENT_SCHEMA_VERSION:
        raise UnsupportedSchemaVersion(
            f"unsupported event schemaVersion {event.get('schemaVersion')!r}; "
            f"expected {EVENT_SCHEMA_VERSION!r}"
        )
    errors = _schema_errors(Draft202012Validator(event_schema()), event)
    if errors:
        raise ValueError("event JSON Schema validation failed: " + "; ".join(errors))
    expected_id = "sha256:" + event_content_hash(event)
    if event["eventId"] != expected_id:
        raise ValueError("eventId does not match canonical event content")
    availability = event["fieldAvailability"]
    if set(availability) != set(TRACKED_FIELDS):
        raise ValueError("fieldAvailability keys do not exactly match TRACKED_FIELDS")
    for path in TRACKED_FIELDS:
        status = availability[path]["status"]
        value = _get_path(event, path)
        if status in {"unavailable", "derived_ambiguous", "not_applicable"} and value is not None:
            raise ValueError(f"{path} is {status} but carries a non-null value")
        if status in {"observed", "derived_exact"} and value is None:
            raise ValueError(f"{path} is {status} but carries null")
    backend_id = event["backend"]["backendId"]
    sequence_basis = event["run"]["sequenceBasis"]
    expected_sequence_basis = (
        "recorded_swap" if backend_id == "historical_frozen_public_commit" else "activation"
    )
    if sequence_basis != expected_sequence_basis:
        raise ValueError(
            f"{backend_id} requires sequenceBasis={expected_sequence_basis}, "
            f"not {sequence_basis}"
        )
    if backend_id == "historical_frozen_public_commit":
        violations = [
            path for path in HISTORICAL_ALWAYS_LOSSY_FIELDS
            if availability[path]["status"] in {"observed", "derived_exact"}
        ]
        if violations:
            raise ValueError("historical adapter manufactured parity for: " + ", ".join(violations))


def _validate_batch_chains(events: Sequence[Mapping[str, Any]]) -> None:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    group_order: list[str] = []
    for event in events:
        batch_id = event["run"]["batch"]["batchId"]
        if batch_id not in groups:
            group_order.append(batch_id)
        groups[batch_id].append(event)
    previous_native: str | None = None
    previous_observable: str | None = None
    for batch_id in group_order:
        group = groups[batch_id]
        widths = {event["run"]["batch"]["width"] for event in group}
        ordinals = [event["run"]["batch"]["ordinal"] for event in group]
        if len(widths) != 1 or len(group) != next(iter(widths)):
            raise ValueError(f"batch {batch_id} does not match declared width")
        if ordinals != list(range(len(group))):
            raise ValueError(f"batch {batch_id} ordinals are not contiguous")
        native_pre = {event["state"]["nativePreHash"] for event in group}
        native_post = {event["state"]["nativePostHash"] for event in group}
        observable_pre = {event["state"]["observablePreHash"] for event in group}
        observable_post = {event["state"]["observablePostHash"] for event in group}
        if len(native_pre) != 1 or len(native_post) != 1:
            raise ValueError(f"batch {batch_id} has inconsistent native hashes")
        if len(observable_pre) != 1 or len(observable_post) != 1:
            raise ValueError(f"batch {batch_id} has inconsistent observable hashes")
        current_native_pre = next(iter(native_pre))
        current_native_post = next(iter(native_post))
        current_observable_pre = next(iter(observable_pre))
        current_observable_post = next(iter(observable_post))
        if previous_native is not None and current_native_pre is not None and current_native_pre != previous_native:
            raise ValueError(f"native post/pre hash chain breaks before batch {batch_id}")
        if previous_observable is not None and current_observable_pre != previous_observable:
            raise ValueError(f"observable post/pre hash chain breaks before batch {batch_id}")
        previous_native = current_native_post
        previous_observable = current_observable_post


def validate_manifest(manifest: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> None:
    if manifest.get("schemaVersion") != TRACE_MANIFEST_SCHEMA_VERSION:
        raise UnsupportedSchemaVersion(
            f"unsupported trace manifest schemaVersion {manifest.get('schemaVersion')!r}"
        )
    errors = _schema_errors(Draft202012Validator(trace_manifest_schema()), manifest)
    if errors:
        raise ValueError("trace manifest JSON Schema validation failed: " + "; ".join(errors))
    if manifest["eventCount"] != len(events):
        raise ValueError("trace manifest eventCount mismatch")
    if manifest["traceContentSha256"] != trace_content_hash(events):
        raise ValueError("trace manifest content hash mismatch")
    identity_material = dict(manifest)
    identity_material.pop("traceId")
    if manifest["traceId"] != "sha256:" + sha256_json(identity_material):
        raise ValueError("traceId does not match canonical manifest content")


def validate_trace(
    bundle: TraceBundle, manifest: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    events = list(bundle.events)
    for event in events:
        validate_event(event)
    indices = [event["run"]["eventIndex"] for event in events]
    if indices != list(range(len(events))):
        raise ValueError("event indices must be contiguous and zero-based")
    for event in events:
        if event["run"]["runId"] != bundle.run_id:
            raise ValueError("event runId differs from bundle")
        if event["run"]["scenarioId"] != bundle.scenario_id:
            raise ValueError("event scenarioId differs from bundle")
        if event["run"]["sequenceBasis"] != bundle.sequence_basis:
            raise ValueError("event sequenceBasis differs from bundle")
        if event["backend"]["backendId"] != bundle.backend:
            raise ValueError("event backend differs from bundle")
    if events:
        _validate_batch_chains(events)
        terminal_batches = {
            event["run"]["batch"]["batchId"]
            for event in events if event["stop"]["terminal"]
        }
        if terminal_batches and terminal_batches != {events[-1]["run"]["batch"]["batchId"]}:
            raise ValueError("terminal metadata appears before the final batch")
        if bundle.stop_reason and not events[-1]["stop"]["terminal"]:
            raise ValueError("nonempty trace does not mark its final batch terminal")
    selected_manifest = dict(manifest) if manifest is not None else bundle.base_manifest()
    validate_manifest(selected_manifest, events)
    return {
        "success": True,
        "eventCount": len(events),
        "eventSchemaVersion": EVENT_SCHEMA_VERSION,
        "manifestSchemaVersion": TRACE_MANIFEST_SCHEMA_VERSION,
        "traceContentSha256": trace_content_hash(events),
        "requiredFieldChecks": len(events) * len(TRACKED_FIELDS),
        "hashChainsValidated": bool(events),
    }
