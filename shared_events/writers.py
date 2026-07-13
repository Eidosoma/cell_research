"""Deterministic Zstandard JSONL and Parquet trace encodings."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .model import TraceBundle, sha256_file
from .schema import canonical_json_bytes, sha256_json
from .validation import validate_manifest, validate_trace


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl_zstd(events: Sequence[Mapping[str, Any]], path: Path) -> None:
    payload = b"".join(canonical_json_bytes(event) + b"\n" for event in events)

    def write(temporary: Path) -> None:
        with pa.OSFile(str(temporary), "wb") as raw:
            with pa.CompressedOutputStream(raw, "zstd") as compressed:
                compressed.write(payload)

    _atomic_write(path, write)


def read_jsonl_zstd(path: Path) -> list[dict[str, Any]]:
    with pa.memory_map(str(path), "r") as raw:
        with pa.CompressedInputStream(raw, "zstd") as compressed:
            payload = compressed.read()
    if not payload:
        return []
    if not payload.endswith(b"\n"):
        raise ValueError("canonical JSONL trace must end with a newline")
    return [json.loads(line) for line in payload.splitlines()]


def _parquet_rows(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": event["schemaVersion"],
            "event_id": event["eventId"],
            "backend_id": event["backend"]["backendId"],
            "run_id": event["run"]["runId"],
            "scenario_id": event["run"]["scenarioId"],
            "event_index": event["run"]["eventIndex"],
            "sequence_basis": event["run"]["sequenceBasis"],
            "batch_id": event["run"]["batch"]["batchId"],
            "batch_width": event["run"]["batch"]["width"],
            "batch_ordinal": event["run"]["batch"]["ordinal"],
            "actor_id": event["actor"]["id"],
            "actor_algotype": event["actor"]["algotype"],
            "proposal_kind": event["proposal"]["kind"],
            "decision_status": event["decision"]["status"],
            "accepted": event["decision"]["accepted"],
            "native_pre_hash": event["state"]["nativePreHash"],
            "native_post_hash": event["state"]["nativePostHash"],
            "observable_pre_hash": event["state"]["observablePreHash"],
            "observable_post_hash": event["state"]["observablePostHash"],
            "stop_reason": event["stop"]["reason"],
            "event_json": canonical_json_bytes(event).decode("utf-8"),
        }
        for event in events
    ]


PARQUET_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("backend_id", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("scenario_id", pa.string()),
        pa.field("event_index", pa.int64()),
        pa.field("sequence_basis", pa.string(), nullable=False),
        pa.field("batch_id", pa.string(), nullable=False),
        pa.field("batch_width", pa.int32(), nullable=False),
        pa.field("batch_ordinal", pa.int32(), nullable=False),
        pa.field("actor_id", pa.string()),
        pa.field("actor_algotype", pa.string()),
        pa.field("proposal_kind", pa.string()),
        pa.field("decision_status", pa.string()),
        pa.field("accepted", pa.bool_()),
        pa.field("native_pre_hash", pa.string()),
        pa.field("native_post_hash", pa.string()),
        pa.field("observable_pre_hash", pa.string()),
        pa.field("observable_post_hash", pa.string()),
        pa.field("stop_reason", pa.string()),
        pa.field("event_json", pa.string(), nullable=False),
    ],
    metadata={
        b"e01.encoding": b"canonical event_json plus searchable projection",
        b"e01.compression": b"zstd",
    },
)


def write_parquet_zstd(events: Sequence[Mapping[str, Any]], path: Path) -> None:
    table = pa.Table.from_pylist(_parquet_rows(events), schema=PARQUET_SCHEMA)

    def write(temporary: Path) -> None:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=9,
            use_dictionary=False,
            write_statistics=True,
            version="2.6",
            data_page_version="1.0",
        )

    _atomic_write(path, write)


def read_parquet_zstd(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["event_json"])
    return [json.loads(value) for value in table.column("event_json").to_pylist()]


def _format_record(path: Path, format_name: str) -> dict[str, Any]:
    return {
        "format": format_name,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "compression": "zstd",
    }


def _assign_manifest_id(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    result.pop("traceId", None)
    result["traceId"] = "sha256:" + sha256_json(result)
    return result


def write_trace_bundle(bundle: TraceBundle, output_prefix: Path) -> dict[str, Any]:
    """Validate, write both encodings, round-trip, and return/write a manifest."""
    validate_trace(bundle)
    jsonl_path = output_prefix.with_suffix(".jsonl.zst")
    parquet_path = output_prefix.with_suffix(".parquet")
    manifest_path = output_prefix.with_suffix(".manifest.json")
    write_jsonl_zstd(bundle.events, jsonl_path)
    write_parquet_zstd(bundle.events, parquet_path)
    jsonl_events = read_jsonl_zstd(jsonl_path)
    parquet_events = read_parquet_zstd(parquet_path)
    expected = [dict(event) for event in bundle.events]
    if jsonl_events != expected:
        raise ValueError("JSONL Zstandard deterministic round-trip mismatch")
    if parquet_events != expected:
        raise ValueError("Parquet Zstandard deterministic round-trip mismatch")
    manifest = bundle.base_manifest()
    manifest["formats"] = [
        _format_record(jsonl_path, "jsonl.zst"),
        _format_record(parquet_path, "parquet.zstd"),
    ]
    manifest = _assign_manifest_id(manifest)
    validate_manifest(manifest, bundle.events)
    _atomic_write(
        manifest_path,
        lambda temporary: temporary.write_bytes(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
        ),
    )
    return manifest
