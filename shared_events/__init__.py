"""Versioned shared event/provenance records for E01 backends."""

from .adapters import adapt_historical_run, adapt_reference_result
from .schema import (
    ADAPTER_VERSION,
    EVENT_SCHEMA_VERSION,
    TRACE_MANIFEST_SCHEMA_VERSION,
    event_schema,
    trace_manifest_schema,
)
from .validation import validate_event, validate_trace
from .writers import read_jsonl_zstd, read_parquet_zstd, write_trace_bundle

__all__ = [
    "ADAPTER_VERSION",
    "EVENT_SCHEMA_VERSION",
    "TRACE_MANIFEST_SCHEMA_VERSION",
    "adapt_historical_run",
    "adapt_reference_result",
    "event_schema",
    "read_jsonl_zstd",
    "read_parquet_zstd",
    "trace_manifest_schema",
    "validate_event",
    "validate_trace",
    "write_trace_bundle",
]
