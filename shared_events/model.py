"""In-memory trace bundle and manifest construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import (
    ADAPTER_VERSION,
    EVENT_SCHEMA_VERSION,
    TRACE_MANIFEST_SCHEMA_VERSION,
    TRACE_SCHEMA_URN,
    sha256_json,
    trace_content_hash,
)


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    path: str
    sha256: str
    classification: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "classification": self.classification,
        }


@dataclass(frozen=True, slots=True)
class TraceBundle:
    events: tuple[Mapping[str, Any], ...]
    run_id: str
    scenario_id: str | None
    backend: str
    sequence_basis: str
    source_artifacts: tuple[SourceArtifact, ...]
    source_run_summary: Mapping[str, Any]
    stop_reason: str | None

    def base_manifest(self) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        unavailable: set[str] = set()
        ambiguous: set[str] = set()
        for event in self.events:
            for path, record in event["fieldAvailability"].items():
                status = record["status"]
                status_counts[status] = status_counts.get(status, 0) + 1
                if status == "unavailable":
                    unavailable.add(path)
                elif status == "derived_ambiguous":
                    ambiguous.add(path)
        content_hash = trace_content_hash(self.events)
        terminal_indices = [
            event["run"]["eventIndex"]
            for event in self.events
            if event["stop"]["terminal"] is True
        ]
        body = {
            "schemaVersion": TRACE_MANIFEST_SCHEMA_VERSION,
            "schemaUri": TRACE_SCHEMA_URN,
            "eventSchemaVersion": EVENT_SCHEMA_VERSION,
            "adapterVersion": ADAPTER_VERSION,
            "runId": self.run_id,
            "scenarioId": self.scenario_id,
            "backend": self.backend,
            "sequenceBasis": self.sequence_basis,
            "eventCount": len(self.events),
            "traceContentSha256": content_hash,
            "formats": [],
            "sourceArtifacts": [item.to_dict() for item in self.source_artifacts],
            "fieldStatusCounts": dict(sorted(status_counts.items())),
            "unavailableFields": sorted(unavailable),
            "ambiguousFields": sorted(ambiguous),
            "stop": {
                "reason": self.stop_reason,
                "terminalEventIndex": terminal_indices[-1] if terminal_indices else None,
            },
            "sourceRunSummary": dict(self.source_run_summary),
        }
        body["traceId"] = "sha256:" + sha256_json(body)
        return body


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_artifact(path: Path, classification: str) -> SourceArtifact:
    return SourceArtifact(str(path), sha256_file(path), classification)


def bundle_with_events(bundle: TraceBundle, events: Sequence[Mapping[str, Any]]) -> TraceBundle:
    return TraceBundle(
        events=tuple(events),
        run_id=bundle.run_id,
        scenario_id=bundle.scenario_id,
        backend=bundle.backend,
        sequence_basis=bundle.sequence_basis,
        source_artifacts=bundle.source_artifacts,
        source_run_summary=bundle.source_run_summary,
        stop_reason=bundle.stop_reason,
    )
