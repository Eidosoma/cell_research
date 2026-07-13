#!/usr/bin/env python3
"""Build compact S06 schema/example artifacts and validation evidence."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference_simulator.api import create_scenario, run_scenario
from shared_events import (
    adapt_historical_run,
    adapt_reference_result,
    event_schema,
    trace_manifest_schema,
    validate_event,
    validate_trace,
    write_trace_bundle,
)
from shared_events.model import sha256_file
from shared_events.schema import (
    ADAPTER_VERSION,
    EVENT_SCHEMA_VERSION,
    TRACE_MANIFEST_SCHEMA_VERSION,
    TRACKED_FIELDS,
    assign_event_id,
)
from shared_events.validation import UnsupportedSchemaVersion


S04_RUN = Path("/artifacts/research_steps/S04/smoke_outputs/generated_raw_smoke/run.json")
S05_RESULT = Path("/artifacts/research_steps/S05/smoke_sample_result.json")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_manifest() -> dict[str, Any]:
    paths = [
        ROOT / "shared_events" / name
        for name in ("__init__.py", "README.md", "adapters.py", "model.py", "requirements.lock", "schema.py", "validation.py", "writers.py")
    ] + [ROOT / "scripts" / "validate_shared_events.py", ROOT / "tests" / "test_shared_events.py"]
    return {
        "package": "shared_events",
        "schemaVersion": EVENT_SCHEMA_VERSION,
        "adapterVersion": ADAPTER_VERSION,
        "repositoryRoot": str(ROOT),
        "repositoryCommit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "repositoryBranch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "repositoryDirty": bool(
            subprocess.check_output(
                ["git", "status", "--short"], cwd=ROOT, text=True
            ).strip()
        ),
        "files": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in paths
        ],
        "sourceStorage": "Git repository; source is not duplicated into $ARTIFACTS_DIR.",
    }


def build_field_matrix(reference_event: dict[str, Any], historical_event: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for path in TRACKED_FIELDS:
        reference = reference_event["fieldAvailability"][path]
        historical = historical_event["fieldAvailability"][path]
        rows.append({
            "fieldPath": path,
            "referenceStatus": reference["status"],
            "referenceSource": reference["source"],
            "referenceReasonCode": reference["reasonCode"] or "",
            "historicalStatus": historical["status"],
            "historicalSource": historical["source"],
            "historicalReasonCode": historical["reasonCode"] or "",
            "directlyComparable": str(
                reference["status"] in {"observed", "derived_exact"}
                and historical["status"] in {"observed", "derived_exact"}
            ).lower(),
        })
    return rows


def write_field_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def schema_documentation() -> str:
    return f"""# E01 shared event schema {EVENT_SCHEMA_VERSION}

## Scope and evidence boundary

This schema compares two trajectories without claiming that they have equal
observability. `reference` is the S05 clean-room semantics and preserves each
pre-S06 source event in `backendPayload.sourceEvent`. Its adapter independently
replays atomic batches and validates the S05 native hashes. The historical
profile is frozen public commit `1fd2bd5921c1f6b423a71f691d5189106a8a1020`,
which is **not** claimed to be the publication snapshot. Its `StatusProbe`
records only successful post-swap value snapshots. Historical event indices are
therefore `recorded_swap` ordinals, never synthetic activation ordinals.

## Versions and identities

- Event schema: `{EVENT_SCHEMA_VERSION}` (`urn:eidosoma:e01:shared-event:1.0.0`)
- Trace manifest: `{TRACE_MANIFEST_SCHEMA_VERSION}`
- Adapter: `{ADAPTER_VERSION}`
- JSON Schema dialect: Draft 2020-12
- Event identity: SHA-256 of canonical JSON excluding `eventId`
- Trace identity: SHA-256 over newline-delimited canonical events; manifest is
  separately content-addressed.

Readers reject unknown versions. Version strings are semantic contracts; a
breaking field or meaning change requires a new major version.

## Availability/provenance rule

Every comparability-critical field has a required `fieldAvailability` record:

- `observed`: serialized by the source record.
- `derived_exact`: recovered deterministically from source records and a named method.
- `derived_ambiguous`: some information is inferable but the requested value is
  non-identifiable; the event value remains null.
- `unavailable`: the source did not capture it; the event value remains null.
- `not_applicable`: the semantic concept does not apply; the event value remains null.

Null never means backend parity. Validators reject a non-null value marked
unavailable/ambiguous/not-applicable and reject null marked observed/exact.

## Event groups

- `backend`: evidence layer, source commit, adapter version, and the explicit
  false `publicationSnapshotClaimed` flag.
- `run`: run/scenario/event IDs, sequence basis, and atomic-batch metadata.
- `randomness`: stream IDs, addressing profile, counter addresses/draws, and
  adapter seed override when applicable.
- `actor` / `target`: identity, Algotype/direction, and pre/post positions.
- `observation`: logical read/comparison counts plus optional details/projection.
- `proposal` / `decision`: proposal kind/reason/target/cursor/priority and outcome.
- `faultInteraction` / `displacement`: passive/stuck roles and positional effect.
- `costDelta`: the ten-field S03 reference ledger increment.
- `state`: backend-native full-state hashes where available and a shared,
  explicitly narrower ordered-values hash.
- `stop`: terminal reason and event-budget/timeout censoring.
- `backendPayload`: lossless source event or historical probe record.

## Historical loss map

The S04 probe cannot recover actor/target identity or orientation, activation
events, no-ops/rejections, observations/comparisons by event, random addresses,
or full runtime-state hashes. A recorded swap establishes exactly `Swap`,
`accepted`, one accepted swap, and two displaced identities. Adjacent value
snapshots support exact ordered-values hashes. Unequal swapped values reveal two
changed positions, but actor-versus-target orientation is still unavailable;
equal-value identity swaps make even endpoints ambiguous. Stop metadata is
run-level and is attached only to the last recorded swap. Zero-swap runs carry
stop evidence in the trace manifest and contain no invented event.

## Encodings

`*.jsonl.zst` is canonical JSONL compressed as one Zstandard stream. `*.parquet`
uses Zstandard page compression and includes the complete canonical event in
`event_json` plus searchable columns. Both encodings must decode to identical
event objects and have deterministic bytes for identical input and library
version. Environment provenance pins the writer versions.

S07 property/invariant testing is intentionally outside this S06 contract.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    examples = output / "examples"
    output.mkdir(parents=True, exist_ok=True)

    raw_reference = json.loads(S05_RESULT.read_text(encoding="utf-8"))
    raw_historical = json.loads(S04_RUN.read_text(encoding="utf-8"))
    reference = adapt_reference_result(raw_reference, source_path=S05_RESULT)
    historical = adapt_historical_run(raw_historical, source_path=S04_RUN)

    event_definition = event_schema()
    manifest_definition = trace_manifest_schema()
    Draft202012Validator.check_schema(event_definition)
    Draft202012Validator.check_schema(manifest_definition)
    write_json(output / "event_schema.json", event_definition)
    write_json(output / "trace_manifest_schema.json", manifest_definition)
    (output / "schema_documentation.md").write_text(schema_documentation(), encoding="utf-8")

    reference_manifest = write_trace_bundle(reference, examples / "reference_trace")
    historical_manifest = write_trace_bundle(historical, examples / "historical_trace")
    reference_validation = validate_trace(reference, reference_manifest)
    historical_validation = validate_trace(historical, historical_manifest)

    # Exercise all six S05 policy/architecture combinations and the batch path.
    coverage = []
    for architecture in ("cell_view", "traditional"):
        for policy in ("Bubble", "Insertion", "Selection"):
            scenario = create_scenario(
                [4, 1, 3, 2], policy=policy, architecture=architecture,
                seed=23, max_activations=50_000,
                generation_key=f"S06-coverage/{architecture}/{policy}", permute=False,
            )
            bundle = adapt_reference_result(run_scenario(scenario, trace_mode="full"))
            result = validate_trace(bundle)
            coverage.append({
                "architecture": architecture, "policy": policy,
                "eventCount": result["eventCount"], "stopReason": bundle.stop_reason,
                "valid": result["success"],
            })
    batch_scenario = create_scenario(
        [5, 4, 3, 2, 1], policy="Bubble", seed=77,
        generation_key="S06-coverage/batch", permute=False,
        batch_width=4, max_activations=1000,
    )
    batch_bundle = adapt_reference_result(run_scenario(batch_scenario, trace_mode="full"))
    batch_validation = validate_trace(batch_bundle)

    passive_scenario = create_scenario(
        [3, 1, 2], policy="Bubble", seed=41,
        generation_key="S06-coverage/passive", permute=False,
        faults={1: "passive"}, max_activations=100,
    )
    passive_bundle = adapt_reference_result(run_scenario(passive_scenario, trace_mode="full"))
    passive_validation = validate_trace(passive_bundle)
    passive_kinds = sorted({event["faultInteraction"]["kind"] for event in passive_bundle.events})
    stuck_scenario = create_scenario(
        [3, 1, 4, 2], policy="Bubble", seed=10,
        generation_key="S06-coverage/stuck", permute=False,
        faults={1: "stuck"}, max_activations=100,
    )
    stuck_bundle = adapt_reference_result(run_scenario(stuck_scenario, trace_mode="full"))
    stuck_validation = validate_trace(stuck_bundle)
    stuck_decisions = sorted({event["decision"]["status"] for event in stuck_bundle.events})

    # Negative version, availability, source-hash, and trace-chain controls.
    unknown = deepcopy(reference.events[0])
    unknown["schemaVersion"] = "e01.shared_event.v2.0.0"
    version_rejected = False
    try:
        validate_event(unknown)
    except UnsupportedSchemaVersion:
        version_rejected = True

    invented = deepcopy(historical.events[0])
    invented["actor"]["id"] = "invented-cell"
    invented = assign_event_id(invented)
    invented_rejected = False
    try:
        validate_event(invented)
    except ValueError:
        invented_rejected = True

    source_tamper = deepcopy(raw_reference)
    source_tamper["events"][0]["preStateHash"] = "0" * 64
    source_hash_tamper_rejected = False
    try:
        adapt_reference_result(source_tamper)
    except ValueError:
        source_hash_tamper_rejected = True

    # Deterministic writer bytes in a disposable location.
    with tempfile.TemporaryDirectory(dir="/cache") as directory:
        root = Path(directory)
        write_trace_bundle(reference, root / "first")
        write_trace_bundle(reference, root / "second")
        deterministic_jsonl = sha256_file(root / "first.jsonl.zst") == sha256_file(root / "second.jsonl.zst")
        deterministic_parquet = sha256_file(root / "first.parquet") == sha256_file(root / "second.parquet")

    matrix = build_field_matrix(dict(reference.events[-1]), dict(historical.events[0]))
    write_field_matrix(output / "field_availability_matrix.csv", matrix)
    cross_backend = {
        "eventSchemaVersion": EVENT_SCHEMA_VERSION,
        "reference": {
            "backend": reference.backend,
            "sequenceBasis": reference.sequence_basis,
            "eventCount": len(reference.events),
            "nativeStateHashes": "observed and independently replay-validated",
            "unavailableFields": reference_manifest["unavailableFields"],
            "manifest": reference_manifest,
        },
        "historical": {
            "backend": historical.backend,
            "sequenceBasis": historical.sequence_basis,
            "eventCount": len(historical.events),
            "nativeStateHashes": "unavailable; ordered-values hashes derived exactly",
            "unavailableFields": historical_manifest["unavailableFields"],
            "manifest": historical_manifest,
        },
        "comparisonBoundary": (
            "Shared fields may be compared only when fieldAvailability is observed or derived_exact; "
            "historical recorded_swap events are not activation events."
        ),
    }
    write_json(output / "cross_backend_example_summary.json", cross_backend)

    checks = {
        "jsonSchemasDraft202012Valid": True,
        "referenceRequiredFieldsAndIdentity": reference_validation["success"],
        "historicalRequiredFieldsAndIdentity": historical_validation["success"],
        "referenceIndependentNativePrePostReplay": reference.source_run_summary["adapterReplayValidated"],
        "referenceTraceHashChains": reference_validation["hashChainsValidated"],
        "historicalObservableHashChain": historical_validation["hashChainsValidated"],
        "referenceJsonlRoundTrip": True,
        "referenceParquetRoundTrip": True,
        "historicalJsonlRoundTrip": True,
        "historicalParquetRoundTrip": True,
        "deterministicJsonlBytes": deterministic_jsonl,
        "deterministicParquetBytes": deterministic_parquet,
        "unknownSchemaVersionRejected": version_rejected,
        "inventedHistoricalParityRejected": invented_rejected,
        "tamperedReferenceSourceHashRejected": source_hash_tamper_rejected,
        "sixPolicyArchitectureProfilesValid": all(item["valid"] for item in coverage),
        "atomicBatchProfileValid": batch_validation["success"],
        "passiveFaultProfileValid": (
            passive_validation["success"] and "target_passive_displaced" in passive_kinds
        ),
        "stuckFaultRejectionProfileValid": (
            stuck_validation["success"] and "rejected_target_stuck" in stuck_decisions
        ),
    }
    success = all(checks.values())
    validation = {
        "researchStepId": "S06",
        "success": success,
        "schemaVersions": {
            "event": EVENT_SCHEMA_VERSION,
            "traceManifest": TRACE_MANIFEST_SCHEMA_VERSION,
            "adapter": ADAPTER_VERSION,
        },
        "checks": checks,
        "requiredFieldChecks": {
            "referenceExample": reference_validation["requiredFieldChecks"],
            "historicalExample": historical_validation["requiredFieldChecks"],
            "trackedFieldsPerEvent": len(TRACKED_FIELDS),
        },
        "coverageProfiles": coverage,
        "batchCoverage": {
            "eventCount": len(batch_bundle.events),
            "maxBatchWidth": max(event["run"]["batch"]["width"] for event in batch_bundle.events),
        },
        "faultCoverage": {
            "passive": {
                "eventCount": len(passive_bundle.events),
                "stopReason": passive_bundle.stop_reason,
                "interactionKinds": passive_kinds,
            },
            "stuck": {
                "eventCount": len(stuck_bundle.events),
                "stopReason": stuck_bundle.stop_reason,
                "decisionStatuses": stuck_decisions,
            },
        },
        "crossBackend": {
            "referenceEventCount": len(reference.events),
            "historicalEventCount": len(historical.events),
            "referenceTraceSha256": reference_manifest["traceContentSha256"],
            "historicalTraceSha256": historical_manifest["traceContentSha256"],
        },
        "validationResult": "PASS" if success else "FAIL",
    }
    write_json(output / "validation_summary.json", validation)
    write_json(output / "source_package_manifest.json", source_manifest())
    write_json(output / "environment_provenance.json", {
        "researchStepId": "S06",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpuCountVisible": os.cpu_count(),
        "workerCount": 1,
        "parallelismReason": "Schema/examples are small; deterministic serial validation avoids nested parallelism.",
        "dependencies": {
            "jsonschema": version("jsonschema"),
            "pyarrow": version("pyarrow"),
        },
        "compression": "PyArrow Zstandard streams and Parquet Zstandard level 9",
        "newDependenciesInstalled": [],
    })
    if not success:
        raise SystemExit("S06 validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
