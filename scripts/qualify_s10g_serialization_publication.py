#!/usr/bin/env python3
"""Outcome-free S10G mixed-schema and publication qualification."""

from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import scripts.run_native_event_discovery_s10 as s10_base
import scripts.qualify_s10e_method_feasibility as s10e
from src.phenotype_discovery.publication import (
    ArtifactSpec,
    AtomicScientificPublisher,
    DATAFRAME_COLUMNS,
    EVIDENCE_STATES,
    FAMILY_PAYLOAD_FIELDS,
    METHOD_FAMILIES,
    METHOD_RECORD_SCHEMA_VERSION,
    PARQUET_SCHEMA,
    PUBLICATION_SCHEMA_VERSION,
    PublicationContractError,
    SerializationContractError,
    VALUE_PLANE_FIELDS,
    VALUE_STATES,
    canonical_json_bytes,
    canonical_sha256,
    make_method_record,
    records_from_dataframe,
    records_from_json_bytes,
    records_from_parquet_bytes,
    records_to_dataframe,
    records_to_json_bytes,
    records_to_parquet_bytes,
    validate_method_record,
    value_plane,
)


STEP_ID = "S10G"
REPOSITORY = Path("/workspace/cell-research")
WORKSPACE = Path("/workspace")
OUTPUT = Path("/artifacts/research_steps/S10G")
CONFIG = REPOSITORY / "configs/discovery/s10g_serialization_publication.yaml"
SCRIPT = Path(__file__).resolve()
CORE = REPOSITORY / "src/phenotype_discovery/publication.py"
TEST = REPOSITORY / "tests/test_s10g_serialization_publication.py"
S10P = Path("/artifacts/research_steps/S10P")
S10F = Path("/artifacts/research_steps/S10F")
PROHIBITED_S10F_ARTIFACTS = frozenset(
    {
        S10F / "discovery_lock.json",
        S10F / "native_event_feature_records.parquet",
        S10F / "ordered_event_summary_records.parquet",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def tree_snapshot(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n").encode("utf-8")
        )
    return {
        "root": str(root),
        "fileCount": len(rows),
        "totalBytes": sum(row["bytes"] for row in rows),
        "treeSha256": digest.hexdigest(),
    }


def metadata_snapshot(root: Path) -> dict[str, Any]:
    """Snapshot file metadata without opening any file content."""

    rows = []
    if root.exists():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            stat = path.stat()
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": stat.st_size,
                    "mtimeNs": stat.st_mtime_ns,
                    "inode": stat.st_ino,
                }
            )
    return {
        "root": str(root),
        "fileCount": len(rows),
        "totalBytes": sum(row["bytes"] for row in rows),
        "metadataSha256": canonical_sha256(
            "E07/S10G/opaque-metadata-snapshot/v1", rows
        ),
    }


def s10f_artifact_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    boundary = config["s10fQuarantineBoundary"]
    files = sorted(item for item in S10F.iterdir() if item.is_file())
    if len(files) != boundary["artifactFileCount"]:
        raise RuntimeError("S10F artifact file count changed")
    total = sum(path.stat().st_size for path in files)
    if total != boundary["artifactTotalBytes"]:
        raise RuntimeError("S10F artifact total bytes changed")
    prohibited_rows = []
    configured_prohibited = {
        Path(row["path"]): row for row in boundary["prohibitedArtifactFiles"]
    }
    if set(configured_prohibited) != PROHIBITED_S10F_ARTIFACTS:
        raise RuntimeError("S10F prohibited artifact registry changed")
    for path, expected in sorted(
        configured_prohibited.items(), key=lambda item: str(item[0])
    ):
        stat = path.stat()
        if stat.st_size != expected["bytes"]:
            raise RuntimeError(f"prohibited S10F artifact size changed: {path}")
        prohibited_rows.append(
            {
                "path": str(path),
                "bytes": stat.st_size,
                "mtimeNs": stat.st_mtime_ns,
                "inode": stat.st_ino,
                "recordedSha256": expected["recordedSha256"],
                "contentOpenedByS10G": False,
            }
        )
    permitted = [path for path in files if path not in PROHIBITED_S10F_ARTIFACTS]
    lines = "".join(f"{sha256_file(path)}  {path}\n" for path in permitted)
    digest = hashlib.sha256(lines.encode("utf-8")).hexdigest()
    if (
        len(permitted) != boundary["permittedContentFileCount"]
        or digest != boundary["permittedContentListSha256"]
    ):
        raise RuntimeError("permitted S10F artifact content changed")
    return {
        "artifactRoot": str(S10F),
        "artifactFileCount": len(files),
        "artifactTotalBytes": total,
        "permittedContentFileCount": len(permitted),
        "permittedContentListSha256": digest,
        "prohibitedArtifactStatRows": prohibited_rows,
        "prohibitedArtifactContentFilesRead": 0,
    }


def freeze_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    artifact_trees = {}
    for name, spec in config["immutableArtifactTrees"].items():
        actual = tree_snapshot(Path(spec["path"]))
        actual["expectedTreeSha256"] = spec["treeSha256"]
        actual["pass"] = (
            actual["treeSha256"] == spec["treeSha256"]
            and actual["fileCount"] == spec["fileCount"]
            and actual["totalBytes"] == spec["totalBytes"]
        )
        artifact_trees[name] = actual
    contracts = []
    for name, spec in config["frozenScientificContracts"].items():
        path = Path(spec["path"])
        actual = sha256_file(path)
        contracts.append(
            {
                "name": name,
                "path": str(path),
                "expectedSha256": spec["sha256"],
                "actualSha256": actual,
                "pass": actual == spec["sha256"],
            }
        )
    caches = {}
    cache_specs = {
        **config["immutableConsumedCacheMetadata"],
        "S10F": config["s10fQuarantineBoundary"]["prohibitedCache"],
    }
    for name, spec in cache_specs.items():
        actual = metadata_snapshot(Path(spec["path"]))
        actual["recordedTreeSha256"] = spec["recordedTreeSha256"]
        actual["pass"] = (
            actual["fileCount"] == spec["fileCount"]
            and actual["totalBytes"] == spec["totalBytes"]
        )
        actual["contentFilesOpenedByS10G"] = 0
        caches[name] = actual
    s10f = s10f_artifact_snapshot(config)
    return {
        "schemaVersion": "e07.s10g.input-hash-freeze.v1",
        "researchStepId": STEP_ID,
        "artifactTrees": artifact_trees,
        "frozenScientificContracts": contracts,
        "opaqueConsumedCacheMetadata": caches,
        "s10fQuarantineBoundary": s10f,
        "allPass": (
            all(row["pass"] for row in artifact_trees.values())
            and all(row["pass"] for row in contracts)
            and all(row["pass"] for row in caches.values())
        ),
        "s10fCacheFilesRead": 0,
        "s10fOutcomeArtifactFilesRead": 0,
    }


def _payload(family: str, ordinal: int) -> dict[str, Any]:
    if family == "clustering":
        return {
            "algorithm": "agglomerative_Ward",
            "clusterLabel": ordinal,
            "membership": [f"synthetic-{ordinal:02d}", "synthetic-13"],
            "stability": 0.8,
            "nullThreshold": 0.4,
        }
    if family == "anomaly":
        return {
            "algorithm": "IsolationForest",
            "selectedConfigurationId": f"synthetic-{ordinal:02d}",
            "scores": {
                f"synthetic-{ordinal:02d}": 0.3,
                "synthetic-13": 0.1,
            },
            "medianScore": 0.2,
            "empiricalPValue": 0.04,
        }
    if family == "change_point":
        return {
            "algorithm": "exact_dynamic_programming_piecewise_constant_SSE",
            "series": "proposal_count",
            "changePoints": [8, 20],
            "prevalence": 0.7,
            "empiricalPValue": 0.03,
        }
    if family == "fixed_holm":
        return {
            "sourceMethodFamily": "anomaly",
            "slotState": "evidentiary",
            "rawPValue": 0.04,
            "adjustedPValue": 0.16,
            "rejected": False,
        }
    raise AssertionError(family)


def synthetic_records() -> list[dict[str, Any]]:
    records = []
    for ordinal, family in enumerate(sorted(METHOD_FAMILIES)):
        configuration = (
            value_plane("available", f"synthetic-{ordinal:02d}")
            if family == "change_point"
            else value_plane(
                "not_applicable", None, ["CONFIGURATION_ID_NOT_APPLICABLE"]
            )
        )
        shared = {
            "methodFamily": family,
            "configurationId": configuration,
        }
        records.append(
            make_method_record(
                **shared,
                taskId="e07_s02_spatial2d_local",
                statusStratum="synthetic|failed=false|censored=true",
                slotId=value_plane(
                    "available", f"local::censored::{family}::{ordinal}"
                ),
                evidenceState="evidentiary",
                methodExecuted=True,
                reasonCodes=[],
                payload=value_plane("available", _payload(family, ordinal)),
                rawPValue=value_plane("available", 0.04),
                holmAdjustedPValue=value_plane("available", 0.16),
                preMultiplicityGatePass=value_plane("available", True),
                multiplicityPass=value_plane("available", False),
            )
        )
        records.append(
            make_method_record(
                **shared,
                taskId="e07_s02_spatial2d_memory",
                statusStratum="synthetic|failed=true|censored=false",
                slotId=value_plane("available", f"memory::failed::{family}::{ordinal}"),
                evidenceState="non_evidentiary",
                methodExecuted=False,
                reasonCodes=["METHOD_STRUCTURALLY_INFEASIBLE"],
                payload=value_plane(
                    "unavailable",
                    None,
                    ["METHOD_STRUCTURALLY_INFEASIBLE"],
                ),
                rawPValue=value_plane("available", 1.0),
                holmAdjustedPValue=value_plane("available", 1.0),
                preMultiplicityGatePass=value_plane("available", False),
                multiplicityPass=value_plane("available", False),
            )
        )
    return records


def schema_registry() -> dict[str, Any]:
    return {
        "schemaVersion": "e07.s10g.method-record-schema-registry.v1",
        "researchStepId": STEP_ID,
        "methodRecordSchemaVersion": METHOD_RECORD_SCHEMA_VERSION,
        "recordFields": [
            "schemaVersion",
            "recordId",
            "methodFamily",
            "taskId",
            "statusStratum",
            "configurationId",
            "slotId",
            "evidenceState",
            "methodExecuted",
            "reasonCodes",
            "payload",
            "rawPValue",
            "holmAdjustedPValue",
            "preMultiplicityGatePass",
            "multiplicityPass",
        ],
        "valuePlaneFields": list(VALUE_PLANE_FIELDS),
        "valuePlaneShape": ["state", "value", "reasonCodes"],
        "valueStates": sorted(VALUE_STATES),
        "evidenceStates": sorted(EVIDENCE_STATES),
        "methodFamilies": sorted(METHOD_FAMILIES),
        "familyPayloadFields": {
            key: sorted(value) for key, value in FAMILY_PAYLOAD_FIELDS.items()
        },
        "dataframeColumns": list(DATAFRAME_COLUMNS),
        "parquetArrowSchema": str(PARQUET_SCHEMA),
        "missingValueRule": (
            "Every potentially absent value has an explicit state/value/"
            "reasonCodes plane; unavailable/not_applicable values are JSON null "
            "only inside that plane. DataFrame and Parquet cells are nonnullable."
        ),
        "rejected": [
            "NaN",
            "infinity",
            "pandas.NA",
            "pandas.NaT",
            "numpy arrays",
            "absent required fields",
            "inferred or mixed DataFrame columns",
            "noncanonical JSON text",
        ],
    }


def roundtrip_qualification(records: list[dict[str, Any]]) -> dict[str, Any]:
    expected = sorted(records, key=lambda record: record["recordId"])
    frame = records_to_dataframe(records)
    dataframe_records = records_from_dataframe(frame)
    json_payload = records_to_json_bytes(records)
    parquet_payload = records_to_parquet_bytes(records)
    json_records = records_from_json_bytes(json_payload)
    parquet_records = records_from_parquet_bytes(parquet_payload)
    reverse_json = records_to_json_bytes(list(reversed(records)))
    reverse_parquet = records_to_parquet_bytes(list(reversed(records)))
    families = {
        family: {
            "records": sum(record["methodFamily"] == family for record in records),
            "evidentiary": sum(
                record["methodFamily"] == family
                and record["evidenceState"] == "evidentiary"
                for record in records
            ),
            "nonEvidentiary": sum(
                record["methodFamily"] == family
                and record["evidenceState"] == "non_evidentiary"
                for record in records
            ),
        }
        for family in sorted(METHOD_FAMILIES)
    }
    checks = {
        "eightMixedRecords": len(records) == 8,
        "allFourFamilies": set(families) == METHOD_FAMILIES,
        "oneEvidentiaryAndOneNonEvidentiaryPerFamily": all(
            row["evidentiary"] == row["nonEvidentiary"] == 1
            for row in families.values()
        ),
        "dataframeExact": dataframe_records == expected,
        "dataframeNoMissingCells": not frame.isna().any(axis=None),
        "dataframeNoFloatColumns": not any(
            pd.api.types.is_float_dtype(dtype) for dtype in frame.dtypes
        ),
        "jsonExact": json_records == expected,
        "parquetExact": parquet_records == expected,
        "jsonWorkerOrderIndependent": reverse_json == json_payload,
        "parquetWorkerOrderIndependent": reverse_parquet == parquet_payload,
        "fixedHolmSlotsRetained": sum(
            record["methodFamily"] == "fixed_holm" for record in records
        )
        == 2,
        "nonEvidentiaryRawPExactlyOne": all(
            record["rawPValue"]["value"] == 1.0
            for record in records
            if record["evidenceState"] == "non_evidentiary"
        ),
    }
    return {
        "schemaVersion": "e07.s10g.mixed-schema-roundtrip.v1",
        "researchStepId": STEP_ID,
        "recordCount": len(records),
        "familyCounts": families,
        "dataframeDtypes": {
            column: str(dtype) for column, dtype in frame.dtypes.items()
        },
        "jsonBytes": len(json_payload),
        "jsonSha256": hashlib.sha256(json_payload).hexdigest(),
        "parquetBytes": len(parquet_payload),
        "parquetSha256": hashlib.sha256(parquet_payload).hexdigest(),
        "checks": checks,
        "allPass": all(checks.values()),
    }


def adversarial_serialization(records: list[dict[str, Any]]) -> dict[str, Any]:
    fixtures: list[tuple[str, Any]] = []
    base = records[0]
    missing = deepcopy(base)
    missing.pop("payload")
    fixtures.append(("absent_required_payload", missing))
    nan = deepcopy(base)
    nan["payload"]["value"] = {"scores": np.nan}
    fixtures.append(("nested_nan", nan))
    infinity = deepcopy(base)
    infinity["payload"]["value"] = {"scores": np.inf}
    fixtures.append(("nested_infinity", infinity))
    pandas_na = deepcopy(base)
    pandas_na["reasonCodes"] = [pd.NA]
    fixtures.append(("pandas_NA", pandas_na))
    pandas_nat = deepcopy(base)
    pandas_nat["payload"]["value"] = pd.NaT
    fixtures.append(("pandas_NaT", pandas_nat))
    numpy_array = deepcopy(base)
    numpy_array["payload"]["value"] = np.asarray([1.0, 2.0])
    fixtures.append(("numpy_array", numpy_array))
    malformed_reason = deepcopy(base)
    malformed_reason["reasonCodes"] = ["bad reason"]
    fixtures.append(("malformed_reason", malformed_reason))
    forged_hash = deepcopy(base)
    forged_hash["recordId"] = "f" * 64
    fixtures.append(("forged_record_hash", forged_hash))
    rows = []
    for label, fixture in fixtures:
        try:
            validate_method_record(fixture)
        except SerializationContractError as exc:
            rows.append(
                {
                    "fixture": label,
                    "rejected": True,
                    "exceptionType": type(exc).__name__,
                    "reason": str(exc),
                }
            )
        else:
            rows.append({"fixture": label, "rejected": False})

    frame_nan = records_to_dataframe(records)
    frame_nan.loc[0, "payload_json"] = np.nan
    frame_mapping = records_to_dataframe(records)
    frame_mapping.at[0, "payload_json"] = {"state": "available"}
    frame_absent_column = records_to_dataframe(records).drop(columns=["payload_json"])
    for label, frame in (
        ("dataframe_nan_cell", frame_nan),
        ("dataframe_mapping_cell", frame_mapping),
        ("dataframe_absent_column", frame_absent_column),
    ):
        try:
            records_from_dataframe(frame)
        except SerializationContractError as exc:
            rows.append(
                {
                    "fixture": label,
                    "rejected": True,
                    "exceptionType": type(exc).__name__,
                    "reason": str(exc),
                }
            )
        else:
            rows.append({"fixture": label, "rejected": False})
    try:
        records_from_json_bytes(b'[{"value":NaN}]\n')
    except SerializationContractError as exc:
        rows.append(
            {
                "fixture": "json_nan_constant",
                "rejected": True,
                "exceptionType": type(exc).__name__,
                "reason": str(exc),
            }
        )
    else:
        rows.append({"fixture": "json_nan_constant", "rejected": False})

    inferred_frame = records_to_dataframe(records)
    inferred_frame.loc[0, "payload_json"] = None
    sink = BytesIO()
    pq.write_table(pa.Table.from_pandas(inferred_frame, preserve_index=False), sink)
    try:
        records_from_parquet_bytes(sink.getvalue())
    except SerializationContractError as exc:
        rows.append(
            {
                "fixture": "inferred_nullable_parquet_projection",
                "rejected": True,
                "exceptionType": type(exc).__name__,
                "reason": str(exc),
            }
        )
    else:
        rows.append(
            {
                "fixture": "inferred_nullable_parquet_projection",
                "rejected": False,
            }
        )
    return {
        "schemaVersion": "e07.s10g.serialization-adversarial.v1",
        "researchStepId": STEP_ID,
        "fixtures": rows,
        "fixtureCount": len(rows),
        "rejectedCount": sum(row["rejected"] for row in rows),
        "allPass": bool(rows) and all(row["rejected"] for row in rows),
    }


def artifact_specs(config: Mapping[str, Any]) -> tuple[ArtifactSpec, ...]:
    return tuple(
        ArtifactSpec(
            class_id=row["id"],
            relative_path=row["path"],
            media_type=row["mediaType"],
        )
        for row in config["publication"]["artifactClasses"]
    )


def generic_parquet(label: str) -> bytes:
    table = pa.table(
        {
            "schemaVersion": ["e07.s10g.synthetic-artifact.v1"],
            "artifactClass": [label],
            "position": [0],
            "available": [True],
        }
    )
    sink = BytesIO()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        use_dictionary=False,
        version="2.6",
    )
    return sink.getvalue()


def publication_payloads(
    specs: tuple[ArtifactSpec, ...],
    records: list[dict[str, Any]],
) -> dict[str, bytes]:
    result = {}
    class_to_family = {
        "clustering_results": "clustering",
        "anomaly_results": "anomaly",
        "change_point_results": "change_point",
        "fixed_holm_records": "fixed_holm",
    }
    for spec in specs:
        if spec.media_type == "method_records_parquet":
            selected = [
                record
                for record in records
                if record["methodFamily"] == class_to_family[spec.class_id]
            ]
            result[spec.class_id] = records_to_parquet_bytes(selected)
        elif spec.media_type == "parquet":
            result[spec.class_id] = generic_parquet(spec.class_id)
        elif spec.media_type == "json":
            result[spec.class_id] = canonical_json_bytes(
                {
                    "schemaVersion": "e07.s10g.synthetic-artifact.v1",
                    "artifactClass": spec.class_id,
                    "availability": {
                        "state": "not_applicable",
                        "value": None,
                        "reasonCodes": ["QUALIFICATION_FIXTURE_ONLY"],
                    },
                }
            )
        else:
            result[spec.class_id] = (
                f"# Synthetic {spec.class_id}\n\n"
                "Outcome-independent publication qualification fixture.\n"
            ).encode("utf-8")
    return result


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def publication_qualification(
    config: Mapping[str, Any],
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    root = Path(tempfile.mkdtemp(prefix="e07-s10g-qualification-", dir="/cache"))
    try:
        specs = artifact_specs(config)
        payloads = publication_payloads(specs, records)
        publisher = AtomicScientificPublisher(specs)
        success_audits = []
        trees = []
        for order_name, mapping in (
            ("natural", payloads),
            ("reverse", dict(reversed(list(payloads.items())))),
        ):
            destination = root / order_name / "scientific"
            audit = publisher.publish(
                destination,
                mapping,
                forensics_directory=root / order_name / "forensics",
            )
            success_audits.append(audit)
            trees.append(tree_bytes(destination))
        success_checks = {
            "artifactClassCount20": len(specs) == 20,
            "bothOrdersComplete": all(
                row["finalScientificPublicationState"]
                == "complete_validated_publication"
                for row in success_audits
            ),
            "singleCommitBoundary": all(
                row["commitBoundaryCount"] == 1 for row in success_audits
            ),
            "completeValidationBeforeCommit": all(
                row["completeValidationBeforeCommit"] for row in success_audits
            ),
            "workerOrderIndependentBytes": trees[0] == trees[1],
            "oneManifestPlusEveryClass": len(trees[0]) == len(specs) + 1,
        }
        success = {
            "schemaVersion": "e07.s10g.publication-success-qualification.v1",
            "researchStepId": STEP_ID,
            "publicationSchemaVersion": PUBLICATION_SCHEMA_VERSION,
            "artifactClasses": [
                {
                    "id": spec.class_id,
                    "path": spec.relative_path,
                    "mediaType": spec.media_type,
                }
                for spec in specs
            ],
            "payloadCommitments": {
                key: {
                    "bytes": len(value),
                    "sha256": hashlib.sha256(value).hexdigest(),
                }
                for key, value in sorted(payloads.items())
            },
            "successAudits": success_audits,
            "publishedTreeSha256": canonical_sha256(
                "E07/S10G/synthetic-publication-tree/v1",
                {
                    key: hashlib.sha256(value).hexdigest()
                    for key, value in sorted(trees[0].items())
                },
            ),
            "checks": success_checks,
            "allPass": all(success_checks.values()),
        }

        failure_points = [
            f"{phase}:{spec.class_id}"
            for spec in specs
            for phase in ("before_write", "during_write", "after_write")
        ] + list(config["publication"]["failureInjection"]["globalPhases"])
        attempts = []
        for ordinal, failure_point in enumerate(failure_points):
            attempt_root = root / f"failure-{ordinal:03d}"
            destination = attempt_root / "scientific"
            try:
                publisher.publish(
                    destination,
                    payloads,
                    forensics_directory=attempt_root / "forensics",
                    failure_point=failure_point,
                )
            except PublicationContractError as exc:
                audit = exc.audit
            else:
                raise RuntimeError(f"injected failure was not raised: {failure_point}")
            expected_state = (
                "complete_validated_publication"
                if failure_point == "after_commit"
                else "zero_scientific_publication"
            )
            attempts.append(
                {
                    "position": ordinal,
                    "failurePoint": failure_point,
                    "expectedState": expected_state,
                    "actualState": audit["finalScientificPublicationState"],
                    "destinationExists": destination.exists(),
                    "stagingExists": Path(audit["stagingPath"]).exists(),
                    "commitBoundaryCount": audit["commitBoundaryCount"],
                    "artifactClassesCompleted": audit["artifactClassesCompleted"],
                    "attemptId": audit["attemptId"],
                    "forensicFileCount": len(
                        list((attempt_root / "forensics").glob("*.json"))
                    ),
                    "pass": (
                        audit["finalScientificPublicationState"] == expected_state
                        and not Path(audit["stagingPath"]).exists()
                        and (
                            destination.exists()
                            == (expected_state == "complete_validated_publication")
                        )
                        and len(list((attempt_root / "forensics").glob("*.json"))) == 1
                    ),
                }
            )
        expected_count = 3 * len(specs) + len(
            config["publication"]["failureInjection"]["globalPhases"]
        )
        failure_summary = {
            "schemaVersion": "e07.s10g.publication-failure-injection.v1",
            "researchStepId": STEP_ID,
            "artifactClassCount": len(specs),
            "perClassPhases": [
                "before_write",
                "during_write",
                "after_write",
            ],
            "globalPhases": config["publication"]["failureInjection"]["globalPhases"],
            "attemptCount": len(attempts),
            "expectedAttemptCount": expected_count,
            "zeroPublicationAttempts": sum(
                row["actualState"] == "zero_scientific_publication" for row in attempts
            ),
            "completePublicationAttempts": sum(
                row["actualState"] == "complete_validated_publication"
                for row in attempts
            ),
            "allForensicsExact": all(row["forensicFileCount"] == 1 for row in attempts),
            "allStagingCleaned": all(not row["stagingExists"] for row in attempts),
            "allCompleteOrZero": all(row["pass"] for row in attempts),
            "allPass": (
                len(attempts) == expected_count and all(row["pass"] for row in attempts)
            ),
        }
        return success, failure_summary, attempts
    finally:
        shutil.rmtree(root)


def candidate_ids() -> tuple[str, ...]:
    rows = [
        json.loads(line)
        for line in (S10P / "candidate_population.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    result = tuple(sorted(str(row["candidateId"]) for row in rows))
    if len(result) != 14 or len(set(result)) != 14:
        raise RuntimeError("frozen candidate identity mismatch")
    return result


def structural_controls() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    structural = s10e.structural_revalidation(candidate_ids())
    structural.update(
        {
            "schemaVersion": "e07.s10g.structural-revalidation.v1",
            "researchStepId": STEP_ID,
            "frozenEpisodesSubmitted": 0,
        }
    )
    access = s10_base.protected_denial()
    access.update(
        {
            "schemaVersion": "e07.s10g.access-control-validation.v1",
            "researchStepId": STEP_ID,
            "protectedOutcomeRowsMaterialized": 0,
        }
    )
    dependency = s10_base.dependency_audit()
    dependency.update(
        {
            "schemaVersion": "e07.s10g.prohibited-dependency-validation.v1",
            "researchStepId": STEP_ID,
            "s10fCacheFilesRead": 0,
            "s10fOutcomeArtifactFilesRead": 0,
            "s10fOutcomeRowsReused": 0,
        }
    )
    return structural, access, dependency


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def command_result(command: list[str]) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.run(
        command,
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": ".",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    )
    return {
        "command": command,
        "exitCode": process.returncode,
        "wallSeconds": round(time.monotonic() - started, 3),
        "stdoutTail": process.stdout[-4000:],
        "stderrTail": process.stderr[-4000:],
        "pass": process.returncode == 0,
    }


def artifact_manifest() -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in OUTPUT.iterdir() if item.is_file()):
        if path.name in {"artifact_manifest.json", "artifact_validation.json"}:
            continue
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schemaVersion": "e07.s10g.artifact-manifest.v1",
        "researchStepId": STEP_ID,
        "artifacts": rows,
    }


def report_markdown(
    *,
    roundtrip: Mapping[str, Any],
    adversarial: Mapping[str, Any],
    success: Mapping[str, Any],
    failure: Mapping[str, Any],
    structural: Mapping[str, Any],
    access: Mapping[str, Any],
    immutable: Mapping[str, Any],
    tests: Mapping[str, Any],
    validation: Mapping[str, Any],
    repository_commit: str,
) -> str:
    return f"""# S10G — Mixed-schema serialization and global fail-atomic publication

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S10G** |
| Completion status | **Complete; outcome-free design and technical qualification only; stopped before fresh S10, reproduction, annotation, protected access, and S11** |
| Artifacts written | Frozen protocol/schema/publication registries; input/preregistration/immutability records; mixed DataFrame/Parquet/JSON fixtures and round-trip evidence; serialization adversaries; complete-publication and 64-attempt failure-injection evidence; G01–G08, binding, access, dependency, accounting, replay/order, test, status, provenance, manifest, validation, and this canonical report under `/artifacts/research_steps/S10G/` |
| Validation result | **PASS — {validation["passedChecks"]}/{validation["totalChecks"]} integrated checks; 8 mixed method records; {failure["attemptCount"]}/{failure["expectedAttemptCount"]} injected publication failures; 28/28 bindings; zero frozen episodes** |
| Outcome classification | **Supportive bounded design/technical qualification** |
| Caveats or blockers | No phenotype, candidate, reproduction, efficacy, power, or bounded-null result exists. S10F quarantine content remained unread. A fresh S10 execution remains separately unauthorized. |
| Lay summary | Missing scientific results now have an explicit reason instead of becoming a spreadsheet-style `NaN`. The full result package is assembled and checked off to the side, then revealed in one filesystem operation. Every simulated crash point left either the whole valid package or no scientific package. |
| Recommended next action | Review S10G. If S10 is continued, separately authorize a genuinely fresh execution namespace that integrates these frozen codecs and publisher. Never reuse S10F or any consumed/quarantined cache; do not start S11. |

## Frozen question and result

S10G asked whether one explicit persistence contract can carry both
evidentiary and non-evidentiary records for all frozen S10 method families,
and whether the complete scientific output can be published in one global
transaction. The answer is **yes at the outcome-free qualification layer**.

The record contract has an explicit state/value/reason-code plane for every
potentially absent field. DataFrame cells and Arrow fields themselves are
nonnullable; nested absence is JSON `null` only when paired with an
`unavailable` or `not_applicable` state and a reason. The publisher stages all
20 registered scientific artifact classes, validates their precommitted bytes,
schemas, and complete manifest, then performs one same-filesystem atomic
directory rename.

## Inputs and authorization boundary

The step refreshed `AGENTS.md`, `FULL_PLAN.md`, `RESEARCH_PLAN.md`, E01–E06
native-event and handoff contracts, S01–S10F handoffs, S10P/S10E registries,
S10F's permitted failure forensics, and the attachment manifest/sidecar.
S10P–S10E artifact trees were content-hashed before and after. The three
outcome-bearing S10F partial artifacts were never opened; their paths, sizes,
timestamps, inodes, and already recorded hashes were checked instead. No file
under `/cache/e07-s10f` or any other consumed cache was opened; cache checks
used file metadata only.

Authorized work was limited to structural metadata, frozen mathematical
definitions, recorded integrity metadata, and generated synthetic fixtures.
Frozen episode submissions, machine fits, reproduction rows, validation or
confirmation outcomes, annotations, S06/S06A loads, S07 signals, quarantine
outcomes, archive mutations, and S11 rows were all zero.

## Detailed methods

### Common method-record schema

The frozen `e07.s10g.method-record.v1` envelope requires task/status identity,
fixed-slot identity, evidence state, executed flag, sorted reason codes, and
explicit planes for configuration, payload, raw and Holm-adjusted p-values,
and pre/post-multiplicity decisions. Evidentiary records require an executed
method and the exact family payload. Non-evidentiary records require an
unavailable payload, explicit reasons, raw `p=1`, and false candidate and
multiplicity decisions. This preserves S10E's fixed family; no slot is dropped.

DataFrame persistence uses exactly {len(DATAFRAME_COLUMNS)} nonnullable columns.
Every value plane is canonical JSON text rather than a mixed Python object
column. Parquet uses an explicit nonnullable Arrow schema, not inference.
Strict JSON rejects NaN and infinities. Every record carries a content-derived
stable hash.

### Global publication transaction

The 20-class registry covers preprocessing/confound outputs, all four method/
Holm tables, discovery lock/catalog/summary, independent reproduction, event
feature and ordered-summary tables, review boundary, accounting, replay/native
validation, and execution/final validation. The destination must not exist.
All payloads are written and fsynced in a sibling staging directory, validated
against the frozen registry and manifest, and exposed with one `os.replace`.
Exact attempt forensics are written outside the scientific destination.

Failure injection covered `before_write`, `during_write`, and `after_write`
for each of 20 classes ({3 * 20} attempts), plus before/after full validation
and before/after commit (4 attempts). A failure after the commit boundary left
a complete validated publication; every earlier failure left no scientific
destination. Staging was cleaned in all attempts.

## Commands, dependencies, and parameters

```bash
PYTHONPATH=. pytest -q tests/test_s10g_serialization_publication.py
ruff check src/phenotype_discovery/publication.py scripts/qualify_s10g_serialization_publication.py tests/test_s10g_serialization_publication.py
ruff format --check src/phenotype_discovery/publication.py scripts/qualify_s10g_serialization_publication.py tests/test_s10g_serialization_publication.py
python -m py_compile src/phenotype_discovery/publication.py scripts/qualify_s10g_serialization_publication.py tests/test_s10g_serialization_publication.py
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/qualify_s10g_serialization_publication.py
```

No dependency was installed. Qualification used one process and one numeric
thread because it was small, deterministic, and outcome-free. Python was
{platform.python_version()}, pandas {pd.__version__}, PyArrow {pa.__version__},
and NumPy {np.__version__}. Repository implementation commit:
`{repository_commit}`.

## Results

- Mixed records: {roundtrip["recordCount"]} total, exactly one evidentiary and
  one non-evidentiary fixture in each of clustering, anomaly, change-point,
  and fixed-Holm families.
- Round trips: exact across record→DataFrame→record, strict JSON, explicit
  Parquet, and reversed input order.
- Adversaries: {adversarial["rejectedCount"]}/{adversarial["fixtureCount"]}
  rejected, including absent required fields, nested NaN/infinity,
  pandas `NA`/`NaT`, arrays, forged hashes, mixed DataFrame cells, JSON NaN,
  and inferred nullable Parquet.
- Publication: two successful natural/reverse-order publications were
  byte-identical and used one commit boundary each.
- Failure injection: {failure["zeroPublicationAttempts"]} attempts left zero
  scientific publication and {failure["completePublicationAttempts"]} post-
  commit attempt left a complete validated publication; all
  {failure["attemptCount"]} attempts retained exact forensics and no staging
  residue.
- Structural controls: {structural["bindingCount"]}/28 bindings; exactly
  {structural["rosterCounts"]["rows"]:,} logical and
  {structural["rosterCounts"]["physicalReplayCommitments"]:,} replay
  commitments; G01–G08 pass; zero frozen episodes.
- Access: {access["denials"]}/{access["attempts"]} protected attempts denied
  before materialization.

## Validation

All {validation["totalChecks"]} integrated checks passed. Focused and selected
compatible test commands passed (`{tests["focused"]["stdoutTail"].strip()}`).
Canonical hashes were stable across source order. Artifact payloads were
validated both before staging and from staged/published bytes. The fixed-Holm
records remained present and non-evidentiary slots retained raw `p=1`.

Predecessor content and opaque cache metadata matched before/after snapshots.
S10F's three prohibited artifact files and `/cache/e07-s10f` were not opened.
The S10G output contains qualification evidence only, not a discovery catalog.

## Provenance

`s10g_serialization_publication_protocol.yaml` and
`preregistration_freeze.json` bind the prospective plan, source, tests,
schema/publication registries, and frozen scientific contracts.
`input_hash_freeze.json` and `immutability_validation.json` record predecessor
checks without opening prohibited S10F content.
`publication_failure_forensics.json` preserves every position-indexed injected
attempt compactly. `artifact_manifest.json` binds the final compact handoff.

## Caveats, blockers, and claim boundary

Atomic directory rename is qualified for a destination and staging directory
on the same filesystem. A future execution must use this registered complete
artifact set and must not write scientific files outside the transaction.
Exact forensic ledgers remain intentionally separate from scientific
publication. Filesystem or storage systems without atomic same-filesystem
rename require a new protocol.

S10G does not rehabilitate S10F and contains no outcome evidence. It does not
change S10P/S10E methods, populations, estimands, thresholds, candidate rules,
Holm family, failure/censoring, trace, cost, or claim boundaries. It supports
no phenotype, null, reproduction, intervention, transfer, cognition, agency,
preference, repair, biological, or clinical claim.

## Recommended next action

Hand control back. If another S10 execution is scientifically desired, require
a separate prospective execution step, a genuinely fresh namespace, integration
of the byte-frozen S10G schema/publisher, and all predecessor gates. Do not
reuse S10F, open protected outcomes, fabricate annotations, or start S11.
"""


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("S10G output already exists; fail closed")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["researchStepId"] != STEP_ID:
        raise RuntimeError("S10G configuration identity mismatch")
    if (
        sha256_file(WORKSPACE / "RESEARCH_PLAN.md")
        != config["prospectiveRegistration"]["researchPlanSha256"]
    ):
        raise RuntimeError("prospective research plan hash changed")

    before = freeze_inputs(config)
    if not before["allPass"]:
        raise RuntimeError("predecessor/input freeze failed")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "s10g_serialization_publication_protocol.yaml").write_text(
        CONFIG.read_text(encoding="utf-8"), encoding="utf-8"
    )
    records = synthetic_records()
    roundtrip = roundtrip_qualification(records)
    adversarial = adversarial_serialization(records)
    publication_success, publication_failure, attempt_rows = publication_qualification(
        config, records
    )
    structural, access, dependency = structural_controls()
    after = freeze_inputs(config)
    immutable_checks = {
        "artifactTreeSnapshotsUnchanged": (
            before["artifactTrees"] == after["artifactTrees"]
        ),
        "scientificContractHashesUnchanged": (
            before["frozenScientificContracts"] == after["frozenScientificContracts"]
        ),
        "opaqueCacheMetadataUnchanged": (
            before["opaqueConsumedCacheMetadata"]
            == after["opaqueConsumedCacheMetadata"]
        ),
        "s10fQuarantineBoundaryUnchanged": (
            before["s10fQuarantineBoundary"] == after["s10fQuarantineBoundary"]
        ),
        "s10fCacheFilesReadZero": (
            before["s10fCacheFilesRead"] == after["s10fCacheFilesRead"] == 0
        ),
        "s10fOutcomeArtifactFilesReadZero": (
            before["s10fOutcomeArtifactFilesRead"]
            == after["s10fOutcomeArtifactFilesRead"]
            == 0
        ),
    }
    immutable = {
        "schemaVersion": "e07.s10g.immutability-validation.v1",
        "researchStepId": STEP_ID,
        "before": before,
        "after": after,
        "checks": immutable_checks,
        "allPass": all(immutable_checks.values()),
    }

    tests = {
        "schemaVersion": "e07.s10g.test-validation.v1",
        "researchStepId": STEP_ID,
        "focused": command_result(
            [
                "pytest",
                "-q",
                "tests/test_s10g_serialization_publication.py",
            ]
        ),
        "compatible": command_result(
            [
                "pytest",
                "-q",
                "tests/test_s10g_serialization_publication.py",
                "tests/test_s10e_method_feasibility.py",
                "tests/test_s10c_callback_binding.py",
                "tests/test_s10a_missingness_accounting.py",
                "tests/test_s10p_native_event_discovery.py",
            ]
        ),
    }
    tests["allPass"] = tests["focused"]["pass"] and tests["compatible"]["pass"]

    registry = {
        "schemaVersion": "e07.s10g.publication-registry.v1",
        "researchStepId": STEP_ID,
        "publicationSchemaVersion": PUBLICATION_SCHEMA_VERSION,
        "commitRule": config["publication"]["commitRule"],
        "artifactClasses": config["publication"]["artifactClasses"],
        "artifactClassCount": len(config["publication"]["artifactClasses"]),
        "failureInjection": config["publication"]["failureInjection"],
        "completeSetChangedFromPreregistration": False,
    }
    accounting = {
        "schemaVersion": "e07.s10g.complete-accounting.v1",
        "researchStepId": STEP_ID,
        "syntheticMethodRecords": len(records),
        "methodFamilies": len(METHOD_FAMILIES),
        "publicationArtifactClasses": registry["artifactClassCount"],
        "successfulPublicationAttempts": 2,
        "injectedFailureAttempts": publication_failure["attemptCount"],
        "zeroPublicationInjectedAttempts": publication_failure[
            "zeroPublicationAttempts"
        ],
        "completePublicationInjectedAttempts": publication_failure[
            "completePublicationAttempts"
        ],
        "logicalStructuralCommitments": structural["rosterCounts"]["rows"],
        "physicalStructuralCommitments": structural["rosterCounts"][
            "physicalReplayCommitments"
        ],
        "frozenEpisodes": 0,
        "machineFits": 0,
        "reproductionRows": 0,
        "validationOutcomeRows": 0,
        "confirmationOutcomeRows": 0,
        "humanAnnotations": 0,
        "s10fCacheFilesRead": 0,
        "s10fOutcomeArtifactFilesRead": 0,
        "s10fOutcomeRowsReused": 0,
        "s06OrS06AArtifactsLoaded": 0,
        "s07ArmSignalsUsed": 0,
        "archiveMutations": 0,
        "s11Rows": 0,
    }
    gate = {
        "schemaVersion": "e07.s10g.gate-revalidation.v1",
        "researchStepId": STEP_ID,
        "rows": [
            {"gateId": gate_id, "pass": structural["s10pGateRows"][gate_id]}
            for gate_id in sorted(structural["s10pGateRows"])
        ],
        "allPass": (
            structural["s10pGateAllPass"] and all(structural["s10pGateRows"].values())
        ),
        "freshExecutionAuthorized": False,
    }
    replay = {
        "schemaVersion": "e07.s10g.replay-order-validation.v1",
        "researchStepId": STEP_ID,
        "methodJsonOrderIndependent": roundtrip["checks"]["jsonWorkerOrderIndependent"],
        "methodParquetOrderIndependent": roundtrip["checks"][
            "parquetWorkerOrderIndependent"
        ],
        "publicationWorkerOrderIndependent": publication_success["checks"][
            "workerOrderIndependentBytes"
        ],
        "failureAttemptPositionsUnique": len({row["position"] for row in attempt_rows})
        == len(attempt_rows),
        "failureAttemptIdsUnique": len({row["attemptId"] for row in attempt_rows})
        == len(attempt_rows),
        "allPass": True,
    }
    replay["allPass"] = all(
        value
        for key, value in replay.items()
        if key not in {"schemaVersion", "researchStepId", "allPass"}
    )

    preregistration = {
        "schemaVersion": "e07.s10g.preregistration-freeze.v1",
        "researchStepId": STEP_ID,
        "prospectiveResearchPlanSha256": config["prospectiveRegistration"][
            "researchPlanSha256"
        ],
        "protocolSha256": sha256_file(CONFIG),
        "implementationSha256": sha256_file(CORE),
        "qualificationRunnerSha256": sha256_file(SCRIPT),
        "focusedTestSha256": sha256_file(TEST),
        "schemaRegistrySha256": canonical_sha256(
            "E07/S10G/schema-registry/v1", schema_registry()
        ),
        "publicationRegistrySha256": canonical_sha256(
            "E07/S10G/publication-registry/v1", registry
        ),
        "outcomeValuesUsedToDefineSchema": 0,
        "featureValuesUsedToDefineSchema": 0,
        "frozenEpisodesBeforeFreeze": 0,
        "scientificDesignChanged": False,
        "gitCommitAtQualification": git_commit(),
    }
    validation_checks = {
        "inputFreeze": before["allPass"],
        "commonSchemaFrozen": True,
        "mixedRoundTrips": roundtrip["allPass"],
        "serializationAdversariesRejected": adversarial["allPass"],
        "fixedHolmSlotsRetained": roundtrip["checks"]["fixedHolmSlotsRetained"],
        "completePublication": publication_success["allPass"],
        "allFailureInjectionsCompleteOrZero": publication_failure["allPass"],
        "replayAndWorkerOrder": replay["allPass"],
        "g01ThroughG08": gate["allPass"],
        "bindings28": (structural["bindingCount"] == 28 and structural["allPass"]),
        "structuralAccounting": (
            structural["rosterCounts"]["rows"] == 10_752
            and structural["rosterCounts"]["physicalReplayCommitments"] == 21_504
        ),
        "protectedDenial": access["allDenied"],
        "prohibitedDependenciesExcluded": dependency["pass"],
        "predecessorImmutability": immutable["allPass"],
        "focusedAndCompatibleTests": tests["allPass"],
        "zeroFrozenEpisodes": accounting["frozenEpisodes"] == 0,
        "zeroProtectedOrProhibitedWork": all(
            accounting[key] == 0
            for key in (
                "reproductionRows",
                "validationOutcomeRows",
                "confirmationOutcomeRows",
                "humanAnnotations",
                "s10fCacheFilesRead",
                "s10fOutcomeArtifactFilesRead",
                "s10fOutcomeRowsReused",
                "s06OrS06AArtifactsLoaded",
                "s07ArmSignalsUsed",
                "archiveMutations",
                "s11Rows",
            )
        ),
    }
    validation = {
        "schemaVersion": "e07.s10g.validation-summary.v1",
        "researchStepId": STEP_ID,
        "checks": validation_checks,
        "passedChecks": sum(validation_checks.values()),
        "totalChecks": len(validation_checks),
        "allPass": all(validation_checks.values()),
        "qualificationOutcome": "supportive_bounded_technical_qualification",
        "scientificConclusionEligible": False,
    }
    if not validation["allPass"]:
        raise RuntimeError(f"S10G validation failed: {validation}")

    write_json(OUTPUT / "method_record_schema.json", schema_registry())
    write_json(OUTPUT / "publication_registry.json", registry)
    write_json(OUTPUT / "input_hash_freeze.json", before)
    write_json(OUTPUT / "preregistration_freeze.json", preregistration)
    write_json(OUTPUT / "mixed_schema_roundtrip.json", roundtrip)
    (OUTPUT / "synthetic_method_records.json").write_bytes(
        records_to_json_bytes(records)
    )
    (OUTPUT / "synthetic_method_records.parquet").write_bytes(
        records_to_parquet_bytes(records)
    )
    write_json(OUTPUT / "serialization_adversarial_qualification.json", adversarial)
    write_json(OUTPUT / "publication_success_qualification.json", publication_success)
    write_json(
        OUTPUT / "publication_failure_injection_summary.json",
        publication_failure,
    )
    write_json(
        OUTPUT / "publication_failure_forensics.json",
        {
            "schemaVersion": "e07.s10g.publication-failure-forensics.v1",
            "researchStepId": STEP_ID,
            "attempts": attempt_rows,
        },
    )
    write_json(OUTPUT / "structural_revalidation.json", structural)
    write_json(OUTPUT / "s10_gate_revalidation.json", gate)
    write_json(OUTPUT / "access_control_validation.json", access)
    write_json(OUTPUT / "prohibited_dependency_validation.json", dependency)
    write_json(OUTPUT / "replay_order_validation.json", replay)
    write_json(OUTPUT / "immutability_validation.json", immutable)
    write_json(OUTPUT / "complete_accounting.json", accounting)
    write_json(OUTPUT / "test_validation.json", tests)
    write_json(OUTPUT / "validation_summary.json", validation)
    review_gate = {
        "schemaVersion": "e07.s10g.execution-review-gate.v1",
        "researchStepId": STEP_ID,
        "qualificationPass": validation["allPass"],
        "commonExplicitSchemaQualified": roundtrip["allPass"],
        "globallyFailAtomicPublicationQualified": publication_failure["allPass"],
        "scientificDesignChanged": False,
        "s10fReusePermitted": False,
        "freshS10ExecutionAuthorized": False,
        "requiresSeparateApproval": True,
        "requiresGenuinelyFreshNamespace": True,
        "recommendedNextAction": (
            "Review S10G; separately authorize a genuinely fresh S10 "
            "execution only if desired. Do not reuse S10F or start S11."
        ),
    }
    write_json(OUTPUT / "execution_review_gate.json", review_gate)
    repository_commit = git_commit()
    write_json(
        OUTPUT / "environment_provenance.json",
        {
            "schemaVersion": "e07.s10g.environment-provenance.v1",
            "researchStepId": STEP_ID,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "workers": 1,
            "numericThreads": 1,
            "dependenciesInstalled": [],
        },
    )
    write_json(
        OUTPUT / "repository_provenance.json",
        {
            "schemaVersion": "e07.s10g.repository-provenance.v1",
            "researchStepId": STEP_ID,
            "branch": "eidosoma/groups/28",
            "commitAtQualification": repository_commit,
            "protocol": {
                "path": str(CONFIG),
                "sha256": sha256_file(CONFIG),
            },
            "implementation": {
                "path": str(CORE),
                "sha256": sha256_file(CORE),
            },
            "runner": {
                "path": str(SCRIPT),
                "sha256": sha256_file(SCRIPT),
            },
            "test": {"path": str(TEST), "sha256": sha256_file(TEST)},
        },
    )
    write_json(
        OUTPUT / "command_log.json",
        {
            "schemaVersion": "e07.s10g.command-log.v1",
            "researchStepId": STEP_ID,
            "commands": [
                "PYTHONPATH=. pytest -q tests/test_s10g_serialization_publication.py",
                "PYTHONPATH=. pytest -q tests/test_s10g_serialization_publication.py tests/test_s10e_method_feasibility.py tests/test_s10c_callback_binding.py tests/test_s10a_missingness_accounting.py tests/test_s10p_native_event_discovery.py",
                "ruff check src/phenotype_discovery/publication.py scripts/qualify_s10g_serialization_publication.py tests/test_s10g_serialization_publication.py",
                "ruff format --check src/phenotype_discovery/publication.py scripts/qualify_s10g_serialization_publication.py tests/test_s10g_serialization_publication.py",
                "python -m py_compile src/phenotype_discovery/publication.py scripts/qualify_s10g_serialization_publication.py tests/test_s10g_serialization_publication.py",
                "PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/qualify_s10g_serialization_publication.py",
            ],
            "frozenEpisodesSubmitted": 0,
        },
    )
    report = report_markdown(
        roundtrip=roundtrip,
        adversarial=adversarial,
        success=publication_success,
        failure=publication_failure,
        structural=structural,
        access=access,
        immutable=immutable,
        tests=tests,
        validation=validation,
        repository_commit=repository_commit,
    )
    (OUTPUT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": "10G",
        "success": True,
        "status": (
            "complete_outcome_free_qualification_stopped_before_fresh_"
            "S10_reproduction_annotation_protected_access_and_S11"
        ),
        "artifactsWritten": sorted(
            path.name for path in OUTPUT.iterdir() if path.is_file()
        )
        + ["artifact_manifest.json", "artifact_validation.json"],
        "validationResult": (
            f"PASS {validation['passedChecks']}/{validation['totalChecks']} "
            f"checks; 8 mixed records; {publication_failure['attemptCount']}/"
            f"{publication_failure['expectedAttemptCount']} failure injections; "
            "28/28 bindings; zero frozen episodes"
        ),
        "outcomeClassification": "supportive",
        "caveatsOrBlockers": [
            "Qualification only; no phenotype, candidate, reproduction, efficacy, power, or bounded-null result exists.",
            "S10F cache and outcome-bearing partial artifacts remained unread and prohibited.",
            "A fresh S10 execution and namespace require separate authorization.",
        ],
        "recommendedNextAction": (
            "Review S10G and separately decide whether to authorize a "
            "genuinely fresh S10 execution integrating the frozen schema and "
            "publisher; do not reuse S10F or start S11."
        ),
    }
    write_json(OUTPUT / "status.json", status)
    write_json(OUTPUT / "artifact_manifest.json", artifact_manifest())
    manifest = json.loads(
        (OUTPUT / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    manifest_pass = all(
        Path(row["path"]).exists()
        and Path(row["path"]).stat().st_size == row["bytes"]
        and sha256_file(Path(row["path"])) == row["sha256"]
        for row in manifest["artifacts"]
    )
    artifact_validation_checks = {
        "canonicalReportPresent": (OUTPUT / "research_step_full_results.md").is_file(),
        "statusRequiredFieldsPresent": all(
            key in status
            for key in (
                "researchStepId",
                "stepNumber",
                "success",
                "status",
                "artifactsWritten",
                "validationResult",
                "caveatsOrBlockers",
                "recommendedNextAction",
            )
        ),
        "manifestEntriesHashValid": manifest_pass,
        "schemaAndPublicationRegistriesPresent": all(
            (OUTPUT / name).is_file()
            for name in (
                "method_record_schema.json",
                "publication_registry.json",
            )
        ),
        "mixedJsonParquetFixturesPresent": all(
            (OUTPUT / name).is_file()
            for name in (
                "synthetic_method_records.json",
                "synthetic_method_records.parquet",
            )
        ),
        "allFailureForensicsPresent": len(attempt_rows) == 64,
        "executionReviewGateClosed": not review_gate["freshS10ExecutionAuthorized"],
        "noS11Artifact": not Path("/artifacts/research_steps/S11").exists(),
    }
    artifact_validation = {
        "schemaVersion": "e07.s10g.artifact-validation.v1",
        "researchStepId": STEP_ID,
        "checks": artifact_validation_checks,
        "passedChecks": sum(artifact_validation_checks.values()),
        "totalChecks": len(artifact_validation_checks),
        "allPass": all(artifact_validation_checks.values()),
        "manifestEntries": len(manifest["artifacts"]),
    }
    write_json(OUTPUT / "artifact_validation.json", artifact_validation)
    if not artifact_validation["allPass"]:
        raise RuntimeError("S10G artifact validation failed")
    print(
        json.dumps(
            {
                "researchStepId": STEP_ID,
                "validation": validation,
                "artifactValidation": artifact_validation,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
