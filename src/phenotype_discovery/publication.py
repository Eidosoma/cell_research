"""Explicit method-record codecs and global fail-atomic publication.

The interfaces in this module are outcome independent.  They preserve explicit
availability states and expose a scientific artifact set only after the full
staged set has passed validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


METHOD_RECORD_SCHEMA_VERSION = "e07.s10g.method-record.v1"
PUBLICATION_SCHEMA_VERSION = "e07.s10g.global-scientific-publication.v1"
EVIDENCE_STATES = frozenset({"evidentiary", "non_evidentiary"})
VALUE_STATES = frozenset({"available", "unavailable", "not_applicable"})
METHOD_FAMILIES = frozenset({"clustering", "anomaly", "change_point", "fixed_holm"})
REASON_CODE = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")

RECORD_FIELDS = (
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
)
VALUE_PLANE_FIELDS = (
    "configurationId",
    "slotId",
    "payload",
    "rawPValue",
    "holmAdjustedPValue",
    "preMultiplicityGatePass",
    "multiplicityPass",
)
DATAFRAME_COLUMNS = (
    "schema_version",
    "record_id",
    "method_family",
    "task_id",
    "status_stratum",
    "configuration_id_json",
    "slot_id_json",
    "evidence_state",
    "method_executed",
    "reason_codes_json",
    "payload_json",
    "raw_p_value_json",
    "holm_adjusted_p_value_json",
    "pre_multiplicity_gate_pass_json",
    "multiplicity_pass_json",
)
BOOL_COLUMNS = frozenset({"method_executed"})
JSON_COLUMNS_TO_RECORD_FIELDS = {
    "configuration_id_json": "configurationId",
    "slot_id_json": "slotId",
    "reason_codes_json": "reasonCodes",
    "payload_json": "payload",
    "raw_p_value_json": "rawPValue",
    "holm_adjusted_p_value_json": "holmAdjustedPValue",
    "pre_multiplicity_gate_pass_json": "preMultiplicityGatePass",
    "multiplicity_pass_json": "multiplicityPass",
}
PARQUET_SCHEMA = pa.schema(
    [
        pa.field(
            column,
            pa.bool_() if column in BOOL_COLUMNS else pa.string(),
            nullable=False,
        )
        for column in DATAFRAME_COLUMNS
    ]
)
FAMILY_PAYLOAD_FIELDS = {
    "clustering": frozenset(
        {
            "algorithm",
            "clusterLabel",
            "membership",
            "stability",
            "nullThreshold",
        }
    ),
    "anomaly": frozenset(
        {
            "algorithm",
            "selectedConfigurationId",
            "scores",
            "medianScore",
            "empiricalPValue",
        }
    ),
    "change_point": frozenset(
        {
            "algorithm",
            "series",
            "changePoints",
            "prevalence",
            "empiricalPValue",
        }
    ),
    "fixed_holm": frozenset(
        {
            "sourceMethodFamily",
            "slotState",
            "rawPValue",
            "adjustedPValue",
            "rejected",
        }
    ),
}


class SerializationContractError(ValueError):
    """A record or table violates the explicit persistence contract."""


class PublicationContractError(RuntimeError):
    """A publication attempt violated the global transaction contract."""

    def __init__(self, message: str, audit: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.audit = dict(audit or {})


def _reject_json_constant(value: str) -> None:
    raise SerializationContractError(f"prohibited JSON constant: {value}")


def validate_json_safe(value: Any, *, path: str = "$") -> None:
    """Reject ambiguous or non-JSON-safe values before any projection."""

    if value is pd.NA or value is pd.NaT:
        raise SerializationContractError(f"{path}: pandas missing scalar prohibited")
    if isinstance(value, (pd.Series, pd.DataFrame, pd.Index)):
        raise SerializationContractError(
            f"{path}: ambiguous pandas container prohibited"
        )
    if isinstance(value, np.ndarray):
        raise SerializationContractError(
            f"{path}: numpy array requires an explicit JSON list"
        )
    if isinstance(value, np.generic):
        validate_json_safe(value.item(), path=path)
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SerializationContractError(
                f"{path}: non-finite floating value prohibited"
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SerializationContractError(
                    f"{path}: every mapping key must be a string"
                )
            validate_json_safe(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_json_safe(item, path=f"{path}[{index}]")
        return
    raise SerializationContractError(
        f"{path}: unsupported JSON value type {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    validate_json_safe(value)
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def canonical_json_text(value: Any) -> str:
    return canonical_json_bytes(value).decode("ascii").rstrip("\n")


def strict_json_loads(text: str | bytes) -> Any:
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    value = json.loads(text, parse_constant=_reject_json_constant)
    validate_json_safe(value)
    return value


def canonical_sha256(domain: str, value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(value))
    return digest.hexdigest()


def value_plane(
    state: str,
    value: Any,
    reason_codes: Sequence[str] = (),
) -> dict[str, Any]:
    plane = {
        "state": state,
        "value": value,
        "reasonCodes": list(reason_codes),
    }
    validate_value_plane(plane, path="$")
    return plane


def _validate_reason_codes(
    reason_codes: Any,
    *,
    required: bool,
    path: str,
) -> None:
    if not isinstance(reason_codes, list):
        raise SerializationContractError(f"{path}: reasonCodes must be a list")
    if any(
        not isinstance(code, str) or REASON_CODE.fullmatch(code) is None
        for code in reason_codes
    ):
        raise SerializationContractError(f"{path}: malformed reason code")
    if reason_codes != sorted(set(reason_codes)):
        raise SerializationContractError(
            f"{path}: reasonCodes must be unique and sorted"
        )
    if required != bool(reason_codes):
        requirement = "required" if required else "prohibited"
        raise SerializationContractError(
            f"{path}: reasonCodes are {requirement} for this state"
        )


def validate_value_plane(value: Any, *, path: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "state",
        "value",
        "reasonCodes",
    }:
        raise SerializationContractError(
            f"{path}: value plane requires exactly state/value/reasonCodes"
        )
    state = value["state"]
    if state not in VALUE_STATES:
        raise SerializationContractError(f"{path}: unknown value state")
    validate_json_safe(value["value"], path=f"{path}.value")
    missing = state != "available"
    _validate_reason_codes(
        value["reasonCodes"],
        required=missing,
        path=f"{path}.reasonCodes",
    )
    if state == "available" and value["value"] is None:
        raise SerializationContractError(f"{path}: available value cannot be null")
    if missing and value["value"] is not None:
        raise SerializationContractError(
            f"{path}: unavailable/not-applicable value must be null"
        )


def _validate_probability_plane(plane: Mapping[str, Any], *, path: str) -> None:
    if plane["state"] != "available":
        return
    value = plane["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SerializationContractError(f"{path}: p-value must be numeric")
    if not 0.0 <= float(value) <= 1.0:
        raise SerializationContractError(f"{path}: p-value outside [0,1]")


def _validate_boolean_plane(plane: Mapping[str, Any], *, path: str) -> None:
    if plane["state"] == "available" and type(plane["value"]) is not bool:
        raise SerializationContractError(f"{path}: value must be boolean")


def validate_method_record(record: Mapping[str, Any]) -> None:
    validate_json_safe(record)
    if set(record) != set(RECORD_FIELDS):
        missing = sorted(set(RECORD_FIELDS) - set(record))
        extra = sorted(set(record) - set(RECORD_FIELDS))
        raise SerializationContractError(
            f"record fields differ; missing={missing}, extra={extra}"
        )
    if record["schemaVersion"] != METHOD_RECORD_SCHEMA_VERSION:
        raise SerializationContractError("method-record schema version mismatch")
    family = record["methodFamily"]
    if family not in METHOD_FAMILIES:
        raise SerializationContractError("unknown method family")
    for field in ("recordId", "taskId", "statusStratum"):
        if not isinstance(record[field], str) or not record[field]:
            raise SerializationContractError(f"{field} must be a nonempty string")
    if not re.fullmatch(r"[0-9a-f]{64}", record["recordId"]):
        raise SerializationContractError("recordId must be a SHA-256 hex digest")
    if record["evidenceState"] not in EVIDENCE_STATES:
        raise SerializationContractError("unknown evidence state")
    if type(record["methodExecuted"]) is not bool:
        raise SerializationContractError("methodExecuted must be boolean")
    _validate_reason_codes(
        record["reasonCodes"],
        required=record["evidenceState"] == "non_evidentiary",
        path="$.reasonCodes",
    )
    for field in VALUE_PLANE_FIELDS:
        validate_value_plane(record[field], path=f"$.{field}")
    _validate_probability_plane(record["rawPValue"], path="$.rawPValue")
    _validate_probability_plane(
        record["holmAdjustedPValue"], path="$.holmAdjustedPValue"
    )
    _validate_boolean_plane(
        record["preMultiplicityGatePass"],
        path="$.preMultiplicityGatePass",
    )
    _validate_boolean_plane(record["multiplicityPass"], path="$.multiplicityPass")

    evidentiary = record["evidenceState"] == "evidentiary"
    if record["methodExecuted"] != evidentiary:
        raise SerializationContractError(
            "methodExecuted must exactly match evidentiary state"
        )
    if evidentiary:
        if record["payload"]["state"] != "available":
            raise SerializationContractError(
                "evidentiary record requires available payload"
            )
        payload = record["payload"]["value"]
        if not isinstance(payload, Mapping):
            raise SerializationContractError("payload must be an object")
        required = FAMILY_PAYLOAD_FIELDS[family]
        if set(payload) != required:
            raise SerializationContractError(
                f"{family} payload fields differ from frozen schema"
            )
    else:
        if record["payload"]["state"] != "unavailable":
            raise SerializationContractError(
                "non-evidentiary record requires unavailable payload"
            )
        if record["rawPValue"] != value_plane("available", 1.0):
            raise SerializationContractError(
                "non-evidentiary record must retain explicit raw p=1"
            )
        if record["preMultiplicityGatePass"] != value_plane("available", False):
            raise SerializationContractError(
                "non-evidentiary record cannot pass the pre-multiplicity gate"
            )
        if record["multiplicityPass"] != value_plane("available", False):
            raise SerializationContractError(
                "non-evidentiary record cannot pass multiplicity"
            )

    if family == "change_point" and record["configurationId"]["state"] != "available":
        raise SerializationContractError(
            "change-point records require configuration identity"
        )
    if (
        family in {"clustering", "anomaly"}
        and record["configurationId"]["state"] != "not_applicable"
    ):
        raise SerializationContractError(
            f"{family} record configuration identity is not applicable"
        )
    if record["slotId"]["state"] != "available":
        raise SerializationContractError("every fixed Holm record needs a slot ID")

    content = {key: record[key] for key in RECORD_FIELDS if key != "recordId"}
    expected_id = canonical_sha256("E07/S10G/method-record/v1", content)
    if record["recordId"] != expected_id:
        raise SerializationContractError("recordId commitment mismatch")


def make_method_record(**fields: Any) -> dict[str, Any]:
    record = {"schemaVersion": METHOD_RECORD_SCHEMA_VERSION, **fields}
    record["recordId"] = "0" * 64
    content = {key: record[key] for key in RECORD_FIELDS if key != "recordId"}
    record["recordId"] = canonical_sha256("E07/S10G/method-record/v1", content)
    validate_method_record(record)
    return record


def _record_to_dataframe_row(record: Mapping[str, Any]) -> dict[str, Any]:
    validate_method_record(record)
    return {
        "schema_version": record["schemaVersion"],
        "record_id": record["recordId"],
        "method_family": record["methodFamily"],
        "task_id": record["taskId"],
        "status_stratum": record["statusStratum"],
        "configuration_id_json": canonical_json_text(record["configurationId"]),
        "slot_id_json": canonical_json_text(record["slotId"]),
        "evidence_state": record["evidenceState"],
        "method_executed": record["methodExecuted"],
        "reason_codes_json": canonical_json_text(record["reasonCodes"]),
        "payload_json": canonical_json_text(record["payload"]),
        "raw_p_value_json": canonical_json_text(record["rawPValue"]),
        "holm_adjusted_p_value_json": canonical_json_text(record["holmAdjustedPValue"]),
        "pre_multiplicity_gate_pass_json": canonical_json_text(
            record["preMultiplicityGatePass"]
        ),
        "multiplicity_pass_json": canonical_json_text(record["multiplicityPass"]),
    }


def records_to_dataframe(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    ordered = sorted(records, key=lambda record: str(record["recordId"]))
    frame = pd.DataFrame(
        [_record_to_dataframe_row(record) for record in ordered],
        columns=DATAFRAME_COLUMNS,
    )
    return validate_dataframe(frame)


def validate_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.has_duplicates:
        raise SerializationContractError("duplicate DataFrame columns prohibited")
    if tuple(frame.columns) != DATAFRAME_COLUMNS:
        raise SerializationContractError(
            "DataFrame must have the exact frozen columns and order"
        )
    rows: list[dict[str, Any]] = []
    for row_index, source in frame.iterrows():
        row: dict[str, Any] = {}
        for column in DATAFRAME_COLUMNS:
            value = source[column]
            if column in BOOL_COLUMNS:
                if type(value) not in (bool, np.bool_):
                    raise SerializationContractError(
                        f"row {row_index} column {column}: boolean required"
                    )
                row[column] = bool(value)
            else:
                if not isinstance(value, str):
                    raise SerializationContractError(
                        f"row {row_index} column {column}: nonnullable string required"
                    )
                row[column] = value
        for column in JSON_COLUMNS_TO_RECORD_FIELDS:
            parsed = strict_json_loads(row[column])
            if canonical_json_text(parsed) != row[column]:
                raise SerializationContractError(
                    f"row {row_index} column {column}: noncanonical JSON text"
                )
        rows.append(row)
    # Reconstructing is part of DataFrame validation, so field mixing cannot
    # survive merely because every cell is technically serializable.
    _dataframe_rows_to_records(rows)
    return frame


def _dataframe_rows_to_records(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        record: dict[str, Any] = {
            "schemaVersion": row["schema_version"],
            "recordId": row["record_id"],
            "methodFamily": row["method_family"],
            "taskId": row["task_id"],
            "statusStratum": row["status_stratum"],
            "evidenceState": row["evidence_state"],
            "methodExecuted": bool(row["method_executed"]),
        }
        for column, field in JSON_COLUMNS_TO_RECORD_FIELDS.items():
            record[field] = strict_json_loads(row[column])
        validate_method_record(record)
        records.append(record)
    if len({record["recordId"] for record in records}) != len(records):
        raise SerializationContractError("duplicate method record IDs prohibited")
    return sorted(records, key=lambda record: record["recordId"])


def records_from_dataframe(frame: pd.DataFrame) -> list[dict[str, Any]]:
    validate_dataframe(frame)
    rows = [
        {column: frame.iloc[index][column] for column in DATAFRAME_COLUMNS}
        for index in range(len(frame))
    ]
    return _dataframe_rows_to_records(rows)


def records_to_json_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    validated = sorted(records, key=lambda record: str(record["recordId"]))
    for record in validated:
        validate_method_record(record)
    if len({record["recordId"] for record in validated}) != len(validated):
        raise SerializationContractError("duplicate method record IDs prohibited")
    return canonical_json_bytes(validated)


def records_from_json_bytes(payload: bytes) -> list[dict[str, Any]]:
    value = strict_json_loads(payload)
    if not isinstance(value, list):
        raise SerializationContractError("method-record JSON must be a list")
    for record in value:
        if not isinstance(record, Mapping):
            raise SerializationContractError("method-record item must be an object")
        validate_method_record(record)
    if value != sorted(value, key=lambda record: record["recordId"]):
        raise SerializationContractError("method-record JSON order is noncanonical")
    if records_to_json_bytes(value) != payload:
        raise SerializationContractError("method-record JSON bytes are noncanonical")
    return list(value)


def records_to_parquet_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    frame = records_to_dataframe(records)
    table = pa.Table.from_pandas(
        frame,
        schema=PARQUET_SCHEMA,
        preserve_index=False,
        safe=True,
    )
    sink = BytesIO()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        version="2.6",
    )
    return sink.getvalue()


def records_from_parquet_bytes(payload: bytes) -> list[dict[str, Any]]:
    table = pq.read_table(BytesIO(payload))
    if table.schema != PARQUET_SCHEMA:
        raise SerializationContractError(f"Parquet schema differs: {table.schema}")
    if any(column.null_count for column in table.columns):
        raise SerializationContractError("Parquet null cells prohibited")
    frame = table.to_pandas()
    return records_from_dataframe(frame)


@dataclass(frozen=True)
class ArtifactSpec:
    class_id: str
    relative_path: str
    media_type: str


def validate_publication_registry(
    specs: Sequence[ArtifactSpec],
) -> tuple[ArtifactSpec, ...]:
    result = tuple(specs)
    class_ids = [spec.class_id for spec in result]
    paths = [spec.relative_path for spec in result]
    if not result or len(set(class_ids)) != len(class_ids):
        raise PublicationContractError("artifact class IDs must be nonempty/unique")
    if len(set(paths)) != len(paths):
        raise PublicationContractError("artifact paths must be unique")
    for spec in result:
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", spec.class_id):
            raise PublicationContractError("malformed artifact class ID")
        pure = PurePosixPath(spec.relative_path)
        if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
            raise PublicationContractError(
                "artifact path must be one safe relative filename"
            )
        if spec.media_type not in {
            "json",
            "markdown",
            "parquet",
            "method_records_parquet",
        }:
            raise PublicationContractError("unknown artifact media type")
    return result


def _validate_generic_parquet(payload: bytes) -> None:
    table = pq.read_table(BytesIO(payload))
    if not table.column_names:
        raise PublicationContractError("generic Parquet requires columns")
    if any(column.null_count for column in table.columns):
        raise PublicationContractError(
            "generic scientific Parquet null cells prohibited"
        )
    for field, column in zip(table.schema, table.columns):
        if pa.types.is_floating(field.type):
            values = column.combine_chunks().to_numpy(zero_copy_only=False)
            if not np.isfinite(values).all():
                raise PublicationContractError(
                    "generic scientific Parquet non-finite value prohibited"
                )


def validate_artifact_payload(spec: ArtifactSpec, payload: bytes) -> None:
    if not isinstance(payload, bytes) or not payload:
        raise PublicationContractError(
            f"{spec.class_id}: payload must be nonempty bytes"
        )
    if spec.media_type == "json":
        strict_json_loads(payload)
    elif spec.media_type == "markdown":
        text = payload.decode("utf-8")
        if not text.strip():
            raise PublicationContractError("empty Markdown prohibited")
    elif spec.media_type == "parquet":
        _validate_generic_parquet(payload)
    elif spec.media_type == "method_records_parquet":
        records_from_parquet_bytes(payload)
    else:  # pragma: no cover - registry validation closes this path
        raise PublicationContractError("unregistered media type")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _strict_json_file(path: Path) -> Any:
    return strict_json_loads(path.read_bytes())


class AtomicScientificPublisher:
    """Stage, validate, and expose a complete artifact set in one rename."""

    def __init__(self, specs: Sequence[ArtifactSpec]):
        self.specs = validate_publication_registry(specs)

    def _manifest(
        self,
        payloads: Mapping[str, bytes],
        *,
        attempt_id: str,
    ) -> dict[str, Any]:
        return {
            "schemaVersion": PUBLICATION_SCHEMA_VERSION,
            "attemptId": attempt_id,
            "artifactCount": len(self.specs),
            "artifacts": [
                {
                    "classId": spec.class_id,
                    "path": spec.relative_path,
                    "mediaType": spec.media_type,
                    "bytes": len(payloads[spec.class_id]),
                    "sha256": hashlib.sha256(payloads[spec.class_id]).hexdigest(),
                }
                for spec in self.specs
            ],
            "commitBoundary": "one_same_filesystem_atomic_directory_rename",
        }

    def validate_complete(
        self,
        destination: Path,
        payloads: Mapping[str, bytes],
        *,
        attempt_id: str,
    ) -> dict[str, Any]:
        expected_names = {spec.relative_path for spec in self.specs} | {
            "publication_manifest.json"
        }
        actual_names = {
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file()
        }
        if actual_names != expected_names:
            raise PublicationContractError(
                "published artifact set is incomplete or contains unknown files"
            )
        for spec in self.specs:
            path = destination / spec.relative_path
            payload = path.read_bytes()
            expected = payloads[spec.class_id]
            if payload != expected:
                raise PublicationContractError(
                    f"{spec.class_id}: published bytes differ from commitment"
                )
            validate_artifact_payload(spec, payload)
        manifest = _strict_json_file(destination / "publication_manifest.json")
        if manifest != self._manifest(payloads, attempt_id=attempt_id):
            raise PublicationContractError("publication manifest mismatch")
        return {
            "complete": True,
            "artifactCount": len(self.specs),
            "manifestSha256": hashlib.sha256(
                canonical_json_bytes(manifest)
            ).hexdigest(),
        }

    @staticmethod
    def _inject(expected: str | None, actual: str) -> None:
        if expected == actual:
            raise PublicationContractError(
                f"deterministic injected failure at {actual}"
            )

    @staticmethod
    def _write_forensic(path: Path, audit: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(canonical_json_bytes(audit))
        os.replace(temporary, path)
        _fsync_directory(path.parent)

    def publish(
        self,
        destination: Path,
        payloads: Mapping[str, bytes],
        *,
        forensics_directory: Path,
        failure_point: str | None = None,
    ) -> dict[str, Any]:
        expected_classes = {spec.class_id for spec in self.specs}
        if set(payloads) != expected_classes:
            raise PublicationContractError(
                "payload classes must exactly match frozen publication registry"
            )
        for spec in self.specs:
            validate_artifact_payload(spec, payloads[spec.class_id])
        if destination.exists():
            raise PublicationContractError(
                "scientific destination must be absent before publication"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        forensics_directory.mkdir(parents=True, exist_ok=True)
        attempt_id = canonical_sha256(
            "E07/S10G/publication-attempt/v1",
            {
                "destinationName": destination.name,
                "failurePoint": failure_point,
                "payloads": {
                    key: hashlib.sha256(payload).hexdigest()
                    for key, payload in sorted(payloads.items())
                },
            },
        )
        stage = destination.parent / f".{destination.name}.stage-{attempt_id}"
        if stage.exists():
            raise PublicationContractError("staging path already exists")
        audit: dict[str, Any] = {
            "schemaVersion": "e07.s10g.publication-attempt-forensics.v1",
            "attemptId": attempt_id,
            "failurePoint": failure_point,
            "scientificDestination": str(destination),
            "stagingPath": str(stage),
            "artifactCount": len(self.specs),
            "artifactClassesCompleted": [],
            "commitBoundaryCount": 0,
            "completeValidationBeforeCommit": False,
            "finalScientificPublicationState": "not_started",
            "exception": None,
        }
        forensic_path = forensics_directory / f"{attempt_id}.json"
        committed = False
        try:
            stage.mkdir(parents=False, exist_ok=False)
            if stage.stat().st_dev != destination.parent.stat().st_dev:
                raise PublicationContractError(
                    "staging and destination are not on one filesystem"
                )
            for spec in self.specs:
                self._inject(failure_point, f"before_write:{spec.class_id}")
                payload = payloads[spec.class_id]
                path = stage / spec.relative_path
                split = max(1, len(payload) // 2)
                with path.open("wb") as handle:
                    handle.write(payload[:split])
                    handle.flush()
                    os.fsync(handle.fileno())
                    self._inject(failure_point, f"during_write:{spec.class_id}")
                    handle.write(payload[split:])
                    handle.flush()
                    os.fsync(handle.fileno())
                validate_artifact_payload(spec, path.read_bytes())
                audit["artifactClassesCompleted"].append(spec.class_id)
                self._inject(failure_point, f"after_write:{spec.class_id}")

            manifest = self._manifest(payloads, attempt_id=attempt_id)
            (stage / "publication_manifest.json").write_bytes(
                canonical_json_bytes(manifest)
            )
            _fsync_directory(stage)
            self._inject(failure_point, "before_full_validation")
            self.validate_complete(stage, payloads, attempt_id=attempt_id)
            audit["completeValidationBeforeCommit"] = True
            self._inject(failure_point, "after_full_validation")
            self._inject(failure_point, "before_commit")
            os.replace(stage, destination)
            audit["commitBoundaryCount"] = 1
            committed = True
            _fsync_directory(destination.parent)
            self._inject(failure_point, "after_commit")
            completed = self.validate_complete(
                destination, payloads, attempt_id=attempt_id
            )
            audit["finalScientificPublicationState"] = "complete_validated_publication"
            audit["completeValidationAfterCommit"] = completed
            self._write_forensic(forensic_path, audit)
            return audit
        except Exception as exc:
            if stage.exists():
                shutil.rmtree(stage)
            complete = False
            if destination.exists():
                self.validate_complete(destination, payloads, attempt_id=attempt_id)
                complete = True
            audit["exception"] = f"{type(exc).__name__}: {exc}"
            audit["finalScientificPublicationState"] = (
                "complete_validated_publication"
                if complete
                else "zero_scientific_publication"
            )
            audit["commitBoundaryCount"] = 1 if committed else 0
            audit["destinationExists"] = destination.exists()
            audit["stagingExists"] = stage.exists()
            self._write_forensic(forensic_path, audit)
            raise PublicationContractError(str(exc), audit) from exc
