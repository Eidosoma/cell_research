from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from src.phenotype_discovery.publication import (
    ArtifactSpec,
    AtomicScientificPublisher,
    DATAFRAME_COLUMNS,
    PublicationContractError,
    SerializationContractError,
    canonical_json_bytes,
    make_method_record,
    records_from_dataframe,
    records_from_json_bytes,
    records_from_parquet_bytes,
    records_to_dataframe,
    records_to_json_bytes,
    records_to_parquet_bytes,
    validate_method_record,
    validate_publication_registry,
    value_plane,
)


CONFIG = (
    Path(__file__).parents[1] / "configs/discovery/s10g_serialization_publication.yaml"
)


def _payload(family: str, ordinal: int) -> dict:
    if family == "clustering":
        return {
            "algorithm": "agglomerative_Ward",
            "clusterLabel": ordinal,
            "membership": [f"candidate-{ordinal:02d}", "candidate-13"],
            "stability": 0.8,
            "nullThreshold": 0.4,
        }
    if family == "anomaly":
        return {
            "algorithm": "IsolationForest",
            "selectedConfigurationId": f"candidate-{ordinal:02d}",
            "scores": {
                f"candidate-{ordinal:02d}": 0.3,
                "candidate-13": 0.1,
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


def method_records() -> list[dict]:
    records = []
    for ordinal, family in enumerate(
        ("clustering", "anomaly", "change_point", "fixed_holm")
    ):
        configuration = (
            value_plane("available", f"candidate-{ordinal:02d}")
            if family == "change_point"
            else value_plane(
                "not_applicable", None, ["CONFIGURATION_ID_NOT_APPLICABLE"]
            )
        )
        records.append(
            make_method_record(
                methodFamily=family,
                taskId="e07_s02_spatial2d_local",
                statusStratum="synthetic|failed=false|censored=true",
                configurationId=configuration,
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
                methodFamily=family,
                taskId="e07_s02_spatial2d_memory",
                statusStratum="synthetic|failed=true|censored=false",
                configurationId=configuration,
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


def publication_specs() -> tuple[ArtifactSpec, ...]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return tuple(
        ArtifactSpec(
            class_id=row["id"],
            relative_path=row["path"],
            media_type=row["mediaType"],
        )
        for row in config["publication"]["artifactClasses"]
    )


def _generic_parquet(label: str) -> bytes:
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


def publication_payloads() -> dict[str, bytes]:
    records = method_records()
    result = {}
    for spec in publication_specs():
        if spec.media_type == "method_records_parquet":
            family = {
                "clustering_results": "clustering",
                "anomaly_results": "anomaly",
                "change_point_results": "change_point",
                "fixed_holm_records": "fixed_holm",
            }[spec.class_id]
            selected = [
                record for record in records if record["methodFamily"] == family
            ]
            result[spec.class_id] = records_to_parquet_bytes(selected)
        elif spec.media_type == "parquet":
            result[spec.class_id] = _generic_parquet(spec.class_id)
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


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_mixed_method_records_round_trip_exactly() -> None:
    records = method_records()
    expected = sorted(records, key=lambda record: record["recordId"])
    frame = records_to_dataframe(records)
    assert tuple(frame.columns) == DATAFRAME_COLUMNS
    assert not frame.isna().any(axis=None)
    assert records_from_dataframe(frame) == expected
    json_payload = records_to_json_bytes(records)
    assert records_from_json_bytes(json_payload) == expected
    parquet_payload = records_to_parquet_bytes(records)
    assert records_from_parquet_bytes(parquet_payload) == expected
    assert records_to_json_bytes(list(reversed(records))) == json_payload
    assert records_to_parquet_bytes(list(reversed(records))) == parquet_payload


@pytest.mark.parametrize(
    "mutator",
    [
        lambda record: record.pop("payload"),
        lambda record: record["payload"].update(value={"scores": np.nan}),
        lambda record: record["payload"].update(value={"scores": np.inf}),
        lambda record: record.update(reasonCodes=[pd.NA]),
        lambda record: record["payload"].update(value=pd.NaT),
    ],
)
def test_record_adversaries_fail_closed(mutator) -> None:
    record = deepcopy(method_records()[0])
    mutator(record)
    with pytest.raises(SerializationContractError):
        validate_method_record(record)


def test_ambiguous_pandas_projection_and_nan_json_fail_closed() -> None:
    frame = records_to_dataframe(method_records())
    frame.loc[0, "payload_json"] = np.nan
    with pytest.raises(SerializationContractError):
        records_from_dataframe(frame)
    frame = records_to_dataframe(method_records())
    frame.at[0, "payload_json"] = {"state": "available"}
    with pytest.raises(SerializationContractError):
        records_from_dataframe(frame)
    with pytest.raises(SerializationContractError):
        records_from_json_bytes(b'[{"value":NaN}]\n')


def test_parquet_inference_or_null_projection_fails_closed() -> None:
    frame = records_to_dataframe(method_records())
    frame.loc[0, "payload_json"] = None
    inferred = pa.Table.from_pandas(frame, preserve_index=False)
    sink = BytesIO()
    pq.write_table(inferred, sink)
    with pytest.raises(SerializationContractError):
        records_from_parquet_bytes(sink.getvalue())


def test_publication_registry_adversaries_fail_closed() -> None:
    specs = publication_specs()
    assert len(validate_publication_registry(specs)) == 20
    with pytest.raises(PublicationContractError):
        validate_publication_registry((*specs, specs[0]))
    with pytest.raises(PublicationContractError):
        validate_publication_registry(
            (ArtifactSpec("unsafe", "../escape.json", "json"),)
        )
    with pytest.raises(PublicationContractError):
        validate_publication_registry(
            (ArtifactSpec("unknown_type", "item.bin", "binary"),)
        )


def test_complete_publication_is_deterministic_and_single_boundary(
    tmp_path: Path,
) -> None:
    specs = publication_specs()
    payloads = publication_payloads()
    publisher = AtomicScientificPublisher(specs)
    audits = []
    trees = []
    for label, mapping in (
        ("natural", payloads),
        ("reverse", dict(reversed(list(payloads.items())))),
    ):
        destination = tmp_path / label / "scientific"
        audit = publisher.publish(
            destination,
            mapping,
            forensics_directory=tmp_path / label / "forensics",
        )
        audits.append(audit)
        trees.append(_tree_bytes(destination))
    assert all(audit["commitBoundaryCount"] == 1 for audit in audits)
    assert all(
        audit["finalScientificPublicationState"] == "complete_validated_publication"
        for audit in audits
    )
    assert trees[0] == trees[1]


def test_every_injected_failure_is_complete_or_zero(tmp_path: Path) -> None:
    specs = publication_specs()
    payloads = publication_payloads()
    publisher = AtomicScientificPublisher(specs)
    failure_points = [
        f"{phase}:{spec.class_id}"
        for spec in specs
        for phase in ("before_write", "during_write", "after_write")
    ] + [
        "before_full_validation",
        "after_full_validation",
        "before_commit",
        "after_commit",
    ]
    assert len(failure_points) == 64
    for ordinal, failure_point in enumerate(failure_points):
        root = tmp_path / f"attempt-{ordinal:03d}"
        destination = root / "scientific"
        with pytest.raises(PublicationContractError) as captured:
            publisher.publish(
                destination,
                payloads,
                forensics_directory=root / "forensics",
                failure_point=failure_point,
            )
        audit = captured.value.audit
        state = audit["finalScientificPublicationState"]
        if failure_point == "after_commit":
            assert state == "complete_validated_publication"
            assert destination.is_dir()
            publisher.validate_complete(
                destination,
                payloads,
                attempt_id=audit["attemptId"],
            )
        else:
            assert state == "zero_scientific_publication"
            assert not destination.exists()
        assert not Path(audit["stagingPath"]).exists()
        forensic_files = list((root / "forensics").glob("*.json"))
        assert len(forensic_files) == 1
        persisted = json.loads(forensic_files[0].read_text(encoding="utf-8"))
        assert persisted["failurePoint"] == failure_point
        assert persisted["finalScientificPublicationState"] == state
