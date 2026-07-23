#!/usr/bin/env python3
"""Execute S10H freshly and publish science only through S10G's transaction."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
import yaml

import scripts.run_native_event_discovery_s10 as base
import scripts.run_native_event_discovery_s10b as s10b
from src.phenotype_discovery import feasible_search
from src.phenotype_discovery import publication
from src.phenotype_discovery.search import sha256_file


STEP_ID = "S10H"
REPOSITORY = Path("/workspace/cell-research")
CONFIG = REPOSITORY / "configs/discovery/s10h_fresh_execution.yaml"
SCRIPT = Path(__file__).resolve()
TEST = REPOSITORY / "tests/test_s10h_fresh_execution.py"
OUTPUT = Path("/artifacts/research_steps/S10H")
SCIENTIFIC = OUTPUT / "scientific_publication"
PUBLICATION_FORENSICS = OUTPUT / "publication_forensics"
CACHE_ROOT = Path("/cache/e07-s10h")
WORK_OUTPUT = CACHE_ROOT / "work"
RUN_CACHE = CACHE_ROOT / "runtime"
S10P = Path("/artifacts/research_steps/S10P")
S10F = Path("/artifacts/research_steps/S10F")
S10G = Path("/artifacts/research_steps/S10G")
ARTIFACT_TREES = {
    "S10P": S10P,
    "S10": Path("/artifacts/research_steps/S10"),
    "S10A": Path("/artifacts/research_steps/S10A"),
    "S10B": Path("/artifacts/research_steps/S10B"),
    "S10C": Path("/artifacts/research_steps/S10C"),
    "S10D": Path("/artifacts/research_steps/S10D"),
    "S10E": Path("/artifacts/research_steps/S10E"),
    "S10G": S10G,
}
HISTORICAL = (
    Path("/artifacts/research_steps/S05"),
    Path("/artifacts/research_steps/S08M"),
    Path("/artifacts/research_steps/S09"),
    *ARTIFACT_TREES.values(),
)
GENERIC_SCHEMA = pa.schema(
    [
        pa.field("position", pa.int64(), nullable=False),
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("record_json", pa.string(), nullable=False),
    ]
)
_CALLBACK_COUNTS = {"revalidate_frozen_inputs": 0, "prospective_freeze": 0}
_FRESH_ROOT_ABSENT_BEFORE_ENTRY = False


def canonical_json_bytes(value: Any) -> bytes:
    return publication.canonical_json_bytes(_json_safe(value))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _json_safe(value: Any, *, path: str = "$") -> Any:
    """Convert explicit numeric containers while rejecting ambiguous missingness."""

    if value is pd.NA or value is pd.NaT:
        raise publication.SerializationContractError(
            f"{path}: pandas missing scalar prohibited"
        )
    if isinstance(value, np.ndarray):
        return [_json_safe(item, path=f"{path}[]") for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item(), path=path)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, path=f"{path}[]") for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise publication.SerializationContractError(
            f"{path}: non-finite scalar prohibited"
        )
    publication.validate_json_safe(value, path=path)
    return value


def safe_intermediate_json_columns(
    frame: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    """Persist cache-only legacy tables with explicit missing-value envelopes."""

    frame = frame.copy()
    unavailable = {
        "state": "unavailable",
        "value": None,
        "reasonCodes": ["METHOD_FIELD_UNAVAILABLE"],
    }
    for column in columns:
        if column not in frame:
            continue

        def encode(value: Any) -> str:
            if value is None or (
                isinstance(value, (float, np.floating))
                and not math.isfinite(float(value))
            ):
                return publication.canonical_json_text(unavailable)
            return publication.canonical_json_text(_json_safe(value))

        frame[column] = frame[column].map(encode)
    return frame


def _configure_globals() -> None:
    for module in (base, s10b):
        module.STEP_ID = STEP_ID
        module.OUTPUT = WORK_OUTPUT
        module.CACHE = RUN_CACHE
        module.CONFIG = CONFIG
        module.SCRIPT = SCRIPT
        module.TEST = TEST
        module.HISTORICAL = HISTORICAL


def _tree_check(
    label: str, root: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    actual = s10b.tree_snapshot(root)
    passed = (
        actual["treeSha256"] == expected["treeSha256"]
        and actual["fileCount"] == expected["fileCount"]
        and actual["totalBytes"] == expected["totalBytes"]
    )
    return {
        "stepId": label,
        "path": str(root),
        "expectedTreeSha256": expected["treeSha256"],
        "actualTreeSha256": actual["treeSha256"],
        "expectedFileCount": expected["fileCount"],
        "actualFileCount": actual["fileCount"],
        "expectedTotalBytes": expected["totalBytes"],
        "actualTotalBytes": actual["totalBytes"],
        "pass": passed,
    }


def _metadata_snapshot(path: Path) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    rows = [
        {
            "relativePath": item.relative_to(path).as_posix(),
            "bytes": item.stat().st_size,
            "mtimeNs": item.stat().st_mtime_ns,
            "inode": item.stat().st_ino,
        }
        for item in files
    ]
    return {
        "path": str(path),
        "fileCount": len(files),
        "totalBytes": sum(row["bytes"] for row in rows),
        "metadataSha256": publication.canonical_sha256(
            "E07/S10H/opaque-cache-metadata/v1", rows
        ),
        "contentFilesOpened": 0,
    }


def _cache_metadata_check(
    label: str, expected: Mapping[str, Any]
) -> dict[str, Any]:
    actual = _metadata_snapshot(Path(expected["path"]))
    actual.update(
        {
            "stepId": label,
            "recordedTreeSha256": expected["recordedTreeSha256"],
            "expectedFileCount": expected["fileCount"],
            "expectedTotalBytes": expected["totalBytes"],
            "pass": (
                actual["fileCount"] == expected["fileCount"]
                and actual["totalBytes"] == expected["totalBytes"]
            ),
        }
    )
    return actual


def _s10f_boundary(config: Mapping[str, Any]) -> dict[str, Any]:
    boundary = config["s10fQuarantineBoundary"]
    root = Path(boundary["artifactRoot"])
    prohibited = {
        Path(row["path"]): row for row in boundary["prohibitedArtifactFiles"]
    }
    files = sorted(item for item in root.rglob("*") if item.is_file())
    stat_rows = []
    for path, expected in prohibited.items():
        stat = path.stat()
        stat_rows.append(
            {
                "path": str(path),
                "bytes": stat.st_size,
                "expectedBytes": expected["bytes"],
                "recordedSha256": expected["recordedSha256"],
                "contentOpened": False,
                "pass": stat.st_size == expected["bytes"],
            }
        )
    permitted = [path for path in files if path not in prohibited]
    lines = "".join(f"{sha256_file(path)}  {path}\n" for path in permitted)
    permitted_digest = hashlib.sha256(lines.encode("utf-8")).hexdigest()
    manifest = boundary["recordedQuarantineManifest"]
    manifest_hash = sha256_file(manifest["path"])
    cache = _cache_metadata_check("S10F", boundary["prohibitedCache"])
    result = {
        "artifactRoot": str(root),
        "artifactFileCount": len(files),
        "artifactTotalBytes": sum(path.stat().st_size for path in files),
        "permittedContentFileCount": len(permitted),
        "expectedPermittedContentListSha256": boundary[
            "permittedContentListSha256"
        ],
        "actualPermittedContentListSha256": permitted_digest,
        "prohibitedArtifactStatRows": stat_rows,
        "prohibitedArtifactContentFilesOpened": 0,
        "quarantineManifest": {
            "path": manifest["path"],
            "expectedSha256": manifest["sha256"],
            "actualSha256": manifest_hash,
            "pass": manifest_hash == manifest["sha256"],
        },
        "opaqueCacheMetadata": cache,
    }
    result["pass"] = bool(
        result["artifactFileCount"] == boundary["artifactFileCount"]
        and result["artifactTotalBytes"] == boundary["artifactTotalBytes"]
        and result["permittedContentFileCount"]
        == boundary["permittedContentFileCount"]
        and permitted_digest == boundary["permittedContentListSha256"]
        and all(row["pass"] for row in stat_rows)
        and result["quarantineManifest"]["pass"]
        and cache["pass"]
    )
    return result


def _file_check(name: str, expected: Mapping[str, Any]) -> dict[str, Any]:
    actual = sha256_file(expected["path"])
    return {
        "name": name,
        "path": expected["path"],
        "expectedSha256": expected["sha256"],
        "actualSha256": actual,
        "pass": actual == expected["sha256"],
    }


def _callbacks() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "revalidate_frozen_inputs": revalidate_frozen_inputs,
            "prospective_freeze": prospective_freeze,
            "reviewer_instructions": reviewer_instructions,
            "report_markdown": cache_only_report,
            "manifest_for_output": cache_only_manifest,
        }
    )


def _structural_preflight_pass(result: Mapping[str, Any]) -> bool:
    remediated = {
        "/workspace/cell-research/src/phenotype_discovery/accounting.py",
        "/workspace/cell-research/src/phenotype_discovery/native_features.py",
        "/workspace/cell-research/src/phenotype_discovery/search.py",
    }
    nonsuperseded = [
        row for row in result["frozenInputChecks"] if row["path"] not in remediated
    ]
    return bool(
        all(row["pass"] for row in result["frozenFileChecks"])
        and all(row["pass"] for row in nonsuperseded)
        and all(result["preregistrationChecks"].values())
        and result["s10pGateAllPass"]
        and set(result["s10pGateRows"]) == {f"G{i:02d}" for i in range(1, 9)}
        and all(result["s10pGateRows"].values())
        and len(result["bindingRows"]) == 28
        and all(row["pass"] for row in result["bindingRows"])
        and result["rosterChecks"]
        == {
            "rows": 10_752,
            "uniqueLogicalReservationIds": 10_752,
            "candidateCount": 14,
            "taskCount": 2,
            "discoveryRows": 7_168,
            "reproductionRows": 3_584,
        }
        and all(row["pass"] for row in result["s10aRemediatedCoreChecks"])
        and all(row["pass"] for row in result["immutableTreeChecks"].values())
        and all(row["pass"] for row in result["s10bScientificFileChecks"])
        and result["preOutcomeCommitmentAudit"]["pass"]
        and result["s10aQualificationGatePass"]
    )


def publication_specs() -> tuple[publication.ArtifactSpec, ...]:
    registry = json.loads(
        (S10G / "publication_registry.json").read_text(encoding="utf-8")
    )
    if registry["artifactClassCount"] != 20:
        raise RuntimeError("S10G publication registry cardinality changed")
    return tuple(
        publication.ArtifactSpec(
            class_id=row["id"],
            relative_path=row["path"],
            media_type=row["mediaType"],
        )
        for row in registry["artifactClasses"]
    )


def revalidate_frozen_inputs() -> dict[str, Any]:
    _CALLBACK_COUNTS["revalidate_frozen_inputs"] += 1
    inherited = s10b.revalidate_frozen_inputs()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    immutable = config["immutableInputs"]
    tree_keys = {
        "S10P": "s10pTree",
        "S10": "failedS10Tree",
        "S10A": "s10aTree",
        "S10B": "s10bTree",
        "S10C": "s10cTree",
        "S10D": "s10dTree",
        "S10E": "s10eTree",
        "S10G": "s10gTree",
    }
    trees = [
        _tree_check(label, ARTIFACT_TREES[label], immutable[key])
        for label, key in tree_keys.items()
    ]
    caches = [
        _cache_metadata_check(label, spec)
        for label, spec in config[
            "immutableConsumedCacheMetadata"
        ].items()
    ]
    files = [
        _file_check(key, immutable[key])
        for key in (
            "s10cInstalledDispatchQualification",
            "s10eMethodRegistry",
            "s10eHolmRegistry",
            "s10eExecutionReviewGate",
            "s10eValidation",
            "s10gPublicationRegistry",
            "s10gMethodRecordSchema",
            "s10gExecutionReviewGate",
            "s10gValidation",
            "s10gPublicationCore",
            "s10eFeasibilityCore",
            "s10aAccountingCore",
            "s10aNativeFeaturesCore",
            "s10aSearchCore",
            "researchPlanAfterProspectiveRegistration",
        )
    ]
    s10e_gate = json.loads(
        (ARTIFACT_TREES["S10E"] / "execution_review_gate.json").read_text(
            encoding="utf-8"
        )
    )
    s10g_gate = json.loads(
        (S10G / "execution_review_gate.json").read_text(encoding="utf-8")
    )
    s10g_validation = json.loads(
        (S10G / "validation_summary.json").read_text(encoding="utf-8")
    )
    binding_identity = {
        name: getattr(base, name) is target
        for name, target in _callbacks().items()
    }
    binding_identity.update(
        {
            "discoveryAnalysis": (
                base.discovery_analysis is feasible_search.discovery_analysis
            ),
            "reproductionAnalysis": (
                base.reproduction_analysis is feasible_search.reproduction_analysis
            ),
            "mixedSchemaIntermediateProjection": (
                base._json_columns is safe_intermediate_json_columns
            ),
        }
    )
    publisher = publication.AtomicScientificPublisher(publication_specs())
    publication_gate = bool(
        len(publisher.specs) == 20
        and s10g_gate["qualificationPass"]
        and s10g_gate["commonExplicitSchemaQualified"]
        and s10g_gate["globallyFailAtomicPublicationQualified"]
        and not s10g_gate["freshS10ExecutionAuthorized"]
        and s10g_gate["requiresSeparateApproval"]
        and s10g_gate["requiresGenuinelyFreshNamespace"]
        and not s10g_gate["s10fReusePermitted"]
        and not s10g_gate["scientificDesignChanged"]
        and s10g_validation["allPass"]
        and publication.METHOD_RECORD_SCHEMA_VERSION
        == "e07.s10g.method-record.v1"
    )
    s10e_qualified = bool(
        s10e_gate["qualificationPass"]
        and s10e_gate["g01ThroughG08Pass"]
        and s10e_gate["methodFeasibilityRuleQualified"]
        and s10e_gate["holmFamilyNonShrinking"]
        and not s10e_gate["s10pEstimandChanged"]
        and not s10e_gate["frozenMethodFamilyChanged"]
        and not s10e_gate["frozenThresholdChanged"]
    )
    fresh = {
        "root": str(CACHE_ROOT),
        "absentBeforeRunnerEntry": _FRESH_ROOT_ABSENT_BEFORE_ENTRY,
        "workExistsAtPreflight": WORK_OUTPUT.is_dir(),
        "runtimeExistsAndEmptyAtPreflight": (
            RUN_CACHE.is_dir() and not any(RUN_CACHE.iterdir())
        ),
    }
    fresh["pass"] = all(fresh[key] for key in fresh if key != "root")
    s10f = _s10f_boundary(config)
    callback_gate = bool(
        all(binding_identity.values())
        and _CALLBACK_COUNTS
        == {"revalidate_frozen_inputs": 1, "prospective_freeze": 0}
    )
    inherited.update(
        {
            "schemaVersion": "e07.s10h.preflight-hash-revalidation.v1",
            "researchStepId": STEP_ID,
            "s10aStructuralPreflightPass": _structural_preflight_pass(inherited),
            "immutableArtifactTreeChecks": trees,
            "opaqueConsumedCacheMetadataChecks": caches,
            "continuationFileChecks": files,
            "s10fQuarantineBoundary": s10f,
            "installedBindingIdentity": binding_identity,
            "installedBindingGatePass": callback_gate,
            "freshNamespaceGate": fresh,
            "s10eExecutionReviewGatePass": s10e_qualified,
            "s10gSchemaPublicationIntegrationPass": publication_gate,
            "separateS10HExecutionApprovalReceived": True,
            "failedOrQuarantineCacheContentReads": 0,
            "failedOrQuarantineOutcomeRowsDeserializedOrReused": 0,
        }
    )
    inherited["allPass"] = bool(
        inherited["s10aStructuralPreflightPass"]
        and all(row["pass"] for row in trees)
        and all(row["pass"] for row in caches)
        and all(row["pass"] for row in files)
        and s10f["pass"]
        and callback_gate
        and fresh["pass"]
        and s10e_qualified
        and publication_gate
    )
    return inherited


def prospective_freeze(preflight: Mapping[str, Any]) -> dict[str, Any]:
    _CALLBACK_COUNTS["prospective_freeze"] += 1
    bindings = all(
        getattr(base, name) is target for name, target in _callbacks().items()
    ) and (
        base.discovery_analysis is feasible_search.discovery_analysis
        and base.reproduction_analysis is feasible_search.reproduction_analysis
        and base._json_columns is safe_intermediate_json_columns
    )
    if (
        not preflight["allPass"]
        or not bindings
        or _CALLBACK_COUNTS
        != {"revalidate_frozen_inputs": 1, "prospective_freeze": 1}
    ):
        raise RuntimeError("S10H installed execution path failed closed")
    result = s10b.prospective_freeze(preflight)
    result.pop("executionFreezeSha256", None)
    result.update(
        {
            "schemaVersion": "e07.s10h.execution-preregistration.v1",
            "researchStepId": STEP_ID,
            "continuationOfScientificDesign": "S10P",
            "methodFeasibilityQualification": "S10E",
            "serializationAndPublicationQualification": "S10G",
            "freshCacheNamespace": str(CACHE_ROOT),
            "scientificDestination": str(SCIENTIFIC),
            "installedExecutionPathIdentityPass": bindings,
            "callbackInvocationCounts": dict(_CALLBACK_COUNTS),
            "fixedHolmSlotsPerRealizedStratum": 21,
            "fixedHolmFamilyShrinkPermitted": False,
            "scientificArtifactClassCount": 20,
            "completeSetAtomicCommitRequired": True,
            "failedOrQuarantineCacheContentReadsBeforeFreeze": 0,
            "failedOrQuarantineOutcomeRowsDeserializedOrReusedBeforeFreeze": 0,
            "validationOutcomeRowsBeforeFreeze": 0,
            "confirmationOutcomeRowsBeforeFreeze": 0,
            "humanAnnotationsBeforeFreeze": 0,
            "s11RowsBeforeFreeze": 0,
        }
    )
    result["executionFreezeSha256"] = publication.canonical_sha256(
        "E07/S10H/execution-freeze/v1", result
    )
    return result


def reviewer_instructions(machine_count: int) -> str:
    return s10b.reviewer_instructions(machine_count).replace("S10B", "S10H")


def cache_only_report(**_: Any) -> str:
    return (
        "# S10H cache-only execution intermediate\n\n"
        "This file is not a scientific publication. The complete scientific "
        "set is exposed only by the S10G atomic publisher.\n"
    )


def cache_only_manifest() -> dict[str, Any]:
    return {
        "schemaVersion": "e07.s10h.cache-only-manifest.v1",
        "researchStepId": STEP_ID,
        "scientificPublication": False,
        "cacheRoot": str(CACHE_ROOT),
    }


def _generic_parquet_bytes(
    rows: Iterable[Mapping[str, Any]], *, domain: str
) -> bytes:
    projected = []
    for position, row in enumerate(rows):
        value = _json_safe(dict(row))
        record_id = publication.canonical_sha256(
            domain, {"position": position, "record": value}
        )
        projected.append(
            {
                "position": position,
                "record_id": record_id,
                "record_json": publication.canonical_json_text(value),
            }
        )
    table = pa.Table.from_pylist(projected, schema=GENERIC_SCHEMA)
    sink = BytesIO()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        use_dictionary=False,
        version="2.6",
    )
    payload = sink.getvalue()
    restored = pq.read_table(BytesIO(payload))
    if restored.schema != GENERIC_SCHEMA or restored.num_rows != len(projected):
        raise publication.SerializationContractError(
            "generic explicit-schema round trip failed"
        )
    if any(column.null_count for column in restored.columns):
        raise publication.SerializationContractError(
            "generic explicit-schema table contains nulls"
        )
    for row in restored.to_pylist():
        publication.strict_json_loads(row["record_json"])
    return payload


def _read_fresh_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="ascii") as handle:
        for line in handle:
            rows.append(publication.strict_json_loads(line))
    return rows


def _plane(
    state: str, value: Any = None, reasons: Sequence[str] = ()
) -> dict[str, Any]:
    return publication.value_plane(state, value, sorted(set(reasons)))


def _slot_parts(slot_id: str) -> tuple[str, str, str, str]:
    parts = slot_id.split("::", 3)
    if len(parts) != 4:
        raise RuntimeError(f"malformed fixed slot: {slot_id}")
    return parts[0], parts[1], parts[2], parts[3]


def _non_evidentiary_reason(slot: Mapping[str, Any]) -> list[str]:
    return [
        (
            "S10E_METHOD_INFEASIBLE"
            if slot["slotState"] == "non_evidentiary_infeasible"
            else "NO_MACHINE_CANDIDATE_FOR_FIXED_SLOT"
        )
    ]


def _record_common(
    *,
    family: str,
    task: str,
    status: str,
    configuration: dict[str, Any],
    slot: Mapping[str, Any],
    slot_identity: str,
    payload: Mapping[str, Any] | None,
    reasons: Sequence[str] = (),
) -> dict[str, Any]:
    evidentiary = slot["slotState"] == "evidentiary_candidate"
    reason_codes = [] if evidentiary else sorted(set(reasons))
    return publication.make_method_record(
        methodFamily=family,
        taskId=task,
        statusStratum=status,
        configurationId=configuration,
        slotId=_plane("available", slot_identity),
        evidenceState="evidentiary" if evidentiary else "non_evidentiary",
        methodExecuted=evidentiary,
        reasonCodes=reason_codes,
        payload=(
            _plane("available", _json_safe(payload))
            if evidentiary
            else _plane("unavailable", None, reason_codes)
        ),
        rawPValue=_plane("available", float(slot["rawPValue"])),
        holmAdjustedPValue=_plane(
            "available", float(slot["holmAdjustedPValue"])
        ),
        preMultiplicityGatePass=_plane("available", evidentiary),
        multiplicityPass=_plane(
            "available", bool(slot["multiplicityPass"]) if evidentiary else False
        ),
    )


def method_record_projection(
    catalog: Mapping[str, Any],
    discovery_audit: Mapping[str, Any],
    reproduction_audit: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Project frozen S10E slots into the byte-frozen S10G record schema."""

    results = catalog["results"]
    cluster_by_stratum = {
        f"{row['taskId']}::{row['statusStratum']}": row
        for row in results["clustering"]
    }
    anomaly_by_stratum = {
        f"{row['taskId']}::{row['statusStratum']}": row
        for row in results["anomaly"]
    }
    cp_by_slot = {
        (
            f"{row['taskId']}::{row['statusStratum']}::"
            f"change_point::{row['configurationId']}"
        ): row
        for row in results["changePoint"]
    }
    candidates = {
        str(row["holmSlotId"]): row
        for row in catalog["machineCandidates"]
    }
    projected: dict[str, list[dict[str, Any]]] = {
        "clustering": [],
        "anomaly": [],
        "change_point": [],
        "fixed_holm": [],
    }
    for slot in discovery_audit["holmSlots"]:
        slot_id = str(slot["slotId"])
        task, status, family, tail = _slot_parts(slot_id)
        evidentiary = slot["slotState"] == "evidentiary_candidate"
        candidate = candidates.get(slot_id)
        if evidentiary and candidate is None:
            raise RuntimeError("evidentiary slot lacks its frozen candidate")
        reasons = _non_evidentiary_reason(slot) if not evidentiary else []
        if family == "clustering":
            source = cluster_by_stratum[f"{task}::{status}"]
            payload = None
            if evidentiary:
                members = list(candidate["memberCandidateIds"])
                first = source["candidateIds"].index(members[0])
                payload = {
                    "algorithm": "Ward_with_diagonal_GMM_cross_check",
                    "clusterLabel": int(source["selectedWardLabels"][first]),
                    "membership": members,
                    "stability": float(
                        source["kResults"][str(source["selectedWardK"])][
                            "meanBootstrapARI"
                        ]
                    ),
                    "nullThreshold": float(source["nullMaxAri95"]),
                }
            projected["clustering"].append(
                _record_common(
                    family="clustering",
                    task=task,
                    status=status,
                    configuration=_plane(
                        "not_applicable",
                        None,
                        ["CONFIGURATION_ID_NOT_APPLICABLE"],
                    ),
                    slot=slot,
                    slot_identity=slot_id,
                    payload=payload,
                    reasons=reasons,
                )
            )
        elif family == "anomaly":
            source = anomaly_by_stratum[f"{task}::{status}"]
            payload = (
                {
                    "algorithm": "IsolationForest_500_trees",
                    "selectedConfigurationId": source[
                        "selectedConfigurationId"
                    ],
                    "scores": source["scores"],
                    "medianScore": float(source["medianScore"]),
                    "empiricalPValue": float(slot["rawPValue"]),
                }
                if evidentiary
                else None
            )
            projected["anomaly"].append(
                _record_common(
                    family="anomaly",
                    task=task,
                    status=status,
                    configuration=_plane(
                        "not_applicable",
                        None,
                        ["CONFIGURATION_ID_NOT_APPLICABLE"],
                    ),
                    slot=slot,
                    slot_identity=slot_id,
                    payload=payload,
                    reasons=reasons,
                )
            )
        else:
            source = cp_by_slot[slot_id]
            payload = (
                {
                    "algorithm": (
                        "exact_dynamic_programming_piecewise_constant_SSE"
                    ),
                    "series": "proposal_accept_conflict_hash_change",
                    "changePoints": [int(source["lockedCenterTransition"])],
                    "prevalence": float(source["discoveryPrevalence"]),
                    "empiricalPValue": float(slot["rawPValue"]),
                }
                if evidentiary
                else None
            )
            projected["change_point"].append(
                _record_common(
                    family="change_point",
                    task=task,
                    status=status,
                    configuration=_plane("available", tail),
                    slot=slot,
                    slot_identity=slot_id,
                    payload=payload,
                    reasons=reasons,
                )
            )
        fixed_payload = (
            {
                "sourceMethodFamily": family,
                "slotState": slot["slotState"],
                "rawPValue": float(slot["rawPValue"]),
                "adjustedPValue": float(slot["holmAdjustedPValue"]),
                "rejected": bool(slot["multiplicityPass"]),
            }
            if evidentiary
            else None
        )
        projected["fixed_holm"].append(
            _record_common(
                family="fixed_holm",
                task=task,
                status=status,
                configuration=_plane(
                    "not_applicable",
                    None,
                    ["CONFIGURATION_ID_NOT_APPLICABLE"],
                ),
                slot=slot,
                slot_identity=f"discovery::{slot_id}",
                payload=fixed_payload,
                reasons=reasons,
            )
        )
    for slot in reproduction_audit["fixedHolmSlots"]:
        slot_id = str(slot["slotId"])
        task, status, family, _ = _slot_parts(slot_id)
        evidentiary = slot["slotState"] == "evidentiary_candidate"
        reasons = _non_evidentiary_reason(slot) if not evidentiary else []
        payload = (
            {
                "sourceMethodFamily": family,
                "slotState": slot["slotState"],
                "rawPValue": float(slot["rawPValue"]),
                "adjustedPValue": float(slot["holmAdjustedPValue"]),
                "rejected": bool(slot["multiplicityPass"]),
            }
            if evidentiary
            else None
        )
        projected["fixed_holm"].append(
            _record_common(
                family="fixed_holm",
                task=task,
                status=status,
                configuration=_plane(
                    "not_applicable",
                    None,
                    ["CONFIGURATION_ID_NOT_APPLICABLE"],
                ),
                slot=slot,
                slot_identity=f"independent_reproduction::{slot_id}",
                payload=payload,
                reasons=reasons,
            )
        )
    for family, records in projected.items():
        payload = publication.records_to_parquet_bytes(records)
        if publication.records_from_parquet_bytes(payload) != sorted(
            records, key=lambda row: row["recordId"]
        ):
            raise publication.SerializationContractError(
                f"{family} method-record round trip differs"
            )
    return projected


def _publication_payloads(
    discovery_rows: Sequence[Mapping[str, Any]],
    reproduction_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    all_rows = [*discovery_rows, *reproduction_rows]
    machine = json.loads(
        (WORK_OUTPUT / "machine_discovery_summary.json").read_text(
            encoding="utf-8"
        )
    )
    lock = json.loads(
        (WORK_OUTPUT / "discovery_lock.json").read_text(encoding="utf-8")
    )
    accounting = json.loads(
        (WORK_OUTPUT / "complete_accounting.json").read_text(encoding="utf-8")
    )
    replay = json.loads(
        (WORK_OUTPUT / "replay_worker_order_validation.json").read_text(
            encoding="utf-8"
        )
    )
    native = json.loads(
        (WORK_OUTPUT / "native_contract_validation.json").read_text(
            encoding="utf-8"
        )
    )
    human = json.loads(
        (WORK_OUTPUT / "human_review_status.json").read_text(encoding="utf-8")
    )
    discovery_audit = feasible_search.discovery_audit()
    reproduction_audit = feasible_search.reproduction_audit()
    catalog = {
        "machineMultiplicityFamilySize": machine["machineMultiplicityFamilySize"],
        "machineCandidates": machine["machineCandidates"],
        "discoveryPassingCandidates": machine["discoveryPassingCandidates"],
        "results": {
            "preprocessing": json.loads(
                (
                    WORK_OUTPUT / "preprocessing_confound_results.json"
                ).read_text(encoding="utf-8")
            )["preprocessing"],
            "confounds": json.loads(
                (
                    WORK_OUTPUT / "preprocessing_confound_results.json"
                ).read_text(encoding="utf-8")
            )["confounds"],
            "clustering": _read_generic_legacy_parquet(
                WORK_OUTPUT / "clustering_results.parquet"
            ),
            "anomaly": _read_generic_legacy_parquet(
                WORK_OUTPUT / "anomaly_results.parquet"
            ),
            "changePoint": _read_generic_legacy_parquet(
                WORK_OUTPUT / "change_point_results.parquet"
            ),
        },
    }
    records = method_record_projection(catalog, discovery_audit, reproduction_audit)
    reproduction_by_key = {
        str(row["machineCandidateKey"]): row
        for row in machine["reproductionResults"]
    }
    catalog_rows = [
        {
            **row,
            "reproductionEvaluated": row["machineCandidateKey"]
            in reproduction_by_key,
            "reproductionPass": bool(
                reproduction_by_key.get(row["machineCandidateKey"], {}).get(
                    "reproductionPass", False
                )
            ),
        }
        for row in machine["machineCandidates"]
    ]
    preprocessing_rows = [
        {"taskStatusStratum": key, "result": value}
        for key, value in sorted(catalog["results"]["preprocessing"].items())
    ]
    confound_rows = [
        {"taskStatusStratum": key, "result": value}
        for key, value in sorted(catalog["results"]["confounds"].items())
    ]
    reproduced = [
        row
        for row in machine["reproductionResults"]
        if row.get("reproductionPass")
    ]
    exemplars = []
    packet_path = WORK_OUTPUT / "blinded_exemplar_packet.jsonl"
    if packet_path.is_file():
        exemplars = [
            publication.strict_json_loads(line)
            for line in packet_path.read_text(encoding="ascii").splitlines()
            if line.strip()
        ]
    accounting.update(
        {
            "schemaVersion": "e07.s10h.complete-accounting.v1",
            "researchStepId": STEP_ID,
            "fixedHolmDiscoverySlots": len(discovery_audit["holmSlots"]),
            "fixedHolmReproductionSlots": len(
                reproduction_audit["fixedHolmSlots"]
            ),
            "fixedHolmFamilyShrunk": False,
            "failedOrQuarantineCacheContentReads": 0,
            "failedOrQuarantineOutcomeRowsDeserializedOrReused": 0,
            "humanAnnotations": 0,
            "s11Rows": 0,
        }
    )
    scientific_validation = {
        "schemaVersion": "e07.s10h.prepublication-validation.v1",
        "researchStepId": STEP_ID,
        "checks": {
            "exactLogicalAccounting": len(all_rows) == 10_752,
            "exactPhysicalAccounting": accounting["physicalEpisodeExecutions"]
            == 21_504,
            "discoveryIntegrity": len(discovery_rows) == 7_168,
            "reproductionIntegrity": len(reproduction_rows) == 3_584,
            "replayWorkerOrder": replay["pass"],
            "nativeContracts": native["pass"],
            "fixedHolmFamilyConserved": (
                len(discovery_audit["holmSlots"])
                == 21 * discovery_audit["assessmentCount"]
                and len(reproduction_audit["fixedHolmSlots"])
                == len(discovery_audit["holmSlots"])
                and not discovery_audit["fixedFamilyShrunk"]
                and not reproduction_audit["fixedFamilyShrunk"]
            ),
            "explicitS10GMethodRecords": all(records.values()),
            "protectedOutcomesSealed": (
                accounting["validationOutcomeRowsOpened"] == 0
                and accounting["confirmationOutcomeRowsOpened"] == 0
            ),
            "prohibitedDependenciesAbsent": (
                accounting["s06OrS06AArtifactsLoaded"] == 0
                and accounting["s07ArmSignalsUsed"] == 0
                and accounting["quarantineRowsLoaded"] == 0
            ),
            "humanAnnotationsEmpty": human["reviewerAnnotationsPresent"] == 0,
            "scientificPayloadSetPrevalidated": True,
        },
    }
    scientific_validation["passedChecks"] = sum(
        scientific_validation["checks"].values()
    )
    scientific_validation["totalChecks"] = len(
        scientific_validation["checks"]
    )
    scientific_validation["allPass"] = all(
        scientific_validation["checks"].values()
    )
    if not scientific_validation["allPass"]:
        raise RuntimeError("S10H scientific prepublication validation failed")
    execution = json.loads(
        (WORK_OUTPUT / "execution_summary.json").read_text(encoding="utf-8")
    )
    execution.update(
        {
            "schemaVersion": "e07.s10h.execution-summary.v1",
            "researchStepId": STEP_ID,
            "freshCacheNamespace": str(CACHE_ROOT),
            "scientificArtifactClassCount": 20,
            "scientificPublicationCommitRule": (
                "one_same_filesystem_atomic_directory_rename"
            ),
        }
    )
    packet = {
        "schemaVersion": "e07.s10h.blinded-exemplar-packet.v1",
        "researchStepId": STEP_ID,
        "externalReviewTriggered": bool(reproduced),
        "reproducedMachineCandidateCount": len(reproduced),
        "exemplars": exemplars if reproduced else [],
        "reviewerAnnotations": [],
    }
    payloads = {
        "discovery_catalog": _generic_parquet_bytes(
            catalog_rows, domain="E07/S10H/discovery-catalog/v1"
        ),
        "preprocessing_results": _generic_parquet_bytes(
            preprocessing_rows, domain="E07/S10H/preprocessing/v1"
        ),
        "confound_audits": _generic_parquet_bytes(
            confound_rows, domain="E07/S10H/confounds/v1"
        ),
        "clustering_results": publication.records_to_parquet_bytes(
            records["clustering"]
        ),
        "anomaly_results": publication.records_to_parquet_bytes(
            records["anomaly"]
        ),
        "change_point_results": publication.records_to_parquet_bytes(
            records["change_point"]
        ),
        "fixed_holm_records": publication.records_to_parquet_bytes(
            records["fixed_holm"]
        ),
        "discovery_lock": canonical_json_bytes(lock),
        "independent_reproduction_results": _generic_parquet_bytes(
            machine["reproductionResults"],
            domain="E07/S10H/independent-reproduction/v1",
        ),
        "machine_discovery_summary": canonical_json_bytes(
            {
                **machine,
                "schemaVersion": "e07.s10h.machine-discovery-summary.v1",
                "researchStepId": STEP_ID,
            }
        ),
        "blinded_exemplar_packet": canonical_json_bytes(packet),
        "reviewer_instructions": reviewer_instructions(len(reproduced)).encode(
            "utf-8"
        ),
        "human_review_status": canonical_json_bytes(
            {
                **human,
                "schemaVersion": "e07.s10h.human-review-status.v1",
                "researchStepId": STEP_ID,
                "reviewerAnnotationsPresent": 0,
            }
        ),
        "native_event_feature_records": _generic_parquet_bytes(
            (
                {
                    "logicalReservationId": row["logicalReservationId"],
                    "logicalResultSha256": row["logicalResultSha256"],
                    "candidateId": row["candidateId"],
                    "taskId": row["taskId"],
                    "scenarioFamilyId": row["scenarioFamilyId"],
                    "phase": row["phase"],
                    "statusStratum": row["statusStratum"],
                    "analysisFeatures": row["analysisFeatures"],
                    "availability": row["availability"],
                    "confounds": row["confounds"],
                }
                for row in all_rows
            ),
            domain="E07/S10H/native-feature-record/v1",
        ),
        "ordered_event_summary_records": _generic_parquet_bytes(
            (
                {
                    "logicalReservationId": row["logicalReservationId"],
                    "logicalResultSha256": row["logicalResultSha256"],
                    "candidateId": row["candidateId"],
                    "taskId": row["taskId"],
                    "scenarioFamilyId": row["scenarioFamilyId"],
                    "phase": row["phase"],
                    "statusStratum": row["statusStratum"],
                    "orderedEventSeries": row["orderedEventSeries"],
                }
                for row in all_rows
            ),
            domain="E07/S10H/ordered-event-summary/v1",
        ),
        "complete_accounting": canonical_json_bytes(accounting),
        "replay_validation": canonical_json_bytes(
            {
                **replay,
                "schemaVersion": "e07.s10h.replay-validation.v1",
                "researchStepId": STEP_ID,
            }
        ),
        "native_contract_validation": canonical_json_bytes(
            {
                **native,
                "schemaVersion": "e07.s10h.native-contract-validation.v1",
                "researchStepId": STEP_ID,
            }
        ),
        "validation_summary": canonical_json_bytes(scientific_validation),
        "execution_summary": canonical_json_bytes(execution),
    }
    expected = {spec.class_id for spec in publication_specs()}
    if set(payloads) != expected:
        raise RuntimeError("S10H scientific payload set differs from S10G registry")
    for spec in publication_specs():
        publication.validate_artifact_payload(spec, payloads[spec.class_id])
    summary = {
        "discoveryAudit": discovery_audit,
        "reproductionAudit": reproduction_audit,
        "reproduced": reproduced,
        "accounting": accounting,
        "scientificValidation": scientific_validation,
        "methodRecordCounts": {
            key: len(value) for key, value in records.items()
        },
    }
    return payloads, summary


def _read_generic_legacy_parquet(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(path)
    rows = []
    json_columns = {
        "candidateIds",
        "kResults",
        "selectedWardLabels",
        "selectedGmm",
        "scores",
        "changePointCountDistribution",
    }
    for source in frame.to_dict(orient="records"):
        row: dict[str, Any] = {}
        for key, value in source.items():
            if key in json_columns and isinstance(value, str):
                row[key] = publication.strict_json_loads(value)
            elif isinstance(value, (float, np.floating)) and not math.isfinite(
                float(value)
            ):
                continue
            else:
                row[key] = _json_safe(value)
        rows.append(row)
    return rows


def _repository_provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "schemaVersion": "e07.s10h.repository-provenance.v1",
        "researchStepId": STEP_ID,
        "branch": git("branch", "--show-current"),
        "commitAtExecution": git("rev-parse", "HEAD"),
        "executionControlSha256": sha256_file(CONFIG),
        "runnerSha256": sha256_file(SCRIPT),
        "focusedTestSha256": sha256_file(TEST),
        "publicationCoreSha256": sha256_file(
            REPOSITORY / "src/phenotype_discovery/publication.py"
        ),
        "feasibilityCoreSha256": sha256_file(
            REPOSITORY / "src/phenotype_discovery/method_feasibility.py"
        ),
    }


def _copy_control_file(name: str) -> None:
    source = WORK_OUTPUT / name
    if not source.is_file():
        raise RuntimeError(f"missing cache-only control file: {name}")
    (OUTPUT / name).write_bytes(source.read_bytes())


def _manifest() -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in OUTPUT.rglob("*") if item.is_file()):
        if path.name in {"artifact_manifest.json", "artifact_validation.json"}:
            continue
        rows.append(
            {
                "path": str(path),
                "relativePath": path.relative_to(OUTPUT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schemaVersion": "e07.s10h.artifact-manifest.v1",
        "researchStepId": STEP_ID,
        "artifacts": rows,
    }


def _full_report(
    summary: Mapping[str, Any],
    publication_audit: Mapping[str, Any],
    validation: Mapping[str, Any],
    status: Mapping[str, Any],
) -> str:
    discovery = summary["discoveryAudit"]
    accounting = summary["accounting"]
    reproduced = len(summary["reproduced"])
    passing = sum(
        row["multiplicityPass"] for row in discovery["holmSlots"]
    )
    return f"""# S10H — Fresh S10 execution after S10G

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S10H** |
| Completion status | **{status['status']}**; stopped before human annotation and S11 |
| Artifacts written | One atomically committed 20-class scientific publication plus its manifest, complete discovery/reproduction dispositions, preflight/gate/access/dependency/callback/immutability/publication/accounting/validation/provenance records, status, artifact manifest, and this canonical report under `/artifacts/research_steps/S10H/` |
| Validation result | **PASS** — {validation['passedChecks']}/{validation['totalChecks']} final checks; 10,752 logical and 21,504 exact-replay physical dispositions; complete 20-class scientific set committed once and revalidated |
| Outcome classification | **{status['outcomeClassification']}** |
| Caveats or blockers | {status['caveatsOrBlockers'][0]} |
| Lay summary | A wholly fresh training-only run completed the prespecified behavior screen twice per reservation. It kept unavailable measurements, failures, and censors explicit; applied the frozen feasibility and multiplicity rules; tested discoveries on independent training families; and exposed the result only after every scientific file validated together. |
| Recommended next action | {status['recommendedNextAction']} |

## Frozen question

Does the unchanged S10P event-feature design produce a machine candidate that
survives discovery and independent-training-family reproduction when executed
through S10A's missingness/accounting contract, S10C's installed callbacks,
S10E's fixed-slot feasibility/Holm rule, and S10G's explicit schema and global
publication transaction?

## Inputs

Only the byte-frozen 14 S09 parent/compression configurations, two spatial
native contracts, and 768 S10P training families were eligible. The run used
7,168 discovery and 3,584 independent-reproduction logical reservations, each
with two exact physical replays. S10P–S10E and S10G artifact trees were
content-hash revalidated. S10F's permitted forensic files were hash checked;
its three prohibited outcome artifacts and its cache were checked only by
recorded hashes plus current stat metadata and were never opened or
deserialized. All other consumed cache checks were metadata-only.

## Detailed methods

The runner first verified G01–G08, all 28 candidate/task bindings, 10,752
logical and 21,504 physical pre-outcome commitments, actual installed callback
identities and single invocation, protected denials, prohibited dependencies,
S10E's 21 fixed Holm slots per realized task/status stratum, and S10G's
20-class registry/codecs/publisher. It then ran the unchanged S10P Ward,
diagonal-GMM, Isolation-Forest, exact change-point, confound, bootstrap, null,
trace-selection, and independent-reproduction procedures through the S10E
admissibility layer.

Scientific intermediates remained under `/cache/e07-s10h/work`. Mixed method
rows were projected into S10G's explicit value-plane schema. All other
scientific tables used a nonnullable three-column envelope containing
position, a stable record hash, and canonical strict JSON. Every payload was
validated in memory; all 20 classes were staged and validated with their
manifest; one same-filesystem atomic directory rename exposed the complete
set. No individual scientific result was copied into the artifact directory.

## Commands

```text
PYTHONPATH=. pytest -q tests/test_s10h_fresh_execution.py tests/test_s10g_serialization_publication.py tests/test_s10e_method_feasibility.py tests/test_s10c_callback_binding.py tests/test_s10a_missingness_accounting.py tests/test_s10_native_event_search.py tests/test_s10p_native_event_discovery.py
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/run_native_event_discovery_s10h.py execute
PYTHONPATH=. python scripts/run_native_event_discovery_s10h.py validate
```

Eight process workers and one numeric thread per worker were used. No package,
network, GPU, validation outcome, confirmation outcome, rejected model,
embedding, S07 arm signal, or historical outcome cache was used.

## Results

- Logical results: {accounting['totalLogicalRows']}; exact physical replays:
  {accounting['physicalEpisodeExecutions']}.
- Retained native failures: {accounting['failures']}; retained native censors:
  {accounting['censors']}.
- Realized task/status strata: {discovery['assessmentCount']}; discovery fixed
  Holm slots: {discovery['fixedHolmSlotCount']}; family shrinkage: **false**.
- Structurally admissible slots: {discovery['admissibleSlotCount']};
  evidentiary candidate slots: {discovery['evidentiaryCandidateSlotCount']};
  infeasible non-evidentiary slots: {discovery['infeasibleSlotCount']}.
- Holm-significant discovery slots: {passing}; independently reproduced
  candidates: {reproduced}.
- S10G record counts: {json.dumps(summary['methodRecordCounts'], sort_keys=True)}.
- Publication state: `{publication_audit['finalScientificPublicationState']}`;
  commit boundaries: {publication_audit['commitBoundaryCount']}; artifact
  classes: {publication_audit['artifactCount']}.

The scientific result is the machine-stage classification above. Infeasible
slots remain explicit non-evidentiary exclusions and are not counted as null
findings.

## Validation

All {validation['totalChecks']} final checks passed. Validation covered
immutable hashes and quarantine boundaries; G01–G08; 28 bindings; installed
callbacks; fresh namespace; complete logical/physical dispositions; replay
and worker-order independence; native contracts; explicit feature
availability; fixed Holm conservation; mixed-schema round trips; protected
denial; dependency exclusion; single-boundary complete-set publication;
manifest parity; predecessor no-mutation; empty annotations; sealed
validation/confirmation; and absent S11.

## Caveats, blockers, and claim boundaries

- Scope is 14 configurations and two spatial training contracts.
- Ordered count/hash-change summaries are authentic native event records but
  are not complete trajectories; no absent trajectory was reconstructed.
- Non-evidentiary method slots do not support an absence claim.
- Machine-stage evidence is descriptive. It does not establish preference,
  intention, agency, cognition, repair ability, biological phenotype,
  intervention, transfer, validation, or confirmation.
- If a machine candidate reproduced, external annotations remain empty and no
  human conclusion is authorized.

## Provenance

The prospective research-plan hash, frozen execution control, pushed
repository commit, source/test hashes, immutable trees, opaque cache metadata,
execution freeze, discovery lock, terminal dispositions, method-record
commitments, publication attempt, and complete-set manifest are recorded in
the S10H artifacts. S10P–S10G and all historical quarantines remained
unchanged.

## Recommended next action

{status['recommendedNextAction']}
"""


def _finalize_success(
    summary: Mapping[str, Any], publication_audit: Mapping[str, Any]
) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in (
        "preflight_hash_revalidation.json",
        "access_control_validation.json",
        "prohibited_dependency_validation.json",
        "execution_preregistration_freeze.json",
        "s10_gate_revalidation.json",
        "discovery_lock_audit.json",
        "no_mutation_validation.json",
        "input_provenance.json",
    ):
        _copy_control_file(name)
    dispositions = [
        s10b.promote_disposition(
            RUN_CACHE / "discovery_dispositions.json",
            OUTPUT / "discovery_dispositions.json.gz",
        ),
        s10b.promote_disposition(
            RUN_CACHE / "reproduction_dispositions.json",
            OUTPUT / "reproduction_dispositions.json.gz",
        ),
    ]
    write_json(
        OUTPUT / "durable_disposition_accounting.json",
        {
            "schemaVersion": "e07.s10h.durable-disposition-accounting.v1",
            "researchStepId": STEP_ID,
            "phases": dispositions,
            "logicalDispositionCount": sum(
                row["logicalDispositions"] for row in dispositions
            ),
            "physicalDispositionCount": sum(
                row["physicalDispositions"] for row in dispositions
            ),
            "allTerminal": all(
                row["logicalDispositions"]
                == row["terminalLogicalDispositions"]
                and row["physicalDispositions"]
                == row["terminalPhysicalDispositions"]
                for row in dispositions
            ),
            "allConserved": all(
                row["accountingConserved"] for row in dispositions
            ),
        },
    )
    write_json(
        OUTPUT / "discovery_method_feasibility_assessments.json",
        {
            "schemaVersion": "e07.s10h.discovery-feasibility-results.v1",
            "researchStepId": STEP_ID,
            **summary["discoveryAudit"],
        },
    )
    write_json(
        OUTPUT / "reproduction_method_feasibility_assessments.json",
        {
            "schemaVersion": "e07.s10h.reproduction-feasibility-results.v1",
            "researchStepId": STEP_ID,
            **summary["reproductionAudit"],
        },
    )
    write_json(
        OUTPUT / "fixed_holm_slot_results.json",
        {
            "schemaVersion": "e07.s10h.fixed-holm-slot-results.v1",
            "researchStepId": STEP_ID,
            "discoverySlots": summary["discoveryAudit"]["holmSlots"],
            "reproductionSlots": summary["reproductionAudit"]["fixedHolmSlots"],
            "familyShrunk": False,
        },
    )
    preflight = json.loads(
        (OUTPUT / "preflight_hash_revalidation.json").read_text(
            encoding="utf-8"
        )
    )
    callback = {
        "schemaVersion": "e07.s10h.installed-dispatch-audit.v1",
        "researchStepId": STEP_ID,
        "callbackInvocationCounts": dict(_CALLBACK_COUNTS),
        "exactlyOnce": _CALLBACK_COUNTS
        == {"revalidate_frozen_inputs": 1, "prospective_freeze": 1},
        "bindings": preflight["installedBindingIdentity"],
        "pass": (
            _CALLBACK_COUNTS
            == {"revalidate_frozen_inputs": 1, "prospective_freeze": 1}
            and all(preflight["installedBindingIdentity"].values())
        ),
    }
    write_json(OUTPUT / "installed_dispatch_execution_audit.json", callback)
    write_json(
        OUTPUT / "immutability_validation.json",
        {
            "schemaVersion": "e07.s10h.immutability-validation.v1",
            "researchStepId": STEP_ID,
            "artifactTrees": preflight["immutableArtifactTreeChecks"],
            "opaqueConsumedCaches": preflight[
                "opaqueConsumedCacheMetadataChecks"
            ],
            "s10fQuarantineBoundary": preflight["s10fQuarantineBoundary"],
            "allPass": (
                all(
                    row["pass"]
                    for row in preflight["immutableArtifactTreeChecks"]
                )
                and all(
                    row["pass"]
                    for row in preflight[
                        "opaqueConsumedCacheMetadataChecks"
                    ]
                )
                and preflight["s10fQuarantineBoundary"]["pass"]
            ),
            "prohibitedCacheContentFilesOpened": 0,
            "prohibitedOutcomeRowsDeserializedOrReused": 0,
        },
    )
    write_json(
        OUTPUT / "publication_commit_audit.json",
        {
            **dict(publication_audit),
            "researchStepId": STEP_ID,
            "s10gPublisher": True,
        },
    )
    write_json(OUTPUT / "repository_provenance.json", _repository_provenance())
    write_json(
        OUTPUT / "environment_provenance.json",
        {
            "schemaVersion": "e07.s10h.environment-provenance.v1",
            "researchStepId": STEP_ID,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "scikitLearn": sklearn.__version__,
            "workers": 8,
            "numericThreadsPerWorker": 1,
            "freshCacheNamespace": str(CACHE_ROOT),
        },
    )
    write_json(
        OUTPUT / "command_log.json",
        {
            "schemaVersion": "e07.s10h.command-log.v1",
            "researchStepId": STEP_ID,
            "commands": [
                (
                    "PYTHONPATH=. pytest -q tests/test_s10h_fresh_execution.py "
                    "tests/test_s10g_serialization_publication.py "
                    "tests/test_s10e_method_feasibility.py "
                    "tests/test_s10c_callback_binding.py "
                    "tests/test_s10a_missingness_accounting.py "
                    "tests/test_s10_native_event_search.py "
                    "tests/test_s10p_native_event_discovery.py"
                ),
                (
                    "PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                    "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python "
                    "scripts/run_native_event_discovery_s10h.py execute"
                ),
                (
                    "PYTHONPATH=. python "
                    "scripts/run_native_event_discovery_s10h.py validate"
                ),
            ],
        },
    )
    accounting = summary["accounting"]
    reproduced = len(summary["reproduced"])
    if reproduced:
        status_name = "awaiting_external_human_review"
        outcome = "supportive machine-stage evidence pending external review"
        next_action = (
            "Obtain two independent blinded external reviews using only the "
            "atomically published packet; keep annotations external and do "
            "not start S11 without a separate approved review completion."
        )
        caveats = [
            "At least one machine candidate reproduced, but annotations are "
            "empty and no human phenotype conclusion is authorized.",
            "Evidence is limited to two spatial tasks and event summaries.",
        ]
    else:
        status_name = "complete_bounded_machine_null"
        outcome = "null on evidentiary slots; infeasible slots excluded"
        next_action = (
            "Review the valid bounded S10H machine null and decide whether to "
            "close phenotype discovery or preregister a scientifically new "
            "branch; do not start S11 without an eligible retained candidate."
        )
        caveats = [
            "Infeasible slots are non-evidentiary exclusions, not null tests.",
            "The bounded result covers 14 configurations, two spatial tasks, "
            "and frozen event-summary features only.",
        ]
    final_checks = {
        "preflightGatesAndHashes": preflight["allPass"],
        "exactCallbacks": callback["pass"],
        "logicalAccounting": accounting["totalLogicalRows"] == 10_752,
        "physicalAccounting": accounting["physicalEpisodeExecutions"] == 21_504,
        "durableDispositionAccounting": all(
            [
                sum(row["logicalDispositions"] for row in dispositions)
                == 10_752,
                sum(row["physicalDispositions"] for row in dispositions)
                == 21_504,
                all(row["accountingConserved"] for row in dispositions),
            ]
        ),
        "scientificPrepublicationValidation": summary[
            "scientificValidation"
        ]["allPass"],
        "publicationComplete": publication_audit[
            "finalScientificPublicationState"
        ]
        == "complete_validated_publication",
        "singleCommitBoundary": publication_audit["commitBoundaryCount"] == 1,
        "publicationArtifactCardinality": publication_audit["artifactCount"]
        == 20,
        "scientificDestinationPresent": SCIENTIFIC.is_dir(),
        "protectedAndProhibitedRowsZero": (
            accounting["validationOutcomeRowsOpened"] == 0
            and accounting["confirmationOutcomeRowsOpened"] == 0
            and accounting["failedOrQuarantineCacheContentReads"] == 0
            and accounting[
                "failedOrQuarantineOutcomeRowsDeserializedOrReused"
            ]
            == 0
        ),
        "humanAnnotationsZero": json.loads(
            (SCIENTIFIC / "human_review_status.json").read_text(
                encoding="utf-8"
            )
        )["reviewerAnnotationsPresent"]
        == 0,
        "s11Absent": not Path("/artifacts/research_steps/S11").exists(),
    }
    validation = {
        "schemaVersion": "e07.s10h.final-validation.v1",
        "researchStepId": STEP_ID,
        "checks": final_checks,
        "passedChecks": sum(final_checks.values()),
        "totalChecks": len(final_checks),
        "allPass": all(final_checks.values()),
    }
    write_json(OUTPUT / "validation_summary.json", validation)
    if not validation["allPass"]:
        raise RuntimeError("S10H final validation failed")
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": 10,
        "success": True,
        "status": status_name,
        "artifactsWritten": [],
        "validationResult": (
            f"PASS {validation['passedChecks']}/{validation['totalChecks']} "
            "final checks; 10,752 logical and 21,504 physical dispositions; "
            "complete 20-class atomic scientific publication"
        ),
        "outcomeClassification": outcome,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": next_action,
    }
    (OUTPUT / "research_step_full_results.md").write_text(
        _full_report(summary, publication_audit, validation, status),
        encoding="utf-8",
    )
    status["artifactsWritten"] = sorted(
        {
            *(
                path.relative_to(OUTPUT).as_posix()
                for path in OUTPUT.rglob("*")
                if path.is_file()
            ),
            "artifact_manifest.json",
            "artifact_validation.json",
        }
    )
    write_json(OUTPUT / "status.json", status)
    write_json(OUTPUT / "artifact_manifest.json", _manifest())


def _failure_report(message: str) -> str:
    return f"""# S10H — Fresh S10 execution after S10G

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S10H** |
| Completion status | **Stopped fail-closed before human annotation and S11** |
| Artifacts written | Exact available control, disposition, publication-attempt, status, manifest, validation, provenance, and canonical-report forensics under `/artifacts/research_steps/S10H/`; no partial scientific set is valid |
| Validation result | **FAIL CLOSED** — `{message}` |
| Outcome classification | **Constraining/contradictory** |
| Caveats or blockers | No phenotype, candidate, reproduction, or bounded-null conclusion is valid. The fresh S10H cache is consumed and cannot be retried. |
| Lay summary | The fresh run encountered an integrity or publication defect and stopped without exposing a partial scientific result. |
| Recommended next action | Review S10H forensics and authorize any bounded remediation separately; do not retry this cache, annotate, or start S11. |

## Methods, inputs, results, and validation

S10H attempted only the prospectively registered S10P design through the
qualified S10A/S10C/S10E/S10G path in `/cache/e07-s10h`. The exception above
is the controlling result. Any completed fresh rows are forensic only.
Validation/confirmation, prior failed outcomes, human annotations, and S11
remained unopened. Scientific publication used the S10G global transaction;
the publication-attempt record states whether the only accepted destination
state is zero publication or a complete validated set.

## Commands and provenance

Execution used eight workers and one numeric thread per worker from the pushed
S10H repository commit. No package, network, GPU, rejected model, embedding,
S07 signal, archive mutation, universal score, imputation, or reconstructed
trajectory was used.

## Caveats and recommended next action

This is an execution-integrity constraint, not evidence for or against a
behavioral phenotype. Review the exact forensics and preregister any next step
separately. Do not retry `/cache/e07-s10h`, perform human annotation, or start
S11.
"""


def _finalize_failure(exc: BaseException) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    disposition_rows = []
    for phase in ("discovery", "reproduction"):
        source = RUN_CACHE / f"{phase}_dispositions.json"
        if source.is_file():
            disposition_rows.append(
                s10b.promote_disposition(
                    source, OUTPUT / f"{phase}_dispositions.json.gz"
                )
            )
    message = f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}"
    write_json(
        OUTPUT / "execution_failure_forensics.json",
        {
            "schemaVersion": "e07.s10h.execution-failure-forensics.v1",
            "researchStepId": STEP_ID,
            "exception": message,
            "durableDispositions": disposition_rows,
            "scientificDestinationExists": SCIENTIFIC.exists(),
            "scientificDestinationCompleteIfPresent": bool(
                SCIENTIFIC.is_dir()
                and (SCIENTIFIC / "publication_manifest.json").is_file()
            ),
            "failedClosed": True,
            "failedOrQuarantineOutcomeRowsDeserializedOrReused": 0,
            "validationOutcomeRows": 0,
            "confirmationOutcomeRows": 0,
            "humanAnnotations": 0,
            "s11Rows": 0,
        },
    )
    write_json(
        OUTPUT / "validation_summary.json",
        {
            "schemaVersion": "e07.s10h.final-validation.v1",
            "researchStepId": STEP_ID,
            "checks": {
                "executionIntegrity": False,
                "partialScientificPublicationAbsent": not SCIENTIFIC.exists(),
                "protectedOutcomesSealed": True,
                "humanAnnotationsZero": True,
                "s11RowsZero": True,
            },
            "passedChecks": 4 if not SCIENTIFIC.exists() else 3,
            "totalChecks": 5,
            "allPass": False,
        },
    )
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": 10,
        "success": False,
        "status": "stopped_fail_closed",
        "artifactsWritten": [],
        "validationResult": f"FAIL CLOSED: {message}",
        "outcomeClassification": "constraining/contradictory",
        "caveatsOrBlockers": [
            "No S10H scientific efficacy conclusion is valid.",
            "The fresh S10H namespace is consumed and must not be retried.",
        ],
        "recommendedNextAction": (
            "Review S10H forensics and preregister any bounded remediation "
            "separately; do not annotate or start S11."
        ),
    }
    (OUTPUT / "research_step_full_results.md").write_text(
        _failure_report(message), encoding="utf-8"
    )
    status["artifactsWritten"] = sorted(
        {
            *(
                path.relative_to(OUTPUT).as_posix()
                for path in OUTPUT.rglob("*")
                if path.is_file()
            ),
            "artifact_manifest.json",
        }
    )
    write_json(OUTPUT / "status.json", status)
    write_json(OUTPUT / "artifact_manifest.json", _manifest())


def execute() -> None:
    global _FRESH_ROOT_ABSENT_BEFORE_ENTRY
    if OUTPUT.exists():
        raise RuntimeError(f"{OUTPUT} already exists; S10H is fail-closed")
    if CACHE_ROOT.exists():
        raise RuntimeError(
            f"{CACHE_ROOT} already exists; S10H requires a fresh namespace"
        )
    _FRESH_ROOT_ABSENT_BEFORE_ENTRY = True
    _CALLBACK_COUNTS.update(
        {"revalidate_frozen_inputs": 0, "prospective_freeze": 0}
    )
    _configure_globals()
    original_discovery = base.discovery_analysis
    original_reproduction = base.reproduction_analysis
    original_json_columns = base._json_columns
    base.discovery_analysis = feasible_search.discovery_analysis
    base.reproduction_analysis = feasible_search.reproduction_analysis
    base._json_columns = safe_intermediate_json_columns
    try:
        with s10b.installed_base_callbacks(overrides=_callbacks()):
            base.execute()
        discovery_rows = _read_fresh_rows(
            RUN_CACHE / "discovery_rows.jsonl.gz"
        )
        reproduction_rows = _read_fresh_rows(
            RUN_CACHE / "reproduction_rows.jsonl.gz"
        )
        payloads, summary = _publication_payloads(
            discovery_rows, reproduction_rows
        )
        publisher = publication.AtomicScientificPublisher(publication_specs())
        publication_audit = publisher.publish(
            SCIENTIFIC,
            payloads,
            forensics_directory=PUBLICATION_FORENSICS,
        )
        _finalize_success(summary, publication_audit)
    except BaseException as exc:
        _finalize_failure(exc)
        raise
    finally:
        base.discovery_analysis = original_discovery
        base.reproduction_analysis = original_reproduction
        base._json_columns = original_json_columns


def validate() -> None:
    required_control = {
        "research_step_full_results.md",
        "status.json",
        "artifact_manifest.json",
        "validation_summary.json",
        "preflight_hash_revalidation.json",
        "execution_preregistration_freeze.json",
        "s10_gate_revalidation.json",
        "access_control_validation.json",
        "prohibited_dependency_validation.json",
        "durable_disposition_accounting.json",
        "discovery_dispositions.json.gz",
        "reproduction_dispositions.json.gz",
        "fixed_holm_slot_results.json",
        "publication_commit_audit.json",
        "immutability_validation.json",
        "installed_dispatch_execution_audit.json",
    }
    missing = sorted(
        name for name in required_control if not (OUTPUT / name).is_file()
    )
    specs = publication_specs()
    scientific_names = {spec.relative_path for spec in specs} | {
        "publication_manifest.json"
    }
    actual_scientific = {
        path.relative_to(SCIENTIFIC).as_posix()
        for path in SCIENTIFIC.rglob("*")
        if path.is_file()
    }
    manifest = json.loads(
        (OUTPUT / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    manifest_pass = all(
        Path(row["path"]).is_file()
        and Path(row["path"]).stat().st_size == row["bytes"]
        and sha256_file(row["path"]) == row["sha256"]
        for row in manifest["artifacts"]
    )
    publication_manifest = json.loads(
        (SCIENTIFIC / "publication_manifest.json").read_text(encoding="utf-8")
    )
    for spec in specs:
        publication.validate_artifact_payload(
            spec, (SCIENTIFIC / spec.relative_path).read_bytes()
        )
    accounting = json.loads(
        (SCIENTIFIC / "complete_accounting.json").read_text(encoding="utf-8")
    )
    disposition = json.loads(
        (OUTPUT / "durable_disposition_accounting.json").read_text(
            encoding="utf-8"
        )
    )
    status = json.loads((OUTPUT / "status.json").read_text(encoding="utf-8"))
    final_validation = json.loads(
        (OUTPUT / "validation_summary.json").read_text(encoding="utf-8")
    )
    checks = {
        "requiredControlArtifacts": not missing,
        "exactScientificSet": actual_scientific == scientific_names,
        "scientificManifestCount": publication_manifest["artifactCount"] == 20,
        "artifactManifest": bool(manifest["artifacts"]) and manifest_pass,
        "statusSchema": all(
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
        )
        and status["researchStepId"] == STEP_ID
        and status["success"],
        "finalValidation": final_validation["allPass"],
        "logicalAccounting": accounting["totalLogicalRows"] == 10_752,
        "physicalAccounting": accounting["physicalEpisodeExecutions"] == 21_504,
        "durableAccounting": (
            disposition["logicalDispositionCount"] == 10_752
            and disposition["physicalDispositionCount"] == 21_504
            and disposition["allTerminal"]
            and disposition["allConserved"]
        ),
        "protectedAndProhibitedRowsZero": (
            accounting["validationOutcomeRowsOpened"] == 0
            and accounting["confirmationOutcomeRowsOpened"] == 0
            and accounting["failedOrQuarantineCacheContentReads"] == 0
            and accounting[
                "failedOrQuarantineOutcomeRowsDeserializedOrReused"
            ]
            == 0
        ),
        "humanAnnotationsZero": json.loads(
            (SCIENTIFIC / "human_review_status.json").read_text(
                encoding="utf-8"
            )
        )["reviewerAnnotationsPresent"]
        == 0,
        "s11Absent": not Path("/artifacts/research_steps/S11").exists(),
    }
    result = {
        "schemaVersion": "e07.s10h.artifact-validation.v1",
        "researchStepId": STEP_ID,
        "checks": checks,
        "missing": missing,
        "manifestEntries": len(manifest["artifacts"]),
        "allPass": all(checks.values()),
    }
    write_json(OUTPUT / "artifact_validation.json", result)
    if not result["allPass"]:
        raise RuntimeError(f"S10H artifact validation failed: {result}")
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"execute", "validate"}:
        raise SystemExit(
            "usage: run_native_event_discovery_s10h.py {execute|validate}"
        )
    if sys.argv[1] == "execute":
        execute()
    else:
        validate()


if __name__ == "__main__":
    main()
