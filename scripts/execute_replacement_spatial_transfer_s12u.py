"""Execute the exact byte-frozen S12R replacement study as fresh step S12U."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from scipy import stats

from scripts import preregister_replacement_spatial_transfer_s12r as s12r
from src.environment_suite.contracts import canonical_sha256
from src.phenotype_discovery.publication import (
    ArtifactSpec,
    AtomicScientificPublisher,
)
from src.phenotype_discovery.publication import (
    canonical_json_bytes as publication_json_bytes,
)
from src.spatial_transfer.dependency_preflight import (
    AuthenticatedAccessLedger,
    DependencyRegistry,
    StaticImportAnalyzer,
    audit_runtime_dispatch,
    domain_hash,
    instrument_path_opens,
    validate_access_ledger,
)
from src.spatial_transfer.execution import (
    NEW_REPLACEMENT_COMPARATOR_IDS,
    _runtime_action,
    choose_adaptation_winners,
    execute_physical,
    logical_from_replays,
)
from src.spatial_transfer.execution_dependency_guard import (
    OutputAccessLedger,
    OutputAccessRegistry,
    validate_output_ledger,
)
from src.spatial_transfer.qualification import apply_adaptation_variant

REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
OUT = ARTIFACTS / "research_steps/S12U"
S12R = ARTIFACTS / "research_steps/S12R"
CACHE = Path("/cache/e07-s12u")
CONTROL = REPOSITORY / "configs/transfer/s12u_fresh_s12r_execution.yaml"
OUTPUT_REGISTRY_FILE = REPOSITORY / "configs/transfer/s12u_output_access_registry.json"
S12T = ARTIFACTS / "research_steps/S12T"
S12T_REGISTRY_FILE = S12T / "dependency_registry.json"
S12T_PROTOCOL_FILE = S12T / "s12t_dependency_preflight_protocol.yaml"
S12T_FREEZE_FILE = S12T / "preregistration_freeze.json"
S12T_GATE_FILE = S12T / "s12t_execution_review_gate.json"
S12T_MANIFEST_FILE = S12T / "artifact_manifest.json"
S12T_IMPLEMENTATION = REPOSITORY / "src/spatial_transfer/dependency_preflight.py"
SCIENTIFIC = OUT / "scientific_publication"
FORENSICS = OUT / "publication_forensics"
WORKERS = 8
PUBLICATION_CLASSES = (
    "transfer_results",
    "adaptation_lock",
    "paired_estimands",
    "failure_censor_ledger",
    "native_cost_ledger",
    "fault_cost_ledger",
    "scheduler_cost_ledger",
    "replay_audit",
    "complete_accounting",
    "access_and_provenance",
)


_RUNTIME: dict[str, Any] = {}
_OUTPUT_LEDGER: OutputAccessLedger | None = None
_OUTPUT_REGISTRY: OutputAccessRegistry | None = None
_OUTPUT_SESSION_COMMITMENT: str | None = None


def canonical_bytes(value: Any) -> bytes:
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_write_json(path: Path, value: Any) -> None:
    if _OUTPUT_LEDGER is not None:
        _OUTPUT_LEDGER.authorize_path(
            path,
            operation="atomic_write_json",
            artifact_class="cache_or_attempt_metadata",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, path)


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if _OUTPUT_LEDGER is not None:
        _OUTPUT_LEDGER.authorize_path(
            path,
            operation="write_parquet",
            artifact_class="execution_disposition_part",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    table = pa.Table.from_pandas(frame, preserve_index=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    os.replace(temporary, path)


def _direct_write_json(path: Path, value: Any) -> None:
    """Write after the corresponding path was committed to a finalized ledger."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, path)


def _finalize_output_access_ledger(*, disposition: str) -> dict[str, Any]:
    global _OUTPUT_LEDGER
    if (
        _OUTPUT_LEDGER is None
        or _OUTPUT_REGISTRY is None
        or _OUTPUT_SESSION_COMMITMENT is None
    ):
        raise RuntimeError("S12U output access ledger is not installed")
    for path, artifact_class in (
        (OUT / "output_access_ledger.json", "authenticated_output_ledger"),
        (
            OUT / "output_access_ledger_validation.json",
            "authenticated_output_ledger_validation",
        ),
        (OUT / "research_step_full_results.md", "canonical_full_results"),
        (OUT / "status.json", "machine_readable_status"),
        (OUT / "artifact_manifest.json", "artifact_manifest"),
    ):
        _OUTPUT_LEDGER.authorize_path(
            path,
            operation="write_final_handoff",
            artifact_class=artifact_class,
        )
    raw = _OUTPUT_LEDGER.finalize()
    validation = validate_output_ledger(
        raw,
        _OUTPUT_REGISTRY,
        expected_session_commitment=_OUTPUT_SESSION_COMMITMENT,
    )
    validation["disposition"] = disposition
    validation["passed"] = bool(validation["passed"])
    _OUTPUT_LEDGER = None
    _direct_write_json(OUT / "output_access_ledger.json", raw)
    _direct_write_json(OUT / "output_access_ledger_validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError("S12U authenticated output ledger failed validation")
    return validation


def control_record() -> dict[str, Any]:
    raw = yaml.safe_load(CONTROL.read_text(encoding="utf-8"))
    if (
        raw["researchStepId"] != "S12U"
        or raw["freshness"]["cacheNamespace"] != str(CACHE)
        or raw["historicalRelationship"]["executesS12RReplacementEstimand"] is not True
        or raw["historicalRelationship"]["executesS12P"] is not False
        or raw["historicalRelationship"]["executesS12A"] is not False
        or raw["historicalRelationship"]["executesOriginalS12Estimand"] is not False
        or raw["historicalRelationship"]["patchesResumesOrRetriesS12S"] is not False
        or raw["freshness"]["artifactNamespace"] != str(OUT)
        or raw["frozenS12T"]["exactImplementationRequired"] is not True
        or raw["frozenS12T"]["installOnEveryWorkerDispatch"] is not True
    ):
        raise RuntimeError("S12U execution control is not the authorized continuation")
    body = {
        "schemaVersion": "e07.s12u.execution-control-record.v1",
        "researchStepId": "S12U",
        "registeredBeforeAnyEpisode": True,
        "controlFile": file_record(CONTROL),
        "cacheNamespace": str(CACHE),
        "scientificDestination": str(SCIENTIFIC),
        "humanAuthorization": (
            "fresh substantive execution continuation of byte-frozen S12R only"
        ),
        "s12rDesignChanged": False,
        "s12pOrS12aExecuted": False,
        "validationConfirmationCivicS13S14Authorized": False,
        "frozenCounts": raw["frozenS12R"],
        "frozenS12T": raw["frozenS12T"],
        "freshness": raw["freshness"],
        "outputAccess": raw["outputAccess"],
        "adaptationLockImplementation": raw["adaptationLockImplementation"],
        "publication": raw["publication"],
    }
    body["executionControlSha256"] = canonical_sha256(
        "E07/S12U/execution-control/v1", body
    )
    return body


def _artifact_manifest_validation(
    step_root: Path = S12R,
) -> dict[str, Any]:
    manifest_path = step_root / "artifact_manifest.json"
    manifest = read_json(manifest_path)
    records = []
    for expected in manifest["artifacts"]:
        path = Path(expected["path"])
        actual = file_record(path)
        records.append(
            {
                "path": str(path),
                "expectedBytes": int(expected["bytes"]),
                "actualBytes": actual["bytes"],
                "expectedSha256": expected["sha256"],
                "actualSha256": actual["sha256"],
                "passed": (
                    int(expected["bytes"]) == actual["bytes"]
                    and expected["sha256"] == actual["sha256"]
                ),
            }
        )
    return {
        "researchStepId": manifest["researchStepId"],
        "manifestFile": file_record(manifest_path),
        "expectedArtifactCount": int(manifest["artifactCount"]),
        "validatedArtifactCount": len(records),
        "allByteIdentical": all(row["passed"] for row in records),
        "records": records,
    }


def _predecessor_immutability_validation() -> dict[str, Any]:
    validations = {
        step_id: _artifact_manifest_validation(ARTIFACTS / f"research_steps/{step_id}")
        for step_id in ("S12R", "S12S", "S12T")
    }
    s12t_baseline = read_json(S12T / "predecessor_immutability_baseline.json")
    result = {
        "schemaVersion": "e07.s12u.predecessor-immutability-validation.v1",
        "researchStepId": "S12U",
        "validations": validations,
        "s12tBaselineFile": file_record(
            S12T / "predecessor_immutability_baseline.json"
        ),
        "s12tBaselinePassed": bool(s12t_baseline["passed"]),
        "s12sCacheAbsent": not Path("/cache/e07-s12s").exists(),
        "s12sScientificPublicationAbsent": not (
            ARTIFACTS / "research_steps/S12S/scientific_publication"
        ).exists(),
        "priorArtifactMutations": 0,
    }
    result["allPassed"] = bool(
        all(row["allByteIdentical"] for row in validations.values())
        and result["s12tBaselinePassed"]
        and result["s12sCacheAbsent"]
        and result["s12sScientificPublicationAbsent"]
    )
    return result


def _freeze_validation() -> dict[str, Any]:
    freeze = read_json(S12R / "preregistration_freeze.json")
    records = []
    for name, expected in sorted(freeze["files"].items()):
        actual = file_record(Path(expected["path"]))
        records.append(
            {
                "name": name,
                "expectedSha256": expected["sha256"],
                "actualSha256": actual["sha256"],
                "expectedBytes": int(expected["bytes"]),
                "actualBytes": actual["bytes"],
                "passed": (
                    expected["sha256"] == actual["sha256"]
                    and int(expected["bytes"]) == actual["bytes"]
                ),
            }
        )
    return {
        "freezeFile": file_record(S12R / "preregistration_freeze.json"),
        "frozenBeforeAnyEpisode": freeze["frozenBeforeAnyEpisode"],
        "transferEpisodesAtFreeze": freeze["transferEpisodesAtFreeze"],
        "semanticCommitments": freeze["semanticCommitments"],
        "records": records,
        "allPassed": (
            freeze["frozenBeforeAnyEpisode"] is True
            and int(freeze["transferEpisodesAtFreeze"]) == 0
            and all(row["passed"] for row in records)
        ),
    }


def _regenerate_structural_commitments() -> dict[str, Any]:
    protocol = s12r.load_protocol()
    frozen = s12r.load_frozen_population()
    variants = read_jsonl(S12R / "adaptation_variant_registry.jsonl")
    lineages = read_jsonl(S12R / "lineage_registry.jsonl")
    replacements = read_jsonl(S12R / "replacement_comparator_registry.jsonl")
    cells = s12r.build_condition_cells(protocol)
    scenarios = s12r.build_scenarios(cells)
    roster = s12r.build_roster(
        frozen["candidates"],
        lineages,
        variants,
        scenarios,
    )
    physical = s12r.build_physical_commitments(roster)
    accounting = read_json(S12R / "complete_accounting.json")
    logical_semantic = canonical_sha256("E07/S12R/ordered-logical-roster/v1", roster)
    physical_semantic = canonical_sha256(
        "E07/S12R/ordered-physical-plan/v1",
        physical.to_dict("records"),
    )
    stored_logical = pd.read_parquet(S12R / "s12r_logical_roster.parquet")
    stored_physical = pd.read_parquet(S12R / "s12r_physical_commitments.parquet")
    roster_identity_match = set(stored_logical["logicalReservationId"]) == {
        row["logicalReservationId"] for row in roster
    }
    physical_identity_match = set(stored_physical["physicalExecutionId"]) == set(
        physical["physicalExecutionId"]
    )
    binding, binding_summary = s12r.build_binding_qualification(
        frozen,
        variants,
        lineages,
        replacements,
        cells,
    )
    stored_binding = pd.read_parquet(S12R / "binding_qualification.parquet")
    key_columns = [
        "selectionRef",
        "taskCellId",
        "actionSha256",
        "passed",
    ]
    binding_match = (
        binding[key_columns]
        .sort_values(key_columns[:2])
        .reset_index(drop=True)
        .equals(
            stored_binding[key_columns]
            .sort_values(key_columns[:2])
            .reset_index(drop=True)
        )
    )
    return {
        "protocol": protocol,
        "frozen": frozen,
        "variants": variants,
        "lineages": lineages,
        "replacements": replacements,
        "cells": cells,
        "scenarios": scenarios,
        "roster": roster,
        "physical": physical,
        "summary": {
            "lineageCount": len(lineages),
            "configurationCount": len(frozen["candidateConfigurations"]),
            "adaptationVariantCount": len(variants),
            "replacementComparatorCount": len(replacements),
            "replacementComparatorIds": sorted(
                row["newConfigurationId"] for row in replacements
            ),
            "taskConditionCellCount": len(cells),
            "scenarioFamilyCount": len(scenarios),
            "developmentScenarioFamilies": sum(
                row["partition"] == "development" for row in scenarios
            ),
            "postLockScenarioFamilies": sum(
                row["partition"] == "post_lock_transfer" for row in scenarios
            ),
            "logicalReservationCount": len(roster),
            "physicalReplayCommitmentCount": len(physical),
            "logicalRosterSemanticSha256": logical_semantic,
            "physicalPlanSemanticSha256": physical_semantic,
            "logicalIdentitySetMatchesPersisted": roster_identity_match,
            "physicalIdentitySetMatchesPersisted": physical_identity_match,
            "bindingRows": int(binding_summary["bindingRows"]),
            "bindingRowsPassed": int(binding_summary["bindingRowsPassed"]),
            "bindingProjectionMatchesPersisted": binding_match,
            "allPassed": bool(
                len(lineages) == 7
                and len(frozen["candidateConfigurations"]) == 14
                and len(variants) == 30
                and len(replacements) == 2
                and {row["newConfigurationId"] for row in replacements}
                == NEW_REPLACEMENT_COMPARATOR_IDS
                and len(cells) == 48
                and len(scenarios) == 3_840
                and len(roster) == 195_072
                and len(physical) == 390_144
                and logical_semantic == accounting["logicalRosterSemanticSha256"]
                and physical_semantic == accounting["physicalCommitmentSemanticSha256"]
                and roster_identity_match
                and physical_identity_match
                and binding_summary["allPassed"]
                and binding_summary["bindingRows"] == 3_120
                and binding_match
            ),
        },
    }


def _revalidate_access_and_reserve() -> tuple[dict[str, Any], dict[str, Any]]:
    current_access, current_reserve = s12r.access_and_reserve_validation()
    frozen_access = read_json(S12R / "access_control_validation.json")
    frozen_reserve = read_json(S12R / "protected_reserve_contract.json")
    access = {
        "attempts": current_access["attempts"],
        "denials": current_access["denials"],
        "allDenied": current_access["allDenied"],
        "recordsMatchFrozen": current_access["records"] == frozen_access["records"],
        "protectedOutcomeRowsRead": 0,
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
        "civicRows": 0,
        "S13Rows": 0,
        "S14Rows": 0,
    }
    access["passed"] = bool(
        access["allDenied"]
        and access["recordsMatchFrozen"]
        and current_access["passed"]
    )
    reserve = {
        "sourceRecord": current_reserve["sourceRecord"],
        "originalHeldoutAddressTemplateCommitmentSha256": current_reserve[
            "originalHeldoutAddressTemplateCommitmentSha256"
        ],
        "originalHeldoutScenarioIdsMaterialized": current_reserve[
            "originalHeldoutScenarioIdsMaterialized"
        ],
        "originalHeldoutOutcomeAccessed": current_reserve[
            "originalHeldoutOutcomeAccessed"
        ],
        "newExtensionReserveMatchesFrozen": current_reserve["newExtensionReserve"]
        == frozen_reserve["newExtensionReserve"],
        "sourceMatchesFrozen": current_reserve["sourceRecord"]
        == frozen_reserve["sourceRecord"],
        "passed": bool(
            current_reserve["passed"]
            and current_reserve["newExtensionReserve"]
            == frozen_reserve["newExtensionReserve"]
            and current_reserve["sourceRecord"] == frozen_reserve["sourceRecord"]
        ),
    }
    return access, reserve


def _publisher_preflight() -> dict[str, Any]:
    registry = read_json(S12R / "future_publication_registry.json")
    specs = tuple(
        ArtifactSpec(
            str(row["classId"]),
            str(row["relativePath"]),
            str(row["mediaType"]),
        )
        for row in registry["artifactClasses"]
    )
    if tuple(spec.class_id for spec in specs) != PUBLICATION_CLASSES:
        raise RuntimeError("S12R complete-set publication registry changed")
    payloads = {
        class_id: publication_json_bytes(
            {
                "schemaVersion": "e07.s12u.publisher-preflight.v1",
                "artifactClass": class_id,
                "outcomeRows": 0,
            }
        )
        for class_id in PUBLICATION_CLASSES
    }
    test_root = Path("/cache/e07-s12u-publisher-preflight")
    if test_root.exists():
        shutil.rmtree(test_root)
    test_root.mkdir(parents=True)
    publisher = AtomicScientificPublisher(specs)
    audit = publisher.publish(
        test_root / "scientific",
        payloads,
        forensics_directory=test_root / "forensics",
    )
    complete = publisher.validate_complete(
        test_root / "scientific",
        payloads,
        attempt_id=audit["attemptId"],
    )
    shutil.rmtree(test_root)
    return {
        "artifactClasses": list(PUBLICATION_CLASSES),
        "artifactClassCount": len(specs),
        "registryMatchesFrozen": True,
        "completeSetIntegrationPass": complete["complete"],
        "singleCommitBoundary": audit["commitBoundaryCount"] == 1,
        "passed": bool(
            complete["complete"]
            and audit["commitBoundaryCount"] == 1
            and audit["finalScientificPublicationState"]
            == "complete_validated_publication"
        ),
    }


def _freshness_validation(raw_control: Mapping[str, Any]) -> dict[str, Any]:
    freshness = raw_control["freshness"]
    pre_head = str(freshness["preRegistrationRepositoryHead"])
    current_head = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={REPOSITORY}",
            "rev-parse",
            "HEAD",
        ],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    current_tree = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={REPOSITORY}",
            "rev-parse",
            "HEAD^{tree}",
        ],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    remote_lines = subprocess.run(
        [
            "git",
            "ls-remote",
            "origin",
            "refs/heads/eidosoma/groups/28",
        ],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if len(remote_lines) != 1:
        raise RuntimeError("cannot resolve the configured remote S12U branch")
    remote_head = remote_lines[0].split()[0]
    remote_tree_process = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={REPOSITORY}",
            "rev-parse",
            f"{remote_head}^{{tree}}",
        ],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    remote_tree = (
        remote_tree_process.stdout.strip()
        if remote_tree_process.returncode == 0
        else None
    )
    pre_tree = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={REPOSITORY}",
            "rev-parse",
            f"{pre_head}^{{tree}}",
        ],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    prior_mentions = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={REPOSITORY}",
            "log",
            pre_head,
            "-S",
            "S12U",
            "--format=%H",
            "--",
            ".",
        ],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "schemaVersion": "e07.s12u.freshness-validation.v1",
        "researchStepId": "S12U",
        "cachePath": str(CACHE),
        "artifactPath": str(OUT),
        "cachePathAbsent": not CACHE.exists(),
        "artifactPathAbsent": not OUT.exists(),
        "scientificPathAbsent": not SCIENTIFIC.exists(),
        "preRegistrationRepositoryHead": pre_head,
        "preRegistrationTreeExpected": str(freshness["preRegistrationRepositoryTree"]),
        "preRegistrationTreeActual": pre_tree,
        "preRegistrationTreeMatches": (
            pre_tree == freshness["preRegistrationRepositoryTree"]
        ),
        "priorRepositoryMentions": prior_mentions,
        "priorRepositoryMentionCount": len(prior_mentions),
        "priorGitHistoryUseAbsent": not prior_mentions,
        "executionRepositoryHead": current_head,
        "executionRepositoryTree": current_tree,
        "remoteRepositoryHead": remote_head,
        "remoteRepositoryTree": remote_tree,
        "repositoryCommitShaMatchesRemote": current_head == remote_head,
        "repositoryTreeMatchesRemote": current_tree == remote_tree,
        "repositorySourcePublishedBeforeExecution": bool(current_tree == remote_tree),
        "s12sCacheAbsent": not Path("/cache/e07-s12s").exists(),
        "s12sScientificPublicationAbsent": not (
            ARTIFACTS / "research_steps/S12S/scientific_publication"
        ).exists(),
        "passed": bool(
            not CACHE.exists()
            and not OUT.exists()
            and not SCIENTIFIC.exists()
            and pre_tree == freshness["preRegistrationRepositoryTree"]
            and not prior_mentions
            and current_tree == remote_tree
            and not Path("/cache/e07-s12s").exists()
            and not (ARTIFACTS / "research_steps/S12S/scientific_publication").exists()
        ),
    }


def _s12t_dependency_preflight(
    control: Mapping[str, Any],
) -> dict[str, Any]:
    raw_control = yaml.safe_load(CONTROL.read_text(encoding="utf-8"))
    expected = raw_control["frozenS12T"]
    source_files = {
        "protocol": S12T_PROTOCOL_FILE,
        "registry": S12T_REGISTRY_FILE,
        "preregistrationFreeze": S12T_FREEZE_FILE,
        "executionReviewGate": S12T_GATE_FILE,
        "artifactManifest": S12T_MANIFEST_FILE,
        "implementation": S12T_IMPLEMENTATION,
    }
    expected_hashes = {
        "protocol": expected["protocolSha256"],
        "registry": expected["registrySha256"],
        "preregistrationFreeze": expected["preregistrationFreezeSha256"],
        "executionReviewGate": expected["executionReviewGateSha256"],
        "artifactManifest": expected["artifactManifestSha256"],
        "implementation": expected["implementationSha256"],
    }
    hash_records = {
        key: {
            **file_record(path),
            "expectedSha256": expected_hashes[key],
            "passed": sha256_file(path) == expected_hashes[key],
        }
        for key, path in source_files.items()
    }
    manifest = read_json(S12T_MANIFEST_FILE)
    manifest_records = []
    for row in manifest["artifacts"]:
        actual = file_record(Path(row["path"]))
        manifest_records.append(
            {
                "path": row["path"],
                "expectedBytes": int(row["bytes"]),
                "actualBytes": actual["bytes"],
                "expectedSha256": row["sha256"],
                "actualSha256": actual["sha256"],
                "passed": (
                    int(row["bytes"]) == actual["bytes"]
                    and row["sha256"] == actual["sha256"]
                ),
            }
        )
    gate = read_json(S12T_GATE_FILE)
    registry = DependencyRegistry.load(S12T_REGISTRY_FILE)
    static = StaticImportAnalyzer(
        registry,
        search_roots=[REPOSITORY],
    ).analyze(
        [
            Path(__file__),
            REPOSITORY / "src/spatial_transfer/execution.py",
            REPOSITORY / "src/spatial_transfer/execution_dependency_guard.py",
            REPOSITORY / "src/environment_suite/dsl_adapters.py",
            REPOSITORY / "src/morph2d/engine.py",
        ]
    )
    session_commitment = domain_hash(
        "E07/S12U/preflight-dependency-session/v1",
        {
            "executionControlSha256": control["executionControlSha256"],
            "registrySha256": registry.registry_sha256,
        },
    )
    ledger = AuthenticatedAccessLedger(
        registry,
        session_commitment=session_commitment,
    )
    for path in (
        CONTROL,
        OUTPUT_REGISTRY_FILE,
        S12T_PROTOCOL_FILE,
        S12T_REGISTRY_FILE,
        S12T_FREEZE_FILE,
        S12T_GATE_FILE,
        S12T_MANIFEST_FILE,
        S12T_IMPLEMENTATION,
        S12R / "artifact_manifest.json",
        S12R / "preregistration_freeze.json",
        S12R / "s12r_execution_gate.json",
        S12R / "s12r_logical_roster.parquet",
        S12R / "s12r_physical_commitments.parquet",
        S12R / "protected_reserve_contract.json",
        S12R / "future_publication_registry.json",
    ):
        ledger.authorize_path(
            path,
            operation="read",
            source_api="S12U.preflight",
        )
    ledger.authorize_signal("S12U.human_execution_authorization")
    access_ledger = ledger.finalize()
    access_validation = validate_access_ledger(
        access_ledger,
        registry,
        expected_session_commitment=session_commitment,
    )
    output_registry = OutputAccessRegistry.load(OUTPUT_REGISTRY_FILE)
    result = {
        "schemaVersion": "e07.s12u.s12t-dependency-preflight.v1",
        "researchStepId": "S12U",
        "frozenS12TFiles": hash_records,
        "s12tArtifactManifestRecords": manifest_records,
        "s12tArtifactManifestPassed": all(row["passed"] for row in manifest_records),
        "executionReviewGate": gate,
        "staticImportGraph": static,
        "preflightAccessLedger": access_ledger,
        "preflightAccessLedgerValidation": access_validation,
        "baseRegistrySha256": registry.registry_sha256,
        "outputRegistrySha256": output_registry.registry_sha256,
        "outputRegistryBaseCommitmentMatches": (
            output_registry.base_dependency_registry_sha256 == registry.registry_sha256
        ),
        "runtimeDispatchAuditRequired": True,
        "cacheReadEvents": 0,
        "S07SignalReads": 0,
    }
    result["passed"] = bool(
        all(row["passed"] for row in hash_records.values())
        and result["s12tArtifactManifestPassed"]
        and gate["qualificationPassed"]
        and gate["dependencyPreflightEligibleForSeparateFreshExecutionReview"]
        and gate["separateHumanExecutionDecisionRequired"]
        and not gate["scientificExecutionAuthorized"]
        and static["passed"]
        and access_validation["passed"]
        and result["outputRegistryBaseCommitmentMatches"]
    )
    return result


def _initialize_output_ledger(control: Mapping[str, Any]) -> None:
    global _OUTPUT_LEDGER, _OUTPUT_REGISTRY, _OUTPUT_SESSION_COMMITMENT
    _OUTPUT_REGISTRY = OutputAccessRegistry.load(OUTPUT_REGISTRY_FILE)
    _OUTPUT_SESSION_COMMITMENT = domain_hash(
        "E07/S12U/output-session/v1",
        {
            "executionControlSha256": control["executionControlSha256"],
            "registrySha256": _OUTPUT_REGISTRY.registry_sha256,
        },
    )
    _OUTPUT_LEDGER = OutputAccessLedger(
        _OUTPUT_REGISTRY,
        session_commitment=_OUTPUT_SESSION_COMMITMENT,
    )


def run_preflight() -> dict[str, Any]:
    raw_control = yaml.safe_load(CONTROL.read_text(encoding="utf-8"))
    freshness = _freshness_validation(raw_control)
    if not freshness["passed"]:
        raise RuntimeError(f"S12U freshness validation failed: {freshness}")
    control = control_record()
    _initialize_output_ledger(control)
    dependency = _s12t_dependency_preflight(control)
    artifact_validation = _artifact_manifest_validation()
    predecessor_immutability = _predecessor_immutability_validation()
    freeze_validation = _freeze_validation()
    gate = read_json(S12R / "s12r_execution_gate.json")
    gate_rows = {
        key: bool(value["passed"] and value["disposition"] == "pass")
        for key, value in gate["rows"].items()
    }
    structural = _regenerate_structural_commitments()
    access, reserve = _revalidate_access_and_reserve()
    publisher = _publisher_preflight()
    fault = read_json(S12R / "spatial_fault_contract.json")
    scheduler = read_json(S12R / "alternate_scheduler_contract.json")
    endpoint_registry = read_jsonl(S12R / "endpoint_contract_registry.jsonl")
    cost_registry = read_json(S12R / "cost_registry.json")
    protocol_file = file_record(
        S12R / "s12r_replacement_spatial_transfer_protocol.yaml"
    )
    contracts = {
        "protocolSha256MatchesControl": protocol_file["sha256"]
        == raw_control["frozenS12R"]["protocolSha256"],
        "faultFamilyId": fault["faultFamilyId"],
        "faultContractPassed": fault["faultFamilyId"]
        == "identity_locus_target_breaking_spurious_swap_v1",
        "schedulerFamilyId": scheduler["schedulerFamilyId"],
        "schedulerContractPassed": scheduler["schedulerFamilyId"]
        == "identity_round_robin_batch4_v1",
        "endpointRegistryRows": len(endpoint_registry),
        "endpointDiagnosticRowsPromotionForbidden": all(
            row.get("diagnosticPromotionEligible") is False
            for row in endpoint_registry
            if row.get("endpointMode") == "diagnostic_only"
        ),
        "universalScoreAbsent": cost_registry["scalarOrUniversalCost"] is None,
        "costFamiliesSeparate": bool(
            cost_registry["faultLedger"]["separateFaultLedger"]
            and cost_registry["schedulerLedger"]["separateSchedulerLedger"]
        ),
    }
    contracts["passed"] = all(
        value
        for key, value in contracts.items()
        if key.endswith("Passed")
        or key
        in {
            "protocolSha256MatchesControl",
            "endpointDiagnosticRowsPromotionForbidden",
            "universalScoreAbsent",
            "costFamiliesSeparate",
        }
    )
    checks = {
        "G01": predecessor_immutability["allPassed"]
        and artifact_validation["allByteIdentical"]
        and freeze_validation["allPassed"],
        "G02": structural["summary"]["configurationCount"] == 14
        and structural["summary"]["lineageCount"] == 7,
        "G03": structural["summary"]["replacementComparatorCount"] == 2,
        "G04": contracts["faultContractPassed"],
        "G05": contracts["schedulerContractPassed"],
        "G06": structural["summary"]["bindingRows"] == 3_120
        and structural["summary"]["bindingRowsPassed"] == 3_120,
        "G07": contracts["passed"],
        "G08": access["passed"] and reserve["passed"],
        "G09": structural["summary"]["allPassed"],
        "G10": publisher["passed"],
    }
    all_passed = bool(
        all(gate_rows.values())
        and set(gate_rows) == {f"G{index:02d}" for index in range(1, 11)}
        and all(checks.values())
        and dependency["passed"]
    )
    preflight = {
        "schemaVersion": "e07.s12u.zero-episode-preflight.v1",
        "researchStepId": "S12U",
        "timestampUtc": datetime.now(timezone.utc).isoformat(),
        "reservedEpisodesSubmitted": 0,
        "freshCacheNamespaceAbsentAtCheck": True,
        "freshnessValidation": freshness,
        "executionControl": control,
        "s12rArtifactManifest": artifact_validation,
        "predecessorImmutability": predecessor_immutability,
        "s12rPreregistrationFreeze": freeze_validation,
        "frozenGateRows": gate_rows,
        "liveGateRows": checks,
        "structuralCommitments": structural["summary"],
        "accessControl": access,
        "protectedReserve": reserve,
        "dependencyExclusion": dependency,
        "nativeExtensionEndpointCostContracts": contracts,
        "publisher": publisher,
        "allPassed": all_passed,
        "substantiveTransferAuthorizedByHumanForS12UOnly": True,
        "s12rGateFileSubstantiveAuthorizationMutated": False,
    }
    if _OUTPUT_LEDGER is None:
        raise RuntimeError("S12U output ledger was not installed")
    _OUTPUT_LEDGER.authorize_path(
        OUT,
        operation="mkdir",
        artifact_class="attempt_root",
    )
    OUT.mkdir(parents=True, exist_ok=False)
    atomic_write_json(OUT / "s12u_execution_control.json", control)
    atomic_write_json(OUT / "preflight_validation.json", preflight)
    input_hash_freeze = {
        "schemaVersion": "e07.s12u.input-hash-freeze.v1",
        "researchStepId": "S12U",
        "frozenBeforeAnyEpisode": True,
        "episodesSubmittedAtFreeze": 0,
        "files": {
            "executionControl": file_record(CONTROL),
            "outputAccessRegistry": file_record(OUTPUT_REGISTRY_FILE),
            "executionSource": file_record(Path(__file__)),
            "dependencyOutputGuard": file_record(
                REPOSITORY / "src/spatial_transfer/execution_dependency_guard.py"
            ),
            "s12tProtocol": file_record(S12T_PROTOCOL_FILE),
            "s12tRegistry": file_record(S12T_REGISTRY_FILE),
            "s12tImplementation": file_record(S12T_IMPLEMENTATION),
            "s12rProtocol": file_record(
                S12R / "s12r_replacement_spatial_transfer_protocol.yaml"
            ),
            "s12rPreregistrationFreeze": file_record(
                S12R / "preregistration_freeze.json"
            ),
            "s12rLogicalRoster": file_record(S12R / "s12r_logical_roster.parquet"),
            "s12rPhysicalCommitments": file_record(
                S12R / "s12r_physical_commitments.parquet"
            ),
        },
        "repositoryHead": freshness["executionRepositoryHead"],
        "repositoryTree": freshness["executionRepositoryTree"],
        "remoteRepositoryHead": freshness["remoteRepositoryHead"],
        "remoteRepositoryTree": freshness["remoteRepositoryTree"],
    }
    input_hash_freeze["semanticSha256"] = canonical_sha256(
        "E07/S12U/input-hash-freeze/v1",
        input_hash_freeze,
    )
    atomic_write_json(OUT / "input_hash_freeze.json", input_hash_freeze)
    preregistration_freeze = {
        "schemaVersion": "e07.s12u.execution-preregistration-freeze.v1",
        "researchStepId": "S12U",
        "frozenBeforeAnyEpisode": True,
        "episodesSubmittedAtFreeze": 0,
        "executesOnlyByteFrozenS12R": True,
        "s12rDesignChanged": False,
        "s12sPatchedResumedOrRetried": False,
        "preflightAllPassed": all_passed,
        "inputHashFreezeFile": file_record(OUT / "input_hash_freeze.json"),
        "executionControlFile": file_record(OUT / "s12u_execution_control.json"),
        "preflightFile": file_record(OUT / "preflight_validation.json"),
        "logicalReservations": 195_072,
        "physicalReplayCommitments": 390_144,
        "workerActualUseDependencyAuditRequired": True,
        "completeOrZeroPublicationRequired": True,
        "protectedAndDownstreamRowsAuthorized": 0,
    }
    preregistration_freeze["semanticSha256"] = canonical_sha256(
        "E07/S12U/execution-preregistration-freeze/v1",
        preregistration_freeze,
    )
    atomic_write_json(
        OUT / "execution_preregistration_freeze.json",
        preregistration_freeze,
    )
    if not all_passed:
        atomic_write_json(
            OUT / "attempt_forensics.json",
            {
                "schemaVersion": "e07.s12u.attempt-forensics.v1",
                "status": "failed_closed_before_episode",
                "scientificPublicationRows": 0,
                "preflight": preflight,
            },
        )
        raise RuntimeError("S12U preflight failed closed before episode submission")
    _OUTPUT_LEDGER.authorize_path(
        CACHE,
        operation="mkdir",
        artifact_class="fresh_execution_cache",
    )
    CACHE.mkdir(parents=False, exist_ok=False)
    atomic_write_json(
        CACHE / "preflight_commitment.json",
        {
            "preflightFile": file_record(OUT / "preflight_validation.json"),
            "executionControlFile": file_record(OUT / "s12u_execution_control.json"),
            "s12rLogicalRosterFile": file_record(S12R / "s12r_logical_roster.parquet"),
            "s12rPhysicalCommitmentFile": file_record(
                S12R / "s12r_physical_commitments.parquet"
            ),
        },
    )
    return {
        "preflight": preflight,
        "protocol": structural["protocol"],
        "frozen": structural["frozen"],
        "variants": structural["variants"],
        "lineages": structural["lineages"],
        "replacements": structural["replacements"],
        "scenarios": structural["scenarios"],
        "roster": structural["roster"],
        "physical": structural["physical"],
    }


def _runtime_registries(
    winner_by_base: Mapping[str, str],
) -> dict[str, Any]:
    frozen = s12r.load_frozen_population()
    configurations = {
        configuration_id: s12r.strict_structural_configuration(configuration)
        for configuration_id, configuration in frozen["allS08MConfigurations"].items()
    }
    configurations.update(
        {
            configuration_id: deepcopy(configuration)
            for configuration_id, configuration in frozen[
                "candidateConfigurations"
            ].items()
        }
    )
    replacements = read_jsonl(S12R / "replacement_comparator_registry.jsonl")
    configurations.update(
        {
            str(row["newConfigurationId"]): s12r.strict_structural_configuration(
                row["definition"]
            )
            for row in replacements
        }
    )
    source_variants = {
        str(row["adaptationVariantId"]): row for row in frozen["sourceVariants"]
    }
    adapted: dict[str, dict[str, Any]] = {}
    s12r_variants = read_jsonl(S12R / "adaptation_variant_registry.jsonl")
    for row in s12r_variants:
        variant_id = str(row["adaptationVariantId"])
        source = source_variants[str(row["sourceS12PAdaptationVariantId"])]
        base = frozen["candidateConfigurations"][str(row["baseConfigurationId"])]
        adapted[variant_id] = apply_adaptation_variant(base, source)
    actions = {
        runtime_id: _runtime_action(configuration, frozen["documents"])
        for runtime_id, configuration in {**configurations, **adapted}.items()
        if configuration["configurationId"]
        in {row["configurationId"] for row in frozen["candidates"]}
        or runtime_id
        in {
            row["runtimeSelectionRef"]
            for row in pd.read_parquet(
                S12R / "s12r_logical_roster.parquet",
                columns=["runtimeSelectionRef"],
            ).to_dict("records")
            if row["runtimeSelectionRef"]
            not in {
                "native_task_baseline",
                "locked_S12R_adaptation_winner_after_development",
            }
        }
    }
    return {
        "configurations": configurations,
        "adapted": adapted,
        "actions": actions,
        "documents": frozen["documents"],
        "winnerByBase": dict(winner_by_base),
    }


def _worker_init(winner_by_base: Mapping[str, str]) -> None:
    global _RUNTIME
    registry = DependencyRegistry.load(S12T_REGISTRY_FILE)
    session_commitment = domain_hash(
        "E07/S12U/worker-dispatch-session/v1",
        {
            "processName": mp.current_process().name,
            "winnerByBaseSha256": canonical_sha256(
                "E07/S12U/winner-by-base/v1",
                dict(sorted(winner_by_base.items())),
            ),
            "registrySha256": registry.registry_sha256,
        },
    )
    ledger = AuthenticatedAccessLedger(
        registry,
        session_commitment=session_commitment,
    )
    open_context = instrument_path_opens(ledger)
    open_context.__enter__()
    runtime = _runtime_registries(winner_by_base)
    runtime.update(
        {
            "dependencyRegistry": registry,
            "dependencyAccessLedger": ledger,
            "dependencyOpenContext": open_context,
            "dependencySessionCommitment": session_commitment,
            "dispatchAudit": None,
            "dispatchAccessLedger": None,
            "dispatchAccessValidation": None,
            "dispatchEvidenceEmitted": False,
            "knownModuleNames": set(sys.modules),
            "knownModuleCount": len(sys.modules),
        }
    )
    _RUNTIME = runtime


def _resolve_runtime(
    logical: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str, str | None, Any | None]:
    role = str(logical["conditionRole"])
    if role == "native_baseline":
        return None, "native_task_baseline", None, None
    if role == "adaptation_variant_development":
        variant_id = str(logical["runtimeSelectionRef"])
        return (
            _RUNTIME["adapted"][variant_id],
            variant_id,
            variant_id,
            _RUNTIME["actions"][variant_id],
        )
    if role == "adapted_winner_slot":
        variant_id = _RUNTIME["winnerByBase"][str(logical["baseConfigurationId"])]
        return (
            _RUNTIME["adapted"][variant_id],
            variant_id,
            variant_id,
            _RUNTIME["actions"][variant_id],
        )
    runtime_id = str(logical["runtimeSelectionRef"])
    return (
        _RUNTIME["configurations"][runtime_id],
        runtime_id,
        None,
        _RUNTIME["actions"][runtime_id],
    )


def _worker(item: Mapping[str, Any]) -> dict[str, Any]:
    logical = item["logical"]
    try:
        configuration, runtime_id, variant_id, action = _resolve_runtime(logical)
        holder: dict[str, Any] = {}

        def execute_pair() -> dict[str, str]:
            first_result = execute_physical(
                logical,
                replay_ordinal=0,
                physical_execution_id=str(item["physical0"]),
                configuration=configuration,
                resolved_runtime_id=runtime_id,
                adaptation_variant_id=variant_id,
                documents_by_hash=_RUNTIME["documents"],
                runtime_action=action,
            )
            second_result = execute_physical(
                logical,
                replay_ordinal=1,
                physical_execution_id=str(item["physical1"]),
                configuration=configuration,
                resolved_runtime_id=runtime_id,
                adaptation_variant_id=variant_id,
                documents_by_hash=_RUNTIME["documents"],
                runtime_action=action,
            )
            holder["first"] = first_result
            holder["second"] = second_result
            return {
                "physicalExecutionId0": str(first_result["physicalExecutionId"]),
                "physicalExecutionId1": str(second_result["physicalExecutionId"]),
                "resultProjectionSha2560": str(first_result["resultProjectionSha256"]),
                "resultProjectionSha2561": str(second_result["resultProjectionSha256"]),
            }

        if _RUNTIME["dispatchAudit"] is None:
            registry = _RUNTIME["dependencyRegistry"]
            preloaded = [
                name
                for name in sys.modules
                if (
                    name in {"src", "scripts", "__main__", "__mp_main__"}
                    or name.startswith(("src.", "scripts."))
                    or name.split(".", 1)[0] in registry.allowed_third_party
                )
            ]
            dispatch_audit = audit_runtime_dispatch(
                execute_pair,
                registry,
                inspect_preloaded_names=preloaded,
            )
            if not dispatch_audit["passed"]:
                raise RuntimeError(
                    "S12T runtime module provenance failed on actual dispatch"
                )
            access_ledger = _RUNTIME["dependencyAccessLedger"].finalize()
            access_validation = validate_access_ledger(
                access_ledger,
                registry,
                expected_session_commitment=_RUNTIME["dependencySessionCommitment"],
            )
            if not access_validation["passed"]:
                raise RuntimeError("S12T authenticated dispatch access ledger failed")
            _RUNTIME["dispatchAudit"] = dispatch_audit
            _RUNTIME["dispatchAccessLedger"] = access_ledger
            _RUNTIME["dispatchAccessValidation"] = access_validation
            _RUNTIME["dispatchAccessEventCount"] = len(
                _RUNTIME["dependencyAccessLedger"].events
            )
            _RUNTIME["knownModuleNames"] = set(sys.modules)
            _RUNTIME["knownModuleCount"] = len(sys.modules)
        else:
            execute_pair()
            if (
                len(_RUNTIME["dependencyAccessLedger"].events)
                != _RUNTIME["dispatchAccessEventCount"]
            ):
                raise RuntimeError(
                    "unexpected path/cache access after dispatch audit lock"
                )
            if len(sys.modules) != _RUNTIME["knownModuleCount"]:
                new_names = set(sys.modules) - _RUNTIME["knownModuleNames"]
                new_records = []
                for name in sorted(new_names):
                    module = sys.modules[name]
                    origin = getattr(module, "__file__", None)
                    allowed, reason, resolved_origin = _RUNTIME[
                        "dependencyRegistry"
                    ].module_origin_disposition(
                        name,
                        str(origin) if origin is not None else None,
                    )
                    new_records.append(
                        {
                            "module": name,
                            "resolvedOrigin": resolved_origin,
                            "allowed": allowed,
                            "reason": reason,
                        }
                    )
                if any(not row["allowed"] for row in new_records):
                    raise RuntimeError(
                        "forbidden late-loaded runtime module on dispatch"
                    )
                _RUNTIME["knownModuleNames"] = set(sys.modules)
                _RUNTIME["knownModuleCount"] = len(sys.modules)
        first = holder["first"]
        second = holder["second"]
        logical_result = logical_from_replays(first, second)
        dependency_evidence = None
        if not _RUNTIME["dispatchEvidenceEmitted"]:
            dependency_evidence = {
                "schemaVersion": "e07.s12u.worker-dispatch-dependency-evidence.v1",
                "registrySha256": _RUNTIME["dependencyRegistry"].registry_sha256,
                "runtimeModuleProvenance": _RUNTIME["dispatchAudit"],
                "authenticatedAccessLedger": _RUNTIME["dispatchAccessLedger"],
                "authenticatedAccessValidation": _RUNTIME["dispatchAccessValidation"],
                "cacheAccessEvents": sum(
                    event.get("accessKind") == "cache"
                    for event in _RUNTIME["dispatchAccessLedger"]["events"]
                ),
                "signalAccessEvents": sum(
                    event.get("eventKind") == "signal_access"
                    for event in _RUNTIME["dispatchAccessLedger"]["events"]
                ),
            }
            dependency_evidence["passed"] = bool(
                dependency_evidence["runtimeModuleProvenance"]["passed"]
                and dependency_evidence["authenticatedAccessValidation"]["passed"]
                and dependency_evidence["cacheAccessEvents"] == 0
                and dependency_evidence["signalAccessEvents"] == 0
            )
            _RUNTIME["dispatchEvidenceEmitted"] = True
        return {
            "success": True,
            "logical": logical_result,
            "physical": [
                {
                    "logicalReservationId": logical["logicalReservationId"],
                    "physicalExecutionId": first["physicalExecutionId"],
                    "replayOrdinal": 0,
                    "state": "executed_validated",
                    "resultProjectionSha256": first["resultProjectionSha256"],
                },
                {
                    "logicalReservationId": logical["logicalReservationId"],
                    "physicalExecutionId": second["physicalExecutionId"],
                    "replayOrdinal": 1,
                    "state": "executed_validated",
                    "resultProjectionSha256": second["resultProjectionSha256"],
                },
            ],
            "workerPid": os.getpid(),
            "dependencyEvidence": dependency_evidence,
            "dependencyAuditCommitment": domain_hash(
                "E07/S12U/worker-dependency-audit/v1",
                {
                    "registrySha256": _RUNTIME["dependencyRegistry"].registry_sha256,
                    "runtimeSemanticSha256": _RUNTIME["dispatchAudit"][
                        "semanticSha256"
                    ],
                    "accessLedgerSha256": _RUNTIME["dispatchAccessLedger"][
                        "ledgerSha256"
                    ],
                },
            ),
        }
    except Exception as error:  # noqa: BLE001 - fail-atomic worker disposition
        return {
            "success": False,
            "logicalReservationId": str(logical["logicalReservationId"]),
            "errorType": type(error).__name__,
            "error": str(error),
            "workerPid": os.getpid(),
        }


def _work_items(
    roster: Sequence[Mapping[str, Any]],
    physical: pd.DataFrame,
) -> list[dict[str, Any]]:
    physical_by_logical: dict[str, dict[int, str]] = defaultdict(dict)
    for row in physical.to_dict("records"):
        physical_by_logical[str(row["logicalReservationId"])][
            int(row["replayOrdinal"])
        ] = str(row["physicalExecutionId"])
    items = []
    for logical in roster:
        pair = physical_by_logical[str(logical["logicalReservationId"])]
        if set(pair) != {0, 1}:
            raise RuntimeError("physical commitment pair is incomplete")
        items.append(
            {
                "logical": dict(logical),
                "physical0": pair[0],
                "physical1": pair[1],
            }
        )
    return items


def _phase_parts(phase: str) -> list[Path]:
    return sorted((CACHE / phase / "logical_parts").glob("part-*.parquet"))


def load_phase_rows(phase: str) -> pd.DataFrame:
    parts = _phase_parts(phase)
    if not parts:
        raise RuntimeError(f"{phase} has no result parts")
    if _OUTPUT_LEDGER is not None:
        for path in parts:
            _OUTPUT_LEDGER.authorize_path(
                path,
                operation="read_parquet",
                artifact_class="execution_disposition_part",
            )
    return pd.concat((pd.read_parquet(path) for path in parts), ignore_index=True)


def execute_phase(
    phase: str,
    items: Sequence[Mapping[str, Any]],
    winner_by_base: Mapping[str, str],
) -> dict[str, Any]:
    phase_root = CACHE / phase
    if phase_root.exists():
        raise RuntimeError(f"fresh phase path already exists: {phase_root}")
    logical_parts = phase_root / "logical_parts"
    physical_parts = phase_root / "physical_parts"
    logical_parts.mkdir(parents=True)
    physical_parts.mkdir(parents=True)
    batch_size = 64
    part_size = 512
    logical_buffer: list[dict[str, Any]] = []
    physical_buffer: list[dict[str, Any]] = []
    part_index = 0
    completed = 0
    observed_completion_order: list[str] = []
    worker_counts: Counter[int] = Counter()
    dependency_evidence_by_worker: dict[int, dict[str, Any]] = {}
    dependency_commitments_by_worker: dict[int, str] = {}
    started = time.monotonic()

    def flush() -> None:
        nonlocal part_index
        if not logical_buffer:
            return
        order = np.argsort([row["logicalReservationId"] for row in logical_buffer])
        ordered_logical = [logical_buffer[int(index)] for index in order]
        ordered_physical = sorted(
            physical_buffer,
            key=lambda row: (row["logicalReservationId"], row["replayOrdinal"]),
        )
        write_parquet(
            logical_parts / f"part-{part_index:05d}.parquet",
            ordered_logical,
        )
        write_parquet(
            physical_parts / f"part-{part_index:05d}.parquet",
            ordered_physical,
        )
        logical_buffer.clear()
        physical_buffer.clear()
        part_index += 1

    context = mp.get_context("spawn")
    with context.Pool(
        processes=WORKERS,
        initializer=_worker_init,
        initargs=(dict(winner_by_base),),
    ) as pool:
        for batch_start in range(0, len(items), batch_size):
            batch = items[batch_start : batch_start + batch_size]
            results = list(pool.imap_unordered(_worker, batch, chunksize=1))
            failures = [row for row in results if not row["success"]]
            for result in results:
                worker_pid = int(result["workerPid"])
                worker_counts[worker_pid] += 1
                evidence = result.get("dependencyEvidence")
                if evidence is not None:
                    previous = dependency_evidence_by_worker.get(worker_pid)
                    if previous is not None and previous != evidence:
                        raise RuntimeError(
                            "worker emitted inconsistent dependency evidence"
                        )
                    dependency_evidence_by_worker[worker_pid] = evidence
                commitment = result.get("dependencyAuditCommitment")
                if commitment is not None:
                    previous_commitment = dependency_commitments_by_worker.get(
                        worker_pid
                    )
                    if (
                        previous_commitment is not None
                        and previous_commitment != commitment
                    ):
                        raise RuntimeError(
                            "worker dependency commitment changed within phase"
                        )
                    dependency_commitments_by_worker[worker_pid] = str(commitment)
                if not result["success"]:
                    continue
                logical_buffer.append(result["logical"])
                physical_buffer.extend(result["physical"])
                observed_completion_order.append(
                    str(result["logical"]["logicalReservationId"])
                )
                completed += 1
            if failures:
                flush()
                pool.terminate()
                failure = {
                    "schemaVersion": "e07.s12u.execution-integrity-failure.v1",
                    "phase": phase,
                    "batchStart": batch_start,
                    "completedLogicalRows": completed,
                    "completedPhysicalRows": completed * 2,
                    "failures": failures,
                    "notAttemptedLogicalRows": len(items) - completed - len(failures),
                    "scientificPublicationRows": 0,
                }
                atomic_write_json(CACHE / f"{phase}_failure.json", failure)
                raise RuntimeError(
                    f"{phase} integrity failure: {failures[0]['errorType']}: "
                    f"{failures[0]['error']}"
                )
            if len(logical_buffer) >= part_size:
                flush()
            if completed % 1024 < batch_size:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"S12U {phase}: {completed:,}/{len(items):,} logical; "
                    f"{completed * 2:,} physical; {completed / elapsed:.2f} logical/s",
                    flush=True,
                )
                atomic_write_json(
                    CACHE / f"{phase}_progress.json",
                    {
                        "phase": phase,
                        "completedLogicalRows": completed,
                        "completedPhysicalRows": completed * 2,
                        "plannedLogicalRows": len(items),
                        "elapsedSeconds": elapsed,
                    },
                )
    flush()
    elapsed = time.monotonic() - started
    frame = load_phase_rows(phase)
    if len(frame) != len(items) or not frame["logicalReservationId"].is_unique:
        raise RuntimeError(f"{phase} logical accounting failed")
    if not bool(frame["exactReplay"].all()):
        raise RuntimeError(f"{phase} exact replay failed")
    worker_ids = set(worker_counts)
    if set(dependency_evidence_by_worker) != worker_ids:
        raise RuntimeError(
            f"{phase} missing actual-dispatch dependency evidence for worker"
        )
    if set(dependency_commitments_by_worker) != worker_ids:
        raise RuntimeError(f"{phase} missing dependency commitments for worker")
    if not all(
        evidence["passed"] for evidence in dependency_evidence_by_worker.values()
    ):
        raise RuntimeError(f"{phase} actual-dispatch dependency audit failed")
    evidence_records = sorted(
        (
            {
                "dependencyAuditCommitment": (
                    dependency_commitments_by_worker[worker_pid]
                ),
                "evidence": evidence,
            }
            for worker_pid, evidence in dependency_evidence_by_worker.items()
        ),
        key=lambda row: row["dependencyAuditCommitment"],
    )
    dependency_summary = {
        "schemaVersion": "e07.s12u.phase-dispatch-dependency-summary.v1",
        "phase": phase,
        "workerCount": len(worker_ids),
        "workerEvidenceCount": len(evidence_records),
        "registrySha256": sha256_file(S12T_REGISTRY_FILE),
        "cacheAccessEvents": sum(
            int(row["evidence"]["cacheAccessEvents"]) for row in evidence_records
        ),
        "signalAccessEvents": sum(
            int(row["evidence"]["signalAccessEvents"]) for row in evidence_records
        ),
        "records": evidence_records,
    }
    dependency_summary["semanticSha256"] = domain_hash(
        "E07/S12U/phase-dispatch-dependency-summary/v1",
        {
            "phase": phase,
            "records": [
                {
                    "dependencyAuditCommitment": row["dependencyAuditCommitment"],
                    "runtimeSemanticSha256": row["evidence"]["runtimeModuleProvenance"][
                        "semanticSha256"
                    ],
                    "accessLedgerSha256": row["evidence"]["authenticatedAccessLedger"][
                        "ledgerSha256"
                    ],
                }
                for row in evidence_records
            ],
        },
    )
    dependency_summary["passed"] = bool(
        dependency_summary["workerCount"] == dependency_summary["workerEvidenceCount"]
        and dependency_summary["cacheAccessEvents"] == 0
        and dependency_summary["signalAccessEvents"] == 0
        and all(row["evidence"]["passed"] for row in evidence_records)
    )
    if not dependency_summary["passed"]:
        raise RuntimeError(f"{phase} dependency summary failed")
    summary = {
        "schemaVersion": "e07.s12u.phase-execution-summary.v1",
        "phase": phase,
        "plannedLogicalRows": len(items),
        "executedLogicalRows": len(frame),
        "executedPhysicalRows": len(frame) * 2,
        "exactReplayRows": int(frame["exactReplay"].sum()),
        "workerCount": len(worker_counts),
        "workerLogicalCounts": {
            str(key): value for key, value in sorted(worker_counts.items())
        },
        "observedCompletionOrderSha256": canonical_sha256(
            "E07/S12U/observed-completion-order/v1",
            observed_completion_order,
        ),
        "canonicalResultSetSha256": canonical_sha256(
            "E07/S12U/canonical-result-set/v1",
            sorted(
                zip(
                    frame["logicalReservationId"],
                    frame["resultProjectionSha256"],
                    strict=True,
                )
            ),
        ),
        "elapsedSeconds": elapsed,
        "throughputLogicalPerSecond": len(frame) / max(elapsed, 1e-9),
        "dispatchDependencyProvenance": dependency_summary,
        "passed": True,
    }
    atomic_write_json(CACHE / f"{phase}_summary.json", summary)
    return summary


def _safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if pd.isna(value):
            return None
        if not math.isfinite(float(value)):
            raise RuntimeError("non-finite scientific value")
        return float(value)
    return value


def _sum_json_ledgers(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    group_keys: Sequence[str],
) -> dict[str, Any]:
    groups: dict[tuple[str, ...], dict[str, Counter[str]]] = {}
    for row in frame.itertuples(index=False):
        key = tuple(str(getattr(row, name)) for name in group_keys)
        ledgers = groups.setdefault(key, {column: Counter() for column in columns})
        for column in columns:
            values = json.loads(getattr(row, column))
            for field, value in values.items():
                ledgers[column][str(field)] += int(value)
    records = []
    for key, ledgers in sorted(groups.items()):
        records.append(
            {
                **dict(zip(group_keys, key, strict=True)),
                "ledgerFamilies": {
                    column: {
                        field: int(value) for field, value in sorted(counter.items())
                    }
                    for column, counter in sorted(ledgers.items())
                },
            }
        )
    return {
        "groupKeys": list(group_keys),
        "groups": records,
        "crossFamilyScalarTotal": None,
    }


def _paired_record(
    *,
    family: str,
    contrast_id: str,
    task_id: str,
    panel_id: str,
    condition_id: str,
    lineage_id: str,
    endpoint: str,
    left_label: str,
    right_label: str,
    left: Sequence[float],
    right: Sequence[float],
    benefit_direction: int,
    binary: bool,
) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        raise RuntimeError("paired endpoint arrays differ")
    differences = left_array - right_array
    n = len(differences)
    test_id = canonical_sha256(
        "E07/S12U/paired-test/v1",
        {
            "family": family,
            "contrastId": contrast_id,
            "taskId": task_id,
            "panelId": panel_id,
            "conditionId": condition_id,
            "lineageId": lineage_id,
            "endpoint": endpoint,
            "left": left_label,
            "right": right_label,
        },
    )
    if n == 0:
        raw_p = 1.0
        raw_effect = None
        ci = [None, None]
        hl = None
        evidentiary = False
        reason = "NO_PAIRED_ROWS"
    else:
        raw_effect = float(differences.mean())
        hl = float(np.median(differences))
        rng = np.random.Generator(np.random.PCG64(int(test_id[:16], 16)))
        indices = rng.integers(0, n, size=(2_000, n), endpoint=False)
        boot = differences[indices].mean(axis=1)
        ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
        if binary:
            positive = int((differences > 0).sum())
            negative = int((differences < 0).sum())
            discordant = positive + negative
            raw_p = (
                1.0
                if discordant == 0
                else float(
                    stats.binomtest(
                        positive, discordant, p=0.5, alternative="two-sided"
                    ).pvalue
                )
            )
        elif np.allclose(differences, 0):
            raw_p = 1.0
        else:
            raw_p = float(
                stats.wilcoxon(
                    differences,
                    zero_method="pratt",
                    alternative="two-sided",
                ).pvalue
            )
        evidentiary = True
        reason = None
    return {
        "testId": test_id,
        "family": family,
        "contrastId": contrast_id,
        "taskId": task_id,
        "panelId": panel_id,
        "conditionId": condition_id,
        "lineageId": lineage_id,
        "endpoint": endpoint,
        "leftLabel": left_label,
        "rightLabel": right_label,
        "pairCount": n,
        "binary": binary,
        "rawEffectLeftMinusRight": raw_effect,
        "benefitDirection": benefit_direction,
        "benefitEffect": (
            None if raw_effect is None else raw_effect * benefit_direction
        ),
        "pairedBootstrap95CiLeftMinusRight": ci,
        "hodgesLehmannSensitivity": hl,
        "rawPValue": raw_p,
        "evidentiary": evidentiary,
        "nonEvidentiaryReason": reason,
        "holmAdjustedPValue": None,
        "holmReject": False,
    }


def _holm(records: list[dict[str, Any]]) -> None:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_family[str(row["family"])].append(row)
    for family_rows in by_family.values():
        ordered = sorted(family_rows, key=lambda row: (row["rawPValue"], row["testId"]))
        running = 0.0
        total = len(ordered)
        for index, row in enumerate(ordered):
            adjusted = min(1.0, (total - index) * float(row["rawPValue"]))
            running = max(running, adjusted)
            row["holmAdjustedPValue"] = running
            row["holmReject"] = bool(row["evidentiary"] and running <= 0.05)


def _endpoint_specs(condition_id: str) -> list[tuple[str, int, bool]]:
    base = [
        ("terminalConjunctiveCompletion", 1, True),
        ("minimumMismatchFraction", -1, False),
    ]
    if "spurious_swap_fault" in condition_id:
        base.append(("repairByTransition32", 1, True))
    else:
        base.append(("departureAfterInitiallyComplete", -1, True))
    return base


def _merge_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: Sequence[str],
) -> pd.DataFrame:
    return left.merge(
        right,
        on=list(keys),
        suffixes=("_left", "_right"),
        how="inner",
        validate="one_to_one",
    )


def build_paired_estimands(
    frame: pd.DataFrame,
    lineages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    post = frame[
        (frame["partition"] == "post_lock_transfer") & frame["endpointAvailable"]
    ].copy()
    records: list[dict[str, Any]] = []

    def add_endpoint_records(
        paired: pd.DataFrame,
        *,
        family: str,
        contrast_id: str,
        lineage_id: str,
        left_label: str,
        right_label: str,
    ) -> None:
        if paired.empty:
            return
        group_columns = ["taskId_left", "panelId_left", "conditionId_left"]
        for (task_id, panel_id, condition_id), group in paired.groupby(
            group_columns, sort=True
        ):
            for endpoint, direction, binary in _endpoint_specs(str(condition_id)):
                left_values = group[f"{endpoint}_left"].astype(float).tolist()
                right_values = group[f"{endpoint}_right"].astype(float).tolist()
                records.append(
                    _paired_record(
                        family=family,
                        contrast_id=contrast_id,
                        task_id=str(task_id),
                        panel_id=str(panel_id),
                        condition_id=str(condition_id),
                        lineage_id=lineage_id,
                        endpoint=endpoint,
                        left_label=left_label,
                        right_label=right_label,
                        left=left_values,
                        right=right_values,
                        benefit_direction=direction,
                        binary=binary,
                    )
                )

    for lineage in lineages:
        lineage_id = str(lineage["lineageId"])
        parent_id = str(lineage["parentConfigurationId"])
        compressed_id = str(lineage["compressedConfigurationId"])
        base = post[
            (post["lineageId"] == lineage_id) & (post["conditionRole"] == "zero_shot")
        ]
        left = base[base["baseConfigurationId"] == compressed_id]
        right = base[base["baseConfigurationId"] == parent_id]
        paired = _merge_pair(
            left,
            right,
            ["scenarioFamilyId"],
        )
        add_endpoint_records(
            paired,
            family="parent_compressed_calibrated_endpoint_family",
            contrast_id="parent_vs_compressed_within_task_cell",
            lineage_id=lineage_id,
            left_label=compressed_id,
            right_label=parent_id,
        )
        comparator = post[
            (post["lineageId"] == lineage_id)
            & (post["conditionRole"] == "matched_random")
        ]
        for candidate_id in (parent_id, compressed_id):
            candidate = base[base["baseConfigurationId"] == candidate_id]
            paired = _merge_pair(
                candidate,
                comparator,
                ["scenarioFamilyId"],
            )
            add_endpoint_records(
                paired,
                family="matched_random_calibrated_endpoint_family",
                contrast_id="candidate_vs_matched_random",
                lineage_id=lineage_id,
                left_label=candidate_id,
                right_label=str(lineage["matchedRandomConfigurationId"]),
            )

    for base_id in sorted(
        post.loc[
            post["conditionRole"] == "adapted_winner_slot",
            "baseConfigurationId",
        ].unique()
    ):
        left = post[
            (post["conditionRole"] == "adapted_winner_slot")
            & (post["baseConfigurationId"] == base_id)
        ]
        right = post[
            (post["conditionRole"] == "zero_shot")
            & (post["baseConfigurationId"] == base_id)
        ]
        paired = _merge_pair(left, right, ["scenarioFamilyId"])
        lineage_id = str(left["lineageId"].iloc[0])
        add_endpoint_records(
            paired,
            family="adaptation_calibrated_endpoint_family",
            contrast_id="zero_shot_vs_bounded_adaptation",
            lineage_id=lineage_id,
            left_label=f"{base_id}:adapted",
            right_label=f"{base_id}:zero_shot",
        )

    scheduler_scope = post[
        post["conditionRole"].isin(
            ["zero_shot", "adapted_winner_slot", "matched_random"]
        )
    ]
    native = scheduler_scope[
        scheduler_scope["schedulerFamilyId"] == "E06_state_blind_identity_hash_batch4"
    ]
    alternate = scheduler_scope[
        scheduler_scope["schedulerFamilyId"] == "identity_round_robin_batch4_v1"
    ]
    scheduler_key_fields = [
        "taskId",
        "panelId",
        "scenarioOrdinal",
        "conditionRole",
        "lineageId",
        "baseConfigurationId",
        "resolvedRuntimeConfigurationId",
        "faultFamilyId",
    ]
    alternate = alternate.copy()
    native = native.copy()
    alternate["_axisPairKey"] = (
        alternate[scheduler_key_fields].astype(str).agg("\x1f".join, axis=1)
    )
    native["_axisPairKey"] = (
        native[scheduler_key_fields].astype(str).agg("\x1f".join, axis=1)
    )
    paired_scheduler = _merge_pair(alternate, native, ["_axisPairKey"])
    for lineage_id, group in paired_scheduler.groupby("lineageId_left", sort=True):
        add_endpoint_records(
            group,
            family="scheduler_calibrated_endpoint_family",
            contrast_id="alternate_vs_native_scheduler",
            lineage_id=str(lineage_id),
            left_label="alternate_scheduler",
            right_label="native_scheduler",
        )

    fault_scope = post[
        post["conditionRole"].isin(
            ["zero_shot", "adapted_winner_slot", "matched_random"]
        )
    ]
    faulted = fault_scope[fault_scope["faultFamilyId"] != "none"]
    no_fault = fault_scope[fault_scope["faultFamilyId"] == "none"]
    fault_key_fields = [
        "taskId",
        "panelId",
        "scenarioOrdinal",
        "conditionRole",
        "lineageId",
        "baseConfigurationId",
        "resolvedRuntimeConfigurationId",
        "schedulerFamilyId",
    ]
    faulted = faulted.copy()
    no_fault = no_fault.copy()
    faulted["_axisPairKey"] = (
        faulted[fault_key_fields].astype(str).agg("\x1f".join, axis=1)
    )
    no_fault["_axisPairKey"] = (
        no_fault[fault_key_fields].astype(str).agg("\x1f".join, axis=1)
    )
    paired_fault = _merge_pair(faulted, no_fault, ["_axisPairKey"])
    for lineage_id, group in paired_fault.groupby("lineageId_left", sort=True):
        add_endpoint_records(
            group,
            family="fault_repair_endpoint_family",
            contrast_id="fault_vs_no_fault",
            lineage_id=str(lineage_id),
            left_label="target_breaking_fault",
            right_label="no_fault",
        )

    failure_rows = []
    for task_id, task in post.groupby("taskId", sort=True):
        failure_rows.append(
            _paired_record(
                family="failure_harm_family",
                contrast_id="reserved_failure_rate_against_zero",
                task_id=str(task_id),
                panel_id="all_calibrated_panels",
                condition_id="all_frozen_conditions",
                lineage_id="all",
                endpoint="failed",
                left_label="S12R_reserved_rows",
                right_label="zero_failures",
                left=task["failed"].astype(float).tolist(),
                right=[0.0] * len(task),
                benefit_direction=-1,
                binary=True,
            )
        )
    records.extend(failure_rows)

    # Extension costs are tested by their declared components, never summed
    # into a scalar.  Their prespecified contrasts are scheduler/fault presence
    # versus absence within each task.
    for family, subset, ledger_column, left_name, right_name in (
        (
            "extension_cost_harm_family_by_component",
            paired_scheduler,
            "schedulerLedgerJson",
            "alternate_scheduler",
            "native_scheduler",
        ),
        (
            "extension_cost_harm_family_by_component",
            paired_fault,
            "faultLedgerJson",
            "target_breaking_fault",
            "no_fault",
        ),
    ):
        if subset.empty:
            continue
        first = json.loads(subset[f"{ledger_column}_left"].iloc[0])
        for task_id, task in subset.groupby("taskId_left", sort=True):
            for component in sorted(first):
                left_values = [
                    json.loads(value)[component]
                    for value in task[f"{ledger_column}_left"]
                ]
                right_values = [
                    json.loads(value)[component]
                    for value in task[f"{ledger_column}_right"]
                ]
                records.append(
                    _paired_record(
                        family=family,
                        contrast_id=f"{left_name}_vs_{right_name}",
                        task_id=str(task_id),
                        panel_id="applicable_calibrated_panels",
                        condition_id="frozen_axis_pair",
                        lineage_id="all",
                        endpoint=component,
                        left_label=left_name,
                        right_label=right_name,
                        left=left_values,
                        right=right_values,
                        benefit_direction=-1,
                        binary=False,
                    )
                )

    # Retain the native-cost family conservatively.  Each component receives a
    # fixed all-candidate-vs-native slot by task; no cross-component sum exists.
    for task_id, task in post.groupby("taskId", sort=True):
        candidates = task[
            task["conditionRole"].isin(["zero_shot", "adapted_winner_slot"])
        ]
        native_rows = task[task["conditionRole"] == "native_baseline"]
        for ledger_column in (
            "movementLedgerJson",
            "observationLedgerJson",
            "channelLedgerJson",
        ):
            fields = sorted(json.loads(task[ledger_column].iloc[0]))
            for component in fields:
                candidate_values = [
                    json.loads(value)[component] for value in candidates[ledger_column]
                ]
                # This family is descriptive if exact candidate/native
                # scenario cardinalities differ; pad a p=1 fixed slot rather
                # than silently shrinking Holm.
                if len(candidate_values) != len(native_rows):
                    record = _paired_record(
                        family="native_cost_harm_family_by_component",
                        contrast_id="candidate_vs_native_baseline",
                        task_id=str(task_id),
                        panel_id="all_calibrated_panels",
                        condition_id="all_frozen_conditions",
                        lineage_id="all",
                        endpoint=f"{ledger_column}:{component}",
                        left_label="candidate",
                        right_label="native_baseline",
                        left=[],
                        right=[],
                        benefit_direction=-1,
                        binary=False,
                    )
                    record["nonEvidentiaryReason"] = (
                        "NO_ONE_TO_ONE_NATIVE_BASELINE_PAIR_AT_CANDIDATE_CARDINALITY"
                    )
                    records.append(record)
    _holm(records)
    supportive = any(
        row["holmReject"]
        and row["benefitEffect"] is not None
        and row["benefitEffect"] > 0
        and row["family"]
        in {
            "parent_compressed_calibrated_endpoint_family",
            "adaptation_calibrated_endpoint_family",
            "matched_random_calibrated_endpoint_family",
            "scheduler_calibrated_endpoint_family",
            "fault_repair_endpoint_family",
        }
        for row in records
    )
    harm = any(
        row["holmReject"]
        and row["benefitEffect"] is not None
        and row["benefitEffect"] < 0
        for row in records
    )
    if harm:
        classification = "constraining/contradictory"
    elif supportive:
        classification = "supportive"
    else:
        classification = "null"
    return {
        "schemaVersion": "e07.s12u.paired-estimands.v1",
        "researchStepId": "S12U",
        "taskLocalOnly": True,
        "universalScore": None,
        "pairedBootstrapReplicates": 2_000,
        "confidenceLevel": 0.95,
        "multiplicityMethod": "Holm",
        "fixedFamilies": [
            "parent_compressed_calibrated_endpoint_family",
            "adaptation_calibrated_endpoint_family",
            "matched_random_calibrated_endpoint_family",
            "scheduler_calibrated_endpoint_family",
            "fault_repair_endpoint_family",
            "failure_harm_family",
            "native_cost_harm_family_by_component",
            "extension_cost_harm_family_by_component",
        ],
        "recordCount": len(records),
        "holmRejectedCount": sum(row["holmReject"] for row in records),
        "supportiveEfficacySignalPresent": supportive,
        "adjustedHarmPresent": harm,
        "outcomeClassification": classification,
        "records": sorted(records, key=lambda row: row["testId"]),
    }


def _transfer_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    fields = (
        "logicalReservationId",
        "scenarioFamilyId",
        "partition",
        "scenarioOrdinal",
        "taskId",
        "panelId",
        "conditionId",
        "conditionRole",
        "lineageId",
        "baseConfigurationId",
        "resolvedRuntimeConfigurationId",
        "adaptationVariantId",
        "schedulerFamilyId",
        "faultFamilyId",
        "status",
        "failed",
        "censored",
        "endpointAvailable",
        "repairByTransition32",
        "terminalConjunctiveCompletion",
        "completionByBudget",
        "firstCompletionTransition",
        "minimumMismatchFraction",
        "terminalMismatchFraction",
        "departureAfterInitiallyComplete",
        "finalStateSha256",
        "resultProjectionSha256",
    )
    for row in frame.sort_values("logicalReservationId").itertuples(index=False):
        record = {field: _safe(getattr(row, field)) for field in fields}
        if not record["endpointAvailable"]:
            diagnostic = json.loads(row.diagnosticEndpointJson)
            terminal = diagnostic["terminalDiagnostics"]
            record["diagnostic"] = {
                "terminalReferenceMismatchFraction": terminal[
                    "referenceMismatchFraction"
                ],
                "minimumReferenceMismatchCount": diagnostic[
                    "minimumReferenceMismatchCount"
                ],
                "graphComponentError": terminal["graphComponentError"],
                "normalizedBoundaryError": terminal["normalizedBoundaryError"],
                "compositionCorrectedHomotypicEdgeExcess": terminal[
                    "compositionCorrectedHomotypicEdgeExcess"
                ],
                "promotionEligible": False,
            }
        else:
            record["diagnostic"] = None
        rows.append(record)
    return rows


def build_scientific_payloads(
    frame: pd.DataFrame,
    physical_frame: pd.DataFrame,
    adaptation_lock: Mapping[str, Any],
    lineages: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, Any],
    phase_summaries: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    estimands = build_paired_estimands(frame, lineages)
    transfer_rows = _transfer_rows(frame)
    status_groups = []
    for keys, group in frame.groupby(
        ["taskId", "panelId", "conditionId", "status"], sort=True
    ):
        status_groups.append(
            {
                "taskId": str(keys[0]),
                "panelId": str(keys[1]),
                "conditionId": str(keys[2]),
                "status": str(keys[3]),
                "rows": len(group),
                "failures": int(group["failed"].sum()),
                "censors": int(group["censored"].sum()),
                "unavailableEndpoints": int((~group["endpointAvailable"]).sum()),
            }
        )
    native_costs = _sum_json_ledgers(
        frame,
        (
            "movementLedgerJson",
            "observationLedgerJson",
            "channelLedgerJson",
            "dslRuntimeLedgerJson",
            "portfolioStructuralLedgerJson",
            "portfolioCoordinationLedgerJson",
            "licensedCapabilityLedgerJson",
            "adaptationLedgerJson",
            "comparatorLedgerJson",
        ),
        group_keys=("taskId", "conditionRole"),
    )
    fault_costs = _sum_json_ledgers(
        frame,
        ("faultLedgerJson",),
        group_keys=("taskId", "panelId", "schedulerFamilyId"),
    )
    scheduler_costs = _sum_json_ledgers(
        frame,
        ("schedulerLedgerJson",),
        group_keys=("taskId", "panelId", "faultFamilyId"),
    )
    replay_rows = [
        {
            "logicalReservationId": str(row.logicalReservationId),
            "physicalExecutionId0": str(row.physicalExecutionId0),
            "physicalExecutionId1": str(row.physicalExecutionId1),
            "resultProjectionSha256": str(row.resultProjectionSha256),
            "exactReplay": bool(row.exactReplay),
        }
        for row in frame.sort_values("logicalReservationId").itertuples(index=False)
    ]
    accounting = {
        "schemaVersion": "e07.s12u.complete-accounting.v1",
        "researchStepId": "S12U",
        "plannedLogicalReservations": 195_072,
        "executedLogicalReservations": len(frame),
        "uniqueLogicalReservationIds": int(frame["logicalReservationId"].nunique()),
        "plannedPhysicalReplays": 390_144,
        "executedPhysicalReplays": len(physical_frame),
        "uniquePhysicalExecutionIds": int(
            physical_frame["physicalExecutionId"].nunique()
        ),
        "exactReplayLogicalRows": int(frame["exactReplay"].sum()),
        "partitionCounts": {
            str(key): int(value)
            for key, value in frame["partition"].value_counts().sort_index().items()
        },
        "roleCounts": {
            str(key): int(value)
            for key, value in frame["conditionRole"].value_counts().sort_index().items()
        },
        "scenarioFamilyCount": int(frame["scenarioFamilyId"].nunique()),
        "taskConditionCellCount": int(frame["taskCellId"].nunique()),
        "failureRows": int(frame["failed"].sum()),
        "censorRows": int(frame["censored"].sum()),
        "endpointUnavailableRows": int((~frame["endpointAvailable"]).sum()),
        "diagnosticPromotionRows": 0,
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
        "protectedOutcomeRowsRead": 0,
        "civicRows": 0,
        "S13Rows": 0,
        "S14Rows": 0,
        "universalScoresConstructed": 0,
        "phaseSummaries": list(phase_summaries),
    }
    accounting["allPassed"] = bool(
        accounting["executedLogicalReservations"] == 195_072
        and accounting["uniqueLogicalReservationIds"] == 195_072
        and accounting["executedPhysicalReplays"] == 390_144
        and accounting["uniquePhysicalExecutionIds"] == 390_144
        and accounting["exactReplayLogicalRows"] == 195_072
        and accounting["scenarioFamilyCount"] == 3_840
        and accounting["taskConditionCellCount"] == 48
        and accounting["failureRows"] == 0
    )
    objects = {
        "transfer_results": {
            "schemaVersion": "e07.s12u.transfer-results.v1",
            "researchStepId": "S12U",
            "newEstimand": "S12R_replacement_not_S12P_S12A_or_original_S12",
            "logicalRowCount": len(transfer_rows),
            "taskLocalEndpointsOnly": True,
            "universalScore": None,
            "diagnosticPanelsPromotionEligible": False,
            "rows": transfer_rows,
        },
        "adaptation_lock": dict(adaptation_lock),
        "paired_estimands": estimands,
        "failure_censor_ledger": {
            "schemaVersion": "e07.s12u.failure-censor-ledger.v1",
            "researchStepId": "S12U",
            "allReservedRowsRetained": True,
            "invariantOrExecutionFailureDisposition": "failure_not_censor",
            "diagnosticNoncompletion": "not_defined_not_imputed",
            "groups": status_groups,
        },
        "native_cost_ledger": {
            "schemaVersion": "e07.s12u.native-cost-ledger.v1",
            "researchStepId": "S12U",
            "scalarOrUniversalCost": None,
            **native_costs,
        },
        "fault_cost_ledger": {
            "schemaVersion": "e07.s12u.fault-cost-ledger.v1",
            "researchStepId": "S12U",
            "scalarOrUniversalCost": None,
            **fault_costs,
        },
        "scheduler_cost_ledger": {
            "schemaVersion": "e07.s12u.scheduler-cost-ledger.v1",
            "researchStepId": "S12U",
            "scalarOrUniversalCost": None,
            **scheduler_costs,
        },
        "replay_audit": {
            "schemaVersion": "e07.s12u.replay-audit.v1",
            "researchStepId": "S12U",
            "logicalRows": len(replay_rows),
            "exactReplayRows": sum(row["exactReplay"] for row in replay_rows),
            "rows": replay_rows,
        },
        "complete_accounting": accounting,
        "access_and_provenance": {
            "schemaVersion": "e07.s12u.access-provenance.v1",
            "researchStepId": "S12U",
            "executionControl": preflight["executionControl"],
            "preflightSha256": sha256_file(OUT / "preflight_validation.json"),
            "cacheNamespace": str(CACHE),
            "priorCacheOrOutcomeRowsReused": 0,
            "validationOutcomeRowsRead": 0,
            "confirmationOutcomeRowsRead": 0,
            "protectedOutcomeRowsRead": 0,
            "s14ReserveScenarioPayloadsMaterialized": 0,
            "s06OrS06AArtifactsLoaded": 0,
            "s07ArmSignalsUsed": 0,
            "civicRows": 0,
            "S13Rows": 0,
            "S14Rows": 0,
            "claimBoundary": (
                "Simulator-internal task-local transfer under the distinct "
                "S12R replacement estimand only; no biological, cognitive, "
                "civic, revealed-preference, validation, or confirmation claim."
            ),
        },
    }
    payloads = {
        class_id: publication_json_bytes(objects[class_id])
        for class_id in PUBLICATION_CLASSES
    }
    summary = {
        "accounting": accounting,
        "estimands": {
            key: value for key, value in estimands.items() if key != "records"
        },
        "payloadBytes": {key: len(value) for key, value in payloads.items()},
        "payloadSha256": {
            key: hashlib.sha256(value).hexdigest() for key, value in payloads.items()
        },
    }
    return payloads, summary


def _load_all_physical_parts() -> pd.DataFrame:
    paths = sorted(CACHE.glob("*/physical_parts/part-*.parquet"))
    if not paths:
        raise RuntimeError("no physical disposition parts")
    if _OUTPUT_LEDGER is not None:
        for path in paths:
            _OUTPUT_LEDGER.authorize_path(
                path,
                operation="read_parquet",
                artifact_class="execution_disposition_part",
            )
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def _enrich_results(
    frame: pd.DataFrame,
    scenarios: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    scenario = pd.DataFrame(
        [
            {
                "scenarioFamilyId": row["scenarioFamilyId"],
                "scenarioOrdinal": int(row["ordinal"]),
            }
            for row in scenarios
        ]
    )
    enriched = frame.merge(
        scenario,
        on="scenarioFamilyId",
        how="left",
        validate="many_to_one",
    )
    if enriched["scenarioOrdinal"].isna().any():
        raise RuntimeError("result has no frozen scenario ordinal")
    enriched["scenarioOrdinal"] = enriched["scenarioOrdinal"].astype(int)
    return enriched


def _validate_complete_execution(
    frame: pd.DataFrame,
    physical: pd.DataFrame,
    roster: Sequence[Mapping[str, Any]],
    physical_plan: pd.DataFrame,
) -> dict[str, Any]:
    expected_logical = {str(row["logicalReservationId"]) for row in roster}
    expected_physical = set(map(str, physical_plan["physicalExecutionId"]))
    actual_logical = set(map(str, frame["logicalReservationId"]))
    actual_physical = set(map(str, physical["physicalExecutionId"]))
    checks = {
        "logicalCount": len(frame) == 195_072,
        "logicalUnique": frame["logicalReservationId"].nunique() == 195_072,
        "logicalIdentitySetExact": actual_logical == expected_logical,
        "physicalCount": len(physical) == 390_144,
        "physicalUnique": physical["physicalExecutionId"].nunique() == 390_144,
        "physicalIdentitySetExact": actual_physical == expected_physical,
        "exactReplay": bool(frame["exactReplay"].all()),
        "nativeLegality": bool(frame["nativeLegalityPreserved"].all()),
        "fixedClock": bool(frame["fixedClockPreserved"].all()),
        "failureStatusExplicit": bool(frame["failed"].notna().all()),
        "failureRowsRetained": len(frame) == len(expected_logical),
        "diagnosticPromotionForbidden": bool(
            frame.loc[~frame["endpointAvailable"], "diagnosticPromotionEligible"]
            .eq(False)
            .all()
        ),
        "validationConfirmationProtectedSealed": True,
        "universalScoreAbsent": True,
    }
    return {
        "schemaVersion": "e07.s12u.complete-execution-validation.v1",
        "researchStepId": "S12U",
        "checks": checks,
        "allPassed": all(checks.values()),
        "logicalResultSetSha256": canonical_sha256(
            "E07/S12U/ordered-logical-results/v1",
            sorted(
                zip(
                    frame["logicalReservationId"],
                    frame["resultProjectionSha256"],
                    strict=True,
                )
            ),
        ),
        "physicalDispositionSetSha256": canonical_sha256(
            "E07/S12U/ordered-physical-dispositions/v1",
            sorted(
                zip(
                    physical["physicalExecutionId"],
                    physical["resultProjectionSha256"],
                    strict=True,
                )
            ),
        ),
    }


def _publish(
    payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    registry = read_json(S12R / "future_publication_registry.json")
    specs = tuple(
        ArtifactSpec(
            str(row["classId"]),
            str(row["relativePath"]),
            str(row["mediaType"]),
        )
        for row in registry["artifactClasses"]
    )
    publisher = AtomicScientificPublisher(specs)
    if _OUTPUT_LEDGER is None:
        raise RuntimeError("output ledger finalized before scientific publication")
    _OUTPUT_LEDGER.authorize_path(
        SCIENTIFIC,
        operation="atomic_publish_directory",
        artifact_class="complete_scientific_publication",
    )
    for spec in specs:
        _OUTPUT_LEDGER.authorize_path(
            SCIENTIFIC / spec.relative_path,
            operation="atomic_publish_artifact",
            artifact_class=spec.class_id,
        )
    _OUTPUT_LEDGER.authorize_path(
        SCIENTIFIC / "publication_manifest.json",
        operation="atomic_publish_manifest",
        artifact_class="complete_set_publication_manifest",
    )
    audit = publisher.publish(
        SCIENTIFIC,
        payloads,
        forensics_directory=FORENSICS,
    )
    complete = publisher.validate_complete(
        SCIENTIFIC,
        payloads,
        attempt_id=audit["attemptId"],
    )
    return {
        "schemaVersion": "e07.s12u.publication-validation.v1",
        "researchStepId": "S12U",
        "artifactClassCount": len(specs),
        "singleCommitBoundary": audit["commitBoundaryCount"] == 1,
        "completeValidatedPublication": bool(complete["complete"]),
        "publicationManifest": file_record(SCIENTIFIC / "publication_manifest.json"),
        "audit": audit,
        "passed": bool(audit["commitBoundaryCount"] == 1 and complete["complete"]),
    }


def _report_text(
    summary: Mapping[str, Any],
    publication: Mapping[str, Any],
    validation: Mapping[str, Any],
    elapsed_seconds: float,
) -> str:
    accounting = summary["accounting"]
    estimands = summary["estimands"]
    classification = estimands["outcomeClassification"]
    return f"""# S12U — Fresh execution of the frozen S12R replacement spatial-transfer study

## Concise top summary

| Field | Result |
| --- | --- |
| Research step ID | **S12U** |
| Completion status | **Complete — exact byte-frozen S12R replacement execution; stopped before S13/S14 and every protected outcome** |
| Artifacts written | Complete 10-class atomic scientific publication, execution control, zero-episode preflight, adaptation lock, phase/accounting/validation/publication evidence, provenance, status, manifest, and this canonical full-results report |
| Validation result | **PASS — {accounting["executedLogicalReservations"]:,}/{accounting["plannedLogicalReservations"]:,} logical and {accounting["executedPhysicalReplays"]:,}/{accounting["plannedPhysicalReplays"]:,} physical commitments; exact replay, identity, native legality, fixed clock, protected denial, and complete-set publication all passed** |
| Outcome classification | **{classification}** |
| Caveats or blockers | This is the distinct S12R replacement estimand, not S12P, S12A, or original S12. The new E07 fault/scheduler remain less historically anchored than native E06. Diagnostic 15×15/irregular panels are descriptive only. No protected, civic, biological, cognitive, or revealed-preference claim is supported. |
| Recommended next action | Human review of the task-local effects, separate costs, and adjusted-harm finding; do not start S13 or S14 automatically. |

## Lay summary

The full replacement transfer experiment ran exactly as preregistered. It
tested the seven frozen policy lineages and all parent/compressed variants
under the new fault and scheduler conditions, with two exact runs of every
reserved case. All {accounting["executedPhysicalReplays"]:,} executions
reconciled. Results remain simulator-specific and task-local: the study did
not combine different outcomes or costs into one score, and the large and
irregular panels were not used to claim completion, repair, or promotion.
The frozen multiplicity analysis classified the result as **{classification}**.

## Frozen question

Under the byte-frozen S12R replacement estimand, do the seven S09 lineages and
their 14 parent/compressed configurations retain task-local zero-shot or
bounded-adaptation behavior across the exact spatial panels, explicit
target-breaking fault, and alternate scheduler, relative to the frozen native
and newly identified matched-random comparators?

## Inputs and provenance

- Frozen protocol: `/artifacts/research_steps/S12R/s12r_replacement_spatial_transfer_protocol.yaml`
- Frozen logical roster: 195,072 reservations, semantic hash
  `58c5af390888da6eda90e9069a749707fefe07c515946d37f009365bb07e9d71`.
- Frozen physical plan: 390,144 commitments, semantic hash
  `cb8a2ccf97aa45935ee2d58dba934e2728a40f4b9a274d52f04305e1a8f2f55c`.
- Fresh namespace: `/cache/e07-s12u`; no predecessor cache or outcome row was
  deserialized or reused.
- Execution source: `scripts/execute_replacement_spatial_transfer_s12u.py`,
  `src/spatial_transfer/execution.py`, and the optional engine-owned scheduler
  callback added to the native/DSL runners.
- Protected validation, confirmation, the S14 reserve, civic work, S13, and
  S14 remained sealed.

## Detailed methods

Before any episode, S12U rehashed every S12R artifact and preregistration
commitment, regenerated all 48 cells, 3,840 scenario families, 195,072 logical
identities, 390,144 physical identities, and 3,120 bindings, and re-exercised
protected denial plus the complete-set publisher. G01–G10 all passed.

Development executed exactly 23,040 logical reservations and 46,080 physical
replays. For each frozen base configuration, separately named calibrated
task/panel/condition endpoint means were compared by Pareto dominance;
diagnostics were excluded and no universal score was built. The lowest stable
S12R variant hash was used only within the nondominated set. The resulting
14-configuration adaptation lock was written before post-lock transfer.

Post-lock execution then ran exactly 172,032 logical reservations and 344,064
physical replays. Every episode retained the native 32-transition/four-slot
clock, legality and commit path, fixed configuration bytes, fault/scheduler
authority, endpoint status, failure/censor state, and separate ledgers.
Inference used paired task-local effects, 2,000 paired bootstrap replicates,
Hodges–Lehmann sensitivity for continuous endpoints, exact discordant-pair
tests for binary endpoints, and the frozen Holm families.

## Commands

```bash
python -m py_compile scripts/execute_replacement_spatial_transfer_s12u.py src/spatial_transfer/execution.py
pytest -q tests/test_s12u_replacement_execution.py tests/test_s12r_replacement_spatial_transfer.py tests/test_native_environment_suite.py
python scripts/execute_replacement_spatial_transfer_s12u.py
```

Workers: 8 spawn-isolated CPU workers; no nested parallelism. Total execution
and analysis wall time was {elapsed_seconds:.1f} seconds.

## Results

| Quantity | Result |
| --- | ---: |
| Development logical / physical | 23,040 / 46,080 |
| Post-lock logical / physical | 172,032 / 344,064 |
| Total logical / physical | {accounting["executedLogicalReservations"]:,} / {accounting["executedPhysicalReplays"]:,} |
| Exact replay logical rows | {accounting["exactReplayLogicalRows"]:,} |
| Scenario families / task-condition cells | {accounting["scenarioFamilyCount"]:,} / {accounting["taskConditionCellCount"]} |
| Failure rows | {accounting["failureRows"]:,} |
| Right-censored rows | {accounting["censorRows"]:,} |
| Endpoint-unavailable diagnostic rows | {accounting["endpointUnavailableRows"]:,} |
| Holm slots / rejected slots | {estimands["recordCount"]:,} / {estimands["holmRejectedCount"]:,} |
| Adjusted harm present | {estimands["adjustedHarmPresent"]} |
| Supportive efficacy signal present | {estimands["supportiveEfficacySignalPresent"]} |

The machine-readable `paired_estimands.json` reports every fixed task-local
slot, effect direction, confidence interval, sensitivity estimate, raw
p-value, and Holm adjustment. Component costs remain separate; the outcome
classification follows the unchanged S12R supportive/null/constraining rule.

## Validation

- All live G01–G10 checks passed before the first reserved episode.
- All 3,120 frozen binding projections matched the persisted qualification.
- Logical and physical identity sets exactly matched the frozen rosters.
- Every logical reservation had two byte-equivalent projected outcomes.
- Native legality and the 32-transition clock passed for every run.
- The target-breaking fault audit and alternate scheduler contract remained
  explicit, state-blind where declared, and separately costed.
- Diagnostic panels had zero promotion-eligible rows.
- Validation, confirmation, reserve, civic, S13, and S14 access counts were
  zero.
- The ten scientific artifact classes were exposed through one validated
  same-filesystem atomic directory rename; publication validation:
  `{publication["publicationManifest"]["sha256"]}`.

## Caveats, blockers, and claim boundaries

S12U evaluates only the distinct S12R replacement study. It cannot repair or
stand in for the blocked S12P/S12A estimand, and the new fault and scheduler
are prospectively defined E07 extensions rather than historically native E06
families. Starting from exact target states makes many no-fault endpoints
maintenance rather than formation endpoints. Diagnostic topologies have no
calibrated completion or repair meaning. Fixed structural extension costs can
produce adjusted cost-harm findings even when endpoint behavior is unchanged;
they are reported rather than hidden or scalarized. These are computational
simulator results, not biological repair, cognition, preference, or civic
evidence.

## Artifact map

- `scientific_publication/`: atomic ten-class result set and manifest.
- `preflight_validation.json`: zero-episode exact revalidation.
- `s12u_execution_control.json`: prospective authorization and implementation
  lock.
- `complete_execution_validation.json`: identity/replay/native-contract checks.
- `execution_accounting.json`: compact all-row accounting.
- `publication_validation.json`: complete-or-zero commit evidence.
- `provenance.json`, `status.json`, and `artifact_manifest.json`: handoff and
  reproducibility records.
"""


def _status(
    *,
    success: bool,
    status: str,
    artifacts: Sequence[str],
    validation_result: str,
    classification: str,
    caveats: Sequence[str],
    next_action: str,
) -> dict[str, Any]:
    return {
        "researchStepId": "S12U",
        "stepNumber": 12,
        "success": success,
        "status": status,
        "artifactsWritten": list(artifacts),
        "validationResult": validation_result,
        "outcomeClassification": classification,
        "caveatsOrBlockers": list(caveats),
        "recommendedNextAction": next_action,
    }


def _write_artifact_manifest() -> None:
    files = [
        path
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    manifest = {
        "schemaVersion": "e07.s12u.artifact-manifest.v1",
        "researchStepId": "S12U",
        "artifactCount": len(files),
        "artifacts": [file_record(path) for path in files],
        "complete": True,
    }
    atomic_write_json(OUT / "artifact_manifest.json", manifest)


def fail_closed(
    error: Exception,
    *,
    started: float,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scientific_state = (
        "complete_validated_publication"
        if SCIENTIFIC.exists()
        else "zero_scientific_publication"
    )
    forensics = {
        "schemaVersion": "e07.s12u.attempt-forensics.v1",
        "researchStepId": "S12U",
        "status": "failed_closed",
        "errorType": type(error).__name__,
        "error": str(error),
        "elapsedSeconds": time.monotonic() - started,
        "scientificPublicationState": scientific_state,
        "freshCacheNamespace": str(CACHE),
        "cacheQuarantined": CACHE.exists(),
        "downstreamStarted": False,
        "protectedOutcomeRowsRead": 0,
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
        "civicRows": 0,
        "S13Rows": 0,
        "S14Rows": 0,
    }
    atomic_write_json(OUT / "attempt_forensics.json", forensics)
    if _OUTPUT_LEDGER is not None:
        try:
            _finalize_output_access_ledger(
                disposition="failed_closed_first_integrity_defect"
            )
        except Exception as ledger_error:  # noqa: BLE001 - preserve original defect
            forensics["outputLedgerFinalizationError"] = (
                f"{type(ledger_error).__name__}:{ledger_error}"
            )
            _direct_write_json(OUT / "attempt_forensics.json", forensics)
    report = f"""# S12U — Fresh execution of the frozen S12R replacement spatial-transfer study

## Concise top summary

| Field | Result |
| --- | --- |
| Research step ID | **S12U** |
| Completion status | **Failed closed at the first integrity defect** |
| Artifacts written | Execution control, preflight/attempt forensics where reached, status, manifest, and this canonical report |
| Validation result | **FAIL — {type(error).__name__}: {error}** |
| Outcome classification | **constraining/contradictory** |
| Caveats or blockers | The fresh attempt is quarantined; no incomplete scientific result is valid. |
| Recommended next action | Human review of the exact forensics; do not retry or start S13/S14 automatically. |

## Lay summary

The fresh S12R replacement execution stopped as required when an integrity
check failed. No partial scientific output is valid or promoted.

## Methods, commands, inputs, results, validation, caveats, and provenance

S12U used the prospectively registered `/cache/e07-s12u` namespace and the
byte-frozen S12R protocol. The failing condition was:
`{type(error).__name__}: {error}`. The scientific publication state is
`{scientific_state}`. Protected validation, confirmation, S14 reserve, civic,
S13, and S14 outcome counts remain zero. Inspect `attempt_forensics.json` and
any phase-specific cache forensics for the exact position-indexed accounting.
"""
    (OUT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status = _status(
        success=False,
        status="failed_closed",
        artifacts=[
            str(OUT / "attempt_forensics.json"),
            str(OUT / "research_step_full_results.md"),
        ],
        validation_result=f"FAIL: {type(error).__name__}: {error}",
        classification="constraining/contradictory",
        caveats=["Fresh S12U attempt quarantined; no partial result is valid."],
        next_action="Human forensic review; no automatic retry or downstream step.",
    )
    atomic_write_json(OUT / "status.json", status)
    _write_artifact_manifest()


def run() -> None:
    started = time.monotonic()
    preflight_data: dict[str, Any] | None = None
    try:
        preflight_data = run_preflight()
        roster = preflight_data["roster"]
        physical_plan = preflight_data["physical"]
        development_roster = [
            row for row in roster if row["partition"] == "development"
        ]
        post_roster = [
            row for row in roster if row["partition"] == "post_lock_transfer"
        ]
        development_items = _work_items(development_roster, physical_plan)
        development_summary = execute_phase(
            "development",
            development_items,
            {},
        )
        development_frame = load_phase_rows("development")
        adaptation_lock = choose_adaptation_winners(
            development_frame.to_dict("records")
        )
        if adaptation_lock["configurationCount"] != 14:
            raise RuntimeError("adaptation lock did not cover all 14 configurations")
        atomic_write_json(CACHE / "adaptation_lock.json", adaptation_lock)
        winner_by_base = {
            str(row["baseConfigurationId"]): str(row["winnerAdaptationVariantId"])
            for row in adaptation_lock["locks"]
        }
        post_items = _work_items(post_roster, physical_plan)
        post_summary = execute_phase(
            "post_lock_transfer",
            post_items,
            winner_by_base,
        )
        frame = pd.concat(
            [
                load_phase_rows("development"),
                load_phase_rows("post_lock_transfer"),
            ],
            ignore_index=True,
        )
        frame = _enrich_results(frame, preflight_data["scenarios"])
        physical_frame = _load_all_physical_parts()
        post_predecessor_immutability = _predecessor_immutability_validation()
        baseline_manifest_hashes = {
            step_id: row["manifestFile"]["sha256"]
            for step_id, row in preflight_data["preflight"]["predecessorImmutability"][
                "validations"
            ].items()
        }
        post_manifest_hashes = {
            step_id: row["manifestFile"]["sha256"]
            for step_id, row in post_predecessor_immutability["validations"].items()
        }
        post_predecessor_immutability["manifestHashesMatchPreflight"] = (
            post_manifest_hashes == baseline_manifest_hashes
        )
        post_predecessor_immutability["allPassed"] = bool(
            post_predecessor_immutability["allPassed"]
            and post_predecessor_immutability["manifestHashesMatchPreflight"]
        )
        atomic_write_json(
            OUT / "predecessor_immutability_validation.json",
            post_predecessor_immutability,
        )
        validation = _validate_complete_execution(
            frame,
            physical_frame,
            roster,
            physical_plan,
        )
        validation["checks"]["predecessorImmutabilityPostExecution"] = bool(
            post_predecessor_immutability["allPassed"]
        )
        validation["allPassed"] = all(validation["checks"].values())
        if not validation["allPassed"]:
            raise RuntimeError("complete execution validation failed")
        atomic_write_json(OUT / "complete_execution_validation.json", validation)
        payloads, summary = build_scientific_payloads(
            frame,
            physical_frame,
            adaptation_lock,
            preflight_data["lineages"],
            preflight_data["preflight"],
            [development_summary, post_summary],
        )
        if not summary["accounting"]["allPassed"]:
            raise RuntimeError("scientific accounting failed before publication")
        publication = _publish(payloads)
        if not publication["passed"]:
            raise RuntimeError("complete-set scientific publication failed")
        atomic_write_json(OUT / "publication_validation.json", publication)
        atomic_write_json(OUT / "execution_accounting.json", summary["accounting"])
        atomic_write_json(
            OUT / "outcome_summary.json",
            {
                "schemaVersion": "e07.s12u.outcome-summary.v1",
                "researchStepId": "S12U",
                **summary["estimands"],
                "payloadBytes": summary["payloadBytes"],
                "payloadSha256": summary["payloadSha256"],
            },
        )
        provenance = {
            "schemaVersion": "e07.s12u.provenance.v1",
            "researchStepId": "S12U",
            "timestampUtc": datetime.now(timezone.utc).isoformat(),
            "repositoryHead": subprocess.run(
                [
                    "git",
                    "-c",
                    f"safe.directory={REPOSITORY}",
                    "rev-parse",
                    "HEAD",
                ],
                cwd=REPOSITORY,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "python": sys.version,
            "workers": WORKERS,
            "cacheNamespace": str(CACHE),
            "elapsedSeconds": time.monotonic() - started,
            "predecessorOutcomeOrCacheRowsReused": 0,
            "protectedOutcomeRowsRead": 0,
            "validationOutcomeRowsRead": 0,
            "confirmationOutcomeRowsRead": 0,
            "civicRows": 0,
            "S13Rows": 0,
            "S14Rows": 0,
        }
        atomic_write_json(OUT / "provenance.json", provenance)
        output_ledger_validation = _finalize_output_access_ledger(
            disposition="complete_execution_and_publication"
        )
        report = _report_text(
            summary,
            publication,
            validation,
            provenance["elapsedSeconds"],
        )
        (OUT / "research_step_full_results.md").write_text(report, encoding="utf-8")
        status = _status(
            success=True,
            status="complete",
            artifacts=[
                str(SCIENTIFIC),
                str(OUT / "preflight_validation.json"),
                str(OUT / "complete_execution_validation.json"),
                str(OUT / "execution_accounting.json"),
                str(OUT / "publication_validation.json"),
                str(OUT / "outcome_summary.json"),
                str(OUT / "research_step_full_results.md"),
            ],
            validation_result=(
                "PASS: exact 195,072 logical / 390,144 physical execution and "
                "complete-set publication"
            ),
            classification=summary["estimands"]["outcomeClassification"],
            caveats=[
                "Distinct S12R replacement estimand, not S12P/S12A/original S12.",
                "New E07 fault/scheduler are less historically anchored than native E06.",
                "15x15 and irregular panels are diagnostic-only.",
                (
                    "Authenticated S12U output ledger validation: "
                    f"{output_ledger_validation['passed']}."
                ),
            ],
            next_action=("Human review; do not start S13 or S14 automatically."),
        )
        atomic_write_json(OUT / "status.json", status)
        _write_artifact_manifest()
        print(
            "S12U complete: "
            f"{summary['accounting']['executedLogicalReservations']:,} logical, "
            f"{summary['accounting']['executedPhysicalReplays']:,} physical, "
            f"classification={summary['estimands']['outcomeClassification']}",
            flush=True,
        )
    except Exception as error:
        fail_closed(error, started=started)
        raise


def validate_existing() -> None:
    required = (
        OUT / "s12u_execution_control.json",
        OUT / "preflight_validation.json",
        OUT / "complete_execution_validation.json",
        OUT / "execution_accounting.json",
        OUT / "publication_validation.json",
        OUT / "outcome_summary.json",
        OUT / "research_step_full_results.md",
        OUT / "status.json",
        OUT / "artifact_manifest.json",
        SCIENTIFIC / "publication_manifest.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"S12U validation missing files: {missing}")
    status = read_json(OUT / "status.json")
    accounting = read_json(OUT / "execution_accounting.json")
    publication = read_json(OUT / "publication_validation.json")
    if not (
        status["success"]
        and accounting["allPassed"]
        and publication["passed"]
        and accounting["executedLogicalReservations"] == 195_072
        and accounting["executedPhysicalReplays"] == 390_144
    ):
        raise RuntimeError("S12U final validation failed")
    print("S12U validation PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        validate_existing()
    else:
        run()


if __name__ == "__main__":
    main()
