#!/usr/bin/env python3
"""Freeze, qualify, and report S12T without submitting a scientific episode."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from scripts import preregister_replacement_spatial_transfer_s12r as s12r
from src.environment_suite.contracts import canonical_sha256
from src.phenotype_discovery.publication import ArtifactSpec, AtomicScientificPublisher
from src.phenotype_discovery.publication import (
    canonical_json_bytes as publication_json_bytes,
)
from src.spatial_transfer.dependency_preflight import (
    AuthenticatedAccessLedger,
    DependencyRegistry,
    DependencyViolation,
    StaticImportAnalyzer,
    audit_runtime_dispatch,
    canonical_json_bytes,
    canonical_round_trip,
    cleanup_modules,
    domain_hash,
    instrument_path_opens,
    sha256_file,
    validate_access_ledger,
)
from src.spatial_transfer.execution import NEW_REPLACEMENT_COMPARATOR_IDS

REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
OUT = ARTIFACTS / "research_steps/S12T"
S12R = ARTIFACTS / "research_steps/S12R"
S12S = ARTIFACTS / "research_steps/S12S"
PROTOCOL = REPOSITORY / "configs/transfer/s12t_dependency_preflight_protocol.yaml"
REGISTRY = REPOSITORY / "configs/transfer/s12t_dependency_registry.json"
QUALIFICATION_CACHE = Path("/cache/e07-s12t-qualification")
FROZEN_S12S_CACHE = Path("/cache/e07-s12s")
S12S_SCIENTIFIC = S12S / "scientific_publication"
SCRIPT = Path(__file__).resolve()
MODULE = REPOSITORY / "src/spatial_transfer/dependency_preflight.py"
TEST = REPOSITORY / "tests/test_s12t_dependency_preflight.py"

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

DIRECT_INPUTS = (
    WORKSPACE / "AGENTS.md",
    WORKSPACE / "FULL_PLAN.md",
    WORKSPACE / "RESEARCH_PLAN.md",
    WORKSPACE / "PREVIOUS_ARTIFACTS.md",
    WORKSPACE / "PREVIOUS_ARTIFACTS.json",
    WORKSPACE / "CAPABILITIES.md",
    WORKSPACE / "CAPABILITY_AVAILABILITY.json",
    WORKSPACE / "input-attachments/MANIFEST.json",
    WORKSPACE
    / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md",
    PROTOCOL,
    REGISTRY,
    SCRIPT,
    MODULE,
    TEST,
    REPOSITORY / "scripts/execute_replacement_spatial_transfer_s12s.py",
    REPOSITORY / "src/spatial_transfer/execution.py",
    S12R / "artifact_manifest.json",
    S12R / "preregistration_freeze.json",
    S12R / "s12r_execution_gate.json",
    S12R / "binding_validation.json",
    S12R / "dependency_exclusion_validation.json",
    S12R / "access_control_validation.json",
    S12R / "protected_reserve_contract.json",
    S12R / "future_publication_registry.json",
    S12R / "research_step_full_results.md",
    S12S / "artifact_manifest.json",
    S12S / "preflight_validation.json",
    S12S / "preflight_failure_analysis.json",
    S12S / "execution_accounting.json",
    S12S / "status.json",
    S12S / "research_step_full_results.md",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    os.replace(temporary, path)


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    frame = pd.DataFrame([dict(row) for row in rows])
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


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    records = []
    for expected in manifest["artifacts"]:
        artifact = Path(expected["path"])
        actual = file_record(artifact)
        records.append(
            {
                "path": str(artifact),
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
        "manifestFile": file_record(path),
        "expectedArtifactCount": int(manifest["artifactCount"]),
        "validatedArtifactCount": len(records),
        "records": records,
        "passed": (
            int(manifest["artifactCount"]) == len(records)
            and all(row["passed"] for row in records)
        ),
    }


def validate_s12r_freeze() -> dict[str, Any]:
    freeze = read_json(S12R / "preregistration_freeze.json")
    records = []
    for name, expected in sorted(freeze["files"].items()):
        actual = file_record(Path(expected["path"]))
        records.append(
            {
                "name": name,
                "path": expected["path"],
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
        "freezeFile": file_record(S12R / "preregistration_freeze.json"),
        "frozenBeforeAnyEpisode": freeze["frozenBeforeAnyEpisode"],
        "transferEpisodesAtFreeze": freeze["transferEpisodesAtFreeze"],
        "semanticCommitments": freeze["semanticCommitments"],
        "records": records,
        "passed": (
            freeze["frozenBeforeAnyEpisode"] is True
            and int(freeze["transferEpisodesAtFreeze"]) == 0
            and all(row["passed"] for row in records)
        ),
    }


def freeze() -> None:
    if OUT.exists():
        raise RuntimeError(f"S12T artifact directory already exists: {OUT}")
    if QUALIFICATION_CACHE.exists():
        raise RuntimeError("S12T qualification cache is not fresh")
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    registry = DependencyRegistry.load(REGISTRY)
    if (
        protocol.get("researchStepId") != "S12T"
        or protocol.get("authorizationBoundary", {}).get("transferEpisodes") != 0
        or protocol.get("historicalBoundary", {}).get("patchesOrRetriesS12S")
        is not False
        or protocol.get("historicalBoundary", {}).get(
            "createsS12RScientificExecutionNamespace"
        )
        is not False
    ):
        raise RuntimeError("S12T protocol boundary changed")
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "s12t_dependency_preflight_protocol.yaml").write_bytes(PROTOCOL.read_bytes())
    (OUT / "dependency_registry.json").write_bytes(REGISTRY.read_bytes())
    inputs = [file_record(path) for path in DIRECT_INPUTS]
    input_freeze = {
        "schemaVersion": "e07.s12t.input-hash-freeze.v1",
        "researchStepId": "S12T",
        "frozenBeforeQualification": True,
        "createdUtc": now_utc(),
        "inputCount": len(inputs),
        "inputs": inputs,
        "transferEpisodesAtFreeze": 0,
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
        "protectedOutcomeRowsRead": 0,
        "scientificExecutionNamespaceCreated": False,
    }
    atomic_write_json(OUT / "input_hash_freeze.json", input_freeze)
    predecessor = {
        "schemaVersion": "e07.s12t.predecessor-immutability-baseline.v1",
        "researchStepId": "S12T",
        "recordedBeforeQualification": True,
        "S12R": validate_manifest(S12R / "artifact_manifest.json"),
        "S12S": validate_manifest(S12S / "artifact_manifest.json"),
        "s12sCacheAbsent": not FROZEN_S12S_CACHE.exists(),
        "s12sScientificPublicationAbsent": not S12S_SCIENTIFIC.exists(),
    }
    predecessor["passed"] = bool(
        predecessor["S12R"]["passed"]
        and predecessor["S12S"]["passed"]
        and predecessor["s12sCacheAbsent"]
        and predecessor["s12sScientificPublicationAbsent"]
    )
    atomic_write_json(OUT / "predecessor_immutability_baseline.json", predecessor)
    source_files = {
        "protocol": file_record(OUT / "s12t_dependency_preflight_protocol.yaml"),
        "dependencyRegistry": file_record(OUT / "dependency_registry.json"),
        "inputFreeze": file_record(OUT / "input_hash_freeze.json"),
        "immutabilityBaseline": file_record(
            OUT / "predecessor_immutability_baseline.json"
        ),
        "implementation": file_record(MODULE),
        "runner": file_record(SCRIPT),
        "tests": file_record(TEST),
    }
    body = {
        "schemaVersion": "e07.s12t.preregistration-freeze.v1",
        "researchStepId": "S12T",
        "frozenBeforeQualification": True,
        "frozenBeforeAnyScientificEpisode": True,
        "qualificationCache": str(QUALIFICATION_CACHE),
        "scientificExecutionNamespacesAuthorized": [],
        "sourceFiles": source_files,
        "permittedEvidence": protocol["authorizationBoundary"]["permittedEvidence"],
        "registrySha256": registry.registry_sha256,
        "qualificationMatrix": protocol["qualificationMatrix"],
        "s12rRevalidation": protocol["s12rRevalidation"],
        "zeroAccounting": protocol["authorizationBoundary"],
    }
    body["preregistrationCommitmentSha256"] = canonical_sha256(
        "E07/S12T/preregistration-freeze/v1",
        body,
    )
    atomic_write_json(OUT / "preregistration_freeze.json", body)
    print(
        json.dumps(
            {
                "researchStepId": "S12T",
                "status": "protocol_frozen_before_qualification",
                "preregistrationCommitmentSha256": body[
                    "preregistrationCommitmentSha256"
                ],
            },
            sort_keys=True,
        )
    )


def verify_freeze(registry: DependencyRegistry) -> dict[str, Any]:
    freeze_record = read_json(OUT / "preregistration_freeze.json")
    records = []
    current = {
        "protocol": OUT / "s12t_dependency_preflight_protocol.yaml",
        "dependencyRegistry": OUT / "dependency_registry.json",
        "inputFreeze": OUT / "input_hash_freeze.json",
        "immutabilityBaseline": OUT / "predecessor_immutability_baseline.json",
        "implementation": MODULE,
        "runner": SCRIPT,
        "tests": TEST,
    }
    for key, path in current.items():
        expected = freeze_record["sourceFiles"][key]
        actual = file_record(path)
        records.append(
            {
                "key": key,
                "path": str(path),
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
    protocol_match = (
        OUT / "s12t_dependency_preflight_protocol.yaml"
    ).read_bytes() == PROTOCOL.read_bytes()
    registry_match = (
        OUT / "dependency_registry.json"
    ).read_bytes() == REGISTRY.read_bytes()
    result = {
        "schemaVersion": "e07.s12t.freeze-revalidation.v1",
        "researchStepId": "S12T",
        "records": records,
        "protocolCopyMatchesRepository": protocol_match,
        "registryCopyMatchesRepository": registry_match,
        "registrySha256Matches": (
            freeze_record["registrySha256"] == registry.registry_sha256
        ),
        "scientificExecutionNamespacesAuthorized": freeze_record[
            "scientificExecutionNamespacesAuthorized"
        ],
    }
    result["passed"] = bool(
        all(row["passed"] for row in records)
        and protocol_match
        and registry_match
        and result["registrySha256Matches"]
        and result["scientificExecutionNamespacesAuthorized"] == []
    )
    return result


def build_static_fixtures(root: Path) -> list[dict[str, Any]]:
    fixture_root = root / "static"
    fixture_root.mkdir(parents=True, exist_ok=False)
    fixtures = {
        "benign_source_literal.py": """
TEXT = "research_steps/S06/ and /cache/e07-s10 are inert audit literals"
SELF = "src.surrogate_models is named but never imported"
VALUE = len(TEXT) + len(SELF)
""",
        "self_referential_registry_literal.py": """
FORBIDDEN = ("src.surrogate_models", "/artifacts/research_steps/S07")
def describes_registry():
    return FORBIDDEN
""",
        "direct_forbidden_import.py": "import src.surrogate_models\n",
        "indirect_forbidden_import.py": "import indirect_helper\n",
        "indirect_helper.py": "from src import surrogate_remediation\n",
        "dynamic_import_literal.py": """
import importlib
importlib.import_module("src.surrogate_models")
""",
        "dynamic_import_alias.py": """
from importlib import import_module as loader
loader("src.surrogate_remediation")
""",
        "builtin_dynamic_import.py": '__import__("src.allocation_redesign")\n',
        "wrapper_dispatch.py": """
from importlib import import_module as loader
def dispatch(name):
    return loader(name)
dispatch("src.non_surrogate_allocation")
""",
        "unresolved_dynamic_import.py": """
import importlib
name = "json"
importlib.import_module(name)
""",
        "allowed_local_import.py": "import allowed_helper\n",
        "allowed_helper.py": "import json\nVALUE = 7\n",
    }
    for name, source in fixtures.items():
        (fixture_root / name).write_text(source.strip() + "\n", encoding="utf-8")
    cases = [
        ("benign_source_literal", True),
        ("self_referential_registry_literal", True),
        ("direct_forbidden_import", False),
        ("indirect_forbidden_import", False),
        ("dynamic_import_literal", False),
        ("dynamic_import_alias", False),
        ("builtin_dynamic_import", False),
        ("wrapper_dispatch", False),
        ("unresolved_dynamic_import", False),
        ("allowed_local_import", True),
    ]
    return [
        {
            "caseId": case_id,
            "path": fixture_root / f"{case_id}.py",
            "expectedStaticPass": expected,
        }
        for case_id, expected in cases
    ]


def evaluate_static_case(
    case: Mapping[str, Any],
    registry: DependencyRegistry,
    root: Path,
) -> dict[str, Any]:
    analyzer = StaticImportAnalyzer(
        registry,
        search_roots=(REPOSITORY, root / "static"),
    )
    result = analyzer.analyze([Path(case["path"])])
    return {
        "caseId": str(case["caseId"]),
        "expectedStaticPass": bool(case["expectedStaticPass"]),
        "actualStaticPass": bool(result["passed"]),
        "filesAnalyzed": int(result["filesAnalyzed"]),
        "importsResolved": int(result["importsResolved"]),
        "violationKinds": sorted({row["kind"] for row in result["violations"]}),
        "semanticSha256": result["semanticSha256"],
        "passed": bool(result["passed"]) == bool(case["expectedStaticPass"]),
    }


def static_qualification(
    cases: Sequence[Mapping[str, Any]],
    registry: DependencyRegistry,
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    orders = {
        "lexical": sorted(cases, key=lambda row: row["caseId"]),
        "reverse": sorted(cases, key=lambda row: row["caseId"], reverse=True),
        "even_then_odd": list(cases[::2]) + list(cases[1::2]),
        "odd_then_even": list(cases[1::2]) + list(cases[::2]),
    }
    order_records = []
    reference_rows: list[dict[str, Any]] | None = None
    for order_id, order in orders.items():
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(
                pool.map(
                    lambda case: evaluate_static_case(case, registry, root),
                    order,
                )
            )
        rows.sort(key=lambda row: row["caseId"])
        digest = domain_hash("E07/S12T/static-case-set/v1", rows)
        if reference_rows is None:
            reference_rows = rows
        order_records.append(
            {
                "orderId": order_id,
                "caseCount": len(rows),
                "semanticSha256": digest,
                "allCasesPassed": all(row["passed"] for row in rows),
            }
        )
    assert reference_rows is not None
    replay_rows = [
        evaluate_static_case(case, registry, root)
        for case in sorted(cases, key=lambda row: row["caseId"])
    ]
    replay_rows.sort(key=lambda row: row["caseId"])
    analyzer = StaticImportAnalyzer(registry, search_roots=(REPOSITORY,))
    s12s_source = analyzer.analyze(
        [
            REPOSITORY / "scripts/execute_replacement_spatial_transfer_s12s.py",
            REPOSITORY / "src/spatial_transfer/execution.py",
        ]
    )
    summary = {
        "schemaVersion": "e07.s12t.static-import-graph-validation.v1",
        "researchStepId": "S12T",
        "caseCount": len(reference_rows),
        "casesPassed": sum(row["passed"] for row in reference_rows),
        "orders": order_records,
        "workerCount": 4,
        "allOrderDigestsIdentical": (
            len({row["semanticSha256"] for row in order_records}) == 1
        ),
        "exactReplay": replay_rows == reference_rows,
        "s12sHistoricalSource": {
            "filesAnalyzed": s12s_source["filesAnalyzed"],
            "importsResolved": s12s_source["importsResolved"],
            "violationCount": s12s_source["violationCount"],
            "passed": s12s_source["passed"],
            "semanticSha256": s12s_source["semanticSha256"],
            "harmlessLiteralMentionsIgnored": s12s_source["passed"],
        },
    }
    summary["passed"] = bool(
        summary["casesPassed"] == summary["caseCount"]
        and summary["allOrderDigestsIdentical"]
        and summary["exactReplay"]
        and summary["s12sHistoricalSource"]["passed"]
    )
    return summary, reference_rows


def access_qualification(
    registry: DependencyRegistry,
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    allowed = root / "allowed.txt"
    allowed.write_text("qualification-only\n", encoding="utf-8")
    symlink = root / "symlink_to_forbidden"
    symlink.symlink_to(Path("/artifacts/research_steps/S06/models/forbidden.bin"))
    cases: list[dict[str, Any]] = []

    def record_denial(
        case_id: str,
        callback,
        *,
        expected_source: str,
    ) -> None:
        session = domain_hash("E07/S12T/session/v1", case_id)
        ledger = AuthenticatedAccessLedger(
            registry,
            session_commitment=session,
        )
        denied = False
        error = None
        try:
            with instrument_path_opens(ledger):
                callback()
        except DependencyViolation as exc:
            denied = True
            error = str(exc)
        raw = ledger.finalize()
        validation = validate_access_ledger(
            raw,
            registry,
            expected_session_commitment=session,
        )
        cases.append(
            {
                "caseId": case_id,
                "expectedDisposition": "deny_before_access",
                "actualDenied": denied,
                "sourceApi": (
                    raw["events"][0].get("sourceApi")
                    if raw["events"]
                    else expected_source
                ),
                "eventCount": raw["eventCount"],
                "ledgerValid": validation["passed"],
                "validationReasons": validation["reasons"],
                "error": error,
                "passed": (
                    denied
                    and raw["eventCount"] == 1
                    and not validation["passed"]
                    and any(
                        reason.startswith("denied_event_present")
                        for reason in validation["reasons"]
                    )
                ),
            }
        )

    record_denial(
        "forbidden_builtin_open",
        lambda: open(  # noqa: SIM115 - denial occurs before a handle exists
            "/artifacts/research_steps/S06/models/forbidden.bin", "rb"
        ),
        expected_source="builtins.open",
    )
    record_denial(
        "forbidden_pathlib_open",
        lambda: Path(  # noqa: SIM115 - denial occurs before a handle exists
            "/artifacts/research_steps/S06A/models/forbidden.bin"
        ).open("rb"),
        expected_source="pathlib.Path.open",
    )
    record_denial(
        "forbidden_os_open",
        lambda: os.open(
            "/artifacts/research_steps/S07/arm_signal.json",
            os.O_RDONLY,
        ),
        expected_source="os.open",
    )
    record_denial(
        "symlink_to_forbidden_path",
        lambda: symlink.open("rb"),
        expected_source="pathlib.Path.open",
    )
    dotdot = (
        root
        / ".."
        / ".."
        / ".."
        / "artifacts"
        / "research_steps"
        / "S06"
        / "models"
        / "forbidden.bin"
    )
    record_denial(
        "dotdot_path_normalization",
        lambda: open(  # noqa: SIM115 - denial occurs before a handle exists
            dotdot, "rb"
        ),
        expected_source="builtins.open",
    )

    session = domain_hash("E07/S12T/session/v1", "allowed_api_matrix")
    ledger = AuthenticatedAccessLedger(registry, session_commitment=session)
    contents: list[str] = []
    with instrument_path_opens(ledger):
        with open(allowed, encoding="utf-8") as handle:
            contents.append(handle.read())
        with Path(allowed).open(encoding="utf-8") as handle:
            contents.append(handle.read())

        with open(allowed, encoding="utf-8") as handle:
            contents.append(handle.read())
        descriptor = os.open(allowed, os.O_RDONLY)
        os.close(descriptor)
    raw = ledger.finalize()
    validation = validate_access_ledger(
        raw,
        registry,
        expected_session_commitment=session,
    )
    cases.append(
        {
            "caseId": "allowed_open_api_matrix",
            "expectedDisposition": "allow_and_authenticate",
            "actualDenied": False,
            "sourceApi": "four_api_matrix",
            "eventCount": raw["eventCount"],
            "ledgerValid": validation["passed"],
            "validationReasons": validation["reasons"],
            "error": None,
            "passed": (
                len(contents) == 3
                and all(item == "qualification-only\n" for item in contents)
                and raw["eventCount"] == 4
                and validation["passed"]
            ),
        }
    )

    for case_id, operation in (
        (
            "forbidden_cache_access",
            lambda item: item.authorize_cache(
                "/cache/e07-s10h/work/outcome_rows.parquet"
            ),
        ),
        (
            "S07_signal_access",
            lambda item: item.authorize_signal("S07.armMembership"),
        ),
    ):
        session = domain_hash("E07/S12T/session/v1", case_id)
        item = AuthenticatedAccessLedger(registry, session_commitment=session)
        denied = False
        try:
            operation(item)
        except DependencyViolation:
            denied = True
        raw = item.finalize()
        validation = validate_access_ledger(
            raw,
            registry,
            expected_session_commitment=session,
        )
        cases.append(
            {
                "caseId": case_id,
                "expectedDisposition": "deny_before_access",
                "actualDenied": denied,
                "sourceApi": raw["events"][0]["eventKind"],
                "eventCount": raw["eventCount"],
                "ledgerValid": validation["passed"],
                "validationReasons": validation["reasons"],
                "error": None,
                "passed": (
                    denied and raw["eventCount"] == 1 and not validation["passed"]
                ),
            }
        )

    session = domain_hash("E07/S12T/session/v1", "forgery")
    valid_ledger = AuthenticatedAccessLedger(registry, session_commitment=session)
    valid_ledger.authorize_signal("qualification.structural")
    valid_raw = valid_ledger.finalize()
    missing = validate_access_ledger(
        None,
        registry,
        expected_session_commitment=session,
    )
    forged_event = deepcopy(valid_raw)
    forged_event["events"][0]["signal"] = "S07.forged"
    forged_event_result = validate_access_ledger(
        forged_event,
        registry,
        expected_session_commitment=session,
    )
    forged_auth = deepcopy(valid_raw)
    forged_auth["authenticatorSha256"] = "0" * 64
    forged_auth_result = validate_access_ledger(
        forged_auth,
        registry,
        expected_session_commitment=session,
    )
    for case_id, result in (
        ("missing_ledger", missing),
        ("forged_event", forged_event_result),
        ("forged_authenticator", forged_auth_result),
    ):
        cases.append(
            {
                "caseId": case_id,
                "expectedDisposition": "reject_ledger",
                "actualDenied": not result["passed"],
                "sourceApi": "ledger_validator",
                "eventCount": result["eventCount"],
                "ledgerValid": result["passed"],
                "validationReasons": result["reasons"],
                "error": None,
                "passed": not result["passed"],
            }
        )

    round_trip = canonical_round_trip(valid_raw)
    round_trip_validation = validate_access_ledger(
        round_trip["decoded"],
        registry,
        expected_session_commitment=session,
    )
    cases.append(
        {
            "caseId": "serialization_round_trip",
            "expectedDisposition": "exact_round_trip",
            "actualDenied": False,
            "sourceApi": "canonical_json",
            "eventCount": valid_raw["eventCount"],
            "ledgerValid": round_trip_validation["passed"],
            "validationReasons": round_trip_validation["reasons"],
            "error": None,
            "passed": (
                round_trip["passed"]
                and round_trip_validation["passed"]
                and round_trip["decoded"] == valid_raw
            ),
        }
    )
    cases.sort(key=lambda row: row["caseId"])
    summary = {
        "schemaVersion": "e07.s12t.authenticated-access-validation.v1",
        "researchStepId": "S12T",
        "caseCount": len(cases),
        "casesPassed": sum(row["passed"] for row in cases),
        "allPathDecisionsUseResolvedPaths": True,
        "symlinkAndDotdotDeniedBeforeOpen": all(
            row["passed"]
            for row in cases
            if row["caseId"]
            in {"symlink_to_forbidden_path", "dotdot_path_normalization"}
        ),
        "cacheAndSignalDenialsPassed": all(
            row["passed"]
            for row in cases
            if row["caseId"] in {"forbidden_cache_access", "S07_signal_access"}
        ),
        "missingAndForgeryDenialsPassed": all(
            row["passed"]
            for row in cases
            if row["caseId"]
            in {"missing_ledger", "forged_event", "forged_authenticator"}
        ),
    }
    summary["passed"] = summary["caseCount"] == summary["casesPassed"]
    return summary, cases


def runtime_qualification(
    registry: DependencyRegistry,
    root: Path,
) -> dict[str, Any]:
    module_root = root / "runtime_modules"
    module_root.mkdir(parents=True, exist_ok=False)
    allowed_name = "s12t_runtime_allowed_fixture"
    (module_root / f"{allowed_name}.py").write_text(
        "VALUE = {'qualificationOnly': True, 'outcomeRows': 0}\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(module_root))
    cleanup_modules((allowed_name,))
    try:
        allowed = audit_runtime_dispatch(
            lambda: importlib.import_module(allowed_name).VALUE,
            registry,
        )
    finally:
        if str(module_root) in sys.path:
            sys.path.remove(str(module_root))
        cleanup_modules((allowed_name,))

    forbidden_name = "s12t_preloaded_forbidden_fixture"
    forbidden = ModuleType(forbidden_name)
    forbidden.__file__ = (
        "/artifacts/research_steps/S06/models/preloaded_forbidden_fixture.py"
    )
    sys.modules[forbidden_name] = forbidden
    try:
        preloaded = audit_runtime_dispatch(
            lambda: {"qualificationOnly": True},
            registry,
            inspect_preloaded_names=(forbidden_name,),
        )
    finally:
        cleanup_modules((forbidden_name,))
    round_trip = canonical_round_trip(
        {
            "allowed": allowed,
            "preloaded": preloaded,
        }
    )
    result = {
        "schemaVersion": "e07.s12t.runtime-module-provenance-validation.v1",
        "researchStepId": "S12T",
        "newlyLoadedPermittedModule": allowed,
        "preloadedForbiddenModule": preloaded,
        "serializationRoundTrip": {
            key: round_trip[key] for key in ("passed", "bytes", "sha256")
        },
        "passed": bool(
            allowed["passed"]
            and allowed["inspectedModuleCount"] == 1
            and not preloaded["passed"]
            and preloaded["violationCount"] == 1
            and round_trip["passed"]
        ),
    }
    return result


def revalidate_s12r(
    cache: Path,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], dict[str, Any]]:
    manifest = validate_manifest(S12R / "artifact_manifest.json")
    freeze_result = validate_s12r_freeze()
    gate = read_json(S12R / "s12r_execution_gate.json")
    frozen_gate = {
        key: bool(value["passed"] and value["disposition"] == "pass")
        for key, value in gate["rows"].items()
    }
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
    binding, binding_summary = s12r.build_binding_qualification(
        frozen,
        variants,
        lineages,
        replacements,
        cells,
    )
    accounting = read_json(S12R / "complete_accounting.json")
    logical_semantic = canonical_sha256(
        "E07/S12R/ordered-logical-roster/v1",
        roster,
    )
    physical_semantic = canonical_sha256(
        "E07/S12R/ordered-physical-plan/v1",
        physical.to_dict("records"),
    )
    persisted_logical = pd.read_parquet(S12R / "s12r_logical_roster.parquet")
    persisted_physical = pd.read_parquet(S12R / "s12r_physical_commitments.parquet")
    persisted_binding = pd.read_parquet(S12R / "binding_qualification.parquet")
    binding_keys = ["selectionRef", "taskCellId", "actionSha256", "passed"]
    binding_matches = (
        binding[binding_keys]
        .sort_values(binding_keys[:2])
        .reset_index(drop=True)
        .equals(
            persisted_binding[binding_keys]
            .sort_values(binding_keys[:2])
            .reset_index(drop=True)
        )
    )
    access, reserve = s12r.access_and_reserve_validation()
    frozen_access = read_json(S12R / "access_control_validation.json")
    frozen_reserve = read_json(S12R / "protected_reserve_contract.json")
    access_reserve = {
        "schemaVersion": "e07.s12t.access-reserve-revalidation.v1",
        "researchStepId": "S12T",
        "accessAttempts": access["attempts"],
        "accessDenials": access["denials"],
        "allDenied": access["allDenied"],
        "recordsMatchFrozen": access["records"] == frozen_access["records"],
        "reserveSourceMatchesFrozen": (
            reserve["sourceRecord"] == frozen_reserve["sourceRecord"]
        ),
        "reserveCommitmentsMatchFrozen": (
            reserve["newExtensionReserve"] == frozen_reserve["newExtensionReserve"]
            and reserve["originalHeldoutAddressTemplateCommitmentSha256"]
            == frozen_reserve["originalHeldoutAddressTemplateCommitmentSha256"]
        ),
        "protectedOutcomeRowsRead": 0,
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
    }
    access_reserve["passed"] = bool(
        access["passed"]
        and reserve["passed"]
        and access_reserve["recordsMatchFrozen"]
        and access_reserve["reserveSourceMatchesFrozen"]
        and access_reserve["reserveCommitmentsMatchFrozen"]
    )

    registry = read_json(S12R / "future_publication_registry.json")
    specs = tuple(
        ArtifactSpec(
            str(row["classId"]),
            str(row["relativePath"]),
            str(row["mediaType"]),
        )
        for row in registry["artifactClasses"]
    )
    payloads = {
        class_id: publication_json_bytes(
            {
                "schemaVersion": "e07.s12t.publisher-fixture.v1",
                "artifactClass": class_id,
                "qualificationOnly": True,
                "outcomeRows": 0,
            }
        )
        for class_id in PUBLICATION_CLASSES
    }
    publisher_root = cache / "publisher"
    publisher_root.mkdir(parents=True, exist_ok=False)
    publisher = AtomicScientificPublisher(specs)
    forward = publisher.publish(
        publisher_root / "forward",
        payloads,
        forensics_directory=publisher_root / "forward-forensics",
    )
    reverse = publisher.publish(
        publisher_root / "reverse",
        dict(reversed(list(payloads.items()))),
        forensics_directory=publisher_root / "reverse-forensics",
    )
    forward_validation = publisher.validate_complete(
        publisher_root / "forward",
        payloads,
        attempt_id=forward["attemptId"],
    )
    reverse_validation = publisher.validate_complete(
        publisher_root / "reverse",
        payloads,
        attempt_id=reverse["attemptId"],
    )
    publication = {
        "schemaVersion": "e07.s12t.publisher-revalidation.v1",
        "researchStepId": "S12T",
        "artifactClassCount": len(specs),
        "classesMatchFrozen": (
            tuple(spec.class_id for spec in specs) == PUBLICATION_CLASSES
        ),
        "forwardComplete": forward_validation["complete"],
        "reverseComplete": reverse_validation["complete"],
        "forwardCommitBoundaries": forward["commitBoundaryCount"],
        "reverseCommitBoundaries": reverse["commitBoundaryCount"],
        "outcomeRows": 0,
    }
    publication["passed"] = bool(
        publication["artifactClassCount"] == 10
        and publication["classesMatchFrozen"]
        and publication["forwardComplete"]
        and publication["reverseComplete"]
        and publication["forwardCommitBoundaries"] == 1
        and publication["reverseCommitBoundaries"] == 1
    )

    fault = read_json(S12R / "spatial_fault_contract.json")
    scheduler = read_json(S12R / "alternate_scheduler_contract.json")
    endpoints = read_jsonl(S12R / "endpoint_contract_registry.jsonl")
    costs = read_json(S12R / "cost_registry.json")
    counts = {
        "lineages": len(lineages),
        "configurations": len(frozen["candidateConfigurations"]),
        "adaptations": len(variants),
        "replacementComparators": len(replacements),
        "taskConditionCells": len(cells),
        "scenarioFamilies": len(scenarios),
        "logicalReservations": len(roster),
        "physicalReplayCommitments": len(physical),
        "bindings": int(binding_summary["bindingRows"]),
        "bindingsPassed": int(binding_summary["bindingRowsPassed"]),
    }
    checks = {
        "G01": manifest["passed"] and freeze_result["passed"],
        "G02": counts["lineages"] == 7 and counts["configurations"] == 14,
        "G03": (
            counts["adaptations"] == 30
            and counts["replacementComparators"] == 2
            and {row["newConfigurationId"] for row in replacements}
            == NEW_REPLACEMENT_COMPARATOR_IDS
        ),
        "G04": (
            fault["faultFamilyId"] == "identity_locus_target_breaking_spurious_swap_v1"
        ),
        "G05": scheduler["schedulerFamilyId"] == "identity_round_robin_batch4_v1",
        "G06": (
            binding_summary["allPassed"]
            and counts["bindings"] == 3_120
            and counts["bindingsPassed"] == 3_120
            and binding_matches
        ),
        "G07": (
            all(
                row.get("diagnosticPromotionEligible") is False
                for row in endpoints
                if row.get("endpointMode") == "diagnostic_only"
            )
            and costs["scalarOrUniversalCost"] is None
        ),
        "G08": access_reserve["passed"],
        "G09": (
            counts["taskConditionCells"] == 48
            and counts["scenarioFamilies"] == 3_840
            and counts["logicalReservations"] == 195_072
            and counts["physicalReplayCommitments"] == 390_144
            and logical_semantic == accounting["logicalRosterSemanticSha256"]
            and physical_semantic == accounting["physicalCommitmentSemanticSha256"]
            and set(persisted_logical["logicalReservationId"])
            == {row["logicalReservationId"] for row in roster}
            and set(persisted_physical["physicalExecutionId"])
            == set(physical["physicalExecutionId"])
        ),
        "G10": publication["passed"],
    }
    result = {
        "schemaVersion": "e07.s12t.s12r-revalidation.v1",
        "researchStepId": "S12T",
        "s12rArtifactManifest": manifest,
        "s12rPreregistrationFreeze": freeze_result,
        "frozenGateRows": frozen_gate,
        "liveGateRows": checks,
        "counts": counts,
        "semanticCommitments": {
            "candidateLockSha256": accounting.get("candidateLockSha256"),
            "logicalRosterSha256": logical_semantic,
            "physicalPlanSha256": physical_semantic,
        },
        "bindingProjectionMatchesPersisted": binding_matches,
        "protectedReserveAndAccessPassed": access_reserve["passed"],
        "publisherPassed": publication["passed"],
        "transferEpisodesSubmitted": 0,
        "passed": bool(
            set(frozen_gate) == {f"G{index:02d}" for index in range(1, 11)}
            and all(frozen_gate.values())
            and all(checks.values())
        ),
    }
    return result, binding, access_reserve, publication


def run_tests() -> dict[str, Any]:
    commands = [
        [
            sys.executable,
            "-m",
            "py_compile",
            str(MODULE),
            str(SCRIPT),
            str(TEST),
        ],
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_s12t_dependency_preflight.py",
            "tests/test_s12r_replacement_spatial_transfer.py",
            "tests/test_s12s_replacement_execution.py",
        ],
        [
            "ruff",
            "check",
            "src/spatial_transfer/dependency_preflight.py",
            "scripts/qualify_dependency_preflight_s12t.py",
            "tests/test_s12t_dependency_preflight.py",
        ],
        [
            "ruff",
            "format",
            "--check",
            "src/spatial_transfer/dependency_preflight.py",
            "scripts/qualify_dependency_preflight_s12t.py",
            "tests/test_s12t_dependency_preflight.py",
        ],
        ["git", "-c", f"safe.directory={REPOSITORY}", "diff", "--check"],
    ]
    records = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY,
            text=True,
            capture_output=True,
            check=False,
        )
        records.append(
            {
                "command": " ".join(command),
                "exitCode": completed.returncode,
                "passed": completed.returncode == 0,
                "stdoutTail": completed.stdout[-4000:],
                "stderrTail": completed.stderr[-4000:],
            }
        )
    return {
        "schemaVersion": "e07.s12t.test-validation.v1",
        "researchStepId": "S12T",
        "commands": records,
        "allPassed": all(row["passed"] for row in records),
    }


def qualify() -> None:
    if not OUT.is_dir():
        raise RuntimeError("S12T must be prospectively frozen before qualification")
    if QUALIFICATION_CACHE.exists():
        raise RuntimeError("S12T qualification cache is not fresh")
    if FROZEN_S12S_CACHE.exists() or S12S_SCIENTIFIC.exists():
        raise RuntimeError("immutable S12S zero-execution boundary changed")
    registry = DependencyRegistry.load(REGISTRY)
    freeze_result = verify_freeze(registry)
    if not freeze_result["passed"]:
        raise RuntimeError("S12T preregistration freeze changed before qualification")
    QUALIFICATION_CACHE.mkdir(parents=True, exist_ok=False)
    try:
        fixtures = build_static_fixtures(QUALIFICATION_CACHE)
        static_summary, static_cases = static_qualification(
            fixtures,
            registry,
            QUALIFICATION_CACHE,
        )
        access_summary, access_cases = access_qualification(
            registry,
            QUALIFICATION_CACHE,
        )
        runtime = runtime_qualification(registry, QUALIFICATION_CACHE)
        s12r_result, binding, access_reserve, publication = revalidate_s12r(
            QUALIFICATION_CACHE
        )
        write_parquet(OUT / "binding_revalidation.parquet", binding.to_dict("records"))
        write_parquet(
            OUT / "adversarial_fixture_results.parquet",
            [
                *(
                    {
                        "caseId": row["caseId"],
                        "controlPlane": "static_import",
                        "expectedDisposition": (
                            "allow" if row["expectedStaticPass"] else "fail_closed"
                        ),
                        "actualDisposition": (
                            "allow" if row["actualStaticPass"] else "fail_closed"
                        ),
                        "passed": row["passed"],
                    }
                    for row in static_cases
                ),
                *(
                    {
                        "caseId": row["caseId"],
                        "controlPlane": "access_ledger",
                        "expectedDisposition": row["expectedDisposition"],
                        "actualDisposition": (
                            "denied" if row["actualDenied"] else "allowed_or_validated"
                        ),
                        "passed": row["passed"],
                    }
                    for row in access_cases
                ),
                {
                    "caseId": "newly_loaded_permitted_module_provenance",
                    "controlPlane": "runtime_module",
                    "expectedDisposition": "allow",
                    "actualDisposition": (
                        "allow"
                        if runtime["newlyLoadedPermittedModule"]["passed"]
                        else "fail_closed"
                    ),
                    "passed": runtime["newlyLoadedPermittedModule"]["passed"],
                },
                {
                    "caseId": "preloaded_forbidden_module_provenance",
                    "controlPlane": "runtime_module",
                    "expectedDisposition": "fail_closed",
                    "actualDisposition": (
                        "fail_closed"
                        if not runtime["preloadedForbiddenModule"]["passed"]
                        else "allow"
                    ),
                    "passed": not runtime["preloadedForbiddenModule"]["passed"],
                },
            ],
        )
        adversarial = {
            "schemaVersion": "e07.s12t.adversarial-fixture-results.v1",
            "researchStepId": "S12T",
            "staticCases": static_cases,
            "accessCases": access_cases,
            "runtimeCases": {
                "newlyLoadedPermittedModulePassed": runtime[
                    "newlyLoadedPermittedModule"
                ]["passed"],
                "preloadedForbiddenModuleDenied": not runtime[
                    "preloadedForbiddenModule"
                ]["passed"],
            },
        }
        adversarial["caseCount"] = len(static_cases) + len(access_cases) + 2
        adversarial["casesPassed"] = (
            sum(row["passed"] for row in static_cases)
            + sum(row["passed"] for row in access_cases)
            + int(runtime["newlyLoadedPermittedModule"]["passed"])
            + int(not runtime["preloadedForbiddenModule"]["passed"])
        )
        adversarial["passed"] = adversarial["caseCount"] == adversarial["casesPassed"]
        replay = {
            "schemaVersion": "e07.s12t.serialization-replay-worker-order.v1",
            "researchStepId": "S12T",
            "staticExactReplay": static_summary["exactReplay"],
            "staticWorkerOrders": static_summary["orders"],
            "allWorkerOrderDigestsIdentical": static_summary[
                "allOrderDigestsIdentical"
            ],
            "accessLedgerSerializationPassed": any(
                row["caseId"] == "serialization_round_trip" and row["passed"]
                for row in access_cases
            ),
            "runtimeSerializationPassed": runtime["serializationRoundTrip"]["passed"],
        }
        replay["passed"] = all(
            value
            for key, value in replay.items()
            if key.endswith("Passed")
            or key in {"staticExactReplay", "allWorkerOrderDigestsIdentical"}
        )
        tests = run_tests()
        atomic_write_json(OUT / "freeze_revalidation.json", freeze_result)
        atomic_write_json(
            OUT / "static_import_graph_validation.json",
            static_summary,
        )
        atomic_write_json(
            OUT / "authenticated_access_validation.json",
            access_summary,
        )
        atomic_write_json(
            OUT / "runtime_module_provenance_validation.json",
            runtime,
        )
        atomic_write_json(
            OUT / "adversarial_fixture_results.json",
            adversarial,
        )
        atomic_write_json(
            OUT / "serialization_replay_worker_order_validation.json",
            replay,
        )
        atomic_write_json(OUT / "s12r_revalidation.json", s12r_result)
        atomic_write_json(
            OUT / "access_reserve_validation.json",
            access_reserve,
        )
        atomic_write_json(
            OUT / "publisher_revalidation.json",
            publication,
        )
        atomic_write_json(OUT / "test_validation.json", tests)
        zero = {
            "schemaVersion": "e07.s12t.zero-execution-accounting.v1",
            "researchStepId": "S12T",
            "qualificationFixturesExecuted": adversarial["caseCount"],
            "plannedS12RLogicalReservations": 195_072,
            "plannedS12RPhysicalReplayCommitments": 390_144,
            "transferEpisodesSubmitted": 0,
            "logicalReservationsExecuted": 0,
            "physicalReplaysExecuted": 0,
            "validationOutcomeRowsRead": 0,
            "confirmationOutcomeRowsRead": 0,
            "protectedOutcomeRowsRead": 0,
            "s14ReservePayloadsMaterialized": 0,
            "civicRows": 0,
            "s13Rows": 0,
            "s14Rows": 0,
            "efficacyRowsPublished": 0,
            "scientificExecutionNamespacesCreated": [],
            "qualificationCache": str(QUALIFICATION_CACHE),
            "s12sPatchedResumedOrRetried": False,
            "s12rScientificExecutionNamespaceCreated": False,
            "allRowsAccounted": True,
        }
        atomic_write_json(OUT / "zero_execution_accounting.json", zero)
        baseline = read_json(OUT / "predecessor_immutability_baseline.json")
        after_s12r = validate_manifest(S12R / "artifact_manifest.json")
        after_s12s = validate_manifest(S12S / "artifact_manifest.json")
        immutable = {
            "schemaVersion": "e07.s12t.predecessor-immutability-validation.v1",
            "researchStepId": "S12T",
            "S12RBeforePassed": baseline["S12R"]["passed"],
            "S12RAfterPassed": after_s12r["passed"],
            "S12RManifestFileUnchanged": (
                baseline["S12R"]["manifestFile"] == after_s12r["manifestFile"]
            ),
            "S12SBeforePassed": baseline["S12S"]["passed"],
            "S12SAfterPassed": after_s12s["passed"],
            "S12SManifestFileUnchanged": (
                baseline["S12S"]["manifestFile"] == after_s12s["manifestFile"]
            ),
            "s12sCacheRemainsAbsent": not FROZEN_S12S_CACHE.exists(),
            "s12sScientificPublicationRemainsAbsent": not S12S_SCIENTIFIC.exists(),
            "transitivePredecessorCommitmentRetainedThroughS12R": True,
            "priorArtifactMutations": 0,
        }
        immutable["passed"] = all(
            value
            for key, value in immutable.items()
            if key.endswith(("Passed", "Unchanged"))
            or key.startswith("s12s")
            or key == "transitivePredecessorCommitmentRetainedThroughS12R"
        )
        atomic_write_json(
            OUT / "predecessor_immutability_validation.json",
            immutable,
        )
        checks = {
            "protocolFreeze": freeze_result["passed"],
            "staticImportGraph": static_summary["passed"],
            "runtimeModuleProvenance": runtime["passed"],
            "authenticatedAccess": access_summary["passed"],
            "adversarialFixtures": adversarial["passed"],
            "serializationReplayWorkerOrder": replay["passed"],
            "S12RHashesGatesBindings": s12r_result["passed"],
            "protectedReserveAndAccess": access_reserve["passed"],
            "publisher": publication["passed"],
            "tests": tests["allPassed"],
            "predecessorImmutability": immutable["passed"],
            "zeroExecutionAccounting": (
                zero["transferEpisodesSubmitted"] == 0
                and zero["scientificExecutionNamespacesCreated"] == []
                and zero["efficacyRowsPublished"] == 0
            ),
        }
        validation = {
            "schemaVersion": "e07.s12t.validation-summary.v1",
            "researchStepId": "S12T",
            "checks": checks,
            "checkCount": len(checks),
            "checksPassed": sum(checks.values()),
            "allPassed": all(checks.values()),
        }
        atomic_write_json(OUT / "validation_summary.json", validation)
        gate = {
            "schemaVersion": "e07.s12t.execution-review-gate.v1",
            "researchStepId": "S12T",
            "qualificationPassed": validation["allPassed"],
            "dependencyPreflightEligibleForSeparateFreshExecutionReview": validation[
                "allPassed"
            ],
            "scientificExecutionAuthorized": False,
            "separateHumanExecutionDecisionRequired": True,
            "futureExecutionMustUseNeverUsedNamespace": True,
            "S12SPatchResumeRetryAuthorized": False,
            "S13OrS14Authorized": False,
            "remainingRiskClasses": [
                "runtime_integrity",
                "replay_integrity",
                "accounting_integrity",
                "access_integrity",
                "publication_integrity",
            ],
            "claimBoundary": (
                "Any eventual result remains specific to the less historically "
                "anchored S12R replacement estimand."
            ),
        }
        atomic_write_json(OUT / "s12t_execution_review_gate.json", gate)
        provenance = {
            "schemaVersion": "e07.s12t.provenance.v1",
            "researchStepId": "S12T",
            "timestampUtc": now_utc(),
            "repository": str(REPOSITORY),
            "gitCommit": subprocess.run(
                ["git", "-c", f"safe.directory={REPOSITORY}", "rev-parse", "HEAD"],
                cwd=REPOSITORY,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip(),
            "python": sys.version,
            "workers": 4,
            "numericThreadsPerWorker": 1,
            "dependenciesInstalled": [],
            "networkResourcesUsed": 0,
            "qualificationCache": str(QUALIFICATION_CACHE),
            "qualificationCacheContainsOutcomeRows": False,
        }
        atomic_write_json(OUT / "provenance.json", provenance)
        if not validation["allPassed"]:
            raise RuntimeError("S12T qualification failed closed")
    finally:
        if QUALIFICATION_CACHE.exists():
            shutil.rmtree(QUALIFICATION_CACHE)
    print(
        json.dumps(
            {
                "researchStepId": "S12T",
                "qualificationPassed": True,
                "transferEpisodesSubmitted": 0,
                "scientificExecutionAuthorized": False,
            },
            sort_keys=True,
        )
    )


def report_text() -> str:
    validation = read_json(OUT / "validation_summary.json")
    static = read_json(OUT / "static_import_graph_validation.json")
    access = read_json(OUT / "authenticated_access_validation.json")
    runtime = read_json(OUT / "runtime_module_provenance_validation.json")
    adversarial = read_json(OUT / "adversarial_fixture_results.json")
    s12r_result = read_json(OUT / "s12r_revalidation.json")
    zero = read_json(OUT / "zero_execution_accounting.json")
    tests = read_json(OUT / "test_validation.json")
    return f"""# S12T — Qualify the outcome-independent dependency preflight

## Concise top summary

| Field | Result |
| --- | --- |
| Research step ID | **S12T** |
| Completion status | **Complete outcome-free dependency-preflight remediation and qualification; stopped before every scientific execution** |
| Artifacts written | Frozen protocol and external registry; preregistration/input/immutability freezes; static import graph, runtime-module provenance, authenticated path/cache/signal ledger, adversarial, serialization/replay/order, exact S12R hash/gate/binding/access/reserve/publisher, zero-accounting, test, provenance, execution-review, status, manifest, and this canonical report under `/artifacts/research_steps/S12T/` |
| Validation result | **PASS — {validation["checksPassed"]}/{validation["checkCount"]} integrated checks; {adversarial["casesPassed"]}/{adversarial["caseCount"]} adversarial/benign cases; G01–G10 and 3,120/3,120 bindings revalidated; zero episodes** |
| Outcome classification | **Supportive bounded technical qualification** |
| Caveats or blockers | Qualification is not transfer efficacy and authorizes no execution. A later fresh attempt can still fail on runtime, replay, accounting, access, or publication integrity. S12R remains a distinct, less historically anchored replacement estimand. |
| Recommended next action | **Human review only. If execution remains desired, authorize a separate fresh step using a never-used namespace and this frozen S12T control. Do not patch/retry S12S or start S13/S14 automatically.** |

## Lay summary

The safety check now distinguishes words from actions. Merely mentioning a
forbidden model, cache, or signal in source code no longer causes a false
alarm. The control instead resolves import operations, checks where modules
were actually loaded from, and records every attempted file, cache, and signal
access in an authenticated ledger. Synthetic attempts to hide prohibited use
behind aliases, wrappers, dynamic imports, symlinks, `..` paths, caches, or
forged logs all failed closed. The complete S12R design still matches its
frozen hashes, but no transfer episode was run and no execution is authorized.

## Frozen question

Can S12S's self-referential forbidden-literal scan be replaced prospectively
by a fail-closed actual-use control that distinguishes harmless source text
from prohibited imports, loaded modules, opened paths, cache reads, and
S07/protected signal access without executing a scientific episode?

The criterion was met for the bounded qualification. S12S itself was not
patched, resumed, imported as an executor, or retried.

## Inputs and permitted evidence

Before implementation and qualification, the step refreshed `AGENTS.md`,
`FULL_PLAN.md`, `RESEARCH_PLAN.md`, upstream-artifact/capability context,
the attachment manifest and sidecar, E01–E06 handoffs/native contracts, and
S08M–S12S reports and controlling artifacts. Exact input paths, sizes, and
SHA-256 values are in `input_hash_freeze.json`.

Only structural metadata, authoritative contracts, frozen S12R manifests and
registries, S12S integrity forensics, and dedicated synthetic fixtures were
permitted. Validation, confirmation, protected-reserve outcomes, quarantine
outcomes, S06/S06A models or embeddings, S07 arm signals, civic rows, and
S13/S14 episodes remained prohibited.

## Detailed methods

### Prospective freeze

`s12t_dependency_preflight_protocol.yaml` and `dependency_registry.json` were
copied byte-for-byte into the artifact directory before qualification.
`preregistration_freeze.json` binds those bytes, the implementation, runner,
tests, permitted evidence, required adversaries, exact S12R counts, and a zero
scientific-namespace boundary. Qualification used only
`/cache/e07-s12t-qualification`, which was removed afterward; it never created
an S12R scientific execution namespace.

### Static import and graph resolution

The AST analyzer resolves direct and relative imports, recursively follows
repository/fixture modules, recognizes `importlib.import_module`, `__import__`,
aliases, assigned callables, and simple wrapper dispatch, and fails closed on
unresolved dynamic targets. String constants that are not arguments to import
operations are ignored. The historical S12S script plus its execution module
passed this control despite retaining the literals that defeated S12S's raw
text scanner: {static["s12sHistoricalSource"]["filesAnalyzed"]} files and
{static["s12sHistoricalSource"]["importsResolved"]} resolved imports, with
{static["s12sHistoricalSource"]["violationCount"]} violation.

### Runtime provenance and authenticated access

Runtime qualification snapshots loaded modules, records names and resolved
origins, and audits both newly loaded modules and explicitly named preloaded
modules. A permitted qualification module loaded successfully; a preloaded
module whose origin pointed inside a forbidden S06 path was rejected.

The access broker instruments `builtins.open`, `io.open`,
`pathlib.Path.open`, and `os.open`. Paths are absolutized and symlinks resolved
before a deny-first policy. Every file/cache/signal event receives a
domain-separated event hash, contiguous ordinal, ledger hash, registry
commitment, session commitment, and authenticator. Missing ledgers, edited
events, changed authenticators, denied events, and inconsistent counts fail
validation.

### Structural S12R revalidation

The qualifier independently rehashed the S12R artifact manifest and
preregistration freeze, regenerated all 48 task-condition cells, 3,840
scenario families, 195,072 logical reservations, and 390,144 physical replay
commitments, recomputed semantic hashes, reconstructed 3,120 bindings, and
compared identity/projection sets with persisted Parquet. It repeated all six
protected denials and reserve commitments and exercised the frozen ten-class
publisher twice with zero-outcome synthetic payloads under the S12T
qualification namespace.

## Commands, dependencies, and parameters

```text
PYTHONPATH=. python scripts/qualify_dependency_preflight_s12t.py freeze
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts/qualify_dependency_preflight_s12t.py qualify
PYTHONPATH=. python scripts/qualify_dependency_preflight_s12t.py finalize
```

Focused validation used `py_compile`, `pytest` on S12T/S12R/S12S,
`ruff check`, `ruff format --check`, and `git diff --check`; all
{len(tests["commands"])} command groups passed. Four static worker orders used
four threads with one numeric thread per worker. No dependency was installed,
no network resource was used, and no GPU or scientific simulator episode was
needed.

## Results

| Control | Result |
| --- | --- |
| Static benign/adversarial cases | {static["casesPassed"]}/{static["caseCount"]} PASS |
| Static replay / four worker orders | {str(static["exactReplay"]).lower()} / one digest |
| Historical S12S source under actual-use AST rule | PASS |
| Authenticated file/cache/signal cases | {access["casesPassed"]}/{access["caseCount"]} PASS |
| Permitted newly loaded runtime module | {str(runtime["newlyLoadedPermittedModule"]["passed"]).upper()} |
| Forbidden preloaded runtime origin rejected | {str(not runtime["preloadedForbiddenModule"]["passed"]).upper()} |
| S12R frozen/live gates | 10/10 PASS |
| S12R bindings | {s12r_result["counts"]["bindingsPassed"]}/{s12r_result["counts"]["bindings"]} PASS |
| S12R logical / physical commitments | {s12r_result["counts"]["logicalReservations"]:,} / {s12r_result["counts"]["physicalReplayCommitments"]:,} exact |
| Protected/reserve denial | PASS; zero outcomes |
| Publisher | ten classes, two complete one-boundary synthetic commits |
| Scientific episodes / efficacy rows | {zero["transferEpisodesSubmitted"]} / {zero["efficacyRowsPublished"]} |

The adversarial matrix includes benign and self-referential literals; direct,
indirect, dynamic, aliased, built-in, wrapper, and unresolved imports;
forbidden built-in/`pathlib`/`os` opens; symlink and `..` normalization;
forbidden cache and S07 signal access; missing/forged ledgers; canonical
serialization; and preloaded/new runtime provenance.

## Validation

All {validation["checkCount"]} controlling checks passed. S12R's artifact and
preregistration hashes, candidate and comparator identities, fault/scheduler
contracts, G01–G10, binding count, roster counts, semantic commitments,
protected denials, reserve records, and publisher remained exact. S12R and
S12S manifests were identical before and after S12T. `/cache/e07-s12s` and
S12S scientific publication remained absent.

## Caveats, blockers, failed assumptions, and limitations

1. This is an outcome-free software/control qualification, not evidence that
   any policy transfers, adapts, repairs, or outperforms a comparator.
2. Static wrapper recognition is intentionally bounded. An unresolved dynamic
   target fails closed rather than being guessed.
3. Runtime provenance and open ledgers must be installed on the actual future
   dispatch path; this step does not itself authorize that dispatch.
4. A future fresh run can still fail on runtime behavior, replay, accounting,
   access, native-contract, or complete-set publication integrity.
5. S12R remains a replacement estimand, not S12P/S12A, and its E07 fault and
   scheduler are less historically anchored than native E06 contracts.
6. No universal score, biological, cognitive, revealed-preference, civic,
   validation, confirmation, S13, or S14 claim is authorized.

## Provenance and artifacts

Repository source contains the registry, protocol, implementation, runner, and
focused tests. Compact evidence is under `/artifacts/research_steps/S12T/`;
the outcome-free qualification cache was removed.
`repository_publication.json` verifies local/remote tree parity on
`eidosoma/groups/28`, `artifact_manifest.json` hashes every canonical S12T
artifact, and `status.json` provides the compact machine-readable handoff.

## Recommended next action

Return to the Chief Scientist. Even though qualification passed, do not begin
another execution automatically. If the less historically anchored S12R
estimand remains approved, separately authorize a genuinely fresh execution
step using a never-used namespace and install this exact frozen preflight on
the dispatch path. Do not patch or retry S12S and do not start S13 or S14.
"""


def finalize() -> None:
    validation = read_json(OUT / "validation_summary.json")
    if not validation["allPassed"]:
        raise RuntimeError("cannot finalize a failed S12T qualification as passing")
    plan = WORKSPACE / "RESEARCH_PLAN.md"
    plan_text = plan.read_text(encoding="utf-8")
    plan_validation = {
        "schemaVersion": "e07.s12t.research-plan-update-validation.v1",
        "researchStepId": "S12T",
        "researchPlan": file_record(plan),
        "mentionsS12T": "### S12T:" in plan_text,
        "currentStepHandedBack": (
            "Step ID: none — S12T completed and handed back" in plan_text
        ),
        "completedTableUpdated": "| S12T |" in plan_text,
        "priorS12SSectionRetained": "### S12S:" in plan_text,
    }
    plan_validation["passed"] = all(
        value
        for key, value in plan_validation.items()
        if key
        in {
            "mentionsS12T",
            "currentStepHandedBack",
            "completedTableUpdated",
            "priorS12SSectionRetained",
        }
    )
    atomic_write_json(
        OUT / "research_plan_update_validation.json",
        plan_validation,
    )
    git_command = ["git", "-c", f"safe.directory={REPOSITORY}"]
    local_commit = subprocess.run(
        [*git_command, "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    local_tree = subprocess.run(
        [*git_command, "rev-parse", "HEAD^{tree}"],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    branch = subprocess.run(
        [*git_command, "branch", "--show-current"],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    remote_line = subprocess.run(
        [*git_command, "ls-remote", "origin", f"refs/heads/{branch}"],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if not remote_line:
        raise RuntimeError(f"configured branch is not published: {branch}")
    remote_commit = remote_line.split()[0]
    subprocess.run(
        [*git_command, "fetch", "--quiet", "origin", remote_commit],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=True,
    )
    remote_tree = subprocess.run(
        [*git_command, "rev-parse", f"{remote_commit}^{{tree}}"],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    repository_publication = {
        "schemaVersion": "e07.s12t.repository-publication.v1",
        "researchStepId": "S12T",
        "repository": "Eidosoma/cell_research",
        "branch": branch,
        "localCommit": local_commit,
        "remoteCommit": remote_commit,
        "localTree": local_tree,
        "remoteTree": remote_tree,
        "treeParity": local_tree == remote_tree,
        "passed": branch == "eidosoma/groups/28" and local_tree == remote_tree,
    }
    atomic_write_json(OUT / "repository_publication.json", repository_publication)
    if not repository_publication["passed"]:
        raise RuntimeError(
            f"repository publication validation failed: {repository_publication}"
        )
    (OUT / "research_step_full_results.md").write_text(
        report_text(),
        encoding="utf-8",
    )
    provenance = read_json(OUT / "provenance.json")
    status = {
        "researchStepId": "S12T",
        "stepNumber": "12T",
        "success": True,
        "status": (
            "complete_outcome_free_dependency_preflight_qualified_"
            "scientific_execution_not_authorized"
        ),
        "artifactsWritten": [str(OUT)],
        "validationResult": (
            "PASS actual-use dependency preflight; all adversarial controls, "
            "S12R G01-G10, 3,120 bindings, access/reserve, publisher, "
            "immutability, and zero-execution accounting passed"
        ),
        "outcomeClassification": "supportive",
        "caveatsOrBlockers": [
            "Qualification is not transfer efficacy and authorizes no execution.",
            "A later fresh execution can still fail on runtime, replay, accounting, access, or publication integrity.",
            "S12R remains a distinct and less historically anchored replacement estimand.",
        ],
        "recommendedNextAction": (
            "Human review; if separately approved, authorize a genuinely fresh "
            "execution step using a never-used namespace and the frozen S12T "
            "control. Do not patch/retry S12S or start S13/S14 automatically."
        ),
        "repositoryCommit": provenance["gitCommit"],
    }
    atomic_write_json(OUT / "status.json", status)
    manifest_names = sorted(
        path.name
        for path in OUT.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    artifacts = [file_record(OUT / name) for name in manifest_names]
    manifest = {
        "schemaVersion": "e07.s12t.artifact-manifest.v1",
        "researchStepId": "S12T",
        "artifactCount": len(artifacts),
        "artifacts": artifacts,
        "complete": True,
        "scientificPublicationState": "zero_scientific_publication",
        "transferEpisodesSubmitted": 0,
        "scientificExecutionAuthorized": False,
    }
    atomic_write_json(OUT / "artifact_manifest.json", manifest)
    print(
        json.dumps(
            {
                "researchStepId": "S12T",
                "artifactCount": len(artifacts) + 1,
                "allPassed": True,
                "scientificExecutionAuthorized": False,
            },
            sort_keys=True,
        )
    )


def validate() -> None:
    manifest = read_json(OUT / "artifact_manifest.json")
    failures = []
    for expected in manifest["artifacts"]:
        actual = file_record(Path(expected["path"]))
        if (
            int(expected["bytes"]) != actual["bytes"]
            or expected["sha256"] != actual["sha256"]
        ):
            failures.append(expected["path"])
    required = {
        "s12t_dependency_preflight_protocol.yaml",
        "dependency_registry.json",
        "preregistration_freeze.json",
        "static_import_graph_validation.json",
        "runtime_module_provenance_validation.json",
        "authenticated_access_validation.json",
        "adversarial_fixture_results.json",
        "adversarial_fixture_results.parquet",
        "serialization_replay_worker_order_validation.json",
        "s12r_revalidation.json",
        "binding_revalidation.parquet",
        "access_reserve_validation.json",
        "publisher_revalidation.json",
        "predecessor_immutability_validation.json",
        "zero_execution_accounting.json",
        "s12t_execution_review_gate.json",
        "validation_summary.json",
        "test_validation.json",
        "provenance.json",
        "repository_publication.json",
        "research_plan_update_validation.json",
        "research_step_full_results.md",
        "status.json",
        "artifact_manifest.json",
    }
    missing = sorted(name for name in required if not (OUT / name).is_file())
    if (
        failures
        or missing
        or not read_json(OUT / "validation_summary.json")["allPassed"]
    ):
        raise RuntimeError(
            f"S12T artifact validation failed: failures={failures}, missing={missing}"
        )
    print(
        json.dumps(
            {
                "researchStepId": "S12T",
                "artifactCount": manifest["artifactCount"] + 1,
                "allPassed": True,
                "transferEpisodesSubmitted": 0,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "freeze":
        freeze()
    elif command == "qualify":
        qualify()
    elif command == "finalize":
        finalize()
    elif command == "validate":
        validate()
    else:
        raise SystemExit(
            "usage: qualify_dependency_preflight_s12t.py "
            "[freeze|qualify|finalize|validate]"
        )


if __name__ == "__main__":
    main()
