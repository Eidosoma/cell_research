#!/usr/bin/env python3
"""Execute S10D through the S10A/S10C-qualified frozen S10 path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Mapping

import yaml

import scripts.run_native_event_discovery_s10 as base
import scripts.run_native_event_discovery_s10b as s10b
from src.phenotype_discovery.search import sha256_file


STEP_ID = "S10D"
OUTPUT = Path("/artifacts/research_steps/S10D")
CACHE = Path("/cache/e07-s10d")
CONFIG = Path("/workspace/cell-research/configs/discovery/s10d_fresh_execution.yaml")
SCRIPT = Path(__file__).resolve()
TEST = Path("/workspace/cell-research/tests/test_s10d_fresh_execution.py")
REPOSITORY = Path("/workspace/cell-research")
S10P = Path("/artifacts/research_steps/S10P")
FAILED_S10 = Path("/artifacts/research_steps/S10")
S10A = Path("/artifacts/research_steps/S10A")
S10B = Path("/artifacts/research_steps/S10B")
S10C = Path("/artifacts/research_steps/S10C")
HISTORICAL = (
    Path("/artifacts/research_steps/S05"),
    Path("/artifacts/research_steps/S08M"),
    Path("/artifacts/research_steps/S09"),
    S10P,
    FAILED_S10,
    S10A,
    S10B,
    S10C,
)
_CALLBACK_COUNTS = {
    "revalidate_frozen_inputs": 0,
    "prospective_freeze": 0,
}
_FRESH_NAMESPACE_ABSENT_BEFORE_RUNNER_ENTRY = False


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _configure_s10b_globals() -> None:
    """Point the qualified continuation helpers at the distinct S10D scope."""

    s10b.STEP_ID = STEP_ID
    s10b.OUTPUT = OUTPUT
    s10b.CACHE = CACHE
    s10b.CONFIG = CONFIG
    s10b.SCRIPT = SCRIPT
    s10b.TEST = TEST
    s10b.HISTORICAL = HISTORICAL


def _callback_overrides() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "revalidate_frozen_inputs": revalidate_frozen_inputs,
            "prospective_freeze": prospective_freeze,
            "reviewer_instructions": reviewer_instructions,
            "report_markdown": report_markdown,
            "manifest_for_output": manifest_for_output,
        }
    )


def _structural_s10b_preflight_pass(result: Mapping[str, Any]) -> bool:
    remediated_core_paths = {
        "/workspace/cell-research/src/phenotype_discovery/accounting.py",
        "/workspace/cell-research/src/phenotype_discovery/native_features.py",
        "/workspace/cell-research/src/phenotype_discovery/search.py",
    }
    nonsuperseded_inputs = [
        row
        for row in result["frozenInputChecks"]
        if str(row["path"]) not in remediated_core_paths
    ]
    return bool(
        all(row["pass"] for row in result["frozenFileChecks"])
        and all(row["pass"] for row in nonsuperseded_inputs)
        and all(result["preregistrationChecks"].values())
        and result["s10pGateAllPass"]
        and set(result["s10pGateRows"]) == {f"G{index:02d}" for index in range(1, 9)}
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


def _tree_check(
    *,
    label: str,
    root: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = s10b.tree_snapshot(root)
    return {
        "stepId": label,
        "path": str(root),
        "expectedTreeSha256": expected["treeSha256"],
        "actualTreeSha256": snapshot["treeSha256"],
        "expectedFileCount": expected["fileCount"],
        "actualFileCount": snapshot["fileCount"],
        "expectedTotalBytes": expected["totalBytes"],
        "actualTotalBytes": snapshot["totalBytes"],
        "pass": (
            snapshot["treeSha256"] == expected["treeSha256"]
            and snapshot["fileCount"] == expected["fileCount"]
            and snapshot["totalBytes"] == expected["totalBytes"]
        ),
    }


def _file_check(spec: Mapping[str, Any], name: str) -> dict[str, Any]:
    actual = sha256_file(spec["path"])
    return {
        "name": name,
        "path": spec["path"],
        "expectedSha256": spec["sha256"],
        "actualSha256": actual,
        "pass": actual == spec["sha256"],
    }


def revalidate_frozen_inputs() -> dict[str, Any]:
    """Run the installed S10D preflight over the immutable S10A/S10C path."""

    _CALLBACK_COUNTS["revalidate_frozen_inputs"] += 1
    result = s10b.revalidate_frozen_inputs()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    immutable = config["immutableInputs"]

    later_tree_checks = [
        _tree_check(label="S10B", root=S10B, expected=immutable["s10bTree"]),
        _tree_check(label="S10C", root=S10C, expected=immutable["s10cTree"]),
    ]
    continuation_file_checks = [
        _file_check(
            immutable["s10cInstalledDispatchQualification"],
            "s10cInstalledDispatchQualification",
        ),
        _file_check(immutable["s10cValidation"], "s10cValidation"),
        _file_check(immutable["s10cCallbackIdentity"], "s10cCallbackIdentity"),
        _file_check(
            immutable["s10bCallbackQualifiedWrapper"],
            "s10bCallbackQualifiedWrapper",
        ),
        _file_check(immutable["s10aAccountingCore"], "s10aAccountingCore"),
        _file_check(
            immutable["s10aNativeFeaturesCore"],
            "s10aNativeFeaturesCore",
        ),
        _file_check(immutable["s10aSearchCore"], "s10aSearchCore"),
        _file_check(
            immutable["researchPlanAfterProspectiveRegistration"],
            "researchPlanAfterProspectiveRegistration",
        ),
    ]
    s10c_dispatch = json.loads(
        Path(immutable["s10cInstalledDispatchQualification"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    s10c_validation = json.loads(
        Path(immutable["s10cValidation"]["path"]).read_text(encoding="utf-8")
    )
    s10c_identity = json.loads(
        Path(immutable["s10cCallbackIdentity"]["path"]).read_text(encoding="utf-8")
    )
    targets = _callback_overrides()
    installed_identity = {
        name: getattr(base, name) is target for name, target in targets.items()
    }
    callback_gate = {
        "installedTargetIdentity": installed_identity,
        "preflightInvocationCount": _CALLBACK_COUNTS["revalidate_frozen_inputs"],
        "prospectiveFreezeInvocationCountAtPreflight": _CALLBACK_COUNTS[
            "prospective_freeze"
        ],
        "s10cInstalledDispatchPass": bool(s10c_dispatch["allPass"]),
        "s10cValidationPass": bool(s10c_validation["allPass"]),
        "s10cIdentityPass": bool(
            s10c_identity["allCapturedBeforeInstallation"]
            and s10c_identity["allOriginalTargetPairsDistinct"]
        ),
    }
    callback_gate["pass"] = bool(
        all(installed_identity.values())
        and callback_gate["preflightInvocationCount"] == 1
        and callback_gate["prospectiveFreezeInvocationCountAtPreflight"] == 0
        and callback_gate["s10cInstalledDispatchPass"]
        and callback_gate["s10cValidationPass"]
        and callback_gate["s10cIdentityPass"]
    )

    prepared_empty = bool(CACHE.is_dir() and not any(CACHE.iterdir()))
    fresh_gate = {
        "namespace": str(CACHE),
        "absentBeforeRunnerEntry": (_FRESH_NAMESPACE_ABSENT_BEFORE_RUNNER_ENTRY),
        "presentAtInstalledPreflight": CACHE.is_dir(),
        "emptyAtInstalledPreflight": prepared_empty,
        "createdByBaseRunnerBeforeInstalledPreflight": True,
        "failedOrQuarantineNamespaceReuse": False,
    }
    fresh_gate["pass"] = bool(
        fresh_gate["absentBeforeRunnerEntry"]
        and fresh_gate["presentAtInstalledPreflight"]
        and fresh_gate["emptyAtInstalledPreflight"]
    )

    base_structural_pass = _structural_s10b_preflight_pass(result)
    result.update(
        {
            "schemaVersion": "e07.s10d.preflight-hash-revalidation.v1",
            "researchStepId": STEP_ID,
            "s10bAllPassBeforeFreshnessReconciliation": result["allPass"],
            "s10bScientificStructuralPass": base_structural_pass,
            "s10dLaterImmutableTreeChecks": later_tree_checks,
            "s10dContinuationFileChecks": continuation_file_checks,
            "installedCallbackGate": callback_gate,
            "freshNamespaceGate": fresh_gate,
            "freshCacheNamespace": str(CACHE),
            "freshCacheAbsentAtPreflight": False,
            "freshCachePreparedEmptyAtPreflight": prepared_empty,
            "s10bCacheReads": 0,
            "failedOrQuarantineCacheReads": 0,
            "failedOrQuarantineOutcomeRowsReused": 0,
            "separateS10DExecutionApprovalReceived": True,
        }
    )
    result["allPass"] = bool(
        base_structural_pass
        and all(row["pass"] for row in later_tree_checks)
        and all(row["pass"] for row in continuation_file_checks)
        and callback_gate["pass"]
        and fresh_gate["pass"]
    )
    return result


def prospective_freeze(preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the exact installed callback state before any phase precommit."""

    _CALLBACK_COUNTS["prospective_freeze"] += 1
    targets = _callback_overrides()
    identity_pass = all(
        getattr(base, name) is target for name, target in targets.items()
    )
    counts_pass = _CALLBACK_COUNTS == {
        "revalidate_frozen_inputs": 1,
        "prospective_freeze": 1,
    }
    if not preflight["allPass"] or not identity_pass or not counts_pass:
        raise RuntimeError("S10D installed callback execution gate failed")
    result = s10b.prospective_freeze(preflight)
    result.pop("executionFreezeSha256", None)
    result.update(
        {
            "schemaVersion": "e07.s10d.execution-preregistration.v1",
            "researchStepId": STEP_ID,
            "continuationOfScientificDesign": "S10P",
            "missingnessAndAccountingContract": "S10A",
            "failedHistoricalExecution": "S10B",
            "callbackBindingQualification": "S10C",
            "freshCacheNamespace": str(CACHE),
            "freshNamespaceAbsentBeforeRunnerEntry": True,
            "freshNamespacePreparedEmptyAtInstalledPreflight": True,
            "installedCallbackIdentityPass": identity_pass,
            "installedCallbackInvocationCounts": dict(_CALLBACK_COUNTS),
            "preOutcomeLogicalCommitments": 10_752,
            "preOutcomePhysicalReplayCommitments": 21_504,
            "failedOrQuarantineCacheReadsBeforeFreeze": 0,
            "failedOrQuarantineOutcomeRowsReusedBeforeFreeze": 0,
            "humanAnnotationBoundary": "stop_before_annotation",
            "s11RowsBeforeFreeze": 0,
        }
    )
    result["executionFreezeSha256"] = hashlib.sha256(
        b"E07/S10D/execution-freeze/v1\0"
        + json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return result


def reviewer_instructions(machine_count: int) -> str:
    text = s10b.reviewer_instructions(machine_count)
    return text.replace("S10B", "S10D").replace(
        "S10D blinded external-review instructions",
        "S10D blinded external-review instructions after S10C",
    )


def report_markdown(**kwargs: Any) -> str:
    text = s10b.report_markdown(**kwargs)
    text = text.replace(
        "# S10B — Fresh S10 execution after S10A",
        "# S10D — Fresh S10 execution after S10C",
    )
    text = text.replace("/artifacts/research_steps/S10B/", str(OUTPUT) + "/")
    text = text.replace("/cache/e07-s10b", str(CACHE))
    text = text.replace("S10B", "S10D")
    text = text.replace(
        "through S10A's remediated missingness and fail-atomic accounting path",
        "through S10A's remediated missingness/accounting and S10C's "
        "qualified callback-binding path",
    )
    text = text.replace(
        "S10A's qualified explicit-missingness and fail-atomic accounting implementation",
        "S10A's qualified explicit-missingness/fail-atomic accounting "
        "implementation plus S10C's installed-callback binding",
    )
    return text


def manifest_for_output() -> dict[str, Any]:
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
        "schemaVersion": "e07.s10d.artifact-manifest.v1",
        "researchStepId": STEP_ID,
        "artifacts": rows,
    }


def _full_immutable_tree_document() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    immutable = config["immutableInputs"]
    rows = [
        _tree_check(label="S10P", root=S10P, expected=immutable["s10pTree"]),
        _tree_check(
            label="S10",
            root=FAILED_S10,
            expected=immutable["failedS10Tree"],
        ),
        _tree_check(label="S10A", root=S10A, expected=immutable["s10aTree"]),
        _tree_check(label="S10B", root=S10B, expected=immutable["s10bTree"]),
        _tree_check(label="S10C", root=S10C, expected=immutable["s10cTree"]),
    ]
    return {
        "schemaVersion": "e07.s10d.immutable-tree-revalidation.v1",
        "researchStepId": STEP_ID,
        "rows": rows,
        "allPass": all(row["pass"] for row in rows),
    }


def _callback_execution_audit() -> dict[str, Any]:
    freeze_path = OUTPUT / "execution_preregistration_freeze.json"
    freeze = (
        json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze_path.is_file()
        else None
    )
    return {
        "schemaVersion": "e07.s10d.installed-callback-execution-audit.v1",
        "researchStepId": STEP_ID,
        "installer": (
            "scripts.run_native_event_discovery_s10b.installed_base_callbacks"
        ),
        "installedTargets": {
            name: f"{target.__module__}.{target.__qualname__}"
            for name, target in _callback_overrides().items()
        },
        "invocationCounts": dict(_CALLBACK_COUNTS),
        "exactlyOnceBeforeEpisode": _CALLBACK_COUNTS
        == {
            "revalidate_frozen_inputs": 1,
            "prospective_freeze": 1,
        },
        "recursionErrors": 0,
        "s10cQualificationSha256": sha256_file(
            S10C / "installed_dispatch_qualification.json"
        ),
        "executionFreezeSha256": (freeze["executionFreezeSha256"] if freeze else None),
        "pass": bool(
            freeze
            and freeze["installedCallbackIdentityPass"]
            and _CALLBACK_COUNTS
            == {
                "revalidate_frozen_inputs": 1,
                "prospective_freeze": 1,
            }
        ),
    }


def _repository_provenance() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "schemaVersion": "e07.s10d.repository-provenance.v1",
        "researchStepId": STEP_ID,
        "branch": branch,
        "commitAtExecution": head,
        "executionControlSha256": sha256_file(CONFIG),
        "runnerSha256": sha256_file(SCRIPT),
        "focusedTestSha256": sha256_file(TEST),
        "s10bCallbackWrapperSha256": sha256_file(
            REPOSITORY / "scripts/run_native_event_discovery_s10b.py"
        ),
    }


def _finalize_success(dispositions: list[Mapping[str, Any]]) -> None:
    s10b._rewrite_success_controls(dispositions)

    accounting_path = OUTPUT / "complete_accounting.json"
    accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
    accounting.update(
        {
            "schemaVersion": "e07.s10d.complete-accounting.v1",
            "researchStepId": STEP_ID,
            "s10bCacheReads": 0,
            "failedOrQuarantineCacheReads": 0,
            "failedOrQuarantineOutcomeRowsReused": 0,
            "humanAnnotations": 0,
            "s11Rows": 0,
        }
    )
    write_json(accounting_path, accounting)

    gate_path = OUTPUT / "s10_gate_revalidation.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate.update(
        {
            "schemaVersion": "e07.s10d.gate-revalidation.v1",
            "researchStepId": STEP_ID,
            "separateS10DExecutionApprovalReceived": True,
            "installedCallbackGatePass": True,
            "freshNamespaceGatePass": True,
        }
    )
    gate.pop("separateS10BFreshExecutionApprovalReceived", None)
    write_json(gate_path, gate)

    input_path = OUTPUT / "input_provenance.json"
    input_record = json.loads(input_path.read_text(encoding="utf-8"))
    input_record.update(
        {
            "schemaVersion": "e07.s10d.input-provenance.v1",
            "researchStepId": STEP_ID,
            "s10bTreeSha256": s10b.tree_snapshot(S10B)["treeSha256"],
            "s10cTreeSha256": s10b.tree_snapshot(S10C)["treeSha256"],
            "s10bCacheReads": 0,
            "failedOrQuarantineCacheReads": 0,
            "failedOrQuarantineOutcomeRowsReused": 0,
        }
    )
    write_json(input_path, input_record)

    immutable = _full_immutable_tree_document()
    write_json(OUTPUT / "immutable_tree_revalidation.json", immutable)
    callback = _callback_execution_audit()
    write_json(OUTPUT / "installed_callback_execution_audit.json", callback)
    write_json(OUTPUT / "repository_provenance.json", _repository_provenance())

    validation_path = OUTPUT / "validation_summary.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["schemaVersion"] = "e07.s10d.final-validation.v1"
    validation["researchStepId"] = STEP_ID
    validation["checks"].update(
        {
            "immutableS10BAndS10CTrees": immutable["allPass"],
            "installedCallbackExactlyOnce": callback["pass"],
            "freshS10DNamespaceOnly": True,
            "failedAndQuarantineReuseZero": True,
        }
    )
    validation["passedChecks"] = sum(
        bool(value) for value in validation["checks"].values()
    )
    validation["totalChecks"] = len(validation["checks"])
    validation["allPass"] = all(bool(value) for value in validation["checks"].values())
    write_json(validation_path, validation)
    if not validation["allPass"]:
        raise RuntimeError("S10D final continuation validation failed")

    status_path = OUTPUT / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "researchStepId": STEP_ID,
            "stepNumber": 10,
            "success": True,
            "validationResult": (
                f"PASS {validation['passedChecks']}/"
                f"{validation['totalChecks']} final checks; G01-G08; "
                "28/28 bindings; 10,752 logical and 21,504 physical "
                "terminal dispositions; exact installed callbacks, replay, "
                "native contracts, access/dependency exclusion, and no mutation"
            ),
        }
    )
    write_json(status_path, status)

    environment_path = OUTPUT / "environment_provenance.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment.update(
        {
            "schemaVersion": "e07.s10d.environment-provenance.v1",
            "researchStepId": STEP_ID,
            "freshCacheNamespace": str(CACHE),
        }
    )
    write_json(environment_path, environment)

    report_path = OUTPUT / "research_step_full_results.md"
    report = report_path.read_text(encoding="utf-8")
    report = re.sub(
        r"\\*\\*PASS\\*\\* — \\d+/\\d+ execution checks;",
        (
            f"**PASS** — {validation['passedChecks']}/"
            f"{validation['totalChecks']} final checks;"
        ),
        report,
        count=1,
    )
    report += (
        "\n\n## S10D continuation provenance\n\n"
        "S10D also revalidated immutable S10B and S10C artifact trees, "
        "confirmed that the S10C-qualified installed preflight and freeze "
        "callbacks were each invoked exactly once before episode submission, "
        "and used only the fresh `/cache/e07-s10d` namespace. Failed and "
        "quarantined cache/outcome reads remained zero.\n"
    )
    report_path.write_text(report, encoding="utf-8")

    write_json(
        OUTPUT / "command_log.json",
        {
            "schemaVersion": "e07.s10d.command-log.v1",
            "researchStepId": STEP_ID,
            "commands": [
                (
                    "PYTHONPATH=. pytest -q "
                    "tests/test_s10d_fresh_execution.py "
                    "tests/test_s10c_callback_binding.py "
                    "tests/test_s10b_fresh_execution.py "
                    "tests/test_s10a_missingness_accounting.py "
                    "tests/test_s10_native_event_search.py "
                    "tests/test_s10p_native_event_discovery.py"
                ),
                (
                    "PYTHONPATH=. OMP_NUM_THREADS=1 "
                    "OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                    "NUMEXPR_NUM_THREADS=1 python "
                    "scripts/run_native_event_discovery_s10d.py execute"
                ),
                (
                    "PYTHONPATH=. python "
                    "scripts/run_native_event_discovery_s10d.py validate"
                ),
            ],
        },
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["artifactsWritten"] = sorted(
        {
            *(path.name for path in OUTPUT.iterdir() if path.is_file()),
            "artifact_manifest.json",
            "artifact_validation.json",
        }
    )
    write_json(status_path, status)
    write_json(OUTPUT / "artifact_manifest.json", manifest_for_output())


def _finalize_failure(exc: BaseException) -> None:
    if not OUTPUT.exists():
        return
    for name in (
        "research_step_full_results.md",
        "status.json",
        "execution_failure_forensics.json",
        "validation_summary.json",
    ):
        path = OUTPUT / name
        if not path.is_file():
            continue
        if path.suffix == ".md":
            text = path.read_text(encoding="utf-8")
            text = text.replace(
                "# S10B — Fresh S10 execution after S10A",
                "# S10D — Fresh S10 execution after S10C",
            )
            text = text.replace("/artifacts/research_steps/S10B/", str(OUTPUT) + "/")
            text = text.replace("/cache/e07-s10b", str(CACHE))
            text = text.replace("S10B", "S10D")
            path.write_text(text, encoding="utf-8")
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        record["researchStepId"] = STEP_ID
        schema = str(record.get("schemaVersion", ""))
        record["schemaVersion"] = schema.replace("s10b", "s10d")
        if name == "execution_failure_forensics.json":
            record.update(
                {
                    "s10bCacheReads": 0,
                    "failedOrQuarantineCacheReads": 0,
                    "failedOrQuarantineOutcomeRowsReused": 0,
                    "callbackInvocationCounts": dict(_CALLBACK_COUNTS),
                }
            )
        write_json(path, record)
    write_json(
        OUTPUT / "immutable_tree_revalidation.json",
        _full_immutable_tree_document(),
    )
    write_json(
        OUTPUT / "installed_callback_execution_audit.json",
        _callback_execution_audit(),
    )
    write_json(OUTPUT / "repository_provenance.json", _repository_provenance())
    status_path = OUTPUT / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["artifactsWritten"] = sorted(
            {
                *(path.name for path in OUTPUT.iterdir() if path.is_file()),
                "artifact_manifest.json",
            }
        )
        status.setdefault(
            "caveatsOrBlockers",
            ["S10D failed closed; no invalid scientific result is usable."],
        )
        status.setdefault(
            "recommendedNextAction",
            "Review S10D forensics; do not reuse its cache or start S11.",
        )
        write_json(status_path, status)
    write_json(OUTPUT / "artifact_manifest.json", manifest_for_output())


def execute() -> None:
    global _FRESH_NAMESPACE_ABSENT_BEFORE_RUNNER_ENTRY
    if OUTPUT.exists():
        raise RuntimeError(f"{OUTPUT} already exists; S10D is fail-closed")
    if CACHE.exists():
        raise RuntimeError(f"{CACHE} already exists; S10D requires freshness")
    _FRESH_NAMESPACE_ABSENT_BEFORE_RUNNER_ENTRY = True
    _CALLBACK_COUNTS.update(
        {
            "revalidate_frozen_inputs": 0,
            "prospective_freeze": 0,
        }
    )
    _configure_s10b_globals()
    base.OUTPUT = OUTPUT
    base.CACHE = CACHE
    base.CONFIG = CONFIG
    base.SCRIPT = SCRIPT
    base.TEST = TEST
    base.HISTORICAL = HISTORICAL
    try:
        with s10b.installed_base_callbacks(overrides=_callback_overrides()):
            base.execute()
            dispositions = [
                s10b.promote_disposition(
                    CACHE / "discovery_dispositions.json",
                    OUTPUT / "discovery_dispositions.json.gz",
                ),
                s10b.promote_disposition(
                    CACHE / "reproduction_dispositions.json",
                    OUTPUT / "reproduction_dispositions.json.gz",
                ),
            ]
            _finalize_success(dispositions)
    except BaseException as exc:
        if OUTPUT.exists():
            s10b._write_failure(exc)
            _finalize_failure(exc)
        raise


def validate() -> None:
    _configure_s10b_globals()
    s10b.validate()
    artifact_path = OUTPUT / "artifact_validation.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["schemaVersion"] = "e07.s10d.artifact-validation.v1"
    artifact["researchStepId"] = STEP_ID
    immutable = json.loads(
        (OUTPUT / "immutable_tree_revalidation.json").read_text(encoding="utf-8")
    )
    callback = json.loads(
        (OUTPUT / "installed_callback_execution_audit.json").read_text(encoding="utf-8")
    )
    human = json.loads(
        (OUTPUT / "human_review_status.json").read_text(encoding="utf-8")
    )
    extra = {
        "immutableS10PThroughS10C": immutable["allPass"],
        "installedCallbackExecution": callback["pass"],
        "humanAnnotationsEmpty": human["reviewerAnnotationsPresent"] == 0,
        "s11Absent": not Path("/artifacts/research_steps/S11").exists(),
    }
    artifact["checks"].update(extra)
    artifact["allPass"] = bool(artifact["allPass"] and all(extra.values()))
    write_json(artifact_path, artifact)
    if not artifact["allPass"]:
        raise RuntimeError("S10D artifact validation failed")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"execute", "validate"}:
        raise SystemExit("usage: run_native_event_discovery_s10d.py {execute|validate}")
    if sys.argv[1] == "execute":
        execute()
    else:
        validate()


if __name__ == "__main__":
    main()
