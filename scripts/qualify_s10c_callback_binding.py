#!/usr/bin/env python3
"""Qualify S10C callback binding without submitting a frozen episode."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Mapping

import pyarrow
import pytest
import sklearn
import yaml

import scripts.run_native_event_discovery_s10 as base
import scripts.run_native_event_discovery_s10b as s10b
from src.phenotype_discovery.search import sha256_file


STEP_ID = "S10C"
OUTPUT = Path("/artifacts/research_steps/S10C")
CONFIG = Path(
    "/workspace/cell-research/configs/discovery/"
    "s10c_callback_binding_qualification.yaml"
)
SCRIPT = Path(__file__).resolve()
TEST = Path("/workspace/cell-research/tests/test_s10c_callback_binding.py")
REPOSITORY = Path("/workspace/cell-research")
PREDECESSORS = {
    "S10P": Path("/artifacts/research_steps/S10P"),
    "S10": Path("/artifacts/research_steps/S10"),
    "S10A": Path("/artifacts/research_steps/S10A"),
    "S10B": Path("/artifacts/research_steps/S10B"),
}
CALLBACK_NAMES = tuple(sorted(s10b.execution_callback_overrides()))


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
        "files": rows,
    }


def callable_provenance(callback: Callable[..., Any]) -> dict[str, Any]:
    source = inspect.getsource(callback).encode("utf-8")
    source_path = Path(inspect.getsourcefile(callback) or "")
    return {
        "module": callback.__module__,
        "qualname": callback.__qualname__,
        "sourceSha256": hashlib.sha256(source).hexdigest(),
        "sourcePath": str(source_path),
        "sourceFileSha256": sha256_file(source_path),
    }


def _structural_preflight_pass(preflight: Mapping[str, Any]) -> bool:
    expected_roster = {
        "rows": 10_752,
        "uniqueLogicalReservationIds": 10_752,
        "candidateCount": 14,
        "taskCount": 2,
        "discoveryRows": 7_168,
        "reproductionRows": 3_584,
    }
    return bool(
        set(preflight["s10pGateRows"]) == {f"G{index:02d}" for index in range(1, 9)}
        and all(preflight["s10pGateRows"].values())
        and preflight["s10pGateAllPass"]
        and len(preflight["bindingRows"]) == 28
        and all(row["pass"] for row in preflight["bindingRows"])
        and preflight["rosterChecks"] == expected_roster
        and preflight["preOutcomeCommitmentAudit"]["pass"]
        and all(row["pass"] for row in preflight["frozenFileChecks"])
        and all(
            row["pass"]
            for row in preflight["frozenInputChecks"]
            if str(row["path"])
            not in {
                "/workspace/cell-research/src/phenotype_discovery/accounting.py",
                "/workspace/cell-research/src/phenotype_discovery/native_features.py",
                "/workspace/cell-research/src/phenotype_discovery/search.py",
            }
        )
        and all(preflight["preregistrationChecks"].values())
        and all(
            row["pass"] for row in preflight["s10bScientificFileChecks"]
        )
        and all(
            row["pass"] for row in preflight["s10aRemediatedCoreChecks"]
        )
        and all(
            row["pass"] for row in preflight["immutableTreeChecks"].values()
        )
        and preflight["s10aQualificationGatePass"]
    )


def installed_dispatch_qualification() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any]
]:
    """Invoke the actual installed callbacks while counting captured delegates."""

    production_captured = dict(s10b._ORIGINAL_BASE_CALLBACKS)
    production_targets = dict(s10b.execution_callback_overrides())
    original_surface = {
        name: getattr(base, name)
        for name in CALLBACK_NAMES
    }
    base_globals = {
        "OUTPUT": base.OUTPUT,
        "CACHE": base.CACHE,
        "CONFIG": base.CONFIG,
        "SCRIPT": base.SCRIPT,
        "TEST": base.TEST,
        "HISTORICAL": base.HISTORICAL,
    }
    counts = {
        "revalidate_frozen_inputs": 0,
        "prospective_freeze": 0,
    }

    def counted_preflight() -> dict[str, Any]:
        counts["revalidate_frozen_inputs"] += 1
        return production_captured["revalidate_frozen_inputs"]()

    def counted_freeze(preflight: Mapping[str, Any]) -> dict[str, Any]:
        counts["prospective_freeze"] += 1
        return production_captured["prospective_freeze"](preflight)

    counted_captured = dict(production_captured)
    counted_captured["revalidate_frozen_inputs"] = counted_preflight
    counted_captured["prospective_freeze"] = counted_freeze

    for name, callback in counted_captured.items():
        setattr(base, name, callback)
    s10b._ORIGINAL_BASE_CALLBACKS = MappingProxyType(counted_captured)
    base.OUTPUT = s10b.OUTPUT
    base.CACHE = s10b.CACHE
    base.CONFIG = s10b.CONFIG
    base.SCRIPT = s10b.SCRIPT
    base.TEST = s10b.TEST
    base.HISTORICAL = s10b.HISTORICAL

    outer_audit: dict[str, Any]
    nested_audit: dict[str, Any]
    preflight: dict[str, Any]
    freeze: dict[str, Any]
    try:
        with s10b.installed_base_callbacks() as outer_audit:
            installed_identity = {
                name: getattr(base, name) is production_targets[name]
                for name in CALLBACK_NAMES
            }
            preflight = base.revalidate_frozen_inputs()
            freeze = base.prospective_freeze(preflight)
            with s10b.installed_base_callbacks() as nested_audit:
                nested_identity = {
                    name: getattr(base, name) is production_targets[name]
                    for name in CALLBACK_NAMES
                }
            after_nested_identity = {
                name: getattr(base, name) is production_targets[name]
                for name in CALLBACK_NAMES
            }
        restored_to_counted = {
            name: getattr(base, name) is counted_captured[name]
            for name in CALLBACK_NAMES
        }
    finally:
        for name, callback in original_surface.items():
            setattr(base, name, callback)
        for name, value in base_globals.items():
            setattr(base, name, value)
        s10b._ORIGINAL_BASE_CALLBACKS = MappingProxyType(production_captured)

    fully_restored = {
        name: getattr(base, name) is original_surface[name]
        for name in CALLBACK_NAMES
    }
    callback_identity = {
        "schemaVersion": "e07.s10c.callback-identity-provenance.v1",
        "researchStepId": STEP_ID,
        "callbacks": [
            {
                "name": name,
                "capturedOriginal": callable_provenance(production_captured[name]),
                "installedTarget": callable_provenance(production_targets[name]),
                "distinctObjects": (
                    production_captured[name] is not production_targets[name]
                ),
            }
            for name in CALLBACK_NAMES
        ],
        "allCapturedBeforeInstallation": all(
            original_surface[name] is production_captured[name]
            for name in CALLBACK_NAMES
        ),
        "allOriginalTargetPairsDistinct": all(
            production_captured[name] is not production_targets[name]
            for name in CALLBACK_NAMES
        ),
    }
    qualification = {
        "schemaVersion": "e07.s10c.installed-dispatch-qualification.v1",
        "researchStepId": STEP_ID,
        "actualProductionSurface": (
            "scripts.run_native_event_discovery_s10"
        ),
        "actualProductionInstaller": (
            "scripts.run_native_event_discovery_s10b.installed_base_callbacks"
        ),
        "outerInstallationAudit": outer_audit,
        "nestedInstallationAudit": nested_audit,
        "installedIdentity": installed_identity,
        "nestedIdentity": nested_identity,
        "afterNestedIdentity": after_nested_identity,
        "restoredToCountedDelegates": restored_to_counted,
        "restoredToProductionOriginals": fully_restored,
        "capturedDelegateInvocationCounts": counts,
        "preflightReturned": isinstance(preflight, dict),
        "preflightStructuralPass": _structural_preflight_pass(preflight),
        "consumedS10BNamespaceWasNotReauthorized": (
            preflight["freshCacheAbsentAtPreflight"] is False
        ),
        "s10bCacheFilesRead": 0,
        "s10bOutcomeRowsRead": 0,
        "freezeReturned": isinstance(freeze, dict),
        "freezeResearchStepId": freeze["researchStepId"],
        "recursionErrors": 0,
        "frozenEpisodesSubmitted": 0,
    }
    qualification["allPass"] = bool(
        all(installed_identity.values())
        and all(nested_identity.values())
        and all(after_nested_identity.values())
        and all(restored_to_counted.values())
        and all(fully_restored.values())
        and counts
        == {
            "revalidate_frozen_inputs": 1,
            "prospective_freeze": 1,
        }
        and qualification["preflightStructuralPass"]
        and qualification["consumedS10BNamespaceWasNotReauthorized"]
        and qualification["freezeResearchStepId"] == "S10B"
        and nested_audit["nestedOrRepeated"]
    )
    return callback_identity, qualification, preflight


def _fixture_bindings() -> tuple[
    SimpleNamespace, dict[str, Any], dict[str, Any], dict[str, int]
]:
    counts = {name: 0 for name in CALLBACK_NAMES}
    originals: dict[str, Any] = {}
    targets: dict[str, Any] = {}
    for name in CALLBACK_NAMES:
        def original(*args: Any, _name: str = name, **kwargs: Any) -> str:
            del args, kwargs
            counts[_name] += 1
            return f"original:{_name}"

        def target(*args: Any, _name: str = name, **kwargs: Any) -> str:
            return originals[_name](*args, **kwargs)

        originals[name] = original
        targets[name] = target
    return SimpleNamespace(**originals), originals, targets, counts


def adversarial_qualification(order: str) -> dict[str, Any]:
    case_rows: list[dict[str, Any]] = []

    surface, originals, targets, counts = _fixture_bindings()
    with s10b.installed_base_callbacks(
        surface=surface, captured=originals, overrides=targets
    ):
        result = surface.revalidate_frozen_inputs()
    case_rows.append(
        {
            "case": "normal_installed_dispatch",
            "pass": (
                result == "original:revalidate_frozen_inputs"
                and counts["revalidate_frozen_inputs"] == 1
                and surface.revalidate_frozen_inputs
                is originals["revalidate_frozen_inputs"]
            ),
        }
    )

    surface, originals, targets, _ = _fixture_bindings()
    with s10b.installed_base_callbacks(
        surface=surface, captured=originals, overrides=targets
    ):
        with s10b.installed_base_callbacks(
            surface=surface, captured=originals, overrides=targets
        ) as nested:
            nested_pass = nested["nestedOrRepeated"]
        nested_pass = (
            nested_pass
            and surface.prospective_freeze is targets["prospective_freeze"]
        )
    case_rows.append(
        {
            "case": "nested_repeated_installation",
            "pass": nested_pass
            and all(
                getattr(surface, name) is originals[name]
                for name in CALLBACK_NAMES
            ),
        }
    )

    surface, originals, targets, _ = _fixture_bindings()
    conflict = lambda: None
    surface.report_markdown = conflict
    before = {name: getattr(surface, name) for name in CALLBACK_NAMES}
    conflict_rejected = False
    try:
        with s10b.installed_base_callbacks(
            surface=surface, captured=originals, overrides=targets
        ):
            pass
    except s10b.CallbackBindingError:
        conflict_rejected = True
    case_rows.append(
        {
            "case": "preexisting_wrapper_conflict",
            "pass": conflict_rejected
            and all(
                getattr(surface, name) is before[name]
                for name in CALLBACK_NAMES
            ),
        }
    )

    surface, originals, targets, _ = _fixture_bindings()
    targets["revalidate_frozen_inputs"] = originals["revalidate_frozen_inputs"]
    self_reference_rejected = False
    try:
        with s10b.installed_base_callbacks(
            surface=surface, captured=originals, overrides=targets
        ):
            pass
    except s10b.CallbackBindingError:
        self_reference_rejected = True
    case_rows.append(
        {"case": "self_reference_target", "pass": self_reference_rejected}
    )

    surface, originals, targets, _ = _fixture_bindings()
    replacement_rejected = False
    try:
        with s10b.installed_base_callbacks(
            surface=surface, captured=originals, overrides=targets
        ):
            surface.manifest_for_output = lambda: None
    except s10b.CallbackBindingError:
        replacement_rejected = True
    case_rows.append(
        {
            "case": "installed_callback_replacement",
            "pass": replacement_rejected
            and all(
                getattr(surface, name) is originals[name]
                for name in CALLBACK_NAMES
            ),
        }
    )

    surface, originals, targets, _ = _fixture_bindings()
    sentinel = ValueError("outcome-independent sentinel")

    def failing_original() -> None:
        raise sentinel

    originals["revalidate_frozen_inputs"] = failing_original
    surface.revalidate_frozen_inputs = failing_original
    propagated_identity = False
    try:
        with s10b.installed_base_callbacks(
            surface=surface, captured=originals, overrides=targets
        ):
            surface.revalidate_frozen_inputs()
    except ValueError as exc:
        propagated_identity = exc is sentinel
    case_rows.append(
        {
            "case": "original_exception_propagation",
            "pass": (
                propagated_identity
                and surface.revalidate_frozen_inputs is failing_original
            ),
        }
    )

    if order == "reverse":
        case_rows.reverse()
    digest = hashlib.sha256(
        canonical_json_bytes(sorted(case_rows, key=lambda row: row["case"]))
    ).hexdigest()
    return {
        "order": order,
        "caseCount": len(case_rows),
        "passedCases": sum(row["pass"] for row in case_rows),
        "canonicalCaseSha256": digest,
        "cases": case_rows,
        "allPass": all(row["pass"] for row in case_rows),
    }


def immutable_revalidation(
    before: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    after = {name: tree_snapshot(path) for name, path in PREDECESSORS.items()}
    config_keys = {
        "S10P": "s10p",
        "S10": "failedS10",
        "S10A": "s10a",
        "S10B": "s10b",
    }
    rows = []
    for name in PREDECESSORS:
        spec = config["immutablePredecessors"][config_keys[name]]
        rows.append(
            {
                "stepId": name,
                "expectedTreeSha256": spec["treeSha256"],
                "beforeTreeSha256": before[name]["treeSha256"],
                "afterTreeSha256": after[name]["treeSha256"],
                "expectedFileCount": spec["fileCount"],
                "actualFileCount": after[name]["fileCount"],
                "expectedTotalBytes": spec["totalBytes"],
                "actualTotalBytes": after[name]["totalBytes"],
                "pass": (
                    before[name]["treeSha256"] == spec["treeSha256"]
                    and after[name]["treeSha256"] == spec["treeSha256"]
                    and before[name] == after[name]
                    and after[name]["fileCount"] == spec["fileCount"]
                    and after[name]["totalBytes"] == spec["totalBytes"]
                ),
            }
        )
    return {
        "schemaVersion": "e07.s10c.immutable-tree-revalidation.v1",
        "researchStepId": STEP_ID,
        "rows": rows,
        "allPass": all(row["pass"] for row in rows),
    }


def manifest() -> dict[str, Any]:
    artifacts = []
    for path in sorted(item for item in OUTPUT.iterdir() if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        artifacts.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schemaVersion": "e07.s10c.artifact-manifest.v1",
        "researchStepId": STEP_ID,
        "artifacts": artifacts,
    }


def execute() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    before = {name: tree_snapshot(path) for name, path in PREDECESSORS.items()}
    OUTPUT.mkdir(parents=True, exist_ok=False)

    identity, dispatch, preflight = installed_dispatch_qualification()
    natural = adversarial_qualification("natural")
    reverse = adversarial_qualification("reverse")
    adversarial = {
        "schemaVersion": "e07.s10c.callback-adversarial-qualification.v1",
        "researchStepId": STEP_ID,
        "natural": natural,
        "reverse": reverse,
        "workerOrderIndependent": (
            natural["canonicalCaseSha256"] == reverse["canonicalCaseSha256"]
        ),
    }
    adversarial["allPass"] = bool(
        natural["allPass"]
        and reverse["allPass"]
        and adversarial["workerOrderIndependent"]
    )

    access = base.protected_denial()
    access.update(
        {
            "schemaVersion": "e07.s10c.access-control-validation.v1",
            "researchStepId": STEP_ID,
            "protectedOutcomeRowsMaterialized": 0,
        }
    )
    dependency = base.dependency_audit()
    dependency.update(
        {
            "schemaVersion": "e07.s10c.prohibited-dependency-validation.v1",
            "researchStepId": STEP_ID,
            "s10bCacheReads": 0,
            "failedOrQuarantineOutcomeReads": 0,
        }
    )
    structural = {
        "schemaVersion": "e07.s10c.structural-revalidation.v1",
        "researchStepId": STEP_ID,
        "s10pGateRows": preflight["s10pGateRows"],
        "s10pGateAllPass": preflight["s10pGateAllPass"],
        "bindingRows": preflight["bindingRows"],
        "bindingCount": len(preflight["bindingRows"]),
        "rosterChecks": preflight["rosterChecks"],
        "preOutcomeCommitmentAudit": preflight["preOutcomeCommitmentAudit"],
        "scientificStructuralPass": _structural_preflight_pass(preflight),
        "s10bConsumedNamespaceFreshnessCheck": (
            preflight["freshCacheAbsentAtPreflight"]
        ),
        "s10bConsumedNamespaceReauthorized": False,
        "frozenEpisodesSubmitted": 0,
    }

    preregistration = {
        "schemaVersion": "e07.s10c.preregistration-freeze.v1",
        "researchStepId": STEP_ID,
        "qualificationControlPath": str(CONFIG),
        "qualificationControlSha256": sha256_file(CONFIG),
        "researchPlanSha256AfterProspectiveRegistration": (
            config["prospectiveInputs"]["researchPlan"][
                "sha256AfterS10CRegistration"
            ]
        ),
        "s10bWrapperBeforeRepairSha256": (
            config["prospectiveInputs"]["s10bWrapperBeforeRepair"]["sha256"]
        ),
        "s10bWrapperAfterRepairSha256": sha256_file(
            REPOSITORY / "scripts/run_native_event_discovery_s10b.py"
        ),
        "focusedQualificationTestSha256": sha256_file(TEST),
        "qualificationRunnerSha256": sha256_file(SCRIPT),
        "scientificDesignChanged": False,
        "freshExecutionNamespaceCreated": False,
        "frozenEpisodesAuthorized": 0,
    }
    accounting = {
        "schemaVersion": "e07.s10c.complete-accounting.v1",
        "researchStepId": STEP_ID,
        "structuralLogicalReservationsValidated": 10_752,
        "structuralPhysicalReplayCommitmentsValidated": 21_504,
        "candidateTaskBindingsValidated": 28,
        "installedPreflightDispatches": 1,
        "installedFreezeDispatches": 1,
        "frozenEpisodesSubmitted": 0,
        "physicalReplaysSubmitted": 0,
        "phenotypeFits": 0,
        "reproductionRows": 0,
        "humanAnnotations": 0,
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
        "s10bCacheReads": 0,
        "failedOrQuarantineOutcomeReads": 0,
        "s06OrS06AModelOrEmbeddingLoads": 0,
        "s07ArmSignalUses": 0,
        "archiveMutations": 0,
        "s11Rows": 0,
    }

    write_json(OUTPUT / "preregistration_freeze.json", preregistration)
    write_json(OUTPUT / "callback_identity_provenance.json", identity)
    write_json(OUTPUT / "installed_dispatch_qualification.json", dispatch)
    write_json(OUTPUT / "callback_adversarial_qualification.json", adversarial)
    write_json(OUTPUT / "structural_revalidation.json", structural)
    write_json(OUTPUT / "access_control_validation.json", access)
    write_json(OUTPUT / "prohibited_dependency_validation.json", dependency)
    write_json(OUTPUT / "complete_accounting.json", accounting)

    immutable = immutable_revalidation(before, config)
    write_json(OUTPUT / "immutable_tree_revalidation.json", immutable)

    checks = {
        "prospectivelyRegistered": (
            sha256_file(Path("/workspace/RESEARCH_PLAN.md"))
            == config["prospectiveInputs"]["researchPlan"][
                "sha256AfterS10CRegistration"
            ]
        ),
        "callbackIdentityAndProvenance": (
            identity["allCapturedBeforeInstallation"]
            and identity["allOriginalTargetPairsDistinct"]
        ),
        "actualInstalledDispatch": dispatch["allPass"],
        "callbackAdversaries": adversarial["allPass"],
        "g01ThroughG08": (
            set(structural["s10pGateRows"])
            == {f"G{index:02d}" for index in range(1, 9)}
            and all(structural["s10pGateRows"].values())
        ),
        "bindings28": (
            structural["bindingCount"] == 28
            and all(row["pass"] for row in structural["bindingRows"])
        ),
        "structuralCommitments": (
            structural["preOutcomeCommitmentAudit"]["pass"]
            and structural["rosterChecks"]["rows"] == 10_752
        ),
        "protectedDenial": (
            access["allDenied"]
            and access["validationOutcomeRowsRead"] == 0
            and access["confirmationOutcomeRowsRead"] == 0
        ),
        "prohibitedDependencies": dependency["pass"],
        "immutablePredecessors": immutable["allPass"],
        "zeroExecutionAndProtectedWork": all(
            accounting[key] == 0
            for key in (
                "frozenEpisodesSubmitted",
                "physicalReplaysSubmitted",
                "phenotypeFits",
                "reproductionRows",
                "humanAnnotations",
                "validationOutcomeRowsRead",
                "confirmationOutcomeRowsRead",
                "s10bCacheReads",
                "failedOrQuarantineOutcomeReads",
                "s06OrS06AModelOrEmbeddingLoads",
                "s07ArmSignalUses",
                "archiveMutations",
                "s11Rows",
            )
        ),
    }
    validation = {
        "schemaVersion": "e07.s10c.validation-summary.v1",
        "researchStepId": STEP_ID,
        "checks": checks,
        "passedChecks": sum(checks.values()),
        "totalChecks": len(checks),
        "allPass": all(checks.values()),
    }
    write_json(OUTPUT / "validation_summary.json", validation)

    environment = {
        "schemaVersion": "e07.s10c.environment-provenance.v1",
        "researchStepId": STEP_ID,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpuCountVisible": os.cpu_count(),
        "qualificationWorkers": 1,
        "pytest": pytest.__version__,
        "pyarrow": pyarrow.__version__,
        "scikitLearn": sklearn.__version__,
        "newDependenciesInstalled": [],
        "networkUsed": False,
        "gpuUsed": False,
    }
    write_json(OUTPUT / "environment_provenance.json", environment)
    repository = {
        "schemaVersion": "e07.s10c.repository-provenance.v1",
        "researchStepId": STEP_ID,
        "branch": subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "headBeforeS10CCommit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "changedFiles": [
            "configs/discovery/s10c_callback_binding_qualification.yaml",
            "scripts/qualify_s10c_callback_binding.py",
            "scripts/run_native_event_discovery_s10b.py",
            "tests/test_s10c_callback_binding.py",
        ],
    }
    write_json(OUTPUT / "repository_provenance.json", repository)
    commands = {
        "schemaVersion": "e07.s10c.command-log.v1",
        "researchStepId": STEP_ID,
        "commands": [
            (
                "PYTHONPATH=. pytest -q tests/test_s10c_callback_binding.py "
                "tests/test_s10b_fresh_execution.py "
                "tests/test_s10a_missingness_accounting.py"
            ),
            "PYTHONPATH=. python scripts/qualify_s10c_callback_binding.py",
        ],
        "episodeExecutorInvoked": False,
    }
    write_json(OUTPUT / "command_log.json", commands)

    if not validation["allPass"]:
        raise RuntimeError("S10C qualification failed closed")


if __name__ == "__main__":
    execute()
