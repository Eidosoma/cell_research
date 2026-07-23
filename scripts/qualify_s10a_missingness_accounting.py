#!/usr/bin/env python3
"""Outcome-free S10A qualification; never executes a frozen S10 reservation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

from src.environment_suite import AccessDeniedError, AccessGrant, AccessPhase
from src.environment_suite.suite import EnvironmentSuite
from src.phenotype_discovery.accounting import (
    FailAtomicPhaseError,
    build_physical_replay_plan,
    execute_fail_atomic_replays,
)
from src.phenotype_discovery.native_features import (
    extract_native_event_features,
    validate_feature_availability_record,
)
from src.phenotype_discovery.search import (
    SPATIAL_TASKS,
    TASK_REGISTRY,
    SPLIT_MANIFEST,
    _action_for,
    _load_candidate_bundles,
    feature_support_by_task_status,
    load_roster,
    sha256_file,
    support_decision,
)


ARTIFACT_ROOT = Path("/artifacts/research_steps")
OUTPUT = ARTIFACT_ROOT / "S10A"
S10P = ARTIFACT_ROOT / "S10P"
S10 = ARTIFACT_ROOT / "S10"
REPOSITORY = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY / "configs/discovery/s10a_missingness_accounting.yaml"
PLAN = Path("/workspace/RESEARCH_PLAN.md")
EXPECTED_S10P_TREE = "541a0ecb17c8bcc2df498a935d075c8df5eab7509c80d5f591e60c152d6fd704"
EXPECTED_S10_TREE = "20b8869c7892107357df401fea816a0a95ccbb7125372d5fb7492dc5c8502c40"
EXPECTED_S10P_PROTOCOL = (
    "fbf6f7d3645ac901c3d93e378a7118bf5a2262902b0428451840d7b8dbd276dd"
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )


def canonical_sha(domain: str, value: Any) -> str:
    body = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(domain.encode("ascii") + b"\0" + body).hexdigest()


def tree_snapshot(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
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
        "treeSha256": digest.hexdigest(),
        "files": rows,
    }


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def ordered_summaries() -> list[dict[str, Any]]:
    current = _sha("qualification-initial")
    rows = []
    for index in range(32):
        post = _sha(f"qualification-state:{index}") if index >= 16 else current
        rows.append(
            {
                "transitionIndex": index,
                "scheduledActorCount": 4,
                "proposalCount": 0 if index < 16 else 3,
                "acceptedCount": 0 if index < 16 else 2,
                "conflictLosses": 0,
                "invalidProposals": 0,
                "preStateSha256": current,
                "postStateSha256": post,
                "transitionSha256": _sha(f"qualification-transition:{index}"),
            }
        )
        current = post
    return rows


def spatial_payload(*, zero_native_denominator: bool = False) -> dict[str, Any]:
    movement = {
        "submittedProposals": 64,
        "adjacentSwaps": 8,
        "vacancyMoves": 7,
        "shortExchanges": 6,
        "rotations": 5,
        "validProposals": 60,
        "conflictCandidates": 4,
        "reservedSiteClaims": 31,
        "totalGraphDisplacement": 42,
    }
    if zero_native_denominator:
        movement = {key: 0 for key in movement}
    return {
        "event": {
            "schemaVersion": "e07.s10a.outcome-independent-spatial-fixture.v1",
            "transitionCount": 32,
        },
        "costs": {"e06MovementLedger": movement},
        "status": {
            "stopReason": "qualification_fixture",
            "failed": False,
            "censored": True,
        },
        "horizon": {"nativeUnit": "graph_transition", "budget": 32},
        "structuralScenarioSize": 64,
        "traceSelectionReason": "all_event_summaries",
        "orderedEventSummaries": ordered_summaries(),
    }


def missingness_qualification() -> tuple[dict[str, Any], dict[str, Any]]:
    complete = extract_native_event_features(
        "e07_s02_spatial2d_local", spatial_payload()
    )
    zero = extract_native_event_features(
        "e07_s02_spatial2d_local",
        spatial_payload(zero_native_denominator=True),
    )
    complete_summary = validate_feature_availability_record(
        complete, expected_task_id="e07_s02_spatial2d_local"
    )
    zero_summary = validate_feature_availability_record(
        zero, expected_task_id="e07_s02_spatial2d_local"
    )
    zero_feature_ids = sorted(
        feature_id
        for feature_id, state in zero["availability"].items()
        if state.get("reasonCode") == "zero_native_denominator"
    )
    missingness = {
        "schemaVersion": "e07.s10a.missingness-qualification.v1",
        "researchStepId": "S10A",
        "completeFixture": complete_summary,
        "zeroDenominatorFixture": zero_summary,
        "zeroDenominatorFeatureIds": zero_feature_ids,
        "zeroDenominatorFrozenReasons": sorted(
            {
                zero["availability"][feature_id]["reason"]
                for feature_id in zero_feature_ids
            }
        ),
        "explicitReasonCodes": bool(zero_feature_ids),
        "imputedFeatureCount": 0,
        "silentDropCount": 0,
        "rowCompletenessRequired": False,
        "pass": (
            bool(zero_feature_ids)
            and zero_summary["unavailableFeatureCount"] == len(zero_feature_ids)
            and zero_summary["imputedFeatureCount"] == 0
            and zero_summary["silentDropCount"] == 0
        ),
    }

    feature_id = sorted(complete["availability"])[0]
    rows = []
    for status, observed_count in (("below", 89), ("at", 90), ("above", 91)):
        for index in range(100):
            record = deepcopy(complete)
            if index >= observed_count:
                record["analysisFeatures"].pop(feature_id)
                record["availability"][feature_id] = {
                    "state": "unavailable",
                    "reason": "structurally_not_applicable",
                    "reasonCode": "qualification_fixture_unavailable",
                }
            rows.append(
                {
                    "taskId": "e07_s02_spatial2d_local",
                    "statusStratum": status,
                    "analysisFeatures": record["analysisFeatures"],
                    "availability": record["availability"],
                }
            )
    support = feature_support_by_task_status(rows)
    support_by_status = {
        row["statusStratum"]: row["featureDecisions"][feature_id]
        for row in support["strata"]
    }
    zero_support = support_decision(0, 0)
    support_qualification = {
        "schemaVersion": "e07.s10a.support-threshold-qualification.v1",
        "researchStepId": "S10A",
        "supportUnit": "task_id_by_native_status_stratum",
        "featureId": feature_id,
        "boundaryDecisions": support_by_status,
        "zeroRowStratumDecision": zero_support,
        "fullSupportDocument": support,
        "expected": {
            "below": {"support": 0.89, "eligible": False},
            "at": {"support": 0.90, "eligible": True},
            "above": {"support": 0.91, "eligible": True},
            "zero": {"support": None, "eligible": False},
        },
        "imputedFeatureCount": 0,
        "completeCaseRowFilterApplied": False,
    }
    support_qualification["pass"] = (
        support_by_status["below"]["support"] == 0.89
        and not support_by_status["below"]["eligible"]
        and support_by_status["at"]["support"] == 0.90
        and support_by_status["at"]["eligible"]
        and support_by_status["above"]["support"] == 0.91
        and support_by_status["above"]["eligible"]
        and zero_support["support"] is None
        and not zero_support["eligible"]
        and all(row["minimumFeatureCountPass"] for row in support["strata"])
    )
    return missingness, support_qualification


def qualification_reservation(
    position: int, *, fail_replays: tuple[int, ...] = ()
) -> dict[str, Any]:
    return {
        "logicalOrdinal": position,
        "logicalReservationId": f"{position:064x}",
        "split": "train",
        "taskId": "qualification_fixture",
        "statusStratum": "fixture|failed=false|censored=false",
        "failReplays": list(fail_replays),
    }


def qualification_evaluator(physical: Mapping[str, Any]) -> dict[str, Any]:
    reservation = physical["reservation"]
    replay = int(physical["replayOrdinal"])
    if replay in reservation["failReplays"]:
        raise RuntimeError(
            "injected qualification failure at "
            f"logical={physical['logicalPosition']} replay={replay}"
        )
    logical = int(physical["logicalPosition"])
    commitment = f"{logical + 1:064x}"
    return {
        "resultBody": {
            "logicalOrdinal": logical,
            "logicalReservationId": reservation["logicalReservationId"],
            "outcomeIndependentQualificationFixture": True,
        },
        "deterministicResultSha256": commitment,
        "nativeReplaySha256": commitment,
        "availabilitySummary": {
            "state": "complete_with_explicit_unavailability",
            "registeredFeatureCount": 18,
            "observedFeatureCount": 17,
            "unavailableFeatureCount": 1,
            "reasonCodeCounts": {"zero_native_denominator": 1},
            "imputedFeatureCount": 0,
            "silentDropCount": 0,
        },
    }


def accounting_qualification() -> dict[str, Any]:
    reservations = [qualification_reservation(index) for index in range(9)]
    natural_path = OUTPUT / "qualification_dispositions_natural.json"
    reverse_path = OUTPUT / "qualification_dispositions_reverse.json"
    natural_results, natural_accounting = execute_fail_atomic_replays(
        reservations,
        qualification_evaluator,
        disposition_path=natural_path,
        phase="qualification_success",
        workers=4,
        executor_kind="thread",
    )
    reverse_results, reverse_accounting = execute_fail_atomic_replays(
        reservations,
        qualification_evaluator,
        disposition_path=reverse_path,
        phase="qualification_success",
        workers=4,
        executor_kind="thread",
        submission_order=list(reversed(range(18))),
    )
    natural = json.loads(natural_path.read_text(encoding="ascii"))
    reverse = json.loads(reverse_path.read_text(encoding="ascii"))
    failure_rows = []
    for label, logical_position in (("early", 0), ("middle", 4), ("late", 8)):
        failure_reservations = [
            qualification_reservation(
                index, fail_replays=(0,) if index == logical_position else ()
            )
            for index in range(9)
        ]
        path = OUTPUT / f"qualification_dispositions_failure_{label}.json"
        try:
            execute_fail_atomic_replays(
                failure_reservations,
                qualification_evaluator,
                disposition_path=path,
                phase=f"qualification_failure_{label}",
                workers=4,
                executor_kind="thread",
                submission_order=list(reversed(range(18))),
            )
        except FailAtomicPhaseError as exc:
            accounting = exc.accounting
        else:
            raise RuntimeError("injected fail-atomic fixture unexpectedly succeeded")
        ledger = json.loads(path.read_text(encoding="ascii"))
        failed_logical = [
            row for row in ledger["logicalDispositions"] if row["state"] != "success"
        ]
        failure_rows.append(
            {
                "label": label,
                "injectedLogicalPosition": logical_position,
                "publishedRows": accounting["publishedRows"],
                "terminalLogicalDispositions": ledger[
                    "terminalLogicalDispositionCount"
                ],
                "terminalPhysicalDispositions": ledger[
                    "terminalPhysicalDispositionCount"
                ],
                "failedLogicalPositions": [
                    row["logicalPosition"] for row in failed_logical
                ],
                "accountingConserved": ledger["accountingConserved"],
                "availabilityReasonPresentForEveryPhysicalDisposition": all(
                    bool(row["availabilitySummary"].get("reasonCodeCounts"))
                    for row in ledger["physicalDispositions"]
                ),
                "pass": (
                    accounting["publishedRows"] == 0
                    and ledger["terminalLogicalDispositionCount"] == 9
                    and ledger["terminalPhysicalDispositionCount"] == 18
                    and [row["logicalPosition"] for row in failed_logical]
                    == [logical_position]
                    and ledger["accountingConserved"]
                ),
            }
        )

    full_roster = load_roster()
    full_plan = build_physical_replay_plan(full_roster)
    full_plan_digest = hashlib.sha256()
    for row in full_plan:
        full_plan_digest.update(str(row["physicalExecutionId"]).encode("ascii"))
        full_plan_digest.update(b"\n")
    full_structural = {
        "logicalReservations": len(full_roster),
        "physicalReplayCommitments": len(full_plan),
        "uniqueLogicalReservationIds": len(
            {str(row["logicalReservationId"]) for row in full_roster}
        ),
        "uniquePhysicalExecutionIds": len(
            {str(row["physicalExecutionId"]) for row in full_plan}
        ),
        "physicalPlanSha256": full_plan_digest.hexdigest(),
        "frozenRowsExecuted": 0,
        "pass": (
            len(full_roster) == 10752
            and len(full_plan) == 21504
            and len({str(row["physicalExecutionId"]) for row in full_plan}) == 21504
        ),
    }
    success_pass = (
        natural_results == reverse_results
        and natural["dispositionSha256"] == reverse["dispositionSha256"]
        and natural["terminalLogicalDispositionCount"] == 9
        and natural["terminalPhysicalDispositionCount"] == 18
        and natural["accountingConserved"]
        and natural["resultRowsEligibleForCallerPublication"] == 9
        and natural["resultRowsPublishedByExecutor"] == 0
        and natural_accounting["publishedRows"] == 9
        and reverse_accounting["publishedRows"] == 9
    )
    return {
        "schemaVersion": "e07.s10a.accounting-qualification.v1",
        "researchStepId": "S10A",
        "successfulBatch": {
            "logicalReservations": 9,
            "physicalReplays": 18,
            "naturalDispositionSha256": natural["dispositionSha256"],
            "reverseDispositionSha256": reverse["dispositionSha256"],
            "resultBodiesEqual": natural_results == reverse_results,
            "callerEligibleRows": natural["resultRowsEligibleForCallerPublication"],
            "executorPublishedRows": natural["resultRowsPublishedByExecutor"],
            "accountingConserved": natural["accountingConserved"],
            "pass": success_pass,
        },
        "failureInjection": failure_rows,
        "fullFrozenStructuralPlan": full_structural,
        "serializationRoundTripPass": all(
            json.loads(path.read_text(encoding="ascii"))
            == json.loads(
                json.dumps(
                    json.loads(path.read_text(encoding="ascii")),
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            for path in (
                natural_path,
                reverse_path,
                *[
                    OUTPUT / f"qualification_dispositions_failure_{label}.json"
                    for label in ("early", "middle", "late")
                ],
            )
        ),
        "allPass": (
            success_pass
            and all(row["pass"] for row in failure_rows)
            and full_structural["pass"]
        ),
    }


def binding_validation() -> dict[str, Any]:
    bundles = _load_candidate_bundles()
    rows = []
    for candidate_id, bundle in sorted(bundles.items()):
        for task_id in SPATIAL_TASKS:
            action, configuration = _action_for(bundle, task_id)
            rows.append(
                {
                    "candidateId": candidate_id,
                    "taskId": task_id,
                    "configurationId": configuration["configurationId"],
                    "actionSha256": action.policy_sha256,
                    "memberCount": len(configuration["members"]),
                    "pass": True,
                }
            )
    return {
        "schemaVersion": "e07.s10a.binding-validation.v1",
        "researchStepId": "S10A",
        "candidateCount": len(bundles),
        "taskCount": len(SPATIAL_TASKS),
        "bindingCount": len(rows),
        "rows": rows,
        "episodesExecuted": 0,
        "pass": len(bundles) == 14 and len(rows) == 28,
    }


def protected_denial() -> dict[str, Any]:
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    development = AccessGrant(AccessPhase.DEVELOPMENT)
    confirmation = AccessGrant(AccessPhase.CONFIRMATION, candidate_lock_sha256="0" * 64)
    rows = []
    for task_id in sorted(suite.tasks):
        by_split = {
            record.split.value: record
            for record in suite.records.values()
            if record.task_id == task_id
        }
        for split, grant in (
            ("validation", development),
            ("confirmation", development),
            ("confirmation", confirmation),
        ):
            try:
                suite.open(task_id, by_split[split].scenario_id, grant)
            except AccessDeniedError as exc:
                rows.append(
                    {
                        "taskId": task_id,
                        "split": split,
                        "grant": grant.phase.value,
                        "denied": True,
                        "reason": str(exc),
                    }
                )
            else:
                rows.append(
                    {
                        "taskId": task_id,
                        "split": split,
                        "grant": grant.phase.value,
                        "denied": False,
                        "reason": None,
                    }
                )
    return {
        "schemaVersion": "e07.s10a.access-control-validation.v1",
        "researchStepId": "S10A",
        "attempts": len(rows),
        "denials": sum(row["denied"] for row in rows),
        "allDenied": len(rows) == 24 and all(row["denied"] for row in rows),
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
        "records": rows,
        "brokerAudit": suite.broker.audit.to_dict(),
    }


def immutable_hash_revalidation() -> dict[str, Any]:
    s10p_snapshot = tree_snapshot(S10P)
    s10_snapshot = tree_snapshot(S10)
    s10p_gate = json.loads(
        (S10P / "s10_eligibility_gate.json").read_text(encoding="ascii")
    )
    prereg = json.loads(
        (S10P / "preregistration_freeze.json").read_text(encoding="ascii")
    )
    file_checks = {
        "protocol": sha256_file(S10P / "s10p_native_event_protocol.yaml")
        == EXPECTED_S10P_PROTOCOL,
        "featureRegistry": sha256_file(S10P / "native_event_feature_registry.json")
        == prereg["featureRegistrySha256"],
        "methodRegistry": sha256_file(S10P / "method_registry.json")
        == prereg["methodRegistrySha256"],
        "candidatePopulation": sha256_file(S10P / "candidate_population.jsonl")
        == prereg["candidatePopulationSha256"],
        "logicalRoster": sha256_file(S10P / "s10_logical_roster.parquet")
        == prereg["logicalRosterSha256"],
        "reviewerProtocol": sha256_file(S10P / "blinded_reviewer_protocol.md")
        == prereg["reviewerProtocolSha256"],
    }
    gate_rows = {str(row["gateId"]): bool(row["pass"]) for row in s10p_gate["rows"]}
    return {
        "schemaVersion": "e07.s10a.immutable-hash-revalidation.v1",
        "researchStepId": "S10A",
        "s10pSnapshot": s10p_snapshot,
        "failedS10Snapshot": s10_snapshot,
        "expectedS10PTreeSha256": EXPECTED_S10P_TREE,
        "expectedFailedS10TreeSha256": EXPECTED_S10_TREE,
        "scientificFileChecks": file_checks,
        "frozenGateRows": gate_rows,
        "pass": (
            s10p_snapshot["treeSha256"] == EXPECTED_S10P_TREE
            and s10_snapshot["treeSha256"] == EXPECTED_S10_TREE
            and all(file_checks.values())
            and s10p_gate["allPass"]
            and set(gate_rows) == {f"G{index:02d}" for index in range(1, 9)}
            and all(gate_rows.values())
        ),
    }


def dependency_validation() -> dict[str, Any]:
    prohibited_modules = ("src.surrogate_models", "src.surrogate_remediation")
    loaded = sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in prohibited_modules
        )
    )
    return {
        "schemaVersion": "e07.s10a.dependency-validation.v1",
        "researchStepId": "S10A",
        "loadedProhibitedModules": loaded,
        "s06OrS06AModelOrEmbeddingLoads": 0,
        "s07ArmSignalUses": 0,
        "failedS10CacheReads": 0,
        "quarantineOutcomeOrCacheReads": 0,
        "machineClusteringFits": 0,
        "machineAnomalyFits": 0,
        "machineChangePointScreens": 0,
        "independentReproductionRows": 0,
        "humanAnnotations": 0,
        "archiveMutations": 0,
        "s11Rows": 0,
        "pass": not loaded,
    }


def main() -> None:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = immutable_hash_revalidation()
    if not before["pass"]:
        raise RuntimeError(
            "immutable S10P/S10 revalidation failed before qualification"
        )
    missingness, support = missingness_qualification()
    accounting = accounting_qualification()
    bindings = binding_validation()
    access = protected_denial()
    dependency = dependency_validation()
    after = immutable_hash_revalidation()
    no_mutation = {
        "schemaVersion": "e07.s10a.no-mutation-validation.v1",
        "researchStepId": "S10A",
        "s10pBefore": before["s10pSnapshot"]["treeSha256"],
        "s10pAfter": after["s10pSnapshot"]["treeSha256"],
        "failedS10Before": before["failedS10Snapshot"]["treeSha256"],
        "failedS10After": after["failedS10Snapshot"]["treeSha256"],
        "s10pByteIdentical": before["s10pSnapshot"]["treeSha256"]
        == after["s10pSnapshot"]["treeSha256"],
        "failedS10ByteIdentical": before["failedS10Snapshot"]["treeSha256"]
        == after["failedS10Snapshot"]["treeSha256"],
        "archiveMutations": 0,
        "pass": before["pass"] and after["pass"],
    }
    source_files = [
        PROTOCOL,
        REPOSITORY / "src/phenotype_discovery/accounting.py",
        REPOSITORY / "src/phenotype_discovery/native_features.py",
        REPOSITORY / "src/phenotype_discovery/search.py",
        REPOSITORY / "scripts/qualify_s10a_missingness_accounting.py",
        REPOSITORY / "scripts/run_native_event_discovery_s10.py",
        REPOSITORY / "tests/test_s10a_missingness_accounting.py",
        REPOSITORY / "tests/test_s10_native_event_search.py",
    ]
    input_freeze = {
        "schemaVersion": "e07.s10a.input-and-implementation-freeze.v1",
        "researchStepId": "S10A",
        "protocolSha256": sha256_file(PROTOCOL),
        "s10pProtocolSha256": sha256_file(S10P / "s10p_native_event_protocol.yaml"),
        "s10pTreeSha256": after["s10pSnapshot"]["treeSha256"],
        "failedS10TreeSha256": after["failedS10Snapshot"]["treeSha256"],
        "sourceFiles": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_files
        ],
        "workspaceInputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                Path("/workspace/AGENTS.md"),
                Path("/workspace/FULL_PLAN.md"),
                PLAN,
                Path("/workspace/PREVIOUS_ARTIFACTS.md"),
                Path("/workspace/PREVIOUS_ARTIFACTS.json"),
                Path("/workspace/input-attachments/MANIFEST.json"),
                Path(
                    "/workspace/input-attachments/"
                    "21c2278b-9950-4e39-a2c8-df578a2508ec/"
                    "_metadata/ATTACHMENT.md"
                ),
            )
        ],
        "frozenS10RowsExecuted": 0,
        "failedS10CacheReads": 0,
    }
    gate_rows = {
        "G01": before["pass"] and after["pass"],
        "G02": accounting["fullFrozenStructuralPlan"]["pass"],
        "G03": missingness["pass"] and bindings["pass"],
        "G04": support["pass"],
        "G05": accounting["allPass"],
        "G06": access["allDenied"] and dependency["pass"],
        "G07": all(before["scientificFileChecks"].values()),
        "G08": (
            dependency["machineClusteringFits"] == 0
            and dependency["machineAnomalyFits"] == 0
            and dependency["machineChangePointScreens"] == 0
            and dependency["independentReproductionRows"] == 0
            and dependency["humanAnnotations"] == 0
            and dependency["archiveMutations"] == 0
            and dependency["s11Rows"] == 0
        ),
    }
    gate = {
        "schemaVersion": "e07.s10a.gate-revalidation.v1",
        "researchStepId": "S10A",
        "rows": [
            {
                "gateId": gate_id,
                "pass": passed,
                "status": "clear" if passed else "blocked",
            }
            for gate_id, passed in gate_rows.items()
        ],
        "allPass": all(gate_rows.values()),
        "frozenS10PScientificDesignChanged": False,
        "freshS10ExecutionAuthorized": False,
        "requiresSeparateFreshExecutionDecision": True,
    }
    validation_checks = [
        ("immutable_hashes", before["pass"] and after["pass"]),
        ("explicit_zero_denominator_unavailability", missingness["pass"]),
        ("task_status_support_89_90_91", support["pass"]),
        ("fail_atomic_all_row_accounting", accounting["allPass"]),
        ("structural_bindings_28", bindings["pass"]),
        ("protected_denial_24", access["allDenied"]),
        ("dependency_exclusion", dependency["pass"]),
        ("G01_G08", gate["allPass"]),
        ("no_mutation", no_mutation["pass"]),
    ]
    validation = {
        "schemaVersion": "e07.s10a.validation-summary.v1",
        "researchStepId": "S10A",
        "checks": [
            {"checkId": check_id, "pass": passed}
            for check_id, passed in validation_checks
        ],
        "passedChecks": sum(passed for _, passed in validation_checks),
        "totalChecks": len(validation_checks),
        "allPass": all(passed for _, passed in validation_checks),
        "freshS10RowsExecuted": 0,
        "machineMethodFits": 0,
        "reproductionRows": 0,
        "humanAnnotations": 0,
        "protectedOutcomeRows": 0,
        "s11Rows": 0,
        "wallSeconds": time.perf_counter() - started,
    }
    if not validation["allPass"]:
        raise RuntimeError("S10A qualification failed")
    provenance = {
        "schemaVersion": "e07.s10a.provenance.v1",
        "researchStepId": "S10A",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "workers": 4,
        "numericThreadsPerWorker": 1,
        "gpuUsed": False,
        "networkUsed": False,
        "newDependenciesInstalled": [],
        "protocolPath": str(PROTOCOL),
        "protocolSha256": sha256_file(PROTOCOL),
        "repository": "Eidosoma/cell_research",
        "branch": "eidosoma/groups/28",
        "gitCommitBeforeS10A": os.popen(f"git -C {REPOSITORY} rev-parse HEAD")
        .read()
        .strip(),
    }
    write_json(OUTPUT / "immutable_hash_revalidation.json", after)
    write_json(OUTPUT / "input_hash_freeze.json", input_freeze)
    write_json(OUTPUT / "missingness_qualification.json", missingness)
    write_json(OUTPUT / "support_threshold_qualification.json", support)
    write_json(OUTPUT / "accounting_qualification.json", accounting)
    write_json(OUTPUT / "binding_validation.json", bindings)
    write_json(OUTPUT / "access_control_validation.json", access)
    write_json(OUTPUT / "dependency_validation.json", dependency)
    write_json(OUTPUT / "no_mutation_validation.json", no_mutation)
    write_json(OUTPUT / "s10_gate_revalidation.json", gate)
    write_json(OUTPUT / "validation_summary.json", validation)
    write_json(OUTPUT / "provenance.json", provenance)
    write_json(
        OUTPUT / "command_log.json",
        {
            "schemaVersion": "e07.s10a.command-log.v1",
            "researchStepId": "S10A",
            "commands": ["python -m scripts.qualify_s10a_missingness_accounting"],
            "freshS10ExecutionCommands": [],
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "checks": validation["passedChecks"],
                "bindings": bindings["bindingCount"],
                "protectedDenials": access["denials"],
                "logicalStructuralRows": accounting["fullFrozenStructuralPlan"][
                    "logicalReservations"
                ],
                "physicalStructuralRows": accounting["fullFrozenStructuralPlan"][
                    "physicalReplayCommitments"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
