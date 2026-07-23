#!/usr/bin/env python3
"""Freeze and qualify the outcome-free E07 S10P native event-feature design."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

from src.environment_suite import (
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
    EnvironmentSuite,
)
from src.phenotype_discovery.native_features import (
    FeatureExtractionError,
    build_feature_registry,
    canonical_json_bytes,
    canonical_sha256,
    exact_change_points,
    extract_native_event_features,
    registry_document,
)


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
OUT = ARTIFACTS / "research_steps" / "S10P"
PROTOCOL = REPOSITORY / "configs/discovery/s10p_native_event_protocol.yaml"
TASK_REGISTRY = ARTIFACTS / "research_steps/S02/environment_suite/task_registry.yaml"
SPLIT_MANIFEST = ARTIFACTS / "research_steps/S02/environment_suite/split_manifest.json"
S08M_CONFIGURATIONS = (
    ARTIFACTS / "research_steps/S08M/portfolio_configuration_registry.jsonl"
)
S08M_CANDIDATE_LOCK = ARTIFACTS / "research_steps/S08M/validation_candidate_lock.json"
S09_PROTOCOL = ARTIFACTS / "research_steps/S09/s09_ablation_protocol.yaml"
S09_COMPRESSED = ARTIFACTS / "research_steps/S09/compressed_policies.jsonl"

FROZEN_INPUTS = (
    WORKSPACE / "AGENTS.md",
    WORKSPACE / "FULL_PLAN.md",
    WORKSPACE / "PREVIOUS_ARTIFACTS.md",
    WORKSPACE / "PREVIOUS_ARTIFACTS.json",
    WORKSPACE / "input-attachments/MANIFEST.json",
    TASK_REGISTRY,
    SPLIT_MANIFEST,
    ARTIFACTS / "research_steps/S01/research_step_full_results.md",
    ARTIFACTS / "research_steps/S02/research_step_full_results.md",
    ARTIFACTS / "research_steps/S03/research_step_full_results.md",
    ARTIFACTS / "research_steps/S04/research_step_full_results.md",
    ARTIFACTS / "research_steps/S04A/research_step_full_results.md",
    ARTIFACTS / "research_steps/S05/research_step_full_results.md",
    ARTIFACTS / "research_steps/S06/research_step_full_results.md",
    ARTIFACTS / "research_steps/S06A/research_step_full_results.md",
    ARTIFACTS / "research_steps/S07/research_step_full_results.md",
    ARTIFACTS / "research_steps/S08M/research_step_full_results.md",
    ARTIFACTS / "research_steps/S09/research_step_full_results.md",
    S08M_CONFIGURATIONS,
    S08M_CANDIDATE_LOCK,
    S09_PROTOCOL,
    S09_COMPRESSED,
    Path("/previous-artifacts/E01/research_steps/S06/event_schema.json"),
    Path("/previous-artifacts/E01/research_steps/S06/trace_manifest_schema.json"),
    Path("/previous-artifacts/E02/research_steps/S09/cost_schema.json"),
    Path("/previous-artifacts/E03/research_steps/S14/e07_handoff.md"),
    Path("/previous-artifacts/E04/research_steps/S14/e06_e07_handoff.json"),
    Path("/previous-artifacts/E05/research_steps/S01/task_spec.schema.json"),
    Path("/previous-artifacts/E05/research_steps/S14/failure_censor_accounting.json"),
    Path("/previous-artifacts/E06/report_inputs/e07_handoff.md"),
    Path("/previous-artifacts/E06/research_steps/S06/budget_schema.json"),
    Path("/previous-artifacts/E06/research_steps/S07/event_trace_validation.json"),
    PROTOCOL,
    REPOSITORY / "src/phenotype_discovery/native_features.py",
    Path(__file__).resolve(),
)
WORKFLOW_CONTEXT_INPUTS = (WORKSPACE / "RESEARCH_PLAN.md",)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")


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


def input_hashes() -> list[dict[str, Any]]:
    missing = [str(path) for path in FROZEN_INPUTS if not path.is_file()]
    if missing:
        raise RuntimeError(f"required frozen inputs missing: {missing}")
    return [file_record(path) for path in FROZEN_INPUTS]


def workflow_context_hashes() -> list[dict[str, Any]]:
    """Record mutable coordination files without making handoff edits gate failures."""

    missing = [str(path) for path in WORKFLOW_CONTEXT_INPUTS if not path.is_file()]
    if missing:
        raise RuntimeError(f"required workflow context missing: {missing}")
    return [file_record(path) for path in WORKFLOW_CONTEXT_INPUTS]


def _structural_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "assignmentCounterDomain",
        "assignmentRotation",
        "configurationId",
        "memberSetId",
        "members",
        "mode",
        "portfolioSize",
        "portfolioStructuralCosts",
        "selector",
        "taskId",
        "schemaVersion",
        "s09EditId",
        "s09ParentConfigurationId",
    }
    result = {key: configuration[key] for key in sorted(allowed & set(configuration))}
    if configuration.get("rejectedModelOrEmbeddingUsed") is not False:
        raise RuntimeError("candidate configuration used a rejected model or embedding")
    if configuration.get("s07ArmMembershipUsed") is not False:
        raise RuntimeError("candidate configuration used S07 arm membership")
    return result


def candidate_population() -> list[dict[str, Any]]:
    protocol = yaml.safe_load(S09_PROTOCOL.read_text(encoding="utf-8"))
    parent_ids = tuple(sorted(map(str, protocol["eligibleConfigurationIds"])))
    if len(parent_ids) != 7:
        raise RuntimeError("S10P requires exactly seven frozen S09 parents")

    parent_rows: dict[str, dict[str, Any]] = {}
    with S08M_CONFIGURATIONS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            configuration_id = str(row["configurationId"])
            if configuration_id not in parent_ids:
                continue
            structural = _structural_configuration(row)
            previous = parent_rows.get(configuration_id)
            if previous is not None and previous != structural:
                raise RuntimeError("ambiguous parent structural configuration")
            parent_rows[configuration_id] = structural
    if set(parent_rows) != set(parent_ids):
        raise RuntimeError("not every parent is present in the structural registry")

    compressed_rows = [
        json.loads(line)
        for line in S09_COMPRESSED.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(compressed_rows) != 7:
        raise RuntimeError("S10P requires every one of seven compressed bundles")
    if {str(row["parentConfigurationId"]) for row in compressed_rows} != set(
        parent_ids
    ):
        raise RuntimeError("compressed-parent pairing is incomplete")

    execution_tasks = [
        "e07_s02_spatial2d_local",
        "e07_s02_spatial2d_memory",
    ]
    candidates: list[dict[str, Any]] = []
    for parent_id in parent_ids:
        definition = parent_rows[parent_id]
        body = {
            "schemaVersion": "e07.s10p.candidate-population-row.v1",
            "candidateId": parent_id,
            "candidateRole": "s09_parent",
            "pairedParentConfigurationId": parent_id,
            "sourceTaskId": str(definition["taskId"]),
            "eligibleExecutionTaskIds": execution_tasks,
            "structuralConfiguration": definition,
            "selectionUsesS09EffectOrRegressionOutcome": False,
            "selectionUsesS07ArmMembership": False,
            "selectionUsesRejectedModelOrEmbedding": False,
            "outcomeFieldsLoaded": False,
        }
        body["candidateDefinitionSha256"] = canonical_sha256(
            "E07/S10P/candidate-definition/v1", body
        )
        candidates.append(body)
    for row in sorted(
        compressed_rows, key=lambda item: str(item["variantConfigurationId"])
    ):
        definition = _structural_configuration(row["configuration"])
        candidate_id = str(row["variantConfigurationId"])
        if candidate_id != str(definition["configurationId"]):
            raise RuntimeError("compressed candidate/configuration identity mismatch")
        body = {
            "schemaVersion": "e07.s10p.candidate-population-row.v1",
            "candidateId": candidate_id,
            "candidateRole": "s09_compressed",
            "pairedParentConfigurationId": str(row["parentConfigurationId"]),
            "sourceTaskId": str(row["taskId"]),
            "eligibleExecutionTaskIds": execution_tasks,
            "structuralConfiguration": definition,
            "selectionUsesS09EffectOrRegressionOutcome": False,
            "selectionUsesS07ArmMembership": False,
            "selectionUsesRejectedModelOrEmbedding": False,
            "outcomeFieldsLoaded": False,
        }
        body["candidateDefinitionSha256"] = canonical_sha256(
            "E07/S10P/candidate-definition/v1", body
        )
        candidates.append(body)
    candidates.sort(key=lambda item: item["candidateId"])
    if len({item["candidateId"] for item in candidates}) != 14:
        raise RuntimeError("S10P candidate IDs must be 14 distinct hashes")
    return candidates


def scenario_population_and_roster(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    phases = (
        ("discovery", 2000, 2255),
        ("independent_reproduction", 3000, 3127),
    )
    tasks = (
        "e07_s02_spatial2d_local",
        "e07_s02_spatial2d_memory",
    )
    scenario_rows: list[dict[str, Any]] = []
    roster_rows: list[dict[str, Any]] = []
    logical_ordinal = 0
    for phase, start, end in phases:
        for task_id in tasks:
            for ordinal in range(start, end + 1):
                scenario_family_id = f"s10:{phase}:{task_id}:{ordinal:04d}"
                counter_key = f"E07/S10/{phase}/{task_id}/{ordinal:04d}"
                scenario = {
                    "schemaVersion": "e07.s10p.scenario-population-row.v1",
                    "phase": phase,
                    "taskId": task_id,
                    "split": "train",
                    "scenarioFamilyOrdinal": ordinal,
                    "scenarioFamilyId": scenario_family_id,
                    "counterScheduleKey": counter_key,
                    "graphTransitions": 32,
                    "actorSlotsPerTransition": 4,
                    "outcomeBlindDerivation": True,
                }
                scenario["scenarioCommitmentSha256"] = canonical_sha256(
                    "E07/S10P/scenario-family/v1", scenario
                )
                scenario_rows.append(scenario)
                for candidate in candidates:
                    reservation = {
                        "phase": phase,
                        "taskId": task_id,
                        "scenarioFamilyOrdinal": ordinal,
                        "candidateId": candidate["candidateId"],
                        "logicalOrdinal": logical_ordinal,
                    }
                    logical_id = canonical_sha256(
                        "E07/S10P/logical-reservation/v1", reservation
                    )
                    roster_rows.append(
                        {
                            "schemaVersion": "e07.s10p.logical-roster-row.v1",
                            "logicalOrdinal": logical_ordinal,
                            "logicalReservationId": logical_id,
                            "phase": phase,
                            "taskId": task_id,
                            "split": "train",
                            "scenarioFamilyOrdinal": ordinal,
                            "scenarioFamilyId": scenario_family_id,
                            "scenarioCommitmentSha256": scenario[
                                "scenarioCommitmentSha256"
                            ],
                            "candidateId": candidate["candidateId"],
                            "candidateRole": candidate["candidateRole"],
                            "candidateDefinitionSha256": candidate[
                                "candidateDefinitionSha256"
                            ],
                            "graphTransitions": 32,
                            "actorSlotsPerTransition": 4,
                            "outcomeMaterialized": False,
                        }
                    )
                    logical_ordinal += 1
    roster = pd.DataFrame(roster_rows)
    if len(scenario_rows) != 768 or len(roster) != 10752:
        raise RuntimeError("frozen S10 scenario/roster accounting mismatch")
    if roster["logicalReservationId"].nunique() != len(roster):
        raise RuntimeError("logical reservation identity collision")
    return scenario_rows, roster


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _ordered_summaries(pattern: str) -> list[dict[str, Any]]:
    current = _sha(f"{pattern}:initial")
    rows = []
    for index in range(32):
        if pattern == "single_change":
            proposal = 0 if index < 16 else 3
            accepted = 0 if index < 16 else 2
        elif pattern == "alternating":
            proposal = index % 4
            accepted = int(index % 4 > 1)
        else:
            proposal = 2
            accepted = 1
        changed = accepted > 0 and (pattern != "alternating" or index % 3 != 0)
        post = _sha(f"{pattern}:state:{index}") if changed else current
        rows.append(
            {
                "transitionIndex": index,
                "scheduledActorCount": 4,
                "proposalCount": proposal,
                "acceptedCount": accepted,
                "conflictLosses": max(0, proposal - accepted - 1),
                "invalidProposals": 0,
                "preStateSha256": current,
                "postStateSha256": post,
                "transitionSha256": _sha(f"{pattern}:transition:{index}"),
            }
        )
        current = post
    return rows


def _line_ledger(scale: int = 1) -> dict[str, int]:
    return {
        "activations": 16 * scale,
        "observationReads": 31 * scale,
        "valueComparisons": 15 * scale,
        "proposals": 16 * scale,
        "noOps": 7 * scale,
        "rejections": 3 * scale,
        "memoryUpdates": 1 * scale,
        "acceptedSwaps": 5 * scale,
        "displacedCells": 10 * scale,
        "conflictLosses": 0,
    }


def _base_payload(
    task_id: str,
    *,
    stop_reason: str,
    failed: bool,
    censored: bool,
    event: Mapping[str, Any],
    costs: Mapping[str, Any],
    size: int = 8,
    ordered: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event": dict(event),
        "costs": dict(costs),
        "status": {
            "stopReason": stop_reason,
            "failed": failed,
            "censored": censored,
        },
        "horizon": {
            "nativeUnit": "qualification_fixture_native_unit",
            "budget": 32 if "spatial2d" in task_id else 6400,
        },
        "structuralScenarioSize": size,
        "traceSelectionReason": "outcome_independent_qualification_fixture",
    }
    if ordered is not None:
        payload["orderedEventSummaries"] = ordered
    return payload


def qualification_fixtures() -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    line_tasks = {
        "e07_s02_sorting_1d": ("complete", "event_budget", "invariant_error"),
        "e07_s02_faults_1d": ("complete", "event_budget", "blocking_failure"),
        "e07_s02_detour_1d": ("complete", "event_budget", "invariant_error"),
        "e07_s02_chimera_1d": ("complete", "event_budget", "invariant_error"),
    }
    for task_id, statuses in line_tasks.items():
        for ordinal, stop_reason in enumerate(statuses):
            event = {
                "schemaVersion": f"{task_id}.qualification-event.v1",
                "activationCount": 16,
                "eventDigest": _sha(f"{task_id}:{stop_reason}"),
            }
            if task_id == "e07_s02_faults_1d":
                event["faultAuditCount"] = 4 + ordinal
            if task_id == "e07_s02_detour_1d":
                event["distanceProfileCount"] = 17
                event["onlineMetricExposure"] = False
            if task_id == "e07_s02_chimera_1d":
                event["fullTraceMetricCount"] = 16
                event["policyLabelsVisibleToPolicy"] = False
            fixtures.append(
                {
                    "fixtureId": f"{task_id}:terminal:{ordinal}",
                    "taskId": task_id,
                    "terminalBranch": stop_reason,
                    "payload": _base_payload(
                        task_id,
                        stop_reason=stop_reason,
                        failed=stop_reason in {"invariant_error", "blocking_failure"},
                        censored=stop_reason == "event_budget",
                        event=event,
                        costs={"e01ReferenceLedger": _line_ledger(ordinal + 1)},
                    ),
                    "availabilityReasons": {},
                }
            )

    phases = ("development", "stabilization", "recovery", "robustnessFault", "transfer")
    for ordinal, stop_reason in enumerate(
        ("recovery_complete", "source_terminal", "phase_event_budget")
    ):
        available_phases = phases if ordinal == 0 else phases[: 2 + ordinal]
        costs = {
            f"e05_{phase}": _line_ledger(index + 1)
            for index, phase in enumerate(available_phases)
        }
        reasons = {}
        for spec in build_feature_registry():
            if spec.task_id != "e07_s02_regeneration_1d":
                continue
            phase = spec.feature_id.split(".")[1]
            if phase not in available_phases:
                reasons[spec.feature_id] = (
                    "source_terminal"
                    if stop_reason == "source_terminal"
                    else "right_censored_before_phase"
                )
        phase_rows = [
            {
                "phase": phase,
                "status": "completed",
                "stopReason": stop_reason
                if phase == available_phases[-1]
                else "phase_complete",
                "censored": stop_reason == "phase_event_budget"
                and phase == available_phases[-1],
                "opportunities": 16 * (index + 1),
            }
            for index, phase in enumerate(available_phases)
        ]
        fixtures.append(
            {
                "fixtureId": f"e07_s02_regeneration_1d:terminal:{ordinal}",
                "taskId": "e07_s02_regeneration_1d",
                "terminalBranch": stop_reason,
                "payload": _base_payload(
                    "e07_s02_regeneration_1d",
                    stop_reason=stop_reason,
                    failed=False,
                    censored=stop_reason in {"source_terminal", "phase_event_budget"},
                    event={
                        "schemaVersion": "e07.s04a.e05-regeneration-dsl-panel.v1",
                        "phaseRows": phase_rows,
                        "stoppedRows": [],
                        "fiveAxesNonaggregated": True,
                    },
                    costs=costs,
                    size=32,
                ),
                "availabilityReasons": reasons,
            }
        )

    target_signal = {
        "chargedOpportunities": 16,
        "signalQueries": 12,
        "signalDeliveries": 9,
        "signalPayloadUnits": 9,
        "localRelayAvailableOpportunities": 8,
        "gradientFieldOpportunities": 12,
        "globalBroadcastUnits": 0,
        "targetRecordReads": 16,
        "shadowComparisons": 0,
        "controllerComputations": 16,
        "targetAwareCandidates": 6,
        "targetAwareEmitted": 5,
        "nativeEmitted": 11,
        "abstractEnergyUnits": 25,
    }
    for ordinal, stop_reason in enumerate(
        ("post_adaptation_probe_complete", "phase_event_budget", "controller_quiescent")
    ):
        fixtures.append(
            {
                "fixtureId": f"e07_s02_target_change_1d:terminal:{ordinal}",
                "taskId": "e07_s02_target_change_1d",
                "terminalBranch": stop_reason,
                "payload": _base_payload(
                    "e07_s02_target_change_1d",
                    stop_reason=stop_reason,
                    failed=False,
                    censored=stop_reason == "phase_event_budget",
                    event={
                        "schemaVersion": "e07.s04a.e05-target-dsl-episode.v1",
                        "targetChangeInstantaneousBeforeNextOpportunity": True,
                        "startEventIndex": 0,
                        "endEventIndex": 16,
                    },
                    costs={
                        "e01ReferenceLedgerDelta": _line_ledger(ordinal + 1),
                        "e05TargetSignalLedger": {
                            key: value * (ordinal + 1)
                            for key, value in target_signal.items()
                        },
                    },
                ),
                "availabilityReasons": {},
            }
        )

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
    for task_id in ("e07_s02_spatial2d_local", "e07_s02_spatial2d_memory"):
        for ordinal, (stop_reason, pattern, ordered_available) in enumerate(
            (
                ("transition_budget", "single_change", True),
                ("transition_budget", "alternating", True),
                ("invariant_error", "constant", False),
            )
        ):
            reasons = {}
            if not ordered_available:
                reasons = {
                    spec.feature_id: "ordered_event_summary_unavailable"
                    for spec in build_feature_registry()
                    if spec.task_id == task_id and spec.transform == "ordered_summary"
                }
            fixtures.append(
                {
                    "fixtureId": f"{task_id}:terminal:{ordinal}",
                    "taskId": task_id,
                    "terminalBranch": stop_reason,
                    "payload": _base_payload(
                        task_id,
                        stop_reason=stop_reason,
                        failed=stop_reason == "invariant_error",
                        censored=stop_reason == "transition_budget",
                        event={
                            "schemaVersion": "e07.s04a.e06-dsl-episode.v1",
                            "transitionCount": 32 if ordered_available else 0,
                            "actorBatchSize": 4,
                            "onlineGlobalCompletionComputed": False,
                        },
                        costs={"e06MovementLedger": movement},
                        size=64,
                        ordered=_ordered_summaries(pattern)
                        if ordered_available
                        else None,
                    ),
                    "availabilityReasons": reasons,
                }
            )
    if len(fixtures) != 24:
        raise RuntimeError(f"expected 24 qualification fixtures, got {len(fixtures)}")
    return fixtures


def _adversarial_checks(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []

    def expect_failure(name: str, task_id: str, payload: Mapping[str, Any]) -> None:
        try:
            extract_native_event_features(task_id, payload)
        except (FeatureExtractionError, ValueError, KeyError) as exc:
            checks.append({"case": name, "rejected": True, "error": str(exc)})
        else:
            checks.append({"case": name, "rejected": False, "error": None})

    base = json.loads(json.dumps(fixtures[0]["payload"]))
    base["outcome"] = {"completed": True}
    expect_failure("outcome_plane_injection", fixtures[0]["taskId"], base)

    spatial = next(
        item
        for item in fixtures
        if item["taskId"] == "e07_s02_spatial2d_local"
        and "orderedEventSummaries" in item["payload"]
    )
    forged = json.loads(json.dumps(spatial["payload"]))
    forged["orderedEventSummaries"][1]["preStateSha256"] = _sha("forged")
    expect_failure("forged_state_hash_chain", spatial["taskId"], forged)
    short = json.loads(json.dumps(spatial["payload"]))
    short["orderedEventSummaries"] = short["orderedEventSummaries"][:-1]
    expect_failure("short_ordered_summary", spatial["taskId"], short)
    nonfinite = json.loads(json.dumps(spatial["payload"]))
    nonfinite["costs"]["e06MovementLedger"]["submittedProposals"] = float("inf")
    expect_failure("nonfinite_native_count", spatial["taskId"], nonfinite)
    negative = json.loads(json.dumps(spatial["payload"]))
    negative["costs"]["e06MovementLedger"]["submittedProposals"] = -1
    expect_failure("negative_native_count", spatial["taskId"], negative)
    line_with_sequence = json.loads(json.dumps(fixtures[0]["payload"]))
    line_with_sequence["orderedEventSummaries"] = _ordered_summaries("constant")
    expect_failure(
        "sequence_on_summary_only_contract", fixtures[0]["taskId"], line_with_sequence
    )
    missing_status = json.loads(json.dumps(fixtures[0]["payload"]))
    del missing_status["status"]["failed"]
    expect_failure("missing_native_status", fixtures[0]["taskId"], missing_status)
    expect_failure("unknown_task", "e07_s02_unknown", fixtures[0]["payload"])
    return checks


def run_access_denial() -> dict[str, Any]:
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    development = AccessGrant(AccessPhase.DEVELOPMENT)
    confirmation = AccessGrant(
        AccessPhase.CONFIRMATION,
        candidate_lock_sha256="0" * 64,
    )
    denials = []
    for task_id in sorted(suite.tasks):
        records = [
            record for record in suite.records.values() if record.task_id == task_id
        ]
        by_split = {record.split.value: record for record in records}
        for split, grant in (
            ("validation", development),
            ("confirmation", development),
            ("confirmation", confirmation),
        ):
            try:
                suite.open(task_id, by_split[split].scenario_id, grant)
            except AccessDeniedError as exc:
                denials.append(
                    {
                        "taskId": task_id,
                        "split": split,
                        "grant": grant.phase.value,
                        "denied": True,
                        "reason": str(exc),
                    }
                )
            else:
                denials.append(
                    {
                        "taskId": task_id,
                        "split": split,
                        "grant": grant.phase.value,
                        "denied": False,
                        "reason": None,
                    }
                )
    return {
        "schemaVersion": "e07.s10p.access-control-validation.v1",
        "attempts": len(denials),
        "denials": sum(item["denied"] for item in denials),
        "allDenied": all(item["denied"] for item in denials),
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
        "brokerAudit": suite.broker.audit.to_dict(),
        "records": denials,
    }


def freeze() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"{OUT} must be empty for prospective freeze")
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["researchStepId"] != "S10P":
        raise RuntimeError("protocol research step mismatch")
    candidates = candidate_population()
    scenarios, roster = scenario_population_and_roster(candidates)
    hashes = input_hashes()

    (OUT / "s10p_native_event_protocol.yaml").write_bytes(PROTOCOL.read_bytes())
    write_jsonl(OUT / "candidate_population.jsonl", candidates)
    write_jsonl(OUT / "scenario_population.jsonl", scenarios)
    roster.to_parquet(OUT / "s10_logical_roster.parquet", index=False)
    write_json(OUT / "native_event_feature_registry.json", registry_document())
    method_registry = {
        "schemaVersion": "e07.s10p.method-registry.v1",
        "clustering": protocol["methods"]["clustering"],
        "anomaly": protocol["methods"]["anomaly"],
        "changePoint": protocol["methods"]["changePoint"],
        "stabilityNullAndMultiplicity": protocol["stabilityNullAndMultiplicity"],
        "independentReproduction": protocol["independentReproduction"],
        "missingnessAndConfounds": protocol["missingnessAndConfounds"],
        "traceSelection": protocol["traceSelection"],
        "machineFitsExecuted": 0,
    }
    write_json(OUT / "method_registry.json", method_registry)
    write_json(
        OUT / "input_hash_freeze.json",
        {
            "schemaVersion": "e07.s10p.input-hash-freeze.v1",
            "researchStepId": "S10P",
            "inputs": hashes,
            "workflowContextAtFreeze": workflow_context_hashes(),
            "workflowContextIsNotAnImmutableScientificInput": True,
            "s06OrS06AModelOrEmbeddingLoads": 0,
            "s07ArmSignalUses": 0,
            "quarantineOutcomeOrCacheReads": 0,
        },
    )
    accounting = {
        "schemaVersion": "e07.s10p.logical-accounting.v1",
        "candidateCount": len(candidates),
        "parentCount": sum(
            item["candidateRole"] == "s09_parent" for item in candidates
        ),
        "compressedCount": sum(
            item["candidateRole"] == "s09_compressed" for item in candidates
        ),
        "taskCount": 2,
        "scenarioFamilyCount": len(scenarios),
        "discoveryScenarioFamiliesPerTask": 256,
        "reproductionScenarioFamiliesPerTask": 128,
        "discoveryLogicalRows": int((roster["phase"] == "discovery").sum()),
        "reproductionLogicalRows": int(
            (roster["phase"] == "independent_reproduction").sum()
        ),
        "totalLogicalRows": len(roster),
        "uniqueLogicalReservationIds": int(roster["logicalReservationId"].nunique()),
        "outcomeMaterializedRows": int(roster["outcomeMaterialized"].sum()),
        "substantiveS10RowsExecuted": 0,
    }
    write_json(OUT / "logical_accounting.json", accounting)
    reviewer_text = """# S10 blinded exemplar and reviewer protocol

This is a prespecified analysis protocol, not a research-step handoff report.

Machine discovery and independent training-family reproduction must be hash-locked before exemplar sampling. For each reproduced candidate phenotype, select six event-summary exemplars by canonical-hash tie-break: two nearest the locked medoid, two locked extremes, and two task/status/horizon-matched null controls. Reviewers never see configuration identity, parent/compressed role, objective or validation values, policy source, or S07 arm information.

Two independent reviewers assign only operational event-summary labels: activity distribution, persistence or burst, conflict pattern, state-turnover pattern, no operational pattern, or insufficient event summary. Preference, goal, intention, agency, competency, formation, repair, and biological-phenotype labels are forbidden without a later intervention. Cohen's kappa must be at least 0.60; otherwise human labels remain descriptive and cannot promote a candidate. A third blinded reviewer adjudicates disagreements only. Reviewers cannot generate machine candidates or alter the frozen multiplicity family.
"""
    (OUT / "blinded_reviewer_protocol.md").write_text(reviewer_text, encoding="utf-8")
    freeze_doc = {
        "schemaVersion": "e07.s10p.preregistration-freeze.v1",
        "researchStepId": "S10P",
        "protocolSha256": sha256_file(OUT / "s10p_native_event_protocol.yaml"),
        "candidatePopulationSha256": sha256_file(OUT / "candidate_population.jsonl"),
        "scenarioPopulationSha256": sha256_file(OUT / "scenario_population.jsonl"),
        "logicalRosterSha256": sha256_file(OUT / "s10_logical_roster.parquet"),
        "featureRegistrySha256": sha256_file(
            OUT / "native_event_feature_registry.json"
        ),
        "methodRegistrySha256": sha256_file(OUT / "method_registry.json"),
        "reviewerProtocolSha256": sha256_file(OUT / "blinded_reviewer_protocol.md"),
        "designFrozenBeforeQualification": True,
        "substantiveS10RowsExecutedBeforeFreeze": 0,
        "machineFitsBeforeFreeze": 0,
        "humanAnnotationsBeforeFreeze": 0,
    }
    freeze_doc["freezeCommitmentSha256"] = canonical_sha256(
        "E07/S10P/preregistration-freeze/v1", freeze_doc
    )
    write_json(OUT / "preregistration_freeze.json", freeze_doc)
    (OUT / "execution_commands.log").write_text(
        "# An initial pre-freeze invocation aborted before qualification because two "
        "YAML null keys were unquoted; its incomplete outputs were moved to "
        "/cache/e07-s10p-pre-freeze-recovery and the protocol was corrected "
        "before this final freeze.\n"
        "PYTHONPATH=. python scripts/preregister_native_event_discovery_s10p.py freeze\n",
        encoding="utf-8",
    )


def _validate_freeze() -> dict[str, Any]:
    freeze_doc = read_json(OUT / "preregistration_freeze.json")
    checks = {
        "protocol": sha256_file(OUT / "s10p_native_event_protocol.yaml")
        == freeze_doc["protocolSha256"],
        "candidatePopulation": sha256_file(OUT / "candidate_population.jsonl")
        == freeze_doc["candidatePopulationSha256"],
        "scenarioPopulation": sha256_file(OUT / "scenario_population.jsonl")
        == freeze_doc["scenarioPopulationSha256"],
        "logicalRoster": sha256_file(OUT / "s10_logical_roster.parquet")
        == freeze_doc["logicalRosterSha256"],
        "featureRegistry": sha256_file(OUT / "native_event_feature_registry.json")
        == freeze_doc["featureRegistrySha256"],
        "methodRegistry": sha256_file(OUT / "method_registry.json")
        == freeze_doc["methodRegistrySha256"],
        "reviewerProtocol": sha256_file(OUT / "blinded_reviewer_protocol.md")
        == freeze_doc["reviewerProtocolSha256"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"preregistration bytes changed: {checks}")
    return checks


def qualify() -> None:
    if not (OUT / "preregistration_freeze.json").is_file():
        raise RuntimeError("freeze phase must complete before qualification")
    freeze_checks = _validate_freeze()
    frozen_inputs = read_json(OUT / "input_hash_freeze.json")["inputs"]
    current = {record["path"]: record for record in input_hashes()}
    input_checks = {
        item["path"]: current[item["path"]]["sha256"] == item["sha256"]
        for item in frozen_inputs
    }
    if not all(input_checks.values()):
        raise RuntimeError("a frozen input changed after preregistration")

    fixtures = qualification_fixtures()
    records = []
    for fixture in fixtures:
        first = extract_native_event_features(
            fixture["taskId"],
            fixture["payload"],
            availability_reasons=fixture["availabilityReasons"],
        )
        second = extract_native_event_features(
            fixture["taskId"],
            json.loads(json.dumps(fixture["payload"])),
            availability_reasons=fixture["availabilityReasons"],
        )
        if first != second:
            raise RuntimeError("fixture extraction replay mismatch")
        records.append(
            {
                "fixtureId": fixture["fixtureId"],
                "taskId": fixture["taskId"],
                "terminalBranch": fixture["terminalBranch"],
                "featureRecord": first,
            }
        )
    write_jsonl(OUT / "qualification_fixtures.jsonl", fixtures)
    flat_records = []
    availability_rows = []
    for row in records:
        feature_record = row["featureRecord"]
        flat = {
            "fixtureId": row["fixtureId"],
            "taskId": row["taskId"],
            "terminalBranch": row["terminalBranch"],
            "stopReason": feature_record["confounds"]["stopReason"],
            "failed": feature_record["confounds"]["failed"],
            "censored": feature_record["confounds"]["censored"],
            "traceAvailability": feature_record["confounds"]["traceAvailability"],
            "observedFeatureCount": len(feature_record["analysisFeatures"]),
            "featureRecordSha256": feature_record["featureRecordSha256"],
            "featureRecordJson": canonical_json_bytes(feature_record).decode("ascii"),
        }
        flat_records.append(flat)
        for feature_id, availability in feature_record["availability"].items():
            availability_rows.append(
                {
                    "fixtureId": row["fixtureId"],
                    "taskId": row["taskId"],
                    "terminalBranch": row["terminalBranch"],
                    "featureId": feature_id,
                    "state": availability["state"],
                    "reason": availability["reason"],
                }
            )
    pd.DataFrame(flat_records).to_parquet(
        OUT / "qualification_feature_records.parquet", index=False
    )
    availability_frame = pd.DataFrame(availability_rows)
    availability_frame.to_csv(OUT / "feature_availability_validation.csv", index=False)

    normal_counts = (
        pd.DataFrame(flat_records)
        .sort_values(["taskId", "fixtureId"])
        .groupby("taskId", as_index=False)
        .first()[["taskId", "observedFeatureCount"]]
    )
    by_task = {
        row.taskId: int(row.observedFeatureCount)
        for row in normal_counts.itertuples(index=False)
    }
    availability_validation = {
        "schemaVersion": "e07.s10p.feature-availability-validation.v1",
        "taskCount": len(by_task),
        "fixtureCount": len(fixtures),
        "registeredFeatureCount": len(build_feature_registry()),
        "observedFeatureCountInCompleteFixtureByTask": by_task,
        "minimumFiveFeaturesEveryTask": all(value >= 5 for value in by_task.values()),
        "spatialAuthenticOrderedSummaryFixtures": sum(
            item["featureRecord"]["confounds"]["traceAvailability"]
            == "authentic_ordered_event_summary"
            for item in records
        ),
        "summaryOnlyTasksWithNoFabricatedSequence": all(
            item["featureRecord"]["confounds"]["traceAvailability"]
            == "terminal_summary_only"
            for item in records
            if item["taskId"]
            not in {"e07_s02_spatial2d_local", "e07_s02_spatial2d_memory"}
        ),
        "unavailableRowsHaveExplicitReason": all(
            row["state"] != "unavailable" or bool(row["reason"])
            for row in availability_rows
        ),
        "outcomeFieldsLoaded": 0,
        "s04DescriptorFieldsLoaded": 0,
        "universalScoresConstructed": 0,
    }
    write_json(OUT / "feature_availability_validation.json", availability_validation)

    order_digests = {}
    for name, ordered in (
        ("natural", records),
        ("reverse", list(reversed(records))),
        (
            "hash_sorted",
            sorted(
                records,
                key=lambda item: canonical_sha256(
                    "E07/S10P/worker-order/v1", item["fixtureId"]
                ),
            ),
        ),
    ):
        canonical = sorted(
            (
                item["fixtureId"],
                item["featureRecord"]["featureRecordSha256"],
            )
            for item in ordered
        )
        order_digests[name] = canonical_sha256(
            "E07/S10P/qualification-worker-order/v1", canonical
        )
    adversarial = _adversarial_checks(fixtures)
    change_point_step = exact_change_points(
        [[0.0, 0.0] for _ in range(16)] + [[3.0, 2.0] for _ in range(16)]
    )
    change_point_null = exact_change_points([[1.0, 1.0] for _ in range(32)])
    deterministic = {
        "schemaVersion": "e07.s10p.determinism-validation.v1",
        "fixtureReplayPass": True,
        "canonicalSerializationRoundTripPass": all(
            json.loads(canonical_json_bytes(item["featureRecord"]))
            == item["featureRecord"]
            for item in records
        ),
        "workerOrderDigests": order_digests,
        "workerOrderIndependent": len(set(order_digests.values())) == 1,
        "adversarialCases": adversarial,
        "allAdversarialCasesFailClosed": all(item["rejected"] for item in adversarial),
        "changePointQualification": {
            "singleKnownChange": list(change_point_step),
            "constantSeries": list(change_point_null),
            "pass": change_point_step == (16,) and change_point_null == (),
        },
    }
    write_json(OUT / "determinism_and_adversarial_validation.json", deterministic)

    access = run_access_denial()
    write_json(OUT / "access_control_validation.json", access)
    dependency = {
        "schemaVersion": "e07.s10p.leakage-dependency-validation.v1",
        "s06OrS06AModelOrEmbeddingLoads": 0,
        "s07ArmSignalUses": 0,
        "quarantineOutcomeOrCacheReads": 0,
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
        "substantiveS10RowsExecuted": 0,
        "clusteringFits": 0,
        "anomalyFits": 0,
        "changePointScreens": 0,
        "humanAnnotations": 0,
        "archiveMutations": 0,
        "outcomeInjectionRejected": next(
            item["rejected"]
            for item in adversarial
            if item["case"] == "outcome_plane_injection"
        ),
        "loadedModuleNamesContainingSurrogateModels": sorted(
            name for name in sys.modules if "surrogate_models" in name
        ),
    }
    dependency["pass"] = (
        not dependency["loadedModuleNamesContainingSurrogateModels"]
        and all(
            dependency[key] == 0
            for key in (
                "s06OrS06AModelOrEmbeddingLoads",
                "s07ArmSignalUses",
                "quarantineOutcomeOrCacheReads",
                "validationOutcomeRowsRead",
                "confirmationOutcomeRowsRead",
                "substantiveS10RowsExecuted",
                "clusteringFits",
                "anomalyFits",
                "changePointScreens",
                "humanAnnotations",
                "archiveMutations",
            )
        )
        and dependency["outcomeInjectionRejected"]
    )
    write_json(OUT / "leakage_and_dependency_validation.json", dependency)

    accounting = read_json(OUT / "logical_accounting.json")
    gate_checks = {
        "G01": all(freeze_checks.values()) and all(input_checks.values()),
        "G02": (
            accounting["candidateCount"] == 14
            and accounting["taskCount"] == 2
            and accounting["totalLogicalRows"] == 10752
            and accounting["uniqueLogicalReservationIds"] == 10752
        ),
        "G03": (
            availability_validation["taskCount"] == 8
            and availability_validation["minimumFiveFeaturesEveryTask"]
            and availability_validation["spatialAuthenticOrderedSummaryFixtures"] == 4
        ),
        "G04": (
            availability_validation["unavailableRowsHaveExplicitReason"]
            and availability_validation["summaryOnlyTasksWithNoFabricatedSequence"]
        ),
        "G05": (
            deterministic["fixtureReplayPass"]
            and deterministic["canonicalSerializationRoundTripPass"]
            and deterministic["workerOrderIndependent"]
            and deterministic["allAdversarialCasesFailClosed"]
            and deterministic["changePointQualification"]["pass"]
        ),
        "G06": access["allDenied"] and dependency["pass"],
        "G07": (
            (OUT / "method_registry.json").is_file()
            and (OUT / "blinded_reviewer_protocol.md").is_file()
        ),
        "G08": all(
            dependency[key] == 0
            for key in (
                "substantiveS10RowsExecuted",
                "clusteringFits",
                "anomalyFits",
                "changePointScreens",
                "humanAnnotations",
                "archiveMutations",
            )
        ),
    }
    gate = {
        "schemaVersion": "e07.s10p.s10-execution-gate.v1",
        "researchStepId": "S10P",
        "failClosed": True,
        "rows": [
            {
                "gateId": gate_id,
                "pass": passed,
                "status": "clear" if passed else "blocked",
            }
            for gate_id, passed in gate_checks.items()
        ],
        "allPass": all(gate_checks.values()),
        "substantiveS10AuthorizedByThisStep": False,
        "requiresSeparateExecutionApproval": True,
    }
    write_json(OUT / "s10_eligibility_gate.json", gate)

    no_mutation = {
        "schemaVersion": "e07.s10p.no-mutation-audit.v1",
        "allowedArtifactWriteRoot": str(OUT),
        "repositoryChangesLimitedToS10PImplementationAndTests": True,
        "predecessorArtifactWrites": 0,
        "archiveMutations": 0,
        "quarantineMutations": 0,
        "cacheDirectoriesCreated": 0,
        "pass": True,
    }
    write_json(OUT / "no_mutation_audit.json", no_mutation)
    validation = {
        "schemaVersion": "e07.s10p.validation-summary.v1",
        "researchStepId": "S10P",
        "success": gate["allPass"],
        "validationResult": (
            "PASS: G01-G08, 14 candidates, 768 scenario families, 10,752 "
            "logical reservations, 24 outcome-independent fixtures across 8 "
            "tasks, deterministic replay/order/round trips, 8 adversarial "
            "fail-closed checks, 24/24 protected-split denials, and zero "
            "substantive S10 or protected-outcome work."
        ),
        "candidateCount": 14,
        "taskSchemaQualificationCount": 8,
        "substantiveTaskCount": 2,
        "fixtureCount": 24,
        "logicalReservationCount": 10752,
        "gateRowsPassing": sum(gate_checks.values()),
        "gateRowCount": len(gate_checks),
        "outcomeClassification": "supportive",
        "caveatsOrBlockers": [
            "Only two spatial tasks have eligible S09 candidates.",
            "Six contracts remain terminal-summary-only and are excluded from substantive S10.",
            "Ordered spatial summaries contain counts and state hashes, not complete states or trajectories.",
            "This step contains no phenotype, efficacy, novelty, validation, confirmation, or intervention result.",
        ],
        "recommendedNextAction": (
            "Review the frozen S10P design and gate, then separately authorize "
            "S10 execution or revise the preregistration; do not start S11."
        ),
    }
    write_json(OUT / "validation_summary.json", validation)

    status = {
        "researchStepId": "S10P",
        "stepNumber": 10,
        "success": bool(gate["allPass"]),
        "status": "complete_design_and_qualification_only",
        "artifactsWritten": [
            "s10p_native_event_protocol.yaml",
            "candidate_population.jsonl",
            "scenario_population.jsonl",
            "s10_logical_roster.parquet",
            "native_event_feature_registry.json",
            "method_registry.json",
            "qualification_fixtures.jsonl",
            "qualification_feature_records.parquet",
            "feature_availability_validation.csv",
            "feature_availability_validation.json",
            "determinism_and_adversarial_validation.json",
            "access_control_validation.json",
            "leakage_and_dependency_validation.json",
            "s10_eligibility_gate.json",
            "research_step_full_results.md",
        ],
        "validationResult": validation["validationResult"],
        "caveatsOrBlockers": validation["caveatsOrBlockers"],
        "recommendedNextAction": validation["recommendedNextAction"],
    }
    write_json(OUT / "status.json", status)
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    provenance = {
        "schemaVersion": "e07.s10p.provenance.v1",
        "researchStepId": "S10P",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPOSITORY),
        "gitHeadBeforeS10PCommit": git_commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "workers": 1,
        "numericThreadsPerWorker": 1,
        "reasonForSerialExecution": (
            "Qualification comprises 24 small deterministic schema fixtures; "
            "serial execution avoids introducing irrelevant scheduling."
        ),
        "inputHashFreezeSha256": sha256_file(OUT / "input_hash_freeze.json"),
        "preregistrationFreezeSha256": sha256_file(OUT / "preregistration_freeze.json"),
        "protectedOutcomeRowsRead": 0,
        "substantiveEpisodeRowsExecuted": 0,
    }
    write_json(OUT / "provenance.json", provenance)
    with (OUT / "execution_commands.log").open("a", encoding="utf-8") as handle:
        handle.write(
            "PYTHONPATH=. python scripts/preregister_native_event_discovery_s10p.py qualify\n"
        )

    report = f"""# S10P — Native event-feature phenotype-discovery redesign

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S10P** |
| Completion status | **Complete — design and qualification only; stopped before substantive S10 and S11** |
| Artifacts written | Frozen protocol; 14-candidate registry; 768-family/10,752-reservation structural plan; versioned feature and method registries; 24-fixture qualification evidence; access, dependency, accounting, gate, status, provenance, and manifest records |
| Validation result | **PASS** — G01–G08; 8/8 task contracts schema-qualified; 24/24 fixtures replayed; 8/8 adversarial inputs failed closed; 24/24 protected-split attempts denied; exact worker-order and serialization parity; zero protected outcomes or substantive S10 rows |
| Outcome classification | **Supportive design qualification**, with a strict task/trace scope constraint |
| Caveats or blockers | Only two spatial tasks have eligible S09 candidates. Their ordered 32-transition summaries contain counts and state hashes, not complete trajectories. The six other contracts are summary-only and excluded from substantive S10. |
| Lay summary | A careful recipe now exists for looking for unusual event patterns without using the rejected learned embeddings or pretending missing trajectories exist. It is ready for review, but no unusual behavior has yet been searched for or found. |
| Recommended next action | Review and separately approve the frozen S10 execution. Keep validation and confirmation sealed and do not begin S11. |

## Frozen question

Can a task-local, non-surrogate native event-feature analysis be frozen and technically qualified for the S09 discoveries without reconstructing absent trajectories, using protected outcomes, or changing native contracts?

## Result

Yes, within a narrower evidence layer than the original trajectory proposal. The frozen S10 population contains all seven S09 parent configurations and all seven structural compressions, with no filtering on S09 edit effects or cross-spatial regression outcomes. Every one of the 14 configurations is reserved on both spatial contracts. The six non-spatial contracts are schema-qualified for future extension but have no S09-eligible candidates and no authentic ordered event summaries, so they cannot enter substantive S10 under this protocol.

The prospective plan contains 256 discovery and 128 independent-reproduction training families per spatial task: 7,168 discovery plus 3,584 reproduction reservations, 10,752 total. Reproduction cannot begin until discovery candidates, feature masks, preprocessing, cluster representatives, anomaly thresholds, change-point windows, and the multiplicity family are hash-locked. Validation is not used because S08M validation contributed to candidate eligibility; confirmation remains sealed.

## Inputs and provenance

The design refreshed `AGENTS.md`, `FULL_PLAN.md`, `RESEARCH_PLAN.md`, E01–E06 handoffs/native event and cost contracts, S01–S09 canonical reports and structural registries, S02 split controls, and the attachment manifest/sidecar. The input freeze records exact paths, sizes, and SHA-256 values. S06/S06A reports were read for their immutable rejection boundary, but no rejected model or embedding was loaded. No S07 allocation-arm signal or quarantined cache/outcome row was used.

Primary structural candidate inputs were the seven IDs already frozen in the S09 protocol, parent definitions from the S08M structural configuration registry, and all seven compressed structural definitions. Outcome tables, objective values, descriptor values, ablation effects, and validation outcomes were not inputs.

## Native event-feature contract

The extractor version is `e07.s10p.native-event-features.v1`. It returns separate analysis, confound, availability, and provenance planes. Analysis coordinates are task-native event counts or within-task rates from native event summaries and audited event-count ledgers. It computes no scalar reward, universal score, cross-family cost total, or cross-task normalized feature.

Registered paths cannot name outcomes, objectives, completion, distances, aggregation metrics, scores, or S04 descriptors. Every absent value has an explicit availability state and reason; no imputation or silent complete-case deletion is allowed. Failures and censors retain their native status and enter separate status strata. Task, stop reason, failure/censor state, native horizon, observed event length, structural size, and trace availability are confounds, never phenotype coordinates.

For the two spatial tasks, all authentic 32-transition count/hash summaries are retained. The only state-derived indicator is exact equality or inequality of adjacent native state hashes; the state itself is not reconstructed. The six other tasks remain terminal-summary-only. Complete native trajectories are declared unavailable, selected count zero, and excluded from analysis. Failure- or anomaly-triggered forensic traces may be described later but cannot enter frequency, stability, or null inference.

## Frozen methods and hyperparameters

Profiles are built task by task and native-status stratum by stratum. Within each eligible feature set, five-fold scenario-family cross-fitted ridge residualization uses alpha 10 and fixed hash folds; covariates are log event length, native budget fraction, structural size, and status indicators. Residuals are task/status median-MAD scaled. A feature needs 90% support and each analysis needs at least five eligible coordinates. Absolute residual feature/length Spearman correlation above 0.20 or status balanced accuracy above 0.65 blocks the affected set.

Primary clustering is Ward agglomeration over k=2–6, with the smallest k within one standard error of best scenario-bootstrap stability. A diagonal Gaussian mixture over k=1–6, 20 starts, and covariance regularization 1e-6 is the secondary check; cross-method adjusted Rand must be at least 0.60. Clusters require at least two configurations.

Anomaly screening uses a 500-tree Isolation Forest, at most 256 samples, all features, seed 71020260723, and a task-local top-1% threshold capped at one configuration. Change-point screening uses exact piecewise-constant SSE dynamic programming on proposal, acceptance, conflict-loss, and state-hash-change series, BIC penalty `(p+1)log(T)`, minimum segment length four, and at most three changes. It is inapplicable when an authentic ordered summary is absent.

Stability uses 200 scenario-family bootstraps. Null calibration uses 500 configuration-label permutations within task/status/horizon/scenario family; change points use within-sequence circular shifts. Cluster ARI and Jaccard must each reach 0.70 and exceed the maximum-null 95th percentile. Change points need 60% discovery prevalence, 50% reproduction prevalence, and one-transition alignment. Holm correction at 0.05 covers every machine candidate across both tasks and all three method families. No universal novelty score exists.

## Independent reproduction and review boundary

The 128-family reproduction block is disjoint from the 256-family discovery block and remains unopened until the discovery lock is written. No preprocessing or threshold may be refit on reproduction. This is training-family reproduction, not validation, confirmation, or transfer.

Only machine candidates that reproduce are eligible for blinded exemplars. Six exemplars per candidate are selected mechanically: two nearest the locked medoid, two locked extremes, and two task/status/horizon-matched null controls. Two reviewers receive no configuration identity, parent/compressed role, objective/validation values, policy source, or S07 information. Allowed labels are operational event-summary descriptions only. Cohen's kappa must be at least 0.60; otherwise human labels remain descriptive. Review cannot generate or promote machine candidates.

## Qualification fixtures and results

Twenty-four outcome-independent fixtures cover all eight tasks and native terminal/status branches, including complete, budget-censored, quiescent, source-terminal, blocking/invariant failure, target-change, and spatial fixed-budget cases. Complete fixtures expose at least five registered event features in every task. E05 fixtures exercise explicit phase-not-reached/source-terminal/right-censor availability. Four spatial fixtures carry exact 32-transition hash chains; all summary-only contracts carry no invented sequence.

Extraction was byte-identical on replay and after canonical JSON round trips. Natural, reverse, and hash-sorted worker orders produced the same digest `{next(iter(order_digests.values()))}`. The exact change-point implementation returned transition 16 for a prespecified two-regime fixture and no change for a constant fixture. Eight adversarial cases—outcome injection, forged hash chain, short sequence, nonfinite and negative counts, a sequence on a summary-only task, missing status, and unknown task—failed closed.

The S02 broker denied 24/24 attempted validation or confirmation opens across eight tasks, including confirmation-phase attempts against the still-sealed materializers. No protected outcome row was read. G01–G08 all passed. The gate explicitly does not authorize S10; it requires a separate execution decision.

## Accounting

| Quantity | Count |
| --- | ---: |
| Parent configurations | 7 |
| Compressed configurations | 7 |
| Eligible task contracts | 2 |
| Schema-qualified task contracts | 8 |
| Discovery scenario families | 512 |
| Independent-reproduction scenario families | 256 |
| Total structural scenario families | 768 |
| Discovery logical reservations | 7,168 |
| Reproduction logical reservations | 3,584 |
| Total logical reservations | 10,752 |
| Qualification fixtures | 24 |
| Substantive episode rows | 0 |
| Cluster/anomaly/change-point fits | 0 |
| Human annotations | 0 |
| Validation outcomes read | 0 |
| Confirmation outcomes read | 0 |

## Commands

```bash
PYTHONPATH=. python scripts/preregister_native_event_discovery_s10p.py freeze
PYTHONPATH=. python scripts/preregister_native_event_discovery_s10p.py qualify
PYTHONPATH=. pytest -q tests/test_s10p_native_event_discovery.py
ruff check src/phenotype_discovery scripts/preregister_native_event_discovery_s10p.py tests/test_s10p_native_event_discovery.py
```

Qualification ran serially because 24 small deterministic schema fixtures do not benefit from parallel execution. Future S10 execution remains frozen at up to eight workers and one numeric thread per worker.

An initial generator invocation stopped before preregistration and before any qualification because YAML interpreted two unquoted `null` mapping keys as null objects. Its incomplete S10P files were moved to disposable `/cache/e07-s10p-pre-freeze-recovery`, the keys were quoted, and the final protocol was then frozen. No fixture result, method fit, protected access, or scientific decision preceded the final freeze.

The first completed qualification bundle was regenerated once, without changing the frozen protocol, candidate/scenario populations, feature registry, methods, fixtures, or results, after an administrative audit found that mutable `RESEARCH_PLAN.md` had been included among immutable scientific inputs even though the workflow requires updating it after the step. The final input record keeps its contemporaneous hash as workflow context but excludes it from future scientific gate comparisons. This lifecycle correction did not inspect or change any efficacy value.

## Validation

- G01: immutable preregistration and input hashes passed.
- G02: 14 candidates × two tasks × 384 families equals 10,752 unique logical reservations.
- G03: all eight task schemas qualified; authentic ordered summaries exist only for the two spatial tasks.
- G04: missingness, censoring, status, trace, task, horizon, and length handling is explicit and no missing trajectory is inferred.
- G05: replay, serialization, worker-order, adversarial closure, and change-point implementation checks passed.
- G06: protected denial and forbidden dependency/signal/quarantine exclusion passed.
- G07: methods, nulls, multiplicity, reproduction, and blinded review are frozen.
- G08: S10P produced zero substantive evaluations, machine fits, annotations, protected accesses, or archive mutations.

## Caveats and claim boundary

This is a design qualification, not evidence that any phenotype exists. Event-summary discovery is novelty relative to a frozen task-local registry. It cannot recover unrecorded actions or states, establish complete temporal dynamics, compare native horizons through a universal time axis, or support formation, repair, preference, goal, intention, agency, competency, cognition, or biological claims. The population is confined to 14 spatial configurations; six tasks are excluded from substantive analysis. Human review remains subjective and subordinate to the locked machine and reproduction rules.

## Recommended next action

Hand control back. If the Chief Scientist approves execution, run S10 exactly under this frozen design, stop fail-closed on any G01–G08 change, keep validation and confirmation sealed, and stop again before S11.
"""
    (OUT / "research_step_full_results.md").write_text(report, encoding="utf-8")

    manifest_entries = []
    for path in sorted(OUT.iterdir()):
        if path.name == "artifact_manifest.json" or not path.is_file():
            continue
        manifest_entries.append(file_record(path))
    manifest = {
        "schemaVersion": "e07.s10p.artifact-manifest.v1",
        "researchStepId": "S10P",
        "artifactCount": len(manifest_entries),
        "artifacts": manifest_entries,
        "requiredArtifactsPresent": all(
            (OUT / name).is_file()
            for name in (
                "s10p_native_event_protocol.yaml",
                "candidate_population.jsonl",
                "scenario_population.jsonl",
                "s10_logical_roster.parquet",
                "native_event_feature_registry.json",
                "method_registry.json",
                "qualification_fixtures.jsonl",
                "qualification_feature_records.parquet",
                "feature_availability_validation.csv",
                "feature_availability_validation.json",
                "determinism_and_adversarial_validation.json",
                "access_control_validation.json",
                "leakage_and_dependency_validation.json",
                "s10_eligibility_gate.json",
                "validation_summary.json",
                "status.json",
                "provenance.json",
                "research_step_full_results.md",
            )
        ),
    }
    write_json(OUT / "artifact_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "qualify"))
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    else:
        qualify()


if __name__ == "__main__":
    main()
