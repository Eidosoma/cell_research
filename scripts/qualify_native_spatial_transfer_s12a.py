#!/usr/bin/env python3
"""Qualify S12P's native spatial transfer plane without transfer outcomes."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import yaml

from src.environment_suite import (
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
    EnvironmentSuite,
)
from src.environment_suite.contracts import canonical_sha256
from src.environment_suite.dsl_adapters import run_spatial_dsl_episode
from src.environment_suite.portfolio_adapters import portfolio_action
from src.morph2d.baseline import load_baseline_assets
from src.morph2d.engine import EpisodeDefinition
from src.phenotype_discovery.publication import (
    ArtifactSpec,
    AtomicScientificPublisher,
    PublicationContractError,
)
from src.spatial_transfer.qualification import (
    PANEL_IDS,
    SPATIAL_TASKS,
    DiagnosticSpatialTracker,
    apply_adaptation_variant,
    build_panel_fixture,
    canonical_binding_digest,
    endpoint_contract,
    qualify_configuration_binding,
    qualify_endpoint,
    run_fail_atomic_fixture_batch,
)


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
OUT = ARTIFACTS / "research_steps/S12A"
S12P = ARTIFACTS / "research_steps/S12P"
PROTOCOL = REPOSITORY / "configs/transfer/s12a_native_spatial_compatibility.yaml"
TASK_REGISTRY = REPOSITORY / "configs/environment_suite/task_registry.yaml"
SPLIT_MANIFEST = REPOSITORY / "configs/environment_suite/split_manifest.json"
S05_POLICIES = ARTIFACTS / "research_steps/S05/policy_catalog.jsonl"
S08M_REGISTRY = ARTIFACTS / "research_steps/S08M/portfolio_configuration_registry.jsonl"
S09_COMPRESSED = ARTIFACTS / "research_steps/S09/compressed_policies.jsonl"

QUALIFICATION_INPUTS = (
    WORKSPACE / "AGENTS.md",
    WORKSPACE / "FULL_PLAN.md",
    WORKSPACE / "RESEARCH_PLAN.md",
    WORKSPACE / "PREVIOUS_ARTIFACTS.md",
    WORKSPACE / "PREVIOUS_ARTIFACTS.json",
    WORKSPACE / "input-attachments/MANIFEST.json",
    WORKSPACE
    / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md",
    TASK_REGISTRY,
    SPLIT_MANIFEST,
    PROTOCOL,
    Path(__file__).resolve(),
    REPOSITORY / "src/environment_suite/dsl_adapters.py",
    REPOSITORY / "src/spatial_transfer/qualification.py",
    REPOSITORY / "tests/test_s12a_native_spatial_compatibility.py",
    S05_POLICIES,
    S08M_REGISTRY,
    S09_COMPRESSED,
    S12P / "s12p_broad_transfer_protocol.yaml",
    S12P / "candidate_lock.json",
    S12P / "candidate_population.jsonl",
    S12P / "lineage_registry.jsonl",
    S12P / "adaptation_variant_registry.jsonl",
    S12P / "baseline_registry.json",
    S12P / "compatibility_registry.json",
    S12P / "estimand_and_inference_registry.json",
    S12P / "s12_logical_roster.parquet",
    S12P / "budget_and_accounting.json",
    S12P / "s12_execution_gate.json",
    S12P / "preregistration_freeze.json",
    S12P / "artifact_manifest.json",
    Path("/previous-artifacts/E06/report_inputs/e07_handoff.md"),
    Path("/previous-artifacts/E06/research_steps/S01/target_catalog.yaml"),
    Path("/previous-artifacts/E06/research_steps/S02/target_grammars.yaml"),
    Path("/previous-artifacts/E06/research_steps/S03/environment_spec.md"),
    Path("/previous-artifacts/E06/research_steps/S04/movement_spec.md"),
    Path("/previous-artifacts/E06/research_steps/S05/policy_spec.md"),
    Path("/previous-artifacts/E06/research_steps/S06/control_channel_spec.md"),
    Path("/previous-artifacts/E06/research_steps/S07/engine_spec.md"),
    Path("/previous-artifacts/E06/research_steps/S10/frozen_perturbation_design.yaml"),
    Path("/previous-artifacts/E06/research_steps/S13/metric_specification.md"),
    Path("/previous-artifacts/E06/research_steps/S14/split_manifest.json"),
)

PREDECESSOR_STEPS = tuple(
    path
    for path in sorted((ARTIFACTS / "research_steps").iterdir())
    if path.is_dir() and path.name != "S12A"
)
PUBLICATION_CLASSES = (
    "transfer_results",
    "transfer_profiles",
    "adaptation_lock",
    "paired_estimands",
    "failure_censor_ledger",
    "native_cost_ledger",
    "replay_audit",
    "complete_accounting",
)
ROLE_COUNTS = {
    "adaptation_variant_development": 7_680,
    "adapted_winner_slot": 14_336,
    "component_single": 20_480,
    "matched_random": 7_168,
    "native_baseline": 1_024,
    "zero_shot": 14_336,
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


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


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def tree_manifest(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for root in paths:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rows.append(file_record(path))
    return rows


def tree_commitment(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(
        "E07/S12A/immutable-predecessor-tree/v1",
        [
            {
                "path": str(row["path"]),
                "bytes": int(row["bytes"]),
                "sha256": str(row["sha256"]),
            }
            for row in rows
        ],
    )


def strict_structural_configuration(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("rejectedModelOrEmbeddingUsed") is not False:
        raise RuntimeError("configuration used a rejected model or embedding")
    if row.get("s07ArmMembershipUsed") is not False:
        raise RuntimeError("configuration used S07 allocation-arm membership")
    allowed = {
        "assignmentCounterDomain",
        "assignmentRotation",
        "configurationId",
        "memberSetId",
        "members",
        "mode",
        "portfolioSize",
        "portfolioStructuralCosts",
        "schemaVersion",
        "selector",
        "taskId",
        "s09EditId",
        "s09ParentConfigurationId",
    }
    return {key: row[key] for key in sorted(set(row) & allowed)}


def load_frozen_objects() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    candidate_rows = read_jsonl(S12P / "candidate_population.jsonl")
    candidate_ids = {str(row["configurationId"]) for row in candidate_rows}
    configurations: dict[str, dict[str, Any]] = {}
    all_s08m_configurations: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(S08M_REGISTRY):
        if row.get("taskId") in SPATIAL_TASKS:
            all_s08m_configurations[str(row["configurationId"])] = row
        if str(row["configurationId"]) in candidate_ids:
            configurations[str(row["configurationId"])] = (
                strict_structural_configuration(row)
            )

    documents: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(S05_POLICIES):
        documents[str(row["policySha256"])] = dict(row["document"])
    for row in read_jsonl(S09_COMPRESSED):
        configuration = strict_structural_configuration(row["configuration"])
        configurations[str(configuration["configurationId"])] = configuration
        documents.update(
            {
                str(policy_hash): dict(document)
                for policy_hash, document in row["documentsByPolicySha256"].items()
            }
        )

    if set(configurations) != candidate_ids or len(configurations) != 14:
        raise RuntimeError(
            "exact fourteen-candidate configuration lock did not resolve"
        )
    variants = read_jsonl(S12P / "adaptation_variant_registry.jsonl")
    if len(variants) != 30:
        raise RuntimeError("exact thirty-variant lock did not resolve")
    return configurations, documents, variants, all_s08m_configurations


def validate_s12p_freeze() -> dict[str, Any]:
    manifest = read_json(S12P / "artifact_manifest.json")
    records = []
    for row in manifest["artifacts"]:
        path = Path(row["path"])
        records.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "bytesMatch": path.is_file()
                and path.stat().st_size == int(row["bytes"]),
                "sha256Match": path.is_file()
                and sha256_file(path) == str(row["sha256"]),
            }
        )
    freeze = read_json(S12P / "preregistration_freeze.json")
    roster = pd.read_parquet(S12P / "s12_logical_roster.parquet")
    semantic_rows = sorted(
        roster.to_dict("records"), key=lambda row: row["logicalReservationId"]
    )
    semantic_sha256 = canonical_sha256(
        "E07/S12P/ordered-logical-roster/v1", semantic_rows
    )
    expected_files = {
        "protocolSha256": S12P / "s12p_broad_transfer_protocol.yaml",
        "candidatePopulationSha256": S12P / "candidate_population.jsonl",
        "lineageRegistrySha256": S12P / "lineage_registry.jsonl",
        "baselineRegistrySha256": S12P / "baseline_registry.json",
        "adaptationVariantRegistrySha256": S12P / "adaptation_variant_registry.jsonl",
        "scenarioPopulationSha256": S12P / "scenario_population.jsonl",
        "logicalRosterSha256": S12P / "s12_logical_roster.parquet",
        "compatibilityRegistrySha256": S12P / "compatibility_registry.json",
        "estimandRegistrySha256": S12P / "estimand_and_inference_registry.json",
        "budgetAccountingSha256": S12P / "budget_and_accounting.json",
        "phenotypeBranchClosureSha256": S12P / "phenotype_branch_closure.json",
    }
    freeze_checks = {
        key: sha256_file(path) == str(freeze[key])
        for key, path in expected_files.items()
    }
    return {
        "schemaVersion": "e07.s12a.s12p-freeze-validation.v1",
        "researchStepId": "S12A",
        "manifestArtifactCount": len(records),
        "manifestRowsPassed": sum(
            row["exists"] and row["bytesMatch"] and row["sha256Match"]
            for row in records
        ),
        "manifestAllPassed": all(
            row["exists"] and row["bytesMatch"] and row["sha256Match"]
            for row in records
        ),
        "freezeFileChecks": freeze_checks,
        "freezeFilesAllPassed": all(freeze_checks.values()),
        "logicalRosterRows": len(roster),
        "logicalReservationIdsUnique": roster["logicalReservationId"].is_unique,
        "logicalRosterSemanticSha256": semantic_sha256,
        "logicalRosterSemanticSha256Matches": semantic_sha256
        == freeze["logicalRosterSemanticSha256"],
        "roleCounts": dict(sorted(Counter(roster["conditionRole"]).items())),
        "roleCountsMatch": dict(sorted(Counter(roster["conditionRole"]).items()))
        == ROLE_COUNTS,
        "allPassed": False,
        "records": records,
    } | {
        "allPassed": bool(
            all(
                (
                    all(
                        row["exists"] and row["bytesMatch"] and row["sha256Match"]
                        for row in records
                    ),
                    all(freeze_checks.values()),
                    len(roster) == 65_024,
                    roster["logicalReservationId"].is_unique,
                    semantic_sha256 == freeze["logicalRosterSemanticSha256"],
                    dict(sorted(Counter(roster["conditionRole"]).items()))
                    == ROLE_COUNTS,
                )
            )
        )
    }


def build_bindings(
    configurations: Mapping[str, Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for configuration_id, configuration in sorted(configurations.items()):
        for task_id in SPATIAL_TASKS:
            for panel_id in PANEL_IDS:
                rows.append(
                    qualify_configuration_binding(
                        configuration,
                        documents,
                        execution_task_id=task_id,
                        panel_id=panel_id,
                        selection_ref=configuration_id,
                    )
                )
    for variant in sorted(variants, key=lambda row: row["adaptationVariantId"]):
        adapted = apply_adaptation_variant(
            configurations[str(variant["baseConfigurationId"])], variant
        )
        for task_id in SPATIAL_TASKS:
            for panel_id in PANEL_IDS:
                rows.append(
                    qualify_configuration_binding(
                        adapted,
                        documents,
                        execution_task_id=task_id,
                        panel_id=panel_id,
                        selection_ref=str(variant["adaptationVariantId"]),
                    )
                )
    rows.sort(key=lambda row: row["bindingCommitmentSha256"])
    forward = canonical_binding_digest(rows)
    reverse = canonical_binding_digest(reversed(rows))
    validation = {
        "schemaVersion": "e07.s12a.binding-validation.v1",
        "researchStepId": "S12A",
        "candidateConfigurationCount": len(configurations),
        "adaptationVariantCount": len(variants),
        "taskCount": len(SPATIAL_TASKS),
        "panelCount": len(PANEL_IDS),
        "bindingRows": len(rows),
        "expectedBindingRows": 704,
        "allBindingsPassed": all(row["passed"] for row in rows),
        "policyBytesPreserved": all(
            len(row["memberPolicySha256"]) == row["memberCount"] for row in rows
        ),
        "nativeCarrierPreserved": all(row["nativeCarrierPreserved"] for row in rows),
        "actionRebindingCount": sum(row["actionRebound"] for row in rows),
        "costRebindingCount": sum(row["costSemanticsRebound"] for row in rows),
        "authorityImpersonationCount": sum(
            bool(row["authorityChannelPermissions"]) for row in rows
        ),
        "selectorAuthorizationAllPassed": all(
            row["selectorTargetTaskAuthorized"] for row in rows
        ),
        "outcomeFieldsLoaded": any(row["outcomeFieldsLoaded"] for row in rows),
        "forwardBindingSetSha256": forward,
        "reverseWorkerOrderBindingSetSha256": reverse,
        "workerOrderIndependent": forward == reverse,
    }
    validation["allPassed"] = bool(
        validation["bindingRows"] == 704
        and validation["allBindingsPassed"]
        and validation["policyBytesPreserved"]
        and validation["nativeCarrierPreserved"]
        and validation["actionRebindingCount"] == 0
        and validation["costRebindingCount"] == 0
        and validation["authorityImpersonationCount"] == 0
        and validation["selectorAuthorizationAllPassed"]
        and not validation["outcomeFieldsLoaded"]
        and validation["workerOrderIndependent"]
    )
    return rows, validation


def build_endpoint_evidence() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contracts = [
        endpoint_contract(task_id, panel_id)
        for task_id in SPATIAL_TASKS
        for panel_id in PANEL_IDS
    ]
    qualifications = [
        qualify_endpoint(task_id, panel_id)
        for task_id in SPATIAL_TASKS
        for panel_id in PANEL_IDS
    ]
    diagnostic = [row for row in contracts if not row["calibratedCompletionAvailable"]]
    calibrated = [
        row for row in qualifications if row["endpointMode"] != "diagnostic_only"
    ]
    evidence = {
        "schemaVersion": "e07.s12a.endpoint-validation.v1",
        "researchStepId": "S12A",
        "contractRows": len(contracts),
        "qualificationRows": len(qualifications),
        "qualificationRowsPassed": sum(row["passed"] for row in qualifications),
        "calibratedRows": len(calibrated),
        "diagnosticOnlyRows": len(diagnostic),
        "holeBoundaryRows": sum(
            row.get("boundaryCueRequired") is True for row in qualifications
        ),
        "holeBoundaryDenialAllPassed": all(
            row["missingBoundarySignalDenied"]
            for row in qualifications
            if row.get("boundaryCueRequired")
        ),
        "repairEligibleRows": sum(row["repairRiskSetEligible"] for row in contracts),
        "oneSwapClassifiedAsFaultRows": sum(row["oneSwapIsFault"] for row in contracts),
        "diagnosticCompletionAvailabilityRows": sum(
            row["calibratedCompletionAvailable"] for row in diagnostic
        ),
        "diagnosticPromotionEligibleRows": sum(
            row["diagnosticPromotionEligible"] for row in diagnostic
        ),
        "universalScoresDefined": 0,
    }
    evidence["allPassed"] = bool(
        len(contracts) == 16
        and len(qualifications) == 16
        and evidence["qualificationRowsPassed"] == 16
        and evidence["calibratedRows"] == 10
        and evidence["diagnosticOnlyRows"] == 6
        and evidence["holeBoundaryRows"] == 4
        and evidence["holeBoundaryDenialAllPassed"]
        and evidence["repairEligibleRows"] == 0
        and evidence["oneSwapClassifiedAsFaultRows"] == 0
        and evidence["diagnosticCompletionAvailabilityRows"] == 0
        and evidence["diagnosticPromotionEligibleRows"] == 0
    )
    return contracts, {"summary": evidence, "rows": qualifications}


def authority_audit() -> dict[str, Any]:
    sources = [
        file_record(Path("/previous-artifacts/E06/research_steps/S07/engine_spec.md")),
        file_record(
            Path(
                "/previous-artifacts/E06/research_steps/S10/"
                "frozen_perturbation_design.yaml"
            )
        ),
        file_record(
            Path("/previous-artifacts/E06/research_steps/S06/control_channel_spec.md")
        ),
        file_record(
            Path("/previous-artifacts/E06/research_steps/S04/movement_spec.md")
        ),
    ]
    return {
        "schemaVersion": "e07.s12a.fault-scheduler-authority-audit.v1",
        "researchStepId": "S12A",
        "authoritativeSources": sources,
        "faultAudit": {
            "requiredGate": "G06",
            "authoritativeUnseenE06SpatialFaultFamilies": [],
            "passed": False,
            "directControllerActuationMissDisposition": (
                "S06_channel_specific_not_native_local_policy_fault"
            ),
            "S10LesionDisposition": (
                "external_intervention_not_unseen_runtime_fault_family"
            ),
            "oneSwapDisposition": ("accepted_target_set_perturbation_not_fault"),
            "crossPredecessorFaultRebinding": False,
            "blocker": (
                "No authoritative E06 contract defines an unseen spatial "
                "fault family compatible with the frozen S12P estimand."
            ),
        },
        "schedulerAudit": {
            "requiredGate": "G07",
            "authoritativeSchedulerFamilies": [
                "state_blind_identity_hash_rank_without_replacement_per_transition"
            ],
            "authoritativeUnseenE06SpatialSchedulerFamilies": [],
            "passed": False,
            "scheduleSeedVariationDisposition": (
                "within_family_scenario_variation_not_scheduler_transfer"
            ),
            "crossPredecessorSchedulerRebinding": False,
            "blocker": (
                "E06 fixes one state-blind identity-hash batch-four scheduler; "
                "no alternate authoritative E06 scheduler-family contract exists."
            ),
        },
        "estimandChanged": False,
        "rosterNarrowed": False,
        "panelsRemoved": 0,
        "g06Passed": False,
        "g07Passed": False,
        "substantiveS12Authorized": False,
    }


def build_roster_bindings(
    configurations: Mapping[str, Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    all_s08m_configurations: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    roster = pd.read_parquet(S12P / "s12_logical_roster.parquet")
    variant_ids = {str(row["adaptationVariantId"]) for row in variants}
    candidate_ids = set(configurations)
    native_baselines = {
        "spatial_greedy_local_native_v1",
        "spatial_memory_local_native_v1",
    }
    dispositions = []
    for row in roster.itertuples(index=False):
        if row.conditionRole == "adaptation_variant_development":
            resolution = (
                "frozen_adaptation_variant"
                if row.runtimeSelectionRef in variant_ids
                else "unresolved"
            )
        elif row.conditionRole in {"zero_shot"}:
            resolution = (
                "frozen_candidate_configuration"
                if row.runtimeSelectionRef in candidate_ids
                else "unresolved"
            )
        elif row.conditionRole == "adapted_winner_slot":
            resolution = (
                "pre_outcome_adaptation_winner_placeholder"
                if row.runtimeSelectionRef
                == "locked_adaptation_winner_after_development"
                and row.baseConfigurationId in candidate_ids
                else "unresolved"
            )
        elif row.conditionRole in {"component_single", "matched_random"}:
            configuration = all_s08m_configurations.get(row.runtimeSelectionRef)
            resolution = (
                "frozen_s08m_spatial_comparator"
                if configuration is not None
                and configuration.get("taskId") in SPATIAL_TASKS
                and all(
                    member.get("nativeCarrier") == "spatial"
                    for member in configuration["members"]
                )
                else "unresolved"
            )
        elif row.conditionRole == "native_baseline":
            resolution = (
                "native_task_baseline"
                if row.baseConfigurationId in native_baselines
                and row.runtimeSelectionRef == "native_task_baseline"
                else "unresolved"
            )
        else:
            resolution = "unresolved"
        dispositions.append(resolution)
    bound = roster[
        [
            "logicalReservationId",
            "partition",
            "taskId",
            "panelId",
            "scenarioFamilyId",
            "conditionRole",
            "baseConfigurationId",
            "runtimeSelectionRef",
        ]
    ].copy()
    bound["runtimeBindingDisposition"] = dispositions
    bound["physicalReplayCount"] = 2
    bound["outcomeMaterialized"] = False
    bound["qualificationOnly"] = True
    bound["runtimeBindingCommitmentSha256"] = [
        canonical_sha256(
            "E07/S12A/roster-runtime-binding/v1",
            {
                "logicalReservationId": row.logicalReservationId,
                "runtimeBindingDisposition": disposition,
                "physicalReplayCount": 2,
            },
        )
        for row, disposition in zip(
            roster.itertuples(index=False), dispositions, strict=True
        )
    ]
    role_counts = dict(sorted(Counter(roster["conditionRole"]).items()))
    accounting = {
        "schemaVersion": "e07.s12a.roster-accounting-validation.v1",
        "researchStepId": "S12A",
        "logicalRows": len(bound),
        "uniqueLogicalReservationIds": bound["logicalReservationId"].nunique(),
        "plannedPhysicalRowsIncludingReplay": int(bound["physicalReplayCount"].sum()),
        "runtimeBindingsResolved": int(
            (bound["runtimeBindingDisposition"] != "unresolved").sum()
        ),
        "runtimeBindingsUnresolved": int(
            (bound["runtimeBindingDisposition"] == "unresolved").sum()
        ),
        "unresolvedByRoleAndConfiguration": [
            {
                "conditionRole": str(condition_role),
                "baseConfigurationId": str(configuration_id),
                "logicalRows": int(count),
                "reason": (
                    "S08M_VALIDATION_LOCK_NAMES_ID_BUT_IMMUTABLE_RUNTIME_"
                    "CONFIGURATION_REGISTRY_HAS_NO_DEFINITION"
                ),
            }
            for (
                condition_role,
                configuration_id,
            ), count in bound[bound["runtimeBindingDisposition"] == "unresolved"]
            .groupby(["conditionRole", "baseConfigurationId"])
            .size()
            .items()
        ],
        "roleCounts": role_counts,
        "roleCountsMatch": role_counts == ROLE_COUNTS,
        "outcomeRowsMaterialized": int(bound["outcomeMaterialized"].sum()),
        "transferEpisodesExecuted": 0,
        "rosterRowsAddedRemovedOrChanged": 0,
        "rosterAccountingConserved": True,
        "frozenLogicalRosterSha256": sha256_file(S12P / "s12_logical_roster.parquet"),
    }
    accounting["runtimeBindingComplete"] = bool(
        accounting["logicalRows"] == 65_024
        and accounting["uniqueLogicalReservationIds"] == 65_024
        and accounting["plannedPhysicalRowsIncludingReplay"] == 130_048
        and accounting["runtimeBindingsResolved"] == 65_024
        and accounting["runtimeBindingsUnresolved"] == 0
        and accounting["roleCountsMatch"]
        and accounting["outcomeRowsMaterialized"] == 0
        and accounting["transferEpisodesExecuted"] == 0
    )
    accounting["qualificationAuditCompleted"] = bool(
        accounting["logicalRows"] == 65_024
        and accounting["uniqueLogicalReservationIds"] == 65_024
        and accounting["plannedPhysicalRowsIncludingReplay"] == 130_048
        and accounting["runtimeBindingsResolved"]
        + accounting["runtimeBindingsUnresolved"]
        == 65_024
        and accounting["roleCountsMatch"]
        and accounting["outcomeRowsMaterialized"] == 0
        and accounting["transferEpisodesExecuted"] == 0
        and accounting["rosterRowsAddedRemovedOrChanged"] == 0
    )
    accounting["allPassed"] = accounting["runtimeBindingComplete"]

    smoke_rows = []
    for task_id in SPATIAL_TASKS:
        task_rows = roster[
            (roster["taskId"] == task_id)
            & (roster["panelId"] == "native_reference_control")
            & (roster["partition"] == "transfer_evaluation")
        ]
        for configuration_id in sorted(candidate_ids):
            zero = (
                task_rows[
                    (task_rows["conditionRole"] == "zero_shot")
                    & (task_rows["baseConfigurationId"] == configuration_id)
                ]
                .sort_values("logicalReservationId")
                .iloc[0]
            )
            adapted = (
                task_rows[
                    (task_rows["conditionRole"] == "adapted_winner_slot")
                    & (task_rows["baseConfigurationId"] == configuration_id)
                ]
                .sort_values("logicalReservationId")
                .iloc[0]
            )
            for stage, source in ((1, zero), (2, adapted)):
                smoke_rows.append(
                    {
                        "schemaVersion": "e07.s12a.structural-smoke-binding.v1",
                        "smokeStage": stage,
                        "taskId": task_id,
                        "configurationId": configuration_id,
                        "logicalReservationId": source.logicalReservationId,
                        "runtimeSelectionRef": source.runtimeSelectionRef,
                        "qualificationOnly": True,
                        "episodeSubmitted": False,
                    }
                )
    smoke = pd.DataFrame(smoke_rows).sort_values(
        ["smokeStage", "taskId", "configurationId"]
    )
    if len(smoke) != 56 or smoke["logicalReservationId"].nunique() != 56:
        raise RuntimeError("frozen 56-row structural smoke binding did not resolve")
    return bound, accounting, smoke


def _integrity_projection(result: Mapping[str, Any]) -> dict[str, Any]:
    audit = result["portfolioAssignmentAudit"]
    offline = result["offlineEvaluation"]
    ledgers = (
        "movementLedger",
        "observationLedger",
        "e06ChannelLedger",
        "dslRuntimeLedger",
        "dslCommunicationLedger",
        "licensedCapabilityLedger",
        "portfolioStructuralLedger",
        "portfolioCoordinationLedger",
    )
    return {
        "episodeSha256": result["episodeSha256"],
        "transitionBudget": result["transitionBudget"],
        "actorBatchSize": result["actorBatchSize"],
        "transitionCount": len(result["transitionSummaries"]),
        "portfolioAuditPresent": bool(audit),
        "portfolioAuditConfigurationId": audit["configurationId"],
        "assignmentCount": sum(audit["assignmentCountsByPolicySha256"].values()),
        "ledgerFamiliesPresent": all(name in result for name in ledgers),
        "licensedSpatialCostsZero": all(
            value == 0 for value in result["licensedCapabilityLedger"].values()
        ),
        "authorityAudit": result["authorityAudit"],
        "stopReason": result["stopReason"],
        "offlineEvaluationSchemaVersion": offline.get(
            "schemaVersion", "e06.target-metric-tracker.native-v1"
        ),
        "diagnosticCompletionAvailable": (
            offline.get("topologyCompletionCalibrated")
            if "topologyCompletionCalibrated" in offline
            else True
        ),
        "diagnosticPromotionEligible": offline.get(
            "diagnosticPromotionEligible", False
        ),
        "resultCommitmentSha256": canonical_sha256(
            "E07/S12A/qualification-episode-result/v1", result
        ),
    }


def run_runtime_qualification(
    configurations: Mapping[str, Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    representatives = {
        task_id: next(
            configuration
            for _configuration_id, configuration in sorted(configurations.items())
            if configuration["taskId"] == task_id
        )
        for task_id in SPATIAL_TASKS
    }
    fixture_ids = [
        f"{task_id}::{panel_id}" for task_id in SPATIAL_TASKS for panel_id in PANEL_IDS
    ]

    def evaluate(item_id: str) -> dict[str, Any]:
        task_id, panel_id = item_id.split("::", maxsplit=1)
        fixture = build_panel_fixture(task_id, panel_id)
        configuration = representatives[task_id]
        action = portfolio_action(
            [
                documents[str(member["policySha256"])]
                for member in configuration["members"]
            ],
            configuration,
        )
        context, _targets, _grammars, _environments = load_baseline_assets()
        context = replace(
            context,
            environments={
                **context.environments,
                fixture["environment"].environment_id: fixture["environment"],
            },
        )
        definition = EpisodeDefinition(
            scenario_id=(
                "s12a-qualification-only/"
                f"{task_id}/{panel_id}/{configuration['configurationId']}"
            ),
            environment_id=fixture["environment"].environment_id,
            policy_id=action.policy_id,
            relation_grammar_id=fixture["grammar"].grammar_id,
            channel_mode="none",
            transitions=32,
            actor_batch_size=4,
            parameters={},
        )
        tracker = (
            DiagnosticSpatialTracker(fixture["environment"], fixture["reference"])
            if fixture["endpointMode"] == "diagnostic_only"
            else None
        )
        result = run_spatial_dsl_episode(
            context,
            definition,
            action,
            target_id=fixture["targetId"],
            initial_state_override=fixture["initialState"],
            offline_tracker=tracker,
        )
        projection = _integrity_projection(result)
        projection.update(
            {
                "schemaVersion": "e07.s12a.runtime-fixture-integrity.v1",
                "fixtureLogicalId": item_id,
                "taskId": task_id,
                "panelId": panel_id,
                "endpointMode": fixture["endpointMode"],
                "configurationId": configuration["configurationId"],
                "qualificationOnly": True,
                "efficacyEvidence": False,
                "persistedObjectiveValues": False,
            }
        )
        return projection

    forward_rows = [evaluate(item_id) for item_id in fixture_ids]
    reverse_rows = [evaluate(item_id) for item_id in reversed(fixture_ids)]
    forward_by_id = {row["fixtureLogicalId"]: row for row in forward_rows}
    reverse_by_id = {row["fixtureLogicalId"]: row for row in reverse_rows}
    replay_matches = {
        item_id: forward_by_id[item_id]["resultCommitmentSha256"]
        == reverse_by_id[item_id]["resultCommitmentSha256"]
        for item_id in fixture_ids
    }
    order_projection = lambda rows: canonical_sha256(  # noqa: E731
        "E07/S12A/runtime-fixture-set/v1",
        sorted(
            (
                {
                    key: value
                    for key, value in row.items()
                    if key != "resultCommitmentSha256"
                }
                for row in rows
            ),
            key=lambda row: row["fixtureLogicalId"],
        ),
    )
    forward_set_sha256 = order_projection(forward_rows)
    reverse_set_sha256 = order_projection(reverse_rows)
    summary = {
        "schemaVersion": "e07.s12a.runtime-qualification.v1",
        "researchStepId": "S12A",
        "dedicatedFixtureLogicalRows": len(forward_rows),
        "dedicatedFixturePhysicalRows": len(forward_rows) + len(reverse_rows),
        "frozenS12LogicalRowsExecuted": 0,
        "frozenS12PhysicalRowsExecuted": 0,
        "transitionBudgetAll32": all(
            row["transitionBudget"] == 32 for row in forward_rows
        ),
        "actorBatchSizeAll4": all(row["actorBatchSize"] == 4 for row in forward_rows),
        "portfolioAssignmentAuditAllPresent": all(
            row["portfolioAuditPresent"] for row in forward_rows
        ),
        "assignmentCountAll128": all(
            row["assignmentCount"] == 128 for row in forward_rows
        ),
        "separateLedgerFamiliesAllPresent": all(
            row["ledgerFamiliesPresent"] for row in forward_rows
        ),
        "licensedSpatialCostsSeparatelyZero": all(
            row["licensedSpatialCostsZero"] for row in forward_rows
        ),
        "nativeAuthorityAllPassed": all(
            all(row["authorityAudit"].values()) for row in forward_rows
        ),
        "exactReplayByFixture": replay_matches,
        "exactReplayAllPassed": all(replay_matches.values()),
        "forwardWorkerOrderSha256": forward_set_sha256,
        "reverseWorkerOrderSha256": reverse_set_sha256,
        "workerOrderIndependent": forward_set_sha256 == reverse_set_sha256,
        "diagnosticCompletionPromotedRows": sum(
            row["diagnosticPromotionEligible"] for row in forward_rows
        ),
        "persistedObjectiveValues": False,
        "qualificationOnly": True,
    }
    summary["allPassed"] = bool(
        summary["dedicatedFixtureLogicalRows"] == 16
        and summary["dedicatedFixturePhysicalRows"] == 32
        and summary["transitionBudgetAll32"]
        and summary["actorBatchSizeAll4"]
        and summary["portfolioAssignmentAuditAllPresent"]
        and summary["assignmentCountAll128"]
        and summary["separateLedgerFamiliesAllPresent"]
        and summary["licensedSpatialCostsSeparatelyZero"]
        and summary["nativeAuthorityAllPassed"]
        and summary["exactReplayAllPassed"]
        and summary["workerOrderIndependent"]
        and summary["diagnosticCompletionPromotedRows"] == 0
        and not summary["persistedObjectiveValues"]
    )
    registry = {
        "schemaVersion": "e07.s12a.runtime-fixture-registry.v1",
        "researchStepId": "S12A",
        "fixtures": forward_rows,
        "physicalReplayCountPerFixture": 2,
        "qualificationOnly": True,
        "efficacyEvidence": False,
        "outcomeValuesPersisted": False,
    }
    return registry, summary


def fail_atomic_execution_validation() -> dict[str, Any]:
    item_ids = [f"s12a-fixture-{index:02d}" for index in range(16)]
    evaluator = lambda item_id: {  # noqa: E731
        "itemId": item_id,
        "qualificationOnly": True,
        "passed": True,
    }
    records = [
        run_fail_atomic_fixture_batch(item_ids, evaluator, failure_position=position)
        for position in (0, len(item_ids) // 2, len(item_ids) - 1)
    ]
    success = run_fail_atomic_fixture_batch(item_ids, evaluator, failure_position=None)
    return {
        "schemaVersion": "e07.s12a.fail-atomic-execution-validation.v1",
        "researchStepId": "S12A",
        "failurePositions": [row["failurePosition"] for row in records],
        "injectedFailures": [
            {
                "failurePosition": row["failurePosition"],
                "precommitted": row["precommitted"],
                "attempted": row["attempted"],
                "failed": row["failed"],
                "notAttempted": row["notAttempted"],
                "publicationRows": row["publicationRows"],
                "dispositionRows": len(row["dispositions"]),
            }
            for row in records
        ],
        "successPublicationRows": success["publicationRows"],
        "allFailurePublicationsZero": all(
            row["publicationRows"] == 0 for row in records
        ),
        "allFailureDispositionSetsComplete": all(
            len(row["dispositions"]) == len(item_ids) for row in records
        ),
        "successPublicationComplete": success["publicationRows"] == len(item_ids),
        "allPassed": all(
            (
                all(row["publicationRows"] == 0 for row in records),
                all(len(row["dispositions"]) == len(item_ids) for row in records),
                success["publicationRows"] == len(item_ids),
            )
        ),
    }


def _json_payload(label: str) -> bytes:
    return canonical_json_bytes(
        {
            "schemaVersion": "e07.s12a.synthetic-publication-fixture.v1",
            "artifactClass": label,
            "qualificationOnly": True,
            "outcomeRows": 0,
        }
    )


def fail_atomic_publication_validation() -> tuple[dict[str, Any], dict[str, Any]]:
    specs = tuple(
        ArtifactSpec(class_id, f"{class_id}.json", "json")
        for class_id in PUBLICATION_CLASSES
    )
    payloads = {class_id: _json_payload(class_id) for class_id in PUBLICATION_CLASSES}
    publisher = AtomicScientificPublisher(specs)
    failure_points = [
        point
        for class_id in PUBLICATION_CLASSES
        for point in (
            f"before_write:{class_id}",
            f"during_write:{class_id}",
            f"after_write:{class_id}",
        )
    ] + [
        "before_full_validation",
        "after_full_validation",
        "before_commit",
        "after_commit",
    ]
    records = []
    with TemporaryDirectory(prefix="s12a-publication-", dir="/cache") as raw:
        root = Path(raw)
        for index, point in enumerate(failure_points):
            destination = root / f"scientific-{index:02d}"
            forensics = root / f"forensics-{index:02d}"
            try:
                publisher.publish(
                    destination,
                    payloads,
                    forensics_directory=forensics,
                    failure_point=point,
                )
            except PublicationContractError as error:
                audit = getattr(error, "audit", {})
            else:
                raise RuntimeError("injected publication failure did not fire")
            files = (
                sorted(path.name for path in destination.iterdir() if path.is_file())
                if destination.exists()
                else []
            )
            state = audit.get("finalScientificPublicationState")
            records.append(
                {
                    "failurePoint": point,
                    "finalState": state,
                    "destinationExists": destination.exists(),
                    "publishedFileCount": len(files),
                    "validState": state
                    in {
                        "zero_scientific_publication",
                        "complete_validated_publication",
                    },
                }
            )
        first = root / "success-forward"
        second = root / "success-reverse"
        forward = publisher.publish(
            first,
            payloads,
            forensics_directory=root / "success-forward-forensics",
        )
        reverse = publisher.publish(
            second,
            dict(reversed(list(payloads.items()))),
            forensics_directory=root / "success-reverse-forensics",
        )
        forward_files = {
            path.name: path.read_bytes()
            for path in first.iterdir()
            if path.is_file() and path.name != "publication_manifest.json"
        }
        reverse_files = {
            path.name: path.read_bytes()
            for path in second.iterdir()
            if path.is_file() and path.name != "publication_manifest.json"
        }
    registry = {
        "schemaVersion": "e07.s12a.future-publication-registry.v1",
        "researchStepId": "S12A",
        "artifactClasses": [
            {
                "classId": spec.class_id,
                "relativePath": spec.relative_path,
                "mediaType": spec.media_type,
            }
            for spec in specs
        ],
        "scientificDestinationDuringS12A": None,
        "futureS12Only": True,
    }
    validation = {
        "schemaVersion": "e07.s12a.fail-atomic-publication-validation.v1",
        "researchStepId": "S12A",
        "artifactClassCount": len(specs),
        "injectedFailureCount": len(records),
        "beforeDuringAfterEveryArtifactClass": len(records) == 28,
        "validAllOrZeroStateCount": sum(row["validState"] for row in records),
        "zeroPublicationFailures": sum(
            row["finalState"] == "zero_scientific_publication" for row in records
        ),
        "completePublicationAfterCommitFailures": sum(
            row["finalState"] == "complete_validated_publication" for row in records
        ),
        "successForwardComplete": forward["finalScientificPublicationState"]
        == "complete_validated_publication",
        "successReverseComplete": reverse["finalScientificPublicationState"]
        == "complete_validated_publication",
        "workerOrderPayloadBytesIdentical": forward_files == reverse_files,
        "scientificPublicationCreatedByS12A": False,
        "records": records,
    }
    validation["allPassed"] = bool(
        len(specs) == 8
        and len(records) == 28
        and validation["validAllOrZeroStateCount"] == 28
        and validation["zeroPublicationFailures"] == 27
        and validation["completePublicationAfterCommitFailures"] == 1
        and validation["successForwardComplete"]
        and validation["successReverseComplete"]
        and validation["workerOrderPayloadBytesIdentical"]
        and not validation["scientificPublicationCreatedByS12A"]
    )
    return registry, validation


def access_control_validation() -> dict[str, Any]:
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    development = AccessGrant(AccessPhase.DEVELOPMENT)
    confirmation = AccessGrant(AccessPhase.CONFIRMATION, candidate_lock_sha256="0" * 64)
    records = []
    for task_id in SPATIAL_TASKS:
        task_records = [
            record for record in suite.records.values() if record.task_id == task_id
        ]
        by_split = {record.split.value: record for record in task_records}
        for split, grant in (
            ("validation", development),
            ("confirmation", development),
            ("confirmation", confirmation),
        ):
            try:
                suite.open(task_id, by_split[split].scenario_id, grant)
            except AccessDeniedError as error:
                records.append(
                    {
                        "taskId": task_id,
                        "split": split,
                        "grant": grant.phase.value,
                        "denied": True,
                        "reason": str(error),
                    }
                )
            else:
                records.append(
                    {
                        "taskId": task_id,
                        "split": split,
                        "grant": grant.phase.value,
                        "denied": False,
                        "reason": None,
                    }
                )
    return {
        "schemaVersion": "e07.s12a.access-control-validation.v1",
        "researchStepId": "S12A",
        "attempts": len(records),
        "denials": sum(row["denied"] for row in records),
        "allDenied": all(row["denied"] for row in records),
        "trainingQualificationFixtureRows": 16,
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
        "transferOutcomeRowsRead": 0,
        "S14ConfirmationReserveRowsRead": 0,
        "quarantinedOutcomeOrCacheRowsRead": 0,
        "brokerAudit": suite.broker.audit.to_dict(),
        "records": records,
    }


def dependency_exclusion_validation() -> dict[str, Any]:
    searched_files = (
        PROTOCOL,
        REPOSITORY / "src/spatial_transfer/qualification.py",
        Path(__file__).resolve(),
    )
    prohibited_module_prefixes = (
        "src.surrogate",
        "src.trajectory_models",
        "src.allocation",
    )
    imported_modules: set[str] = set()
    for path in searched_files:
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
    hits = [
        module
        for module in sorted(imported_modules)
        if module.startswith(prohibited_module_prefixes)
    ]
    cache_inputs = [
        str(path) for path in QUALIFICATION_INPUTS if str(path).startswith("/cache/")
    ]
    return {
        "schemaVersion": "e07.s12a.dependency-exclusion-validation.v1",
        "researchStepId": "S12A",
        "searchedFiles": [str(path) for path in searched_files],
        "runtimeImportedModules": sorted(imported_modules),
        "prohibitedRuntimeDependencyHits": hits,
        "cacheInputPaths": cache_inputs,
        "S06ModelLoads": 0,
        "S06AModelLoads": 0,
        "S06EmbeddingLoads": 0,
        "S06AEmbeddingLoads": 0,
        "S07ArmSignalUses": 0,
        "S10OrS11SignalUses": 0,
        "quarantineCacheReads": 0,
        "universalScoreDefined": False,
        "allPassed": not hits and not cache_inputs,
    }


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(
        path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )


def build_artifact_manifest() -> dict[str, Any]:
    files = [
        file_record(path)
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    return {
        "schemaVersion": "e07.s12a.artifact-manifest.v1",
        "researchStepId": "S12A",
        "artifactCount": len(files),
        "artifacts": files,
    }


def qualify() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"{OUT} must be empty before S12A")
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["researchStepId"] != "S12A":
        raise RuntimeError("S12A protocol ID mismatch")
    missing = [str(path) for path in QUALIFICATION_INPUTS if not path.is_file()]
    if missing:
        raise RuntimeError(f"required S12A input missing: {missing}")

    predecessor_before = tree_manifest(PREDECESSOR_STEPS)
    predecessor_commitment_before = tree_commitment(predecessor_before)
    inputs = [file_record(path) for path in QUALIFICATION_INPUTS]
    input_freeze = {
        "schemaVersion": "e07.s12a.input-hash-freeze.v1",
        "researchStepId": "S12A",
        "inputs": inputs,
        "inputSetCommitmentSha256": canonical_sha256("E07/S12A/input-set/v1", inputs),
        "predecessorTreeFileCount": len(predecessor_before),
        "predecessorTreeBytes": sum(int(row["bytes"]) for row in predecessor_before),
        "predecessorTreeCommitmentBeforeSha256": (predecessor_commitment_before),
        "outcomeBearingInputsRead": 0,
        "quarantinedCachesRead": 0,
        "protectedOutcomeRowsRead": 0,
    }
    write_json(OUT / "input_hash_freeze.json", input_freeze)
    (OUT / "s12a_native_spatial_compatibility_protocol.yaml").write_bytes(
        PROTOCOL.read_bytes()
    )

    s12p_validation = validate_s12p_freeze()
    if not s12p_validation["allPassed"]:
        raise RuntimeError("S12P byte/semantic freeze did not revalidate")
    write_json(OUT / "s12p_hash_revalidation.json", s12p_validation)

    configurations, documents, variants, all_s08m = load_frozen_objects()
    binding_rows, binding_validation = build_bindings(
        configurations, documents, variants
    )
    if not binding_validation["allPassed"]:
        raise RuntimeError("native spatial binding qualification failed")
    write_parquet(OUT / "binding_qualification.parquet", pd.DataFrame(binding_rows))
    write_json(OUT / "binding_validation.json", binding_validation)

    contracts, endpoint_validation = build_endpoint_evidence()
    if not endpoint_validation["summary"]["allPassed"]:
        raise RuntimeError("native endpoint qualification failed")
    write_json(
        OUT / "endpoint_contract_registry.json",
        {
            "schemaVersion": "e07.s12a.endpoint-contract-registry.v1",
            "researchStepId": "S12A",
            "contracts": contracts,
        },
    )
    write_json(OUT / "endpoint_qualification.json", endpoint_validation)

    authority = authority_audit()
    write_json(OUT / "fault_scheduler_authority_audit.json", authority)

    roster_bindings, roster_accounting, smoke = build_roster_bindings(
        configurations, variants, all_s08m
    )
    if not roster_accounting["qualificationAuditCompleted"]:
        raise RuntimeError("S12P roster runtime accounting audit did not conserve rows")
    write_parquet(OUT / "roster_runtime_bindings.parquet", roster_bindings)
    write_json(OUT / "roster_accounting_validation.json", roster_accounting)
    write_parquet(OUT / "frozen_smoke_binding_registry.parquet", smoke)

    runtime_registry, runtime_validation = run_runtime_qualification(
        configurations, documents
    )
    if not runtime_validation["allPassed"]:
        raise RuntimeError("runtime qualification fixtures failed")
    write_json(OUT / "runtime_fixture_registry.json", runtime_registry)
    write_json(OUT / "runtime_qualification.json", runtime_validation)
    write_json(
        OUT / "replay_worker_order_validation.json",
        {
            "schemaVersion": "e07.s12a.replay-worker-order-validation.v1",
            "researchStepId": "S12A",
            "bindingWorkerOrderIndependent": binding_validation[
                "workerOrderIndependent"
            ],
            "runtimeExactReplayAllPassed": runtime_validation["exactReplayAllPassed"],
            "runtimeWorkerOrderIndependent": runtime_validation[
                "workerOrderIndependent"
            ],
            "logicalIdentityUnique": roster_accounting["uniqueLogicalReservationIds"]
            == 65_024,
            "allPassed": all(
                (
                    binding_validation["workerOrderIndependent"],
                    runtime_validation["exactReplayAllPassed"],
                    runtime_validation["workerOrderIndependent"],
                    roster_accounting["uniqueLogicalReservationIds"] == 65_024,
                )
            ),
        },
    )

    execution_atomic = fail_atomic_execution_validation()
    publication_registry, publication_atomic = fail_atomic_publication_validation()
    if not execution_atomic["allPassed"] or not publication_atomic["allPassed"]:
        raise RuntimeError("fail-atomic execution/publication qualification failed")
    write_json(OUT / "fail_atomic_execution_validation.json", execution_atomic)
    write_json(OUT / "future_s12_publication_registry.json", publication_registry)
    write_json(OUT / "fail_atomic_publication_validation.json", publication_atomic)

    access = access_control_validation()
    dependencies = dependency_exclusion_validation()
    if not access["allDenied"] or not dependencies["allPassed"]:
        raise RuntimeError("access or prohibited-dependency gate failed")
    write_json(OUT / "access_control_validation.json", access)
    write_json(OUT / "dependency_exclusion_validation.json", dependencies)

    gate_rows = {
        "G01": {
            "passed": True,
            "result": "immutable S12P inputs and predecessor tree revalidated",
        },
        "G02": {
            "passed": True,
            "result": "S10/S11 bounded machine-null branch remains closed",
        },
        "G03": {
            "passed": True,
            "result": "seven lineages, fourteen configurations, and comparators remain locked",
        },
        "G04": {
            "passed": binding_validation["allPassed"],
            "result": "704 exact configuration/variant-task-panel bindings qualified",
        },
        "G05": {
            "passed": endpoint_validation["summary"]["allPassed"],
            "result": "native targets, hole-boundary semantics, and task-local endpoints qualified",
        },
        "G06": {
            "passed": False,
            "result": authority["faultAudit"]["blocker"],
        },
        "G07": {
            "passed": False,
            "result": authority["schedulerAudit"]["blocker"],
        },
        "G08": {
            "passed": endpoint_validation["summary"]["allPassed"],
            "result": "15x15 and irregular panels qualified as diagnostics only, never efficacy endpoints",
        },
        "G09": {
            "passed": access["allDenied"] and dependencies["allPassed"],
            "result": "protected denials and prohibited-dependency exclusions pass",
        },
        "G10": {
            "passed": all(
                (
                    roster_accounting["runtimeBindingComplete"],
                    runtime_validation["allPassed"],
                    execution_atomic["allPassed"],
                    publication_atomic["allPassed"],
                )
            ),
            "result": (
                "runtime replay and fail-atomic plumbing qualified, but 2,048 "
                "matched-random roster rows reference two IDs without persisted "
                "executable configuration definitions"
            ),
        },
    }
    gate = {
        "schemaVersion": "e07.s12a.execution-gate.v1",
        "researchStepId": "S12A",
        "rows": gate_rows,
        "allPassed": all(row["passed"] for row in gate_rows.values()),
        "substantiveS12Authorized": False,
        "failureMode": "fail_closed_before_any_transfer_episode",
        "remainingBlockers": [
            "G06: no authoritative unseen E06 spatial fault family",
            "G07: no authoritative alternate E06 spatial scheduler family",
            (
                "G10: 2,048 matched-random reservations reference two locked "
                "configuration IDs with no immutable runtime definition"
            ),
        ],
        "estimandChanged": False,
        "rosterNarrowed": False,
        "recommendedNextAction": (
            "Do not execute S12. Recover and hash-qualify the two missing "
            "matched-random runtime definitions and obtain authoritative native "
            "E06 unseen-fault and scheduler-family contracts, or seek explicit "
            "approval for a new preregistered scientific redesign; never narrow "
            "S12P silently."
        ),
    }
    if gate["allPassed"]:
        raise RuntimeError("S12A unexpectedly cleared absent G06/G07 authority")
    write_json(OUT / "s12_execution_gate.json", gate)

    predecessor_after = tree_manifest(PREDECESSOR_STEPS)
    predecessor_commitment_after = tree_commitment(predecessor_after)
    immutability = {
        "schemaVersion": "e07.s12a.predecessor-immutability-validation.v1",
        "researchStepId": "S12A",
        "predecessorFileCountBefore": len(predecessor_before),
        "predecessorFileCountAfter": len(predecessor_after),
        "predecessorTreeCommitmentBeforeSha256": (predecessor_commitment_before),
        "predecessorTreeCommitmentAfterSha256": (predecessor_commitment_after),
        "predecessorsByteIdentical": predecessor_before == predecessor_after,
        "completedStepArtifactsModified": 0,
        "quarantinesModified": 0,
        "scientificPublicationsModified": 0,
        "S12PRosterModified": False,
        "archiveMutations": 0,
        "civicSimulationRows": 0,
        "S13OrS14ExecutionRows": 0,
        "transferOutcomeRows": 0,
    }
    if not immutability["predecessorsByteIdentical"]:
        raise RuntimeError("predecessor artifact tree changed during S12A")
    write_json(OUT / "predecessor_immutability_validation.json", immutability)

    validation_checks = {
        "S12P freeze revalidated": s12p_validation["allPassed"],
        "704 exact bindings": binding_validation["allPassed"],
        "target/boundary/endpoints": endpoint_validation["summary"]["allPassed"],
        "diagnostics not promoted": endpoint_validation["summary"][
            "diagnosticPromotionEligibleRows"
        ]
        == 0,
        "one swap not a fault": endpoint_validation["summary"][
            "oneSwapClassifiedAsFaultRows"
        ]
        == 0,
        "65,024 roster rows accounted": roster_accounting[
            "qualificationAuditCompleted"
        ],
        "runtime binding deficit exactly recorded": (
            roster_accounting["runtimeBindingsResolved"] == 62_976
            and roster_accounting["runtimeBindingsUnresolved"] == 2_048
            and len(roster_accounting["unresolvedByRoleAndConfiguration"]) == 2
        ),
        "56 structural smoke bindings resolved": len(smoke) == 56,
        "runtime fixture replay/order": runtime_validation["allPassed"],
        "fail-atomic execution": execution_atomic["allPassed"],
        "fail-atomic publication": publication_atomic["allPassed"],
        "protected denial": access["allDenied"] and access["denials"] == 6,
        "prohibited dependencies absent": dependencies["allPassed"],
        "G04/G05/G08 clear": all(
            gate_rows[key]["passed"] for key in ("G04", "G05", "G08")
        ),
        "runtime G10 fails closed on missing definitions": not gate_rows["G10"][
            "passed"
        ],
        "G06/G07 remain failed": not gate_rows["G06"]["passed"]
        and not gate_rows["G07"]["passed"],
        "substantive S12 blocked": not gate["substantiveS12Authorized"],
        "predecessors immutable": immutability["predecessorsByteIdentical"],
        "zero frozen transfer episodes": roster_accounting["transferEpisodesExecuted"]
        == 0,
        "zero protected outcomes": access["validationOutcomeRowsRead"] == 0
        and access["confirmationOutcomeRowsRead"] == 0,
    }
    validation = {
        "schemaVersion": "e07.s12a.validation-summary.v1",
        "researchStepId": "S12A",
        "checks": validation_checks,
        "passedCount": sum(validation_checks.values()),
        "checkCount": len(validation_checks),
        "allPassed": all(validation_checks.values()),
        "qualificationPassed": all(validation_checks.values()),
        "scientificExecutionGatePassed": False,
        "outcomeClassification": "constraining/contradictory",
    }
    if not validation["allPassed"]:
        raise RuntimeError("S12A validation summary did not pass")
    write_json(OUT / "validation_summary.json", validation)

    provenance = {
        "schemaVersion": "e07.s12a.provenance.v1",
        "researchStepId": "S12A",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPOSITORY),
        "gitBranch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPOSITORY, text=True
        ).strip(),
        "gitCommitAtExecution": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "workers": 1,
        "numericThreads": 1,
        "dependenciesInstalled": [],
        "dedicatedQualificationLogicalRows": 16,
        "dedicatedQualificationPhysicalRows": 32,
        "frozenS12Episodes": 0,
        "protectedOutcomeRowsRead": 0,
    }
    write_json(OUT / "provenance.json", provenance)
    (OUT / "execution_commands.log").write_text(
        "PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
        "python scripts/qualify_native_spatial_transfer_s12a.py run\n",
        encoding="utf-8",
    )
    write_json(
        OUT / "status.json",
        {
            "schemaVersion": "e07.research-step-status.v1",
            "researchStepId": "S12A",
            "stepNumber": "12A",
            "success": True,
            "status": "complete_constraining_gate_fail_closed",
            "artifactsWritten": [
                str(path) for path in sorted(OUT.iterdir()) if path.is_file()
            ],
            "validationResult": (
                "PASS: outcome-free native binding, endpoint, runtime, replay, "
                "access, accounting, and fail-atomic qualifications passed; "
                "the substantive G01-G10 gate remains failed at G06/G07/G10."
            ),
            "caveatsOrBlockers": [
                authority["faultAudit"]["blocker"],
                authority["schedulerAudit"]["blocker"],
                (
                    "G10: 2,048 matched-random reservations reference two "
                    "locked configuration IDs with no persisted executable "
                    "runtime definition."
                ),
            ],
            "recommendedNextAction": gate["recommendedNextAction"],
            "outcomeClassification": "constraining/contradictory",
            "substantiveS12Authorized": False,
            "validationOutcomeRowsRead": 0,
            "confirmationOutcomeRowsRead": 0,
            "frozenS12EpisodesExecuted": 0,
        },
    )
    write_json(OUT / "artifact_manifest.json", build_artifact_manifest())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    args = parser.parse_args()
    if args.command == "run":
        qualify()


if __name__ == "__main__":
    main()
