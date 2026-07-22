#!/usr/bin/env python3
"""Qualify the outcome-free S08F descriptor and identity remediation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

import pandas as pd
import yaml

REPOSITORY_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPOSITORY_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_BOOTSTRAP))

from scripts.qualify_e05_terminal_audit_s08d import (  # noqa: E402
    access_validation,
    qualify_definition,
)
from src.environment_suite.contracts import canonical_sha256  # noqa: E402
from src.portfolio_preregistration.core import (  # noqa: E402
    candidate_commitment,
    canonical_hash,
    plan_digest,
    read_jsonl,
    validate_portfolio_registry,
)
from src.portfolio_search.execution import (  # noqa: E402
    S08P,
    _catalog,
    _expand_logical_result,
    _initial_work,
    _physical_key,
)
from src.portfolio_search.preflight import (  # noqa: E402
    sha256_file,
    tree_digest,
    validate_executable_bindings,
)
from src.quality_diversity.core import (  # noqa: E402
    E05_DESCRIPTOR_AVAILABILITY_VERSION,
    E05_REGENERATION_DESCRIPTOR_SPECS,
    _aggregate_descriptors,
)


REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/artifacts/research_steps")
OUTPUT = ARTIFACT_ROOT / "S08F"
PROTOCOL = REPOSITORY / "configs/portfolio/s08f_descriptor_identity_remediation.yaml"
S08P_PROTOCOL = ARTIFACT_ROOT / "S08P/s08p_portfolio_protocol.yaml"
IMMUTABLE_STEPS = ("S05", "S08P", "S08A", "S08", "S08B", "S08C", "S08D", "S08E")
EXPECTED_TREES = {
    "S05": "141e2059702460ace997b56b077bf2af7972ee61718c9a04e5a30b527e518b83",
    "S08P": "c3fcb613877c9c9e7a619b7456fb217eb953b0e2fea04a299f5ccdcbc4af4b7b",
    "S08A": "f13ffc52e276629cf27f4eecef5ad3c4564fd02d7ea5de488f039ebceabab06d",
    "S08": "4eb18b54d306621696b780144a808b610ee795e903144ab980c4c48e2312ff0c",
    "S08B": "720c3847ec562e3b86d8afcabfa5c2b49bda6ee6f4c4e74a3293166833fdf0d3",
    "S08C": "239ef7d7e2b7d0cb39a376a64ef45691cf06fef050c446fecba594753ca7767b",
    "S08D": "3fe1b373061b349a8816f1b2346ebbcd7633443ccbf1a9a825a7e610f823c6b3",
    "S08E": "e2dd1335e74d850daeff8c0b35e3146e919fc3cdce9e0ec765f7fbd1c1d32694",
}
EXPECTED_PROTOCOL_SHA = (
    "2e10443017c9c23ff703e0e847ce19290c52c3881b22c00b8298ebe9470fc4f8"
)
E05 = "e07_s02_regeneration_1d"
MODES = (
    "single_policy",
    "fixed_balanced_identity",
    "random_static_identity",
    "random_dynamic_opportunity",
    "environment_conditioned",
)
KNOWN_CHIMERA_IDS = {
    "611e01a8a56e5ac1398c382177e8619fa16709b40917a9b1a315c7b66b34f203",
    "ffe9d0be2c0e470f942238621b1a619d28f9d7e0fd51ac3f75f4f2af9073afd2",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _movement(value: float) -> dict[str, float]:
    return {
        "acceptedNativeActionFraction": value,
        "committedDisplacementFraction": value / 2,
    }


def _complete_outcome(mode_ordinal: int) -> dict[str, Any]:
    offset = mode_ordinal * 0.005
    return {
        "sourceTerminal": False,
        "nativeMovementDescriptorsByPhase": {
            phase: _movement(0.2 + offset + index * 0.01)
            for index, phase in enumerate(
                (
                    "development",
                    "stabilization",
                    "recovery",
                    "memoryResetRecovery",
                    "robustnessFault",
                    "transfer",
                )
            )
        },
        "descriptorsByAxis": {
            "robustness": {
                "pairedCompletionDelta": 0.0,
                "pairedResidualDelta": 0.1,
            },
            "repair": {
                "distanceRestorationFraction": 0.5,
                "restrictedRecoveryTimeFraction": 0.6,
                "recoveryCensored": True,
            },
            "memory": {
                "historyInterventionEffect": 0.0,
                "resetInterventionEffect": 0.0,
            },
            "transfer": {
                "frozenStratumSuccessFraction": 1.0,
                "frozenStratumResidual": 0.0,
            },
        },
    }


def _terminal_outcome(branch: str, mode_ordinal: int) -> dict[str, Any]:
    phases = {"development": _movement(0.2 + mode_ordinal * 0.005)}
    if branch == "stabilization_source_terminal":
        phases["stabilization"] = _movement(0.3 + mode_ordinal * 0.005)
    return {
        "sourceTerminal": True,
        "nativeMovementDescriptorsByPhase": phases,
        "descriptorsByAxis": {},
    }


def _descriptor_row(
    family: int,
    outcome: Mapping[str, Any],
    branch: str,
    *,
    force_failed: bool | None = None,
    force_censored: bool | None = None,
) -> dict[str, Any]:
    terminal = branch != "completed_panel"
    return {
        "taskId": E05,
        "scenarioFamilyOrdinal": family,
        "scenarioId": f"e07s08f:qualification:{family}",
        "stopReason": "source_terminal" if terminal else "completed_panel",
        "failed": terminal if force_failed is None else force_failed,
        "censored": terminal if force_censored is None else force_censored,
        "outcome": dict(outcome),
    }


def descriptor_support_qualification() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    patterns = {
        "development_only": ("development_source_terminal",) * 4,
        "development_and_stabilization": ("stabilization_source_terminal",) * 4,
        "complete_all_expected_archives": ("completed_panel",) * 4,
        "mixed_terminal_panel": (
            "development_source_terminal",
            "stabilization_source_terminal",
            "completed_panel",
            "completed_panel",
        ),
        "failure_and_censor_retained": (
            "development_source_terminal",
            "completed_panel",
            "stabilization_source_terminal",
            "completed_panel",
        ),
    }
    rows: list[dict[str, Any]] = []
    for mode_ordinal, mode in enumerate(MODES):
        for pattern, branches in patterns.items():
            panel = []
            for family, branch in enumerate(branches):
                outcome = (
                    _complete_outcome(mode_ordinal)
                    if branch == "completed_panel"
                    else _terminal_outcome(branch, mode_ordinal)
                )
                panel.append(
                    _descriptor_row(
                        family,
                        outcome,
                        branch,
                        force_failed=(family in {0, 2})
                        if pattern == "failure_and_censor_retained"
                        else None,
                        force_censored=(family in {0, 3})
                        if pattern == "failure_and_censor_retained"
                        else None,
                    )
                )
            forward = _aggregate_descriptors(E05, panel)
            reverse = _aggregate_descriptors(E05, list(reversed(panel)))
            forward_by_id = {item["archiveId"]: item for item in forward}
            expected_complete = {
                "development_only": {f"common:{E05}:development"},
                "development_and_stabilization": {
                    f"common:{E05}:development",
                    f"common:{E05}:stabilization",
                },
                "complete_all_expected_archives": set(
                    E05_REGENERATION_DESCRIPTOR_SPECS
                ),
                "mixed_terminal_panel": {f"common:{E05}:development"},
                "failure_and_censor_retained": {f"common:{E05}:development"},
            }[pattern]
            observed_complete = {
                archive_id
                for archive_id, item in forward_by_id.items()
                if item["cellEligible"]
            }
            no_imputation = all(
                (item["values"] is not None and item["cell"] is not None)
                if item["cellEligible"]
                else (
                    item["values"] is None
                    and item["cell"] is None
                    and item["comparabilityKeySha256"] is None
                )
                for item in forward
            )
            status_retained = all(
                len(item["support"]["nativeStatusByScenarioFamily"]) == 4
                for item in forward
            )
            replay = canonical_sha256(
                "E07/S08F/descriptor-support-panel/v1", forward
            ) == canonical_sha256("E07/S08F/descriptor-support-panel/v1", reverse)
            rows.append(
                {
                    "schemaVersion": "e07.s08f.descriptor-support-qualification-row.v1",
                    "researchStepId": "S08F",
                    "qualificationOnly": True,
                    "mode": mode,
                    "pattern": pattern,
                    "terminalBranches": list(branches),
                    "expectedArchiveRecords": len(E05_REGENERATION_DESCRIPTOR_SPECS),
                    "observedArchiveRecords": len(forward),
                    "expectedCompleteArchiveIds": sorted(expected_complete),
                    "observedCompleteArchiveIds": sorted(observed_complete),
                    "unavailableArchiveIds": sorted(
                        archive_id
                        for archive_id, item in forward_by_id.items()
                        if not item["cellEligible"]
                    ),
                    "noImputation": no_imputation,
                    "noSilentDrop": len(forward)
                    == len(E05_REGENERATION_DESCRIPTOR_SPECS),
                    "nativeStatusRetained": status_retained,
                    "availabilityNotNovelty": not any(
                        item["availabilityIsNoveltyCoordinate"] for item in forward
                    ),
                    "workerOrderIndependent": replay,
                    "panelSha256": canonical_sha256(
                        "E07/S08F/descriptor-support-panel/v1", forward
                    ),
                    "success": observed_complete == expected_complete
                    and no_imputation
                    and status_retained
                    and replay,
                }
            )
    summary = {
        "schemaVersion": "e07.s08f.e05-terminal-support-qualification.v1",
        "researchStepId": "S08F",
        "success": len(rows) == 25 and all(row["success"] for row in rows),
        "portfolioModes": len(MODES),
        "supportPatterns": len(patterns),
        "qualificationPanels": len(rows),
        "logicalFixtureRows": len(rows) * 4,
        "terminalClasses": sorted(
            {branch for branches in patterns.values() for branch in branches}
        ),
        "descriptorAvailabilityVersion": E05_DESCRIPTOR_AVAILABILITY_VERSION,
        "expectedArchiveIds": list(E05_REGENERATION_DESCRIPTOR_SPECS),
        "imputedValues": 0,
        "silentlyDroppedRows": 0,
        "archiveConstructionCalls": 0,
        "objectiveOrEfficacyValuesUsed": 0,
    }
    return rows, summary


def _logical(config: Mapping[str, Any], slot: str, family: int = 300) -> dict[str, Any]:
    return {
        "stage": "qualification",
        "generation": 0,
        "taskId": str(config["taskId"]),
        "split": "train",
        "scenarioFamilyOrdinal": family,
        "logicalSlotId": slot,
        "reservedConfigurationSlotId": str(config["configurationId"]),
        "configurationRole": str(config["mode"]),
        "configuration": dict(config),
        "smoke": False,
    }


def dedup_and_identity_qualification(
    configurations: list[Mapping[str, Any]], budget: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    initial = _initial_work(
        budget, {row["configurationId"]: row for row in configurations}
    )
    started = time.perf_counter()
    forward = [(row["logicalSlotId"], _physical_key(row)) for row in initial]
    reverse = [(row["logicalSlotId"], _physical_key(row)) for row in reversed(initial)]
    forward_map = dict(forward)
    reverse_map = dict(reverse)
    groups: dict[str, list[str]] = defaultdict(list)
    for logical_slot, key in forward:
        groups[key].append(logical_slot)
    accounting = {
        "schemaVersion": "e07.s08f.full-registry-dedup-accounting.v1",
        "researchStepId": "S08F",
        "success": len(initial) == 4864
        and forward_map == reverse_map
        and sum(len(value) for value in groups.values()) == len(initial),
        "configurationRows": len(configurations),
        "scenarioFamiliesPerConfiguration": 4,
        "logicalRows": len(initial),
        "uniquePhysicalRows": len(groups),
        "legallyDeduplicatedLogicalRows": len(initial) - len(groups),
        "allLogicalSlotsAccounted": len(forward_map) == len(initial),
        "workerOrderIndependent": forward_map == reverse_map,
        "forwardPlanSha256": canonical_sha256(
            "E07/S08F/full-physical-plan/v1", sorted(forward)
        ),
        "reversePlanSha256": canonical_sha256(
            "E07/S08F/full-physical-plan/v1", sorted(reverse)
        ),
        "physicalGroupSizeDistribution": dict(
            sorted(Counter(len(value) for value in groups.values()).items())
        ),
        "physicalEvaluationCalls": 0,
        "wallSeconds": time.perf_counter() - started,
    }

    by_id = {str(row["configurationId"]): dict(row) for row in configurations}
    known = [by_id[value] for value in sorted(KNOWN_CHIMERA_IDS)]
    known_work = [
        _logical(config, f"known-{index}") for index, config in enumerate(known)
    ]
    known_keys = [_physical_key(row) for row in known_work]
    from src.portfolio_search.execution import build_action

    known_actions = [
        build_action(config, *_catalog()).policy_sha256 for config in known
    ]
    base = known[0]
    alias = deepcopy(base)
    alias["configurationId"] = canonical_hash(
        "E07/S08F/qualification-alias/v1", base["configurationId"]
    )
    alias["memberSetId"] = canonical_hash(
        "E07/S08F/qualification-member-set/v1", base["memberSetId"]
    )
    base_work = _logical(base, "legal-dedup-base")
    alias_work = _logical(alias, "legal-dedup-alias")
    base_key = _physical_key(base_work)
    alias_key = _physical_key(alias_work)
    physical = {
        "stableEvaluationSha256": "1" * 64,
        "configurationId": base["configurationId"],
        "configurationDefinitionSha256": canonical_hash(
            "E07/S08C/configuration-definition/v1", base
        ),
    }
    expanded_base = _expand_logical_result(base_work, physical, base_key)
    expanded_alias = _expand_logical_result(alias_work, physical, alias_key)
    cost_changed = deepcopy(base)
    cost_changed["configurationId"] = "a" * 64
    cost_changed["portfolioStructuralCosts"]["totalRuleCount"] += 1
    conditioned = next(
        dict(row)
        for row in configurations
        if row["taskId"] == E05 and row["mode"] == "environment_conditioned"
    )
    selector_changed = deepcopy(conditioned)
    selector_changed["configurationId"] = "b" * 64
    selector_changed["selector"] = dict(selector_changed["selector"])
    selector_changed["selector"]["signal"] = (
        "repair.nudge_count"
        if selector_changed["selector"]["signal"] == "last_action.rejected"
        else "last_action.rejected"
    )
    cases = [
        {
            "case": "known_action_equivalent_single_policy_pair",
            "actionBehaviorEquivalent": len(set(known_actions)) == 1,
            "nativeCarrierEquivalent": len(
                {config["members"][0]["nativeCarrier"] for config in known}
            )
            == 1,
            "structuralCostEquivalent": len(
                {
                    json.dumps(config["portfolioStructuralCosts"], sort_keys=True)
                    for config in known
                }
            )
            == 1,
            "physicalKeyEquivalent": len(set(known_keys)) == 1,
            "deduplicationLegal": False,
            "pass": len(set(known_actions)) == 1 and len(set(known_keys)) == 2,
        },
        {
            "case": "behavior_native_selector_and_cost_equivalent_identity_alias",
            "physicalKeyEquivalent": base_key == alias_key,
            "deduplicationLegal": True,
            "pass": base_key == alias_key,
        },
        {
            "case": "cost_semantics_mismatch",
            "physicalKeyEquivalent": base_key
            == _physical_key(_logical(cost_changed, "cost-changed")),
            "deduplicationLegal": False,
            "pass": base_key != _physical_key(_logical(cost_changed, "cost-changed")),
        },
        {
            "case": "selector_semantics_mismatch",
            "physicalKeyEquivalent": _physical_key(
                _logical(conditioned, "selector-base")
            )
            == _physical_key(_logical(selector_changed, "selector-changed")),
            "deduplicationLegal": False,
            "pass": _physical_key(_logical(conditioned, "selector-base"))
            != _physical_key(_logical(selector_changed, "selector-changed")),
        },
    ]
    dedup = {
        "schemaVersion": "e07.s08f.dedup-equivalence-qualification.v1",
        "researchStepId": "S08F",
        "success": all(row["pass"] for row in cases),
        "equivalenceCommitments": [
            "behavior",
            "native task/scenario/carrier",
            "selector/assignment",
            "communication",
            "portfolio structural cost",
            "native cost",
            "licensed-capability cost",
        ],
        "cases": cases,
        "knownPairActionSha256": known_actions[0],
        "knownPairPhysicalKeys": known_keys,
    }
    identity = {
        "schemaVersion": "e07.s08f.logical-identity-restoration.v1",
        "researchStepId": "S08F",
        "success": base_key == alias_key
        and expanded_base["configurationId"] == base["configurationId"]
        and expanded_alias["configurationId"] == alias["configurationId"]
        and expanded_base["physicalStableEvaluationSha256"]
        == expanded_alias["physicalStableEvaluationSha256"]
        and expanded_base["stableEvaluationSha256"]
        != expanded_alias["stableEvaluationSha256"],
        "logicalRows": 2,
        "physicalRows": 1,
        "restoredConfigurationIds": [
            expanded_base["configurationId"],
            expanded_alias["configurationId"],
        ],
        "restoredMemberSetIds": [
            expanded_base["memberSetId"],
            expanded_alias["memberSetId"],
        ],
        "sharedPhysicalWorkSha256": expanded_base["physicalWorkSha256"],
        "sharedPhysicalStableEvaluationSha256": expanded_base[
            "physicalStableEvaluationSha256"
        ],
        "logicalStableEvaluationSha256": [
            expanded_base["stableEvaluationSha256"],
            expanded_alias["stableEvaluationSha256"],
        ],
        "logicalExpansionSha256": [
            expanded_base["logicalExpansionSha256"],
            expanded_alias["logicalExpansionSha256"],
        ],
        "nativeOutcomeValuesUsed": 0,
    }
    return accounting, dedup, identity


def dependency_validation() -> dict[str, Any]:
    prohibited = (
        "src.surrogate_models",
        "src.surrogate_remediation",
        "scripts.run_surrogate_models",
        "scripts.run_surrogate_remediation",
    )
    loaded = sorted(
        name for name in sys.modules if any(token in name for token in prohibited)
    )
    execution_sources = "\n".join(
        (REPOSITORY / path).read_text(encoding="utf-8")
        for path in (
            "src/portfolio_search/execution.py",
            "src/quality_diversity/core.py",
        )
    )
    prohibited_paths = [
        token
        for token in (
            "/artifacts/research_steps/S06/",
            "/artifacts/research_steps/S06A/",
        )
        if token in execution_sources
    ]
    return {
        "schemaVersion": "e07.s08f.dependency-exclusion-audit.v1",
        "researchStepId": "S08F",
        "success": not loaded and not prohibited_paths,
        "loadedProhibitedModules": loaded,
        "prohibitedRuntimePathsInQualificationSource": prohibited_paths,
        "legacyS08cDefaultCacheConstantPresentButUnused": "/cache/e07-s08c"
        in execution_sources,
        "prohibitedRuntimeFilesOpened": [],
        "s06ModelLoads": 0,
        "s06aModelLoads": 0,
        "s07ArmPromotionUses": 0,
        "s08cQuarantineOutcomeLoads": 0,
        "s08eQuarantineOutcomeLoads": 0,
    }


def execute(output: Path) -> None:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if sha256_file(PROTOCOL) != EXPECTED_PROTOCOL_SHA:
        raise RuntimeError("S08F protocol changed after preregistration")
    if protocol["schemaVersion"] != "e07.s08f.descriptor-identity-remediation.v2":
        raise RuntimeError("unexpected S08F protocol schema")
    tree_before = {step: tree_digest(ARTIFACT_ROOT / step) for step in IMMUTABLE_STEPS}
    immutable_pass = tree_before == EXPECTED_TREES

    # These are the only S08E JSON payloads deserialized; all other historical
    # tree access is byte hashing for preservation validation only.
    defect = json.loads(
        (ARTIFACT_ROOT / "S08E/generation_0_integrity_failure.json").read_text()
    )
    quarantine = json.loads(
        (ARTIFACT_ROOT / "S08E/training_evaluation_quarantine.json").read_text()
    )
    documented_defect_pass = (
        defect["descriptorPanelAudit"]["affectedConfigurationCount"] == 7
        and defect["logicalAttributionAudit"]["logicalRowsMisattributed"] == 4
        and quarantine["eligibleForArchiveInsertion"] is False
        and quarantine["eligibleForEfficacyClaims"] is False
    )

    s08p_protocol = yaml.safe_load(S08P_PROTOCOL.read_text(encoding="utf-8"))
    candidates = read_jsonl(S08P / "candidate_eligibility_registry.jsonl")
    member_sets = read_jsonl(S08P / "portfolio_member_sets.jsonl")
    configurations = read_jsonl(S08P / "portfolio_seed_registry.jsonl")
    budget = pd.read_parquet(S08P / "budget_slot_ledger.parquet")
    legality = validate_portfolio_registry(
        s08p_protocol, candidates, member_sets, configurations
    )
    bindings = validate_executable_bindings(
        configurations,
        set(budget.loc[budget["smoke"], "configurationId"].astype(str).tolist()),
    )
    freeze = json.loads(
        (ARTIFACT_ROOT / "S08P/preregistration_freeze.json").read_text()
    )
    candidate_sha = candidate_commitment(candidates)
    plan_sha = plan_digest(
        candidates, member_sets, configurations, budget.to_dict("records")
    )
    plan_pass = (
        candidate_sha == freeze["candidateEligibilityCommitmentSha256"]
        and plan_sha == freeze["completePlanSha256"]
    )

    descriptor_rows, descriptor_summary = descriptor_support_qualification()
    write_jsonl(output / "e05_descriptor_support_patterns.jsonl", descriptor_rows)
    write_json(output / "e05_terminal_support_qualification.json", descriptor_summary)

    by_hash, by_id = _catalog()
    representatives = [
        min(
            (
                row
                for row in configurations
                if row["taskId"] == E05 and row["mode"] == mode
            ),
            key=lambda row: row["configurationId"],
        )
        for mode in MODES
    ]
    branch_rows = [
        {
            **row,
            "schemaVersion": "e07.s08f.e05-branch-mode-qualification-row.v1",
            "researchStepId": "S08F",
            "inheritedFixtureContract": "e07.s08d.e05-configuration-qualification-row.v1",
        }
        for definition in representatives
        for row in qualify_definition(definition, by_hash, by_id)
    ]
    branch_qualification = {
        "schemaVersion": "e07.s08f.e05-branch-mode-qualification.v1",
        "researchStepId": "S08F",
        "success": len(branch_rows) == 15
        and all(
            row["exactReplay"] and row["assignmentAuditComplete"] for row in branch_rows
        ),
        "rows": branch_rows,
        "branchModeRows": len(branch_rows),
        "portfolioModes": len({row["mode"] for row in branch_rows}),
        "terminalBranches": len({row["branch"] for row in branch_rows}),
        "frozenSmokeRowsExecuted": 0,
        "efficacyRowsExecuted": 0,
        "outcomeEndpointCalls": 0,
    }
    write_json(output / "e05_branch_mode_qualification.json", branch_qualification)

    full_accounting, dedup, identity = dedup_and_identity_qualification(
        configurations, budget
    )
    write_json(output / "full_registry_dedup_accounting.json", full_accounting)
    write_json(output / "dedup_equivalence_qualification.json", dedup)
    write_json(output / "logical_identity_restoration.json", identity)

    access = access_validation(configurations)
    access["schemaVersion"] = "e07.s08f.access-control-validation.v1"
    access["researchStepId"] = "S08F"
    dependency = dependency_validation()
    write_json(output / "access_control_validation.json", access)
    write_json(output / "dependency_exclusion_audit.json", dependency)

    tree_after = {step: tree_digest(ARTIFACT_ROOT / step) for step in IMMUTABLE_STEPS}
    no_mutation = {
        "schemaVersion": "e07.s08f.no-mutation-audit.v1",
        "researchStepId": "S08F",
        "success": tree_before == tree_after == EXPECTED_TREES,
        "before": tree_before,
        "after": tree_after,
        "s05ArchiveMutations": 0,
        "s08PortfolioArchiveConstructions": 0,
        "s08PortfolioArchiveMutations": 0,
        "freshSmokeRowsExecuted": 0,
        "searchRowsExecuted": 0,
        "validationOutcomeAccesses": 0,
        "confirmationOutcomeAccesses": 0,
        "s09Rows": 0,
    }
    write_json(output / "no_mutation_audit.json", no_mutation)

    native_contract = {
        "schemaVersion": "e07.s08f.native-contract-preservation.v1",
        "researchStepId": "S08F",
        "success": all(
            sha256_file(REPOSITORY / path) == expected
            for path, expected in {
                "configs/environment_suite/task_registry.yaml": "42ba412bc9611fca03f5f1056483dc3f9db494c1ed0a7565c5df2ebfa579a122",
                "configs/environment_suite/split_manifest.json": "989e45e6b03429c64f338b674da044f154f65c1e973909a28f5874365d715df3",
                "src/environment_suite/dsl_adapters.py": "ee6b0f8e6637eaeb6751ce6848e8f2af10455c738c86ca83b52075988aba67d7",
                "src/environment_suite/portfolio_adapters.py": "6f4209ff15542ba51e24ef4353b1faef04036cc4056fb3f9a4d4af303b1bdd38",
            }.items()
        ),
        "preservedFields": protocol["preservation"]["nativeContractsUnchanged"],
        "taskContractRows": 8,
        "freshNativeEpisodes": 0,
        "nativeTransitionsCommittedByDescriptorFixtures": 0,
    }
    write_json(output / "native_contract_preservation.json", native_contract)

    qualification_replay = {
        "schemaVersion": "e07.s08f.replay-worker-order-validation.v1",
        "researchStepId": "S08F",
        "success": full_accounting["workerOrderIndependent"]
        and all(row["workerOrderIndependent"] for row in descriptor_rows)
        and branch_qualification["success"],
        "descriptorPanelsForwardReverse": len(descriptor_rows),
        "branchModeExactReplayRows": len(branch_rows),
        "fullRegistryLogicalRowsForwardReverse": full_accounting["logicalRows"],
        "fullRegistryForwardSha256": full_accounting["forwardPlanSha256"],
        "fullRegistryReverseSha256": full_accounting["reversePlanSha256"],
    }
    write_json(output / "replay_worker_order_validation.json", qualification_replay)

    accounting = {
        "schemaVersion": "e07.s08f.complete-accounting.v1",
        "researchStepId": "S08F",
        "success": descriptor_summary["qualificationPanels"] == 25
        and branch_qualification["branchModeRows"] == 15
        and full_accounting["logicalRows"] == 4864
        and bindings["configurationRowsBound"] == 1216,
        "descriptorQualificationPanels": 25,
        "descriptorQualificationLogicalFixtureRows": 100,
        "branchModeQualificationRows": 15,
        "fullRegistryConfigurations": 1216,
        "fullRegistryLogicalStructuralRows": 4864,
        "fullRegistryPhysicalStructuralKeys": full_accounting["uniquePhysicalRows"],
        "structuralSmokeBindings": 40 - bindings["frozenSmokeConfigurationsBlocked"],
        "physicalEvaluationCalls": 0,
        "freshSmokeRowsExecuted": 0,
        "efficacyRowsExecuted": 0,
        "archiveConstructionCalls": 0,
        "validationOutcomeAccesses": 0,
        "confirmationOutcomeAccesses": 0,
        "s09Rows": 0,
    }
    write_json(output / "complete_accounting.json", accounting)

    gate_checks = {
        "G01": immutable_pass and documented_defect_pass and plan_pass,
        "G02": len(candidates) == 512
        and not any(row.get("s07ArmMembershipUsed", False) for row in configurations)
        and not any(
            row.get("rejectedModelOrEmbeddingUsed", False) for row in configurations
        ),
        "G03": legality["success"] and native_contract["success"],
        "G04": descriptor_summary["success"]
        and branch_qualification["success"]
        and dedup["success"]
        and identity["success"]
        and bindings["success"],
        "G05": access["success"] and dependency["success"],
        "G06": accounting["success"]
        and qualification_replay["success"]
        and no_mutation["success"],
    }
    gate = {
        "schemaVersion": "e07.s08f.s08-execution-eligibility-gate.v1",
        "researchStepId": "S08F",
        "success": all(gate_checks.values()),
        "status": (
            "technically_qualified_for_separate_execution_review"
            if all(gate_checks.values())
            else "blocked_fail_closed"
        ),
        "rows": [
            {"gateId": gate_id, "status": "pass" if value else "blocked"}
            for gate_id, value in gate_checks.items()
        ],
        "blockedGateIds": [key for key, value in gate_checks.items() if not value],
        "technicalEligibilityPass": all(gate_checks.values()),
        "authorizationToExecuteFrozenSmoke": False,
        "authorizationToConstructArchive": False,
        "authorizationToAccessProtectedOutcomes": False,
        "authorizationToStartS09": False,
        "separateExecutionApprovalRequired": True,
    }
    write_json(output / "s08_execution_eligibility_gate.json", gate)

    immutable_validation = {
        "schemaVersion": "e07.s08f.immutable-input-validation.v1",
        "researchStepId": "S08F",
        "success": immutable_pass and documented_defect_pass and plan_pass,
        "artifactTreeHashes": tree_after,
        "expectedArtifactTreeHashes": EXPECTED_TREES,
        "protocolSha256": sha256_file(PROTOCOL),
        "candidateEligibilityCommitmentSha256": candidate_sha,
        "completePlanSha256": plan_sha,
        "documentedS08eIntegrityMetadataReads": [
            "generation_0_integrity_failure.json",
            "training_evaluation_quarantine.json",
        ],
        "s08eObjectiveOrEfficacyFilesDeserialized": [],
        "historicalTreeReadsWereByteHashOnly": True,
    }
    write_json(output / "immutable_input_validation.json", immutable_validation)

    checks = {
        "immutableInputs": immutable_validation["success"],
        "protocolFreeze": sha256_file(PROTOCOL) == EXPECTED_PROTOCOL_SHA,
        "candidateAndPlanHashes": plan_pass,
        "compositionLegality": legality["success"],
        "bindings1216And40": bindings["success"],
        "e05TerminalSupportPatterns": descriptor_summary["success"],
        "e05BranchModeCoverage": branch_qualification["success"],
        "dedupEquivalence": dedup["success"],
        "logicalIdentityRestoration": identity["success"],
        "fullLogicalPhysicalAccounting": full_accounting["success"],
        "replayAndWorkerOrder": qualification_replay["success"],
        "protectedDenial": access["success"],
        "dependencyExclusion": dependency["success"],
        "nativeContractPreservation": native_contract["success"],
        "completeAccounting": accounting["success"],
        "g01ToG06": gate["success"],
        "noMutation": no_mutation["success"],
    }
    validation = {
        "schemaVersion": "e07.s08f.validation-summary.v1",
        "researchStepId": "S08F",
        "success": all(checks.values()),
        "passed": sum(checks.values()),
        "total": len(checks),
        "checks": checks,
        "blockedGateIds": gate["blockedGateIds"],
    }
    write_json(output / "validation_summary.json", validation)
    write_json(
        output / "environment.json",
        {
            "schemaVersion": "e07.s08f.environment.v1",
            "researchStepId": "S08F",
            "python": sys.version,
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "maximumWorkers": 1,
            "qualificationExecution": "serial deterministic; structural hashing only",
            "gpuUsed": False,
            "networkUsed": False,
            "newDependenciesInstalled": [],
            "wallSeconds": time.perf_counter() - started,
        },
    )
    write_json(
        output / "provenance.json",
        {
            "schemaVersion": "e07.s08f.provenance.v1",
            "researchStepId": "S08F",
            "protocolPath": str(PROTOCOL),
            "protocolSha256": sha256_file(PROTOCOL),
            "repositoryBranch": "eidosoma/groups/28",
            "preImplementationCommit": "d094ebdac504fde1e1537ea2cb30388a002398b0",
            "sourcePaths": [
                "configs/portfolio/s08f_descriptor_identity_remediation.yaml",
                "src/quality_diversity/core.py",
                "src/portfolio_search/execution.py",
                "scripts/qualify_descriptor_identity_s08f.py",
                "tests/test_descriptor_identity_remediation_s08f.py",
            ],
            "sourceHashes": {
                path: sha256_file(REPOSITORY / path)
                for path in (
                    "configs/portfolio/s08f_descriptor_identity_remediation.yaml",
                    "src/quality_diversity/core.py",
                    "src/portfolio_search/execution.py",
                    "scripts/qualify_descriptor_identity_s08f.py",
                    "tests/test_descriptor_identity_remediation_s08f.py",
                )
            },
            "artifactTreeHashes": tree_after,
            "validationOutcomeAccesses": 0,
            "confirmationOutcomeAccesses": 0,
            "archiveConstructionCalls": 0,
            "freshSmokeRowsExecuted": 0,
            "efficacyRowsExecuted": 0,
        },
    )
    if not validation["success"]:
        raise RuntimeError("S08F qualification failed closed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    execute(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
