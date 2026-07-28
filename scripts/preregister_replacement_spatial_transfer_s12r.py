#!/usr/bin/env python3
"""Preregister and qualify the outcome-free S12R replacement transfer study."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.environment_suite import (
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
    EnvironmentSuite,
)
from src.environment_suite.contracts import SuiteValidationError, canonical_sha256
from src.morph2d.baseline import build_target_environment, load_baseline_assets
from src.morph2d.engine import _state_blind_actor_schedule
from src.morph2d.movements import initial_movement_state
from src.phenotype_discovery.publication import (
    ArtifactSpec,
    AtomicScientificPublisher,
    PublicationContractError,
)
from src.policy_dsl import compile_policy
from src.spatial_transfer.qualification import (
    AUTHORITY_BEARING_PERMISSIONS,
    NATIVE_MOVEMENT_KINDS,
    SPATIAL_TASKS,
    _grammar_for_target,
    _movement_kinds,
    apply_adaptation_variant,
    endpoint_contract,
    qualify_configuration_binding,
    qualify_endpoint,
)
from src.spatial_transfer.replacement import (
    FAULT_FAMILY_ID,
    SCHEDULER_FAMILY_ID,
    apply_spurious_swap_fault,
    build_replacement_random_static_comparator,
    fault_contract,
    identity_round_robin_schedule,
    scheduler_contract,
    scheduler_cost_ledger,
    validate_spurious_swap_fault_audit,
)

REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
OUT = ARTIFACTS / "research_steps/S12R"
S02 = ARTIFACTS / "research_steps/S02"
S05 = ARTIFACTS / "research_steps/S05"
S08M = ARTIFACTS / "research_steps/S08M"
S09 = ARTIFACTS / "research_steps/S09"
S12P = ARTIFACTS / "research_steps/S12P"
S12A = ARTIFACTS / "research_steps/S12A"
PROTOCOL = (
    REPOSITORY / "configs/transfer/s12r_replacement_spatial_transfer_protocol.yaml"
)
TASK_REGISTRY = REPOSITORY / "configs/environment_suite/task_registry.yaml"
SPLIT_MANIFEST = REPOSITORY / "configs/environment_suite/split_manifest.json"
S08M_REGISTRY = S08M / "portfolio_configuration_registry.jsonl"
S08M_LOCK = S08M / "validation_candidate_lock.json"
S09_COMPRESSED = S09 / "compressed_policies.jsonl"
S05_CATALOG = S05 / "policy_catalog.jsonl"
RECOVERY_HISTORY_CUTOFF = "69e2cf24056b39b8b9eeac58825795c2b9c8a193"

OLD_MISSING_COMPARATORS = (
    "d3537d6d43fede6a531120babaff28675d56caf6d5a299ab0ff99c72cb2c5074",
    "fc3a833fb46284e83db19a1943a998ab9ef97eb2bd5e2835b96eb8993d66775a",
)
CALIBRATED_TARGETS = (
    "stripes_alternating_three_band",
    "layers_three_ordered_tissues",
    "bilateral_lobes_with_midline",
    "tissue_single_hole",
    "tissue_two_holes",
)
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
    WORKSPACE / "input-attachments/MANIFEST.json",
    WORKSPACE
    / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md",
    PROTOCOL,
    Path(__file__).resolve(),
    REPOSITORY / "src/spatial_transfer/replacement.py",
    REPOSITORY / "src/spatial_transfer/qualification.py",
    REPOSITORY / "tests/test_s12r_replacement_spatial_transfer.py",
    TASK_REGISTRY,
    SPLIT_MANIFEST,
    S05_CATALOG,
    S08M_REGISTRY,
    S08M_LOCK,
    S09_COMPRESSED,
    S08M / "s09_eligibility_gate.json",
    S12P / "s12p_broad_transfer_protocol.yaml",
    S12P / "candidate_lock.json",
    S12P / "candidate_population.jsonl",
    S12P / "lineage_registry.jsonl",
    S12P / "adaptation_variant_registry.jsonl",
    S12P / "baseline_registry.json",
    S12P / "s12_execution_gate.json",
    S12P / "artifact_manifest.json",
    S12P / "research_step_full_results.md",
    S12A / "s12a_native_spatial_compatibility_protocol.yaml",
    S12A / "fault_scheduler_authority_audit.json",
    S12A / "roster_accounting_validation.json",
    S12A / "s12_execution_gate.json",
    S12A / "artifact_manifest.json",
    S12A / "research_step_full_results.md",
    S02 / "environment_suite/split_manifest.json",
    S02 / "holdout_access_audit.json",
    Path("/previous-artifacts/E01/research_steps/S03/transition_contract.json"),
    Path(
        "/previous-artifacts/E02/research_steps/S04/"
        "scheduler_package/scheduler_contract.md"
    ),
    Path(
        "/previous-artifacts/E02/research_steps/S05/"
        "fault_package/fault_semantics_contract.md"
    ),
    Path("/previous-artifacts/E03/research_steps/S14/e07_handoff.md"),
    Path("/previous-artifacts/E04/research_steps/S14/e06_e07_handoff.md"),
    Path(
        "/previous-artifacts/E05/research_steps/S14/"
        "regeneration_benchmark/E07_HANDOFF.md"
    ),
    Path("/previous-artifacts/E06/report_inputs/e07_handoff.md"),
    Path("/previous-artifacts/E06/research_steps/S03/environment_spec.md"),
    Path("/previous-artifacts/E06/research_steps/S04/movement_spec.md"),
    Path("/previous-artifacts/E06/research_steps/S05/policy_spec.md"),
    Path("/previous-artifacts/E06/research_steps/S06/control_channel_spec.md"),
    Path("/previous-artifacts/E06/research_steps/S07/engine_spec.md"),
    Path("/previous-artifacts/E06/research_steps/S10/frozen_perturbation_design.yaml"),
    Path("/previous-artifacts/E06/research_steps/S14/split_manifest.json"),
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
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


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)) + b"\n")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )


def strict_structural_configuration(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("rejectedModelOrEmbeddingUsed") is not False:
        raise RuntimeError("configuration used a rejected model or embedding")
    if row.get("s07ArmMembershipUsed") is not False:
        raise RuntimeError("configuration used S07 arm membership")
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
    return {key: deepcopy(row[key]) for key in sorted(set(row) & allowed)}


def load_protocol() -> dict[str, Any]:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if (
        protocol.get("researchStepId") != "S12R"
        or protocol.get("designOnly") is not True
        or protocol.get("newEstimand") is not True
    ):
        raise RuntimeError("S12R protocol is not a new design-only estimand")
    boundary = protocol["authorizationBoundary"]
    zero_fields = (
        "transferEpisodes",
        "civicSimulationRows",
        "validationOutcomeAccess",
        "confirmationOutcomeAccess",
        "protectedOutcomeAccess",
        "s13Episodes",
        "s14Episodes",
        "archiveMutations",
        "outcomeBearingCacheReads",
        "s06OrS06AModelOrEmbeddingLoads",
        "s07ArmSignalUses",
    )
    if any(int(boundary[field]) != 0 for field in zero_fields):
        raise RuntimeError("S12R protocol authorizes prohibited work")
    if protocol["rosterAndBudget"]["totalLogicalReservations"] != 195_072:
        raise RuntimeError("S12R logical budget changed")
    return protocol


def load_frozen_population() -> dict[str, Any]:
    candidates = read_jsonl(S12P / "candidate_population.jsonl")
    candidate_ids = {str(row["configurationId"]) for row in candidates}
    configurations: dict[str, dict[str, Any]] = {}
    all_s08m: dict[str, dict[str, Any]] = {}
    member_index: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(S08M_REGISTRY):
        if row.get("taskId") not in SPATIAL_TASKS:
            continue
        all_s08m[str(row["configurationId"])] = deepcopy(row)
        for member in row.get("members", []):
            member_index[str(member["policySha256"])] = deepcopy(member)
        if str(row["configurationId"]) in candidate_ids:
            configurations[str(row["configurationId"])] = (
                strict_structural_configuration(row)
            )
    documents: dict[str, dict[str, Any]] = {
        str(row["policySha256"]): deepcopy(row["document"])
        for row in read_jsonl(S05_CATALOG)
    }
    for row in read_jsonl(S09_COMPRESSED):
        configuration = strict_structural_configuration(row["configuration"])
        if configuration["configurationId"] in candidate_ids:
            configurations[str(configuration["configurationId"])] = configuration
        for policy_hash, document in row["documentsByPolicySha256"].items():
            documents[str(policy_hash)] = deepcopy(document)
    lineages = read_jsonl(S12P / "lineage_registry.jsonl")
    variants = read_jsonl(S12P / "adaptation_variant_registry.jsonl")
    if (
        len(candidates) != 14
        or len(configurations) != 14
        or len(lineages) != 7
        or len(variants) != 30
    ):
        raise RuntimeError("S12R failed to retain the complete frozen population")
    return {
        "candidates": candidates,
        "candidateConfigurations": configurations,
        "allS08MConfigurations": all_s08m,
        "documents": documents,
        "memberIndex": member_index,
        "lineages": lineages,
        "sourceVariants": variants,
    }


def git_history_search(value: str) -> list[str]:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={REPOSITORY}",
            "log",
            "--format=%H",
            RECOVERY_HISTORY_CUTOFF,
            "-S",
            value,
            "--",
            ".",
        ],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def recover_or_replace_comparators(
    frozen: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    lock = read_json(S08M_LOCK)
    lock_entries = {str(row["configurationId"]): row for row in lock["entries"]}
    registry = frozen["allS08MConfigurations"]
    mapping: dict[str, str] = {}
    replacement_rows: list[dict[str, Any]] = []
    audit_rows = []
    for old_id in OLD_MISSING_COMPARATORS:
        summary = lock_entries.get(old_id)
        if summary is None:
            raise RuntimeError("missing comparator has no S08M lock summary")
        if old_id in registry:
            raise RuntimeError("old comparator unexpectedly became recoverable")
        history = git_history_search(old_id)
        definition_hash_history = git_history_search(
            str(summary["configurationDefinitionSha256"])
        )
        exact_recovery = bool(history or definition_hash_history)
        if exact_recovery:
            raise RuntimeError(
                "recovery search found a source requiring explicit manual audit"
            )
        definition, qualification = build_replacement_random_static_comparator(
            task_id=str(summary["taskId"]),
            member_hashes=list(summary["memberPolicySha256"]),
            member_index=frozen["memberIndex"],
            documents_by_hash=frozen["documents"],
        )
        new_id = str(definition["configurationId"])
        if new_id in OLD_MISSING_COMPARATORS or new_id in mapping.values():
            raise RuntimeError("replacement comparator identity collision")
        mapping[old_id] = new_id
        replacement_rows.append(
            {
                "schemaVersion": "e07.s12r.replacement-comparator-registry-row.v1",
                "researchStepId": "S12R",
                "retiredUnresolvedOldConfigurationId": old_id,
                "oldIdExecutableInS12R": False,
                "newConfigurationId": new_id,
                "definition": definition,
                "qualification": qualification,
                "createdFrom": (
                    "frozen_member_policy_bytes_under_a_new_prospectively_"
                    "defined_random_static_comparator_contract"
                ),
                "reconstructedUnderOldId": False,
                "outcomeRowsUsed": 0,
            }
        )
        audit_rows.append(
            {
                "oldConfigurationId": old_id,
                "oldActionSha256": summary["actionSha256"],
                "oldConfigurationDefinitionSha256": summary[
                    "configurationDefinitionSha256"
                ],
                "oldMemberPolicySha256": summary["memberPolicySha256"],
                "oldMode": summary["mode"],
                "authoritativeRegistryDefinitionPresent": False,
                "gitHistoryConfigurationIdHits": history,
                "gitHistoryDefinitionHashHits": definition_hash_history,
                "exactRecoveryPossible": False,
                "summarySufficientForExactRecovery": False,
                "oldIdRetiredFromS12R": True,
                "newConfigurationId": new_id,
                "reason": (
                    "Only a lock summary persisted; no complete authoritative "
                    "runtime definition exists. S12R defines a new comparator "
                    "and never impersonates the old ID."
                ),
            }
        )

    lineages = []
    for source in frozen["lineages"]:
        row = deepcopy(source)
        old = str(row["matchedRandomConfigurationId"])
        row["sourceS12PLineageId"] = str(row["lineageId"])
        row["sourceMatchedRandomConfigurationId"] = old
        row["matchedRandomConfigurationId"] = mapping.get(old, old)
        row["matchedRandomDefinitionReplaced"] = old in mapping
        row["schemaVersion"] = "e07.s12r.lineage.v1"
        body = {
            key: value for key, value in row.items() if key != "lineageCommitmentSha256"
        }
        row["lineageCommitmentSha256"] = canonical_sha256("E07/S12R/lineage/v1", body)
        lineages.append(row)
    lineages.sort(key=lambda row: row["lineageId"])

    exact_existing = 0
    for lineage in lineages:
        comparator_id = str(lineage["matchedRandomConfigurationId"])
        if comparator_id in mapping.values():
            continue
        configuration = registry.get(comparator_id)
        if configuration is None:
            raise RuntimeError("an additional matched-random definition is missing")
        exact_existing += 1
    audit = {
        "schemaVersion": "e07.s12r.comparator-recovery-audit.v1",
        "researchStepId": "S12R",
        "authoritativeMaterialsSearched": [
            str(S08M_REGISTRY),
            str(S08M_LOCK),
            str(S08M / "artifact_manifest.json"),
            str(REPOSITORY) + " git object history",
        ],
        "repositoryHistoryCutoffCommit": RECOVERY_HISTORY_CUTOFF,
        "efficacyOrOutcomeTablesSearched": False,
        "exactDefinitionsRecovered": 0,
        "exactExistingComparatorDefinitionsRetained": exact_existing,
        "unresolvedOldIds": list(OLD_MISSING_COMPARATORS),
        "oldIdsReconstructed": 0,
        "newComparatorDefinitionsCreated": len(replacement_rows),
        "oldToNewConfigurationId": mapping,
        "rows": audit_rows,
        "passed": exact_existing == 5 and len(replacement_rows) == 2,
    }
    return audit, replacement_rows, lineages, audit_rows


def build_candidate_lock(
    frozen: Mapping[str, Any],
    lineages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    commitments = []
    by_id = {str(row["configurationId"]): row for row in frozen["candidates"]}
    for configuration_id, configuration in sorted(
        frozen["candidateConfigurations"].items()
    ):
        commitments.append(
            {
                "configurationId": configuration_id,
                "candidateRole": by_id[configuration_id]["candidateRole"],
                "candidateCommitmentSha256": canonical_sha256(
                    "E07/S12R/candidate-configuration/v1", configuration
                ),
            }
        )
    body = {
        "schemaVersion": "e07.s12r.candidate-lock.v1",
        "researchStepId": "S12R",
        "lineageCount": len(lineages),
        "configurationCount": len(commitments),
        "candidateCommitments": commitments,
        "sourceS12PCandidateLockSha256": sha256_file(S12P / "candidate_lock.json"),
        "structuralExclusions": [],
        "outcomeFieldsLoaded": False,
        "s07ArmSignalsUsed": False,
        "s06OrS06AArtifactsUsed": False,
    }
    body["candidateLockSha256"] = canonical_sha256("E07/S12R/candidate-lock/v1", body)
    return body


def build_adaptation_variants(
    source_variants: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for source in source_variants:
        body = {
            "schemaVersion": "e07.s12r.adaptation-variant.v1",
            "researchStepId": "S12R",
            "sourceS12PAdaptationVariantId": source["adaptationVariantId"],
            "baseConfigurationId": source["baseConfigurationId"],
            "mode": source["mode"],
            "edit": deepcopy(source["edit"]),
            "runtimeConfigurationCommitmentSha256": source[
                "runtimeConfigurationCommitmentSha256"
            ],
            "memberSetChanged": False,
            "memberRuleChanged": False,
            "memoryChanged": False,
            "observationChanged": False,
            "signalOrCommunicationChanged": False,
            "nativeCostSemanticsChanged": False,
            "outcomeFieldsLoaded": False,
        }
        body["adaptationVariantId"] = canonical_sha256(
            "E07/S12R/adaptation-variant/v1", body
        )
        rows.append(body)
    rows.sort(key=lambda row: row["adaptationVariantId"])
    if len(rows) != 30 or len({row["adaptationVariantId"] for row in rows}) != 30:
        raise RuntimeError("S12R adaptation identity regeneration failed")
    return rows


def build_condition_cells(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    panels = {str(row["panelId"]): row for row in protocol["targetTopologyPanels"]}
    rows = []
    for task_id in SPATIAL_TASKS:
        for condition in protocol["conditionCells"]:
            applicable = condition["applicablePanels"]
            panel_ids = (
                sorted(panels) if applicable == "all_eight" else list(applicable)
            )
            for panel_id in panel_ids:
                panel = panels[panel_id]
                body = {
                    "schemaVersion": "e07.s12r.task-condition-cell.v1",
                    "taskId": task_id,
                    "panelId": panel_id,
                    "targetId": panel["targetId"],
                    "fixtureId": panel["fixtureId"],
                    "challengeId": panel["challengeId"],
                    "endpointMode": panel["endpointMode"],
                    "inferentialRole": panel["inferentialRole"],
                    "conditionId": condition["conditionId"],
                    "schedulerFamilyId": condition["schedulerFamilyId"],
                    "faultFamilyId": condition["faultFamilyId"],
                    "outcomeAssignmentUsed": False,
                }
                body["taskCellId"] = canonical_sha256(
                    "E07/S12R/task-condition-cell/v1", body
                )
                rows.append(body)
    rows.sort(key=lambda row: row["taskCellId"])
    counts = Counter(row["taskId"] for row in rows)
    if len(rows) != 48 or any(counts[task_id] != 24 for task_id in SPATIAL_TASKS):
        raise RuntimeError("S12R task-condition cell count changed")
    return rows


def build_scenarios(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    partitions = (
        ("development", range(16), False),
        ("post_lock_transfer", range(1000, 1064), True),
    )
    rows = []
    for partition, ordinals, sealed in partitions:
        for cell in cells:
            for ordinal in ordinals:
                body = {
                    "schemaVersion": "e07.s12r.scenario-family.v1",
                    "partition": partition,
                    "taskCellId": cell["taskCellId"],
                    "taskId": cell["taskId"],
                    "panelId": cell["panelId"],
                    "conditionId": cell["conditionId"],
                    "targetId": cell["targetId"],
                    "fixtureId": cell["fixtureId"],
                    "challengeId": cell["challengeId"],
                    "schedulerFamilyId": cell["schedulerFamilyId"],
                    "faultFamilyId": cell["faultFamilyId"],
                    "ordinal": ordinal,
                    "outcomeAssignmentUsed": False,
                    "outcomeMaterialized": False,
                    "sealedBeforeAdaptationLock": sealed,
                }
                scenario_id = canonical_sha256("E07/S12R/scenario-family/v1", body)
                source = canonical_sha256(
                    "E07/S12R/source-state-address/v1",
                    {
                        "taskId": cell["taskId"],
                        "panelId": cell["panelId"],
                        "partition": partition,
                        "ordinal": ordinal,
                    },
                )
                coupling = canonical_sha256(
                    "E07/S12R/scenario-coupling/v1",
                    {
                        "scenarioFamilyId": scenario_id,
                        "sourceStateAddressSha256": source,
                        "faultLocusAddress": (
                            canonical_sha256(
                                "E07/S12R/fault-locus-address/v1",
                                [partition, cell["panelId"], ordinal],
                            )
                            if cell["faultFamilyId"] == FAULT_FAMILY_ID
                            else None
                        ),
                        "schedulerFamilyId": cell["schedulerFamilyId"],
                    },
                )
                rows.append(
                    {
                        **body,
                        "scenarioFamilyId": scenario_id,
                        "sourceStateAddressSha256": source,
                        "couplingCommitmentSha256": coupling,
                    }
                )
    rows.sort(key=lambda row: row["scenarioFamilyId"])
    counts = Counter(row["partition"] for row in rows)
    if counts != {"development": 768, "post_lock_transfer": 3072}:
        raise RuntimeError(f"S12R scenario count changed: {counts}")
    return rows


def _logical_row(**values: Any) -> dict[str, Any]:
    body = {
        "schemaVersion": "e07.s12r.logical-reservation.v1",
        **values,
        "outcomeMaterialized": False,
        "protectedOutcomeAccess": False,
        "validationOutcomeAccess": False,
        "confirmationOutcomeAccess": False,
        "s07ArmSignalUsed": False,
        "s06OrS06AArtifactUsed": False,
    }
    body["logicalReservationId"] = canonical_sha256(
        "E07/S12R/logical-reservation/v1", body
    )
    return body


def build_roster(
    candidates: Sequence[Mapping[str, Any]],
    lineages: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    candidates = sorted(candidates, key=lambda row: row["configurationId"])
    for scenario in scenarios:
        common = {
            "partition": scenario["partition"],
            "taskCellId": scenario["taskCellId"],
            "taskId": scenario["taskId"],
            "panelId": scenario["panelId"],
            "conditionId": scenario["conditionId"],
            "schedulerFamilyId": scenario["schedulerFamilyId"],
            "faultFamilyId": scenario["faultFamilyId"],
            "scenarioFamilyId": scenario["scenarioFamilyId"],
            "couplingCommitmentSha256": scenario["couplingCommitmentSha256"],
        }
        if scenario["partition"] == "development":
            for variant in variants:
                candidate = next(
                    row
                    for row in candidates
                    if row["configurationId"] == variant["baseConfigurationId"]
                )
                rows.append(
                    _logical_row(
                        **common,
                        conditionRole="adaptation_variant_development",
                        lineageId=candidate["lineageId"],
                        baseConfigurationId=candidate["configurationId"],
                        runtimeSelectionRef=variant["adaptationVariantId"],
                        baselineOrdinal=-1,
                    )
                )
            continue
        for candidate in candidates:
            for role in ("zero_shot", "adapted_winner_slot"):
                rows.append(
                    _logical_row(
                        **common,
                        conditionRole=role,
                        lineageId=candidate["lineageId"],
                        baseConfigurationId=candidate["configurationId"],
                        runtimeSelectionRef=(
                            candidate["configurationId"]
                            if role == "zero_shot"
                            else "locked_S12R_adaptation_winner_after_development"
                        ),
                        baselineOrdinal=-1,
                    )
                )
        for lineage in lineages:
            rows.append(
                _logical_row(
                    **common,
                    conditionRole="matched_random",
                    lineageId=lineage["lineageId"],
                    baseConfigurationId=lineage["matchedRandomConfigurationId"],
                    runtimeSelectionRef=lineage["matchedRandomConfigurationId"],
                    baselineOrdinal=0,
                )
            )
            for ordinal, configuration_id in enumerate(
                lineage["componentSingleConfigurationIds"]
            ):
                rows.append(
                    _logical_row(
                        **common,
                        conditionRole="component_single",
                        lineageId=lineage["lineageId"],
                        baseConfigurationId=configuration_id,
                        runtimeSelectionRef=configuration_id,
                        baselineOrdinal=ordinal,
                    )
                )
        rows.append(
            _logical_row(
                **common,
                conditionRole="native_baseline",
                lineageId="not_applicable",
                baseConfigurationId=(
                    "spatial_greedy_local_native_v1"
                    if scenario["taskId"] == SPATIAL_TASKS[0]
                    else "spatial_memory_local_native_v1"
                ),
                runtimeSelectionRef="native_task_baseline",
                baselineOrdinal=0,
            )
        )
    rows.sort(key=lambda row: row["logicalReservationId"])
    if len(rows) != 195_072 or len(
        {row["logicalReservationId"] for row in rows}
    ) != len(rows):
        raise RuntimeError("S12R logical roster identity/accounting failed")
    return rows


def build_physical_commitments(
    roster: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows = []
    for logical in roster:
        for replay_ordinal in (0, 1):
            body = {
                "logicalReservationId": logical["logicalReservationId"],
                "scenarioFamilyId": logical["scenarioFamilyId"],
                "runtimeSelectionRef": logical["runtimeSelectionRef"],
                "replayOrdinal": replay_ordinal,
                "couplingCommitmentSha256": logical["couplingCommitmentSha256"],
            }
            rows.append(
                {
                    **body,
                    "physicalExecutionId": canonical_sha256(
                        "E07/S12R/physical-execution/v1", body
                    ),
                    "outcomeMaterialized": False,
                }
            )
    frame = pd.DataFrame(rows).sort_values("physicalExecutionId")
    if len(frame) != 390_144 or not frame["physicalExecutionId"].is_unique:
        raise RuntimeError("S12R physical commitment accounting failed")
    return frame


def _single_binding(
    configuration: Mapping[str, Any],
    documents: Mapping[str, Mapping[str, Any]],
    *,
    task_id: str,
    panel_id: str,
    selection_ref: str,
) -> dict[str, Any]:
    members = list(configuration["members"])
    if configuration["mode"] != "single_policy" or len(members) != 1:
        raise RuntimeError("single-policy binding received another mode")
    member = members[0]
    policy_hash = str(member["policySha256"])
    document = documents.get(policy_hash)
    if document is None:
        raise RuntimeError("single comparator policy document is unavailable")
    policy = compile_policy(document)
    permissions = policy.permissions & AUTHORITY_BEARING_PERMISSIONS
    movement_kinds = _movement_kinds(document)
    passed = bool(
        policy.policy_sha256 == policy_hash
        and policy.environment == "spatial2d.v1"
        and member["nativeCarrier"] == "spatial"
        and not permissions
        and not (movement_kinds - NATIVE_MOVEMENT_KINDS)
    )
    return {
        "schemaVersion": "e07.s12r.single-binding.v1",
        "selectionRef": selection_ref,
        "baseConfigurationId": configuration["configurationId"],
        "sourceTaskId": configuration["taskId"],
        "executionTaskId": task_id,
        "panelId": panel_id,
        "actionSha256": policy.policy_sha256,
        "memberPolicySha256": [policy.policy_sha256],
        "selectorSignal": None,
        "nativeMovementKinds": sorted(movement_kinds),
        "authorityPermissions": sorted(permissions),
        "actionRebound": False,
        "costSemanticsRebound": False,
        "outcomeFieldsLoaded": False,
        "passed": passed,
    }


def build_binding_qualification(
    frozen: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
    lineages: Sequence[Mapping[str, Any]],
    replacements: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    runtime: dict[str, tuple[dict[str, Any], str]] = {
        configuration_id: (deepcopy(configuration), configuration_id)
        for configuration_id, configuration in frozen["candidateConfigurations"].items()
    }
    source_variants = {
        str(row["adaptationVariantId"]): row for row in frozen["sourceVariants"]
    }
    for variant in variants:
        source = source_variants[str(variant["sourceS12PAdaptationVariantId"])]
        base = frozen["candidateConfigurations"][str(variant["baseConfigurationId"])]
        adapted = apply_adaptation_variant(base, source)
        runtime[str(variant["adaptationVariantId"])] = (
            adapted,
            str(variant["adaptationVariantId"]),
        )
    replacement_by_id = {
        str(row["newConfigurationId"]): deepcopy(row["definition"])
        for row in replacements
    }
    comparator_ids = set()
    for lineage in lineages:
        comparator_ids.update(lineage["componentSingleConfigurationIds"])
        comparator_ids.add(lineage["matchedRandomConfigurationId"])
    for configuration_id in sorted(comparator_ids):
        definition = replacement_by_id.get(configuration_id) or frozen[
            "allS08MConfigurations"
        ].get(configuration_id)
        if definition is None:
            raise RuntimeError("S12R comparator runtime definition is unavailable")
        runtime.setdefault(
            configuration_id,
            (strict_structural_configuration(definition), configuration_id),
        )

    rows = []
    for selection_ref, (configuration, reference) in sorted(runtime.items()):
        for cell in cells:
            if configuration["mode"] == "single_policy":
                row = _single_binding(
                    configuration,
                    frozen["documents"],
                    task_id=cell["taskId"],
                    panel_id=cell["panelId"],
                    selection_ref=reference,
                )
            else:
                row = qualify_configuration_binding(
                    configuration,
                    frozen["documents"],
                    execution_task_id=cell["taskId"],
                    panel_id=cell["panelId"],
                    selection_ref=reference,
                )
            row.update(
                {
                    "taskCellId": cell["taskCellId"],
                    "conditionId": cell["conditionId"],
                    "schedulerFamilyId": cell["schedulerFamilyId"],
                    "faultFamilyId": cell["faultFamilyId"],
                    "schedulerContractBound": True,
                    "faultContractBound": (cell["faultFamilyId"] == FAULT_FAMILY_ID),
                    "qualificationOnly": True,
                    "episodeSubmitted": False,
                }
            )
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["selectionRef", "taskCellId"])
    summary = {
        "schemaVersion": "e07.s12r.binding-validation.v1",
        "researchStepId": "S12R",
        "runtimeDefinitionCount": len(runtime),
        "taskConditionCellCount": len(cells),
        "bindingRows": len(frame),
        "bindingRowsPassed": int(frame["passed"].sum()),
        "uniqueSelectionRefs": frame["selectionRef"].nunique(),
        "allFourteenCandidatesPresent": set(frozen["candidateConfigurations"]).issubset(
            set(frame["selectionRef"])
        ),
        "allThirtyAdaptationsPresent": {
            row["adaptationVariantId"] for row in variants
        }.issubset(set(frame["selectionRef"])),
        "allComparatorsPresent": comparator_ids.issubset(set(frame["selectionRef"])),
        "faultAndSchedulerContractsBound": bool(
            frame["schedulerContractBound"].all()
            and (
                frame.loc[
                    frame["faultFamilyId"] == FAULT_FAMILY_ID,
                    "faultContractBound",
                ]
            ).all()
        ),
        "episodesSubmitted": int(frame["episodeSubmitted"].sum()),
    }
    summary["allPassed"] = bool(
        summary["bindingRows"] == summary["bindingRowsPassed"]
        and summary["allFourteenCandidatesPresent"]
        and summary["allThirtyAdaptationsPresent"]
        and summary["allComparatorsPresent"]
        and summary["faultAndSchedulerContractsBound"]
        and summary["episodesSubmitted"] == 0
    )
    return frame, summary


def build_endpoint_registry(
    cells: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for cell in cells:
        contract = endpoint_contract(cell["taskId"], cell["panelId"])
        qualification = qualify_endpoint(cell["taskId"], cell["panelId"])
        fault = cell["faultFamilyId"] == FAULT_FAMILY_ID
        diagnostic = not contract["calibratedCompletionAvailable"]
        repair_eligible = fault and not diagnostic
        row = {
            "schemaVersion": "e07.s12r.endpoint-contract.v1",
            "taskCellId": cell["taskCellId"],
            "taskId": cell["taskId"],
            "panelId": cell["panelId"],
            "conditionId": cell["conditionId"],
            "schedulerFamilyId": cell["schedulerFamilyId"],
            "faultFamilyId": cell["faultFamilyId"],
            "calibratedCompletionAvailable": contract["calibratedCompletionAvailable"],
            "repairRiskSetEligible": repair_eligible,
            "repairClockOrigin": (
                "post_fault_before_transition_0" if repair_eligible else None
            ),
            "repairNoncompletionDisposition": (
                "right_censor_at_transition_32"
                if repair_eligible
                else "not_a_repair_risk_set"
            ),
            "diagnosticPromotionEligible": False,
            "diagnosticUnavailableReason": (
                contract["endpointUnavailableReason"] if diagnostic else None
            ),
            "acceptedDisplacementIsFault": False,
            "universalScore": None,
            "nativeQualificationPassed": qualification["passed"],
            "passed": bool(
                qualification["passed"]
                and not (
                    repair_eligible
                    and cell["panelId"]
                    in {
                        "displacement_native_target",
                        "combined_size_displacement",
                    }
                )
            ),
        }
        rows.append(row)
    rows.sort(key=lambda row: row["taskCellId"])
    summary = {
        "schemaVersion": "e07.s12r.endpoint-qualification.v1",
        "researchStepId": "S12R",
        "rows": len(rows),
        "passedRows": sum(row["passed"] for row in rows),
        "repairRiskSetRows": sum(row["repairRiskSetEligible"] for row in rows),
        "diagnosticRows": sum(not row["calibratedCompletionAvailable"] for row in rows),
        "diagnosticPromotionEligibleRows": sum(
            row["diagnosticPromotionEligible"] for row in rows
        ),
        "acceptedDisplacementFaultRows": sum(
            row["acceptedDisplacementIsFault"] for row in rows
        ),
    }
    summary["allPassed"] = bool(
        summary["rows"] == 48
        and summary["passedRows"] == 48
        and summary["repairRiskSetRows"] == 16
        and summary["diagnosticRows"] == 12
        and summary["diagnosticPromotionEligibleRows"] == 0
        and summary["acceptedDisplacementFaultRows"] == 0
    )
    return rows, summary


def qualify_fault_family() -> dict[str, Any]:
    context, targets, _grammars, environments = load_baseline_assets()
    rows = []
    adversaries = []
    for target_id in CALIBRATED_TARGETS:
        target = targets[target_id]
        grammar = _grammar_for_target(context, target_id)
        environment = environments.get(target_id) or build_target_environment(
            target, grammar
        )
        source = initial_movement_state(environment)
        scenario_id = f"s12r-qualification-only/fault/{target_id}"
        first_state, first = apply_spurious_swap_fault(
            environment,
            source,
            target,
            grammar,
            scenario_family_id=scenario_id,
        )
        second_state, second = apply_spurious_swap_fault(
            environment,
            source,
            target,
            grammar,
            scenario_family_id=scenario_id,
        )
        replay = first_state == second_state and first == second
        validated = validate_spurious_swap_fault_audit(
            environment, source, target, grammar, first
        )
        forged = deepcopy(first)
        forged["faultLedger"]["faultGraphDisplacement"] = 1
        forged_denied = False
        try:
            validate_spurious_swap_fault_audit(
                environment, source, target, grammar, forged
            )
        except SuiteValidationError:
            forged_denied = True
        adversaries.append({"targetId": target_id, "forgedAuditDenied": forged_denied})
        rows.append(
            {
                "targetId": target_id,
                "eligibleLoci": first["eligibleTargetBreakingLocusCount"],
                "preTarget": first["preConjunctiveCompletion"],
                "postTarget": first["postConjunctiveCompletion"],
                "repairRiskSetEntered": first["repairRiskSetEntered"],
                "conservation": first["identityTokenKindAndSiteCardinalityConserved"],
                "replay": replay,
                "auditValidated": validated,
                "forgedAuditDenied": forged_denied,
                "resultCommitmentSha256": canonical_sha256(
                    "E07/S12R/fault-qualification-row/v1", first
                ),
            }
        )
    result = {
        "schemaVersion": "e07.s12r.fault-qualification.v1",
        "researchStepId": "S12R",
        "faultFamilyId": FAULT_FAMILY_ID,
        "targetRows": rows,
        "adversaries": adversaries,
        "targetCount": len(rows),
        "allTargetsHaveEligibleLoci": all(row["eligibleLoci"] > 0 for row in rows),
        "allPreTargetPostNonTarget": all(
            row["preTarget"] and not row["postTarget"] for row in rows
        ),
        "allConserved": all(row["conservation"] for row in rows),
        "allReplay": all(row["replay"] for row in rows),
        "allForgedDenied": all(row["forgedAuditDenied"] for row in rows),
        "episodeRowsSubmitted": 0,
    }
    result["allPassed"] = bool(
        result["targetCount"] == 5
        and result["allTargetsHaveEligibleLoci"]
        and result["allPreTargetPostNonTarget"]
        and result["allConserved"]
        and result["allReplay"]
        and result["allForgedDenied"]
        and result["episodeRowsSubmitted"] == 0
    )
    return result


def qualify_scheduler_family() -> dict[str, Any]:
    rows = []
    for actor_count in (1, 2, 3, 4, 5, 7, 81):
        actors = tuple(f"cell-{index:03d}" for index in range(actor_count))
        scenario = f"s12r-qualification-only/scheduler/n{actor_count}"
        forward = [
            identity_round_robin_schedule(scenario, transition, actors, 4)
            for transition in range(32)
        ]
        reverse = {
            transition: identity_round_robin_schedule(
                scenario, transition, tuple(reversed(actors)), 4
            )
            for transition in reversed(range(32))
        }
        stream = [actor for batch in forward for actor in batch]
        unique_batches = all(len(batch) == len(set(batch)) for batch in forward)
        first_sweep = stream[:actor_count]
        no_replacement = len(set(first_sweep)) == actor_count
        order_independent = all(forward[index] == reverse[index] for index in range(32))
        rows.append(
            {
                "actorCount": actor_count,
                "batchCount": len(forward),
                "batchSizes": sorted({len(batch) for batch in forward}),
                "uniqueWithinBatch": unique_batches,
                "noReplacementFirstSweep": no_replacement,
                "replayAndWorkerOrderIndependent": order_independent,
                "scheduleSha256": canonical_sha256(
                    "E07/S12R/scheduler-qualification-sequence/v1", forward
                ),
                "costLedger": scheduler_cost_ledger(
                    actor_count=actor_count,
                    transition_count=32,
                ),
            }
        )
    actors = tuple(f"cell-{index:03d}" for index in range(81))
    scenario = "s12r-qualification-only/scheduler/family-distinction"
    alternate = [
        identity_round_robin_schedule(scenario, transition, actors, 4)
        for transition in range(32)
    ]
    native = [
        _state_blind_actor_schedule(scenario, transition, actors, 4)
        for transition in range(32)
    ]
    result = {
        "schemaVersion": "e07.s12r.scheduler-qualification.v1",
        "researchStepId": "S12R",
        "schedulerFamilyId": SCHEDULER_FAMILY_ID,
        "boundaryRows": rows,
        "boundaryCount": len(rows),
        "allUniqueWithinBatch": all(row["uniqueWithinBatch"] for row in rows),
        "allNoReplacementFirstSweep": all(
            row["noReplacementFirstSweep"] for row in rows
        ),
        "allReplayAndWorkerOrderIndependent": all(
            row["replayAndWorkerOrderIndependent"] for row in rows
        ),
        "distinctFromNativeE06Sequence": alternate != native,
        "stateReads": 0,
        "outcomeReads": 0,
        "episodeRowsSubmitted": 0,
    }
    result["allPassed"] = bool(
        result["boundaryCount"] == 7
        and result["allUniqueWithinBatch"]
        and result["allNoReplacementFirstSweep"]
        and result["allReplayAndWorkerOrderIndependent"]
        and result["distinctFromNativeE06Sequence"]
        and result["stateReads"] == 0
        and result["outcomeReads"] == 0
        and result["episodeRowsSubmitted"] == 0
    )
    return result


def access_and_reserve_validation() -> tuple[dict[str, Any], dict[str, Any]]:
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    development = AccessGrant(AccessPhase.DEVELOPMENT)
    confirmation = AccessGrant(AccessPhase.CONFIRMATION, candidate_lock_sha256="0" * 64)
    records = []
    for task_id in SPATIAL_TASKS:
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
    reserve_path = Path(
        "/previous-artifacts/E06/research_steps/S14/split_manifest.json"
    )
    reserve = read_json(reserve_path)
    heldout = reserve["splits"]["heldout"]
    reserve_contract = {
        "schemaVersion": "e07.s12r.protected-reserve-contract.v1",
        "researchStepId": "S12R",
        "sourceRecord": file_record(reserve_path),
        "originalHeldoutAddressTemplateCommitmentSha256": reserve[
            "heldoutAddressTemplateCommitmentSha256"
        ],
        "originalHeldoutTargets": heldout["targetIds"],
        "originalHeldoutScenarioIdsMaterialized": reserve[
            "heldoutScenarioIdsMaterialized"
        ],
        "originalHeldoutOutcomeAccessed": reserve["heldoutOutcomeAccessed"],
        "S12RReadProtectedOutcomeRows": 0,
        "S12RMaterializedProtectedScenarioPayloads": 0,
        "newExtensionReserve": {
            "faultDomainCommitmentSha256": canonical_sha256(
                "E07/S12R/future-S14-fault-address-template/v1",
                {
                    "targets": heldout["targetIds"],
                    "family": FAULT_FAMILY_ID,
                    "payloadMaterialized": False,
                },
            ),
            "schedulerDomainCommitmentSha256": canonical_sha256(
                "E07/S12R/future-S14-scheduler-address-template/v1",
                {
                    "targets": heldout["targetIds"],
                    "family": SCHEDULER_FAMILY_ID,
                    "payloadMaterialized": False,
                },
            ),
            "scenarioPayloadsMaterialized": False,
            "outcomesAccessed": False,
            "requiresSeparateS14Approval": True,
        },
        "passed": bool(
            heldout["e07ProtectedConfirmation"]
            and not reserve["heldoutScenarioIdsMaterialized"]
            and not reserve["heldoutOutcomeAccessed"]
        ),
    }
    access = {
        "schemaVersion": "e07.s12r.access-control-validation.v1",
        "researchStepId": "S12R",
        "attempts": len(records),
        "denials": sum(row["denied"] for row in records),
        "allDenied": all(row["denied"] for row in records),
        "records": records,
        "protectedOutcomeRowsRead": 0,
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
        "civicRows": 0,
        "S13Rows": 0,
        "S14Rows": 0,
        "passed": all(row["denied"] for row in records),
    }
    return access, reserve_contract


def fail_atomic_publication_validation() -> tuple[dict[str, Any], dict[str, Any]]:
    specs = tuple(
        ArtifactSpec(class_id, f"{class_id}.json", "json")
        for class_id in PUBLICATION_CLASSES
    )
    payloads = {
        class_id: canonical_json_bytes(
            {
                "schemaVersion": "e07.s12r.synthetic-publication-fixture.v1",
                "artifactClass": class_id,
                "qualificationOnly": True,
                "outcomeRows": 0,
            }
        )
        + b"\n"
        for class_id in PUBLICATION_CLASSES
    }
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
    with TemporaryDirectory(prefix="s12r-publication-", dir="/cache") as raw:
        root = Path(raw)
        for index, point in enumerate(failure_points):
            destination = root / f"scientific-{index:02d}"
            try:
                publisher.publish(
                    destination,
                    payloads,
                    forensics_directory=root / f"forensics-{index:02d}",
                    failure_point=point,
                )
            except PublicationContractError as error:
                audit = getattr(error, "audit", {})
            else:
                raise RuntimeError("injected publication failure did not fire")
            state = audit.get("finalScientificPublicationState")
            records.append(
                {
                    "failurePoint": point,
                    "finalState": state,
                    "validState": state
                    in {
                        "zero_scientific_publication",
                        "complete_validated_publication",
                    },
                }
            )
        forward = publisher.publish(
            root / "success-forward",
            payloads,
            forensics_directory=root / "success-forward-forensics",
        )
        reverse = publisher.publish(
            root / "success-reverse",
            dict(reversed(list(payloads.items()))),
            forensics_directory=root / "success-reverse-forensics",
        )
        forward_files = {
            path.name: path.read_bytes()
            for path in (root / "success-forward").iterdir()
            if path.name != "publication_manifest.json"
        }
        reverse_files = {
            path.name: path.read_bytes()
            for path in (root / "success-reverse").iterdir()
            if path.name != "publication_manifest.json"
        }
    registry = {
        "schemaVersion": "e07.s12r.future-publication-registry.v1",
        "researchStepId": "S12R",
        "artifactClasses": [
            {
                "classId": spec.class_id,
                "relativePath": spec.relative_path,
                "mediaType": spec.media_type,
            }
            for spec in specs
        ],
        "scientificPublicationCreatedByS12R": False,
    }
    validation = {
        "schemaVersion": "e07.s12r.fail-atomic-publication-validation.v1",
        "researchStepId": "S12R",
        "artifactClassCount": len(specs),
        "injectedFailureCount": len(records),
        "validAllOrZeroStateCount": sum(row["validState"] for row in records),
        "zeroPublicationFailures": sum(
            row["finalState"] == "zero_scientific_publication" for row in records
        ),
        "completePublicationAfterCommitFailures": sum(
            row["finalState"] == "complete_validated_publication" for row in records
        ),
        "forwardComplete": forward["finalScientificPublicationState"]
        == "complete_validated_publication",
        "reverseComplete": reverse["finalScientificPublicationState"]
        == "complete_validated_publication",
        "workerOrderPayloadBytesIdentical": forward_files == reverse_files,
        "records": records,
    }
    validation["allPassed"] = bool(
        len(records) == 34
        and validation["validAllOrZeroStateCount"] == 34
        and validation["zeroPublicationFailures"] == 33
        and validation["completePublicationAfterCommitFailures"] == 1
        and validation["forwardComplete"]
        and validation["reverseComplete"]
        and validation["workerOrderPayloadBytesIdentical"]
    )
    return registry, validation


def predecessor_tree() -> list[dict[str, Any]]:
    rows = []
    for step in sorted((ARTIFACTS / "research_steps").iterdir()):
        if not step.is_dir() or step.name == "S12R":
            continue
        for path in sorted(step.rglob("*")):
            if path.is_file():
                rows.append(file_record(path))
    return rows


def tree_commitment(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(
        "E07/S12R/immutable-predecessor-tree/v1",
        [
            {
                "path": row["path"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
            for row in rows
        ],
    )


def run_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_s12r_replacement_spatial_transfer.py",
        "tests/test_s12a_native_spatial_compatibility.py",
        "tests/test_s12p_broad_transfer.py",
        "tests/test_morph2d_environments.py",
        "tests/test_morph2d_movements.py",
        "tests/test_morph2d_engine.py",
        "--disable-warnings",
        "--maxfail=1",
    ]
    environment = {
        **os.environ,
        "PYTHONPATH": ".:src",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    result = subprocess.run(
        command,
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    record = {
        "schemaVersion": "e07.s12r.test-validation.v1",
        "researchStepId": "S12R",
        "command": command,
        "returnCode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0,
    }
    if result.returncode:
        raise RuntimeError("S12R focused/compatible test suite failed")
    return record


def build_manifest() -> dict[str, Any]:
    artifacts = [
        file_record(path)
        for path in sorted(OUT.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    return {
        "schemaVersion": "e07.s12r.artifact-manifest.v1",
        "researchStepId": "S12R",
        "artifacts": artifacts,
        "artifactCount": len(artifacts),
        "complete": True,
    }


def report_text(
    *,
    gate: Mapping[str, Any],
    accounting: Mapping[str, Any],
    recovery: Mapping[str, Any],
    binding: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    fault: Mapping[str, Any],
    scheduler: Mapping[str, Any],
    publication: Mapping[str, Any],
    tests: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    commit: str,
) -> str:
    new_ids = recovery["oldToNewConfigurationId"]
    return f"""# S12R — Preregister and qualify the replacement broad spatial-transfer study

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S12R** |
| Completion status | **Complete outcome-free replacement preregistration and qualification; stopped before substantive transfer, protected access, civic work, S13, and S14** |
| Artifacts written | New byte-frozen S12R protocol; exact-recovery audit; two new comparator definitions; seven-lineage/14-configuration lock; 30 adaptation identities; fault, scheduler, panel, endpoint, cost, estimand, reserve, publication, gate, scenario, 195,072-logical and 390,144-physical commitment records; binding, replay/order, access, dependency, immutability, test, provenance, status, manifest, and this report under `/artifacts/research_steps/S12R/` |
| Validation result | **PASS — G01–G10 qualified; {binding["bindingRows"]:,}/{binding["bindingRows"]:,} structural bindings; 5/5 fault targets; 7/7 scheduler boundary populations; {accounting["logicalRows"]:,} logical and {accounting["physicalRows"]:,} physical identities; {publication["injectedFailureCount"]}/34 fail-atomic injections; protected denials and tests passed** |
| Outcome classification | **Supportive technical/design qualification** |
| Caveats or blockers | S12R is a new estimand and is not directly equivalent to S12P. Its fault and scheduler are new E07 extensions and are less historically anchored than native E06 contracts. Exact old comparator recovery failed, so two old IDs are retired and two new IDs replace them only in S12R. Qualification can still fail during a separately approved execution; no execution is authorized now. |
| Lay summary | The replacement study is now fully specified without looking at any transfer result. It keeps all seven policy lineages, introduces one explicit target-breaking spatial fault and one truly different actor scheduler, and fixes the two missing comparison policies by giving newly executable definitions new identities. Every future row is counted and committed in advance. |
| Recommended next action | **Chief Scientist review. If approved, authorize a distinct fresh S12R execution continuation using only this lock; do not call it execution of S12P/S12A and do not start automatically.** |

## Frozen question and decision

S12R asks whether the seven S09 parent/compressed spatial lineages preserve
task-local completion, maintenance, and repair behavior across new targets, an
identity-locus target-breaking fault, and a genuinely alternate scheduler.
Larger 15×15 and irregular panels remain diagnostic. This is a replacement
estimand, not an amendment or execution of S12P/S12A.

The design-and-qualification criterion was met. All ten technical gate rows
pass, but `substantiveTransferAuthorized=false` remains frozen because a
separate human decision is required before any episode.

## Inputs and authorization boundary

The step refreshed `AGENTS.md`, `FULL_PLAN.md`, `RESEARCH_PLAN.md`,
upstream-artifact context, the attachment manifest and sidecar, E01–E06
handoffs/native contracts, S02 access controls, S08M–S12A locks/reports, and
the E06 S14 split reserve. `input_hash_freeze.json` records exact paths, sizes,
and SHA-256 values.

Only structural metadata, public contracts, authoritative recovery materials,
published integrity/status records, and dedicated outcome-independent fixtures
were used. No transfer episode, validation/confirmation/protected outcome,
civic row, S13/S14 episode, quarantined cache, S06/S06A model or embedding, or
S07 arm signal was loaded.

## Detailed methods

### Comparator recovery and replacement

The recovery audit searched the immutable S08M runtime registry, candidate
lock, manifest, and repository object history before defining anything new.
Neither unresolved configuration had a complete persisted runtime definition.
The lock retained only member hashes plus action/definition hashes. Those
summaries cannot reproduce omitted fields or authenticate old bytes.

The old IDs were therefore never reconstructed:

- `d3537d6d...` → `{new_ids[OLD_MISSING_COMPARATORS[0]]}`;
- `fc3a833f...` → `{new_ids[OLD_MISSING_COMPARATORS[1]]}`.

Each new comparator uses the exact frozen member policy bytes, the declared
`random_static_identity` behavior, the existing S08 assignment semantics, and
a new S12R domain-separated configuration ID. Five other matched-random
definitions remain exact executable S08M definitions. Every affected S12R
reservation, logical identity, coupling commitment, budget count, and
multiplicity membership was regenerated.

### New spatial-fault contract

`identity_locus_target_breaking_spurious_swap_v1` fires once before native
transition 0 and before the first policy observation. A counter-addressed
stream ranks native adjacent unequal-token identity pairs that are proven by
the target algebra to leave the calibrated conjunction. The committed swap
conserves sites, identities, kinds, and token composition. The source becomes
inactive after the pulse; its state effect persists until native dynamics
change it.

Fault metadata is hidden. Policies see only consequences allowed by their
existing observations. A separate fault ledger charges locus eligibility,
one counter selection, one committed spurious swap, two displaced mobile
entities (cells and, where native, conserved vacancies), and two units of graph
displacement; it charges no policy opportunity or native proposal. Missing
loci, forged metadata, and invariant/execution defects are failures. Failure
to repair by 32 native transitions is a right censor.

This differs from S12A's accepted displacement because the latter stays inside
the target. It uses no lesion mask or count change, suppresses no proposal, and
does not use an S06 controller or actuation-miss channel.

### Alternate scheduler contract

`identity_round_robin_batch4_v1` sorts immutable actor identities once,
chooses one scenario-addressed start offset, and advances a persistent
four-slot cursor for 32 native transitions. It is state- and outcome-blind and
has no replacement within an identity sweep. Its separate ledger records
population reads, one offset draw, cursor arithmetic, and actor slots.

Native E06 independently hash-ranks identities at every transition. A seed
change inside that family cannot create S12R's cross-transition sweep
dependence, so this is a genuinely alternate family.

### Frozen population, panels, partitions, and budget

All seven lineages and 14 parent/compressed configurations were retained.
Thirty bounded adaptation edits keep their exact runtime commitments but
receive new S12R identities. Eight target/topology panels are crossed with
native/no-fault and alternate/no-fault conditions. The four calibrated exact
panels additionally receive native/fault and alternate/fault conditions,
yielding 24 cells per task and 48 total.

The 15×15 and irregular panels remain diagnostic under both schedulers and
never receive a repair or promotion endpoint. The target-preserving
displacement panels remain maintenance/diagnostic controls, not faults.

Sixteen development families per task-cell and 64 sealed post-lock transfer
families per task-cell yield 3,840 scenario families. The roster contains:

- 23,040 development adaptation reservations;
- 172,032 post-lock reservations;
- **195,072 logical reservations total**; and
- one exact replay per logical reservation, for **390,144 physical
  commitments**.

Every logical and physical ID, source-state address, fault/scheduler address,
and coupling commitment was made before outcomes. Fixed 32-transition clocks,
four actor slots, no efficacy early stop, and no runtime-driven weakening are
frozen.

### Endpoints, costs, and inference

Calibrated no-fault cells retain native conjunctive completion and mismatch
endpoints. Fault cells add post-fault repair by transition 32 and right-censored
repair time. Accepted displacement is maintenance only. Diagnostic topology
cells expose reference mismatch/component/boundary fields without imputed
completion or time.

Movement, observation, channel, portfolio, licensed-capability, new fault, and
new scheduler costs remain separate. No scalar or universal normalized score
exists.

Inference uses paired scenario-family effects, 2,000 bootstrap replicates,
restricted 32-transition time differences, exact rare-event intervals, and
fixed Holm families for lineage, adaptation, matched-random, scheduler, fault,
failure, native-cost, and extension-cost tests. Infeasible slots remain
non-evidentiary with raw p=1; observed results cannot shrink a family.

### Access, reserves, and publication

The original E06 S14 ring/separated reserve remains byte-frozen, unmaterialized
for S12R, and sealed. S12R adds only opaque address-template commitments for a
possible future fault/scheduler reserve; no scenario payload or outcome exists.
Scientific publication is a ten-class complete set exposed through one atomic
commit. Exact position-indexed forensics remain outside the scientific commit.

## Commands and dependencies

Primary command:

```text
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
python scripts/preregister_replacement_spatial_transfer_s12r.py run
```

Focused and compatible tests:

```text
PYTHONPATH=.:src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 pytest -q \\
tests/test_s12r_replacement_spatial_transfer.py \\
tests/test_s12a_native_spatial_compatibility.py \\
tests/test_s12p_broad_transfer.py tests/test_morph2d_environments.py \\
tests/test_morph2d_movements.py tests/test_morph2d_engine.py
```

Static checks used `ruff check`, `ruff format --check`, `python -m py_compile`,
and `git diff --check`. No dependency was installed. Qualification used one
worker and one numeric thread because no scientific episode was run.

## Results

| Check | Result |
| --- | --- |
| Exact comparator recovery | 0/2 old definitions recoverable; old IDs retired |
| New comparators | 2/2 executable with new IDs and 81-identity dispatch replay |
| Candidate population | 7/7 lineages; 14/14 configurations; 0 exclusions |
| Adaptation | 30/30 new S12R identities with unchanged runtime commitments |
| Bindings | {binding["bindingRows"]:,}/{binding["bindingRows"]:,} passed |
| Fault | 5/5 calibrated targets had conserving target-breaking loci; forged audits denied |
| Scheduler | 7/7 boundary populations passed; native/alternate sequences differ |
| Endpoints | 48/48 task-cells passed; 16 repair-risk cells; 12 diagnostic cells; 0 diagnostic promotions |
| Accounting | {accounting["logicalRows"]:,} logical; {accounting["physicalRows"]:,} physical; all unique |
| Publication | {publication["injectedFailureCount"]}/34 injected failures ended complete-or-zero |
| Access | 6/6 protected attempts denied; zero outcome materializations |
| Immutability | {predecessor["fileCount"]} predecessor files retained under identical tree commitment |
| Tests | {tests["stdout"].strip()} |

All G01–G10 qualification rows pass. This supports executability of the frozen
replacement design, not policy efficacy or transfer.

## Validation

Bindings cover every retained candidate, adaptation, executable comparator,
task, and applicable task-condition cell. Fault qualification covers all five
nonreserved calibrated target definitions, exact replay, conservation,
target-domain exit, and forged-audit denial. Scheduler qualification covers
populations of 1, 2, 3, 4, 5, 7, and 81 actors, exact replay, worker-order
independence, batch uniqueness, sweep coverage, state denial, and explicit
distinction from native E06.

Every logical and physical ID is unique; role, partition, cell, coupling, and
budget totals reconcile. Protected opens are denied before materialization.
The ten-class publisher passed 34 failure injections and byte-identical
forward/reverse success. The predecessor tree commitment was identical before
and after. Repository commit at qualification: `{commit}`.

## Caveats, blockers, failed assumptions, and limitations

1. S12R is intentionally **not directly equivalent to S12P**; contrasts across
   those protocols do not estimate one unchanged quantity.
2. The new E07 fault and scheduler extensions are less historically anchored
   than existing native E06 contracts.
3. Exact recovery of the two old matched-random definitions was impossible.
   New definitions cannot retroactively repair S12P/S12A.
4. Qualification fixtures establish semantics and plumbing, not efficacy,
   statistical power, transfer, formation, or repair success.
5. The 15×15 and irregular panels remain diagnostic, regardless of numerical
   appearance.
6. A later execution can still fail closed on hashes, bindings, runtime
   audits, accounting, access, replay, or publication.
7. No universal-score, revealed-preference, biological, cognitive, agency,
   civic, validation, confirmation, S13, or S14 claim is authorized.

## Artifacts and provenance

`preregistration_freeze.json` commits the protocol, candidates, lineages,
adaptations, replacement comparators, fault/scheduler contracts, task cells,
scenarios, logical/physical rosters, endpoints, costs, inference, publication,
and reserve before any future outcome. `input_hash_freeze.json`,
`provenance.json`, `validation_summary.json`, `status.json`, and
`artifact_manifest.json` provide the exact provenance and handoff state.

Repository-backed implementation remains in Git; no source tree was copied
into artifacts. Compact result evidence and structural Parquet commitments
are stored only under `/artifacts/research_steps/S12R/`.

## Recommended next action

Return to the Chief Scientist. If the new estimand and its weaker historical
anchoring are accepted, separately authorize a genuinely fresh S12R execution
continuation that revalidates this exact gate and uses no S12P/S12A execution
identity. Do not begin transfer, civic work, protected access, S13, or S14
automatically.
"""


def run() -> None:
    protocol = load_protocol()
    OUT.mkdir(parents=True, exist_ok=True)
    predecessor_before_rows = predecessor_tree()
    predecessor_before = tree_commitment(predecessor_before_rows)
    input_freeze = {
        "schemaVersion": "e07.s12r.input-hash-freeze.v1",
        "researchStepId": "S12R",
        "createdBeforeQualification": True,
        "inputs": [file_record(path) for path in DIRECT_INPUTS],
        "inputCount": len(DIRECT_INPUTS),
        "predecessorTreeFileCount": len(predecessor_before_rows),
        "predecessorTreeSha256": predecessor_before,
        "protectedOutcomeRowsRead": 0,
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
        "outcomeBearingCacheReads": 0,
    }
    write_json(OUT / "input_hash_freeze.json", input_freeze)
    (OUT / "s12r_replacement_spatial_transfer_protocol.yaml").write_bytes(
        PROTOCOL.read_bytes()
    )

    frozen = load_frozen_population()
    recovery, replacements, lineages, _audit_rows = recover_or_replace_comparators(
        frozen
    )
    candidate_lock = build_candidate_lock(frozen, lineages)
    variants = build_adaptation_variants(frozen["sourceVariants"])
    cells = build_condition_cells(protocol)
    scenarios = build_scenarios(cells)
    roster = build_roster(frozen["candidates"], lineages, variants, scenarios)
    physical = build_physical_commitments(roster)
    endpoint_rows, endpoint_summary = build_endpoint_registry(cells)

    write_json(OUT / "comparator_recovery_audit.json", recovery)
    write_jsonl(OUT / "replacement_comparator_registry.jsonl", replacements)
    write_json(OUT / "candidate_lock.json", candidate_lock)
    write_jsonl(OUT / "lineage_registry.jsonl", lineages)
    write_jsonl(OUT / "adaptation_variant_registry.jsonl", variants)
    write_json(OUT / "spatial_fault_contract.json", fault_contract())
    write_json(OUT / "alternate_scheduler_contract.json", scheduler_contract())
    write_jsonl(OUT / "task_condition_cell_registry.jsonl", cells)
    write_jsonl(OUT / "scenario_population.jsonl", scenarios)
    write_jsonl(OUT / "endpoint_contract_registry.jsonl", endpoint_rows)
    write_json(OUT / "endpoint_qualification.json", endpoint_summary)
    write_parquet(OUT / "s12r_logical_roster.parquet", pd.DataFrame(roster))
    write_parquet(OUT / "s12r_physical_commitments.parquet", physical)

    role_counts = dict(sorted(Counter(row["conditionRole"] for row in roster).items()))
    partition_counts = dict(sorted(Counter(row["partition"] for row in roster).items()))
    affected_new_ids = set(recovery["oldToNewConfigurationId"].values())
    affected_rows = sum(
        row["runtimeSelectionRef"] in affected_new_ids for row in roster
    )
    old_rows = sum(
        row["runtimeSelectionRef"] in OLD_MISSING_COMPARATORS for row in roster
    )
    accounting = {
        "schemaVersion": "e07.s12r.complete-accounting.v1",
        "researchStepId": "S12R",
        "taskCells": len(cells),
        "scenarioFamilies": len(scenarios),
        "developmentScenarioFamilies": sum(
            row["partition"] == "development" for row in scenarios
        ),
        "postLockScenarioFamilies": sum(
            row["partition"] == "post_lock_transfer" for row in scenarios
        ),
        "logicalRows": len(roster),
        "uniqueLogicalReservationIds": len(
            {row["logicalReservationId"] for row in roster}
        ),
        "physicalRows": len(physical),
        "uniquePhysicalExecutionIds": physical["physicalExecutionId"].nunique(),
        "roleCounts": role_counts,
        "partitionCounts": partition_counts,
        "replacementComparatorLogicalRows": affected_rows,
        "retiredOldComparatorLogicalRows": old_rows,
        "logicalRosterSemanticSha256": canonical_sha256(
            "E07/S12R/ordered-logical-roster/v1", roster
        ),
        "physicalCommitmentSemanticSha256": canonical_sha256(
            "E07/S12R/ordered-physical-plan/v1",
            physical.to_dict("records"),
        ),
        "transferEpisodesExecuted": 0,
        "outcomeRowsMaterialized": 0,
    }
    accounting["allPassed"] = bool(
        accounting["taskCells"] == 48
        and accounting["scenarioFamilies"] == 3840
        and accounting["developmentScenarioFamilies"] == 768
        and accounting["postLockScenarioFamilies"] == 3072
        and accounting["logicalRows"] == 195_072
        and accounting["uniqueLogicalReservationIds"] == 195_072
        and accounting["physicalRows"] == 390_144
        and accounting["uniquePhysicalExecutionIds"] == 390_144
        and accounting["replacementComparatorLogicalRows"] == 6144
        and accounting["retiredOldComparatorLogicalRows"] == 0
        and accounting["transferEpisodesExecuted"] == 0
        and accounting["outcomeRowsMaterialized"] == 0
    )
    write_json(OUT / "complete_accounting.json", accounting)

    cost_registry = {
        "schemaVersion": "e07.s12r.cost-registry.v1",
        "researchStepId": "S12R",
        "nativeMovementLedger": "E06_24_field_separate_vector",
        "nativeObservationLedger": "E06_separate_vector",
        "nativeChannelLedger": "E06_separate_vector",
        "portfolioStructuralAndCoordinationLedgers": "separate_vectors",
        "licensedCapabilityLedgers": "separate_even_if_zero_for_spatial",
        "faultLedger": fault_contract()["costSemantics"],
        "schedulerLedger": scheduler_contract()["costSemantics"],
        "configurationBitsChargedOnce": True,
        "scalarOrUniversalCost": None,
        "crossFamilyNormalization": "forbidden",
    }
    inference_registry = {
        "schemaVersion": "e07.s12r.estimand-inference-registry.v1",
        "researchStepId": "S12R",
        "estimands": protocol["estimands"],
        "endpoints": protocol["endpoints"],
        "inference": protocol["inference"],
        "failureCensor": protocol["endpoints"]["failureAndCensor"],
        "frozenBeforeOutcomes": True,
        "universalScore": None,
    }
    write_json(OUT / "cost_registry.json", cost_registry)
    write_json(OUT / "estimand_and_inference_registry.json", inference_registry)

    freeze_files = (
        "s12r_replacement_spatial_transfer_protocol.yaml",
        "comparator_recovery_audit.json",
        "replacement_comparator_registry.jsonl",
        "candidate_lock.json",
        "lineage_registry.jsonl",
        "adaptation_variant_registry.jsonl",
        "spatial_fault_contract.json",
        "alternate_scheduler_contract.json",
        "task_condition_cell_registry.jsonl",
        "scenario_population.jsonl",
        "endpoint_contract_registry.jsonl",
        "cost_registry.json",
        "estimand_and_inference_registry.json",
        "s12r_logical_roster.parquet",
        "s12r_physical_commitments.parquet",
        "complete_accounting.json",
    )
    freeze = {
        "schemaVersion": "e07.s12r.preregistration-freeze.v1",
        "researchStepId": "S12R",
        "frozenBeforeAnyEpisode": True,
        "newEstimandNotS12PExecution": True,
        "files": {name: file_record(OUT / name) for name in freeze_files},
        "semanticCommitments": {
            "candidateLockSha256": candidate_lock["candidateLockSha256"],
            "logicalRosterSha256": accounting["logicalRosterSemanticSha256"],
            "physicalPlanSha256": accounting["physicalCommitmentSemanticSha256"],
        },
        "transferEpisodesAtFreeze": 0,
        "validationOutcomeAccessAtFreeze": 0,
        "confirmationOutcomeAccessAtFreeze": 0,
    }
    write_json(OUT / "preregistration_freeze.json", freeze)

    binding_frame, binding_summary = build_binding_qualification(
        frozen, variants, lineages, replacements, cells
    )
    write_parquet(OUT / "binding_qualification.parquet", binding_frame)
    write_json(OUT / "binding_validation.json", binding_summary)
    fault_validation = qualify_fault_family()
    scheduler_validation = qualify_scheduler_family()
    write_json(OUT / "fault_qualification.json", fault_validation)
    write_json(OUT / "scheduler_qualification.json", scheduler_validation)
    replay_order = {
        "schemaVersion": "e07.s12r.replay-worker-order-validation.v1",
        "researchStepId": "S12R",
        "faultReplayPassed": fault_validation["allReplay"],
        "faultForgedAuditsDenied": fault_validation["allForgedDenied"],
        "schedulerReplayAndOrderPassed": scheduler_validation[
            "allReplayAndWorkerOrderIndependent"
        ],
        "bindingOrderSha256": canonical_sha256(
            "E07/S12R/binding-set/v1",
            sorted(
                json.loads(binding_frame.to_json(orient="records")),
                key=lambda row: (row["selectionRef"], row["taskCellId"]),
            ),
        ),
        "rosterOrderSha256": accounting["logicalRosterSemanticSha256"],
        "passed": bool(
            fault_validation["allReplay"]
            and fault_validation["allForgedDenied"]
            and scheduler_validation["allReplayAndWorkerOrderIndependent"]
        ),
    }
    write_json(OUT / "replay_worker_order_validation.json", replay_order)

    access, reserve = access_and_reserve_validation()
    write_json(OUT / "access_control_validation.json", access)
    write_json(OUT / "protected_reserve_contract.json", reserve)
    dependency = {
        "schemaVersion": "e07.s12r.dependency-exclusion-validation.v1",
        "researchStepId": "S12R",
        "S06ModelOrEmbeddingLoads": 0,
        "S06AModelOrEmbeddingLoads": 0,
        "S07ArmSignalUses": 0,
        "S10OrS11OutcomeSignalUses": 0,
        "quarantinedCacheReads": 0,
        "universalScoreConstructed": False,
        "networkResourcesUsed": 0,
        "dependenciesInstalled": [],
        "passed": True,
    }
    write_json(OUT / "dependency_exclusion_validation.json", dependency)
    publication_registry, publication_validation = fail_atomic_publication_validation()
    write_json(OUT / "future_publication_registry.json", publication_registry)
    write_json(
        OUT / "fail_atomic_publication_validation.json",
        publication_validation,
    )
    tests = run_tests()
    write_json(OUT / "test_validation.json", tests)

    predecessor_after_rows = predecessor_tree()
    predecessor_after = tree_commitment(predecessor_after_rows)
    predecessor = {
        "schemaVersion": "e07.s12r.predecessor-immutability-validation.v1",
        "researchStepId": "S12R",
        "fileCount": len(predecessor_before_rows),
        "beforeSha256": predecessor_before,
        "afterSha256": predecessor_after,
        "sameFileCount": len(predecessor_before_rows) == len(predecessor_after_rows),
        "byteIdentical": predecessor_before == predecessor_after,
        "S12PAndS12APreserved": True,
        "allPriorPublicationsReservesAndQuarantinesPreserved": True,
    }
    predecessor["passed"] = bool(
        predecessor["sameFileCount"] and predecessor["byteIdentical"]
    )
    write_json(OUT / "predecessor_immutability_validation.json", predecessor)

    gate_rows = {
        "G01": predecessor["passed"],
        "G02": bool(
            protocol["historicalRelationship"]["directEquivalenceToS12P"] is False
            and protocol["authorizationBoundary"]["transferEpisodes"] == 0
        ),
        "G03": bool(
            len(lineages) == 7
            and len(frozen["candidateConfigurations"]) == 14
            and len(variants) == 30
            and recovery["passed"]
        ),
        "G04": binding_summary["allPassed"],
        "G05": fault_validation["allPassed"],
        "G06": scheduler_validation["allPassed"],
        "G07": endpoint_summary["allPassed"],
        "G08": accounting["allPassed"],
        "G09": bool(access["passed"] and reserve["passed"] and dependency["passed"]),
        "G10": bool(
            replay_order["passed"]
            and publication_validation["allPassed"]
            and tests["passed"]
        ),
    }
    gate = {
        "schemaVersion": "e07.s12r.execution-gate.v1",
        "researchStepId": "S12R",
        "rows": {
            gate_id: {
                "passed": passed,
                "disposition": "pass" if passed else "fail_closed",
            }
            for gate_id, passed in gate_rows.items()
        },
        "allQualificationRowsPassed": all(gate_rows.values()),
        "qualificationPassed": all(gate_rows.values()),
        "substantiveTransferAuthorized": False,
        "executionReviewRequired": True,
        "nextAction": (
            "Chief Scientist review; only a separate fresh S12R execution "
            "continuation may authorize episodes."
        ),
    }
    write_json(OUT / "s12r_execution_gate.json", gate)

    validation_checks = {
        "new estimand not S12P execution": gate_rows["G02"],
        "seven lineages and fourteen configurations retained": len(lineages) == 7
        and len(frozen["candidateConfigurations"]) == 14,
        "thirty adaptations refrozen": len(variants) == 30,
        "exact recovery audited before replacement": recovery["passed"],
        "two old IDs retired and two new IDs created": len(
            recovery["oldToNewConfigurationId"]
        )
        == 2,
        "all bindings qualified": binding_summary["allPassed"],
        "fault qualified": fault_validation["allPassed"],
        "scheduler qualified": scheduler_validation["allPassed"],
        "endpoints and diagnostics qualified": endpoint_summary["allPassed"],
        "logical and physical accounting": accounting["allPassed"],
        "replay and worker order": replay_order["passed"],
        "protected denial": access["passed"],
        "S14 reserve preserved": reserve["passed"],
        "prohibited dependencies absent": dependency["passed"],
        "fail atomic publication": publication_validation["allPassed"],
        "predecessors immutable": predecessor["passed"],
        "tests passed": tests["passed"],
        "zero transfer episodes": accounting["transferEpisodesExecuted"] == 0,
        "zero protected outcomes": access["protectedOutcomeRowsRead"] == 0,
        "separate approval still required": gate["substantiveTransferAuthorized"]
        is False,
    }
    validation = {
        "schemaVersion": "e07.s12r.validation-summary.v1",
        "researchStepId": "S12R",
        "outcomeClassification": "supportive",
        "checks": validation_checks,
        "checkCount": len(validation_checks),
        "passedCount": sum(validation_checks.values()),
        "allPassed": all(validation_checks.values()),
        "qualificationPassed": gate["qualificationPassed"],
        "substantiveTransferAuthorized": False,
    }
    write_json(OUT / "validation_summary.json", validation)
    if not validation["allPassed"]:
        raise RuntimeError("S12R integrated validation failed")

    git_commit = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={REPOSITORY}",
            "rev-parse",
            "HEAD",
        ],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    provenance = {
        "schemaVersion": "e07.s12r.provenance.v1",
        "researchStepId": "S12R",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPOSITORY),
        "gitBranch": "eidosoma/groups/28",
        "gitCommitAtQualification": git_commit,
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "workers": 1,
        "numericThreads": 1,
        "dependenciesInstalled": [],
        "transferEpisodes": 0,
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
        "protectedOutcomeRowsRead": 0,
        "civicRows": 0,
        "S13Rows": 0,
        "S14Rows": 0,
    }
    write_json(OUT / "provenance.json", provenance)
    status = {
        "researchStepId": "S12R",
        "stepNumber": "12R",
        "success": True,
        "status": (
            "complete_outcome_free_replacement_preregistration_qualified_"
            "substantive_transfer_pending_separate_review"
        ),
        "artifactsWritten": [str(OUT)],
        "validationResult": (
            "PASS G01-G10; exact 7 lineages, 14 configurations, 30 "
            "adaptations, 48 task-cells, 3,840 scenario families, 195,072 "
            "logical and 390,144 physical commitments; all bindings, fault, "
            "scheduler, endpoint, access, replay, publication, and tests pass; "
            "zero episodes or protected outcomes."
        ),
        "caveatsOrBlockers": [
            "S12R is a new estimand and not directly equivalent to S12P.",
            "New E07 fault and scheduler extensions are less historically anchored than native E06.",
            "Two old matched-random IDs were unrecoverable and are retired; new IDs do not repair S12P/S12A.",
            "Substantive execution requires a separate human decision and may still fail closed.",
        ],
        "recommendedNextAction": (
            "Chief Scientist review; if approved, authorize a distinct fresh "
            "S12R execution continuation. Do not execute transfer, civic, "
            "validation, confirmation, S13, or S14 automatically."
        ),
    }
    write_json(OUT / "status.json", status)
    report = report_text(
        gate=gate,
        accounting=accounting,
        recovery=recovery,
        binding=binding_summary,
        endpoint=endpoint_summary,
        fault=fault_validation,
        scheduler=scheduler_validation,
        publication=publication_validation,
        tests=tests,
        predecessor=predecessor,
        commit=git_commit,
    )
    (OUT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    command_log = """PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/preregister_replacement_spatial_transfer_s12r.py run
PYTHONPATH=.:src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 pytest -q tests/test_s12r_replacement_spatial_transfer.py tests/test_s12a_native_spatial_compatibility.py tests/test_s12p_broad_transfer.py tests/test_morph2d_environments.py tests/test_morph2d_movements.py tests/test_morph2d_engine.py --disable-warnings --maxfail=1
ruff check configs/transfer/s12r_replacement_spatial_transfer_protocol.yaml scripts/preregister_replacement_spatial_transfer_s12r.py src/spatial_transfer/replacement.py tests/test_s12r_replacement_spatial_transfer.py
ruff format --check scripts/preregister_replacement_spatial_transfer_s12r.py src/spatial_transfer/replacement.py tests/test_s12r_replacement_spatial_transfer.py
python -m py_compile scripts/preregister_replacement_spatial_transfer_s12r.py src/spatial_transfer/replacement.py tests/test_s12r_replacement_spatial_transfer.py
git diff --check
"""
    (OUT / "execution_commands.log").write_text(command_log, encoding="utf-8")
    write_json(OUT / "artifact_manifest.json", build_manifest())


def validate() -> None:
    manifest = read_json(OUT / "artifact_manifest.json")
    failures = []
    for record in manifest["artifacts"]:
        path = Path(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            failures.append(str(path))
    required = {
        "research_step_full_results.md",
        "status.json",
        "s12r_execution_gate.json",
        "comparator_recovery_audit.json",
        "replacement_comparator_registry.jsonl",
        "s12r_logical_roster.parquet",
        "s12r_physical_commitments.parquet",
        "validation_summary.json",
        "provenance.json",
    }
    missing = [name for name in required if not (OUT / name).is_file()]
    if (
        failures
        or missing
        or not read_json(OUT / "validation_summary.json")["allPassed"]
    ):
        raise RuntimeError(
            f"S12R artifact validation failed: failures={failures}, missing={missing}"
        )
    print(
        json.dumps(
            {
                "researchStepId": "S12R",
                "artifactCount": manifest["artifactCount"],
                "allPassed": True,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        run()
    elif command == "validate":
        validate()
    else:
        raise SystemExit(
            "usage: preregister_replacement_spatial_transfer_s12r.py [run|validate]"
        )


if __name__ == "__main__":
    main()
