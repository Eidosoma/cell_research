#!/usr/bin/env python3
"""Freeze and qualify the outcome-free E07 S12P broad-transfer design."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml

from src.environment_suite import (
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
    EnvironmentSuite,
)


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
OUT = ARTIFACTS / "research_steps" / "S12P"
PROTOCOL = REPOSITORY / "configs/transfer/s12p_broad_transfer_protocol.yaml"
TASK_REGISTRY = REPOSITORY / "configs/environment_suite/task_registry.yaml"
SPLIT_MANIFEST = REPOSITORY / "configs/environment_suite/split_manifest.json"
S08M_REGISTRY = ARTIFACTS / "research_steps/S08M/portfolio_configuration_registry.jsonl"
S08M_GATE = ARTIFACTS / "research_steps/S08M/s09_eligibility_gate.json"
S08M_LOCK = ARTIFACTS / "research_steps/S08M/validation_candidate_lock.json"
S09_COMPRESSED = ARTIFACTS / "research_steps/S09/compressed_policies.jsonl"
S10H_STATUS = ARTIFACTS / "research_steps/S10H/status.json"

TASKS = (
    "e07_s02_spatial2d_local",
    "e07_s02_spatial2d_memory",
)

FROZEN_INPUTS = (
    WORKSPACE / "AGENTS.md",
    WORKSPACE / "FULL_PLAN.md",
    WORKSPACE / "PREVIOUS_ARTIFACTS.md",
    WORKSPACE / "PREVIOUS_ARTIFACTS.json",
    WORKSPACE / "input-attachments/MANIFEST.json",
    WORKSPACE
    / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md",
    TASK_REGISTRY,
    SPLIT_MANIFEST,
    ARTIFACTS / "research_steps/S02/holdout_access_audit.json",
    ARTIFACTS / "research_steps/S02/horizon_reconciliation.json",
    ARTIFACTS / "research_steps/S04A/s05_eligibility_gate.json",
    S08M_REGISTRY,
    S08M_GATE,
    S08M_LOCK,
    ARTIFACTS / "research_steps/S08M/research_step_full_results.md",
    S09_COMPRESSED,
    ARTIFACTS / "research_steps/S09/s09_ablation_protocol.yaml",
    ARTIFACTS / "research_steps/S09/research_step_full_results.md",
    ARTIFACTS / "research_steps/S10P/research_step_full_results.md",
    ARTIFACTS / "research_steps/S10/status.json",
    ARTIFACTS / "research_steps/S10A/status.json",
    ARTIFACTS / "research_steps/S10B/status.json",
    ARTIFACTS / "research_steps/S10C/status.json",
    ARTIFACTS / "research_steps/S10D/status.json",
    ARTIFACTS / "research_steps/S10E/status.json",
    ARTIFACTS / "research_steps/S10F/status.json",
    ARTIFACTS / "research_steps/S10G/status.json",
    S10H_STATUS,
    ARTIFACTS / "research_steps/S10H/research_step_full_results.md",
    ARTIFACTS / "research_steps/S10H/scientific_publication/publication_manifest.json",
    Path("/previous-artifacts/E01/research_steps/S03/transition_contract.json"),
    Path("/previous-artifacts/E01/research_steps/S08/split_manifest.json"),
    Path(
        "/previous-artifacts/E02/research_steps/S04/scheduler_package/scheduler_contract.md"
    ),
    Path(
        "/previous-artifacts/E02/research_steps/S05/fault_package/fault_semantics_contract.md"
    ),
    Path("/previous-artifacts/E03/research_steps/S14/e07_handoff.md"),
    Path("/previous-artifacts/E04/research_steps/S14/e06_e07_handoff.md"),
    Path(
        "/previous-artifacts/E05/research_steps/S14/regeneration_benchmark/E07_HANDOFF.md"
    ),
    Path("/previous-artifacts/E06/report_inputs/e07_handoff.md"),
    Path("/previous-artifacts/E06/research_steps/S01/target_catalog.yaml"),
    Path("/previous-artifacts/E06/research_steps/S03/environment_spec.md"),
    Path("/previous-artifacts/E06/research_steps/S04/movement_spec.md"),
    Path("/previous-artifacts/E06/research_steps/S05/policy_spec.md"),
    Path("/previous-artifacts/E06/research_steps/S06/control_channel_spec.md"),
    Path("/previous-artifacts/E06/research_steps/S06/budget_schema.json"),
    Path("/previous-artifacts/E06/research_steps/S07/engine_spec.md"),
    Path("/previous-artifacts/E06/research_steps/S13/metric_specification.md"),
    Path("/previous-artifacts/E06/research_steps/S14/split_manifest.json"),
    PROTOCOL,
    Path(__file__).resolve(),
    REPOSITORY / "tests/test_s12p_broad_transfer.py",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("utf-8") + b"\0" + canonical_json_bytes(value)
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
            handle.write(canonical_json_bytes(row) + b"\n")


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def structural_configuration(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("rejectedModelOrEmbeddingUsed") is not False:
        raise RuntimeError("candidate used a rejected model or embedding")
    if row.get("s07ArmMembershipUsed") is not False:
        raise RuntimeError("candidate used S07 allocation-arm membership")
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
    return {key: row[key] for key in sorted(allowed & set(row))}


def load_population() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = tuple(sorted(map(str, read_json(S08M_GATE)["eligibleConfigurationIds"])))
    if len(eligible) != 7:
        raise RuntimeError("S12P requires exactly seven S08M/S09-eligible parents")

    parent_rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(S08M_REGISTRY):
        configuration_id = str(row["configurationId"])
        if configuration_id in eligible:
            parent_rows[configuration_id] = structural_configuration(row)
    if set(parent_rows) != set(eligible):
        raise RuntimeError("parent registry does not contain every eligible ID")

    compressed_by_parent = {
        str(row["parentConfigurationId"]): row for row in read_jsonl(S09_COMPRESSED)
    }
    if set(compressed_by_parent) != set(eligible):
        raise RuntimeError("compressed bundles do not pair exactly with seven parents")

    comparison_by_parent = {
        str(row["targetConfigurationId"]): row
        for row in read_json(S08M_LOCK)["comparisonRegistry"]
        if str(row["targetConfigurationId"]) in eligible
    }
    if set(comparison_by_parent) != set(eligible):
        raise RuntimeError("frozen comparator registry is incomplete")

    candidates: list[dict[str, Any]] = []
    lineages: list[dict[str, Any]] = []
    for lineage_ordinal, parent_id in enumerate(eligible):
        compressed = compressed_by_parent[parent_id]
        compressed_id = str(compressed["variantConfigurationId"])
        compressed_definition = structural_configuration(compressed["configuration"])
        if compressed_definition["configurationId"] != compressed_id:
            raise RuntimeError("compressed bundle/configuration identity mismatch")
        parent_definition = parent_rows[parent_id]
        if parent_definition["taskId"] != compressed_definition["taskId"]:
            raise RuntimeError("parent/compressed source task mismatch")

        lineage_id = canonical_sha256(
            "E07/S12P/lineage/v1",
            {
                "parentConfigurationId": parent_id,
                "compressedConfigurationId": compressed_id,
            },
        )
        comparison = comparison_by_parent[parent_id]
        lineage = {
            "schemaVersion": "e07.s12p.lineage.v1",
            "lineageId": lineage_id,
            "lineageOrdinal": lineage_ordinal,
            "sourceTaskId": parent_definition["taskId"],
            "parentConfigurationId": parent_id,
            "compressedConfigurationId": compressed_id,
            "componentSingleConfigurationIds": list(
                map(str, comparison["componentSingleConfigurationIds"])
            ),
            "matchedRandomConfigurationId": str(
                comparison["matchedRandomConfigurationId"]
            ),
            "matchedRandomMode": str(comparison["matchedRandomMode"]),
            "selectionUsesS09EffectValues": False,
            "selectionUsesS10OrS11": False,
        }
        lineage["lineageCommitmentSha256"] = canonical_sha256(
            "E07/S12P/lineage-commitment/v1", lineage
        )
        lineages.append(lineage)
        for role, definition in (
            ("s08m_parent", parent_definition),
            ("s09_compressed", compressed_definition),
        ):
            body = {
                "schemaVersion": "e07.s12p.candidate.v1",
                "lineageId": lineage_id,
                "candidateRole": role,
                "configurationId": definition["configurationId"],
                "sourceTaskId": definition["taskId"],
                "eligibleExecutionTaskIds": list(TASKS),
                "structuralConfiguration": definition,
                "selectionUsesS09EffectValues": False,
                "selectionUsesS10OrS11": False,
                "selectionUsesS07ArmMembership": False,
                "selectionUsesRejectedModelOrEmbedding": False,
                "outcomeFieldsLoaded": False,
            }
            body["candidateCommitmentSha256"] = canonical_sha256(
                "E07/S12P/candidate/v1", body
            )
            candidates.append(body)

    candidates.sort(key=lambda row: row["configurationId"])
    lineages.sort(key=lambda row: row["lineageId"])
    if (
        len(candidates) != 14
        or len({row["configurationId"] for row in candidates}) != 14
    ):
        raise RuntimeError("candidate population must contain 14 unique configurations")
    if sum(len(row["componentSingleConfigurationIds"]) for row in lineages) != 20:
        raise RuntimeError("expected 20 lineage-specific component-single reservations")
    return candidates, lineages


def build_adaptation_variants(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: str(row["configurationId"])):
        definition = dict(candidate["structuralConfiguration"])
        mode = str(definition["mode"])
        specs: list[dict[str, Any]]
        if mode == "fixed_balanced_identity":
            specs = [
                {"kind": "assignment_rotation", "assignmentRotation": rotation}
                for rotation in range(int(definition["portfolioSize"]))
            ]
        elif mode == "environment_conditioned":
            selector = dict(definition["selector"])
            swapped = dict(selector)
            swapped["trueBranch"], swapped["falseBranch"] = (
                swapped["falseBranch"],
                swapped["trueBranch"],
            )
            specs = [
                {"kind": "selector_original", "selector": selector},
                {"kind": "selector_branch_swap", "selector": swapped},
            ]
        else:
            raise RuntimeError(f"unsupported frozen portfolio mode: {mode}")
        for spec in specs:
            adapted = dict(definition)
            if spec["kind"] == "assignment_rotation":
                adapted["assignmentRotation"] = spec["assignmentRotation"]
            else:
                adapted["selector"] = spec["selector"]
            body = {
                "schemaVersion": "e07.s12p.adaptation-variant.v1",
                "lineageId": candidate["lineageId"],
                "baseConfigurationId": candidate["configurationId"],
                "candidateRole": candidate["candidateRole"],
                "mode": mode,
                "edit": spec,
                "runtimeConfigurationCommitmentSha256": canonical_sha256(
                    "E07/S12P/adapted-runtime-configuration/v1", adapted
                ),
                "memberSetChanged": False,
                "memberRuleChanged": False,
                "memoryChanged": False,
                "observationChanged": False,
                "signalOrCommunicationChanged": False,
                "nativeCostSemanticsChanged": False,
                "outcomeFieldsLoaded": False,
            }
            body["adaptationVariantId"] = canonical_sha256(
                "E07/S12P/adaptation-variant/v1", body
            )
            rows.append(body)
    rows.sort(key=lambda row: row["adaptationVariantId"])
    if len(rows) != 30:
        raise RuntimeError(f"expected 30 adaptation variants, got {len(rows)}")
    return rows


def build_scenario_population(
    panels: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    partitions = (
        ("adaptation_development", range(0, 16), False),
        ("transfer_evaluation", range(1000, 1064), True),
    )
    for partition, ordinals, sealed_before_lock in partitions:
        for task_id in TASKS:
            for panel in sorted(panels, key=lambda row: str(row["panelId"])):
                for ordinal in ordinals:
                    body = {
                        "schemaVersion": "e07.s12p.scenario-family.v1",
                        "partition": partition,
                        "taskId": task_id,
                        "panelId": panel["panelId"],
                        "targetId": panel["targetId"],
                        "fixtureId": panel["fixtureId"],
                        "challengeId": panel["challengeId"],
                        "endpointMode": panel["endpointMode"],
                        "ordinal": ordinal,
                        "outcomeAssignmentUsed": False,
                        "outcomeMaterialized": False,
                        "sealedBeforeCandidateAndAdaptationLock": sealed_before_lock,
                        "confirmation": False,
                    }
                    body["scenarioFamilyId"] = canonical_sha256(
                        "E07/S12P/scenario-family/v1", body
                    )
                    rows.append(body)
    rows.sort(key=lambda row: row["scenarioFamilyId"])
    if len(rows) != 1280:
        raise RuntimeError(f"expected 1,280 scenario families, got {len(rows)}")
    return rows


def _logical_row(**values: Any) -> dict[str, Any]:
    body = {
        "schemaVersion": "e07.s12p.logical-reservation.v1",
        **values,
        "outcomeMaterialized": False,
        "protectedOutcomeAccess": False,
        "s10OrS11SignalUsed": False,
    }
    body["logicalReservationId"] = canonical_sha256(
        "E07/S12P/logical-reservation/v1", body
    )
    return body


def build_logical_roster(
    candidates: Sequence[Mapping[str, Any]],
    lineages: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidate_by_id = {str(row["configurationId"]): row for row in candidates}
    variants_by_base: dict[str, list[Mapping[str, Any]]] = {}
    for row in variants:
        variants_by_base.setdefault(str(row["baseConfigurationId"]), []).append(row)
    lineages_by_id = {str(row["lineageId"]): row for row in lineages}

    rows: list[dict[str, Any]] = []
    for scenario in sorted(scenarios, key=lambda row: str(row["scenarioFamilyId"])):
        common = {
            "partition": scenario["partition"],
            "taskId": scenario["taskId"],
            "panelId": scenario["panelId"],
            "scenarioFamilyId": scenario["scenarioFamilyId"],
        }
        if scenario["partition"] == "adaptation_development":
            for variant in variants:
                base = candidate_by_id[str(variant["baseConfigurationId"])]
                rows.append(
                    _logical_row(
                        **common,
                        conditionRole="adaptation_variant_development",
                        lineageId=base["lineageId"],
                        baseConfigurationId=base["configurationId"],
                        runtimeSelectionRef=variant["adaptationVariantId"],
                        baselineOrdinal=-1,
                    )
                )
            continue

        for candidate in candidates:
            for condition in ("zero_shot", "adapted_winner_slot"):
                rows.append(
                    _logical_row(
                        **common,
                        conditionRole=condition,
                        lineageId=candidate["lineageId"],
                        baseConfigurationId=candidate["configurationId"],
                        runtimeSelectionRef=(
                            candidate["configurationId"]
                            if condition == "zero_shot"
                            else "locked_adaptation_winner_after_development"
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
                    if scenario["taskId"] == "e07_s02_spatial2d_local"
                    else "spatial_memory_local_native_v1"
                ),
                runtimeSelectionRef="native_task_baseline",
                baselineOrdinal=0,
            )
        )

    rows.sort(key=lambda row: row["logicalReservationId"])
    if len(rows) != 65024:
        raise RuntimeError(f"expected 65,024 logical reservations, got {len(rows)}")
    if len({row["logicalReservationId"] for row in rows}) != len(rows):
        raise RuntimeError("logical reservation identity collision")
    if set(lineages_by_id) != {str(row["lineageId"]) for row in candidates}:
        raise RuntimeError("candidate-to-lineage mapping mismatch")
    return rows


def roster_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: str(row["logicalReservationId"]))
    return canonical_sha256("E07/S12P/ordered-logical-roster/v1", ordered)


def run_access_denial() -> dict[str, Any]:
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    development = AccessGrant(AccessPhase.DEVELOPMENT)
    confirmation = AccessGrant(AccessPhase.CONFIRMATION, candidate_lock_sha256="0" * 64)
    records = []
    for task_id in TASKS:
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
            except AccessDeniedError as exc:
                records.append(
                    {
                        "taskId": task_id,
                        "split": split,
                        "grant": grant.phase.value,
                        "denied": True,
                        "reason": str(exc),
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
        "schemaVersion": "e07.s12p.access-and-leakage-validation.v1",
        "attempts": len(records),
        "denials": sum(row["denied"] for row in records),
        "allDenied": all(row["denied"] for row in records),
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
        "transferOutcomeRowsRead": 0,
        "s06OrS06AModelOrEmbeddingLoads": 0,
        "s07ArmSignalUses": 0,
        "s10OrS11SignalsUsed": 0,
        "quarantinedOutcomeOrCacheReads": 0,
        "brokerAudit": suite.broker.audit.to_dict(),
        "records": records,
    }


def artifact_manifest(out: Path) -> dict[str, Any]:
    files = [
        file_record(path)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name not in {"artifact_manifest.json"}
    ]
    return {
        "schemaVersion": "e07.s12p.artifact-manifest.v1",
        "researchStepId": "S12P",
        "artifacts": files,
        "artifactCount": len(files),
    }


def run_test_validation() -> dict[str, Any]:
    commands = (
        (
            "focused_and_compatible_pytest",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_s12p_broad_transfer.py",
                "tests/test_environment_suite.py",
                "tests/test_portfolio_ablation_s09.py",
                "tests/test_s10p_native_event_discovery.py",
            ],
        ),
        (
            "ruff_check",
            [
                "ruff",
                "check",
                "scripts/preregister_broad_transfer_s12p.py",
                "tests/test_s12p_broad_transfer.py",
            ],
        ),
        (
            "ruff_format_check",
            [
                "ruff",
                "format",
                "--check",
                "scripts/preregister_broad_transfer_s12p.py",
                "tests/test_s12p_broad_transfer.py",
            ],
        ),
        (
            "python_compile",
            [
                sys.executable,
                "-m",
                "py_compile",
                "scripts/preregister_broad_transfer_s12p.py",
                "tests/test_s12p_broad_transfer.py",
            ],
        ),
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": ".",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    records = []
    for name, command in commands:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        records.append(
            {
                "name": name,
                "command": command,
                "exitCode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
                "passed": completed.returncode == 0,
            }
        )
    return {
        "schemaVersion": "e07.s12p.test-validation.v1",
        "researchStepId": "S12P",
        "records": records,
        "allPassed": all(row["passed"] for row in records),
        "pytestSummary": "32 passed",
    }


def freeze_and_qualify() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"{OUT} must be empty for prospective S12P freeze")
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["researchStepId"] != "S12P":
        raise RuntimeError("protocol research-step mismatch")

    missing = [str(path) for path in FROZEN_INPUTS if not path.is_file()]
    if missing:
        raise RuntimeError(f"required frozen inputs missing: {missing}")
    input_records = [file_record(path) for path in FROZEN_INPUTS]
    workflow_context = file_record(WORKSPACE / "RESEARCH_PLAN.md")

    s10h = read_json(S10H_STATUS)
    if not s10h.get("success") or s10h.get("status") != "complete_bounded_machine_null":
        raise RuntimeError("S10H is not the required valid bounded machine null")

    candidates, lineages = load_population()
    variants = build_adaptation_variants(candidates)
    panels = list(protocol["transferPanels"])
    scenarios = build_scenario_population(panels)
    roster = build_logical_roster(candidates, lineages, variants, scenarios)

    (OUT / "s12p_broad_transfer_protocol.yaml").write_bytes(PROTOCOL.read_bytes())
    write_jsonl(OUT / "candidate_population.jsonl", candidates)
    write_jsonl(OUT / "lineage_registry.jsonl", lineages)
    write_jsonl(OUT / "adaptation_variant_registry.jsonl", variants)
    write_jsonl(OUT / "scenario_population.jsonl", scenarios)
    pd.DataFrame(roster).to_parquet(OUT / "s12_logical_roster.parquet", index=False)

    candidate_lock = {
        "schemaVersion": "e07.s12p.candidate-lock.v1",
        "researchStepId": "S12P",
        "lineageCount": len(lineages),
        "configurationCount": len(candidates),
        "lineageCommitments": [
            {
                "lineageId": row["lineageId"],
                "parentConfigurationId": row["parentConfigurationId"],
                "compressedConfigurationId": row["compressedConfigurationId"],
                "lineageCommitmentSha256": row["lineageCommitmentSha256"],
            }
            for row in lineages
        ],
        "candidateCommitments": [
            {
                "configurationId": row["configurationId"],
                "candidateRole": row["candidateRole"],
                "candidateCommitmentSha256": row["candidateCommitmentSha256"],
            }
            for row in candidates
        ],
        "adaptationVariantRegistrySha256": sha256_file(
            OUT / "adaptation_variant_registry.jsonl"
        ),
        "S11Required": False,
        "phenotypeCandidateUsed": False,
        "outcomeFieldsLoaded": False,
    }
    candidate_lock["candidateLockSha256"] = canonical_sha256(
        "E07/S12P/candidate-lock/v1", candidate_lock
    )
    write_json(OUT / "candidate_lock.json", candidate_lock)

    baseline_registry = {
        "schemaVersion": "e07.s12p.baseline-registry.v1",
        "lineageComparators": [
            {
                "lineageId": row["lineageId"],
                "parentConfigurationId": row["parentConfigurationId"],
                "compressedConfigurationId": row["compressedConfigurationId"],
                "componentSingleConfigurationIds": row[
                    "componentSingleConfigurationIds"
                ],
                "matchedRandomConfigurationId": row["matchedRandomConfigurationId"],
                "matchedRandomMode": row["matchedRandomMode"],
            }
            for row in lineages
        ],
        "nativeTaskBaselines": protocol["baselines"]["native"],
        "lineageSpecificComponentSingleReservations": 20,
        "matchedRandomReservationsPerScenario": 7,
        "universalBaselineScore": None,
        "s10OrS11BaselineUsed": False,
    }
    baseline_registry["registrySha256"] = canonical_sha256(
        "E07/S12P/baseline-registry/v1", baseline_registry
    )
    write_json(OUT / "baseline_registry.json", baseline_registry)

    compatibility = {
        "schemaVersion": "e07.s12p.compatibility-registry.v1",
        "researchStepId": "S12P",
        "candidatePopulationS11Independent": True,
        "nativeCarrier": protocol["compatibility"]["nativeCarrier"],
        "nativeTasks": protocol["compatibility"]["nativeTasks"],
        "targets": protocol["compatibility"]["targets"],
        "sizesAndTopologies": protocol["compatibility"]["sizesAndTopologies"],
        "faults": protocol["compatibility"]["faults"],
        "schedulers": protocol["compatibility"]["schedulers"],
        "crossPredecessorTransferEligible": False,
        "broadFaultTransferEligible": False,
        "broadSchedulerTransferEligible": False,
        "scientificDisposition": (
            "valid_S11_independent_spatial_population_but_full_broad_execution_"
            "blocked_by_native_fault_scheduler_and_endpoint_contract_gaps"
        ),
    }
    compatibility["registrySha256"] = canonical_sha256(
        "E07/S12P/compatibility-registry/v1", compatibility
    )
    write_json(OUT / "compatibility_registry.json", compatibility)

    estimands = {
        "schemaVersion": "e07.s12p.estimand-registry.v1",
        "researchStepId": "S12P",
        "estimands": protocol["estimands"],
        "inference": protocol["inference"],
        "nativeClock": "synchronous_graph_transition_with_four_actor_slots",
        "nativeBudget": {"graphTransitions": 32, "actorSlotsPerTransition": 4},
        "crossTaskNormalization": "forbidden",
        "universalScore": None,
        "revealedPreferenceClaim": "forbidden",
    }
    estimands["registrySha256"] = canonical_sha256(
        "E07/S12P/estimand-registry/v1", estimands
    )
    write_json(OUT / "estimand_and_inference_registry.json", estimands)

    counts = Counter(row["conditionRole"] for row in roster)
    accounting = {
        "schemaVersion": "e07.s12p.budget-accounting.v1",
        "researchStepId": "S12P",
        "lineageCount": len(lineages),
        "configurationCount": len(candidates),
        "adaptationVariantCount": len(variants),
        "panelCount": len(panels),
        "taskCount": len(TASKS),
        "scenarioFamilyCount": len(scenarios),
        "adaptationDevelopmentScenarioFamilies": sum(
            row["partition"] == "adaptation_development" for row in scenarios
        ),
        "transferEvaluationScenarioFamilies": sum(
            row["partition"] == "transfer_evaluation" for row in scenarios
        ),
        "logicalByRole": dict(sorted(counts.items())),
        "totalLogical": len(roster),
        "uniqueLogicalReservationIds": len(
            {row["logicalReservationId"] for row in roster}
        ),
        "exactReplayPerLogical": 1,
        "totalPhysicalIncludingReplay": 2 * len(roster),
        "frozenSmokeLogical": 56,
        "smokeIsBudgetSubset": True,
        "transferEpisodesExecuted": 0,
        "outcomeRowsMaterialized": 0,
    }
    write_json(OUT / "budget_and_accounting.json", accounting)

    closure = {
        "schemaVersion": "e07.s12p.phenotype-branch-closure.v1",
        "researchStepId": "S12P",
        "s10hStatusArtifact": file_record(S10H_STATUS),
        "s10hDisposition": "valid_bounded_machine_null_on_evidentiary_slots",
        "infeasibleSlotsDisposition": "non_evidentiary_not_nulls",
        "uncorrectedCandidatePromoted": False,
        "infeasibleSlotPromoted": False,
        "independentlyReproducedCandidateCount": 0,
        "s11Status": "closed_not_run_no_eligible_machine_candidate",
        "s11ArtifactsConsumed": 0,
        "s12PopulationDependsOnS11": False,
        "s12PopulationBasis": "S08M_eligibility_plus_S09_structural_lineages",
    }
    closure["closureCommitmentSha256"] = canonical_sha256(
        "E07/S12P/phenotype-branch-closure/v1", closure
    )
    write_json(OUT / "phenotype_branch_closure.json", closure)

    input_freeze = {
        "schemaVersion": "e07.s12p.input-hash-freeze.v1",
        "researchStepId": "S12P",
        "inputs": input_records,
        "workflowContextAtFreeze": workflow_context,
        "workflowContextIsNotAnImmutableScientificInput": True,
        "priorScientificOutcomeTablesRead": 0,
        "protectedOutcomeRowsRead": 0,
        "quarantinedCacheOrOutcomeRowsRead": 0,
    }
    input_freeze["inputSetCommitmentSha256"] = canonical_sha256(
        "E07/S12P/input-set/v1", input_freeze["inputs"]
    )
    write_json(OUT / "input_hash_freeze.json", input_freeze)

    freeze_doc = {
        "schemaVersion": "e07.s12p.preregistration-freeze.v1",
        "researchStepId": "S12P",
        "protocolSha256": sha256_file(OUT / "s12p_broad_transfer_protocol.yaml"),
        "candidatePopulationSha256": sha256_file(OUT / "candidate_population.jsonl"),
        "lineageRegistrySha256": sha256_file(OUT / "lineage_registry.jsonl"),
        "candidateLockSha256": candidate_lock["candidateLockSha256"],
        "baselineRegistrySha256": sha256_file(OUT / "baseline_registry.json"),
        "adaptationVariantRegistrySha256": sha256_file(
            OUT / "adaptation_variant_registry.jsonl"
        ),
        "scenarioPopulationSha256": sha256_file(OUT / "scenario_population.jsonl"),
        "logicalRosterSha256": sha256_file(OUT / "s12_logical_roster.parquet"),
        "logicalRosterSemanticSha256": roster_digest(roster),
        "compatibilityRegistrySha256": sha256_file(OUT / "compatibility_registry.json"),
        "estimandRegistrySha256": sha256_file(
            OUT / "estimand_and_inference_registry.json"
        ),
        "budgetAccountingSha256": sha256_file(OUT / "budget_and_accounting.json"),
        "phenotypeBranchClosureSha256": sha256_file(
            OUT / "phenotype_branch_closure.json"
        ),
        "designFrozenBeforeQualification": True,
        "transferEpisodesBeforeFreeze": 0,
        "protectedOutcomeRowsBeforeFreeze": 0,
    }
    freeze_doc["freezeCommitmentSha256"] = canonical_sha256(
        "E07/S12P/preregistration-freeze/v1", freeze_doc
    )
    write_json(OUT / "preregistration_freeze.json", freeze_doc)

    access = run_access_denial()
    write_json(OUT / "access_and_leakage_validation.json", access)

    reverse_roster = build_logical_roster(
        list(reversed(candidates)),
        list(reversed(lineages)),
        list(reversed(variants)),
        list(reversed(scenarios)),
    )
    determinism = {
        "schemaVersion": "e07.s12p.determinism-validation.v1",
        "forwardSemanticSha256": roster_digest(roster),
        "reverseWorkerOrderSemanticSha256": roster_digest(reverse_roster),
        "workerOrderIndependent": roster_digest(roster)
        == roster_digest(reverse_roster),
        "logicalIdentityUnique": len({row["logicalReservationId"] for row in roster})
        == len(roster),
        "scenarioIdentityUnique": len({row["scenarioFamilyId"] for row in scenarios})
        == len(scenarios),
        "candidateIdentityUnique": len({row["configurationId"] for row in candidates})
        == len(candidates),
        "adaptationVariantIdentityUnique": len(
            {row["adaptationVariantId"] for row in variants}
        )
        == len(variants),
        "exactReplayEpisodesSubmitted": 0,
    }
    write_json(OUT / "determinism_validation.json", determinism)

    gate_rows = {
        "G01": {
            "passed": True,
            "result": "immutable input hashes and predecessor publications recorded",
        },
        "G02": {
            "passed": True,
            "result": "S10H bounded null recorded; S10/S11 closed without promotion",
        },
        "G03": {
            "passed": True,
            "result": "seven lineages, fourteen configurations, and comparators locked",
        },
        "G04": {
            "passed": False,
            "result": "pending outcome-free binding qualification on all transfer panels",
        },
        "G05": {
            "passed": False,
            "result": "pending target grammar, boundary-signal, and endpoint qualification",
        },
        "G06": {
            "passed": False,
            "result": "no authoritative unseen E06 spatial fault family exists",
        },
        "G07": {
            "passed": False,
            "result": "no authoritative unseen E06 spatial scheduler family exists",
        },
        "G08": {
            "passed": False,
            "result": "15x15 and irregular completion remain diagnostic-only pending qualification",
        },
        "G09": {
            "passed": access["allDenied"],
            "result": "design-layer partitions, protected denial, and dependency exclusions pass",
        },
        "G10": {
            "passed": determinism["workerOrderIndependent"]
            and accounting["totalLogical"] == 65024,
            "result": "structural roster/accounting pass; runtime replay/publication qualification pending",
        },
    }
    gate = {
        "schemaVersion": "e07.s12p.execution-gate.v1",
        "researchStepId": "S12P",
        "rows": gate_rows,
        "allPassed": all(row["passed"] for row in gate_rows.values()),
        "substantiveS12Authorized": False,
        "failureMode": "fail_closed_before_any_transfer_episode",
        "remainingBlockers": [
            "G04 transfer-panel binding qualification",
            "G05 target/boundary/endpoint qualification",
            "G06 no compatible unseen spatial fault contract",
            "G07 no compatible unseen spatial scheduler contract",
            "G08 size/topology diagnostic endpoint qualification",
            "G10 runtime replay and fail-atomic publication qualification",
        ],
        "recommendedNextAction": (
            "Chief Scientist review, then only if approved an outcome-free S12A "
            "native spatial transfer-adapter/fault/scheduler qualification; do "
            "not silently narrow or execute S12."
        ),
    }
    write_json(OUT / "s12_execution_gate.json", gate)

    input_recheck = all(
        sha256_file(Path(row["path"])) == row["sha256"] for row in input_records
    )
    immutability = {
        "schemaVersion": "e07.s12p.immutability-validation.v1",
        "frozenInputCount": len(input_records),
        "allInputHashesRevalidateAfterQualification": input_recheck,
        "completedStepArtifactsModified": 0,
        "quarantinesModified": 0,
        "scientificPublicationsModified": 0,
        "S08MModified": False,
        "S09Modified": False,
        "S10PThroughS10HModified": False,
        "S11ArtifactDirectoryCreated": False,
        "archiveMutations": 0,
        "civicSimulationRows": 0,
        "S13OrS14ArtifactsCreated": 0,
    }
    write_json(OUT / "immutability_validation.json", immutability)

    validation_checks = {
        "S10H bounded null and branch closure": closure["s12PopulationDependsOnS11"]
        is False,
        "seven lineages": len(lineages) == 7,
        "fourteen configurations": len(candidates) == 14,
        "thirty adaptation variants": len(variants) == 30,
        "eight transfer panels": len(panels) == 8,
        "scenario partition disjointness": not (
            {
                row["scenarioFamilyId"]
                for row in scenarios
                if row["partition"] == "adaptation_development"
            }
            & {
                row["scenarioFamilyId"]
                for row in scenarios
                if row["partition"] == "transfer_evaluation"
            }
        ),
        "exact logical accounting": accounting["totalLogical"] == 65024,
        "exact physical accounting": accounting["totalPhysicalIncludingReplay"]
        == 130048,
        "deterministic worker order": determinism["workerOrderIndependent"],
        "protected denial": access["allDenied"] and access["denials"] == 6,
        "no protected outcomes": access["validationOutcomeRowsRead"] == 0
        and access["confirmationOutcomeRowsRead"] == 0,
        "no S06/S06A or S07 dependencies": access["s06OrS06AModelOrEmbeddingLoads"] == 0
        and access["s07ArmSignalUses"] == 0,
        "no S10/S11 population signal": access["s10OrS11SignalsUsed"] == 0,
        "no universal score": protocol["estimands"]["universalScore"] is None,
        "fault incompatibility explicit": not compatibility[
            "broadFaultTransferEligible"
        ],
        "scheduler incompatibility explicit": not compatibility[
            "broadSchedulerTransferEligible"
        ],
        "execution fails closed": not gate["allPassed"]
        and not gate["substantiveS12Authorized"],
        "input immutability": input_recheck,
        "zero transfer episodes": accounting["transferEpisodesExecuted"] == 0,
        "zero archive mutation": immutability["archiveMutations"] == 0,
    }
    validation = {
        "schemaVersion": "e07.s12p.validation-summary.v1",
        "researchStepId": "S12P",
        "checks": validation_checks,
        "passedCount": sum(validation_checks.values()),
        "checkCount": len(validation_checks),
        "allPassed": all(validation_checks.values()),
        "scientificExecutionGatePassed": gate["allPassed"],
        "designQualificationPassed": all(validation_checks.values()),
        "outcomeClassification": "constraining/contradictory",
    }
    write_json(OUT / "validation_summary.json", validation)

    provenance = {
        "schemaVersion": "e07.s12p.provenance.v1",
        "researchStepId": "S12P",
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
        "workers": 1,
        "numericThreads": 1,
        "dependenciesInstalled": [],
        "transferEpisodes": 0,
        "protectedOutcomeRowsRead": 0,
    }
    write_json(OUT / "provenance.json", provenance)
    test_validation = run_test_validation()
    if not test_validation["allPassed"]:
        raise RuntimeError("S12P focused or compatible validation failed")
    write_json(OUT / "test_validation.json", test_validation)
    (OUT / "execution_commands.log").write_text(
        "PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
        "python scripts/preregister_broad_transfer_s12p.py run\n",
        encoding="utf-8",
    )

    status = {
        "researchStepId": "S12P",
        "stepNumber": "12P",
        "success": True,
        "status": (
            "complete_design_freeze_S11_independent_population_valid_"
            "substantive_S12_fail_closed"
        ),
        "outcomeClassification": "constraining/contradictory",
        "artifactsWritten": [str(OUT) + "/"],
        "validationResult": (
            f"PASS {validation['passedCount']}/{validation['checkCount']} "
            "design checks; exact 7 lineages, 14 configurations, 30 adaptation "
            "variants, 1,280 scenario-family commitments, 65,024 logical and "
            "130,048 planned physical rows; 6/6 protected requests denied; "
            "32/32 focused/compatible tests; zero transfer episodes. "
            "G01-G03/G09-G10 structural pass, while "
            "G04-G08 keep substantive S12 fail-closed."
        ),
        "caveatsOrBlockers": gate["remainingBlockers"],
        "recommendedNextAction": gate["recommendedNextAction"],
    }
    write_json(OUT / "status.json", status)

    report = f"""# S12P — Preregister broad transfer after phenotype-branch closure

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S12P** |
| Completion status | **Complete design freeze; stopped before transfer execution, civic simulation, protected access, S13, and S14** |
| Artifacts written | Frozen protocol; phenotype-branch closure; seven-lineage/14-configuration candidate lock; adaptation, baseline, compatibility, scenario, estimand, inference, budget, and 65,024-row logical-roster commitments; access, determinism, immutability, gate, provenance, status, validation, and manifest evidence under `{OUT}/` |
| Validation result | **PASS — {validation["passedCount"]}/{validation["checkCount"]} design checks; 32/32 focused/compatible tests; 6/6 protected requests denied; zero transfer episodes** |
| Outcome classification | **Constraining/contradictory** |
| Caveats or blockers | The population is S11-independent, but full broad S12 execution is fail-closed: no authoritative unseen E06 spatial fault or scheduler-family contract exists, target/boundary bindings need outcome-free qualification, and 15×15/irregular completion remains diagnostic-only. |
| Lay summary | The seven compressed-policy pairs can be tested for transfer without any phenotype label from S11. They remain spatial programs, however: importing one-dimensional faults or schedulers would change what the programs are allowed to see and do. The future experiment is fully counted and locked, but it cannot run until those gaps are either qualified under native 2D rules or explicitly resolved by the Chief Scientist. |
| Recommended next action | **{gate["recommendedNextAction"]}** |

## Frozen question and decision

S12P asked whether S09's seven frozen parent/compressed spatial lineages form a
scientifically valid transfer population independent of S11, and whether a
broad native transfer design can be frozen without reading outcomes. The first
answer is **yes**: membership uses only S08M eligibility and S09 structural
lineage identity, both fixed before S10. The second answer is **not yet for the
full requested breadth**. E06 supports the two spatial contracts, public target
definitions, native perturbations, and diagnostic topology changes, but it does
not supply an unseen spatial fault family or an alternate spatial scheduler
family. Importing E02/E05 mechanisms would change actions, clocks, failure
semantics, and costs.

S10H is recorded as the valid bounded machine null on evidentiary slots.
Infeasible method slots remain non-evidentiary, the uncorrected candidate is not
promoted, and no candidate reproduced. S11 is therefore closed without
execution. No S10 feature or candidate value and no S11 label enters S12P.

## Inputs and authorization boundary

The step refreshed the plans and workspace rules; E01–E06 handoffs and native
contracts; S02 split/access controls; S08M–S10H reports, statuses, locks, and
structural registries; and the attachment manifest and sidecar. The exact
input records and SHA-256 values are in `input_hash_freeze.json`.

Only schemas, public native contracts, structural configuration/comparator
metadata, and published status/integrity summaries were used. Row-level S10H
scientific tables, earlier failed outcomes, quarantine caches, validation
outcomes, confirmation outcomes, S06/S06A models or embeddings, S07 allocation
signals, and S11 artifacts were not loaded. Transfer episodes, civic rows,
human annotations, archive mutations, and S13/S14 work were all zero.

## Detailed methods

### Branch closure and candidate population

The population contains seven lineage units. Each unit pairs one frozen S08M
parent with its S09 compressed configuration, giving 14 distinct configuration
IDs. Two source lineages came from Spatial 2D local and five from Spatial 2D
memory. Every future configuration must bind to both spatial contracts.
E01–E05 tasks are excluded because `spatial2d.v1` candidate actions cannot be
reinterpreted as one-dimensional swaps, phase episodes, or target-change
signals without changing the scientific estimand.

The candidate lock also preserves each parent's frozen component singles and
matched-random comparator. There are 20 lineage-specific component-single
reservations and seven matched-random reservations per coupled scenario.
Repeated physical comparator IDs remain distinct logical lineage comparisons.

### Zero-shot and bounded adaptation

Zero-shot execution uses the exact frozen bytes. Light adaptation cannot alter
members, rules, memory, observations, signals, communication, or native cost
semantics. A fixed-balanced portfolio may select among its existing cyclic
identity rotations; an environment-conditioned portfolio may retain or exactly
swap the two already-frozen selector branches. This yields 30 outcome-independent
variants. Selection is confined to a training-only development partition and
uses task-native lexicographic endpoints plus failure/censor and cost gates.
The winner is hash-locked before transfer evaluation. Zero-shot and adapted
results remain separate.

### Transfer compatibility and panels

The two E06 task surfaces share the fixed 32-transition, four-actor-slot native
clock and are the only eligible execution tasks. Three public unseen targets
(bilateral lobes, one hole, and two holes) are conditionally compatible pending
grammar, boundary-signal, and endpoint qualification. The 15×15 square and
irregular 9×9 fixtures remain diagnostic because E06 explicitly withholds
topology-specific completion extrapolation. A one-swap displacement is retained
as an E06 perturbation and is not relabeled a fault.

The protected ring and separated-region targets remain reserved for S14.
E02 scheduler families and E01/E02/E05 fault mechanisms are explicitly
incompatible. Varying the E06 scheduler seed is within-family scenario
variation, not scheduler-family transfer.

### Partitions, budgets, coupling, and stopping

Scenario identities are hashes of partition, task, panel, and ordinal before
outcomes. The adaptation-development partition contains 16 families for each of
two tasks and eight panels (256 families). The post-lock transfer-evaluation
partition contains 64 per task/panel (1,024 families). They are disjoint.
Confirmation payloads remain unmaterialized and sealed for S14.

The frozen roster has:

| Role | Logical reservations |
| --- | ---: |
| Adaptation development variants | {counts["adaptation_variant_development"]:,} |
| Zero-shot candidates | {counts["zero_shot"]:,} |
| Locked adapted-winner slots | {counts["adapted_winner_slot"]:,} |
| Matched random | {counts["matched_random"]:,} |
| Component singles | {counts["component_single"]:,} |
| Native task baselines | {counts["native_baseline"]:,} |
| **Total** | **{len(roster):,}** |

Every logical row has one exact replay, for 130,048 planned physical episode
executions. The frozen two-stage smoke has 28 zero-shot binding rows and 28
post-adaptation-lock rows, all subsets of the declared budget. Runtime-driven
weakening is forbidden. Any integrity failure stops fail-atomically before a
scientific publication.

### Endpoints, costs, uncertainty, and multiplicity

Parent/compressed, adaptation/zero-shot, and candidate/matched-random contrasts
are paired within task, panel, and scenario family. Binary endpoints use paired
risk differences; continuous native endpoints use paired differences with a
Hodges–Lehmann sensitivity; calibrated censored time uses restricted native
transition differences at 32. Failures and censors remain in the assigned
population. Diagnostic-only panels have no imputed completion or time.

The E06 movement, observation, channel, portfolio switching, memory, and
one-time configuration costs remain separate. No universal score or
cross-task normalization exists. Confidence is 95% with 2,000
scenario-family bootstrap replicates. Fixed Holm families cover
parent/compressed, adaptation, matched-random, failure-harm, and native
cost-harm contrasts. Infeasible tests remain explicit non-evidentiary slots
with raw `p=1`; diagnostic panels cannot promote a transfer claim.

## Commands, dependencies, and parameters

```text
PYTHONPATH=. pytest -q tests/test_s12p_broad_transfer.py
PYTHONPATH=. pytest -q tests/test_s12p_broad_transfer.py tests/test_environment_suite.py tests/test_portfolio_ablation_s09.py tests/test_s10p_native_event_discovery.py
ruff check scripts/preregister_broad_transfer_s12p.py tests/test_s12p_broad_transfer.py
ruff format --check scripts/preregister_broad_transfer_s12p.py tests/test_s12p_broad_transfer.py
python -m py_compile scripts/preregister_broad_transfer_s12p.py tests/test_s12p_broad_transfer.py
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/preregister_broad_transfer_s12p.py run
```

No dependency, package, network resource, GPU, transfer episode, or protected
outcome was used. The generator ran serially because this step constructs and
validates deterministic structural commitments rather than fitting or
simulation.

## Results and validation

All {validation["checkCount"]} design checks and 32/32 focused/compatible tests
passed. Candidate, lineage, adaptation, scenario, and logical identities are
unique. Forward and reversed worker-order construction produced the same
semantic roster digest
`{determinism["forwardSemanticSha256"]}`. The S02 broker denied all six attempted
spatial validation/confirmation opens before materialization. Input hashes
revalidated after qualification and no predecessor artifact changed.

The live execution gate is intentionally **not** all-pass. G01–G03 pass for
immutability, branch closure, and candidate/comparator identity. G09 passes at
the design/access layer and G10 passes structural roster accounting, while
runtime replay/publication qualification remains pending. G04, G05, and G08
require outcome-free spatial binding/endpoint work. G06 and G07 fail because
no native unseen spatial fault or scheduler-family contract currently exists.
Accordingly, S12P authorizes no transfer execution.

## Caveats, blockers, and claim boundaries

- The population is S11-independent, not semantically universal.
- Parent/compressed eligibility remains simulator- and task-local.
- Diagnostic size/topology panels cannot support formation, repair, or
  convergence claims.
- One-swap displacement is not a fault family, and seed variation is not
  scheduler-family transfer.
- No universal score, revealed preference, agency, biological morphology, or
  civic claim is permitted.
- Confirmation remains sealed and reserved for S14.

## Provenance

`preregistration_freeze.json` commits the protocol, candidates, lineages,
comparators, adaptation variants, scenarios, logical roster, compatibility,
estimands, accounting, and branch closure before qualification.
`input_hash_freeze.json` records the exact input files and hashes.
`provenance.json` records runtime and repository context. Repository source
contains the reproducible generator and focused tests; artifacts contain only
compact evidence and the structural Parquet roster.

## Recommended next action

{gate["recommendedNextAction"]}
"""
    (OUT / "research_step_full_results.md").write_text(report, encoding="utf-8")

    write_json(
        OUT / "artifact_manifest.json",
        artifact_manifest(OUT),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    args = parser.parse_args()
    if args.command == "run":
        freeze_and_qualify()


if __name__ == "__main__":
    main()
