#!/usr/bin/env python3
"""Qualify S08L using structural metadata and outcome-free fixtures only."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from src.environment_suite import (
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
    EnvironmentSuite,
    Split,
)
from src.environment_suite.contracts import canonical_sha256
from src.portfolio_preregistration.core import (
    TASK_IDS,
    candidate_commitment,
    checked_protocol,
    plan_digest,
    read_jsonl,
    validate_portfolio_registry,
)
from src.portfolio_search.execution import (
    S08P,
    _catalog,
    _expand_logical_result,
    _initial_work,
    _physical_key,
    build_action,
    matched_random_definition,
)
from src.portfolio_search.identity import (
    IDENTITY_PLANE_VERSION,
    IdentityPlaneError,
    bind_identity_plane,
    validate_bound_identity_record,
    validate_bound_identity_roster,
    validate_persisted_logical_identity_rows,
)
from src.portfolio_search.preflight import (
    sha256_file,
    tree_digest,
    validate_executable_bindings,
)


REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/artifacts/research_steps")
OUTPUT = ARTIFACT_ROOT / "S08L"
PROTOCOL = REPOSITORY / "configs/portfolio/s08l_logical_identity_plane.yaml"
PREREGISTRATION = OUTPUT / "preregistration_freeze.json"
INPUT_FREEZE = OUTPUT / "input_hash_freeze.json"
TASK_REGISTRY = REPOSITORY / "configs/environment_suite/task_registry.yaml"
SPLIT_MANIFEST = REPOSITORY / "configs/environment_suite/split_manifest.json"
EXPECTED_PROTOCOL_SHA256 = (
    "254bbf7e24f5f05576582fee7659dff7b94236a8901884e70f27e93ebe2ff904"
)
IMMUTABLE_STEPS = (
    "S05",
    "S08P",
    "S08A",
    "S08",
    "S08B",
    "S08C",
    "S08D",
    "S08E",
    "S08F",
    "S08G",
    "S08H",
    "S08I",
    "S08J",
    "S08K",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                dict(row),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _strip_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value) for key, value in row.items() if key != "identityPlane"
    }


def _precomputed_resolver(
    physical_by_slot: Mapping[str, str],
):
    def resolve(row: Mapping[str, Any]) -> str:
        return physical_by_slot[str(row["logicalSlotId"])]

    return resolve


def _identity_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(
        "E07/S08L/identity-roster-digest/v1",
        sorted(
            (
                str(row["logicalSlotId"]),
                row["identityPlane"]["reservationSlotIdentitySha256"],
                row["identityPlane"]["runtimeConfigurationIdentitySha256"],
                row["identityPlane"]["physicalExecutionIdentitySha256"],
                row["identityPlane"]["logicalResultIdentitySha256"],
                row["identityPlane"]["physicalToLogicalExpansionCommitmentSha256"],
                row["identityPlane"]["logicalIdentityBindingSha256"],
            )
            for row in rows
        ),
    )


def _adaptive_fixture_work(
    budget: pd.DataFrame,
    configurations: Sequence[Mapping[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Compile six structural batches without archive or outcome input."""

    by_task_target: dict[str, list[Mapping[str, Any]]] = {}
    for task_id in TASK_IDS:
        by_task_target[task_id] = sorted(
            (
                row
                for row in configurations
                if row["taskId"] == task_id
                and row["mode"]
                in {"fixed_balanced_identity", "environment_conditioned"}
            ),
            key=lambda row: (row["mode"], row["configurationId"]),
        )
    batches: list[list[dict[str, Any]]] = []
    for generation in range(1, 7):
        frame = budget[
            (budget["stage"] == "adaptive") & (budget["generation"] == generation)
        ]
        rows: list[dict[str, Any]] = []
        for task_id in TASK_IDS:
            task_frame = frame[frame["taskId"] == task_id]
            pair_groups = task_frame.groupby("pairedSlotId", sort=True)
            targets = by_task_target[task_id]
            for pair_ordinal, (pair_id, pair_rows) in enumerate(pair_groups):
                # Structural only: no objective, archive, status, S07 arm, or
                # quarantined value enters this deterministic fixture choice.
                target = dict(targets[(generation + pair_ordinal) % len(targets)])
                comparator = matched_random_definition(target)
                definitions = {
                    "target_portfolio": target,
                    "matched_random_dynamic": comparator,
                }
                for slot in pair_rows.to_dict("records"):
                    role = str(slot["configurationRole"])
                    rows.append(
                        {
                            "stage": "adaptive",
                            "generation": generation,
                            "taskId": task_id,
                            "split": "train",
                            "scenarioFamilyOrdinal": int(slot["scenarioFamilyOrdinal"]),
                            "logicalSlotId": str(slot["logicalSlotId"]),
                            "logicalSlotOrdinal": int(slot["logicalSlotOrdinal"]),
                            "reservedConfigurationSlotId": str(slot["configurationId"]),
                            "configurationRole": role,
                            "pairedSlotId": str(pair_id),
                            "configuration": definitions[role],
                            "smoke": False,
                        }
                    )
        if len(rows) != 1024:
            raise RuntimeError(
                f"adaptive fixture generation {generation} has {len(rows)} rows"
            )
        batches.append(
            bind_identity_plane(rows, physical_identity_resolver=_physical_key)
        )
    return batches


def _persisted_projection(
    bound: Mapping[str, Any], physical_result_sha: str
) -> dict[str, Any]:
    physical = {
        "stableEvaluationSha256": physical_result_sha,
        "configurationId": bound["configuration"]["configurationId"],
        "configurationDefinitionSha256": canonical_sha256(
            "qualification-physical-definition",
            bound["configuration"]["configurationId"],
        ),
    }
    return _expand_logical_result(
        bound,
        physical,
        bound["identityPlane"]["physicalExecution"]["physicalDedupEquivalenceSha256"],
    )


def adversarial_qualification(
    configurations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = next(
        row
        for row in configurations
        if row["taskId"] == "e07_s02_chimera_1d" and row["mode"] == "single_policy"
    )
    alias = deepcopy(base)
    alias["configurationId"] = canonical_sha256(
        "E07/S08L/complete-equivalence-alias/v1", base["configurationId"]
    )
    alias["memberSetId"] = canonical_sha256(
        "E07/S08L/complete-equivalence-member-set/v1", base["memberSetId"]
    )

    def work(
        ordinal: int,
        configuration: Mapping[str, Any],
        *,
        role: str,
        reservation: str,
    ) -> dict[str, Any]:
        return {
            "stage": "qualification",
            "generation": 1,
            "taskId": str(configuration["taskId"]),
            "split": "train",
            "scenarioFamilyOrdinal": 300,
            "logicalSlotOrdinal": ordinal,
            "logicalSlotId": canonical_sha256("E07/S08L/adversary-slot/v1", ordinal),
            "reservedConfigurationSlotId": reservation,
            "configurationRole": role,
            "pairedSlotId": "s08l-adversary-pair",
            "configuration": dict(configuration),
            "smoke": False,
        }

    valid = [
        work(
            0,
            base,
            role="target_portfolio",
            reservation="frozen-target-reservation",
        ),
        work(
            1,
            base,
            role="matched_random_dynamic",
            reservation="frozen-comparator-reservation",
        ),
        work(
            2,
            alias,
            role="component_single",
            reservation="frozen-component-reservation",
        ),
    ]
    bound = bind_identity_plane(valid, physical_identity_resolver=_physical_key)
    physical_ids = {
        row["identityPlane"]["physicalExecutionIdentitySha256"] for row in bound
    }
    valid_audit = validate_bound_identity_roster(
        bound, physical_identity_resolver=_physical_key
    )
    persisted = [_persisted_projection(row, "1" * 64) for row in bound]
    persisted_audit = validate_persisted_logical_identity_rows(persisted)

    cases: list[dict[str, Any]] = [
        {
            "case": "repeated_same_configuration_same_scenario_distinct_slots",
            "expected": "pass",
            "observed": "pass"
            if len(
                {
                    bound[index]["identityPlane"]["logicalResultIdentitySha256"]
                    for index in (0, 1)
                }
            )
            == 2
            else "fail",
        },
        {
            "case": "complete_equivalence_configuration_alias",
            "expected": "pass",
            "observed": "pass" if len(physical_ids) == 1 else "fail",
        },
        {
            "case": "adaptive_reservation_runtime_identity_separation",
            "expected": "pass",
            "observed": "pass"
            if all(
                row["reservedConfigurationSlotId"]
                != row["configuration"]["configurationId"]
                for row in bound
            )
            else "fail",
        },
        {
            "case": "persisted_projection_restoration_and_distinct_hashes",
            "expected": "pass",
            "observed": "pass" if persisted_audit["success"] else "fail",
        },
    ]

    attacks: dict[str, list[dict[str, Any]]] = {}
    missing = deepcopy(bound)
    missing[0].pop("identityPlane")
    attacks["missing_binding"] = missing
    copied = deepcopy(bound)
    copied[1]["identityPlane"] = deepcopy(copied[0]["identityPlane"])
    attacks["copied_binding"] = copied
    for name, field in (
        ("forged_reservation_commitment", "reservationSlotIdentitySha256"),
        ("forged_runtime_commitment", "runtimeConfigurationIdentitySha256"),
        ("forged_physical_commitment", "physicalExecutionIdentitySha256"),
        ("forged_binding_commitment", "logicalIdentityBindingSha256"),
    ):
        attack = deepcopy(bound)
        attack[0]["identityPlane"][field] = "0" * 64
        attacks[name] = attack
    expansion = deepcopy(bound)
    expansion[0]["identityPlane"]["physicalToLogicalExpansion"]["members"].pop()
    attacks["ambiguous_expansion_membership"] = expansion
    field_loss = json.loads(json.dumps(bound, sort_keys=True))
    field_loss[0]["identityPlane"]["runtimeConfiguration"].pop("mode")
    attacks["serialization_field_loss"] = field_loss
    duplicate = [_strip_identity(bound[0]), _strip_identity(bound[0])]
    duplicate_failed = False
    try:
        bind_identity_plane(duplicate, physical_identity_resolver=_physical_key)
    except IdentityPlaneError:
        duplicate_failed = True
    cases.append(
        {
            "case": "duplicate_logical_slot",
            "expected": "fail_closed",
            "observed": "fail_closed" if duplicate_failed else "accepted",
        }
    )
    for name, attack in attacks.items():
        audit = validate_bound_identity_roster(
            attack, physical_identity_resolver=_physical_key
        )
        cases.append(
            {
                "case": name,
                "expected": "fail_closed",
                "observed": "fail_closed" if not audit["success"] else "accepted",
                "errorCount": len(audit["errors"]),
            }
        )

    single_local = validate_bound_identity_record(
        bound[0], physical_identity_resolver=_physical_key
    )
    success = (
        valid_audit["success"]
        and persisted_audit["success"]
        and single_local["success"]
        and all(row["expected"] == row["observed"] for row in cases)
    )
    return (
        {
            "schemaVersion": "e07.s08l.adversarial-qualification.v1",
            "researchStepId": "S08L",
            "success": success,
            "validLogicalRows": len(bound),
            "validPhysicalExecutionIdentities": len(physical_ids),
            "validLogicalResultIdentities": len(
                {row["identityPlane"]["logicalResultIdentitySha256"] for row in bound}
            ),
            "persistedProjectionAudit": persisted_audit,
            "caseCount": len(cases),
            "failClosedCaseCount": sum(
                row["expected"] == "fail_closed" for row in cases
            ),
            "outcomeValuesUsed": 0,
        },
        cases,
    )


def access_validation(
    configurations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    by_hash, by_id = _catalog()
    representative = {
        task: min(
            (
                row
                for row in configurations
                if row["taskId"] == task and row["mode"] == "single_policy"
            ),
            key=lambda row: row["configurationId"],
        )
        for task in TASK_IDS
    }
    rows = []
    for task_id, configuration in representative.items():
        action = build_action(configuration, by_hash, by_id)
        for record in suite.records.values():
            if record.task_id != task_id or record.split is Split.TRAIN:
                continue
            materializations_before = suite.broker.audit.materializer_invocations
            try:
                suite.open_for_action(
                    task_id,
                    record.scenario_id,
                    AccessGrant(
                        AccessPhase.VALIDATION
                        if record.split is Split.VALIDATION
                        else AccessPhase.CONFIRMATION,
                        "0" * 64 if record.split is Split.CONFIRMATION else None,
                    ),
                    action,
                )
                denied = False
            except AccessDeniedError:
                denied = True
            rows.append(
                {
                    "taskId": task_id,
                    "split": record.split.value,
                    "deniedBeforeMaterialization": denied
                    and suite.broker.audit.materializer_invocations
                    == materializations_before,
                }
            )
    return {
        "schemaVersion": "e07.s08l.access-control-validation.v1",
        "researchStepId": "S08L",
        "success": len(rows) == 16
        and all(row["deniedBeforeMaterialization"] for row in rows),
        "rows": rows,
        "validationOutcomeAccesses": 0,
        "confirmationOutcomeAccesses": 0,
        "s08kCacheReads": 0,
    }


def dependency_validation() -> dict[str, Any]:
    prohibited_prefixes = (
        "src.surrogate_models",
        "src.surrogate_remediation",
    )
    loaded = sorted(
        name for name in sys.modules if name.startswith(prohibited_prefixes)
    )
    source_paths = (
        REPOSITORY / "src/portfolio_search/identity.py",
        REPOSITORY / "src/portfolio_search/execution.py",
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_paths)
    prohibited_artifact_paths = [
        token
        for token in (
            "/artifacts/research_steps/S06/models",
            "/artifacts/research_steps/S06/embeddings",
            "/artifacts/research_steps/S06A/models",
            "/artifacts/research_steps/S06A/embeddings",
        )
        if token in source
    ]
    return {
        "schemaVersion": "e07.s08l.dependency-exclusion-audit.v1",
        "researchStepId": "S08L",
        "success": not loaded and not prohibited_artifact_paths,
        "loadedProhibitedModules": loaded,
        "prohibitedArtifactPathsInIdentityRuntime": prohibited_artifact_paths,
        "s06ModelLoads": 0,
        "s06EmbeddingLoads": 0,
        "s06aModelLoads": 0,
        "s06aEmbeddingLoads": 0,
        "s07ArmPromotionUses": 0,
        "historicalQuarantineOutcomeReads": 0,
        "s08kCacheReads": 0,
    }


def main() -> int:
    started = time.perf_counter()
    protocol_sha = sha256_file(PROTOCOL)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("S08L protocol changed after preregistration")
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if preregistration["protocolSha256"] != protocol_sha:
        raise RuntimeError("S08L preregistration no longer binds the protocol")
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["schemaVersion"] != (
        "e07.s08l.logical-identity-plane-preregistration.v1"
    ):
        raise RuntimeError("unexpected S08L protocol")
    input_freeze = json.loads(INPUT_FREEZE.read_text(encoding="utf-8"))
    expected_trees = input_freeze["historicalArtifactTreeSha256"]
    before = {step: tree_digest(ARTIFACT_ROOT / step) for step in IMMUTABLE_STEPS}
    immutable_before = before == expected_trees

    s08p_protocol = checked_protocol(S08P / "s08p_portfolio_protocol.yaml")
    candidates = read_jsonl(S08P / "candidate_eligibility_registry.jsonl")
    member_sets = read_jsonl(S08P / "portfolio_member_sets.jsonl")
    configurations = read_jsonl(S08P / "portfolio_seed_registry.jsonl")
    budget = pd.read_parquet(S08P / "budget_slot_ledger.parquet")
    legality = validate_portfolio_registry(
        s08p_protocol, candidates, member_sets, configurations
    )
    s08p_freeze = json.loads(
        (S08P / "preregistration_freeze.json").read_text(encoding="utf-8")
    )
    candidate_sha = candidate_commitment(candidates)
    complete_plan_sha = plan_digest(
        candidates, member_sets, configurations, budget.to_dict("records")
    )
    smoke_ids = set(budget.loc[budget["smoke"], "configurationId"].astype(str).tolist())
    bindings = validate_executable_bindings(configurations, smoke_ids)
    bindings.update(
        {
            "schemaVersion": "e07.s08l.configuration-binding-validation.v1",
            "researchStepId": "S08L",
        }
    )

    configuration_by_id = {str(row["configurationId"]): row for row in configurations}
    initial = _initial_work(budget, configuration_by_id)
    adaptive_batches = _adaptive_fixture_work(budget, configurations)
    batches = [initial, *adaptive_batches]
    bound = [row for batch in batches for row in batch]
    if len(bound) != 11008:
        raise RuntimeError("identity qualification did not cover 11,008 slots")
    physical_by_slot = {
        str(row["logicalSlotId"]): row["identityPlane"]["physicalExecution"][
            "physicalDedupEquivalenceSha256"
        ]
        for row in bound
    }
    resolver = _precomputed_resolver(physical_by_slot)
    batch_audits = [
        validate_bound_identity_roster(batch, physical_identity_resolver=resolver)
        for batch in batches
    ]
    replay_batches = [
        bind_identity_plane(
            [_strip_identity(row) for row in batch],
            physical_identity_resolver=resolver,
        )
        for batch in batches
    ]
    reverse_batches = [
        bind_identity_plane(
            [_strip_identity(row) for row in reversed(batch)],
            physical_identity_resolver=resolver,
        )
        for batch in batches
    ]
    replay = [row for batch in replay_batches for row in batch]
    reverse = [row for batch in reverse_batches for row in batch]
    primary_digest = _identity_digest(bound)
    replay_digest = _identity_digest(replay)
    reverse_digest = _identity_digest(reverse)
    round_trip = json.loads(json.dumps(bound, sort_keys=True, allow_nan=False))
    round_trip_digest = _identity_digest(round_trip)

    planes = [row["identityPlane"] for row in bound]
    reservation_ids = {row["reservationSlotIdentitySha256"] for row in planes}
    runtime_ids = {row["runtimeConfigurationIdentitySha256"] for row in planes}
    physical_ids = {row["physicalExecutionIdentitySha256"] for row in planes}
    logical_ids = {row["logicalResultIdentitySha256"] for row in planes}
    logical_slots = {str(row["logicalSlotId"]) for row in bound}
    expansion_group_keys = {
        (
            plane["physicalExecutionIdentitySha256"],
            plane["physicalToLogicalExpansionCommitmentSha256"],
        )
        for plane in planes
    }
    group_size_distribution: Counter[int] = Counter()
    seen_groups: set[tuple[str, str]] = set()
    expansion_members = 0
    for plane in planes:
        key = (
            plane["physicalExecutionIdentitySha256"],
            plane["physicalToLogicalExpansionCommitmentSha256"],
        )
        if key in seen_groups:
            continue
        seen_groups.add(key)
        count = int(plane["physicalToLogicalExpansion"]["logicalResultCount"])
        group_size_distribution[count] += 1
        expansion_members += count
    mismatch_count = sum(
        str(row["reservedConfigurationSlotId"])
        != str(row["configuration"]["configurationId"])
        for row in bound
    )
    adaptive_role_counts = Counter(
        str(row["configurationRole"]) for row in bound if row["stage"] == "adaptive"
    )
    full_qualification = {
        "schemaVersion": "e07.s08l.full-structural-identity-qualification.v1",
        "researchStepId": "S08L",
        "identityPlaneVersion": IDENTITY_PLANE_VERSION,
        "success": all(row["success"] for row in batch_audits)
        and len(logical_slots) == 11008
        and len(reservation_ids) == 11008
        and len(logical_ids) == 11008
        and expansion_members == 11008
        and primary_digest == replay_digest == reverse_digest == round_trip_digest,
        "batches": [
            {
                "stage": "initial" if index == 0 else "adaptive",
                "generation": index,
                **audit,
            }
            for index, audit in enumerate(batch_audits)
        ],
        "logicalReservations": len(bound),
        "uniqueLogicalSlots": len(logical_slots),
        "uniqueReservationSlotIdentities": len(reservation_ids),
        "uniqueRuntimeConfigurationIdentities": len(runtime_ids),
        "uniquePhysicalExecutionIdentitiesAcrossBatches": len(physical_ids),
        "uniqueLogicalResultIdentities": len(logical_ids),
        "physicalExpansionGroups": len(expansion_group_keys),
        "physicalDedupSavedLogicalRowsAcrossBatches": sum(
            audit["physicalDedupSavedLogicalRows"] for audit in batch_audits
        ),
        "physicalExpansionMemberCount": expansion_members,
        "reservationRuntimeReferenceMismatchRows": mismatch_count,
        "adaptiveRoleCounts": dict(sorted(adaptive_role_counts.items())),
        "groupSizeDistribution": {
            str(size): count for size, count in sorted(group_size_distribution.items())
        },
        "primaryIdentityDigestSha256": primary_digest,
        "replayIdentityDigestSha256": replay_digest,
        "reverseIdentityDigestSha256": reverse_digest,
        "roundTripIdentityDigestSha256": round_trip_digest,
        "outcomeValuesUsed": 0,
        "episodeEvaluations": 0,
    }
    serialization = {
        "schemaVersion": "e07.s08l.serialization-replay-worker-order.v1",
        "researchStepId": "S08L",
        "success": primary_digest
        == replay_digest
        == reverse_digest
        == round_trip_digest,
        "logicalRows": len(bound),
        "batchCount": len(batches),
        "primaryIdentityDigestSha256": primary_digest,
        "replayIdentityDigestSha256": replay_digest,
        "reverseIdentityDigestSha256": reverse_digest,
        "roundTripIdentityDigestSha256": round_trip_digest,
        "serialization": "canonical JSON round trip",
        "workerOrderSimulation": "forward versus reverse within each frozen batch",
    }
    accounting = {
        "schemaVersion": "e07.s08l.complete-accounting.v1",
        "researchStepId": "S08L",
        "success": len(configurations) == 1216
        and int(budget["smoke"].sum()) == 40
        and len(initial) == 4864
        and sum(len(batch) for batch in adaptive_batches) == 6144
        and len(bound) == 11008
        and expansion_members == 11008,
        "configurationBindings": len(configurations),
        "frozenSmokeStructuralBindings": int(budget["smoke"].sum()),
        "initialLogicalReservations": len(initial),
        "adaptiveLogicalReservations": sum(len(batch) for batch in adaptive_batches),
        "totalLogicalReservations": len(bound),
        "physicalExpansionMemberCount": expansion_members,
        "qualificationFixtureRows": len(bound),
        "episodeEvaluations": 0,
        "frozenSmokeRowsExecuted": 0,
        "efficacyRowsRead": 0,
        "archiveConstructionCalls": 0,
        "validationOutcomeAccesses": 0,
        "confirmationOutcomeAccesses": 0,
        "s09Actions": 0,
        "s08kCacheReads": 0,
    }

    adversarial, adversarial_cases = adversarial_qualification(configurations)
    access = access_validation(configurations)
    dependency = dependency_validation()

    candidate_pass = (
        len(candidates) == 512
        and candidate_sha == s08p_freeze["candidateEligibilityCommitmentSha256"]
        and all(
            row["eligible"]
            and not row["eligibilityUsesS07ArmMembership"]
            and not row["eligibilityUsesRejectedModelOrEmbedding"]
            and not row["promotionEvidence"]
            for row in candidates
        )
    )
    plan_pass = (
        complete_plan_sha == s08p_freeze["completePlanSha256"]
        and len(budget) == 11008
        and set(budget["split"]) == {"train"}
        and budget.groupby("generation").size().to_dict()
        == {0: 4864, 1: 1024, 2: 1024, 3: 1024, 4: 1024, 5: 1024, 6: 1024}
    )
    g04_pass = (
        full_qualification["success"]
        and adversarial["success"]
        and serialization["success"]
        and bindings["success"]
    )
    gate_rows = [
        {
            "gateId": "G01",
            "requirement": "frozen inputs and S08P-S08K historical artifacts unchanged",
            "status": "pass" if immutable_before else "blocked",
        },
        {
            "gateId": "G02",
            "requirement": "512 canonical candidates and runtime configurations remain authoritative",
            "status": "pass" if candidate_pass else "blocked",
        },
        {
            "gateId": "G03",
            "requirement": "S08P design, legality, plan, and complete-equivalence semantics unchanged",
            "status": "pass"
            if plan_pass and legality["illegalCompositions"] == 0
            else "blocked",
        },
        {
            "gateId": "G04",
            "requirement": "outcome-independent four-plane logical identity contract qualified fail closed",
            "status": "pass" if g04_pass else "blocked",
        },
        {
            "gateId": "G05",
            "requirement": "protected access and rejected dependencies denied",
            "status": "pass"
            if access["success"] and dependency["success"]
            else "blocked",
        },
        {
            "gateId": "G06",
            "requirement": "reservation, logical, physical, and zero-execution accounting complete",
            "status": "pass" if accounting["success"] else "blocked",
        },
    ]
    blocked = [row["gateId"] for row in gate_rows if row["status"] != "pass"]
    gate = {
        "schemaVersion": "e07.s08l.s08-execution-eligibility-gate.v1",
        "researchStepId": "S08L",
        "success": not blocked,
        "technicalEligibilityPass": not blocked,
        "executionAuthorized": False,
        "freshExecutionRequiresSeparateApproval": True,
        "blockedGateIds": blocked,
        "rows": gate_rows,
        "recommendedNextAction": (
            "Review S08L and, only under separate approval, preregister a genuinely fresh S08 execution namespace."
            if not blocked
            else "Return for review without smoke or execution; do not weaken S08P."
        ),
    }

    after = {step: tree_digest(ARTIFACT_ROOT / step) for step in IMMUTABLE_STEPS}
    immutable = {
        "schemaVersion": "e07.s08l.immutable-input-validation.v1",
        "researchStepId": "S08L",
        "success": immutable_before and before == after,
        "expectedTreeSha256": expected_trees,
        "beforeTreeSha256": before,
        "afterTreeSha256": after,
        "s08kApprovedIntegrityMetadataFilesRead": [
            "/artifacts/research_steps/S08K/logical_identity_failure_forensics.json",
            "/artifacts/research_steps/S08K/logical_identity_restoration_audit.json",
            "/artifacts/research_steps/S08K/quarantine_manifest.json",
            "/artifacts/research_steps/S08K/research_step_full_results.md",
        ],
        "s08kCacheReads": 0,
        "s08kOutcomeOrEfficacyFieldsRead": 0,
    }
    no_mutation = {
        "schemaVersion": "e07.s08l.no-mutation-audit.v1",
        "researchStepId": "S08L",
        "success": before == after,
        "historicalTreesUnchanged": before == after,
        "s05ArchiveMutations": 0,
        "portfolioArchiveMutations": 0,
        "historicalArtifactMutations": 0,
        "cacheNamespacesCreated": 0,
        "episodeEvaluations": 0,
        "smokeRowsExecuted": 0,
        "validationOutcomeAccesses": 0,
        "confirmationOutcomeAccesses": 0,
        "s09Actions": 0,
    }
    # G01 is controlling and includes the final no-mutation audit.
    if not immutable["success"] or not no_mutation["success"]:
        gate_rows[0]["status"] = "blocked"
        blocked = sorted(
            {row["gateId"] for row in gate_rows if row["status"] != "pass"}
        )
        gate.update(
            {
                "success": False,
                "technicalEligibilityPass": False,
                "blockedGateIds": blocked,
                "rows": gate_rows,
            }
        )

    identity_spec = {
        "schemaVersion": "e07.s08l.logical-identity-plane-spec.v1",
        "researchStepId": "S08L",
        "success": protocol["identityContract"]["version"] == IDENTITY_PLANE_VERSION,
        "identityPlaneVersion": IDENTITY_PLANE_VERSION,
        "generatedBeforeOutcome": True,
        "reservationConfigurationReferenceSemantics": protocol["identityContract"][
            "reservationSlot"
        ]["configurationReferenceSemantics"],
        "identities": {
            "reservationSlot": "exact frozen S08P budget address",
            "runtimeConfiguration": "immutable definition actually dispatched",
            "physicalExecution": "complete-equivalence native work",
            "logicalResult": "one reserved attribution of physical work",
        },
        "physicalExpansionMembership": (
            "sorted complete logical-result membership committed before the row outcome"
        ),
        "failClosedOn": [
            "missing metadata",
            "copied metadata",
            "duplicate logical slot",
            "ambiguous physical membership",
            "inconsistent metadata",
            "forged commitment",
        ],
        "scientificDesignChanged": False,
        "s08pEstimandsChanged": False,
        "s08pComparatorsChanged": False,
        "s08pSelectorsChanged": False,
        "s08pCostsChanged": False,
        "s08pBudgetsChanged": False,
        "s08pStatisticalRulesChanged": False,
    }

    samples = []
    for batch_name, batch in (
        ("initial", initial),
        ("adaptive_generation_1", adaptive_batches[0]),
    ):
        by_role: dict[str, Mapping[str, Any]] = {}
        for row in batch:
            by_role.setdefault(str(row["configurationRole"]), row)
        for role, row in sorted(by_role.items()):
            samples.append(
                {
                    "batch": batch_name,
                    "configurationRole": role,
                    "logicalSlotId": row["logicalSlotId"],
                    "reservedConfigurationSlotId": row["reservedConfigurationSlotId"],
                    "runtimeConfigurationId": row["configuration"]["configurationId"],
                    "identityPlane": row["identityPlane"],
                }
            )

    overall_success = all(
        (
            identity_spec["success"],
            full_qualification["success"],
            adversarial["success"],
            serialization["success"],
            accounting["success"],
            bindings["success"],
            access["success"],
            dependency["success"],
            immutable["success"],
            no_mutation["success"],
            gate["success"],
        )
    )
    validation = {
        "schemaVersion": "e07.s08l.validation-summary.v1",
        "researchStepId": "S08L",
        "success": overall_success,
        "checks": {
            "prospectiveProtocol": protocol_sha == EXPECTED_PROTOCOL_SHA256,
            "identitySpecification": identity_spec["success"],
            "fullStructuralRoster": full_qualification["success"],
            "adversarialFailClosed": adversarial["success"],
            "serializationReplayWorkerOrder": serialization["success"],
            "completeAccounting": accounting["success"],
            "all1216Bindings": bindings["success"],
            "protectedDenial": access["success"],
            "dependencyExclusion": dependency["success"],
            "immutableHistoricalTrees": immutable["success"],
            "noMutation": no_mutation["success"],
            "g01ToG06": gate["success"],
        },
        "scientificDesignChanged": False,
        "outcomeValuesUsed": 0,
        "episodeEvaluations": 0,
        "archiveConstructionCalls": 0,
        "protectedOutcomeAccesses": 0,
        "wallSeconds": time.perf_counter() - started,
    }

    _write_json(OUTPUT / "identity_plane_spec.json", identity_spec)
    _write_json(
        OUTPUT / "full_structural_identity_qualification.json",
        full_qualification,
    )
    _write_json(OUTPUT / "adversarial_identity_qualification.json", adversarial)
    _write_jsonl(OUTPUT / "adversarial_identity_cases.jsonl", adversarial_cases)
    _write_json(
        OUTPUT / "serialization_replay_worker_order_validation.json",
        serialization,
    )
    _write_json(OUTPUT / "complete_accounting.json", accounting)
    _write_json(OUTPUT / "configuration_binding_validation.json", bindings)
    _write_json(OUTPUT / "access_control_validation.json", access)
    _write_json(OUTPUT / "dependency_exclusion_audit.json", dependency)
    _write_json(OUTPUT / "immutable_input_validation.json", immutable)
    _write_json(OUTPUT / "no_mutation_audit.json", no_mutation)
    _write_json(OUTPUT / "s08_execution_eligibility_gate.json", gate)
    _write_jsonl(OUTPUT / "logical_identity_samples.jsonl", samples)
    _write_json(OUTPUT / "validation_summary.json", validation)
    _write_json(
        OUTPUT / "environment.json",
        {
            "schemaVersion": "e07.s08l.environment.v1",
            "researchStepId": "S08L",
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "workers": 1,
            "numericThreadsPerWorker": 1,
            "gpuUsed": False,
            "networkUsed": False,
            "newDependenciesInstalled": [],
        },
    )
    _write_json(
        OUTPUT / "provenance.json",
        {
            "schemaVersion": "e07.s08l.provenance.v1",
            "researchStepId": "S08L",
            "protocolPath": str(PROTOCOL),
            "protocolSha256": protocol_sha,
            "inputFreezePath": str(INPUT_FREEZE),
            "inputFreezeSha256": sha256_file(INPUT_FREEZE),
            "preregistrationPath": str(PREREGISTRATION),
            "preregistrationSha256": sha256_file(PREREGISTRATION),
            "repositoryHeadBeforeQualification": input_freeze["repository"][
                "headCommit"
            ],
            "scriptPath": str(Path(__file__).resolve()),
            "scriptSha256": sha256_file(Path(__file__).resolve()),
            "s08kCacheRead": False,
            "s08kOutcomeOrEfficacyFieldsRead": False,
        },
    )
    _write_json(
        OUTPUT / "status.json",
        {
            "researchStepId": "S08L",
            "stepNumber": 8,
            "success": overall_success,
            "status": "complete_qualified"
            if overall_success
            else "complete_fail_closed",
            "artifactsWritten": [
                "identity_plane_spec.json",
                "full_structural_identity_qualification.json",
                "adversarial_identity_qualification.json",
                "adversarial_identity_cases.jsonl",
                "serialization_replay_worker_order_validation.json",
                "complete_accounting.json",
                "configuration_binding_validation.json",
                "access_control_validation.json",
                "dependency_exclusion_audit.json",
                "immutable_input_validation.json",
                "no_mutation_audit.json",
                "s08_execution_eligibility_gate.json",
                "logical_identity_samples.jsonl",
                "validation_summary.json",
                "environment.json",
                "provenance.json",
                "status.json",
            ],
            "validationResult": "PASS" if overall_success else "FAIL CLOSED",
            "caveatsOrBlockers": (
                [
                    "Qualification only; no efficacy or execution authorization.",
                    "A separately approved genuinely fresh S08 execution is required.",
                    "Every historical quarantine remains prohibited.",
                ]
                if overall_success
                else [f"Blocked gates: {gate['blockedGateIds']}"]
            ),
            "recommendedNextAction": (
                "Hand control back for separate fresh-execution review; do not start S09."
                if overall_success
                else "Review the fail-closed qualification; do not execute S08 or start S09."
            ),
        },
    )
    return 0 if overall_success else 4


if __name__ == "__main__":
    raise SystemExit(main())
