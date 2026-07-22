"""S08 pre-smoke integrity and executable-binding audit.

The frozen S08P budget cannot be entered unless every initial configuration,
including every one of the 40 smoke rows, binds through the qualified S08A
dispatcher.  This module deliberately owns no episode runner, outcome reader,
search/archive implementation, validation materializer, or protected unseal
path.  A failed gate therefore produces audit evidence and stops at zero
evaluations.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from src.environment_suite import (
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
    EnvironmentSuite,
)
from src.environment_suite.dsl_adapters import dsl_action
from src.environment_suite.portfolio_adapters import PortfolioDispatcher
from src.policy_dsl import compile_policy
from src.portfolio_preregistration.core import (
    candidate_commitment,
    canonical_hash,
    checked_protocol,
    plan_digest,
    validate_portfolio_registry,
    verify_frozen_inputs,
)


REPOSITORY = Path(__file__).resolve().parents[2]
CONTROL_PATH = REPOSITORY / "configs/portfolio/s08_execution_control.yaml"
TASK_REGISTRY = REPOSITORY / "configs/environment_suite/task_registry.yaml"
SPLIT_MANIFEST = REPOSITORY / "configs/environment_suite/split_manifest.json"
ARTIFACT_ROOT = Path("/artifacts/research_steps")
S05 = ARTIFACT_ROOT / "S05"
S08P = ARTIFACT_ROOT / "S08P"
S08A = ARTIFACT_ROOT / "S08A"
S08 = ARTIFACT_ROOT / "S08"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(path: str | Path) -> str:
    root = Path(path)
    rows = [
        (str(item.relative_to(root)), sha256_file(item), item.stat().st_size)
        for item in sorted(
            candidate for candidate in root.rglob("*") if candidate.is_file()
        )
    ]
    return canonical_hash("E07/S08A/immutable-tree/v1", rows)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _manifest_validation(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for item in manifest["artifacts"]:
        path = root / item["path"]
        actual = sha256_file(path) if path.is_file() else None
        rows.append(
            {
                "path": str(path),
                "expectedSha256": item["sha256"],
                "actualSha256": actual,
                "pass": actual == item["sha256"],
            }
        )
    return {
        "root": str(root),
        "artifactRows": len(rows),
        "success": bool(rows) and all(row["pass"] for row in rows),
        "rows": rows,
    }


def _load_catalog() -> tuple[
    dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]
]:
    rows = read_jsonl(S05 / "policy_catalog.jsonl")
    by_hash = {str(row["policySha256"]): row["document"] for row in rows}
    by_id = {str(row["policyId"]): row["document"] for row in rows}
    return by_hash, by_id


def _single_action(
    task_id: str,
    configuration: Mapping[str, Any],
    by_hash: Mapping[str, Mapping[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]],
):
    member = configuration["members"][0]
    document = by_hash[str(member["policySha256"])]
    compiled = compile_policy(document)
    if compiled.policy_sha256 != member["policySha256"]:
        raise RuntimeError("single member no longer recompiles to its frozen hash")
    if task_id != "e07_s02_chimera_1d":
        return dsl_action([document])
    carrier = str(member["nativeCarrier"])
    if carrier == "Bubble":
        counterpart = by_id["insertion_cell_view_v1"]
        documents = [document, counterpart]
        bindings = {
            "Bubble": str(document["policyId"]),
            "Insertion": str(counterpart["policyId"]),
        }
    elif carrier == "Insertion":
        counterpart = by_id["bubble_cell_view_v1"]
        documents = [counterpart, document]
        bindings = {
            "Bubble": str(counterpart["policyId"]),
            "Insertion": str(document["policyId"]),
        }
    else:
        raise RuntimeError("Chimera single has a non-native carrier")
    return dsl_action(documents, native_policy_bindings=bindings)


def validate_executable_bindings(
    configurations: Sequence[Mapping[str, Any]],
    smoke_configuration_ids: set[str],
) -> dict[str, Any]:
    """Bind every frozen initial row without materializing a scenario."""

    by_hash, by_id = _load_catalog()
    candidate_rows = read_jsonl(S08P / "candidate_eligibility_registry.jsonl")
    candidate_hashes = {(row["taskId"], row["policySha256"]) for row in candidate_rows}
    errors: list[dict[str, Any]] = []
    mode_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    successful = 0
    for configuration in sorted(
        configurations,
        key=lambda row: (row["taskId"], row["mode"], row["configurationId"]),
    ):
        task_id = str(configuration["taskId"])
        mode = str(configuration["mode"])
        mode_counts[mode] += 1
        task_counts[task_id] += 1
        try:
            if any(
                (task_id, member["policySha256"]) not in candidate_hashes
                for member in configuration["members"]
            ):
                raise RuntimeError(
                    "configuration references an ineligible task-policy pair"
                )
            if mode == "single_policy":
                _single_action(task_id, configuration, by_hash, by_id)
            else:
                documents = [
                    by_hash[member["policySha256"]]
                    for member in configuration["members"]
                ]
                policies = {
                    item.policy_id: item for item in map(compile_policy, documents)
                }
                if len(policies) != len(documents):
                    raise RuntimeError("portfolio member policy IDs are not unique")
                PortfolioDispatcher(configuration, policies)
            successful += 1
        except Exception as exc:  # Audit every frozen row instead of stopping at first.
            carrier_counts = Counter(
                str(member["nativeCarrier"]) for member in configuration["members"]
            )
            errors.append(
                {
                    "taskId": task_id,
                    "mode": mode,
                    "configurationId": configuration["configurationId"],
                    "memberSetId": configuration["memberSetId"],
                    "portfolioSize": int(configuration["portfolioSize"]),
                    "nativeCarrierCounts": dict(sorted(carrier_counts.items())),
                    "isFrozenSmokeConfiguration": configuration["configurationId"]
                    in smoke_configuration_ids,
                    "errorType": type(exc).__name__,
                    "error": str(exc),
                }
            )
    smoke_errors = [row for row in errors if row["isFrozenSmokeConfiguration"]]
    return {
        "schemaVersion": "e07.s08.configuration-binding-validation.v1",
        "researchStepId": "S08",
        "success": not errors,
        "configurationRowsChecked": len(configurations),
        "configurationRowsBound": successful,
        "configurationRowsBlocked": len(errors),
        "modeCounts": dict(sorted(mode_counts.items())),
        "taskCounts": dict(sorted(task_counts.items())),
        "blockedByTask": dict(sorted(Counter(row["taskId"] for row in errors).items())),
        "blockedByMode": dict(sorted(Counter(row["mode"] for row in errors).items())),
        "frozenSmokeConfigurationsBlocked": len(smoke_errors),
        "frozenSmokeConfigurationIdsBlocked": sorted(
            row["configurationId"] for row in smoke_errors
        ),
        "errors": errors,
        "scenarioMaterializations": 0,
        "episodeEvaluations": 0,
    }


def _access_denial_validation(
    configurations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    by_hash, by_id = _load_catalog()
    singles = {
        row["taskId"]: row
        for row in configurations
        if row["mode"] == "single_policy"
        and row["configurationId"]
        == min(
            item["configurationId"]
            for item in configurations
            if item["taskId"] == row["taskId"] and item["mode"] == "single_policy"
        )
    }
    records_by_task_split = {
        (record.task_id, record.split.value): record
        for record in suite.records.values()
    }
    before = suite.broker.audit.to_dict()
    rows = []
    for task_id in sorted(singles):
        action = _single_action(task_id, singles[task_id], by_hash, by_id)
        for split, grant in (
            ("validation", AccessGrant(AccessPhase.VALIDATION)),
            (
                "confirmation",
                AccessGrant(AccessPhase.CONFIRMATION, "0" * 64),
            ),
        ):
            record = records_by_task_split[(task_id, split)]
            denied = False
            try:
                suite.open_for_action(task_id, record.scenario_id, grant, action)
            except AccessDeniedError:
                denied = True
            rows.append(
                {
                    "taskId": task_id,
                    "split": split,
                    "scenarioId": record.scenario_id,
                    "deniedBeforeMaterialization": denied,
                }
            )
    after = suite.broker.audit.to_dict()
    success = all(row["deniedBeforeMaterialization"] for row in rows) and (
        before["materializerInvocations"] == after["materializerInvocations"]
    )
    return {
        "schemaVersion": "e07.s08.access-control-validation.v1",
        "researchStepId": "S08",
        "success": success,
        "rows": rows,
        "brokerAuditBefore": before,
        "brokerAuditAfter": after,
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
        "protectedOutcomeMaterializations": 0,
    }


def run_preflight(control_path: str | Path = CONTROL_PATH) -> dict[str, Any]:
    control = yaml.safe_load(Path(control_path).read_text(encoding="utf-8"))
    if control.get("schemaVersion") != "e07.s08.execution-control.v1":
        raise RuntimeError("unexpected S08 execution-control schema")
    if not control.get("failClosedBeforeSmoke"):
        raise RuntimeError("S08 control must fail closed before smoke")
    before_trees = {"S05": tree_digest(S05), "S08P": tree_digest(S08P)}
    file_checks = []
    for step, files in control["expectedFrozenFiles"].items():
        root = ARTIFACT_ROOT / step
        for name, expected_raw in sorted(files.items()):
            expected = str(expected_raw).lower()
            path = root / name
            actual = sha256_file(path)
            file_checks.append(
                {
                    "step": step,
                    "path": str(path),
                    "expectedSha256": expected,
                    "actualSha256": actual,
                    "pass": actual == expected,
                }
            )
    protocol = checked_protocol(S08P / "s08p_portfolio_protocol.yaml")
    frozen_input_checks = verify_frozen_inputs(protocol)
    manifest_checks = [_manifest_validation(S08P), _manifest_validation(S08A)]

    candidates = read_jsonl(S08P / "candidate_eligibility_registry.jsonl")
    member_sets = read_jsonl(S08P / "portfolio_member_sets.jsonl")
    configurations = read_jsonl(S08P / "portfolio_seed_registry.jsonl")
    budget = pd.read_parquet(S08P / "budget_slot_ledger.parquet")
    budget_rows = budget.to_dict(orient="records")
    legality = validate_portfolio_registry(
        protocol, candidates, member_sets, configurations
    )
    frozen = json.loads((S08P / "preregistration_freeze.json").read_text())
    candidate_hash = candidate_commitment(candidates)
    computed_plan = plan_digest(candidates, member_sets, configurations, budget_rows)
    smoke_ids = set(budget.loc[budget["smoke"], "configurationId"].astype(str))
    bindings = validate_executable_bindings(configurations, smoke_ids)
    access = _access_denial_validation(configurations)

    historical_gate = json.loads(
        (S08A / "s08_execution_eligibility_gate.json").read_text(encoding="utf-8")
    )
    historical_rows = {row["gateId"]: row for row in historical_gate["rows"]}
    required_gates = list(control["requiredGateIds"])
    historical_gate_pass = (
        set(historical_rows) == set(required_gates)
        and all(historical_rows[key]["status"] == "pass" for key in required_gates)
        and historical_gate.get("technicalEligibilityPass") is True
    )
    loaded_prohibited = sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in control["prohibitedModulePrefixes"]
        )
    )
    dependency = {
        "schemaVersion": "e07.s08.dependency-exclusion-audit.v1",
        "researchStepId": "S08",
        "success": not loaded_prohibited,
        "loadedProhibitedModules": loaded_prohibited,
        "rejectedModelArtifactsOpened": 0,
        "rejectedEmbeddingArtifactsOpened": 0,
        "pseudoLabelsUsed": 0,
        "warmStartsUsed": 0,
        "distillationUses": 0,
        "s07ArmPromotionUses": sum(
            bool(row.get("s07ArmMembershipUsed")) for row in configurations
        ),
    }
    accounting = {
        "schemaVersion": "e07.s08.pre-smoke-budget-accounting.v1",
        "researchStepId": "S08",
        "success": len(budget) == int(control["frozenCounts"]["trainingLogicalRows"])
        and int(budget["smoke"].sum())
        == int(control["frozenCounts"]["smokeLogicalRows"])
        and set(budget["split"]) == {"train"},
        "frozenTrainingLogicalRows": len(budget),
        "frozenSmokeLogicalRows": int(budget["smoke"].sum()),
        "validationLogicalCeiling": int(
            control["frozenCounts"]["validationLogicalCeiling"]
        ),
        "confirmationLogicalRows": int(
            control["frozenCounts"]["confirmationLogicalRows"]
        ),
        "smokeLogicalRowsExecuted": 0,
        "substantiveTrainingLogicalRowsExecuted": 0,
        "validationLogicalRowsExecuted": 0,
        "confirmationLogicalRowsExecuted": 0,
        "portfolioArchiveMutations": 0,
    }
    hash_success = (
        all(row["pass"] for row in file_checks)
        and all(item["success"] for item in manifest_checks)
        and before_trees == control["expectedTrees"]
        and candidate_hash == frozen["candidateEligibilityCommitmentSha256"]
        and computed_plan == frozen["completePlanSha256"]
    )
    candidate_fields_pass = all(
        row.get("eligibilityUsesS07ArmMembership") is False
        and row.get("eligibilityUsesRejectedModelOrEmbedding") is False
        and row.get("promotionEvidence") is False
        and row.get("eligible") is True
        for row in candidates
    )
    gate_rows = [
        {
            "gateId": "G01",
            "requirement": "all frozen input hashes and rejection decisions unchanged",
            "status": "pass" if hash_success else "blocked",
        },
        {
            "gateId": "G02",
            "requirement": "512 authoritative candidates recompile and exclude S07 arm efficacy",
            "status": "pass"
            if len(candidates) == 512 and candidate_fields_pass
            else "blocked",
        },
        {
            "gateId": "G03",
            "requirement": "portfolio compositions, selectors, costs, and paired plan are legal",
            "status": "pass" if legality["illegalCompositions"] == 0 else "blocked",
        },
        {
            "gateId": "G04",
            "requirement": "every frozen initial/smoke portfolio binds through the qualified multi-policy adapter",
            "status": "pass" if bindings["success"] else "blocked",
            "basis": (
                "full frozen registry executable-binding audit; static S08P legality and "
                "qualification-only S08A fixtures are insufficient when they omit frozen carrier cardinalities"
            ),
        },
        {
            "gateId": "G05",
            "requirement": "protected access denied and rejected models/embeddings isolated",
            "status": "pass"
            if access["success"] and dependency["success"]
            else "blocked",
        },
        {
            "gateId": "G06",
            "requirement": "smoke, training, validation ceiling, and stopping budgets completely accounted",
            "status": "pass" if accounting["success"] else "blocked",
        },
    ]
    blocked = [row["gateId"] for row in gate_rows if row["status"] != "pass"]
    after_trees = {"S05": tree_digest(S05), "S08P": tree_digest(S08P)}
    no_mutation = before_trees == after_trees == control["expectedTrees"]
    return {
        "schemaVersion": "e07.s08.pre-smoke-result.v1",
        "researchStepId": "S08",
        "success": not blocked,
        "status": "eligible_for_smoke" if not blocked else "blocked_before_smoke",
        "blockedGateIds": blocked,
        "gateRows": gate_rows,
        "historicalS08AGatePass": historical_gate_pass,
        "hashValidation": {
            "success": hash_success,
            "fileChecks": file_checks,
            "frozenInputChecks": frozen_input_checks,
            "manifestChecks": manifest_checks,
            "candidateEligibilityCommitmentExpected": frozen[
                "candidateEligibilityCommitmentSha256"
            ],
            "candidateEligibilityCommitmentActual": candidate_hash,
            "completePlanSha256Expected": frozen["completePlanSha256"],
            "completePlanSha256Actual": computed_plan,
            "treesBefore": before_trees,
            "treesAfter": after_trees,
        },
        "legalityValidation": legality,
        "bindingValidation": bindings,
        "accessValidation": access,
        "dependencyValidation": dependency,
        "accounting": accounting,
        "noMutation": {
            "success": no_mutation,
            "s05Unchanged": before_trees["S05"] == after_trees["S05"],
            "s08pUnchanged": before_trees["S08P"] == after_trees["S08P"],
            "s05ArchiveMutations": 0,
            "portfolioArchiveMutations": 0,
        },
        "frozenSmoke": {
            "plannedRows": 40,
            "executedRows": 0,
            "started": False,
            "pass": False,
            "blockedConfigurationIds": bindings["frozenSmokeConfigurationIdsBlocked"],
        },
        "substantiveSearchStarted": False,
        "trainingOutcomeEvaluations": 0,
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
        "recommendedNextAction": (
            "Preregister and authorize a bounded S08B adapter remediation that permits "
            "one-member native-carrier strata without weakening Chimera carrier authority; "
            "requalify the exact 56 affected frozen configurations and all 40 smoke rows, "
            "then return for new S08 execution approval."
        ),
    }
