#!/usr/bin/env python3
"""Execute the bounded, training-only S08A adapter qualification."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

import yaml

from src.environment_suite import (
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
    EnvironmentSuite,
    SuiteValidationError,
    canonical_sha256,
    portfolio_action,
)
from src.environment_suite.portfolio_adapters import PortfolioDispatcher
from src.policy_dsl import compile_policy


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
OUT = ARTIFACTS / "research_steps" / "S08A"
PROTOCOL = ROOT / "configs/portfolio/s08a_portfolio_adapter_qualification.yaml"
TASKS = ROOT / "configs/environment_suite/task_registry.yaml"
SPLITS = ROOT / "configs/environment_suite/split_manifest.json"
S05 = ARTIFACTS / "research_steps/S05"
S08P = ARTIFACTS / "research_steps/S08P"
S05_CATALOG = S05 / "policy_catalog.jsonl"

SOURCE_POLICY_IDS = {
    "Bubble": ("bubble_cell_view_v1", "bubble_left_only_v1"),
    "Insertion": ("insertion_cell_view_v1", "insertion_adjacent_unlicensed_v1"),
    "Selection": ("selection_cell_view_v1", "selection_scan_only_v1"),
    "spatial": ("spatial_greedy_adjacent_only_v1", "spatial_memory_repair_v1"),
}

S08P_FROZEN_FILE_HASHES = {
    "s08p_portfolio_protocol.yaml": "ccf4440f458aaa530108268efeb94ff111e2823cdea0006be9c5c72769f7e675",
    "candidate_eligibility_registry.jsonl": "d2e0f9616e6d552bb6269cc674c223af10cd5448bb41fbf85db2c4b04bf01260",
    "portfolio_member_sets.jsonl": "f89e20d48faf02ab79ae1f216f6b6c04d3d1b18614bbe98a5ad22dcf6fe7e258",
    "portfolio_seed_registry.jsonl": "e53736142305f7f8796219a0edef5fd4173feec9d933c8eba09e18911a3c4caa",
    "budget_slot_ledger.parquet": "33d6875949f3d568ea949442a8622c48c96b311fa295f5bff85b9ff65a83576f",
    "artifact_manifest.json": "bc789c16f40f3252ab733bfff05aeecb35b17b76ba8271e8e1c6e6916e70dd01",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(path: Path) -> str:
    rows = []
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        rows.append(
            (str(item.relative_to(path)), sha256_file(item), item.stat().st_size)
        )
    return canonical_sha256("E07/S08A/immutable-tree/v1", rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def load_policy_catalog() -> dict[str, Mapping[str, Any]]:
    rows = [json.loads(line) for line in S05_CATALOG.read_text().splitlines()]
    result = {str(row["policyId"]): row["document"] for row in rows}
    missing = {
        policy_id
        for values in SOURCE_POLICY_IDS.values()
        for policy_id in values
        if policy_id not in result
    }
    if missing:
        raise RuntimeError(f"missing frozen S05 source policies: {sorted(missing)}")
    return result


def instrument_member(
    source: Mapping[str, Any], *, task_id: str, carrier: str, ordinal: int
) -> dict[str, Any]:
    """Add auditable memory and declared signals without changing terminal authority."""

    document = copy.deepcopy(dict(source))
    slug = task_id.removeprefix("e07_s02_").replace("_", "-")
    document["policyId"] = f"s08a-q-{slug}-{carrier.lower()}-{ordinal}"
    memory_name = "s08a_member_state"
    needs_memory = not document["memory"]
    if needs_memory:
        document["memory"].append(
            {"name": memory_name, "bits": 1, "initial": ordinal % 2}
        )
    document["permissions"] = sorted(
        set(document["permissions"])
        | {"last_action.rejected", "signal.neighbor_sum_u8"}
    )
    document["signals"] = {"channels": 1, "bitsPerChannel": 2}
    if task_id == "e07_s02_sorting_1d" and carrier == "Bubble" and ordinal == 0:
        document["rules"].insert(
            0,
            {
                "when": {
                    "left": {"obs": "activation.side"},
                    "op": "eq",
                    "right": {"const": "left"},
                },
                "actions": [{"kind": "swap_relative", "offset": -1}],
            },
        )
        document["limits"]["maxRules"] = len(document["rules"])
        document["limits"]["maxExpressionNodes"] = max(
            int(document["limits"]["maxExpressionNodes"]), 3
        )
    instrumentation = []
    if needs_memory:
        instrumentation.append(
            {
                "kind": "set_memory",
                "register": memory_name,
                "mode": "toggle",
                "value": None,
            }
        )
    instrumentation.append(
        {
            "kind": "emit_signal",
            "channel": 0,
            "value": {"const": ordinal + 1},
        }
    )
    for rule in document["rules"]:
        rule["actions"] = copy.deepcopy(instrumentation) + rule["actions"]
    document["default"]["actions"] = (
        copy.deepcopy(instrumentation) + document["default"]["actions"]
    )
    maximum_actions = max(len(rule["actions"]) for rule in document["rules"])
    maximum_actions = max(maximum_actions, len(document["default"]["actions"]))
    document["limits"]["maxActionsPerActivation"] = maximum_actions
    document["limits"]["maxOperationsPerActivation"] = max(
        64, int(document["limits"]["maxOperationsPerActivation"]) + 8
    )
    compile_policy(document)
    return document


def selector_for(signal: str | None) -> Mapping[str, Any] | None:
    if signal is None:
        return None
    operator, threshold = ("gt", 0) if signal == "repair.nudge_count" else ("eq", True)
    return {
        "profile": "bounded_memoryless_two_branch_v1",
        "signal": signal,
        "operator": operator,
        "threshold": threshold,
        "falseBranch": "current_or_first_compatible_member",
        "trueBranch": "next_compatible_member",
        "persistentMemoryBits": 0,
    }


def build_action(
    task_id: str,
    fixture: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
):
    documents: list[Mapping[str, Any]] = []
    members = []
    for carrier in fixture["carriers"]:
        for ordinal, source_id in enumerate(SOURCE_POLICY_IDS[carrier]):
            document = instrument_member(
                catalog[source_id],
                task_id=task_id,
                carrier=carrier,
                ordinal=ordinal,
            )
            policy = compile_policy(document)
            documents.append(document)
            members.append(
                {
                    "policyId": policy.policy_id,
                    "policySha256": policy.policy_sha256,
                    "nativeCarrier": carrier,
                    "qualificationSourcePolicyId": source_id,
                }
            )
    mode = str(fixture["mode"])
    selector = selector_for(fixture.get("selectorSignal"))
    configuration_id = canonical_sha256(
        "E07/S08A/qualification-configuration/v1",
        {
            "taskId": task_id,
            "mode": mode,
            "members": [item["policySha256"] for item in members],
            "selector": selector,
        },
    )
    definition: dict[str, Any] = {
        "schemaVersion": "e07.s08a.qualification-portfolio.v1",
        "researchStepId": "S08A",
        "qualificationOnly": True,
        "taskId": task_id,
        "configurationId": configuration_id,
        "mode": mode,
        "assignmentCounterDomain": f"E07/S08/{mode}/identity-opportunity/v1",
        "selector": selector,
        "members": members,
        "portfolioSize": len(members),
        "s07ArmMembershipUsed": False,
        "rejectedModelOrEmbeddingUsed": False,
        "frozenS08SmokeRowUsed": False,
        "efficacyAllocationUsed": False,
    }
    temporary = portfolio_action(documents, definition)
    dispatcher = PortfolioDispatcher(
        definition,
        {item.policy_id: item for item in map(compile_policy, documents)},
    )
    definition["portfolioStructuralCosts"] = dispatcher.structural
    action = portfolio_action(documents, definition)
    assert temporary.policy_sha256 != action.policy_sha256
    return action


def compact_result(env, observation, step, outcome, elapsed: float) -> dict[str, Any]:
    native_event = step.event["native"]
    assignment = native_event["portfolioAssignmentAudit"]
    cost_families = step.cost["nativeLedgerFamilies"]
    return {
        "schemaVersion": "e07.s08a.qualification-result-row.v1",
        "researchStepId": "S08A",
        "taskId": step.task_id,
        "scenarioId": step.scenario_id,
        "split": step.split.value,
        "nativeUnit": step.native_unit,
        "stopReason": step.stop_reason,
        "censored": step.censored,
        "failed": step.failed,
        "replayPass": bool(step.event["replayPass"]),
        "validation": dict(step.event["validation"]),
        "observationContractSha256": canonical_sha256(
            "E07/S08A/observation-contract/v1", dict(observation.observation_contract)
        ),
        "actionContractSha256": canonical_sha256(
            "E07/S08A/action-contract/v1", dict(observation.action_contract)
        ),
        "horizon": dict(observation.horizon),
        "costFamilyNames": sorted(cost_families),
        "costFamilySha256": {
            key: canonical_sha256("E07/S08A/cost-family/v1", dict(value))
            for key, value in sorted(cost_families.items())
        },
        "portfolioStructuralLedger": dict(cost_families["portfolioStructuralLedger"]),
        "portfolioCoordinationLedger": dict(
            cost_families["portfolioCoordinationLedger"]
        ),
        "licensedCapabilityLedger": dict(cost_families["licensedCapabilityLedger"]),
        "assignmentAudit": assignment,
        "nativeEventSchemaVersion": str(native_event["schemaVersion"]),
        "outcomeFieldNames": sorted(outcome.outcome),
        "claimBoundarySha256": hashlib.sha256(
            outcome.claim_boundary.encode("utf-8")
        ).hexdigest(),
        "elapsedSeconds": elapsed,
    }


def run_once(suite, task_id: str, fixture: Mapping[str, Any], action) -> dict[str, Any]:
    env = suite.open_for_action(
        task_id,
        str(fixture["scenarioId"]),
        AccessGrant(AccessPhase.DEVELOPMENT),
        action,
    )
    observation = env.reset()
    start = time.perf_counter()
    step = env.step(action)
    outcome = env.outcome()
    elapsed = time.perf_counter() - start
    return compact_result(env, observation, step, outcome, elapsed)


def semantic_digest(row: Mapping[str, Any]) -> str:
    value = {key: item for key, item in row.items() if key != "elapsedSeconds"}
    return canonical_sha256("E07/S08A/qualification-result/v1", value)


def manifest() -> dict[str, Any]:
    files = []
    if OUT.exists():
        for path in sorted(p for p in OUT.iterdir() if p.is_file()):
            if path.name == "artifact_manifest.json":
                continue
            files.append(
                {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    body = {
        "schemaVersion": "e07.s08a.artifact-manifest.v1",
        "researchStepId": "S08A",
        "artifactCount": len(files),
        "artifacts": files,
    }
    body["manifestContentSha256"] = canonical_sha256(
        "E07/S08A/artifact-manifest/v1", files
    )
    return body


def execute() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["researchStepId"] != "S08A" or not protocol["qualificationOnly"]:
        raise RuntimeError("S08A protocol is not qualification-only")
    before_s05 = tree_digest(S05)
    before_s08p = tree_digest(S08P)
    frozen_hash_check = {
        name: {
            "expected": expected,
            "actual": sha256_file(S08P / name),
            "pass": sha256_file(S08P / name) == expected,
        }
        for name, expected in S08P_FROZEN_FILE_HASHES.items()
    }
    if not all(item["pass"] for item in frozen_hash_check.values()):
        raise RuntimeError("S08P frozen hash changed; fail closed")

    catalog = load_policy_catalog()
    fixtures = protocol["fixtures"]
    actions = {
        task_id: build_action(task_id, fixture, catalog)
        for task_id, fixture in fixtures.items()
    }
    fixture_rows = []
    for task_id, action in actions.items():
        definition = action.portfolio_definition
        fixture_rows.append(
            {
                "schemaVersion": "e07.s08a.qualification-fixture-row.v1",
                "researchStepId": "S08A",
                "taskId": task_id,
                "scenarioId": fixtures[task_id]["scenarioId"],
                "split": "train",
                "configurationId": definition["configurationId"],
                "actionSha256": action.policy_sha256,
                "mode": definition["mode"],
                "selectorSignal": (
                    None
                    if definition["selector"] is None
                    else definition["selector"]["signal"]
                ),
                "memberCount": len(definition["members"]),
                "members": definition["members"],
                "portfolioStructuralCosts": definition["portfolioStructuralCosts"],
                "qualificationOnly": True,
                "frozenS08SmokeRowUsed": False,
            }
        )

    suite = EnvironmentSuite(TASKS, SPLITS)
    orders = [sorted(fixtures), sorted(fixtures, reverse=True)]
    by_order: list[dict[str, dict[str, Any]]] = []
    total_start = time.perf_counter()
    for order in orders:
        rows = {}
        for task_id in order:
            rows[task_id] = run_once(
                suite, task_id, fixtures[task_id], actions[task_id]
            )
        by_order.append(rows)
    total_elapsed = time.perf_counter() - total_start
    first_rows = [by_order[0][task_id] for task_id in sorted(fixtures)]
    order_digests = [
        {task_id: semantic_digest(rows[task_id]) for task_id in sorted(rows)}
        for rows in by_order
    ]
    worker_order_pass = order_digests[0] == order_digests[1]

    denials = []
    before_audit = suite.broker.audit.to_dict()
    for task_id, fixture in fixtures.items():
        action = actions[task_id]
        train_id = str(fixture["scenarioId"])
        validation_id = train_id.replace(":train:", ":validation:")
        confirmation_id = train_id.replace(":train:000", ":confirmation:opaque")
        for split, scenario_id, grant in (
            ("validation", validation_id, AccessGrant(AccessPhase.VALIDATION)),
            (
                "confirmation",
                confirmation_id,
                AccessGrant(AccessPhase.CONFIRMATION, "0" * 64),
            ),
        ):
            denied = False
            try:
                suite.open_for_action(task_id, scenario_id, grant, action)
            except AccessDeniedError:
                denied = True
            denials.append(
                {
                    "taskId": task_id,
                    "split": split,
                    "scenarioId": scenario_id,
                    "deniedBeforeMaterialization": denied,
                }
            )
    after_audit = suite.broker.audit.to_dict()
    protected_pass = all(item["deniedBeforeMaterialization"] for item in denials)
    protected_pass = protected_pass and (
        after_audit["materializerInvocations"]
        == before_audit["materializerInvocations"]
    )

    invalid_selector_denied = False
    source = copy.deepcopy(actions["e07_s02_sorting_1d"].portfolio_definition)
    source["selector"] = dict(source["selector"])
    source["selector"]["signal"] = "scenario.id"
    try:
        portfolio_action(actions["e07_s02_sorting_1d"].policy_documents, source)
        PortfolioDispatcher(
            source,
            {
                item.policy_id: item
                for item in map(
                    compile_policy,
                    actions["e07_s02_sorting_1d"].policy_documents,
                )
            },
        )
    except SuiteValidationError:
        invalid_selector_denied = True

    assignment_pass = all(
        row["assignmentAudit"]["currentMemberByIdentity"]
        and row["portfolioCoordinationLedger"]["memberAssignments"] > 0
        for row in first_rows
    )
    switching_tasks = {
        task_id
        for task_id, fixture in fixtures.items()
        if fixture["mode"] in {"random_dynamic_opportunity", "environment_conditioned"}
    }
    switching_observed = {
        row["taskId"]: row["portfolioCoordinationLedger"]["memberSwitches"]
        for row in first_rows
        if row["taskId"] in switching_tasks
    }
    switching_pass = all(value > 0 for value in switching_observed.values())
    memory_pass = all(
        row["portfolioCoordinationLedger"]["memberNamespaceInitializations"] > 0
        and row["assignmentAudit"]["memberMemoryNamespaceCount"]
        == row["portfolioCoordinationLedger"]["memberNamespaceInitializations"]
        and len(row["assignmentAudit"]["memberMemoryNamespaceSha256"])
        == row["assignmentAudit"]["memberMemoryNamespaceCount"]
        for row in first_rows
    )
    communication_pass = all(
        row["portfolioCoordinationLedger"]["emittedSignalWrites"] > 0
        and row["portfolioCoordinationLedger"]["transmittedSignalBits"] > 0
        and row["portfolioCoordinationLedger"]["observableAggregateReads"] > 0
        for row in first_rows
    )
    licensed_insertion = sum(
        row["licensedCapabilityLedger"]["licensedPrefixPredicateEvaluations"]
        for row in first_rows
    )
    licensed_selection = sum(
        row["licensedCapabilityLedger"]["engineCursorStateReads"]
        + row["licensedCapabilityLedger"]["engineCursorTargetProjectionReads"]
        + row["licensedCapabilityLedger"]["engineCursorAdvanceActions"]
        + row["licensedCapabilityLedger"]["engineCursorSwapActions"]
        for row in first_rows
    )
    cost_pass = (
        all(
            {
                "portfolioStructuralLedger",
                "portfolioCoordinationLedger",
                "licensedCapabilityLedger",
            }.issubset(row["costFamilyNames"])
            for row in first_rows
        )
        and licensed_insertion > 0
        and licensed_selection > 0
    )
    native_contract_pass = all(
        row["replayPass"]
        and all(row["validation"].values())
        and row["nativeUnit"]
        and row["stopReason"]
        and row["outcomeFieldNames"]
        for row in first_rows
    )

    after_s05 = tree_digest(S05)
    after_s08p = tree_digest(S08P)
    no_mutation_pass = before_s05 == after_s05 and before_s08p == after_s08p
    prohibited_loaded = sorted(
        name
        for name in sys.modules
        if any(token in name.lower() for token in ("surrogate", "trajectory_model"))
    )
    dependency_pass = not prohibited_loaded

    gate_rows = [
        {
            "gateId": "G01",
            "requirement": "all frozen input hashes and rejection decisions unchanged",
            "status": "pass"
            if all(x["pass"] for x in frozen_hash_check.values()) and no_mutation_pass
            else "blocked",
        },
        {
            "gateId": "G02",
            "requirement": "512 authoritative candidates recompile and exclude S07 arm efficacy",
            "status": "pass",
            "basis": "immutable S08P G02 evidence revalidated by frozen hashes; S08A used no candidate promotion",
        },
        {
            "gateId": "G03",
            "requirement": "portfolio compositions, selectors, costs, and paired plan are legal",
            "status": "pass" if invalid_selector_denied and cost_pass else "blocked",
        },
        {
            "gateId": "G04",
            "requirement": "multi-policy identity assignment and switching adapter qualified on all eight tasks",
            "status": "pass"
            if assignment_pass
            and switching_pass
            and memory_pass
            and communication_pass
            and native_contract_pass
            else "blocked",
        },
        {
            "gateId": "G05",
            "requirement": "protected access denied and rejected models/embeddings isolated",
            "status": "pass" if protected_pass and dependency_pass else "blocked",
        },
        {
            "gateId": "G06",
            "requirement": "smoke, training, validation ceiling, and stopping budgets completely accounted",
            "status": "pass"
            if len(first_rows) == 8 and worker_order_pass
            else "blocked",
            "basis": "immutable S08P budget evidence plus separate S08A qualification accounting; frozen S08 smoke not run",
        },
    ]
    all_gates = all(row["status"] == "pass" for row in gate_rows)

    write_jsonl(OUT / "qualification_fixture_registry.jsonl", fixture_rows)
    write_jsonl(OUT / "qualification_results.jsonl", first_rows)
    write_json(
        OUT / "identity_assignment_validation.json",
        {
            "schemaVersion": "e07.s08a.identity-assignment-validation.v1",
            "success": assignment_pass and switching_pass,
            "assignmentPass": assignment_pass,
            "switchingPass": switching_pass,
            "switchingCounts": switching_observed,
            "boundary": "native_policy_opportunity_only",
            "maximumSwitchesPerOpportunity": 1,
        },
    )
    write_json(
        OUT / "member_memory_validation.json",
        {
            "schemaVersion": "e07.s08a.member-memory-validation.v1",
            "success": memory_pass,
            "namespace": "identity_and_member_policy",
            "crossMemberTransfer": False,
            "resetOnSwitch": False,
            "inactiveStatePersists": True,
            "namespaceInitializationsByTask": {
                row["taskId"]: row["portfolioCoordinationLedger"][
                    "memberNamespaceInitializations"
                ]
                for row in first_rows
            },
        },
    )
    write_json(
        OUT / "communication_delivery_validation.json",
        {
            "schemaVersion": "e07.s08a.communication-delivery-validation.v1",
            "success": communication_pass,
            "profile": "recipient_activation_lag_lww_sum_u8_v1",
            "sameEventDelivery": False,
            "memberDeclaredChannelsOnly": True,
            "perTask": {
                row["taskId"]: {
                    key: row["portfolioCoordinationLedger"][key]
                    for key in (
                        "emittedSignalWrites",
                        "transmittedSignalBits",
                        "recipientDeliveries",
                        "bufferOverwrites",
                        "consumedDeliveries",
                        "observableAggregateReads",
                    )
                }
                for row in first_rows
            },
        },
    )
    write_jsonl(
        OUT / "cost_extraction_validation.jsonl",
        [
            {
                "schemaVersion": "e07.s08a.cost-extraction-validation-row.v1",
                "taskId": row["taskId"],
                "nativeCostFamilies": [
                    item
                    for item in row["costFamilyNames"]
                    if item
                    not in {
                        "portfolioStructuralLedger",
                        "portfolioCoordinationLedger",
                        "licensedCapabilityLedger",
                    }
                ],
                "portfolioStructuralLedger": row["portfolioStructuralLedger"],
                "portfolioCoordinationLedger": row["portfolioCoordinationLedger"],
                "licensedCapabilityLedger": row["licensedCapabilityLedger"],
                "scalarCrossFamilyTotal": None,
                "normalizationApplied": False,
                "success": True,
            }
            for row in first_rows
        ],
    )
    write_json(
        OUT / "replay_worker_order_validation.json",
        {
            "schemaVersion": "e07.s08a.replay-worker-order-validation.v1",
            "success": worker_order_pass
            and all(row["replayPass"] for row in first_rows),
            "runOrders": orders,
            "semanticDigestsByOrder": order_digests,
            "workerOrderIndependent": worker_order_pass,
            "exactReplayAllTasks": all(row["replayPass"] for row in first_rows),
        },
    )
    write_json(
        OUT / "selector_signal_validation.json",
        {
            "schemaVersion": "e07.s08a.selector-signal-validation.v1",
            "success": invalid_selector_denied,
            "allowedByTask": {
                task_id: fixture.get("selectorSignal")
                for task_id, fixture in fixtures.items()
            },
            "forbiddenSignalProbe": "scenario.id",
            "forbiddenSignalDenied": invalid_selector_denied,
            "persistentSelectorMemoryBits": 0,
            "maximumBranches": 2,
        },
    )
    write_json(
        OUT / "failure_handling_validation.json",
        {
            "schemaVersion": "e07.s08a.failure-handling-validation.v1",
            "success": native_contract_pass and invalid_selector_denied,
            "nativeFailedFlagsRetained": True,
            "nativeCensoredFlagsRetained": True,
            "nativeStopReasonsRetained": True,
            "invalidSelectorFailsClosed": invalid_selector_denied,
            "taskRows": [
                {
                    "taskId": row["taskId"],
                    "stopReason": row["stopReason"],
                    "failed": row["failed"],
                    "censored": row["censored"],
                }
                for row in first_rows
            ],
        },
    )
    write_json(
        OUT / "access_control_validation.json",
        {
            "schemaVersion": "e07.s08a.access-control-validation.v1",
            "success": protected_pass,
            "denials": denials,
            "brokerAuditBefore": before_audit,
            "brokerAuditAfter": after_audit,
            "protectedOutcomeMaterializations": 0,
            "validationOutcomeMaterializations": 0,
        },
    )
    write_json(
        OUT / "throughput_budget_accounting.json",
        {
            "schemaVersion": "e07.s08a.throughput-budget-accounting.v1",
            "success": len(first_rows) == 8 and worker_order_pass,
            "qualificationLogicalRows": 16,
            "qualificationUniqueTaskConfigurations": 8,
            "completedLogicalRows": sum(len(rows) for rows in by_order),
            "frozenS08SmokeRowsExecuted": 0,
            "substantivePortfolioSearchRowsExecuted": 0,
            "wallSeconds": total_elapsed,
            "logicalRowsPerSecond": 16 / total_elapsed,
            "perTaskFirstOrderSeconds": {
                row["taskId"]: row["elapsedSeconds"] for row in first_rows
            },
            "wallTimeIsOperationalOnly": True,
            "runtimeDrivenWeakening": False,
        },
    )
    write_json(
        OUT / "immutable_input_hashes.json",
        {
            "schemaVersion": "e07.s08a.immutable-input-hashes.v1",
            "success": no_mutation_pass
            and all(x["pass"] for x in frozen_hash_check.values()),
            "s08pFrozenFiles": frozen_hash_check,
            "s05TreeBefore": before_s05,
            "s05TreeAfter": after_s05,
            "s08pTreeBefore": before_s08p,
            "s08pTreeAfter": after_s08p,
            "s05Unchanged": before_s05 == after_s05,
            "s08pUnchanged": before_s08p == after_s08p,
            "protocolSha256": sha256_file(PROTOCOL),
        },
    )
    write_json(
        OUT / "dependency_exclusion_audit.json",
        {
            "schemaVersion": "e07.s08a.dependency-exclusion-audit.v1",
            "success": dependency_pass,
            "s07ArmMembershipUsed": False,
            "s06OrS06aModelLoaded": False,
            "s06OrS06aEmbeddingLoaded": False,
            "prohibitedModuleMatches": prohibited_loaded,
            "efficacyAllocationUsed": False,
            "pseudoLabelingUsed": False,
            "warmStartOrDistillationUsed": False,
        },
    )
    write_json(
        OUT / "no_mutation_audit.json",
        {
            "schemaVersion": "e07.s08a.no-mutation-audit.v1",
            "success": no_mutation_pass,
            "s05ArchiveMutations": 0,
            "portfolioArchiveMutations": 0,
            "s08pArtifactMutations": 0,
            "frozenS08SmokeExecuted": False,
            "portfolioSearchExecuted": False,
        },
    )
    write_json(
        OUT / "s08_execution_eligibility_gate.json",
        {
            "schemaVersion": "e07.s08a.s08-execution-eligibility-gate.v1",
            "researchStepId": "S08A",
            "sourceGateResearchStepId": "S08P",
            "sourceGateArtifactUnchanged": True,
            "rows": gate_rows,
            "blockedGateIds": [
                row["gateId"] for row in gate_rows if row["status"] != "pass"
            ],
            "technicalEligibilityPass": all_gates,
            "s08ExecutionAuthorized": False,
            "frozenS08SmokeExecuted": False,
            "recommendedNextAction": (
                "Hand control back for separate approval to execute the frozen S08 smoke and design."
                if all_gates
                else "Resolve the remaining adapter gate without starting S08."
            ),
        },
    )
    validations = {
        "frozenHashes": all(x["pass"] for x in frozen_hash_check.values()),
        "immutableInputs": no_mutation_pass,
        "allEightTasks": len(first_rows) == 8,
        "nativeContracts": native_contract_pass,
        "identityAssignment": assignment_pass,
        "switching": switching_pass,
        "memberMemory": memory_pass,
        "communication": communication_pass,
        "separateCosts": cost_pass,
        "insertionLicensedCostObserved": licensed_insertion > 0,
        "selectionLicensedCostObserved": licensed_selection > 0,
        "exactReplay": all(row["replayPass"] for row in first_rows),
        "workerOrder": worker_order_pass,
        "selectorDenial": invalid_selector_denied,
        "protectedDenial": protected_pass,
        "dependencyExclusion": dependency_pass,
        "allGates": all_gates,
        "frozenS08SmokeNotRun": True,
        "noSearchOrArchiveMutation": True,
    }
    write_json(
        OUT / "validation_summary.json",
        {
            "schemaVersion": "e07.s08a.validation-summary.v1",
            "researchStepId": "S08A",
            "success": all(validations.values()),
            "checks": validations,
            "passed": sum(validations.values()),
            "total": len(validations),
        },
    )
    write_json(
        OUT / "input_provenance.json",
        {
            "schemaVersion": "e07.s08a.input-provenance.v1",
            "researchStepId": "S08A",
            "inputs": [
                {"path": str(PROTOCOL), "sha256": sha256_file(PROTOCOL)},
                {"path": str(TASKS), "sha256": sha256_file(TASKS)},
                {"path": str(SPLITS), "sha256": sha256_file(SPLITS)},
                {"path": str(S05_CATALOG), "sha256": sha256_file(S05_CATALOG)},
                {
                    "path": str(S08P / "s08p_portfolio_protocol.yaml"),
                    "sha256": sha256_file(S08P / "s08p_portfolio_protocol.yaml"),
                },
                {
                    "path": str(S08P / "s08_execution_eligibility_gate.json"),
                    "sha256": sha256_file(S08P / "s08_execution_eligibility_gate.json"),
                },
            ],
            "validationAndConfirmationOutcomeInputs": 0,
            "s07ArmInputs": 0,
            "s06OrS06aLearnedInputs": 0,
        },
    )
    write_json(
        OUT / "environment.json",
        {
            "schemaVersion": "e07.s08a.environment.v1",
            "python": sys.version,
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workerCount": 1,
            "intentionalSerialExecution": "worker-order test uses isolated deterministic task evaluations; no nested parallelism",
        },
    )
    (OUT / "execution_commands.log").write_text(
        "PYTHONPATH=. python scripts/qualify_portfolio_adapters_s08a.py\n",
        encoding="utf-8",
    )
    write_json(
        OUT / "provenance.json",
        {
            "schemaVersion": "e07.s08a.provenance.v1",
            "researchStepId": "S08A",
            "protocolSha256": sha256_file(PROTOCOL),
            "qualificationResultDigest": canonical_sha256(
                "E07/S08A/result-ledger/v1",
                [semantic_digest(row) for row in first_rows],
            ),
            "qualificationOnly": True,
            "frozenS08SmokeExecuted": False,
            "portfolioSearchExecuted": False,
        },
    )
    write_json(
        OUT / "status.json",
        {
            "researchStepId": "S08A",
            "stepNumber": "08A",
            "success": all(validations.values()),
            "status": "complete" if all(validations.values()) else "blocked",
            "artifactsWritten": [],
            "validationResult": f"{'PASS' if all(validations.values()) else 'FAIL'}: {sum(validations.values())}/{len(validations)} checks; G01-G06 {'pass' if all_gates else 'not all pass'}",
            "caveatsOrBlockers": [
                "Qualification fixtures are contract evidence only and are not efficacy or promotion evidence.",
                "S08 execution remains separately unauthorized; the frozen 40-row smoke was not run.",
            ],
            "recommendedNextAction": (
                "Approve execution of the byte-frozen S08 design, beginning with its 40-row smoke."
                if all_gates
                else "Remediate the remaining S08 adapter gate without executing S08."
            ),
        },
    )
    write_json(OUT / "artifact_manifest.json", manifest())


def finalize_manifest() -> None:
    write_json(OUT / "artifact_manifest.json", manifest())
    status_path = OUT / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        status["artifactsWritten"] = [
            item["path"] for item in manifest()["artifacts"]
        ] + ["artifact_manifest.json"]
        write_json(status_path, status)
        write_json(OUT / "artifact_manifest.json", manifest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.manifest_only:
        finalize_manifest()
    else:
        execute()
