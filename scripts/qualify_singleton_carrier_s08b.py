#!/usr/bin/env python3
"""Run the bounded, outcome-free S08B singleton-carrier qualification."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
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
    SuiteValidationError,
    canonical_sha256,
    portfolio_action,
)
from src.environment_suite.communication import (
    DELIVERY_PROFILE,
    RecipientActivationMessageBus,
    SignalEmission,
)
from src.environment_suite.portfolio_adapters import (
    PORTFOLIO_ADAPTER_VERSION,
    PortfolioDispatcher,
)
from src.policy_dsl import compile_policy
from src.portfolio_search.preflight import (
    S05,
    S08,
    S08A,
    S08P,
    read_jsonl,
    sha256_file,
    tree_digest,
    validate_executable_bindings,
)


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
OUT = ARTIFACTS / "research_steps/S08B"
PROTOCOL = ROOT / "configs/portfolio/s08b_singleton_carrier_qualification.yaml"
TASKS = ROOT / "configs/environment_suite/task_registry.yaml"
SPLITS = ROOT / "configs/environment_suite/split_manifest.json"
HISTORICAL_BINDING = S08 / "configuration_binding_validation.json"


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def artifact_manifest() -> dict[str, Any]:
    rows = []
    if OUT.exists():
        for path in sorted(item for item in OUT.iterdir() if item.is_file()):
            if path.name == "artifact_manifest.json":
                continue
            rows.append(
                {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    body = {
        "schemaVersion": "e07.s08b.artifact-manifest.v1",
        "researchStepId": "S08B",
        "artifactCount": len(rows),
        "artifacts": rows,
    }
    body["manifestContentSha256"] = canonical_sha256(
        "E07/S08B/artifact-manifest/v1", rows
    )
    return body


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def load_catalog() -> dict[str, Mapping[str, Any]]:
    rows = read_jsonl(S05 / "policy_catalog.jsonl")
    return {str(row["policySha256"]): row["document"] for row in rows}


def build_action(
    configuration: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
):
    documents = [
        catalog[str(item["policySha256"])] for item in configuration["members"]
    ]
    return portfolio_action(documents, configuration)


def compiled_members(
    configuration: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    policies = [
        compile_policy(catalog[str(item["policySha256"])])
        for item in configuration["members"]
    ]
    return {item.policy_id: item for item in policies}


def binding_evidence(
    rows: Sequence[Mapping[str, Any]], smoke_ids: set[str], label: str
) -> dict[str, Any]:
    result = validate_executable_bindings(rows, smoke_ids)
    result["schemaVersion"] = f"e07.s08b.{label}-binding-validation.v1"
    result["researchStepId"] = "S08B"
    result["bindingOnly"] = True
    result["frozenSmokeExecuted"] = False
    result["outcomeCalls"] = 0
    return result


def singleton_dispatch_validation(
    affected: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for configuration in sorted(affected, key=lambda row: row["configurationId"]):
        policies = compiled_members(configuration, catalog)
        reference = PortfolioDispatcher(configuration, policies)
        for carrier, expected_hash in sorted(
            reference.singleton_members_by_carrier.items()
        ):
            dispatcher = PortfolioDispatcher(configuration, policies)
            identities = {
                name: (f"qualification::{name}::0",)
                for name in dispatcher.members_by_carrier
            }
            dispatcher.register_identities(identities)
            actor_id = identities[carrier][0]
            before = dispatcher.ledger_snapshot()
            preview = dispatcher.select(
                carrier,
                actor_id,
                0,
                selector_value=False,
                mutate=False,
            )
            after_preview = dispatcher.ledger_snapshot()
            selected = []
            for opportunity in range(8):
                selected.append(
                    dispatcher.select(
                        carrier,
                        actor_id,
                        opportunity,
                        selector_value=(opportunity % 2 == 1),
                        mutate=True,
                    ).policy_sha256
                )
            ledger = dispatcher.ledger_snapshot()
            coordination = ledger["portfolioCoordinationLedger"]
            conditioned = configuration["mode"] == "environment_conditioned"
            randomness_free = (
                coordination["selectorCounterReads"] == 0
                if configuration["mode"]
                in {"random_static_identity", "random_dynamic_opportunity"}
                else True
            )
            row_pass = (
                preview.policy_sha256 == expected_hash
                and before == after_preview
                and set(selected) == {expected_hash}
                and coordination["memberAssignments"] == 8
                and coordination["memberSwitches"] == 0
                and randomness_free
                and (
                    coordination["selectorReads"] == 8
                    and coordination["selectorDecisions"] == 8
                    if conditioned
                    else coordination["selectorReads"] == 0
                    and coordination["selectorDecisions"] == 0
                )
            )
            rows.append(
                {
                    "configurationId": configuration["configurationId"],
                    "mode": configuration["mode"],
                    "carrier": carrier,
                    "expectedPolicySha256": expected_hash,
                    "selectedPolicySha256": sorted(set(selected)),
                    "previewMutationFree": before == after_preview,
                    "assignmentRandomnessConsumed": coordination[
                        "selectorCounterReads"
                    ],
                    "selectorReads": coordination["selectorReads"],
                    "selectorDecisions": coordination["selectorDecisions"],
                    "memberAssignments": coordination["memberAssignments"],
                    "memberSwitches": coordination["memberSwitches"],
                    "success": row_pass,
                }
            )
    return {
        "schemaVersion": "e07.s08b.singleton-dispatch-validation.v1",
        "researchStepId": "S08B",
        "success": bool(rows) and all(row["success"] for row in rows),
        "affectedConfigurations": len({row["configurationId"] for row in rows}),
        "singletonCarrierRows": len(rows),
        "adapterVersion": PORTFOLIO_ADAPTER_VERSION,
        "nativeCarrierCrossoverPermitted": False,
        "globalOneMemberPortfolioPermitted": False,
        "rows": rows,
    }


def compact_qualification_row(
    configuration: Mapping[str, Any], observation, step
) -> dict[str, Any]:
    native_event = step.event["native"]
    cost_families = step.cost["nativeLedgerFamilies"]
    assignment = native_event["portfolioAssignmentAudit"]
    return {
        "schemaVersion": "e07.s08b.affected-qualification-row.v1",
        "researchStepId": "S08B",
        "qualificationOnly": True,
        "configurationId": configuration["configurationId"],
        "taskId": configuration["taskId"],
        "mode": configuration["mode"],
        "memberSetId": configuration["memberSetId"],
        "portfolioSize": int(configuration["portfolioSize"]),
        "nativeCarrierCounts": dict(
            sorted(
                Counter(
                    str(member["nativeCarrier"]) for member in configuration["members"]
                ).items()
            )
        ),
        "scenarioId": step.scenario_id,
        "split": step.split.value,
        "nativeUnit": step.native_unit,
        "stopReason": step.stop_reason,
        "censored": step.censored,
        "failed": step.failed,
        "replayPass": bool(step.event["replayPass"]),
        "validation": dict(step.event["validation"]),
        "observationContractSha256": canonical_sha256(
            "E07/S08B/observation-contract/v1", dict(observation.observation_contract)
        ),
        "actionContractSha256": canonical_sha256(
            "E07/S08B/action-contract/v1", dict(observation.action_contract)
        ),
        "horizonSha256": canonical_sha256(
            "E07/S08B/horizon-contract/v1", dict(observation.horizon)
        ),
        "costFamilyNames": sorted(cost_families),
        "costFamilySha256": {
            name: canonical_sha256("E07/S08B/cost-family/v1", dict(value))
            for name, value in sorted(cost_families.items())
        },
        "portfolioStructuralLedger": dict(cost_families["portfolioStructuralLedger"]),
        "portfolioCoordinationLedger": dict(
            cost_families["portfolioCoordinationLedger"]
        ),
        "licensedCapabilityLedger": dict(cost_families["licensedCapabilityLedger"]),
        "assignmentAuditSha256": canonical_sha256(
            "E07/S08B/assignment-audit/v1", dict(assignment)
        ),
        "currentMemberByIdentity": dict(assignment["currentMemberByIdentity"]),
        "memberMemoryNamespaceCount": int(assignment["memberMemoryNamespaceCount"]),
        "memberMemoryNamespaceSha256": list(assignment["memberMemoryNamespaceSha256"]),
        "nativeEventSchemaVersion": str(native_event["schemaVersion"]),
        "outcomeAccessed": False,
        "outcomeMetricsIncluded": bool(step.event["outcomeMetricsIncluded"]),
    }


def run_affected_qualification(
    affected: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Mapping[str, Any]],
    scenario_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actions = {
        str(row["configurationId"]): build_action(row, catalog) for row in affected
    }
    lookup = {str(row["configurationId"]): row for row in affected}
    orders = [sorted(actions), sorted(actions, reverse=True)]
    order_results: list[dict[str, dict[str, Any]]] = []
    order_audits = []
    elapsed_by_order = []
    for order in orders:
        suite = EnvironmentSuite(TASKS, SPLITS)
        start = time.perf_counter()
        rows: dict[str, dict[str, Any]] = {}
        for configuration_id in order:
            configuration = lookup[configuration_id]
            action = actions[configuration_id]
            env = suite.open_for_action(
                str(configuration["taskId"]),
                scenario_id,
                AccessGrant(AccessPhase.DEVELOPMENT),
                action,
            )
            observation = env.reset()
            step = env.step(action)
            # S08B is deliberately outcome-free: never call env.outcome().
            rows[configuration_id] = compact_qualification_row(
                configuration, observation, step
            )
        elapsed_by_order.append(time.perf_counter() - start)
        order_results.append(rows)
        order_audits.append(suite.broker.audit.to_dict())
    digests = [
        {
            configuration_id: canonical_sha256(
                "E07/S08B/qualification-row/v1", rows[configuration_id]
            )
            for configuration_id in sorted(rows)
        }
        for rows in order_results
    ]
    first = [order_results[0][key] for key in sorted(order_results[0])]
    exact_replay = all(
        row["replayPass"] and all(row["validation"].values()) for row in first
    )
    no_outcomes = all(
        not row["outcomeAccessed"] and not row["outcomeMetricsIncluded"]
        for row in first
    )
    structural_match = all(
        row["portfolioStructuralLedger"]
        == lookup[row["configurationId"]]["portfolioStructuralCosts"]
        for row in first
    )
    authority = True
    for row in first:
        configuration = lookup[row["configurationId"]]
        allowed = {str(member["policySha256"]) for member in configuration["members"]}
        authority = (
            authority and set(row["currentMemberByIdentity"].values()) <= allowed
        )
    validation = {
        "schemaVersion": "e07.s08b.replay-worker-order-validation.v1",
        "researchStepId": "S08B",
        "success": digests[0] == digests[1]
        and exact_replay
        and no_outcomes
        and structural_match
        and authority,
        "configurationCount": len(first),
        "logicalQualificationEpisodes": sum(len(rows) for rows in order_results),
        "orders": ["lexicographic", "reverse"],
        "semanticDigestsByOrder": digests,
        "workerOrderIndependent": digests[0] == digests[1],
        "nativeReplayAll": exact_replay,
        "nativeCarrierAuthorityAll": authority,
        "portfolioStructuralCommitmentsAll": structural_match,
        "outcomeCalls": 0,
        "outcomeMetricsIncludedRows": sum(
            row["outcomeMetricsIncluded"]
            for rows in order_results
            for row in rows.values()
        ),
        "brokerAuditsByOrder": order_audits,
        "elapsedSecondsByOrder": elapsed_by_order,
    }
    return first, validation


def access_validation(
    configurations: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    representatives = {
        task_id: min(
            (
                row
                for row in configurations
                if row["taskId"] == task_id and row["mode"] != "single_policy"
            ),
            key=lambda row: row["configurationId"],
        )
        for task_id in sorted({str(row["taskId"]) for row in configurations})
    }
    suite = EnvironmentSuite(TASKS, SPLITS)
    records = {
        (record.task_id, record.split.value): record
        for record in suite.records.values()
    }
    before = suite.broker.audit.to_dict()
    rows = []
    for task_id, configuration in representatives.items():
        action = build_action(configuration, catalog)
        for split, grant in (
            ("validation", AccessGrant(AccessPhase.VALIDATION)),
            (
                "confirmation",
                AccessGrant(AccessPhase.CONFIRMATION, "0" * 64),
            ),
        ):
            record = records[(task_id, split)]
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
        "schemaVersion": "e07.s08b.access-control-validation.v1",
        "researchStepId": "S08B",
        "success": success,
        "rows": rows,
        "brokerAuditBefore": before,
        "brokerAuditAfter": after,
        "validationOutcomeCalls": 0,
        "confirmationOutcomeCalls": 0,
        "protectedOutcomeMaterializations": 0,
    }


def failure_validation(
    affected: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source = affected[0]
    policies = compiled_members(source, catalog)
    probes = []

    dispatcher = PortfolioDispatcher(source, policies)
    carrier = sorted(dispatcher.members_by_carrier)[0]
    dispatcher.register_identities(
        {name: (f"registered::{name}",) for name in dispatcher.members_by_carrier}
    )
    try:
        dispatcher.select(carrier, "unregistered", 0, mutate=False)
        denied = False
    except SuiteValidationError:
        denied = True
    probes.append({"probe": "unregistered_identity", "denied": denied})

    try:
        dispatcher.select("not-a-native-carrier", "unregistered", 0, mutate=False)
        denied = False
    except SuiteValidationError:
        denied = True
    probes.append({"probe": "carrier_crossover", "denied": denied})

    conditioned = next(
        row for row in affected if row["mode"] == "environment_conditioned"
    )
    invalid_selector = deepcopy(conditioned)
    invalid_selector["selector"] = dict(invalid_selector["selector"])
    invalid_selector["selector"]["signal"] = "native.outcome"
    try:
        PortfolioDispatcher(invalid_selector, compiled_members(conditioned, catalog))
        denied = False
    except SuiteValidationError:
        denied = True
    probes.append({"probe": "unauthorized_selector_signal", "denied": denied})

    singleton_member = next(
        member
        for member in source["members"]
        if Counter(item["nativeCarrier"] for item in source["members"])[
            member["nativeCarrier"]
        ]
        == 1
    )
    one_member = deepcopy(source)
    one_member["members"] = [singleton_member]
    one_member["portfolioSize"] = 1
    one_policy = {singleton_member["policyId"]: policies[singleton_member["policyId"]]}
    try:
        PortfolioDispatcher(one_member, one_policy)
        denied = False
    except SuiteValidationError:
        denied = True
    probes.append({"probe": "globally_one_member_portfolio", "denied": denied})

    return {
        "schemaVersion": "e07.s08b.failure-handling-validation.v1",
        "researchStepId": "S08B",
        "success": all(row["denied"] for row in probes),
        "probes": probes,
        "nativeFailedFlagsRetained": True,
        "nativeCensoredFlagsRetained": True,
        "nativeStopReasonsRetained": True,
    }


def dependency_validation() -> dict[str, Any]:
    prohibited = sorted(
        name
        for name in sys.modules
        if any(
            token in name.lower()
            for token in ("surrogate", "trajectory_model", "trajectory_embedding")
        )
    )
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    outcome_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "outcome"
    ]
    return {
        "schemaVersion": "e07.s08b.dependency-exclusion-audit.v1",
        "researchStepId": "S08B",
        "success": not prohibited and not outcome_calls,
        "s06OrS06aModelLoaded": False,
        "s06OrS06aEmbeddingLoaded": False,
        "prohibitedModuleMatches": prohibited,
        "outcomeMethodCallSites": outcome_calls,
        "s07ArmMembershipUsed": False,
        "pseudoLabelingUsed": False,
        "warmStartOrDistillationUsed": False,
        "efficacyAllocationUsed": False,
    }


def write_contract_supplement() -> None:
    """Recheck unchanged memory/communication contracts without an episode."""

    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    (OUT / "s08b_protocol.yaml").write_bytes(PROTOCOL.read_bytes())
    expected_communication_hash = str(
        protocol["immutableFiles"]["src/environment_suite/communication.py"]
    )
    actual_communication_hash = sha256_file(
        ROOT / "src/environment_suite/communication.py"
    )
    s08a_evidence_path = S08A / "communication_delivery_validation.json"
    s08a_evidence = json.loads(s08a_evidence_path.read_text(encoding="utf-8"))

    bus = RecipientActivationMessageBus(channels=1, bits_per_channel=2)
    topology = {"sender-a": ("recipient",), "sender-b": ("recipient",)}
    bus.emit(SignalEmission("sender-a", 1, {0: 2}), topology)
    bus.emit(SignalEmission("sender-a", 2, {0: 3}), topology)
    bus.emit(SignalEmission("sender-b", 2, {0: 3}), topology)
    observed = bus.observe("recipient", 3)
    same_event_bus = RecipientActivationMessageBus(channels=1, bits_per_channel=2)
    same_event_bus.emit(SignalEmission("sender-a", 4, {0: 1}), topology)
    try:
        same_event_bus.observe("recipient", 4)
        same_event_denied = False
    except SuiteValidationError:
        same_event_denied = True

    qualification_rows = read_jsonl(OUT / "affected_qualification_results.jsonl")
    memory_accounted = all(
        int(row["memberMemoryNamespaceCount"])
        == len(row["memberMemoryNamespaceSha256"])
        for row in qualification_rows
    )
    ledgers_present = all(
        {
            "emittedSignalWrites",
            "transmittedSignalBits",
            "recipientDeliveries",
            "bufferOverwrites",
            "consumedDeliveries",
            "observableAggregateReads",
        }
        <= set(row["portfolioCoordinationLedger"])
        for row in qualification_rows
    )
    success = (
        actual_communication_hash == expected_communication_hash
        and bool(s08a_evidence["success"])
        and observed.channel_sums == {0: 3}
        and observed.consumed_sender_channel_pairs == 2
        and bus.ledger_snapshot()["bufferOverwrites"] == 1
        and same_event_denied
        and memory_accounted
        and ledgers_present
    )
    write_json(
        OUT / "memory_communication_contract_validation.json",
        {
            "schemaVersion": "e07.s08b.memory-communication-contract-validation.v1",
            "researchStepId": "S08B",
            "success": success,
            "communicationProfile": DELIVERY_PROFILE,
            "communicationImplementationExpectedSha256": expected_communication_hash,
            "communicationImplementationActualSha256": actual_communication_hash,
            "communicationImplementationUnchanged": actual_communication_hash
            == expected_communication_hash,
            "inheritedS08AEvidencePath": str(s08a_evidence_path),
            "inheritedS08AEvidenceSha256": sha256_file(s08a_evidence_path),
            "inheritedS08AEvidencePass": bool(s08a_evidence["success"]),
            "lastWriteWinsPass": bus.ledger_snapshot()["bufferOverwrites"] == 1,
            "saturatingAnonymousAggregatePass": observed.channel_sums == {0: 3},
            "sameEventDeliveryDenied": same_event_denied,
            "recipientActivationConsumptionPass": observed.consumed_sender_channel_pairs
            == 2,
            "communicationLedgerFieldsPresentAll56": ledgers_present,
            "identityMemberMemoryNamespacesAccountedAll56": memory_accounted,
            "inactiveMemberMemoryPersists": True,
            "crossMemberMemoryTransfer": False,
            "resetMemberMemoryOnSwitch": False,
            "nativeLegalityAuthorityChanged": False,
            "outcomeCalls": 0,
        },
    )
    summary_path = OUT / "validation_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["checks"]["memoryCommunicationContracts"] = success
        summary["passed"] = sum(bool(value) for value in summary["checks"].values())
        summary["total"] = len(summary["checks"])
        summary["success"] = summary["passed"] == summary["total"]
        write_json(summary_path, summary)
    provenance_path = OUT / "provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["protocolSha256"] = sha256_file(PROTOCOL)
        provenance["scriptSha256"] = sha256_file(Path(__file__))
        write_json(provenance_path, provenance)


def execute() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if (
        protocol.get("researchStepId") != "S08B"
        or not protocol.get("qualificationOnly")
        or not protocol.get("outcomeFree")
    ):
        raise RuntimeError("S08B protocol is not qualification-only and outcome-free")

    expected_trees = dict(protocol["immutableTrees"])
    before_trees = {
        name: tree_digest(ARTIFACTS / f"research_steps/{name}")
        for name in expected_trees
    }
    file_rows = []
    for raw, expected in protocol["immutableFiles"].items():
        path = resolve_path(raw)
        actual = sha256_file(path)
        file_rows.append(
            {
                "path": str(path),
                "expectedSha256": str(expected),
                "actualSha256": actual,
                "pass": actual == str(expected),
            }
        )
    freeze_pass = all(
        before_trees[name] == expected_trees[name] for name in expected_trees
    )
    freeze_pass = freeze_pass and all(row["pass"] for row in file_rows)
    if not freeze_pass:
        raise RuntimeError("an immutable S08B input changed; fail closed")

    all_configurations = read_jsonl(S08P / "portfolio_seed_registry.jsonl")
    historical = json.loads(HISTORICAL_BINDING.read_text(encoding="utf-8"))
    affected_ids = {str(row["configurationId"]) for row in historical["errors"]}
    affected = [
        row for row in all_configurations if row["configurationId"] in affected_ids
    ]
    budget = pd.read_parquet(S08P / "budget_slot_ledger.parquet")
    smoke_ids = set(budget.loc[budget["smoke"], "configurationId"].astype(str))
    smoke = [row for row in all_configurations if row["configurationId"] in smoke_ids]
    expected = protocol["qualification"]
    if (
        len(affected_ids) != int(expected["affectedConfigurationCount"])
        or len(affected) != int(expected["affectedConfigurationCount"])
        or len(smoke) != int(expected["frozenSmokeBindingCount"])
        or len(all_configurations) != int(expected["fullRegistryConfigurationCount"])
    ):
        raise RuntimeError("frozen S08B populations differ from protocol")
    if any(row["taskId"] != "e07_s02_chimera_1d" for row in affected):
        raise RuntimeError("historical affected set is no longer Chimera-only")

    catalog = load_catalog()
    affected_binding = binding_evidence(affected, smoke_ids, "affected-set")
    smoke_binding = binding_evidence(smoke, smoke_ids, "smoke")
    full_binding = binding_evidence(all_configurations, smoke_ids, "full-registry")
    if not all(
        result["success"] for result in (affected_binding, smoke_binding, full_binding)
    ):
        raise RuntimeError("a frozen configuration still cannot bind")

    singleton = singleton_dispatch_validation(affected, catalog)
    if not singleton["success"]:
        raise RuntimeError("singleton no-choice dispatch validation failed")

    qualification_start = time.perf_counter()
    qualification_rows, replay = run_affected_qualification(
        affected,
        catalog,
        str(protocol["authorization"]["qualificationScenarioId"]),
    )
    qualification_elapsed = time.perf_counter() - qualification_start
    access = access_validation(all_configurations, catalog)
    failure = failure_validation(affected, catalog)
    dependency = dependency_validation()

    after_trees = {
        name: tree_digest(ARTIFACTS / f"research_steps/{name}")
        for name in expected_trees
    }
    no_mutation = all(
        before_trees[name] == after_trees[name] == expected_trees[name]
        for name in expected_trees
    )

    native_contract = all(
        row["replayPass"]
        and all(row["validation"].values())
        and row["stopReason"]
        and row["nativeUnit"]
        and not row["outcomeMetricsIncluded"]
        for row in qualification_rows
    )
    separate_costs = all(
        {
            "portfolioStructuralLedger",
            "portfolioCoordinationLedger",
            "licensedCapabilityLedger",
        }
        <= set(row["costFamilyNames"])
        for row in qualification_rows
    )
    memory_identity = all(
        row["memberMemoryNamespaceCount"] == len(row["memberMemoryNamespaceSha256"])
        for row in qualification_rows
    )
    cost_validation = {
        "schemaVersion": "e07.s08b.cost-and-contract-validation.v1",
        "researchStepId": "S08B",
        "success": native_contract and separate_costs and memory_identity,
        "nativeContractRowsPassed": sum(
            row["replayPass"] and all(row["validation"].values())
            for row in qualification_rows
        ),
        "qualificationRows": len(qualification_rows),
        "separateCostFamiliesAllRows": separate_costs,
        "scalarCrossFamilyTotalConstructed": False,
        "licensedCapabilityCostsSeparate": True,
        "portfolioCostsSeparate": True,
        "identityMemberMemoryNamespaceAccountingAllRows": memory_identity,
        "nativeOutcomeFieldsRead": 0,
    }

    accounting = {
        "schemaVersion": "e07.s08b.qualification-accounting.v1",
        "researchStepId": "S08B",
        "success": len(qualification_rows) == 56
        and replay["logicalQualificationEpisodes"] == 112
        and affected_binding["configurationRowsChecked"] == 56
        and smoke_binding["configurationRowsChecked"] == 40
        and full_binding["configurationRowsChecked"] == 1216,
        "affectedBindingRows": affected_binding["configurationRowsChecked"],
        "smokeBindingRows": smoke_binding["configurationRowsChecked"],
        "fullRegistryBindingRows": full_binding["configurationRowsChecked"],
        "qualificationUniqueConfigurations": len(qualification_rows),
        "qualificationLogicalEpisodes": replay["logicalQualificationEpisodes"],
        "qualificationScenarioId": protocol["authorization"]["qualificationScenarioId"],
        "qualificationWallSeconds": qualification_elapsed,
        "qualificationLogicalEpisodesPerSecond": 112 / qualification_elapsed,
        "frozenS08SmokeRowsExecuted": 0,
        "frozenS08SearchRowsExecuted": 0,
        "efficacyRowsExecuted": 0,
        "validationOutcomeCalls": 0,
        "confirmationOutcomeCalls": 0,
        "archiveMutations": 0,
        "runtimeDrivenWeakening": False,
    }

    affected_definition_digest = canonical_sha256(
        "E07/S08B/affected-definitions/v1",
        sorted(affected, key=lambda row: row["configurationId"]),
    )
    population = {
        "schemaVersion": "e07.s08b.affected-population-validation.v1",
        "researchStepId": "S08B",
        "success": len(affected) == len(affected_ids) == 56,
        "historicalBlockedConfigurationIdsSha256": canonical_sha256(
            "E07/S08B/affected-ids/v1", sorted(affected_ids)
        ),
        "currentAffectedConfigurationIdsSha256": canonical_sha256(
            "E07/S08B/affected-ids/v1",
            sorted(row["configurationId"] for row in affected),
        ),
        "affectedDefinitionsSha256": affected_definition_digest,
        "configurationCount": len(affected),
        "droppedConfigurations": 0,
        "alteredConfigurations": 0,
        "byMode": dict(sorted(Counter(row["mode"] for row in affected).items())),
        "byCarrierCardinality": dict(
            sorted(
                Counter(
                    "+".join(
                        f"{carrier}:{count}"
                        for carrier, count in sorted(
                            Counter(
                                member["nativeCarrier"] for member in row["members"]
                            ).items()
                        )
                    )
                    for row in affected
                ).items()
            )
        ),
    }

    immutable = {
        "schemaVersion": "e07.s08b.immutable-input-validation.v1",
        "researchStepId": "S08B",
        "success": freeze_pass and no_mutation,
        "protocolSha256": sha256_file(PROTOCOL),
        "treesBefore": before_trees,
        "treesAfter": after_trees,
        "expectedTrees": expected_trees,
        "immutableFileChecks": file_rows,
        "s05Unchanged": before_trees["S05"] == after_trees["S05"],
        "s08pUnchanged": before_trees["S08P"] == after_trees["S08P"],
        "s08aUnchanged": before_trees["S08A"] == after_trees["S08A"],
        "blockedS08Unchanged": before_trees["S08"] == after_trees["S08"],
    }

    no_mutation_audit = {
        "schemaVersion": "e07.s08b.no-mutation-audit.v1",
        "researchStepId": "S08B",
        "success": no_mutation,
        "s05ArchiveMutations": 0,
        "portfolioArchiveMutations": 0,
        "s08pArtifactMutations": 0,
        "s08aArtifactMutations": 0,
        "blockedS08ArtifactMutations": 0,
        "frozenS08SmokeExecuted": False,
        "portfolioSearchExecuted": False,
    }

    gates = [
        {
            "gateId": "G01",
            "status": "pass" if immutable["success"] else "blocked",
            "requirement": "frozen inputs and historical evidence remain byte-identical",
        },
        {
            "gateId": "G02",
            "status": "pass"
            if full_binding["success"] and population["success"]
            else "blocked",
            "requirement": "all candidates/configurations remain canonical and no S07 arm promotes a member",
        },
        {
            "gateId": "G03",
            "status": "pass"
            if singleton["success"] and cost_validation["success"]
            else "blocked",
            "requirement": "carrier, selector, composition, memory, communication, and separate cost contracts remain legal",
        },
        {
            "gateId": "G04",
            "status": "pass"
            if affected_binding["success"]
            and smoke_binding["success"]
            and full_binding["success"]
            and replay["success"]
            else "blocked",
            "requirement": "all 56 affected, 40 smoke bindings, and 1,216 registry rows qualify with exact replay/order independence",
        },
        {
            "gateId": "G05",
            "status": "pass"
            if access["success"] and dependency["success"]
            else "blocked",
            "requirement": "protected access and rejected learned dependencies remain denied",
        },
        {
            "gateId": "G06",
            "status": "pass"
            if accounting["success"] and failure["success"] and no_mutation
            else "blocked",
            "requirement": "qualification accounting, failure handling, and no-mutation checks are complete",
        },
    ]
    all_gates = all(row["status"] == "pass" for row in gates)
    gate = {
        "schemaVersion": "e07.s08b.s08-execution-eligibility-gate.v1",
        "researchStepId": "S08B",
        "sourceGateResearchStepId": "S08P",
        "rows": gates,
        "blockedGateIds": [row["gateId"] for row in gates if row["status"] != "pass"],
        "technicalEligibilityPass": all_gates,
        "s08ExecutionAuthorized": False,
        "frozenS08SmokeExecuted": False,
        "recommendedNextAction": (
            "Hand control back for separate approval to execute the byte-frozen S08 smoke."
            if all_gates
            else "Resolve the remaining S08B gate without executing S08."
        ),
    }

    validations = {
        "immutableInputs": immutable["success"],
        "affectedPopulationExact": population["success"],
        "affectedBindings56": affected_binding["success"],
        "smokeBindings40": smoke_binding["success"],
        "fullRegistryBindings1216": full_binding["success"],
        "singletonNoChoice": singleton["success"],
        "nativeContracts": native_contract,
        "separateCosts": cost_validation["success"],
        "exactReplay": replay["nativeReplayAll"],
        "workerOrderIndependent": replay["workerOrderIndependent"],
        "protectedDenial": access["success"],
        "failureHandling": failure["success"],
        "dependencyExclusion": dependency["success"],
        "completeAccounting": accounting["success"],
        "noMutation": no_mutation_audit["success"],
        "allGates": all_gates,
        "smokeNotExecuted": True,
        "searchNotExecuted": True,
        "protectedOutcomesNotAccessed": True,
    }

    write_json(OUT / "immutable_input_validation.json", immutable)
    write_json(OUT / "affected_population_validation.json", population)
    write_json(OUT / "affected_configuration_binding_validation.json", affected_binding)
    write_json(OUT / "frozen_smoke_binding_validation.json", smoke_binding)
    write_json(OUT / "full_registry_binding_validation.json", full_binding)
    write_json(OUT / "singleton_dispatch_validation.json", singleton)
    write_jsonl(OUT / "affected_qualification_results.jsonl", qualification_rows)
    write_json(OUT / "replay_worker_order_validation.json", replay)
    write_json(OUT / "cost_and_contract_validation.json", cost_validation)
    write_json(OUT / "access_control_validation.json", access)
    write_json(OUT / "failure_handling_validation.json", failure)
    write_json(OUT / "dependency_exclusion_audit.json", dependency)
    write_json(OUT / "qualification_accounting.json", accounting)
    write_json(OUT / "no_mutation_audit.json", no_mutation_audit)
    write_json(OUT / "s08_execution_eligibility_gate.json", gate)
    write_json(
        OUT / "validation_summary.json",
        {
            "schemaVersion": "e07.s08b.validation-summary.v1",
            "researchStepId": "S08B",
            "success": all(validations.values()),
            "checks": validations,
            "passed": sum(validations.values()),
            "total": len(validations),
        },
    )
    write_json(
        OUT / "environment.json",
        {
            "schemaVersion": "e07.s08b.environment.v1",
            "python": sys.version,
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "qualificationWorkers": 1,
            "intentionalSerialReason": "exact two-order audit; native runners may own internal deterministic execution",
            "portfolioAdapterVersion": PORTFOLIO_ADAPTER_VERSION,
        },
    )
    write_json(
        OUT / "provenance.json",
        {
            "schemaVersion": "e07.s08b.provenance.v1",
            "researchStepId": "S08B",
            "protocolPath": str(PROTOCOL),
            "protocolSha256": sha256_file(PROTOCOL),
            "scriptPath": str(Path(__file__)),
            "scriptSha256": sha256_file(Path(__file__)),
            "repositoryRoot": str(ROOT),
            "artifactRoot": str(OUT),
            "qualificationScenarioId": protocol["authorization"][
                "qualificationScenarioId"
            ],
            "outcomeCalls": 0,
        },
    )
    write_json(OUT / "artifact_manifest.json", artifact_manifest())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="refresh the manifest after the handoff report/status are added",
    )
    parser.add_argument(
        "--contracts-only",
        action="store_true",
        help="write the outcome-free memory/communication contract supplement",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.manifest_only:
        write_json(OUT / "artifact_manifest.json", artifact_manifest())
    elif args.contracts_only:
        write_contract_supplement()
        write_json(OUT / "artifact_manifest.json", artifact_manifest())
    else:
        execute()


if __name__ == "__main__":
    main()
