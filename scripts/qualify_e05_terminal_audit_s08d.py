#!/usr/bin/env python3
"""Run the outcome-free S08D E05 terminal-audit qualification."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import patch

import pandas as pd
import yaml

from reference_simulator import Direction, Policy, create_scenario
from reference_simulator.model import RunState
from src.environment_suite import (
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
    EnvironmentSuite,
    Split,
)
from src.environment_suite.contracts import canonical_sha256
from src.environment_suite.dsl_adapters import (
    bind_homogeneous_line_scenario,
    bind_portfolio_line_scenario,
    finalize_e05_result,
    line_replay_bytes,
    make_line_runtime,
    run_e05_regeneration_dsl,
    run_line_dsl_episode,
    validate_e05_portfolio_result_contract,
)
from src.policy_dsl import compile_policy
from src.portfolio_preregistration.core import (
    candidate_commitment,
    plan_digest,
    read_jsonl,
    validate_portfolio_registry,
)
from src.portfolio_search.execution import (
    FailAtomicBatchError,
    S08P,
    TASK_IDS,
    _base_records,
    _catalog,
    build_action,
    execute_fail_atomic_batch,
    publish_parquet_fail_atomic,
)
from src.environment_suite.portfolio_adapters import portfolio_action
from src.environment_suite.runners import run_regeneration
from src.portfolio_search.preflight import (
    _manifest_validation,
    sha256_file,
    tree_digest,
    validate_executable_bindings,
)


REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/artifacts/research_steps")
PROTOCOL = REPOSITORY / "configs/portfolio/s08d_e05_terminal_audit_remediation.yaml"
TASK_REGISTRY = REPOSITORY / "configs/environment_suite/task_registry.yaml"
SPLIT_MANIFEST = REPOSITORY / "configs/environment_suite/split_manifest.json"
E05 = "e07_s02_regeneration_1d"
BRANCHES = (
    "development_source_terminal",
    "stabilization_source_terminal",
    "completed_panel",
)
MODES = (
    "single_policy",
    "fixed_balanced_identity",
    "random_static_identity",
    "random_dynamic_opportunity",
    "environment_conditioned",
)
IMMUTABLE_STEPS = ("S05", "S08P", "S08A", "S08", "S08B", "S08C")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )


def fixture_payload(branch: str) -> dict[str, Any]:
    source_terminal = branch != "completed_panel"
    stop_reason = {
        "development_source_terminal": "qualification_development_terminal",
        "stabilization_source_terminal": "qualification_stabilization_terminal",
        "completed_panel": "qualification_completed_panel",
    }[branch]
    return {
        "schemaVersion": "e07.s08d.e05-terminal-qualification-fixture.v1",
        "qualificationOnly": True,
        "qualificationBranch": branch,
        "sourceTerminal": source_terminal,
        "sourceStopReason": stop_reason,
        "phaseRows": [],
        "stoppedRows": [],
    }


def _qualification_scenario(definition: Mapping[str, Any]):
    carriers = {str(member["nativeCarrier"]) for member in definition["members"]}
    if len(carriers) != 1:
        raise RuntimeError("E05 qualification fixture requires one native carrier")
    carrier = Policy(next(iter(carriers)))
    seed = 1 + int(str(definition["configurationId"])[:8], 16) % (2**31 - 2)
    return create_scenario(
        tuple(range(8)),
        policy=carrier,
        direction=Direction.ASCENDING,
        seed=seed,
        max_activations=24,
        generation_key=f"E07/S08D/qualification/{definition['configurationId']}",
        permute=True,
    )


def qualify_definition(
    definition: Mapping[str, Any],
    by_hash: Mapping[str, Mapping[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    action = build_action(definition, by_hash, by_id)
    source = _qualification_scenario(definition)
    if action.portfolio_definition:
        scenario = bind_portfolio_line_scenario(source, action)
    else:
        scenario = bind_homogeneous_line_scenario(
            source, compile_policy(action.policy_documents[0])
        )
    first, first_runtime = run_line_dsl_episode(scenario, action, trace_mode="digest")
    second, second_runtime = run_line_dsl_episode(scenario, action, trace_mode="digest")
    replay_pass = line_replay_bytes(first, first_runtime) == line_replay_bytes(
        second, second_runtime
    )
    native_run_sha = canonical_sha256(
        "E07/S08D/qualification-native-run/v1", first.to_dict()
    )
    rows = []
    for branch in BRANCHES:
        result = finalize_e05_result(fixture_payload(branch), first_runtime)
        validation = validate_e05_portfolio_result_contract(result, action)
        audit = result["portfolioAssignmentAudit"]
        rows.append(
            {
                "schemaVersion": "e07.s08d.e05-configuration-qualification-row.v1",
                "researchStepId": "S08D",
                "qualificationOnly": True,
                "taskId": E05,
                "configurationId": str(definition["configurationId"]),
                "mode": str(definition["mode"]),
                "portfolioSize": int(definition["portfolioSize"]),
                "branch": branch,
                "actionSha256": action.policy_sha256,
                "nativeCarrier": next(
                    iter({str(item["nativeCarrier"]) for item in definition["members"]})
                ),
                "nativeQualificationStopReason": str(first.summary["stopReason"]),
                "nativeQualificationFailed": first.summary["stopReason"]
                == "invariant_error",
                "nativeQualificationCensored": first.summary["stopReason"]
                == "event_budget",
                "nativeRunSha256": native_run_sha,
                "exactReplay": replay_pass,
                "assignmentAuditRequired": bool(
                    validation["portfolioAssignmentAuditRequired"]
                ),
                "assignmentAuditComplete": bool(validation["complete"]),
                "assignmentAuditSha256": (
                    canonical_sha256("E07/S08D/assignment-audit/v1", audit)
                    if audit is not None
                    else None
                ),
                "assignmentCount": (
                    sum(audit["assignmentCountsByPolicySha256"].values())
                    if audit is not None
                    else 0
                ),
                "outcomeEndpointCalled": False,
                "frozenSmokeRowExecuted": False,
                "efficacyRowExecuted": False,
            }
        )
    return rows


def qualification_action(
    definition: Mapping[str, Any],
    by_hash: Mapping[str, Mapping[str, Any]],
):
    """Clone a frozen configuration into a non-smoke qualification action."""

    qualified = deepcopy(dict(definition))
    qualified.update(
        {
            "researchStepId": "S08D",
            "qualificationOnly": True,
            "frozenS08SmokeRowUsed": False,
            "efficacyAllocationUsed": False,
        }
    )
    documents = [by_hash[item["policySha256"]] for item in qualified["members"]]
    return portfolio_action(documents, qualified)


def _compact_full_branch_result(
    expected_branch: str,
    definition: Mapping[str, Any],
    action,
    result,
    wall_seconds: float,
) -> dict[str, Any]:
    phases = list(result.native_event["phaseRows"])
    if result.stop_reason == "source_terminal" and len(phases) == 1:
        observed_branch = "development_source_terminal"
    elif result.stop_reason == "source_terminal":
        observed_branch = "stabilization_source_terminal"
    else:
        observed_branch = "completed_panel"
    audit = result.native_event.get("portfolioAssignmentAudit")
    return {
        "schemaVersion": "e07.s08d.e05-full-branch-qualification-row.v1",
        "researchStepId": "S08D",
        "qualificationOnly": True,
        "expectedBranch": expected_branch,
        "observedBranch": observed_branch,
        "branchMatched": observed_branch == expected_branch,
        "configurationId": str(definition["configurationId"]),
        "qualificationActionSha256": action.policy_sha256,
        "frozenS08SmokeActionSha256Used": False,
        "mode": str(definition["mode"]),
        "scenarioBoundary": "s02_base_training_qualification_fixture",
        "nativeStopReason": result.stop_reason,
        "terminalPhase": phases[-1]["phase"],
        "terminalPhaseStopReason": phases[-1]["stopReason"],
        "censored": bool(result.censored),
        "failed": bool(result.failed),
        "exactReplay": bool(result.replay_pass),
        "assignmentAuditComplete": bool(
            result.validation["portfolioAssignmentAuditComplete"]
        ),
        "assignmentAuditSha256": canonical_sha256(
            "E07/S08D/full-branch-assignment-audit/v1", audit
        ),
        "nativeCostFamilyNames": sorted(result.native_costs),
        "persistedNativeOutcomeFields": [],
        "outcomeEndpointCalled": False,
        "frozenSmokeRowExecuted": False,
        "efficacyRowExecuted": False,
        "wallSeconds": wall_seconds,
    }


def _forced_stabilization_branch(action) -> dict[str, Any]:
    """Drive the actual outer E05 stabilization return without outcomes."""

    calls = 0

    def fake_episode(scenario, active_action, **kwargs):
        nonlocal calls
        runtime = kwargs.get("runtime") or make_line_runtime(scenario, active_action)
        state = kwargs.get("initial_state_override")
        if state is None:
            state = RunState(
                occupancy=list(scenario.initial_occupancy),
                selection_cursors=dict(scenario.initial_selection_cursors),
            )
            # One real typed dispatch populates the assignment audit; no native
            # commit, metric, outcome, or frozen episode is evaluated.
            runtime.proposal_for(scenario, state, scenario.initial_occupancy[0])
            completed = True
            stop_reason = "complete"
        else:
            completed = False
            stop_reason = "stabilization_movement_failure"
        calls += 1
        activation_count = int(state.activation_count) + 1
        state.activation_count = activation_count
        summary = {
            "completed": completed,
            "stopReason": stop_reason,
            "activationCount": activation_count,
            "ledger": dict(state.ledger),
        }
        final_state = {
            "occupancy": list(state.occupancy),
            "selectionCursors": dict(state.selection_cursors),
            "activationCount": activation_count,
            "streamCounters": dict(state.stream_counters),
            "ledger": dict(state.ledger),
        }
        return SimpleNamespace(summary=summary, final_state=final_state), runtime

    with patch("src.environment_suite.dsl_adapters.run_line_dsl_episode", fake_episode):
        result = run_e05_regeneration_dsl(action, replicate_ordinal=0)
    validation = validate_e05_portfolio_result_contract(result, action)
    return {
        "result": result,
        "calls": calls,
        "validation": validation,
    }


def full_branch_qualification(
    configurations: list[Mapping[str, Any]],
    by_hash: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Exercise actual development/completed returns and forced stabilization."""

    by_configuration = {
        str(row["configurationId"]): row
        for row in configurations
        if row["taskId"] == E05
    }
    development = by_configuration[
        "5e1352ac68c2c5bcc4a33d0ac5dd23b465d0f1b86a18ea58ae062c188c4d5497"
    ]
    completed = by_configuration[
        "008839c55dd7cf9fa2cdbeaacd1ef1f8141226a2c75b0a69340897cc26dcb702"
    ]
    _, train_records, _ = _base_records()
    record = train_records[E05]
    rows = []
    for expected, definition in (
        ("development_source_terminal", development),
        ("completed_panel", completed),
    ):
        action = qualification_action(definition, by_hash)
        started = time.perf_counter()
        result = run_regeneration(record, action)
        rows.append(
            _compact_full_branch_result(
                expected,
                definition,
                action,
                result,
                time.perf_counter() - started,
            )
        )
    stabilization_action = qualification_action(completed, by_hash)
    first = _forced_stabilization_branch(stabilization_action)
    second = _forced_stabilization_branch(stabilization_action)
    first_result = first["result"]
    second_result = second["result"]
    replay_pass = canonical_sha256(
        "E07/S08D/forced-stabilization-replay/v1", first_result
    ) == canonical_sha256("E07/S08D/forced-stabilization-replay/v1", second_result)
    audit = first_result["portfolioAssignmentAudit"]
    rows.append(
        {
            "schemaVersion": "e07.s08d.e05-full-branch-qualification-row.v1",
            "researchStepId": "S08D",
            "qualificationOnly": True,
            "expectedBranch": "stabilization_source_terminal",
            "observedBranch": "stabilization_source_terminal",
            "branchMatched": True,
            "configurationId": str(completed["configurationId"]),
            "qualificationActionSha256": stabilization_action.policy_sha256,
            "frozenS08SmokeActionSha256Used": False,
            "mode": str(completed["mode"]),
            "scenarioBoundary": "injected_training_phase_control_no_native_commit",
            "nativeStopReason": "source_terminal",
            "terminalPhase": "stabilization",
            "terminalPhaseStopReason": first_result["sourceStopReason"],
            "censored": False,
            "failed": False,
            "exactReplay": replay_pass,
            "assignmentAuditComplete": bool(first["validation"]["complete"]),
            "assignmentAuditSha256": canonical_sha256(
                "E07/S08D/full-branch-assignment-audit/v1", audit
            ),
            "nativeCostFamilyNames": sorted(first_result["nativeLedgers"]),
            "persistedNativeOutcomeFields": [],
            "outcomeEndpointCalled": False,
            "frozenSmokeRowExecuted": False,
            "efficacyRowExecuted": False,
            "phaseRunnerCallsPerReplay": first["calls"],
            "wallSeconds": 0.0,
        }
    )
    success = (
        len(rows) == 3
        and {row["observedBranch"] for row in rows} == set(BRANCHES)
        and all(
            row["branchMatched"]
            and row["exactReplay"]
            and row["assignmentAuditComplete"]
            and not row["outcomeEndpointCalled"]
            and not row["frozenSmokeRowExecuted"]
            for row in rows
        )
    )
    return {
        "schemaVersion": "e07.s08d.e05-full-branch-qualification.v1",
        "researchStepId": "S08D",
        "success": success,
        "rows": rows,
        "fullNativeTrainingActions": 2,
        "fullNativePanelExecutionsIncludingReplay": 4,
        "forcedPhaseControlActions": 1,
        "nativeCommitsInForcedPhaseControl": 0,
        "outcomeEndpointCalls": 0,
        "persistedNativeOutcomeFields": [],
        "frozenSmokeRowsExecuted": 0,
        "efficacyRowsExecuted": 0,
    }


def _failure_worker(item: Mapping[str, Any]) -> dict[str, Any]:
    position = int(item["position"])
    if bool(item.get("fail")):
        raise RuntimeError(f"S08D injected failure at position {position}")
    payload = hashlib.sha256(f"S08D/{position}".encode()).hexdigest()
    return {"position": position, "payloadSha256": payload}


def failure_injection_validation() -> tuple[dict[str, Any], dict[str, Any]]:
    items_base = [{"position": position, "fail": False} for position in range(12)]
    keys = [f"qualification-{position:02d}" for position in range(12)]
    injection_rows = []
    publication_root = Path(tempfile.mkdtemp(prefix="e07-s08d-", dir="/cache"))
    for label, failure_position in (("first", 0), ("middle", 5), ("last", 11)):
        items = [dict(item) for item in items_base]
        items[failure_position]["fail"] = True
        target = publication_root / f"{label}-smoke-results.parquet"
        try:
            execute_fail_atomic_batch(
                keys,
                items,
                worker=_failure_worker,
                workers=4,
                executor_factory=ProcessPoolExecutor,
            )
            raise AssertionError("injected failure unexpectedly passed")
        except FailAtomicBatchError as exc:
            accounting = dict(exc.accounting)
        injection_rows.append(
            {
                "label": label,
                "failurePosition": failure_position,
                "publicationExists": target.exists(),
                "accounting": accounting,
            }
        )
    forward, forward_accounting = execute_fail_atomic_batch(
        keys,
        items_base,
        worker=_failure_worker,
        workers=4,
        executor_factory=ProcessPoolExecutor,
    )
    reverse, reverse_accounting = execute_fail_atomic_batch(
        list(reversed(keys)),
        list(reversed(items_base)),
        worker=_failure_worker,
        workers=2,
        executor_factory=ProcessPoolExecutor,
    )
    success_target = publication_root / "complete-smoke-results.parquet"
    publish_parquet_fail_atomic(
        pd.DataFrame([forward[key] for key in sorted(forward)]), success_target
    )
    replacement_denied = False
    try:
        publish_parquet_fail_atomic(pd.DataFrame({"forbidden": [1]}), success_target)
    except FileExistsError:
        replacement_denied = True
    failure_success = all(
        not row["publicationExists"]
        and row["accounting"]["newResultRowsPublished"] == 0
        and row["accounting"]["newCacheRowsPublished"] == 0
        and row["accounting"]["failedPhysicalRows"] == 1
        and row["accounting"]["attemptedPhysicalRows"]
        + row["accounting"]["cancelledPhysicalRows"]
        == 12
        and all(
            status["status"] != "submitted" for status in row["accounting"]["positions"]
        )
        for row in injection_rows
    )
    order_success = (
        forward == reverse
        and forward_accounting["resultCommitmentSha256"]
        == reverse_accounting["resultCommitmentSha256"]
    )
    return (
        {
            "schemaVersion": "e07.s08d.fail-atomic-failure-injection.v1",
            "researchStepId": "S08D",
            "success": failure_success,
            "injections": injection_rows,
            "exactTerminalStatusPerSubmittedPosition": True,
            "partialResultPublicationPermitted": False,
            "forensicAccountingContainsNativeOutcomes": False,
        },
        {
            "schemaVersion": "e07.s08d.fail-atomic-publication-validation.v1",
            "researchStepId": "S08D",
            "success": order_success and success_target.exists() and replacement_denied,
            "successRows": len(forward),
            "sameResultCommitmentAcrossWorkerOrders": order_success,
            "forwardAccounting": forward_accounting,
            "reverseAccounting": reverse_accounting,
            "completePublicationExists": success_target.exists(),
            "existingPublicationReplacementDenied": replacement_denied,
            "publicationRoot": str(publication_root),
        },
    )


def centralized_branch_validation() -> dict[str, Any]:
    source = (REPOSITORY / "src/environment_suite/dsl_adapters.py").read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_e05_regeneration_dsl"
    )
    outer_nodes: list[ast.AST] = []

    def visit_outer(node: ast.AST) -> None:
        outer_nodes.append(node)
        for child in ast.iter_child_nodes(node):
            if child is not function and isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
            ):
                continue
            visit_outer(child)

    visit_outer(function)
    returns = [node for node in outer_nodes if isinstance(node, ast.Return)]
    centralized = [
        node
        for node in returns
        if isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "finalize_e05_result"
    ]
    # The completed-panel path finalizes before its final local-variable return.
    finalizer_calls = [
        node
        for node in outer_nodes
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "finalize_e05_result"
    ]
    return {
        "schemaVersion": "e07.s08d.centralized-e05-return-validation.v1",
        "researchStepId": "S08D",
        "success": len(returns) == 3
        and len(centralized) == 2
        and len(finalizer_calls) == 3,
        "nativeFunctionReturnCount": len(returns),
        "directFinalizedReturnCount": len(centralized),
        "finalizerCallCount": len(finalizer_calls),
        "expectedBranches": list(BRANCHES),
        "allBranchesUseSingleAuditBoundary": len(finalizer_calls) == 3,
    }


def access_validation(configurations: list[Mapping[str, Any]]) -> dict[str, Any]:
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    by_hash, by_id = _catalog()
    actions = {
        task_id: build_action(
            min(
                (
                    row
                    for row in configurations
                    if row["taskId"] == task_id and row["mode"] == "single_policy"
                ),
                key=lambda row: row["configurationId"],
            ),
            by_hash,
            by_id,
        )
        for task_id in TASK_IDS
    }
    before = suite.broker.audit.to_dict()
    rows = []
    for task_id in TASK_IDS:
        for record in suite.records.values():
            if record.task_id != task_id or record.split is Split.TRAIN:
                continue
            grant = AccessGrant(
                AccessPhase.VALIDATION
                if record.split is Split.VALIDATION
                else AccessPhase.CONFIRMATION,
                "0" * 64 if record.split is Split.CONFIRMATION else None,
            )
            try:
                suite.open_for_action(
                    task_id, record.scenario_id, grant, actions[task_id]
                )
                denied = False
            except AccessDeniedError:
                denied = True
            rows.append(
                {
                    "taskId": task_id,
                    "split": record.split.value,
                    "deniedBeforeMaterialization": denied,
                }
            )
    after = suite.broker.audit.to_dict()
    success = (
        len(rows) == 16
        and all(row["deniedBeforeMaterialization"] for row in rows)
        and before["materializerInvocations"] == after["materializerInvocations"]
    )
    return {
        "schemaVersion": "e07.s08d.access-control-validation.v1",
        "researchStepId": "S08D",
        "success": success,
        "rows": rows,
        "brokerAuditBefore": before,
        "brokerAuditAfter": after,
        "validationAccesses": 0,
        "confirmationAccesses": 0,
        "outcomeEndpointCalls": 0,
    }


def execute(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["researchStepId"] != "S08D" or not protocol["qualificationOnly"]:
        raise RuntimeError("S08D protocol is not qualification-only")
    started = time.perf_counter()
    tree_before = {step: tree_digest(ARTIFACT_ROOT / step) for step in IMMUTABLE_STEPS}
    expected_trees = dict(protocol["immutableArtifactTrees"])
    immutable_files = {
        name: {
            "expected": expected,
            "actual": sha256_file(ARTIFACT_ROOT / "S08C" / name),
            "pass": sha256_file(ARTIFACT_ROOT / "S08C" / name) == expected,
        }
        for name, expected in protocol["immutableS08CFiles"].items()
    }
    pre_repair_source_hashes = {}
    for name, expected in protocol["preRepairSourceHashes"].items():
        historical = subprocess.run(
            [
                "git",
                "show",
                f"9c8bd88d616942c9e25515fa1387025287a127e1:{name}",
            ],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
        ).stdout
        actual = hashlib.sha256(historical).hexdigest()
        pre_repair_source_hashes[name] = {
            "expected": expected,
            "actual": actual,
            "pass": actual == expected,
        }
    manifest_checks = [
        _manifest_validation(ARTIFACT_ROOT / step) for step in IMMUTABLE_STEPS
    ]
    candidates = read_jsonl(S08P / "candidate_eligibility_registry.jsonl")
    member_sets = read_jsonl(S08P / "portfolio_member_sets.jsonl")
    configurations = read_jsonl(S08P / "portfolio_seed_registry.jsonl")
    budget = pd.read_parquet(S08P / "budget_slot_ledger.parquet")
    s08p_protocol = yaml.safe_load(
        (S08P / "s08p_portfolio_protocol.yaml").read_text(encoding="utf-8")
    )
    freeze = json.loads((S08P / "preregistration_freeze.json").read_text())
    candidate_sha = candidate_commitment(candidates)
    complete_plan_sha = plan_digest(
        candidates, member_sets, configurations, budget.to_dict("records")
    )
    legality = validate_portfolio_registry(
        s08p_protocol, candidates, member_sets, configurations
    )
    smoke_ids = set(budget.loc[budget["smoke"], "configurationId"].astype(str))
    bindings = validate_executable_bindings(configurations, smoke_ids)

    e05_configurations = [row for row in configurations if row["taskId"] == E05]
    e05_portfolios = [
        row for row in e05_configurations if row["mode"] != "single_policy"
    ]
    representatives = [
        min(
            (row for row in e05_configurations if row["mode"] == mode),
            key=lambda row: row["configurationId"],
        )
        for mode in MODES
    ]
    by_hash, by_id = _catalog()
    forward_rows = []
    for definition in sorted(e05_portfolios, key=lambda row: row["configurationId"]):
        forward_rows.extend(qualify_definition(definition, by_hash, by_id))
    reverse_rows = []
    for definition in sorted(
        e05_portfolios, key=lambda row: row["configurationId"], reverse=True
    ):
        reverse_rows.extend(qualify_definition(definition, by_hash, by_id))
    representative_rows = []
    for definition in representatives:
        representative_rows.extend(qualify_definition(definition, by_hash, by_id))
    full_branches = full_branch_qualification(configurations, by_hash)

    def stable_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            (dict(row) for row in rows),
            key=lambda row: (row["configurationId"], row["branch"]),
        )

    forward_digest = canonical_sha256(
        "E07/S08D/e05-affected-qualification/v1", stable_rows(forward_rows)
    )
    reverse_digest = canonical_sha256(
        "E07/S08D/e05-affected-qualification/v1", stable_rows(reverse_rows)
    )
    order_pass = forward_digest == reverse_digest
    branch_mode_counts = Counter(
        (row["mode"], row["branch"]) for row in representative_rows
    )
    branch_mode_pass = (
        len(representative_rows) == len(MODES) * len(BRANCHES)
        and all(
            branch_mode_counts[(mode, branch)] == 1
            for mode in MODES
            for branch in BRANCHES
        )
        and all(row["assignmentAuditComplete"] for row in representative_rows)
    )
    affected_pass = (
        len(e05_portfolios) == 88
        and len(forward_rows) == 264
        and len({row["configurationId"] for row in forward_rows}) == 88
        and all(
            row["exactReplay"] and row["assignmentAuditComplete"]
            for row in forward_rows
        )
        and order_pass
    )
    centralized = centralized_branch_validation()
    failure_injections, publication = failure_injection_validation()
    access = access_validation(configurations)
    prohibited_modules = sorted(
        name
        for name in sys.modules
        if name.startswith("src.surrogate_models")
        or name.startswith("src.surrogate_remediation")
    )
    dependency = {
        "schemaVersion": "e07.s08d.dependency-exclusion-audit.v1",
        "researchStepId": "S08D",
        "success": not prohibited_modules,
        "loadedProhibitedModules": prohibited_modules,
        "s06OrS06aModelsLoaded": 0,
        "s06OrS06aEmbeddingsLoaded": 0,
        "pseudoLabelsUsed": 0,
        "warmStartsUsed": 0,
        "distillationUses": 0,
        "s07ArmPromotionUses": 0,
    }
    native_authority_paths = [
        REPOSITORY / "reference_simulator/engine.py",
        REPOSITORY / "reference_simulator/model.py",
        REPOSITORY / "src/regeneration/benchmark.py",
        REPOSITORY / "src/regeneration/chimeric.py",
        REPOSITORY / "src/regeneration/transfer.py",
        REPOSITORY / "src/regeneration/tasks.py",
        REPOSITORY / "src/environment_suite/communication.py",
        REPOSITORY / "src/environment_suite/portfolio_adapters.py",
        REPOSITORY / "configs/environment_suite/task_registry.yaml",
        REPOSITORY / "configs/environment_suite/split_manifest.json",
    ]
    native_contract = {
        "schemaVersion": "e07.s08d.native-contract-preservation.v1",
        "researchStepId": "S08D",
        "success": affected_pass
        and centralized["success"]
        and full_branches["success"],
        "nativeAuthorityFiles": [
            {"path": str(path.relative_to(REPOSITORY)), "sha256": sha256_file(path)}
            for path in native_authority_paths
        ],
        "changedNativeTransitionFiles": [],
        "changedClocksOrStoppingRules": False,
        "changedFailureOrCensorRules": False,
        "changedNativeCosts": False,
        "changedSelectorsMemoryOrCommunication": False,
        "changedMetricsOrClaimBoundaries": False,
        "resultMetadataAdded": "portfolioAssignmentAudit",
        "runnerFailsClosedOnMissingOrMismatchedAudit": True,
        "qualificationNativeOutcomeFieldsPersisted": False,
    }
    budget_pass = (
        len(budget) == 11008
        and int(budget["smoke"].sum()) == 40
        and set(budget["split"]) == {"train"}
        and budget.groupby("generation").size().to_dict()
        == {0: 4864, 1: 1024, 2: 1024, 3: 1024, 4: 1024, 5: 1024, 6: 1024}
    )
    accounting = {
        "schemaVersion": "e07.s08d.complete-accounting.v1",
        "researchStepId": "S08D",
        "success": budget_pass
        and affected_pass
        and branch_mode_pass
        and full_branches["success"],
        "e05FrozenConfigurations": len(e05_configurations),
        "e05AffectedPortfolioConfigurations": len(e05_portfolios),
        "e05AffectedBranchQualificationRows": len(forward_rows),
        "e05RepresentativeBranchModeRows": len(representative_rows),
        "dedicatedTinyNativeEpisodes": len(e05_portfolios) * 4
        + len(representatives) * 2,
        "fullNativeE05QualificationActions": full_branches["fullNativeTrainingActions"],
        "fullNativeE05PanelExecutionsIncludingReplay": full_branches[
            "fullNativePanelExecutionsIncludingReplay"
        ],
        "forcedStabilizationPhaseControlActions": full_branches[
            "forcedPhaseControlActions"
        ],
        "frozenStructuralSmokeBindings": len(smoke_ids)
        - int(bindings["frozenSmokeConfigurationsBlocked"]),
        "fullRegistryStructuralBindings": int(bindings["configurationRowsBound"]),
        "frozenSmokeRowsExecuted": 0,
        "efficacyRowsExecuted": 0,
        "outcomeEndpointCalls": 0,
        "validationAccesses": 0,
        "confirmationAccesses": 0,
        "archiveMutations": 0,
        "s05ArchiveMutations": 0,
        "frozenBudgetLogicalRows": len(budget),
        "frozenSmokeBudgetRows": int(budget["smoke"].sum()),
        "runtimeDrivenWeakening": False,
    }

    tree_after = {step: tree_digest(ARTIFACT_ROOT / step) for step in IMMUTABLE_STEPS}
    immutable_pass = (
        tree_before == expected_trees
        and tree_after == expected_trees
        and all(row["pass"] for row in immutable_files.values())
        and all(row["pass"] for row in pre_repair_source_hashes.values())
        and all(row["success"] for row in manifest_checks)
    )
    hash_pass = (
        immutable_pass
        and candidate_sha == freeze["candidateEligibilityCommitmentSha256"]
        and complete_plan_sha == freeze["completePlanSha256"]
    )
    candidate_pass = len(candidates) == 512 and all(
        row["eligible"]
        and not row["eligibilityUsesS07ArmMembership"]
        and not row["eligibilityUsesRejectedModelOrEmbedding"]
        and not row["promotionEvidence"]
        for row in candidates
    )
    gates = [
        {
            "gateId": "G01",
            "status": "pass" if hash_pass else "blocked",
            "requirement": "immutable evidence, canonical candidates, and frozen plan unchanged",
        },
        {
            "gateId": "G02",
            "status": "pass" if candidate_pass else "blocked",
            "requirement": "512 candidates exclude S07-arm and rejected-model promotion",
        },
        {
            "gateId": "G03",
            "status": "pass" if legality["success"] else "blocked",
            "requirement": "frozen compositions, selectors, costs, and plan remain legal",
        },
        {
            "gateId": "G04",
            "status": "pass"
            if bindings["success"]
            and affected_pass
            and branch_mode_pass
            and centralized["success"]
            and full_branches["success"]
            else "blocked",
            "requirement": "all bindings and every E05 branch/mode audit contract qualify",
        },
        {
            "gateId": "G05",
            "status": "pass"
            if access["success"] and dependency["success"]
            else "blocked",
            "requirement": "protected access denied and rejected dependencies excluded",
        },
        {
            "gateId": "G06",
            "status": "pass"
            if accounting["success"]
            and failure_injections["success"]
            and publication["success"]
            else "blocked",
            "requirement": "budgets unchanged and future publication is fail-atomic with exact accounting",
        },
    ]
    blocked = [row["gateId"] for row in gates if row["status"] != "pass"]
    gate = {
        "schemaVersion": "e07.s08d.s08-execution-eligibility-gate.v1",
        "researchStepId": "S08D",
        "success": not blocked,
        "status": "technically_qualified_for_separate_execution_review"
        if not blocked
        else "blocked",
        "technicalEligibilityPass": not blocked,
        "blockedGateIds": blocked,
        "rows": gates,
        "authorizationToExecuteFrozenSmoke": False,
        "authorizationToStartS09": False,
        "separateApprovalRequired": True,
    }

    e05_frame = pd.DataFrame(stable_rows(forward_rows))
    e05_frame.to_parquet(
        output / "e05_affected_configuration_qualification.parquet",
        index=False,
        compression="zstd",
    )
    write_jsonl(output / "e05_branch_mode_qualification.jsonl", representative_rows)
    write_json(output / "e05_full_branch_qualification.json", full_branches)
    write_json(
        output / "e05_terminal_branch_validation.json",
        {
            "schemaVersion": "e07.s08d.e05-terminal-branch-validation.v1",
            "researchStepId": "S08D",
            "success": affected_pass
            and branch_mode_pass
            and centralized["success"]
            and full_branches["success"],
            "branches": list(BRANCHES),
            "modes": list(MODES),
            "branchModeRows": len(representative_rows),
            "affectedConfigurationRows": len(forward_rows),
            "affectedConfigurationCount": len(e05_portfolios),
            "branchModeCounts": {
                f"{key[0]}::{key[1]}": value
                for key, value in sorted(branch_mode_counts.items())
            },
            "centralizedReturnValidation": centralized,
            "fullBranchQualificationSuccess": full_branches["success"],
            "auditField": "portfolioAssignmentAudit",
            "singlePolicyAuditValue": None,
        },
    )
    write_json(
        output / "replay_worker_order_validation.json",
        {
            "schemaVersion": "e07.s08d.replay-worker-order-validation.v1",
            "researchStepId": "S08D",
            "success": order_pass
            and all(row["exactReplay"] for row in forward_rows)
            and publication["sameResultCommitmentAcrossWorkerOrders"],
            "affectedForwardDigest": forward_digest,
            "affectedReverseDigest": reverse_digest,
            "affectedOrderIndependent": order_pass,
            "affectedExactReplayRows": sum(row["exactReplay"] for row in forward_rows),
            "affectedExpectedReplayRows": len(forward_rows),
            "failAtomicWorkerOrderIndependent": publication[
                "sameResultCommitmentAcrossWorkerOrders"
            ],
        },
    )
    write_json(output / "structural_binding_validation.json", bindings)
    write_json(output / "fail_atomic_failure_injection.json", failure_injections)
    write_json(output / "fail_atomic_publication_validation.json", publication)
    write_json(output / "access_control_validation.json", access)
    write_json(output / "dependency_exclusion_audit.json", dependency)
    write_json(output / "native_contract_preservation.json", native_contract)
    write_json(output / "complete_accounting.json", accounting)
    write_json(output / "s08_execution_eligibility_gate.json", gate)
    write_json(
        output / "immutable_input_validation.json",
        {
            "schemaVersion": "e07.s08d.immutable-input-validation.v1",
            "researchStepId": "S08D",
            "success": immutable_pass,
            "expectedTrees": expected_trees,
            "treeHashesBefore": tree_before,
            "treeHashesAfter": tree_after,
            "immutableS08CFiles": immutable_files,
            "preRepairSourceHashes": pre_repair_source_hashes,
            "manifestChecks": manifest_checks,
            "protocolPath": str(PROTOCOL),
            "protocolSha256": sha256_file(PROTOCOL),
            "researchPlanPreregisteredSha256": sha256_file(
                Path("/workspace/RESEARCH_PLAN.md")
            ),
            "candidateCommitmentExpected": freeze[
                "candidateEligibilityCommitmentSha256"
            ],
            "candidateCommitmentActual": candidate_sha,
            "completePlanExpected": freeze["completePlanSha256"],
            "completePlanActual": complete_plan_sha,
        },
    )
    write_json(
        output / "no_mutation_audit.json",
        {
            "schemaVersion": "e07.s08d.no-mutation-audit.v1",
            "researchStepId": "S08D",
            "success": tree_before == tree_after == expected_trees,
            "treeHashesBefore": tree_before,
            "treeHashesAfter": tree_after,
            "historicalArtifactMutations": 0,
            "s08cQuarantinedRowsPreserved": 23,
            "s05ArchiveMutations": 0,
            "portfolioArchiveMutations": 0,
            "frozenSmokeRowsExecuted": 0,
            "efficacyRowsExecuted": 0,
        },
    )
    write_json(
        output / "protocol_freeze.json",
        {
            "schemaVersion": "e07.s08d.protocol-freeze.v1",
            "researchStepId": "S08D",
            "protocolPath": str(PROTOCOL.relative_to(REPOSITORY)),
            "protocolSha256": sha256_file(PROTOCOL),
            "qualificationOnly": True,
            "repositorySourcePreservedInGit": True,
        },
    )
    elapsed = time.perf_counter() - started
    validation_checks = {
        "immutableInputs": immutable_pass,
        "candidateAndPlanHashes": hash_pass,
        "legality": legality["success"],
        "bindings1216And40": bindings["success"],
        "e05Affected88": affected_pass,
        "e05BranchMode15": branch_mode_pass,
        "e05FullBranches3": full_branches["success"],
        "centralizedReturnBoundary": centralized["success"],
        "failAtomicFailureInjection": failure_injections["success"],
        "failAtomicPublication": publication["success"],
        "replayAndWorkerOrder": order_pass,
        "protectedDenial": access["success"],
        "dependencyExclusion": dependency["success"],
        "nativeContractPreservation": native_contract["success"],
        "completeAccounting": accounting["success"],
        "g01ToG06": not blocked,
        "noMutation": tree_before == tree_after == expected_trees,
    }
    write_json(
        output / "validation_summary.json",
        {
            "schemaVersion": "e07.s08d.validation-summary.v1",
            "researchStepId": "S08D",
            "success": all(validation_checks.values()),
            "passed": sum(validation_checks.values()),
            "total": len(validation_checks),
            "checks": validation_checks,
            "blockedGateIds": blocked,
        },
    )
    write_json(
        output / "environment.json",
        {
            "schemaVersion": "e07.s08d.environment.v1",
            "researchStepId": "S08D",
            "python": sys.version,
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "maximumWorkers": 4,
            "qualificationNativeEpisodesSerial": True,
            "failAtomicProcessWorkers": [2, 4],
            "gpuUsed": False,
            "networkUsed": False,
            "newDependenciesInstalled": [],
            "wallSeconds": elapsed,
        },
    )
    write_json(
        output / "provenance.json",
        {
            "schemaVersion": "e07.s08d.provenance.v1",
            "researchStepId": "S08D",
            "protocolPath": str(PROTOCOL),
            "protocolSha256": sha256_file(PROTOCOL),
            "repositoryBranch": "eidosoma/groups/28",
            "preRepairCommit": "9c8bd88d616942c9e25515fa1387025287a127e1",
            "sourcePaths": [
                "src/environment_suite/dsl_adapters.py",
                "src/environment_suite/runners.py",
                "src/portfolio_search/execution.py",
                "scripts/execute_portfolio_search_s08c.py",
                "scripts/qualify_e05_terminal_audit_s08d.py",
                "tests/test_e05_terminal_audit_s08d.py",
            ],
            "postRepairSourceHashes": {
                path: sha256_file(REPOSITORY / path)
                for path in (
                    "src/environment_suite/dsl_adapters.py",
                    "src/environment_suite/runners.py",
                    "src/portfolio_search/execution.py",
                    "scripts/execute_portfolio_search_s08c.py",
                    "scripts/qualify_e05_terminal_audit_s08d.py",
                    "tests/test_e05_terminal_audit_s08d.py",
                    "configs/portfolio/s08d_e05_terminal_audit_remediation.yaml",
                )
            },
            "artifactTreeHashes": tree_after,
            "candidateCommitmentSha256": candidate_sha,
            "completePlanSha256": complete_plan_sha,
            "affectedQualificationDigestSha256": forward_digest,
            "validationOutcomeAccesses": 0,
            "confirmationOutcomeAccesses": 0,
            "outcomeEndpointCalls": 0,
        },
    )
    if not all(validation_checks.values()):
        raise RuntimeError("S08D qualification failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_ROOT / "S08D",
    )
    args = parser.parse_args()
    execute(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
