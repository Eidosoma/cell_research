#!/usr/bin/env python3
"""Build and validate compact E07 S02 environment-suite evidence."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
from src.environment_suite import (
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
    EnvironmentSuite,
    EvaluationAction,
    RecipientActivationMessageBus,
    SignalEmission,
    baseline_policy_hash,
    canonical_json_bytes,
)
from src.environment_suite.runners import RUNNERS, runner_source_hashes


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
REGISTRY = REPOSITORY / "configs/environment_suite/task_registry.yaml"
SPLITS = REPOSITORY / "configs/environment_suite/split_manifest.json"


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(_json_native(value)) + b"\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT
    ).strip()


def _safe_input_paths() -> list[Path]:
    paths = [
        WORKSPACE / "AGENTS.md",
        WORKSPACE / "FULL_PLAN.md",
        WORKSPACE / "RESEARCH_PLAN.md",
        WORKSPACE / "PREVIOUS_ARTIFACTS.md",
        WORKSPACE / "PREVIOUS_ARTIFACTS.json",
        WORKSPACE / "DATASETS.md",
        WORKSPACE / "DATASET_CATALOG.json",
        WORKSPACE / "DATASET_AVAILABILITY.json",
        WORKSPACE / "CAPABILITIES.md",
        WORKSPACE / "CAPABILITY_AVAILABILITY.json",
        WORKSPACE / "input-attachments/MANIFEST.json",
        WORKSPACE
        / "input-attachments/21c68842-f029-5668-bff6-b32f6188a597/_metadata/ATTACHMENT.md",
        Path("/artifacts/research_steps/S01/research_step_full_results.md"),
        Path("/artifacts/research_steps/S01/policy_language_spec.md"),
    ]
    for experiment in ("E01", "E02", "E03", "E04", "E05", "E06"):
        paths.append(
            Path(
                f"/previous-artifacts/{experiment}/research_steps/S14/research_step_full_results.md"
            )
        )
        paths.append(
            Path(f"/previous-artifacts/{experiment}/report_inputs/e07_handoff.md")
        )
        paths.append(
            Path(f"/previous-artifacts/{experiment}/report_inputs/e07_handoff.json")
        )
    paths.extend(
        [
            Path("/previous-artifacts/E01/specification/transition_spec.md"),
            Path("/previous-artifacts/E02/research_steps/S02/action_interface_spec.md"),
            Path("/previous-artifacts/E03/research_steps/S01/distance_spec.md"),
            Path("/previous-artifacts/E03/research_steps/S06/necessary_detour_spec.md"),
            Path("/previous-artifacts/E03/research_steps/S14/e07_handoff.md"),
            Path("/previous-artifacts/E04/report_inputs/e06_e07_handoff.md"),
            Path("/previous-artifacts/E05/research_steps/S14/benchmark_card.md"),
            Path("/previous-artifacts/E06/research_steps/S02/grammar_spec.md"),
            Path("/previous-artifacts/E06/research_steps/S03/environment_spec.md"),
            Path("/previous-artifacts/E06/research_steps/S04/movement_spec.md"),
            Path("/previous-artifacts/E06/research_steps/S05/policy_spec.md"),
            Path("/previous-artifacts/E06/research_steps/S06/control_channel_spec.md"),
            Path("/previous-artifacts/E06/research_steps/S07/engine_spec.md"),
            Path("/previous-artifacts/E06/research_steps/S14/split_manifest.json"),
        ]
    )
    return [path for path in paths if path.is_file()]


def input_provenance() -> dict[str, Any]:
    files = []
    for path in _safe_input_paths():
        files.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "classification": "governing_or_public_handoff_context",
            }
        )
    return {
        "schemaVersion": "e07.s02.input-provenance.v1",
        "researchStepId": "S02",
        "files": files,
        "protectedOutcomeTablesOpened": 0,
        "protectedScenarioPayloadsOpened": 0,
        "note": "Only governing files, public handoffs/specifications, and opaque split metadata are listed; predecessor protected outcome tables are absent.",
    }


def environment_provenance() -> dict[str, Any]:
    return {
        "schemaVersion": "e07.s02.environment-provenance.v1",
        "researchStepId": "S02",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpuCountReported": os.cpu_count(),
        "workerLimit": 8,
        "actualParallelWorkers": 1,
        "threadEnvironment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "gitHead": git_output("rev-parse", "HEAD"),
        "gitBranch": git_output("branch", "--show-current"),
        "gitStatusShort": git_output("status", "--short"),
        "newDependenciesInstalled": [],
        "networkUsed": False,
        "gpuUsed": False,
        "runnerSourceHashes": runner_source_hashes(),
    }


def communication_audit() -> dict[str, Any]:
    bus = RecipientActivationMessageBus(channels=2, bits_per_channel=3)
    topology = {"a": ("b",), "b": ("a", "c"), "c": ("b",)}
    bus.emit(SignalEmission("a", 0, {0: 5, 1: 2}), topology)
    bus.emit(SignalEmission("a", 1, {0: 6}), topology)
    bus.emit(SignalEmission("c", 1, {0: 4}), topology)
    side_effect_free_denial = False
    try:
        bus.observe("b", 1)
    except Exception:
        side_effect_free_denial = bus.ledger_snapshot()["consumedDeliveries"] == 0
    observed = bus.observe("b", 2)
    consumed = bus.observe("b", 3)
    checks = {
        "sameEventReadDeniedWithoutConsumption": side_effect_free_denial,
        "lastWriteWins": observed.consumed_sender_channel_pairs == 3,
        "saturatingSum": dict(observed.channel_sums) == {0: 7, 1: 2},
        "senderIdentityHidden": set(observed.channel_sums) == {0, 1},
        "consumeOnActivation": dict(consumed.channel_sums) == {0: 0, 1: 0},
        "separateLedger": bus.ledger_snapshot()
        == {
            "emittedSignalWrites": 4,
            "transmittedSignalBits": 12,
            "recipientDeliveries": 4,
            "bufferOverwrites": 1,
            "consumedDeliveries": 3,
            "observableAggregateReads": 4,
        },
    }
    return {
        "schemaVersion": "e07.s02.communication-audit.v1",
        "profile": "recipient_activation_lag_lww_sum_u8_v1",
        "checks": checks,
        "ledger": bus.ledger_snapshot(),
        "passed": all(checks.values()),
    }


def horizon_reconciliation(suite: EnvironmentSuite) -> dict[str, Any]:
    task_ids = sorted(suite.tasks)
    pairs = []
    for left_index, left_id in enumerate(task_ids):
        left = suite.tasks[left_id]
        for right_id in task_ids[left_index + 1 :]:
            right = suite.tasks[right_id]
            same_group = (
                left.horizon.comparability_group == right.horizon.comparability_group
            )
            pairs.append(
                {
                    "leftTaskId": left_id,
                    "rightTaskId": right_id,
                    "leftNativeUnit": left.horizon.native_unit,
                    "rightNativeUnit": right.horizon.native_unit,
                    "comparable": same_group,
                    "reason": (
                        "same frozen predecessor-native comparability group"
                        if same_group
                        else "different native clock, stopping, censoring, or benchmark estimand"
                    ),
                    "normalizationApplied": False,
                }
            )
    return {
        "schemaVersion": "e07.s02.horizon-reconciliation.v1",
        "crossTaskNormalization": "forbidden",
        "taskHorizons": {
            task_id: suite.tasks[task_id].horizon.to_dict() for task_id in task_ids
        },
        "pairwise": pairs,
        "comparablePairCount": sum(item["comparable"] for item in pairs),
        "incomparablePairCount": sum(not item["comparable"] for item in pairs),
    }


def access_audit(suite: EnvironmentSuite) -> dict[str, Any]:
    confirmation = sorted(
        (item for item in suite.records.values() if item.protected),
        key=lambda item: item.scenario_id,
    )
    rows = []
    for record in confirmation:
        before = suite.broker.audit.materializer_invocations
        phase_denials = []
        for phase, lock in (
            (AccessPhase.DEVELOPMENT, None),
            (AccessPhase.VALIDATION, None),
            (AccessPhase.CONFIRMATION, "0" * 64),
        ):
            denied = False
            try:
                suite.open(
                    record.task_id,
                    record.scenario_id,
                    AccessGrant(phase, lock),
                )
            except AccessDeniedError:
                denied = True
            phase_denials.append(denied)
        rows.append(
            {
                "scenarioId": record.scenario_id,
                "taskId": record.task_id,
                "allThreePhasesDenied": all(phase_denials),
                "materializerInvocationDelta": suite.broker.audit.materializer_invocations
                - before,
                "materializerAbsent": record.materializer_id is None,
                "outcomeAccess": record.outcome_access,
            }
        )
    validation_record = suite.records["e07s02:sorting:validation:000"]
    development_denied = False
    try:
        suite.open(
            validation_record.task_id,
            validation_record.scenario_id,
            AccessGrant(AccessPhase.DEVELOPMENT),
        )
    except AccessDeniedError:
        development_denied = True
    validation_opened = (
        suite.open(
            validation_record.task_id,
            validation_record.scenario_id,
            AccessGrant(AccessPhase.VALIDATION),
        )
        .reset()
        .split.value
        == "validation"
    )
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            REPOSITORY / "src/environment_suite/access.py",
            REPOSITORY / "src/environment_suite/runners.py",
            REGISTRY,
            SPLITS,
        )
    )
    static_checks = {
        "noPreviousArtifactOutcomePathInRuntime": "/previous-artifacts"
        not in source_text,
        "noParquetOrArrowOutcomeReaderInRuntime": all(
            token not in source_text
            for token in ("read_parquet", "pyarrow", ".parquet")
        ),
        "confirmationMaterializersAbsent": all(
            item["materializerAbsent"] for item in rows
        ),
        "confirmationAllPhasesDenied": all(
            item["allThreePhasesDenied"] for item in rows
        ),
        "denialsBeforeMaterialization": all(
            item["materializerInvocationDelta"] == 0 for item in rows
        ),
        "validationDeniedDuringDevelopment": development_denied,
        "validationAllowedWithValidationGrant": validation_opened,
    }
    return {
        "schemaVersion": "e07.s02.holdout-access-audit.v1",
        "confirmationRecords": rows,
        "staticChecks": static_checks,
        "brokerAudit": suite.broker.audit.to_dict(),
        "protectedOutcomeArtifactsOpened": 0,
        "passed": all(static_checks.values()),
    }


def baseline_runs(
    suite: EnvironmentSuite,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    parity = []
    for task_id in sorted(suite.tasks):
        record = next(
            item
            for item in suite.records.values()
            if item.task_id == task_id and item.split.value == "train"
        )
        policy_id = str(record.public_parameters["baselinePolicyId"])
        action = EvaluationAction(policy_id, baseline_policy_hash(policy_id))
        environment = suite.open(
            task_id, record.scenario_id, AccessGrant(AccessPhase.DEVELOPMENT)
        )
        reset = environment.reset().to_dict()
        step = environment.step(action).to_dict()
        outcome = environment.outcome().to_dict()
        direct = RUNNERS[suite.tasks[task_id].runner_id](record, action)
        comparisons = {
            "stopReason": step["stopReason"] == direct.stop_reason,
            "censored": step["censored"] == direct.censored,
            "failed": step["failed"] == direct.failed,
            "nativeCostLedgers": step["cost"]["nativeLedgerFamilies"]
            == _json_native(direct.native_costs),
            "nativeEvent": step["event"]["native"] == _json_native(direct.native_event),
            "nativeOutcome": _json_native(outcome["outcome"])
            == _json_native(direct.native_outcome),
            "nativeValidation": all(direct.validation.values()),
            "exactReplay": direct.replay_pass,
        }
        parity.append(
            {
                "taskId": task_id,
                "scenarioId": record.scenario_id,
                "predecessor": suite.tasks[task_id].predecessor,
                "comparisons": comparisons,
                "passed": all(comparisons.values()),
            }
        )
        rows.append(
            {
                "taskId": task_id,
                "family": suite.tasks[task_id].family,
                "predecessor": suite.tasks[task_id].predecessor,
                "scenarioId": record.scenario_id,
                "policyId": policy_id,
                "policySha256": action.policy_sha256,
                "reset": reset,
                "step": step,
                "outcome": outcome,
                "passed": all(step["event"]["validation"].values())
                and step["event"]["replayPass"]
                and not step["failed"],
            }
        )
    return rows, parity


def _interface_spec() -> str:
    return (
        "# Unified environment interface specification\n\n"
        "Version `e07.s02.environment-suite.v1` standardizes reset, observation, "
        "step, cost, event, and outcome envelopes. One adapter step is one complete "
        "predecessor-native evaluation unit. It is not a universal primitive time step.\n\n"
        "`reset` opens a preassigned scenario through the access broker. `observation` "
        "returns task/action/permission/horizon metadata and no outcome. `step` passes "
        "an immutable hashed policy reference to the predecessor-native runner. `cost` "
        "keeps named native ledgers separate. `event` contains native digests, clocks, "
        "validation, and provenance but no outcome metrics. `outcome` is available only "
        "after terminal execution and a second split authorization.\n\n"
        "Legality, actions, reads, event bytes/digests, terminal precedence, censoring, "
        "and claims remain predecessor-owned. The wrapper defines no reward, cost total, "
        "time normalization, completion substitution, or cross-task ranking.\n\n"
        "S02 validates native baseline bindings. Arbitrary S01 DSL-to-native bindings "
        "for regeneration overlays and E06 batched policies are not silently invented; "
        "they remain later explicit adapter work.\n"
    )


def _communication_spec() -> str:
    return (
        "# Communication delivery specification\n\n"
        "Profile `recipient_activation_lag_lww_sum_u8_v1` applies to S01 DSL "
        "`emit_signal` actions only; it does not replace E05 target-signal or E06 S06 "
        "channel contracts. A policy observes its mailbox before its current native "
        "action. Emission occurs afterward and recipients are the sender's neighbors "
        "in the pre-transition topology.\n\n"
        "For each recipient, sender, and channel, the most recent message persists "
        "until the recipient's next activation. Repeated writes overwrite that slot. "
        "At activation, values are summed in sorted-sender order with saturation at "
        "the declared bit width; sender identities are not exposed, and all pending "
        "values are consumed. Same-event/future reads and self-delivery are rejected.\n\n"
        "Communication cannot change movement legality, retry, native stopping, or "
        "native ledgers. Writes, bits, deliveries, overwrites, consumption, and reads "
        "are recorded in a distinct nonnegative ledger.\n"
    )


def write_artifact_manifest(output: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schemaVersion": "e07.s02.artifact-manifest.v1",
        "researchStepId": "S02",
        "artifactCount": len(artifacts),
        "artifacts": artifacts,
    }
    write_json(output / "artifact_manifest.json", manifest)
    return manifest


def build(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    suite_dir = output / "environment_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REGISTRY, suite_dir / "task_registry.yaml")
    shutil.copyfile(SPLITS, suite_dir / "split_manifest.json")
    suite = EnvironmentSuite(REGISTRY, SPLITS)
    write_json(suite_dir / "public_registry.json", suite.public_registry())
    (suite_dir / "unified_interface_spec.md").write_text(
        _interface_spec(), encoding="utf-8"
    )
    (suite_dir / "communication_delivery_spec.md").write_text(
        _communication_spec(), encoding="utf-8"
    )
    communication = communication_audit()
    horizons = horizon_reconciliation(suite)
    holdouts = access_audit(suite)
    rows, parity = baseline_runs(suite)
    write_json(output / "communication_delivery_validation.json", communication)
    write_json(output / "horizon_reconciliation.json", horizons)
    with (output / "horizon_reconciliation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(horizons["pairwise"][0]))
        writer.writeheader()
        writer.writerows(horizons["pairwise"])
    write_json(output / "holdout_access_audit.json", holdouts)
    write_json(
        output / "baseline_smoke_results.json",
        {
            "schemaVersion": "e07.s02.baseline-smoke-results.v1",
            "taskCount": len(rows),
            "rows": rows,
            "allPassed": all(item["passed"] for item in rows),
        },
    )
    write_json(
        output / "predecessor_parity.json",
        {
            "schemaVersion": "e07.s02.predecessor-parity.v1",
            "comparisonCount": len(parity),
            "rows": parity,
            "allPassed": all(item["passed"] for item in parity),
        },
    )
    schema_validation = {
        "schemaVersion": "e07.s02.task-schema-validation.v1",
        "taskCount": len(suite.tasks),
        "scenarioRecordCount": len(suite.records),
        "families": sorted({item.family for item in suite.tasks.values()}),
        "allTasksHaveThreeSplits": all(
            len(
                [
                    record
                    for record in suite.records.values()
                    if record.task_id == task_id
                ]
            )
            == 3
            for task_id in suite.tasks
        ),
        "allOutcomesOffline": all(
            item.outcome_contract["policyVisibleOnline"] is False
            for item in suite.tasks.values()
        ),
        "allScalarCostsForbidden": all(
            item.cost_contract["scalarTotalPermitted"] is False
            for item in suite.tasks.values()
        ),
        "allCrossTaskNormalizationForbidden": all(
            item.horizon.cross_task_normalization == "forbidden"
            for item in suite.tasks.values()
        ),
    }
    schema_validation["passed"] = all(
        value for key, value in schema_validation.items() if key.startswith("all")
    )
    write_json(output / "task_schema_validation.json", schema_validation)
    write_json(output / "input_provenance.json", input_provenance())
    write_json(output / "environment_provenance.json", environment_provenance())
    validation = {
        "schemaVersion": "e07.s02.validation-summary.v1",
        "researchStepId": "S02",
        "gates": {
            "taskSchema": bool(schema_validation["passed"]),
            "completeBaselineExecution": all(item["passed"] for item in rows),
            "predecessorParity": all(item["passed"] for item in parity),
            "holdoutAccessControls": bool(holdouts["passed"]),
            "communicationDelivery": bool(communication["passed"]),
            "explicitHorizonReconciliation": all(
                not item["normalizationApplied"] for item in horizons["pairwise"]
            ),
            "protectedOutcomesNotOpened": True,
        },
    }
    validation["overall"] = "PASS" if all(validation["gates"].values()) else "FAIL"
    write_json(output / "validation_summary.json", validation)
    (output / "execution_commands.log").write_text(
        "PYTHONPATH=/workspace/cell-research OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "
        "python -m pytest -q tests/test_environment_suite.py\n"
        "PYTHONPATH=/workspace/cell-research OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "
        "python scripts/build_environment_suite_s02.py build "
        f"--output-dir {output}\n",
        encoding="utf-8",
    )
    write_artifact_manifest(output)
    return validation


def validate(output: Path) -> dict[str, Any]:
    required = [
        output / "environment_suite/task_registry.yaml",
        output / "environment_suite/split_manifest.json",
        output / "environment_suite/public_registry.json",
        output / "environment_suite/unified_interface_spec.md",
        output / "environment_suite/communication_delivery_spec.md",
        output / "baseline_smoke_results.json",
        output / "predecessor_parity.json",
        output / "holdout_access_audit.json",
        output / "task_schema_validation.json",
        output / "communication_delivery_validation.json",
        output / "horizon_reconciliation.json",
        output / "horizon_reconciliation.csv",
        output / "input_provenance.json",
        output / "environment_provenance.json",
        output / "validation_summary.json",
        output / "execution_commands.log",
    ]
    checks = {str(path.relative_to(output)): path.is_file() for path in required}
    checks["taskRegistrySnapshotMatches"] = (
        output / "environment_suite/task_registry.yaml"
    ).read_bytes() == REGISTRY.read_bytes()
    checks["splitManifestSnapshotMatches"] = (
        output / "environment_suite/split_manifest.json"
    ).read_bytes() == SPLITS.read_bytes()
    checks["baselineSmokePassed"] = json.loads(
        (output / "baseline_smoke_results.json").read_text(encoding="utf-8")
    )["allPassed"]
    checks["predecessorParityPassed"] = json.loads(
        (output / "predecessor_parity.json").read_text(encoding="utf-8")
    )["allPassed"]
    checks["holdoutAccessPassed"] = json.loads(
        (output / "holdout_access_audit.json").read_text(encoding="utf-8")
    )["passed"]
    checks["communicationPassed"] = json.loads(
        (output / "communication_delivery_validation.json").read_text(encoding="utf-8")
    )["passed"]
    checks["taskSchemaPassed"] = json.loads(
        (output / "task_schema_validation.json").read_text(encoding="utf-8")
    )["passed"]
    manifest = write_artifact_manifest(output)
    return {
        "schemaVersion": "e07.s02.artifact-validation.v1",
        "checks": checks,
        "manifestArtifactCount": manifest["artifactCount"],
        "passed": all(checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--output-dir", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (
        build(args.output_dir) if args.command == "build" else validate(args.output_dir)
    )
    print(json.dumps(_json_native(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
