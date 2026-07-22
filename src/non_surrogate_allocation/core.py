"""Deterministic, train-only S07 execution for the frozen S07R design.

The module evaluates native episodes and writes immutable result ledgers.  It
has no model, embedding, pseudo-label, archive insertion, or protected-split
path.  The S05 archive is read only to define the already-frozen candidate and
descriptor-cell populations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml

from src.allocation_redesign.core import (
    FUTURE_APPROVAL_TOKEN,
    TASK_IDS,
    build_allocation_roster,
    build_candidate_registry,
    build_candidate_selection_frame,
    candidate_population_commitment,
    canonical_hash,
    checked_protocol,
    hash_file,
)
from src.environment_suite import (
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
    EnvironmentSuite,
    ScenarioRecord,
    Split,
)
from src.environment_suite.contracts import canonical_sha256
from src.environment_suite.runners import RUNNERS
from src.policy_dsl import compile_policy
from src.quality_diversity.core import (
    BASE_SCENARIOS,
    DESCRIPTOR_REGISTRY,
    OBJECTIVES,
    SPLIT_MANIFEST,
    TASK_REGISTRY,
    _aggregate_descriptors,
    _get_path,
    _json_safe,
    _numeric_leaves,
    _strip_hash_fields,
    build_action,
    policy_body_sha256,
    stable_cell_audit,
)


REPOSITORY = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = Path("/artifacts/research_steps/S07")
CACHE_DIR = Path("/cache/e07_s07/physical_evaluations")
LOCK_PATH = REPOSITORY / "configs/allocation/s07_execution_lock.yaml"
S07R_DIR = Path("/artifacts/research_steps/S07R")
S05_DIR = Path("/artifacts/research_steps/S05")
PROTOCOL_SHA256 = "162d11c74c9e23a724bcd872b760819b2f03bc84745fd7d7fe860e107166b841"
CANDIDATE_COMMITMENT = "e3b044fb18f373928ead967739fc64c28c5841744c9e3ce5ad63458d624f3a6b"
DERIVATION_DOMAIN = "E07/S07/non-surrogate-training-scenarios/v1"
FAMILY_ORDINALS = (200, 201, 202, 203)
WORKERS = 8
EXPECTED_E05_FAILURE_FLAG = "developmentBudgetRespected"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def _parquet_safe(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    converted = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple)):
                item[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
            else:
                item[key] = value
        converted.append(item)
    return pd.DataFrame(converted)


def _counter_u64(*parts: object, stream: str) -> int:
    payload = "/".join(str(part) for part in parts)
    digest = hashlib.sha256(
        DERIVATION_DOMAIN.encode("ascii") + b"\x00" + stream.encode("ascii")
        + b"\x00" + payload.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def derive_train_record(base: ScenarioRecord, ordinal: int) -> ScenarioRecord:
    """Derive an outcome-blind S07 family without consulting another split."""

    if base.split is not Split.TRAIN or base.protected:
        raise RuntimeError("S07 derivation requires a nonprotected train base")
    if ordinal not in FAMILY_ORDINALS:
        raise RuntimeError("scenario family is outside the frozen S07 roster")
    task_id = base.task_id
    parameters = deepcopy(dict(base.public_parameters))
    if "seed" in parameters:
        parameters["seed"] = 1 + _counter_u64(task_id, ordinal, stream="scenario-seed") % (2**31 - 2)
    if task_id == "e07_s02_faults_1d":
        parameters["faultIndex"] = _counter_u64(task_id, ordinal, stream="fault-index") % len(parameters["values"])
    elif task_id == "e07_s02_chimera_1d":
        parameters["replicate"] = ordinal
    elif task_id == "e07_s02_regeneration_1d":
        parameters["replicateOrdinal"] = ordinal
    elif task_id.startswith("e07_s02_spatial2d_"):
        parameters["counterScheduleKey"] = f"{DERIVATION_DOMAIN}/{task_id}/{ordinal}"
    scenario_id = f"e07s07:{task_id.removeprefix('e07_s02_')}:train:{ordinal:03d}"
    return ScenarioRecord(
        scenario_id=scenario_id,
        task_id=task_id,
        split=Split.TRAIN,
        materializer_id=f"s07_train_family_{ordinal:03d}",
        public_parameters=parameters,
        protected=False,
        outcome_access="development",
        predecessor_partition=f"S07_outcome_blind_family_from_{base.scenario_id}",
    )


def _catalog() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = _read_jsonl(S05_DIR / "policy_catalog.jsonl")
    catalog = {row["policySha256"]: row for row in rows}
    faithful = {}
    for row in rows:
        if row["policyId"] == "bubble_cell_view_v1":
            faithful["Bubble"] = row["document"]
        elif row["policyId"] == "insertion_cell_view_v1":
            faithful["Insertion"] = row["document"]
    if set(faithful) != {"Bubble", "Insertion"}:
        raise RuntimeError("faithful E04 carrier documents are missing")
    return catalog, faithful


def _scenario_for(task_id: str, ordinal: int) -> tuple[ScenarioRecord, dict[str, Any]]:
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    base = suite.records[BASE_SCENARIOS[task_id]]
    record = derive_train_record(base, ordinal)
    commitment = canonical_hash(
        "E07/S07/materialized-scenario-commitment/v1", record.public_dict()
    )
    return record, {
        "scenarioId": record.scenario_id,
        "scenarioCommitmentSha256": commitment,
        "baseScenarioId": base.scenario_id,
        "derivationDomain": DERIVATION_DOMAIN,
        "parameters": dict(record.public_parameters),
    }


def _manifest_audit(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = []
    for spec in manifest["artifacts"]:
        path = directory / spec["path"]
        actual = hash_file(path)
        if actual != spec["sha256"] or path.stat().st_size != spec["bytes"]:
            raise RuntimeError(f"artifact manifest mismatch: {path}")
        checked.append({"path": str(path), "sha256": actual, "bytes": path.stat().st_size})
    return {
        "manifestPath": str(manifest_path),
        "manifestSha256": hash_file(manifest_path),
        "entryCount": len(checked),
        "allEntriesPass": True,
    }


def _input_gate(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if hash_file(REPOSITORY / "configs/allocation/s07r_non_surrogate_protocol.yaml") != PROTOCOL_SHA256:
        raise RuntimeError("frozen S07R protocol hash changed")
    if hash_file(S07R_DIR / "s07r_non_surrogate_protocol.yaml") != PROTOCOL_SHA256:
        raise RuntimeError("S07R artifact protocol hash changed")
    prereg = json.loads((S07R_DIR / "preregistration_freeze.json").read_text())
    candidates = build_candidate_registry(protocol)
    commitment = candidate_population_commitment(candidates)
    if commitment != CANDIDATE_COMMITMENT or prereg["candidatePopulationCommitmentSha256"] != commitment:
        raise RuntimeError("candidate commitment changed")
    s05_manifest = _manifest_audit(S05_DIR)
    s07r_manifest = _manifest_audit(S07R_DIR)
    for key in ("baseEvaluationLedger", "policyCatalog", "archiveEntries", "lineageLedger", "generationPlanLedger"):
        spec = protocol["authorizedPopulation"][key]
        if hash_file(spec["path"]) != spec["sha256"]:
            raise RuntimeError(f"authoritative S05 source changed: {key}")
    decisions = {
        "S06": json.loads(Path("/artifacts/research_steps/S06/deployment_decision.json").read_text()),
        "S06A": json.loads(Path("/artifacts/research_steps/S06A/deployment_decision.json").read_text()),
    }
    if any(value.get("eligibleForS07", value.get("deploymentEligible", False)) for value in decisions.values()):
        raise RuntimeError("a rejected modeling decision unexpectedly became eligible")
    forbidden_loaded = sorted(
        name for name in sys.modules
        if name.startswith(("src.surrogate_models", "src.surrogate_remediation"))
    )
    if forbidden_loaded:
        raise RuntimeError("rejected model code loaded in S07 process")
    return {
        "schemaVersion": "e07.s07.input-gate.v1",
        "researchStepId": "S07",
        "success": True,
        "checkedAtUtc": _utc_now(),
        "protocolSha256": PROTOCOL_SHA256,
        "executionLockSha256": hash_file(LOCK_PATH),
        "candidatePopulationCommitmentSha256": commitment,
        "S05Manifest": s05_manifest,
        "S07RManifest": s07r_manifest,
        "rejectedModelDecisionEligibility": {key: False for key in decisions},
        "rejectedModulesLoaded": [],
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
        "archiveMutations": 0,
    }


def _access_denial_audit() -> dict[str, Any]:
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    catalog, faithful = _catalog()
    bubble = next(row for row in catalog.values() if row["policyId"] == "bubble_cell_view_v1")
    denials = []
    before = suite.broker.audit.materializer_invocations
    for task_id in TASK_IDS:
        action = build_action(task_id, bubble["document"], faithful) if task_id != "e07_s02_chimera_1d" else build_action(task_id, bubble["document"], faithful)
        for split in (Split.VALIDATION, Split.CONFIRMATION):
            record = next(row for row in suite.records.values() if row.task_id == task_id and row.split is split)
            denied = False
            try:
                suite.open_for_action(task_id, record.scenario_id, AccessGrant(AccessPhase.DEVELOPMENT), action)
            except AccessDeniedError:
                denied = True
            if not denied:
                raise RuntimeError(f"protected-split denial failed: {record.scenario_id}")
            denials.append({"taskId": task_id, "scenarioId": record.scenario_id, "split": split.value, "denied": denied})
    delta = suite.broker.audit.materializer_invocations - before
    if delta != 0:
        raise RuntimeError("protected-split denial invoked a materializer")
    return {
        "schemaVersion": "e07.s07.access-denial-audit.v1",
        "researchStepId": "S07",
        "success": len(denials) == 16,
        "denials": denials,
        "materializerInvocationDelta": delta,
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
    }


def _plan_digest(frame: Sequence[Mapping[str, Any]], roster: Sequence[Mapping[str, Any]]) -> str:
    return canonical_hash(
        "E07/S07/approved-plan/v1",
        {
            "selectionFrame": sorted(row["selectionFrameRowId"] for row in frame),
            "logicalRoster": sorted(row["logicalRowId"] for row in roster),
        },
    )


def freeze_execution_plan(output_dir: Path = ARTIFACT_DIR) -> dict[str, Any]:
    """Revalidate every gate, generate the exact plans, and freeze hashes."""

    protocol = checked_protocol()
    gate = _input_gate(protocol)
    access = _access_denial_audit()
    candidates = build_candidate_registry(protocol)
    frame = build_candidate_selection_frame(candidates, protocol, approval_token=FUTURE_APPROVAL_TOKEN)
    roster = build_allocation_roster(candidates, protocol, approval_token=FUTURE_APPROVAL_TOKEN)
    protocol_hash = hash_file(REPOSITORY / "configs/allocation/s07r_non_surrogate_protocol.yaml")
    scenario_lookup = {}
    scenario_registry = []
    for task_id in TASK_IDS:
        for ordinal in FAMILY_ORDINALS:
            _, scenario = _scenario_for(task_id, ordinal)
            scenario_lookup[(task_id, ordinal)] = scenario
            scenario_registry.append({"taskId": task_id, "scenarioFamilyOrdinal": ordinal, "split": "train", "protected": False, **scenario})
    for row in frame:
        row["protocolSha256"] = protocol_hash
    for row in roster:
        row["protocolSha256"] = protocol_hash
        row["scenarioCommitmentSha256"] = scenario_lookup[(row["taskId"], row["scenarioFamilyOrdinal"])]["scenarioCommitmentSha256"]
        row["plannedWorkerId"] = int(row["physicalRowId"][:8], 16) % WORKERS
    smoke_ids = set()
    for arm_id in sorted(protocol["arms"]):
        for task_id in TASK_IDS:
            chosen = min(
                (row for row in roster if row["armId"] == arm_id and row["taskId"] == task_id and row["scenarioFamilyOrdinal"] == 200),
                key=lambda row: row["selectionRank"],
            )
            smoke_ids.add(chosen["logicalRowId"])
    if len(smoke_ids) != 24:
        raise RuntimeError("smoke logical roster is not 24 rows")
    for row in roster:
        row["smokeLogicalIndicator"] = row["logicalRowId"] in smoke_ids
    physical = {}
    for row in roster:
        item = physical.setdefault(
            row["physicalRowId"],
            {
                "physicalRowId": row["physicalRowId"], "taskId": row["taskId"],
                "policySha256": row["policySha256"],
                "scenarioFamilyOrdinal": row["scenarioFamilyOrdinal"],
                "scenarioCommitmentSha256": row["scenarioCommitmentSha256"],
                "plannedWorkerId": row["plannedWorkerId"], "logicalMultiplicity": 0,
                "overlapArmIds": row["overlapArmIds"], "smokeIndicator": False,
            },
        )
        item["logicalMultiplicity"] += 1
        item["smokeIndicator"] = item["smokeIndicator"] or row["smokeLogicalIndicator"]
    shuffled = list(candidates)
    random.Random(707702).shuffle(shuffled)
    frame_shuffled = build_candidate_selection_frame(shuffled, protocol, approval_token=FUTURE_APPROVAL_TOKEN)
    roster_shuffled = build_allocation_roster(shuffled, protocol, approval_token=FUTURE_APPROVAL_TOKEN)
    worker_order = {
        "schemaVersion": "e07.s07.worker-order-plan-validation.v1",
        "success": _plan_digest(frame, roster) == _plan_digest(frame_shuffled, roster_shuffled),
        "canonicalDigestSha256": _plan_digest(frame, roster),
        "shuffledDigestSha256": _plan_digest(frame_shuffled, roster_shuffled),
    }
    if not worker_order["success"]:
        raise RuntimeError("plan changed under candidate worker order")
    output_dir.mkdir(parents=True, exist_ok=True)
    _parquet_safe(frame).to_parquet(output_dir / "candidate_selection_frame.parquet", index=False)
    _parquet_safe(roster).to_parquet(output_dir / "allocation_logical_roster.parquet", index=False)
    physical_rows = sorted(physical.values(), key=lambda row: row["physicalRowId"])
    _parquet_safe(physical_rows).to_parquet(output_dir / "physical_evaluation_plan.parquet", index=False)
    _write_json(output_dir / "scenario_registry.json", {"schemaVersion": "e07.s07.scenario-registry.v1", "rows": scenario_registry})
    _write_json(output_dir / "gate_revalidation.json", gate)
    _write_json(output_dir / "access_control_validation.json", access)
    _write_json(output_dir / "worker_order_plan_validation.json", worker_order)
    freeze = {
        "schemaVersion": "e07.s07.execution-freeze.v1",
        "researchStepId": "S07", "success": True, "frozenAtUtc": _utc_now(),
        "approvedS07RProtocolSha256": protocol_hash,
        "executionLockSha256": hash_file(LOCK_PATH),
        "candidatePopulationCommitmentSha256": candidate_population_commitment(candidates),
        "candidateSelectionRows": len(frame), "logicalRosterRows": len(roster),
        "physicalEvaluationRows": len(physical_rows), "smokeLogicalRows": len(smoke_ids),
        "smokePhysicalRows": sum(row["smokeIndicator"] for row in physical_rows),
        "candidateSelectionFrameSha256": hash_file(output_dir / "candidate_selection_frame.parquet"),
        "logicalRosterSha256": hash_file(output_dir / "allocation_logical_roster.parquet"),
        "physicalEvaluationPlanSha256": hash_file(output_dir / "physical_evaluation_plan.parquet"),
        "scenarioRegistrySha256": hash_file(output_dir / "scenario_registry.json"),
        "planDigestSha256": worker_order["canonicalDigestSha256"],
        "S05ArchiveManifestSha256Before": hash_file(S05_DIR / "archive/archive_manifest.json"),
        "S05ArchiveEntriesSha256Before": hash_file(S05_DIR / "archive/archive_entries.jsonl"),
        "S05EvaluationLedgerSha256Before": hash_file(S05_DIR / "evaluation_ledger.jsonl"),
        "validationOutcomeEvaluations": 0, "confirmationOutcomeEvaluations": 0,
        "archiveMutations": 0,
    }
    if len(frame) != 1536 or len(roster) != 3072:
        raise RuntimeError("frozen plan row count mismatch")
    _write_json(output_dir / "execution_freeze.json", freeze)
    return freeze


def evaluate_physical_work(work: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one physical native row; suitable for spawn/fork workers."""

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    task_id = str(work["taskId"])
    ordinal = int(work["scenarioFamilyOrdinal"])
    document = dict(work["document"])
    faithful = {key: dict(value) for key, value in work["faithfulDocuments"].items()}
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    base = suite.records[BASE_SCENARIOS[task_id]]
    record = derive_train_record(base, ordinal)
    action = build_action(task_id, document, faithful)
    task = suite.tasks[task_id]
    started = time.perf_counter()
    result = RUNNERS[task.runner_id](record, action)
    elapsed = time.perf_counter() - started
    outcome, outcome_nonfinite = _json_safe(dict(result.native_outcome))
    costs, cost_nonfinite = _json_safe({key: dict(value) for key, value in result.native_costs.items()})
    event, event_nonfinite = _json_safe(dict(result.native_event))
    provenance, provenance_nonfinite = _json_safe(dict(result.provenance))
    stable = {
        "schemaVersion": "e07.s07.physical-evaluation-row.v1",
        "researchStepId": "S07", "physicalRowId": work["physicalRowId"],
        "taskId": task_id, "policyId": document["policyId"],
        "policySha256": compile_policy(document).policy_sha256,
        "policyBodySha256": policy_body_sha256(document),
        "scenarioId": record.scenario_id, "scenarioFamilyOrdinal": ordinal,
        "scenarioCommitmentSha256": canonical_hash("E07/S07/materialized-scenario-commitment/v1", record.public_dict()),
        "split": record.split.value, "protected": record.protected,
        "actionId": action.policy_id, "actionSha256": action.policy_sha256,
        "nativeUnit": task.horizon.native_unit, "nativeHorizon": task.horizon.to_dict(),
        "stopReason": result.stop_reason, "failed": bool(result.failed),
        "censored": bool(result.censored), "replayPass": bool(result.replay_pass),
        "validation": {key: bool(value) for key, value in result.validation.items()},
        "outcome": outcome, "nativeLedgerFamilies": costs, "nativeEvent": event,
        "provenance": provenance,
        "nativeNonFiniteValues": outcome_nonfinite + cost_nonfinite + event_nonfinite + provenance_nonfinite,
        "claimBoundary": task.claim_boundary,
        "licensedCapabilityCostsSeparate": {
            key: value for key, value in _numeric_leaves(costs).items()
            if "licensed" in key.lower() or "cursor" in key.lower() or "prefix" in key.lower()
        },
        "scenarioDerivation": {"baseScenarioId": base.scenario_id, "domain": DERIVATION_DOMAIN, "outcomeBlind": True, "publicParameters": dict(record.public_parameters)},
    }
    stable["semanticProjectionSha256"] = hashlib.sha256(
        b"E07/S07/semantic-evaluation/v1\x00" + _json_bytes(_strip_hash_fields({
            "scenarioFamilyOrdinal": ordinal, "stopReason": result.stop_reason,
            "failed": result.failed, "censored": result.censored, "outcome": outcome,
            "nativeLedgerFamilies": costs, "nativeEvent": event,
        }))
    ).hexdigest()
    stable["resultSha256"] = canonical_sha256("E07/S07/physical-evaluation/v1", stable)
    stable["elapsedSeconds"] = elapsed
    stable["workerPid"] = os.getpid()
    return stable


def validate_physical_result(row: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    errors = []
    if row.get("physicalRowId") != expected.get("physicalRowId"):
        errors.append("physical_row_id")
    if row.get("policySha256") != expected.get("policySha256"):
        errors.append("policy_hash")
    if row.get("scenarioCommitmentSha256") != expected.get("scenarioCommitmentSha256"):
        errors.append("scenario_commitment")
    if row.get("split") != "train" or row.get("protected"):
        errors.append("protected_split")
    if not row.get("replayPass"):
        errors.append("replay")
    validation = row.get("validation", {})
    false_flags = {key for key, value in validation.items() if not value}
    allowed = (
        row.get("taskId") == "e07_s02_regeneration_1d"
        and row.get("failed") and row.get("stopReason") == "source_terminal"
        and false_flags == {EXPECTED_E05_FAILURE_FLAG}
    )
    if false_flags and not allowed:
        errors.append("native_validation")
    if not row.get("stopReason") or not row.get("nativeLedgerFamilies") or not row.get("nativeEvent"):
        errors.append("native_contract_fields")
    stable = dict(row)
    result_hash = stable.pop("resultSha256", None)
    stable.pop("elapsedSeconds", None)
    stable.pop("workerPid", None)
    if canonical_sha256("E07/S07/physical-evaluation/v1", stable) != result_hash:
        errors.append("result_hash")
    return errors


def _read_physical_plan(output_dir: Path) -> list[dict[str, Any]]:
    return pd.read_parquet(output_dir / "physical_evaluation_plan.parquet").to_dict("records")


def _work_items(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    catalog, faithful = _catalog()
    items = []
    for row in rows:
        source = catalog[row["policySha256"]]
        items.append({**dict(row), "document": source["document"], "faithfulDocuments": faithful})
    return items


def _cache_path(physical_id: str) -> Path:
    return CACHE_DIR / f"{physical_id}.json"


def _evaluate_rows(rows: Sequence[Mapping[str, Any]], *, workers: int = WORKERS) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    expected = {row["physicalRowId"]: row for row in rows}
    results = []
    pending = []
    for work in _work_items(rows):
        cache_path = _cache_path(work["physicalRowId"])
        if cache_path.exists():
            row = json.loads(cache_path.read_text())
            errors = validate_physical_result(row, expected[row["physicalRowId"]])
            if errors:
                raise RuntimeError(f"invalid cached result {row['physicalRowId']}: {errors}")
            results.append(row)
        else:
            pending.append(work)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(evaluate_physical_work, work): work for work in pending}
        for future in as_completed(future_map):
            work = future_map[future]
            row = future.result()
            errors = validate_physical_result(row, expected[row["physicalRowId"]])
            if errors:
                raise RuntimeError(f"native integrity failure {row['physicalRowId']}: {errors}")
            _write_json(_cache_path(row["physicalRowId"]), row)
            results.append(row)
    return sorted(results, key=lambda row: row["physicalRowId"]), {
        "requestedPhysicalRows": len(rows), "cacheHits": len(rows) - len(pending),
        "executedPhysicalRows": len(pending), "workers": workers,
        "wallSeconds": time.perf_counter() - started,
    }


def run_smoke(output_dir: Path = ARTIFACT_DIR) -> dict[str, Any]:
    freeze = json.loads((output_dir / "execution_freeze.json").read_text())
    if hash_file(output_dir / "physical_evaluation_plan.parquet") != freeze["physicalEvaluationPlanSha256"]:
        raise RuntimeError("physical plan changed before smoke")
    rows = [row for row in _read_physical_plan(output_dir) if row["smokeIndicator"]]
    results, accounting = _evaluate_rows(rows)
    logical = pd.read_parquet(output_dir / "allocation_logical_roster.parquet")
    smoke_logical = logical[logical["smokeLogicalIndicator"]]
    errors = {row["physicalRowId"]: validate_physical_result(row, next(item for item in rows if item["physicalRowId"] == row["physicalRowId"])) for row in results}
    errors = {key: value for key, value in errors.items() if value}
    gate = {
        "schemaVersion": "e07.s07.smoke-gate.v1", "researchStepId": "S07",
        "success": not errors and len(smoke_logical) == 24 and len(results) == len(rows),
        "completedAtUtc": _utc_now(), "logicalRows": len(smoke_logical),
        "physicalRows": len(results), "integrityErrors": errors,
        "failedRowsRetained": sum(row["failed"] for row in results),
        "censoredRowsRetained": sum(row["censored"] for row in results),
        "allReplayPass": all(row["replayPass"] for row in results),
        "accounting": accounting, "withinFrozenBudget": True,
        "substantiveExecutionAuthorized": not errors and len(smoke_logical) == 24,
    }
    _write_jsonl(output_dir / "smoke_physical_results.jsonl", results)
    _write_json(output_dir / "smoke_gate.json", gate)
    if not gate["success"]:
        raise RuntimeError("S07 smoke gate failed")
    return gate


def run_substantive(output_dir: Path = ARTIFACT_DIR) -> dict[str, Any]:
    smoke = json.loads((output_dir / "smoke_gate.json").read_text())
    if not smoke.get("success") or not smoke.get("substantiveExecutionAuthorized"):
        raise RuntimeError("substantive S07 requires a passing smoke gate")
    freeze = json.loads((output_dir / "execution_freeze.json").read_text())
    for name, key in (
        ("candidate_selection_frame.parquet", "candidateSelectionFrameSha256"),
        ("allocation_logical_roster.parquet", "logicalRosterSha256"),
        ("physical_evaluation_plan.parquet", "physicalEvaluationPlanSha256"),
    ):
        if hash_file(output_dir / name) != freeze[key]:
            raise RuntimeError(f"frozen plan changed before substantive execution: {name}")
    rows = _read_physical_plan(output_dir)
    results, accounting = _evaluate_rows(rows)
    if len(results) != freeze["physicalEvaluationRows"]:
        raise RuntimeError("incomplete physical evaluation ledger")
    _write_jsonl(output_dir / "physical_evaluation_ledger.jsonl", results)
    complete = {
        "schemaVersion": "e07.s07.execution-accounting.v1", "researchStepId": "S07",
        "success": True, "completedAtUtc": _utc_now(),
        "logicalRows": freeze["logicalRosterRows"], "physicalRows": len(results),
        "logicalMinusPhysicalReuse": freeze["logicalRosterRows"] - len(results),
        "failedPhysicalRowsRetained": sum(row["failed"] for row in results),
        "censoredPhysicalRowsRetained": sum(row["censored"] for row in results),
        "replayPassRows": sum(row["replayPass"] for row in results),
        "execution": accounting,
        "physicalLedgerSha256": hash_file(output_dir / "physical_evaluation_ledger.jsonl"),
    }
    _write_json(output_dir / "execution_accounting.json", complete)
    return complete
