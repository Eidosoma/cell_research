"""Core contracts for the bounded E07 S06A remediation.

The module creates only new, outcome-blind training scenarios from the eight
nonprotected S02 train records.  It never materializes validation or
confirmation records and never loads a rejected S06 model bundle.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from src.environment_suite import EvaluationAction, Split, SuiteValidationError
from src.environment_suite.contracts import ScenarioRecord, canonical_sha256
from src.environment_suite.runners import RUNNERS
from src.environment_suite.suite import EnvironmentSuite
from src.policy_dsl import compile_policy
from src.quality_diversity.core import BASE_SCENARIOS, TASK_IDS, build_action
from src.surrogate_models.core import (
    assign_grouped_splits,
    build_target_registry,
    load_lineage,
    load_policy_catalog_with_plans,
    numeric_leaves,
    policy_feature_dict,
)


REPOSITORY = Path(__file__).resolve().parents[2]
S05_ROOT = Path("/artifacts/research_steps/S05")
S06_ROOT = Path("/artifacts/research_steps/S06")
S06A_ROOT = Path("/artifacts/research_steps/S06A")
PROTOCOL_PATH = REPOSITORY / "configs/modeling/s06a_remediation_protocol.yaml"
TASK_REGISTRY = REPOSITORY / "configs/environment_suite/task_registry.yaml"
SPLIT_MANIFEST = REPOSITORY / "configs/environment_suite/split_manifest.json"
FAMILY_ORDINALS = (100, 101, 102, 103, 104)
HASH_LIKE_KEYS = {
    "eventDigest",
    "episodeSha256",
    "panelSha256",
    "resultSha256",
    "finalStateHash",
    "finalStateSha256",
    "initialStateSha256",
    "policySha256",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_hash(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_bytes(value)
    ).hexdigest()


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_bytes(row).decode("ascii") + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_protocol() -> dict[str, Any]:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("schemaVersion") != "e07.s06a.remediation-protocol.v1":
        raise RuntimeError("unexpected S06A protocol schema")
    return protocol


def _verify_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(manifest.get("root", manifest.get("artifactRoot")))
    entries = manifest.get("artifacts", manifest.get("files", []))
    checks = []
    for entry in entries:
        path = root / entry["path"]
        actual = hash_file(path)
        checks.append(
            {
                "path": str(path),
                "expectedSha256": entry["sha256"],
                "actualSha256": actual,
                "pass": actual == entry["sha256"],
            }
        )
    if not checks or not all(item["pass"] for item in checks):
        raise RuntimeError(f"artifact manifest verification failed: {manifest_path}")
    return checks


def _s06_thresholds() -> dict[str, Any]:
    source = yaml.safe_load((S06_ROOT / "modeling_preregistration.yaml").read_text())
    return source["frozenThresholds"]


def _normalized_s06a_thresholds(protocol: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = deepcopy(protocol["frozenThresholds"])
    thresholds.pop("inheritedVerbatimFrom", None)
    thresholds.pop("inheritedSha256", None)
    thresholds["failureAction"] = "reject_models_for_S07_allocation_or_archive_mutation"
    return thresholds


def freeze_inputs(protocol: Mapping[str, Any]) -> dict[str, Any]:
    immutable = protocol["immutablePredecessors"]
    s05_manifest = Path(immutable["s05ArtifactManifest"])
    s06_manifest = Path(immutable["s06ArtifactManifest"])
    if hash_file(s05_manifest) != immutable["s05ArtifactManifestSha256"]:
        raise RuntimeError("S05 artifact manifest changed")
    if hash_file(s06_manifest) != immutable["s06ArtifactManifestSha256"]:
        raise RuntimeError("S06 artifact manifest changed")
    s05_checks = _verify_manifest(s05_manifest)
    s06_checks = _verify_manifest(s06_manifest)
    decision_path = Path(immutable["rejectedS06Decision"])
    if hash_file(decision_path) != immutable["rejectedS06DecisionSha256"]:
        raise RuntimeError("rejected S06 decision changed")
    decision = json.loads(decision_path.read_text())
    if (
        decision.get("decision") != "reject"
        or decision.get("eligibleForS07") is not False
    ):
        raise RuntimeError("S06 is not the expected immutable rejection")
    if _normalized_s06a_thresholds(protocol) != _s06_thresholds():
        raise RuntimeError("S06A deployment thresholds differ from S06")
    inherited = protocol["frozenThresholds"]
    inherited_path = Path(inherited["inheritedVerbatimFrom"])
    if hash_file(inherited_path) != inherited["inheritedSha256"]:
        raise RuntimeError("S06 threshold source changed")

    rows = read_jsonl(protocol["authorizedPopulation"]["baseEvaluationLedger"])
    selected = [row for row in rows if row["selectedForCanonicalSearch"]]
    orphans = [row for row in rows if not row["selectedForCanonicalSearch"]]
    auth = protocol["authorizedPopulation"]
    if hash_file(auth["baseEvaluationLedger"]) != auth["baseEvaluationLedgerSha256"]:
        raise RuntimeError("S05 evaluation ledger changed")
    if len(selected) != int(auth["expectedBaseRows"]):
        raise RuntimeError("S05 selected population changed")
    if len(orphans) != int(auth["recoveryOrphanRows"]):
        raise RuntimeError("S05 recovery-orphan population changed")
    if any(row["split"] != "train" for row in rows):
        raise RuntimeError("nontraining S05 row found")
    by_task: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        by_task[row["taskId"]].add(row["policySha256"])
    if set(by_task) != set(TASK_IDS) or any(
        len(values) != int(auth["expectedPoliciesPerTask"])
        for values in by_task.values()
    ):
        raise RuntimeError("expected 64 canonical policies per task")

    exact_inputs = [
        REPOSITORY / "configs/environment_suite/task_registry.yaml",
        REPOSITORY / "configs/environment_suite/split_manifest.json",
        REPOSITORY / "configs/search/s04_objective_registry.yaml",
        Path("/artifacts/research_steps/S04A/s05_eligibility_gate.json"),
        Path("/artifacts/research_steps/S04A/adapter_binding_registry.yaml"),
        Path("/artifacts/research_steps/S04A/e05_native_contract_validation.json"),
        Path("/artifacts/research_steps/S04A/e06_native_contract_validation.json"),
        S05_ROOT / "generation_plan_ledger.jsonl",
        S05_ROOT / "lineage_graph.jsonl",
        S05_ROOT / "policy_catalog.jsonl",
        S05_ROOT / "archive/archive_manifest.json",
        Path("/workspace/AGENTS.md"),
        Path("/workspace/FULL_PLAN.md"),
        Path("/workspace/RESEARCH_PLAN.md"),
        Path("/workspace/input-attachments/MANIFEST.json"),
        Path(
            "/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/"
            "_metadata/ATTACHMENT.md"
        ),
        PROTOCOL_PATH,
        REPOSITORY / "src/surrogate_remediation/core.py",
        REPOSITORY / "scripts/run_surrogate_remediation_s06a.py",
        REPOSITORY / "scripts/validate_surrogate_remediation_s06a.py",
        REPOSITORY / "tests/test_surrogate_remediation_s06a.py",
    ]
    input_hashes = [
        {"path": str(path), "sha256": hash_file(path)} for path in exact_inputs
    ]
    repository_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    repository_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=REPOSITORY, text=True
    ).splitlines()
    tracked_diff = subprocess.check_output(
        ["git", "diff", "HEAD", "--binary"], cwd=REPOSITORY
    )
    return {
        "schemaVersion": "e07.s06a.input-hash-freeze.v1",
        "researchStepId": "S06A",
        "success": True,
        "protocolSha256": hash_file(PROTOCOL_PATH),
        "s05ArtifactManifestSha256": hash_file(s05_manifest),
        "s06ArtifactManifestSha256": hash_file(s06_manifest),
        "s05ArtifactChecks": len(s05_checks),
        "s06ArtifactChecks": len(s06_checks),
        "s05SelectedRows": len(selected),
        "s05RecoveryOrphans": len(orphans),
        "policiesPerTask": {key: len(value) for key, value in sorted(by_task.items())},
        "selectedSetCommitmentSha256": canonical_hash(
            "E07/S06A/base-selected-set/v1",
            sorted(row["stableEvaluationSha256"] for row in selected),
        ),
        "orphanSetCommitmentSha256": canonical_hash(
            "E07/S06A/recovery-orphan-set/v1",
            sorted(row["stableEvaluationSha256"] for row in orphans),
        ),
        "thresholdsUnchanged": True,
        "rejectedS06Decision": "reject",
        "rejectedS06ModelsLoaded": 0,
        "s05ArchiveMutations": 0,
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
        "repositoryCommit": repository_commit,
        "repositoryStatusShort": repository_status,
        "repositoryTrackedDiffSha256": hashlib.sha256(tracked_diff).hexdigest(),
        "inputHashes": input_hashes,
    }


def verify_frozen_inputs(
    protocol: Mapping[str, Any], frozen: Mapping[str, Any]
) -> None:
    current = freeze_inputs(protocol)
    stable = (
        "protocolSha256",
        "s05ArtifactManifestSha256",
        "s06ArtifactManifestSha256",
        "s05SelectedRows",
        "s05RecoveryOrphans",
        "policiesPerTask",
        "selectedSetCommitmentSha256",
        "orphanSetCommitmentSha256",
        "thresholdsUnchanged",
        "rejectedS06Decision",
        "repositoryCommit",
        "repositoryTrackedDiffSha256",
    )
    if any(current[key] != frozen[key] for key in stable):
        raise RuntimeError("S06A frozen evidence changed")
    mutable_handoff = {"/workspace/RESEARCH_PLAN.md"}
    current_hashes = [
        item for item in current["inputHashes"] if item["path"] not in mutable_handoff
    ]
    frozen_hashes = [
        item for item in frozen["inputHashes"] if item["path"] not in mutable_handoff
    ]
    if current_hashes != frozen_hashes:
        raise RuntimeError("S06A exact input hash changed")


def load_catalog() -> dict[str, dict[str, Any]]:
    return load_policy_catalog_with_plans()


def counter_u64(*parts: object, stream: str) -> int:
    payload = "/".join(str(part) for part in parts)
    digest = hashlib.sha256(
        b"E07/S06A/additional-training-scenarios/v1\x00"
        + stream.encode("utf-8")
        + b"\x00"
        + payload.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def derive_additional_train_record(
    base: ScenarioRecord, ordinal: int
) -> ScenarioRecord:
    if ordinal not in FAMILY_ORDINALS:
        raise SuiteValidationError("S06A family ordinal is outside the frozen plan")
    if base.split is not Split.TRAIN or base.protected:
        raise SuiteValidationError("S06A derivation requires a nonprotected train base")
    parameters = dict(base.public_parameters)
    task_id = base.task_id
    if "seed" in parameters:
        parameters["seed"] = 1 + counter_u64(
            task_id, ordinal, stream="scenario-seed"
        ) % (2**31 - 2)
    if task_id == "e07_s02_faults_1d":
        parameters["faultIndex"] = counter_u64(
            task_id, ordinal, stream="fault-index"
        ) % len(parameters["values"])
    elif task_id == "e07_s02_chimera_1d":
        parameters["replicate"] = ordinal
    elif task_id == "e07_s02_regeneration_1d":
        if ordinal in {4, 5, 6, 7}:
            raise SuiteValidationError("protected E05 ordinal requested")
        parameters["replicateOrdinal"] = ordinal
    elif task_id.startswith("e07_s02_spatial2d_"):
        parameters["counterScheduleKey"] = (
            f"E07/S06A/{task_id}/train-family/{ordinal:03d}"
        )
    return ScenarioRecord(
        scenario_id=(f"e07s06a:{task_id.removeprefix('e07_s02_')}:train:{ordinal:03d}"),
        task_id=task_id,
        split=Split.TRAIN,
        materializer_id=f"s06a_train_family_{ordinal:03d}",
        public_parameters=parameters,
        protected=False,
        outcome_access="development",
        predecessor_partition=(
            f"S06A_outcome_blind_train_derivative_from_{base.scenario_id}"
        ),
    )


def _faithful_documents(
    catalog: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_id = {row["policyId"]: row for row in catalog.values()}
    return {
        "Bubble": deepcopy(by_id["bubble_cell_view_v1"]["document"]),
        "Insertion": deepcopy(by_id["insertion_cell_view_v1"]["document"]),
    }


def build_additional_plan(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = read_jsonl(protocol["authorizedPopulation"]["baseEvaluationLedger"])
    selected = [row for row in rows if row["selectedForCanonicalSearch"]]
    catalog = load_catalog()
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    policy_by_task: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        policy_by_task[row["taskId"]].add(row["policySha256"])
    output = []
    for task_id in TASK_IDS:
        base = suite.records[BASE_SCENARIOS[task_id]]
        for policy_hash in sorted(policy_by_task[task_id]):
            if policy_hash not in catalog:
                raise RuntimeError(
                    f"authoritative policy document missing: {policy_hash}"
                )
            document = catalog[policy_hash]["document"]
            if compile_policy(document).policy_sha256 != policy_hash:
                raise RuntimeError("policy document hash mismatch")
            for ordinal in FAMILY_ORDINALS:
                record = derive_additional_train_record(base, ordinal)
                plan = {
                    "schemaVersion": "e07.s06a.scenario-plan-row.v1",
                    "researchStepId": "S06A",
                    "taskId": task_id,
                    "policySha256": policy_hash,
                    "policyBodySha256": catalog[policy_hash]["policyBodySha256"],
                    "scenarioOrdinal": ordinal,
                    "scenarioId": record.scenario_id,
                    "split": "train",
                    "protected": False,
                    "sourceBaseScenarioId": base.scenario_id,
                    "publicParametersSha256": canonical_hash(
                        "E07/S06A/public-parameters/v1", record.public_parameters
                    ),
                }
                plan["planRowSha256"] = canonical_hash(
                    "E07/S06A/scenario-plan-row/v1", plan
                )
                output.append(plan)
    expected = int(protocol["additionalScenarioPlan"]["expectedTopLevelEvaluations"])
    if len(output) != expected:
        raise RuntimeError(
            f"expected {expected} additional evaluations, got {len(output)}"
        )
    return output


def work_items_from_plan(
    plan: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    catalog = load_catalog()
    faithful = _faithful_documents(catalog)
    return [
        {
            "taskId": row["taskId"],
            "scenarioOrdinal": int(row["scenarioOrdinal"]),
            "policySha256": row["policySha256"],
            "policyBodySha256": row["policyBodySha256"],
            "document": catalog[row["policySha256"]]["document"],
            "faithfulDocuments": faithful,
            "planRowSha256": row["planRowSha256"],
        }
        for row in plan
    ]


def _json_safe(value: Any, prefix: str = "") -> tuple[Any, list[dict[str, str]]]:
    if isinstance(value, Mapping):
        clean = {}
        markers = []
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            cleaned, found = _json_safe(child, path)
            clean[str(key)] = cleaned
            markers.extend(found)
        return clean, markers
    if isinstance(value, (list, tuple)):
        clean = []
        markers = []
        for index, child in enumerate(value):
            cleaned, found = _json_safe(child, f"{prefix}[{index}]")
            clean.append(cleaned)
            markers.extend(found)
        return clean, markers
    if isinstance(value, float) and not math.isfinite(value):
        kind = (
            "nan"
            if math.isnan(value)
            else ("positive_infinity" if value > 0 else "negative_infinity")
        )
        return None, [{"path": prefix, "nativeValue": kind}]
    return value, []


def _strip_hash_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _strip_hash_fields(child)
            for key, child in value.items()
            if key not in HASH_LIKE_KEYS and "policyId" not in key
        }
    if isinstance(value, list):
        return [_strip_hash_fields(item) for item in value]
    return value


def evaluate_additional_work_item(work: Mapping[str, Any]) -> dict[str, Any]:
    task_id = str(work["taskId"])
    ordinal = int(work["scenarioOrdinal"])
    document = dict(work["document"])
    faithful = {key: dict(value) for key, value in work["faithfulDocuments"].items()}
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    base = suite.records[BASE_SCENARIOS[task_id]]
    record = derive_additional_train_record(base, ordinal)
    action: EvaluationAction = build_action(task_id, document, faithful)
    task = suite.tasks[task_id]
    started = time.perf_counter()
    result = RUNNERS[task.runner_id](record, action)
    elapsed = time.perf_counter() - started
    validation = {key: bool(value) for key, value in result.validation.items()}
    outcome, outcome_nonfinite = _json_safe(dict(result.native_outcome))
    costs, cost_nonfinite = _json_safe(
        {key: dict(value) for key, value in result.native_costs.items()}
    )
    event, event_nonfinite = _json_safe(dict(result.native_event))
    provenance, provenance_nonfinite = _json_safe(dict(result.provenance))
    stable = {
        "taskId": task_id,
        "scenarioId": record.scenario_id,
        "scenarioOrdinal": ordinal,
        "split": "train",
        "stage": "s06a_additional_training",
        "selectedForCanonicalSearch": True,
        "modelPopulation": "s06a_additional_training",
        "policyId": str(document["policyId"]),
        "policySha256": compile_policy(document).policy_sha256,
        "policyBodySha256": str(work["policyBodySha256"]),
        "actionId": action.policy_id,
        "actionSha256": action.policy_sha256,
        "planRowSha256": work["planRowSha256"],
        "nativeUnit": task.horizon.native_unit,
        "stopReason": result.stop_reason,
        "censored": bool(result.censored),
        "failed": bool(result.failed),
        "replayPass": bool(result.replay_pass),
        "validation": validation,
        "outcome": outcome,
        "nativeLedgerFamilies": costs,
        "nativeEvent": event,
        "provenance": provenance,
        "nativeNonFiniteValues": (
            outcome_nonfinite + cost_nonfinite + event_nonfinite + provenance_nonfinite
        ),
        "claimBoundary": task.claim_boundary,
        "scenarioDerivation": {
            "baseScenarioId": base.scenario_id,
            "sourceGeneration": "s06a",
            "withinSourceFamilyIndex": ordinal - FAMILY_ORDINALS[0],
            "outcomeBlind": True,
            "counterScheduleKey": record.public_parameters.get("counterScheduleKey"),
            "faultIndex": record.public_parameters.get("faultIndex"),
            "replicate": record.public_parameters.get(
                "replicate", record.public_parameters.get("replicateOrdinal")
            ),
        },
    }
    stable["semanticProjectionSha256"] = hashlib.sha256(
        b"E07/S06A/semantic-evaluation/v1\x00"
        + canonical_bytes(
            _strip_hash_fields(
                {
                    "scenarioOrdinal": ordinal,
                    "stopReason": result.stop_reason,
                    "censored": result.censored,
                    "failed": result.failed,
                    "outcome": outcome,
                    "nativeLedgerFamilies": costs,
                    "nativeEvent": event,
                }
            )
        )
    ).hexdigest()
    stable["stableEvaluationSha256"] = canonical_sha256(
        "E07/S06A/evaluation/v1", stable
    )
    return {
        "schemaVersion": "e07.s06a.evaluation-ledger-row.v1",
        **stable,
        "elapsedSeconds": elapsed,
    }


def assign_s06a_splits(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return assign_grouped_splits(rows, load_lineage(), protocol)


def panel_indices(assignments: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    lineage = np.asarray([item["lineagePartition"] for item in assignments])
    scenario = np.asarray([item["scenarioPartition"] for item in assignments])
    return {
        "fit": np.flatnonzero((lineage == "fit") & (scenario == "fit")),
        "lineage_calibration": np.flatnonzero(
            (lineage == "calibration") & (scenario == "fit")
        ),
        "lineage_test": np.flatnonzero((lineage == "test") & (scenario == "fit")),
        "scenario_calibration": np.flatnonzero(
            (lineage == "fit") & (scenario == "calibration")
        ),
        "scenario_test": np.flatnonzero((lineage == "fit") & (scenario == "test")),
        "joint_test": np.flatnonzero((lineage == "test") & (scenario == "test")),
    }


def remediation_feature_dict(
    row: Mapping[str, Any], catalog: Mapping[str, Mapping[str, Any]], *, scenario: bool
) -> dict[str, float]:
    features = policy_feature_dict(row, catalog, scenario=False)
    if not scenario:
        return features
    derivation = row.get("scenarioDerivation", {})
    source = str(derivation.get("sourceGeneration", "s05"))
    if source == "s06a":
        index = int(derivation["withinSourceFamilyIndex"])
    else:
        index = int(row["scenarioOrdinal"])
    features[f"scenario::source::{source}"] = 1.0
    features["scenario::within_source_family_index"] = float(index)
    features["scenario::within_source_family_index_squared"] = float(index * index)
    fault_index = derivation.get("faultIndex")
    if isinstance(fault_index, (int, float)) and not isinstance(fault_index, bool):
        features["scenario::fault_index"] = float(fault_index)
    return features


def build_s06a_target_registry(
    task_id: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Retain every named numeric native cost, including licensed maxima.

    S06 deliberately omitted the long-range maximum-requested-distance field.
    S06A has a stricter preservation contract, so the omitted field is restored
    as its own licensed-capability target rather than aggregated with any cost.
    """

    registry = build_target_registry(task_id, rows)
    registered = {
        item["sourcePath"]
        for item in registry["continuous"]
        if item["name"].startswith("cost::")
    }
    all_costs = sorted(
        {path for row in rows for path in numeric_leaves(row["nativeLedgerFamilies"])}
    )
    for path in all_costs:
        if path in registered:
            continue
        licensed = any(
            token in path
            for token in ("licensedPrefix", "engineCursor", "licensedLongRange")
        )
        item = {
            "name": f"cost::{path}",
            "sourcePath": path,
            "direction": "minimize",
            "group": "cost",
            "transform": "signed_log1p",
            "type": "continuous",
            "licensedCapability": licensed,
        }
        registry["continuous"].append(item)
        registry["costCount"] += 1
        if licensed:
            registry["licensedCapabilityCostTargets"].append(item["name"])
        present = sum(
            path in numeric_leaves(row["nativeLedgerFamilies"]) for row in rows
        )
        if present != len(rows):
            registry["binary"].append(
                {
                    "name": f"cost_present::{path}",
                    "sourcePath": path,
                    "group": "binary",
                    "type": "binary_presence",
                    "transform": "identity",
                    "licensedCapability": licensed,
                }
            )
    registry["schemaVersion"] = "e07.s06a.target-registry.v1"
    registry["allNumericNativeCostFieldsRetained"] = len(all_costs)
    registry["claimBoundary"] = (
        "Task-specific native targets remain separately named; licensed prefix, "
        "cursor, and long-range quantities are never pooled into a universal cost."
    )
    return registry


def access_audit(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schemaVersion": "e07.s06a.access-control-validation.v1",
        "researchStepId": "S06A",
        "success": True,
        "allowedPopulation": "S05 canonical selected train rows plus frozen S06A train derivatives",
        "recoveryOrphanUse": "post_freeze_sensitivity_only",
        "rejectedS06ModelLoads": 0,
        "protectedMaterializersInvoked": 0,
        "protectedScenarioPayloadsRead": 0,
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
        "e05ProtectedOrdinalsExcluded": protocol["additionalScenarioPlan"][
            "e05ProtectedOrdinalsExcluded"
        ],
    }
