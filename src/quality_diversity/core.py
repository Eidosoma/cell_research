"""Core contracts for the frozen E07 S05 Pareto MAP-Elites search.

The module deliberately has no validation/confirmation outcome reader.  New
scenario rows are outcome-blind resamples of the eight frozen S02 training
records, and every policy still executes through the S04A native adapters.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import yaml

from src.environment_suite import EvaluationAction, Split, SuiteValidationError
from src.environment_suite.contracts import ScenarioRecord, canonical_sha256
from src.environment_suite.dsl_adapters import dsl_action
from src.environment_suite.runners import RUNNERS
from src.environment_suite.suite import EnvironmentSuite
from src.objective_design import require_s05_eligible
from src.policy_dsl import PolicyValidationError, compile_policy


REPOSITORY = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path("/artifacts")
TASK_REGISTRY = REPOSITORY / "configs/environment_suite/task_registry.yaml"
SPLIT_MANIFEST = REPOSITORY / "configs/environment_suite/split_manifest.json"
DESCRIPTOR_REGISTRY = REPOSITORY / "configs/search/s04_descriptor_registry.yaml"
SEED_LIBRARY = ARTIFACT_ROOT / "research_steps/S03/seed_library.jsonl"

TASK_IDS = (
    "e07_s02_sorting_1d",
    "e07_s02_faults_1d",
    "e07_s02_detour_1d",
    "e07_s02_chimera_1d",
    "e07_s02_regeneration_1d",
    "e07_s02_target_change_1d",
    "e07_s02_spatial2d_local",
    "e07_s02_spatial2d_memory",
)

BASE_SCENARIOS = {
    "e07_s02_sorting_1d": "e07s02:sorting:train:000",
    "e07_s02_faults_1d": "e07s02:faults:train:000",
    "e07_s02_detour_1d": "e07s02:detour:train:000",
    "e07_s02_chimera_1d": "e07s02:chimera:train:000",
    "e07_s02_regeneration_1d": "e07s02:regeneration:train:000",
    "e07_s02_target_change_1d": "e07s02:target-change:train:000",
    "e07_s02_spatial2d_local": "e07s02:spatial-local:train:000",
    "e07_s02_spatial2d_memory": "e07s02:spatial-memory:train:000",
}

COMPLEXITY_FIELDS = (
    "ruleCount",
    "expressionNodes",
    "actionCount",
    "persistentMemoryBits",
    "outboundSignalBits",
    "worstCaseOperations",
    "canonicalBytes",
)

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

E05_DESCRIPTOR_AVAILABILITY_VERSION = "e07.s08f.e05-descriptor-availability.v1"
E05_REGENERATION_DESCRIPTOR_SPECS: dict[str, dict[str, Any]] = {
    **{
        f"common:e07_s02_regeneration_1d:{phase}": {
            "kind": "common",
            "fields": (
                "acceptedNativeActionFraction",
                "committedDisplacementFraction",
            ),
            "censorFields": (),
        }
        for phase in (
            "development",
            "stabilization",
            "recovery",
            "memoryResetRecovery",
            "robustnessFault",
            "transfer",
        )
    },
    "phenotype:e05:robustness": {
        "kind": "e05:robustness",
        "fields": ("pairedCompletionDelta", "pairedResidualDelta"),
        "censorFields": (),
    },
    "phenotype:e05:repair": {
        "kind": "e05:repair",
        "fields": (
            "distanceRestorationFraction",
            "restrictedRecoveryTimeFraction",
        ),
        "censorFields": ("recoveryCensored",),
    },
    "phenotype:e05:memory": {
        "kind": "e05:memory",
        "fields": ("historyInterventionEffect", "resetInterventionEffect"),
        "censorFields": (),
    },
    "phenotype:e05:transfer": {
        "kind": "e05:transfer",
        "fields": ("frozenStratumSuccessFraction", "frozenStratumResidual"),
        "censorFields": (),
    },
}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_safe(value: Any, prefix: str = "") -> tuple[Any, list[dict[str, str]]]:
    """Encode native undefined numeric diagnostics without losing provenance."""

    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        markers: list[dict[str, str]] = []
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            cleaned, found = _json_safe(child, path)
            clean[str(key)] = cleaned
            markers.extend(found)
        return clean, markers
    if isinstance(value, (list, tuple)):
        clean_items = []
        markers = []
        for index, child in enumerate(value):
            cleaned, found = _json_safe(child, f"{prefix}[{index}]")
            clean_items.append(cleaned)
            markers.extend(found)
        return clean_items, markers
    if isinstance(value, float) and not math.isfinite(value):
        kind = (
            "nan"
            if math.isnan(value)
            else ("positive_infinity" if value > 0 else "negative_infinity")
        )
        return None, [{"path": prefix, "nativeValue": kind}]
    return value, []


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def counter_u64(*parts: object, stream: str = "default") -> int:
    """Domain-separated counter derivation independent of process order."""

    payload = "/".join(str(item) for item in parts)
    digest = hashlib.sha256(
        b"E07/S05/quality-diversity/v1\x00"
        + stream.encode("utf-8")
        + b"\x00"
        + payload.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPOSITORY / path


def validate_gate_and_inputs(protocol_path: str | Path) -> dict[str, Any]:
    """Fail closed on any frozen hash, gate row, or evidence change."""

    protocol_path = Path(protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schemaVersion") != "e07.s05.qd-protocol.v1":
        raise RuntimeError("unexpected S05 protocol schema")
    checked: list[dict[str, Any]] = []

    for item in protocol["frozenInputs"].values():
        path = _resolve(str(item["path"]))
        actual = _sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"frozen input changed: {path}")
        checked.append({"path": str(path), "sha256": actual})

    eligibility = protocol["eligibility"]
    for path_key, hash_key in (
        ("liveGatePath", "liveGateSha256"),
        ("evidenceGatePath", "evidenceGateSha256"),
        ("evidenceManifestPath", "evidenceManifestSha256"),
    ):
        path = _resolve(str(eligibility[path_key]))
        actual = _sha256_file(path)
        if actual != eligibility[hash_key]:
            raise RuntimeError(f"eligibility input changed: {path}")
        checked.append({"path": str(path), "sha256": actual})

    live_path = _resolve(str(eligibility["liveGatePath"]))
    evidence_path = _resolve(str(eligibility["evidenceGatePath"]))
    live = yaml.safe_load(live_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    require_s05_eligible(live)
    require_s05_eligible(evidence)
    expected_rows = set(eligibility["requiredRows"])
    live_rows = dict(live.get("computedRequirementStatus", {}))
    evidence_rows = dict(evidence.get("computedRequirementStatus", {}))
    if set(live_rows) != expected_rows or live_rows != evidence_rows:
        raise RuntimeError("live and evidence eligibility rows disagree")
    if any(value != "pass" for value in live_rows.values()):
        raise RuntimeError("at least one S05 eligibility row is not pass")

    manifest_path = _resolve(str(eligibility["evidenceManifestPath"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_root = Path(manifest["root"])
    manifest_checks = []
    for item in manifest["artifacts"]:
        path = manifest_root / item["path"]
        actual = _sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"S04A evidence artifact changed: {path}")
        manifest_checks.append({"path": str(path), "sha256": actual})

    leakage = json.loads(
        (manifest_root / "leakage_replay_validation.json").read_text(encoding="utf-8")
    )
    if not leakage.get("success") or not all(leakage["checks"].values()):
        raise RuntimeError("S04A leakage evidence no longer passes")
    if Path("/artifacts/research_steps/S06").exists():
        raise RuntimeError("S06 already exists; S05 refuses to run out of order")

    return {
        "schemaVersion": "e07.s05.gate-revalidation.v1",
        "researchStepId": "S05",
        "success": True,
        "protocolSha256": _sha256_file(protocol_path),
        "gateSchemaVersion": live["schemaVersion"],
        "gateDecision": live["decision"],
        "gateRows": live_rows,
        "frozenInputsChecked": checked,
        "evidenceArtifactsChecked": len(manifest_checks),
        "evidenceArtifacts": manifest_checks,
        "protectedRequestsPreviouslyDenied": len(leakage["rows"]),
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
        "s06DirectoryAbsent": True,
    }


def load_seed_records(path: str | Path = SEED_LIBRARY) -> list[dict[str, Any]]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            compiled = compile_policy(item["canonicalPolicy"])
            if compiled.policy_sha256 != item["policySha256"]:
                raise RuntimeError(f"seed hash mismatch: {item['policyId']}")
            records.append(item)
    if len(records) != 30:
        raise RuntimeError("S05 requires the complete 30-policy S03 seed library")
    return records


def policy_body_sha256(document: Mapping[str, Any]) -> str:
    body = deepcopy(dict(document))
    body.pop("policyId", None)
    return hashlib.sha256(b"E07/S05/policy-body/v1\x00" + _json_bytes(body)).hexdigest()


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _choose(values: Sequence[Any], address: int) -> Any | None:
    return None if not values else values[address % len(values)]


def mutate_policy(
    parent_document: Mapping[str, Any],
    operator_id: str,
    *,
    address: int,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Apply one frozen, permission-nonexpanding mutation and compile it."""

    document = deepcopy(dict(parent_document))
    before_permissions = tuple(document["permissions"])
    detail: dict[str, Any] = {"operatorId": operator_id}
    changed = False

    if operator_id == "delete_rule":
        if len(document["rules"]) > 1:
            index = address % len(document["rules"])
            document["rules"].pop(index)
            detail["ruleIndex"] = index
            changed = True
    elif operator_id == "rotate_rule_order":
        if len(document["rules"]) > 1:
            shift = 1 + address % (len(document["rules"]) - 1)
            document["rules"] = document["rules"][shift:] + document["rules"][:shift]
            detail["shift"] = shift
            changed = True
    elif operator_id == "duplicate_rule":
        if (
            document["rules"]
            and len(document["rules"]) < document["limits"]["maxRules"]
        ):
            index = address % len(document["rules"])
            document["rules"].insert(index + 1, deepcopy(document["rules"][index]))
            detail["ruleIndex"] = index
            changed = True
    elif operator_id == "flip_comparison":
        comparisons = [
            item
            for item in _walk_dicts(document["rules"])
            if item.get("op") in {"eq", "ne", "lt", "le", "gt", "ge"}
        ]
        target = _choose(comparisons, address)
        if target is not None:
            mapping = {
                "eq": "ne",
                "ne": "eq",
                "lt": "gt",
                "gt": "lt",
                "le": "ge",
                "ge": "le",
            }
            detail.update({"from": target["op"], "to": mapping[target["op"]]})
            target["op"] = mapping[target["op"]]
            changed = True
    elif operator_id == "nudge_literal":
        literals = [
            item
            for item in _walk_dicts(document["rules"])
            if set(item) == {"const"} and isinstance(item["const"], (bool, int))
        ]
        target = _choose(literals, address)
        if target is not None:
            old = target["const"]
            if isinstance(old, bool):
                target["const"] = not old
            else:
                delta = -1 if (address // max(1, len(literals))) % 2 else 1
                target["const"] = max(-(2**31), min(2**31 - 1, old + delta))
            detail.update({"from": old, "to": target["const"]})
            changed = target["const"] != old
    elif operator_id == "terminal_noop":
        terminals = [
            item
            for item in _walk_dicts(
                {"rules": document["rules"], "default": document["default"]}
            )
            if item.get("kind")
            in {"swap_relative", "swap_cursor", "advance_cursor", "move_candidate"}
        ]
        target = _choose(terminals, address)
        if target is not None:
            detail["from"] = target["kind"]
            target.clear()
            target["kind"] = "noop"
            changed = True
    elif operator_id == "flip_relative_offset":
        actions = [
            item
            for item in _walk_dicts(
                {"rules": document["rules"], "default": document["default"]}
            )
            if item.get("kind") == "swap_relative"
        ]
        target = _choose(actions, address)
        if target is not None:
            old = int(target["offset"])
            target["offset"] = -old
            detail.update({"from": old, "to": -old})
            changed = old != 0
    elif operator_id == "flip_candidate_selector":
        actions = [
            item
            for item in _walk_dicts(
                {"rules": document["rules"], "default": document["default"]}
            )
            if item.get("kind") == "move_candidate"
            and item["selector"]["mode"] in {"argmax", "argmin"}
        ]
        target = _choose(actions, address)
        if target is not None:
            old = target["selector"]["mode"]
            target["selector"]["mode"] = "argmin" if old == "argmax" else "argmax"
            detail.update({"from": old, "to": target["selector"]["mode"]})
            changed = True
    elif operator_id == "toggle_candidate_positive_requirement":
        actions = [
            item
            for item in _walk_dicts(
                {"rules": document["rules"], "default": document["default"]}
            )
            if item.get("kind") == "move_candidate"
            and item["selector"]["mode"] in {"argmax", "argmin"}
        ]
        target = _choose(actions, address)
        if target is not None:
            old = bool(target["selector"]["requirePositive"])
            target["selector"]["requirePositive"] = not old
            detail.update({"from": old, "to": not old})
            changed = True
    elif operator_id == "drop_movement_kind":
        actions = [
            item
            for item in _walk_dicts(
                {"rules": document["rules"], "default": document["default"]}
            )
            if item.get("kind") == "move_candidate"
            and len(item["allowedMovementKinds"]) > 1
        ]
        target = _choose(actions, address)
        if target is not None:
            kinds = target["allowedMovementKinds"]
            index = address % len(kinds)
            detail["removed"] = kinds[index]
            kinds.pop(index)
            changed = True
    elif operator_id == "nudge_memory_initial":
        target = _choose(document["memory"], address)
        if target is not None:
            old = int(target["initial"])
            maximum = (1 << int(target["bits"])) - 1
            delta = -1 if (address // max(1, len(document["memory"]))) % 2 else 1
            target["initial"] = max(0, min(maximum, old + delta))
            detail.update(
                {"register": target["name"], "from": old, "to": target["initial"]}
            )
            changed = old != target["initial"]
    else:
        raise ValueError(f"unknown mutation operator {operator_id!r}")

    if not changed or tuple(document["permissions"]) != before_permissions:
        return None
    body_hash = policy_body_sha256(document)
    profile = "line" if document["environment"] == "line1d.v1" else "spatial"
    document["policyId"] = f"s05_{profile}_{body_hash[:20]}"
    try:
        compiled = compile_policy(document)
    except (PolicyValidationError, SuiteValidationError, ValueError):
        return None
    detail["offspringBodySha256"] = body_hash
    detail["offspringPolicySha256"] = compiled.policy_sha256
    return document, detail


def _line_carrier(document: Mapping[str, Any]) -> str:
    permissions = set(document["permissions"])
    actions = [
        item
        for item in _walk_dicts(
            {"rules": document["rules"], "default": document["default"]}
        )
        if "kind" in item
    ]
    if permissions & {
        "selection.cursor_in_bounds",
        "selection.cursor_at_actor",
        "selection.target.value",
        "selection.target.stuck",
    } or any(item["kind"] in {"swap_cursor", "advance_cursor"} for item in actions):
        return "Selection"
    if "activation.side" in permissions:
        return "Bubble"
    return "Insertion"


def compatible_with_task(task_id: str, document: Mapping[str, Any]) -> bool:
    environment = document["environment"]
    if task_id.startswith("e07_s02_spatial2d_"):
        return environment == "spatial2d.v1"
    if environment != "line1d.v1":
        return False
    if task_id == "e07_s02_chimera_1d":
        return _line_carrier(document) in {"Bubble", "Insertion"}
    return True


def build_action(
    task_id: str,
    document: Mapping[str, Any],
    faithful_documents: Mapping[str, Mapping[str, Any]],
) -> EvaluationAction:
    if task_id != "e07_s02_chimera_1d":
        return dsl_action([document])
    carrier = _line_carrier(document)
    if carrier == "Bubble":
        documents = [document, faithful_documents["Insertion"]]
        bindings = {
            "Bubble": str(document["policyId"]),
            "Insertion": str(faithful_documents["Insertion"]["policyId"]),
        }
    elif carrier == "Insertion":
        documents = [faithful_documents["Bubble"], document]
        bindings = {
            "Bubble": str(faithful_documents["Bubble"]["policyId"]),
            "Insertion": str(document["policyId"]),
        }
    else:
        raise SuiteValidationError("Selection-carrier policies are not E04-compatible")
    return dsl_action(documents, native_policy_bindings=bindings)


def derive_train_record(base: ScenarioRecord, ordinal: int) -> ScenarioRecord:
    """Outcome-blind training resample that never consults another split."""

    if base.split is not Split.TRAIN or base.protected:
        raise SuiteValidationError("S05 resampling requires a nonprotected train base")
    task_id = base.task_id
    parameters = dict(base.public_parameters)
    if "seed" in parameters:
        parameters["seed"] = 1 + counter_u64(
            task_id, ordinal, stream="scenario-seed"
        ) % (2**31 - 2)
    if task_id == "e07_s02_faults_1d":
        parameters["faultIndex"] = counter_u64(
            task_id, ordinal, stream="fault-index"
        ) % len(parameters["values"])
    elif task_id == "e07_s02_chimera_1d":
        parameters["replicate"] = int(ordinal)
    elif task_id == "e07_s02_regeneration_1d":
        if ordinal not in {0, 1, 2, 3}:
            raise SuiteValidationError(
                "E05 protected replicate ordinals may not be used"
            )
        parameters["replicateOrdinal"] = int(ordinal)
    elif task_id.startswith("e07_s02_spatial2d_"):
        parameters["counterScheduleKey"] = (
            f"E07/S05/{task_id}/train-resample/{ordinal:03d}"
        )
    scenario_id = f"e07s05:{task_id.removeprefix('e07_s02_')}:train:{ordinal:03d}"
    return ScenarioRecord(
        scenario_id=scenario_id,
        task_id=task_id,
        split=Split.TRAIN,
        materializer_id=f"s05_train_resample_{ordinal:03d}",
        public_parameters=parameters,
        protected=False,
        outcome_access="development",
        predecessor_partition=(f"S05_outcome_blind_resample_from_{base.scenario_id}"),
    )


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


def evaluate_work_item(work: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one complete native train episode; safe for process workers."""

    task_id = str(work["taskId"])
    ordinal = int(work["scenarioOrdinal"])
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
    validation = {key: bool(value) for key, value in result.validation.items()}
    outcome, outcome_nonfinite = _json_safe(dict(result.native_outcome))
    native_costs, cost_nonfinite = _json_safe(
        {key: dict(value) for key, value in result.native_costs.items()}
    )
    native_event, event_nonfinite = _json_safe(dict(result.native_event))
    provenance, provenance_nonfinite = _json_safe(dict(result.provenance))
    stable = {
        "taskId": task_id,
        "scenarioId": record.scenario_id,
        "scenarioOrdinal": ordinal,
        "split": record.split.value,
        "stage": str(work["stage"]),
        "policyId": str(document["policyId"]),
        "policySha256": compile_policy(document).policy_sha256,
        "policyBodySha256": policy_body_sha256(document),
        "actionId": action.policy_id,
        "actionSha256": action.policy_sha256,
        "nativeUnit": task.horizon.native_unit,
        "stopReason": result.stop_reason,
        "censored": bool(result.censored),
        "failed": bool(result.failed),
        "replayPass": bool(result.replay_pass),
        "validation": validation,
        "outcome": outcome,
        "nativeLedgerFamilies": native_costs,
        "nativeEvent": native_event,
        "provenance": provenance,
        "nativeNonFiniteValues": (
            outcome_nonfinite + cost_nonfinite + event_nonfinite + provenance_nonfinite
        ),
        "claimBoundary": task.claim_boundary,
        "scenarioDerivation": {
            "baseScenarioId": base.scenario_id,
            "outcomeBlind": True,
            "counterScheduleKey": record.public_parameters.get("counterScheduleKey"),
            "seed": record.public_parameters.get("seed"),
            "faultIndex": record.public_parameters.get("faultIndex"),
            "replicate": record.public_parameters.get(
                "replicate", record.public_parameters.get("replicateOrdinal")
            ),
        },
    }
    stable["semanticProjectionSha256"] = hashlib.sha256(
        b"E07/S05/semantic-evaluation/v1\x00"
        + _json_bytes(
            _strip_hash_fields(
                {
                    "scenarioOrdinal": ordinal,
                    "stopReason": result.stop_reason,
                    "censored": result.censored,
                    "failed": result.failed,
                    "outcome": outcome,
                    "nativeLedgerFamilies": native_costs,
                    "nativeEvent": native_event,
                }
            )
        )
    ).hexdigest()
    stable["stableEvaluationSha256"] = canonical_sha256("E07/S05/evaluation/v1", stable)
    return {
        "schemaVersion": "e07.s05.evaluation-ledger-row.v1",
        **stable,
        "elapsedSeconds": elapsed,
    }


def _get_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        current = current[part]
    return current


OBJECTIVES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "e07_s02_sorting_1d": (
        ("sorting_completion", "completed", "maximize"),
        (
            "sorting_terminal_normalized_kendall",
            "derived.finalNormalizedKendallDistance",
            "minimize",
        ),
    ),
    "e07_s02_faults_1d": (
        ("fault_completion", "completed", "maximize"),
        (
            "fault_terminal_normalized_kendall",
            "derived.finalNormalizedKendallDistance",
            "minimize",
        ),
        (
            "paired_fault_completion_delta",
            "paired.completedFaultMinusMatchedClean",
            "maximize",
        ),
        (
            "paired_fault_residual_delta",
            "paired.residualFaultMinusMatchedClean",
            "minimize",
        ),
    ),
    "e07_s02_detour_1d": (
        ("detour_completion", "completed", "maximize"),
        (
            "adjacent_descents_final",
            "finalDistanceProfile.normalized_adjacent_descents",
            "minimize",
        ),
        (
            "inversion_final",
            "finalDistanceProfile.normalized_kendall_distance",
            "minimize",
        ),
        (
            "footrule_final",
            "finalDistanceProfile.normalized_spearman_footrule",
            "minimize",
        ),
        (
            "max_rank_error_final",
            "finalDistanceProfile.normalized_maximum_rank_error",
            "minimize",
        ),
        (
            "duplicate_transport_final",
            "finalDistanceProfile.normalized_duplicate_aware_earth_movers_distance",
            "minimize",
        ),
    ),
    "e07_s02_chimera_1d": (
        ("chimera_completion", "completed", "maximize"),
        (
            "corrected_null_adjusted_peak",
            "dynamicNull.compositionCorrectedPeakExcess",
            "maximize",
        ),
        (
            "corrected_null_adjusted_positive_area",
            "dynamicNull.fixedHorizonPositiveAreaMean",
            "maximize",
        ),
    ),
    "e07_s02_regeneration_1d": (
        ("regeneration_success", "success", "maximize"),
        ("regeneration_restricted_time", "restrictedTime", "minimize"),
        ("regeneration_final_distance", "finalDistance", "minimize"),
    ),
    "e07_s02_target_change_1d": (
        ("target_completion", "targetCompleted", "maximize"),
        ("restricted_adaptation_time", "restrictedAdaptationTime", "minimize"),
        ("post_hit_departure", "postHitAnyDeparture", "minimize"),
    ),
    "e07_s02_spatial2d_local": (
        ("conjunctive_completion", "conjunctiveCompletion", "maximize"),
        (
            "s01_global_mismatch",
            "s01GlobalCompletionAudit.normalizedMismatch",
            "minimize",
        ),
        ("s02_local_acceptance", "s02LocalGrammar.accepted", "maximize"),
        ("repair_success", "repairCompletion", "maximize"),
    ),
    "e07_s02_spatial2d_memory": (
        ("conjunctive_completion", "conjunctiveCompletion", "maximize"),
        (
            "s01_global_mismatch",
            "s01GlobalCompletionAudit.normalizedMismatch",
            "minimize",
        ),
        ("s02_local_acceptance", "s02LocalGrammar.accepted", "maximize"),
        ("repair_success", "repairCompletion", "maximize"),
    ),
}


def _numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    leaves: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_numeric_leaves(child, path))
    elif isinstance(value, bool):
        leaves[prefix] = float(value)
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        leaves[prefix] = float(value)
    return leaves


@lru_cache(maxsize=1)
def _descriptor_edges() -> dict[str, Any]:
    registry = yaml.safe_load(DESCRIPTOR_REGISTRY.read_text(encoding="utf-8"))
    common = {
        item["id"]: tuple(float(value) for value in item["binEdges"])
        for item in registry["commonArchive"]["dimensions"]
    }
    phenotype = registry["taskPhenotypeArchives"]
    return {"common": common, "phenotype": phenotype}


def _descriptor_sets(task_id: str, outcome: Mapping[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    if task_id == "e07_s02_regeneration_1d":
        for phase, item in outcome["nativeMovementDescriptorsByPhase"].items():
            values.append(
                {
                    "archiveId": f"common:{task_id}:{phase}",
                    "values": dict(item),
                    "kind": "common",
                }
            )
    else:
        native_descriptors = (
            outcome["nativeMovementDescriptors"]
            if task_id == "e07_s02_target_change_1d"
            else outcome["descriptors"]
        )
        common = {
            key: native_descriptors[key]
            for key in (
                "acceptedNativeActionFraction",
                "committedDisplacementFraction",
            )
        }
        values.append(
            {"archiveId": f"common:{task_id}", "values": dict(common), "kind": "common"}
        )

    if task_id == "e07_s02_faults_1d":
        values.append(
            {
                "archiveId": "phenotype:fault:paired",
                "kind": "fault",
                "values": {
                    "paired_completion_delta": outcome["paired"][
                        "completedFaultMinusMatchedClean"
                    ],
                    "paired_residual_delta": outcome["paired"][
                        "residualFaultMinusMatchedClean"
                    ],
                },
            }
        )
    elif task_id == "e07_s02_detour_1d":
        for metric, item in outcome["metricSpecificExcursions"].items():
            values.append(
                {
                    "archiveId": f"phenotype:detour:{metric}",
                    "kind": "detour",
                    "values": {
                        "maximum_positive_excursion_fraction": item[
                            "maximumPositiveExcursionFraction"
                        ],
                        "worsening_event_fraction": item["worseningEventFraction"],
                    },
                }
            )
    elif task_id == "e07_s02_chimera_1d":
        item = outcome["dynamicNull"]
        values.append(
            {
                "archiveId": "phenotype:chimera:corrected",
                "kind": "chimera",
                "values": {
                    "composition_corrected_null_excess_peak": item[
                        "compositionCorrectedPeakExcess"
                    ],
                    "fixed_horizon_positive_null_excess_area_mean": item[
                        "fixedHorizonPositiveAreaMean"
                    ],
                },
            }
        )
    elif task_id == "e07_s02_regeneration_1d":
        for axis, item in outcome["descriptorsByAxis"].items():
            values.append(
                {
                    "archiveId": f"phenotype:e05:{axis}",
                    "kind": f"e05:{axis}",
                    "values": dict(item),
                }
            )
    elif task_id == "e07_s02_target_change_1d":
        item = outcome["descriptorsByAxis"]["plasticity_target_adaptation"]
        values.append(
            {
                "archiveId": "phenotype:e05:plasticity_target_adaptation",
                "kind": "e05:plasticity_target_adaptation",
                "values": dict(item),
            }
        )
    elif task_id.startswith("e07_s02_spatial2d_"):
        item = outcome["descriptors"]
        values.append(
            {
                "archiveId": f"phenotype:e06:{task_id}",
                "kind": "e06",
                "values": {
                    "committed_movement_kind_entropy": item[
                        "committedMovementKindEntropy"
                    ],
                    "state_turnover_fraction": item["stateTurnoverFraction"],
                },
            }
        )
    return values


def _edges_for(kind: str, fields: Sequence[str]) -> dict[str, tuple[float, ...]]:
    registry = _descriptor_edges()
    if kind == "common":
        aliases = {
            "acceptedNativeActionFraction": "accepted_native_action_fraction",
            "committedDisplacementFraction": "committed_displacement_fraction",
        }
        return {field: registry["common"][aliases[field]] for field in fields}
    phenotype = registry["phenotype"]
    if kind == "fault":
        rows = phenotype["e07_s02_faults_1d"]["dimensions"]
    elif kind == "detour":
        rows = phenotype["e07_s02_detour_1d"]["dimensions"]
    elif kind == "chimera":
        rows = phenotype["e07_s02_chimera_1d"]["dimensions"]
    elif kind == "e06":
        rows = phenotype["e06_bounded_square"]["dimensions"]
    else:
        # S04 freezes E05 field names/bounds but not explicit bin edges.  Use
        # the common ten-bin numeric contract for bounded fractions/signed
        # effects, retaining censor flags outside the coordinate.
        signed = {
            "pairedCompletionDelta",
            "pairedResidualDelta",
            "historyInterventionEffect",
            "resetInterventionEffect",
        }
        return {
            field: (
                (-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0)
                if field in signed
                else (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
            )
            for field in fields
            if not field.lower().endswith("censored")
        }
    indexed = {
        item["id"]: tuple(float(value) for value in item["binEdges"]) for item in rows
    }
    return {field: indexed[field] for field in fields}


def _bin(value: float, edges: Sequence[float]) -> int:
    if not math.isfinite(value) or value < edges[0] or value > edges[-1]:
        raise ValueError("descriptor outside frozen bounds")
    if value == edges[-1]:
        return len(edges) - 2
    for index, (left, right) in enumerate(zip(edges, edges[1:])):
        if left <= value < right:
            return index
    raise AssertionError("unreachable descriptor bin")


def _aggregate_descriptors(
    task_id: str, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    descriptor_rows: dict[str, list[dict[str, Any]]] = {}
    observed_by_family: dict[str, dict[str, dict[str, Any]]] = {}
    for row_index, row in enumerate(rows):
        family = str(
            row.get("scenarioFamilyOrdinal", row.get("scenarioOrdinal", row_index))
        )
        observed_by_family[family] = {}
        for item in _descriptor_sets(task_id, row["outcome"]):
            descriptor_rows.setdefault(item["archiveId"], []).append(item)
            observed_by_family[family][item["archiveId"]] = item
    if task_id == "e07_s02_regeneration_1d":
        return _aggregate_e05_descriptors(rows, observed_by_family)
    descriptors = []
    for archive_id, items in sorted(descriptor_rows.items()):
        if len(items) != len(rows):
            raise ValueError("descriptor missing from scenario panel")
        coordinate_fields = [
            key for key in items[0]["values"] if not key.lower().endswith("censored")
        ]
        means = {
            key: sum(float(item["values"][key]) for item in items) / len(items)
            for key in coordinate_fields
        }
        edges = _edges_for(items[0]["kind"], coordinate_fields)
        cell = tuple(_bin(means[key], edges[key]) for key in coordinate_fields)
        flags = {
            key: any(bool(item["values"][key]) for item in items)
            for key in items[0]["values"]
            if key.lower().endswith("censored")
        }
        descriptors.append(
            {
                "archiveId": archive_id,
                "kind": items[0]["kind"],
                "fields": coordinate_fields,
                "values": means,
                "cell": list(cell),
                "censorFlags": flags,
            }
        )
    return descriptors


def _e05_native_status(
    row: Mapping[str, Any], family: str, *, descriptor_available: bool
) -> dict[str, Any]:
    outcome = row["outcome"]
    return {
        "scenarioFamilyOrdinal": family,
        "scenarioId": row.get("scenarioId"),
        "stopReason": row.get("stopReason"),
        "sourceTerminal": bool(outcome.get("sourceTerminal", False)),
        "failed": bool(row.get("failed", False)),
        "censored": bool(row.get("censored", False)),
        "descriptorAvailable": descriptor_available,
    }


def _aggregate_e05_descriptors(
    rows: Sequence[Mapping[str, Any]],
    observed_by_family: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Retain terminal-dependent E05 support without manufacturing coordinates.

    Every possible native E05 regeneration descriptor produces one record.  A
    numeric task-local cell exists only when the same descriptor contract is
    present and finite in every coupled scenario family.  Missing phase/axis
    support remains explicit native terminal/failure/censor metadata.
    """

    families = tuple(
        sorted(
            observed_by_family,
            key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
        )
    )
    row_by_family = {
        str(row.get("scenarioFamilyOrdinal", row.get("scenarioOrdinal", index))): row
        for index, row in enumerate(rows)
    }
    descriptors: list[dict[str, Any]] = []
    for archive_id, spec in E05_REGENERATION_DESCRIPTOR_SPECS.items():
        expected_fields = tuple(spec["fields"])
        censor_fields = tuple(spec["censorFields"])
        observed_families: list[str] = []
        missing_families: list[str] = []
        reasons: set[str] = set()
        items: list[Mapping[str, Any]] = []
        native_status: list[dict[str, Any]] = []
        for family in families:
            item = observed_by_family[family].get(archive_id)
            available = item is not None
            native_status.append(
                _e05_native_status(
                    row_by_family[family], family, descriptor_available=available
                )
            )
            if item is None:
                missing_families.append(family)
                reasons.add("missing_native_phase_or_axis_support")
                continue
            observed_families.append(family)
            items.append(item)
            if item["kind"] != spec["kind"]:
                reasons.add("descriptor_kind_mismatch")
            values = item["values"]
            if set(values) != set(expected_fields) | set(censor_fields):
                reasons.add("descriptor_field_contract_mismatch")
            for field in expected_fields:
                value = values.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    reasons.add("nonfinite_or_nonnumeric_coordinate")
            for field in censor_fields:
                if not isinstance(values.get(field), bool):
                    reasons.add("invalid_censor_flag")
        complete = len(items) == len(rows) and not reasons
        support = {
            "expectedScenarioFamilyOrdinals": list(families),
            "observedScenarioFamilyOrdinals": observed_families,
            "missingScenarioFamilyOrdinals": missing_families,
            "nativeStatusByScenarioFamily": native_status,
            "sourceTerminalRows": sum(item["sourceTerminal"] for item in native_status),
            "failedRows": sum(item["failed"] for item in native_status),
            "censoredRows": sum(item["censored"] for item in native_status),
        }
        base = {
            "archiveId": archive_id,
            "kind": str(spec["kind"]),
            "fields": list(expected_fields),
            "descriptorAvailabilityVersion": E05_DESCRIPTOR_AVAILABILITY_VERSION,
            "supportState": "complete" if complete else "unavailable",
            "cellEligible": complete,
            "availabilityIsNoveltyCoordinate": False,
            "support": support,
            "availabilityReasonCodes": [] if complete else sorted(reasons),
        }
        if not complete:
            descriptors.append(
                {
                    **base,
                    "values": None,
                    "cell": None,
                    "censorFlags": None,
                    "comparabilityKeySha256": None,
                }
            )
            continue
        means = {
            field: sum(float(item["values"][field]) for item in items) / len(items)
            for field in expected_fields
        }
        edges = _edges_for(str(spec["kind"]), expected_fields)
        cell = tuple(_bin(means[field], edges[field]) for field in expected_fields)
        censor_flags = {
            field: any(bool(item["values"][field]) for item in items)
            for field in censor_fields
        }
        comparability = {
            "taskId": "e07_s02_regeneration_1d",
            "archiveId": archive_id,
            "descriptorAvailabilityVersion": E05_DESCRIPTOR_AVAILABILITY_VERSION,
            "coordinateFields": list(expected_fields),
            "scenarioFamilyOrdinals": list(families),
            "edges": {field: list(edges[field]) for field in expected_fields},
        }
        descriptors.append(
            {
                **base,
                "values": means,
                "cell": list(cell),
                "censorFlags": censor_flags,
                "comparabilityKeySha256": canonical_sha256(
                    "E07/S08F/E05-descriptor-comparability/v1", comparability
                ),
            }
        )
    return descriptors


def aggregate_candidate(
    policy: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not rows:
        raise ValueError("candidate aggregation requires evaluation rows")
    task_id = str(rows[0]["taskId"])
    if any(row["taskId"] != task_id for row in rows):
        raise ValueError("candidate aggregation cannot cross tasks")
    objectives: dict[str, float] = {}
    directions: dict[str, str] = {}
    for objective_id, path, direction in OBJECTIVES[task_id]:
        observed = [float(_get_path(row["outcome"], path)) for row in rows]
        objectives[objective_id] = sum(observed) / len(observed)
        directions[objective_id] = direction

    cost_rows = [_numeric_leaves(row["nativeLedgerFamilies"]) for row in rows]
    cost_fields = sorted({key for item in cost_rows for key in item})
    costs = {
        key: sum(item.get(key, 0.0) for item in cost_rows) / len(cost_rows)
        for key in cost_fields
        if not key.endswith("licensedLongRangeMaximumRequestedDistance")
    }
    compiled = compile_policy(policy["document"])
    complexity = {
        key: float(compiled.complexity.to_dict()[key]) for key in COMPLEXITY_FIELDS
    }
    descriptors = _aggregate_descriptors(task_id, rows)

    quality_min = {
        **{
            f"performance.{key}": (-value if directions[key] == "maximize" else value)
            for key, value in objectives.items()
        },
        **{f"cost.{key}": value for key, value in costs.items()},
        **{f"complexity.{key}": value for key, value in complexity.items()},
    }
    behavior_signature = hashlib.sha256(
        b"E07/S05/panel-semantic-equivalence/v1\x00"
        + _json_bytes(
            [
                {
                    "scenarioOrdinal": row["scenarioOrdinal"],
                    "semanticProjectionSha256": row["semanticProjectionSha256"],
                }
                for row in sorted(rows, key=lambda item: item["scenarioOrdinal"])
            ]
        )
    ).hexdigest()
    return {
        "schemaVersion": "e07.s05.candidate-aggregate.v1",
        "taskId": task_id,
        "policyId": policy["policyId"],
        "policySha256": policy["policySha256"],
        "policyBodySha256": policy["policyBodySha256"],
        "origin": policy["origin"],
        "generation": policy["generation"],
        "scenarioOrdinals": sorted(int(row["scenarioOrdinal"]) for row in rows),
        "scenarioCount": len(rows),
        "failedCount": sum(bool(row["failed"]) for row in rows),
        "censoredCount": sum(bool(row["censored"]) for row in rows),
        "validNativeEpisodes": all(
            not row["failed"] and row["replayPass"] and all(row["validation"].values())
            for row in rows
        ),
        "objectives": objectives,
        "objectiveDirections": directions,
        "costVector": costs,
        "complexityVector": complexity,
        "qualityMinimization": quality_min,
        "descriptors": descriptors,
        "boundedSemanticSha256": behavior_signature,
        "evaluationHashes": [
            row["stableEvaluationSha256"]
            for row in sorted(rows, key=lambda item: item["scenarioOrdinal"])
        ],
    }


def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    a = left["qualityMinimization"]
    b = right["qualityMinimization"]
    keys = set(a) | set(b)
    no_worse = all(
        float(a.get(key, 0.0)) <= float(b.get(key, 0.0)) + 1e-12 for key in keys
    )
    strictly = any(
        float(a.get(key, 0.0)) < float(b.get(key, 0.0)) - 1e-12 for key in keys
    )
    return no_worse and strictly


def finite_behavior_deduplicate(
    aggregates: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for item in aggregates:
        groups.setdefault(str(item["boundedSemanticSha256"]), []).append(item)
    retained = []
    audit = []
    for signature, members in sorted(groups.items()):
        ordered = sorted(
            members,
            key=lambda item: (
                tuple(item["complexityVector"][key] for key in COMPLEXITY_FIELDS),
                item["policySha256"],
            ),
        )
        retained.append(ordered[0])
        audit.append(
            {
                "boundedSemanticSha256": signature,
                "memberPolicySha256": [item["policySha256"] for item in ordered],
                "retainedPolicySha256": ordered[0]["policySha256"],
                "discardedPolicySha256": [item["policySha256"] for item in ordered[1:]],
                "finiteTrainingSupportOnly": True,
            }
        )
    return retained, audit


def archive_insert(
    archive: dict[tuple[str, tuple[int, ...]], list[Mapping[str, Any]]],
    candidate: Mapping[str, Any],
    *,
    descriptor_filter: set[str] | None = None,
) -> dict[str, int]:
    stats = {"newCells": 0, "frontAdditions": 0, "dominatedRemoved": 0, "rejected": 0}
    if not candidate["validNativeEpisodes"]:
        stats["rejected"] += len(candidate["descriptors"])
        return stats
    for descriptor in candidate["descriptors"]:
        if descriptor.get("cellEligible", True) is not True:
            stats["rejected"] += 1
            continue
        if (
            descriptor_filter is not None
            and descriptor["archiveId"] not in descriptor_filter
        ):
            continue
        key = (descriptor["archiveId"], tuple(descriptor["cell"]))
        front = list(archive.get(key, []))
        if any(dominates(item, candidate) for item in front):
            stats["rejected"] += 1
            continue
        equal = [
            item
            for item in front
            if item["qualityMinimization"] == candidate["qualityMinimization"]
        ]
        if (
            equal
            and min(item["policySha256"] for item in equal) < candidate["policySha256"]
        ):
            stats["rejected"] += 1
            continue
        survivors = [
            item
            for item in front
            if not dominates(candidate, item)
            and not (
                item["qualityMinimization"] == candidate["qualityMinimization"]
                and candidate["policySha256"] < item["policySha256"]
            )
        ]
        stats["dominatedRemoved"] += len(front) - len(survivors)
        if not front:
            stats["newCells"] += 1
        survivors.append(candidate)
        archive[key] = sorted(
            {item["policySha256"]: item for item in survivors}.values(),
            key=lambda item: item["policySha256"],
        )
        stats["frontAdditions"] += 1
    return stats


def stable_cell_audit(
    policy: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
) -> list[dict[str, Any]]:
    full = aggregate_candidate(policy, rows)
    by_archive = {item["archiveId"]: item for item in full["descriptors"]}
    matches = {
        archive_id: 0
        for archive_id, item in by_archive.items()
        if item.get("cellEligible", True) is True
    }
    task_id = str(rows[0]["taskId"])
    for replicate in range(bootstrap_replicates):
        sampled = [
            rows[
                counter_u64(
                    policy["policySha256"],
                    replicate,
                    draw,
                    stream="descriptor-bootstrap",
                )
                % len(rows)
            ]
            for draw in range(len(rows))
        ]
        sampled_by_archive = {
            item["archiveId"]: item for item in _aggregate_descriptors(task_id, sampled)
        }
        for archive_id, reference in by_archive.items():
            if archive_id in matches:
                sampled = sampled_by_archive[archive_id]
                matches[archive_id] += bool(
                    sampled.get("cellEligible", True) is True
                    and sampled["cell"] == reference["cell"]
                )
    return [
        (
            {
                "archiveId": archive_id,
                "referenceCell": reference["cell"],
                "supportState": reference.get("supportState", "complete"),
                "cellEligible": reference.get("cellEligible", True),
                "sameCellProbability": (
                    matches[archive_id] / bootstrap_replicates
                    if archive_id in matches
                    else None
                ),
                "bootstrapReplicates": bootstrap_replicates,
            }
        )
        for archive_id, reference in sorted(by_archive.items())
    ]


def bootstrap_objective_intervals(
    policy: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
) -> dict[str, dict[str, float]]:
    task_id = str(rows[0]["taskId"])
    observed = {
        objective_id: [float(_get_path(row["outcome"], path)) for row in rows]
        for objective_id, path, _ in OBJECTIVES[task_id]
    }
    samples: dict[str, list[float]] = {key: [] for key in observed}
    for replicate in range(bootstrap_replicates):
        indices = [
            (
                counter_u64(
                    policy["policySha256"],
                    replicate,
                    draw,
                    stream="objective-bootstrap",
                )
                % len(rows)
            )
            for draw in range(len(rows))
        ]
        for key, values in observed.items():
            samples[key].append(sum(values[index] for index in indices) / len(indices))

    def quantile(values: Sequence[float], probability: float) -> float:
        ordered = sorted(values)
        index = probability * (len(ordered) - 1)
        left = math.floor(index)
        right = math.ceil(index)
        if left == right:
            return ordered[left]
        weight = index - left
        return ordered[left] * (1 - weight) + ordered[right] * weight

    point = {key: sum(values) / len(values) for key, values in observed.items()}
    return {
        key: {
            "mean": float(point[key]),
            "lower95": quantile(values, 0.025),
            "upper95": quantile(values, 0.975),
        }
        for key, values in samples.items()
    }
