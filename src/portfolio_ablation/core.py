"""Outcome-independent edit generation and paired S09 spatial evaluation."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from src.environment_suite import (
    AccessGrant,
    AccessPhase,
    Split,
    SuiteValidationError,
    portfolio_action,
)
from src.environment_suite.access import AccessBroker
from src.environment_suite.contracts import ScenarioRecord, canonical_sha256
from src.environment_suite.runners import RUNNERS
from src.environment_suite.suite import EnvironmentSuite
from src.policy_dsl import compile_policy
from src.portfolio_preregistration.core import canonical_hash
from src.portfolio_search.preflight import sha256_file, tree_digest
from src.quality_diversity.core import policy_body_sha256


REPOSITORY = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path("/artifacts/research_steps")
S02 = ARTIFACT_ROOT / "S02"
S05 = ARTIFACT_ROOT / "S05"
S08M = ARTIFACT_ROOT / "S08M"
TASK_REGISTRY = REPOSITORY / "configs/environment_suite/task_registry.yaml"
SPLIT_MANIFEST = REPOSITORY / "configs/environment_suite/split_manifest.json"
PROTOCOL_PATH = REPOSITORY / "configs/portfolio/s09_ablation_compression.yaml"
SPATIAL_TASKS = (
    "e07_s02_spatial2d_local",
    "e07_s02_spatial2d_memory",
)
EDIT_ORDER = (
    "member_delete",
    "selector_remove_to_fixed_balanced",
    "selector_branch_flip_control",
    "fixed_assignment_rotation_control",
    "rule_delete",
    "memory_freeze_register",
    "observation_predicate_erase",
    "sensing_delta_to_counter_index",
    "sensing_restrict_adjacent_radius_one",
    "sensing_remove_positive_filter",
    "exact_runtime_sham",
)
ENDPOINTS = {
    "conjunctive_completion": (
        ("conjunctiveCompletion",),
        "maximize",
        0.0,
    ),
    "s01_global_mismatch": (
        ("s01GlobalCompletionAudit", "normalizedMismatch"),
        "minimize",
        0.01,
    ),
    "s02_local_acceptance": (
        ("s02LocalGrammar", "accepted"),
        "maximize",
        0.0,
    ),
    "formation_success": (("formationCompletion",), "maximize", 0.0),
    "repair_success": (("repairCompletion",), "maximize", 0.0),
}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def checked_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if protocol.get("schemaVersion") != "e07.s09.ablation-compression-protocol.v1":
        raise RuntimeError("unexpected S09 protocol schema")
    ids = list(protocol["eligibleConfigurationIds"])
    if len(ids) != 7 or len(set(ids)) != 7:
        raise RuntimeError("S09 protocol must name exactly seven unique parents")
    if tuple(protocol["authorizationBoundary"]["allowedTasks"]) != SPATIAL_TASKS:
        raise RuntimeError("S09 protocol task scope changed")
    if protocol["authorizationBoundary"]["allowedSplit"] != "train":
        raise RuntimeError("S09 must remain training-only")
    return protocol


def validate_s09_inputs(
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the seven-parent lock without loading prohibited artifacts."""

    protocol = dict(protocol or checked_protocol())
    checked = []
    for name, spec in protocol["frozenInputs"].items():
        path = Path(spec["path"])
        actual = sha256_file(path)
        expected = spec.get("sha256")
        if expected is not None and actual != expected:
            raise RuntimeError(f"frozen S09 input changed: {name}")
        checked.append(
            {
                "name": name,
                "path": str(path),
                "sha256": actual,
                "bytes": path.stat().st_size,
            }
        )
    gate = json.loads(
        Path(protocol["frozenInputs"]["s08mEligibilityGate"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    protocol_ids = sorted(protocol["eligibleConfigurationIds"])
    gate_ids = sorted(gate.get("eligibleConfigurationIds", []))
    if protocol_ids != gate_ids:
        raise RuntimeError("S08M eligibility population changed")
    eligible_rows = [row for row in gate["rows"] if row.get("s09Eligible")]
    if len(eligible_rows) != 7:
        raise RuntimeError("S08M gate no longer contains seven passing rows")
    if any(row["taskId"] not in SPATIAL_TASKS for row in eligible_rows):
        raise RuntimeError("non-spatial parent entered S09")
    required = {
        "completePairedValidation",
        "exactReplayAndContractPass",
        "failureCensorGuardPass",
        "frozenHashesPass",
        "holmSignificantBenefitVersusMatchedRandom",
        "holmSignificantBenefitVersusRelevantSingles",
        "noHolmSignificantHarmBeyondMargin",
        "separatePortfolioCostsDisclosed",
    }
    if any(not all(row.get(key) is True for key in required) for row in eligible_rows):
        raise RuntimeError("S08M S09 gate conjunction changed")
    return {
        "schemaVersion": "e07.s09.input-validation.v1",
        "researchStepId": "S09",
        "success": True,
        "eligibleConfigurationIds": protocol_ids,
        "eligibleRows": eligible_rows,
        "candidateLockSha256": sha256_file(
            protocol["frozenInputs"]["s08mCandidateLock"]["path"]
        ),
        "checkedFiles": checked,
        "s06OrS06AArtifactLoads": 0,
        "s07ArmSignalUses": 0,
        "validationOutcomeAccesses": 0,
        "confirmationOutcomeAccesses": 0,
    }


def load_parent_bundles(
    protocol: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    protocol = dict(protocol or checked_protocol())
    allowed = set(protocol["eligibleConfigurationIds"])
    configs = {
        row["configurationId"]: row
        for row in _read_jsonl(
            protocol["frozenInputs"]["s08mConfigurationRegistry"]["path"]
        )
        if row.get("configurationId") in allowed
    }
    if set(configs) != allowed:
        raise RuntimeError("not every frozen S09 parent has a configuration row")
    catalog = {
        row["policySha256"]: row
        for row in _read_jsonl(protocol["frozenInputs"]["s05PolicyCatalog"]["path"])
    }
    bundles = []
    for configuration_id in sorted(allowed):
        config = deepcopy(configs[configuration_id])
        if config["taskId"] not in SPATIAL_TASKS:
            raise RuntimeError("parent task left the frozen spatial scope")
        documents = {}
        for member in config["members"]:
            policy_hash = member["policySha256"]
            row = catalog.get(policy_hash)
            if row is None:
                raise RuntimeError("parent member lacks authoritative S05 document")
            compiled = compile_policy(row["document"])
            if compiled.policy_sha256 != policy_hash:
                raise RuntimeError("parent member does not recompile")
            documents[policy_hash] = deepcopy(row["document"])
        action = portfolio_action(
            [documents[item["policySha256"]] for item in config["members"]],
            config,
        )
        bundles.append(
            {
                "parentConfigurationId": configuration_id,
                "taskId": config["taskId"],
                "configuration": config,
                "documentsByPolicySha256": documents,
                "parentActionSha256": action.policy_sha256,
            }
        )
    counts = defaultdict(int)
    for row in bundles:
        counts[row["taskId"]] += 1
    if dict(counts) != {
        "e07_s02_spatial2d_local": 2,
        "e07_s02_spatial2d_memory": 5,
    }:
        raise RuntimeError("S09 parent task counts changed")
    return bundles


def _structural_costs(
    members: Sequence[Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
    selector: Mapping[str, Any] | None,
) -> dict[str, int]:
    compiled = [compile_policy(documents[item["policySha256"]]) for item in members]
    selector_bytes = len(_json_bytes(selector)) if selector else 0
    return {
        "uniqueMemberCount": len(compiled),
        "totalCanonicalMemberBytes": sum(
            item.complexity.canonical_bytes for item in compiled
        ),
        "totalRuleCount": sum(item.complexity.rule_count for item in compiled),
        "totalExpressionNodes": sum(
            item.complexity.expression_nodes for item in compiled
        ),
        "totalPersistentMemoryBits": sum(
            item.complexity.persistent_memory_bits for item in compiled
        ),
        "totalOutboundSignalBits": sum(
            item.complexity.outbound_signal_bits for item in compiled
        ),
        "selectorBranchCount": 2 if selector else 0,
        "selectorExpressionNodes": 3 if selector else 0,
        "selectorCanonicalBytes": selector_bytes,
    }


def _member_projection(
    document: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    compiled = compile_policy(document)
    return {
        "policyId": compiled.policy_id,
        "policySha256": compiled.policy_sha256,
        "policyBodySha256": policy_body_sha256(document),
        "nativeCarrier": "spatial",
        "boundedSemanticGroup": canonical_hash(
            "E07/S09/unproven-bounded-semantic-singleton/v1",
            [compiled.policy_sha256],
        ),
        "s09SourcePolicySha256": source.get(
            "s09SourcePolicySha256", source["policySha256"]
        ),
    }


def _variant_id(parent_id: str, operator: str, target: Mapping[str, Any]) -> str:
    return canonical_sha256(
        "E07/S09/edit/v1",
        {"parentConfigurationId": parent_id, "operator": operator, "target": target},
    )


def _finish_variant(
    parent: Mapping[str, Any],
    config: dict[str, Any],
    documents: dict[str, dict[str, Any]],
    *,
    operator: str,
    category: str,
    target: Mapping[str, Any],
    causal_edit: bool,
    control_kind: str | None = None,
) -> dict[str, Any]:
    parent_id = str(parent["parentConfigurationId"])
    edit_id = _variant_id(parent_id, operator, target)
    config = deepcopy(config)
    config["schemaVersion"] = "e07.s08c.adaptive-portfolio.v1"
    config["researchStepId"] = "S09"
    config["evaluationAuthorized"] = True
    config["outcomeMaterialized"] = False
    config["s07ArmMembershipUsed"] = False
    config["rejectedModelOrEmbeddingUsed"] = False
    config["s09ParentConfigurationId"] = parent_id
    config["s09EditId"] = edit_id
    config["portfolioSize"] = len(config["members"])
    config["memberSetId"] = canonical_sha256(
        "E07/S09/member-set/v1",
        [
            parent_id,
            edit_id,
            [item["policySha256"] for item in config["members"]],
        ],
    )
    config["portfolioStructuralCosts"] = _structural_costs(
        config["members"], documents, config.get("selector")
    )
    definition_for_hash = deepcopy(config)
    definition_for_hash.pop("configurationId", None)
    config["configurationId"] = canonical_sha256(
        "E07/S09/runtime-configuration/v1", definition_for_hash
    )
    ordered_documents = [documents[item["policySha256"]] for item in config["members"]]
    action = portfolio_action(ordered_documents, config)
    before = dict(parent["configuration"]["portfolioStructuralCosts"])
    after = dict(config["portfolioStructuralCosts"])
    delta = {key: after[key] - before[key] for key in sorted(before)}
    return {
        "schemaVersion": "e07.s09.ablation-edit.v1",
        "researchStepId": "S09",
        "editId": edit_id,
        "parentConfigurationId": parent_id,
        "taskId": parent["taskId"],
        "category": category,
        "operator": operator,
        "target": dict(target),
        "causalEdit": causal_edit,
        "controlKind": control_kind,
        "configuration": config,
        "documentsByPolicySha256": documents,
        "variantConfigurationId": config["configurationId"],
        "variantActionSha256": action.policy_sha256,
        "parentStructuralCosts": before,
        "variantStructuralCosts": after,
        "structuralCostDelta": delta,
        "structuralBytesReduced": -delta["totalCanonicalMemberBytes"],
        "globalMinimalityClaimPermitted": False,
    }


def _rename_document(document: Mapping[str, Any], edit_id: str) -> dict[str, Any]:
    result = deepcopy(dict(document))
    result["policyId"] = f"s09_{edit_id[:20]}"
    return result


def _replace_member_document(
    parent: Mapping[str, Any],
    member_index: int,
    document: Mapping[str, Any],
    edit_id: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    config = deepcopy(parent["configuration"])
    documents = deepcopy(parent["documentsByPolicySha256"])
    old_member = config["members"][member_index]
    old_hash = old_member["policySha256"]
    renamed = _rename_document(document, edit_id)
    compiled = compile_policy(renamed)
    documents.pop(old_hash)
    documents[compiled.policy_sha256] = renamed
    config["members"][member_index] = _member_projection(renamed, old_member)
    return config, documents


def _simplify_condition(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    if "all" in value:
        children = [_simplify_condition(item) for item in value["all"]]
        if any(item == {"const": False} for item in children):
            return {"const": False}
        children = [item for item in children if item != {"const": True}]
        if not children:
            return {"const": True}
        if len(children) == 1:
            return children[0]
        return {"all": children}
    if "any" in value:
        children = [_simplify_condition(item) for item in value["any"]]
        if any(item == {"const": True} for item in children):
            return {"const": True}
        children = [item for item in children if item != {"const": False}]
        if not children:
            return {"const": False}
        if len(children) == 1:
            return children[0]
        return {"any": children}
    if "not" in value:
        child = _simplify_condition(value["not"])
        if child == {"const": True}:
            return {"const": False}
        if child == {"const": False}:
            return {"const": True}
        return {"not": child}
    return {key: _simplify_condition(child) for key, child in value.items()}


def _contains_operand(value: Any, key: str, name: str) -> bool:
    if isinstance(value, Mapping):
        if value.get(key) == name:
            return True
        return any(_contains_operand(child, key, name) for child in value.values())
    if isinstance(value, list):
        return any(_contains_operand(child, key, name) for child in value)
    return False


def _erase_observation_condition(value: Any, observation: str) -> Any:
    if not isinstance(value, Mapping):
        return value
    if (
        "op" in value
        and (
            _contains_operand(value.get("left"), "obs", observation)
            or _contains_operand(value.get("right"), "obs", observation)
        )
    ):
        return {"const": True}
    return _simplify_condition(
        {
            key: (
                [_erase_observation_condition(item, observation) for item in child]
                if isinstance(child, list)
                else _erase_observation_condition(child, observation)
            )
            for key, child in value.items()
        }
    )


def _replace_memory_operand(value: Any, register: str, initial: int) -> Any:
    if isinstance(value, Mapping):
        if value.get("mem") == register and len(value) == 1:
            return {"const": initial}
        return {
            key: _replace_memory_operand(child, register, initial)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_replace_memory_operand(child, register, initial) for child in value]
    return value


def _walk_actions(document: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for rule in document["rules"]:
        yield from rule["actions"]
    yield from document["default"]["actions"]


def _uses_observation(document: Mapping[str, Any], observation: str) -> bool:
    if _contains_operand(document, "obs", observation):
        return True
    return any(
        action.get("selector", {}).get("indexObservation") == observation
        for action in _walk_actions(document)
    )


def _normalize_policy_limits(document: dict[str, Any]) -> None:
    movement_kinds = {"swap_relative", "swap_cursor", "move_candidate"}
    if not any(
        action.get("kind") in movement_kinds for action in _walk_actions(document)
    ):
        document["limits"]["maxMovementRadius"] = 0


def build_edit_registry(
    parents: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enumerate every edit from structure only; never inspect an outcome."""

    parents = list(parents or load_parent_bundles())
    rows: list[dict[str, Any]] = []
    coverage = []
    for parent in parents:
        parent_id = parent["parentConfigurationId"]
        config = parent["configuration"]
        documents = parent["documentsByPolicySha256"]
        parent_rows: list[dict[str, Any]] = []

        # Exact runtime sham: only an ignored audit tag and resulting identities differ.
        target = {"auditTag": "exact_runtime_sham"}
        edit_id = _variant_id(parent_id, "exact_runtime_sham", target)
        sham_config = deepcopy(config)
        sham_config["s09ShamAuditTag"] = edit_id
        parent_rows.append(
            _finish_variant(
                parent,
                sham_config,
                deepcopy(documents),
                operator="exact_runtime_sham",
                category="sham",
                target=target,
                causal_edit=False,
                control_kind="exact_runtime_sham",
            )
        )

        if len(config["members"]) > 2:
            for member_index, member in enumerate(config["members"]):
                target = {
                    "memberIndex": member_index,
                    "policySha256": member["policySha256"],
                }
                new_config = deepcopy(config)
                removed = new_config["members"].pop(member_index)
                new_documents = deepcopy(documents)
                new_documents.pop(removed["policySha256"])
                parent_rows.append(
                    _finish_variant(
                        parent,
                        new_config,
                        new_documents,
                        operator="member_delete",
                        category="member",
                        target=target,
                        causal_edit=True,
                    )
                )

        if config["mode"] == "environment_conditioned":
            target = {"selector": "remove_to_fixed_balanced"}
            new_config = deepcopy(config)
            new_config["mode"] = "fixed_balanced_identity"
            new_config["selector"] = None
            new_config["assignmentCounterDomain"] = (
                "E07/S08/fixed_balanced_identity/identity-opportunity/v1"
            )
            new_config["assignmentRotation"] = int(
                new_config.get("assignmentRotation", 0)
            ) % len(new_config["members"])
            parent_rows.append(
                _finish_variant(
                    parent,
                    new_config,
                    deepcopy(documents),
                    operator="selector_remove_to_fixed_balanced",
                    category="selector",
                    target=target,
                    causal_edit=True,
                )
            )
            target = {"selector": "branch_flip"}
            new_config = deepcopy(config)
            selector = deepcopy(new_config["selector"])
            selector["trueBranch"], selector["falseBranch"] = (
                selector["falseBranch"],
                selector["trueBranch"],
            )
            new_config["selector"] = selector
            parent_rows.append(
                _finish_variant(
                    parent,
                    new_config,
                    deepcopy(documents),
                    operator="selector_branch_flip_control",
                    category="selector",
                    target=target,
                    causal_edit=False,
                    control_kind="complexity_matched_selector_direction",
                )
            )
        elif config["mode"] == "fixed_balanced_identity":
            target = {"assignmentRotationDelta": 1}
            new_config = deepcopy(config)
            new_config["assignmentRotation"] = (
                int(new_config.get("assignmentRotation", 0)) + 1
            ) % len(new_config["members"])
            parent_rows.append(
                _finish_variant(
                    parent,
                    new_config,
                    deepcopy(documents),
                    operator="fixed_assignment_rotation_control",
                    category="selector",
                    target=target,
                    causal_edit=False,
                    control_kind="complexity_matched_identity_assignment",
                )
            )

        for member_index, member in enumerate(config["members"]):
            old_hash = member["policySha256"]
            document = documents[old_hash]
            for rule_index in (
                range(len(document["rules"]))
                if len(document["rules"]) > 1
                else ()
            ):
                target = {
                    "memberIndex": member_index,
                    "sourcePolicySha256": old_hash,
                    "ruleIndex": rule_index,
                }
                edit_id = _variant_id(parent_id, "rule_delete", target)
                edited = deepcopy(document)
                edited["rules"].pop(rule_index)
                _normalize_policy_limits(edited)
                new_config, new_documents = _replace_member_document(
                    parent, member_index, edited, edit_id
                )
                parent_rows.append(
                    _finish_variant(
                        parent,
                        new_config,
                        new_documents,
                        operator="rule_delete",
                        category="rule",
                        target=target,
                        causal_edit=True,
                    )
                )
            for memory in document["memory"]:
                register = memory["name"]
                initial = int(memory["initial"])
                target = {
                    "memberIndex": member_index,
                    "sourcePolicySha256": old_hash,
                    "register": register,
                    "initial": initial,
                }
                edit_id = _variant_id(
                    parent_id, "memory_freeze_register", target
                )
                edited = _replace_memory_operand(document, register, initial)
                edited["memory"] = [
                    item for item in edited["memory"] if item["name"] != register
                ]
                action_bundles = [
                    rule["actions"] for rule in edited["rules"]
                ] + [edited["default"]["actions"]]
                for actions in action_bundles:
                    actions[:] = [
                        action
                        for action in actions
                        if not (
                            action.get("kind") == "set_memory"
                            and action.get("register") == register
                        )
                    ]
                new_config, new_documents = _replace_member_document(
                    parent, member_index, edited, edit_id
                )
                parent_rows.append(
                    _finish_variant(
                        parent,
                        new_config,
                        new_documents,
                        operator="memory_freeze_register",
                        category="memory",
                        target=target,
                        causal_edit=True,
                    )
                )
            condition_observations = sorted(
                permission
                for permission in document["permissions"]
                if any(
                    _contains_operand(rule["when"], "obs", permission)
                    for rule in document["rules"]
                )
            )
            for observation in condition_observations:
                target = {
                    "memberIndex": member_index,
                    "sourcePolicySha256": old_hash,
                    "observation": observation,
                }
                edit_id = _variant_id(
                    parent_id, "observation_predicate_erase", target
                )
                edited = deepcopy(document)
                for rule in edited["rules"]:
                    rule["when"] = _erase_observation_condition(
                        rule["when"], observation
                    )
                if not _uses_observation(edited, observation):
                    edited["permissions"] = [
                        value
                        for value in edited["permissions"]
                        if value != observation
                    ]
                new_config, new_documents = _replace_member_document(
                    parent, member_index, edited, edit_id
                )
                parent_rows.append(
                    _finish_variant(
                        parent,
                        new_config,
                        new_documents,
                        operator="observation_predicate_erase",
                        category="observation",
                        target=target,
                        causal_edit=True,
                    )
                )
            has_delta_selector = any(
                action.get("kind") == "move_candidate"
                and action.get("selector", {}).get("field")
                == "candidate.local_relation_delta"
                for action in _walk_actions(document)
            )
            if has_delta_selector:
                target = {
                    "memberIndex": member_index,
                    "sourcePolicySha256": old_hash,
                    "sensing": "candidate_delta_to_counter_index",
                }
                edit_id = _variant_id(
                    parent_id, "sensing_delta_to_counter_index", target
                )
                edited = deepcopy(document)
                for action in _walk_actions(edited):
                    if (
                        action.get("kind") == "move_candidate"
                        and action.get("selector", {}).get("field")
                        == "candidate.local_relation_delta"
                    ):
                        action["selector"] = {
                            "mode": "index",
                            "field": None,
                            "requirePositive": False,
                            "indexObservation": "counter.choice_u8",
                        }
                if "counter.choice_u8" not in edited["permissions"]:
                    edited["permissions"].append("counter.choice_u8")
                    edited["permissions"].sort()
                if not _contains_operand(
                    edited, "obs", "candidate.local_relation_delta"
                ) and not any(
                    action.get("selector", {}).get("field")
                    == "candidate.local_relation_delta"
                    for action in _walk_actions(edited)
                ):
                    edited["permissions"] = [
                        value
                        for value in edited["permissions"]
                        if value != "candidate.local_relation_delta"
                    ]
                new_config, new_documents = _replace_member_document(
                    parent, member_index, edited, edit_id
                )
                parent_rows.append(
                    _finish_variant(
                        parent,
                        new_config,
                        new_documents,
                        operator="sensing_delta_to_counter_index",
                        category="sensing",
                        target=target,
                        causal_edit=True,
                    )
                )
            nonlocal_movement = any(
                action.get("kind") == "move_candidate"
                and set(action.get("allowedMovementKinds", []))
                - {"adjacent_swap"}
                for action in _walk_actions(document)
            )
            if nonlocal_movement:
                target = {
                    "memberIndex": member_index,
                    "sourcePolicySha256": old_hash,
                    "sensing": "restrict_adjacent_radius_one",
                }
                edit_id = _variant_id(
                    parent_id, "sensing_restrict_adjacent_radius_one", target
                )
                edited = deepcopy(document)
                for action in _walk_actions(edited):
                    if action.get("kind") == "move_candidate":
                        action["allowedMovementKinds"] = ["adjacent_swap"]
                edited["limits"]["maxMovementRadius"] = 1
                new_config, new_documents = _replace_member_document(
                    parent, member_index, edited, edit_id
                )
                parent_rows.append(
                    _finish_variant(
                        parent,
                        new_config,
                        new_documents,
                        operator="sensing_restrict_adjacent_radius_one",
                        category="sensing",
                        target=target,
                        causal_edit=True,
                    )
                )
            positive_filters = any(
                action.get("kind") == "move_candidate"
                and action.get("selector", {}).get("requirePositive") is True
                for action in _walk_actions(document)
            )
            if positive_filters:
                target = {
                    "memberIndex": member_index,
                    "sourcePolicySha256": old_hash,
                    "sensing": "remove_positive_filter",
                }
                edit_id = _variant_id(
                    parent_id, "sensing_remove_positive_filter", target
                )
                edited = deepcopy(document)
                for action in _walk_actions(edited):
                    if action.get("kind") == "move_candidate":
                        action["selector"]["requirePositive"] = False
                new_config, new_documents = _replace_member_document(
                    parent, member_index, edited, edit_id
                )
                parent_rows.append(
                    _finish_variant(
                        parent,
                        new_config,
                        new_documents,
                        operator="sensing_remove_positive_filter",
                        category="sensing",
                        target=target,
                        causal_edit=True,
                    )
                )

        if len({row["editId"] for row in parent_rows}) != len(parent_rows):
            raise RuntimeError("duplicate S09 edit ID")
        rows.extend(parent_rows)
        all_channels_zero = all(
            document["signals"] == {"channels": 0, "bitsPerChannel": 0}
            for document in documents.values()
        )
        coverage.append(
            {
                "parentConfigurationId": parent_id,
                "taskId": parent["taskId"],
                "portfolioSize": len(config["members"]),
                "editCount": len(parent_rows),
                "editCountsByCategory": dict(
                    sorted(
                        (
                            category,
                            sum(row["category"] == category for row in parent_rows),
                        )
                        for category in {row["category"] for row in parent_rows}
                    )
                ),
                "memberDeletionAtCardinalityFloor": len(config["members"]) == 2,
                "memberPoliciesAtRuleCardinalityFloor": sum(
                    len(document["rules"]) == 1
                    for document in documents.values()
                ),
                "allMemberSignalChannelsZero": all_channels_zero,
                "signalAblationStatus": (
                    "structurally_inapplicable_no_channels"
                    if all_channels_zero
                    else "registered"
                ),
                "communicationAblationStatus": (
                    "structurally_inapplicable_no_channels"
                    if all_channels_zero
                    else "registered"
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            row["parentConfigurationId"],
            EDIT_ORDER.index(row["operator"]),
            row["editId"],
        )
    )
    return rows, {
        "schemaVersion": "e07.s09.edit-coverage.v1",
        "researchStepId": "S09",
        "success": True,
        "parentCount": len(parents),
        "editCount": len(rows),
        "causalEditCount": sum(row["causalEdit"] for row in rows),
        "controlCount": sum(not row["causalEdit"] for row in rows),
        "parents": coverage,
        "signalAblationsFabricated": 0,
        "communicationAblationsFabricated": 0,
    }


def compose_causal_edits(
    parent: Mapping[str, Any], edits: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Rebuild a cumulative variant from one frozen parent and original targets."""

    if not edits:
        raise ValueError("a cumulative S09 variant requires at least one edit")
    parent_id = str(parent["parentConfigurationId"])
    selected = sorted(
        edits,
        key=lambda row: (
            EDIT_ORDER.index(str(row["operator"])),
            str(row["editId"]),
        ),
    )
    if any(
        row["parentConfigurationId"] != parent_id or not row["causalEdit"]
        for row in selected
    ):
        raise RuntimeError("cumulative edits must be causal children of one parent")
    if len({row["editId"] for row in selected}) != len(selected):
        raise RuntimeError("cumulative edit set contains duplicates")

    config = deepcopy(parent["configuration"])
    original_documents = parent["documentsByPolicySha256"]
    delete_sources = {
        str(row["target"]["policySha256"])
        for row in selected
        if row["operator"] == "member_delete"
    }
    if len(config["members"]) - len(delete_sources) < 2:
        raise RuntimeError("cumulative member deletion reached cardinality floor")
    config["members"] = [
        member
        for member in config["members"]
        if member["policySha256"] not in delete_sources
    ]
    if sum(
        row["operator"] == "selector_remove_to_fixed_balanced" for row in selected
    ) > 1:
        raise RuntimeError("duplicate cumulative selector removal")
    if any(
        row["operator"] == "selector_remove_to_fixed_balanced" for row in selected
    ):
        if config["mode"] != "environment_conditioned":
            raise RuntimeError("selector removal is incompatible with current mode")
        config["mode"] = "fixed_balanced_identity"
        config["selector"] = None
        config["assignmentCounterDomain"] = (
            "E07/S08/fixed_balanced_identity/identity-opportunity/v1"
        )
        config["assignmentRotation"] = int(
            config.get("assignmentRotation", 0)
        ) % len(config["members"])

    composition_id = canonical_sha256(
        "E07/S09/composition/v1",
        {"parentConfigurationId": parent_id, "editIds": [row["editId"] for row in selected]},
    )
    documents: dict[str, dict[str, Any]] = {}
    rebuilt_members = []
    for source_member in config["members"]:
        source_hash = str(source_member["policySha256"])
        document = deepcopy(original_documents[source_hash])
        member_edits = [
            row
            for row in selected
            if row["target"].get("sourcePolicySha256") == source_hash
        ]
        rule_indexes = sorted(
            {
                int(row["target"]["ruleIndex"])
                for row in member_edits
                if row["operator"] == "rule_delete"
            },
            reverse=True,
        )
        if len(document["rules"]) - len(rule_indexes) < 1:
            raise RuntimeError("cumulative rule deletion reached DSL floor")
        for rule_index in rule_indexes:
            document["rules"].pop(rule_index)
        for row in member_edits:
            if row["operator"] != "memory_freeze_register":
                continue
            register = str(row["target"]["register"])
            initial = int(row["target"]["initial"])
            document = _replace_memory_operand(document, register, initial)
            document["memory"] = [
                item for item in document["memory"] if item["name"] != register
            ]
            action_bundles = [
                rule["actions"] for rule in document["rules"]
            ] + [document["default"]["actions"]]
            for actions in action_bundles:
                actions[:] = [
                    action
                    for action in actions
                    if not (
                        action.get("kind") == "set_memory"
                        and action.get("register") == register
                    )
                ]
        for row in member_edits:
            if row["operator"] != "observation_predicate_erase":
                continue
            observation = str(row["target"]["observation"])
            for rule in document["rules"]:
                rule["when"] = _erase_observation_condition(
                    rule["when"], observation
                )
            if not _uses_observation(document, observation):
                document["permissions"] = [
                    item
                    for item in document["permissions"]
                    if item != observation
                ]
        for row in member_edits:
            operator = row["operator"]
            if operator == "sensing_delta_to_counter_index":
                for action in _walk_actions(document):
                    if (
                        action.get("kind") == "move_candidate"
                        and action.get("selector", {}).get("field")
                        == "candidate.local_relation_delta"
                    ):
                        action["selector"] = {
                            "mode": "index",
                            "field": None,
                            "requirePositive": False,
                            "indexObservation": "counter.choice_u8",
                        }
                if "counter.choice_u8" not in document["permissions"]:
                    document["permissions"].append("counter.choice_u8")
                    document["permissions"].sort()
                if not _uses_observation(
                    document, "candidate.local_relation_delta"
                ):
                    document["permissions"] = [
                        item
                        for item in document["permissions"]
                        if item != "candidate.local_relation_delta"
                    ]
            elif operator == "sensing_restrict_adjacent_radius_one":
                for action in _walk_actions(document):
                    if action.get("kind") == "move_candidate":
                        action["allowedMovementKinds"] = ["adjacent_swap"]
                document["limits"]["maxMovementRadius"] = 1
            elif operator == "sensing_remove_positive_filter":
                for action in _walk_actions(document):
                    if action.get("kind") == "move_candidate":
                        action["selector"]["requirePositive"] = False
        _normalize_policy_limits(document)
        document["policyId"] = (
            f"s09_{composition_id[:12]}_{source_hash[:7]}"
        )
        compiled = compile_policy(document)
        documents[compiled.policy_sha256] = document
        rebuilt_members.append(_member_projection(document, source_member))
    config["members"] = rebuilt_members
    config["assignmentRotation"] = int(
        config.get("assignmentRotation", 0)
    ) % len(rebuilt_members)
    target = {
        "compositionId": composition_id,
        "editIds": [row["editId"] for row in selected],
    }
    result = _finish_variant(
        parent,
        config,
        documents,
        operator="cumulative_delta_debug",
        category="cumulative",
        target=target,
        causal_edit=True,
    )
    result["componentEditIds"] = target["editIds"]
    result["compositionId"] = composition_id
    return result


def _base_train_records() -> tuple[dict[str, Any], dict[str, ScenarioRecord]]:
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    base = {}
    for record in suite.records.values():
        if record.task_id in SPATIAL_TASKS and record.split is Split.TRAIN:
            base[record.task_id] = record
    if set(base) != set(SPATIAL_TASKS):
        raise RuntimeError("spatial training bases are incomplete")
    return suite.tasks, base


def derive_training_record(
    base: ScenarioRecord, family: int, *, panel: str
) -> ScenarioRecord:
    if base.split is not Split.TRAIN or base.protected:
        raise SuiteValidationError("S09 derivation requires an unprotected train base")
    parameters = dict(base.public_parameters)
    parameters["counterScheduleKey"] = (
        f"E07/S09/{panel}/{base.task_id}/train-family/{int(family):04d}"
    )
    scenario_id = (
        f"e07s09:{base.task_id.removeprefix('e07_s02_')}:{panel}:{int(family):04d}"
    )
    return ScenarioRecord(
        scenario_id=scenario_id,
        task_id=base.task_id,
        split=Split.TRAIN,
        materializer_id=f"s09_{panel}_{int(family):04d}",
        public_parameters=parameters,
        protected=False,
        outcome_access="development",
        predecessor_partition=f"S09_outcome_blind_train_{panel}_{base.scenario_id}",
    )


def _endpoint_projection(outcome: Mapping[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name, (path, _direction, _margin) in ENDPOINTS.items():
        value: Any = outcome
        for part in path:
            if value is None or not isinstance(value, Mapping):
                value = None
                break
            value = value.get(part)
        if value is None:
            result[name] = None
        elif isinstance(value, bool):
            result[name] = float(value)
        else:
            result[name] = float(value)
    return result


def _numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_numeric_leaves(child, path))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = float(value)
    return result


def evaluate_work_item(work: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one paired condition on an outcome-blind training family."""

    configuration = deepcopy(work["configuration"])
    task_id = str(work["taskId"])
    if task_id not in SPATIAL_TASKS or work.get("split") != "train":
        raise SuiteValidationError("S09 work left the frozen train-only spatial scope")
    configuration["taskId"] = task_id
    documents = {
        key: deepcopy(value)
        for key, value in work["documentsByPolicySha256"].items()
    }
    ordered_documents = [
        documents[item["policySha256"]] for item in configuration["members"]
    ]
    action = portfolio_action(ordered_documents, configuration)
    tasks, bases = _base_train_records()
    record = derive_training_record(
        bases[task_id], int(work["scenarioFamilyOrdinal"]), panel=str(work["panel"])
    )
    broker = AccessBroker({record.scenario_id: record}, confirmation_unsealed=False)
    grant = AccessGrant(AccessPhase.DEVELOPMENT)
    broker.authorize_scenario(record.scenario_id, grant)
    started = time.perf_counter()
    result = RUNNERS[tasks[task_id].runner_id](record, action)
    elapsed = time.perf_counter() - started
    broker.authorize_outcome(record.scenario_id, grant)
    false_validation = sorted(
        key for key, value in result.validation.items() if not value
    )
    contract_pass = (
        result.replay_pass
        and not false_validation
        and bool(result.stop_reason)
        and bool(result.native_costs)
        and bool(result.native_event)
    )
    outcome = _json_safe(dict(result.native_outcome))
    costs = _json_safe(
        {key: dict(value) for key, value in result.native_costs.items()}
    )
    event = _json_safe(dict(result.native_event))
    assignment = event.get("portfolioAssignmentAudit", {})
    projection = {
        "schemaVersion": "e07.s09.paired-evaluation-row.v1",
        "researchStepId": "S09",
        "panel": str(work["panel"]),
        "conditionRole": str(work["conditionRole"]),
        "parentConfigurationId": str(work["parentConfigurationId"]),
        "editId": work.get("editId"),
        "configurationId": str(configuration["configurationId"]),
        "taskId": task_id,
        "split": "train",
        "scenarioFamilyOrdinal": int(work["scenarioFamilyOrdinal"]),
        "scenarioId": record.scenario_id,
        "scenarioCommitmentSha256": canonical_sha256(
            "E07/S09/scenario/v1",
            {
                "scenarioId": record.scenario_id,
                "taskId": task_id,
                "family": int(work["scenarioFamilyOrdinal"]),
                "panel": str(work["panel"]),
                "parameters": dict(record.public_parameters),
            },
        ),
        "pairId": str(work["pairId"]),
        "actionSha256": action.policy_sha256,
        "memberPolicySha256": [
            item["policySha256"] for item in configuration["members"]
        ],
        "stopReason": result.stop_reason,
        "censored": bool(result.censored),
        "failed": bool(result.failed),
        "replayPass": bool(result.replay_pass),
        "nativeContractPass": bool(contract_pass),
        "nativeContractErrors": false_validation,
        "endpointProjection": _endpoint_projection(outcome),
        "finalStateSha256": outcome.get("finalStateSha256"),
        "nativeLedgerFamilies": costs,
        "nativeCostLeaves": _numeric_leaves(costs),
        "nativeEventSha256": canonical_sha256("E07/S09/native-event/v1", event),
        "assignmentCountsByPolicySha256": assignment.get(
            "assignmentCountsByPolicySha256", {}
        ),
        "memberSwitches": assignment.get("portfolioCoordinationLedger", {}).get(
            "memberSwitches", 0
        ),
        "memberMemoryNamespaceCount": assignment.get(
            "memberMemoryNamespaceCount", 0
        ),
        "claimBoundary": tasks[task_id].claim_boundary,
        "accessBoundary": "development_training",
        "accessAudit": broker.audit.to_dict(),
    }
    projection["resultSha256"] = canonical_sha256(
        "E07/S09/paired-evaluation-result/v1", projection
    )
    projection["elapsedSeconds"] = elapsed
    projection["workerPid"] = os.getpid()
    return projection


def make_work_roster(
    parents: Sequence[Mapping[str, Any]],
    edits: Sequence[Mapping[str, Any]],
    *,
    panel: str,
    families: Sequence[int],
) -> list[dict[str, Any]]:
    """Build one deduplicated condition roster and explicit parent/edit pairs."""

    parent_map = {row["parentConfigurationId"]: row for row in parents}
    conditions: dict[tuple[str, str], dict[str, Any]] = {}
    for edit in edits:
        parent_id = edit["parentConfigurationId"]
        parent = parent_map[parent_id]
        conditions[(parent_id, parent_id)] = {
            "conditionRole": "parent",
            "parentConfigurationId": parent_id,
            "editId": None,
            "configuration": parent["configuration"],
            "documentsByPolicySha256": parent["documentsByPolicySha256"],
            "taskId": parent["taskId"],
        }
        conditions[(parent_id, edit["editId"])] = {
            "conditionRole": (
                "control" if edit["controlKind"] else "edited"
            ),
            "parentConfigurationId": parent_id,
            "editId": edit["editId"],
            "configuration": edit["configuration"],
            "documentsByPolicySha256": edit["documentsByPolicySha256"],
            "taskId": edit["taskId"],
        }
    rows = []
    for (parent_id, condition_id), condition in sorted(conditions.items()):
        for family in families:
            rows.append(
                {
                    **deepcopy(condition),
                    "panel": panel,
                    "split": "train",
                    "scenarioFamilyOrdinal": int(family),
                    "pairId": canonical_sha256(
                        "E07/S09/pair/v1",
                        [panel, parent_id, int(family)],
                    ),
                    "conditionId": condition_id,
                }
            )
    return rows


def execute_roster(
    roster: Sequence[Mapping[str, Any]], *, workers: int = 8
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between one and eight")
    if len(
        {
            (
                row["panel"],
                row["parentConfigurationId"],
                row["conditionId"],
                row["taskId"],
                row["scenarioFamilyOrdinal"],
            )
            for row in roster
        }
    ) != len(roster):
        raise RuntimeError("S09 roster has duplicate logical work")
    results: list[dict[str, Any]] = []
    failures = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(evaluate_work_item, row): index
            for index, row in enumerate(roster)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                results.append(future.result())
            except BaseException as exc:
                failures.append(
                    {
                        "position": index,
                        "errorType": f"{type(exc).__module__}.{type(exc).__qualname__}",
                        "errorMessage": str(exc),
                    }
                )
    if failures:
        raise RuntimeError(
            "S09 fail-atomic roster failed: "
            + json.dumps(failures[:3], sort_keys=True)
        )
    results.sort(
        key=lambda row: (
            row["panel"],
            row["parentConfigurationId"],
            row["editId"] or "",
            row["taskId"],
            row["scenarioFamilyOrdinal"],
        )
    )
    return results, {
        "logicalRows": len(roster),
        "physicalRows": len(roster),
        "failedWorkerRows": 0,
        "publishedRows": len(results),
        "failAtomic": True,
        "workers": workers,
        "wallSeconds": time.perf_counter() - started,
    }


def _bootstrap_interval(
    values: np.ndarray, *, seed: int, replicates: int = 2000
) -> tuple[float, float]:
    if len(values) == 0:
        return math.nan, math.nan
    if np.all(values == values[0]):
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 250):
        count = min(250, replicates - start)
        draw = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[draw].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def _sign_flip_p(
    values: np.ndarray, *, seed: int, direction: str, replicates: int = 2000
) -> float:
    if len(values) == 0:
        return math.nan
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(replicates):
        statistic = float((values * rng.choice((-1.0, 1.0), len(values))).mean())
        if direction == "benefit":
            count += statistic >= observed
        else:
            count += statistic <= observed
    return (count + 1) / (replicates + 1)


def _holm(rows: list[dict[str, Any]], p_field: str, output_field: str) -> None:
    finite = [
        (index, float(row[p_field]))
        for index, row in enumerate(rows)
        if row.get(p_field) is not None and math.isfinite(float(row[p_field]))
    ]
    ordered = sorted(finite, key=lambda item: item[1])
    adjusted = [0.0] * len(ordered)
    running = 0.0
    total = len(ordered)
    for rank, (_index, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p_value))
        adjusted[rank] = running
    for (index, _), value in zip(ordered, adjusted, strict=True):
        rows[index][output_field] = value


def compare_paired_rows(
    rows: Sequence[Mapping[str, Any]],
    edits: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = 2000,
    sign_flip_replicates: int = 2000,
    seed: int = 70920260723,
) -> list[dict[str, Any]]:
    """Calculate task-local paired edit-minus-parent effects and guards."""

    edit_map = {row["editId"]: row for row in edits}
    grouped: dict[
        tuple[str, str, str, str], dict[str, Mapping[str, Any]]
    ] = defaultdict(dict)
    for row in rows:
        condition = row["editId"] or row["parentConfigurationId"]
        key = (
            row["panel"],
            row["parentConfigurationId"],
            row["taskId"],
            str(row["scenarioFamilyOrdinal"]),
        )
        grouped[key][condition] = row
    effects = []
    for edit_id, edit in sorted(edit_map.items()):
        parent_id = edit["parentConfigurationId"]
        relevant = [
            pair
            for (panel, pid, task, _family), pair in grouped.items()
            if pid == parent_id
            and task == edit["taskId"]
            and parent_id in pair
            and edit_id in pair
        ]
        if not relevant:
            continue
        integrity = all(
            pair[parent_id]["nativeContractPass"]
            and pair[edit_id]["nativeContractPass"]
            and pair[parent_id]["replayPass"]
            and pair[edit_id]["replayPass"]
            for pair in relevant
        )
        failed_diff = np.array(
            [
                float(pair[edit_id]["failed"]) - float(pair[parent_id]["failed"])
                for pair in relevant
            ]
        )
        censor_diff = np.array(
            [
                float(pair[edit_id]["censored"])
                - float(pair[parent_id]["censored"])
                for pair in relevant
            ]
        )
        endpoint_rows = []
        exact = True
        for pair in relevant:
            left = pair[parent_id]
            right = pair[edit_id]
            exact &= (
                left["stopReason"] == right["stopReason"]
                and left["failed"] == right["failed"]
                and left["censored"] == right["censored"]
                and left["endpointProjection"] == right["endpointProjection"]
                and left["finalStateSha256"] == right["finalStateSha256"]
            )
        for endpoint, (_path, direction, margin) in ENDPOINTS.items():
            values = []
            for pair in relevant:
                parent_value = pair[parent_id]["endpointProjection"][endpoint]
                edit_value = pair[edit_id]["endpointProjection"][endpoint]
                if parent_value is None or edit_value is None:
                    continue
                difference = float(edit_value) - float(parent_value)
                values.append(difference if direction == "maximize" else -difference)
            if not values:
                endpoint_rows.append(
                    {
                        "endpoint": endpoint,
                        "availablePairs": 0,
                        "unavailablePairs": len(relevant),
                        "direction": direction,
                        "margin": margin,
                        "meanBenefitDifference": None,
                        "ciLower": None,
                        "ciUpper": None,
                        "benefitP": None,
                        "harmP": None,
                        "equivalencePass": True,
                    }
                )
                continue
            array = np.asarray(values, dtype=float)
            endpoint_seed = int.from_bytes(
                hashlib.sha256(
                    f"{seed}:{edit_id}:{endpoint}".encode("ascii")
                ).digest()[:8],
                "big",
            )
            lower, upper = _bootstrap_interval(
                array, seed=endpoint_seed, replicates=bootstrap_replicates
            )
            endpoint_rows.append(
                {
                    "endpoint": endpoint,
                    "availablePairs": len(array),
                    "unavailablePairs": len(relevant) - len(array),
                    "direction": direction,
                    "margin": margin,
                    "meanBenefitDifference": float(array.mean()),
                    "ciLower": lower,
                    "ciUpper": upper,
                    "benefitP": _sign_flip_p(
                        array,
                        seed=endpoint_seed + 1,
                        direction="benefit",
                        replicates=sign_flip_replicates,
                    ),
                    "harmP": _sign_flip_p(
                        array,
                        seed=endpoint_seed + 2,
                        direction="harm",
                        replicates=sign_flip_replicates,
                    ),
                    "equivalencePass": lower >= -margin,
                }
            )
        bounded = (
            integrity
            and all(row["equivalencePass"] for row in endpoint_rows)
            and float(failed_diff.mean()) <= 0.0
            and float(censor_diff.mean()) <= 0.0
            and (
                not edit["causalEdit"]
                or all(
                    value <= 0
                    for key, value in edit["structuralCostDelta"].items()
                    if key
                    in {
                        "uniqueMemberCount",
                        "totalCanonicalMemberBytes",
                        "totalRuleCount",
                        "totalExpressionNodes",
                        "totalPersistentMemoryBits",
                        "totalOutboundSignalBits",
                        "selectorBranchCount",
                        "selectorExpressionNodes",
                        "selectorCanonicalBytes",
                    }
                )
            )
        )
        effects.append(
            {
                "schemaVersion": "e07.s09.ablation-effect.v1",
                "researchStepId": "S09",
                "panel": relevant[0][edit_id]["panel"],
                "parentConfigurationId": parent_id,
                "editId": edit_id,
                "variantConfigurationId": edit["variantConfigurationId"],
                "taskId": edit["taskId"],
                "category": edit["category"],
                "operator": edit["operator"],
                "causalEdit": edit["causalEdit"],
                "controlKind": edit["controlKind"],
                "pairedScenarios": len(relevant),
                "nativeContractAndReplayPass": integrity,
                "exactFunctionalEquivalence": bool(exact),
                "boundedFunctionalEquivalence": bool(bounded),
                "failureRiskDifference": float(failed_diff.mean()),
                "censorRiskDifference": float(censor_diff.mean()),
                "causalHarmBeyondMargin": False,
                "structuralBytesReduced": edit["structuralBytesReduced"],
                "structuralCostDelta": edit["structuralCostDelta"],
                "endpointEffects": endpoint_rows,
                "globalMinimalityClaimPermitted": False,
            }
        )
    multiplicity_families: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for effect in effects:
        multiplicity_families[
            (effect["parentConfigurationId"], effect["category"])
        ].extend(effect["endpointEffects"])
    for family_rows in multiplicity_families.values():
        _holm(family_rows, "benefitP", "holmBenefitP")
        _holm(family_rows, "harmP", "holmHarmP")
    for effect in effects:
        effect["causalHarmBeyondMargin"] = any(
            row["availablePairs"] > 0
            and row["ciUpper"] < -row["margin"]
            and row.get("holmHarmP", 1.0) <= 0.05
            for row in effect["endpointEffects"]
        )
    return effects


def tree_hashes(paths: Sequence[str | Path]) -> dict[str, str]:
    return {str(path): tree_digest(path) for path in paths}
