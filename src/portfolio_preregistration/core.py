"""Fail-closed, outcome-free planning for heterogeneous S08 portfolios.

This module deliberately has no episode runner, result reader, archive writer,
model loader, or protected-split materializer.  It recompiles the immutable S05
candidate documents and emits structural portfolio and budget commitments only.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from src.policy_dsl import compile_policy


REPOSITORY = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPOSITORY / "configs/portfolio/s08p_portfolio_protocol.yaml"
ARTIFACT_DIR = Path("/artifacts/research_steps/S08P")
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
CHIMERA_TASK = "e07_s02_chimera_1d"
MODE_ORDER = (
    "single_policy",
    "fixed_balanced_identity",
    "random_static_identity",
    "random_dynamic_opportunity",
    "environment_conditioned",
)


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


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def checked_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if protocol.get("schemaVersion") != "e07.s08p.portfolio-preregistration.v1":
        raise RuntimeError("unexpected S08P protocol schema")
    if (
        protocol.get("researchStepId") != "S08P"
        or protocol.get("designOnly") is not True
    ):
        raise RuntimeError("S08P must remain a design-only research step")
    boundary = protocol.get("authorizationBoundary", {})
    zero_fields = (
        "trainingScenarioMaterializations",
        "validationScenarioMaterializations",
        "confirmationScenarioMaterializations",
        "episodeEvaluations",
        "portfolioArchiveMutations",
        "s05ArchiveMutations",
    )
    if boundary.get("substantivePortfolioEvaluationsAuthorized") is not False:
        raise RuntimeError("S08P unexpectedly authorizes portfolio evaluation")
    if any(int(boundary.get(field, -1)) != 0 for field in zero_fields):
        raise RuntimeError("S08P authorization boundary contains nonzero execution")
    if int(protocol["searchDesign"]["totalTrainingLogicalBudget"]) != 11008:
        raise RuntimeError("S08P training budget changed")
    return protocol


def verify_frozen_inputs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    checked = []
    for name, spec in sorted(protocol["frozenInputs"].items()):
        path = Path(spec["path"])
        actual = hash_file(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"frozen input changed: {name}")
        checked.append(
            {
                "name": name,
                "path": str(path),
                "sha256": actual,
                "bytes": path.stat().st_size,
            }
        )
    return checked


def _native_carrier(document: Mapping[str, Any]) -> str:
    if document["environment"] == "spatial2d.v1":
        return "spatial"
    permissions = set(document["permissions"])
    actions = [
        action for rule in document["rules"] for action in rule["actions"]
    ] + list(document["default"]["actions"])
    if permissions & {
        "selection.cursor_in_bounds",
        "selection.cursor_at_actor",
        "selection.target.value",
        "selection.target.stuck",
    } or any(action["kind"] in {"advance_cursor", "swap_cursor"} for action in actions):
        return "Selection"
    if "activation.side" in permissions:
        return "Bubble"
    return "Insertion"


def _semantic_groups(path: str | Path) -> dict[tuple[str, str], str]:
    audit = json.loads(Path(path).read_text(encoding="utf-8"))
    result: dict[tuple[str, str], str] = {}
    for group in audit["finiteSemanticFinal"]:
        for policy_hash in group["memberPolicySha256"]:
            result[(group["taskId"], policy_hash)] = group["boundedSemanticSha256"]
    return result


def build_candidate_eligibility_registry(
    protocol: Mapping[str, Any],
    *,
    source_order: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Freeze all 512 structural candidates without S07 arm or efficacy fields."""

    frozen = protocol["frozenInputs"]
    source_rows = (
        list(source_order)
        if source_order is not None
        else read_jsonl(frozen["s07rCandidateRegistry"]["path"])
    )
    catalog_rows = read_jsonl(frozen["s05PolicyCatalog"]["path"])
    catalog = {row["policySha256"]: row for row in catalog_rows}
    lineage_rows = read_jsonl(frozen["s05LineageLedger"]["path"])
    lineage = {row["policySha256"]: row for row in lineage_rows}
    semantics = _semantic_groups(frozen["s05DuplicateAudit"]["path"])

    forbidden_exact = {
        "armId",
        "armMembership",
        "selected",
        "inclusionProbability",
        "nativeOutcome",
        "nativeCosts",
        "elapsedSeconds",
        "objectiveValues",
    }
    result = []
    for source in source_rows:
        if forbidden_exact & set(source):
            raise RuntimeError(
                "S07 arm, selection, or efficacy field entered eligibility"
            )
        if source.get("allocationEfficacyFieldsIncluded") is not False:
            raise RuntimeError(
                "candidate projection unexpectedly contains efficacy fields"
            )
        if source.get("recoveryOrphan") is not False:
            raise RuntimeError("recovery orphan entered S08 candidate population")
        task_id = source["taskId"]
        policy_hash = source["policySha256"]
        catalog_row = catalog.get(policy_hash)
        if catalog_row is None:
            raise RuntimeError("S08 candidate lacks an authoritative policy document")
        compiled = compile_policy(catalog_row["document"])
        if compiled.policy_sha256 != policy_hash:
            raise RuntimeError("S08 candidate policy does not recompile to its hash")
        expected_environment = (
            "spatial2d.v1" if task_id.startswith("e07_s02_spatial2d_") else "line1d.v1"
        )
        if compiled.environment != expected_environment:
            raise RuntimeError("candidate environment/task mismatch")
        lineage_row = lineage.get(policy_hash, {})
        semantic_group = semantics.get(
            (task_id, policy_hash),
            canonical_hash("E07/S08P/semantic-singleton/v1", [task_id, policy_hash]),
        )
        result.append(
            {
                "schemaVersion": "e07.s08p.candidate-eligibility-row.v1",
                "researchStepId": "S08P",
                "taskId": task_id,
                "policyId": catalog_row["policyId"],
                "policySha256": policy_hash,
                "policyBodySha256": catalog_row["policyBodySha256"],
                "environment": compiled.environment,
                "nativeCarrier": _native_carrier(catalog_row["document"]),
                "origin": catalog_row["origin"],
                "family": catalog_row["family"],
                "generation": int(catalog_row["generation"]),
                "licensedCapabilityProfile": source["licensedCapabilityProfile"],
                "complexityQuartile": int(source["complexityQuartile"]),
                "complexity": compiled.complexity.to_dict(),
                "frozenS05ArchiveMemberships": list(source["archiveMemberships"]),
                "frozenS05NativeStatusStratum": source["nativeStatusStratum"],
                "boundedSemanticGroup": semantic_group,
                "lineageParents": sorted(lineage_row.get("parents", [])),
                "eligible": True,
                "eligibilityUsesOutcome": False,
                "eligibilityUsesS07ArmMembership": False,
                "eligibilityUsesRejectedModelOrEmbedding": False,
                "promotionEvidence": False,
            }
        )
    result.sort(key=lambda row: (row["taskId"], row["policySha256"]))
    counts = Counter(row["taskId"] for row in result)
    if set(counts) != set(TASK_IDS) or any(counts[task] != 64 for task in TASK_IDS):
        raise RuntimeError("S08 eligibility is not exactly 64 candidates per task")
    if len({(row["taskId"], row["policySha256"]) for row in result}) != 512:
        raise RuntimeError("S08 eligibility contains duplicate task-policy pairs")
    return result


def candidate_commitment(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "taskId",
        "policySha256",
        "policyBodySha256",
        "environment",
        "nativeCarrier",
        "licensedCapabilityProfile",
        "complexityQuartile",
        "boundedSemanticGroup",
        "lineageParents",
    )
    payload = [
        {field: row[field] for field in fields}
        for row in sorted(rows, key=lambda item: (item["taskId"], item["policySha256"]))
    ]
    return canonical_hash("E07/S08P/candidate-eligibility/v1", payload)


def _directly_related(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left["policySha256"] in set(right["lineageParents"]) or right[
        "policySha256"
    ] in set(left["lineageParents"])


def _member_is_compatible(
    candidate: Mapping[str, Any], picked: Sequence[Mapping[str, Any]]
) -> bool:
    return all(
        candidate["policySha256"] != item["policySha256"]
        and candidate["policyBodySha256"] != item["policyBodySha256"]
        and candidate["boundedSemanticGroup"] != item["boundedSemanticGroup"]
        and not _directly_related(candidate, item)
        for item in picked
    )


def _carrier_pattern(
    task_id: str, size: int, index: int, available: Sequence[str]
) -> list[str]:
    if task_id == CHIMERA_TASK:
        if size == 2:
            return ["Bubble", "Insertion"]
        if size == 3:
            return (
                ["Bubble", "Bubble", "Insertion"]
                if index % 2 == 0
                else ["Bubble", "Insertion", "Insertion"]
            )
        return ["Bubble", "Bubble", "Insertion", "Insertion"]
    carriers = [carrier for carrier in available if carrier]
    carrier = carriers[index % len(carriers)]
    return [carrier] * size


def _choose_member_set(
    task_id: str,
    task_rows: Sequence[Mapping[str, Any]],
    *,
    size: int,
    index: int,
    prior_sets: set[tuple[str, ...]],
) -> tuple[list[Mapping[str, Any]], str]:
    by_carrier: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in task_rows:
        by_carrier[row["nativeCarrier"]].append(row)
    available = sorted(
        carrier for carrier, values in by_carrier.items() if len(values) >= size
    )
    if task_id == CHIMERA_TASK:
        available = ["Bubble", "Insertion"]
    if not available:
        raise RuntimeError(f"no size-{size} carrier-compatible set for {task_id}")
    strategy = "descriptor_complementary" if index < 4 else "counter_structural"
    for retry in range(128):
        pattern = _carrier_pattern(task_id, size, index + retry, available)
        picked: list[Mapping[str, Any]] = []
        covered_memberships: set[str] = set()
        covered_quartiles: set[int] = set()
        covered_origins: set[str] = set()
        covered_families: set[str] = set()
        for slot, carrier in enumerate(pattern):
            options = [
                row for row in by_carrier[carrier] if _member_is_compatible(row, picked)
            ]
            if not options:
                break

            def rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
                counter = canonical_hash(
                    "E07/S08P/member-choice/v1",
                    {
                        "taskId": task_id,
                        "size": size,
                        "index": index,
                        "retry": retry,
                        "slot": slot,
                        "policySha256": row["policySha256"],
                    },
                )
                if strategy == "descriptor_complementary":
                    new_cells = len(
                        set(row["frozenS05ArchiveMemberships"]) - covered_memberships
                    )
                    return (
                        -new_cells,
                        -(int(row["complexityQuartile"] not in covered_quartiles)),
                        -(int(row["origin"] not in covered_origins)),
                        -(int(row["family"] not in covered_families)),
                        counter,
                        row["policySha256"],
                    )
                return (counter, row["policySha256"])

            chosen = min(options, key=rank)
            picked.append(chosen)
            covered_memberships.update(chosen["frozenS05ArchiveMemberships"])
            covered_quartiles.add(int(chosen["complexityQuartile"]))
            covered_origins.add(str(chosen["origin"]))
            covered_families.add(str(chosen["family"]))
        if len(picked) != size:
            continue
        key = tuple(sorted(item["policySha256"] for item in picked))
        if key in prior_sets:
            continue
        prior_sets.add(key)
        return picked, strategy
    raise RuntimeError(f"could not build unique legal member set for {task_id}")


def _portfolio_costs(
    members: Sequence[Mapping[str, Any]], selector: Mapping[str, Any] | None
) -> dict[str, int]:
    complexity = [item["complexity"] for item in members]
    return {
        "uniqueMemberCount": len(members),
        "totalCanonicalMemberBytes": sum(
            int(item["canonicalBytes"]) for item in complexity
        ),
        "totalRuleCount": sum(int(item["ruleCount"]) for item in complexity),
        "totalExpressionNodes": sum(
            int(item["expressionNodes"]) for item in complexity
        ),
        "totalPersistentMemoryBits": sum(
            int(item["persistentMemoryBits"]) for item in complexity
        ),
        "totalOutboundSignalBits": sum(
            int(item["outboundSignalBits"]) for item in complexity
        ),
        "selectorBranchCount": 2 if selector else 0,
        "selectorExpressionNodes": 3 if selector else 0,
        "selectorCanonicalBytes": len(canonical_bytes(selector)) if selector else 0,
    }


def _configuration(
    task_id: str,
    mode: str,
    members: Sequence[Mapping[str, Any]],
    *,
    member_set_id: str,
    selector_signal: str | None = None,
) -> dict[str, Any]:
    selector = None
    if mode == "environment_conditioned":
        if selector_signal is None:
            raise RuntimeError("conditioned configuration lacks a selector signal")
        selector = {
            "profile": "bounded_memoryless_two_branch_v1",
            "signal": selector_signal,
            "operator": "eq" if selector_signal == "last_action.rejected" else "gt",
            "threshold": True if selector_signal == "last_action.rejected" else 0,
            "trueBranch": "next_compatible_member",
            "falseBranch": "current_or_first_compatible_member",
            "persistentMemoryBits": 0,
        }
    member_projection = [
        {
            "policyId": item["policyId"],
            "policySha256": item["policySha256"],
            "policyBodySha256": item["policyBodySha256"],
            "nativeCarrier": item["nativeCarrier"],
            "boundedSemanticGroup": item["boundedSemanticGroup"],
        }
        for item in sorted(
            members, key=lambda row: (row["nativeCarrier"], row["policySha256"])
        )
    ]
    payload = {
        "taskId": task_id,
        "mode": mode,
        "memberSetId": member_set_id,
        "members": member_projection,
        "selector": selector,
        "assignmentCounterDomain": (
            None
            if mode == "single_policy"
            else f"E07/S08/{mode}/identity-opportunity/v1"
        ),
    }
    config_id = canonical_hash("E07/S08P/portfolio-configuration/v1", payload)
    return {
        "schemaVersion": "e07.s08p.portfolio-seed-row.v1",
        "researchStepId": "S08P",
        "taskId": task_id,
        "configurationId": config_id,
        **payload,
        "portfolioSize": len(members),
        "portfolioStructuralCosts": _portfolio_costs(members, selector),
        "evaluationAuthorized": False,
        "outcomeMaterialized": False,
        "s07ArmMembershipUsed": False,
        "rejectedModelOrEmbeddingUsed": False,
    }


def build_portfolio_seed_registry(
    protocol: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_task[row["taskId"]].append(row)
    member_sets: list[dict[str, Any]] = []
    configurations: list[dict[str, Any]] = []
    selector_by_task = protocol["selectorSignals"]["byTask"]
    for task_id in TASK_IDS:
        task_rows = sorted(by_task[task_id], key=lambda row: row["policySha256"])
        for row in task_rows:
            member_set_id = canonical_hash(
                "E07/S08P/single-member-set/v1", [task_id, row["policySha256"]]
            )
            configurations.append(
                _configuration(
                    task_id, "single_policy", [row], member_set_id=member_set_id
                )
            )
        prior_sets: set[tuple[str, ...]] = set()
        task_sets = []
        for size in (2, 3, 4):
            for index in range(8):
                picked, strategy = _choose_member_set(
                    task_id,
                    task_rows,
                    size=size,
                    index=index,
                    prior_sets=prior_sets,
                )
                member_hashes = sorted(item["policySha256"] for item in picked)
                member_set_id = canonical_hash(
                    "E07/S08P/member-set/v1", [task_id, member_hashes]
                )
                set_row = {
                    "schemaVersion": "e07.s08p.member-set-row.v1",
                    "researchStepId": "S08P",
                    "taskId": task_id,
                    "memberSetId": member_set_id,
                    "portfolioSize": size,
                    "seedOrdinalWithinSize": index,
                    "selectionStrategy": strategy,
                    "members": [
                        {
                            "policySha256": item["policySha256"],
                            "nativeCarrier": item["nativeCarrier"],
                            "boundedSemanticGroup": item["boundedSemanticGroup"],
                        }
                        for item in sorted(
                            picked,
                            key=lambda row: (row["nativeCarrier"], row["policySha256"]),
                        )
                    ],
                    "frozenArchiveCellsCovered": sorted(
                        {
                            cell
                            for item in picked
                            for cell in item["frozenS05ArchiveMemberships"]
                        }
                    ),
                    "efficacyRankUsed": False,
                    "s07ArmMembershipUsed": False,
                }
                member_sets.append(set_row)
                task_sets.append((set_row, picked))
                for mode in (
                    "fixed_balanced_identity",
                    "random_static_identity",
                    "random_dynamic_opportunity",
                ):
                    configurations.append(
                        _configuration(
                            task_id, mode, picked, member_set_id=member_set_id
                        )
                    )
        conditioned = []
        for set_row, picked in task_sets:
            carrier_counts = Counter(item["nativeCarrier"] for item in picked)
            if max(carrier_counts.values()) >= 2:
                conditioned.append((set_row, picked))
        if len(conditioned) < 16:
            raise RuntimeError(f"insufficient conditioned member sets for {task_id}")
        signals = list(selector_by_task[task_id])
        for ordinal, (set_row, picked) in enumerate(conditioned[:16]):
            configurations.append(
                _configuration(
                    task_id,
                    "environment_conditioned",
                    picked,
                    member_set_id=set_row["memberSetId"],
                    selector_signal=signals[ordinal % len(signals)],
                )
            )
    configurations.sort(
        key=lambda row: (
            TASK_IDS.index(row["taskId"]),
            MODE_ORDER.index(row["mode"]),
            row["configurationId"],
        )
    )
    member_sets.sort(
        key=lambda row: (
            TASK_IDS.index(row["taskId"]),
            row["portfolioSize"],
            row["seedOrdinalWithinSize"],
        )
    )
    return member_sets, configurations


def validate_portfolio_registry(
    protocol: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    member_sets: Sequence[Mapping[str, Any]],
    configurations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_map = {(row["taskId"], row["policySha256"]): row for row in candidates}
    errors = []
    for row in configurations:
        members = [
            candidate_map.get((row["taskId"], item["policySha256"]))
            for item in row["members"]
        ]
        if any(item is None for item in members):
            errors.append(f"{row['configurationId']}: absent candidate")
            continue
        members = [item for item in members if item is not None]
        hashes = [item["policySha256"] for item in members]
        bodies = [item["policyBodySha256"] for item in members]
        semantic = [item["boundedSemanticGroup"] for item in members]
        if (
            len(set(hashes)) != len(hashes)
            or len(set(bodies)) != len(bodies)
            or len(set(semantic)) != len(semantic)
        ):
            errors.append(f"{row['configurationId']}: duplicate member")
        if any(
            _directly_related(left, right)
            for index, left in enumerate(members)
            for right in members[index + 1 :]
        ):
            errors.append(f"{row['configurationId']}: direct lineage relation")
        if row["mode"] != "single_policy":
            carriers = {item["nativeCarrier"] for item in members}
            if row["taskId"] == CHIMERA_TASK:
                if not {"Bubble", "Insertion"}.issubset(carriers):
                    errors.append(f"{row['configurationId']}: chimera carrier coverage")
            elif len(carriers) != 1:
                errors.append(f"{row['configurationId']}: carrier mixing")
        selector = row.get("selector")
        if row["mode"] == "environment_conditioned":
            allowed = set(protocol["selectorSignals"]["byTask"][row["taskId"]])
            if not selector or selector.get("signal") not in allowed:
                errors.append(f"{row['configurationId']}: illegal selector")
        elif selector is not None:
            errors.append(f"{row['configurationId']}: unexpected selector")
        if row["evaluationAuthorized"] or row["outcomeMaterialized"]:
            errors.append(f"{row['configurationId']}: design-only boundary violated")
    config_counts = Counter(row["taskId"] for row in configurations)
    mode_counts = {
        task: Counter(row["mode"] for row in configurations if row["taskId"] == task)
        for task in TASK_IDS
    }
    expected_modes = protocol["searchDesign"]["initialConfigurationsPerTask"]
    for task_id in TASK_IDS:
        if config_counts[task_id] != int(expected_modes["total"]):
            errors.append(f"{task_id}: configuration count")
        for mode in MODE_ORDER:
            if mode_counts[task_id][mode] != int(expected_modes[mode]):
                errors.append(f"{task_id}: {mode} count")
    if len(member_sets) != 8 * 24:
        errors.append("member-set count")
    if len({row["configurationId"] for row in configurations}) != len(configurations):
        errors.append("duplicate configuration ID")
    if errors:
        raise RuntimeError(
            "portfolio legality validation failed: " + "; ".join(errors[:10])
        )
    return {
        "schemaVersion": "e07.s08p.composition-legality-validation.v1",
        "researchStepId": "S08P",
        "success": True,
        "candidateRows": len(candidates),
        "memberSetRows": len(member_sets),
        "configurationRows": len(configurations),
        "configurationsPerTask": dict(sorted(config_counts.items())),
        "modeCountsByTask": {
            task: dict(sorted(counts.items())) for task, counts in mode_counts.items()
        },
        "canonicalDuplicateConfigurations": 0,
        "illegalCompositions": 0,
        "forbiddenSelectorSignals": 0,
        "s07ArmMembershipUses": 0,
        "outcomeMaterializations": 0,
    }


def build_budget_slots(
    protocol: Mapping[str, Any], configurations: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    families = [
        int(value) for value in protocol["searchDesign"]["scenarioFamilies"]["ordinals"]
    ]
    rows: list[dict[str, Any]] = []
    by_task_mode: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for config in configurations:
        by_task_mode[(config["taskId"], config["mode"])].append(config)
        for family in families:
            rows.append(
                {
                    "schemaVersion": "e07.s08p.budget-slot.v1",
                    "researchStepId": "S08P",
                    "stage": "initial",
                    "taskId": config["taskId"],
                    "generation": 0,
                    "configurationRole": config["mode"],
                    "configurationId": config["configurationId"],
                    "pairedSlotId": None,
                    "scenarioFamilyOrdinal": family,
                    "split": "train",
                    "smoke": False,
                    "evaluationAuthorized": False,
                    "outcomeMaterialized": False,
                }
            )
    smoke_ids = {
        min(by_task_mode[(task, mode)], key=lambda row: row["configurationId"])[
            "configurationId"
        ]
        for task in TASK_IDS
        for mode in MODE_ORDER
    }
    for row in rows:
        row["smoke"] = (
            row["scenarioFamilyOrdinal"] == families[0]
            and row["configurationId"] in smoke_ids
        )

    adaptive = protocol["searchDesign"]["adaptiveSearch"]
    for generation in range(1, int(adaptive["generations"]) + 1):
        for task_id in TASK_IDS:
            for ordinal in range(int(adaptive["targetPortfoliosPerTaskGeneration"])):
                pair_id = canonical_hash(
                    "E07/S08P/adaptive-pair-slot/v1",
                    [task_id, generation, ordinal],
                )
                for role in ("target_portfolio", "matched_random_dynamic"):
                    placeholder_id = canonical_hash(
                        "E07/S08P/adaptive-configuration-slot/v1",
                        [task_id, generation, ordinal, role],
                    )
                    for family in families:
                        rows.append(
                            {
                                "schemaVersion": "e07.s08p.budget-slot.v1",
                                "researchStepId": "S08P",
                                "stage": "adaptive",
                                "taskId": task_id,
                                "generation": generation,
                                "configurationRole": role,
                                "configurationId": placeholder_id,
                                "pairedSlotId": pair_id,
                                "scenarioFamilyOrdinal": family,
                                "split": "train",
                                "smoke": False,
                                "evaluationAuthorized": False,
                                "outcomeMaterialized": False,
                            }
                        )
    for index, row in enumerate(rows):
        row["logicalSlotOrdinal"] = index
        row["logicalSlotId"] = canonical_hash(
            "E07/S08P/logical-budget-slot/v1",
            {
                "stage": row["stage"],
                "taskId": row["taskId"],
                "generation": row["generation"],
                "configurationRole": row["configurationRole"],
                "configurationId": row["configurationId"],
                "scenarioFamilyOrdinal": row["scenarioFamilyOrdinal"],
            },
        )
    expected = int(protocol["searchDesign"]["totalTrainingLogicalBudget"])
    if len(rows) != expected:
        raise RuntimeError(f"budget ledger has {len(rows)} rows, expected {expected}")
    if sum(row["smoke"] for row in rows) != int(
        protocol["searchDesign"]["smoke"]["rows"]
    ):
        raise RuntimeError("smoke is not exactly 40 within-budget rows")
    if any(row["split"] != "train" or row["evaluationAuthorized"] for row in rows):
        raise RuntimeError("budget ledger violates design-only training boundary")
    return rows


def plan_digest(
    candidates: Sequence[Mapping[str, Any]],
    member_sets: Sequence[Mapping[str, Any]],
    configurations: Sequence[Mapping[str, Any]],
    budget_slots: Sequence[Mapping[str, Any]],
) -> str:
    return canonical_hash(
        "E07/S08P/complete-plan/v1",
        {
            "candidateCommitment": candidate_commitment(candidates),
            "memberSets": [row["memberSetId"] for row in member_sets],
            "configurations": [row["configurationId"] for row in configurations],
            "budgetSlots": [row["logicalSlotId"] for row in budget_slots],
        },
    )


def iter_policy_hashes(configurations: Iterable[Mapping[str, Any]]) -> Iterable[str]:
    for row in configurations:
        for member in row["members"]:
            yield str(member["policySha256"])
