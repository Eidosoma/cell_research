"""Fail-closed, non-surrogate allocation design for E07 S07R.

This module can summarize the frozen S05 candidate population and, only after
an explicit future approval token, create an S07 allocation roster. It has no
episode runner, model loader, embedding reader, archive writer, or protected
split materializer. S07R uses the roster generator only on synthetic fixtures
to validate determinism; no real roster is generated in S07R.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from src.policy_dsl import compile_policy


REPOSITORY = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPOSITORY / "configs/allocation/s07r_non_surrogate_protocol.yaml"
FUTURE_APPROVAL_TOKEN = "S07_EXECUTION_SEPARATELY_APPROVED"
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
    if protocol.get("schemaVersion") != "e07.s07r.non-surrogate-allocation-protocol.v1":
        raise RuntimeError("unexpected S07R protocol schema")
    if (
        protocol.get("researchStepId") != "S07R"
        or protocol.get("designOnly") is not True
    ):
        raise RuntimeError("S07R protocol must remain design-only")
    gate = protocol.get("S07ExecutionApprovalGate", {})
    if gate.get("approved") is not False or gate.get("episodesEvaluated") != 0:
        raise RuntimeError("S07R protocol unexpectedly authorizes execution")
    if gate.get("allocationRosterWritten") is not False:
        raise RuntimeError("S07R protocol unexpectedly authorizes a roster")
    return protocol


def _contains_cursor_action(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("kind") in {"advance_cursor", "swap_cursor"}:
            return True
        return any(_contains_cursor_action(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_cursor_action(child) for child in value)
    return False


def _capability_profile(document: Mapping[str, Any]) -> str:
    permissions = set(document["permissions"])
    if "line.prefix_ordered" in permissions:
        return "insertion_prefix"
    if any(
        item.startswith("selection.") for item in permissions
    ) or _contains_cursor_action(document):
        return "selection_cursor_long_range"
    if document["environment"] == "spatial2d.v1":
        return "spatial_candidate"
    return "local_line"


def _status_stratum(rows: Sequence[Mapping[str, Any]]) -> str:
    signatures = {
        (bool(row["failed"]), bool(row["censored"]), str(row["stopReason"]))
        for row in rows
    }
    if any(bool(row["failed"]) for row in rows):
        return "failure_observed"
    if any(bool(row["censored"]) for row in rows):
        return "censor_observed"
    if len(signatures) > 1:
        return "mixed_terminal_signature"
    return "single_terminal_signature"


def _status_signature(row: Mapping[str, Any]) -> str:
    return f"failed={int(bool(row['failed']))}|censored={int(bool(row['censored']))}|stop={row['stopReason']}"


def _complexity_quartiles(rows: list[dict[str, Any]]) -> None:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["taskId"]].append(row)
    for task_id, task_rows in by_task.items():
        if len(task_rows) != 64:
            raise RuntimeError(f"{task_id} does not have 64 candidate rows")
        ordered = sorted(
            task_rows,
            key=lambda item: (
                item["complexity"]["worstCaseOperations"],
                item["complexity"]["canonicalBytes"],
                item["policySha256"],
            ),
        )
        for rank, row in enumerate(ordered):
            row["complexityQuartile"] = rank // 16
            row["complexityRankWithinTask"] = rank


def build_candidate_registry(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project authorized S05 rows to an outcome-free allocation registry.

    Only native failure/censor/terminal status is retained from the evaluation
    ledger. Task objectives, native costs, elapsed time, and event payloads are
    deliberately absent from the returned rows.
    """

    sources = protocol["authorizedPopulation"]
    for key in ("baseEvaluationLedger", "policyCatalog", "archiveEntries"):
        spec = sources[key]
        if hash_file(spec["path"]) != spec["sha256"]:
            raise RuntimeError(f"frozen S05 source changed: {key}")

    evaluation_rows = read_jsonl(sources["baseEvaluationLedger"]["path"])
    selected = [row for row in evaluation_rows if row["selectedForCanonicalSearch"]]
    orphans = [row for row in evaluation_rows if not row["selectedForCanonicalSearch"]]
    if len(orphans) != int(sources["recoveryOrphansExcluded"]):
        raise RuntimeError("S05 recovery-orphan count changed")
    if any(row["split"] != "train" for row in selected):
        raise RuntimeError("nontraining row in S07 candidate population")

    catalog_rows = read_jsonl(sources["policyCatalog"]["path"])
    catalog = {row["policySha256"]: row for row in catalog_rows}
    archive_rows = read_jsonl(sources["archiveEntries"]["path"])
    memberships: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in archive_rows:
        membership = (
            f"{row['archiveId']}|{canonical_bytes(row['cell']).decode('ascii')}"
        )
        memberships[(row["taskId"], row["policySha256"])].add(membership)

    panels: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        panels[(row["taskId"], row["policySha256"])].append(row)
    expected_pairs = int(sources["candidateCount"])
    if len(panels) != expected_pairs:
        raise RuntimeError("S05 selected task-policy population changed")

    result: list[dict[str, Any]] = []
    for (task_id, policy_hash), panel in sorted(panels.items()):
        if policy_hash not in catalog:
            raise RuntimeError("selected S05 policy lacks a catalog document")
        source = catalog[policy_hash]
        compiled = compile_policy(source["document"])
        if compiled.policy_sha256 != policy_hash:
            raise RuntimeError("catalog policy hash does not recompile")
        status_counts = Counter(_status_signature(row) for row in panel)
        archive_memberships = sorted(memberships[(task_id, policy_hash)])
        result.append(
            {
                "schemaVersion": "e07.s07r.candidate-population-row.v1",
                "researchStepId": "S07R",
                "taskId": task_id,
                "policyId": source["policyId"],
                "policySha256": policy_hash,
                "policyBodySha256": source["policyBodySha256"],
                "origin": source["origin"],
                "family": source["family"],
                "generation": int(source["generation"]),
                "licensedCapabilityProfile": _capability_profile(source["document"]),
                "nativeStatusStratum": _status_stratum(panel),
                "nativeStatusSignatureCounts": dict(sorted(status_counts.items())),
                "archiveMemberships": archive_memberships,
                "archiveMembershipCount": len(archive_memberships),
                "archived": bool(archive_memberships),
                "complexity": compiled.complexity.to_dict(),
                "allocationEfficacyFieldsIncluded": False,
                "recoveryOrphan": False,
            }
        )
    _complexity_quartiles(result)
    by_task = Counter(row["taskId"] for row in result)
    if set(by_task) != set(TASK_IDS) or any(value != 64 for value in by_task.values()):
        raise RuntimeError("candidate registry is not 64 policies for each of 8 tasks")
    return sorted(result, key=lambda row: (row["taskId"], row["policySha256"]))


def candidate_population_commitment(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_hash(
        "E07/S07R/candidate-population/v1",
        [
            {
                "taskId": row["taskId"],
                "policySha256": row["policySha256"],
                "status": row["nativeStatusStratum"],
                "capability": row["licensedCapabilityProfile"],
                "complexityQuartile": row["complexityQuartile"],
                "archiveMemberships": row["archiveMemberships"],
            }
            for row in sorted(
                rows, key=lambda item: (item["taskId"], item["policySha256"])
            )
        ],
    )


def derive_rare_status_registry(
    candidate_rows: Sequence[Mapping[str, Any]], threshold: float = 0.10
) -> dict[str, Any]:
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for row in candidate_rows:
        by_task[row["taskId"]].update(row["nativeStatusSignatureCounts"])
    tasks: dict[str, Any] = {}
    for task_id in TASK_IDS:
        counts = by_task[task_id]
        total = sum(counts.values())
        signatures = []
        for signature, count in sorted(counts.items()):
            prevalence = count / total
            signatures.append(
                {
                    "signature": signature,
                    "count": count,
                    "denominator": total,
                    "prevalence": prevalence,
                    "rareByFrozenRule": prevalence <= threshold
                    or signature.startswith("failed=1|"),
                }
            )
        tasks[task_id] = {"evaluationRows": total, "signatures": signatures}
    return {
        "schemaVersion": "e07.s07r.rare-status-registry.v1",
        "researchStepId": "S07R",
        "source": "S05 selected training rows only",
        "prevalenceThreshold": threshold,
        "unseenFutureSignaturesAreRare": True,
        "failedFutureEpisodesAreRare": True,
        "tasks": tasks,
    }


def build_scenario_registry(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    scenario = protocol["scenarioPopulation"]
    rows = []
    for task_id in TASK_IDS:
        for ordinal in scenario["familyOrdinals"]:
            rows.append(
                {
                    "schemaVersion": "e07.s07r.scenario-family-template.v1",
                    "researchStepId": "S07R",
                    "taskId": task_id,
                    "familyOrdinal": int(ordinal),
                    "split": "train",
                    "protected": False,
                    "derivationDomain": scenario["derivationDomain"],
                    "materialized": False,
                    "outcomeRead": False,
                    "templateCommitmentSha256": canonical_hash(
                        "E07/S07R/scenario-template/v1",
                        {
                            "taskId": task_id,
                            "familyOrdinal": int(ordinal),
                            "split": "train",
                            "derivationDomain": scenario["derivationDomain"],
                        },
                    ),
                }
            )
    if len(rows) != int(scenario["taskFamilyTemplateCount"]):
        raise RuntimeError("scenario template count changed")
    return rows


def _marginal_balance_select(
    candidates: Sequence[Mapping[str, Any]], count: int, fields: Sequence[str]
) -> list[Mapping[str, Any]]:
    remaining = list(candidates)
    selected: list[Mapping[str, Any]] = []
    field_counts: dict[str, Counter[Any]] = {field: Counter() for field in fields}
    while len(selected) < count:
        if not remaining:
            raise RuntimeError("fixed stratum quota is infeasible")

        def score(row: Mapping[str, Any]) -> tuple[Fraction, str]:
            total = sum(
                (Fraction(1, 1 + field_counts[field][row[field]]) for field in fields),
                start=Fraction(0, 1),
            )
            return total, row["policySha256"]

        best_score = max(score(row)[0] for row in remaining)
        best = min(
            (row for row in remaining if score(row)[0] == best_score),
            key=lambda row: row["policySha256"],
        )
        selected.append(best)
        remaining.remove(best)
        for field in fields:
            field_counts[field][best[field]] += 1
    return selected


def _uniform_select(
    task_rows: Sequence[Mapping[str, Any]], arm: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    seed = int(arm["publicSeed"])
    domain = str(arm["counterDomain"])
    ranked = sorted(
        task_rows,
        key=lambda row: (
            canonical_hash(
                domain,
                {
                    "seed": seed,
                    "taskId": row["taskId"],
                    "policySha256": row["policySha256"],
                },
            ),
            row["policySha256"],
        ),
    )
    return ranked[: int(arm["candidatesPerTask"])]


def _descriptor_select(
    task_rows: Sequence[Mapping[str, Any]], arm: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    archived = [row for row in task_rows if row["archived"]]
    unarchived = [row for row in task_rows if not row["archived"]]
    archived_quota = int(arm["archivedQuotaPerTask"])
    unarchived_quota = int(arm["unarchivedQuotaPerTask"])
    if len(archived) < archived_quota or len(unarchived) < unarchived_quota:
        raise RuntimeError("descriptor archived/unarchived quota is infeasible")
    covered: set[str] = set()
    selected: list[Mapping[str, Any]] = []
    remaining = list(archived)
    while len(selected) < archived_quota:
        best = min(
            remaining,
            key=lambda row: (
                -len(set(row["archiveMemberships"]) - covered),
                len(set(row["archiveMemberships"]) & covered),
                row["policySha256"],
            ),
        )
        selected.append(best)
        covered.update(best["archiveMemberships"])
        remaining.remove(best)
    selected.extend(
        _marginal_balance_select(
            unarchived,
            unarchived_quota,
            ("licensedCapabilityProfile", "nativeStatusStratum", "complexityQuartile"),
        )
    )
    return selected


def _contract_select(
    task_rows: Sequence[Mapping[str, Any]], arm: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    enriched = []
    for row in task_rows:
        item = dict(row)
        item["originFamily"] = f"{row['origin']}|{row['family']}"
        enriched.append(item)
    return _marginal_balance_select(
        enriched,
        int(arm["candidatesPerTask"]),
        (
            "licensedCapabilityProfile",
            "nativeStatusStratum",
            "complexityQuartile",
            "originFamily",
        ),
    )


def build_allocation_roster(
    candidate_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    approval_token: str | None = None,
) -> list[dict[str, Any]]:
    """Build the future S07 logical roster only with separate approval.

    S07R never supplies the approval token for the real candidate population.
    """

    if approval_token != FUTURE_APPROVAL_TOKEN:
        raise PermissionError(
            "S07 allocation roster requires separate execution approval"
        )
    selection_frame = build_candidate_selection_frame(
        candidate_rows, protocol, approval_token=approval_token
    )
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_task[row["taskId"]].append(row)
    candidate_lookup = {
        (row["taskId"], row["policySha256"]): row for row in candidate_rows
    }
    selected_by_arm: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for frame_row in selection_frame:
        if frame_row["selected"]:
            selected_by_arm[frame_row["armId"]][frame_row["taskId"]].append(
                candidate_lookup[(frame_row["taskId"], frame_row["policySha256"])]
            )

    logical: list[dict[str, Any]] = []
    for family in protocol["scenarioPopulation"]["familyOrdinals"]:
        for arm_id in sorted(selected_by_arm):
            for task_id in sorted(by_task):
                chosen = selected_by_arm[arm_id][task_id]
                ranked = sorted(
                    chosen,
                    key=lambda row: (
                        canonical_hash(
                            "E07/S07/non-surrogate-logical-order/v1",
                            {
                                "armId": arm_id,
                                "taskId": task_id,
                                "family": family,
                                "policySha256": row["policySha256"],
                            },
                        ),
                        row["policySha256"],
                    ),
                )
                chosen_hashes = {row["policySha256"] for row in chosen}
                for rank, row in enumerate(ranked):
                    inclusion = (
                        0.5
                        if arm_id == "uniform_randomized"
                        else (1.0 if row["policySha256"] in chosen_hashes else 0.0)
                    )
                    physical_id = canonical_hash(
                        "E07/S07/physical-row/v1",
                        {
                            "taskId": task_id,
                            "policySha256": row["policySha256"],
                            "familyOrdinal": family,
                        },
                    )
                    payload = {
                        "armId": arm_id,
                        "taskId": task_id,
                        "policySha256": row["policySha256"],
                        "scenarioFamilyOrdinal": int(family),
                        "selectionRank": rank,
                        "inclusionProbability": inclusion,
                        "deterministicSelectionIndicator": (
                            None if arm_id == "uniform_randomized" else 1
                        ),
                        "randomizedSelectionIndicator": (
                            1 if arm_id == "uniform_randomized" else None
                        ),
                        "selectionMechanismDeterministic": arm_id
                        != "uniform_randomized",
                        "selectionStrata": {
                            "licensedCapabilityProfile": row[
                                "licensedCapabilityProfile"
                            ],
                            "nativeStatusStratum": row["nativeStatusStratum"],
                            "complexityQuartile": row["complexityQuartile"],
                            "archived": row["archived"],
                        },
                        "physicalRowId": physical_id,
                    }
                    payload["logicalRowId"] = canonical_hash(
                        "E07/S07/logical-row/v1", payload
                    )
                    logical.append(payload)
    expected = int(protocol["budgetAndStopping"]["totalLogicalEvaluations"])
    if len(logical) != expected:
        raise RuntimeError("logical allocation budget mismatch")
    overlaps: dict[str, set[str]] = defaultdict(set)
    for row in logical:
        overlaps[row["physicalRowId"]].add(row["armId"])
    for row in logical:
        row["overlapArmIds"] = sorted(overlaps[row["physicalRowId"]])
        row["reuseIndicator"] = len(row["overlapArmIds"]) > 1
    return logical


def build_candidate_selection_frame(
    candidate_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    approval_token: str | None = None,
) -> list[dict[str, Any]]:
    """Create the complete future 3 x 512 candidate selection frame."""

    if approval_token != FUTURE_APPROVAL_TOKEN:
        raise PermissionError(
            "S07 allocation roster requires separate execution approval"
        )
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_task[row["taskId"]].append(row)
    arms = protocol["arms"]
    selectors = {
        "uniform_randomized": lambda rows: _uniform_select(
            rows, arms["uniform_randomized"]
        ),
        "descriptor_cell_coverage": lambda rows: _descriptor_select(
            rows, arms["descriptor_cell_coverage"]
        ),
        "contract_status_stratified": lambda rows: _contract_select(
            rows, arms["contract_status_stratified"]
        ),
    }
    frame = []
    for arm_id, selector in sorted(selectors.items()):
        for task_id, task_rows in sorted(by_task.items()):
            selected = selector(task_rows)
            ranks = {row["policySha256"]: rank for rank, row in enumerate(selected)}
            for row in sorted(task_rows, key=lambda item: item["policySha256"]):
                indicator = row["policySha256"] in ranks
                probability = (
                    0.5
                    if arm_id == "uniform_randomized"
                    else (1.0 if indicator else 0.0)
                )
                payload = {
                    "armId": arm_id,
                    "taskId": task_id,
                    "policySha256": row["policySha256"],
                    "selected": indicator,
                    "selectionRank": ranks.get(row["policySha256"]),
                    "inclusionProbability": probability,
                    "deterministicSelectionIndicator": (
                        None if arm_id == "uniform_randomized" else int(indicator)
                    ),
                    "randomizedSelectionIndicator": (
                        int(indicator) if arm_id == "uniform_randomized" else None
                    ),
                    "selectionMechanismDeterministic": arm_id != "uniform_randomized",
                    "selectionStrata": {
                        "licensedCapabilityProfile": row["licensedCapabilityProfile"],
                        "nativeStatusStratum": row["nativeStatusStratum"],
                        "complexityQuartile": row["complexityQuartile"],
                        "archived": row["archived"],
                    },
                }
                payload["selectionFrameRowId"] = canonical_hash(
                    "E07/S07/candidate-selection-frame/v1", payload
                )
                frame.append(payload)
    expected = len(protocol["arms"]) * int(
        protocol["authorizedPopulation"]["candidateCount"]
    )
    if len(frame) != expected:
        raise RuntimeError("candidate selection frame budget mismatch")
    return frame


def feasibility_summary(
    candidate_rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_task[row["taskId"]].append(row)
    tasks = {}
    for task_id in TASK_IDS:
        rows = by_task[task_id]
        archived = sum(bool(row["archived"]) for row in rows)
        unarchived = len(rows) - archived
        tasks[task_id] = {
            "candidateCount": len(rows),
            "archivedCount": archived,
            "unarchivedCount": unarchived,
            "descriptorQuotaFeasible": archived >= 16 and unarchived >= 16,
            "capabilityCounts": dict(
                sorted(
                    Counter(row["licensedCapabilityProfile"] for row in rows).items()
                )
            ),
            "statusCounts": dict(
                sorted(Counter(row["nativeStatusStratum"] for row in rows).items())
            ),
            "complexityQuartileCounts": dict(
                sorted(Counter(str(row["complexityQuartile"]) for row in rows).items())
            ),
            "archiveCellUniverseCount": len(
                {membership for row in rows for membership in row["archiveMemberships"]}
            ),
        }
    budget = protocol["budgetAndStopping"]
    return {
        "schemaVersion": "e07.s07r.feasibility-summary.v1",
        "researchStepId": "S07R",
        "success": all(
            item["descriptorQuotaFeasible"] and item["candidateCount"] == 64
            for item in tasks.values()
        ),
        "candidatePopulationCount": len(candidate_rows),
        "taskCount": len(tasks),
        "scenarioTemplateCount": int(
            protocol["scenarioPopulation"]["taskFamilyTemplateCount"]
        ),
        "logicalEvaluationBudget": int(budget["totalLogicalEvaluations"]),
        "physicalEvaluationLowerBound": int(
            budget["possiblePhysicalEvaluationsLowerBound"]
        ),
        "physicalEvaluationUpperBound": int(
            budget["possiblePhysicalEvaluationsUpperBound"]
        ),
        "actualAllocationRosterRows": 0,
        "tasks": tasks,
    }


def _synthetic_candidates(
    order: Iterable[int], task_id: str = "synthetic_task"
) -> list[dict[str, Any]]:
    rows = []
    for index in order:
        archived = index % 2 == 0
        rows.append(
            {
                "taskId": task_id,
                "policySha256": f"{index:064x}",
                "licensedCapabilityProfile": (
                    "insertion_prefix" if index % 4 == 0 else "local_line"
                ),
                "nativeStatusStratum": (
                    "censor_observed" if index % 5 == 0 else "single_terminal_signature"
                ),
                "complexityQuartile": index // 16,
                "archived": archived,
                "archiveMemberships": ([f"common|[{index % 8},0]"] if archived else []),
                "origin": "synthetic",
                "family": f"family_{index % 3}",
            }
        )
    return rows


def synthetic_worker_order_validation(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Exercise all selection rules without allocating any real S05 candidate."""

    digests = []
    frame_counts = []
    roster_counts = []
    physical_counts = []
    orders = [
        list(range(64)),
        list(reversed(range(64))),
        list(range(0, 64, 2)) + list(range(1, 64, 2)),
        sorted(range(64), key=lambda value: canonical_hash("synthetic-order", value)),
    ]
    for order in orders:
        rows = [
            row
            for task_index in range(8)
            for row in _synthetic_candidates(
                order, task_id=f"synthetic_task_{task_index}"
            )
        ]
        frame = build_candidate_selection_frame(
            rows, protocol, approval_token=FUTURE_APPROVAL_TOKEN
        )
        roster = build_allocation_roster(
            rows, protocol, approval_token=FUTURE_APPROVAL_TOKEN
        )
        frame_counts.append(len(frame))
        roster_counts.append(len(roster))
        physical_counts.append(len({row["physicalRowId"] for row in roster}))
        digests.append(
            canonical_hash(
                "E07/S07R/synthetic-plan-validation/v1",
                {
                    "selectionFrame": sorted(
                        row["selectionFrameRowId"] for row in frame
                    ),
                    "logicalRoster": sorted(row["logicalRowId"] for row in roster),
                },
            )
        )
    return {
        "schemaVersion": "e07.s07r.worker-order-validation.v1",
        "researchStepId": "S07R",
        "success": (
            len(set(digests)) == 1
            and set(frame_counts) == {1536}
            and set(roster_counts) == {3072}
            and all(1024 <= count <= 3072 for count in physical_counts)
        ),
        "fixtureOnly": True,
        "realCandidateAllocationPerformed": False,
        "workerOrdersTested": len(orders),
        "syntheticCandidateSelectionFrameRows": frame_counts[0],
        "syntheticLogicalRosterRows": roster_counts[0],
        "syntheticPhysicalRows": physical_counts[0],
        "completeAccountingPass": (
            set(frame_counts) == {1536}
            and set(roster_counts) == {3072}
            and all(1024 <= count <= 3072 for count in physical_counts)
        ),
        "selectionDigestSha256": digests[0],
        "allDigests": digests,
    }
