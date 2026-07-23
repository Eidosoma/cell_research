#!/usr/bin/env python3
"""Execute the frozen S09 spatial ablation and compression protocol."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.environment_suite import (
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
    Split,
)
from src.environment_suite.access import AccessBroker
from src.environment_suite.contracts import canonical_sha256
from src.environment_suite.suite import EnvironmentSuite
from src.portfolio_ablation.core import (
    ARTIFACT_ROOT,
    EDIT_ORDER,
    PROTOCOL_PATH,
    S08M,
    SPATIAL_TASKS,
    build_edit_registry,
    checked_protocol,
    compare_paired_rows,
    compose_causal_edits,
    execute_roster,
    load_parent_bundles,
    make_work_roster,
    validate_s09_inputs,
)
from src.portfolio_search.preflight import sha256_file, tree_digest


OUTPUT = ARTIFACT_ROOT / "S09"
CACHE = Path("/cache/e07-s09")
RAW = CACHE / "raw"
HISTORICAL_STEPS = (
    "S05",
    "S08",
    "S08A",
    "S08B",
    "S08C",
    "S08D",
    "S08E",
    "S08F",
    "S08G",
    "S08H",
    "S08I",
    "S08J",
    "S08K",
    "S08L",
    "S08P",
)


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="ascii") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_row_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        kept = {
            key: value
            for key, value in row.items()
            if key not in {"elapsedSeconds", "workerPid"}
        }
        digest.update(
            json.dumps(
                kept,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def historical_snapshot() -> dict[str, Any]:
    s08m_freeze = read_json(S08M / "input_hash_freeze.json")
    expected = s08m_freeze["expectedTrees"]
    observed = {
        step: tree_digest(ARTIFACT_ROOT / step) for step in HISTORICAL_STEPS
    }
    mismatches = {
        step: {"expected": expected[step], "observed": observed[step]}
        for step in HISTORICAL_STEPS
        if observed[step] != expected[step]
    }
    if mismatches:
        raise RuntimeError(f"historical artifact tree changed: {mismatches}")
    return {
        "historicalExpectedTrees": expected,
        "historicalObservedTrees": observed,
        "s08mTreeSha256": tree_digest(S08M),
        "historicalQuarantineOutcomeRowsLoaded": 0,
        "historicalQuarantineCacheReads": 0,
    }


def rebase_cumulative_edit(
    original_parent: Mapping[str, Any],
    current: Mapping[str, Any],
    candidate: Mapping[str, Any],
    component_edits: Sequence[Mapping[str, Any]],
    stage: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage_parent_id = canonical_sha256(
        "E07/S09/cumulative-stage-parent/v1",
        {
            "originalParentConfigurationId": original_parent[
                "parentConfigurationId"
            ],
            "acceptedEditIds": [
                row["editId"] for row in component_edits[:-1]
            ],
            "stage": stage,
        },
    )
    stage_parent = {
        "parentConfigurationId": stage_parent_id,
        "taskId": original_parent["taskId"],
        "configuration": current["configuration"],
        "documentsByPolicySha256": current["documentsByPolicySha256"],
    }
    stage_edit = deepcopy(candidate)
    stage_edit_id = canonical_sha256(
        "E07/S09/cumulative-stage-edit/v1",
        {
            "stageParentId": stage_parent_id,
            "componentEditIds": candidate["componentEditIds"],
        },
    )
    stage_edit["editId"] = stage_edit_id
    stage_edit["parentConfigurationId"] = stage_parent_id
    stage_edit["originalParentConfigurationId"] = original_parent[
        "parentConfigurationId"
    ]
    stage_edit["sourceFirstOrderEditId"] = component_edits[-1]["editId"]
    before = current["configuration"]["portfolioStructuralCosts"]
    after = candidate["configuration"]["portfolioStructuralCosts"]
    stage_edit["parentStructuralCosts"] = before
    stage_edit["variantStructuralCosts"] = after
    stage_edit["structuralCostDelta"] = {
        key: after[key] - before[key] for key in sorted(before)
    }
    stage_edit["structuralBytesReduced"] = -stage_edit[
        "structuralCostDelta"
    ]["totalCanonicalMemberBytes"]
    return stage_parent, stage_edit


def original_as_current(parent: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "configuration": deepcopy(parent["configuration"]),
        "documentsByPolicySha256": deepcopy(
            parent["documentsByPolicySha256"]
        ),
    }


def run_cumulative(
    parents: Sequence[Mapping[str, Any]],
    edits: Sequence[Mapping[str, Any]],
    screening_effects: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    families: Sequence[int],
    bootstrap_replicates: int,
    sign_flip_replicates: int,
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    edit_map = {row["editId"]: row for row in edits}
    all_rows: list[dict[str, Any]] = []
    all_effects: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    finals: list[dict[str, Any]] = []
    for parent in parents:
        parent_id = parent["parentConfigurationId"]
        candidates = [
            edit_map[effect["editId"]]
            for effect in screening_effects
            if effect["parentConfigurationId"] == parent_id
            and effect["causalEdit"]
            and effect["boundedFunctionalEquivalence"]
            and effect["structuralBytesReduced"] >= 0
        ]
        candidates.sort(
            key=lambda row: (
                -int(row["structuralBytesReduced"]),
                EDIT_ORDER.index(row["operator"]),
                row["editId"],
            )
        )
        accepted: list[dict[str, Any]] = []
        current = original_as_current(parent)
        for candidate_index, first_order in enumerate(candidates):
            if len(accepted) >= 8:
                break
            proposed = accepted + [first_order]
            try:
                composed = compose_causal_edits(parent, proposed)
                stage_parent, stage_edit = rebase_cumulative_edit(
                    parent,
                    current,
                    composed,
                    proposed,
                    candidate_index,
                )
            except RuntimeError as exc:
                decisions.append(
                    {
                        "parentConfigurationId": parent_id,
                        "sourceFirstOrderEditId": first_order["editId"],
                        "operator": first_order["operator"],
                        "decision": "incompatible_after_prior_acceptance",
                        "reason": str(exc),
                    }
                )
                continue
            roster = make_work_roster(
                [stage_parent],
                [stage_edit],
                panel="cumulative",
                families=families,
            )
            rows, accounting = execute_roster(roster, workers=workers)
            effects = compare_paired_rows(
                rows,
                [stage_edit],
                bootstrap_replicates=bootstrap_replicates,
                sign_flip_replicates=sign_flip_replicates,
                seed=seed,
            )
            if len(effects) != 1:
                raise RuntimeError("cumulative stage did not yield one effect")
            effect = effects[0]
            effect["originalParentConfigurationId"] = parent_id
            effect["sourceFirstOrderEditId"] = first_order["editId"]
            effect["stage"] = candidate_index
            effect["accounting"] = accounting
            accept = bool(effect["boundedFunctionalEquivalence"])
            decisions.append(
                {
                    "parentConfigurationId": parent_id,
                    "sourceFirstOrderEditId": first_order["editId"],
                    "stageEditId": stage_edit["editId"],
                    "operator": first_order["operator"],
                    "decision": "accepted" if accept else "rejected",
                    "reason": (
                        "bounded_equivalence_on_fresh_cumulative_panel"
                        if accept
                        else "cumulative_bounded_equivalence_failed"
                    ),
                    "componentEditIdsIfAccepted": (
                        list(composed["componentEditIds"]) if accept else None
                    ),
                    "logicalRows": accounting["logicalRows"],
                }
            )
            all_rows.extend(rows)
            all_effects.extend(effects)
            if accept:
                accepted.append(first_order)
                current = {
                    "configuration": composed["configuration"],
                    "documentsByPolicySha256": composed[
                        "documentsByPolicySha256"
                    ],
                }
        if accepted:
            final = compose_causal_edits(parent, accepted)
        else:
            final = {
                "configuration": deepcopy(parent["configuration"]),
                "documentsByPolicySha256": deepcopy(
                    parent["documentsByPolicySha256"]
                ),
                "variantConfigurationId": parent_id,
                "componentEditIds": [],
                "structuralBytesReduced": 0,
                "structuralCostDelta": {
                    key: 0
                    for key in parent["configuration"][
                        "portfolioStructuralCosts"
                    ]
                },
            }
        finals.append(
            {
                "parentConfigurationId": parent_id,
                "taskId": parent["taskId"],
                "acceptedEditIds": [row["editId"] for row in accepted],
                **final,
            }
        )
    return all_rows, all_effects, decisions, finals


def cross_spatial_roster(
    parents: Sequence[Mapping[str, Any]],
    finals: Sequence[Mapping[str, Any]],
    families: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final_map = {row["parentConfigurationId"]: row for row in finals}
    pseudo_parents = []
    pseudo_edits = []
    for parent in parents:
        original_id = parent["parentConfigurationId"]
        final = final_map[original_id]
        for task_id in SPATIAL_TASKS:
            pair_parent_id = canonical_sha256(
                "E07/S09/cross-spatial-parent/v1",
                [original_id, task_id],
            )
            pseudo_parent = {
                "parentConfigurationId": pair_parent_id,
                "taskId": task_id,
                "configuration": deepcopy(parent["configuration"]),
                "documentsByPolicySha256": deepcopy(
                    parent["documentsByPolicySha256"]
                ),
            }
            pseudo_parents.append(pseudo_parent)
            before = parent["configuration"]["portfolioStructuralCosts"]
            after = final["configuration"]["portfolioStructuralCosts"]
            delta = {key: after[key] - before[key] for key in sorted(before)}
            pseudo_edits.append(
                {
                    "editId": canonical_sha256(
                        "E07/S09/cross-spatial-final/v1",
                        [original_id, task_id, final["acceptedEditIds"]],
                    ),
                    "parentConfigurationId": pair_parent_id,
                    "originalParentConfigurationId": original_id,
                    "taskId": task_id,
                    "category": "cross_spatial_regression",
                    "operator": "final_compression_cross_spatial",
                    "causalEdit": True,
                    "controlKind": None,
                    "configuration": deepcopy(final["configuration"]),
                    "documentsByPolicySha256": deepcopy(
                        final["documentsByPolicySha256"]
                    ),
                    "variantConfigurationId": final[
                        "variantConfigurationId"
                    ],
                    "parentStructuralCosts": before,
                    "variantStructuralCosts": after,
                    "structuralCostDelta": delta,
                    "structuralBytesReduced": -delta[
                        "totalCanonicalMemberBytes"
                    ],
                }
            )
    return (
        make_work_roster(
            pseudo_parents,
            pseudo_edits,
            panel="crossSpatialRegression",
            families=families,
        ),
        pseudo_edits,
    )


def flatten_effects(
    phases: Sequence[tuple[str, Sequence[Mapping[str, Any]]]]
) -> list[dict[str, Any]]:
    rows = []
    for phase, effects in phases:
        for effect in effects:
            for endpoint in effect["endpointEffects"]:
                rows.append(
                    {
                        "phase": phase,
                        "panel": effect["panel"],
                        "parentConfigurationId": effect[
                            "parentConfigurationId"
                        ],
                        "originalParentConfigurationId": effect.get(
                            "originalParentConfigurationId",
                            effect["parentConfigurationId"],
                        ),
                        "editId": effect["editId"],
                        "sourceFirstOrderEditId": effect.get(
                            "sourceFirstOrderEditId"
                        ),
                        "variantConfigurationId": effect[
                            "variantConfigurationId"
                        ],
                        "taskId": effect["taskId"],
                        "category": effect["category"],
                        "operator": effect["operator"],
                        "causalEdit": effect["causalEdit"],
                        "controlKind": effect["controlKind"],
                        "pairedScenarios": effect["pairedScenarios"],
                        "nativeContractAndReplayPass": effect[
                            "nativeContractAndReplayPass"
                        ],
                        "exactFunctionalEquivalence": effect[
                            "exactFunctionalEquivalence"
                        ],
                        "boundedFunctionalEquivalence": effect[
                            "boundedFunctionalEquivalence"
                        ],
                        "failureRiskDifference": effect[
                            "failureRiskDifference"
                        ],
                        "censorRiskDifference": effect[
                            "censorRiskDifference"
                        ],
                        "causalHarmBeyondMargin": effect[
                            "causalHarmBeyondMargin"
                        ],
                        "structuralBytesReduced": effect[
                            "structuralBytesReduced"
                        ],
                        "structuralCostDeltaJson": json.dumps(
                            effect["structuralCostDelta"],
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        **endpoint,
                        "globalMinimalityClaimPermitted": False,
                    }
                )
    return rows


def paired_cost_summary(
    rows: Sequence[Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    effect_ids = {row["editId"] for row in effects}
    grouped: dict[
        tuple[str, str, int], dict[str, Mapping[str, Any]]
    ] = defaultdict(dict)
    for row in rows:
        condition = row["editId"] or row["parentConfigurationId"]
        grouped[
            (
                row["parentConfigurationId"],
                row["taskId"],
                int(row["scenarioFamilyOrdinal"]),
            )
        ][condition] = row
    output = []
    for effect in effects:
        edit_id = effect["editId"]
        if edit_id not in effect_ids:
            continue
        values: dict[str, list[float]] = defaultdict(list)
        for (parent_id, task_id, _family), pair in grouped.items():
            if (
                parent_id != effect["parentConfigurationId"]
                or task_id != effect["taskId"]
                or parent_id not in pair
                or edit_id not in pair
            ):
                continue
            leaves = set(pair[parent_id]["nativeCostLeaves"]) | set(
                pair[edit_id]["nativeCostLeaves"]
            )
            for leaf in leaves:
                values[leaf].append(
                    float(pair[edit_id]["nativeCostLeaves"].get(leaf, 0.0))
                    - float(
                        pair[parent_id]["nativeCostLeaves"].get(leaf, 0.0)
                    )
                )
        for field, differences in sorted(values.items()):
            output.append(
                {
                    "parentConfigurationId": effect[
                        "parentConfigurationId"
                    ],
                    "editId": edit_id,
                    "taskId": effect["taskId"],
                    "costField": field,
                    "pairedRows": len(differences),
                    "meanEditMinusParent": sum(differences)
                    / len(differences),
                    "minimum": min(differences),
                    "maximum": max(differences),
                }
            )
    return output


def access_denial_validation() -> dict[str, Any]:
    suite = EnvironmentSuite(
        "configs/environment_suite/task_registry.yaml",
        "configs/environment_suite/split_manifest.json",
    )
    protected = [
        row
        for row in suite.records.values()
        if row.task_id in SPATIAL_TASKS and row.split is not Split.TRAIN
    ]
    broker = AccessBroker(suite.records, confirmation_unsealed=False)
    denials = []
    for record in protected:
        for kind in ("scenario", "outcome"):
            try:
                if kind == "scenario":
                    broker.authorize_scenario(
                        record.scenario_id, AccessGrant(AccessPhase.DEVELOPMENT)
                    )
                else:
                    broker.authorize_outcome(
                        record.scenario_id, AccessGrant(AccessPhase.DEVELOPMENT)
                    )
            except AccessDeniedError as exc:
                denials.append(
                    {
                        "scenarioId": record.scenario_id,
                        "split": record.split.value,
                        "requestKind": kind,
                        "denied": True,
                        "reason": str(exc),
                    }
                )
            else:
                raise RuntimeError("protected S09 access unexpectedly succeeded")
    return {
        "schemaVersion": "e07.s09.access-control-validation.v1",
        "researchStepId": "S09",
        "success": len(denials) == len(protected) * 2,
        "denialCount": len(denials),
        "denials": denials,
        "trainingOutcomeAccessOnly": True,
        "validationScenarioAccesses": 0,
        "validationOutcomeAccesses": 0,
        "confirmationScenarioAccesses": 0,
        "confirmationOutcomeAccesses": 0,
        "s06OrS06AArtifactLoads": 0,
        "s07ArmSignalUses": 0,
        "brokerAudit": broker.audit.to_dict(),
    }


def validate_worker_order(
    parents: Sequence[Mapping[str, Any]],
    edits: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], int]:
    shams = [row for row in edits if row["operator"] == "exact_runtime_sham"]
    roster = make_work_roster(
        parents,
        shams,
        panel="screening",
        families=(626, 627),
    )
    forward, first_accounting = execute_roster(roster, workers=1)
    reverse, second_accounting = execute_roster(
        list(reversed(roster)), workers=8
    )
    left = {
        (
            row["parentConfigurationId"],
            row["editId"],
            row["scenarioFamilyOrdinal"],
        ): row["resultSha256"]
        for row in forward
    }
    right = {
        (
            row["parentConfigurationId"],
            row["editId"],
            row["scenarioFamilyOrdinal"],
        ): row["resultSha256"]
        for row in reverse
    }
    if left != right:
        raise RuntimeError("worker-order validation changed result commitments")
    return (
        {
            "schemaVersion": "e07.s09.replay-worker-order-validation.v1",
            "researchStepId": "S09",
            "success": True,
            "rowsPerOrder": len(forward),
            "forwardWorkers": 1,
            "reverseWorkers": 8,
            "forwardDigest": canonical_row_digest(forward),
            "reverseDigest": canonical_row_digest(reverse),
            "resultCommitmentsEqual": True,
            "allReplayPass": all(row["replayPass"] for row in forward + reverse),
            "forwardAccounting": first_accounting,
            "reverseAccounting": second_accounting,
        },
        len(forward) + len(reverse),
    )


def main() -> int:
    started = time.time()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if CACHE.exists():
        raise RuntimeError(f"fresh S09 cache already exists: {CACHE}")
    CACHE.mkdir(parents=True)
    RAW.mkdir()
    protocol = checked_protocol()
    protocol_sha = sha256_file(PROTOCOL_PATH)
    if (
        protocol_sha
        != "0d65cf26ace4f91ace0ad57a88ecd29384a3dd48232ea1f466f36f475c6d2a75"
    ):
        raise RuntimeError("S09 protocol changed after preregistration")
    shutil.copy2(PROTOCOL_PATH, CACHE / "s09_protocol_frozen.yaml")
    before = historical_snapshot()
    input_validation = validate_s09_inputs(protocol)
    parents = load_parent_bundles(protocol)
    edits, coverage = build_edit_registry(parents)
    write_json(CACHE / "input_validation.json", input_validation)
    write_jsonl(CACHE / "edit_registry.jsonl", edits)
    write_json(CACHE / "edit_coverage.json", coverage)

    panels = protocol["scenarioPanels"]
    screening_families = range(
        int(panels["screening"]["ordinals"][0]),
        int(panels["screening"]["ordinals"][1]) + 1,
    )
    cumulative_families = range(
        int(panels["cumulative"]["ordinals"][0]),
        int(panels["cumulative"]["ordinals"][1]) + 1,
    )
    regression_families = range(
        int(panels["crossSpatialRegression"]["ordinals"][0]),
        int(panels["crossSpatialRegression"]["ordinals"][1]) + 1,
    )
    workers = int(protocol["execution"]["workers"])
    stats = protocol["statistics"]

    screening_roster = make_work_roster(
        parents,
        edits,
        panel="screening",
        families=screening_families,
    )
    screening_rows, screening_accounting = execute_roster(
        screening_roster, workers=workers
    )
    write_jsonl(RAW / "screening_rows.jsonl", screening_rows)
    screening_effects = compare_paired_rows(
        screening_rows,
        edits,
        bootstrap_replicates=int(stats["pairedBootstrapReplicates"]),
        sign_flip_replicates=int(stats["pairedSignFlipReplicates"]),
        seed=int(stats["seed"]),
    )
    shams = [
        row for row in screening_effects if row["operator"] == "exact_runtime_sham"
    ]
    if len(shams) != 7 or not all(
        row["exactFunctionalEquivalence"] for row in shams
    ):
        raise RuntimeError("S09 exact-runtime sham gate failed")

    cumulative_rows, cumulative_effects, decisions, finals = run_cumulative(
        parents,
        edits,
        screening_effects,
        workers=workers,
        families=cumulative_families,
        bootstrap_replicates=int(stats["pairedBootstrapReplicates"]),
        sign_flip_replicates=int(stats["pairedSignFlipReplicates"]),
        seed=int(stats["seed"]),
    )
    write_jsonl(RAW / "cumulative_rows.jsonl", cumulative_rows)
    write_jsonl(RAW / "cumulative_decisions.jsonl", decisions)

    regression_roster, regression_edits = cross_spatial_roster(
        parents, finals, regression_families
    )
    regression_rows, regression_accounting = execute_roster(
        regression_roster, workers=workers
    )
    write_jsonl(RAW / "cross_spatial_rows.jsonl", regression_rows)
    write_jsonl(RAW / "cross_spatial_rows.jsonl", regression_rows)
    regression_effects = compare_paired_rows(
        regression_rows,
        regression_edits,
        bootstrap_replicates=int(stats["pairedBootstrapReplicates"]),
        sign_flip_replicates=int(stats["pairedSignFlipReplicates"]),
        seed=int(stats["seed"]),
    )
    regression_edit_map = {
        row["editId"]: row for row in regression_edits
    }
    for effect in regression_effects:
        effect["originalParentConfigurationId"] = regression_edit_map[
            effect["editId"]
        ]["originalParentConfigurationId"]
    worker_order, worker_order_rows = validate_worker_order(parents, edits)
    access = access_denial_validation()

    cross_by_original: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for effect in regression_effects:
        cross_by_original[effect["originalParentConfigurationId"]].append(
            effect
        )
    certificates = []
    for parent in parents:
        parent_id = parent["parentConfigurationId"]
        final = next(
            row for row in finals if row["parentConfigurationId"] == parent_id
        )
        first_order = [
            row
            for row in screening_effects
            if row["parentConfigurationId"] == parent_id
        ]
        parent_decisions = [
            row for row in decisions if row["parentConfigurationId"] == parent_id
        ]
        cross = cross_by_original[parent_id]
        certificates.append(
            {
                "schemaVersion": "e07.s09.bounded-minimality-certificate.v1",
                "researchStepId": "S09",
                "parentConfigurationId": parent_id,
                "taskId": parent["taskId"],
                "eligibleParent": True,
                "acceptedEditIds": final["acceptedEditIds"],
                "acceptedEditCount": len(final["acceptedEditIds"]),
                "firstOrderRemovableCount": sum(
                    row["causalEdit"]
                    and row["boundedFunctionalEquivalence"]
                    for row in first_order
                ),
                "firstOrderNecessaryWithinPanelCount": sum(
                    row["causalEdit"] and row["causalHarmBeyondMargin"]
                    for row in first_order
                ),
                "cumulativeRejectedCount": sum(
                    row["decision"] == "rejected" for row in parent_decisions
                ),
                "cumulativeIncompatibleCount": sum(
                    row["decision"]
                    == "incompatible_after_prior_acceptance"
                    for row in parent_decisions
                ),
                "finalConfigurationId": final[
                    "variantConfigurationId"
                ],
                "parentStructuralCosts": parent["configuration"][
                    "portfolioStructuralCosts"
                ],
                "finalStructuralCosts": final["configuration"][
                    "portfolioStructuralCosts"
                ],
                "structuralBytesReduced": final[
                    "structuralBytesReduced"
                ],
                "crossSpatialRegressionPass": len(cross) == 2
                and all(
                    row["boundedFunctionalEquivalence"]
                    and row["nativeContractAndReplayPass"]
                    for row in cross
                ),
                "crossSpatialTaskEffects": [
                    {
                        "taskId": row["taskId"],
                        "boundedFunctionalEquivalence": row[
                            "boundedFunctionalEquivalence"
                        ],
                        "exactFunctionalEquivalence": row[
                            "exactFunctionalEquivalence"
                        ],
                    }
                    for row in cross
                ],
                "boundedClaim": (
                    "locally compressed under the frozen edit order and finite "
                    "training panels"
                ),
                "globalMinimalityClaimPermitted": False,
                "unsearchedMechanisms": [
                    "new_rule_synthesis",
                    "threshold_tuning",
                    "arbitrary_program_rewrite",
                    "nonregistered_joint_edits",
                    "confirmation_behavior",
                ],
            }
        )

    flattened = flatten_effects(
        (
            ("first_order_screening", screening_effects),
            ("cumulative_delta_debug", cumulative_effects),
            ("cross_spatial_regression", regression_effects),
        )
    )
    cost_rows = (
        paired_cost_summary(screening_rows, screening_effects)
        + paired_cost_summary(cumulative_rows, cumulative_effects)
        + paired_cost_summary(regression_rows, regression_effects)
    )
    all_eval_rows = screening_rows + cumulative_rows + regression_rows
    native_contract = {
        "schemaVersion": "e07.s09.native-contract-validation.v1",
        "researchStepId": "S09",
        "success": all(
            row["nativeContractPass"] and row["replayPass"]
            for row in all_eval_rows
        ),
        "evaluatedRows": len(all_eval_rows),
        "nativeContractPassRows": sum(
            row["nativeContractPass"] for row in all_eval_rows
        ),
        "replayPassRows": sum(row["replayPass"] for row in all_eval_rows),
        "failuresRetained": sum(row["failed"] for row in all_eval_rows),
        "censorsRetained": sum(row["censored"] for row in all_eval_rows),
        "stopReasons": dict(
            sorted(Counter(row["stopReason"] for row in all_eval_rows).items())
        ),
        "tasks": dict(
            sorted(Counter(row["taskId"] for row in all_eval_rows).items())
        ),
        "claimBoundaries": sorted(
            {row["claimBoundary"] for row in all_eval_rows}
        ),
        "universalScoreConstructed": False,
    }
    if not native_contract["success"]:
        raise RuntimeError("S09 native contract or replay validation failed")
    total_rows = (
        14
        + len(screening_rows)
        + len(cumulative_rows)
        + len(regression_rows)
        + worker_order_rows
    )
    accounting = {
        "schemaVersion": "e07.s09.evaluation-accounting.v1",
        "researchStepId": "S09",
        "success": True,
        "preflightThroughputRowsNotUsedForInference": 14,
        "preflightNote": (
            "One post-preregistration family-500 exact-sham throughput check; "
            "only aggregate runtime and integrity were observed, and all "
            "inferential rows were freshly executed."
        ),
        "screening": screening_accounting,
        "cumulative": {
            "attempts": sum(
                row["decision"] in {"accepted", "rejected"}
                for row in decisions
            ),
            "incompatibleStructuralSkips": sum(
                row["decision"] == "incompatible_after_prior_acceptance"
                for row in decisions
            ),
            "logicalRows": len(cumulative_rows),
            "physicalRows": len(cumulative_rows),
        },
        "crossSpatialRegression": regression_accounting,
        "workerOrderValidationRows": worker_order_rows,
        "totalPhysicalEpisodeRowsIncludingNoninferentialChecks": total_rows,
        "inferentialEpisodeRows": len(all_eval_rows),
        "publishedArtifactOutcomeRows": 0,
        "rawOutcomeCacheRoot": str(RAW),
        "rawScreeningSha256": sha256_file(RAW / "screening_rows.jsonl"),
        "rawCumulativeSha256": sha256_file(RAW / "cumulative_rows.jsonl"),
        "rawCrossSpatialSha256": sha256_file(
            RAW / "cross_spatial_rows.jsonl"
        ),
        "failAtomicPanelPublication": True,
        "workers": workers,
        "numericThreadsPerWorker": 1,
        "runtimeDrivenWeakening": False,
    }
    after = historical_snapshot()
    immutable = {
        "schemaVersion": "e07.s09.hash-immutability-validation.v1",
        "researchStepId": "S09",
        "success": before == after,
        "protocolSha256": protocol_sha,
        "candidateLockSha256": input_validation["candidateLockSha256"],
        "eligibleConfigurationIds": sorted(
            protocol["eligibleConfigurationIds"]
        ),
        "before": before,
        "after": after,
        "s08mUnchanged": before["s08mTreeSha256"]
        == after["s08mTreeSha256"],
        "historicalArtifactsUnchanged": before[
            "historicalObservedTrees"
        ]
        == after["historicalObservedTrees"],
        "s05Mutations": 0,
        "s08mMutations": 0,
        "historicalQuarantineMutations": 0,
    }
    if not immutable["success"]:
        raise RuntimeError("S09 mutated a frozen predecessor artifact")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(PROTOCOL_PATH, OUTPUT / "s09_ablation_protocol.yaml")
    write_json(OUTPUT / "input_validation.json", input_validation)
    write_jsonl(OUTPUT / "edit_registry.jsonl", edits)
    write_json(OUTPUT / "edit_coverage.json", coverage)
    compressed_rows = []
    for final in finals:
        compressed_rows.append(
            {
                "schemaVersion": "e07.s09.compressed-policy-bundle.v1",
                "researchStepId": "S09",
                "parentConfigurationId": final[
                    "parentConfigurationId"
                ],
                "taskId": final["taskId"],
                "acceptedEditIds": final["acceptedEditIds"],
                "configuration": final["configuration"],
                "documentsByPolicySha256": final[
                    "documentsByPolicySha256"
                ],
                "variantConfigurationId": final[
                    "variantConfigurationId"
                ],
                "structuralBytesReduced": final[
                    "structuralBytesReduced"
                ],
                "structuralCostDelta": final["structuralCostDelta"],
                "claimBoundary": (
                    "finite-panel task-local compression; no global minimality"
                ),
            }
        )
    write_jsonl(OUTPUT / "compressed_policies.jsonl", compressed_rows)
    write_jsonl(OUTPUT / "minimality_certificates.jsonl", certificates)
    write_jsonl(OUTPUT / "cumulative_decisions.jsonl", decisions)
    frame = pd.DataFrame(flattened)
    frame.to_csv(OUTPUT / "ablation_effects.csv", index=False)
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        OUTPUT / "ablation_effects.parquet",
        compression="zstd",
    )
    write_json(OUTPUT / "evaluation_accounting.json", accounting)
    write_json(OUTPUT / "access_control_validation.json", access)
    write_json(
        OUTPUT / "replay_worker_order_validation.json", worker_order
    )
    write_json(OUTPUT / "native_contract_validation.json", native_contract)
    write_json(
        OUTPUT / "cost_accounting.json",
        {
            "schemaVersion": "e07.s09.cost-accounting.v1",
            "researchStepId": "S09",
            "success": True,
            "analysis": "fieldwise paired edit-minus-parent only",
            "universalCostTotalConstructed": False,
            "licensedCapabilityCostsKeptSeparate": True,
            "rows": cost_rows,
        },
    )
    write_json(OUTPUT / "hash_and_immutability_validation.json", immutable)
    write_json(
        OUTPUT / "validation_summary.json",
        {
            "schemaVersion": "e07.s09.validation-summary.v1",
            "researchStepId": "S09",
            "success": True,
            "eligibleParentCount": len(parents),
            "eligibleTaskCounts": dict(
                sorted(Counter(row["taskId"] for row in parents).items())
            ),
            "firstOrderEditCount": len(edits),
            "firstOrderCausalCount": sum(row["causalEdit"] for row in edits),
            "exactShamPassCount": sum(
                row["exactFunctionalEquivalence"] for row in shams
            ),
            "firstOrderRemovableCount": sum(
                row["causalEdit"]
                and row["boundedFunctionalEquivalence"]
                for row in screening_effects
            ),
            "firstOrderNecessaryWithinPanelCount": sum(
                row["causalEdit"] and row["causalHarmBeyondMargin"]
                for row in screening_effects
            ),
            "cumulativeAcceptedCount": sum(
                row["decision"] == "accepted" for row in decisions
            ),
            "compressedParentCount": sum(
                bool(row["acceptedEditIds"]) for row in finals
            ),
            "crossSpatialPassCount": sum(
                row["crossSpatialRegressionPass"] for row in certificates
            ),
            "allNativeContractsAndReplayPass": native_contract["success"],
            "workerOrderIndependent": worker_order["success"],
            "accessDenialPass": access["success"],
            "immutableInputsPass": immutable["success"],
            "signalAblationsFabricated": 0,
            "communicationAblationsFabricated": 0,
            "validationOutcomeAccesses": 0,
            "confirmationOutcomeAccesses": 0,
            "s06OrS06AArtifactLoads": 0,
            "s07ArmSignalUses": 0,
            "globalMinimalityClaimPermitted": False,
        },
    )
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    write_json(
        OUTPUT / "provenance.json",
        {
            "schemaVersion": "e07.s09.provenance.v1",
            "researchStepId": "S09",
            "dateUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "repository": "/workspace/cell-research",
            "sourceCommitBeforeS09": git_commit,
            "protocolPath": str(PROTOCOL_PATH),
            "protocolSha256": protocol_sha,
            "command": (
                "PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                "OPENBLAS_NUM_THREADS=1 "
                "python scripts/run_s09_ablation_compression.py"
            ),
            "python": sys.version,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "workers": workers,
            "wallSeconds": time.time() - started,
            "rawCacheRoot": str(RAW),
            "rawCacheTreeSha256": tree_digest(RAW),
            "artifactCodePaths": [
                "configs/portfolio/s09_ablation_compression.yaml",
                "src/portfolio_ablation/core.py",
                "scripts/run_s09_ablation_compression.py",
                "tests/test_portfolio_ablation_s09.py",
            ],
        },
    )
    write_json(
        OUTPUT / "status.json",
        {
            "researchStepId": "S09",
            "stepNumber": 9,
            "success": True,
            "status": "complete",
            "artifactsWritten": sorted(
                [
                    path.name
                    for path in OUTPUT.iterdir()
                    if path.is_file()
                ]
                + [
                    "artifact_manifest.json",
                    "research_step_full_results.md",
                ]
            ),
            "validationResult": (
                "PASS: seven-parent lock, exact shams, native contracts, "
                "replay/order, access denial, accounting, and immutability"
            ),
            "outcomeClassification": (
                "supportive"
                if any(row["acceptedEditIds"] for row in finals)
                else "null"
            ),
            "caveatsOrBlockers": [
                "Finite training panels and frozen edit registry only.",
                "No global minimality, cross-task efficacy, biological, validation, or confirmation claim.",
                "Signal and communication edits were structurally inapplicable because all member channels are zero.",
            ],
            "recommendedNextAction": (
                "Hand control back before S10; separately preregister any "
                "non-embedding event-feature S10 redesign."
            ),
        },
    )
    print(
        json.dumps(
            {
                "success": True,
                "output": str(OUTPUT),
                "screeningRows": len(screening_rows),
                "cumulativeRows": len(cumulative_rows),
                "crossRows": len(regression_rows),
                "acceptedEdits": sum(
                    len(row["acceptedEditIds"]) for row in finals
                ),
                "wallSeconds": time.time() - started,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
