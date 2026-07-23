#!/usr/bin/env python3
"""Resume S09 artifact finalization after an outcome-independent projection fault."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.run_s09_ablation_compression import (
    CACHE,
    OUTPUT,
    PROTOCOL_PATH,
    RAW,
    access_denial_validation,
    cross_spatial_roster,
    flatten_effects,
    historical_snapshot,
    paired_cost_summary,
    validate_worker_order,
    write_json,
    write_jsonl,
)
from src.portfolio_ablation.core import (
    build_edit_registry,
    checked_protocol,
    compare_paired_rows,
    compose_causal_edits,
    load_parent_bundles,
    validate_s09_inputs,
)
from src.portfolio_search.preflight import sha256_file, tree_digest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def reconstruct_cumulative(
    parents: list[dict[str, Any]],
    edits: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    bootstrap_replicates: int,
    sign_flip_replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parent_map = {row["parentConfigurationId"]: row for row in parents}
    edit_map = {row["editId"]: row for row in edits}
    rows_by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stage_parent_by_edit: dict[str, str] = {}
    for row in rows:
        if row["editId"] is not None:
            stage_parent_by_edit[row["editId"]] = row[
                "parentConfigurationId"
            ]
            rows_by_stage[row["editId"]].append(row)
        else:
            rows_by_stage[row["parentConfigurationId"]].append(row)
    accepted_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    current_costs = {
        row["parentConfigurationId"]: deepcopy(
            row["configuration"]["portfolioStructuralCosts"]
        )
        for row in parents
    }
    effects = []
    for stage, decision in enumerate(decisions):
        if decision["decision"] == "incompatible_after_prior_acceptance":
            continue
        parent_id = decision["parentConfigurationId"]
        source = edit_map[decision["sourceFirstOrderEditId"]]
        proposed = accepted_by_parent[parent_id] + [source]
        composed = compose_causal_edits(parent_map[parent_id], proposed)
        stage_edit_id = decision["stageEditId"]
        stage_parent_id = stage_parent_by_edit[stage_edit_id]
        before = current_costs[parent_id]
        after = composed["configuration"]["portfolioStructuralCosts"]
        delta = {key: after[key] - before[key] for key in sorted(before)}
        metadata = {
            "editId": stage_edit_id,
            "parentConfigurationId": stage_parent_id,
            "originalParentConfigurationId": parent_id,
            "sourceFirstOrderEditId": source["editId"],
            "taskId": source["taskId"],
            "category": "cumulative",
            "operator": "cumulative_delta_debug",
            "causalEdit": True,
            "controlKind": None,
            "variantConfigurationId": composed[
                "variantConfigurationId"
            ],
            "structuralCostDelta": delta,
            "structuralBytesReduced": -delta[
                "totalCanonicalMemberBytes"
            ],
        }
        stage_rows = (
            rows_by_stage[stage_parent_id] + rows_by_stage[stage_edit_id]
        )
        effect = compare_paired_rows(
            stage_rows,
            [metadata],
            bootstrap_replicates=bootstrap_replicates,
            sign_flip_replicates=sign_flip_replicates,
            seed=seed,
        )
        if len(effect) != 1:
            raise RuntimeError("failed to reconstruct one cumulative effect")
        effect[0]["originalParentConfigurationId"] = parent_id
        effect[0]["sourceFirstOrderEditId"] = source["editId"]
        effect[0]["stage"] = stage
        effects.extend(effect)
        if decision["decision"] == "accepted":
            if not effect[0]["boundedFunctionalEquivalence"]:
                raise RuntimeError(
                    "recovered cumulative decision no longer passes"
                )
            accepted_by_parent[parent_id].append(source)
            current_costs[parent_id] = deepcopy(after)
        elif decision["decision"] == "rejected":
            if effect[0]["boundedFunctionalEquivalence"]:
                raise RuntimeError(
                    "recovered cumulative rejection no longer fails"
                )
        else:
            raise RuntimeError("unknown cumulative decision")

    finals = []
    for parent in parents:
        parent_id = parent["parentConfigurationId"]
        accepted = accepted_by_parent[parent_id]
        if accepted:
            final = compose_causal_edits(parent, accepted)
        else:
            costs = parent["configuration"]["portfolioStructuralCosts"]
            final = {
                "configuration": deepcopy(parent["configuration"]),
                "documentsByPolicySha256": deepcopy(
                    parent["documentsByPolicySha256"]
                ),
                "variantConfigurationId": parent_id,
                "componentEditIds": [],
                "structuralBytesReduced": 0,
                "structuralCostDelta": {key: 0 for key in costs},
            }
        finals.append(
            {
                "parentConfigurationId": parent_id,
                "taskId": parent["taskId"],
                "acceptedEditIds": [row["editId"] for row in accepted],
                **final,
            }
        )
    return effects, finals


def main() -> int:
    started = time.time()
    if OUTPUT.exists():
        raise RuntimeError("S09 artifact output already exists")
    required_raw = {
        "screening": RAW / "screening_rows.jsonl",
        "cumulative": RAW / "cumulative_rows.jsonl",
        "decisions": RAW / "cumulative_decisions.jsonl",
        "cross": RAW / "cross_spatial_rows.jsonl",
    }
    if not CACHE.exists() or any(
        not path.is_file() for path in required_raw.values()
    ):
        raise RuntimeError("complete S09 recovery cache is unavailable")
    frozen_protocol = CACHE / "s09_protocol_frozen.yaml"
    if sha256_file(frozen_protocol) != sha256_file(PROTOCOL_PATH):
        raise RuntimeError("recovery protocol differs from frozen protocol")

    protocol = checked_protocol()
    stats = protocol["statistics"]
    before = historical_snapshot()
    input_validation = validate_s09_inputs(protocol)
    parents = load_parent_bundles(protocol)
    edits, coverage = build_edit_registry(parents)
    screening_rows = read_jsonl(required_raw["screening"])
    cumulative_rows = read_jsonl(required_raw["cumulative"])
    decisions = read_jsonl(required_raw["decisions"])
    regression_rows = read_jsonl(required_raw["cross"])
    if (len(screening_rows), len(cumulative_rows), len(regression_rows)) != (
        16896,
        26624,
        7168,
    ):
        raise RuntimeError("raw S09 panel cardinality changed")

    screening_effects = compare_paired_rows(
        screening_rows,
        edits,
        bootstrap_replicates=int(stats["pairedBootstrapReplicates"]),
        sign_flip_replicates=int(stats["pairedSignFlipReplicates"]),
        seed=int(stats["seed"]),
    )
    shams = [
        row
        for row in screening_effects
        if row["operator"] == "exact_runtime_sham"
    ]
    if len(shams) != 7 or not all(
        row["exactFunctionalEquivalence"] for row in shams
    ):
        raise RuntimeError("recovered exact-sham gate failed")
    cumulative_effects, finals = reconstruct_cumulative(
        parents,
        edits,
        decisions,
        cumulative_rows,
        bootstrap_replicates=int(stats["pairedBootstrapReplicates"]),
        sign_flip_replicates=int(stats["pairedSignFlipReplicates"]),
        seed=int(stats["seed"]),
    )
    _roster, regression_edits = cross_spatial_roster(
        parents,
        finals,
        range(
            int(
                protocol["scenarioPanels"]["crossSpatialRegression"][
                    "ordinals"
                ][0]
            ),
            int(
                protocol["scenarioPanels"]["crossSpatialRegression"][
                    "ordinals"
                ][1]
            )
            + 1,
        ),
    )
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

    worker_order, recovery_worker_rows = validate_worker_order(
        parents, edits
    )
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
            row
            for row in decisions
            if row["parentConfigurationId"] == parent_id
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
                    row["causalEdit"]
                    and row["causalHarmBeyondMargin"]
                    for row in first_order
                ),
                "cumulativeRejectedCount": sum(
                    row["decision"] == "rejected"
                    for row in parent_decisions
                ),
                "cumulativeIncompatibleCount": sum(
                    row["decision"]
                    == "incompatible_after_prior_acceptance"
                    for row in parent_decisions
                ),
                "frozenRemovableCandidatesNotAttemptedAfterStop": (
                    sum(
                        row["causalEdit"]
                        and row["boundedFunctionalEquivalence"]
                        for row in first_order
                    )
                    - len(final["acceptedEditIds"])
                    - sum(
                        row["decision"] == "rejected"
                        for row in parent_decisions
                    )
                    - sum(
                        row["decision"]
                        == "incompatible_after_prior_acceptance"
                        for row in parent_decisions
                    )
                ),
                "deltaDebuggingStopReason": (
                    "eight_accepted_edits"
                    if len(final["acceptedEditIds"]) == 8
                    else "all_individually_equivalent_candidates_attempted"
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
    all_rows = screening_rows + cumulative_rows + regression_rows
    native_contract = {
        "schemaVersion": "e07.s09.native-contract-validation.v1",
        "researchStepId": "S09",
        "success": all(
            row["nativeContractPass"] and row["replayPass"]
            for row in all_rows
        ),
        "evaluatedRows": len(all_rows),
        "nativeContractPassRows": sum(
            row["nativeContractPass"] for row in all_rows
        ),
        "replayPassRows": sum(row["replayPass"] for row in all_rows),
        "failuresRetained": sum(row["failed"] for row in all_rows),
        "censorsRetained": sum(row["censored"] for row in all_rows),
        "stopReasons": dict(
            sorted(Counter(row["stopReason"] for row in all_rows).items())
        ),
        "tasks": dict(
            sorted(Counter(row["taskId"] for row in all_rows).items())
        ),
        "claimBoundaries": sorted(
            {row["claimBoundary"] for row in all_rows}
        ),
        "universalScoreConstructed": False,
    }
    if not native_contract["success"]:
        raise RuntimeError("recovered native-contract gate failed")
    accounting = {
        "schemaVersion": "e07.s09.evaluation-accounting.v1",
        "researchStepId": "S09",
        "success": True,
        "preflightThroughputRowsNotUsedForInference": 14,
        "screening": {
            "logicalRows": len(screening_rows),
            "physicalRows": len(screening_rows),
            "publishedRows": len(screening_rows),
            "failAtomic": True,
        },
        "cumulative": {
            "attempts": sum(
                row["decision"] in {"accepted", "rejected"}
                for row in decisions
            ),
            "incompatibleStructuralSkips": sum(
                row["decision"]
                == "incompatible_after_prior_acceptance"
                for row in decisions
            ),
            "logicalRows": len(cumulative_rows),
            "physicalRows": len(cumulative_rows),
            "failAtomic": True,
        },
        "crossSpatialRegression": {
            "logicalRows": len(regression_rows),
            "physicalRows": len(regression_rows),
            "publishedRows": len(regression_rows),
            "failAtomic": True,
        },
        "initialPostprocessingWorkerOrderRows": 56,
        "recoveryWorkerOrderRows": recovery_worker_rows,
        "totalPhysicalEpisodeRowsIncludingNoninferentialChecks": (
            14
            + len(screening_rows)
            + len(cumulative_rows)
            + len(regression_rows)
            + 56
            + recovery_worker_rows
        ),
        "inferentialEpisodeRows": len(all_rows),
        "rawOutcomeCacheRoot": str(RAW),
        "rawScreeningSha256": sha256_file(required_raw["screening"]),
        "rawCumulativeSha256": sha256_file(required_raw["cumulative"]),
        "rawCrossSpatialSha256": sha256_file(required_raw["cross"]),
        "postprocessingRecovery": {
            "required": True,
            "fault": (
                "cross-task effect projection omitted the already-frozen "
                "original parent ID"
            ),
            "substantivePanelsRerun": 0,
            "scientificDesignChanged": False,
            "rawRowsReusedForFinalizationOnly": len(all_rows),
        },
        "workers": 8,
        "numericThreadsPerWorker": 1,
        "runtimeDrivenWeakening": False,
    }
    after = historical_snapshot()
    immutable = {
        "schemaVersion": "e07.s09.hash-immutability-validation.v1",
        "researchStepId": "S09",
        "success": before == after,
        "protocolSha256": sha256_file(PROTOCOL_PATH),
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
        raise RuntimeError("predecessor hash changed during recovery")

    OUTPUT.mkdir(parents=True)
    (OUTPUT / "s09_ablation_protocol.yaml").write_bytes(
        PROTOCOL_PATH.read_bytes()
    )
    write_json(OUTPUT / "input_validation.json", input_validation)
    write_jsonl(OUTPUT / "edit_registry.jsonl", edits)
    write_json(OUTPUT / "edit_coverage.json", coverage)
    write_jsonl(
        OUTPUT / "compressed_policies.jsonl",
        [
            {
                "schemaVersion": "e07.s09.compressed-policy-bundle.v1",
                "researchStepId": "S09",
                "parentConfigurationId": row["parentConfigurationId"],
                "taskId": row["taskId"],
                "acceptedEditIds": row["acceptedEditIds"],
                "configuration": row["configuration"],
                "documentsByPolicySha256": row[
                    "documentsByPolicySha256"
                ],
                "variantConfigurationId": row[
                    "variantConfigurationId"
                ],
                "structuralBytesReduced": row[
                    "structuralBytesReduced"
                ],
                "structuralCostDelta": row["structuralCostDelta"],
                "claimBoundary": (
                    "finite-panel task-local compression; no global minimality"
                ),
            }
            for row in finals
        ],
    )
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
    summary = {
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
            row["causalEdit"] and row["boundedFunctionalEquivalence"]
            for row in screening_effects
        ),
        "firstOrderNecessaryWithinPanelCount": sum(
            row["causalEdit"] and row["causalHarmBeyondMargin"]
            for row in screening_effects
        ),
        "cumulativeAcceptedCount": sum(
            row["decision"] == "accepted" for row in decisions
        ),
        "cumulativeRejectedCount": sum(
            row["decision"] == "rejected" for row in decisions
        ),
        "cumulativeIncompatibleCount": sum(
            row["decision"] == "incompatible_after_prior_acceptance"
            for row in decisions
        ),
        "frozenRemovableCandidatesNotAttemptedAfterStop": sum(
            row["frozenRemovableCandidatesNotAttemptedAfterStop"]
            for row in certificates
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
        "postprocessingRecoveryWithoutSubstantiveRerun": True,
    }
    write_json(OUTPUT / "validation_summary.json", summary)
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
            "protocolSha256": sha256_file(PROTOCOL_PATH),
            "substantiveCommand": (
                "PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                "OPENBLAS_NUM_THREADS=1 "
                "python scripts/run_s09_ablation_compression.py"
            ),
            "recoveryCommand": (
                "PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                "OPENBLAS_NUM_THREADS=1 "
                "python scripts/finalize_s09_from_cache.py"
            ),
            "postprocessingFault": (
                "missing originalParentConfigurationId projection after all "
                "substantive panels; corrected without outcome or design change"
            ),
            "substantivePanelsRerunDuringRecovery": 0,
            "python": sys.version,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "workers": 8,
            "recoveryWallSeconds": time.time() - started,
            "rawCacheRoot": str(RAW),
            "rawCacheTreeSha256": tree_digest(RAW),
            "artifactCodePaths": [
                "configs/portfolio/s09_ablation_compression.yaml",
                "src/portfolio_ablation/core.py",
                "scripts/run_s09_ablation_compression.py",
                "scripts/finalize_s09_from_cache.py",
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
                "PASS: seven-parent lock, 7/7 exact shams, native contracts, "
                "replay/order, access denial, accounting, and immutability"
            ),
            "outcomeClassification": "supportive",
            "caveatsOrBlockers": [
                "Finite training panels and frozen edit registry only.",
                "No global minimality, cross-task efficacy, biological, validation, or confirmation claim.",
                "Signal and communication edits were structurally inapplicable because all member channels are zero.",
                "A post-processing provenance-key fault required cache-only finalization; no substantive panel was rerun or redesigned.",
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
                "crossSpatialPassCount": summary[
                    "crossSpatialPassCount"
                ],
                "recoveryWallSeconds": time.time() - started,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
