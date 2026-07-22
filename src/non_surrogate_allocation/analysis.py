"""Prespecified analysis for the approved S07 non-surrogate allocation."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta

from src.allocation_redesign.core import TASK_IDS, canonical_hash, hash_file
from src.non_surrogate_allocation.core import (
    ARTIFACT_DIR,
    FAMILY_ORDINALS,
    LOCK_PATH,
    S05_DIR,
    _catalog,
    _numeric_leaves,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from src.quality_diversity.core import OBJECTIVES, _get_path, stable_cell_audit


ARMS = (
    "uniform_randomized",
    "descriptor_cell_coverage",
    "contract_status_stratified",
)
FIXED_ARMS = ("descriptor_cell_coverage", "contract_status_stratified")
STABILITY_REPLICATES = 1000
STABILITY_THRESHOLD = 0.8
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 707702


def _status_signature(row: Mapping[str, Any]) -> str:
    return f"failed={int(bool(row['failed']))}|censored={int(bool(row['censored']))}|stop={row['stopReason']}"


def _cell_key(task_id: str, archive_id: str, cell: Sequence[int]) -> str:
    return f"{task_id}|{archive_id}|{json.dumps(list(cell), separators=(',', ':'))}"


def _auc(curve: Sequence[float]) -> float:
    if len(curve) != 4:
        raise ValueError("S07 discovery AUC requires exactly four waves")
    return float(np.trapezoid([0.0, *curve], x=[0.0, 0.25, 0.5, 0.75, 1.0]))


def _holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def _curve_contribution_weight(first_wave_index: int) -> float:
    # AUC of cumulative discovery yield y_j = D_j / (32*j), evaluated at
    # normalized checkpoints 0,.25,.5,.75,1 by trapezoids.
    y_coefficients = (0.25, 0.25, 0.25, 0.125)
    return sum(
        y_coefficients[index] / (32.0 * (index + 1))
        for index in range(first_wave_index, 4)
    )


def _descriptor_panel_audits(
    physical: Mapping[str, Mapping[str, Any]],
    logical_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    catalog, _ = _catalog()
    selected_pairs = sorted({(row["taskId"], row["policySha256"]) for row in logical_rows})
    physical_by_key = {
        (row["taskId"], row["policySha256"], int(row["scenarioFamilyOrdinal"])): row
        for row in physical.values()
    }
    audits: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for task_id, policy_hash in selected_pairs:
        panel = []
        for wave_index, ordinal in enumerate(FAMILY_ORDINALS):
            panel.append(physical_by_key[(task_id, policy_hash, ordinal)])
            integrity_valid = all(
                not row["failed"] and row["replayPass"] and all(row["validation"].values())
                for row in panel
            )
            # S04 explicitly marked singleton stability not estimable.
            if len(panel) < 2 or not integrity_valid:
                audits[(task_id, policy_hash, ordinal)] = []
                continue
            result = stable_cell_audit(
                catalog[policy_hash], panel,
                bootstrap_replicates=STABILITY_REPLICATES,
            )
            audits[(task_id, policy_hash, ordinal)] = [
                {**item, "stable": item["sameCellProbability"] >= STABILITY_THRESHOLD}
                for item in result
            ]
    return audits


def _discovery_curves(
    physical: Mapping[str, Mapping[str, Any]],
    logical_rows: Sequence[Mapping[str, Any]],
    descriptor_audits: Mapping[tuple[str, str, int], Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rare_registry = json.loads(
        (Path("/artifacts/research_steps/S07R/rare_status_registry.json")).read_text()
    )
    rare_known = {
        task_id: {
            item["signature"] for item in value["signatures"] if item["rareByFrozenRule"]
        }
        for task_id, value in rare_registry["tasks"].items()
    }
    known_all = {
        task_id: {item["signature"] for item in value["signatures"]}
        for task_id, value in rare_registry["tasks"].items()
    }
    occupied = {
        _cell_key(row["taskId"], row["archiveId"], row["cell"])
        for row in _read_jsonl(S05_DIR / "archive/archive_entries.jsonl")
    }
    by_arm_task_family: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for logical in logical_rows:
        by_arm_task_family[(logical["armId"], logical["taskId"], int(logical["scenarioFamilyOrdinal"]))].append(logical)
    rare_curves = []
    descriptor_curves = []
    first_discovery = {"rare": {}, "descriptor": {}}
    episode_flags: dict[str, dict[str, Any]] = defaultdict(lambda: {"rare": False, "descriptor": False, "reasons": []})
    for arm_id in ARMS:
        for task_id in TASK_IDS:
            rare_seen: set[str] = set()
            descriptor_seen: set[str] = set()
            for wave_index, ordinal in enumerate(FAMILY_ORDINALS):
                new_rare: set[str] = set()
                new_descriptor: set[str] = set()
                for logical in by_arm_task_family[(arm_id, task_id, ordinal)]:
                    result = physical[logical["physicalRowId"]]
                    signature = _status_signature(result)
                    is_rare = bool(result["failed"]) or signature not in known_all[task_id] or signature in rare_known[task_id]
                    if is_rare:
                        new_rare.add(signature)
                        episode_flags[result["physicalRowId"]]["rare"] = True
                        episode_flags[result["physicalRowId"]]["reasons"].append(f"rare_status:{signature}")
                    for audit in descriptor_audits[(task_id, logical["policySha256"], ordinal)]:
                        if not audit["stable"]:
                            continue
                        key = _cell_key(task_id, audit["archiveId"], audit["referenceCell"])
                        if key not in occupied:
                            new_descriptor.add(key)
                            episode_flags[result["physicalRowId"]]["descriptor"] = True
                            episode_flags[result["physicalRowId"]]["reasons"].append(f"shadow_descriptor:{key}")
                truly_new_rare = new_rare - rare_seen
                truly_new_descriptor = new_descriptor - descriptor_seen
                rare_seen.update(new_rare)
                descriptor_seen.update(new_descriptor)
                first_discovery["rare"][(arm_id, task_id, ordinal)] = len(truly_new_rare)
                first_discovery["descriptor"][(arm_id, task_id, ordinal)] = len(truly_new_descriptor)
                logical_n = 32 * (wave_index + 1)
                rare_curves.append({
                    "armId": arm_id, "taskId": task_id, "scenarioFamilyOrdinal": ordinal,
                    "waveIndex": wave_index + 1, "cumulativeLogicalEvaluations": logical_n,
                    "newDistinctDiscoveries": len(truly_new_rare),
                    "cumulativeDistinctDiscoveries": len(rare_seen),
                    "cumulativeDiscoveryYield": len(rare_seen) / logical_n,
                })
                descriptor_curves.append({
                    "armId": arm_id, "taskId": task_id, "scenarioFamilyOrdinal": ordinal,
                    "waveIndex": wave_index + 1, "cumulativeLogicalEvaluations": logical_n,
                    "newDistinctDiscoveries": len(truly_new_descriptor),
                    "cumulativeDistinctDiscoveries": len(descriptor_seen),
                    "cumulativeDiscoveryYield": len(descriptor_seen) / logical_n,
                    "singletonStabilityExcluded": True,
                })
    return rare_curves, descriptor_curves, {"firstDiscovery": first_discovery, "episodeFlags": episode_flags}


def _primary_comparisons(first: Mapping[str, Mapping[tuple[str, str, int], int]]) -> dict[str, Any]:
    block_rows = []
    for estimand in ("rare", "descriptor"):
        for arm_id in ARMS:
            for task_id in TASK_IDS:
                for wave_index, ordinal in enumerate(FAMILY_ORDINALS):
                    count = first[estimand][(arm_id, task_id, ordinal)]
                    block_rows.append({
                        "estimand": estimand, "armId": arm_id, "taskId": task_id,
                        "scenarioFamilyOrdinal": ordinal,
                        "firstDiscoveryCount": count,
                        "aucContribution": count * _curve_contribution_weight(wave_index),
                    })
    lookup = {(row["estimand"], row["armId"], row["taskId"], row["scenarioFamilyOrdinal"]): row["aucContribution"] for row in block_rows}
    blocks = [(task_id, ordinal) for task_id in TASK_IDS for ordinal in FAMILY_ORDINALS]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    tests = []
    raw_p = {}
    for estimand in ("rare", "descriptor"):
        for fixed in FIXED_ARMS:
            differences = np.array([
                lookup[(estimand, fixed, task_id, ordinal)] - lookup[(estimand, "uniform_randomized", task_id, ordinal)]
                for task_id, ordinal in blocks
            ])
            point = float(differences.sum() / len(TASK_IDS))
            samples = np.empty(BOOTSTRAP_REPLICATES)
            for index in range(BOOTSTRAP_REPLICATES):
                draw = rng.integers(0, len(blocks), size=len(blocks))
                samples[index] = float(differences[draw].mean() * len(FAMILY_ORDINALS))
            p = min(1.0, 2 * min(float(np.mean(samples <= 0)), float(np.mean(samples >= 0))))
            test_id = f"{estimand}:{fixed}-uniform_randomized"
            raw_p[test_id] = p
            tests.append({
                "testId": test_id, "estimand": estimand, "fixedArmId": fixed,
                "comparatorArmId": "uniform_randomized", "differenceAuc": point,
                "lower95": float(np.quantile(samples, 0.025)),
                "upper95": float(np.quantile(samples, 0.975)),
                "rawTwoSidedBootstrapP": p,
                "interpretation": "paired descriptive fixed-arm contrast; not a causal population contrast",
            })
    adjusted = _holm(raw_p)
    for test in tests:
        test["HolmAdjustedP"] = adjusted[test["testId"]]
        test["HolmRejectAt0.05"] = adjusted[test["testId"]] <= 0.05
    arm_auc = []
    for estimand in ("rare", "descriptor"):
        for arm_id in ARMS:
            task_values = []
            for task_id in TASK_IDS:
                value = sum(lookup[(estimand, arm_id, task_id, ordinal)] for ordinal in FAMILY_ORDINALS)
                task_values.append(value)
                arm_auc.append({"estimand": estimand, "armId": arm_id, "taskId": task_id, "discoveryYieldAuc": value})
            arm_auc.append({"estimand": estimand, "armId": arm_id, "taskId": "equal_task_mean", "discoveryYieldAuc": float(np.mean(task_values))})
    return {
        "schemaVersion": "e07.s07.primary-comparisons.v1", "tests": tests,
        "armAuc": arm_auc, "blockRows": block_rows,
        "bootstrapReplicates": BOOTSTRAP_REPLICATES, "bootstrapSeed": BOOTSTRAP_SEED,
        "resamplingUnit": "paired_task_by_scenario_family_block",
        "multiplicity": "Holm familywise 0.05 across four frozen tests",
    }


def _uniform_population_estimates(
    physical: Mapping[str, Mapping[str, Any]],
    logical_rows: Sequence[Mapping[str, Any]],
    flags: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = defaultdict(set)
    rare_y = defaultdict(bool)
    descriptor_y = defaultdict(bool)
    for row in logical_rows:
        if row["armId"] != "uniform_randomized":
            continue
        key = (row["taskId"], row["policySha256"])
        selected[row["taskId"]].add(row["policySha256"])
        rare_y[key] = rare_y[key] or flags[row["physicalRowId"]]["rare"]
        descriptor_y[key] = descriptor_y[key] or flags[row["physicalRowId"]]["descriptor"]
    result = []
    for task_id in TASK_IDS:
        policies = sorted(selected[task_id])
        if len(policies) != 32:
            raise RuntimeError("uniform task sample is not 32 of 64 candidates")
        for estimand, values in (("candidate_ever_rare", rare_y), ("candidate_ever_shadow_descriptor", descriptor_y)):
            observed = np.array([float(values[(task_id, policy)]) for policy in policies])
            mean = float(observed.mean())
            variance = float(observed.var(ddof=1)) if len(observed) > 1 else 0.0
            se = math.sqrt((1 - 32 / 64) * variance / 32)
            result.append({
                "taskId": task_id, "estimand": estimand, "sampleN": 32,
                "populationN": 64, "inclusionProbability": 0.5,
                "HorvitzThompsonTotal": float(observed.sum() / 0.5),
                "estimatedPopulationRate": mean, "finitePopulationCorrectedSe": se,
                "lower95": max(0.0, mean - 1.96 * se), "upper95": min(1.0, mean + 1.96 * se),
                "probabilitySampledComparator": True,
            })
    return result


def _native_tables(
    physical: Mapping[str, Mapping[str, Any]], logical_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in logical_rows:
        groups[(row["armId"], row["taskId"])].append(physical[row["physicalRowId"]])
    objectives = []
    costs = []
    status = []
    for (arm_id, task_id), rows in sorted(groups.items()):
        for objective_id, path, direction in OBJECTIVES[task_id]:
            values = [float(_get_path(row["outcome"], path)) for row in rows]
            objectives.append({
                "armId": arm_id, "taskId": task_id, "objectiveId": objective_id,
                "direction": direction, "nativeMean": float(np.mean(values)),
                "nativeMedian": float(np.median(values)), "rowCount": len(values),
                "crossTaskNormalization": False,
            })
        flattened = [_numeric_leaves(row["nativeLedgerFamilies"]) for row in rows]
        for field in sorted({key for item in flattened for key in item}):
            values = [item.get(field, 0.0) for item in flattened]
            costs.append({
                "armId": arm_id, "taskId": task_id, "nativeCostField": field,
                "nativeMean": float(np.mean(values)), "nativeMedian": float(np.median(values)),
                "licensedCapabilityCost": any(token in field.lower() for token in ("licensed", "prefix", "cursor")),
                "crossFamilyScalarTotal": False,
            })
        status.append({
            "armId": arm_id, "taskId": task_id, "logicalRows": len(rows),
            "failedRows": sum(row["failed"] for row in rows),
            "censoredRows": sum(row["censored"] for row in rows),
            "validRows": sum(not row["failed"] and row["replayPass"] and all(row["validation"].values()) for row in rows),
            "stopReasonCounts": json.dumps(dict(sorted(Counter(row["stopReason"] for row in rows).items())), sort_keys=True),
            "nativeUnit": rows[0]["nativeUnit"],
        })
    return objectives, costs, status


def _zero_event_bounds(curves: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    final = [row for row in curves if row["waveIndex"] == 4]
    for row in final:
        count = int(row["cumulativeDistinctDiscoveries"])
        if count == 0:
            result.append({
                "armId": row["armId"], "taskId": row["taskId"],
                "logicalEvaluations": row["cumulativeLogicalEvaluations"],
                "events": 0,
                "oneSidedExact95UpperProbability": float(beta.ppf(0.95, 1, row["cumulativeLogicalEvaluations"])),
                "note": "binomial bound is a bounded diagnostic; distinct discoveries are not independent trials",
            })
    return result


def analyze(output_dir: Path = ARTIFACT_DIR) -> dict[str, Any]:
    freeze = json.loads((output_dir / "execution_freeze.json").read_text())
    accounting = json.loads((output_dir / "execution_accounting.json").read_text())
    if not accounting.get("success") or hash_file(output_dir / "physical_evaluation_ledger.jsonl") != accounting["physicalLedgerSha256"]:
        raise RuntimeError("complete immutable S07 physical ledger required")
    physical_rows = _read_jsonl(output_dir / "physical_evaluation_ledger.jsonl")
    physical = {row["physicalRowId"]: row for row in physical_rows}
    logical_rows = pd.read_parquet(output_dir / "allocation_logical_roster.parquet").to_dict("records")
    descriptor_audits = _descriptor_panel_audits(physical, logical_rows)
    stability_rows = []
    for (task_id, policy_hash, ordinal), audits in descriptor_audits.items():
        for row in audits:
            stability_rows.append({"taskId": task_id, "policySha256": policy_hash, "scenarioFamilyOrdinal": ordinal, **row})
    rare_curves, descriptor_curves, discovery = _discovery_curves(physical, logical_rows, descriptor_audits)
    primary = _primary_comparisons(discovery["firstDiscovery"])
    uniform = _uniform_population_estimates(physical, logical_rows, discovery["episodeFlags"])
    objectives, costs, status = _native_tables(physical, logical_rows)
    logical_ledger = []
    for row in logical_rows:
        result = physical[row["physicalRowId"]]
        logical_ledger.append({
            **row, "workerId": result["workerPid"], "resultSha256": result["resultSha256"],
            "failed": result["failed"], "censored": result["censored"],
            "stopReason": result["stopReason"], "replayPass": result["replayPass"],
            "rareEventIndicator": discovery["episodeFlags"][row["physicalRowId"]]["rare"],
            "shadowDescriptorIndicator": discovery["episodeFlags"][row["physicalRowId"]]["descriptor"],
        })
    pd.DataFrame(logical_ledger).to_parquet(output_dir / "logical_evaluation_ledger.parquet", index=False)
    pd.DataFrame(rare_curves).to_parquet(output_dir / "rare_event_discovery_curves.parquet", index=False)
    pd.DataFrame(descriptor_curves).to_parquet(output_dir / "shadow_descriptor_discovery_curves.parquet", index=False)
    pd.DataFrame(stability_rows).to_parquet(output_dir / "descriptor_stability_audit.parquet", index=False)
    pd.DataFrame(objectives).to_parquet(output_dir / "native_objective_summary.parquet", index=False)
    pd.DataFrame(costs).to_parquet(output_dir / "native_cost_summary.parquet", index=False)
    pd.DataFrame(status).to_parquet(output_dir / "failure_censor_summary.parquet", index=False)
    _write_json(output_dir / "primary_comparisons.json", primary)
    _write_json(output_dir / "uniform_probability_sample_estimates.json", {"schemaVersion": "e07.s07.uniform-estimates.v1", "rows": uniform})
    _write_json(output_dir / "zero_rare_event_bounds.json", {"schemaVersion": "e07.s07.zero-event-bounds.v1", "rows": _zero_event_bounds(rare_curves)})
    retained = []
    trace_manifest = []
    for row in physical_rows:
        flags = discovery["episodeFlags"][row["physicalRowId"]]
        base = int(hashlib.sha256(("E07/S07/native-event-retention/v1\x00" + row["physicalRowId"]).encode()).hexdigest(), 16) / 2**256 < 0.05
        reasons = []
        if base: reasons.append("deterministic_5_percent")
        if row["failed"]: reasons.append("failure")
        if row["censored"]: reasons.append("censor")
        if flags["rare"]: reasons.append("rare_status_discovery")
        if flags["descriptor"]: reasons.append("shadow_descriptor_discovery")
        if reasons:
            retained.append(row)
            trace_manifest.append({"physicalRowId": row["physicalRowId"], "selectionReasons": sorted(set(reasons)), "selectionProbabilityForBaseSample": 0.05, "fullNativeTraceAvailable": False, "retainedPayload": "complete_native_event_outcome_cost_summary"})
    _write_jsonl(output_dir / "selected_native_event_records.jsonl", retained)
    _write_json(output_dir / "trace_retention_manifest.json", {"schemaVersion": "e07.s07.trace-retention.v1", "selectedPhysicalRows": len(retained), "rows": trace_manifest})
    trace_audit = {
        "schemaVersion": "e07.s07.trace-availability-audit.v1", "success": True,
        "physicalRowsWithNativeEventSummary": len(physical_rows),
        "completeNativeTrajectoryRows": 0, "completeNativeTracesAvailable": False,
        "S10Gate": "requires redesign as native event-feature discovery or remains blocked",
        "rejectedEventSummaryEmbeddingsUsed": False,
    }
    _write_json(output_dir / "trace_availability_audit.json", trace_audit)
    result = {
        "schemaVersion": "e07.s07.analysis-summary.v1", "researchStepId": "S07",
        "success": True, "logicalRows": len(logical_rows), "physicalRows": len(physical_rows),
        "primaryTests": primary["tests"],
        "anyHolmSignificantFixedArmContrast": any(row["HolmRejectAt0.05"] for row in primary["tests"]),
        "fixedArmContrastsDescriptiveOnly": True,
        "uniformArmProbabilitySampledComparator": True,
        "descriptorSingletonStabilityExcluded": True,
        "failuresAndCensorsRetained": True,
        "executionLockSha256": hash_file(LOCK_PATH),
    }
    _write_json(output_dir / "analysis_summary.json", result)
    return result
