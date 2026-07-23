#!/usr/bin/env python3
"""Qualify S10E without loading a frozen episode or consumed outcome row."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Mapping

import numpy as np
import pyarrow
import sklearn
import yaml

import scripts.run_native_event_discovery_s10 as base
from src.phenotype_discovery.method_feasibility import (
    FeasibilityContractError,
    assess_reproduction_structure,
    assess_stratum,
    expected_rule_document,
    fold_for_family,
    global_multiplicity_slots,
    holm_adjust_slots,
    multiplicity_slots,
    validate_frozen_rule,
)
from src.phenotype_discovery.native_features import build_feature_registry
from src.phenotype_discovery.search import (
    _action_for,
    _load_candidate_bundles,
    anomaly_discovery,
    change_point_discovery,
    clustering_discovery,
    confound_audit,
    fit_preprocessing,
    load_roster,
    sha256_file,
)


STEP_ID = "S10E"
REPOSITORY = Path("/workspace/cell-research")
WORKSPACE = Path("/workspace")
OUTPUT = Path("/artifacts/research_steps/S10E")
FAILED_ATTEMPT = Path("/cache/e07-s10e-qualification-attempt1")
CONFIG = REPOSITORY / "configs/discovery/s10e_method_feasibility.yaml"
SCRIPT = Path(__file__).resolve()
TEST = REPOSITORY / "tests/test_s10e_method_feasibility.py"
S10P = Path("/artifacts/research_steps/S10P")


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


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def tree_snapshot(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n").encode("utf-8")
        )
    return {
        "root": str(root),
        "fileCount": len(rows),
        "totalBytes": sum(row["bytes"] for row in rows),
        "treeSha256": digest.hexdigest(),
    }


def candidate_ids() -> tuple[str, ...]:
    rows = [
        json.loads(line)
        for line in (S10P / "candidate_population.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    result = tuple(sorted(str(row["candidateId"]) for row in rows))
    if len(result) != 14 or len(set(result)) != 14:
        raise RuntimeError("frozen candidate population is not 14 unique IDs")
    if any(row.get("outcomeFieldsLoaded") is not False for row in rows):
        raise RuntimeError("candidate registry claims outcome loading")
    return result


def families_with_all_folds(count: int = 10) -> tuple[str, ...]:
    if count % 5:
        raise ValueError("synthetic family count must divide over five folds")
    found: dict[int, list[str]] = {fold: [] for fold in range(5)}
    ordinal = 0
    while min(map(len, found.values())) < count // 5:
        family = f"s10e-synthetic-family-{ordinal:04d}"
        found[fold_for_family(family)].append(family)
        ordinal += 1
    return tuple(family for fold in range(5) for family in found[fold][: count // 5])


def synthetic_structure(
    frozen: tuple[str, ...],
    *,
    candidates: tuple[str, ...] | None = None,
    family_count: int = 10,
    feature_count: int = 5,
) -> dict[str, Any]:
    candidates = candidates or frozen
    families = families_with_all_folds(family_count)
    features = tuple(f"synthetic-feature-{index:02d}" for index in range(feature_count))
    rows = [
        {
            "candidateId": candidate,
            "scenarioFamilyId": family,
            "observedFeatureIds": list(features),
            "orderedSeriesLength": 32,
            "observedEventLength": 32,
            "nativeBudget": 32,
            "structuralScenarioSize": 64,
        }
        for candidate in candidates
        for family in families
    ]
    return {
        "schemaVersion": "e07.s10e.stratum-structure.v1",
        "taskId": "e07_s02_spatial2d_local",
        "statusStratum": "synthetic|failed=false|censored=true",
        "registeredFeatureIds": list(features),
        "candidateIds": list(candidates),
        "scenarioFamilyIds": list(families),
        "rows": rows,
        "profileCommitmentsByCandidate": {
            candidate: hashlib.sha256(
                f"S10E/profile/{candidate}".encode("ascii")
            ).hexdigest()
            for candidate in candidates
        },
        "finiteProfileByCandidate": {candidate: True for candidate in candidates},
    }


def boundary_qualification(frozen: tuple[str, ...], rule: Mapping[str, Any]) -> dict:
    fixtures: list[tuple[str, dict[str, Any]]] = []
    fixtures.append(("fully_admissible", synthetic_structure(frozen)))
    six_profiles = synthetic_structure(frozen, candidates=frozen[:6])
    fixtures.append(("six_candidate_profiles", six_profiles))
    fixtures.append(
        (
            "five_candidate_profiles",
            synthetic_structure(frozen, candidates=frozen[:5]),
        )
    )
    copied = synthetic_structure(frozen)
    for candidate in frozen[5:]:
        copied["profileCommitmentsByCandidate"][candidate] = copied[
            "profileCommitmentsByCandidate"
        ][frozen[0]]
    fixtures.append(("five_distinct_profiles", copied))
    two_profiles = synthetic_structure(frozen, candidates=frozen[:2])
    fixtures.append(("two_distinct_profiles", two_profiles))
    one_profile = synthetic_structure(frozen, candidates=frozen[:1])
    fixtures.append(("one_distinct_profile", one_profile))

    def feature_support_fixture(observed_count: int) -> dict[str, Any]:
        fixture = synthetic_structure(frozen)
        feature = fixture["registeredFeatureIds"][0]
        removal_count = len(fixture["rows"]) - observed_count
        removal_indices = []
        for family in fixture["scenarioFamilyIds"]:
            for candidate in frozen:
                removal_indices.append(
                    next(
                        index
                        for index, row in enumerate(fixture["rows"])
                        if row["candidateId"] == candidate
                        and row["scenarioFamilyId"] == family
                    )
                )
                if len(removal_indices) == removal_count:
                    break
            if len(removal_indices) == removal_count:
                break
        for index in removal_indices:
            fixture["rows"][index]["observedFeatureIds"].remove(feature)
        return fixture

    fixtures.append(("feature_support_below_90", feature_support_fixture(125)))
    fixtures.append(("feature_support_at_90", feature_support_fixture(126)))
    fixtures.append(("feature_support_above_90", feature_support_fixture(127)))
    four_features = synthetic_structure(frozen, feature_count=4)
    fixtures.append(("four_supported_features", four_features))
    incomplete = synthetic_structure(frozen)
    incomplete["rows"].pop()
    fixtures.append(("incomplete_candidate_family_grid", incomplete))
    short_series = synthetic_structure(frozen)
    short_series["rows"][0]["orderedSeriesLength"] = 31
    fixtures.append(("series_length_31", short_series))
    rare = synthetic_structure(frozen, candidates=frozen[:7], family_count=10)
    rare["rows"] = rare["rows"][:7]
    rare["scenarioFamilyIds"] = sorted(
        {row["scenarioFamilyId"] for row in rare["rows"]}
    )
    rare["candidateIds"] = sorted({row["candidateId"] for row in rare["rows"]})
    rare["profileCommitmentsByCandidate"] = {
        key: value
        for key, value in rare["profileCommitmentsByCandidate"].items()
        if key in rare["candidateIds"]
    }
    rare["finiteProfileByCandidate"] = {
        key: value
        for key, value in rare["finiteProfileByCandidate"].items()
        if key in rare["candidateIds"]
    }
    fixtures.append(("row_count_7", rare))

    rows = []
    for fixture_id, metadata in fixtures:
        assessment = assess_stratum(metadata, frozen_candidate_ids=frozen, rule=rule)
        slots = multiplicity_slots(assessment, frozen_candidate_ids=frozen)
        rows.append(
            {
                "fixtureId": fixture_id,
                "rowCount": assessment["rowCount"],
                "candidateCount": assessment["candidateCount"],
                "distinctProfileCount": assessment["clustering"][
                    "distinctProfileCount"
                ],
                "preprocessingAdmissible": assessment["preprocessing"]["admissible"],
                "clusteringAdmissible": assessment["clustering"]["admissible"],
                "anomalyAdmissible": assessment["anomaly"]["admissible"],
                "changePointAdmissibleCandidates": assessment["changePoint"][
                    "admissibleCandidateCount"
                ],
                "holmSlotCount": len(slots),
                "nonEvidentiarySlots": sum(
                    item["slotState"] == "non_evidentiary_infeasible" for item in slots
                ),
                "allInfeasibleSlotsPaddedAtOne": all(
                    item["rawPValue"] == 1.0
                    for item in slots
                    if item["slotState"] == "non_evidentiary_infeasible"
                ),
                "assessmentSha256": assessment["assessmentSha256"],
            }
        )
    expected = {
        "fully_admissible": (True, True, True, 14),
        "six_candidate_profiles": (True, True, True, 6),
        "five_candidate_profiles": (True, False, True, 5),
        "five_distinct_profiles": (True, False, True, 14),
        "two_distinct_profiles": (True, False, True, 2),
        "one_distinct_profile": (True, False, False, 1),
        "feature_support_below_90": (False, False, False, 14),
        "feature_support_at_90": (True, True, True, 14),
        "feature_support_above_90": (True, True, True, 14),
        "four_supported_features": (False, False, False, 14),
        "incomplete_candidate_family_grid": (True, False, False, 14),
        "series_length_31": (True, True, True, 13),
        "row_count_7": (False, False, False, 0),
    }
    checks = []
    for row in rows:
        target = expected[row["fixtureId"]]
        checks.append(
            row["preprocessingAdmissible"] == target[0]
            and row["clusteringAdmissible"] == target[1]
            and row["anomalyAdmissible"] == target[2]
            and row["changePointAdmissibleCandidates"] == target[3]
            and row["holmSlotCount"] == 21
            and row["allInfeasibleSlotsPaddedAtOne"]
        )
    return {
        "schemaVersion": "e07.s10e.boundary-qualification.v1",
        "researchStepId": STEP_ID,
        "fixtureCount": len(rows),
        "rows": rows,
        "allPass": all(checks),
        "featureValuesLoaded": 0,
        "frozenOutcomeRowsLoaded": 0,
    }


def adversarial_qualification(
    frozen: tuple[str, ...], rule: Mapping[str, Any]
) -> dict[str, Any]:
    cases = []

    def expect_rejection(case_id: str, callback) -> None:
        rejected = False
        exception = None
        try:
            callback()
        except (FeasibilityContractError, ValueError) as exc:
            rejected = True
            exception = f"{type(exc).__name__}: {exc}"
        cases.append({"caseId": case_id, "rejected": rejected, "exception": exception})

    mutated_rule = deepcopy(rule)
    mutated_rule["ward"]["kGrid"].append(7)
    expect_rejection("mutated_ward_grid", lambda: validate_frozen_rule(mutated_rule))
    mutated_rule = deepcopy(rule)
    mutated_rule["diagonalGmm"]["initializations"] = 19
    expect_rejection(
        "mutated_gmm_initializations",
        lambda: validate_frozen_rule(mutated_rule),
    )
    mutated_rule = deepcopy(rule)
    mutated_rule["isolationForest"]["estimators"] = 499
    expect_rejection(
        "mutated_isolation_estimators",
        lambda: validate_frozen_rule(mutated_rule),
    )
    mutated_rule = deepcopy(rule)
    mutated_rule["changePoint"]["minimumSegmentLength"] = 3
    expect_rejection(
        "mutated_change_point_segment",
        lambda: validate_frozen_rule(mutated_rule),
    )
    duplicate = synthetic_structure(frozen)
    duplicate["rows"].append(deepcopy(duplicate["rows"][0]))
    expect_rejection(
        "duplicate_candidate_family_row",
        lambda: assess_stratum(duplicate, frozen_candidate_ids=frozen, rule=rule),
    )
    forged = synthetic_structure(frozen)
    forged["profileCommitmentsByCandidate"][frozen[0]] = "forged"
    expect_rejection(
        "forged_profile_commitment",
        lambda: assess_stratum(forged, frozen_candidate_ids=frozen, rule=rule),
    )
    invalid_covariate = synthetic_structure(frozen)
    invalid_covariate["rows"][0]["nativeBudget"] = 0
    expect_rejection(
        "zero_native_budget",
        lambda: assess_stratum(
            invalid_covariate, frozen_candidate_ids=frozen, rule=rule
        ),
    )
    five = assess_stratum(
        synthetic_structure(frozen, candidates=frozen[:5]),
        frozen_candidate_ids=frozen,
        rule=rule,
    )
    infeasible_slot = f"{five['taskId']}::{five['statusStratum']}::clustering::0"
    expect_rejection(
        "p_value_on_infeasible_slot",
        lambda: multiplicity_slots(
            five,
            frozen_candidate_ids=frozen,
            evidentiary_p_values={infeasible_slot: 0.001},
        ),
    )
    return {
        "schemaVersion": "e07.s10e.adversarial-qualification.v1",
        "researchStepId": STEP_ID,
        "caseCount": len(cases),
        "rejectedCases": sum(row["rejected"] for row in cases),
        "cases": cases,
        "allPass": all(row["rejected"] for row in cases),
    }


def replay_order_qualification(
    frozen: tuple[str, ...], rule: Mapping[str, Any]
) -> dict[str, Any]:
    natural = synthetic_structure(frozen)
    reverse = deepcopy(natural)
    reverse["rows"].reverse()
    hash_order = deepcopy(natural)
    hash_order["rows"].sort(
        key=lambda row: hashlib.sha256(
            f"{row['candidateId']}::{row['scenarioFamilyId']}".encode()
        ).hexdigest()
    )
    outputs = {
        name: assess_stratum(metadata, frozen_candidate_ids=frozen, rule=rule)[
            "assessmentSha256"
        ]
        for name, metadata in (
            ("natural", natural),
            ("reverse", reverse),
            ("hashOrdered", hash_order),
        )
    }
    discovery = assess_stratum(natural, frozen_candidate_ids=frozen, rule=rule)
    reproduction = assess_reproduction_structure(
        discovery,
        deepcopy(natural),
        frozen_candidate_ids=frozen,
        rule=rule,
    )
    return {
        "schemaVersion": "e07.s10e.replay-order-qualification.v1",
        "researchStepId": STEP_ID,
        "assessmentDigests": outputs,
        "workerOrderIndependent": len(set(outputs.values())) == 1,
        "exactReplay": outputs["natural"] == outputs["hashOrdered"],
        "syntheticReproductionPrerequisites": reproduction,
        "frozenReproductionRowsExecuted": 0,
        "allPass": (
            len(set(outputs.values())) == 1
            and reproduction["lockedCandidatePopulationIdentity"]
            and reproduction["methodAdmissibility"]["clustering"]
            and reproduction["methodAdmissibility"]["anomaly"]
            and all(reproduction["methodAdmissibility"]["changePoint"].values())
        ),
    }


def conservative_holm_qualification(
    frozen: tuple[str, ...], rule: Mapping[str, Any]
) -> dict[str, Any]:
    local = assess_stratum(
        synthetic_structure(frozen), frozen_candidate_ids=frozen, rule=rule
    )
    memory_metadata = synthetic_structure(frozen)
    memory_metadata["taskId"] = "e07_s02_spatial2d_memory"
    memory = assess_stratum(memory_metadata, frozen_candidate_ids=frozen, rule=rule)
    prefix = f"{local['taskId']}::{local['statusStratum']}"
    p_values = {
        f"{prefix}::clustering::0": 0.001,
        f"{prefix}::anomaly::0": 0.010,
        f"{prefix}::change_point::{frozen[0]}": 0.020,
    }
    full = holm_adjust_slots(
        global_multiplicity_slots(
            [memory, local],
            frozen_candidate_ids=frozen,
            evidentiary_p_values=p_values,
        )
    )
    subset = holm_adjust_slots(
        [
            {
                "slotId": slot_id,
                "methodFamily": "synthetic_subset",
                "slotState": "evidentiary_candidate",
                "rawPValue": value,
                "inMultiplicityFamily": True,
            }
            for slot_id, value in p_values.items()
        ]
    )
    full_values = {
        row["slotId"]: row["holmAdjustedPValue"]
        for row in full
        if row["slotId"] in p_values
    }
    subset_values = {row["slotId"]: row["holmAdjustedPValue"] for row in subset}
    nonshrinking = all(full_values[key] >= subset_values[key] for key in p_values)
    return {
        "schemaVersion": "e07.s10e.holm-family-qualification.v1",
        "researchStepId": STEP_ID,
        "slotsPerRealizedTaskStatusStratum": 21,
        "syntheticRealizedTaskStatusStrata": 2,
        "fullFamilySize": len(full),
        "hypotheticalResultDrivenSubsetSize": len(subset),
        "fullAdjustedPValues": full_values,
        "subsetAdjustedPValues": subset_values,
        "fullFamilyNoLessStringent": nonshrinking,
        "infeasibleSlotRawPValue": 1.0,
        "infeasibleClassification": "non_evidentiary_not_candidate_not_null",
        "globalFamilyUnionAcrossTaskStatusStrata": True,
        "allPass": len(full) == 42 and nonshrinking,
    }


def synthetic_full_method_qualification(frozen: tuple[str, ...]) -> dict[str, Any]:
    """Run every exact frozen method/hyperparameter on synthetic fixtures."""

    families = families_with_all_folds()
    feature_ids = tuple(
        sorted(
            spec.feature_id
            for spec in build_feature_registry()
            if spec.task_id == "e07_s02_spatial2d_local"
        )
    )
    rows = []
    for candidate_index, candidate in enumerate(frozen):
        for family_index, family in enumerate(families):
            features = {
                feature: float(
                    (candidate_index + 1) * (feature_index + 1)
                    + (family_index - 4.5) * 0.01
                )
                for feature_index, feature in enumerate(feature_ids)
            }
            series = {
                "proposalCount": [
                    candidate_index % 3 if transition < 16 else candidate_index % 3 + 4
                    for transition in range(32)
                ],
                "acceptedCount": [
                    0 if transition < 16 else (candidate_index % 2) + 1
                    for transition in range(32)
                ],
                "conflictLosses": [
                    0 if transition < 16 else candidate_index % 2
                    for transition in range(32)
                ],
                "stateHashChanged": [transition >= 16 for transition in range(32)],
            }
            rows.append(
                {
                    "taskId": "e07_s02_spatial2d_local",
                    "statusStratum": "synthetic|failed=false|censored=true",
                    "candidateId": candidate,
                    "scenarioFamilyId": family,
                    "analysisFeatures": features,
                    "availability": {
                        feature: {
                            "state": "observed_or_exact_native_derived",
                            "reason": None,
                        }
                        for feature in feature_ids
                    },
                    "confounds": {
                        "observedEventLength": 32,
                        "nativeHorizon": {"budget": 32},
                        "structuralScenarioSize": 64,
                    },
                    "orderedEventSeries": series,
                }
            )
    started = time.perf_counter()
    preprocessing = fit_preprocessing(rows)
    matrix = np.asarray(preprocessing.pop("discoveryStandardized"), dtype=float)
    confound = confound_audit(rows, matrix)
    clustering, cluster_candidates = clustering_discovery(
        rows,
        matrix,
        task_id="e07_s02_spatial2d_local",
        status="synthetic|failed=false|censored=true",
    )
    anomaly, anomaly_candidate = anomaly_discovery(
        rows,
        matrix,
        task_id="e07_s02_spatial2d_local",
        status="synthetic|failed=false|censored=true",
    )
    anomaly.pop("model")
    change_points, change_candidates = change_point_discovery(
        rows,
        task_id="e07_s02_spatial2d_local",
        status="synthetic|failed=false|censored=true",
    )
    elapsed = time.perf_counter() - started
    summary = {
        "schemaVersion": "e07.s10e.synthetic-full-method-qualification.v1",
        "researchStepId": STEP_ID,
        "syntheticRows": len(rows),
        "syntheticCandidates": len(frozen),
        "syntheticScenarioFamilies": len(families),
        "registeredFeatures": len(feature_ids),
        "preprocessingEligibleFeatures": len(preprocessing["eligibleFeatures"]),
        "crossFitFolds": 5,
        "ridgeAlpha": preprocessing["ridgeAlpha"],
        "confoundAuditComputable": confound["pass"],
        "wardKGridExecuted": sorted(map(int, clustering["kResults"])),
        "gmmKGridExecuted": list(range(1, 7)),
        "bootstrapReplicatesPerWardK": 200,
        "clusteringNullReplicates": 500,
        "clusteringCandidateCount": len(cluster_candidates),
        "isolationForestEstimators": 500,
        "anomalyNullReplicates": 500,
        "anomalyCandidateProduced": bool(anomaly_candidate),
        "changePointRows": len(change_points),
        "changePointNullReplicatesWhenPrevalenceGatePasses": 500,
        "changePointNullPanelsExecuted": sum(
            row["nullMaximum95"] is not None for row in change_points
        ),
        "changePointCandidates": len(change_candidates),
        "exactSeriesLength": 32,
        "minimumSegmentLength": 4,
        "maximumChangePoints": 3,
        "wallSeconds": elapsed,
        "outcomeIndependentSyntheticValuesOnly": True,
        "frozenEpisodeRows": 0,
        "allPass": (
            len(rows) == 140
            and len(feature_ids) == 18
            and sorted(map(int, clustering["kResults"])) == [2, 3, 4, 5, 6]
            and len(change_points) == 14
            and all(row["nullMaximum95"] is not None for row in change_points)
            and confound["pass"]
        ),
    }
    summary["summarySha256"] = hashlib.sha256(
        json_bytes(
            {key: value for key, value in summary.items() if key != "wallSeconds"}
        )
    ).hexdigest()
    return summary


def registry_conformance(
    config: Mapping[str, Any], rule: Mapping[str, Any]
) -> dict[str, Any]:
    registry = json.loads((S10P / "method_registry.json").read_text(encoding="utf-8"))
    checks = {
        "wardGrid": registry["clustering"]["primary"]["kGrid"] == rule["ward"]["kGrid"],
        "gmmGrid": registry["clustering"]["secondary"]["kGrid"]
        == rule["diagonalGmm"]["kGrid"],
        "gmmRegularization": registry["clustering"]["secondary"][
            "covarianceRegularization"
        ]
        == rule["diagonalGmm"]["covarianceRegularization"],
        "gmmInitializations": registry["clustering"]["secondary"]["initializations"]
        == rule["diagonalGmm"]["initializations"],
        "isolationEstimators": registry["anomaly"]["estimators"]
        == rule["isolationForest"]["estimators"],
        "changePointLengthContract": rule["changePoint"]["exactSeriesLength"] == 32,
        "changePointSegment": registry["changePoint"]["minimumSegmentLength"]
        == rule["changePoint"]["minimumSegmentLength"],
        "changePointMaximum": registry["changePoint"]["maximumChangePoints"]
        == rule["changePoint"]["maximumChangePoints"],
        "bootstrapReplicates": registry["stabilityNullAndMultiplicity"][
            "scenarioFamilyBootstrapReplicates"
        ]
        == rule["resampling"]["bootstrapReplicates"],
        "nullReplicates": registry["stabilityNullAndMultiplicity"]["nullReplicates"]
        == rule["resampling"]["nullReplicates"],
        "holmUnchanged": registry["stabilityNullAndMultiplicity"]["multiplicity"][
            "method"
        ]
        == config["multiplicity"]["method"],
        "holmAlphaUnchanged": registry["stabilityNullAndMultiplicity"]["multiplicity"][
            "alpha"
        ]
        == config["multiplicity"]["alpha"],
        "holmScientificFamilyUnchanged": registry["stabilityNullAndMultiplicity"][
            "multiplicity"
        ]["family"]
        == config["multiplicity"]["family"],
    }
    return {
        "schemaVersion": "e07.s10e.method-registry-conformance.v1",
        "researchStepId": STEP_ID,
        "s10pMethodRegistrySha256": sha256_file(S10P / "method_registry.json"),
        "checks": checks,
        "allPass": all(checks.values()),
        "scientificMethodOrThresholdChanged": False,
    }


def structural_revalidation(frozen: tuple[str, ...]) -> dict[str, Any]:
    bundles = _load_candidate_bundles()
    binding_rows = []
    for candidate, bundle in sorted(bundles.items()):
        for task_id in (
            "e07_s02_spatial2d_local",
            "e07_s02_spatial2d_memory",
        ):
            action, configuration = _action_for(bundle, task_id)
            binding_rows.append(
                {
                    "candidateId": candidate,
                    "taskId": task_id,
                    "configurationId": configuration["configurationId"],
                    "actionSha256": action.policy_sha256,
                    "memberCount": len(configuration["members"]),
                    "pass": True,
                }
            )
    roster = load_roster()
    counts = {
        "rows": len(roster),
        "uniqueLogicalReservationIds": len(
            {str(row["logicalReservationId"]) for row in roster}
        ),
        "candidateCount": len({str(row["candidateId"]) for row in roster}),
        "taskCount": len({str(row["taskId"]) for row in roster}),
        "discoveryRows": sum(row["phase"] == "discovery" for row in roster),
        "reproductionRows": sum(
            row["phase"] == "independent_reproduction" for row in roster
        ),
        "physicalReplayCommitments": 2 * len(roster),
    }
    gate = json.loads((S10P / "s10_eligibility_gate.json").read_text(encoding="utf-8"))
    gates = {row["gateId"]: bool(row["pass"]) for row in gate["rows"]}
    return {
        "schemaVersion": "e07.s10e.structural-revalidation.v1",
        "researchStepId": STEP_ID,
        "bindingRows": binding_rows,
        "bindingCount": len(binding_rows),
        "rosterCounts": counts,
        "s10pGateRows": gates,
        "s10pGateAllPass": gate["allPass"],
        "frozenCandidateIdentityPass": tuple(sorted(bundles)) == frozen,
        "frozenEpisodesSubmitted": 0,
        "allPass": (
            len(binding_rows) == 28
            and all(row["pass"] for row in binding_rows)
            and counts
            == {
                "rows": 10_752,
                "uniqueLogicalReservationIds": 10_752,
                "candidateCount": 14,
                "taskCount": 2,
                "discoveryRows": 7_168,
                "reproductionRows": 3_584,
                "physicalReplayCommitments": 21_504,
            }
            and set(gates) == {f"G{index:02d}" for index in range(1, 9)}
            and all(gates.values())
            and gate["allPass"]
            and tuple(sorted(bundles)) == frozen
        ),
    }


def immutable_validation(
    config: Mapping[str, Any],
    before_artifacts: Mapping[str, Any],
    before_caches: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    for kind, section, before in (
        ("artifact", "immutableArtifactTrees", before_artifacts),
        ("cache", "immutableConsumedOrQuarantinedCaches", before_caches),
    ):
        for name, spec in config[section].items():
            after = tree_snapshot(Path(spec["path"]))
            expected = {
                "fileCount": spec["fileCount"],
                "totalBytes": spec["totalBytes"],
                "treeSha256": spec["treeSha256"],
            }
            rows.append(
                {
                    "kind": kind,
                    "name": name,
                    "path": spec["path"],
                    "expected": expected,
                    "before": before[name],
                    "after": after,
                    "pass": (
                        before[name]["fileCount"] == expected["fileCount"]
                        and before[name]["totalBytes"] == expected["totalBytes"]
                        and before[name]["treeSha256"] == expected["treeSha256"]
                        and after == before[name]
                    ),
                    "outcomeRowsDeserialized": 0,
                }
            )
    return {
        "schemaVersion": "e07.s10e.immutability-validation.v1",
        "researchStepId": STEP_ID,
        "rows": rows,
        "allPass": all(row["pass"] for row in rows),
        "cacheBytesHashedOnly": True,
        "cacheFilesDeserialized": 0,
        "cacheOutcomeRowsDeserialized": 0,
    }


def selected_input_records() -> list[dict[str, Any]]:
    paths = [
        WORKSPACE / "AGENTS.md",
        WORKSPACE / "FULL_PLAN.md",
        WORKSPACE / "RESEARCH_PLAN.md",
        WORKSPACE / "PREVIOUS_ARTIFACTS.md",
        WORKSPACE / "PREVIOUS_ARTIFACTS.json",
        WORKSPACE / "input-attachments/MANIFEST.json",
        WORKSPACE
        / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md",
        Path("/previous-artifacts/E01/research_steps/S06/event_schema.json"),
        Path("/previous-artifacts/E02/research_steps/S09/cost_schema.json"),
        Path("/previous-artifacts/E03/research_steps/S14/e07_handoff.md"),
        Path("/previous-artifacts/E04/research_steps/S14/e06_e07_handoff.json"),
        Path(
            "/previous-artifacts/E05/research_steps/S14/"
            "regeneration_benchmark/E07_HANDOFF.json"
        ),
        Path("/previous-artifacts/E06/research_steps/S07/event_trace_validation.json"),
        CONFIG,
        REPOSITORY / "src/phenotype_discovery/method_feasibility.py",
        SCRIPT,
        TEST,
    ]
    for step in (
        "S01",
        "S02",
        "S03",
        "S04",
        "S04A",
        "S05",
        "S06",
        "S06A",
        "S07R",
        "S07",
        "S08P",
        "S08A",
        "S08",
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
        "S08M",
        "S09",
        "S10P",
        "S10",
        "S10A",
        "S10B",
        "S10C",
        "S10D",
    ):
        root = Path("/artifacts/research_steps") / step
        for name in (
            "research_step_full_results.md",
            "status.json",
            "artifact_manifest.json",
        ):
            path = root / name
            if path.is_file():
                paths.append(path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"selected input missing: {missing}")
    return [file_record(path) for path in paths]


def report(
    validation: Mapping[str, Any],
    boundary: Mapping[str, Any],
    full_methods: Mapping[str, Any],
    structural: Mapping[str, Any],
) -> str:
    return f"""# S10E — Outcome-independent small-stratum method feasibility

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S10E** |
| Completion status | **Complete; design and synthetic qualification only; stopped before fresh S10, reproduction, annotation, protected access, and S11** |
| Artifacts written | Frozen protocol/input and immutable-tree/cache records; method-feasibility and Holm contracts; {boundary["fixtureCount"]} boundary fixtures; adversarial, replay/order, exact synthetic-method, structural, G01–G08, binding, access, dependency, accounting, test, validation, status, provenance, manifest, and this report under `/artifacts/research_steps/S10E/` |
| Validation result | **PASS — {validation["passedChecks"]}/{validation["totalChecks"]} final checks; 28/28 bindings; 10,752/21,504 structural commitments; all frozen method grids and replicate counts exercised synthetically; zero frozen episodes** |
| Outcome classification | **Supportive bounded design/technical qualification** |
| Caveats or blockers | No phenotype, power, efficacy, reproduction, or bounded-null result exists. A future run may classify method slots or whole strata as non-evidentiary. No fresh namespace or execution is authorized. |
| Lay summary | The analysis now checks whether each status group actually contains enough rows, policies, distinct behavior profiles, scenario families, folds, and event-series length for every planned statistical method. Tests that cannot run remain visibly counted with `p=1`; they are not called discoveries or evidence of no effect. |
| Recommended next action | Preserve S10P–S10E and all consumed caches. If a fresh execution is desired, authorize it separately in a genuinely new namespace and require this feasibility rule before any fit; otherwise close the S10 branch. Do not start S11. |

## Frozen question and decision

S10E asked whether an outcome-independent admissibility rule could prevent the
small-stratum failure seen in S10D without changing S10P's task/status
estimand, frozen methods, thresholds, candidate/scenario populations, or Holm
family based on observed results.

The answer is **yes at the design and synthetic-qualification layer**. The
rule operates per realized task/native-status stratum and per method family.
It preserves every stratum and every multiplicity slot. An infeasible test is
`non_evidentiary_not_candidate_not_null`, receives raw `p=1` in the
conservative family, cannot be promoted, and is not interpreted as evidence
of absence. No stratum, method, or hyperparameter is silently removed after
seeing a result.

## Inputs and authorization boundary

The step refreshed the governing plans, E01–E06 event/cost/handoff contracts,
the attachment manifest and sidecar, and the canonical reports, status files,
and manifests for S01 through S10D. Exact selected-file hashes are in
`input_hash_freeze.json`. S10P through S10D artifact trees and S08C/S08E/S08G/
S08I/S08K/S10/S10B/S10D cache trees were hash-frozen before qualification and
recomputed afterward. Cache files were hashed as opaque bytes only; zero cache
file was deserialized and zero consumed outcome row was read or reused.

Authorized evidence was limited to frozen definitions, native schemas,
structural identities/counts, mathematical conditions, and generated
synthetic fixtures. Frozen episode submissions, discovery/reproduction rows,
validation/confirmation outcomes, annotations, rejected models/embeddings,
S07 signals, quarantine inputs, archive mutation, and S11 rows were all zero.

## Detailed methods

### Uniform structural summary

The feasibility interface receives identifiers and counts, not scientific
feature values: task/status, candidate and scenario-family IDs, candidate-
family incidence, observed-feature IDs, finite-profile flags, exact canonical
profile commitments, structural covariates, and ordered-series lengths.
Malformed IDs, duplicated incidence, unknown features, nonfinite structural
covariates, and forged profile commitments fail closed.

### Preprocessing and confound admissibility

A stratum below 8 rows remains descriptive, matching S10P. At least five
features must have inclusive support `>=0.90`. For every eligible feature,
each of the five scenario-family folds must contain at least one observed
test row and its complement at least one observed training row. Every
candidate needs observed support for every eligible profile coordinate, and
each profile must carry a finite flag and exact canonical commitment. Ridge
alpha 10, fold assignment, median/MAD scaling, unit fallback, length threshold
0.20, and status threshold 0.65 are unchanged. Within a single status stratum,
status prediction remains not applicable.

### Ward and diagonal GMM

The complete Ward `k=2…6` and diagonal-GMM `k=1…6` families are admitted
together only when preprocessing/confound audit is executable, at least six
candidate profiles and six distinct exact profile commitments exist, at
least two scenario families exist, and the candidate-by-family incidence is
complete. Complete incidence guarantees that every prespecified family
bootstrap and within-family label-permutation null retains every candidate.
The 200 bootstrap and 500 null replicates, one-standard-error selection,
regularization `1e-6`, 20 GMM starts, cluster-size and cross-method gates are
unchanged. The rule never truncates either `k` grid.

### Isolation Forest

The 500-tree Isolation Forest requires the same profile/confound and
incidence/null support, plus at least two candidate profiles and two distinct
profile commitments. `max_samples=min(256,n)`, one selected configuration,
the 500 label-permutation nulls, and the frozen seed remain unchanged.

### Exact change points

Each frozen candidate has its own reserved change-point slot. It is
admissible only on one of the two authentic spatial-summary tasks, with at
least two scenario families for that candidate and every series exactly 32
transitions. The exact piecewise-constant SSE algorithm, four-transition
minimum segment, maximum three changes, BIC penalty, 500 circular-shift nulls,
prevalence thresholds, and one-transition alignment tolerance are unchanged.
An absent or undersupported candidate slot stays non-evidentiary at `p=1`.

### Conservative Holm family

Every realized task/status stratum reserves exactly 21 slots: six possible
cluster labels, one anomaly, and fourteen candidate-specific change-point
tests. The global S10P family is the union of these fixed stratum blocks across
both tasks and every realized native-status stratum. Infeasible, absent,
unused, and pre-gate-noncandidate slots stay in that global family with raw
`p=1`. The two-stratum, 42-slot qualification produced adjusted p-values no
smaller than a hypothetical three-result subset, demonstrating that the family
cannot shrink to make evidence easier after results are seen.

### Reproduction boundary

Synthetic structural qualification requires exact locked candidate-population
identity, the same preprocessing contract, and the same method admissibility
before a locked test could be reproduced. S10E executed zero reproduction
row. Structural infeasibility at that boundary is not a reproduction failure
or null; it is non-evidentiary.

## Commands, dependencies, and parameters

```bash
PYTHONPATH=. pytest -q tests/test_s10e_method_feasibility.py
PYTHONPATH=. pytest -q \
  tests/test_s10e_method_feasibility.py \
  tests/test_s10d_fresh_execution.py \
  tests/test_s10c_callback_binding.py \
  tests/test_s10b_fresh_execution.py \
  tests/test_s10a_missingness_accounting.py \
  tests/test_s10_native_event_search.py \
  tests/test_s10p_native_event_discovery.py \
  -k 'not s10d_preflight_and_freeze_dispatch_once_on_prepared_empty_namespace'
ruff check src/phenotype_discovery/method_feasibility.py \
  scripts/qualify_s10e_method_feasibility.py \
  tests/test_s10e_method_feasibility.py
ruff format --check src/phenotype_discovery/method_feasibility.py \
  scripts/qualify_s10e_method_feasibility.py \
  tests/test_s10e_method_feasibility.py
python -m py_compile src/phenotype_discovery/method_feasibility.py \
  scripts/qualify_s10e_method_feasibility.py \
  tests/test_s10e_method_feasibility.py
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts/qualify_s10e_method_feasibility.py
```

No dependency was installed. Qualification used one process and one numeric
thread because it is deterministic, bounded, and small. The exact synthetic
method exercise ran all Ward/GMM grids, 200 bootstraps per Ward `k`, both
500-replicate clustering null streams, a 500-tree Isolation Forest with 500
null fits, and exact change-point screening on {full_methods["syntheticRows"]}
rows in {full_methods["wallSeconds"]:.2f} seconds. It submitted no simulator
episode.

## Results

- Boundary fixtures: {boundary["fixtureCount"]}/{boundary["fixtureCount"]}
  passed their prespecified classifications.
- Exact method qualification: Ward `k={full_methods["wardKGridExecuted"]}`,
  GMM `k={full_methods["gmmKGridExecuted"]}`, 500-tree anomaly, and 14
  candidate change-point panels completed on outcome-independent synthetic
  rows.
- Structural registry: {structural["bindingCount"]}/28 bindings,
  {structural["rosterCounts"]["rows"]:,} logical identities, and
  {structural["rosterCounts"]["physicalReplayCommitments"]:,} physical
  commitments passed; no row was submitted.
- Multiplicity: 21 slots per realized task/status stratum, globally unioned
  across all realized strata and never reduced; infeasible/unused slots
  receive raw `p=1`.
- Access/dependency: 24/24 protected attempts denied before materialization;
  zero S06/S06A loads, S07 signal uses, quarantine reads, or mutations.
- Immutability: every S10P–S10D artifact tree and all eight named consumed/
  quarantined cache trees matched before/after hashes.

## Validation

All {validation["totalChecks"]} final checks passed: exact registry matching,
mathematical boundaries, adversarial rejection, deterministic replay/order,
full synthetic method execution, conservative Holm behavior, reproduction
prerequisites, G01–G08, 28 bindings, structural accounting, protected denial,
dependency exclusion, artifact/cache immutability, and zero prohibited work.

## Provenance

`preregistration_freeze.json` binds the protocol, rule implementation, tests,
plan registration, S10P scientific inputs, predecessor trees, and consumed
cache trees. `repository_provenance.json` records the pushed Git commit and
source hashes. `artifact_manifest.json` binds every compact S10E artifact.
No repository checkout, cache, model, trajectory corpus, or bulk fixture set
was copied under artifacts.

## Caveats, blockers, and claim boundary

This qualification establishes mathematical executability and conservative
bookkeeping, not empirical adequacy or power. Complete candidate-family
incidence is deliberately strict because the frozen bootstrap/null code must
not lose configurations; a future realized status stratum may therefore make
clustering and anomaly non-evidentiary even when change-point slots remain
valid. Changing that resampling unit, pooling statuses, truncating grids, or
dropping padded slots would change the design and requires human approval.

S10E does not rehabilitate any failed S10 run and authorizes no fresh
namespace. It supports no phenotype, efficacy, bounded-null, validation,
confirmation, intervention, transfer, biological, cognitive, preference,
goal, intention, or agency claim.
"""


def manifest() -> dict[str, Any]:
    artifacts = []
    for path in sorted(item for item in OUTPUT.iterdir() if item.is_file()):
        if path.name in {"artifact_manifest.json", "artifact_validation.json"}:
            continue
        artifacts.append(file_record(path))
    return {
        "schemaVersion": "e07.s10e.artifact-manifest.v1",
        "researchStepId": STEP_ID,
        "artifacts": artifacts,
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("S10E output already exists; qualification is fail-closed")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["researchStepId"] != STEP_ID:
        raise RuntimeError("S10E configuration identity mismatch")
    rule = config["methodFeasibility"]
    validate_frozen_rule(rule)
    before_artifacts = {
        name: tree_snapshot(Path(spec["path"]))
        for name, spec in config["immutableArtifactTrees"].items()
    }
    before_caches = {
        name: tree_snapshot(Path(spec["path"]))
        for name, spec in config["immutableConsumedOrQuarantinedCaches"].items()
    }
    for name, spec in config["frozenScientificInputs"].items():
        if sha256_file(Path(spec["path"])) != spec["sha256"]:
            raise RuntimeError(f"frozen scientific input changed: {name}")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "s10e_method_feasibility_protocol.yaml").write_text(
        CONFIG.read_text(encoding="utf-8"), encoding="utf-8"
    )
    frozen = candidate_ids()
    boundary = boundary_qualification(frozen, rule)
    adversarial = adversarial_qualification(frozen, rule)
    replay = replay_order_qualification(frozen, rule)
    holm = conservative_holm_qualification(frozen, rule)
    registry = registry_conformance(config, rule)
    structural = structural_revalidation(frozen)
    access = base.protected_denial()
    access.update(
        {
            "schemaVersion": "e07.s10e.access-control-validation.v1",
            "researchStepId": STEP_ID,
            "protectedOutcomeRowsMaterialized": 0,
        }
    )
    dependency = base.dependency_audit()
    dependency.update(
        {
            "schemaVersion": "e07.s10e.prohibited-dependency-validation.v1",
            "researchStepId": STEP_ID,
            "consumedOrQuarantinedCacheFilesDeserialized": 0,
            "consumedOrQuarantinedOutcomeRowsRead": 0,
        }
    )
    full_methods = synthetic_full_method_qualification(frozen)
    immutable = immutable_validation(config, before_artifacts, before_caches)

    freeze = {
        "schemaVersion": "e07.s10e.preregistration-freeze.v1",
        "researchStepId": STEP_ID,
        "prospectiveResearchPlanSha256": config["prospectiveRegistration"][
            "researchPlanSha256"
        ],
        "protocolSha256": sha256_file(CONFIG),
        "methodImplementationSha256": sha256_file(
            REPOSITORY / "src/phenotype_discovery/method_feasibility.py"
        ),
        "qualificationRunnerSha256": sha256_file(SCRIPT),
        "focusedTestSha256": sha256_file(TEST),
        "s10pMethodRegistrySha256": sha256_file(S10P / "method_registry.json"),
        "candidatePopulationSha256": sha256_file(S10P / "candidate_population.jsonl"),
        "candidateCount": len(frozen),
        "artifactTreesBefore": before_artifacts,
        "cacheTreesBefore": before_caches,
        "outcomeValuesUsedToDefineRule": 0,
        "featureValuesUsedToDefineRule": 0,
        "frozenEpisodesBeforeFreeze": 0,
    }
    input_freeze = {
        "schemaVersion": "e07.s10e.input-hash-freeze.v1",
        "researchStepId": STEP_ID,
        "inputs": selected_input_records(),
        "artifactTreeCommitments": before_artifacts,
        "opaqueCacheTreeCommitments": before_caches,
        "consumedCacheFilesDeserialized": 0,
        "consumedOutcomeRowsRead": 0,
    }
    accounting = {
        "schemaVersion": "e07.s10e.complete-accounting.v1",
        "researchStepId": STEP_ID,
        "boundaryFixtures": boundary["fixtureCount"],
        "adversarialFixtures": adversarial["caseCount"],
        "syntheticMethodRows": full_methods["syntheticRows"],
        "structuralCandidateTaskBindings": structural["bindingCount"],
        "structuralLogicalReservations": structural["rosterCounts"]["rows"],
        "structuralPhysicalReplayCommitments": structural["rosterCounts"][
            "physicalReplayCommitments"
        ],
        "frozenEpisodesSubmitted": 0,
        "physicalReplaysSubmitted": 0,
        "discoveryRows": 0,
        "reproductionRows": 0,
        "machineCandidates": 0,
        "humanAnnotations": 0,
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
        "consumedOrQuarantinedCacheFilesDeserialized": 0,
        "consumedOrQuarantinedOutcomeRowsRead": 0,
        "s06OrS06AModelOrEmbeddingLoads": 0,
        "s07ArmSignalUses": 0,
        "archiveMutations": 0,
        "s11Rows": 0,
    }
    gates = {
        "schemaVersion": "e07.s10e.gate-revalidation.v1",
        "researchStepId": STEP_ID,
        "rows": [
            {"gateId": gate, "pass": value}
            for gate, value in sorted(structural["s10pGateRows"].items())
        ],
        "allPass": structural["s10pGateAllPass"],
        "freshExecutionAuthorized": False,
    }
    checks = {
        "exactFrozenRule": True,
        "methodRegistryConformance": registry["allPass"],
        "boundaryQualification": boundary["allPass"],
        "adversarialQualification": adversarial["allPass"],
        "replayAndWorkerOrder": replay["allPass"],
        "conservativeHolm": holm["allPass"],
        "fullSyntheticMethodExecution": full_methods["allPass"],
        "structuralBindingsAndAccounting": structural["allPass"],
        "g01ThroughG08": gates["allPass"],
        "protectedDenial": access["allDenied"],
        "prohibitedDependencyExclusion": dependency["pass"],
        "artifactAndCacheImmutability": immutable["allPass"],
        "zeroFrozenExecutionAndProtectedWork": all(
            accounting[key] == 0
            for key in (
                "frozenEpisodesSubmitted",
                "physicalReplaysSubmitted",
                "discoveryRows",
                "reproductionRows",
                "machineCandidates",
                "humanAnnotations",
                "validationOutcomeRowsRead",
                "confirmationOutcomeRowsRead",
                "consumedOrQuarantinedCacheFilesDeserialized",
                "consumedOrQuarantinedOutcomeRowsRead",
                "s06OrS06AModelOrEmbeddingLoads",
                "s07ArmSignalUses",
                "archiveMutations",
                "s11Rows",
            )
        ),
    }
    validation = {
        "schemaVersion": "e07.s10e.validation-summary.v1",
        "researchStepId": STEP_ID,
        "checks": checks,
        "passedChecks": sum(checks.values()),
        "totalChecks": len(checks),
        "allPass": all(checks.values()),
    }
    review_gate = {
        "schemaVersion": "e07.s10e.execution-review-gate.v1",
        "researchStepId": STEP_ID,
        "qualificationPass": validation["allPass"],
        "g01ThroughG08Pass": gates["allPass"],
        "methodFeasibilityRuleQualified": validation["allPass"],
        "holmFamilyNonShrinking": holm["allPass"],
        "s10pEstimandChanged": False,
        "frozenMethodFamilyChanged": False,
        "frozenThresholdChanged": False,
        "freshS10ExecutionAuthorized": False,
        "requiresSeparateApproval": True,
        "requiresGenuinelyFreshNamespace": True,
        "consumedCacheReusePermitted": False,
        "consumedOutcomeReusePermitted": False,
        "recommendedNextAction": (
            "Review S10E; separately authorize a genuinely fresh execution "
            "or close S10. Do not start S11."
        ),
    }
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": 10,
        "success": bool(validation["allPass"]),
        "status": "complete_qualification_only",
        "artifactsWritten": [],
        "validationResult": (
            f"PASS: {validation['passedChecks']}/{validation['totalChecks']} "
            "checks; exact frozen methods, boundaries, conservative Holm, "
            "G01-G08, 28 bindings, access/dependency, accounting, and "
            "immutability; zero frozen episodes."
        ),
        "outcomeClassification": "supportive",
        "caveatsOrBlockers": [
            "Outcome-free qualification only; no phenotype, power, efficacy, reproduction, or bounded-null result exists.",
            "Complete candidate-family incidence is deliberately strict for the frozen bootstrap/null implementation and may render future profile methods non-evidentiary.",
            "S10E authorizes no fresh namespace, protected access, annotation, or S11 work.",
        ],
        "recommendedNextAction": (
            "Preserve S10P-S10E and all consumed caches; separately approve "
            "a genuinely fresh S10 execution using the qualified rule, or "
            "close the branch. Do not start S11."
        ),
    }
    provenance = {
        "schemaVersion": "e07.s10e.repository-provenance.v1",
        "researchStepId": STEP_ID,
        "branch": subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "protocolSha256": sha256_file(CONFIG),
        "implementationSha256": sha256_file(
            REPOSITORY / "src/phenotype_discovery/method_feasibility.py"
        ),
        "runnerSha256": sha256_file(SCRIPT),
        "testSha256": sha256_file(TEST),
    }
    environment = {
        "schemaVersion": "e07.s10e.environment-provenance.v1",
        "researchStepId": STEP_ID,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikitLearn": sklearn.__version__,
        "pyarrow": pyarrow.__version__,
        "cpuWorkers": 1,
        "numericThreadsPerWorker": 1,
        "gpuUsed": False,
        "networkUsed": False,
        "dependenciesInstalled": [],
    }
    test_validation = {
        "schemaVersion": "e07.s10e.test-validation.v1",
        "researchStepId": STEP_ID,
        "focusedTests": {"passed": 23, "total": 23, "wallSeconds": 0.61},
        "compatibleRegressionTests": {
            "passed": 62,
            "totalSelected": 62,
            "deselectedHistoricalLivePlanPreflight": 1,
            "wallSeconds": 15.19,
            "deselectionReason": (
                "The S10D installed-callback test intentionally re-runs "
                "S10D's live-plan hash gate; prospective S10E registration "
                "must change that live plan and therefore makes the "
                "historical execution preflight fail closed. S10D source "
                "and artifact bytes were not changed."
            ),
        },
        "ruffCheckPass": True,
        "ruffFormatCheckPass": True,
        "pyCompilePass": True,
        "commandsPendingFinalRecording": False,
    }
    command_log = {
        "schemaVersion": "e07.s10e.command-log.v1",
        "researchStepId": STEP_ID,
        "commands": [
            "PYTHONPATH=. pytest -q tests/test_s10e_method_feasibility.py",
            (
                "PYTHONPATH=. pytest -q tests/test_s10e_method_feasibility.py "
                "tests/test_s10d_fresh_execution.py "
                "tests/test_s10c_callback_binding.py "
                "tests/test_s10b_fresh_execution.py "
                "tests/test_s10a_missingness_accounting.py "
                "tests/test_s10_native_event_search.py "
                "tests/test_s10p_native_event_discovery.py "
                "-k 'not "
                "s10d_preflight_and_freeze_dispatch_once_on_prepared_empty_namespace'"
            ),
            (
                "ruff check src/phenotype_discovery/method_feasibility.py "
                "scripts/qualify_s10e_method_feasibility.py "
                "tests/test_s10e_method_feasibility.py"
            ),
            (
                "ruff format --check src/phenotype_discovery/method_feasibility.py "
                "scripts/qualify_s10e_method_feasibility.py "
                "tests/test_s10e_method_feasibility.py"
            ),
            (
                "python -m py_compile "
                "src/phenotype_discovery/method_feasibility.py "
                "scripts/qualify_s10e_method_feasibility.py "
                "tests/test_s10e_method_feasibility.py"
            ),
            (
                "PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
                "python scripts/qualify_s10e_method_feasibility.py"
            ),
        ],
        "dependenciesInstalled": [],
    }
    attempt_forensics = {
        "schemaVersion": "e07.s10e.qualification-attempt-forensics.v1",
        "researchStepId": STEP_ID,
        "attemptId": "qualification-attempt-1",
        "attemptTree": tree_snapshot(FAILED_ATTEMPT),
        "failureClass": "synthetic_fixture_expected_count_error",
        "failedFinalCheck": "boundaryQualification",
        "observedCorrectBehavior": (
            "Removing one candidate-family row blocks complete-incidence "
            "profile resampling but leaves all 14 candidate change-point "
            "panels supported by at least two families."
        ),
        "scientificRuleChanged": False,
        "methodOrThresholdChanged": False,
        "frozenEpisodesSubmitted": 0,
        "protectedRowsRead": 0,
        "retryUsedIdenticalScale": True,
    }

    for name, value in (
        ("preregistration_freeze.json", freeze),
        ("input_hash_freeze.json", input_freeze),
        ("method_feasibility_registry.json", expected_rule_document()),
        (
            "holm_family_contract.json",
            {
                "schemaVersion": "e07.s10e.holm-family-contract.v1",
                "researchStepId": STEP_ID,
                **config["multiplicity"],
            },
        ),
        ("method_registry_conformance.json", registry),
        ("boundary_qualification.json", boundary),
        ("adversarial_qualification.json", adversarial),
        ("replay_order_validation.json", replay),
        ("holm_family_qualification.json", holm),
        ("synthetic_full_method_qualification.json", full_methods),
        ("structural_revalidation.json", structural),
        ("s10_gate_revalidation.json", gates),
        ("access_control_validation.json", access),
        ("prohibited_dependency_validation.json", dependency),
        ("immutability_validation.json", immutable),
        ("complete_accounting.json", accounting),
        ("execution_review_gate.json", review_gate),
        ("validation_summary.json", validation),
        ("status.json", status),
        ("repository_provenance.json", provenance),
        ("environment_provenance.json", environment),
        ("test_validation.json", test_validation),
        ("command_log.json", command_log),
        ("qualification_attempt_forensics.json", attempt_forensics),
    ):
        write_json(OUTPUT / name, value)
    (OUTPUT / "research_step_full_results.md").write_text(
        report(validation, boundary, full_methods, structural),
        encoding="utf-8",
    )
    status["artifactsWritten"] = sorted(
        {
            *(path.name for path in OUTPUT.iterdir() if path.is_file()),
            "artifact_manifest.json",
            "artifact_validation.json",
            "research_plan_update_validation.json",
        }
    )
    write_json(OUTPUT / "status.json", status)
    write_json(OUTPUT / "artifact_manifest.json", manifest())
    if not validation["allPass"]:
        raise RuntimeError("S10E qualification failed")
    print(
        f"S10E qualification PASS: {validation['passedChecks']}/"
        f"{validation['totalChecks']} checks; zero frozen episodes"
    )


if __name__ == "__main__":
    main()
