"""S10F execution of frozen S10P methods through the S10E feasibility rule.

This module does not alter a scientific method, threshold, or population.  It
adds the prespecified outcome-independent admissibility check before each
frozen method is called and expands the one global Holm family to all 21 fixed
slots in every realized task/native-status stratum.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import IsolationForest

from src.phenotype_discovery import method_feasibility as feasibility
from src.phenotype_discovery import search
from src.phenotype_discovery.native_features import canonical_sha256


_LAST_DISCOVERY_AUDIT: dict[str, Any] | None = None
_LAST_REPRODUCTION_AUDIT: dict[str, Any] | None = None


def _frozen_candidate_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    candidate_ids = sorted({str(row["candidateId"]) for row in rows})
    if len(candidate_ids) != feasibility.FROZEN_CANDIDATE_COUNT:
        raise feasibility.FeasibilityContractError(
            "S10F requires the 14 byte-frozen candidate identities"
        )
    return candidate_ids


def _registered_features(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(name) for row in rows for name in row["availability"]})


def _observed_features(row: Mapping[str, Any]) -> list[str]:
    return sorted(
        str(feature_id)
        for feature_id, state in row["availability"].items()
        if state["state"] == "observed_or_exact_native_derived"
    )


def _row_structure(row: Mapping[str, Any]) -> dict[str, Any]:
    horizon = row["confounds"]["nativeHorizon"]
    return {
        "candidateId": str(row["candidateId"]),
        "scenarioFamilyId": str(row["scenarioFamilyId"]),
        "observedFeatureIds": _observed_features(row),
        "orderedSeriesLength": len(row["orderedEventSeries"]["proposalCount"]),
        "observedEventLength": float(row["confounds"]["observedEventLength"]),
        "nativeBudget": float(horizon.get("budget", 32)),
        "structuralScenarioSize": float(
            row["confounds"]["structuralScenarioSize"]
        ),
    }


def _profile_planes(
    candidates: Sequence[str],
    profiles: np.ndarray | None,
) -> tuple[dict[str, str], dict[str, bool]]:
    if profiles is None or profiles.shape[0] != len(candidates):
        return (
            {
                candidate: canonical_sha256(
                    "E07/S10F/unavailable-profile/v1", candidate
                )
                for candidate in candidates
            },
            {candidate: False for candidate in candidates},
        )
    commitments, finite = {}, {}
    for index, candidate in enumerate(candidates):
        profile = np.asarray(profiles[index], dtype=float)
        is_finite = bool(np.all(np.isfinite(profile)))
        commitments[candidate] = canonical_sha256(
            "E07/S10/configuration-profiles/v1",
            profile.tolist() if is_finite else {"candidateId": candidate},
        )
        finite[candidate] = is_finite
    return commitments, finite


def stratum_structure(
    rows: Sequence[Mapping[str, Any]],
    *,
    profiles: np.ndarray | None = None,
    provisional_distinct_profiles: bool = False,
) -> dict[str, Any]:
    """Project rows into S10E's value-free structural interface."""

    if not rows:
        raise feasibility.FeasibilityContractError("empty stratum")
    task_ids = {str(row["taskId"]) for row in rows}
    statuses = {str(row["statusStratum"]) for row in rows}
    if len(task_ids) != 1 or len(statuses) != 1:
        raise feasibility.FeasibilityContractError(
            "structure must represent one task/status stratum"
        )
    candidates = sorted({str(row["candidateId"]) for row in rows})
    families = sorted({str(row["scenarioFamilyId"]) for row in rows})
    if provisional_distinct_profiles:
        commitments = {
            candidate: canonical_sha256(
                "E07/S10F/provisional-structural-profile/v1", candidate
            )
            for candidate in candidates
        }
        finite = {candidate: True for candidate in candidates}
    else:
        commitments, finite = _profile_planes(candidates, profiles)
    return {
        "schemaVersion": "e07.s10e.stratum-structure.v1",
        "taskId": next(iter(task_ids)),
        "statusStratum": next(iter(statuses)),
        "registeredFeatureIds": _registered_features(rows),
        "candidateIds": candidates,
        "scenarioFamilyIds": families,
        "rows": [_row_structure(row) for row in rows],
        "profileCommitmentsByCandidate": commitments,
        "finiteProfileByCandidate": finite,
    }


def _assessment_with_confound(
    assessment: Mapping[str, Any],
    confound: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = deepcopy(dict(assessment))
    if confound is None:
        return result
    result["observedConfoundAudit"] = dict(confound)
    if confound["pass"]:
        return result
    reason = "FROZEN_POSTFIT_CONFOUND_GATE_FAILED"
    result["confoundAudit"] = {
        **result["confoundAudit"],
        "admissible": False,
        "classification": "non_evidentiary_not_candidate_not_null",
        "reasonCodes": sorted(
            set([*result["confoundAudit"]["reasonCodes"], reason])
        ),
    }
    for family in ("clustering", "anomaly"):
        result[family] = {
            **result[family],
            "admissible": False,
            "classification": "non_evidentiary_not_candidate_not_null",
            "reasonCodes": sorted(set([*result[family]["reasonCodes"], reason])),
        }
    result.pop("assessmentSha256", None)
    result["assessmentSha256"] = feasibility.canonical_sha256(
        "E07/S10F/stratum-feasibility-assessment/v1", result
    )
    return result


def _not_run_result(
    assessment: Mapping[str, Any], family: str
) -> dict[str, Any]:
    decision = assessment[family]
    return {
        "schemaVersion": f"e07.s10f.{family}-non-evidentiary.v1",
        "taskId": assessment["taskId"],
        "statusStratum": assessment["statusStratum"],
        "analysisState": decision["classification"],
        "methodExecuted": False,
        "reasonCodes": decision["reasonCodes"],
    }


def _cp_not_run_result(
    assessment: Mapping[str, Any], candidate_id: str
) -> dict[str, Any]:
    decision = assessment["changePoint"]["candidateDecisions"][candidate_id]
    return {
        "schemaVersion": "e07.s10f.change-point-non-evidentiary.v1",
        "taskId": assessment["taskId"],
        "statusStratum": assessment["statusStratum"],
        "configurationId": candidate_id,
        "analysisState": decision["classification"],
        "methodExecuted": False,
        "reasonCodes": decision["reasonCodes"],
        "rawPValue": 1.0,
        "preMultiplicityGatePass": False,
    }


def _slot_for_candidate(
    candidate: Mapping[str, Any],
    cluster_slots_by_stratum: Mapping[str, Mapping[str, str]],
) -> str:
    prefix = f"{candidate['taskId']}::{candidate['statusStratum']}"
    if candidate["methodFamily"] == "clustering":
        return cluster_slots_by_stratum[prefix][candidate["machineCandidateKey"]]
    if candidate["methodFamily"] == "anomaly":
        return f"{prefix}::anomaly::0"
    return f"{prefix}::change_point::{candidate['configurationId']}"


def discovery_analysis(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run only S10E-admissible frozen discovery methods."""

    global _LAST_DISCOVERY_AUDIT
    frozen_candidates = _frozen_candidate_ids(rows)
    all_results: dict[str, Any] = {
        "preprocessing": {},
        "confounds": {},
        "clustering": [],
        "anomaly": [],
        "changePoint": [],
        "rareStatusStrata": [],
    }
    machine_candidates: list[dict[str, Any]] = []
    assessments: list[dict[str, Any]] = []
    runtime_locks: dict[str, Any] = {}
    cluster_slots: dict[str, dict[str, str]] = {}

    for task_id in search.SPATIAL_TASKS:
        task_rows = [row for row in rows if row["taskId"] == task_id]
        statuses = sorted({str(row["statusStratum"]) for row in task_rows})
        for status in statuses:
            stratum = [
                row for row in task_rows if row["statusStratum"] == status
            ]
            key = f"{task_id}::{status}"
            provisional = feasibility.assess_stratum(
                stratum_structure(
                    stratum, provisional_distinct_profiles=True
                ),
                frozen_candidate_ids=frozen_candidates,
            )
            preprocessing = None
            matrix = None
            candidate_ids = sorted({str(row["candidateId"]) for row in stratum})
            profiles = None
            confound = None
            if provisional["preprocessing"]["admissible"]:
                preprocessing = search.fit_preprocessing(stratum)
                matrix = np.asarray(
                    preprocessing.pop("discoveryStandardized"), dtype=float
                )
                candidate_ids, profiles = search.configuration_profiles(
                    stratum, matrix
                )
                confound = search.confound_audit(stratum, matrix)
            final = feasibility.assess_stratum(
                stratum_structure(stratum, profiles=profiles),
                frozen_candidate_ids=frozen_candidates,
            )
            final = _assessment_with_confound(final, confound)
            assessments.append(final)

            if final["rareStatusDescriptiveOnly"]:
                all_results["rareStatusStrata"].append(
                    {
                        "taskId": task_id,
                        "statusStratum": status,
                        "rowCount": len(stratum),
                        "analysis": "descriptive_only_non_evidentiary",
                    }
                )
            if preprocessing is not None:
                all_results["preprocessing"][key] = preprocessing
            else:
                all_results["preprocessing"][key] = {
                    "analysisState": final["preprocessing"]["classification"],
                    "reasonCodes": final["preprocessing"]["reasonCodes"],
                }
            all_results["confounds"][key] = (
                {
                    **dict(confound),
                    "frozenObservedAuditPass": bool(confound["pass"]),
                    "pass": True,
                    "analysisState": (
                        "evidentiary_audit"
                        if confound["pass"]
                        else "non_evidentiary_not_candidate_not_null"
                    ),
                    "failedAuditMethodsExcluded": not bool(confound["pass"]),
                }
                if confound is not None
                else {
                    "pass": True,
                    "analysisState": "non_evidentiary_not_candidate_not_null",
                    "reasonCodes": final["confoundAudit"]["reasonCodes"],
                }
            )

            cluster_result = _not_run_result(final, "clustering")
            cluster_candidates: list[dict[str, Any]] = []
            anomaly_result = _not_run_result(final, "anomaly")
            anomaly_candidate: dict[str, Any] | None = None
            anomaly_model: IsolationForest | None = None
            if final["clustering"]["admissible"]:
                if matrix is None:
                    raise AssertionError("admissible clustering lacks matrix")
                cluster_result, cluster_candidates = search.clustering_discovery(
                    stratum, matrix, task_id=task_id, status=status
                )
                cluster_result["methodExecuted"] = True
                cluster_result["analysisState"] = "evidentiary_test"
            if final["anomaly"]["admissible"]:
                if matrix is None:
                    raise AssertionError("admissible anomaly lacks matrix")
                anomaly_result, anomaly_candidate = search.anomaly_discovery(
                    stratum, matrix, task_id=task_id, status=status
                )
                anomaly_model = anomaly_result.pop("model")
                anomaly_result["methodExecuted"] = True
                anomaly_result["analysisState"] = "evidentiary_test"
            all_results["clustering"].append(cluster_result)
            all_results["anomaly"].append(anomaly_result)

            cp_eligible = [
                candidate
                for candidate in frozen_candidates
                if final["changePoint"]["candidateDecisions"][candidate][
                    "admissible"
                ]
            ]
            cp_rows = [
                row for row in stratum if row["candidateId"] in cp_eligible
            ]
            cp_results: list[dict[str, Any]] = []
            cp_candidates: list[dict[str, Any]] = []
            if cp_rows:
                cp_results, cp_candidates = search.change_point_discovery(
                    cp_rows, task_id=task_id, status=status
                )
                for item in cp_results:
                    item["methodExecuted"] = True
                    item["analysisState"] = "evidentiary_test"
            completed_cp = {
                str(item["configurationId"]) for item in cp_results
            }
            cp_results.extend(
                _cp_not_run_result(final, candidate)
                for candidate in frozen_candidates
                if candidate not in completed_cp
            )
            all_results["changePoint"].extend(cp_results)

            ordered_cluster_candidates = sorted(
                cluster_candidates,
                key=lambda item: str(item["machineCandidateKey"]),
            )
            cluster_slots[key] = {
                item["machineCandidateKey"]: f"{key}::clustering::{index}"
                for index, item in enumerate(ordered_cluster_candidates)
            }
            machine_candidates.extend(ordered_cluster_candidates)
            if anomaly_candidate is not None:
                machine_candidates.append(anomaly_candidate)
            machine_candidates.extend(cp_candidates)
            runtime_locks[key] = {
                "preprocessing": preprocessing,
                "candidateIds": candidate_ids,
                "profiles": (
                    profiles.tolist() if profiles is not None else None
                ),
                "cluster": cluster_result,
                "anomalyModel": anomaly_model,
                "anomaly": anomaly_result,
                "feasibilityAssessment": final,
            }

    p_values: dict[str, float] = {}
    candidate_slots: dict[str, str] = {}
    for candidate in machine_candidates:
        slot_id = _slot_for_candidate(candidate, cluster_slots)
        if slot_id in p_values:
            raise RuntimeError("multiple machine candidates claimed one Holm slot")
        p_values[slot_id] = float(candidate["rawPValue"])
        candidate_slots[candidate["machineCandidateKey"]] = slot_id
    slots = feasibility.global_multiplicity_slots(
        assessments,
        frozen_candidate_ids=frozen_candidates,
        evidentiary_p_values=p_values,
    )
    adjusted_slots = feasibility.holm_adjust_slots(slots)
    adjusted_by_id = {row["slotId"]: row for row in adjusted_slots}
    adjusted_candidates = []
    for candidate in machine_candidates:
        slot = adjusted_by_id[candidate_slots[candidate["machineCandidateKey"]]]
        adjusted_candidates.append(
            {
                **candidate,
                "holmSlotId": slot["slotId"],
                "holmAdjustedPValue": slot["holmAdjustedPValue"],
                "discoveryMultiplicityPass": slot["multiplicityPass"],
            }
        )
    passing = [
        row for row in adjusted_candidates if row["discoveryMultiplicityPass"]
    ]
    lock_document = {
        "schemaVersion": "e07.s10f.discovery-lock.v1",
        "researchStepId": "S10F",
        "candidateDefinitions": adjusted_candidates,
        "discoveryPassingCandidateKeys": sorted(
            row["machineCandidateKey"] for row in passing
        ),
        "featureSupportMasksAndPreprocessing": {
            key: value["preprocessing"]
            for key, value in runtime_locks.items()
        },
        "configurationProfiles": {
            key: {
                "candidateIds": value["candidateIds"],
                "profiles": value["profiles"],
            }
            for key, value in runtime_locks.items()
        },
        "clusterRepresentatives": {
            key: value["cluster"] for key, value in runtime_locks.items()
        },
        "anomalyThresholds": {
            key: value["anomaly"] for key, value in runtime_locks.items()
        },
        "changePointWindows": [
            {
                "machineCandidateKey": row["machineCandidateKey"],
                "center": row["lockedCenterTransition"],
            }
            for row in adjusted_candidates
            if row["methodFamily"] == "change_point"
        ],
        "s10eFeasibilityAssessments": assessments,
        "fixedHolmSlots": adjusted_slots,
        "multiplicityFamily": {
            "method": "Holm",
            "alpha": 0.05,
            "size": len(adjusted_slots),
            "candidateKeys": sorted(
                row["machineCandidateKey"] for row in adjusted_candidates
            ),
            "fixedSlotIds": sorted(row["slotId"] for row in adjusted_slots),
            "infeasibleSlotRawPValue": 1.0,
            "unusedOrNoncandidateSlotRawPValue": 1.0,
            "familyShrunk": False,
        },
        "reproductionRefitPermitted": False,
    }
    lock_document["discoveryLockSha256"] = canonical_sha256(
        "E07/S10F/discovery-lock/v1", lock_document
    )
    lock_document["_runtimeLocks"] = runtime_locks
    catalog = {
        "schemaVersion": "e07.s10f.discovery-analysis.v1",
        "machineMultiplicityFamilySize": len(adjusted_slots),
        "machineCandidates": adjusted_candidates,
        "discoveryPassingCandidates": passing,
        "results": all_results,
    }
    _LAST_DISCOVERY_AUDIT = {
        "schemaVersion": "e07.s10f.discovery-feasibility-audit.v1",
        "assessmentCount": len(assessments),
        "fixedHolmSlotCount": len(adjusted_slots),
        "admissibleSlotCount": sum(
            row["structurallyAdmissible"] for row in adjusted_slots
        ),
        "evidentiaryCandidateSlotCount": sum(
            row["slotState"] == "evidentiary_candidate"
            for row in adjusted_slots
        ),
        "infeasibleSlotCount": sum(
            row["slotState"] == "non_evidentiary_infeasible"
            for row in adjusted_slots
        ),
        "unusedOrNoncandidateSlotCount": sum(
            row["slotState"] == "feasible_but_unused_or_noncandidate"
            for row in adjusted_slots
        ),
        "fixedFamilyShrunk": False,
        "assessments": assessments,
        "holmSlots": adjusted_slots,
    }
    return catalog, lock_document


def reproduction_analysis(
    rows: Sequence[Mapping[str, Any]],
    discovery_catalog: Mapping[str, Any],
    discovery_lock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Run locked reproduction only where S10E remains admissible."""

    global _LAST_REPRODUCTION_AUDIT
    frozen_candidates = _frozen_candidate_ids(rows)
    runtime_locks = discovery_lock["_runtimeLocks"]
    passing = {
        row["machineCandidateKey"]: row
        for row in discovery_catalog["discoveryPassingCandidates"]
    }
    results: list[dict[str, Any]] = []
    reproduction_assessments: dict[str, Any] = {}
    anomaly_slot_p_values: dict[str, float] = {}

    for key, candidate in sorted(passing.items()):
        lock_key = f"{candidate['taskId']}::{candidate['statusStratum']}"
        task_rows = [
            row
            for row in rows
            if row["taskId"] == candidate["taskId"]
            and row["statusStratum"] == candidate["statusStratum"]
        ]
        if not task_rows:
            results.append(
                {
                    **candidate,
                    "reproductionPass": False,
                    "analysisState": (
                        "non_evidentiary_not_candidate_not_null"
                    ),
                    "failureReason": "locked_status_stratum_absent",
                }
            )
            continue
        lock = runtime_locks[lock_key]
        matrix = None
        profiles = None
        candidate_ids = sorted({str(row["candidateId"]) for row in task_rows})
        if lock["preprocessing"] is not None:
            matrix = search.apply_preprocessing(
                task_rows, lock["preprocessing"]
            )
            try:
                candidate_ids, profiles = search.configuration_profiles(
                    task_rows, matrix
                )
            except RuntimeError:
                profiles = None
        metadata = stratum_structure(task_rows, profiles=profiles)
        reproduction_structure = feasibility.assess_reproduction_structure(
            lock["feasibilityAssessment"],
            metadata,
            frozen_candidate_ids=frozen_candidates,
        )
        reproduction_assessments[lock_key] = reproduction_structure
        if candidate["methodFamily"] == "clustering":
            admissible = reproduction_structure["methodAdmissibility"][
                "clustering"
            ]
        elif candidate["methodFamily"] == "anomaly":
            admissible = reproduction_structure["methodAdmissibility"]["anomaly"]
        else:
            admissible = reproduction_structure["methodAdmissibility"][
                "changePoint"
            ][str(candidate["configurationId"])]
        if not admissible:
            results.append(
                {
                    **candidate,
                    "reproductionPass": False,
                    "analysisState": (
                        "non_evidentiary_not_candidate_not_null"
                    ),
                    "failureReason": "S10E_REPRODUCTION_METHOD_INADMISSIBLE",
                }
            )
            continue
        if candidate_ids != lock["candidateIds"] and candidate[
            "methodFamily"
        ] in {"clustering", "anomaly"}:
            results.append(
                {
                    **candidate,
                    "reproductionPass": False,
                    "analysisState": (
                        "non_evidentiary_not_candidate_not_null"
                    ),
                    "failureReason": "configuration_population_changed",
                }
            )
            continue
        if candidate["methodFamily"] == "clustering":
            if profiles is None:
                raise AssertionError("admissible cluster reproduction lacks profiles")
            labels = lock["cluster"]["selectedWardLabels"]
            assigned = search._locked_cluster_assignment(
                np.asarray(lock["profiles"], dtype=float), labels, profiles
            )
            member_set = set(candidate["memberCandidateIds"])
            discovery_label = next(
                labels[index]
                for index, item in enumerate(candidate_ids)
                if item in member_set
            )
            reproduced_set = {
                candidate_ids[index]
                for index, label in enumerate(assigned)
                if int(label) == int(discovery_label)
            }
            jaccard = len(member_set & reproduced_set) / max(
                1, len(member_set | reproduced_set)
            )
            results.append(
                {
                    **candidate,
                    "analysisState": "evidentiary_reproduction",
                    "reproductionMetric": "locked_membership_Jaccard",
                    "reproductionValue": float(jaccard),
                    "reproductionThreshold": 0.70,
                    "reproductionPass": bool(jaccard >= 0.70),
                    "reproducedMemberCandidateIds": sorted(reproduced_set),
                }
            )
        elif candidate["methodFamily"] == "anomaly":
            if profiles is None:
                raise AssertionError("admissible anomaly reproduction lacks profiles")
            model: IsolationForest = lock["anomalyModel"]
            scores = -model.decision_function(profiles)
            index = candidate_ids.index(str(candidate["configurationId"]))
            score = float(scores[index])
            same_direction = score > float(np.median(scores))
            grouped: dict[str, list[int]] = defaultdict(list)
            for row_index, row in enumerate(task_rows):
                grouped[str(row["scenarioFamilyId"])].append(row_index)
            original = np.asarray(
                [str(row["candidateId"]) for row in task_rows], dtype=object
            )
            null_max = []
            for replicate in range(search.NULL_REPLICATES):
                rng = np.random.default_rng(
                    search.stable_seed(
                        "S10", "reproduction-anomaly-null", key, replicate
                    )
                )
                permuted = original.copy()
                for indices in grouped.values():
                    values = permuted[indices].copy()
                    rng.shuffle(values)
                    permuted[indices] = values
                null_rows = [
                    {**row, "candidateId": str(permuted[row_index])}
                    for row_index, row in enumerate(task_rows)
                ]
                _, null_profiles = search.configuration_profiles(
                    null_rows, matrix
                )
                null_max.append(
                    float(np.max(-model.decision_function(null_profiles)))
                )
            p_value = (
                1 + sum(value >= score for value in null_max)
            ) / (search.NULL_REPLICATES + 1)
            anomaly_slot_p_values[candidate["holmSlotId"]] = float(p_value)
            results.append(
                {
                    **candidate,
                    "analysisState": "evidentiary_reproduction",
                    "reproductionMetric": "locked_isolation_score",
                    "reproductionValue": score,
                    "sameDirection": bool(same_direction),
                    "reproductionRawPValue": float(p_value),
                    "reproductionPassPreHolm": bool(same_direction),
                }
            )
        else:
            candidate_rows = [
                row
                for row in task_rows
                if row["candidateId"] == candidate["configurationId"]
            ]
            points = [search.row_change_points(row) for row in candidate_rows]
            center = int(candidate["lockedCenterTransition"])
            prevalence = sum(
                any(abs(point - center) <= 1 for point in row_points)
                for row_points in points
            ) / len(points)
            results.append(
                {
                    **candidate,
                    "analysisState": "evidentiary_reproduction",
                    "reproductionMetric": "locked_window_prevalence",
                    "reproductionValue": float(prevalence),
                    "reproductionThreshold": 0.50,
                    "reproductionPass": bool(prevalence >= 0.50),
                }
            )

    reproduction_slots = [
        {
            **dict(slot),
            "rawPValue": (
                anomaly_slot_p_values.get(str(slot["slotId"]), 1.0)
            ),
            "slotState": (
                "evidentiary_candidate"
                if str(slot["slotId"]) in anomaly_slot_p_values
                else (
                    "non_evidentiary_infeasible"
                    if not slot["structurallyAdmissible"]
                    else "feasible_but_unused_or_noncandidate"
                )
            ),
        }
        for slot in discovery_lock["fixedHolmSlots"]
    ]
    adjusted = feasibility.holm_adjust_slots(reproduction_slots)
    by_slot = {row["slotId"]: row for row in adjusted}
    for result in results:
        if (
            result["methodFamily"] == "anomaly"
            and "reproductionRawPValue" in result
        ):
            slot = by_slot[result["holmSlotId"]]
            result["reproductionHolmAdjustedPValue"] = slot[
                "holmAdjustedPValue"
            ]
            result["reproductionPass"] = bool(
                result["reproductionPassPreHolm"]
                and slot["multiplicityPass"]
            )
    _LAST_REPRODUCTION_AUDIT = {
        "schemaVersion": "e07.s10f.reproduction-feasibility-audit.v1",
        "assessments": reproduction_assessments,
        "fixedHolmSlots": adjusted,
        "fixedFamilyShrunk": False,
        "infeasibleResultsClassifiedNonEvidentiary": all(
            row.get("analysisState")
            != "non_evidentiary_not_candidate_not_null"
            or not row.get("reproductionPass")
            for row in results
        ),
    }
    return results


def discovery_audit() -> dict[str, Any]:
    if _LAST_DISCOVERY_AUDIT is None:
        raise RuntimeError("discovery feasibility audit is not available")
    return deepcopy(_LAST_DISCOVERY_AUDIT)


def reproduction_audit() -> dict[str, Any]:
    if _LAST_REPRODUCTION_AUDIT is None:
        raise RuntimeError("reproduction feasibility audit is not available")
    return deepcopy(_LAST_REPRODUCTION_AUDIT)
