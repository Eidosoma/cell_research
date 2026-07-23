"""Outcome-independent method-feasibility rules for the frozen S10P design.

The module consumes structural summaries only.  It never loads episode rows,
feature values, protected outcomes, failed caches, models, or embeddings.
Future execution code may construct the summary after native rows exist, but
must do so without exposing values to this interface: observed-feature
identities, exact profile commitments, incidence, lengths, and finite flags are
enough to decide whether a frozen test is mathematically executable.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


WARD_K_GRID = (2, 3, 4, 5, 6)
GMM_K_GRID = (1, 2, 3, 4, 5, 6)
CROSS_FIT_FOLDS = 5
MINIMUM_ROW_COUNT = 8
MINIMUM_FEATURE_SUPPORT = 0.90
MINIMUM_ELIGIBLE_FEATURES = 5
MINIMUM_RESAMPLING_FAMILIES = 2
MINIMUM_CHANGE_POINT_FAMILIES = 2
EXACT_SERIES_LENGTH = 32
MINIMUM_SEGMENT_LENGTH = 4
MAXIMUM_CHANGE_POINTS = 3
FROZEN_CANDIDATE_COUNT = 14
CLUSTERING_SLOTS = 6
ANOMALY_SLOTS = 1
CHANGE_POINT_SLOTS = FROZEN_CANDIDATE_COUNT
SLOTS_PER_STRATUM = CLUSTERING_SLOTS + ANOMALY_SLOTS + CHANGE_POINT_SLOTS
SPATIAL_TASKS = frozenset({"e07_s02_spatial2d_local", "e07_s02_spatial2d_memory"})


class FeasibilityContractError(ValueError):
    """Raised when structural metadata or the frozen rule is malformed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(domain: str, value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(value))
    return digest.hexdigest()


def fold_for_family(scenario_family_id: str) -> int:
    return (
        int(hashlib.sha256(scenario_family_id.encode("utf-8")).hexdigest(), 16)
        % CROSS_FIT_FOLDS
    )


def expected_rule_document() -> dict[str, Any]:
    """Return the exact frozen mathematical rule represented by this module."""

    return {
        "rareStatusRowThreshold": MINIMUM_ROW_COUNT,
        "minimumFeatureSupportInclusive": MINIMUM_FEATURE_SUPPORT,
        "minimumEligibleFeatures": MINIMUM_ELIGIBLE_FEATURES,
        "crossFitFolds": CROSS_FIT_FOLDS,
        "crossFitMinimumObservedTrainRowsPerFeatureFold": 1,
        "crossFitMinimumObservedTestRowsPerFeatureFold": 1,
        "configurationProfile": {
            "requireFiniteCoordinateForEveryEligibleFeature": True,
            "identity": "exact_canonical_profile_commitment_no_tolerance_tuning",
        },
        "resampling": {
            "requireCompleteCandidateByScenarioFamilyIncidence": True,
            "minimumScenarioFamilies": MINIMUM_RESAMPLING_FAMILIES,
            "bootstrapReplicates": 200,
            "nullReplicates": 500,
        },
        "ward": {
            "kGrid": list(WARD_K_GRID),
            "minimumCandidateProfiles": max(WARD_K_GRID),
            "minimumDistinctProfiles": max(WARD_K_GRID),
        },
        "diagonalGmm": {
            "kGrid": list(GMM_K_GRID),
            "minimumCandidateProfiles": max(GMM_K_GRID),
            "minimumDistinctProfiles": max(GMM_K_GRID),
            "covarianceRegularization": 0.000001,
            "initializations": 20,
        },
        "isolationForest": {
            "estimators": 500,
            "minimumCandidateProfiles": 2,
            "minimumDistinctProfiles": 2,
            "maxSamples": "min_256_and_n",
        },
        "changePoint": {
            "authenticTasks": sorted(SPATIAL_TASKS),
            "exactSeriesLength": EXACT_SERIES_LENGTH,
            "minimumSegmentLength": MINIMUM_SEGMENT_LENGTH,
            "maximumChangePoints": MAXIMUM_CHANGE_POINTS,
            "minimumScenarioFamiliesPerCandidate": MINIMUM_CHANGE_POINT_FAMILIES,
        },
        "reproduction": {
            "requireLockedCandidatePopulationIdentity": True,
            "requireLockedPreprocessingIdentity": True,
            "requireSameMethodAdmissibility": True,
        },
    }


def validate_frozen_rule(rule: Mapping[str, Any]) -> None:
    expected = expected_rule_document()
    if dict(rule) != expected:
        raise FeasibilityContractError(
            "method-feasibility rule differs from the frozen implementation"
        )


def _validate_sha256(value: Any, field: str) -> str:
    text = str(value)
    if len(text) != 64:
        raise FeasibilityContractError(f"{field} is not a SHA-256 commitment")
    try:
        bytes.fromhex(text)
    except ValueError as exc:
        raise FeasibilityContractError(f"{field} is not a SHA-256 commitment") from exc
    return text


def _reasoned(pass_value: bool, *reasons: str) -> dict[str, Any]:
    return {
        "admissible": bool(pass_value),
        "classification": (
            "evidentiary_test_admissible"
            if pass_value
            else "non_evidentiary_not_candidate_not_null"
        ),
        "reasonCodes": [] if pass_value else sorted(set(reasons)),
    }


def _parse_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schemaVersion",
        "taskId",
        "statusStratum",
        "registeredFeatureIds",
        "candidateIds",
        "scenarioFamilyIds",
        "rows",
        "profileCommitmentsByCandidate",
        "finiteProfileByCandidate",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise FeasibilityContractError(f"missing structural fields: {missing}")
    if metadata["schemaVersion"] != "e07.s10e.stratum-structure.v1":
        raise FeasibilityContractError("unknown stratum-structure schema")
    task_id = str(metadata["taskId"])
    status = str(metadata["statusStratum"])
    candidates = tuple(map(str, metadata["candidateIds"]))
    families = tuple(map(str, metadata["scenarioFamilyIds"]))
    features = tuple(map(str, metadata["registeredFeatureIds"]))
    if not task_id or not status:
        raise FeasibilityContractError("task and status identifiers are required")
    if len(candidates) != len(set(candidates)):
        raise FeasibilityContractError("candidate identifiers are not unique")
    if len(families) != len(set(families)):
        raise FeasibilityContractError("scenario-family identifiers are not unique")
    if len(features) != len(set(features)):
        raise FeasibilityContractError("registered feature identifiers are not unique")
    if not candidates or not families or not features:
        raise FeasibilityContractError(
            "candidate, scenario-family, and feature registries must be nonempty"
        )

    candidate_set, family_set, feature_set = (
        set(candidates),
        set(families),
        set(features),
    )
    parsed_rows: list[dict[str, Any]] = []
    pairs: set[tuple[str, str]] = set()
    for index, raw in enumerate(metadata["rows"]):
        if not isinstance(raw, Mapping):
            raise FeasibilityContractError(f"row {index} is not a mapping")
        required_row = {
            "candidateId",
            "scenarioFamilyId",
            "observedFeatureIds",
            "orderedSeriesLength",
            "observedEventLength",
            "nativeBudget",
            "structuralScenarioSize",
        }
        missing_row = required_row - set(raw)
        if missing_row:
            raise FeasibilityContractError(
                f"row {index} is missing {sorted(missing_row)}"
            )
        candidate = str(raw["candidateId"])
        family = str(raw["scenarioFamilyId"])
        if candidate not in candidate_set or family not in family_set:
            raise FeasibilityContractError(
                "row references an undeclared candidate or scenario family"
            )
        pair = (candidate, family)
        if pair in pairs:
            raise FeasibilityContractError("duplicate candidate-family row")
        pairs.add(pair)
        observed = tuple(map(str, raw["observedFeatureIds"]))
        if len(observed) != len(set(observed)):
            raise FeasibilityContractError("row repeats an observed feature")
        if not set(observed).issubset(feature_set):
            raise FeasibilityContractError("row observes an unregistered feature")
        event_length = float(raw["observedEventLength"])
        native_budget = float(raw["nativeBudget"])
        scenario_size = float(raw["structuralScenarioSize"])
        if not (
            math.isfinite(event_length)
            and math.isfinite(native_budget)
            and math.isfinite(scenario_size)
            and event_length >= 0
            and native_budget > 0
            and scenario_size > 0
        ):
            raise FeasibilityContractError("row contains invalid structural covariates")
        series_length = raw["orderedSeriesLength"]
        if series_length is not None:
            if isinstance(series_length, bool) or int(series_length) != series_length:
                raise FeasibilityContractError("ordered series length must be integral")
            series_length = int(series_length)
            if series_length < 0:
                raise FeasibilityContractError(
                    "ordered series length cannot be negative"
                )
        parsed_rows.append(
            {
                "candidateId": candidate,
                "scenarioFamilyId": family,
                "observedFeatureIds": observed,
                "orderedSeriesLength": series_length,
                "observedEventLength": event_length,
                "nativeBudget": native_budget,
                "structuralScenarioSize": scenario_size,
            }
        )

    declared_pairs = {
        (row["candidateId"], row["scenarioFamilyId"]) for row in parsed_rows
    }
    if declared_pairs != pairs:
        raise AssertionError("unreachable pair-parser mismatch")
    if {row["candidateId"] for row in parsed_rows} != candidate_set:
        raise FeasibilityContractError("a declared candidate has no row")
    if {row["scenarioFamilyId"] for row in parsed_rows} != family_set:
        raise FeasibilityContractError("a declared scenario family has no row")

    commitments_raw = metadata["profileCommitmentsByCandidate"]
    finite_raw = metadata["finiteProfileByCandidate"]
    if not isinstance(commitments_raw, Mapping) or not isinstance(finite_raw, Mapping):
        raise FeasibilityContractError("profile planes must be mappings")
    unknown_profile_keys = (
        set(map(str, commitments_raw)) | set(map(str, finite_raw))
    ) - candidate_set
    if unknown_profile_keys:
        raise FeasibilityContractError("profile plane references unknown candidates")
    commitments = {
        str(candidate): _validate_sha256(value, f"profile:{candidate}")
        for candidate, value in commitments_raw.items()
    }
    finite = {str(candidate): bool(value) for candidate, value in finite_raw.items()}
    return {
        "taskId": task_id,
        "statusStratum": status,
        "registeredFeatureIds": features,
        "candidateIds": candidates,
        "scenarioFamilyIds": families,
        "rows": parsed_rows,
        "pairs": pairs,
        "profileCommitmentsByCandidate": commitments,
        "finiteProfileByCandidate": finite,
    }


def assess_stratum(
    metadata: Mapping[str, Any],
    *,
    frozen_candidate_ids: Sequence[str],
    rule: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess every frozen method without inspecting scientific feature values."""

    if rule is not None:
        validate_frozen_rule(rule)
    frozen = tuple(map(str, frozen_candidate_ids))
    if len(frozen) != FROZEN_CANDIDATE_COUNT or len(set(frozen)) != len(frozen):
        raise FeasibilityContractError(
            "Holm bookkeeping requires 14 unique frozen candidate IDs"
        )
    parsed = _parse_metadata(metadata)
    candidates = parsed["candidateIds"]
    families = parsed["scenarioFamilyIds"]
    rows = parsed["rows"]
    row_count = len(rows)
    rare = row_count < MINIMUM_ROW_COUNT
    expected_grid = {
        (candidate, family) for candidate in candidates for family in families
    }
    complete_grid = parsed["pairs"] == expected_grid

    feature_decisions: dict[str, Any] = {}
    eligible: list[str] = []
    observed_by_feature: dict[str, list[dict[str, Any]]] = {}
    for feature in parsed["registeredFeatureIds"]:
        observed_rows = [row for row in rows if feature in row["observedFeatureIds"]]
        observed_by_feature[feature] = observed_rows
        support = len(observed_rows) / row_count
        is_eligible = support >= MINIMUM_FEATURE_SUPPORT
        feature_decisions[feature] = {
            "observedCount": len(observed_rows),
            "denominatorCount": row_count,
            "support": support,
            "threshold": MINIMUM_FEATURE_SUPPORT,
            "comparison": "greater_than_or_equal",
            "eligible": is_eligible,
        }
        if is_eligible:
            eligible.append(feature)

    fold_checks: list[dict[str, Any]] = []
    empty_train = False
    empty_test = False
    candidate_feature_gaps: list[str] = []
    for feature in eligible:
        observed_rows = observed_by_feature[feature]
        for fold in range(CROSS_FIT_FOLDS):
            test = sum(
                fold_for_family(row["scenarioFamilyId"]) == fold
                for row in observed_rows
            )
            train = len(observed_rows) - test
            fold_checks.append(
                {
                    "featureId": feature,
                    "fold": fold,
                    "observedTrainRows": train,
                    "observedTestRows": test,
                    "pass": train >= 1 and test >= 1,
                }
            )
            empty_train |= train < 1
            empty_test |= test < 1
        for candidate in candidates:
            if not any(row["candidateId"] == candidate for row in observed_rows):
                candidate_feature_gaps.append(f"{candidate}:{feature}")

    profile_keys_complete = set(parsed["profileCommitmentsByCandidate"]) == set(
        candidates
    ) and set(parsed["finiteProfileByCandidate"]) == set(candidates)
    finite_profiles = profile_keys_complete and all(
        parsed["finiteProfileByCandidate"].values()
    )
    distinct_profiles = len(set(parsed["profileCommitmentsByCandidate"].values()))
    preprocessing_reasons: list[str] = []
    if rare:
        preprocessing_reasons.append("ROW_COUNT_BELOW_FROZEN_EIGHT")
    if len(eligible) < MINIMUM_ELIGIBLE_FEATURES:
        preprocessing_reasons.append("FEWER_THAN_FIVE_FEATURES_AT_90_PERCENT")
    if empty_train:
        preprocessing_reasons.append("EMPTY_CROSSFIT_TRAIN_FOLD")
    if empty_test:
        preprocessing_reasons.append("EMPTY_CROSSFIT_TEST_FOLD")
    if candidate_feature_gaps:
        preprocessing_reasons.append("CANDIDATE_PROFILE_FEATURE_UNOBSERVED")
    if not profile_keys_complete:
        preprocessing_reasons.append("PROFILE_COMMITMENT_PLANE_INCOMPLETE")
    if not finite_profiles:
        preprocessing_reasons.append("NONFINITE_OR_UNCONFIRMED_PROFILE")
    preprocessing = _reasoned(not preprocessing_reasons, *preprocessing_reasons)
    preprocessing.update(
        {
            "eligibleFeatureCount": len(eligible),
            "eligibleFeatureIds": sorted(eligible),
            "featureDecisions": feature_decisions,
            "crossFitChecks": fold_checks,
            "candidateFeatureGaps": sorted(candidate_feature_gaps),
        }
    )

    confound_reasons: list[str] = []
    if not preprocessing["admissible"]:
        confound_reasons.append("PREPROCESSING_INADMISSIBLE")
    if row_count < 2:
        confound_reasons.append("FEWER_THAN_TWO_ROWS_FOR_POSTFIT_AUDIT")
    confound = _reasoned(not confound_reasons, *confound_reasons)
    confound.update(
        {
            "lengthAuditComputable": row_count >= 2,
            "statusPrediction": "not_applicable_within_one_status_stratum",
            "taskPrediction": "not_applicable_no_cross_task_pooling",
        }
    )

    clustering_reasons: list[str] = []
    if not confound["admissible"]:
        clustering_reasons.append("PROFILE_CONFOUND_PIPELINE_INADMISSIBLE")
    if len(candidates) < max(WARD_K_GRID):
        clustering_reasons.append("FEWER_THAN_SIX_CANDIDATE_PROFILES")
    if distinct_profiles < max(WARD_K_GRID):
        clustering_reasons.append("FEWER_THAN_SIX_DISTINCT_PROFILE_COMMITMENTS")
    if len(families) < MINIMUM_RESAMPLING_FAMILIES:
        clustering_reasons.append("FEWER_THAN_TWO_SCENARIO_FAMILIES")
    if not complete_grid:
        clustering_reasons.append("INCOMPLETE_CANDIDATE_FAMILY_INCIDENCE")
    clustering = _reasoned(not clustering_reasons, *clustering_reasons)
    clustering.update(
        {
            "candidateProfileCount": len(candidates),
            "distinctProfileCount": distinct_profiles,
            "scenarioFamilyCount": len(families),
            "completeCandidateFamilyIncidence": complete_grid,
            "ward": [
                {
                    "k": k,
                    "admissible": (
                        clustering["admissible"]
                        and len(candidates) >= k
                        and distinct_profiles >= k
                    ),
                }
                for k in WARD_K_GRID
            ],
            "diagonalGmm": [
                {
                    "k": k,
                    "admissible": (
                        clustering["admissible"]
                        and len(candidates) >= k
                        and distinct_profiles >= k
                    ),
                }
                for k in GMM_K_GRID
            ],
            "bootstrapReplicates": 200,
            "nullReplicates": 500,
        }
    )

    anomaly_reasons: list[str] = []
    if not confound["admissible"]:
        anomaly_reasons.append("PROFILE_CONFOUND_PIPELINE_INADMISSIBLE")
    if len(candidates) < 2:
        anomaly_reasons.append("FEWER_THAN_TWO_CANDIDATE_PROFILES")
    if distinct_profiles < 2:
        anomaly_reasons.append("FEWER_THAN_TWO_DISTINCT_PROFILE_COMMITMENTS")
    if len(families) < MINIMUM_RESAMPLING_FAMILIES:
        anomaly_reasons.append("FEWER_THAN_TWO_SCENARIO_FAMILIES")
    if not complete_grid:
        anomaly_reasons.append("INCOMPLETE_CANDIDATE_FAMILY_INCIDENCE")
    anomaly = _reasoned(not anomaly_reasons, *anomaly_reasons)
    anomaly.update(
        {
            "candidateProfileCount": len(candidates),
            "distinctProfileCount": distinct_profiles,
            "scenarioFamilyCount": len(families),
            "completeCandidateFamilyIncidence": complete_grid,
            "estimators": 500,
            "maxSamples": min(256, len(candidates)),
            "nullReplicates": 500,
        }
    )

    change_rows = {candidate: [] for candidate in frozen}
    for row in rows:
        change_rows[row["candidateId"]].append(row)
    change_candidates: dict[str, Any] = {}
    for candidate in frozen:
        candidate_rows = change_rows[candidate]
        reasons: list[str] = []
        if rare:
            reasons.append("ROW_COUNT_BELOW_FROZEN_EIGHT")
        if parsed["taskId"] not in SPATIAL_TASKS:
            reasons.append("AUTHENTIC_ORDERED_SUMMARY_TASK_REQUIRED")
        if not candidate_rows:
            reasons.append("CANDIDATE_ABSENT_FROM_STATUS_STRATUM")
        unique_candidate_families = {row["scenarioFamilyId"] for row in candidate_rows}
        if (
            candidate_rows
            and len(unique_candidate_families) < MINIMUM_CHANGE_POINT_FAMILIES
        ):
            reasons.append("FEWER_THAN_TWO_SCENARIO_FAMILIES_FOR_PREVALENCE")
        if candidate_rows and any(
            row["orderedSeriesLength"] != EXACT_SERIES_LENGTH for row in candidate_rows
        ):
            reasons.append("ORDERED_SERIES_LENGTH_NOT_EXACTLY_32")
        if EXACT_SERIES_LENGTH < ((MAXIMUM_CHANGE_POINTS + 1) * MINIMUM_SEGMENT_LENGTH):
            reasons.append("SERIES_TOO_SHORT_FOR_MAXIMUM_SEGMENT_PARTITION")
        decision = _reasoned(not reasons, *reasons)
        decision.update(
            {
                "rowCount": len(candidate_rows),
                "scenarioFamilyCount": len(unique_candidate_families),
                "seriesLengths": sorted(
                    {
                        row["orderedSeriesLength"]
                        for row in candidate_rows
                        if row["orderedSeriesLength"] is not None
                    }
                ),
            }
        )
        change_candidates[candidate] = decision
    change_point = {
        "admissibleCandidateCount": sum(
            row["admissible"] for row in change_candidates.values()
        ),
        "candidateDecisions": change_candidates,
        "exactSeriesLength": EXACT_SERIES_LENGTH,
        "minimumSegmentLength": MINIMUM_SEGMENT_LENGTH,
        "maximumChangePoints": MAXIMUM_CHANGE_POINTS,
        "nullReplicates": 500,
    }

    result = {
        "schemaVersion": "e07.s10e.stratum-feasibility-assessment.v1",
        "taskId": parsed["taskId"],
        "statusStratum": parsed["statusStratum"],
        "rowCount": row_count,
        "candidateIds": list(candidates),
        "candidateCount": len(candidates),
        "scenarioFamilyCount": len(families),
        "rareStatusDescriptiveOnly": rare,
        "preprocessing": preprocessing,
        "confoundAudit": confound,
        "clustering": clustering,
        "anomaly": anomaly,
        "changePoint": change_point,
        "outcomeValuesLoaded": False,
        "featureValuesLoaded": False,
        "classificationRule": "infeasible_is_non_evidentiary_not_candidate_not_null",
    }
    result["assessmentSha256"] = canonical_sha256(
        "E07/S10E/stratum-feasibility-assessment/v1", result
    )
    return result


def multiplicity_slots(
    assessment: Mapping[str, Any],
    *,
    frozen_candidate_ids: Sequence[str],
    evidentiary_p_values: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Create 21 fixed slots; infeasible and unused slots receive p=1."""

    frozen = tuple(map(str, frozen_candidate_ids))
    if len(frozen) != FROZEN_CANDIDATE_COUNT or len(set(frozen)) != len(frozen):
        raise FeasibilityContractError(
            "Holm bookkeeping requires 14 unique frozen candidate IDs"
        )
    provided = dict(evidentiary_p_values or {})
    prefix = f"{assessment['taskId']}::{assessment['statusStratum']}"
    definitions: list[tuple[str, bool, str]] = []
    for index in range(CLUSTERING_SLOTS):
        definitions.append(
            (
                f"{prefix}::clustering::{index}",
                bool(assessment["clustering"]["admissible"]),
                "clustering",
            )
        )
    definitions.append(
        (
            f"{prefix}::anomaly::0",
            bool(assessment["anomaly"]["admissible"]),
            "anomaly",
        )
    )
    for candidate in frozen:
        definitions.append(
            (
                f"{prefix}::change_point::{candidate}",
                bool(
                    assessment["changePoint"]["candidateDecisions"][candidate][
                        "admissible"
                    ]
                ),
                "change_point",
            )
        )
    known = {item[0] for item in definitions}
    if not set(provided).issubset(known):
        raise FeasibilityContractError("p-value supplied for an unknown Holm slot")
    rows = []
    for slot_id, admissible, family in definitions:
        if slot_id in provided:
            p_value = float(provided[slot_id])
            if not admissible:
                raise FeasibilityContractError(
                    "p-value supplied for a structurally infeasible slot"
                )
            if not math.isfinite(p_value) or not 0 <= p_value <= 1:
                raise FeasibilityContractError("p-value must be finite in [0,1]")
            state = "evidentiary_candidate"
        else:
            p_value = 1.0
            state = (
                "feasible_but_unused_or_noncandidate"
                if admissible
                else "non_evidentiary_infeasible"
            )
        rows.append(
            {
                "slotId": slot_id,
                "methodFamily": family,
                "structurallyAdmissible": admissible,
                "slotState": state,
                "rawPValue": p_value,
                "inMultiplicityFamily": True,
            }
        )
    if len(rows) != SLOTS_PER_STRATUM or len({row["slotId"] for row in rows}) != len(
        rows
    ):
        raise AssertionError("fixed Holm slot cardinality changed")
    return rows


def global_multiplicity_slots(
    assessments: Sequence[Mapping[str, Any]],
    *,
    frozen_candidate_ids: Sequence[str],
    evidentiary_p_values: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Union all realized task/status slots into S10P's one global Holm family."""

    if not assessments:
        raise FeasibilityContractError(
            "global Holm family requires at least one realized task/status stratum"
        )
    stratum_keys = [
        (str(row["taskId"]), str(row["statusStratum"])) for row in assessments
    ]
    if len(stratum_keys) != len(set(stratum_keys)):
        raise FeasibilityContractError(
            "global Holm family repeats a task/status stratum"
        )
    empty_slots = [
        slot
        for assessment in sorted(
            assessments,
            key=lambda row: (str(row["taskId"]), str(row["statusStratum"])),
        )
        for slot in multiplicity_slots(
            assessment,
            frozen_candidate_ids=frozen_candidate_ids,
        )
    ]
    slot_by_id = {str(row["slotId"]): row for row in empty_slots}
    if len(slot_by_id) != len(empty_slots):
        raise AssertionError("global Holm slot identities are not unique")
    provided = dict(evidentiary_p_values or {})
    if not set(provided).issubset(slot_by_id):
        raise FeasibilityContractError("p-value supplied outside global Holm family")
    for slot_id, value in provided.items():
        slot = slot_by_id[slot_id]
        if not slot["structurallyAdmissible"]:
            raise FeasibilityContractError(
                "p-value supplied for a structurally infeasible global slot"
            )
        p_value = float(value)
        if not math.isfinite(p_value) or not 0 <= p_value <= 1:
            raise FeasibilityContractError("p-value must be finite in [0,1]")
        slot["rawPValue"] = p_value
        slot["slotState"] = "evidentiary_candidate"
    return empty_slots


def holm_adjust_slots(slots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not slots:
        raise FeasibilityContractError("Holm family cannot be empty")
    ordered = sorted(
        enumerate(slots),
        key=lambda item: (float(item[1]["rawPValue"]), str(item[1]["slotId"])),
    )
    adjusted = [1.0] * len(slots)
    running = 0.0
    total = len(slots)
    for rank, (original_index, row) in enumerate(ordered):
        raw = float(row["rawPValue"])
        if not math.isfinite(raw) or not 0 <= raw <= 1:
            raise FeasibilityContractError("Holm input p-values must be in [0,1]")
        value = min(1.0, (total - rank) * raw)
        running = max(running, value)
        adjusted[original_index] = running
    return [
        {
            **dict(row),
            "holmAdjustedPValue": adjusted[index],
            "multiplicityPass": bool(
                row["slotState"] == "evidentiary_candidate" and adjusted[index] <= 0.05
            ),
        }
        for index, row in enumerate(slots)
    ]


def assess_reproduction_structure(
    discovery_assessment: Mapping[str, Any],
    reproduction_metadata: Mapping[str, Any],
    *,
    frozen_candidate_ids: Sequence[str],
    rule: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Qualify locked reproduction structure without executing reproduction."""

    reproduction = assess_stratum(
        reproduction_metadata,
        frozen_candidate_ids=frozen_candidate_ids,
        rule=rule,
    )
    discovery_candidates = (
        tuple(map(str, discovery_assessment["candidateIds"]))
        if ("candidateIds" in discovery_assessment)
        else None
    )
    parsed_reproduction = _parse_metadata(reproduction_metadata)
    reproduction_candidates = tuple(parsed_reproduction["candidateIds"])
    population_identity = (
        discovery_candidates is None or discovery_candidates == reproduction_candidates
    )
    methods = {
        "clustering": bool(
            population_identity
            and discovery_assessment["clustering"]["admissible"]
            and reproduction["clustering"]["admissible"]
        ),
        "anomaly": bool(
            population_identity
            and discovery_assessment["anomaly"]["admissible"]
            and reproduction["anomaly"]["admissible"]
        ),
        "changePoint": {
            candidate: bool(
                discovery_assessment["changePoint"]["candidateDecisions"][candidate][
                    "admissible"
                ]
                and reproduction["changePoint"]["candidateDecisions"][candidate][
                    "admissible"
                ]
            )
            for candidate in map(str, frozen_candidate_ids)
        },
    }
    return {
        "schemaVersion": "e07.s10e.reproduction-feasibility-assessment.v1",
        "lockedCandidatePopulationIdentity": population_identity,
        "methodAdmissibility": methods,
        "frozenReproductionRowsExecuted": 0,
        "classificationRule": "infeasible_is_non_evidentiary_not_reproduction_failure",
    }
