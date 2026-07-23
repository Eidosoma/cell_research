from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from src.phenotype_discovery.method_feasibility import (
    FeasibilityContractError,
    SLOTS_PER_STRATUM,
    assess_reproduction_structure,
    assess_stratum,
    expected_rule_document,
    fold_for_family,
    global_multiplicity_slots,
    holm_adjust_slots,
    multiplicity_slots,
    validate_frozen_rule,
)


FROZEN = tuple(f"candidate-{index:02d}" for index in range(14))
FEATURES = tuple(f"feature-{index:02d}" for index in range(5))


def _families_with_all_folds(count: int = 10) -> tuple[str, ...]:
    found: dict[int, list[str]] = {fold: [] for fold in range(5)}
    ordinal = 0
    while min(map(len, found.values())) < count // 5:
        candidate = f"family-{ordinal:04d}"
        found[fold_for_family(candidate)].append(candidate)
        ordinal += 1
    return tuple(family for fold in range(5) for family in found[fold][: count // 5])


def structure(
    *,
    candidates: tuple[str, ...] = FROZEN,
    families: tuple[str, ...] | None = None,
    features: tuple[str, ...] = FEATURES,
) -> dict:
    families = families or _families_with_all_folds()
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
            candidate: hashlib.sha256(candidate.encode()).hexdigest()
            for candidate in candidates
        },
        "finiteProfileByCandidate": {candidate: True for candidate in candidates},
    }


def test_exact_rule_document_is_accepted() -> None:
    validate_frozen_rule(expected_rule_document())


def test_rule_mutation_fails_closed() -> None:
    mutated = deepcopy(expected_rule_document())
    mutated["ward"]["kGrid"].append(7)
    with pytest.raises(FeasibilityContractError):
        validate_frozen_rule(mutated)


def test_fully_supported_stratum_admits_every_method() -> None:
    result = assess_stratum(
        structure(), frozen_candidate_ids=FROZEN, rule=expected_rule_document()
    )
    assert result["preprocessing"]["admissible"]
    assert result["confoundAudit"]["admissible"]
    assert result["clustering"]["admissible"]
    assert all(row["admissible"] for row in result["clustering"]["ward"])
    assert all(row["admissible"] for row in result["clustering"]["diagonalGmm"])
    assert result["anomaly"]["admissible"]
    assert result["changePoint"]["admissibleCandidateCount"] == 14


def test_rare_status_is_descriptive_and_non_evidentiary() -> None:
    metadata = structure(candidates=FROZEN[:7], families=("only-family",))
    result = assess_stratum(metadata, frozen_candidate_ids=FROZEN)
    assert result["rareStatusDescriptiveOnly"]
    assert not result["preprocessing"]["admissible"]
    assert not result["clustering"]["admissible"]
    assert not result["anomaly"]["admissible"]
    assert result["changePoint"]["admissibleCandidateCount"] == 0
    slots = multiplicity_slots(result, frozen_candidate_ids=FROZEN)
    assert len(slots) == SLOTS_PER_STRATUM
    assert all(row["rawPValue"] == 1.0 for row in slots)
    assert all(row["slotState"] == "non_evidentiary_infeasible" for row in slots)


def test_five_profiles_block_full_clustering_grid_but_not_anomaly() -> None:
    metadata = structure(candidates=FROZEN[:5])
    result = assess_stratum(metadata, frozen_candidate_ids=FROZEN)
    assert not result["clustering"]["admissible"]
    assert "FEWER_THAN_SIX_CANDIDATE_PROFILES" in result["clustering"]["reasonCodes"]
    assert result["anomaly"]["admissible"]
    assert result["changePoint"]["admissibleCandidateCount"] == 5


def test_copied_profile_commitment_blocks_clustering() -> None:
    metadata = structure()
    metadata["profileCommitmentsByCandidate"][FROZEN[-1]] = metadata[
        "profileCommitmentsByCandidate"
    ][FROZEN[0]]
    result = assess_stratum(metadata, frozen_candidate_ids=FROZEN)
    assert result["clustering"]["distinctProfileCount"] == 13
    assert result["clustering"]["admissible"]
    for candidate in FROZEN[5:]:
        metadata["profileCommitmentsByCandidate"][candidate] = metadata[
            "profileCommitmentsByCandidate"
        ][FROZEN[0]]
    result = assess_stratum(metadata, frozen_candidate_ids=FROZEN)
    assert not result["clustering"]["admissible"]
    assert (
        "FEWER_THAN_SIX_DISTINCT_PROFILE_COMMITMENTS"
        in result["clustering"]["reasonCodes"]
    )
    assert result["anomaly"]["admissible"]


def test_incomplete_candidate_family_grid_blocks_resampling_methods() -> None:
    metadata = structure()
    metadata["rows"].pop()
    result = assess_stratum(metadata, frozen_candidate_ids=FROZEN)
    assert not result["clustering"]["admissible"]
    assert not result["anomaly"]["admissible"]
    assert (
        "INCOMPLETE_CANDIDATE_FAMILY_INCIDENCE" in result["clustering"]["reasonCodes"]
    )


def test_four_supported_features_block_profile_methods_only() -> None:
    metadata = structure(features=FEATURES[:4])
    result = assess_stratum(metadata, frozen_candidate_ids=FROZEN)
    assert not result["preprocessing"]["admissible"]
    assert not result["clustering"]["admissible"]
    assert not result["anomaly"]["admissible"]
    assert result["changePoint"]["admissibleCandidateCount"] == 14


@pytest.mark.parametrize(
    ("observed_count", "eligible"),
    ((125, False), (126, True), (127, True)),
)
def test_feature_support_boundary_is_inclusive(
    observed_count: int, eligible: bool
) -> None:
    metadata = structure()
    feature = FEATURES[0]
    removal_count = len(metadata["rows"]) - observed_count
    removal_indices = []
    for family in metadata["scenarioFamilyIds"]:
        for candidate in FROZEN:
            removal_indices.append(
                next(
                    index
                    for index, row in enumerate(metadata["rows"])
                    if row["candidateId"] == candidate
                    and row["scenarioFamilyId"] == family
                )
            )
            if len(removal_indices) == removal_count:
                break
        if len(removal_indices) == removal_count:
            break
    for index in removal_indices:
        metadata["rows"][index]["observedFeatureIds"].remove(feature)
    result = assess_stratum(metadata, frozen_candidate_ids=FROZEN)
    decision = result["preprocessing"]["featureDecisions"][feature]
    assert decision["observedCount"] == observed_count
    assert decision["denominatorCount"] == 140
    assert decision["eligible"] is eligible


def test_missing_crossfit_test_fold_fails_closed() -> None:
    same_fold = []
    ordinal = 0
    while len(same_fold) < 2:
        family = f"same-fold-{ordinal}"
        if fold_for_family(family) == 0:
            same_fold.append(family)
        ordinal += 1
    metadata = structure(families=tuple(same_fold))
    result = assess_stratum(metadata, frozen_candidate_ids=FROZEN)
    assert not result["preprocessing"]["admissible"]
    assert "EMPTY_CROSSFIT_TEST_FOLD" in result["preprocessing"]["reasonCodes"]


def test_missing_candidate_feature_blocks_profile_construction() -> None:
    metadata = structure()
    for row in metadata["rows"]:
        if row["candidateId"] == FROZEN[0]:
            row["observedFeatureIds"].remove(FEATURES[0])
    result = assess_stratum(metadata, frozen_candidate_ids=FROZEN)
    assert (
        "CANDIDATE_PROFILE_FEATURE_UNOBSERVED" in result["preprocessing"]["reasonCodes"]
    )


def test_non_32_transition_series_blocks_only_affected_change_slot() -> None:
    metadata = structure()
    metadata["rows"][0]["orderedSeriesLength"] = 31
    result = assess_stratum(metadata, frozen_candidate_ids=FROZEN)
    assert not result["changePoint"]["candidateDecisions"][FROZEN[0]]["admissible"]
    assert result["changePoint"]["candidateDecisions"][FROZEN[1]]["admissible"]
    assert result["clustering"]["admissible"]


def test_one_family_candidate_is_not_a_change_point_prevalence_test() -> None:
    metadata = structure()
    keep_family = metadata["scenarioFamilyIds"][0]
    metadata["rows"] = [
        row
        for row in metadata["rows"]
        if row["candidateId"] != FROZEN[0] or row["scenarioFamilyId"] == keep_family
    ]
    result = assess_stratum(metadata, frozen_candidate_ids=FROZEN)
    decision = result["changePoint"]["candidateDecisions"][FROZEN[0]]
    assert not decision["admissible"]
    assert "FEWER_THAN_TWO_SCENARIO_FAMILIES_FOR_PREVALENCE" in decision["reasonCodes"]


def test_malformed_and_forged_metadata_fail_closed() -> None:
    duplicate = structure()
    duplicate["rows"].append(deepcopy(duplicate["rows"][0]))
    with pytest.raises(FeasibilityContractError, match="duplicate"):
        assess_stratum(duplicate, frozen_candidate_ids=FROZEN)
    forged = structure()
    forged["profileCommitmentsByCandidate"][FROZEN[0]] = "not-a-hash"
    with pytest.raises(FeasibilityContractError, match="SHA-256"):
        assess_stratum(forged, frozen_candidate_ids=FROZEN)
    unknown = structure()
    unknown["rows"][0]["observedFeatureIds"].append("unknown")
    with pytest.raises(FeasibilityContractError, match="unregistered"):
        assess_stratum(unknown, frozen_candidate_ids=FROZEN)


def test_worker_order_does_not_change_assessment() -> None:
    natural = structure()
    reverse = deepcopy(natural)
    reverse["rows"].reverse()
    left = assess_stratum(natural, frozen_candidate_ids=FROZEN)
    right = assess_stratum(reverse, frozen_candidate_ids=FROZEN)
    assert left["assessmentSha256"] == right["assessmentSha256"]


def test_holm_family_never_shrinks_for_infeasible_methods() -> None:
    full = assess_stratum(structure(), frozen_candidate_ids=FROZEN)
    constrained_metadata = structure()
    for candidate in FROZEN[5:]:
        constrained_metadata["profileCommitmentsByCandidate"][candidate] = (
            constrained_metadata["profileCommitmentsByCandidate"][FROZEN[0]]
        )
    constrained = assess_stratum(constrained_metadata, frozen_candidate_ids=FROZEN)
    assert len(multiplicity_slots(full, frozen_candidate_ids=FROZEN)) == 21
    slots = multiplicity_slots(constrained, frozen_candidate_ids=FROZEN)
    assert len(slots) == 21
    assert sum(row["slotState"] == "non_evidentiary_infeasible" for row in slots) == 6
    assert all(
        row["rawPValue"] == 1.0
        for row in slots
        if row["slotState"] == "non_evidentiary_infeasible"
    )


def test_p_value_cannot_be_assigned_to_infeasible_slot() -> None:
    metadata = structure(candidates=FROZEN[:5])
    result = assess_stratum(metadata, frozen_candidate_ids=FROZEN)
    slot = f"{result['taskId']}::{result['statusStratum']}::clustering::0"
    with pytest.raises(FeasibilityContractError, match="infeasible"):
        multiplicity_slots(
            result,
            frozen_candidate_ids=FROZEN,
            evidentiary_p_values={slot: 0.001},
        )


def test_conservative_holm_is_no_less_stringent_than_subset() -> None:
    result = assess_stratum(structure(), frozen_candidate_ids=FROZEN)
    prefix = f"{result['taskId']}::{result['statusStratum']}"
    p_values = {
        f"{prefix}::clustering::0": 0.001,
        f"{prefix}::anomaly::0": 0.01,
        f"{prefix}::change_point::{FROZEN[0]}": 0.02,
    }
    full = holm_adjust_slots(
        multiplicity_slots(
            result, frozen_candidate_ids=FROZEN, evidentiary_p_values=p_values
        )
    )
    full_adjusted = {
        row["slotId"]: row["holmAdjustedPValue"]
        for row in full
        if row["slotId"] in p_values
    }
    subset = holm_adjust_slots(
        [
            {
                "slotId": slot_id,
                "methodFamily": "synthetic",
                "slotState": "evidentiary_candidate",
                "rawPValue": p_value,
                "inMultiplicityFamily": True,
            }
            for slot_id, p_value in p_values.items()
        ]
    )
    assert all(
        full_adjusted[row["slotId"]] >= row["holmAdjustedPValue"] for row in subset
    )


def test_global_holm_unions_realized_task_status_strata_without_shrinking() -> None:
    local_metadata = structure()
    memory_metadata = structure()
    memory_metadata["taskId"] = "e07_s02_spatial2d_memory"
    local = assess_stratum(local_metadata, frozen_candidate_ids=FROZEN)
    memory = assess_stratum(memory_metadata, frozen_candidate_ids=FROZEN)
    local_slot = (
        f"{local['taskId']}::{local['statusStratum']}::change_point::{FROZEN[0]}"
    )
    rows = global_multiplicity_slots(
        [memory, local],
        frozen_candidate_ids=FROZEN,
        evidentiary_p_values={local_slot: 0.001},
    )
    adjusted = holm_adjust_slots(rows)
    selected = next(row for row in adjusted if row["slotId"] == local_slot)
    assert len(rows) == 42
    assert selected["holmAdjustedPValue"] == pytest.approx(0.042)
    assert sum(row["rawPValue"] == 1.0 for row in rows) == 41


def test_global_holm_rejects_duplicate_task_status_stratum() -> None:
    assessment = assess_stratum(structure(), frozen_candidate_ids=FROZEN)
    with pytest.raises(FeasibilityContractError, match="repeats"):
        global_multiplicity_slots([assessment, assessment], frozen_candidate_ids=FROZEN)


def test_synthetic_reproduction_requires_locked_population_identity() -> None:
    discovery_metadata = structure()
    discovery = assess_stratum(discovery_metadata, frozen_candidate_ids=FROZEN)
    identical = assess_reproduction_structure(
        discovery,
        deepcopy(discovery_metadata),
        frozen_candidate_ids=FROZEN,
    )
    assert identical["lockedCandidatePopulationIdentity"]
    assert identical["methodAdmissibility"]["clustering"]
    reordered = deepcopy(discovery_metadata)
    reordered["candidateIds"].reverse()
    mismatch = assess_reproduction_structure(
        discovery, reordered, frozen_candidate_ids=FROZEN
    )
    assert not mismatch["lockedCandidatePopulationIdentity"]
    assert not mismatch["methodAdmissibility"]["clustering"]
    assert mismatch["frozenReproductionRowsExecuted"] == 0
